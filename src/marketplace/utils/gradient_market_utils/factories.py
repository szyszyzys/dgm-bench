import copy
import logging
from typing import Callable, Optional, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset

# Import the specific generator classes needed for the generic poisoner
from src.attacks.gradient_market.poison_attack.attack_utils import (
    LabelFlipGenerator, PoisonGenerator
)
from src.common_utils.constants.enums import PoisonType
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, RuntimeDataConfig
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import LabelFlipConfig
from src.common_utils.model_utils import _log_param_stats
# Import the specific seller classes
from src.participants.seller.gradient_seller import (
    GradientSeller, AdvancedBackdoorAdversarySeller,
    AdvancedPoisoningAdversarySeller, AdaptiveAttackerSeller, DrowningAttackerSeller,
    LLMAdversarySeller, LazyGradientSeller
)
from src.model.models import TextCNN


class SellerFactory:
    """Handles creating and configuring different seller types from a unified AppConfig."""

    # Map the ENUM directly to the class for better type safety
    ADVERSARY_CLASS_MAP = {
        PoisonType.IMAGE_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.TEXT_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.TABULAR_BACKDOOR: AdvancedBackdoorAdversarySeller,
        PoisonType.LABEL_FLIP: AdvancedPoisoningAdversarySeller,
    }

    def __init__(self, cfg: AppConfig, model_factory: Callable[[], nn.Module], num_classes: int,
                 **kwargs):
        """Initializes the factory with the main application config and runtime args."""
        self.cfg = cfg
        self.model_factory = model_factory
        self.num_classes = num_classes  # Store it
        self.runtime_kwargs = kwargs

    def _create_poison_generator(self, poison_type_str: str) -> Optional[PoisonGenerator]:
        """
        Factory method to create a poison generator from a string identifier.
        This helper centralizes the creation logic for all poison types.
        """
        adv_cfg = self.cfg.adversary_seller_config
        model_type = self.cfg.experiment.dataset_type
        device = self.cfg.experiment.device

        try:
            poison_type = PoisonType(poison_type_str)
        except ValueError:
            logging.warning(f"Invalid poison type string: '{poison_type_str}'")
            return None

        # If it's any kind of backdoor, delegate to the single source of truth.
        if 'backdoor' in poison_type.value:
            # 1. Make a copy of kwargs to avoid modifying the original.
            kwargs_for_generator = self.runtime_kwargs.copy()
            # 2. Safely remove the conflicting key.
            kwargs_for_generator.pop('model_type', None)

            # 3. Call the static method on the seller class
            return AdvancedBackdoorAdversarySeller._create_poison_generator(
                adv_cfg=adv_cfg,
                model_type=model_type,
                device=device,
                **kwargs_for_generator
            )

        elif poison_type == PoisonType.LABEL_FLIP:
            active_params = adv_cfg.poisoning.label_flip_params
            generator_cfg = LabelFlipConfig(
                num_classes=self.num_classes,
                attack_mode=active_params.mode.value,
                target_label=active_params.target_label
            )
            return LabelFlipGenerator(generator_cfg)

        logging.warning(
            f"Could not create a generator for poison type '{poison_type_str}' "
            f"with model type '{model_type}'. Check configuration."
        )
        return None

    def create_seller(self,
                      seller_id: str,
                      dataset: Dataset,
                      is_adversary: bool,
                      collate_fn: Callable = None):
        """Creates a seller instance, assembling configs and dependencies on the fly."""
        # Detect LLM mode from experiment config
        is_llm = self.cfg.experiment.dataset_type == "llm"
        data_cfg = RuntimeDataConfig(
            dataset=dataset,
            num_classes=self.num_classes,
            collate_fn=collate_fn,
            dataset_type="llm" if is_llm else "standard",
            llm_task=self.runtime_kwargs.get("llm_task") if is_llm else None,
        )
        training_cfg = self.cfg.training
        base_kwargs = {
            "seller_id": seller_id,
            "data_config": data_cfg,
            "training_config": training_cfg,
            "model_factory": self.model_factory,
            "save_path": self.cfg.experiment.save_path,
            "device": self.cfg.experiment.device,
        }

        if not is_adversary:
            benign_kwargs = self.runtime_kwargs.copy()
            benign_kwargs.pop('validation_loader', None)
            # Pass DP config to benign sellers (adversaries don't get DP noise)
            if hasattr(self.cfg, 'dp') and self.cfg.dp.enabled:
                benign_kwargs['dp_config'] = self.cfg.dp

            # --- Lazy / Free-Rider injection (Step 11) -----------------------
            # If lazy_seller.is_active, deterministically promote a fraction of
            # benign sellers to LazyGradientSeller. We use a stable hash on
            # (seller_id, seed) so the same sellers are lazy across reruns.
            lazy_cfg = getattr(self.cfg.adversary_seller_config, 'lazy_seller', None)
            if lazy_cfg is not None and getattr(lazy_cfg, 'is_active', False):
                import hashlib
                h = hashlib.md5(f"{seller_id}|{self.cfg.seed}".encode()).hexdigest()
                # Map first 8 hex chars -> [0, 1)
                bucket = int(h[:8], 16) / float(1 << 32)
                if bucket < float(lazy_cfg.lazy_rate):
                    logging.info(
                        f"Creating LazyGradientSeller for {seller_id} "
                        f"(strategy={lazy_cfg.strategy}, rate={lazy_cfg.lazy_rate})"
                    )
                    return LazyGradientSeller(
                        lazy_config=lazy_cfg,
                        **base_kwargs,
                        **benign_kwargs,
                    )

            return GradientSeller(**base_kwargs, **benign_kwargs)

        # --- LLM-specific attacks (GAP, Alignment Poisoning) ---
        if is_llm:
            llm_attack_cfg = self.cfg.data.llm.llm_attack
            if llm_attack_cfg and llm_attack_cfg.attack_type not in ("none", "", None):
                logging.info(f"Creating LLMAdversarySeller for {seller_id} "
                             f"(attack={llm_attack_cfg.attack_type})")
                extra_kwargs = self.runtime_kwargs.copy()
                extra_kwargs.pop('validation_loader', None)
                return LLMAdversarySeller(
                    llm_attack_config=llm_attack_cfg,
                    tokenizer=self.runtime_kwargs.get("tokenizer"),
                    **base_kwargs,
                    **extra_kwargs,
                )

        # The Adaptive Attacker is the "Superset". It can do drowning/stealthy blend internally.
        if self.cfg.adversary_seller_config.adaptive_attack.is_active:
            logging.info(f"Creating AdaptiveAttackerSeller for {seller_id}")

            # Ensure validation_loader is extracted from runtime_kwargs if present
            return AdaptiveAttackerSeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                model_type=self.cfg.experiment.dataset_type,
                **self.runtime_kwargs  # Contains validation_loader
            )

        if self.cfg.adversary_seller_config.drowning_attack.is_active:
            logging.info(f"Creating DrowningAttackerSeller for {seller_id}")
            return DrowningAttackerSeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                model_type=self.cfg.experiment.dataset_type,  # Pass model type just in case
                **self.runtime_kwargs
            )

        # --- Standard Poisoning Logic (Backdoor/Label Flip) ---
        attack_type = self.cfg.adversary_seller_config.poisoning.type
        AdversaryClass = self.ADVERSARY_CLASS_MAP.get(attack_type)

        if not AdversaryClass:
            logging.warning(f"Attack type '{attack_type.value}' is invalid. Creating a benign seller.")
            return GradientSeller(**base_kwargs, **self.runtime_kwargs)

        if AdversaryClass is AdvancedBackdoorAdversarySeller:
            logging.info(f"Creating AdvancedBackdoorAdversarySeller for {seller_id}")
            kwargs_for_seller = self.runtime_kwargs.copy()
            kwargs_for_seller.pop('model_type', None)

            return AdvancedBackdoorAdversarySeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                model_type=self.cfg.experiment.dataset_type,
                **kwargs_for_seller
            )

        elif AdversaryClass is AdvancedPoisoningAdversarySeller:
            logging.info(f"Creating AdvancedPoisoningAdversarySeller for {seller_id}")
            poison_generator = self._create_poison_generator(attack_type.value)  # Fixed call

            if not poison_generator:
                # Use create_generic_poison_generator if _create_poison_generator returned None
                # or handle fallback
                logging.warning(f"Could not create generator. Creating benign seller.")
                return GradientSeller(**base_kwargs, **self.runtime_kwargs)

            return AdvancedPoisoningAdversarySeller(
                **base_kwargs,
                adversary_config=self.cfg.adversary_seller_config,
                poison_generator=poison_generator,
                **self.runtime_kwargs
            )

        logging.warning(f"Unhandled adversary class '{AdversaryClass.__name__}'. Creating benign seller.")
        return GradientSeller(**base_kwargs, **self.runtime_kwargs)


def create_generic_poison_generator(cfg: AppConfig, **kwargs: Dict[str, Any]) -> Optional[PoisonGenerator]:
    """
    A simplified factory that creates generators for generic poisoning attacks.
    ** Backdoor logic has been removed, as that is now handled inside the
    ** AdvancedBackdoorAdversarySeller class itself.
    """
    poison_cfg = cfg.adversary_seller_config.poisoning
    active_params = poison_cfg.active_params

    if not active_params:
        return None

    if poison_cfg.type == PoisonType.LABEL_FLIP:
        generator_cfg = LabelFlipConfig(
            num_classes=cfg.experiment.num_classes,
            attack_mode=active_params.mode.value,
            target_label=active_params.target_label
        )
        return LabelFlipGenerator(generator_cfg)

    # This function would be extended for other *generic* attacks like noise, etc.
    logging.warning(f"Unhandled poison type in generic generator factory: {poison_cfg.type}")
    return None


class StatefulModelFactory:
    """
    A stateful model factory that maintains a reference to the global model.
    Every call to the factory returns a new model initialized with current global weights.
    Works across text, image, and tabular modalities.
    """

    def __init__(self, base_factory: Callable[[], nn.Module], device: str):
        """
        Args:
            base_factory: Function that creates a fresh model instance
            device: Device to place models on
        """
        self.base_factory = base_factory
        self.device = device
        self.global_model = None
        logging.info("✅ Created stateful model factory (will maintain global model reference)")

    def __call__(self) -> nn.Module:
        """
        Create a new model instance.
        If global model exists, copy its weights to the new model.

        Returns:
            nn.Module with current global weights (if set)
        """
        # Create fresh model
        model = self.base_factory()

        # Copy global weights if available
        if self.global_model is not None:
            model.load_state_dict(self.global_model.state_dict())
            logging.debug("Model created with current global weights")
        else:
            logging.debug("Model created with fresh initialization (no global model set yet)")

        return model

    def set_global_model(self, model: nn.Module):
        """
        Register the global model with this factory.
        All subsequent factory calls will copy weights from this model.

        Args:
            model: The global model to maintain reference to
        """
        self.global_model = model
        logging.info("✅ Global model registered with factory")

    def get_global_model(self) -> Optional[nn.Module]:
        """Get reference to the current global model."""
        return self.global_model


class LLMModelFactory:
    """
    Factory for LLM PEFT models (LoRA / Soft Prompt).

    The frozen base model is loaded once and cached at module level.
    Each factory call creates a fresh PEFT adapter with current global LoRA weights.
    """

    @staticmethod
    def create_model(llm_cfg, device: str = "cpu") -> nn.Module:
        """Create a FederatedLoRA or FederatedSoftPrompt model."""
        from src.model.llm_models import FederatedLoRA, FederatedSoftPrompt

        if llm_cfg.peft_method == "soft_prompt":
            return FederatedSoftPrompt(llm_cfg.base_model_name, llm_cfg, device=device)
        else:
            return FederatedLoRA(llm_cfg.base_model_name, llm_cfg, device=device)

    @staticmethod
    def create_factory(llm_cfg, device: str = "cpu") -> Callable[[], nn.Module]:
        """
        Returns a zero-argument callable that creates a fresh PEFT adapter.
        Compatible with StatefulModelFactory.
        """
        def factory():
            return LLMModelFactory.create_model(llm_cfg, device=device)
        return factory


class TextModelFactory:
    """Factory class for creating text models."""

    @staticmethod
    def create_model(dataset_name: str, num_classes: int, vocab_size: int,
                     padding_idx: int, device: str = 'cpu', **model_kwargs) -> nn.Module:
        """Create a text model based on dataset configuration."""
        logging.info(f"🧠 Creating text model for dataset: {dataset_name}")

        target_device = torch.device(device)
        model: nn.Module

        # Instantiate the model on CPU
        match dataset_name.lower():
            case "ag_news" | "trec":
                embed_dim = model_kwargs.get("embed_dim", 100)
                num_filters = model_kwargs.get("num_filters", 100)
                filter_sizes = model_kwargs.get("filter_sizes", [3, 4, 5])
                dropout = model_kwargs.get("dropout", 0.5)

                if not isinstance(filter_sizes, list):
                    raise TypeError("'filter_sizes' must be a list")

                model = TextCNN(
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    num_filters=num_filters,
                    filter_sizes=filter_sizes,
                    num_class=num_classes,
                    dropout=dropout,
                    padding_idx=padding_idx
                )
            case _:
                raise NotImplementedError(f"Model not found for dataset {dataset_name}")

        # Move to target device
        model = model.to(target_device)
        logging.info(f"--- Text Model moved to {target_device} ---")
        log_dtype = model.embedding.weight.dtype
        _log_param_stats(model, "embedding.weight", f"After .to({target_device}) ({log_dtype})")

        # Verify no NaN/Inf
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logging.error(f"--- ❌ VERIFICATION FAILED FOR {name} ---")
                _log_param_stats(model, name, "FAILURE")
                raise RuntimeError(f"NaN/Inf in parameter '{name}' after initialization!")

        logging.info("--- ✅ Text Model verification PASSED ---")

        # Log model details
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"  -> Instantiated model class: '{model.__class__.__name__}'")
        logging.info(f"  -> Total parameters: {num_params:,}")
        logging.info(f"  -> Trainable parameters: {num_trainable:,}")

        return model

    @staticmethod
    def create_factory(dataset_name: str, num_classes: int, vocab_size: int,
                       padding_idx: int, device: str = 'cpu', **model_kwargs) -> Callable[[], nn.Module]:
        """
        Create a zero-argument factory function with frozen parameters.

        Args:
            dataset_name: Name of the dataset (determines model architecture)
            num_classes: Number of output classes
            vocab_size: Size of vocabulary
            padding_idx: Padding index for embeddings
            device: Device to place models on
            **model_kwargs: Additional model configuration (embed_dim, num_filters, etc.)

        Returns:
            A callable that takes no arguments and returns a model instance
        """
        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        # Deep copy all parameters to freeze them at factory creation time
        frozen_dataset_name = str(dataset_name)
        frozen_num_classes = int(num_classes)
        frozen_vocab_size = int(vocab_size)
        frozen_padding_idx = int(padding_idx)
        frozen_device = str(device)
        frozen_model_kwargs = copy.deepcopy(model_kwargs)

        def model_factory() -> nn.Module:
            """Zero-argument factory that creates a model with frozen config."""
            return TextModelFactory.create_model(
                dataset_name=frozen_dataset_name,
                num_classes=frozen_num_classes,
                vocab_size=frozen_vocab_size,
                padding_idx=frozen_padding_idx,
                device=frozen_device,
                **frozen_model_kwargs
            )

        # Validate factory creates valid models
        try:
            test_model = model_factory()
            num_params = sum(p.numel() for p in test_model.parameters())
            logging.info(f"Text model factory created and validated:")
            logging.info(f"  - Dataset: {frozen_dataset_name}")
            logging.info(f"  - Architecture: {test_model.__class__.__name__}")
            logging.info(f"  - Parameters: {num_params:,}")
            logging.info(f"  - Vocabulary size: {frozen_vocab_size}")
            logging.info(f"  - Output classes: {frozen_num_classes}")
            del test_model  # Clean up
        except Exception as e:
            logging.error(f"Failed to create test model from factory: {e}")
            raise

        return model_factory
