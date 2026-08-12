import logging
import random
import torch
from abc import ABC, abstractmethod
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from typing import Any, Optional
from typing import Dict, Tuple, List

from src.common_utils.constants import ImageTriggerType, ImageTriggerLocation
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import LabelFlipConfig, BackdoorImageConfig, BackdoorTextConfig, \
    BackdoorTabularConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PoisonGenerator(ABC):
    """
    Abstract base class for all data poisoning attack generators.
    This defines a standard interface for applying a poison to a single data sample.
    """

    @abstractmethod
    def apply(self, data: Any, label: int) -> Tuple[Any, int]:
        """
        Apply the poisoning logic to a single data sample.

        Args:
            data (Any): The feature part of the sample (e.g., image tensor).
            label (int): The original label of the sample.

        Returns:
            A tuple (poisoned_data, poisoned_label).
        """
        pass


class BackdoorImageGenerator(PoisonGenerator):
    # The __init__ signature is now much simpler
    # --- Dataset-aware trigger auto-switch -----------------------------------
    # The configured trigger (BLENDED_PATCH / alpha=0.2 / 5x5) works for CIFAR
    # but vanishes against FEMNIST's white background (ASR drops to ~5% = random
    # baseline). When we see a 28x28 input (FEMNIST-style), we transparently
    # swap in a stronger CHECKERBOARD / alpha=0.8 / 4x4 trigger. CIFAR cells
    # (32x32 or larger) keep the original configured trigger untouched so
    # existing CIFAR results remain comparable.
    _FEMNIST_SPATIAL = 28
    _FEMNIST_TRIGGER_SIZE = (4, 4)
    _FEMNIST_BLEND_ALPHA = 0.8

    def __init__(self, config: BackdoorImageConfig, device: torch.device):
        if not (0.0 <= config.blend_alpha <= 1.0):
            raise ValueError("blend_alpha must be between 0.0 and 1.0")

        self.config = config
        self.target_label = config.target_label
        self.blend_alpha = config.blend_alpha
        self.location = config.location
        self.randomize_location = config.randomize_location
        self.device = device
        self.trigger_pattern = self._generate_trigger_pattern(
            config.trigger_type, config.channels, config.trigger_size, device
        )
        # Pre-build the FEMNIST override trigger so the per-batch hot path
        # does zero extra work until it actually fires.
        self._femnist_trigger_pattern = self._generate_trigger_pattern(
            ImageTriggerType.CHECKERBOARD,
            config.channels,
            self._FEMNIST_TRIGGER_SIZE,
            device,
        )

    def apply(self, data: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """
        Applies the trigger to an image tensor and returns the poisoned sample.
        This method fulfills the PoisonGenerator interface.
        """
        if not isinstance(data, torch.Tensor) or data.dim() < 3:
            raise TypeError(f"Expected data to be a tensor with at least 3 dimensions (C, H, W), but got {data.shape}")

        # Auto-switch to the FEMNIST override for 28x28 inputs.
        H, W = data.shape[-2:]
        if H == self._FEMNIST_SPATIAL and W == self._FEMNIST_SPATIAL:
            poisoned_image = self._apply_trigger_to_tensor(
                data, self._femnist_trigger_pattern, alpha=self._FEMNIST_BLEND_ALPHA
            )
        else:
            poisoned_image = self._apply_trigger_to_tensor(data, self.trigger_pattern)
        return poisoned_image, self.target_label

    def _apply_trigger_to_tensor(self, image: torch.Tensor, trigger: torch.Tensor,
                                 alpha: Optional[float] = None) -> torch.Tensor:
        """Internal logic to blend a trigger pattern onto an image tensor.

        `alpha` lets callers override self.blend_alpha without mutating state
        (used by the FEMNIST auto-switch path).
        """
        image = image.clone().float()
        _, H, W = image.shape
        _, h, w = trigger.shape

        if self.randomize_location:
            y = torch.randint(0, max(H - h, 1), (1,)).item()
            x = torch.randint(0, max(W - w, 1), (1,)).item()
        else:
            # Use a mapping for cleaner lookup
            location_coords = {
                ImageTriggerLocation.TOP_LEFT: (0, 0),
                ImageTriggerLocation.CENTER: ((H - h) // 2, (W - w) // 2),
                ImageTriggerLocation.BOTTOM_RIGHT: (H - h, W - w)
            }
            # Default to CENTER if location is not in the map, though enum ensures it will be
            y, x = location_coords.get(self.location, location_coords[ImageTriggerLocation.CENTER])

        # Ensure the trigger is on the same device as the image
        trigger = trigger.to(image.device)

        # Blend the trigger into the selected region
        blend_alpha = self.blend_alpha if alpha is None else alpha
        region = image[:, y:y + h, x:x + w]
        blended = (1.0 - blend_alpha) * region + blend_alpha * trigger
        image[:, y:y + h, x:x + w] = torch.clamp(blended, 0.0, 1.0)
        return image

    @staticmethod
    def _generate_trigger_pattern(trigger_type: ImageTriggerType, channels: int, size: Tuple[int, int],
                                  device: torch.device) -> torch.Tensor:
        """Generates a trigger pattern tensor on the SPECIFIED device."""
        h, w = size
        if trigger_type == ImageTriggerType.BLENDED_PATCH:
            # Use the 'device' argument when creating the tensor
            return torch.ones(channels, h, w, device=device)
        elif trigger_type == ImageTriggerType.CHECKERBOARD:
            # Move intermediate tensors to the correct device
            coords = torch.arange(h, device=device).unsqueeze(1) + torch.arange(w, device=device).unsqueeze(0)
            checkerboard = (coords % 2 == 0).float()
            return checkerboard.unsqueeze(0).repeat(channels, 1, 1)
        elif trigger_type == ImageTriggerType.NOISE:
            return torch.rand(channels, h, w, device=device)
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")

    def update_trigger(self, new_trigger: torch.Tensor):
        """Allows for dynamically updating the trigger, e.g., after optimization."""
        # Let the trigger live on its current device (e.g., the GPU)
        self.trigger_pattern = new_trigger.clone()

    def get_trigger(self) -> torch.Tensor:
        """Returns a copy of the current trigger pattern."""
        return self.trigger_pattern.clone()


class LabelFlipGenerator(PoisonGenerator):
    """
    A poison generator that only flips the label, leaving the data unchanged.
    """

    def __init__(self, config: LabelFlipConfig):
        if not isinstance(config.num_classes, int) or config.num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than 1.")
        if config.attack_mode not in ["fixed_target", "random_different"]:
            raise ValueError("attack_mode must be 'fixed_target' or 'random_different'")

        self.config = config

        if self.config.attack_mode == "fixed_target":
            if not (0 <= self.config.target_label < self.config.num_classes):
                raise ValueError(f"target_label must be in [0, {self.config.num_classes - 1}]")
            logging.info(f"Initialized LabelFlipGenerator (mode: fixed_target, target: {self.config.target_label})")
        else:
            logging.info(f"Initialized LabelFlipGenerator (mode: random_different)")

    def apply(self, data: Any, label: int) -> Tuple[Any, int]:
        """
        Returns the original data with a flipped label.
        """
        if self.config.attack_mode == "fixed_target":
            if label == self.config.target_label:
                return data, (label + 1) % self.config.num_classes
            return data, self.config.target_label
        else:  # 'random_different'
            possible_targets = list(range(self.config.num_classes))
            possible_targets.remove(label)
            return data, random.choice(possible_targets)


class BackdoorTextGenerator(PoisonGenerator):
    """
    Generates backdoor attacks for text data by inserting trigger words/phrases.
    (Updated for modern torchtext)
    """

    def __init__(self, config: BackdoorTextConfig):
        """
        Initializes the text backdoor generator from a configuration object.
        """
        # --- MODIFICATION #1: Check the object type instead of the old attribute ---
        if not isinstance(config.vocab, Vocab):
            raise TypeError("Provided vocab object must be a torchtext.vocab.Vocab instance.")

        if not config.trigger_content or not isinstance(config.trigger_content, str):
            raise ValueError("trigger_content must be a non-empty string.")

        self.config = config
        self.vocab = config.vocab
        # This tokenizer should ideally be consistent with the one used to build the vocab
        self.tokenizer = get_tokenizer('basic_english')

        self.trigger_token_ids = self._string_to_ids(config.trigger_content)

        if not self.trigger_token_ids:
            raise ValueError(f"Trigger content '{config.trigger_content}' could not be converted to token IDs.")

        logging.info(
            f"Initialized BackdoorTextGenerator with trigger: '{config.trigger_content}' -> {self.trigger_token_ids}")

    def _string_to_ids(self, text: str) -> List[int]:
        """Converts a string to a list of token IDs using the modern vocab object."""
        tokens = self.tokenizer(text)
        # The modern vocab object is callable
        return self.vocab(tokens)

    def apply(self, data: Any, label: int) -> Tuple[Any, int]:
        """
        Applies the backdoor trigger to a single data sample,
        switching logic based on input type (torch.Tensor or list).

        Args:
            data (Any): The original data, either a 1D torch.Tensor or a List[int]
            label (int): The original label.

        Returns:
            A tuple of (poisoned_data, poisoned_label), matching the input type.
        """
        # Poison the sample by changing its label to the target label
        poisoned_label = self.config.target_label
        trigger_len = len(self.trigger_token_ids)

        # --- BRANCH 1: TENSOR INPUT (e.g., from evaluate_attack_performance) ---
        # Uses REPLACEMENT logic to preserve sequence length for torch.stack()
        if isinstance(data, torch.Tensor):
            poisoned_data = data.clone()

            # Create trigger tensor on the same device as the data
            trigger_tensor = torch.tensor(self.trigger_token_ids,
                                          dtype=torch.long,
                                          device=poisoned_data.device)

            # Ensure sequence is long enough to hold the trigger
            if len(poisoned_data) < trigger_len:
                logging.warning(
                    f"Data sample length ({len(poisoned_data)}) is shorter than "
                    f"trigger length ({trigger_len}). Skipping trigger insertion for this sample."
                )
                return poisoned_data, poisoned_label  # Return (tensor, int)

            # Replace tokens at the specified location
            if self.config.location == "start":
                poisoned_data[:trigger_len] = trigger_tensor
            elif self.config.location == "end":
                poisoned_data[-trigger_len:] = trigger_tensor
            else:  # Default to end
                poisoned_data[-trigger_len:] = trigger_tensor

            return poisoned_data, poisoned_label  # Return (tensor, int)

        # --- BRANCH 2: LIST INPUT ---
        # Uses REPLACEMENT logic (same as tensor branch) to preserve sequence length
        elif isinstance(data, list):
            poisoned_data = list(data)  # shallow copy
            if len(poisoned_data) < trigger_len:
                logging.warning(
                    f"Data sample length ({len(poisoned_data)}) is shorter than "
                    f"trigger length ({trigger_len}). Skipping trigger insertion."
                )
                return poisoned_data, poisoned_label

            if self.config.location == "start":
                poisoned_data[:trigger_len] = self.trigger_token_ids
            else:  # "end" or default
                poisoned_data[-trigger_len:] = self.trigger_token_ids

            return poisoned_data, poisoned_label  # Return (list, int)

        # --- ERROR BRANCH ---
        else:
            raise TypeError(
                f"BackdoorTextGenerator.apply received unsupported data type: {type(data)}. "
                "Expected torch.Tensor or list."
            )


class BackdoorTabularGenerator(PoisonGenerator):
    """Applies a feature-based trigger to tabular data."""

    def __init__(self, config: BackdoorTabularConfig, feature_to_idx: Dict[str, int]):
        self.config = config

        self.target_label = config.target_label

        self.trigger_map: List[Tuple[int, float]] = []
        for feature_name, trigger_value in config.trigger_conditions.items():
            if feature_name not in feature_to_idx:
                raise ValueError(f"Trigger feature '{feature_name}' not found in feature map.")
            feature_index = feature_to_idx[feature_name]
            self.trigger_map.append((feature_index, float(trigger_value)))

        logging.info(f"BackdoorTabularGenerator initialized:")
        logging.info(f"  - Target Label: {self.target_label}")
        logging.info(f"  - Trigger Map (Index, Value): {self.trigger_map}")

    def apply(self, data: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """
        Applies the feature-based trigger to a tabular data tensor.
        (This method's logic is correct, no changes needed)
        """
        if not isinstance(data, torch.Tensor) or data.dim() != 1:
            raise TypeError(f"Expected data to be a 1D tensor, but got shape {data.shape}")

        poisoned_data = data.clone()

        # Overwrite the feature values with the trigger values
        for index, value in self.trigger_map:
            poisoned_data[index] = value

        return poisoned_data, self.target_label