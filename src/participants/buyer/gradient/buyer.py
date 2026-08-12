import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.participants.seller.gradient_seller import GradientSeller
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import BuyerAttackConfig


class MaliciousBuyerProxy(GradientSeller):
    """
    A proxy that acts on behalf of a malicious buyer to submit a manipulated root gradient.
    It inherits from GradientSeller to reuse the gradient computation logic but overrides the
    behavior based on the configured attack type.
    """

    def __init__(self, *args, attack_config: BuyerAttackConfig, num_classes: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_cfg = attack_config
        self.num_classes = num_classes  # 🆕 ADD THIS

        if not self.attack_cfg.is_active:
            raise ValueError("MaliciousBuyerProxy created but is_active is False in config.")

        self.round_counter = 0

        # For class-based exclusion, cache the filtered dataset
        self.exclusion_dataset = None
        if self.attack_cfg.attack_type == "class_exclusion":
            self._prepare_class_exclusion_dataset()

        logging.info(
            f"[{self.seller_id}] Initialized as MaliciousBuyerProxy. "
            f"Attack mode: '{self.attack_cfg.attack_type}', "
            f"Dataset classes: {self.num_classes}."  # 🆕 LOG THIS
        )

    def _prepare_class_exclusion_dataset(self):
        """
        Pre-filter the buyer's dataset to only include samples from targeted classes.
        This creates a biased baseline that will systematically favor sellers with similar data.
        """
        target_classes = self.attack_cfg.exclusion_target_classes
        exclude_classes = self.attack_cfg.exclusion_exclude_classes

        if not target_classes and not exclude_classes:
            logging.error(
                "Class exclusion attack active but neither target_classes "
                "nor exclude_classes specified! Using full dataset."
            )
            self.exclusion_dataset = self.dataset
            return

        indices = []
        for i, (_, label) in enumerate(self.dataset):
            # Include logic: Either in target_classes OR not in exclude_classes
            if target_classes:
                # Positive selection: only include these classes
                if label in target_classes:
                    indices.append(i)
            elif exclude_classes:
                # Negative selection: exclude these classes
                if label not in exclude_classes:
                    indices.append(i)

        if not indices:
            logging.warning(
                f"Class filtering resulted in empty dataset! "
                f"Target classes: {target_classes}, Exclude classes: {exclude_classes}. "
                f"Using full dataset as fallback."
            )
            self.exclusion_dataset = self.dataset
        else:
            self.exclusion_dataset = Subset(self.dataset, indices)
            logging.info(
                f"[{self.seller_id}][Class Exclusion] Filtered dataset: "
                f"{len(indices)}/{len(self.dataset)} samples. "
                f"Target: {target_classes}, Exclude: {exclude_classes}"
            )

    def get_gradient_for_upload(
            self,
            all_seller_gradients: Dict[str, List[torch.Tensor]] = None,
            target_seller_id: str = None
    ) -> Tuple[Optional[List[torch.Tensor]], Dict[str, Any]]:
        self.round_counter += 1
        stats = {'attack_strategy': self.attack_cfg.attack_type}

        # --- Attack 1: Marketplace Denial-of-Service (DoS) ---
        if self.attack_cfg.attack_type == "dos":
            logging.info(f"[{self.seller_id}][DoS Attack] Submitting zero-gradient.")
            honest_gradient, stats = super().get_gradient_for_upload()
            if honest_gradient is None:
                return None, {}
            zero_gradient = [torch.zeros_like(p) for p in honest_gradient]
            return zero_gradient, stats

        # --- Attack 2: Economic Starvation ---
        elif self.attack_cfg.attack_type == "starvation":
            target_classes = self.attack_cfg.starvation_classes
            if not target_classes:
                logging.error("Starvation attack active but starvation_classes is empty!")
                return super().get_gradient_for_upload()

            logging.info(
                f"[{self.seller_id}][Starvation Attack] "
                f"Crafting biased gradient for classes: {target_classes}."
            )
            indices = [i for i, (_, label) in enumerate(self.dataset) if label in target_classes]
            if not indices:
                logging.warning(
                    f"No samples found for target classes {target_classes}. "
                    f"Returning zero-gradient."
                )
                honest_gradient, stats = super().get_gradient_for_upload()
                if honest_gradient is None:
                    return None, {}
                zero_gradient = [torch.zeros_like(p) for p in honest_gradient]
                return zero_gradient, stats

            biased_dataset = Subset(self.dataset, indices)
            local_model = self.model_factory().to(self.device)
            return self._compute_local_grad(local_model, biased_dataset)

        # --- Attack 3: Trust Erosion (Pivoting Objective) ---
        elif self.attack_cfg.attack_type == "erosion":
            logging.info(f"[{self.seller_id}][Trust Erosion Attack] Submitting random gradient.")
            honest_gradient, _ = super().get_gradient_for_upload()
            if honest_gradient is None:
                return None, {}
            random_gradient = [torch.randn_like(p) for p in honest_gradient]
            return random_gradient, stats

        # --- Attack 4: Class-Based Exclusion (IMPROVED VERSION) ---
        elif self.attack_cfg.attack_type == "class_exclusion":
            """
            A more realistic exclusion attack that systematically biases against sellers
            with data on specific classes by using a filtered baseline dataset.

            Unlike orthogonal_pivot which requires knowing a specific seller's gradient,
            this attack works by training the root gradient only on a subset of classes,
            causing similarity-based defenses to favor sellers with matching class distributions.
            """
            logging.info(
                f"[{self.seller_id}][Class Exclusion Attack] Round {self.round_counter}: "
                f"Computing biased root gradient from filtered dataset."
            )

            if self.exclusion_dataset is None:
                logging.error("Exclusion dataset not initialized! Falling back to honest behavior.")
                return super().get_gradient_for_upload()

            # Compute gradient using only the class-filtered dataset
            local_model = self.model_factory().to(self.device)
            biased_gradient, grad_stats = self._compute_local_grad(local_model, self.exclusion_dataset)

            # Add attack-specific stats
            stats.update(grad_stats)
            stats['filtered_dataset_size'] = len(self.exclusion_dataset)
            stats['original_dataset_size'] = len(self.dataset)
            stats['exclusion_ratio'] = len(self.exclusion_dataset) / len(self.dataset)

            # Optional: Apply additional scaling to amplify bias
            if hasattr(self.attack_cfg, 'exclusion_gradient_scale'):
                scale = self.attack_cfg.exclusion_gradient_scale
                biased_gradient = [g * scale for g in biased_gradient]
                stats['gradient_scale_applied'] = scale
                logging.info(f"[{self.seller_id}] Applied gradient scaling: {scale:.2f}")

            return biased_gradient, stats

        # --- Attack 5: Oscillating Objective (NEW - OPTIONAL) ---
        elif self.attack_cfg.attack_type == "oscillating":
            """
            Enhanced oscillating attack with multiple strategies:
            1. 'binary_flip': Alternates between two fixed class sets
            2. 'rotating': Cycles through multiple class subsets
            3. 'random_walk': Randomly selects class subset each round
            4. 'adversarial_drift': Gradually shifts focus to destabilize trust
            """
            strategy = self.attack_cfg.oscillation_strategy  # Default: "binary_flip"
            local_model = self.model_factory().to(self.device)
            if strategy == "binary_flip":
                # Strategy 1: Simple A/B oscillation
                oscillating_gradient, stats = self._oscillate_binary_flip(local_model, stats)

            elif strategy == "rotating":
                # Strategy 2: Cycle through multiple phases
                oscillating_gradient, stats = self._oscillate_rotating(local_model, stats)

            elif strategy == "random_walk":
                # Strategy 3: Random class selection each round
                oscillating_gradient, stats = self._oscillate_random_walk(local_model, stats)

            elif strategy == "adversarial_drift":
                # Strategy 4: Gradually shift to break trust
                oscillating_gradient, stats = self._oscillate_adversarial_drift(local_model, stats)

            else:
                logging.error(f"Unknown oscillation strategy: {strategy}. Using binary_flip.")
                oscillating_gradient, stats = self._oscillate_binary_flip(local_model, stats)

            return oscillating_gradient, stats

        # --- LEGACY: Keep orthogonal_pivot for comparison (mark as deprecated) ---
        elif self.attack_cfg.attack_type == "orthogonal_pivot":

            logging.info(
                f"[{self.seller_id}][Orthogonal Pivot Attack] "
                f"Crafting gradient to exclude '{target_seller_id}'."
            )
            if not all_seller_gradients or target_seller_id not in all_seller_gradients:
                logging.error(
                    f"Targeted exclusion failed: Target '{target_seller_id}' gradient not provided, current gradients: {all_seller_gradients.keys()}."
                )
                return super().get_gradient_for_upload()

            target_gradient = all_seller_gradients[target_seller_id]
            g_ideal, _ = super().get_gradient_for_upload()
            if g_ideal is None:
                return None, {}

            g_orth = self._calculate_orthogonal_pivot(g_ideal, target_gradient)
            if g_orth is None:
                return super().get_gradient_for_upload()

            pivoted_gradient = [p_ideal + p_orth for p_ideal, p_orth in zip(g_ideal, g_orth)]
            return pivoted_gradient, stats

        # Fallback to honest behavior
        logging.warning(f"[{self.seller_id}] No valid attack type matched. Behaving honestly.")
        return super().get_gradient_for_upload()

    def _oscillate_binary_flip(
            self,
            global_model: nn.Module,
            stats: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Strategy 1: Simple A/B flip between two class sets.
        Most basic but effective for destabilizing similarity-based defenses.
        """
        period = self.attack_cfg.oscillation_period
        phase = (self.round_counter // period) % 2

        if phase == 0:
            target_classes = self.attack_cfg.oscillation_classes_a
            phase_name = "A"
        else:
            target_classes = self.attack_cfg.oscillation_classes_b
            phase_name = "B"

        logging.info(
            f"[{self.seller_id}][Oscillating-BinaryFlip] Round {self.round_counter}, "
            f"Phase {phase_name} (period={period}): Classes {target_classes}"
        )

        gradient, grad_stats = self._compute_gradient_for_classes(
            global_model, target_classes
        )

        stats.update(grad_stats)
        stats['oscillation_strategy'] = 'binary_flip'
        stats['oscillation_phase'] = phase_name
        stats['target_classes'] = target_classes
        stats['period'] = period

        return gradient, stats

    def _oscillate_rotating(
            self,
            global_model: nn.Module,
            stats: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Strategy 2: Rotate through multiple class subsets (e.g., 4 phases).
        Creates more complex selection instability.
        """
        period = self.attack_cfg.oscillation_period
        class_subsets = self.attack_cfg.oscillation_class_subsets  # List of lists

        if not class_subsets:
            logging.error("Rotating strategy requires oscillation_class_subsets config!")
            return super().get_gradient_for_upload()

        num_phases = len(class_subsets)
        current_phase = (self.round_counter // period) % num_phases
        target_classes = class_subsets[current_phase]

        logging.info(
            f"[{self.seller_id}][Oscillating-Rotating] Round {self.round_counter}, "
            f"Phase {current_phase + 1}/{num_phases}: Classes {target_classes}"
        )

        gradient, grad_stats = self._compute_gradient_for_classes(
            global_model, target_classes
        )

        stats.update(grad_stats)
        stats['oscillation_strategy'] = 'rotating'
        stats['oscillation_phase'] = current_phase + 1
        stats['total_phases'] = num_phases
        stats['target_classes'] = target_classes

        return gradient, stats

    def _oscillate_random_walk(
            self,
            global_model: nn.Module,
            stats: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Strategy 3: Randomly select a subset of classes each round.
        Most unpredictable, hardest for defenses to adapt to.
        """
        num_classes = self.attack_cfg.num_classes  # Total classes in dataset
        subset_size = self.attack_cfg.oscillation_subset_size  # e.g., 3 classes

        # Randomly sample classes for this round
        all_classes = list(range(num_classes))
        target_classes = sorted(
            torch.randperm(num_classes)[:subset_size].tolist()
        )

        logging.info(
            f"[{self.seller_id}][Oscillating-RandomWalk] Round {self.round_counter}: "
            f"Randomly selected classes {target_classes}"
        )

        gradient, grad_stats = self._compute_gradient_for_classes(
            global_model, target_classes
        )

        stats.update(grad_stats)
        stats['oscillation_strategy'] = 'random_walk'
        stats['target_classes'] = target_classes
        stats['subset_size'] = subset_size

        return gradient, stats

    def _oscillate_adversarial_drift(
            self,
            global_model: nn.Module,
            stats: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Strategy 4: Gradually drift the class focus over time.
        Starts with full dataset, slowly narrows to subset, then expands again.
        Designed to first gain trust, then slowly poison the reference.
        """
        total_rounds = self.attack_cfg.oscillation_drift_total_rounds  # e.g., 50

        if self.round_counter <= total_rounds // 3:
            # Phase 1: Honest (rounds 1-16)
            target_classes = list(range(self.attack_cfg.num_classes))
            phase_name = "honest"

        elif self.round_counter <= 2 * total_rounds // 3:
            # Phase 2: Gradual narrowing (rounds 17-33)
            # Linearly reduce class set size
            progress = (self.round_counter - total_rounds // 3) / (total_rounds // 3)
            num_classes_to_keep = int((1 - progress * 0.7) * self.attack_cfg.num_classes)
            target_classes = list(range(max(2, num_classes_to_keep)))
            phase_name = "narrowing"

        else:
            # Phase 3: Narrow focus (rounds 34-50)
            target_classes = self.attack_cfg.oscillation_classes_a  # e.g., [0, 1, 2]
            phase_name = "poisoned"

        logging.info(
            f"[{self.seller_id}][Oscillating-AdversarialDrift] Round {self.round_counter}, "
            f"Phase '{phase_name}': Classes {target_classes}"
        )

        gradient, grad_stats = self._compute_gradient_for_classes(
            global_model, target_classes
        )

        stats.update(grad_stats)
        stats['oscillation_strategy'] = 'adversarial_drift'
        stats['oscillation_phase'] = phase_name
        stats['target_classes'] = target_classes

        return gradient, stats

    def _compute_gradient_for_classes(
            self,
            global_model: nn.Module,
            target_classes: List[int]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Helper: Computes gradient using only samples from specified classes.
        """
        indices = [
            i for i, (_, label) in enumerate(self.dataset)
            if label in target_classes
        ]

        if not indices:
            logging.warning(
                f"No samples found for classes {target_classes}. "
                f"Using random gradient as fallback."
            )
            honest_gradient, _ = super().get_gradient_for_upload()
            random_gradient = [torch.randn_like(p) for p in honest_gradient]
            return random_gradient, {'fallback': 'random', 'reason': 'no_samples'}

        filtered_dataset = Subset(self.dataset, indices)
        local_model = self.model_factory().to(self.device)
        local_model.load_state_dict(global_model.state_dict())

        gradient, grad_stats = self._compute_local_grad(local_model, filtered_dataset)
        grad_stats['filtered_samples'] = len(indices)
        grad_stats['original_dataset_size'] = len(self.dataset)

        return gradient, grad_stats

    def _calculate_orthogonal_pivot(
            self,
            base_gradient: List[torch.Tensor],
            target_gradient: List[torch.Tensor]
    ) -> Optional[List[torch.Tensor]]:
        """
        [LEGACY METHOD - KEPT FOR BACKWARD COMPATIBILITY]
        Computes a gradient-like noise vector that is orthogonal to the target_gradient.
        """
        g_orth_list = []
        try:
            g_noise = [torch.randn_like(p) for p in base_gradient]

            for p_noise, p_target in zip(g_noise, target_gradient):
                v_noise = p_noise.flatten()
                v_target = p_target.flatten()

                dot_target_target = torch.dot(v_target, v_target)
                if dot_target_target < 1e-9:
                    proj = torch.zeros_like(v_target)
                else:
                    proj = (torch.dot(v_target, v_noise) / dot_target_target) * v_target

                v_orth = v_noise - proj
                g_orth_list.append(v_orth.reshape(p_target.shape))

            return g_orth_list
        except Exception as e:
            logging.error(f"Failed to compute orthogonal pivot: {e}", exc_info=True)
            return None
