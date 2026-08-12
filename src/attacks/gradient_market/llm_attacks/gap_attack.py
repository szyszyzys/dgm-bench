"""
Gradient Assembly Poisoning (GAP) Attack
=========================================

Exploits the structural vulnerability of LoRA's decoupled A×B factorization.
Crafts matrices that individually appear benign to robust aggregators but
compose into a malicious weight update upon multiplication.

The attacker solves:
    min ||A_mal × B_mal - W_target||
    s.t. ||A_mal - μ_A|| < δ  and  ||B_mal - μ_B|| < δ

where W_target is the desired malicious weight shift, μ is the benign
distribution center, and δ is the defense's detection threshold.

This attack is specific to LoRA-based federated learning and is orthogonal
to all existing gradient-level attacks (which treat gradients as flat vectors).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GAPConfig:
    """Configuration for Gradient Assembly Poisoning attack."""
    target_label: int = 0
    poison_lr: float = 1e-3
    poison_epochs: int = 5           # Epochs to train the target malicious model
    projection_steps: int = 50       # Optimization steps for constrained factorization
    projection_lr: float = 0.01
    stealth_budget: float = 1.0      # δ: max L2 distance from benign mean
    scale_to_benign_norm: bool = True
    # Which LoRA layers to attack (None = all)
    target_layer_keywords: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


class GAPAttack:
    """
    Gradient Assembly Poisoning for LoRA-based federated LLM finetuning.

    The attack operates in three phases:
    1. Extract the target malicious weight shift W_target by training on poison data
    2. Estimate the benign distribution center from observed global updates
    3. Solve a constrained SVD factorization to produce stealthy A, B matrices

    Usage:
        gap = GAPAttack(config, model)
        poisoned_gradient = gap.craft_poisoned_gradient(
            honest_gradient, benign_mean_gradient, poison_dataset
        )
    """

    def __init__(self, config: GAPConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._benign_history: List[List[torch.Tensor]] = []

    def update_benign_estimate(self, global_gradient: List[torch.Tensor]):
        """
        Track the global aggregated gradient across rounds to estimate
        the benign distribution center μ.
        """
        self._benign_history.append([g.detach().clone() for g in global_gradient])
        # Keep last 5 rounds for running average
        if len(self._benign_history) > 5:
            self._benign_history.pop(0)

    def _get_benign_mean(self) -> Optional[List[torch.Tensor]]:
        """Compute running mean of observed global gradients."""
        if not self._benign_history:
            return None
        mean = []
        for i in range(len(self._benign_history[0])):
            stacked = torch.stack([h[i] for h in self._benign_history])
            mean.append(stacked.mean(dim=0))
        return mean

    def craft_poisoned_gradient(
        self,
        model: nn.Module,
        honest_gradient: List[torch.Tensor],
        poison_loader,
        benign_mean: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Main attack entry point.

        Args:
            model: The current global FederatedLoRA model
            honest_gradient: The gradient this seller would normally submit
            poison_loader: DataLoader with poisoned/backdoor data
            benign_mean: Estimated mean of benign seller gradients (optional)

        Returns:
            Poisoned gradient (List[torch.Tensor]) that looks benign individually
            but composes into a malicious update
        """
        if benign_mean is None:
            benign_mean = self._get_benign_mean()
        if benign_mean is None:
            # No history yet — fall back to honest gradient as reference
            benign_mean = honest_gradient

        # Phase 1: Extract target malicious weight shift
        w_target = self._compute_target_shift(model, poison_loader)

        # Phase 2: Identify LoRA parameter pairs (A, B) in the gradient
        lora_pairs, other_indices = self._identify_lora_pairs(model, honest_gradient)

        # Phase 3: For each LoRA pair, solve constrained factorization
        poisoned_gradient = [g.clone() for g in honest_gradient]

        for pair_info in lora_pairs:
            a_idx, b_idx = pair_info["a_idx"], pair_info["b_idx"]
            a_shape, b_shape = pair_info["a_shape"], pair_info["b_shape"]
            layer_name = pair_info["name"]

            # Extract the target W for this layer
            w_target_layer = self._extract_layer_target(
                w_target, layer_name, a_shape, b_shape
            )
            if w_target_layer is None:
                continue

            # Solve: find A_mal, B_mal such that A_mal × B_mal ≈ W_target
            # and ||A_mal - μ_A|| < δ, ||B_mal - μ_B|| < δ
            a_benign = benign_mean[a_idx]
            b_benign = benign_mean[b_idx]

            a_mal, b_mal = self._constrained_factorize(
                w_target_layer, a_benign, b_benign, a_shape, b_shape
            )

            poisoned_gradient[a_idx] = a_mal
            poisoned_gradient[b_idx] = b_mal
            logger.debug(f"GAP: Poisoned layer {layer_name}")

        # Optional: scale entire gradient to match benign norm
        if self.config.scale_to_benign_norm:
            poisoned_gradient = self._match_norm(poisoned_gradient, benign_mean)

        return poisoned_gradient

    def _compute_target_shift(self, model, poison_loader) -> Dict[str, torch.Tensor]:
        """
        Phase 1: Train on poison data to extract the ideal malicious weight shift.
        Returns dict mapping parameter names to their desired shifts.
        """
        import copy
        target_model = copy.deepcopy(model)
        target_model.train()

        trainable = [(n, p) for n, p in target_model.named_parameters() if p.requires_grad]
        initial_state = {n: p.data.clone() for n, p in trainable}

        optimizer = torch.optim.AdamW(
            [p for _, p in trainable], lr=self.config.poison_lr
        )

        for epoch in range(self.config.poison_epochs):
            for batch in poison_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels", input_ids.clone()).to(self.device)

                outputs = target_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute shift: final - initial (note: opposite sign from FedAvg pseudo-grad)
        w_target = {}
        for name, param in trainable:
            w_target[name] = param.data - initial_state[name]

        return w_target

    def _identify_lora_pairs(
        self, model, gradient: List[torch.Tensor]
    ) -> Tuple[List[Dict], List[int]]:
        """
        Identify pairs of (lora_A, lora_B) parameters in the gradient list.
        Returns list of pair info dicts and indices of non-LoRA parameters.
        """
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        pairs = []
        paired_indices = set()

        for i, name in enumerate(param_names):
            if "lora_A" in name:
                # Find matching lora_B
                b_name = name.replace("lora_A", "lora_B")
                for j, other_name in enumerate(param_names):
                    if other_name == b_name:
                        # Check if this layer should be attacked
                        should_attack = not self.config.target_layer_keywords or any(
                            kw in name for kw in self.config.target_layer_keywords
                        )
                        if should_attack:
                            pairs.append({
                                "name": name.split(".lora_A")[0],
                                "a_idx": i, "b_idx": j,
                                "a_shape": gradient[i].shape,
                                "b_shape": gradient[j].shape,
                            })
                            paired_indices.update({i, j})
                        break

        other_indices = [i for i in range(len(gradient)) if i not in paired_indices]
        logger.info(f"GAP: Found {len(pairs)} LoRA pairs to attack")
        return pairs, other_indices

    def _extract_layer_target(
        self, w_target: Dict, layer_name: str,
        a_shape: torch.Size, b_shape: torch.Size
    ) -> Optional[torch.Tensor]:
        """Extract and reshape W_target for a specific LoRA layer pair."""
        a_key = None
        b_key = None
        for key in w_target:
            if layer_name in key and "lora_A" in key:
                a_key = key
            elif layer_name in key and "lora_B" in key:
                b_key = key

        if a_key is None or b_key is None:
            return None

        # W_target for this layer = B_target × A_target (LoRA convention: W = B @ A)
        a_target = w_target[a_key]
        b_target = w_target[b_key]
        return b_target @ a_target  # (out_features, in_features)

    def _constrained_factorize(
        self,
        w_target: torch.Tensor,
        a_benign: torch.Tensor,
        b_benign: torch.Tensor,
        a_shape: torch.Size,
        b_shape: torch.Size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the constrained matrix factorization via projected gradient descent.

        Minimize: ||B_mal @ A_mal - W_target||²
        Subject to: ||A_mal - A_benign|| ≤ δ  and  ||B_mal - B_benign|| ≤ δ

        Uses SVD initialization + projected gradient descent.
        """
        rank = a_shape[0]  # LoRA rank = rows of A = cols of B
        delta = self.config.stealth_budget

        # Initialize via truncated SVD of W_target
        U, S, Vh = torch.linalg.svd(w_target.float(), full_matrices=False)
        # A_init = sqrt(S[:r]) @ Vh[:r, :]  (rank x in_features)
        # B_init = U[:, :r] @ sqrt(S[:r])   (out_features x rank)
        sqrt_s = torch.sqrt(S[:rank] + 1e-10)
        a_init = (sqrt_s.unsqueeze(1) * Vh[:rank, :]).to(a_benign.dtype)
        b_init = (U[:, :rank] * sqrt_s.unsqueeze(0)).to(b_benign.dtype)

        # Clamp to be within δ of benign
        a_mal = self._project_to_ball(a_init, a_benign, delta).clone().requires_grad_(True)
        b_mal = self._project_to_ball(b_init, b_benign, delta).clone().requires_grad_(True)

        optimizer = torch.optim.Adam([a_mal, b_mal], lr=self.config.projection_lr)

        for step in range(self.config.projection_steps):
            optimizer.zero_grad()
            reconstruction = b_mal @ a_mal
            loss = torch.nn.functional.mse_loss(reconstruction, w_target.float())
            loss.backward()
            optimizer.step()

            # Project back into feasible set after each step
            with torch.no_grad():
                a_mal.data = self._project_to_ball(a_mal.data, a_benign, delta)
                b_mal.data = self._project_to_ball(b_mal.data, b_benign, delta)

        return a_mal.detach().to(a_benign.dtype), b_mal.detach().to(b_benign.dtype)

    @staticmethod
    def _project_to_ball(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
        """Project x onto the L2 ball of given radius around center."""
        diff = x - center.float()
        norm = diff.norm()
        if norm > radius:
            diff = diff * (radius / norm)
        return (center.float() + diff).to(x.dtype)

    @staticmethod
    def _match_norm(gradient: List[torch.Tensor], reference: List[torch.Tensor]) -> List[torch.Tensor]:
        """Scale gradient to match the L2 norm of the reference."""
        grad_norm = sum(g.float().norm() ** 2 for g in gradient) ** 0.5
        ref_norm = sum(g.float().norm() ** 2 for g in reference) ** 0.5
        if grad_norm > 1e-10:
            scale = ref_norm / grad_norm
            return [g * scale for g in gradient]
        return gradient
