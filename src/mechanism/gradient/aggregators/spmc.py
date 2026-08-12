"""
SPMC Aggregator — Server-side Margin Contribution Aggregation (MCAgg) only.

Reference:
    "SPMC: Self-Purifying Federated Backdoor Defense via Margin Contribution."
    ICML 2025.

This implements the MCAgg component (server-side robust aggregation).
The client-side self-purification component (knowledge distillation) is
orthogonal to the aggregation rule and is not included here.

Algorithm (per round):
    1. For each seller i, compute the "leave-one-out coalition" average
       (mean of all other sellers' updates).
    2. Margin score_i = cos_sim(update_i, coalition_avg_i).
    3. Purify: score_i = max(0, score_i)   [ReLU cutoff].
    4. Weights = softmax(scores / temperature).
    5. Aggregate: g = sum(w_i * update_i).
"""

import logging
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F

from src.common_utils import flatten_tensor, clip_gradient_update
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class SPMCAggregator(BaseAggregator):
    """
    Server-side Margin Contribution Aggregation from SPMC.

    Uses leave-one-out cosine similarity as each seller's margin score,
    applies ReLU to reject divergent sellers, then softmax-weights the
    remaining contributions.
    """

    def __init__(self, *args, temperature: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if temperature <= 0:
            raise ValueError(f"SPMC temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"=== SPMC MCAgg Aggregation (Round {global_epoch}) ===")

        seller_ids = list(seller_updates.keys())
        if not seller_ids:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(seller_ids)

        # ------------------------------------------------------------------
        # Single seller: nothing to compare against → return as-is
        # ------------------------------------------------------------------
        if num_sellers == 1:
            sid = seller_ids[0]
            return (
                seller_updates[sid],
                seller_ids,
                [],
                {"aggregation_method": "spmc", "fallback": "single_seller"},
            )

        # ------------------------------------------------------------------
        # 0. Clip updates to prevent scaling attacks (cosine sim is scale-invariant)
        # ------------------------------------------------------------------
        processed_updates = {}
        for sid, upd in seller_updates.items():
            if self.clip_norm is not None and self.clip_norm > 0:
                processed_updates[sid] = clip_gradient_update(upd, self.clip_norm)
            else:
                processed_updates[sid] = upd

        # ------------------------------------------------------------------
        # 1. Flatten all seller updates and stack
        # ------------------------------------------------------------------
        flat_updates = {sid: flatten_tensor(processed_updates[sid]) for sid in seller_ids}
        stacked = torch.stack([flat_updates[sid] for sid in seller_ids])  # (N, D)
        total_sum = stacked.sum(dim=0)  # (D,)

        # ------------------------------------------------------------------
        # 2. Leave-one-out cosine similarity (margin scores)
        # ------------------------------------------------------------------
        margin_scores = torch.zeros(num_sellers, device=stacked.device)

        for i in range(num_sellers):
            coalition_sum = total_sum - stacked[i]
            coalition_avg = coalition_sum / (num_sellers - 1)
            margin_scores[i] = F.cosine_similarity(
                stacked[i].unsqueeze(0),
                coalition_avg.unsqueeze(0),
            ).squeeze()

        # ------------------------------------------------------------------
        # 3. ReLU purification: zero-out divergent sellers
        #    Guard against NaN from near-zero coalition averages
        # ------------------------------------------------------------------
        if torch.isnan(margin_scores).any():
            logger.warning("NaN detected in SPMC margin scores. Replacing with 0.0.")
            margin_scores = torch.nan_to_num(margin_scores, nan=0.0)
        purified_scores = torch.relu(margin_scores)

        # ------------------------------------------------------------------
        # 4. Softmax weighting with temperature
        #    Mask out zero-scored (rejected) sellers with -inf so softmax
        #    gives them exactly zero weight, not exp(0/T).
        # ------------------------------------------------------------------
        masked_scores = purified_scores.clone()
        masked_scores[purified_scores == 0] = float('-inf')
        weights = F.softmax(masked_scores / self.temperature, dim=0)
        # If ALL sellers are rejected, softmax produces NaN → fall back to uniform
        if torch.isnan(weights).any():
            logger.warning("All sellers rejected by SPMC. Using uniform weights as fallback.")
            weights = torch.ones(num_sellers, device=stacked.device) / num_sellers

        # ------------------------------------------------------------------
        # 5. Weighted aggregation (per-layer)
        # ------------------------------------------------------------------
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            w = weights[i].item()
            if w > 0:
                for agg_g, sell_g in zip(aggregated_gradient, processed_updates[sid]):
                    agg_g.add_(sell_g, alpha=w)

        # ------------------------------------------------------------------
        # 6. Stats & detection metrics
        # ------------------------------------------------------------------
        selected_ids = [sid for i, sid in enumerate(seller_ids) if purified_scores[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if purified_scores[i] == 0]

        known_adv = [sid for sid in seller_ids if "adv" in sid]
        detected_adv = [sid for sid in outlier_ids if "adv" in sid]
        false_pos = [sid for sid in outlier_ids if "bn" in sid]
        num_benign = len(seller_ids) - len(known_adv)

        aggregation_stats = {
            "aggregation_method": "spmc",
            "round": global_epoch,
            "temperature": self.temperature,
            "margin_scores": {sid: ms.item() for sid, ms in zip(seller_ids, margin_scores)},
            "purified_scores": {sid: ps.item() for sid, ps in zip(seller_ids, purified_scores)},
            "seller_weights": {sid: w.item() for sid, w in zip(seller_ids, weights)},
            "avg_margin_score": margin_scores.mean().item(),
            "std_margin_score": margin_scores.std().item(),
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),
            "adversary_detection_rate": (
                len(detected_adv) / len(known_adv) if known_adv else 0.0
            ),
            "false_positive_rate": (
                len(false_pos) / num_benign if num_benign > 0 else 0.0
            ),
        }

        logger.info(
            f"Margin scores — mean: {margin_scores.mean():.4f}, "
            f"selected: {len(selected_ids)}/{num_sellers}, "
            f"ADR: {aggregation_stats['adversary_detection_rate']:.1%}"
        )

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats
