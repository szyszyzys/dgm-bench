"""
DAVED + FWIF Aggregator — Software simulation of hardware-economic filtering.

Algorithm (per round):
    1. Clip seller gradients (standard clip_norm).
    2. Compute empirical diagonal Fisher from root gradient's classification head.
    3. Magnitude Gate: reject sellers whose gradient norm (relative to market
       median) falls below a dynamic threshold.
    4. Angle Gate: compute Fisher-Weighted Inner Product (FWIF) between each
       seller's head gradient and the root head gradient.  If FWIF < 0
       (angle > 90 deg), project ONLY the head component and re-score.
    5. Aggregate surviving gradients with weights proportional to FWIF scores.
"""

import logging
from typing import Dict, List, Tuple, Any

import torch

from src.common_utils import clip_gradient_update, flatten_tensor
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class DAVEDAggregator(BaseAggregator):
    """
    DAVED + FWIF Aggregation.

    Extracts the classification head to compute Fisher-Weighted Inner Products.
    Projects the head if FWIF < 0, preserving the backbone.
    Rejects updates that fail a relative-median L2 norm gate.
    Aggregates with continuous FWIF-proportional weights.
    """
    requires_root_gradient = True

    def __init__(self, *args,
                 head_layer_indices: List[int] = None,
                 l2_relative_alpha: float = 0.1,
                 proj_lambda: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.head_indices = head_layer_indices if head_layer_indices is not None else [-2, -1]
        self.l2_relative_alpha = l2_relative_alpha
        self.proj_lambda = proj_lambda

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_head(self, gradient: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return the classification-head parameter tensors."""
        n = len(gradient)
        return [gradient[i % n] for i in self.head_indices]

    def _is_head_index(self, idx: int, n_params: int) -> bool:
        """Check whether parameter index *idx* belongs to the head."""
        return idx in self.head_indices or (idx - n_params) in self.head_indices

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"=== DAVED+FWIF Aggregation (Round {global_epoch}) ===")
        seller_ids = list(seller_updates.keys())

        if not seller_ids or root_gradient is None:
            logger.warning("No seller updates or missing root gradient.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        n_params = len(root_gradient)
        clip_enabled = self.clip_norm is not None and self.clip_norm > 0

        # ==============================================================
        # 0.  Clip root gradient (same as every other aggregator)
        # ==============================================================
        if clip_enabled:
            root_gradient = clip_gradient_update(root_gradient, self.clip_norm)

        # ==============================================================
        # 1.  Empirical diagonal Fisher on the head
        # ==============================================================
        root_head_flat = flatten_tensor(self._extract_head(root_gradient))
        F_diag = root_head_flat ** 2
        # Normalise so that no single parameter dominates via cubic scaling
        F_diag = F_diag / (F_diag.sum() + 1e-12)

        # Denominator for projection: <g_root_head, g_root_head>_F
        root_fwif = torch.sum(root_head_flat * F_diag * root_head_flat)

        # ==============================================================
        # 2.  Clip seller gradients + compute norms for magnitude gate
        # ==============================================================
        clipped_updates: Dict[str, List[torch.Tensor]] = {}
        raw_norms: Dict[str, float] = {}

        for sid, upd in seller_updates.items():
            if clip_enabled:
                upd = clip_gradient_update(upd, self.clip_norm)
            clipped_updates[sid] = upd
            raw_norms[sid] = torch.norm(flatten_tensor(upd)).item()

        # Dynamic threshold: reject if norm < alpha * median(norms)
        norms_tensor = torch.tensor(list(raw_norms.values()))
        median_norm = torch.median(norms_tensor).item() if len(norms_tensor) > 0 else 0.0
        lazy_threshold = self.l2_relative_alpha * median_norm

        # ==============================================================
        # 3.  Per-seller gates
        # ==============================================================
        processed_updates: Dict[str, List[torch.Tensor]] = {}
        fwif_scores: Dict[str, float] = {}
        cos_similarities: Dict[str, float] = {}
        slashed_sids: List[str] = []
        projected_sids: List[str] = []

        for sid in seller_ids:
            G_full = clipped_updates[sid]

            # --- Magnitude gate ---
            if raw_norms[sid] < lazy_threshold:
                logger.debug(f"[{sid}] slashed (lazy): norm {raw_norms[sid]:.4f} < {lazy_threshold:.4f}")
                slashed_sids.append(sid)
                fwif_scores[sid] = 0.0
                cos_similarities[sid] = 0.0
                continue

            # --- Angle gate (FWIF on head) ---
            g_d_head_flat = flatten_tensor(self._extract_head(G_full))

            cross_fwif = torch.sum(g_d_head_flat * F_diag * root_head_flat)

            # Also compute standard full-gradient cosine similarity for logging
            flat_full = flatten_tensor(G_full)
            flat_root = flatten_tensor(root_gradient)
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_full.unsqueeze(0), flat_root.unsqueeze(0)
            ).item()
            cos_similarities[sid] = cos_sim

            if cross_fwif < 0:
                # --- Head-only projection ---
                proj_scalar = self.proj_lambda * (cross_fwif / (root_fwif + 1e-9))

                G_projected = []
                for idx, (p_d, p_g) in enumerate(zip(G_full, root_gradient)):
                    if self._is_head_index(idx, n_params):
                        G_projected.append(p_d - proj_scalar * p_g)
                    else:
                        G_projected.append(p_d)

                processed_updates[sid] = G_projected
                projected_sids.append(sid)

                # Re-score the corrected head so projected sellers still contribute
                g_corrected_head = flatten_tensor(self._extract_head(G_projected))
                re_score = torch.sum(g_corrected_head * F_diag * root_head_flat)
                fwif_scores[sid] = max(0.0, re_score.item())
            else:
                # Naturally aligned
                processed_updates[sid] = G_full
                fwif_scores[sid] = cross_fwif.item()

        # ==============================================================
        # 4.  Weighted aggregation (FWIF-proportional)
        # ==============================================================
        surviving_sids = list(processed_updates.keys())
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]

        total_fwif = sum(fwif_scores[sid] for sid in surviving_sids)
        weights: Dict[str, float] = {}

        if total_fwif > 1e-9:
            for sid in surviving_sids:
                w = fwif_scores[sid] / total_fwif
                weights[sid] = w
                if w > 0:
                    for agg_g, sell_g in zip(aggregated_gradient, processed_updates[sid]):
                        agg_g.add_(sell_g, alpha=w)
        else:
            # All survivors were projected and scored 0 — uniform fallback
            logger.warning("All FWIF scores ≈ 0 after projection — uniform fallback.")
            n_surv = max(1, len(surviving_sids))
            for sid in surviving_sids:
                weights[sid] = 1.0 / n_surv
                for agg_g, sell_g in zip(aggregated_gradient, processed_updates[sid]):
                    agg_g.add_(sell_g, alpha=1.0 / n_surv)

        # Also give slashed sellers 0 weight for completeness
        for sid in slashed_sids:
            weights[sid] = 0.0

        # ==============================================================
        # 5.  Detection metrics (standard framework fields)
        # ==============================================================
        selected_ids = surviving_sids
        outlier_ids = slashed_sids

        known_adv = [sid for sid in seller_ids if "adv" in sid]
        detected_adv = [sid for sid in outlier_ids if "adv" in sid]
        false_pos = [sid for sid in outlier_ids if "bn" in sid]
        num_benign = len(seller_ids) - len(known_adv)

        aggregation_stats = {
            "aggregation_method": "daved",
            "round": global_epoch,
            "clip_norm": self.clip_norm if clip_enabled else None,
            "lazy_threshold": lazy_threshold,
            "median_norm": median_norm,
            "fwif_scores": {sid: fwif_scores.get(sid, 0.0) for sid in seller_ids},
            "cosine_similarities": cos_similarities,
            "seller_weights": weights,
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),
            "num_projected": len(projected_sids),
            "projected_sellers": projected_sids,
            "slashed_sellers": slashed_sids,
            "adversary_detection_rate": (
                len(detected_adv) / len(known_adv) if known_adv else 0.0
            ),
            "false_positive_rate": (
                len(false_pos) / num_benign if num_benign > 0 else 0.0
            ),
        }

        logger.info(
            f"DAVED — selected: {len(selected_ids)}/{len(seller_ids)}, "
            f"projected: {len(projected_sids)}, slashed: {len(slashed_sids)}, "
            f"ADR: {aggregation_stats['adversary_detection_rate']:.1%}"
        )

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats
