"""
RFLPA Aggregator — robustness component only (without secure aggregation crypto).

Reference:
    Mai, Yan & Pang. "RFLPA: A Robust Federated Learning Framework against
    Poisoning Attacks with Secure Aggregation." NeurIPS 2024.
    https://github.com/NusIoraPrivacy/RFLPA

Key difference from FLTrust:
    FLTrust requires a trusted root dataset on the server to compute a
    reference gradient.  RFLPA instead uses the *previous round's aggregated
    gradient* as the reference signal, making it fully self-bootstrapping —
    no trusted data needed.

Algorithm (per round):
    1. Clip each seller gradient element-wise to [-clip, clip].
    2. Normalize each seller gradient to have the same L2 norm as the
       reference (previous global update).
    3. Compute trust score  TS_i = max(0, cos_sim(g_bar_i, g_ref)).
    4. Aggregate:  g = sum(TS_i * g_bar_i) / sum(TS_i).
    5. Store the aggregated gradient as next round's reference.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F

from src.common_utils import clip_gradient_update, flatten_tensor
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class RFLPAAggregator(BaseAggregator):
    """
    Robust Federated Learning against Poisoning Attacks (RFLPA).

    Uses the previous round's aggregated gradient as a trust reference,
    then applies cosine-similarity trust scoring with norm normalization.
    """

    def __init__(self, *args, element_clip: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.element_clip = element_clip
        # Store previous round's aggregated gradient as reference
        self._prev_global_gradient: Optional[List[torch.Tensor]] = None

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"=== RFLPA Aggregation (Round {global_epoch}) ===")

        seller_ids = list(seller_updates.keys())
        if not seller_ids:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        # ------------------------------------------------------------------
        # Round 0 bootstrap: no previous gradient yet → plain FedAvg
        # ------------------------------------------------------------------
        if self._prev_global_gradient is None:
            logger.info("Round 0: no reference gradient yet — bootstrapping with FedAvg.")
            agg, sel, out, stats = self._fedavg(seller_ids, seller_updates)
            self._prev_global_gradient = [g.clone() for g in agg]
            stats["rflpa_bootstrap"] = True
            return agg, sel, out, stats

        # ------------------------------------------------------------------
        # 1. Reference signal = previous round's aggregated gradient
        # ------------------------------------------------------------------
        ref_gradient = self._prev_global_gradient
        flat_ref = flatten_tensor(ref_gradient)
        ref_norm = torch.norm(flat_ref)

        if ref_norm < 1e-9:
            logger.warning("Reference gradient norm ≈ 0 — falling back to FedAvg.")
            agg, sel, out, stats = self._fedavg(seller_ids, seller_updates)
            self._prev_global_gradient = [g.clone() for g in agg]
            stats["rflpa_fallback"] = "zero_ref"
            return agg, sel, out, stats

        # ------------------------------------------------------------------
        # 2. Element-wise clip + norm-normalize each seller update
        # ------------------------------------------------------------------
        processed_updates: Dict[str, List[torch.Tensor]] = {}

        for sid, upd in seller_updates.items():
            # a) Global L2 clip (reuse existing infrastructure)
            if self.clip_norm is not None and self.clip_norm > 0:
                clipped = clip_gradient_update(upd, self.clip_norm)
            else:
                clipped = upd

            # b) RFLPA element-wise clip: clamp every element to [-c, c]
            clipped = [torch.clamp(p, -self.element_clip, self.element_clip) for p in clipped]

            # c) Norm-normalize to match reference gradient norm
            flat = flatten_tensor(clipped)
            seller_norm = torch.norm(flat)
            if seller_norm > 1e-9:
                scaler = (ref_norm / seller_norm).item()
                processed_updates[sid] = [p * scaler for p in clipped]
            else:
                processed_updates[sid] = clipped

        # ------------------------------------------------------------------
        # 3. Compute cosine-similarity trust scores
        # ------------------------------------------------------------------
        flat_sellers = {sid: flatten_tensor(upd) for sid, upd in processed_updates.items()}
        seller_stack = torch.stack([flat_sellers[sid] for sid in seller_ids])

        cos_sim = F.cosine_similarity(
            seller_stack,
            flat_ref.unsqueeze(0),
            dim=1,
        )
        trust_scores = torch.relu(cos_sim)  # TS_i = max(0, cos_sim)

        # ------------------------------------------------------------------
        # 4. Weighted aggregation
        # ------------------------------------------------------------------
        total_trust = trust_scores.sum()
        if total_trust > 1e-9:
            weights = trust_scores / total_trust
        else:
            logger.warning("All trust scores ≈ 0 — using uniform weights.")
            weights = torch.ones_like(trust_scores) / len(trust_scores)

        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            w = weights[i].item()
            if w > 0:
                for agg_g, sell_g in zip(aggregated_gradient, processed_updates[sid]):
                    agg_g.add_(sell_g, alpha=w)

        # Store for next round's reference
        self._prev_global_gradient = [g.clone() for g in aggregated_gradient]

        # ------------------------------------------------------------------
        # 5. Stats & detection metrics
        # ------------------------------------------------------------------
        selected_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] == 0]

        known_adv = [sid for sid in seller_ids if "adv" in sid]
        detected_adv = [sid for sid in outlier_ids if "adv" in sid]
        false_pos = [sid for sid in outlier_ids if "bn" in sid]
        num_benign = len(seller_ids) - len(known_adv)

        aggregation_stats = {
            "aggregation_method": "rflpa",
            "round": global_epoch,
            "element_clip": self.element_clip,
            "ref_gradient_norm": ref_norm.item(),
            "trust_scores": {sid: ts.item() for sid, ts in zip(seller_ids, trust_scores)},
            "cosine_similarities": {sid: cs.item() for sid, cs in zip(seller_ids, cos_sim)},
            "seller_weights": {sid: w.item() for sid, w in zip(seller_ids, weights)},
            "avg_trust_score": trust_scores.mean().item(),
            "std_trust_score": trust_scores.std().item(),
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
            f"Trust scores — mean: {trust_scores.mean():.4f}, "
            f"selected: {len(selected_ids)}/{len(seller_ids)}, "
            f"ADR: {aggregation_stats['adversary_detection_rate']:.1%}"
        )

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats

    # ------------------------------------------------------------------
    def _fedavg(self, seller_ids, seller_updates):
        """Simple mean fallback for the bootstrap round."""
        n = len(seller_ids)
        agg = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for sid in seller_ids:
            for a, g in zip(agg, seller_updates[sid]):
                a.add_(g, alpha=1.0 / n)
        return agg, seller_ids, [], {"aggregation_method": "rflpa", "fallback": "fedavg"}
