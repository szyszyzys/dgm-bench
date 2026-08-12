from typing import Dict, Tuple, List, Any

import torch

from src.mechanism.gradient.aggregators.fltrust import FLTrustAggregator
from src.mechanism.gradient.aggregators.base_aggregator import logger


class FLTrustThresholdAggregator(FLTrustAggregator):
    """Thresholded FLTrust — selection-integrity baseline (E1b).

    Runs FLTrust's trust-score computation UNCHANGED (via the parent's
    _compute_trust_scores), then converts the soft trust weighting into an
    explicit accept/reject decision:

        accept seller i (R_i = 1)  iff  trust_score_i >= tau

    Accepted sellers are aggregated with FLTrust's trust-proportional weights
    renormalized over the accepted set; rejected sellers contribute nothing
    and are unpaid. This makes selection integrity an explicit objective so
    the MSR-vs-Acc frontier can be swept over tau.
    """

    requires_root_gradient = True

    def __init__(self, *args, tau: float = 0.5, clip_norm=None, **kwargs):
        super().__init__(*args, clip_norm=clip_norm, **kwargs)
        if not (0.0 <= tau <= 1.0):
            raise ValueError(
                f"fltrust_threshold.tau must be in [0, 1] (trust scores are "
                f"ReLU'd cosine similarities); got {tau}."
            )
        self.tau = tau

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        logger.info(f"=== FLTrust-Threshold Aggregation (Round {global_epoch}, tau={self.tau}) ===")
        logger.info(f"Processing {len(seller_updates)} seller updates")

        (processed_updates, _flat_trust_gradient, trust_norm, seller_ids,
         cos_sim, trust_scores, clip_enabled) = self._compute_trust_scores(
            seller_updates, root_gradient
        )

        if not seller_ids:
            logger.warning("No valid seller updates to aggregate.")
            return [torch.zeros_like(p) for p in self.global_model.parameters()], [], [], {}

        # --- Explicit accept/reject decision: R_i = 1 iff trust_i >= tau ---
        accept_mask = trust_scores >= self.tau
        selected_ids = [sid for i, sid in enumerate(seller_ids) if accept_mask[i]]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if not accept_mask[i]]

        logger.info(f"Trust scores - Mean: {trust_scores.mean().item():.4f}, "
                    f"Max: {trust_scores.max().item():.4f}; "
                    f"accepted {len(selected_ids)}/{len(seller_ids)} at tau={self.tau}")

        # --- Weights: FLTrust's trust-proportional weighting, renormalized
        # over the accepted set only. Rejected sellers get weight 0.
        masked_trust = trust_scores * accept_mask.to(trust_scores.dtype)
        total_trust = masked_trust.sum()
        if total_trust > 1e-9:
            weights = masked_trust / total_trust
        else:
            logger.warning(f"No seller passed tau={self.tau}; returning zero update.")
            weights = torch.zeros_like(trust_scores)

        aggregation_stats = {
            'aggregation_method': 'fltrust_threshold',
            'round': global_epoch,
            'tau': self.tau,
            'clip_enabled': clip_enabled,
            'clip_norm': self.clip_norm if clip_enabled else None,
            'trust_gradient_norm': trust_norm.item(),
            'avg_trust_score': trust_scores.mean().item(),
            'std_trust_score': trust_scores.std().item(),
            'min_trust_score': trust_scores.min().item(),
            'max_trust_score': trust_scores.max().item(),
            'median_trust_score': trust_scores.median().item(),
            'trust_scores': {sid: score.item() for sid, score in zip(seller_ids, trust_scores)},
            'cosine_similarities': {sid: sim.item() for sid, sim in zip(seller_ids, cos_sim)},
            'seller_weights': {sid: w.item() for sid, w in zip(seller_ids, weights)},
            'total_trust_sum': total_trust.item(),
        }

        # --- Weighted aggregation over accepted sellers only ---
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            weight = weights[i].item()
            if weight > 0:
                for agg_grad, seller_grad in zip(aggregated_gradient, processed_updates[sid]):
                    agg_grad.add_(seller_grad, alpha=weight)

        # --- Detection stats (same conventions as FLTrust) ---
        aggregation_stats.update({
            'num_sellers': len(seller_ids),
            'num_selected': len(selected_ids),
            'num_outliers': len(outlier_ids),
            'selection_rate': len(selected_ids) / len(seller_ids) if seller_ids else 0,
            'outlier_rate': len(outlier_ids) / len(seller_ids) if seller_ids else 0,
            'known_adversaries': [sid for sid in seller_ids if 'adv' in sid],
            'detected_adversaries': [sid for sid in outlier_ids if 'adv' in sid],
            'missed_adversaries': [sid for sid in selected_ids if 'adv' in sid],
            'false_positives': [sid for sid in outlier_ids if 'bn' in sid],
        })

        num_known_adv = len(aggregation_stats['known_adversaries'])
        num_detected_adv = len(aggregation_stats['detected_adversaries'])
        num_benign = len(seller_ids) - num_known_adv
        num_false_pos = len(aggregation_stats['false_positives'])

        aggregation_stats['adversary_detection_rate'] = (
            num_detected_adv / num_known_adv if num_known_adv > 0 else 0
        )
        aggregation_stats['false_positive_rate'] = (
            num_false_pos / num_benign if num_benign > 0 else 0
        )

        logger.info(f"Adversary detection: {num_detected_adv}/{num_known_adv}; "
                    f"false positives: {num_false_pos}/{num_benign}")

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats
