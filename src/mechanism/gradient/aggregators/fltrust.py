from typing import Dict, Tuple, List, Any

import torch
import torch.nn.functional as F

from src.common_utils import clip_gradient_update, flatten_tensor
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator, logger


class FLTrustAggregator(BaseAggregator):
    requires_root_gradient = True

    def __init__(self, *args, clip_norm=None, **kwargs):
        super().__init__(*args, clip_norm=clip_norm, **kwargs)

    def _compute_trust_scores(
            self,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
    ):
        """FLTrust steps 1-3: clip + rescale updates, then per-seller trust scores.

        Returns (processed_updates, flat_trust_gradient, trust_norm, seller_ids,
        cos_sim, trust_scores, clip_enabled); seller_ids is empty when no valid
        update survives. Extracted unchanged from aggregate() so the
        fltrust_threshold variant (E1b) can reuse the identical computation.
        """
        # --- 1. Process Root Gradient (Clip & Get Norm) ---
        clip_enabled = self.clip_norm is not None and self.clip_norm > 0

        if clip_enabled:
            trust_gradient = clip_gradient_update(root_gradient, self.clip_norm)
        else:
            trust_gradient = root_gradient

        # Flatten root gradient to compute norm and similarity
        flat_trust_gradient = flatten_tensor(trust_gradient)
        trust_norm = torch.norm(flat_trust_gradient)

        # Safety: Prevent division errors if root gradient is exactly zero
        if trust_norm < 1e-9:
            trust_norm = torch.tensor(1.0, device=self.device)

        logger.info(f"Trust gradient norm: {trust_norm.item():.4f}")

        # --- 2. Process Seller Updates (Clip & Rescale) ---
        processed_updates = {}

        for sid, upd in seller_updates.items():
            # A. Clip (if enabled)
            if clip_enabled:
                clipped_upd = clip_gradient_update(upd, self.clip_norm)
            else:
                clipped_upd = upd

            # B. FLTrust Normalization Step
            # Rescale the seller update so its L2 norm equals the root gradient's L2 norm.
            # This neutralizes "Model Replacement" attacks (huge norms)
            # and prevents "IID Explosion" (double scaling).
            flat_upd = flatten_tensor(clipped_upd)
            seller_norm = torch.norm(flat_upd)

            if seller_norm > 1e-9:
                scaler = trust_norm / seller_norm
                # Apply scaler to every tensor in the list
                processed_updates[sid] = [p * scaler for p in clipped_upd]
            else:
                # If seller gradient is 0, keep it 0
                processed_updates[sid] = clipped_upd

        # --- 3. Compute Trust Scores ---
        # Flatten the processed (rescaled) updates for cosine similarity
        flat_seller_updates = {sid: flatten_tensor(upd) for sid, upd in processed_updates.items()}
        seller_ids = list(flat_seller_updates.keys())

        if not seller_ids:
            return processed_updates, flat_trust_gradient, trust_norm, [], None, None, clip_enabled

        # Stack vectors for batch processing
        seller_updates_stack = torch.stack(list(flat_seller_updates.values()))

        # Compute Cosine Similarity
        cos_sim = F.cosine_similarity(
            seller_updates_stack,
            flat_trust_gradient.unsqueeze(0),
            dim=1
        )

        # ReLU: Drop negative trust scores to zero
        trust_scores = torch.relu(cos_sim)

        return (processed_updates, flat_trust_gradient, trust_norm, seller_ids,
                cos_sim, trust_scores, clip_enabled)

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        """
        FLTrust aggregation with magnitude normalization.

        Fix: Scales client gradients to match the root gradient norm BEFORE aggregation.
        This prevents IID gradient explosion.
        """
        logger.info(f"=== FLTrust Aggregation (Round {global_epoch}) ===")
        logger.info(f"Processing {len(seller_updates)} seller updates")

        (processed_updates, flat_trust_gradient, trust_norm, seller_ids,
         cos_sim, trust_scores, clip_enabled) = self._compute_trust_scores(
            seller_updates, root_gradient
        )

        if not seller_ids:
            logger.warning("No valid seller updates to aggregate.")
            return [torch.zeros_like(p) for p in self.global_model.parameters()], [], [], {}

        # --- 4. Logging & Stats ---
        logger.info(f"Trust scores - Mean: {trust_scores.mean().item():.4f}, "
                    f"Std: {trust_scores.std().item():.4f}, "
                    f"Min: {trust_scores.min().item():.4f}, "
                    f"Max: {trust_scores.max().item():.4f}")

        aggregation_stats = {
            'aggregation_method': 'fltrust',
            'round': global_epoch,
            'clip_enabled': clip_enabled,
            'clip_norm': self.clip_norm if clip_enabled else None,
            'trust_gradient_norm': trust_norm.item(),
            'avg_trust_score': trust_scores.mean().item(),
            'std_trust_score': trust_scores.std().item(),
            'min_trust_score': trust_scores.min().item(),
            'max_trust_score': trust_scores.max().item(),
            'median_trust_score': trust_scores.median().item(),
            'trust_scores': {sid: score.item() for sid, score in zip(seller_ids, trust_scores)},
            'cosine_similarities': {sid: sim.item() for sid, sim in zip(seller_ids, cos_sim)}
        }

        # --- 5. Calculate Weights ---
        total_trust = trust_scores.sum()
        if total_trust > 1e-9:
            weights = trust_scores / total_trust
        else:
            logger.warning("Total trust score near zero! Using uniform weights.")
            weights = torch.ones_like(trust_scores) / len(trust_scores)

        aggregation_stats['seller_weights'] = {sid: w.item() for sid, w in zip(seller_ids, weights)}
        aggregation_stats['total_trust_sum'] = total_trust.item()

        # --- 6. Weighted Aggregation ---
        # Note: We aggregate the 'processed_updates' which are already rescaled to the correct norm.
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]

        for i, sid in enumerate(seller_ids):
            weight = weights[i].item()
            if weight > 0:
                for agg_grad, seller_grad in zip(aggregated_gradient, processed_updates[sid]):
                    agg_grad.add_(seller_grad, alpha=weight)
                # logger.debug(f"Seller {sid}: trust={trust_scores[i].item():.4f}, weight={weight:.4f}")

        # --- 7. Metrics & Detection Stats ---
        selected_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if weights[i] == 0]

        if outlier_ids:
            logger.info(f"Outlier IDs (Trust=0): {outlier_ids}")

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

        logger.info(f"Adversary detection: {num_detected_adv}/{num_known_adv} "
                    f"({aggregation_stats['adversary_detection_rate']:.1%})")
        logger.info(f"False positives: {num_false_pos}/{num_benign} "
                    f"({aggregation_stats['false_positive_rate']:.1%})")

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats