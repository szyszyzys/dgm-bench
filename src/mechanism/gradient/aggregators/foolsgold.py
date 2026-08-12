import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class FoolsGoldAggregator(BaseAggregator):
    """
    FoolsGold: Mitigating Sybils in Federated Learning Poisoning (arXiv 2018).

    Defeats Sybil clusters by analyzing historical gradient similarity.
    Sellers that consistently submit similar gradients across rounds are
    penalized, as this indicates coordinated (Sybil) behavior.

    Mechanism:
      1. Maintain a running sum of each seller's historical gradients.
      2. Compute pairwise cosine similarity on L2-normalized histories.
      3. For each seller, find max similarity to any other seller.
      4. Penalize: w_i = 1 - max_sim_i. Pardon if max_sim < pardon_threshold.
      5. Normalize weights and compute weighted average of current updates.

    Args:
        pardon_threshold: Sellers with max similarity below this are pardoned
            (weight reset to 1.0). Default 0.5.
    """

    def __init__(self, *args, pardon_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pardon_threshold = pardon_threshold
        # Persistent history: seller_id -> accumulated gradient sum (flat tensor)
        self._history: Dict[str, torch.Tensor] = {}

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- FoolsGold Aggregation (Epoch {global_epoch}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(valid_sellers)

        # =====================================================================
        # Step 1: Update History Memory
        # =====================================================================
        for sid in valid_sellers:
            flat = torch.cat([p.view(-1).detach().cpu() for p in seller_updates[sid]])
            if sid in self._history:
                self._history[sid] = self._history[sid] + flat
            else:
                self._history[sid] = flat.clone()

        # =====================================================================
        # Step 2: Pairwise Cosine Similarity on Normalized Histories
        # =====================================================================
        # Normalize histories by L2 norm before computing similarity
        # (handles sellers that joined at different rounds)
        history_vectors = []
        for sid in valid_sellers:
            h = self._history[sid]
            h_norm = torch.norm(h, p=2)
            if h_norm > 1e-9:
                history_vectors.append(h / h_norm)
            else:
                history_vectors.append(h)

        history_stack = torch.stack(history_vectors)  # (N, D)

        # Cosine similarity matrix (already L2-normalized, so just dot product)
        cos_sim = torch.mm(history_stack, history_stack.t())  # (N, N)

        # =====================================================================
        # Step 3: Find Max Similarity for Each Seller
        # =====================================================================
        # Mask self-similarity
        mask = torch.ones_like(cos_sim) - torch.eye(num_sellers)
        masked_sim = cos_sim * mask - (1 - mask) * 1e9  # set diagonal to -inf
        max_sim, _ = torch.max(masked_sim, dim=1)  # (N,)

        # =====================================================================
        # Step 4: Compute Weights with Pardoning
        # =====================================================================
        weights = torch.clamp(1.0 - max_sim, min=0.0)  # w_i = 1 - max_sim_i

        # Pardon: if max similarity is low enough, the seller is clearly unique
        for i in range(num_sellers):
            if max_sim[i].item() < self.pardon_threshold:
                weights[i] = 1.0

        # =====================================================================
        # Step 5: Normalize and Aggregate
        # =====================================================================
        weight_sum = weights.sum()
        if weight_sum < 1e-9:
            # All sellers penalized to zero — fall back to uniform
            logger.warning("FoolsGold: All weights near zero. Falling back to uniform.")
            weights = torch.ones(num_sellers)
            weight_sum = weights.sum()

        norm_weights = weights / weight_sum  # (N,)

        # Weighted average of current-round updates
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(valid_sellers):
            w = norm_weights[i].item()
            for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                agg_grad.add_(upd_grad, alpha=w)

        # Classify outliers: sellers with weight < 0.1 * max_weight
        max_weight = norm_weights.max().item()
        threshold = 0.1 * max_weight if max_weight > 0 else 0.0
        selected_sids = [sid for i, sid in enumerate(valid_sellers) if norm_weights[i].item() >= threshold]
        outlier_sids = [sid for i, sid in enumerate(valid_sellers) if norm_weights[i].item() < threshold]

        aggregation_stats = {
            "weights": {sid: norm_weights[i].item() for i, sid in enumerate(valid_sellers)},
            "max_similarities": {sid: max_sim[i].item() for i, sid in enumerate(valid_sellers)},
            "num_pardoned": sum(1 for i in range(num_sellers) if max_sim[i].item() < self.pardon_threshold),
            "num_penalized": len(outlier_sids),
        }

        logger.info(
            f"FoolsGold: {len(selected_sids)} accepted, {len(outlier_sids)} penalized. "
            f"Pardoned: {aggregation_stats['num_pardoned']}/{num_sellers}."
        )

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
