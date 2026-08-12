import logging
from typing import Dict, List, Tuple, Any

import torch

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class MultiKrumAggregator(BaseAggregator):
    """
    Multi-Krum aggregation (Blanchard et al., 2017).

    For each seller, computes a "Krum score" = sum of squared distances to
    its (n - f - 2) nearest neighbors, where f is the assumed number of
    Byzantine sellers.  Selects the `m_selected` sellers with the lowest
    scores and averages their updates.

    When m_selected == 1 this reduces to standard Krum.
    Requires: num_sellers >= 2 * num_byzantine + 3.
    """

    def __init__(self, *args, num_byzantine: int = 0, m_selected: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_byzantine = num_byzantine
        self.m_selected = m_selected

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- MultiKrum Aggregation (Epoch {global_epoch}, f={self.num_byzantine}, "
                    f"m={self.m_selected}) ---")

        valid_sellers = list(seller_updates.keys())
        num_sellers = len(valid_sellers)

        if not valid_sellers:
            logger.warning("No valid seller updates received for MultiKrum aggregation.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        # --- Validate parameters against actual seller count ---
        num_byzantine = self.num_byzantine
        m_selected = self.m_selected

        num_neighbors = num_sellers - num_byzantine - 2
        if num_neighbors <= 0:
            logger.warning(
                f"Too few sellers ({num_sellers}) for Krum with f={num_byzantine}. "
                f"Need >= {2 * num_byzantine + 3}. Falling back to FedAvg."
            )
            return self._fallback_fedavg(valid_sellers, seller_updates, num_sellers)

        # Clamp m_selected to valid range
        max_m = num_sellers - num_byzantine
        if m_selected > max_m:
            logger.warning(f"m_selected={m_selected} > {max_m}. Clamping to {max_m}.")
            m_selected = max_m

        # --- 1. Pairwise squared Euclidean distances (layer-by-layer to save memory) ---
        num_layers = len(seller_updates[valid_sellers[0]])
        sq_distances = torch.zeros((num_sellers, num_sellers), device=self.device)
        for layer_idx in range(num_layers):
            layer_tensors = torch.stack([seller_updates[sid][layer_idx].view(-1) for sid in valid_sellers])
            layer_dists = torch.cdist(layer_tensors, layer_tensors, p=2.0)
            sq_distances.add_(layer_dists ** 2)
        distances = torch.sqrt(sq_distances)

        # --- 2. Krum scores: sum of squared distances to nearest neighbors ---
        sorted_distances, _ = torch.sort(distances, dim=1)
        # Skip index 0 (self-distance = 0), square per original Blanchard et al.
        krum_scores = torch.sum(sorted_distances[:, 1:num_neighbors + 1] ** 2, dim=1)

        # --- 4. Select top-m sellers with lowest scores ---
        _, top_indices = torch.topk(krum_scores, m_selected, largest=False)
        selected_indices = top_indices.tolist()

        selected_sids = [valid_sellers[i] for i in selected_indices]
        outlier_sids = [sid for sid in valid_sellers if sid not in selected_sids]

        # --- 5. Average the selected sellers' updates (per-layer) ---
        num_params = len(seller_updates[valid_sellers[0]])
        aggregated_gradient = []
        for param_idx in range(num_params):
            param_sum = torch.zeros_like(seller_updates[valid_sellers[0]][param_idx])
            for sid in selected_sids:
                param_sum.add_(seller_updates[sid][param_idx])
            aggregated_gradient.append(param_sum / m_selected)

        aggregation_stats = {
            "num_byzantine": num_byzantine,
            "m_selected": m_selected,
            "krum_scores": {valid_sellers[i]: krum_scores[i].item() for i in range(num_sellers)},
            "selected_sellers": selected_sids,
        }

        logger.info(f"Selected {m_selected}/{num_sellers} sellers: {selected_sids}")

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats

    def _fallback_fedavg(self, valid_sellers, seller_updates, num_sellers):
        """Simple mean fallback when Krum preconditions aren't met."""
        num_params = len(seller_updates[valid_sellers[0]])
        aggregated_gradient = []
        for param_idx in range(num_params):
            param_sum = torch.zeros_like(seller_updates[valid_sellers[0]][param_idx])
            for sid in valid_sellers:
                param_sum.add_(seller_updates[sid][param_idx])
            aggregated_gradient.append(param_sum / num_sellers)
        return aggregated_gradient, valid_sellers, [], {"fallback": "fedavg"}
