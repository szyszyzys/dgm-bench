import logging
from typing import Dict, List, Tuple, Any

import torch

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class TrimmedMeanAggregator(BaseAggregator):
    """
    Coordinate-wise Trimmed Mean aggregation.

    For each parameter coordinate, sorts the values across all sellers,
    trims the top and bottom `trim_ratio` fraction, and averages the rest.
    This is robust to up to `trim_ratio` fraction of Byzantine sellers.
    """

    def __init__(self, *args, trim_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_ratio = trim_ratio

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- TrimmedMean Aggregation (Epoch {global_epoch}, trim={self.trim_ratio}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received for TrimmedMean aggregation.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(valid_sellers)
        trim_count = int(num_sellers * self.trim_ratio)

        # Guard: need at least 1 seller remaining after trimming both ends
        if num_sellers - 2 * trim_count < 1:
            logger.warning(
                f"trim_ratio={self.trim_ratio} would trim all {num_sellers} sellers. "
                f"Falling back to simple mean (trim_count=0)."
            )
            trim_count = 0

        # Number of parameter tensors (layers)
        num_params = len(seller_updates[valid_sellers[0]])

        aggregated_gradient = []
        for param_idx in range(num_params):
            # Stack this layer's updates from all sellers: shape (num_sellers, *param_shape)
            stacked = torch.stack([seller_updates[sid][param_idx] for sid in valid_sellers], dim=0)
            original_shape = stacked.shape[1:]

            # Flatten to (num_sellers, num_elements) for coordinate-wise sorting
            flat = stacked.view(num_sellers, -1)

            if trim_count == 0:
                agg = torch.mean(flat, dim=0)
            else:
                sorted_flat, _ = torch.sort(flat, dim=0)
                trimmed = sorted_flat[trim_count: num_sellers - trim_count, :]
                agg = torch.mean(trimmed, dim=0)

            aggregated_gradient.append(agg.view(original_shape))

        selected_sids = valid_sellers
        outlier_sids = []
        aggregation_stats = {
            "trim_ratio": self.trim_ratio,
            "trim_count": trim_count,
            "effective_sellers": num_sellers - 2 * trim_count,
        }

        logger.info(
            f"Aggregated {num_sellers} sellers (trimmed {trim_count} from each end, "
            f"{num_sellers - 2 * trim_count} effective)."
        )

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
