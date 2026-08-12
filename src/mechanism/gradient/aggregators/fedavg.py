import logging
from typing import Dict, List, Tuple, Any

import torch

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class FedAvgAggregator(BaseAggregator):
    """
    Implements the standard Federated Averaging (FedAvg) algorithm.
    It performs a simple, unweighted average of all valid seller updates.
    """

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- FedAvg Aggregation (Epoch {global_epoch}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received for FedAvg aggregation.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            # Return an empty stats dictionary
            return zero_grad, [], [], {}

        num_valid_sellers = len(valid_sellers)
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]

        # Perform the simple average
        for sid in valid_sellers:
            update = seller_updates[sid]
            for agg_grad, upd_grad in zip(aggregated_gradient, update):
                agg_grad.add_(upd_grad, alpha=1 / num_valid_sellers)

        logger.info(f"Aggregated updates from {num_valid_sellers} sellers.")

        selected_sids = valid_sellers
        outlier_sids = []

        aggregation_stats = {}

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
