import copy
import logging
from typing import Dict, List, Any, Optional  # ✅ 1. Import Optional

import torch
from torch.utils.data import DataLoader

# Import the Aggregator type hint
from src.mechanism.gradient.aggregator import Aggregator


class RoundBasedLOOEvaluator:
    """
    Calculates per-round marginal contribution using Leave-One-Out (LOO).

    This is an ONLINE but computationally EXPENSIVE method.
    It works by evaluating N+1 potential model updates every round
    (1 for all sellers, and N for "all-sellers-except-one").

    The "value" is the *immediate* performance gain on the buyer's data.
    """

    def __init__(self,
                 aggregator_object: 'Aggregator',
                 buyer_root_loader: DataLoader,
                 device: str):
        """
        Args:
            aggregator_object: The actual Aggregator instance. We need this
                               to call its 'aggregate' and 'apply_gradient'
                               logic repeatedly.
            buyer_root_loader: The buyer's private validation set.
            device: The device to run evaluations on (e.g., 'cuda').
        """
        self.aggregator = aggregator_object
        self.buyer_loader = buyer_root_loader
        self.device = device
        logging.info("RoundBasedLOOEvaluator initialized.")

    def _get_performance(self, model: torch.nn.Module) -> float:
        """Helper function to evaluate a model's accuracy on the buyer's data."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Iterate over the batch, not the unpacked items
            for batch in self.buyer_loader:

                try:
                    if len(batch) == 3:  # Text data
                        labels, data, _ = batch
                    else:  # Image/Tabular
                        data, labels = batch
                except Exception as e:
                    logging.warning(f"Skipping batch in performance eval due to unpack error: {e}")
                    continue

                data, labels = data.to(self.device), labels.to(self.device)

                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (100 * correct / total) if total > 0 else 0.0

    def _get_post_update_performance(
            self,
            original_model: torch.nn.Module,
            seller_gradients: Dict[str, List[torch.Tensor]],
            round_number: int,
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ 2. Accept buyer_gradient here
    ) -> float:
        """
        Simulates a full aggregation and update, then returns the
        performance of the *updated* model.
        """
        # 1. Create a deep copy of the model to avoid changing the real one
        temp_model = copy.deepcopy(original_model).to(self.device)

        # 2. Save and swap aggregator state to avoid corrupting stateful aggregators
        #    (e.g., RFLPA's _prev_global_gradient, MartFL's baseline_id).
        #    CRITICAL: restore the LIVE global_model object by reference, never a
        #    deepcopy — rebinding it to a clone desyncs the aggregator from the
        #    StatefulModelFactory the sellers load from and diverges training (see
        #    the same fix in kernel_shapely.py). Deep-copy every field except the model.
        live_global_model = self.aggregator.strategy.global_model
        original_strategy_state = {
            k: copy.deepcopy(v)
            for k, v in self.aggregator.strategy.__dict__.items()
            if k != "global_model"
        }
        self.aggregator.strategy.global_model = temp_model  # Temporarily swap

        try:
            agg_grad, _, _, _ = self.aggregator.aggregate(
                global_epoch=round_number,
                seller_updates=seller_gradients,
                root_gradient=buyer_gradient,
            )

            # 3. Simulate applying the gradient
            if agg_grad:
                try:
                    self.aggregator.apply_gradient(agg_grad)
                except Exception as e:
                    logging.error(f"LOO: Failed to apply temp gradient: {e}")

            # 4. Evaluate the temporary model's performance
            performance = self._get_performance(temp_model)
        finally:
            # 5. Restore the aggregator strategy state, keeping the ORIGINAL
            #    global_model object (by reference) so it stays the same module
            #    the model factory holds.
            self.aggregator.strategy.__dict__.update(original_strategy_state)
            self.aggregator.strategy.global_model = live_global_model
            del temp_model  # Clean up memory

        return performance

    def evaluate_round(
            self,
            round_number: int,
            current_global_model: torch.nn.Module,
            seller_gradients: Dict[str, List[torch.Tensor]],
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ 5. Accept buyer_gradient
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs the full LOO evaluation for the current round.

        Returns:
            A dictionary mapping {seller_id: {'marginal_contrib_loo': score}}
        """
        logging.info("Starting round-based LOO evaluation...")
        seller_ids = list(seller_gradients.keys())
        valuations = {sid: {} for sid in seller_ids}

        # 1. Get baseline performance (Value of all sellers)
        # This is the performance *after* updating with ALL gradients
        logging.debug("LOO: Calculating baseline (all sellers)...")
        baseline_performance = self._get_post_update_performance(
            current_global_model,
            seller_gradients,
            round_number,
            buyer_gradient=buyer_gradient  # ✅ 6. Pass gradient down
        )

        logging.debug(f"LOO Baseline Performance: {baseline_performance:.2f}%")

        # 2. Get performance for each "Leave-One-Out" coalition
        for seller_to_remove in seller_ids:
            logging.debug(f"LOO: Calculating contribution for {seller_to_remove}...")

            # Create the "leave-one-out" gradient dictionary
            loo_gradients = {
                sid: grad for sid, grad in seller_gradients.items()
                if sid != seller_to_remove
            }

            if not loo_gradients:
                # If this was the only seller, the LOO value is 0
                loo_performance = self._get_performance(current_global_model)
            else:
                loo_performance = self._get_post_update_performance(
                    current_global_model,
                    loo_gradients,
                    round_number,
                    buyer_gradient=buyer_gradient  # ✅ 7. Pass gradient down
                )

            # Marginal contribution = (Value with all) - (Value without seller)
            marginal_contrib = baseline_performance - loo_performance
            valuations[seller_to_remove]['marginal_contrib_loo'] = marginal_contrib
            logging.debug(
                f"LOO {seller_to_remove}: {baseline_performance:.2f} - {loo_performance:.2f} = {marginal_contrib:.2f}")

        logging.info("Round-based LOO evaluation complete.")
        return valuations