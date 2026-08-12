import copy
import logging
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader


class InfluenceEvaluator:
    """
    Calculates per-round contribution using a fast, online
    Influence Function approximation (also known as TracIn).

    It measures the direct impact of a seller's gradient on
    a small batch of the buyer's validation data.
    """

    def __init__(self,
                 buyer_root_loader: DataLoader,
                 device: str,
                 learning_rate: float):
        """
        Args:
            buyer_root_loader: The buyer's private validation set.
            device: The device to run evaluations on (e.g., 'cuda').
            learning_rate: The server-side learning rate used for
                           applying gradients.
        """
        self.buyer_loader = buyer_root_loader
        self.device = device
        self.lr = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()  # Or your default loss
        logging.info("InfluenceEvaluator initialized.")

    def _get_batch_loss(self, model: torch.nn.Module,
                        data: torch.Tensor,
                        labels: torch.Tensor) -> float:
        """Helper to get loss on a single batch."""
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            return self.loss_fn(outputs, labels).item()

    @torch.no_grad()
    def evaluate_round(
            self,
            current_global_model: torch.nn.Module,
            seller_gradients: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs the fast influence evaluation for the current round.
        """
        logging.info("Starting fast influence evaluation...")
        valuations = {sid: {} for sid in seller_gradients.keys()}

        try:
            # 1. Get a single, representative batch from the buyer
            batch = next(iter(self.buyer_loader))

            # Apply the same logic as train_local_model
            if len(batch) == 3:  # Text data
                labels, data, _ = batch
            else:  # Image/Tabular
                data, labels = batch

            data, labels = data.to(self.device), labels.to(self.device)

        except StopIteration:
            logging.error("Buyer loader is empty, cannot run influence evaluation.")
            return valuations

        # 2. Get the baseline loss on the "before" model
        current_global_model.eval()
        loss_before = self._get_batch_loss(current_global_model, data, labels)

        # 3. Get "after" loss for each seller's hypothetical update
        for sid, grad_list in seller_gradients.items():
            if not grad_list:
                valuations[sid]['influence_score'] = 0.0
                continue

            # 4. Create a temporary model state
            # We apply the update 'in-place' on a copy
            temp_model = copy.deepcopy(current_global_model)
            temp_model.train()  # Set to train mode for param updates

            # 5. Apply the seller's pseudo-gradient (initial - final)
            # Use lr=1.0 to match server-side apply_gradient behavior:
            #   param = param - 1.0 * pseudo_grad = final_weights
            with torch.no_grad():
                for i, p in enumerate(temp_model.parameters()):
                    p.sub_(grad_list[i].data)

            # 6. Calculate the loss on the "after" model
            loss_after = self._get_batch_loss(temp_model, data, labels)

            # 7. Influence = improvement in loss
            influence_score = loss_before - loss_after
            valuations[sid]['influence_score'] = influence_score

            del temp_model  # Clean up memory

        logging.info("Fast influence evaluation complete.")
        return valuations