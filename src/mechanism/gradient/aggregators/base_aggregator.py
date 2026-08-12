import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Abstract base class for all aggregation strategies."""

    requires_root_gradient: bool = False

    def __init__(self, global_model: nn.Module, device: torch.device,
                 loss_fn: nn.Module, buyer_data_loader: DataLoader, clip_norm: float, **kwargs):
        self.global_model = global_model
        self.device = device
        self.loss_fn = loss_fn
        self.buyer_data_loader = buyer_data_loader
        self.clip_norm = clip_norm
        self._buyer_data_iter = None  # Initialize iterator as None

    @abstractmethod
    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict]:
        """
        The core aggregation logic. Must be implemented by all subclasses.
        Returns (aggregated_gradient, selected_seller_ids, outlier_ids, aggregation_stats).
        """
        raise NotImplementedError

    def apply_gradient(self, aggregated_gradient: List[torch.Tensor], learning_rate: float = 1.0):
        """
        Applies the aggregated pseudo-gradient (weight difference) to the global model.

        For FedAvg with weight differences:
            learning_rate should be 1.0 (default)
            Update: global_weights = global_weights - 1.0 * averaged_weight_difference
                  = global_weights - (sum of weight_differences) / num_clients

        Args:
            aggregated_gradient: Average of (initial_weights - final_weights) from clients
            learning_rate: Server-side learning rate (default=1.0 for standard FedAvg)
        """
        if not aggregated_gradient:
            logger.warning("apply_gradient called with empty gradient. Skipping.")
            return

        with torch.no_grad():
            for param, pseudo_grad in zip(self.global_model.parameters(), aggregated_gradient):
                # FedAvg update: param = param - learning_rate * pseudo_grad
                # With lr=1.0: param = param - pseudo_grad
                # Since pseudo_grad = (old - new), this becomes: param = param - (old - new) = new
                param.add_(pseudo_grad, alpha=-learning_rate)

        logger.debug(f"Applied gradient with learning_rate={learning_rate}")

    def _get_model_from_update(self, update: List[torch.Tensor]) -> nn.Module:
        """Reconstruct a model from a pseudo-gradient update (initial - final)."""
        temp_model = copy.deepcopy(self.global_model)
        with torch.no_grad():
            for model_p, update_p in zip(temp_model.parameters(), update):
                # update = (initial_weights - final_weights), so final = global - update
                model_p.sub_(update_p)
        return temp_model

    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float, float, float]:
        """Evaluates a model using the buyer's data loader."""
        model.to(self.device)
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []
        if not self.buyer_data_loader:
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in self.buyer_data_loader:
                # Check the batch format to handle different data types
                if len(batch) == 3:  # Text data: (labels, texts, lengths)
                    labels, data, _ = batch
                else:  # Image/Tabular data: (data, labels)
                    data, labels = batch

                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)

                total_loss += self.loss_fn(outputs, labels).item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += data.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0

        acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        kappa = cohen_kappa_score(all_labels, all_preds)

        # Returning 4 values as the original function signature expects
        return avg_loss, acc, kappa, 0.0

    def _compute_trust_gradient(self) -> list[torch.Tensor]:
        """
        Computes the gradient on the server's trusted (buyer's) dataset.
        This is a common utility needed by multiple robust aggregators.
        """
        logger.info("Computing trust gradient on server's data...")
        # Use a deepcopy to avoid altering the main model's state
        model_copy = copy.deepcopy(self.global_model).to(self.device)
        model_copy.train()

        try:
            # Get a single batch from the trusted data loader
            batch = next(iter(self.buyer_data_loader))
            if len(batch) == 3:  # Text data: (labels, texts, lengths)
                labels, inputs, _ = batch
            else:  # Image/Tabular data: (data, labels)
                inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
        except StopIteration:
            logger.error("The trusted (buyer) data loader is empty!")
            raise

        model_copy.zero_grad()
        outputs = model_copy(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()

        # Extract and return the gradients
        trust_gradient = [p.grad.clone().detach() for p in model_copy.parameters()]
        logger.info("Trust gradient computed successfully.")
        return trust_gradient
    # ==========================================================
