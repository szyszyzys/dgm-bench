import copy
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class DeepSightAggregator(BaseAggregator):
    """
    DeepSight: Detecting Poisoned Models in Federated Learning (NDSS 2022).

    Dual-metric outlier detection:
      1. NEUP (Neuron Excitation Under Perturbation): detects backdoor triggers
         wired into feature extraction layers by comparing penultimate activations.
      2. DDUP (Decision Distortion Under Perturbation): detects classification head
         manipulation by comparing final-layer weight/bias differences.
      3. Sellers flagged by either metric are rejected; the rest are FedAvg-aggregated.

    Args:
        threshold_c: Number of standard deviations for the density threshold.
            Higher = more permissive. Default 2.0.
        neighbor_ratio: Fraction of sellers used for density estimation.
            Default 0.5 (N/2 nearest neighbors).
    """

    requires_root_gradient: bool = False

    def __init__(self, *args, threshold_c: float = 2.0,
                 neighbor_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_c = threshold_c
        self.neighbor_ratio = neighbor_ratio

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- DeepSight Aggregation (Epoch {global_epoch}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(valid_sellers)
        if num_sellers < 3:
            logger.warning(f"DeepSight needs >= 3 sellers for density estimation. Got {num_sellers}. Using FedAvg.")
            return self._fedavg_fallback(seller_updates, valid_sellers)

        if not self.buyer_data_loader:
            logger.warning("DeepSight requires calibration data (buyer_data_loader). Using FedAvg.")
            return self._fedavg_fallback(seller_updates, valid_sellers)

        # =====================================================================
        # Step 1-2: Extract NEUPs (penultimate layer activation differences)
        # =====================================================================
        neup_vectors = self._extract_neups(seller_updates, valid_sellers)

        # =====================================================================
        # Step 3: Extract DDUPs (classification head weight/bias differences)
        # =====================================================================
        ddup_vectors = self._extract_ddups(seller_updates, valid_sellers)

        # =====================================================================
        # Step 4: Dual-Threshold Filtering
        # =====================================================================
        neup_outliers = set()
        ddup_outliers = set()

        if neup_vectors is not None and len(neup_vectors) == num_sellers:
            neup_outlier_indices = self._density_filter(neup_vectors, "NEUP")
            neup_outliers = {valid_sellers[i] for i in neup_outlier_indices}

        if ddup_vectors is not None and len(ddup_vectors) == num_sellers:
            ddup_outlier_indices = self._density_filter(ddup_vectors, "DDUP")
            ddup_outliers = {valid_sellers[i] for i in ddup_outlier_indices}

        # Bipartite decision: reject if flagged by either metric
        rejected = neup_outliers | ddup_outliers
        selected_sids = [sid for sid in valid_sellers if sid not in rejected]
        outlier_sids = [sid for sid in valid_sellers if sid in rejected]

        logger.info(
            f"DeepSight: NEUP outliers={len(neup_outliers)}, DDUP outliers={len(ddup_outliers)}, "
            f"Total rejected={len(rejected)}/{num_sellers}"
        )

        # =====================================================================
        # Step 5: Clean Aggregation (FedAvg on inliers)
        # =====================================================================
        if not selected_sids:
            logger.warning("DeepSight rejected all sellers. Falling back to FedAvg on all.")
            selected_sids = valid_sellers
            outlier_sids = []

        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for sid in selected_sids:
            for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                agg_grad.add_(upd_grad, alpha=1.0 / len(selected_sids))

        aggregation_stats = {
            "num_surviving": len(selected_sids),
            "num_neup_outliers": len(neup_outliers),
            "num_ddup_outliers": len(ddup_outliers),
            "neup_outliers": list(neup_outliers),
            "ddup_outliers": list(ddup_outliers),
        }

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats

    def _extract_neups(self, seller_updates, valid_sellers) -> np.ndarray:
        """
        Extract NEUP vectors by comparing penultimate layer activations
        between each seller's reconstructed model and the global model.
        """
        # Find the penultimate layer
        penultimate_layer = self._find_penultimate_layer(self.global_model)
        if penultimate_layer is None:
            logger.warning("Could not identify penultimate layer. Skipping NEUP.")
            return None

        # Get global model activations
        global_act = self._get_activations(self.global_model, penultimate_layer)
        if global_act is None:
            return None

        neup_vectors = []
        for sid in valid_sellers:
            # Reconstruct seller model: W_seller = W_global - pseudo_grad
            # (pseudo_grad = initial - final, so final = global - pseudo_grad)
            seller_model = self._get_model_from_update(seller_updates[sid])
            seller_act = self._get_activations(seller_model, penultimate_layer)
            if seller_act is None:
                neup_vectors.append(np.zeros_like(global_act))
            else:
                neup_vectors.append(seller_act - global_act)
            del seller_model

        return np.array(neup_vectors)

    def _extract_ddups(self, seller_updates, valid_sellers) -> np.ndarray:
        """
        Extract DDUP vectors from classification head weight/bias differences.
        """
        # Find the final linear layer
        final_layer = self._find_final_linear(self.global_model)
        if final_layer is None:
            logger.warning("Could not identify final linear layer. Skipping DDUP.")
            return None

        # Get global head params
        global_weight = final_layer.weight.data.clone()
        global_bias = final_layer.bias.data.clone() if final_layer.bias is not None else None

        ddup_vectors = []
        for sid in valid_sellers:
            seller_model = self._get_model_from_update(seller_updates[sid])
            seller_final = self._find_final_linear(seller_model)

            delta_weight = seller_final.weight.data - global_weight
            # Per-class L2 norm of weight differences
            weight_norms = torch.norm(delta_weight, p=2, dim=1).cpu().numpy()

            if global_bias is not None and seller_final.bias is not None:
                delta_bias = (seller_final.bias.data - global_bias).cpu().numpy()
                ddup_vec = np.concatenate([weight_norms, delta_bias])
            else:
                ddup_vec = weight_norms

            ddup_vectors.append(ddup_vec)
            del seller_model

        return np.array(ddup_vectors)

    def _density_filter(self, vectors: np.ndarray, metric_name: str) -> List[int]:
        """
        Density-based outlier detection using pairwise Euclidean distances.
        Returns list of outlier indices.
        """
        n = len(vectors)
        k = max(1, int(n * self.neighbor_ratio))

        # Pairwise Euclidean distance matrix
        # vectors: (N, D)
        diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]  # (N, N, D)
        dist_matrix = np.linalg.norm(diff, axis=2)  # (N, N)

        # For each seller, find density score: max distance to k nearest neighbors
        density_scores = []
        for i in range(n):
            dists = np.sort(dist_matrix[i])  # sort ascending
            # Skip self (dist=0), take k nearest
            neighbor_dists = dists[1:k + 1]
            density_scores.append(np.max(neighbor_dists) if len(neighbor_dists) > 0 else 0.0)

        density_scores = np.array(density_scores)
        mu = np.mean(density_scores)
        sigma = np.std(density_scores)
        threshold = mu + self.threshold_c * sigma

        outlier_indices = [i for i in range(n) if density_scores[i] > threshold]

        logger.info(
            f"DeepSight {metric_name}: threshold={threshold:.4f} "
            f"(mu={mu:.4f}, sigma={sigma:.4f}, C={self.threshold_c}), "
            f"outliers={len(outlier_indices)}/{n}"
        )

        return outlier_indices

    def _get_activations(self, model: nn.Module, target_layer_name: str) -> np.ndarray:
        """
        Run calibration data through the model and extract averaged
        activations from the target layer using a forward hook.
        """
        model.to(self.device)
        model.eval()

        activations = []

        # Find the layer by name
        target_module = dict(model.named_modules()).get(target_layer_name)
        if target_module is None:
            logger.warning(f"Target layer '{target_layer_name}' not found in model.")
            return None

        def hook_fn(module, input, output):
            # Handle both tuple and tensor outputs
            out = output if isinstance(output, torch.Tensor) else output[0]
            activations.append(out.detach().cpu())

        handle = target_module.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                for batch in self.buyer_data_loader:
                    if len(batch) == 3:
                        labels, data, _ = batch
                    else:
                        data, labels = batch
                    data = data.to(self.device)
                    model(data)
        finally:
            handle.remove()

        if not activations:
            return None

        # Average across all calibration samples
        all_acts = torch.cat(activations, dim=0)  # (total_samples, ...)
        avg_act = all_acts.mean(dim=0).flatten().numpy()
        return avg_act

    def _find_penultimate_layer(self, model: nn.Module) -> str:
        """Find the name of the penultimate layer (layer before final Linear)."""
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        if len(linear_layers) >= 2:
            # Penultimate = the layer before the last linear
            # We want the module that feeds INTO the second-to-last linear
            return linear_layers[-2]

        # Fallback: look for last non-linear layer before the final one
        all_layers = [(name, module) for name, module in model.named_modules()
                      if name and not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict))]
        if len(all_layers) >= 2:
            return all_layers[-2][0]

        return None

    def _find_final_linear(self, model: nn.Module) -> nn.Linear:
        """Find the final Linear layer (classification head)."""
        final_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                final_linear = module
        return final_linear

    def _fedavg_fallback(self, seller_updates, valid_sellers):
        """Simple FedAvg when DeepSight can't run."""
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for sid in valid_sellers:
            for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                agg_grad.add_(upd_grad, alpha=1.0 / len(valid_sellers))
        return aggregated_gradient, valid_sellers, [], {"fallback": "fedavg"}
