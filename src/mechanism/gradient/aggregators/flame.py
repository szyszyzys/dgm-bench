import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class FLAMEAggregator(BaseAggregator):
    """
    FLAME: Taming Backdoors in Federated Learning (USENIX Security 2022).

    Three-stage defense:
      1. HDBSCAN clustering on pairwise cosine distances to filter outliers.
      2. Adaptive clipping to the median L2 norm of surviving gradients.
      3. Differential Privacy noise calibrated to the clipping bound.

    Args:
        noise_scale: Multiplier for DP Gaussian noise (lambda). Default 0.001.
        min_cluster_size_ratio: HDBSCAN min_cluster_size as a fraction of N.
            Default 0.5 means min_cluster_size = N/2 + 1 (majority cluster).
    """

    def __init__(self, *args, noise_scale: float = 0.001,
                 min_cluster_size_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale
        self.min_cluster_size_ratio = min_cluster_size_ratio

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- FLAME Aggregation (Epoch {global_epoch}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(valid_sellers)

        # --- Flatten all gradients ---
        flat_updates = {}
        for sid in valid_sellers:
            flat_updates[sid] = torch.cat([p.view(-1) for p in seller_updates[sid]])

        seller_ids = list(flat_updates.keys())
        stacked = torch.stack([flat_updates[sid] for sid in seller_ids])  # (N, D)

        # =====================================================================
        # Stage 1: HDBSCAN Clustering on Cosine Distances
        # =====================================================================
        selected_indices, outlier_indices, cluster_labels = self._hdbscan_filter(
            stacked, num_sellers
        )

        selected_sids = [seller_ids[i] for i in selected_indices]
        outlier_sids = [seller_ids[i] for i in outlier_indices]

        if not selected_sids:
            # Fallback: if HDBSCAN rejects everyone, use all sellers (FedAvg fallback)
            logger.warning("FLAME: HDBSCAN rejected all sellers. Falling back to FedAvg.")
            selected_sids = seller_ids
            selected_indices = list(range(num_sellers))

        logger.info(
            f"FLAME Stage 1: {len(selected_sids)}/{num_sellers} sellers survived clustering. "
            f"Outliers: {outlier_sids}"
        )

        # =====================================================================
        # Stage 2: Adaptive Clipping to Median Norm
        # =====================================================================
        surviving_flat = stacked[selected_indices]  # (M, D)
        norms = torch.norm(surviving_flat, p=2, dim=1)  # (M,)
        median_norm = torch.median(norms).item()

        # Clip each surviving gradient to median norm
        clipped = []
        for i in range(surviving_flat.shape[0]):
            g = surviving_flat[i]
            g_norm = norms[i].item()
            if g_norm > median_norm and g_norm > 1e-9:
                g = g * (median_norm / g_norm)
            clipped.append(g)

        clipped_stack = torch.stack(clipped)  # (M, D)

        # =====================================================================
        # Stage 3: Aggregate + DP Noise
        # =====================================================================
        aggregated_flat = torch.mean(clipped_stack, dim=0)  # (D,)

        # Add Gaussian noise: N(0, (lambda * S_t)^2)
        if self.noise_scale > 0 and median_norm > 0:
            noise_std = self.noise_scale * median_norm
            noise = torch.randn_like(aggregated_flat) * noise_std
            aggregated_flat = aggregated_flat + noise
            logger.info(f"FLAME Stage 3: Added DP noise (std={noise_std:.6f}, median_norm={median_norm:.4f})")
        else:
            logger.info(f"FLAME Stage 3: No noise added (noise_scale={self.noise_scale}, median_norm={median_norm:.4f})")

        # --- Unflatten back to parameter shapes ---
        aggregated_gradient = []
        offset = 0
        for p in self.global_model.parameters():
            numel = p.numel()
            aggregated_gradient.append(aggregated_flat[offset:offset + numel].view(p.shape))
            offset += numel

        aggregation_stats = {
            "num_surviving": len(selected_sids),
            "num_outliers": len(outlier_sids),
            "median_norm": median_norm,
            "noise_std": self.noise_scale * median_norm if median_norm > 0 else 0.0,
            "cluster_labels": cluster_labels,
        }

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats

    def _hdbscan_filter(self, stacked: torch.Tensor, num_sellers: int):
        """
        Run HDBSCAN on pairwise cosine distances.
        Returns (selected_indices, outlier_indices, cluster_labels_list).
        """
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            logger.warning("hdbscan not installed. Falling back to all-accept (no clustering).")
            return list(range(num_sellers)), [], [-1] * num_sellers

        # Compute pairwise cosine distance matrix: 1 - cos_sim
        normed = F.normalize(stacked, p=2, dim=1)
        cos_sim = torch.mm(normed, normed.t())  # (N, N)
        cos_dist = (1.0 - cos_sim).clamp(min=0.0).cpu().numpy()

        # HDBSCAN clustering
        min_cluster_size = max(2, int(num_sellers * self.min_cluster_size_ratio) + 1)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="precomputed",
            allow_single_cluster=True,
        )
        cluster_labels = clusterer.fit_predict(cos_dist)

        # Find the largest cluster (the assumed benign majority)
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        if not unique_labels:
            # All points labeled as noise — fallback
            logger.warning("FLAME: HDBSCAN marked all sellers as noise. Accepting all.")
            return list(range(num_sellers)), [], cluster_labels.tolist()

        # Find largest cluster
        largest_label = max(unique_labels, key=lambda lbl: np.sum(cluster_labels == lbl))
        largest_count = np.sum(cluster_labels == largest_label)

        selected_indices = [i for i in range(num_sellers) if cluster_labels[i] == largest_label]
        outlier_indices = [i for i in range(num_sellers) if cluster_labels[i] != largest_label]

        logger.info(
            f"FLAME HDBSCAN: {len(unique_labels)} clusters found. "
            f"Largest cluster (label={largest_label}): {largest_count} sellers. "
            f"Noise: {np.sum(cluster_labels == -1)}."
        )

        return selected_indices, outlier_indices, cluster_labels.tolist()
