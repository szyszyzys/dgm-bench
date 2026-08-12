import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.common_utils import clip_gradient_update
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator
from src.mechanism.gradient.aggregators.skymask_utils.classify import GMM2
from src.mechanism.gradient.aggregators.skymask_utils.models import create_masknet
from src.mechanism.gradient.aggregators.skymask_utils.mytorch import myconv2d, mylinear

logger = logging.getLogger(__name__)


def train_masknet_small(masknet: nn.Module, server_data_loader, epochs: int, lr: float, grad_clip: float,
                        device: torch.device) -> nn.Module:
    """
    Helper function to train the SkyMask MaskNet for the 'Small Dataset' variant.

    FIXED: Convergence check is now performed every 10 epochs to align with official logic
    and prevent premature stopping due to small-batch noise.
    """
    masknet = masknet.to(device)
    optimizer = optim.SGD(masknet.parameters(), lr=lr)
    loss_fn = F.nll_loss

    UPDATES_PER_EPOCH = 4

    # Track loss for official convergence check
    prev_loss = 1e4

    logger.info(f"Starting MaskNet (Small) Training: Epochs={epochs}, LR={lr}, Updates/Epoch={UPDATES_PER_EPOCH}")

    # Create an iterator so we can pull batches manually
    data_iter = iter(server_data_loader)

    for epoch in range(epochs):
        masknet.train()

        current_loss = 0.0

        # --- THE TRICK LOOP ---
        for _ in range(UPDATES_PER_EPOCH):
            try:
                X, y = next(data_iter)
            except StopIteration:
                # If we hit the end of the large dataset, restart
                data_iter = iter(server_data_loader)
                X, y = next(data_iter)

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = masknet(X)
            loss = loss_fn(output, y)
            loss.backward()

            # Manual gradient clipping
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
            optimizer.step()

            current_loss = loss.item()

        if epoch % 10 == 0:
            loss_diff = prev_loss - current_loss

            # Only stop if we have trained at least 10 epochs AND improvement is small
            if epoch > 0 and loss_diff < 1e-2:
                logger.info(f"MaskNet Converged at epoch {epoch} (Loss diff {loss_diff:.4f}). Stopping.")
                break

            # Update prev_loss only at the check points
            prev_loss = current_loss

    logger.info("MaskNet Training Finished.")
    return masknet


class SkymaskSmallAggregator(BaseAggregator):
    """
    Implements the SkyMask (Small Dataset Variant) aggregation strategy.
    Fixed to handle PCA/GMM convergence errors when masks are identical.
    """
    requires_root_gradient = True

    def __init__(self,
                 clip: bool,
                 sm_model_type: str,
                 mask_epochs: int,
                 mask_lr: float,
                 mask_clip: float,
                 mask_threshold: float,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.clip = clip
        self.sm_model_type = sm_model_type
        self.mask_epochs = mask_epochs
        self.mask_lr = mask_lr
        self.mask_clip = mask_clip
        self.mask_threshold = mask_threshold
        logger.info(f"SkymaskSmallAggregator initialized with mask_epochs={self.mask_epochs}, mask_lr={self.mask_lr}")

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- SkyMask (Small) Aggregation (Epoch {global_epoch}) ---")

        # 1. Compute full model parameters
        global_params = [p.data.clone() for p in self.global_model.parameters()]
        worker_params = []
        seller_ids = list(seller_updates.keys())
        processed_updates = {}

        for sid in seller_ids:
            update = clip_gradient_update(seller_updates[sid], self.clip_norm) if self.clip else seller_updates[sid]
            processed_updates[sid] = update
            worker_params.append([p_glob - p_upd for p_glob, p_upd in zip(global_params, update)])

        # Buyer parameters: pseudo_grad = (initial - final), so final = global - pseudo_grad
        buyer_params = [p_glob - p_upd for p_glob, p_upd in zip(global_params, root_gradient)]
        worker_params.append(buyer_params)

        # 2. Determine model type
        sm_model_type = self.sm_model_type if (self.sm_model_type and self.sm_model_type != 'None') else 'flexiblecnn'

        # 3. Create and train MaskNet
        masknet = create_masknet(worker_params, sm_model_type, self.device)
        if masknet is None:
            raise RuntimeError(f"Failed to create masknet with type '{sm_model_type}'.")

        masknet = train_masknet_small(
            masknet,
            self.buyer_data_loader,
            self.mask_epochs,
            self.mask_lr,
            self.mask_clip,
            self.device
        )

        # 4. Extract SOFT masks (Floats)
        seller_masks_np = []
        valid_mask_indices = []  # Track which seller indices produced valid masks

        for i in range(len(seller_ids)):
            seller_mask_layers = []
            for layer in masknet.modules():
                if isinstance(layer, (myconv2d, mylinear)):
                    if hasattr(layer, 'weight_mask'):
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.weight_mask[i].data)))
                    if hasattr(layer, 'bias_mask') and layer.bias_mask is not None:
                        seller_mask_layers.append(torch.flatten(torch.sigmoid(layer.bias_mask[i].data)))

            if not seller_mask_layers:
                logger.warning(f"Seller {seller_ids[i]} produced no mask layers — treating as outlier.")
                continue

            flat_mask = torch.cat(seller_mask_layers).cpu().numpy()
            flat_mask = (flat_mask > self.mask_threshold).astype(float)

            seller_masks_np.append(flat_mask)
            valid_mask_indices.append(i)
        masks_matrix = np.array(seller_masks_np) if seller_masks_np else np.empty((0, 0))

        # --- ROBUST CLUSTERING BLOCK (VARIANCE FIX) ---

        # 1. Sanitize
        if np.isnan(masks_matrix).any():
            logger.warning("NaNs detected in seller masks. Replacing with 0.5.")
            masks_matrix = np.nan_to_num(masks_matrix, nan=0.5)

        # 2. VARIANCE CHECK (The Fix)
        # Instead of checking unique rows, we check if the max variance across sellers is essentially zero.
        # If variance is < 1e-5, PCA will divide by zero.
        max_variance = np.max(np.var(masks_matrix, axis=0)) if masks_matrix.size > 0 else 0.0

        # Also require at least 2 valid masks to cluster
        if len(valid_mask_indices) < 2 or max_variance < 1e-6:
            logger.warning(f"⚠️ Low Mask Variance ({max_variance:.6f}). Skipping GMM (selecting all).")
            gmm_labels = np.ones(len(valid_mask_indices))
        else:
            try:
                # 3. Run GMM
                gmm_labels = GMM2(masks_matrix)

                # 4. Enforce Honest Majority Rule
                count_0 = np.sum(gmm_labels == 0)
                count_1 = np.sum(gmm_labels == 1)

                if count_0 > count_1:
                    logger.info(f"GMM Flip: Cluster 0 is majority ({count_0} vs {count_1}). Swapping labels.")
                    gmm_labels = 1 - gmm_labels

            except Exception as e:
                # Catch any residual sklearn errors
                logger.error(f"GMM2 Clustering Exception: {e}. Defaulting to select all.")
                gmm_labels = np.ones(len(valid_mask_indices))

        # Build a label map: seller index -> gmm label (sellers without masks default to outlier)
        gmm_label_map = {}
        for gmm_idx, seller_idx in enumerate(valid_mask_indices):
            gmm_label_map[seller_idx] = gmm_labels[gmm_idx]

        # 5. Aggregate
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        selected_sids, outlier_sids = [], []
        inlier_updates = []

        aggregation_stats = {}

        for i, sid in enumerate(seller_ids):
            is_inlier = gmm_label_map.get(i, 0) == 1
            aggregation_stats[f"skymask_gmm_label_{sid}"] = int(is_inlier)

            if is_inlier:
                selected_sids.append(sid)
                inlier_updates.append(processed_updates[sid])
            else:
                outlier_sids.append(sid)

        if inlier_updates:
            num_inliers = len(inlier_updates)
            for update in inlier_updates:
                for agg_grad, upd_grad in zip(aggregated_gradient, update):
                    agg_grad.add_(upd_grad, alpha=1 / num_inliers)

        aggregation_stats["skymask_num_selected"] = len(selected_sids)
        aggregation_stats["skymask_num_rejected"] = len(outlier_sids)

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
