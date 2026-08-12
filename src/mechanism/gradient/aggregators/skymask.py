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


def train_masknet(masknet: nn.Module, server_data_loader, epochs: int, lr: float, grad_clip: float,
                  device: torch.device) -> nn.Module:
    """Helper function to train the SkyMask MaskNet."""
    masknet = masknet.to(device)
    optimizer = optim.SGD(masknet.parameters(), lr=lr)
    loss_fn = F.nll_loss

    logger.info(f"Starting MaskNet Training: Epochs={epochs}, LR={lr}")
    for epoch in range(epochs):
        masknet.train()
        for X, y in server_data_loader:
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
    logger.info("MaskNet Training Finished.")
    return masknet


class SkymaskAggregator(BaseAggregator):
    """
    Implements the SkyMask aggregation strategy.

    This method trains a special neural network (MaskNet) on the server's trusted
    data to learn which parameters of a seller's update are beneficial. It then uses
    these learned "masks" to cluster sellers into benign and malicious groups.
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

        # Pass all common arguments (global_model, device, etc.) up to the BaseAggregator
        super().__init__(*args, **kwargs)

        # Handle the specific parameters for Skymask
        self.clip = clip
        self.sm_model_type = sm_model_type
        self.mask_epochs = mask_epochs
        self.mask_lr = mask_lr
        self.mask_clip = mask_clip
        self.mask_threshold = mask_threshold
        logger.info(f"SkymaskAggregator initialized with mask_epochs={self.mask_epochs}, mask_lr={self.mask_lr}")

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],  # <-- Now a required, named argument
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- SkyMask Aggregation (Epoch {global_epoch}) ---")

        if self.buyer_data_loader is None:
            raise ValueError("SkyMask requires a buyer_data_loader to train the MaskNet, but None was provided.")

        # 1. Compute full model parameters
        global_params = [p.data.clone() for p in self.global_model.parameters()]
        worker_params = []
        seller_ids = list(seller_updates.keys())
        processed_updates = {}

        for sid in seller_ids:
            update = clip_gradient_update(seller_updates[sid], self.clip_norm) if self.clip else seller_updates[sid]
            processed_updates[sid] = update
            worker_params.append([p_glob - p_upd for p_glob, p_upd in zip(global_params, update)])

        # Here, we use the passed-in root_gradient to construct the buyer's parameters
        # pseudo_grad = (initial - final), so final = global - pseudo_grad
        buyer_params = [p_glob - p_upd for p_glob, p_upd in zip(global_params, root_gradient)]
        worker_params.append(buyer_params)

        # 2. Determine the correct model type
        if self.sm_model_type == 'None' or self.sm_model_type is None or self.sm_model_type == '':
            sm_model_type = 'flexiblecnn'
            logger.warning(f"sm_model_type not set. Using 'dynamic' which auto-adapts to model architecture.")
        else:
            sm_model_type = self.sm_model_type
            logger.info(f"Using explicitly set sm_model_type: {sm_model_type}")

        # 3. Create and train the MaskNet
        masknet = create_masknet(worker_params, sm_model_type, self.device)

        if masknet is None:
            # NO FALLBACK - raise error instead
            raise RuntimeError(
                f"Failed to create masknet with type '{sm_model_type}'. "
                f"This is a configuration error. Check your model architecture matches the sm_model_type. "
                f"Available types: cnn, resnet18, resnet20, lr, lenet, cifarcnn, flexiblecnn"
            )

        masknet = train_masknet(masknet, self.buyer_data_loader, self.mask_epochs, self.mask_lr, self.mask_clip,
                                self.device)

        # 4. Extract masks and classify with GMM
        seller_masks_np = []
        valid_mask_indices = []  # Track which seller indices produced valid masks
        t = torch.tensor([self.mask_threshold], device=self.device)
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
            flat_mask = (torch.cat(seller_mask_layers) > t).float()
            seller_masks_np.append(flat_mask.cpu().numpy())
            valid_mask_indices.append(i)

        masks_matrix = np.array(seller_masks_np) if seller_masks_np else np.array([])
        max_variance = np.max(np.var(masks_matrix, axis=0)) if masks_matrix.size > 0 else 0.0

        if len(seller_masks_np) < 2 or max_variance < 1e-6:
            if max_variance < 1e-6 and len(seller_masks_np) >= 2:
                logger.warning(f"⚠️ Low Mask Variance ({max_variance:.6f}). Skipping GMM (selecting all).")
            gmm_labels = np.ones(len(seller_masks_np))
        else:
            try:
                gmm_labels = GMM2(seller_masks_np)
            except Exception as e:
                logger.error(f"GMM2 Clustering Exception: {e}. Defaulting to select all.")
                gmm_labels = np.ones(len(seller_masks_np))

        # Build a label map: seller index -> gmm label (sellers without masks default to outlier)
        gmm_label_map = {}
        for gmm_idx, seller_idx in enumerate(valid_mask_indices):
            gmm_label_map[seller_idx] = gmm_labels[gmm_idx]

        # 5. Aggregate using inliers
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
            logger.info(f"Aggregating {num_inliers} inlier updates.")
            for update in inlier_updates:
                for agg_grad, upd_grad in zip(aggregated_gradient, update):
                    agg_grad.add_(upd_grad, alpha=1 / num_inliers)

        aggregation_stats["skymask_num_selected"] = len(selected_sids)
        aggregation_stats["skymask_num_rejected"] = len(outlier_sids)

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats
