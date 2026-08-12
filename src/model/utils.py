#!/usr/bin/env python3
"""
fl_training.py

This file defines:
  - A simple neural network model (SimpleMLP) for demonstration.
  - A local training routine that trains a local copy of the global model on local data.
  - Functions to compute the gradient update as the difference between the trained and initial model parameters.
  - A helper to flatten the gradient into a single vector (useful for federated aggregation).

Usage (within a seller’s get_gradient method):
    flat_update, data_size = local_training_and_get_gradient(model, train_dataset, batch_size=16, device, local_epochs=1, lr=0.01)
    # flat_update is a numpy array representing the update from local training.
"""

import logging
from typing import List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.common_utils.model_utils import _log_param_stats
# from model.text_model import TEXTCNN
from src.model.models import LeNet, TextCNN, SimpleCNN


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1,
                      max_grad_norm: float = 1.0,
                      use_amp: bool = False) -> Tuple[nn.Module, Union[float, None]]:
    logging.info(f"--- ⚡️ Running with CORRECTED GradScaler + Autocast ---")
    model.train()
    batch_losses_all = []

    # Convert to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    scaler = GradScaler() if use_amp else None

    if not train_loader or len(train_loader) == 0:
        logging.warning("train_loader is empty or None. Skipping training.")
        return model, None

    for epoch in range(epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 3:  # Text data
                    labels, data, _ = batch_data
                else:  # Image/Tabular
                    data, labels = batch_data

                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if data.size(0) <= 1:
                    # This is expected if drop_last=False, log at debug level
                    logging.debug(
                        f"Skipping batch {batch_idx} (size {data.size(0)}) to avoid potential BatchNorm error.")
                    continue
                # Once-per-epoch sanity log. Only no-sync attributes (.shape,
                # .dtype) — no .min()/.max()/.mean()/.isnan() etc. Each of
                # those is a GPU→CPU sync that costs ~1ms per call, and with
                # 10 clients × N rounds × 2 local epochs per cell they add up
                # to a non-trivial fraction of total time. Tensor stats were
                # diagnostic-only; the per-batch loss check below is the
                # actual safety net.
                if batch_idx == 0:
                    logging.info(f"🔍 Input shape: {data.shape}, dtype: {data.dtype}")

                optimizer.zero_grad(set_to_none=True)  # set_to_none=True is slightly more memory efficient

                # Autocast ONLY wraps the forward pass
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                # Single scalar GPU sync via torch.isfinite on the loss.
                # Previously this loop did 6 tensor-wide syncs per batch:
                #   - 2× isnan(data).any() + isinf(data).any()  (input check)
                #   - 2× isnan(data).any() + isinf(data).any()  (DUPLICATE)
                #   - 1× isnan(outputs).any() + isinf(outputs).any()  (output)
                # All redundant: any NaN in input → NaN in outputs → NaN in
                # loss → caught here. Scalar isfinite() is ~100× cheaper than
                # tensor.isnan().any(), and on a small CNN those syncs were
                # the dominant cost (GPU idle waiting for sync result).
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss ({loss.item()}) in batch {batch_idx}. Skipping update.")
                    continue

                # Backward pass and optimizer steps
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

                batch_losses_all.append(loss.item())

            except Exception as e:
                logging.warning(f"Error in batch {batch_idx}: {e}", exc_info=False)
                continue

    if not batch_losses_all:
        logging.error("❌ CRITICAL: No batches were successfully processed. This indicates severe training instability.")
        logging.error(
            "   Possible causes: (1) Input data not normalized, (2) Learning rate too high, (3) Model architecture issue")
        # Return model with a dummy loss instead of None to prevent crashes
        return model, float('inf')  # Changed from None
    else:
        overall_avg_loss = np.mean(batch_losses_all)
        logging.info(
            f"Finished local training. Total successful batches: {len(batch_losses_all)}. "
            f"Overall Avg Loss: {overall_avg_loss:.4f}"
        )
        return model, overall_avg_loss


def flatten_gradients(grad_list: List[torch.Tensor]) -> np.ndarray:
    """
    Flatten a list of gradient tensors into a single 1D numpy array.
    """
    flat_grad = torch.cat([g.view(-1) for g in grad_list])
    return flat_grad.detach().cpu().numpy()


def test_local_model(model: nn.Module,
                     test_loader: DataLoader,
                     criterion: nn.Module,
                     device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if not test_loader or len(test_loader) == 0:
        logging.warning("test_loader is empty or None. Returning NaN metrics.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    with torch.no_grad():
        for batch_idx, batch_items in enumerate(test_loader):
            try:
                # Handle different data formats
                if len(batch_items) == 3:  # Text data: (labels, texts, lengths)
                    batch_labels, batch_data = batch_items[0], batch_items[1]
                else:  # Image/Tabular data: (data, labels)
                    batch_data, batch_labels = batch_items[0], batch_items[1]

                batch_data = batch_data.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                if torch.isfinite(loss):
                    total_loss += loss.item() * batch_data.size(0)
                    _, predicted = torch.max(outputs, dim=1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_data.size(0)
                else:
                    logging.warning(f"Non-finite loss in test batch {batch_idx}. Skipping.")

            except Exception as e:
                logging.warning(f"Error processing test batch {batch_idx}: {e}. Skipping.", exc_info=False)
                continue

    if total_samples == 0:
        logging.warning("No samples were successfully processed during testing.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


def log_param_stats(params: List[torch.Tensor], prefix: str = ""):
    """Logs key statistics for a specific parameter tensor to debug instability."""
    TENSOR_INDEX_TO_DEBUG = 12
    if len(params) > TENSOR_INDEX_TO_DEBUG:
        p = params[TENSOR_INDEX_TO_DEBUG]
        # Check if the tensor is on a CUDA device and has elements before calculating stats
        if p.is_cuda and p.numel() > 0:
            stats = {
                'shape': p.shape,
                'min': torch.min(p).item(),
                'max': torch.max(p).item(),
                'mean': torch.mean(p).item(),
                'has_nan': torch.isnan(p).any().item(),
                'has_inf': torch.isinf(p).any().item()
            }
            logging.info(f"  -> Stats for {prefix} Param[{TENSOR_INDEX_TO_DEBUG}]: {stats}")
        else:
            logging.info(f"  -> {prefix} Param[{TENSOR_INDEX_TO_DEBUG}] is on CPU or empty, skipping stats.")


def local_training_and_get_gradient(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 0.01,
        opt_str: str = "SGD",
        eps=1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0005
) -> Tuple[Optional[List[torch.Tensor]], Optional[float]]:
    """
    Performs local training and returns the WEIGHT DIFFERENCE (pseudo-gradient).

    Returns: (initial_weights - final_weights)

    This is the standard FedAvg approach where the server aggregates weight differences
    and applies them directly (with server learning rate = 1.0).
    """
    logging.debug(f"Starting local training: {local_epochs} epochs, lr={lr}, optimizer={opt_str}")

    # Store initial parameters
    initial_params = [p.data.clone() for p in model.parameters()]

    print("--- Inspecting Model Parameters PRE-TRAINING ---")
    log_param_stats(initial_params, prefix="Initial")

    # Train model
    model.train()

    criterion = nn.CrossEntropyLoss()
    if opt_str.upper() == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    try:
        trained_model, avg_train_loss = train_local_model(
            model, train_loader, criterion, optimizer, device, epochs=local_epochs
        )
        if trained_model is None or avg_train_loss is None:
            logging.error("train_local_model returned None!")
            return None, None
    except Exception as e:
        logging.error(f"Error during local training: {e}", exc_info=True)
        return None, None

    # Compute pseudo-gradients (weight differences)
    pseudo_gradients: List[torch.Tensor] = []
    with torch.no_grad():
        trained_params = [p.data for p in trained_model.parameters()]

        print("--- Inspecting Model Parameters POST-TRAINING ---")
        log_param_stats(trained_params, prefix="Trained")

        if len(trained_params) != len(initial_params):
            logging.error(f"Parameter count mismatch!")
            return None, None

        # ✅ FedAvg-style: Return weight difference
        # pseudo_gradient = initial_weights - final_weights
        for init_p, trained_p in zip(initial_params, trained_params):
            if init_p.shape != trained_p.shape:
                logging.error(f"Shape mismatch: {init_p.shape} vs {trained_p.shape}")
                return None, None

            # Weight difference (points in the direction of improvement)
            pseudo_grad = init_p - trained_p
            pseudo_gradients.append(pseudo_grad)

    if not pseudo_gradients:
        logging.error("Generated empty pseudo-gradient list!")
        return None, None

    logging.debug(f"Training complete. Loss: {avg_train_loss:.4f}, Pseudo-gradients: {len(pseudo_gradients)}")

    return pseudo_gradients, avg_train_loss


# --- 1. Update the MODEL_REGISTRY to include text models ---
MODEL_REGISTRY = {
    # Image Models
    "simple_cnn": {
        "class": SimpleCNN,
        "domain": "image",
        "supported_datasets": ["cifar", "celeba", "camelyon16"],
    },
    "lenet": {
        "class": LeNet,
        "domain": "image",
        "supported_datasets": ["mnist", "fmnist", "cifar", "celeba", "camelyon16"],
    },
    # Text Models
    "text_cnn": {
        "class": TextCNN,
        "domain": "text",
        "supported_datasets": ["ag_news", "trec"],
        # Define required parameters for text models
        "required_params": ["vocab_size", "num_classes", "padding_idx"]
    }
}


def get_image_model(
        model_name: str,
        num_classes: int,
        in_channels: int,
        device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Gets an initialized model instance from the registry using provided parameters.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Model '{model_name}' is not in the registry. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]["class"]

    print(
        f"Instantiating model '{model_name}' with {in_channels} channels and {num_classes} classes.")
    model = model_class(in_channels=in_channels, num_classes=num_classes)

    if device:
        model.to(torch.device(device) if isinstance(device, str) else device)
        print(f"Model moved to device: {device}")

    return model


def get_text_model(
        dataset_name: str,
        num_classes: int,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        **model_kwargs: Any
) -> nn.Module:
    print(f"🧠 Creating text model for dataset: {dataset_name}")
    model: nn.Module

    # --- 1. Instantiate the model on CPU ---
    target_device = torch.device(device) if device else torch.device('cpu')  # Determine target device early

    dataset_lower = dataset_name.lower()
    if dataset_lower in ("ag_news", "trec"):
        if vocab_size is None: raise ValueError("`vocab_size` required.")
        if padding_idx is None: raise ValueError("`padding_idx` required.")
        embed_dim = model_kwargs.get("embed_dim", 100)
        num_filters = model_kwargs.get("num_filters", 100)
        filter_sizes = model_kwargs.get("filter_sizes", [3, 4, 5])
        dropout = model_kwargs.get("dropout", 0.5)
        if not isinstance(filter_sizes, list): raise TypeError("'filter_sizes' must be a list")

        # Create on CPU
        model = TextCNN(
            vocab_size=vocab_size, embed_dim=embed_dim, num_filters=num_filters,
            filter_sizes=filter_sizes, num_class=num_classes, dropout=dropout,
            padding_idx=padding_idx
        )
    else:
        raise NotImplementedError(f"Model not found for dataset {dataset_name}")

    model = model.to(target_device)
    logging.info(f"--- Text Model moved to {target_device} ---")
    log_dtype = model.embedding.weight.dtype  # Get actual dtype after move
    _log_param_stats(model, "embedding.weight", f"After .to({target_device}) ({log_dtype})")

    # --- 5. VERIFY ---
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logging.error(f"--- ❌ VERIFICATION FAILED FOR {name} ---")
            _log_param_stats(model, name, "FAILURE")
            raise RuntimeError(f"NaN/Inf in parameter '{name}' after initialization!")

    logging.info("--- ✅ Text Model verification PASSED ---")
    return model
