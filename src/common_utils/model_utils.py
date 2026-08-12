import logging

import torch
from torch import nn


def _log_param_stats(model: nn.Module, param_name: str, stage: str):
    """Helper function to log the stats of a specific parameter."""
    try:
        # Find the parameter by name
        param = dict(model.named_parameters())[param_name]
        stats = {
            "device": param.device,
            "dtype": param.dtype,
            "min": param.min().item(),
            "max": param.max().item(),
            "mean": param.mean().item(),
            "has_nan": torch.isnan(param).any().item(),
            "has_inf": torch.isinf(param).any().item(),
        }
        logging.info(f"--- STATS ({stage}) for '{param_name}': {stats}")
    except KeyError:
        # Try to find the first Linear layer's weight instead
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                logging.info(f"--- STATS ({stage}) for '{name}': ...")
                break
        else:
            logging.debug(f"--- STATS ({stage}): No suitable param found for logging.")


def init_weights(m):
    """
    Applies the correct weight initialization to different layer types.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        logging.info(f"--- ⚡️ Applying Kaiming Uniform to: {m}")
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        logging.info(f"--- ⚡️ Initializing BatchNorm: {m}")
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if m.padding_idx is not None:
            nn.init.constant_(m.weight[m.padding_idx], 0)
