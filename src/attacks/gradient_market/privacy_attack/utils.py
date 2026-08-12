import logging
import re
from typing import Any, Callable, Dict
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset

from src.common_utils import set_seed
# --- 1. Import the necessary functions from your project ---
# Make sure your project structure allows these imports.
from experiments.gradient_market.automate_exp.config_parser import load_config
from experiments.gradient_market.run_exp import setup_data_and_model

# In attack_analysis.py

# --- Global cache to avoid re-loading and re-splitting data for the same run ---
_ground_truth_cache = {}


def get_run_config(run_data: Dict[str, Any]) -> "AppConfig":
    """Loads the specific config file saved for a given experiment run."""
    config_path = run_data['run_path'] / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found in {run_data['run_path']}. "
                                "Please re-run the experiment with the code modification to save it.")
    return load_config(str(config_path))


def reconstruct_run_setup(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the seed and config to deterministically reconstruct the data splits and model factory.
    This is the core of the ground truth reconstruction.
    """
    run_path_str = str(run_data['run_path'])

    # Check cache first to avoid redundant computation
    if run_path_str in _ground_truth_cache:
        return _ground_truth_cache[run_path_str]

    # 1. Load the exact config used for the run
    cfg = get_run_config(run_data)

    # 2. Parse the seed from the run directory name
    match = re.search(r'seed_(\d+)', run_path_str)
    if not match:
        raise ValueError(f"Could not parse seed from path: {run_path_str}")
    seed = int(match.group(1))

    # 3. SET THE SEED. This is critical for reproducibility.
    set_seed(seed)

    # 4. Call your original setup function. It will now produce the exact same
    #    data splits and model factory as it did during the experiment.
    _, seller_loaders, _, model_factory, _, _ = setup_data_and_model(cfg)

    # 5. Store the results in the cache
    reconstructed_setup = {
        "model_factory": model_factory,
        "seller_loaders": seller_loaders,
        "full_dataset": next(iter(seller_loaders.values())).dataset.dataset
    }
    _ground_truth_cache[run_path_str] = reconstructed_setup

    return reconstructed_setup


# --- IMPLEMENTATION OF YOUR THREE FACTORY FUNCTIONS ---

def model_factory(run_data: Dict[str, Any]) -> Callable[[], nn.Module]:
    """
    Returns an instance of your model architecture by reconstructing the run setup.
    """
    setup = reconstruct_run_setup(run_data)
    return setup["model_factory"]


def get_buyer_dataset(run_data: Dict[str, Any]) -> Dataset:
    """
    Returns the buyer's root dataset (the full dataset before splitting).
    """
    setup = reconstruct_run_setup(run_data)
    return setup["full_dataset"]


def get_seller_ground_truth(run_data: Dict[str, Any], seller_id: str) -> Dict[str, Any]:
    """
    Returns ground truth info for a seller by reconstructing the data split.
    """
    setup = reconstruct_run_setup(run_data)
    seller_loaders = setup["seller_loaders"]

    # The seller_id in logs is like 'adv_0', 'bn_1'. The loader key is the numeric ID.
    seller_numeric_id = int(seller_id.split('_')[1])

    if seller_numeric_id not in seller_loaders:
        raise ValueError(f"Could not find loader for seller ID {seller_numeric_id}")

    # The seller_loader.dataset is a torch.utils.data.Subset object.
    # The .indices attribute holds the list of indices from the original dataset.
    seller_subset: Subset = seller_loaders[seller_numeric_id].dataset
    member_indices = seller_subset.indices

    return {"member_indices": member_indices}


def get_loss_and_pred(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Calculates loss and predictions for a given model and data loader."""
    model.eval()
    losses = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            losses.extend(loss.cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.array(losses), np.array(predictions), np.array(true_labels)


def extract_mia_features(
        global_model_state_dict: Dict[str, Any],
        individual_grad: torch.Tensor,
        sample_index: int,
        true_label: int,
        dataset: Dataset,  # The full dataset to get specific sample
        seller_round_history: pd.DataFrame,  # For weights/payments
        round_number: int,
        device: str,
        ModelClass: nn.Module,
        learning_rate: float
) -> Optional[Dict[str, Any]]:
    """
    Extracts features for Membership Inference Attack for a single sample and seller.
    Assumes the global model is trained on CrossEntropyLoss.
    """
    model = ModelClass().to(device)
    model.load_state_dict(global_model_state_dict)
    model.eval()

    # Get the specific sample
    try:
        sample_input, _ = dataset[sample_index]
        sample_input = sample_input.unsqueeze(0).to(device)  # Add batch dimension
    except IndexError:
        logging.error(f"Sample index {sample_index} not found in dataset.")
        return None

    # Calculate pre-update loss and confidence
    with torch.no_grad():
        outputs_pre = model(sample_input)
        loss_pre = nn.CrossEntropyLoss()(outputs_pre, torch.tensor([true_label]).to(device)).item()
        confidence_pre = torch.softmax(outputs_pre, dim=1)[0, true_label].item()

    # Simulate model update with individual gradient
    # This requires applying the gradient to the global model parameters
    # This is a simplification; a more accurate way would be to get the *model update* from the seller
    # However, for gradient-based attacks, we use the gradient directly.
    # We are simulating the *effect* if this gradient was applied directly.
    temp_model = ModelClass().to(device)
    temp_model.load_state_dict(global_model_state_dict)
    optimizer = torch.optim.SGD(temp_model.parameters(), lr=learning_rate)

    # Apply individual_grad to temp_model (requires careful handling)
    # This part is tricky. A global model update from a *single* seller gradient is not typical in FL.
    # More realistic: The buyer sees the *aggregated* gradient, and wants to infer membership
    # based on the *difference* in global model state, or by trying to approximate a seller's model.
    # For a direct "buyer sees individual grad" scenario:
    with torch.no_grad():
        for i, (name, param) in enumerate(temp_model.named_parameters()):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.data.copy_(individual_grad[i])  # Assuming individual_grad is a list/tuple of tensors

    optimizer.step()

    # Calculate post-update loss and confidence
    temp_model.eval()
    with torch.no_grad():
        outputs_post = temp_model(sample_input)
        loss_post = nn.CrossEntropyLoss()(outputs_post, torch.tensor([true_label]).to(device)).item()
        confidence_post = torch.softmax(outputs_post, dim=1)[0, true_label].item()

    # Marketplace Features
    # Filter seller_round_history for the current round
    round_data = seller_round_history[seller_round_history['round'] == round_number]
    assigned_weight = round_data['assigned_weight'].iloc[0] if not round_data.empty else 0.0
    payment_received = round_data['payment_received'].iloc[0] if not round_data.empty else 0.0

    features = {
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "loss_diff": loss_pre - loss_post,
        "confidence_pre": confidence_pre,
        "confidence_post": confidence_post,
        "confidence_diff": confidence_post - confidence_pre,
        "assigned_weight": assigned_weight,
        "payment_received": payment_received,
        "gradient_norm": torch.norm(torch.cat([g.flatten() for g in individual_grad])).item()  # Norm of flattened grad
        # Add more features like statistics of gradient per layer if needed
    }
    return features


def extract_pia_features(
        individual_grad: torch.Tensor,
        seller_round_history: pd.DataFrame,
        round_number: int,
        ModelClass: nn.Module,  # Needed to interpret gradient structure
        device: str
) -> Optional[Dict[str, Any]]:
    """
    Extracts features for Property Inference Attack for a single seller.
    Focuses on gradient statistics and marketplace signals.
    """
    if individual_grad is None:
        return None

    features = {}

    # Gradient Statistics (example features)
    flattened_grad = torch.cat([g.flatten() for g in individual_grad])
    features["grad_norm"] = torch.norm(flattened_grad).item()
    features["grad_mean"] = torch.mean(flattened_grad).item()
    features["grad_std"] = torch.std(flattened_grad).item()
    features["grad_min"] = torch.min(flattened_grad).item()
    features["grad_max"] = torch.max(flattened_grad).item()

    # Marketplace Features
    round_data = seller_round_history[seller_round_history['round'] == round_number]
    features["assigned_weight"] = round_data['assigned_weight'].iloc[0] if not round_data.empty else 0.0
    features["payment_received"] = round_data['payment_received'].iloc[0] if not round_data.empty else 0.0

    # You could add more sophisticated features here, e.g.:
    # - Cosine similarity of this grad to a 'reference grad' for a known property
    # - Sub-layer specific gradient norms
    # - Changes in buyer model performance on a 'probe set' after this gradient (more complex)

    return features
