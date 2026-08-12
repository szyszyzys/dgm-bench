import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D NumPy arrays.
    """
    dot_val = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return dot_val / (norm1 * norm2)


def flatten_np(param_tensors: List[torch.Tensor]) -> np.ndarray:
    """Flatten a list of PyTorch tensors into a single 1D NumPy array."""
    flat = []
    for t in param_tensors:
        flat.append(t.view(-1).detach().cpu().numpy())
    return np.concatenate(flat, axis=0)


def unflatten_np(flat_params: np.ndarray, shapes: List[torch.Size]) -> List[np.ndarray]:
    """Unflatten a 1D NumPy array into a list of arrays with specified shapes."""
    idx = 0
    result = []
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        segment = flat_params[idx:idx + size]
        idx += size
        result.append(segment.reshape(shape))
    return result


def global_clip_np(grad: np.ndarray, clip_norm: float) -> np.ndarray:
    """Global L2 norm clipping for a flattened NumPy gradient."""
    norm = np.linalg.norm(grad)
    if norm > clip_norm:
        grad = grad * (clip_norm / norm)
    return grad


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    flat_params = []
    for key, param in state_dict.items():
        flat_params.append(param.detach().cpu().numpy().ravel())
    return np.concatenate(flat_params)


def unflatten_state_dict(model, flat_params: np.ndarray) -> dict:
    new_state_dict = {}
    pointer = 0
    for key, param in model.state_dict().items():
        numel = param.numel()
        # Slice the flat_params to match this parameter's number of elements.
        param_flat = flat_params[pointer:pointer + numel]
        # Reshape to the original shape.
        new_state_dict[key] = torch.tensor(param_flat.reshape(param.shape), dtype=param.dtype)
        pointer += numel
    return new_state_dict


class FederatedEarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, monitor='loss'):
        """
        Args:
            patience (int): Number of rounds to wait after last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            monitor (str): Metric to monitor, 'loss' for minimizing metric or 'acc' for maximizing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0

    def update(self, current_value):
        """
        Update the stopper with the latest metric value.

        Returns:
            bool: True if training should be stopped, otherwise False.
        """
        # For loss, lower is better; for accuracy, higher is better.
        if self.best_score is None:
            self.best_score = current_value
            return False

        if self.monitor == 'loss':
            improvement = self.best_score - current_value
        elif self.monitor == 'acc':
            improvement = current_value - self.best_score
        else:
            raise ValueError("Monitor must be 'loss' or 'acc'")

        if improvement > self.min_delta:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def set_seed(seed: int):
    """Set the seed for random, numpy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects deterministic algorithms when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")


def get_device(args) -> str:
    """
    Returns a string representing the device based on available GPUs and args.gpu_ids.
    The --gpu_ids argument should be a comma-separated string of GPU indices (e.g., "0,1,2").
    """
    if torch.cuda.is_available():
        # Parse the gpu_ids argument into a list of integers.
        gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
        # Set CUDA_VISIBLE_DEVICES so that only these GPUs are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        # After setting CUDA_VISIBLE_DEVICES, PyTorch remaps GPUs starting from 0.
        device_str = "cuda:0"
        print(f"[INFO] Using GPUs: {gpu_ids}. Default device set to {device_str}.")
    else:
        device_str = "cpu"
        print("[INFO] CUDA not available. Using CPU.")
    return device_str


def clear_work_path(path):
    """
    Delete all files and subdirectories in the specified path.
    """
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# ===================================================================
# Tensor and Gradient Manipulation
# ===================================================================

def flatten_tensor(param_list: List[torch.Tensor]) -> torch.Tensor:
    """Flattens a list of tensors into a single 1D tensor."""
    if not param_list: return torch.tensor([])
    return torch.cat([p.flatten() for p in param_list])


def unflatten_tensor(flat_tensor: torch.Tensor, original_shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Reconstructs a list of tensors with original shapes from a flattened tensor."""
    result = []
    offset = 0
    for shape in original_shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        segment = flat_tensor[offset: offset + num_elements]
        result.append(segment.reshape(shape))
        offset += num_elements
    return result


def add_gradient_updates(grad_accumulator: List[torch.Tensor], grad_update: List[torch.Tensor], weight: float = 1.0):
    """In-place: grad_accumulator[i] += weight * grad_update[i]."""
    for acc, upd in zip(grad_accumulator, grad_update):
        if upd is not None:
            acc.add_(upd, alpha=weight)


def clip_gradient_update(grad_update: List[torch.Tensor], clip_norm: Optional[float]) -> List[torch.Tensor]:
    """
    Performs L2-norm clipping on a list of gradient tensors.
    This scales the entire gradient if its total norm exceeds clip_norm.
    """
    # If clip_norm is None or 0, it means "no clipping", so return the original.
    if not clip_norm or clip_norm <= 0:
        return grad_update

    try:
        # We must operate on the flat vector to get the true total norm
        # Ensure gradients are on the same device before cat
        device = grad_update[0].device
        flat_grad = torch.cat([p.data.view(-1).to(device) for p in grad_update])

        # Calculate the total L2 norm
        total_norm = torch.norm(flat_grad, p=2)

        # Calculate the scaling factor
        # (Add 1e-9 to avoid division by zero)
        clip_coef = clip_norm / (total_norm + 1e-9)

        # If the norm is already less than clip_norm, clip_coef will be > 1.
        # We only scale *down*, so we take the minimum.
        if clip_coef < 1.0:
            # Apply the scaling to all original gradient tensors
            return [p.data * clip_coef for p in grad_update]
        else:
            # Norm is already within bounds, return original
            return grad_update

    except Exception as e:
        logging.error(f"Error during L2 norm clipping (norm={clip_norm}): {e}. Returning original gradient.")
        return grad_update


# ===================================================================
# Metrics and Training Control
# ===================================================================

def calculate_kappa(y_true: List, y_pred: List) -> float:
    """Calculates Cohen's Kappa score."""
    return cohen_kappa_score(y_true, y_pred)


class ExperimentLoader:
    def __init__(self, experiment_root_path: str):
        self.experiment_root_path = Path(experiment_root_path)
        if not self.experiment_root_path.exists():
            raise FileNotFoundError(f"Experiment root path not found: {experiment_root_path}")

        self.runs = self._find_runs()
        logging.info(f"Found {len(self.runs)} experiment runs in {self.experiment_root_path}")

    def _find_runs(self) -> List[Path]:
        """Identifies all 'run_X_seed_Y' subdirectories."""
        run_paths = []
        for item in self.experiment_root_path.iterdir():
            if item.is_dir() and item.name.startswith("run_") and "seed" in item.name:
                run_paths.append(item)
        return sorted(run_paths)  # Sort for consistent order

    def load_run_data(self, run_path: Path) -> Dict[str, Any]:
        """
        Loads all available log data for a single experiment run.
        This includes marketplace logs, all seller logs, and provides gradient paths.
        """
        run_data = {
            "run_path": run_path,
            "marketplace": {},
            "sellers": {},
            "gradient_paths": {}  # Stores paths, not actual tensors to avoid memory issues
        }

        logging.info(f"Loading data for run: {run_path.name}")

        # Load Marketplace Logs
        marketplace_log_path = run_path / "training_log.csv"
        if marketplace_log_path.exists():
            run_data["marketplace"]["training_log"] = pd.read_csv(marketplace_log_path)
        else:
            logging.warning(f"Marketplace training log not found for {run_path.name}")

        final_metrics_path = run_path / "final_metrics.json"
        if final_metrics_path.exists():
            with open(final_metrics_path, 'r') as f:
                run_data["marketplace"]["final_metrics"] = json.load(f)
        else:
            logging.warning(f"Marketplace final metrics not found for {run_path.name}")

        # Load Seller Logs
        sellers_dir = run_path / "sellers"
        if sellers_dir.exists():
            for seller_dir in sellers_dir.iterdir():
                if seller_dir.is_dir():
                    seller_id = seller_dir.name
                    seller_history_path = seller_dir / "history" / "round_history.csv"
                    if seller_history_path.exists():
                        run_data["sellers"][seller_id] = pd.read_csv(seller_history_path)
                    else:
                        logging.warning(f"Seller {seller_id} history not found for {run_path.name}")
        else:
            logging.warning(f"Sellers directory not found for {run_path.name}")

        # Collect Gradient Paths
        gradients_base_dir = run_path / "individual_gradients"
        if gradients_base_dir.exists():
            for round_dir in gradients_base_dir.iterdir():
                if round_dir.is_dir() and round_dir.name.startswith("round_"):
                    round_number = int(round_dir.name.split('_')[1])
                    run_data["gradient_paths"][round_number] = {
                        "aggregated_grad": round_dir / "aggregated_grad.pt",
                        "individual_grads": {}
                    }
                    for grad_file in round_dir.iterdir():
                        if grad_file.is_file() and grad_file.name.endswith(
                                "_grad.pt") and not grad_file.name.startswith("aggregated_"):
                            seller_id = grad_file.name.replace("_grad.pt", "")
                            run_data["gradient_paths"][round_number]["individual_grads"][seller_id] = grad_file
        else:
            logging.warning(f"Individual gradients directory not found for {run_path.name}")

        return run_data

    def load_all_runs_data(self) -> List[Dict[str, Any]]:
        """Loads data for all experiment runs."""
        all_runs_data = []
        for run_path in self.runs:
            all_runs_data.append(self.load_run_data(run_path))
        return all_runs_data

    def get_aggregated_final_metrics(self) -> pd.DataFrame:
        """Collects and aggregates final_metrics from all runs."""
        all_final_metrics = []
        for run_path in self.runs:
            final_metrics_path = run_path / "final_metrics.json"
            if final_metrics_path.exists():
                with open(final_metrics_path, 'r') as f:
                    metrics = json.load(f)
                    metrics['run_name'] = run_path.name
                    all_final_metrics.append(metrics)
            else:
                logging.warning(f"Final metrics not found for run: {run_path.name}")
        return pd.DataFrame(all_final_metrics)

    def load_gradient(self, run_data: Dict[str, Any], round_number: int, seller_id: Optional[str] = None) -> Any:
        """
        Loads a specific gradient tensor.
        If seller_id is None, loads the aggregated gradient.
        """
        if round_number not in run_data["gradient_paths"]:
            logging.error(f"Gradients for round {round_number} not found in {run_data['run_path'].name}")
            return None

        round_grads_info = run_data["gradient_paths"][round_number]

        if seller_id is None:
            grad_path = round_grads_info["aggregated_grad"]
        else:
            if seller_id not in round_grads_info["individual_grads"]:
                logging.error(
                    f"Individual gradient for seller {seller_id} in round {round_number} not found in {run_data['run_path'].name}")
                return None
            grad_path = round_grads_info["individual_grads"][seller_id]

        if grad_path.exists():
            return torch.load(grad_path)
        else:
            logging.error(f"Gradient file not found at {grad_path}")
            return None
