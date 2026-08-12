# --- Imports ---
import hashlib
import logging
import numpy as np
import os
import random
import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from typing import (Any, Dict, List, Optional, Tuple, Callable)

# --- HuggingFace datasets dynamic import ---
try:
    from datasets import load_dataset as hf_load

    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# --- Configure logging once ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class StandardFormatDataset(Dataset):
    """A wrapper for text data that provides a uniform interface."""

    def __init__(self, data: List[Tuple[Any, Any]], label_first: bool = True):
        self.data = data
        self.label_first = label_first
        label_index = 0 if self.label_first else 1

        processed_labels = []
        for item in self.data:
            label = item[label_index]

            # Normalize the label format (handle lists/tuples)
            if isinstance(label, (list, tuple)):
                processed_labels.append(label[0])
            else:
                processed_labels.append(label)

        self.targets = np.array(processed_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This method can remain simple; its job is just to return one sample.
        return self.data[idx]


def collate_batch(batch, padding_value):
    labels = []
    texts = []
    lengths = []

    for text_sequence, label in batch:
        labels.append(label)
        texts.append(torch.tensor(text_sequence, dtype=torch.int64))
        lengths.append(len(text_sequence))  # This will now work correctly

    labels = torch.tensor(labels, dtype=torch.int64)
    texts = pad_sequence(texts, batch_first=True, padding_value=padding_value)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    return labels, texts, lengths


def get_cache_path(cache_dir: str, prefix: str, params: Tuple) -> str:
    """Generates a unique cache file path based on parameters."""
    os.makedirs(cache_dir, exist_ok=True)
    param_string = "_".join(map(str, params)).replace("/", "_")
    key = hashlib.md5(param_string.encode()).hexdigest()
    return os.path.join(cache_dir, f"{prefix}_{key}.cache")


def generate_buyer_bias_distribution(num_classes: int, bias_type: str, alpha: float = 0.5) -> Dict[int, float]:
    """
    Generates a class distribution for the buyer.
    (This function replaces the placeholder).
    """
    if bias_type == "dirichlet":
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive.")
        proportions = np.random.dirichlet([alpha] * num_classes)
        return {i: proportions[i] for i in range(num_classes)}
    elif bias_type == "uniform":
        return {i: 1.0 / num_classes for i in range(num_classes)}
    else:
        raise ValueError(f"Unsupported buyer_bias_type: {bias_type}")


def get_text_property_indices(dataset: Any, property_key: str, text_field: str) -> Tuple[List[int], List[int]]:
    """
    Scans a Hugging Face dataset to find indices of samples that contain a specific keyword.

    Returns:
        A tuple of two lists: (indices_with_property, indices_without_property)
    """
    logger.info(f"Scanning dataset for text property (keyword): '{property_key}'...")
    prop_true_indices, prop_false_indices = [], []
    # The property key is case-insensitive for robustness
    keyword = property_key.lower()

    for i, item in enumerate(dataset):
        text_content = item.get(text_field, "")
        if isinstance(text_content, str) and keyword in text_content.lower():
            prop_true_indices.append(i)
        else:
            prop_false_indices.append(i)

    logger.info(f"Found {len(prop_true_indices)} samples with the property and {len(prop_false_indices)} without.")
    return prop_true_indices, prop_false_indices


# --- Main Data Loading and Processing Function (Refactored) ---
@dataclass
class ProcessedTextData:
    """Holds the results of the data processing pipeline."""
    buyer_loader: Optional[DataLoader]
    seller_loaders: Dict[int, Optional[DataLoader]]
    test_loader: Optional[DataLoader]
    class_names: List[str]
    vocab: Vocab
    pad_idx: int
    num_classes: int
    collate_fn: Callable


# --- Data Splitting Logic (Previously defined but unused, now integrated) ---


def _calculate_target_counts(total_samples: int, proportions: Dict[int, float]) -> Dict[int, int]:
    """Helper to calculate exact sample counts per class from float proportions."""
    counts = {cls: int(round(prop * total_samples)) for cls, prop in proportions.items()}
    diff = total_samples - sum(counts.values())
    if diff != 0:
        sorted_classes = sorted(proportions, key=proportions.get, reverse=(diff > 0))
        for i in range(abs(diff)):
            counts[sorted_classes[i % len(sorted_classes)]] += 1 if diff > 0 else -1
    return counts


def construct_text_buyer_set(dataset: List[Tuple[int, Any]], buyer_count: int, buyer_data_mode: str,
                             buyer_bias_distribution: Optional[Dict], seed: int) -> np.ndarray:
    """Constructs the buyer index set based on the specified mode."""
    random.seed(seed)
    np.random.seed(seed)
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)

    if buyer_count <= 0: return np.array([], dtype=int)
    if buyer_count >= total_samples: return all_indices

    if buyer_data_mode == "random":
        return np.random.choice(all_indices, buyer_count, replace=False)

    elif buyer_data_mode == "biased":
        if buyer_bias_distribution is None:
            raise ValueError("`buyer_bias_distribution` is required for 'biased' mode.")

        targets = dataset.targets  # This is fast, robust, and much cleaner
        target_counts = _calculate_target_counts(buyer_count, buyer_bias_distribution)

        buyer_indices_list = []
        for cls, needed_count in target_counts.items():
            class_indices = np.where(targets == cls)[0]
            if len(class_indices) < needed_count:
                logging.warning(f"Class {cls}: Needed {needed_count} but only {len(class_indices)} available.")

            chosen_indices = np.random.choice(class_indices, min(needed_count, len(class_indices)), replace=False)
            buyer_indices_list.extend(chosen_indices)

        # Fill remaining if any class was undersampled
        remaining_needed = buyer_count - len(buyer_indices_list)
        if remaining_needed > 0:
            available_pool = np.setdiff1d(all_indices, np.array(buyer_indices_list))
            fill_indices = np.random.choice(available_pool, min(remaining_needed, len(available_pool)), replace=False)
            buyer_indices_list.extend(fill_indices)

        return np.array(buyer_indices_list)
    else:
        raise ValueError(f"Unknown buyer_data_mode: {buyer_data_mode}")


def split_text_dataset_martfl_discovery(
        dataset: StandardFormatDataset,
        buyer_count: int,
        num_clients: int,
        noise_factor: float,
        buyer_data_mode: str,
        buyer_bias_distribution: Optional[Dict],
        seed: int
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Simulates a data split where seller distributions are noisy mimics of the buyer's.

    Args:
        noise_factor: Quality of discovery, range [0, 1].
                     1.0 = perfect match, 0.0 = maximum noise
    """
    random.seed(seed)
    np.random.seed(seed)

    logging.info(f"Starting martFL discovery split: {len(dataset)} samples, "
                 f"{num_clients} clients, noise_factor={noise_factor:.2f}")

    # 1. Extract and validate labels
    all_indices = np.arange(len(dataset))
    targets = []

    for i, item in enumerate(dataset):
        label = item[1]

        # Handle different label formats
        if isinstance(label, (list, tuple)):
            if len(label) == 0:
                raise ValueError(f"Empty label at index {i}")
            label = label[0]

        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = label.item()

        # Validate it's now a scalar
        if not isinstance(label, (int, float, np.integer)):
            raise ValueError(f"Cannot convert label at index {i} to scalar: {type(label)}")

        targets.append(int(label))

    targets = np.array(targets, dtype=np.int64)
    unique_classes = sorted(list(np.unique(targets)))
    num_classes = len(unique_classes)

    logging.info(f"Found {num_classes} unique classes: {unique_classes}")

    # 2. Construct buyer set
    buyer_indices = construct_text_buyer_set(
        dataset, buyer_count, buyer_data_mode, buyer_bias_distribution, seed
    )
    logging.info(f"Buyer set: {len(buyer_indices)} samples")

    # 3. Define seller pool and calculate buyer's distribution
    seller_pool_indices = np.setdiff1d(all_indices, buyer_indices, assume_unique=True)
    np.random.shuffle(seller_pool_indices)

    logging.info(f"Seller pool: {len(seller_pool_indices)} samples")

    if len(seller_pool_indices) < num_clients:
        raise ValueError(f"Not enough samples for {num_clients} clients. "
                         f"Only {len(seller_pool_indices)} available.")

    # Calculate buyer's class distribution
    if len(buyer_indices) > 0:
        buyer_targets = targets[buyer_indices]
        buyer_class_counts = {cls: np.sum(buyer_targets == cls) for cls in unique_classes}
        base_proportions = np.array([buyer_class_counts.get(c, 0) / len(buyer_indices)
                                     for c in unique_classes])
        logging.info(f"Buyer distribution: {dict(zip(unique_classes, base_proportions))}")
    else:
        logging.warning("Buyer set is empty! Using uniform distribution.")
        base_proportions = np.array([1.0 / num_classes] * num_classes)

    # Ensure no zero proportions (causes Dirichlet issues)
    base_proportions = np.maximum(base_proportions, 1e-6)
    base_proportions /= base_proportions.sum()

    # 4. Calculate safe concentration parameter
    # Map noise_factor [0, 1] to concentration [0.1, 100]
    # Higher noise_factor -> higher concentration -> less noise
    if not (0 <= noise_factor <= 1):
        raise ValueError(f"noise_factor must be in [0, 1], got {noise_factor}")

    if noise_factor >= 0.999:
        concentration = 100.0  # Cap at high value
    elif noise_factor <= 0.001:
        concentration = 0.1  # Cap at low value
    else:
        # Scale: 0 -> 0.1, 0.5 -> 1, 0.9 -> 10, 0.99 -> 100
        concentration = 10 ** (2 * noise_factor - 1)

    logging.info(f"Using Dirichlet concentration: {concentration:.2f}")

    # 5. Create pool organized by class
    seller_pool_by_class = {
        cls: list(seller_pool_indices[targets[seller_pool_indices] == cls])
        for cls in unique_classes
    }

    # Log available samples per class
    for cls, indices in seller_pool_by_class.items():
        logging.debug(f"Class {cls}: {len(indices)} samples available")

    # 6. Calculate samples per client
    samples_per_client = len(seller_pool_indices) // num_clients
    min_samples_per_client = max(10, samples_per_client // 2)  # Minimum threshold

    logging.info(f"Target samples per client: {samples_per_client}")

    # 7. Assign samples to clients
    seller_splits = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        if samples_per_client == 0:
            logging.warning(f"samples_per_client is 0! Skipping client {client_id}")
            continue

        # Generate noisy distribution
        alpha = base_proportions * concentration
        noisy_proportions = np.random.dirichlet(alpha)

        # Calculate target counts
        target_counts = _calculate_target_counts(
            samples_per_client,
            dict(zip(unique_classes, noisy_proportions))
        )

        # Sample from pool
        for cls, count in target_counts.items():
            available = seller_pool_by_class.get(cls, [])
            num_to_take = min(count, len(available))

            if num_to_take > 0:
                assigned_samples = available[:num_to_take]
                seller_splits[client_id].extend(assigned_samples)
                seller_pool_by_class[cls] = available[num_to_take:]

    # 8. Distribute remaining samples more fairly
    remaining_indices = [idx for cls_indices in seller_pool_by_class.values()
                         for idx in cls_indices]

    if remaining_indices:
        logging.info(f"Distributing {len(remaining_indices)} remaining samples")
        np.random.shuffle(remaining_indices)

        # Distribute to clients with fewest samples first
        client_sizes = [(cid, len(indices)) for cid, indices in seller_splits.items()]
        client_sizes.sort(key=lambda x: x[1])  # Sort by size ascending

        for idx in remaining_indices:
            smallest_client = client_sizes[0][0]
            seller_splits[smallest_client].append(idx)
            # Update size and re-sort
            client_sizes[0] = (smallest_client, client_sizes[0][1] + 1)
            client_sizes.sort(key=lambda x: x[1])

    # 9. Validate and log results
    logging.info("Client data distribution:")
    for client_id in range(num_clients):
        indices = seller_splits[client_id]
        if len(indices) < min_samples_per_client:
            logging.warning(f"Client {client_id} has only {len(indices)} samples "
                            f"(below minimum {min_samples_per_client})")

        if len(indices) > 0:
            client_targets = targets[indices]
            class_dist = {cls: np.sum(client_targets == cls) for cls in unique_classes}
            logging.info(f"  Client {client_id}: {len(indices)} samples, "
                         f"distribution: {class_dist}")
        else:
            logging.error(f"Client {client_id} has NO samples!")

    # Final validation
    total_assigned = sum(len(indices) for indices in seller_splits.values())
    if total_assigned != len(seller_pool_indices):
        logging.warning(f"Sample count mismatch: assigned {total_assigned}, "
                        f"pool had {len(seller_pool_indices)}")

    return buyer_indices, seller_splits
