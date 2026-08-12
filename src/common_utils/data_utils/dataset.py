# --- Imports ---
import hashlib
import json
import logging
import os
import pickle
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import (Callable, Generator)

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchvision.transforms import v2 as transforms

from src.common_utils.data_utils.data_partitioner import FederatedDataPartitioner, _extract_targets
from src.common_utils.data_utils.image_data_processor import save_data_statistics, CelebACustom
from src.common_utils.data_utils.text_data_processor import ProcessedTextData, get_cache_path, \
    collate_batch, StandardFormatDataset
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig

try:
    from datasets import load_dataset as hf_load

    hf_datasets_available = True
except ImportError:
    hf_datasets_available = False
    logging.warning("HuggingFace 'datasets' library not found. Some dataset loading will fail.")

# --- Setup logging for better feedback ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# In a new file, e.g., common/datasets/wrappers.py
import re
from torch.utils.data import Dataset


class UnifiedDatasetWrapper(Dataset):
    """Wraps various dataset objects to provide a uniform interface."""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.dataset_type = type(original_dataset).__name__

        # --- Create a uniform .targets attribute ---
        if hasattr(original_dataset, 'targets'):
            # For standard torchvision datasets like CIFAR
            self.targets = np.array(original_dataset.targets)
        elif hasattr(original_dataset, 'labels'):
            # For some custom datasets
            self.targets = np.array(original_dataset.labels)
        elif hasattr(original_dataset, 'get_targets'):
            # For our CelebACustom class
            self.targets = original_dataset.get_targets()
        else:
            raise TypeError(f"Could not find a 'targets' or 'labels' attribute on {self.dataset_type}")

        # --- Promote `writer_ids` for FEMNIST natural partitioning ---
        # FEMNIST's _FEMNISTWrapper exposes a per-sample `writer_ids` list so
        # the partitioner can group samples by handwriting author. Without
        # this promotion, the partitioner sees `UnifiedDatasetWrapper` (which
        # does not have writer_ids itself) and crashes with
        #     "Natural partitioning requires dataset with 'writer_ids' attribute".
        # Mirror what we do for `targets` above: forward the attribute up
        # one level so the partitioner can read it directly.
        if hasattr(original_dataset, 'writer_ids'):
            self.writer_ids = original_dataset.writer_ids

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[idx]

    def has_property(self, idx: int, property_key: str) -> bool:
        """
        A uniform method to check if a sample has a given property.
        The logic that was in the partitioner now lives here.
        """
        # A. Generic handler for class-based properties
        if property_key.startswith("class_in_"):
            target_classes = set(map(int, re.findall(r'\d+', property_key)))
            sample_label = self.targets[idx]
            return sample_label in target_classes

        # B. Specific handler for datasets with their own logic (like CelebA)
        elif hasattr(self.original_dataset, 'has_property'):
            return self.original_dataset.has_property(idx, property_key=property_key)

        # C. Fallback error
        else:
            raise NotImplementedError(
                f"Property check for key '{property_key}' is not implemented for {self.dataset_type}"
            )


class _FEMNISTWrapper(torch.utils.data.Dataset):
    """Wraps a HuggingFace FEMNIST split into a standard (image, label) dataset.

    Defined at module level (not inside _get_dataset_loaders) so that
    multiprocessing/joblib-based defenses (e.g. DeepSight's clustering) can
    pickle dataset references. Local classes are not picklable, which caused
    `Can't pickle local object '_get_dataset_loaders.<locals>._FEMNISTWrapper'`
    when DeepSight tried to fan workers out over the dataset.
    """
    def __init__(self, hf_split, transform, label_key, writer_key):
        self.transform = transform
        self.label_key = label_key
        self.writer_key = writer_key
        # Column access — NOT row iteration. Iterating `for row in hf_split`
        # forces HuggingFace's format_row to decode every feature in each row,
        # including the `image` column, which triggers PIL PNG decoding for
        # all ~340K FEMNIST samples (twice — once for targets, once for
        # writer_ids). py-spy traced ~100% of FEMNIST runtime to this exact
        # listcomp via `format_row → decode_example → PIL.Image`. Pulling the
        # column directly skips per-row formatting and never touches the image
        # column. Semantics are byte-identical: same values, same order.
        self.targets = [int(x) for x in hf_split[label_key]]
        self.writer_ids = (
            list(hf_split[writer_key]) if writer_key else [None] * len(hf_split)
        )
        # Pre-decode all images to a uint8 numpy array ONCE, here in __init__.
        # This is the difference between "every batch decodes 64 PNGs" and
        # "every batch slices a numpy array". py-spy on a post-fix worker
        # showed 80%+ of training time was per-sample PNG decode in
        # _FEMNISTWrapper.__getitem__ → HF format_row → PIL.Image.open. With
        # ~340K FEMNIST samples at 28×28 grayscale, the cached array is
        # ~270 MB per wrapper — pickled once to each DataLoader subprocess
        # at first iteration, then sliced essentially for free thereafter.
        # We discard self.hf_split after the decode so it's not pickled.
        images_pil = hf_split["image"]   # column access; one-time PNG decode
        n = len(images_pil)
        # Probe the first image to get shape, then allocate and fill in place.
        first = np.asarray(images_pil[0].convert("L"), dtype=np.uint8)
        self._image_shape = first.shape  # (H, W)
        self._images = np.empty((n,) + first.shape, dtype=np.uint8)
        self._images[0] = first
        for i in range(1, n):
            self._images[i] = np.asarray(images_pil[i].convert("L"), dtype=np.uint8)
        # Drop references so the HF Arrow table and PIL list can be freed.
        del images_pil
        self.hf_split = None  # no longer needed; __getitem__ uses self._images

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # idx may be numpy.int64 (from numpy-array Subset indices); cast to int.
        idx = int(idx)
        # Pure tensor path — no PIL, no torchvision transforms, no HF.
        # The original transform was [ToImage, ToDtype(float32, scale=True),
        # Resize(28,28)].  FEMNIST images are already 28×28, so Resize is a
        # no-op.  ToImage + ToDtype(scale=True) is equivalent to uint8→float32
        # division by 255.  Doing this with torch ops avoids the PIL roundtrip
        # and the torchvision dispatch overhead that py-spy showed at ~26% of
        # training time even after the pre-decode fix.
        img = torch.from_numpy(self._images[idx].copy()).unsqueeze(0).float().div_(255.0)
        # Expand 1-channel to 3-channel for compatibility with FlexibleCNN
        img = img.expand(3, -1, -1).contiguous()
        return img, self.targets[idx]


def _get_dataset_loaders(dataset_name: str, data_root: str) -> Tuple[Dataset, Dataset, int]:
    """
    A helper factory to load standard torchvision datasets and get their properties.
    This makes adding new datasets much easier.
    """
    if dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        raw_train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        raw_test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),  # Mean - correct
                (0.2470, 0.2435, 0.2616)  # Std - FIXED!
            )
        ])
        raw_train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        raw_test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "FEMNIST":
        # FEMNIST from LEAF benchmark — 62 classes (10 digits + 26 lower + 26 upper)
        # Loaded via HuggingFace: "flwrlabs/femnist"
        # Each sample has a writer_id for natural non-IID partitioning.
        from datasets import load_dataset as hf_load_dataset

        hf_ds = hf_load_dataset("flwrlabs/femnist")
        # flwrlabs/femnist only has a "train" split; split it 90/10 for train/test
        if "test" in hf_ds:
            train_split = hf_ds["train"]
            test_split = hf_ds["test"]
        else:
            split = hf_ds["train"].train_test_split(test_size=0.1, seed=42)
            train_split = split["train"]
            test_split = split["test"]

        femnist_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((28, 28)),
        ])

        # flwrlabs/femnist uses "character" (ClassLabel) and "writer_id" (string).
        # Detect the label field name in case the schema differs.
        label_field = "character" if "character" in train_split.column_names else (
            "label" if "label" in train_split.column_names else None
        )
        if label_field is None:
            raise KeyError(
                f"FEMNIST: no label column found. "
                f"Available columns: {train_split.column_names}"
            )
        writer_field = "writer_id" if "writer_id" in train_split.column_names else None

        raw_train_set = _FEMNISTWrapper(train_split, femnist_transform, label_field, writer_field)
        raw_test_set = _FEMNISTWrapper(test_split, femnist_transform, label_field, writer_field)
        num_classes = 62
    elif dataset_name == "CelebA":
        # Note: CelebA does not have a standard test set split in torchvision
        # We handle this by partitioning the 'train' set into train/test splits.
        # The property_key for CelebA is handled in the main function.
        raw_train_set = CelebACustom(root=data_root, split='train', download=True)
        raw_test_set = CelebACustom(root=data_root, split='test', download=True)
        num_classes = 2  # Assuming binary classification on a single attribute
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    train_set = UnifiedDatasetWrapper(raw_train_set)
    test_set = UnifiedDatasetWrapper(raw_test_set)

    return train_set, test_set, num_classes


# In common/datasets/dataset.py

def get_image_dataset(cfg: AppConfig) -> Tuple[DataLoader, Dict[int, DataLoader], DataLoader, Dict, int]:
    """
    A unified function to load, partition, and prepare federated image datasets.
    This version includes caching for the data partitioning step.
    """
    logger.info(f"--- Starting Federated Dataset Setup for '{cfg.experiment.dataset_name}' ---")
    image_cfg = cfg.data.image
    if not image_cfg:
        raise ValueError("Image data configuration ('data.image') is missing.")

    # --- [NEW] VERIFICATION LOGGING ---
    logger.info("📋 Dataset Configuration Verification:")
    logger.info(f"   > Dataset Name: {cfg.experiment.dataset_name}")
    logger.info(f"   > N Sellers:    {cfg.experiment.n_sellers}")
    logger.info(f"   > SELLER Split: Strategy='{image_cfg.strategy}', Alpha={image_cfg.dirichlet_alpha}")
    logger.info(
        f"   > BUYER Split:  Strategy='{image_cfg.buyer_strategy}', Alpha={image_cfg.buyer_dirichlet_alpha}, Ratio={image_cfg.buyer_ratio}")
    if cfg.experiment.use_subset:
        logger.warning(f"   > ⚠️  SUBSET MODE: Using only {cfg.experiment.subset_size} samples!")
    # ----------------------------------

    cache_dir = Path(cfg.data_root) / ".cache"
    # parents=True: create the data_root too if it doesn't exist yet. Without it,
    # mkdir raises FileNotFoundError ("No such file or directory: 'data/.cache'")
    # whenever data_root is missing relative to the run's working directory.
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- UPDATED CACHE KEY ---
    # Include all relevant fields from ImageDataConfig in the hash
    config_params = {
        "dataset": cfg.experiment.dataset_name,
        "n_sellers": cfg.experiment.n_sellers,
        "seed": cfg.seed,
        "use_subset": cfg.experiment.use_subset,
        "subset_size": cfg.experiment.subset_size if cfg.experiment.use_subset else None,
        # Seller params
        "seller_strategy": image_cfg.strategy,
        "seller_dirichlet_alpha": image_cfg.dirichlet_alpha,
        "seller_property_skew_params": asdict(image_cfg.property_skew) if image_cfg.property_skew else None,
        # Buyer params
        "buyer_ratio": image_cfg.buyer_ratio,
        "buyer_strategy": image_cfg.buyer_strategy,
        "buyer_dirichlet_alpha": image_cfg.buyer_dirichlet_alpha,
    }

    # Convert dict to a canonical string and hash it
    config_string = json.dumps(config_params, sort_keys=True)
    config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()
    cache_file = cache_dir / f"{config_hash}.pkl"

    # Load base dataset
    train_set, test_set, num_classes = _get_dataset_loaders(cfg.experiment.dataset_name, cfg.data_root)
    if cfg.experiment.dataset_name == "CelebA":
        property_key = image_cfg.property_skew.property_key
        train_set.set_target_attribute(property_key)
        test_set.set_target_attribute(property_key)
    if cfg.experiment.use_subset:
        num_samples = min(cfg.experiment.subset_size, len(train_set))
        train_set = Subset(train_set, list(range(num_samples)))

    if cache_file.exists() and cfg.use_cache:  # Check use_cache flag
        logger.info(f"✅ Found cached data split. Loading from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        buyer_indices = cached_data['buyer_indices']
        seller_splits = cached_data['seller_splits']
        test_indices = cached_data['test_indices']
        client_properties = cached_data['client_properties']

    else:
        if not cfg.use_cache:
            logger.info("Cache disabled by config. Running partitioning...")
        else:
            logger.info(f"❗️ No cached data split found (Hash: {config_hash}). Running partitioning...")
            logger.info(f"   -> Partitioning Sellers with {image_cfg.strategy} (alpha={image_cfg.dirichlet_alpha})")
            logger.info(
                f"   -> Partitioning Buyer with {image_cfg.buyer_strategy} (alpha={image_cfg.buyer_dirichlet_alpha})")

        # Initialize the partitioner with the (potentially subsetted) training dataset
        partitioner = FederatedDataPartitioner(
            dataset=train_set, num_clients=cfg.experiment.n_sellers, seed=cfg.seed
        )

        # Pass the whole image_cfg object to the updated partition method
        partitioner.partition(data_config=image_cfg)

        # Get all three sets of indices from the partitioner
        buyer_indices, seller_splits, test_indices = partitioner.get_splits()
        client_properties = partitioner.client_properties

        # --- Save the newly created split to the cache file ---
        if cfg.use_cache:
            logger.info(f"💾 Saving new data split to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'buyer_indices': buyer_indices,
                    'seller_splits': seller_splits,
                    'test_indices': test_indices,
                    'client_properties': client_properties
                }, f)

    # ===========================================================================
    # Generate statistics and save to a DEDICATED path
    logger.info("Generating and saving image data split statistics...")

    config_hash = Path(cache_file).stem
    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"

    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        test_indices=test_indices,
        client_properties=client_properties,
        targets=_extract_targets(train_set),
        save_filepath=stats_save_path
    )

    # Create DataLoaders
    batch_size = cfg.training.batch_size
    actual_dataset = train_set.dataset if isinstance(train_set, Subset) else train_set

    workers = min(cfg.data.num_workers, 4) if cfg.data.num_workers > 0 else 2

    # persistent_workers=True keeps DataLoader subprocesses alive across epochs,
    # so HF datasets / PIL / transforms only import once instead of every epoch.
    # Safe for the image path: all image transforms are deterministic
    # (ToImage/ToDtype/Normalize/Resize — no Random* augmentations) and dataset
    # __getitem__ methods do not consume RNG. The DataLoader sampler's shuffle
    # order lives in the main process and is unaffected. The tabular path
    # already uses persistent_workers=True (tabular_data_processor.py:284) —
    # this just makes the image path consistent.
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "prefetch_factor": 4,
        "persistent_workers": workers > 0,
        "pin_memory": True
    }

    logger.info(f"⚡️ DataLoader: workers={workers}, persistent={workers > 0}, pin_memory=True")

    buyer_loader = DataLoader(
        Subset(actual_dataset, buyer_indices),
        shuffle=True,
        **loader_kwargs
    ) if buyer_indices.size > 0 else None

    seller_loaders = {
        cid: DataLoader(
            Subset(actual_dataset, indices),
            shuffle=True,
            **loader_kwargs
        )
        for cid, indices in seller_splits.items() if indices
    }

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **loader_kwargs
    ) if test_set else None

    logger.info(f"✅ Federated dataset setup complete. Using {num_classes} classes.")

    return buyer_loader, seller_loaders, test_loader, stats, num_classes


# --- Helper to extract targets from StandardFormatDataset ---
def _extract_text_targets(dataset: StandardFormatDataset) -> np.ndarray:
    """Extracts targets from the StandardFormatDataset."""
    return dataset.targets


# --- Helper to partition Dirichlet (adapted for StandardFormatDataset) ---
def _partition_text_dirichlet(
        seller_pool_indices: np.ndarray,
        targets: np.ndarray,  # Full dataset targets
        num_clients: int,
        alpha: float,
        client_indices_dict: Dict[int, List[int]]  # Pass dict to modify directly
):
    """Partitions text data indices using Dirichlet."""
    logger.info(
        f"Partitioning {len(seller_pool_indices)} seller samples for {num_clients} clients using Dirichlet (alpha={alpha})...")
    try:
        pool_targets = targets[seller_pool_indices]
    except IndexError as e:
        logger.error(
            f"IndexError accessing targets. Max index: {seller_pool_indices.max() if len(seller_pool_indices) > 0 else 'N/A'}, Targets length: {len(targets)}")
        raise e

    unique_targets = np.unique(targets)
    n_classes = len(unique_targets)
    logger.info(f"Found {n_classes} unique classes in total dataset.")

    label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

    class_to_pool_indices = {}
    present_classes = np.unique(pool_targets)
    for label in present_classes:
        class_to_pool_indices[label] = seller_pool_indices[np.where(pool_targets == label)[0]]

    assigned_indices_count = 0
    indices_available = set(seller_pool_indices)

    for class_label in present_classes:
        class_indices_in_pool = class_to_pool_indices[class_label]
        np.random.shuffle(class_indices_in_pool)
        try:
            class_dist_index = np.where(unique_targets == class_label)[0][0]
        except IndexError:
            logger.error(f"Class label {class_label} present in pool but not in unique targets. Skipping.")
            continue

        proportions = label_distribution[class_dist_index]
        total_class_samples = len(class_indices_in_pool)
        samples_per_client = (proportions * total_class_samples).astype(int)
        samples_per_client[-1] = total_class_samples - np.sum(samples_per_client[:-1])  # Adjust last
        samples_per_client = np.maximum(0, samples_per_client)  # Ensure non-negative
        diff = total_class_samples - samples_per_client.sum()  # Ensure sum matches
        samples_per_client[-1] += diff

        start_idx = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            if num_samples == 0: continue
            end_idx = start_idx + num_samples
            assigned_indices = class_indices_in_pool[start_idx:end_idx].tolist()
            client_indices_dict[client_id].extend(assigned_indices)  # Modify the passed dict
            assigned_indices_count += len(assigned_indices)
            start_idx = end_idx

    logger.info(f"Dirichlet distribution assigned {assigned_indices_count} / {len(seller_pool_indices)} samples.")
    # Distribute remaining samples
    remaining_indices = list(
        indices_available - set(idx for indices in client_indices_dict.values() for idx in indices))
    if remaining_indices:
        logger.warning(f"Distributing {len(remaining_indices)} remaining Dirichlet samples.")
        np.random.shuffle(remaining_indices)
        extra_indices_split = np.array_split(np.array(remaining_indices), num_clients)
        for client_id in range(num_clients):
            client_indices_dict[client_id].extend(extra_indices_split[client_id].tolist())


# --- Helper for IID Split ---
def _partition_text_iid(
        seller_pool_indices: np.ndarray,
        num_clients: int,
        client_indices_dict: Dict[int, List[int]]
):
    """Partitions seller data evenly and randomly (IID)."""
    logger.info(f"Partitioning {len(seller_pool_indices)} samples for {num_clients} clients (IID)...")
    np.random.shuffle(seller_pool_indices)
    split_indices = np.array_split(seller_pool_indices, num_clients)
    for client_id in range(num_clients):
        client_indices_dict[client_id].extend(split_indices[client_id].tolist())


# --- Helper to Split Buyer Pool ---
def _split_buyer_pool_for_root_test(
        buyer_pool_all_data: np.ndarray,
        root_set_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to split a pool of indices into root and test sets."""
    np.random.shuffle(buyer_pool_all_data)
    split_idx = int(len(buyer_pool_all_data) * root_set_fraction)
    buyer_root_indices = buyer_pool_all_data[:split_idx]
    test_indices = buyer_pool_all_data[split_idx:]
    logger.info(
        f"Buyer pool ({len(buyer_pool_all_data)}) split: "
        f"{len(buyer_root_indices)} root (buyer), {len(test_indices)} test. "
        f"Fraction: {root_set_fraction:.2f}"
    )
    return buyer_root_indices, test_indices


# The get_text_dataset function remains unchanged as it was already robust.
def get_text_dataset(cfg: AppConfig) -> ProcessedTextData:
    """
    Loads, processes, caches, and splits a text dataset according to the provided AppConfig.
    """
    # 1. --- Input Validation and Setup ---
    exp_cfg = cfg.experiment
    train_cfg = cfg.training
    text_cfg = cfg.data.text

    if not text_cfg:
        raise ValueError("Text data configuration ('data.text') is missing from the AppConfig.")

    vocab_cfg = text_cfg.vocab
    buyer_percentage = text_cfg.buyer_ratio

    if not (0.0 <= buyer_percentage <= 1.0):
        raise ValueError("buyer_percentage must be between 0 and 1.")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    app_cache_dir = os.path.join(cfg.data_root, ".cache", "get_text_data_set_cache")
    os.makedirs(app_cache_dir, exist_ok=True)
    logging.info(f"Using cache directory: {app_cache_dir}")

    tokenizer = get_tokenizer('basic_english')

    # 2. --- Load Raw Dataset ---
    if exp_cfg.dataset_name.lower() == "ag_news":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for AG_NEWS.")
        ds = hf_load("ag_news", cache_dir=cfg.data_root)
        train_ds_hf, test_ds_hf = ds["train"], ds["test"]
        num_classes, class_names = 4, ['World', 'Sports', 'Business', 'Sci/Tech']
        text_field, label_field = "text", "label"
    elif exp_cfg.dataset_name.lower() == "trec":
        if not hf_datasets_available:
            raise ImportError("HuggingFace 'datasets' library required for TREC.")
        # 1. DEFINE the field names first
        text_field, label_field = "text", "coarse_label"

        # 2. LOAD the Hugging Face dataset
        ds = hf_load("trec", cache_dir=cfg.data_root)
        train_ds_hf, test_ds_hf = ds["train"], ds["test"]

        # 3. GET the actual class information from the dataset
        dataset_features = ds['train'].features[label_field]
        class_names = dataset_features.names
        num_classes = len(class_names)  # This is the true number of classes in the data

    else:
        raise ValueError(f"Unsupported dataset: {exp_cfg.dataset_name}")

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:
            text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
            if isinstance(text_content, str):
                yield (text_content, label_content) if label_fld else text_content

    vocab_cache_params = (
        exp_cfg.dataset_name, vocab_cfg.min_freq, vocab_cfg.unk_token, vocab_cfg.pad_token
    )
    vocab_cache_file = get_cache_path(app_cache_dir, "vocab", vocab_cache_params)
    vocab: Optional[Vocab] = None

    if cfg.use_cache and os.path.exists(vocab_cache_file):
        try:
            logging.info(f"Loading vocabulary from cache: {vocab_cache_file}")
            vocab, pad_idx, unk_idx = torch.load(vocab_cache_file, weights_only=False)
            if not isinstance(vocab, Vocab):
                raise TypeError("Cached object is not a Vocab.")
            logging.info(f"Vocabulary loaded. Size: {len(vocab.get_itos())}")
        except Exception as e:
            logging.warning(f"Failed to load vocab from cache: {e}. Rebuilding.")
            vocab = None

    if vocab is None:
        logging.info("Building vocabulary...")

        def yield_tokens(data_iterator):
            # hf_iterator yields (text, label) tuples — take the text, not the
            # label. The previous unpacking (for _, text) tokenized the integer
            # label and crashed torchtext with "'int' object has no attribute
            # 'lower'", which killed every text (TREC) cell in Table 5.
            for text, _ in data_iterator:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(
            yield_tokens(hf_iterator(train_ds_hf, text_field, label_field)),
            min_freq=vocab_cfg.min_freq,
            specials=[vocab_cfg.unk_token, vocab_cfg.pad_token],
            special_first=True
        )

        unk_idx = vocab[vocab_cfg.unk_token]
        vocab.set_default_index(unk_idx)
        pad_idx = vocab[vocab_cfg.pad_token]

        logging.info(f"Vocabulary built. Size: {len(vocab)}. UNK index: {unk_idx}, PAD index: {pad_idx}.")

        if cfg.use_cache:
            torch.save((vocab, pad_idx, unk_idx), vocab_cache_file)
            logging.info(f"Vocabulary saved to cache: {vocab_cache_file}")

    unk_idx, pad_idx = vocab[vocab_cfg.unk_token], vocab[vocab_cfg.pad_token]

    def hf_iterator(dataset_obj, text_fld, label_fld=None) -> Generator[Any, None, None]:
        for ex in dataset_obj:
            text_content, label_content = ex.get(text_fld), ex.get(label_fld) if label_fld else None
            if isinstance(text_content, str):
                yield (text_content, label_content) if label_fld else text_content

    # 4. --- Numericalize Data ---
    def numericalize_dataset(data_iterator_func: Callable, split_name: str) -> List[Tuple[int, List[int]]]:
        cache_params = (exp_cfg.dataset_name, vocab_cache_file, split_name)
        cache_path = get_cache_path(app_cache_dir, f"num_{split_name}", cache_params)

        logging.info(f"Numericalizing {split_name} data...")
        processed_data = []
        text_pipeline = lambda x: vocab(tokenizer(x))

        for text, label in data_iterator_func():
            if text and label is not None:
                processed_text = text_pipeline(text)
                if processed_text:
                    processed_data.append((processed_text, label))

        if cfg.use_cache:
            with open(cache_path, "wb") as f: pickle.dump(processed_data, f)
        return processed_data

    processed_train_data = numericalize_dataset(lambda: hf_iterator(train_ds_hf, text_field, label_field), "train")
    processed_test_data = numericalize_dataset(lambda: hf_iterator(test_ds_hf, text_field, label_field), "test")
    if not processed_train_data:
        raise ValueError("Training data is empty after processing.")
    standardized_train_data = StandardFormatDataset(processed_train_data, label_first=False)
    standardized_test_data = StandardFormatDataset(processed_test_data, label_first=False)

    # 5. --- Split Data ---
    cache_dir_splits = Path(cfg.data_root) / ".cache_splits"  # Separate cache dir for splits
    cache_dir_splits.mkdir(exist_ok=True)

    # Create cache key based on partitioning parameters
    split_config_params = {
        "dataset": exp_cfg.dataset_name,
        "n_sellers": exp_cfg.n_sellers,
        "seed": cfg.seed,
        "seller_strategy": text_cfg.strategy,
        "seller_dirichlet_alpha": text_cfg.dirichlet_alpha if text_cfg.strategy == 'dirichlet' else None,
        "seller_prop_skew_params": asdict(
            text_cfg.property_skew) if text_cfg.strategy == 'property-skew' and text_cfg.property_skew else None,
        "buyer_ratio": text_cfg.buyer_ratio,
        "buyer_strategy": text_cfg.buyer_strategy,
        "buyer_dirichlet_alpha": text_cfg.buyer_dirichlet_alpha if text_cfg.buyer_strategy == 'dirichlet' else None,
    }
    split_config_string = json.dumps(split_config_params, sort_keys=True)
    split_config_hash = hashlib.md5(split_config_string.encode('utf-8')).hexdigest()
    split_cache_file = cache_dir_splits / f"{split_config_hash}.pkl"

    if cfg.use_cache and split_cache_file.exists():
        logger.info(f"✅ Loading split indices from cache: {split_cache_file}")
        with open(split_cache_file, "rb") as f:
            cached_data = pickle.load(f)
        buyer_indices = cached_data['buyer_indices']
        seller_splits = cached_data['seller_splits']
        test_indices = cached_data['test_indices']
        client_properties = cached_data.get('client_properties', {})  # Load properties if available

    else:
        if not cfg.use_cache:
            logger.info("Cache disabled. Running partitioning...")
        else:
            logger.info(f"❗️ No cached split found or cache disabled. Running partitioning...")

        # --- Perform Partitioning ---
        all_train_indices = np.arange(len(standardized_train_data))
        np.random.shuffle(all_train_indices)

        # Stage 1: Split into Buyer Pool and Seller Pool
        buyer_pool_size = int(len(all_train_indices) * text_cfg.buyer_ratio)
        # Add size checks
        if buyer_pool_size == 0 and text_cfg.buyer_ratio > 0:
            logger.warning("Buyer ratio resulted in 0 samples for buyer pool.")
        elif buyer_pool_size >= len(all_train_indices):
            raise ValueError("Buyer ratio too high, leaves no data for sellers.")

        buyer_pool_indices_all = all_train_indices[:buyer_pool_size]
        seller_pool_indices = all_train_indices[buyer_pool_size:]
        logger.info(f"Initial split: {len(buyer_pool_indices_all)} buyer pool, {len(seller_pool_indices)} seller pool.")

        # Stage 2: Handle Buyer Pool (Split Root/Test)
        # Assuming IID buyer for text simplicity for now
        if text_cfg.buyer_strategy.lower() == 'iid':
            buyer_indices, test_indices = _split_buyer_pool_for_root_test(buyer_pool_indices_all)
            logger.info(f"Buyer strategy: IID.")
        elif text_cfg.buyer_strategy.lower() == 'dirichlet':
            logger.warning("Buyer strategy 'dirichlet' for text currently treated as IID.")
            buyer_indices, test_indices = _split_buyer_pool_for_root_test(buyer_pool_indices_all)
        else:
            raise ValueError(f"Unsupported buyer strategy: {text_cfg.buyer_strategy}")

        # Stage 3: Partition Seller Pool
        logger.info(f"Partitioning seller pool using strategy: '{text_cfg.strategy}'")
        seller_splits = {i: [] for i in range(exp_cfg.n_sellers)}  # Initialize empty dict
        client_properties = {}  # Initialize empty properties

        if text_cfg.strategy.lower() == 'dirichlet':
            all_train_targets = _extract_text_targets(standardized_train_data)
            _partition_text_dirichlet(
                seller_pool_indices,
                all_train_targets,
                exp_cfg.n_sellers,
                text_cfg.dirichlet_alpha,
                seller_splits  # Pass dict to be modified
            )
        elif text_cfg.strategy.lower() == 'iid':
            _partition_text_iid(seller_pool_indices, exp_cfg.n_sellers, seller_splits)
        elif text_cfg.strategy.lower() == 'property-skew':
            # Property skew for text is complex as it needs raw text.
            # This logic should ideally be here, but requires passing raw_train_ds_hf
            # and text_field. Let's assume it's handled for now or raise error.
            raise NotImplementedError("Text property skew partitioning needs refactoring "
                                      "to access raw text data within this function.")
            # If refactored:
            # client_properties = _partition_text_property_skew(...) # This func would need access to raw text
        elif text_cfg.strategy.lower() == 'discovery':
            # Keep your existing discovery split logic if needed, adapting it
            # to operate on seller_pool_indices and populate seller_splits.
            # You'll need to pass standardized_train_data and extract targets within it.
            logger.info("Using 'discovery' split strategy for sellers...")
            train_targets_subset = _extract_text_targets(Subset(standardized_train_data, seller_pool_indices))
            # Adapt split_text_dataset_martfl_discovery or call relevant parts
            # Example adaptation (replace with your actual logic):
            # temp_dataset = Subset(standardized_train_data, seller_pool_indices)
            # _, seller_splits_relative = split_text_dataset_martfl_discovery(...) # Operates on subset indices
            # Convert relative indices back to absolute indices based on seller_pool_indices
            # seller_splits = {cid: seller_pool_indices[rel_indices].tolist() for cid, rel_indices in seller_splits_relative.items()}
            raise NotImplementedError("Adapt 'discovery' split logic for the new structure.")

        else:
            raise ValueError(f"Unknown client partitioning strategy: {text_cfg.strategy}")

        # --- Save split to cache ---
        if cfg.use_cache:
            logger.info(f"💾 Saving new data split to cache: {split_cache_file}")
            with open(split_cache_file, "wb") as f:
                pickle.dump({
                    'buyer_indices': buyer_indices,
                    'seller_splits': seller_splits,
                    'test_indices': test_indices,
                    'client_properties': client_properties
                }, f)

    # 6. --- Generate and Save Statistics ---
    logger.info("Generating and saving data split statistics...")
    config_hash = Path(split_cache_file).stem
    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"
    all_train_targets = _extract_text_targets(standardized_train_data)  # Get targets again
    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        test_indices=test_indices,
        client_properties=client_properties,
        targets=all_train_targets,
        save_filepath=stats_save_path
    )

    # 7. --- Create DataLoaders ---
    collate_fn = partial(collate_batch, padding_value=pad_idx)
    batch_size = train_cfg.batch_size

    workers = min(cfg.data.num_workers, 4) if cfg.data.num_workers > 0 else 2

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": False,
        "collate_fn": collate_fn
    }

    logger.info(f"⚡️ Text Loader: workers={workers}, pin=True")

    buyer_loader = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        buyer_loader = DataLoader(
            Subset(standardized_train_data, buyer_indices.tolist()),
            shuffle=True,
            **loader_kwargs   # <--- Applies optimization AND collate_fn
        )
    if buyer_loader is None: logger.warning("Buyer loader is None.")

    seller_loaders = {}
    non_empty_clients = 0
    for i in range(exp_cfg.n_sellers):
        indices = seller_splits.get(i)
        if indices is not None and len(indices) > 0:
            seller_loaders[i] = DataLoader(
                Subset(standardized_train_data, indices),
                shuffle=True,
                **loader_kwargs  # <--- Applies optimization AND collate_fn
            )
            non_empty_clients += 1
        else:
            seller_loaders[i] = None
            logger.warning(f"Client {i} has no data assigned.")
    logger.info(f"Created DataLoaders for {non_empty_clients} / {exp_cfg.n_sellers} sellers.")

    test_loader = None
    if test_indices is not None and len(test_indices) > 0:
        test_loader = DataLoader(
            Subset(standardized_train_data, test_indices.tolist()),
            shuffle=False,
            **loader_kwargs  # <--- Applies optimization AND collate_fn
        )
    if test_loader is None: logger.warning("Test loader is None.")
    # --- PERFORMANCE OPTIMIZATION END ---

    logging.info("DataLoader creation complete.")

    return ProcessedTextData(
        buyer_loader=buyer_loader,
        seller_loaders=seller_loaders,
        test_loader=test_loader,
        class_names=class_names,
        vocab=vocab,
        pad_idx=pad_idx,
        num_classes=num_classes,
        collate_fn=collate_fn
    )