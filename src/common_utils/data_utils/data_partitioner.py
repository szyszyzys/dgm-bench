import logging
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List, Tuple  # Added Optional

from src.common_utils.data_utils.image_data_processor import CelebACustom
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import PropertySkewParams, TabularDataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_targets(dataset: Dataset) -> np.ndarray:
    # ... (function remains the same) ...
    # Case 1: The dataset is a Subset wrapper
    if isinstance(dataset, Subset):
        underlying_dataset = dataset.dataset
        indices = dataset.indices

        # If the underlying dataset is our custom CelebA...
        if isinstance(underlying_dataset, CelebACustom):
            prop_idx = underlying_dataset.property_idx
            return underlying_dataset.attr[indices, prop_idx].numpy()

        # For other datasets, try to get targets from the underlying dataset
        if hasattr(underlying_dataset, "targets"):
            # Handle potential list or tensor targets robustly
            targets_ = getattr(underlying_dataset, "targets")
            if isinstance(targets_, torch.Tensor):
                targets_ = targets_.numpy()
            return np.array(targets_)[indices]

        # Fallback for subsets of datasets without a .targets attribute
        return np.array([underlying_dataset[i][1] for i in indices])

    # Case 2: The dataset is the full object itself
    if isinstance(dataset, CelebACustom):
        prop_idx = dataset.property_idx
        return dataset.attr[:, prop_idx].numpy()

    # Fallback for other full datasets
    if hasattr(dataset, "targets"):
        # Handle potential list or tensor targets robustly
        targets_ = getattr(dataset, "targets")
        if isinstance(targets_, torch.Tensor):
            targets_ = targets_.numpy()
        return np.array(targets_)

    # Original (and slowest) fallback if no .targets attribute is found
    return np.array([dataset[i][1] for i in range(len(dataset))])


class FederatedDataPartitioner:
    """Handles the partitioning of a dataset for a federated learning scenario."""

    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        # Ensure _extract_targets is available in your scope
        self.targets = _extract_targets(dataset)
        self.buyer_indices: np.ndarray = np.array([], dtype=int)
        self.test_indices: np.ndarray = np.array([], dtype=int)
        self.client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
        self.client_properties: Dict[int, str] = {}
        random.seed(seed)
        np.random.seed(seed)

    def _split_buyer_pool_for_root_test(self, buyer_pool_all_data: np.ndarray, root_set_fraction: float = 0.5):
        """Helper to split a pool of indices into root and test sets."""
        np.random.shuffle(buyer_pool_all_data)
        split_idx = int(len(buyer_pool_all_data) * root_set_fraction)
        self.buyer_indices = buyer_pool_all_data[:split_idx]
        self.test_indices = buyer_pool_all_data[split_idx:]
        logger.info(
            f"Buyer pool ({len(buyer_pool_all_data)}) split: "
            f"{len(self.buyer_indices)} root (buyer), {len(self.test_indices)} test. "
            f"Fraction: {root_set_fraction:.2f}"
        )

    def _select_buyer_indices_dirichlet(self, available_indices: np.ndarray, size: int, alpha: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Selects a subset of indices for the Buyer based on a Dirichlet distribution.
        Returns (buyer_indices, remaining_seller_indices).
        """
        unique_classes = np.unique(self.targets)
        n_classes = len(unique_classes)

        # 1. Draw a class distribution for this specific Buyer (e.g., highly skewed)
        #
        buyer_class_probs = np.random.dirichlet([alpha] * n_classes)

        # 2. Organize available indices by class
        class_indices_map = {c: [] for c in unique_classes}
        # Filter available targets to match available_indices
        # (Optimization: In a clean run, available_indices is usually the whole set,
        # but we handle the subset case for safety)
        subset_targets = self.targets[available_indices]

        for idx_in_subset, global_idx in enumerate(available_indices):
            label = subset_targets[idx_in_subset]
            class_indices_map[label].append(global_idx)

        buyer_selected = []

        # 3. Select samples per class based on drawn probabilities
        # We try to fulfill the exact 'size' required.

        # Calculate target counts per class
        class_counts = (buyer_class_probs * size).astype(int)
        diff = size - np.sum(class_counts)
        if diff > 0:
            # Add to the majority class or random class
            class_counts[np.argmax(buyer_class_probs)] += diff

        # 4. Fetch indices
        for i, label in enumerate(unique_classes):
            needed = class_counts[i]
            available = class_indices_map[label]

            # If we need more than we have, take all and warn (unlikely if buyer ratio is small)
            if needed > len(available):
                logger.warning(
                    f"Buyer needed {needed} of class {label}, but only {len(available)} available. Taking all.")
                buyer_selected.extend(available)
            else:
                # Shuffle locally to ensure randomness within the class
                np.random.shuffle(available)
                buyer_selected.extend(available[:needed])

        # If we are still short (due to running out of specific classes), fill randomly from remainder
        buyer_selected = np.array(buyer_selected)
        current_count = len(buyer_selected)

        if current_count < size:
            logger.info(f"Dirichlet selection resulted in {current_count}/{size}. Filling remainder randomly.")
            # Find what is left
            all_set = set(available_indices)
            selected_set = set(buyer_selected)
            remainder = list(all_set - selected_set)
            np.random.shuffle(remainder)
            needed = size - current_count
            buyer_selected = np.concatenate([buyer_selected, remainder[:needed]])

        # 5. Determine Seller Pool (Everything not in Buyer)
        # Use boolean mask for speed if indices are aligned, but set diff is safer here
        buyer_set = set(buyer_selected)
        seller_remaining = np.array([i for i in available_indices if i not in buyer_set])

        return buyer_selected.astype(int), seller_remaining.astype(int)

    def partition(self, data_config: Any):
        """Main partitioning dispatcher based on the provided data config."""

        # Imports strictly for type checking if needed, otherwise rely on duck typing
        # if not isinstance(data_config, (ImageDataConfig, TextDataConfig)):
        #     raise TypeError("data_config must be an instance of ImageDataConfig or TextDataConfig")

        buyer_ratio = data_config.buyer_ratio
        buyer_strategy = data_config.buyer_strategy
        seller_strategy = data_config.strategy

        # Handle Subset or Full Dataset
        available_indices = np.array(
            self.dataset.indices if isinstance(self.dataset, Subset) else np.arange(len(self.dataset)))

        # Initial shuffle ensures randomness for IID logic
        np.random.shuffle(available_indices)

        total_samples = len(available_indices)
        buyer_pool_size = int(total_samples * buyer_ratio)

        logger.info(f"Partitioning: Total={total_samples}, Buyer Target={buyer_pool_size}, Strategy={buyer_strategy}")

        # --- Stage 1: Split into Buyer Pool and Seller Pool ---

        buyer_pool_indices = None
        seller_pool_indices = None

        if buyer_strategy.lower() == 'iid':
            # Standard Random Split
            buyer_pool_indices = available_indices[:buyer_pool_size]
            seller_pool_indices = available_indices[buyer_pool_size:]
            logger.info("Buyer strategy: IID (Random Split).")

        elif buyer_strategy.lower() == 'dirichlet':
            # Non-IID Split
            # We look for 'buyer_dirichlet_alpha' in the config, defaulting to 100.0 (effectively IID)
            buyer_alpha = getattr(data_config, 'buyer_dirichlet_alpha', 100.0)
            logger.info(f"Buyer strategy: Dirichlet (Alpha={buyer_alpha}).")

            buyer_pool_indices, seller_pool_indices = self._select_buyer_indices_dirichlet(
                available_indices, buyer_pool_size, buyer_alpha
            )
        else:
            raise ValueError(f"Unsupported buyer strategy: {buyer_strategy}")

        # Sanity Check
        if len(buyer_pool_indices) == 0 and buyer_ratio > 0:
            logger.warning(f"Buyer pool is empty despite ratio {buyer_ratio}.")

        # --- Stage 2: Handle Buyer Pool (Root vs Test) ---
        # Now that we have the pool (IID or Skewed), we split it for the buyer's internal usage
        root_fraction = 0.5
        if hasattr(data_config, 'buyer_config') and data_config.buyer_config:
            root_fraction = data_config.buyer_config.get("root_set_fraction", 0.5)

        self._split_buyer_pool_for_root_test(buyer_pool_indices, root_fraction)

        # --- Stage 3: Partition Seller Pool Among Clients ---
        logger.info(
            f"Partitioning seller pool ({len(seller_pool_indices)} samples) using strategy: '{seller_strategy}'")

        if seller_strategy.lower() == 'property-skew':
            # (Existing Property Skew Logic ...)
            if hasattr(data_config, 'property_skew') and data_config.property_skew:
                self._partition_property_skew(seller_pool_indices, data_config.property_skew)
            else:
                raise ValueError("Property skew selected but config missing.")

        elif seller_strategy.lower() == 'dirichlet':
            alpha = data_config.dirichlet_alpha
            self._partition_by_dirichlet(seller_pool_indices, alpha)

        elif seller_strategy.lower() == 'iid':
            self._partition_iid(seller_pool_indices)

        elif seller_strategy.lower() == 'natural':
            # Natural partitioning by writer/user ID (e.g., FEMNIST)
            self._partition_by_natural_id(seller_pool_indices)

        else:
            raise ValueError(f"Unknown client partitioning strategy: {seller_strategy}")

        return self

    def _partition_by_natural_id(self, seller_pool_indices: np.ndarray):
        """
        Partitions seller data by natural user/writer IDs stored in the dataset.
        Expects the underlying dataset to have a `writer_ids` attribute (e.g., FEMNIST).
        Groups samples by writer, then assigns writers round-robin to num_clients sellers.
        """
        # Locate the writer_ids attribute. Walk through any wrapper layers
        # (Subset, UnifiedDatasetWrapper, ...) until we find an object that
        # actually carries the per-sample writer ids. The previous version
        # only handled a single Subset layer, which broke the FEMNIST path
        # because the dataset is wrapped as
        #     UnifiedDatasetWrapper(_FEMNISTWrapper)
        # and UnifiedDatasetWrapper does not promote arbitrary attributes by
        # default. We now also promote writer_ids inside UnifiedDatasetWrapper
        # itself (see src/common_utils/data_utils/dataset.py), and as a
        # belt-and-braces fallback we walk the wrapper chain here too.
        def _find_writer_ids(obj):
            visited = set()
            while obj is not None and id(obj) not in visited:
                visited.add(id(obj))
                if hasattr(obj, 'writer_ids'):
                    return obj.writer_ids
                # Common wrapper attribute names that point at an inner dataset.
                inner = (getattr(obj, 'dataset', None)
                         or getattr(obj, 'original_dataset', None)
                         or getattr(obj, 'wrapped', None))
                if inner is None:
                    return None
                obj = inner
            return None

        writer_ids_raw = _find_writer_ids(self.dataset)
        if writer_ids_raw is None:
            raise ValueError(
                "Natural partitioning requires dataset with 'writer_ids' attribute. "
                "Walked the wrapper chain and could not find one. Make sure "
                "UnifiedDatasetWrapper is promoting writer_ids from the inner "
                "dataset (see dataset.py:UnifiedDatasetWrapper.__init__)."
            )

        writer_ids = np.array(writer_ids_raw)
        pool_writers = writer_ids[seller_pool_indices]

        # Group indices by writer
        from collections import defaultdict
        writer_to_indices = defaultdict(list)
        for pool_pos, global_idx in enumerate(seller_pool_indices):
            wid = pool_writers[pool_pos]
            writer_to_indices[wid].append(global_idx)

        unique_writers = list(writer_to_indices.keys())
        np.random.shuffle(unique_writers)

        logger.info(f"Natural partition: {len(unique_writers)} writers -> {self.num_clients} sellers")

        # Assign writers round-robin to sellers
        for i, wid in enumerate(unique_writers):
            client_id = i % self.num_clients
            self.client_indices[client_id].extend(writer_to_indices[wid])

        # Log distribution
        sizes = [len(self.client_indices[c]) for c in range(self.num_clients)]
        logger.info(f"Natural partition sizes: min={min(sizes)}, max={max(sizes)}, "
                    f"mean={np.mean(sizes):.0f}, writers_per_seller~{len(unique_writers)/self.num_clients:.1f}")

    def _partition_iid(self, seller_pool_indices: np.ndarray):
        """Partitions seller data evenly and randomly (IID)."""
        logger.info(f"Partitioning {len(seller_pool_indices)} samples for {self.num_clients} clients (IID)...")
        np.random.shuffle(seller_pool_indices)
        split_indices = np.array_split(seller_pool_indices, self.num_clients)
        for client_id in range(self.num_clients):
            self.client_indices[client_id].extend(split_indices[client_id].tolist())

    def _partition_by_dirichlet(self, seller_pool_indices: np.ndarray, alpha: float):
        """
        Partitions data among clients using a Dirichlet distribution to simulate
        Non-IID data heterogeneity.
        """
        logger.info(
            f"Partitioning {len(seller_pool_indices)} seller samples for {self.num_clients} clients using Dirichlet (alpha={alpha})..."
        )
        # Filter targets to only include those in the seller pool
        # Need to handle potential errors if indices are out of bounds
        try:
            pool_targets = self.targets[seller_pool_indices]
        except IndexError as e:
            logger.error(
                f"IndexError accessing targets with seller_pool_indices. Max index: {seller_pool_indices.max() if len(seller_pool_indices) > 0 else 'N/A'}, Targets length: {len(self.targets)}")
            raise e

        unique_targets = np.unique(self.targets)  # Use all potential targets for consistency
        n_classes = len(unique_targets)
        min_target = unique_targets.min()
        max_target = unique_targets.max()
        logger.info(f"Found {n_classes} unique classes in total dataset (min={min_target}, max={max_target}).")

        # Ensure label_distribution covers all potential classes even if some aren't in seller pool
        # Use indices relative to the min_target if labels aren't 0-indexed
        label_distribution = np.random.dirichlet([alpha] * self.num_clients, n_classes)

        # Map actual class labels present in the pool to their indices in the pool
        class_to_pool_indices = {}
        present_classes = np.unique(pool_targets)
        for label in present_classes:
            class_to_pool_indices[label] = seller_pool_indices[np.where(pool_targets == label)[0]]

        # Distribute indices for each class present in the pool across all clients
        assigned_indices_count = 0
        indices_available = set(seller_pool_indices)

        for class_label in present_classes:
            class_indices_in_pool = class_to_pool_indices[class_label]
            np.random.shuffle(class_indices_in_pool)

            # Find the index corresponding to this class label in the full distribution
            try:
                # Assuming unique_targets is sorted
                class_dist_index = np.where(unique_targets == class_label)[0][0]
            except IndexError:
                logger.error(
                    f"Class label {class_label} present in pool but not found in unique targets of full dataset. Skipping.")
                continue

            proportions = label_distribution[class_dist_index]  # Get proportions for this class

            # Calculate samples per client for this class
            total_class_samples = len(class_indices_in_pool)
            samples_per_client = (proportions * total_class_samples).astype(int)
            # Adjust last client to account for rounding errors
            samples_per_client[-1] = total_class_samples - np.sum(samples_per_client[:-1])
            # Ensure non-negative counts
            samples_per_client = np.maximum(0, samples_per_client)
            # Ensure sum matches total
            diff = total_class_samples - samples_per_client.sum()
            samples_per_client[-1] += diff

            start_idx = 0
            for client_id in range(self.num_clients):
                num_samples = samples_per_client[client_id]
                if num_samples == 0: continue  # Skip if no samples assigned

                end_idx = start_idx + num_samples
                assigned_indices = class_indices_in_pool[start_idx:end_idx].tolist()
                self.client_indices[client_id].extend(assigned_indices)
                assigned_indices_count += len(assigned_indices)
                # Remove assigned indices to track leftovers
                # indices_available.difference_update(assigned_indices) # Can be slow
                start_idx = end_idx

        logger.info(f"Dirichlet distribution assigned {assigned_indices_count} / {len(seller_pool_indices)} samples.")
        # Optional: Distribute any remaining samples if rounding caused issues (less likely with adjustment)
        remaining_indices = list(
            indices_available - set(idx for indices in self.client_indices.values() for idx in indices))
        if remaining_indices:
            logger.warning(f"Distributing {len(remaining_indices)} remaining samples due to rounding.")
            np.random.shuffle(remaining_indices)
            extra_indices_split = np.array_split(np.array(remaining_indices), self.num_clients)
            for client_id in range(self.num_clients):
                self.client_indices[client_id].extend(extra_indices_split[client_id].tolist())

    # --- _partition_property_skew (largely the same, ensure has_property exists) ---
    def _partition_property_skew(self, seller_pool_indices: np.ndarray, config: PropertySkewParams):
        """Partitions sellers based on the prevalence of a specific data property."""
        logger.info(
            f"Partitioning {len(seller_pool_indices)} seller samples using property skew: '{config.property_key}'")
        wrapped_dataset = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset

        # --- CRITICAL CHECK: Ensure the dataset object has the required method ---
        if not hasattr(wrapped_dataset, 'has_property'):
            raise AttributeError(
                f"Dataset type {type(wrapped_dataset)} does not have a 'has_property' method required for property skew.")

        num_low_prevalence = self.num_clients - config.num_high_prevalence_clients - config.num_security_attackers

        # --- Define client groups ---
        client_ids = list(range(self.num_clients))
        np.random.shuffle(client_ids)
        # ... (rest of group definition logic is fine) ...
        high_clients = client_ids[:config.num_high_prevalence_clients]
        low_clients = client_ids[
                      config.num_high_prevalence_clients: config.num_high_prevalence_clients + num_low_prevalence]
        security_clients = client_ids[config.num_high_prevalence_clients + num_low_prevalence:]
        for cid in high_clients: self.client_properties[cid] = f"High-Prevalence ({config.property_key})"
        for cid in low_clients: self.client_properties[cid] = f"Low-Prevalence ({config.property_key})"
        for cid in security_clients: self.client_properties[cid] = "Security-Attacker (Standard-Prevalence)"

        # --- Separate seller data by property ---
        prop_true_indices, prop_false_indices = [], []
        for idx in seller_pool_indices:
            try:
                if wrapped_dataset.has_property(idx, property_key=config.property_key):
                    prop_true_indices.append(idx)
                else:
                    prop_false_indices.append(idx)
            except Exception as e:
                logger.error(f"Error checking property for index {idx}: {e}")
                # Decide how to handle errors - skip index, assume false? Assuming false for now.
                prop_false_indices.append(idx)

        np.random.shuffle(prop_true_indices)
        np.random.shuffle(prop_false_indices)
        logger.info(
            f"Separated seller pool: {len(prop_true_indices)} with property, {len(prop_false_indices)} without.")

        # --- Distribute data ---
        # Need to handle potential division by zero if num_clients is 0
        if self.num_clients == 0:
            logger.warning("Number of clients is 0, cannot distribute data.")
            return

        samples_per_client = len(seller_pool_indices) // self.num_clients
        if samples_per_client == 0 and len(seller_pool_indices) > 0:
            logger.warning(
                f"Seller pool size ({len(seller_pool_indices)}) is smaller than num_clients ({self.num_clients}). Some clients might get 0 samples initially.")

        true_ptr, false_ptr = 0, 0

        # --- assign_data function (remains the same, but needs safety checks) ---
        def assign_data(client_list, prevalence):
            nonlocal true_ptr, false_ptr
            total_true_available = len(prop_true_indices)
            total_false_available = len(prop_false_indices)

            for client_id in client_list:
                # Calculate desired number, ensuring it doesn't exceed total samples
                num_prop_true_desired = int(samples_per_client * prevalence)
                num_prop_false_desired = samples_per_client - num_prop_true_desired

                # Calculate actually available number
                num_prop_true_avail = total_true_available - true_ptr
                num_prop_false_avail = total_false_available - false_ptr

                # Take minimum of desired and available
                num_prop_true_actual = min(num_prop_true_desired, num_prop_true_avail)
                num_prop_false_actual = min(num_prop_false_desired, num_prop_false_avail)

                # Adjust one if the sum is less than samples_per_client due to availability limits
                current_total = num_prop_true_actual + num_prop_false_actual
                if current_total < samples_per_client:
                    # Try to add more true samples if possible
                    can_add_true = total_true_available - (true_ptr + num_prop_true_actual)
                    needed = samples_per_client - current_total
                    add_true = min(needed, can_add_true)
                    num_prop_true_actual += add_true
                    current_total += add_true

                    # Try to add more false samples if still needed
                    if current_total < samples_per_client:
                        can_add_false = total_false_available - (false_ptr + num_prop_false_actual)
                        needed = samples_per_client - current_total
                        add_false = min(needed, can_add_false)
                        num_prop_false_actual += add_false

                # Assign samples with the property
                end_true = true_ptr + num_prop_true_actual
                if end_true > total_true_available: end_true = total_true_available  # Boundary check
                self.client_indices[client_id].extend(prop_true_indices[true_ptr:end_true])
                true_ptr = end_true

                # Assign samples without the property
                end_false = false_ptr + num_prop_false_actual
                if end_false > total_false_available: end_false = total_false_available  # Boundary check
                self.client_indices[client_id].extend(prop_false_indices[false_ptr:end_false])
                false_ptr = end_false

        assign_data(high_clients, config.high_prevalence_ratio)
        assign_data(low_clients, config.low_prevalence_ratio)
        assign_data(security_clients, config.standard_prevalence_ratio)

        # Optional: Distribute remaining samples if division wasn't perfect
        remaining_true = prop_true_indices[true_ptr:]
        remaining_false = prop_false_indices[false_ptr:]
        remaining_all = np.concatenate((remaining_true, remaining_false))
        if len(remaining_all) > 0:
            logger.warning(f"Distributing {len(remaining_all)} remaining property skew samples.")
            np.random.shuffle(remaining_all)
            extra_indices_split = np.array_split(remaining_all, self.num_clients)
            for client_id in range(self.num_clients):
                self.client_indices[client_id].extend(extra_indices_split[client_id].tolist())

    def get_splits(self) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
        """Returns the final buyer (root), client, and test index splits."""
        return self.buyer_indices, self.client_indices, self.test_indices


class TabularDataPartitioner:
    """
    Handles partitioning of tabular data, decoupling buyer and seller distribution.
    """

    def __init__(self, dataset: Dataset, features: pd.DataFrame, targets: pd.Series,
                 num_clients: int, seed: int = 42):
        self.dataset = dataset
        self.features = features
        self.targets = targets  # This should be the pandas Series for stratification
        self.num_clients = num_clients
        self.seed = seed  # Use the seed
        np.random.seed(seed)
        random.seed(seed)  # Also seed python's random

        self.buyer_indices: np.ndarray = np.array([], dtype=int)  # Final buyer root indices
        self.test_indices: np.ndarray = np.array([], dtype=int)  # Final test indices
        self.client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

        logging.info(f"TabularDataPartitioner initialized with {len(self.dataset)} samples.")

    def partition(self, data_config: TabularDataConfig):
        """
        Partitions the dataset based on the provided TabularDataConfig.
        """
        buyer_ratio = data_config.buyer_ratio
        buyer_strategy = data_config.buyer_strategy
        seller_strategy = data_config.strategy

        all_indices = np.arange(len(self.dataset))
        np.random.shuffle(all_indices)

        # --- Stage 1: Split into Buyer Pool and Seller Pool ---
        buyer_pool_size = int(len(all_indices) * buyer_ratio)
        # Add checks similar to FederatedDataPartitioner
        if buyer_pool_size == 0 and buyer_ratio > 0:
            logger.warning(f"Buyer ratio {buyer_ratio} resulted in 0 samples for buyer pool.")
        elif buyer_pool_size >= len(all_indices):
            raise ValueError(f"Buyer ratio {buyer_ratio} is too high, leaves no data for sellers.")

        buyer_pool_indices = all_indices[:buyer_pool_size]
        seller_pool_indices = all_indices[buyer_pool_size:]
        logger.info(
            f"Initial split: {len(buyer_pool_indices)} buyer pool, {len(seller_pool_indices)} seller pool."
        )

        # --- Stage 2: Handle Buyer Pool (Split into Root/Test) ---
        # Assuming buyer is generally IID for tabular for now
        if buyer_strategy.lower() == 'iid':
            # Split this pool into the final buyer (root) and test sets
            root_fraction = data_config.buyer_config.get("root_set_fraction", 0.5) if hasattr(data_config,
                                                                                              'buyer_config') and data_config.buyer_config else 0.5
            self._split_buyer_pool_for_root_test(buyer_pool_indices, root_fraction)  # Using default 0.5 fraction
            logger.info(f"Buyer strategy: IID. Using randomly sampled indices.")
        elif buyer_strategy.lower() == 'dirichlet':
            logger.warning(f"Buyer strategy 'dirichlet' requested for tabular, treating as IID for now.")
            root_fraction = data_config.buyer_config.get("root_set_fraction", 0.5) if hasattr(data_config,
                                                                                              'buyer_config') and data_config.buyer_config else 0.5
            self._split_buyer_pool_for_root_test(buyer_pool_indices, root_fraction)
        else:
            raise ValueError(f"Unsupported buyer strategy: {buyer_strategy}")

        # --- Stage 3: Partition Seller Pool Among Clients ---
        logger.info(f"Partitioning seller pool using strategy: '{seller_strategy}'")
        if seller_strategy.lower() == 'property_skew':
            # Use the parameters directly from the config
            self._partition_property_skew(seller_pool_indices, data_config.property_skew)
        elif seller_strategy.lower() == 'dirichlet':
            alpha = data_config.dirichlet_alpha
            self._partition_dirichlet(seller_pool_indices, alpha)
        elif seller_strategy.lower() == 'iid':
            self._partition_iid(seller_pool_indices)  # Use IID method
        else:
            raise ValueError(f"Unknown partitioning strategy: {seller_strategy}")

        # Log final client sizes
        for client_id in range(self.num_clients):
            logger.debug(f"Client {client_id} final size: {len(self.client_indices[client_id])}")

        return self

    # --- ADDED _split_buyer_pool_for_root_test (Identical to FederatedDataPartitioner) ---
    def _split_buyer_pool_for_root_test(self, buyer_pool_all_data: np.ndarray, root_set_fraction: float = 0.5):
        """Helper to split a pool of indices into root and test sets."""
        np.random.shuffle(buyer_pool_all_data)
        split_idx = int(len(buyer_pool_all_data) * root_set_fraction)
        self.buyer_indices = buyer_pool_all_data[:split_idx]  # Final buyer (root) set
        self.test_indices = buyer_pool_all_data[split_idx:]
        logger.info(
            f"Buyer pool ({len(buyer_pool_all_data)}) split: "
            f"{len(self.buyer_indices)} root (buyer), {len(self.test_indices)} test. "
            f"Fraction: {root_set_fraction:.2f}"
        )

    # --- ADDED _partition_iid (Identical to FederatedDataPartitioner) ---
    def _partition_iid(self, seller_pool_indices: np.ndarray):
        """Partitions seller data evenly and randomly (IID)."""
        logger.info(f"Partitioning {len(seller_pool_indices)} samples for {self.num_clients} clients (IID)...")
        np.random.shuffle(seller_pool_indices)
        split_indices = np.array_split(seller_pool_indices, self.num_clients)
        for client_id in range(self.num_clients):
            self.client_indices[client_id].extend(split_indices[client_id].tolist())

    # --- _partition_dirichlet (Adapted for pandas targets) ---
    def _partition_dirichlet(self, seller_pool_indices: np.ndarray, alpha: float):
        """Partitions seller data using a Dirichlet distribution."""
        logger.info(
            f"Partitioning {len(seller_pool_indices)} seller samples for {self.num_clients} clients using Dirichlet (alpha={alpha})..."
        )

        # Use pandas iloc to get targets corresponding to seller indices
        try:
            pool_targets = self.targets.iloc[seller_pool_indices].to_numpy()
        except IndexError as e:
            logger.error(
                f"IndexError accessing targets with seller_pool_indices. Max index: {seller_pool_indices.max() if len(seller_pool_indices) > 0 else 'N/A'}, Targets length: {len(self.targets)}")
            raise e

        unique_targets = np.unique(self.targets)  # Use all potential targets
        n_classes = len(unique_targets)
        min_target = unique_targets.min()
        max_target = unique_targets.max()
        logger.info(f"Found {n_classes} unique classes in total dataset (min={min_target}, max={max_target}).")

        # --- Rest of Dirichlet logic is identical to FederatedDataPartitioner ---
        label_distribution = np.random.dirichlet([alpha] * self.num_clients, n_classes)
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
                logger.error(f"Class label {class_label} present in pool but not found in unique targets. Skipping.")
                continue
            proportions = label_distribution[class_dist_index]
            total_class_samples = len(class_indices_in_pool)
            samples_per_client = (proportions * total_class_samples).astype(int)
            samples_per_client[-1] = total_class_samples - np.sum(samples_per_client[:-1])
            samples_per_client = np.maximum(0, samples_per_client)
            diff = total_class_samples - samples_per_client.sum()
            samples_per_client[-1] += diff

            start_idx = 0
            for client_id in range(self.num_clients):
                num_samples = samples_per_client[client_id]
                if num_samples == 0: continue
                end_idx = start_idx + num_samples
                assigned_indices = class_indices_in_pool[start_idx:end_idx].tolist()
                self.client_indices[client_id].extend(assigned_indices)
                assigned_indices_count += len(assigned_indices)
                start_idx = end_idx

        logger.info(f"Dirichlet distribution assigned {assigned_indices_count} / {len(seller_pool_indices)} samples.")
        remaining_indices = list(
            indices_available - set(idx for indices in self.client_indices.values() for idx in indices))
        if remaining_indices:
            logger.warning(f"Distributing {len(remaining_indices)} remaining Dirichlet samples.")
            np.random.shuffle(remaining_indices)
            extra_indices_split = np.array_split(np.array(remaining_indices), self.num_clients)
            for client_id in range(self.num_clients):
                self.client_indices[client_id].extend(extra_indices_split[client_id].tolist())

    # --- _partition_property_skew (Adapted for pandas features) ---
    def _partition_property_skew(self, seller_pool_indices: np.ndarray, config: PropertySkewParams):
        """Partitions sellers based on a binary feature (property)."""

        # Now you can read directly from the dataclass object
        if not config:
            raise ValueError("Property skew strategy selected, but 'property_skew' config is None.")

        # Read directly from dataclass attributes
        prop_key = config.property_key
        high_prevalence_ratio = config.high_prevalence_ratio
        low_prevalence_ratio = config.low_prevalence_ratio
        standard_prevalence_ratio = config.standard_prevalence_ratio
        num_high_prevalence_clients = config.num_high_prevalence_clients
        num_security_attackers = config.num_security_attackers

        logger.info(f"Partitioning {len(seller_pool_indices)} seller samples using property skew: '{prop_key}'")

        # --- Define client groups ---
        # The rest of your function logic (defining num_low_prevalence,
        # separating data, and calling assign_data) works perfectly
        # as-is because it uses the local variables defined above.
        num_low_prevalence = self.num_clients - num_high_prevalence_clients - num_security_attackers
        if num_low_prevalence < 0:
            raise ValueError("Sum of high prevalence clients and security attackers exceeds total clients.")

        client_ids = list(range(self.num_clients))
        np.random.shuffle(client_ids)
        high_clients = client_ids[:num_high_prevalence_clients]
        low_clients = client_ids[num_high_prevalence_clients: num_high_prevalence_clients + num_low_prevalence]
        security_clients = client_ids[num_high_prevalence_clients + num_low_prevalence:]
        # Add client properties dictionary if needed for tracking
        self.client_properties: Dict[int, str] = {}  # Initialize if not present
        for cid in high_clients: self.client_properties[cid] = f"High-Prevalence ({prop_key})"
        for cid in low_clients: self.client_properties[cid] = f"Low-Prevalence ({prop_key})"
        for cid in security_clients: self.client_properties[cid] = "Security-Attacker (Standard-Prevalence)"

        # --- Separate seller data by property using pandas ---
        try:
            seller_features = self.features.iloc[seller_pool_indices]
            # Assuming the property key corresponds to a binary feature (0 or 1)
            prop_true_indices = seller_features[seller_features[prop_key] == 1].index.to_numpy()
            prop_false_indices = seller_features[seller_features[prop_key] == 0].index.to_numpy()
        except KeyError:
            logger.error(f"Property key '{prop_key}' not found as a column in the features DataFrame.")
            raise
        except Exception as e:
            logger.error(f"Error separating data by property '{prop_key}': {e}")
            raise

        np.random.shuffle(prop_true_indices)
        np.random.shuffle(prop_false_indices)
        logger.info(
            f"Separated seller pool: {len(prop_true_indices)} with property, {len(prop_false_indices)} without.")

        # --- Distribute data using assign_data helper (Identical logic as FederatedDataPartitioner) ---
        if self.num_clients == 0:
            logger.warning("Number of clients is 0, cannot distribute data.")
            return
        samples_per_client = len(seller_pool_indices) // self.num_clients
        if samples_per_client == 0 and len(seller_pool_indices) > 0:
            logger.warning(
                f"Seller pool size ({len(seller_pool_indices)}) < num_clients ({self.num_clients}). Some clients might get 0 samples.")

        true_ptr, false_ptr = 0, 0

        # --- assign_data function (Identical to FederatedDataPartitioner version with safety checks) ---
        def assign_data(client_list, prevalence):
            nonlocal true_ptr, false_ptr
            total_true_available = len(prop_true_indices)
            total_false_available = len(prop_false_indices)

            for client_id in client_list:
                num_prop_true_desired = int(samples_per_client * prevalence)
                num_prop_false_desired = samples_per_client - num_prop_true_desired
                num_prop_true_avail = total_true_available - true_ptr
                num_prop_false_avail = total_false_available - false_ptr
                num_prop_true_actual = min(num_prop_true_desired, num_prop_true_avail)
                num_prop_false_actual = min(num_prop_false_desired, num_prop_false_avail)
                current_total = num_prop_true_actual + num_prop_false_actual
                if current_total < samples_per_client:
                    can_add_true = total_true_available - (true_ptr + num_prop_true_actual)
                    needed = samples_per_client - current_total
                    add_true = min(needed, can_add_true)
                    num_prop_true_actual += add_true
                    current_total += add_true
                    if current_total < samples_per_client:
                        can_add_false = total_false_available - (false_ptr + num_prop_false_actual)
                        needed = samples_per_client - current_total
                        add_false = min(needed, can_add_false)
                        num_prop_false_actual += add_false
                end_true = true_ptr + num_prop_true_actual
                if end_true > total_true_available: end_true = total_true_available
                self.client_indices[client_id].extend(prop_true_indices[true_ptr:end_true])
                true_ptr = end_true
                end_false = false_ptr + num_prop_false_actual
                if end_false > total_false_available: end_false = total_false_available
                self.client_indices[client_id].extend(prop_false_indices[false_ptr:end_false])
                false_ptr = end_false

        assign_data(high_clients, high_prevalence_ratio)
        assign_data(low_clients, low_prevalence_ratio)
        assign_data(security_clients, standard_prevalence_ratio)

        # Optional: Distribute remaining samples
        remaining_true = prop_true_indices[true_ptr:]
        remaining_false = prop_false_indices[false_ptr:]
        remaining_all = np.concatenate((remaining_true, remaining_false))
        if len(remaining_all) > 0:
            logger.warning(f"Distributing {len(remaining_all)} remaining property skew samples.")
            np.random.shuffle(remaining_all)
            extra_indices_split = np.array_split(remaining_all, self.num_clients)
            for client_id in range(self.num_clients):
                self.client_indices[client_id].extend(extra_indices_split[client_id].tolist())

    # --- get_splits (remains the same) ---
    def get_splits(self) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
        """Returns the final buyer (root), client, and test index splits."""
        return self.buyer_indices, self.client_indices, self.test_indices
