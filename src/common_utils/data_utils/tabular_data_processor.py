import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from urllib import request

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from ucimlrepo import fetch_ucirepo

from src.common_utils.data_utils.data_partitioner import TabularDataPartitioner
from src.common_utils.data_utils.image_data_processor import save_data_statistics
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, TabularDataConfig

logger = logging.getLogger(__name__)


def _build_tabular_npz_from_tgz(name: str, tgz_url: str, out_path: str) -> None:
    """Download the canonical privacytrustlab .tgz for Texas-100 / Purchase-100 and
    build the ``features`` / ``labels`` .npz the loader expects (one-time, cached).

    The public source only ships .tgz archives, not the .npz the loader reads, so we
    fetch, parse, and cache. Labels are stored 0-indexed (the source files are
    1-indexed). Purchase-100: one text file, column 0 = label, columns 1.. = 600
    binary features. Texas-100: ``texas/100/feats`` (6169 binary features) +
    ``texas/100/labels``.
    """
    logger.info(f"  - Building {name} .npz from {tgz_url} (one-time) ...")
    with request.urlopen(tgz_url, timeout=300) as resp:
        raw = resp.read()
    lname = name.lower()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        if "texas" in lname:
            feats = np.loadtxt(tf.extractfile("texas/100/feats"), delimiter=",", dtype=np.float32)
            labels = np.loadtxt(tf.extractfile("texas/100/labels"), dtype=np.int64)
        elif "purchase" in lname:
            arr = np.loadtxt(tf.extractfile("dataset_purchase"), delimiter=",", dtype=np.int64)
            labels, feats = arr[:, 0], arr[:, 1:].astype(np.float32)
        else:
            raise ValueError(f"No .tgz build recipe for tabular dataset '{name}'.")
    labels = labels.astype(np.int64)
    if labels.min() >= 1:
        labels = labels - 1  # store 0-indexed so CrossEntropy targets are in range
    assert feats.shape[0] == labels.shape[0], (feats.shape, labels.shape)
    np.savez_compressed(out_path, features=feats, labels=labels)
    logger.info(f"  - Wrote {out_path}: features {feats.shape}, labels {labels.shape}, "
                f"{len(np.unique(labels))} classes")


def _load_and_prepare_tabular_df(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """A unified helper to load and prepare a tabular DataFrame from various sources."""
    dataset_name = config['name']
    logger.info(f"📦 Loading and preparing the '{dataset_name}' tabular dataset...")

    # --- Fetch data based on its source type ---
    if config['source_type'] == 'uci':
        dataset_obj = fetch_ucirepo(id=config['uci_id'])
        df = pd.concat([dataset_obj.data.features, dataset_obj.data.targets], axis=1)
        df.columns = ["".join(c if c.isalnum() else '_' for c in str(x)) for x in df.columns]
    elif config['source_type'] in ['url', 'local_csv']:
        path = config['url'] if config['source_type'] == 'url' else config['path']
        df = pd.read_csv(path, header='infer' if config.get('has_header', True) else None)
    elif config['source_type'] == 'numpy':
        # Save downloads in project data/ dir, not CWD
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = project_root / "data" / "tabular"
        data_dir.mkdir(parents=True, exist_ok=True)
        local_filename = str(data_dir / f"{dataset_name.lower()}.npz")
        if not os.path.exists(local_filename):
            if config.get('tgz_url'):
                # canonical source ships a .tgz -> download, parse, cache as .npz
                _build_tabular_npz_from_tgz(dataset_name, config['tgz_url'], local_filename)
            else:
                logger.info(f"  - Downloading {dataset_name} dataset from {config['url']}...")
                request.urlretrieve(config['url'], local_filename)
        data = np.load(local_filename)
        features, labels = data[config['data_key']], data[config['labels_key']]
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        df[config['target_column']] = labels
    else:
        raise ValueError(f"Unsupported source_type: {config['source_type']}")

    # --- Apply generic preprocessing steps from config ---
    if 'query' in config: df = df.query(config['query']).dropna()
    if 'missing_value_placeholder' in config:
        df.replace(config['missing_value_placeholder'], np.nan, inplace=True)
        df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if config['target_column'] in categorical_cols:
        categorical_cols.remove(config['target_column'])

    logger.info(f"  - ✅ Dataset '{dataset_name}' loaded successfully. Shape: {df.shape}")
    return df, categorical_cols


def get_tabular_dataset(cfg: AppConfig) -> Tuple[
    Optional[DataLoader], Dict[str, DataLoader], Optional[DataLoader], int, int, Dict[str, int]]:
    logger.info(f"--- Starting Federated Tabular Dataset Setup for '{cfg.experiment.dataset_name}' ---")

    # --- Ensure tabular config exists ---
    tabular_cfg = cfg.data.tabular
    if not tabular_cfg or not isinstance(tabular_cfg, TabularDataConfig):
        raise ValueError("Tabular data configuration ('data.tabular') is missing or invalid.")
    # --- End Check ---

    # 1. Load dataset-specific configuration and raw data
    config_path = Path(tabular_cfg.dataset_config_path)
    if not config_path.is_absolute():
        # Resolve relative to project root (two levels up from src/common_utils/data_utils/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        all_tabular_configs = yaml.safe_load(f)
    dataset_cfg = all_tabular_configs[cfg.experiment.dataset_name.lower()]
    df, categorical_cols = _load_and_prepare_tabular_df(config=dataset_cfg)
    target_col = dataset_cfg['target_column']
    if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found")

    # 2. Preprocess Data (Dummify, Scale, Split)
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )
    X_train_pd = X_train_pd.copy()
    X_test_pd = X_test_pd.copy()

    numerical_cols = X_train_pd.select_dtypes(include=np.number).columns
    if numerical_cols.empty:
        logger.error("CRITICAL: No numerical columns found!")
        raise ValueError("Dataset must contain at least one numerical column")

    logger.info(f"Dataset: {cfg.experiment.dataset_name}")
    logger.info(f"Total features: {len(numerical_cols)}")
    logger.info(f"Training samples: {len(X_train_pd)}, Test samples: {len(X_test_pd)}")

    # Log statistics BEFORE scaling
    logger.info("📊 Data statistics BEFORE scaling (first 5 numerical columns):")
    cols_to_check = numerical_cols[:5]
    train_stats_before = X_train_pd[cols_to_check].describe().loc[['mean', 'std', 'min', 'max']]
    test_stats_before = X_test_pd[cols_to_check].describe().loc[['mean', 'std', 'min', 'max']]
    logger.info(f"\n--- X_train (raw) ---\n{train_stats_before.to_string()}")
    logger.info(f"\n--- X_test (raw) ---\n{test_stats_before.to_string()}")

    # 3. Analyze feature types for Purchase100 and Texas100
    logger.info("🔍 Analyzing feature distribution for privacy benchmark dataset...")

    # Check sparsity
    sparsity = (X_train_pd[numerical_cols] == 0).sum().sum() / (len(X_train_pd) * len(numerical_cols))
    logger.info(f"Data sparsity: {sparsity:.2%} of values are zero")

    # Check value range
    data_min = X_train_pd[numerical_cols].min().min()
    data_max = X_train_pd[numerical_cols].max().max()
    logger.info(f"Value range: [{data_min}, {data_max}]")

    # Identify feature types
    binary_cols = [col for col in numerical_cols if X_train_pd[col].nunique() <= 2]
    bounded_cols = [col for col in numerical_cols
                    if X_train_pd[col].min() >= 0 and X_train_pd[col].max() <= 1
                    and col not in binary_cols]
    continuous_cols = [col for col in numerical_cols
                       if col not in binary_cols and col not in bounded_cols]

    logger.info(f"Feature breakdown:")
    logger.info(f"  - Binary features (0/1): {len(binary_cols)}")
    logger.info(f"  - Bounded features [0,1]: {len(bounded_cols)}")
    logger.info(f"  - Continuous features: {len(continuous_cols)}")

    # 4. Apply appropriate scaling based on dataset
    dataset_name_lower = cfg.experiment.dataset_name.lower()

    if 'purchase' in dataset_name_lower or 'texas' in dataset_name_lower:
        logger.info("📋 Detected privacy benchmark dataset (Purchase100/Texas100)")
        logger.info("   These datasets typically contain binary or bounded features")

        if sparsity > 0.5:  # Very sparse data
            logger.info("   ⚠️ High sparsity detected - skipping StandardScaler to preserve structure")
            logger.info("   Using data as-is (already in [0,1] range)")

        elif len(continuous_cols) > 0:
            # Scale only truly continuous features
            logger.info(f"   Scaling {len(continuous_cols)} continuous features with StandardScaler")
            scaler = StandardScaler()

            X_train_pd[continuous_cols] = X_train_pd[continuous_cols].astype(np.float64)
            X_test_pd[continuous_cols] = X_test_pd[continuous_cols].astype(np.float64)

            X_train_pd[continuous_cols] = scaler.fit_transform(X_train_pd[continuous_cols])
            X_test_pd[continuous_cols] = scaler.transform(X_test_pd[continuous_cols])

            # Clip extreme outliers (protect against rare extreme values)
            X_train_pd[continuous_cols] = X_train_pd[continuous_cols].clip(-10, 10)
            X_test_pd[continuous_cols] = X_test_pd[continuous_cols].clip(-10, 10)
            logger.info(f"   Clipped continuous features to [-10, 10]")
        else:
            logger.info("   ✅ All features are binary/bounded - no scaling needed")

    else:
        # Default behavior for other tabular datasets
        logger.info("Applying StandardScaler to all numerical features")
        scaler = StandardScaler()

        X_train_pd[numerical_cols] = X_train_pd[numerical_cols].astype(np.float64)
        X_test_pd[numerical_cols] = X_test_pd[numerical_cols].astype(np.float64)

        X_train_pd[numerical_cols] = scaler.fit_transform(X_train_pd[numerical_cols])
        X_test_pd[numerical_cols] = scaler.transform(X_test_pd[numerical_cols])

    # Log statistics AFTER scaling
    logger.info("📊 Data statistics AFTER processing (first 5 numerical columns):")
    train_stats_after = X_train_pd[cols_to_check].describe().loc[['mean', 'std', 'min', 'max']]
    test_stats_after = X_test_pd[cols_to_check].describe().loc[['mean', 'std', 'min', 'max']]
    logger.info(f"\n--- X_train ---\n{train_stats_after.to_string()}")
    logger.info(f"\n--- X_test ---\n{test_stats_after.to_string()}")
    feature_to_idx = {col: i for i, col in enumerate(X_train_pd.columns)}

    # 5. Convert to PyTorch Datasets
    X_train_tensor = torch.tensor(X_train_pd.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_pd.values, dtype=torch.long)

    if torch.isnan(X_train_tensor).any() or torch.isinf(X_train_tensor).any():
        logger.error("NaN/Inf detected in training data after processing!")
        nan_mask = torch.isnan(X_train_tensor).any(dim=0)
        inf_mask = torch.isinf(X_train_tensor).any(dim=0)
        problem_cols = [col for i, col in enumerate(X_train_pd.columns)
                        if nan_mask[i] or inf_mask[i]]
        logger.error(f"Problematic columns: {problem_cols}")
        raise ValueError("Cannot proceed with NaN/Inf values in training data. Check data quality.")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(X_test_pd.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_pd.values, dtype=torch.long)

    if torch.isnan(X_test_tensor).any() or torch.isinf(X_test_tensor).any():
        logger.error("NaN/Inf detected in test data after processing!")
        nan_mask = torch.isnan(X_test_tensor).any(dim=0)
        inf_mask = torch.isinf(X_test_tensor).any(dim=0)
        problem_cols = [col for i, col in enumerate(X_test_pd.columns)
                        if nan_mask[i] or inf_mask[i]]
        logger.error(f"Problematic columns: {problem_cols}")
        raise ValueError("Cannot proceed with NaN/Inf values in test data. Check data quality.")

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    input_dim = X_train_tensor.shape[1]
    num_classes = len(torch.unique(y_train_tensor))

    # 4. Partition the training data using the updated partitioner call
    logger.info("Partitioning training data...")
    partitioner = TabularDataPartitioner(
        dataset=train_dataset,
        features=X_train_pd,  # Pass original X_train dataframe for property skew
        targets=y_train_pd,  # Pass original y_train series for stratification/skew
        num_clients=cfg.experiment.n_sellers,
        seed=cfg.seed
    )

    partitioner.partition(data_config=tabular_cfg)

    buyer_indices, seller_splits, test_indices_from_partition = partitioner.get_splits()
    client_properties = getattr(partitioner, 'client_properties', {})  # Get properties if available

    # 5. Save data split statistics (including test indices from partition)
    logger.info("Generating and saving tabular data split statistics...")
    # --- Create cache key based on TabularDataConfig ---
    tabular_split_params = {
        "dataset": cfg.experiment.dataset_name,
        "n_sellers": cfg.experiment.n_sellers,
        "seed": cfg.seed,
        # Seller params
        "seller_strategy": tabular_cfg.strategy,
        "seller_dirichlet_alpha": tabular_cfg.dirichlet_alpha if tabular_cfg.strategy == 'dirichlet' else None,
        "seller_property_skew_params": tabular_cfg.property_skew if tabular_cfg.strategy == 'property-skew' else None,
        # Buyer params
        "buyer_ratio": tabular_cfg.buyer_ratio,
        "buyer_strategy": tabular_cfg.buyer_strategy,
        "buyer_dirichlet_alpha": tabular_cfg.buyer_dirichlet_alpha if tabular_cfg.buyer_strategy == 'dirichlet' else None,
    }
    config_string = json.dumps(tabular_split_params, sort_keys=True)
    config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()
    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"

    save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        test_indices=test_indices_from_partition,  # Pass the new test indices
        client_properties=client_properties,
        targets=y_train_pd.values,  # Use original numpy array of train targets
        save_filepath=stats_save_path
    )

    # 6. Create final DataLoaders
    batch_size = cfg.training.batch_size

    # --- PERFORMANCE OPTIMIZATION START ---
    workers = max(cfg.data.num_workers, 0)  # Respect explicit 0 (main-process loading)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 8
        loader_kwargs["persistent_workers"] = True

    logger.info(f"⚡️ Tabular Loader Optimized: workers={workers}, prefetch=8, pin=True")

    buyer_loader = None
    if buyer_indices is not None and len(buyer_indices) > 0:
        buyer_loader = DataLoader(
            Subset(train_dataset, buyer_indices.tolist()),
            shuffle=True,
            **loader_kwargs  # <--- Applies the optimizations
        )
    if buyer_loader is None: logger.warning("Buyer loader is None.")

    seller_loaders = {}
    non_empty_clients = 0
    for client_id, indices in seller_splits.items():
        if indices is not None and len(indices) > 0:
            seller_loaders[f"{client_id}"] = DataLoader(
                Subset(train_dataset, indices),
                shuffle=True,
                **loader_kwargs  # <--- Applies the optimizations
            )
            non_empty_clients += 1
        else:
            logger.warning(f"Client {client_id} has no data.")
    logger.info(f"Created DataLoaders for {non_empty_clients} / {cfg.experiment.n_sellers} sellers.")

    # Use the proper held-out test set from the sklearn train/test split,
    # not a subset of the training data (which would inflate test metrics).
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    # --- PERFORMANCE OPTIMIZATION END ---

    logger.info(f"✅ Federated tabular dataset setup complete.")
    # Return the test_loader derived from the training data partition
    return buyer_loader, seller_loaders, test_loader, num_classes, input_dim, feature_to_idx