# run_exp.py — Core experiment runner for federated learning experiments

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.common_utils import get_image_dataset, get_text_dataset
from src.common_utils import get_tabular_dataset
from src.common_utils import set_seed
from torch.utils.data import DataLoader, random_split

from src.participants.buyer.gradient.buyer import MaliciousBuyerProxy
from src.common_utils.evaluation.evaluators import create_evaluators, CleanEvaluator
from experiments.gradient_market.automate_exp.config_parser import load_config
from src.marketplace.market.markplace_gradient import DataMarketplaceFederated
from src.mechanism.gradient.aggregator import Aggregator
from src.marketplace.utils.gradient_market_utils.factories import SellerFactory, StatefulModelFactory, TextModelFactory
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, RuntimeDataConfig
from src.model.image_model import ImageModelFactory
from src.model.model_configs import get_image_model_config
from src.model.tabular_model import TabularModelFactory, TabularConfigManager
from src.participants.seller.gradient_seller import (
    GradientSeller, SybilCoordinator
)


def setup_data_and_model(cfg: AppConfig, device):
    """Loads dataset and creates a model factory from the AppConfig."""
    dataset_name = cfg.experiment.dataset_name
    dataset_type = cfg.experiment.dataset_type

    if dataset_type == "text":
        processed_data = get_text_dataset(cfg)
        buyer_loader = processed_data.buyer_loader
        seller_loaders = processed_data.seller_loaders
        test_loader = processed_data.test_loader
        num_classes = processed_data.num_classes
        vocab = processed_data.vocab
        collate_fn = processed_data.collate_fn
        pad_idx = processed_data.pad_idx

        if not seller_loaders:
            logging.error("get_text_dataset returned empty seller_loaders!")
            raise ValueError("get_text_dataset returned no sellers. Check data partitioning.")

        base_factory = TextModelFactory.create_factory(
            dataset_name=dataset_name,
            num_classes=num_classes,
            vocab_size=len(vocab),
            padding_idx=pad_idx,
            device=device
        )

        # Wrap in stateful factory
        model_factory = StatefulModelFactory(base_factory, device)
        seller_extra_args = {"vocab": vocab, "pad_idx": pad_idx}

    elif dataset_type == "image":
        buyer_loader, seller_loaders, test_loader, stats, num_classes = get_image_dataset(cfg)

        if not seller_loaders or len(seller_loaders) == 0:
            logging.error("get_image_dataset returned empty seller_loaders!")
            logging.error(f"Config values:")
            logging.error(f"  - n_sellers: {cfg.experiment.n_sellers}")
            logging.error(f"  - dataset_name: {cfg.experiment.dataset_name}")
            logging.error(f"  - data_distribution: {getattr(cfg.experiment, 'data_distribution', 'N/A')}")
            raise ValueError(
                "get_image_dataset returned empty seller_loaders. "
                "Check your data partitioning configuration."
            )

        logging.info(f"Seller loaders created: {len(seller_loaders)}")
        logging.info(f"Seller IDs: {list(seller_loaders.keys())}")

        empty_loaders = []
        for sid, loader in seller_loaders.items():
            if loader is None or len(loader.dataset) == 0:
                empty_loaders.append(sid)

        if empty_loaders:
            logging.error(f"Found {len(empty_loaders)} empty loaders: {empty_loaders}")
            raise ValueError(f"Sellers have no data: {empty_loaders}")

        config_name = cfg.experiment.image_model_config_name
        logging.info(f"Loading image model config: '{config_name}'")

        image_model_config = get_image_model_config(config_name)
        logging.info(
            f"Loaded config for '{image_model_config.model_name}' with recipe '{image_model_config.config_name}'")

        sample_data, _ = next(iter(test_loader))
        in_channels = sample_data.shape[1]  # Derive from data (3 for RGB, 1 for grayscale)
        image_size = tuple(sample_data.shape[2:])

        logging.info(f"Image model parameters:")
        logging.info(f"  - Input channels: {in_channels}")
        logging.info(f"  - Image size: {image_size}")
        logging.info(f"  - Classes: {num_classes}")

        base_factory = ImageModelFactory.create_factory(
            model_name=image_model_config.model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            config=image_model_config,
            device=device
        )

        # Wrap in stateful factory
        model_factory = StatefulModelFactory(base_factory, device)

        collate_fn = None
        seller_extra_args = {}

    elif dataset_type == "tabular":
        buyer_loader, seller_loaders, test_loader, num_classes, input_dim, feature_to_idx = get_tabular_dataset(cfg)

        if not seller_loaders:
            logging.error("get_tabular_dataset returned empty seller_loaders!")
            raise ValueError("get_tabular_dataset returned no sellers. Check data partitioning.")

        config_manager = TabularConfigManager(config_dir=cfg.data.tabular.model_config_dir)
        tabular_model_config = config_manager.get_config_by_name(cfg.experiment.tabular_model_config_name)

        base_factory = TabularModelFactory.create_factory(
            model_name=tabular_model_config.model_name,
            input_dim=input_dim,
            num_classes=num_classes,
            config=tabular_model_config,
            device=device
        )

        # Wrap in stateful factory
        model_factory = StatefulModelFactory(base_factory, device)

        collate_fn = None
        seller_extra_args = {"feature_to_idx": feature_to_idx}

    elif dataset_type == "llm":
        from src.common_utils.data_utils.llm_dataset import get_llm_dataset
        from src.marketplace.utils.gradient_market_utils.factories import LLMModelFactory

        llm_cfg = cfg.data.llm
        if llm_cfg is None:
            raise ValueError("dataset_type='llm' but data.llm config is missing.")

        processed = get_llm_dataset(cfg)
        buyer_loader = processed.buyer_loader
        seller_loaders = processed.seller_loaders
        test_loader = processed.test_loader
        collate_fn = processed.collate_fn
        num_classes = 0  # LLMs don't have fixed classes

        if not seller_loaders:
            raise ValueError("get_llm_dataset returned no sellers. Check data/min_samples config.")

        base_factory = LLMModelFactory.create_factory(llm_cfg, device=device)
        model_factory = StatefulModelFactory(base_factory, device)

        seller_extra_args = {
            "tokenizer": processed.tokenizer,
            "llm_task": processed.task,
        }

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    cfg.experiment.num_classes = num_classes

    logging.info("=" * 60)
    logging.info(f"Data and model setup complete:")
    logging.info(f"  - Dataset: {dataset_name}")
    logging.info(f"  - Classes: {num_classes}")
    logging.info(f"  - Sellers: {len(seller_loaders)}")
    logging.info(f"  - Test samples: {len(test_loader.dataset) if test_loader else 0}")
    logging.info("=" * 60)

    logging.info("Data loaded for '%s'. Number of classes: %d", dataset_name, cfg.experiment.num_classes)

    logging.info("Attempting to create a validation set by splitting buyer data...")
    validation_loader = None

    if buyer_loader and len(buyer_loader.dataset) > 1:
        buyer_dataset = buyer_loader.dataset
        val_size = max(1, int(0.2 * len(buyer_dataset)))  # 20% for validation, keep 80% for buyer gradient
        train_size = len(buyer_dataset) - val_size

        if train_size > 0:
            generator = torch.Generator().manual_seed(cfg.seed)
            train_subset, val_subset = random_split(buyer_dataset, [train_size, val_size], generator=generator)

            # --- OPTIMIZATION START ---
            loader_kwargs = {
                "batch_size": cfg.training.batch_size,
                "num_workers": getattr(cfg.training, 'num_workers', 0),
                "pin_memory": True,
                "collate_fn": collate_fn,
                "persistent_workers": False,
            }

            buyer_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
            validation_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
            logging.info(f"  -> New buyer data size (for aggregator): {len(train_subset)}")
            logging.info(f"  -> Validation set size: {len(val_subset)}")
        else:
            logging.warning("Buyer dataset is too small to split. Using full buyer dataset for aggregator.")
            validation_loader = buyer_loader

    if validation_loader is None:
        logging.warning(
            "Could not create validation set from buyer data. Falling back to using the TEST SET as the validation set for the Oracle.")
        validation_loader = test_loader

    return buyer_loader, seller_loaders, test_loader, validation_loader, model_factory, seller_extra_args, collate_fn, num_classes


def generate_marketplace_report(save_path: Path, marketplace, total_rounds):
    """Generate comprehensive marketplace analysis report."""

    report = {
        'experiment_summary': {
            'total_sellers': len(marketplace.sellers),
            'total_rounds': total_rounds,
            'adversary_rate': sum(1 for s in marketplace.sellers.values() if 'adv' in s.seller_id) / len(
                marketplace.sellers)
        },
        'seller_summaries': {}
    }

    # Per-seller summary
    for sid, seller in marketplace.sellers.items():
        selection_history = getattr(seller, 'selection_history', [])
        reward_history = getattr(seller, 'reward_history', [])

        report['seller_summaries'][sid] = {
            'type': 'adversary' if 'adv' in sid else 'benign',
            'selection_rate': sum(1 for h in selection_history if h['selected']) / len(
                selection_history) if selection_history else 0,
            'outlier_rate': sum(1 for h in selection_history if h.get('outlier', False)) / len(
                selection_history) if selection_history else 0,
            'total_reward': sum(r['reward'] for r in reward_history) if reward_history else 0
        }

    # Save report
    with open(save_path / "marketplace_report.json", 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)

    logging.info(f"Marketplace report saved to {save_path / 'marketplace_report.json'}")


def initialize_sellers(
        cfg: AppConfig,
        marketplace,
        client_loaders,
        model_factory,
        seller_extra_args,
        collate_fn,
        num_classes: int,
        validation_loader=None
):
    """
    Creates and registers all sellers in the marketplace using the factory.

    Args:
        cfg: Application configuration
        marketplace: DataMarketplaceFederated instance
        client_loaders: Dict mapping client_id -> DataLoader
        model_factory: Factory function that creates model instances
        seller_extra_args: Extra arguments for seller creation
        collate_fn: Collate function for data loading
        num_classes: Number of output classes
    """
    logging.info("=" * 60)
    logging.info("🏪 Initializing Sellers")
    logging.info("=" * 60)

    # Validate inputs
    if not client_loaders:
        raise ValueError("client_loaders is empty! Cannot create sellers.")

    n_sellers = len(client_loaders)
    n_adversaries = int(n_sellers * cfg.experiment.adv_rate)

    # Validate adversary count
    if n_adversaries > n_sellers:
        logging.warning(
            f"⚠️  Requested {n_adversaries} adversaries but only {n_sellers} sellers available. "
            f"Capping at {n_sellers}."
        )
        n_adversaries = n_sellers

    logging.info(f"Configuration:")
    logging.info(f"  - Total sellers: {n_sellers}")
    logging.info(f"  - Adversary rate: {cfg.experiment.adv_rate:.1%}")
    logging.info(f"  - Adversaries: {n_adversaries}")
    logging.info(f"  - Benign: {n_sellers - n_adversaries}")
    logging.info(f"  - Sybil enabled: {cfg.adversary_seller_config.sybil.is_sybil}")

    # Select adversary IDs (first n_adversaries)
    all_client_ids = list(client_loaders.keys())
    adversary_ids = all_client_ids[:n_adversaries]

    logging.info(f"\n📋 Adversary IDs: {adversary_ids}")
    if validation_loader:
        seller_extra_args['validation_loader'] = validation_loader
    # Create seller factory
    seller_factory = SellerFactory(
        cfg=cfg,
        model_factory=model_factory,
        num_classes=num_classes,
        **seller_extra_args
    )

    # Track creation statistics
    created_adversaries = 0
    created_benign = 0
    registered_sybils = 0
    failed_creations = 0

    # Create and register sellers
    logging.info(f"\n🏗️  Creating sellers...")

    for cid, loader in client_loaders.items():
        is_adv = cid in adversary_ids
        seller_type = "adversary" if is_adv else "benign"
        seller_id = f"{'adv' if is_adv else 'bn'}_{cid}"

        try:
            # Validate loader has data
            if loader.dataset is None or len(loader.dataset) == 0:
                logging.error(f"  ❌ {seller_id}: Empty dataset! Skipping.")
                failed_creations += 1
                continue

            # Create seller
            seller = seller_factory.create_seller(
                seller_id=seller_id,
                dataset=loader.dataset,
                is_adversary=is_adv,
                collate_fn=collate_fn
            )

            # Validate seller was created properly
            if seller is None:
                logging.error(f"  ❌ {seller_id}: Factory returned None! Skipping.")
                failed_creations += 1
                continue

            # Validate seller has model_factory
            if not hasattr(seller, 'model_factory') or seller.model_factory is None:
                logging.error(f"  ❌ {seller_id}: No model_factory attribute! Skipping.")
                failed_creations += 1
                continue

            # Register seller in marketplace
            marketplace.register_seller(seller.seller_id, seller)

            if is_adv:
                created_adversaries += 1
            else:
                created_benign += 1

            is_registered_as_sybil = False  # Flag for logging
            if is_adv and cfg.adversary_seller_config.sybil.is_sybil:
                should_be_sybil = getattr(seller, 'is_sybil', True)

                if should_be_sybil:
                    # --- THIS IS THE CRITICAL MISSING LINE ---
                    if marketplace.sybil_coordinator:  # Make sure coordinator exists
                        marketplace.sybil_coordinator.register_seller(seller)
                        registered_sybils += 1  # Increment the counter HERE
                        is_registered_as_sybil = True
                        logging.debug(f"      -> Registered {seller_id} with SybilCoordinator.")
                    else:
                        logging.warning(
                            f"      -> Sybil attack enabled, but SybilCoordinator not found in marketplace!")
            logging.info(
                f"  ✅ {seller_id} ({seller_type}): "
                f"{len(loader.dataset)} samples"
                f"{' [SYBIL Registered]' if is_registered_as_sybil else ''}"  # Corrected log message
            )

        except Exception as e:
            logging.error(f"  ❌ {seller_id}: Creation failed: {e}", exc_info=True)
            failed_creations += 1

    # Summary
    logging.info("=" * 60)
    logging.info("📊 Seller Initialization Summary:")
    logging.info(f"  - Total created: {created_adversaries + created_benign}/{n_sellers}")
    logging.info(f"  - Adversaries: {created_adversaries}/{n_adversaries}")
    logging.info(f"  - Benign: {created_benign}/{n_sellers - n_adversaries}")
    logging.info(f"  - Registered Sybils: {registered_sybils}")
    logging.info(f"  - Failed: {failed_creations}")
    logging.info("=" * 60)

    # Validate we created enough sellers
    total_created = created_adversaries + created_benign
    if total_created == 0:
        raise RuntimeError("❌ Failed to create any sellers!")

    if total_created < n_sellers * 0.5:  # Less than 50% succeeded
        logging.warning(
            f"⚠️  Only created {total_created}/{n_sellers} sellers ({total_created / n_sellers:.1%}). "
            f"Experiment may not run as expected."
        )

    # Verify marketplace state
    registered_count = len(marketplace.sellers)
    if registered_count != total_created:
        logging.error(
            f"❌ Mismatch: Created {total_created} sellers but marketplace has {registered_count}!"
        )

    logging.info(f"✅ Seller initialization complete!\n")


def save_marketplace_analysis_data_incremental(save_path: Path, r: Dict):
    """
    Saves marketplace data incrementally, appending one round at a time.
    """
    round_num = r['round']

    # 1. Round-level aggregate metrics
    round_df = pd.DataFrame([{
        'round': round_num,
        'timestamp': r['timestamp'],
        'duration_sec': r['duration_sec'],
        'aggregation_latency_sec': r.get('aggregation_latency_sec', 0),
        'aggregation_peak_vram_mb': r.get('aggregation_peak_vram_mb', 0),
        'selection_rate': r.get('selection_rate', 0),
        'outlier_rate': r.get('outlier_rate', 0),
        'avg_gradient_norm': r.get('avg_gradient_norm', 0),
        'adversary_detection_rate': r.get('adversary_detection_rate', 0),
        'false_positive_rate': r.get('false_positive_rate', 0),
        'round_upload_mb': r.get('round_upload_mb', 0),
        'round_upload_mb_accepted': r.get('round_upload_mb_accepted', 0),
    }])
    path = save_path / "round_aggregates.csv"
    round_df.to_csv(path, mode='a', header=not path.exists(), index=False)

    # 2. Per-seller per-round metrics
    seller_round_records = []
    for key, value in r.items():
        if key.startswith('seller_') and '_' in key[7:]:
            parts = key.split('_', 2)
            if len(parts) == 3:
                _, sid, metric = parts
                seller_round_records.append({
                    'round': round_num,
                    'seller_id': sid,
                    'metric': metric,
                    'value': value
                })

    if seller_round_records:
        seller_df = pd.DataFrame(seller_round_records)
        path = save_path / "seller_round_metrics_flat.csv"  # Save in flat format
        seller_df.to_csv(path, mode='a', header=not path.exists(), index=False)

    # 3. Selection history
    selection_records = []
    for sid in r.get('selected_seller_ids', []):
        selection_records.append({'round': round_num, 'seller_id': sid, 'selected': True, 'outlier': False})
    for sid in r.get('outlier_seller_ids', []):
        selection_records.append({'round': round_num, 'seller_id': sid, 'selected': False, 'outlier': True})

    if selection_records:
        path = save_path / "selection_history.csv"
        pd.DataFrame(selection_records).to_csv(path, mode='a', header=not path.exists(), index=False)


def run_final_evaluation_and_logging(
        cfg: AppConfig,
        final_model: nn.Module,
        test_loader,
        evaluators,
        marketplace=None,
        experiment_start_time: Optional[float] = None,
):
    """Performs final evaluation and saves experiment artifacts."""
    logging.info("=" * 60)
    logging.info("Final Evaluation and Logging")
    logging.info("=" * 60)

    save_path = Path(cfg.experiment.save_path)

    final_metrics = {}  # <-- BUG 1: This line was missing

    # Get the final round count from the log file
    log_path = save_path / "training_log.csv"
    completed_rounds = 0
    if log_path.exists():
        try:
            completed_rounds = len(pd.read_csv(log_path))
        except Exception as e:
            logging.warning(f"Could not read training_log.csv to get round count: {e}")
            completed_rounds = 'unknown_read_error'

    # Run all evaluators
    logging.info("Performing final evaluation on test set...")
    for evaluator in evaluators:
        try:
            metrics = evaluator.evaluate(final_model, test_loader)
            final_metrics.update(metrics)
        except Exception as e:
            logging.error(f"Evaluator {evaluator.__class__.__name__} failed: {e}")

    logging.info(f"Final metrics: {final_metrics}")

    # Add final metadata
    final_end_time = time.time()
    final_metrics['timestamp'] = final_end_time
    final_metrics['completed_rounds'] = completed_rounds

    # --- System performance metrics (added for SIGMOD benchmark) ---
    # All entries are best-effort: any failure is logged at debug level and
    # the field is simply omitted, so this block can never break the run.
    if experiment_start_time is not None:
        wall_clock_seconds = final_end_time - experiment_start_time
        final_metrics['start_timestamp'] = experiment_start_time
        final_metrics['wall_clock_seconds'] = wall_clock_seconds
        if isinstance(completed_rounds, int) and completed_rounds > 0 and wall_clock_seconds > 0:
            final_metrics['throughput_rounds_per_sec'] = completed_rounds / wall_clock_seconds
            final_metrics['avg_seconds_per_round'] = wall_clock_seconds / completed_rounds

    # Peak GPU memory (per device) — torch.cuda.max_memory_allocated returns
    # the high-water mark since the last reset_peak_memory_stats call, which
    # we issue at the start of run_attack so this is THIS experiment's peak.
    try:
        if torch.cuda.is_available():
            device_idx = torch.device(cfg.experiment.device).index
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            peak_alloc_bytes = torch.cuda.max_memory_allocated(device_idx)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device_idx)
            final_metrics['peak_gpu_memory_allocated_mb'] = peak_alloc_bytes / (1024 * 1024)
            final_metrics['peak_gpu_memory_reserved_mb'] = peak_reserved_bytes / (1024 * 1024)
            final_metrics['gpu_device_index'] = int(device_idx)
            final_metrics['gpu_device_name'] = torch.cuda.get_device_name(device_idx)
    except Exception as _e:
        logging.debug(f"Could not record GPU memory metrics: {_e}")

    # Peak host RSS memory — Linux-only via resource.getrusage. ru_maxrss is
    # in kilobytes on Linux, bytes on macOS; we report MB to be uniform.
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: KB; convert to MB
        final_metrics['peak_host_rss_mb'] = ru.ru_maxrss / 1024.0
    except Exception as _e:
        logging.debug(f"Could not record host RSS metric: {_e}")

    # Aggregate per-round system metrics from round_aggregates.csv
    try:
        agg_csv = save_path / "round_aggregates.csv"
        if agg_csv.exists():
            _rdf = pd.read_csv(agg_csv)
            if 'aggregation_latency_sec' in _rdf.columns:
                _lat = _rdf['aggregation_latency_sec'].dropna()
                if len(_lat) > 0:
                    final_metrics['avg_aggregation_latency_sec'] = float(_lat.mean())
                    final_metrics['max_aggregation_latency_sec'] = float(_lat.max())
                    final_metrics['median_aggregation_latency_sec'] = float(_lat.median())
            if 'aggregation_peak_vram_mb' in _rdf.columns:
                _vram = _rdf['aggregation_peak_vram_mb'].dropna()
                if len(_vram) > 0:
                    final_metrics['max_aggregation_peak_vram_mb'] = float(_vram.max())
            if 'round_upload_mb' in _rdf.columns:
                _wasted = _rdf['round_upload_mb'] - _rdf.get('round_upload_mb_accepted', 0)
                final_metrics['total_wasted_upload_mb'] = float(_wasted.sum())
    except Exception as _e:
        logging.debug(f"Could not aggregate per-round system metrics: {_e}")

    # Add Cost of Convergence (CoC) if tracked
    if marketplace is not None:
        cumulative_bytes = getattr(marketplace, '_cumulative_upload_bytes', None)
        if cumulative_bytes is not None:
            final_metrics['total_upload_bytes'] = cumulative_bytes
            final_metrics['total_upload_mb'] = cumulative_bytes / (1024 * 1024)
            if isinstance(completed_rounds, int) and completed_rounds > 0:
                final_metrics['avg_upload_mb_per_round'] = (cumulative_bytes / (1024 * 1024)) / completed_rounds
        cumulative_bytes_accepted = getattr(marketplace, '_cumulative_upload_bytes_accepted', None)
        if cumulative_bytes_accepted is not None:
            final_metrics['total_upload_bytes_accepted'] = cumulative_bytes_accepted
            final_metrics['total_upload_mb_accepted'] = cumulative_bytes_accepted / (1024 * 1024)

    # Save final metrics atomically
    save_json_atomic(final_metrics, save_path / "final_metrics.json")

    # Save final model atomically
    save_model_atomic(final_model.state_dict(), save_path / "final_model.pth")

    logging.info("Final evaluation complete")


def _convert_numpy_keys(obj):
    """Recursively convert numpy-typed dict keys (and values) to native Python.
    json.JSONEncoder.default() is only called for values, not keys, so dicts
    with np.int64 keys will fail without this preprocessing step."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (np.integer, np.floating, np.bool_)):
                k = k.item()
            elif isinstance(k, torch.Tensor):
                k = k.item() if k.ndim == 0 else str(k.tolist())
            new_dict[k] = _convert_numpy_keys(v)
        return new_dict
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_keys(x) for x in obj]
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle common numpy types.
    Converts np.int64 -> int, np.float64 -> float, and np.ndarray -> list.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json_atomic(data, filepath):
    """Save JSON with atomic write to prevent corruption."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        suffix='.tmp',
        prefix=filepath.stem
    )
    try:
        with os.fdopen(temp_fd, 'w') as f:
            # Preprocess to convert numpy keys, then use encoder for values
            json.dump(_convert_numpy_keys(data), f, indent=2, cls=NumpyJSONEncoder)

        shutil.move(temp_path, filepath)
        logging.debug(f"Saved (atomic): {filepath}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save {filepath}: {e}")


def save_dataframe_atomic(df, filepath):
    """Save DataFrame with atomic write."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_path = filepath.with_suffix('.tmp')
    try:
        df.to_csv(temp_path, index=False)
        shutil.move(temp_path, filepath)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_model_atomic(state_dict, filepath):
    """Save model with atomic write."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_path = filepath.with_suffix('.tmp')
    try:
        torch.save(state_dict, temp_path)
        shutil.move(temp_path, filepath)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def run_training_loop(cfg, marketplace, validation_loader, test_loader, evaluators,
                      on_round_end=None):
    """Training loop with incremental saving.

    on_round_end: optional callable(round_record, marketplace) invoked after
    each round's record is saved. Used by the E2 multi-task runner to apply
    payment accounting and benign-seller attrition between rounds; None (the
    default) preserves the original single-task behavior exactly.
    """
    save_path = Path(cfg.experiment.save_path)
    log_path = save_path / "training_log.csv"
    log_columns = _training_log_columns(cfg)

    # Initialize CSV with ALL desired headers if it doesn't exist
    if not log_path.exists():
        try:
            pd.DataFrame(columns=log_columns).to_csv(log_path, index=False)
        except Exception as e:
            logging.error(f"Failed to initialize {log_path} with header: {e}")

    # --- 1. EARLY STOPPING INITIALIZATION ---
    patience = cfg.experiment.patience
    patience_counter = 0
    best_validation_loss = float('inf')
    best_model_state = None
    best_model_round = 0

    if cfg.experiment.use_early_stopping:
        logging.info(f"✅ Early stopping enabled with patience: {patience}")
        if not validation_loader:
            logging.warning(
                "⚠️ Early stopping is enabled, but no validation loader was provided! Cannot perform early stopping.")

    # --- 2. MAIN TRAINING LOOP ---
    for round_num in range(1, cfg.experiment.global_rounds + 1):
        # Get reference to global model (always use the current state)
        global_model = marketplace.global_model

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Round {round_num}/{cfg.experiment.global_rounds}")
        logging.info(f"{'=' * 60}")

        # Train one federated round
        round_record, agg_grad = marketplace.train_federated_round(
            round_number=round_num,
            global_model=global_model,  # Pass for reference (not strictly needed with stateful factory)
            validation_loader=validation_loader,
            ground_truth_dict={}
        )

        # --- 3. VALIDATION FOR EARLY STOPPING ---
        if cfg.experiment.use_early_stopping and validation_loader:
            all_val_metrics = {}
            try:
                current_global_model = marketplace.global_model

                from src.common_utils.evaluation.evaluators import BaseEvaluator
                for evaluator in evaluators:
                    metrics_subset = evaluator.evaluate(current_global_model, validation_loader)

                    # Prefix utility evaluators with val_ for early stopping
                    evaluator_name = type(evaluator).__name__
                    if evaluator_name in ("CleanEvaluator", "PerplexityEvaluator"):
                        prefixed_metrics = {f"val_{k}": v for k, v in metrics_subset.items()}
                        all_val_metrics.update(prefixed_metrics)
                    else:
                        all_val_metrics.update(metrics_subset)

                current_loss = all_val_metrics.get('val_loss')
                if current_loss is not None:
                    logging.info(f"📊 Validation Loss: {current_loss:.4f}")

                    if current_loss < best_validation_loss:
                        best_validation_loss = current_loss
                        patience_counter = 0
                        # Save the current best model state
                        best_model_state = {k: v.cpu().clone() for k, v in current_global_model.state_dict().items()}
                        best_model_round = round_num
                        logging.info(f"  ✅ New best model! (loss: {current_loss:.4f})")
                    else:
                        patience_counter += 1
                        logging.info(f"  ⏳ No improvement. Patience: {patience_counter}/{patience}")

                # Add validation metrics to round record
                round_record.update(all_val_metrics)

            except Exception as e:
                logging.error(f"❌ Validation failed in round {round_num}: {e}", exc_info=True)
                # Do NOT increment patience_counter on validation failure —
                # transient errors (e.g., OOM) should not trigger early stopping

        # --- 4. CHECK EARLY STOPPING ---
        if cfg.experiment.use_early_stopping and patience_counter >= patience:
            logging.warning(f"🛑 EARLY STOPPING: No improvement for {patience} rounds. Halting at round {round_num}.")
            break

        # --- 5. PERIODIC TEST EVALUATION ---
        if round_num % cfg.experiment.eval_frequency == 0:
            logging.info(f"📈 Running test evaluation at round {round_num}...")
            eval_metrics = {}

            # IMPORTANT: Use the CURRENT global model for test evaluation
            current_global_model = marketplace.global_model

            for evaluator in evaluators:
                metrics_subset = evaluator.evaluate(current_global_model, test_loader)
                prefixed_metrics = {f"test_{k}": v for k, v in metrics_subset.items()}
                eval_metrics.update(prefixed_metrics)

            # Log test metrics
            logging.info("📊 Test Metrics:")
            for key, value in eval_metrics.items():
                logging.info(f"  {key}: {value:.4f}")

            # Add to round record
            round_record.update(eval_metrics)

            # Save evaluation
            eval_path = save_path / "evaluations" / f"round_{round_num}.json"
            save_json_atomic(eval_metrics, eval_path)

        # --- 6. SAVE ROUND DATA INCREMENTALLY ---
        save_round_incremental(round_record, save_path, columns=log_columns)
        save_seller_metrics_incremental(save_path, round_record)
        # Re-enabled: writes round_aggregates.csv with per-round
        # aggregation_latency_sec / aggregation_peak_vram_mb /
        # round_upload_mb / round_upload_mb_accepted. Without this, the
        # post-experiment aggregation in run_final_evaluation_and_logging
        # (line ~567) silently produces final_metrics.json files missing
        # avg/max/median aggregation latency, which then breaks
        # extract_step8_system_scalability.py's broker-latency columns.
        save_marketplace_analysis_data_incremental(save_path, round_record)
        detailed_stats = round_record.get('detailed_aggregation_stats')

        # Save the detailed log *only if* it's not empty
        if detailed_stats:
            # Create the save directory
            agg_stats_save_path = save_path / "agg_stats"
            agg_stats_save_path.mkdir(parents=True, exist_ok=True)

            # Save the *entire* nested dictionary to a JSON file
            save_json_atomic(
                detailed_stats,
                agg_stats_save_path / f"round_{round_record['round']}.json"
            )
        # --- 7. GENERATE REPORTS ---
        generate_marketplace_report(save_path, marketplace, cfg.experiment.global_rounds)

        # Seller summaries
        for sid, seller in marketplace.sellers.items():
            seller.save_marketplace_summary()

        # --- 7b. OPTIONAL OUTER-LOOP HOOK (E2 multi-task runner) ---
        if on_round_end is not None:
            on_round_end(round_record, marketplace)

    # --- 8. LOAD BEST MODEL IF EARLY STOPPING WAS USED ---
    final_global_model = marketplace.global_model

    if best_model_state:
        logging.info(f"✅ Loading best model from round {best_model_round} (Val Loss: {best_validation_loss:.4f})")
        final_global_model.load_state_dict(best_model_state)
    else:
        logging.info("ℹ️ No early stopping occurred - using model from final round")

    return final_global_model


_csv_headers_cache = {}
TRAINING_LOG_COLUMNS = [
    # Core Round Info
    'round', 'timestamp', 'duration_sec',

    # Aggregation/Selection Summary
    'num_total_sellers', 'num_selected', 'num_outliers',
    'selection_rate', 'outlier_rate',

    # Defense Performance
    'num_known_adversaries', 'num_detected_adversaries', 'num_benign_outliers',
    'adversary_detection_rate', 'false_positive_rate',

    # Validation Metrics (every round)
    'val_loss', 'val_acc', 'asr', 'B-Acc', 'B-F1',

    # Test Metrics (periodic)
    'test_loss', 'test_acc', 'test_asr', 'test_B-Acc', 'test_B-F1',

    # Buyer Attack Info
    "buyer_attack_active", "buyer_attack_type", "buyer_attack_stats",

    # Server Attack Info
    "attack_performed", "attack_victim", "attack_success",

    # Gradient Stats (from default evaluator)
    'avg_gradient_norm', 'std_gradient_norm', 'min_gradient_norm', 'max_gradient_norm',

    # --- ALL VALUATION METRICS (Default & Optional) ---
    'avg_sim_to_buyer', 'std_sim_to_buyer', 'min_sim_to_buyer', 'max_sim_to_buyer',
    'avg_sim_to_oracle', 'std_sim_to_oracle', 'min_sim_to_oracle', 'max_sim_to_oracle',
    'avg_sim_to_aggregate_cgsv', 'std_sim_to_aggregate_cgsv',

]


def _training_log_columns(cfg) -> list:
    """training_log.csv column set for this run.

    The default set is TRAINING_LOG_COLUMNS, byte-identical to the original
    log header. The E1a continuous-payment mode opts into one extra column:
    adversary_revenue_share (the continuous analogue of MSR).
    """
    columns = list(TRAINING_LOG_COLUMNS)
    if getattr(cfg.valuation, 'payment_model', 'quality_based') == "weighted":
        columns.append('adversary_revenue_share')
    return columns


def save_round_incremental(round_record: Dict, save_path: Path, columns: list = None):
    """
    Saves only the predefined TRAINING_LOG_COLUMNS (or the caller-provided
    column set) from the round_record to training_log.csv incrementally.
    """
    log_path = Path(save_path) / "training_log.csv"
    if columns is None:
        columns = TRAINING_LOG_COLUMNS

    try:
        # 1. Select only the desired columns, using .get() for safety
        #    Use None (which becomes NaN in Pandas) if a key is missing.
        filtered_record = {
            col: round_record.get(col, None) for col in columns
        }

        # 2. Create DataFrame using the defined columns to ensure order and structure
        df = pd.DataFrame([filtered_record], columns=columns)

        # 3. Append to existing file or create new
        file_exists = log_path.exists() and log_path.stat().st_size > 0

        df.to_csv(
            log_path,
            mode='a',  # Always append
            header=not file_exists,  # Write header only if file doesn't exist yet (or is empty)
            index=False
        )
    except Exception as e:
        logging.error(f"Error writing round {round_record.get('round', 'N/A')} to {log_path}: {e}")
        logging.error(f"Available keys in round_record: {list(round_record.keys())}")


SELLER_LOG_COLUMNS = [
    'round', 'seller_id', 'selected', 'outlier',
    'sim_to_oracle_root', 'sim_to_buyer_root',
    'gradient_norm', 'train_loss', 'num_samples', 'weight',
    # Payment read-out. `price_paid` is what the pricing scheme actually paid
    # this round and `selection_score` is the filter's raw per-seller score;
    # both are needed to audit E1a/E2 payment claims from the CSV alone
    # (previously they existed only inside valuations.jsonl).
    'price_paid', 'selection_score',
]

# The per-seller dicts come straight from `seller_valuations`, whose key names
# differ from the historical CSV column names. Without this mapping the
# reindex below silently produced three permanently-empty columns
# (sim_to_oracle_root, sim_to_buyer_root, weight) in every run ever logged.
SELLER_LOG_ALIASES = {
    'sim_to_oracle_root': 'sim_to_oracle',
    'sim_to_buyer_root': 'sim_to_buyer',
    'weight': 'aggregation_weight',
}


def save_seller_metrics_incremental(save_path: Path, round_record: Dict):
    """
    Saves the detailed per-seller metrics to a separate 'seller_metrics.csv' file.
    """
    seller_metrics_list = round_record.get('detailed_seller_metrics')
    if not seller_metrics_list:
        return  # No seller metrics to save this round

    log_path = save_path / "seller_metrics.csv"

    try:
        # Convert the list of dicts to a DataFrame
        df = pd.DataFrame(seller_metrics_list)

        # Fill the legacy column names from their real source keys before the
        # reindex drops everything not named in SELLER_LOG_COLUMNS.
        for legacy, source in SELLER_LOG_ALIASES.items():
            if source in df.columns:
                if legacy not in df.columns or df[legacy].isna().all():
                    df[legacy] = df[source]

        # Check if file exists to write header
        file_exists = os.path.isfile(log_path)

        # Ensure column order and handle missing columns. If we are appending to
        # a CSV written by an older build, keep ITS header so the appended rows
        # stay aligned — never widen a file mid-run.
        columns = SELLER_LOG_COLUMNS
        if file_exists:
            try:
                with open(log_path, 'r') as fh:
                    existing = fh.readline().strip().split(',')
                if existing and existing != ['']:
                    columns = existing
            except OSError:
                pass
        df = df.reindex(columns=columns)

        # Append to CSV
        df.to_csv(
            log_path,
            mode='a',
            header=not file_exists,
            index=False,
            float_format='%.6e'  # Use scientific notation for precision
        )
    except Exception as e:
        logging.error(f"Failed to save incremental seller metrics: {e}")


def initialize_root_sellers(cfg, marketplace, buyer_loader, validation_loader, model_factory):
    """
    Creates and attaches the two 'virtual' sellers for root gradient computation.
    Conditionally replaces the buyer seller with a malicious proxy if an attack is active.
    """
    logging.info("--- Initializing Root Gradient Sellers ---")

    # 1. Conditionally create the "Buyer Seller"
    if cfg.buyer_attack_config.is_active:
        # --- MALICIOUS PATH ---
        logging.warning("🚨 Buyer-side attack is active! Creating MaliciousBuyerProxy.")
        marketplace.buyer_seller = MaliciousBuyerProxy(
            seller_id='malicious_buyer_proxy',
            attack_config=cfg.buyer_attack_config,  # Pass the specific attack config
            data_config=RuntimeDataConfig(
                dataset=buyer_loader.dataset,
                num_classes=marketplace.num_classes,
                collate_fn=getattr(buyer_loader, 'collate_fn', None)
            ),
            training_config=cfg.training,
            model_factory=model_factory,
            save_path=cfg.experiment.save_path,
            device=cfg.experiment.device,
            num_classes=marketplace.num_classes  # 🆕 ADD THIS for oscillating/class_exclusion attacks
        )
    else:
        # --- HONEST PATH ---
        logging.info("🛒 Creating honest virtual 'Buyer Seller'...")
        marketplace.buyer_seller = marketplace.SellerClass(
            seller_id='virtual_buyer',
            data_config=RuntimeDataConfig(
                dataset=buyer_loader.dataset,
                num_classes=marketplace.num_classes,
                collate_fn=getattr(buyer_loader, 'collate_fn', None)
            ),
            training_config=cfg.training,
            model_factory=model_factory,
            save_path=cfg.experiment.save_path,
            device=cfg.experiment.device)

    # 2. Create the "Oracle Seller" (no changes needed)
    logging.info("🧪 Creating virtual 'Oracle Seller'...")
    marketplace.oracle_seller = marketplace.SellerClass(
        seller_id='virtual_oracle',
        data_config=RuntimeDataConfig(
            dataset=validation_loader.dataset,
            num_classes=marketplace.num_classes,
            collate_fn=getattr(validation_loader, 'collate_fn', None)
        ),
        training_config=cfg.training,
        model_factory=model_factory,
        save_path=cfg.experiment.save_path,
        device=cfg.experiment.device
    )
    logging.info("✅ Root gradient sellers initialized.")


def run_attack(cfg: AppConfig):
    """
    Orchestrates the entire experiment from a single config object.

    Note: Caching is handled by the experiment orchestrator, not here.
    This function always runs the full experiment when called.

    Args:
        cfg: Application configuration containing all experiment parameters
    """
    save_path = Path(cfg.experiment.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- System performance instrumentation (added for SIGMOD benchmark) ---
    # Capture wall-clock start so we can compute training duration and
    # throughput. Reset CUDA peak-memory counter so torch.cuda.max_memory_*
    # at the end reflects only this experiment's footprint, not whatever was
    # allocated before by another cell that shared the GPU.
    _experiment_start_time = time.time()
    try:
        if torch.cuda.is_available():
            _device_idx = torch.device(cfg.experiment.device).index
            if _device_idx is not None:
                torch.cuda.reset_peak_memory_stats(_device_idx)
            else:
                torch.cuda.reset_peak_memory_stats()
    except Exception as _e:
        logging.debug(f"Could not reset CUDA peak memory stats: {_e}")

    logging.info("=" * 80)
    logging.info(f"🚀 Starting Experiment")
    logging.info(f"   Dataset: {cfg.experiment.dataset_name}")
    logging.info(f"   Model: {cfg.experiment.model_structure}")
    logging.info(f"   Device: {cfg.experiment.device}")
    logging.info(f"   Save Path: {save_path}")
    logging.info("=" * 80)

    try:
        # 1. Save configuration snapshot for reproducibility
        _save_config_for_reproducibility(cfg, save_path)

        # 2. Data and Model Setup
        buyer_loader, seller_loaders, test_loader, validation_loader, model_factory, seller_extra_args, collate_fn, num_classes = \
            setup_data_and_model(cfg, device=cfg.experiment.device)

        # Create global model (first call returns fresh model with random weights)
        global_model = model_factory().to(cfg.experiment.device)
        logging.info(f"✅ Global model created and moved to {cfg.experiment.device}")

        # Register global model with the stateful factory
        model_factory.set_global_model(global_model)
        logging.info("✅ Model factory now maintains global model reference")
        logging.info(f"--- Global Model Architecture ---\n{global_model}")

        # 3. FL Component Initialization
        loss_fn = nn.CrossEntropyLoss()
        aggregator = Aggregator(
            global_model=global_model,
            device=torch.device(cfg.experiment.device),
            loss_fn=loss_fn,
            buyer_data_loader=buyer_loader,
            agg_config=cfg.aggregation
        )
        factory_model_id = id(model_factory.global_model)
        aggregator_model_id = id(aggregator.strategy.global_model)

        logging.info(f"🔍 Model Reference Check:")
        logging.info(f"  Factory model ID:     {factory_model_id}")
        logging.info(f"  Aggregator model ID:  {aggregator_model_id}")
        logging.info(f"  Same object? {factory_model_id == aggregator_model_id}")

        sybil_coordinator = SybilCoordinator(cfg.adversary_seller_config.sybil, aggregator)
        evaluators = create_evaluators(cfg, cfg.experiment.device, **seller_extra_args)

        # 4. Marketplace Initialization
        if cfg.experiment.dataset_type == "llm":
            # LLM loaders yield dicts, not tensors — shape inference not needed
            input_shape = (1,)  # Placeholder — not used for LLM aggregation
        else:
            sample_data = _get_sample_data(test_loader, seller_loaders)
            input_shape = tuple(sample_data.shape[1:])

        marketplace = DataMarketplaceFederated(
            cfg=cfg,
            aggregator=aggregator,
            sellers={},
            input_shape=input_shape,
            SellerClass=GradientSeller,
            validation_loader=validation_loader,
            model_factory=model_factory,  # Pass the stateful factory
            num_classes=num_classes,
            sybil_coordinator=sybil_coordinator,
        )

        # 5. Seller Initialization
        initialize_sellers(
            cfg,
            marketplace,
            seller_loaders,
            model_factory,
            seller_extra_args,
            collate_fn,
            num_classes,
            validation_loader=validation_loader
        )

        initialize_root_sellers(
            cfg, marketplace, buyer_loader, validation_loader, model_factory
        )

        # 6. Federated Training Loop
        logging.info("🏋️ Starting federated training...")
        final_model = run_training_loop(
            cfg, marketplace, validation_loader, test_loader, evaluators
        )

        # 7. Final Evaluation and Artifact Saving
        logging.info("📊 Running final evaluation and saving results...")
        run_final_evaluation_and_logging(
            cfg, final_model, test_loader, evaluators, marketplace,
            experiment_start_time=_experiment_start_time,
        )

        # 8. Save seller-specific results
        for sid, seller in marketplace.sellers.items():
            seller.save_round_history_csv()

        # 9. Mark experiment as successfully completed
        _mark_experiment_success(save_path)

        logging.info("=" * 80)
        logging.info("✅ Experiment Finished Successfully")
        logging.info(f"   Results saved to: {save_path}")
        logging.info("=" * 80)

        return None

    except Exception as e:
        logging.error("=" * 80)
        logging.error(f"❌ Experiment Failed: {e}")
        logging.error("=" * 80)
        _mark_experiment_failed(save_path, str(e))
        raise


def _save_config_for_reproducibility(cfg: AppConfig, save_path: Path):
    """
    Save the configuration to disk for reproducibility and debugging.

    Also writes `run_provenance.json` next to it: a content hash of the exact
    config used, the git commit, and the seed. Without this a result directory
    cannot be tied back to the configuration that produced it, which makes
    "did this number drift, or did the config change?" unanswerable after the
    fact.
    """
    import json
    config_dict = None
    try:
        config_path = save_path / "config_snapshot.json"
        with open(config_path, 'w') as f:
            # Assuming your AppConfig has a to_dict() method
            # If not, you might need to use dataclasses.asdict() or similar
            if hasattr(cfg, 'to_dict'):
                config_dict = cfg.to_dict()
                json.dump(config_dict, f, indent=2)
            elif hasattr(cfg, '__dict__'):
                # Fallback: try to serialize the object's dictionary
                config_dict = json.loads(json.dumps(cfg.__dict__, default=str))
                json.dump(config_dict, f, indent=2, default=str)
            else:
                logging.warning("Could not serialize config - no to_dict() method available")

        logging.info(f"📝 Configuration snapshot saved to: {config_path}")
    except Exception as e:
        logging.warning(f"⚠️  Could not save config snapshot: {e}")

    _save_run_provenance(cfg, save_path, config_dict)


def _save_run_provenance(cfg: AppConfig, save_path: Path, config_dict):
    """Write run_provenance.json (config hash + git commit + seed + host)."""
    import hashlib
    import json
    import platform
    import subprocess
    try:
        if config_dict is None:
            config_sha = None
        else:
            # sort_keys so the hash is stable across dict ordering.
            canonical = json.dumps(config_dict, sort_keys=True, default=str)
            config_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # Try the module's own directory first, then the process cwd — the
        # experiment runner is normally launched from the repo root, and the
        # module path can point outside the work tree when it is staged.
        _candidates = [Path(__file__).resolve().parent, Path.cwd()]

        def _git(*args):
            for cwd in _candidates:
                try:
                    out = subprocess.check_output(
                        ["git", *args], cwd=str(cwd),
                        stderr=subprocess.DEVNULL, timeout=10,
                    ).decode().strip()
                    if out or args[0] == "status":
                        return out
                except Exception:
                    continue
            return None

        seed = None
        for attr in ("seed", "random_seed"):
            seed = getattr(getattr(cfg, "experiment", cfg), attr, None)
            if seed is not None:
                break

        provenance = {
            "config_sha256": config_sha,
            "seed": seed,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "timestamp": time.time(),
        }
        with open(save_path / "run_provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
        logging.info(f"🔖 Provenance: config_sha256={str(config_sha)[:12]} "
                     f"commit={provenance['git_commit']} seed={seed}")
    except Exception as e:
        logging.warning(f"⚠️  Could not write run provenance: {e}")


def _get_sample_data(test_loader, seller_loaders):
    """
    Get a sample batch of data for shape inference.
    Tries test_loader first, then falls back to seller loaders.
    This version is robust to different numbers of returned items from the loader.
    """
    sample_data = None

    # --- Try test loader first ---
    if test_loader:
        try:
            # Get the whole batch first without unpacking it immediately
            batch = next(iter(test_loader))

            # Check how many items the loader returned to handle different data types
            if len(batch) == 3:  # This is the text case: (labels, texts, lengths)
                sample_data = batch[1]  # The actual data is the second item ('texts')
            else:  # Assume the standard (data, label) case for image/tabular
                sample_data = batch[0]  # The data is the first item

            logging.info("✅ Sample data obtained from test loader")

        except StopIteration:
            logging.warning("⚠️  Test loader is available but empty")

    # --- Fall back to seller loaders if needed ---
    if sample_data is None:
        logging.info("🔍 No test data found. Trying seller loaders...")
        for sid, loader in seller_loaders.items():
            if loader:
                try:
                    # Apply the same robust logic here
                    batch = next(iter(loader))
                    if len(batch) == 3:
                        sample_data = batch[1]
                    else:
                        sample_data = batch[0]

                    logging.info(f"✅ Sample data obtained from seller {sid} loader")
                    break
                except StopIteration:
                    continue

    if sample_data is None:
        raise RuntimeError(
            "❌ Could not retrieve a sample data batch from any available loader. "
            "Please check your data loaders."
        )

    return sample_data


def _mark_experiment_success(save_path: Path):
    """
    Create a success marker file to indicate the experiment completed successfully.
    This is used by the orchestrator for caching.
    """
    success_marker = save_path / ".success"
    try:
        success_marker.touch()
        logging.info(f"✅ Success marker created: {success_marker}")
    except Exception as e:
        logging.warning(f"⚠️  Could not create success marker: {e}")


def _mark_experiment_failed(save_path: Path, error_message: str):
    """
    Create a failure marker with error information for debugging.
    """
    try:
        failed_marker = save_path / ".failed"
        with open(failed_marker, 'w') as f:
            f.write(f"Experiment failed with error:\n{error_message}\n")
        logging.info(f"❌ Failure marker created: {failed_marker}")
    except Exception as e:
        logging.warning(f"⚠️  Could not create failure marker: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment from Config File")
    parser.add_argument("config", help="Path to the YAML configuration file")
    cli_args = parser.parse_args()

    app_config = load_config(cli_args.config)

    initial_seed = app_config.seed
    for i in range(app_config.n_samples):
        run_cfg = copy.deepcopy(app_config)
        current_seed = initial_seed + i
        run_cfg.seed = current_seed  # Propagate seed so internal splits use it
        set_seed(current_seed)

        run_save_path = Path(run_cfg.experiment.save_path) / f"run_{i}_seed_{current_seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        run_cfg.experiment.save_path = str(run_save_path)

        logging.info(f"\n{'=' * 20} Starting Run {i + 1}/{app_config.n_samples} (Seed: {current_seed}) {'=' * 20}")
        run_attack(run_cfg)


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Belt-and-suspenders against "Shared memory manager connection has timed
    # out": prefer FD-based sharing over /dev/shm filename-based sharing for any
    # tensor that does cross a process boundary. With the seller loader now at
    # num_workers=0 this is rarely exercised, but it costs nothing and protects
    # multi-worker code paths on cluster nodes with a small/contended /dev/shm.
    try:
        import torch.multiprocessing as _torch_mp
        _torch_mp.set_sharing_strategy("file_descriptor")
    except Exception as _e:
        logging.debug(f"Could not set torch sharing strategy: {_e}")
    main()
