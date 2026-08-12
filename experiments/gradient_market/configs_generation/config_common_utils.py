# ==============================================================================
# --- 1. Golden Training HPs (from Step 1) & Tuned Defense HPs (from Step 3) ---
# ==============================================================================
# These are loaded from JSON files produced by auto_update_golden_params.py.
# If no JSON file exists, hardcoded defaults below are used as fallback.
# Run:  python experiments/gradient_market/auto_update_golden_params.py --step 1
#       python experiments/gradient_market/auto_update_golden_params.py --step 3

import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from src.common_utils.constants import PoisonType
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig

_PARAMS_DIR = Path(__file__).resolve().parent / "tuned_params"
_GOLDEN_PARAMS_FILE = _PARAMS_DIR / "golden_training_params.json"
_TUNED_DEFENSE_FILE = _PARAMS_DIR / "tuned_defense_params.json"


def _load_json_params(filepath: Path) -> Dict[str, Any]:
    """Load params from a JSON file, stripping metadata keys (prefixed with _)."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        raw = json.load(f)
    # Strip metadata keys like _best_acc, _data_setting, _score, etc.
    cleaned = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            cleaned[key] = {k: v for k, v in val.items() if not k.startswith("_")}
        else:
            cleaned[key] = val
    return cleaned


# --- Hardcoded fallback defaults (used when JSON files don't exist) ---
_DEFAULT_GOLDEN_TRAINING_PARAMS = {
    "cifar10_cnn": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "cifar10_resnet18": {
        "training.optimizer": "SGD", "training.learning_rate": 0.1, "training.local_epochs": 2,
        "training.momentum": 0.9, "training.weight_decay": 5e-4,
    },
    "cifar100_cnn": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "cifar100_resnet18": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "mlp_texas100_baseline": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "mlp_purchase100_baseline": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 5,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "textcnn_trec_baseline": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
    "femnist_cnn": {
        "training.optimizer": "Adam", "training.learning_rate": 0.001, "training.local_epochs": 2,
        "training.momentum": 0.0, "training.weight_decay": 0.0,
    },
}

_DEFAULT_TUNED_DEFENSE_PARAMS = {
    "fedavg_cifar100_cnn_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_cifar100_cnn_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_cifar10_cnn_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_cifar10_cnn_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_texas100_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_mlp_texas100_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fedavg_textcnn_trec_baseline_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_textcnn_trec_baseline_labelflip": {'aggregation.method': 'fedavg'},
    "fltrust_cifar100_cnn_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_cifar100_cnn_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_cifar10_cnn_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_cifar10_cnn_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_texas100_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "fltrust_mlp_texas100_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_textcnn_trec_baseline_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_textcnn_trec_baseline_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "martfl_cifar100_cnn_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "martfl_cifar100_cnn_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_cifar10_cnn_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0},
    "martfl_cifar10_cnn_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_texas100_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "martfl_mlp_texas100_baseline_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 7},
    "martfl_textcnn_trec_baseline_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "martfl_textcnn_trec_baseline_labelflip": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "skymask_cifar100_cnn_backdoor": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar100_cnn_labelflip": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar10_cnn_backdoor": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_cifar10_cnn_labelflip": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar100_cnn_backdoor": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar100_cnn_labelflip": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar10_cnn_backdoor": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_cifar10_cnn_labelflip": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "trimmed_mean_cifar10_cnn_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_cifar10_cnn_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_cifar100_cnn_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_cifar100_cnn_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_mlp_texas100_baseline_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_mlp_texas100_baseline_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_textcnn_trec_baseline_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_textcnn_trec_baseline_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "multi_krum_cifar10_cnn_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_cifar10_cnn_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_cifar100_cnn_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_cifar100_cnn_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_mlp_texas100_baseline_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_mlp_texas100_baseline_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_textcnn_trec_baseline_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_textcnn_trec_baseline_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "rflpa_cifar10_cnn_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_cifar10_cnn_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_cifar100_cnn_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_cifar100_cnn_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_mlp_texas100_baseline_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_mlp_texas100_baseline_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_textcnn_trec_baseline_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_textcnn_trec_baseline_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "spmc_cifar10_cnn_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_cifar10_cnn_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_cifar100_cnn_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_cifar100_cnn_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_mlp_texas100_baseline_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_mlp_texas100_baseline_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_textcnn_trec_baseline_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_textcnn_trec_baseline_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "daved_cifar10_cnn_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_cifar10_cnn_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_cifar100_cnn_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_cifar100_cnn_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_mlp_texas100_baseline_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_mlp_texas100_baseline_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_textcnn_trec_baseline_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_textcnn_trec_baseline_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    # FEMNIST (uses same defaults as cifar100_cnn — will be refined after Step 3)
    "fedavg_femnist_cnn_backdoor": {'aggregation.method': 'fedavg'},
    "fedavg_femnist_cnn_labelflip": {'aggregation.method': 'fedavg'},
    "fltrust_femnist_cnn_backdoor": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 5.0},
    "fltrust_femnist_cnn_labelflip": {'aggregation.method': 'fltrust', 'aggregation.clip_norm': 3.0},
    "martfl_femnist_cnn_backdoor": {'aggregation.method': 'martfl', 'aggregation.clip_norm': 5.0, 'aggregation.martfl.max_k': 3},
    "martfl_femnist_cnn_labelflip": {'aggregation.method': 'martfl', 'aggregation.martfl.max_k': 3},
    "skymask_femnist_cnn_backdoor": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_femnist_cnn_labelflip": {'aggregation.method': 'skymask', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 50, 'aggregation.skymask.mask_lr': 0.001, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_femnist_cnn_backdoor": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "skymask_small_femnist_cnn_labelflip": {'aggregation.method': 'skymask_small', 'aggregation.clip_norm': 10.0, 'aggregation.skymask.mask_epochs': 20, 'aggregation.skymask.mask_lr': 0.5, 'aggregation.skymask.mask_threshold': 0.5},
    "trimmed_mean_femnist_cnn_backdoor": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "trimmed_mean_femnist_cnn_labelflip": {'aggregation.method': 'trimmed_mean', 'aggregation.trimmed_mean.trim_ratio': 0.1},
    "multi_krum_femnist_cnn_backdoor": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "multi_krum_femnist_cnn_labelflip": {'aggregation.method': 'multi_krum', 'aggregation.multi_krum.num_byzantine': 3, 'aggregation.multi_krum.m_selected': 3},
    "rflpa_femnist_cnn_backdoor": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "rflpa_femnist_cnn_labelflip": {'aggregation.method': 'rflpa', 'aggregation.rflpa.element_clip': 1.0, 'aggregation.clip_norm': 5.0},
    "spmc_femnist_cnn_backdoor": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "spmc_femnist_cnn_labelflip": {'aggregation.method': 'spmc', 'aggregation.spmc.temperature': 0.5},
    "daved_femnist_cnn_backdoor": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},
    "daved_femnist_cnn_labelflip": {'aggregation.method': 'daved', 'aggregation.daved.l2_relative_alpha': 0.1, 'aggregation.daved.proj_lambda': 1.0, 'aggregation.clip_norm': 5.0},

    # ===== NEW DEFENSES (defaults — will be refined after step 3 tuning) =====
    # --- FLAME ---
    "flame_cifar100_cnn_backdoor": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_cifar100_cnn_labelflip": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_femnist_cnn_backdoor": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_femnist_cnn_labelflip": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_mlp_texas100_baseline_backdoor": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_mlp_texas100_baseline_labelflip": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_textcnn_trec_baseline_backdoor": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    "flame_textcnn_trec_baseline_labelflip": {'aggregation.method': 'flame', 'aggregation.flame.noise_scale': 0.001, 'aggregation.flame.min_cluster_size_ratio': 0.5},
    # --- DeepSight ---
    "deepsight_cifar100_cnn_backdoor": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_cifar100_cnn_labelflip": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_femnist_cnn_backdoor": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_femnist_cnn_labelflip": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_mlp_texas100_baseline_backdoor": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_mlp_texas100_baseline_labelflip": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_textcnn_trec_baseline_backdoor": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    "deepsight_textcnn_trec_baseline_labelflip": {'aggregation.method': 'deepsight', 'aggregation.deepsight.threshold_c': 2.0, 'aggregation.deepsight.neighbor_ratio': 0.5},
    # --- Bulyan ---
    "bulyan_cifar100_cnn_backdoor": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_cifar100_cnn_labelflip": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_femnist_cnn_backdoor": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_femnist_cnn_labelflip": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_mlp_texas100_baseline_backdoor": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_mlp_texas100_baseline_labelflip": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_textcnn_trec_baseline_backdoor": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    "bulyan_textcnn_trec_baseline_labelflip": {'aggregation.method': 'bulyan', 'aggregation.bulyan.num_byzantine': 1},
    # --- FoolsGold ---
    "foolsgold_cifar100_cnn_backdoor": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_cifar100_cnn_labelflip": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_femnist_cnn_backdoor": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_femnist_cnn_labelflip": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_mlp_texas100_baseline_backdoor": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_mlp_texas100_baseline_labelflip": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_mlp_purchase100_baseline_backdoor": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_mlp_purchase100_baseline_labelflip": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_textcnn_trec_baseline_backdoor": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
    "foolsgold_textcnn_trec_baseline_labelflip": {'aggregation.method': 'foolsgold', 'aggregation.foolsgold.pardon_threshold': 0.5},
}

# --- Load from JSON if available, otherwise use hardcoded defaults ---
_loaded_golden = _load_json_params(_GOLDEN_PARAMS_FILE)
GOLDEN_TRAINING_PARAMS = _loaded_golden if _loaded_golden else _DEFAULT_GOLDEN_TRAINING_PARAMS

_loaded_defense = _load_json_params(_TUNED_DEFENSE_FILE)
TUNED_DEFENSE_PARAMS = _loaded_defense if _loaded_defense else _DEFAULT_TUNED_DEFENSE_PARAMS

if _loaded_golden:
    print(f"[config] Loaded GOLDEN_TRAINING_PARAMS from {_GOLDEN_PARAMS_FILE.name} ({len(GOLDEN_TRAINING_PARAMS)} models)")
if _loaded_defense:
    print(f"[config] Loaded TUNED_DEFENSE_PARAMS from {_TUNED_DEFENSE_FILE.name} ({len(TUNED_DEFENSE_PARAMS)} combos)")

def get_tuned_defense_params(
        defense_name: str,
        model_config_name: str,
        attack_state: str,
        explicit_attack_type: Optional[str] = None,
        default_attack_type_for_tuning: str = "backdoor"
) -> Optional[Dict[str, Any]]:  # <-- Return Optional
    """
    Intelligently retrieves the correct tuned defense parameters from the
    global TUNED_DEFENSE_PARAMS dictionary.
    """

    if defense_name == "fedavg":
        return {"aggregation.method": "fedavg"}

    attack_type_key = default_attack_type_for_tuning  # Start with the default

    if explicit_attack_type:
        # If Step 5 or 12 passes "backdoor" or "labelflip", use it.
        attack_type_key = explicit_attack_type
    elif attack_state == "no_attack":
        # For a no_attack run (like in Step 4), use the default.
        attack_type_key = default_attack_type_for_tuning

    # Build the specific key
    tuned_params_key = f"{defense_name}_{model_config_name}_{attack_type_key}"

    if tuned_params_key not in TUNED_DEFENSE_PARAMS:
        print(f"!!!!!!!!!! FATAL WARNING !!!!!!!!!!!")
        print(f"  Could not find tuned params for key: '{tuned_params_key}'")
        print(f"  Please check your TUNED_DEFENSE_PARAMS in config_common_utils.py")

        return None

    return TUNED_DEFENSE_PARAMS[tuned_params_key]

# ==============================================================================
# --- Defense + attack scoping (single source of truth) ---
# ==============================================================================
# To drop a defense or attack from ALL experiment steps, remove it from
# the appropriate set below. Every step generator imports ENABLED_DEFENSES /
# ENABLED_ATTACK_TYPES from this module and filters its target list.
#
# Currently dropped:
#   - skymask_small : redundant with skymask, doubled GPU cost
#   - daved         : not in current paper scope
#   - labelflip     : tuning on backdoor alone is sufficient
ENABLED_DEFENSES = {
    "fedavg",
    "fltrust",
    "martfl",
    "skymask",
    "trimmed_mean",
    "multi_krum",
    "rflpa",
    "spmc",
    # "daved",
    "flame",
    "deepsight",
    "bulyan",
    "foolsgold",
    # "skymask_small",   # disabled
}

ENABLED_ATTACK_TYPES = {
    "backdoor",
    # "labelflip",       # disabled
}


def is_defense_enabled(defense_name: str) -> bool:
    return defense_name in ENABLED_DEFENSES


def is_attack_enabled(attack_type: str) -> bool:
    return attack_type in ENABLED_ATTACK_TYPES


def filter_enabled_defenses(defenses):
    """Return only the defenses in `defenses` that appear in ENABLED_DEFENSES."""
    return [d for d in defenses if d in ENABLED_DEFENSES]


def filter_enabled_attacks(attacks):
    return [a for a in attacks if a in ENABLED_ATTACK_TYPES]


# Helper lists — these are the canonical lists every generator imports.
# They are AUTOMATICALLY filtered against ENABLED_DEFENSES, so dropping a
# defense from the set above removes it everywhere.
_ALL_DEFENSES_RAW = [
    "fedavg", "fltrust", "martfl", "skymask", "skymask_small",
    "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
    "flame", "deepsight", "bulyan", "foolsgold",
]
# FLAME / DeepSight / Bulyan / FoolsGold are now first-class members of the
# canonical defense lists. Every "all-defenses" generator (steps 3, 4, 5, 6, 7,
# 8, 9, 10, 15, 16, 17) iterates over IMAGE_DEFENSES / TEXT_TABULAR_DEFENSES
# and will pick them up automatically. Deep-dive steps (13 = drowning,
# 14 = MartFL collusion) hardcode their own defense lists and are unaffected.
_IMAGE_DEFENSES_RAW = [
    "fedavg",  # no-defense baseline — required for headline table
    "fltrust", "martfl", "skymask", "skymask_small",
    "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
    "flame", "deepsight", "bulyan", "foolsgold",
]
_TEXT_TABULAR_DEFENSES_RAW = [
    "fedavg",  # no-defense baseline — required for headline table
    "fltrust", "martfl", "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
    "flame", "deepsight", "bulyan", "foolsgold",
]
_NEW_IMAGE_DEFENSES_RAW = ["flame", "deepsight", "bulyan", "foolsgold"]
_NEW_TEXT_TABULAR_DEFENSES_RAW = ["flame", "deepsight", "bulyan", "foolsgold"]

ALL_DEFENSES = filter_enabled_defenses(_ALL_DEFENSES_RAW)
IMAGE_DEFENSES = filter_enabled_defenses(_IMAGE_DEFENSES_RAW)
TEXT_TABULAR_DEFENSES = filter_enabled_defenses(_TEXT_TABULAR_DEFENSES_RAW)

# NEW_*_DEFENSES are kept as standalone lists so step 19 can still target only
# the new four when run on its own. They're a SUBSET of IMAGE_DEFENSES /
# TEXT_TABULAR_DEFENSES now, not a complement.
NEW_IMAGE_DEFENSES = filter_enabled_defenses(_NEW_IMAGE_DEFENSES_RAW)
NEW_TEXT_TABULAR_DEFENSES = filter_enabled_defenses(_NEW_TEXT_TABULAR_DEFENSES_RAW)

# FOCUSED_DEFENSES — used by the in-depth analysis steps (4, 5, 6, 7, 8, 9)
# to limit cost. The "headline" steps (3 tuning, 10 main summary) still use
# the full IMAGE_DEFENSES list. Step 13 (drowning) and Step 14 (collusion)
# already hardcode their own deep-dive defense subsets.
#
# Rationale: the user is on a deadline. The in-depth steps cross-cut the same
# defenses as the main summary, so collapsing them to a focused 2-defense set
# (FLTrust + MartFL — the paper's primary subjects) cuts ~50% of the priority
# pipeline cost without weakening the headline result. Other defenses' main-
# summary results remain in Step 10 (Table X) for cross-paradigm comparison.
#
# To re-expand later: replace this list with IMAGE_DEFENSES at the call site.
_FOCUSED_DEFENSES_RAW = ["fltrust", "martfl"]
FOCUSED_DEFENSES = filter_enabled_defenses(_FOCUSED_DEFENSES_RAW)

# ==============================================================================
# FILTERING_DEFENSES — used by the in-depth analysis steps to evaluate the
# subset of defenses that perform explicit per-seller filtering, i.e. that
# produce a per-seller accept/reject (or weight) decision per round, not just
# a robust aggregate.
# ==============================================================================
# This scope expresses the paper's central claim: in a marketplace, simply
# tolerating malicious gradients in the aggregate is not enough — honest
# sellers must be paid and adversarial sellers must not be, so a defense must
# *identify* who the malicious sellers are. Defenses like Trimmed-Mean, RFLPA,
# and SPMC produce a robust aggregate without producing per-seller decisions
# and are therefore reported only in the headline main-summary table (Step 10),
# not in the in-depth analyses (Steps 4, 5, 7, 8, 9, 15).
#
# Defense families covered:
#   - Baseline                    : fedavg
#   - Trust-weighted              : fltrust, martfl
#   - Distance-based selection    : multi_krum, bulyan
#   - Clustering / similarity     : flame, deepsight, foolsgold
#   - Mask-based (image only)     : skymask
#
# Image-only variant includes skymask. Tabular/text variant drops it because
# the mask-learning meta-optimization assumes a CNN substrate.
_FILTERING_DEFENSES_RAW = [
    "fedavg",        # no-defense baseline
    "fltrust",       # trust-weighted (per-seller trust score)
    "martfl",        # clustering / trusted-set selection
    "multi_krum",    # distance-based selection (k closest)
    "bulyan",        # Krum + coordinate trim, explicit per-seller selection
    "flame",         # clustering by gradient similarity
    "deepsight",     # neuron-level clustering
    "foolsgold",     # historical cosine similarity (anti-sybil)
    "skymask",       # mask-based per-seller filtering (image-only)
]
FILTERING_DEFENSES = filter_enabled_defenses(_FILTERING_DEFENSES_RAW)
FILTERING_DEFENSES_IMAGE = FILTERING_DEFENSES
FILTERING_DEFENSES_NON_IMAGE = [d for d in FILTERING_DEFENSES if "skymask" not in d]

# ==============================================================================
# --- Dataset scoping (single source of truth for which datasets to run) ---
# ==============================================================================
# Currently scoped to CIFAR100 + Texas100 + TREC to keep iteration cycles short.
# To re-enable FEMNIST and Purchase100, uncomment the relevant entries below.
# Every multi-dataset step generator (Step 1, 3, 10, 14, 19) imports
# ENABLED_DATASETS / ENABLED_MODEL_CONFIGS from this module and filters its
# target list before generating configs.
ENABLED_DATASETS = {
    "CIFAR100",
    "Texas100",
    "TREC",
    "FEMNIST",       # re-enabled for all-dataset main-conclusion runs
    "Purchase100",   # re-enabled for all-dataset main-conclusion runs
}

ENABLED_MODEL_CONFIGS = {
    "cifar100_cnn",
    "mlp_texas100_baseline",
    "textcnn_trec_baseline",
    "femnist_cnn",
    "mlp_purchase100_baseline",
}


def is_dataset_enabled(dataset_name: str) -> bool:
    """True if the dataset should be included in the current experiment scope."""
    return dataset_name in ENABLED_DATASETS


def is_model_config_enabled(model_config_name: str) -> bool:
    """True if the model_config should be included in the current experiment scope."""
    return model_config_name in ENABLED_MODEL_CONFIGS


def filter_enabled_targets(targets, dataset_key="dataset_name"):
    """Filter a list of target dicts (from a TUNING_CONFIGS-style structure) to
    only those whose dataset name is in ENABLED_DATASETS. Use this in step
    generators to scope which datasets get configs generated for them."""
    return [t for t in targets if t.get(dataset_key) in ENABLED_DATASETS]

# ==============================================================================
# --- 3. Define Shared Parameters & Modifiers ---
# ==============================================================================
NUM_SEEDS_PER_CONFIG = 2  # Reduced from 3 — still enough for error bars
DEFAULT_ADV_RATE = 0.3
DEFAULT_POISON_RATE = 0.5  # Match defense tuning


def disable_all_attacks(config: AppConfig) -> AppConfig:
    """
    This modifier disables all attack flags across the entire configuration
    to create a purely benign (non-adversarial) experiment setting.
    """

    # 1. Set main adversary rate to 0
    config.experiment.adv_rate = 0.0

    # 2. Disable all Adversarial Seller attacks
    adv_seller_cfg = config.adversary_seller_config
    adv_seller_cfg.poisoning.type = PoisonType.NONE
    adv_seller_cfg.sybil.is_sybil = False
    adv_seller_cfg.adaptive_attack.is_active = False
    adv_seller_cfg.drowning_attack.is_active = False
    adv_seller_cfg.mimicry_attack.is_active = False

    # 3. Disable all Server attacks (e.g., gradient inversion)
    config.server_attack_config.attack_name = "none"

    # 4. Disable all Buyer attacks
    config.buyer_attack_config.is_active = False
    config.buyer_attack_config.attack_type = "none"

    return config

# --- Valuation Config Helper ---
def enable_valuation(config: AppConfig, influence: bool = True, loo: bool = False, kernelshap: bool = False,
                     loo_freq: int = 10, kshap_freq: int = 20,
                     kshap_samples: int = 500) -> AppConfig:  # <-- ADD kshap_samples HERE

    config.valuation.run_influence = influence
    config.valuation.run_loo = loo
    config.valuation.run_kernelshap = kernelshap
    config.valuation.loo_frequency = loo_freq
    config.valuation.kernelshap_frequency = kshap_freq

    # Add this line to actually use the new parameter
    config.valuation.kernelshap_samples = kshap_samples

    return config


def use_sybil_attack_strategy(strategy: str, **kwargs) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier function that enables Sybil attack with a specific strategy
    and optional strategy-specific parameters.

    Args:
        strategy: "oracle_blend", "systematic_probe", "mimic", "pivot", etc.
        **kwargs: Additional parameters for the strategy (e.g., blend_alpha for oracle)
    """

    def modifier(config: AppConfig) -> AppConfig:
        sybil_cfg = config.adversary_seller_config.sybil
        sybil_cfg.is_sybil = True
        sybil_cfg.gradient_default_mode = strategy

        # Add any extra strategy-specific parameters
        # Example: Oracle blending factor
        if strategy == "oracle_blend":
            sybil_cfg.oracle_blend_alpha = kwargs.get("blend_alpha", 0.1)  # Default 10% malicious

        # Example: Parameters for systematic probing (if needed in config)
        # if strategy == "systematic_probe":
        #     sybil_cfg.probe_num_variations = kwargs.get("num_probes", 5)
        #     sybil_cfg.probe_noise_scale = kwargs.get("probe_scale", 0.01)

        # Ensure base poisoning is active (Sybil modifies *how* poison is delivered)
        # Assuming the attack modifier (e.g., use_image_backdoor_attack) is also applied
        # If not, you might need to set config.adversary_seller_config.poisoning.type here too.

        return config

    return modifier
