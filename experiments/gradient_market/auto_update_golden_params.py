"""
Auto-Analyze Step 1 & Step 3 Results
=====================================

Scans experiment results, finds optimal hyperparameters, saves them to JSON files,
which config_common_utils.py loads automatically.

Usage:
    # Analyze step 1 → golden_training_params.json
    python experiments/gradient_market/auto_update_golden_params.py --step 1

    # Analyze step 3 → tuned_defense_params.json
    python experiments/gradient_market/auto_update_golden_params.py --step 3

    # Dry run (print results without writing)
    python experiments/gradient_market/auto_update_golden_params.py --step 1 --dry_run

    # Both steps
    python experiments/gradient_market/auto_update_golden_params.py --step all
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PARAMS_DIR = SCRIPT_DIR / "configs_generation" / "tuned_params"

GOLDEN_PARAMS_FILE = PARAMS_DIR / "golden_training_params.json"
TUNED_DEFENSE_FILE = PARAMS_DIR / "tuned_defense_params.json"

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# Step 1: GOLDEN_TRAINING_PARAMS
# ============================================================================

# Regex for step 1 scenario folders
# e.g. step1_tune_fedavg_image_CIFAR10_flexiblecnn_noniid
STEP1_SCENARIO_RE = re.compile(
    r"step1_tune_fedavg_(?P<modality>\w+?)_(?P<dataset>\w+?)_(?P<model_arch>\w+?)_(?P<data_setting>iid|noniid)$"
)

# Regex for HP folder names: opt_Adam_lr_0.001_epochs_2
HP_RE = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[0-9.eE\-+]+)_epochs_(?P<epochs>\d+)")

# (modality, dataset, model_arch) -> model_config_name
MODEL_CONFIG_MAP = {
    ("image", "FEMNIST", "flexiblecnn"): "femnist_cnn",
    ("image", "CIFAR100", "flexiblecnn"): "cifar100_cnn",
    ("tabular", "Texas100", "mlp"): "mlp_texas100_baseline",
    ("tabular", "Purchase100", "mlp"): "mlp_purchase100_baseline",
    ("text", "TREC", "textcnn"): "textcnn_trec_baseline",
}

OPTIMIZER_DEFAULTS = {
    "Adam": {"training.momentum": 0.0, "training.weight_decay": 0.0},
    "SGD": {"training.momentum": 0.9, "training.weight_decay": 5e-4},
}


def _collect_step1_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Collect all step 1 metrics from result directories."""
    records = []

    search_dirs = [results_dir]
    alt = PROJECT_ROOT / "new_results_nolocalclip"
    if alt.exists():
        search_dirs.append(alt)

    for base in search_dirs:
        if not base.exists():
            continue
        for scenario_dir in sorted(base.iterdir()):
            if not scenario_dir.is_dir():
                continue
            m = STEP1_SCENARIO_RE.match(scenario_dir.name)
            if not m:
                continue

            modality, dataset, model_arch = m.group("modality"), m.group("dataset"), m.group("model_arch")
            data_setting = m.group("data_setting")
            model_config_name = MODEL_CONFIG_MAP.get((modality, dataset, model_arch))
            if not model_config_name:
                continue

            for metrics_file in scenario_dir.rglob("final_metrics.json"):
                hp_match = HP_RE.search(str(metrics_file))
                if not hp_match:
                    continue
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                acc = metrics.get("acc")
                if acc is None:
                    continue
                records.append({
                    "model_config_name": model_config_name,
                    "data_setting": data_setting,
                    "optimizer": hp_match.group("optimizer"),
                    "lr": float(hp_match.group("lr")),
                    "epochs": int(hp_match.group("epochs")),
                    "acc": acc,
                })
    return records


def analyze_step1(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Find best training HPs per model_config_name from step 1 results."""
    records = _collect_step1_metrics(results_dir)
    if not records:
        return {}

    # Find best acc per model, preferring noniid results since downstream
    # experiments use Dirichlet non-IID partitioning
    best: Dict[str, Dict[str, Any]] = {}
    for r in records:
        name = r["model_config_name"]
        current_is_noniid = r["data_setting"] == "noniid"
        existing = best.get(name)
        # Prefer noniid over iid; within same setting, prefer higher accuracy
        if (existing is None
                or (current_is_noniid and existing["_data_setting"] != "noniid")
                or (current_is_noniid == (existing["_data_setting"] == "noniid") and r["acc"] > existing["_best_acc"])):
            defaults = OPTIMIZER_DEFAULTS.get(r["optimizer"], {})
            best[name] = {
                "training.optimizer": r["optimizer"],
                "training.learning_rate": r["lr"],
                "training.local_epochs": r["epochs"],
                **defaults,
                "_best_acc": round(r["acc"], 4),
                "_data_setting": r["data_setting"],
            }
    return best


# ============================================================================
# Step 3: TUNED_DEFENSE_PARAMS
# ============================================================================

# Regex for step 3 scenario folders
# e.g. step3_tune_fltrust_backdoor_image_CIFAR10_cifar10_cnn
STEP3_SCENARIO_RE = re.compile(
    r"step3_tune_(?P<defense>\w+?)_(?P<attack>backdoor|labelflip)_(?P<modality>\w+?)_(?P<dataset>\w+?)_(?P<model_config>\w+)$"
)

# Regex patterns for defense HP folder names (must match run_parallel_experiment.py)
DEFENSE_HP_PATTERNS = {
    "fltrust": [
        ("aggregation.clip_norm", re.compile(r"aggregation\.clip_norm_([0-9.]+|None)")),
    ],
    "martfl": [
        ("aggregation.martfl.max_k", re.compile(r"aggregation\.martfl\.max_k_(\d+)")),
        ("aggregation.clip_norm", re.compile(r"aggregation\.clip_norm_([0-9.]+|None)")),
    ],
    "skymask": [
        ("aggregation.skymask.mask_epochs", re.compile(r"aggregation\.skymask\.mask_epochs_(\d+)")),
        ("aggregation.skymask.mask_lr", re.compile(r"aggregation\.skymask\.mask_lr_([0-9.eE\-+]+)")),
        ("aggregation.skymask.mask_threshold", re.compile(r"aggregation\.skymask\.mask_threshold_([0-9.]+)")),
        ("aggregation.clip_norm", re.compile(r"aggregation\.clip_norm_([0-9.]+|None)")),
    ],
    "skymask_small": [
        ("aggregation.skymask.mask_epochs", re.compile(r"aggregation\.skymask\.mask_epochs_(\d+)")),
        ("aggregation.skymask.mask_lr", re.compile(r"aggregation\.skymask\.mask_lr_([0-9.eE\-+]+)")),
        ("aggregation.skymask.mask_threshold", re.compile(r"aggregation\.skymask\.mask_threshold_([0-9.]+)")),
        ("aggregation.clip_norm", re.compile(r"aggregation\.clip_norm_([0-9.]+|None)")),
    ],
    "trimmed_mean": [
        ("aggregation.trimmed_mean.trim_ratio", re.compile(r"aggregation\.trimmed_mean\.trim_ratio_([0-9.]+)")),
    ],
    "multi_krum": [
        ("aggregation.multi_krum.num_byzantine", re.compile(r"aggregation\.multi_krum\.num_byzantine_(\d+)")),
        ("aggregation.multi_krum.m_selected", re.compile(r"aggregation\.multi_krum\.m_selected_(\d+)")),
    ],
    "rflpa": [
        ("aggregation.rflpa.element_clip", re.compile(r"aggregation\.rflpa\.element_clip_([0-9.]+)")),
        ("aggregation.clip_norm", re.compile(r"aggregation\.clip_norm_([0-9.]+|None)")),
    ],
    "spmc": [
        ("aggregation.spmc.temperature", re.compile(r"aggregation\.spmc\.temperature_([0-9.]+)")),
    ],
    "daved": [
        ("aggregation.daved.l2_relative_alpha", re.compile(r"aggregation\.daved\.l2_relative_alpha_([0-9.eE\-+]+)")),
        ("aggregation.daved.proj_lambda", re.compile(r"aggregation\.daved\.proj_lambda_([0-9.eE\-+]+)")),
    ],
}


def _parse_hp_folder(defense: str, folder_name: str) -> Optional[Dict[str, Any]]:
    """Extract defense HP values from a folder name."""
    patterns = DEFENSE_HP_PATTERNS.get(defense)
    if not patterns:
        return None

    params: Dict[str, Any] = {"aggregation.method": defense}
    for param_key, pattern in patterns:
        m = pattern.search(folder_name)
        if m:
            val_str = m.group(1)
            if val_str == "None":
                params[param_key] = None
            elif "." in val_str or "e" in val_str.lower():
                params[param_key] = float(val_str)
            else:
                params[param_key] = int(val_str)
    return params


def _collect_step3_metrics(results_dir: Path) -> List[Dict[str, Any]]:
    """Collect all step 3 metrics from result directories."""
    records = []

    if not results_dir.exists():
        return records

    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = STEP3_SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue

        defense = m.group("defense")
        attack = m.group("attack")
        model_config = m.group("model_config")

        for metrics_file in scenario_dir.rglob("final_metrics.json"):
            # Find the HP folder in the path (first directory under scenario)
            rel = metrics_file.relative_to(scenario_dir)
            if not rel.parts:
                continue
            hp_folder = rel.parts[0]

            hp_params = _parse_hp_folder(defense, hp_folder)
            if not hp_params:
                continue

            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            acc = metrics.get("acc")
            if acc is None:
                continue

            asr = metrics.get("asr", 0.0)

            records.append({
                "key": f"{defense}_{model_config}_{attack}",
                "defense": defense,
                "attack": attack,
                "model_config": model_config,
                "acc": acc,
                "asr": asr if asr is not None else 0.0,
                "hp_params": hp_params,
                "hp_folder": hp_folder,
            })

    return records


def analyze_step3(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Find best defense HPs per (defense, model, attack) combo.

    Selection criteria:
    - For backdoor: maximize acc while minimizing asr. Score = acc - asr
    - For labelflip: maximize acc. Score = acc
    """
    records = _collect_step3_metrics(results_dir)
    if not records:
        return {}

    # Group by key, then pick best per key
    # Average across seeds for the same HP combo
    grouped: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["key"]][r["hp_folder"]].append(r)

    best: Dict[str, Dict[str, Any]] = {}

    for key, hp_groups in grouped.items():
        best_score = -float("inf")
        best_entry = None

        for hp_folder, runs in hp_groups.items():
            avg_acc = sum(r["acc"] for r in runs) / len(runs)
            avg_asr = sum(r["asr"] for r in runs) / len(runs)
            attack = runs[0]["attack"]

            if attack == "backdoor":
                score = avg_acc - avg_asr  # want high acc, low asr
            else:
                score = avg_acc  # labelflip: just want high acc

            if score > best_score:
                best_score = score
                # Keep all params including None values (None means "no clipping" etc.)
                params = dict(runs[0]["hp_params"])
                best_entry = {
                    **params,
                    "_best_acc": round(avg_acc, 4),
                    "_best_asr": round(avg_asr, 4),
                    "_score": round(best_score, 4),
                    "_n_seeds": len(runs),
                }

        if best_entry:
            best[key] = best_entry

    # Also add fedavg entries (no tuning needed)
    model_configs = set()
    for r in records:
        model_configs.add(r["model_config"])
    for mc in model_configs:
        for attack in ["backdoor", "labelflip"]:
            fedavg_key = f"fedavg_{mc}_{attack}"
            if fedavg_key not in best:
                best[fedavg_key] = {"aggregation.method": "fedavg"}

    return best


# ============================================================================
# Save / Load helpers
# ============================================================================

def save_params(params: Dict, filepath: Path):
    """Save params dict to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    print(f"  Saved to: {filepath}")


def load_params(filepath: Path) -> Dict:
    """Load params dict from JSON file. Returns empty dict if file doesn't exist."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        return json.load(f)


def _print_step1_table(best: Dict[str, Dict[str, Any]]):
    print(f"\n  {'Model':<30s} {'Optimizer':<8s} {'LR':<10s} {'Epochs':<8s} {'Acc':<8s} {'Setting':<8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for name in sorted(best):
        p = best[name]
        print(f"  {name:<30s} {p['training.optimizer']:<8s} {p['training.learning_rate']:<10g} "
              f"{p['training.local_epochs']:<8d} {p['_best_acc']:<8.4f} {p.get('_data_setting', '?'):<8s}")


def _print_step3_table(best: Dict[str, Dict[str, Any]]):
    print(f"\n  {'Key':<50s} {'Acc':<8s} {'ASR':<8s} {'Score':<8s} {'Seeds':<6s}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for key in sorted(best):
        p = best[key]
        if p.get("aggregation.method") == "fedavg" and "_best_acc" not in p:
            continue  # Skip auto-generated fedavg placeholders
        acc = p.get("_best_acc", "?")
        asr = p.get("_best_asr", "?")
        score = p.get("_score", "?")
        seeds = p.get("_n_seeds", "?")
        print(f"  {key:<50s} {acc:<8} {asr:<8} {score:<8} {seeds:<6}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auto-analyze experiment results and save best HPs to JSON.")
    parser.add_argument("--step", type=str, required=True, choices=["1", "3", "all"],
                        help="Which step to analyze: 1, 3, or all")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Base directory containing result folders")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print results without writing files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    steps = ["1", "3"] if args.step == "all" else [args.step]

    for step in steps:
        if step == "1":
            print(f"\n{'='*60}")
            print("  Analyzing Step 1: Training HP Tuning")
            print(f"{'='*60}")
            print(f"  Scanning: {results_dir}")

            best = analyze_step1(results_dir)
            if not best:
                print("\n  No step 1 results found.")
                continue

            print(f"\n  Found best HPs for {len(best)} model(s):")
            _print_step1_table(best)

            if args.dry_run:
                print(f"\n  [DRY RUN] Would save to: {GOLDEN_PARAMS_FILE}")
            else:
                save_params(best, GOLDEN_PARAMS_FILE)
                print(f"\n  Done! config_common_utils.py will load these automatically.")

        elif step == "3":
            print(f"\n{'='*60}")
            print("  Analyzing Step 3: Defense HP Tuning")
            print(f"{'='*60}")
            print(f"  Scanning: {results_dir}")

            best = analyze_step3(results_dir)
            if not best:
                print("\n  No step 3 results found.")
                continue

            print(f"\n  Found best HPs for {len(best)} defense/model/attack combo(s):")
            _print_step3_table(best)

            if args.dry_run:
                print(f"\n  [DRY RUN] Would save to: {TUNED_DEFENSE_FILE}")
            else:
                save_params(best, TUNED_DEFENSE_FILE)
                print(f"\n  Done! config_common_utils.py will load these automatically.")

    if not args.dry_run:
        print(f"\n  Parameter files saved to: {PARAMS_DIR}/")


if __name__ == "__main__":
    main()
