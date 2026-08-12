"""
Validate Parameter Coverage
=============================

Checks that golden_training_params.json and tuned_defense_params.json
cover all model/defense/attack combinations needed by downstream steps.

Also recovers parameters from old CSV results if available.

Usage:
    # Validate coverage (reports gaps)
    python experiments/gradient_market/validate_params.py --check

    # Recover from old results (resbackup/)
    python experiments/gradient_market/validate_params.py --recover

    # Both
    python experiments/gradient_market/validate_params.py --check --recover
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PARAMS_DIR = SCRIPT_DIR / "configs_generation" / "tuned_params"
GOLDEN_FILE = PARAMS_DIR / "golden_training_params.json"
DEFENSE_FILE = PARAMS_DIR / "tuned_defense_params.json"

# Pull the enabled-dataset/defense/attack scope from config_common_utils so
# this validator stays in sync with whatever the generators are producing.
sys.path.insert(0, str(SCRIPT_DIR / "configs_generation"))
try:
    from config_common_utils import (  # noqa: E402
        ENABLED_MODEL_CONFIGS, ENABLED_DEFENSES, ENABLED_ATTACK_TYPES,
    )
except Exception:
    # Fallback (only used if config_common_utils can't be imported for some reason)
    ENABLED_MODEL_CONFIGS = {
        "cifar100_cnn", "mlp_texas100_baseline", "textcnn_trec_baseline",
    }
    ENABLED_DEFENSES = {
        "fedavg", "fltrust", "martfl", "skymask",
        "trimmed_mean", "multi_krum", "rflpa", "spmc",
        "flame", "deepsight", "bulyan", "foolsgold",
    }
    ENABLED_ATTACK_TYPES = {"backdoor"}

# ============================================================================
# What downstream steps NEED
# ============================================================================

# Models that Step 1 tunes (must all appear in golden_training_params).
# Filtered to ENABLED_MODEL_CONFIGS so disabled datasets don't show as gaps.
_ALL_MODELS = [
    "femnist_cnn",
    "cifar100_cnn",
    "mlp_texas100_baseline",
    "mlp_purchase100_baseline",
    "textcnn_trec_baseline",
]
STEP1_MODELS = [m for m in _ALL_MODELS if m in ENABLED_MODEL_CONFIGS]
STEP3_MODELS = [m for m in _ALL_MODELS if m in ENABLED_MODEL_CONFIGS]

# Defenses tuned in Step 3 (fedavg doesn't need tuning).
# Filtered to ENABLED_DEFENSES so disabled defenses don't show as gaps.
_STEP3_DEFENSES_RAW = [
    "fltrust", "martfl", "skymask", "skymask_small",
    "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
]
STEP3_DEFENSES = [d for d in _STEP3_DEFENSES_RAW if d in ENABLED_DEFENSES]

# Attack types in Step 3, filtered to ENABLED_ATTACK_TYPES.
_STEP3_ATTACKS_RAW = ["backdoor", "labelflip"]
STEP3_ATTACKS = [a for a in _STEP3_ATTACKS_RAW if a in ENABLED_ATTACK_TYPES]

# Required golden training param keys
REQUIRED_GOLDEN_KEYS = [
    "training.optimizer",
    "training.learning_rate",
    "training.local_epochs",
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ============================================================================
# CHECK: Validate coverage
# ============================================================================

def check_golden_coverage(golden: Dict) -> List[str]:
    """Check that golden_training_params covers all required models."""
    issues = []

    if not golden:
        issues.append("golden_training_params.json is EMPTY or MISSING (using hardcoded defaults)")
        return issues

    for model in STEP1_MODELS:
        if model not in golden:
            issues.append(f"MISSING model: '{model}' not in golden_training_params")
            continue

        params = golden[model]
        for key in REQUIRED_GOLDEN_KEYS:
            if key not in params:
                issues.append(f"MISSING key: '{model}' -> '{key}'")

    # Check for unexpected models (not necessarily a problem)
    extra = set(golden.keys()) - set(STEP1_MODELS)
    if extra:
        logger.info(f"  Extra models in golden params (OK): {extra}")

    return issues


def check_defense_coverage(defense_params: Dict) -> List[str]:
    """Check that tuned_defense_params covers all (defense, model, attack) combos."""
    issues = []

    if not defense_params:
        issues.append("tuned_defense_params.json is EMPTY or MISSING (using hardcoded defaults)")
        return issues

    for model in STEP3_MODELS:
        for attack in STEP3_ATTACKS:
            # FedAvg always needs an entry
            fedavg_key = f"fedavg_{model}_{attack}"
            if fedavg_key not in defense_params:
                issues.append(f"MISSING: '{fedavg_key}'")

            for defense in STEP3_DEFENSES:
                key = f"{defense}_{model}_{attack}"
                if key not in defense_params:
                    issues.append(f"MISSING: '{key}'")
                else:
                    params = defense_params[key]
                    if "aggregation.method" not in params:
                        issues.append(f"BAD: '{key}' missing 'aggregation.method'")

    return issues


def run_check():
    """Main validation check."""
    logger.info("=" * 60)
    logger.info("  Parameter Coverage Validation")
    logger.info("=" * 60)

    # Check golden training params
    logger.info("\n--- Golden Training Params ---")
    golden = load_json(GOLDEN_FILE)
    if golden:
        logger.info(f"  Source: {GOLDEN_FILE.name}")
        logger.info(f"  Models: {len(golden)}")
    else:
        logger.info("  Source: hardcoded defaults (no JSON file)")
        # Load defaults for validation
        try:
            from experiments.gradient_market.configs_generation.config_common_utils import GOLDEN_TRAINING_PARAMS
            golden = GOLDEN_TRAINING_PARAMS
            logger.info(f"  Defaults loaded: {len(golden)} models")
        except ImportError:
            golden = {}

    golden_issues = check_golden_coverage(golden)
    if golden_issues:
        for issue in golden_issues:
            logger.warning(f"  [!] {issue}")
    else:
        logger.info("  [OK] All models covered")

    # Show current values
    for model in STEP1_MODELS:
        if model in golden:
            p = golden[model]
            opt = p.get("training.optimizer", "?")
            lr = p.get("training.learning_rate", "?")
            ep = p.get("training.local_epochs", "?")
            acc = p.get("_best_acc", "?")
            logger.info(f"    {model:<30s} {opt:<6s} lr={lr:<8} ep={ep}  acc={acc}")
        else:
            logger.info(f"    {model:<30s} MISSING")

    # Check tuned defense params
    logger.info("\n--- Tuned Defense Params ---")
    defense = load_json(DEFENSE_FILE)
    if defense:
        logger.info(f"  Source: {DEFENSE_FILE.name}")
        logger.info(f"  Entries: {len(defense)}")
    else:
        logger.info("  Source: hardcoded defaults (no JSON file)")
        try:
            from experiments.gradient_market.configs_generation.config_common_utils import TUNED_DEFENSE_PARAMS
            defense = TUNED_DEFENSE_PARAMS
            logger.info(f"  Defaults loaded: {len(defense)} entries")
        except ImportError:
            defense = {}

    defense_issues = check_defense_coverage(defense)
    if defense_issues:
        # Group by type for readability
        missing = [i for i in defense_issues if "MISSING" in i]
        bad = [i for i in defense_issues if "BAD" in i]
        if missing:
            logger.warning(f"  [!] {len(missing)} missing entries:")
            for m in missing[:10]:
                logger.warning(f"      {m}")
            if len(missing) > 10:
                logger.warning(f"      ... and {len(missing) - 10} more")
        if bad:
            for b in bad:
                logger.warning(f"  [!] {b}")
    else:
        logger.info("  [OK] All defense/model/attack combos covered")

    # Summary
    total_issues = len(golden_issues) + len(defense_issues)
    logger.info(f"\n{'=' * 60}")
    if total_issues == 0:
        logger.info("  RESULT: All parameters validated successfully")
    else:
        logger.warning(f"  RESULT: {total_issues} issue(s) found")
        if not GOLDEN_FILE.exists() or not DEFENSE_FILE.exists():
            logger.info("  Note: Using hardcoded defaults is fine for now.")
            logger.info("  JSON files will be auto-generated after Steps 1 and 3 complete.")
    logger.info("=" * 60)

    return total_issues


# ============================================================================
# RECOVER: Extract params from old CSV results
# ============================================================================

OLD_RESULTS_PATHS = [
    PROJECT_ROOT / "resbackup" / "results" / "summary_avg.csv",
    PROJECT_ROOT / "resbackup" / "result_new" / "summary_avg.csv",
    PROJECT_ROOT / "resbackup" / "result_new" / "summary_individual_runs.csv",
]

# Map old CSV column names to new model config names
OLD_DATASET_MAP = {
    "CIFAR": "cifar10_cnn",
    "cifar": "cifar10_cnn",
    "CIFAR10": "cifar10_cnn",
    "CIFAR100": "cifar100_cnn",
    "FMNIST": None,  # Not used in current benchmark
    "TREC": "textcnn_trec_baseline",
    "trec": "textcnn_trec_baseline",
}

OLD_DEFENSE_MAP = {
    "fedavg": "fedavg",
    "fltrust": "fltrust",
    "martfl": "martfl",
    "skymask": "skymask",
}

OLD_ATTACK_MAP = {
    "single": "backdoor",
    "label_flip": "labelflip",
    "None": "backdoor",  # No attack = use backdoor tuning as default
}


def recover_golden_from_csv() -> Dict[str, Dict[str, Any]]:
    """
    Extract best training HPs from old CSV results.
    Looks for benign FedAvg runs (adv_rate=0) and picks the best accuracy.
    """
    recovered = {}

    for csv_path in OLD_RESULTS_PATHS:
        if not csv_path.exists():
            continue
        logger.info(f"  Scanning: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"    Failed to read: {e}")
            continue

        # Normalize column names
        df.columns = df.columns.str.strip()

        # Filter for benign FedAvg baseline runs
        if "ADV_RATE" in df.columns:
            benign = df[df["ADV_RATE"] == 0.0]
        else:
            benign = df

        if "AGGREGATION_METHOD" in df.columns:
            benign = benign[benign["AGGREGATION_METHOD"].str.lower() == "fedavg"]

        if benign.empty:
            continue

        # Extract best accuracy per dataset
        acc_col = None
        for col in ["FINAL_CLEAN_ACC", "FINAL_MAIN_ACC", "final_acc"]:
            if col in benign.columns:
                acc_col = col
                break

        if acc_col is None:
            continue

        dataset_col = "DATASET" if "DATASET" in benign.columns else None
        if dataset_col is None:
            continue

        for dataset_name, group in benign.groupby(dataset_col):
            model_name = OLD_DATASET_MAP.get(str(dataset_name).strip())
            if model_name is None:
                continue

            best_row = group.loc[group[acc_col].idxmax()]
            best_acc = best_row[acc_col]

            if model_name not in recovered or best_acc > recovered[model_name].get("_best_acc", 0):
                # Old results don't track optimizer/lr/epochs per run in summary
                # Use defaults based on what the old code typically used
                recovered[model_name] = {
                    "training.optimizer": "Adam",
                    "training.learning_rate": 0.001,
                    "training.local_epochs": 2,
                    "training.momentum": 0.0,
                    "training.weight_decay": 0.0,
                    "_best_acc": round(float(best_acc), 4),
                    "_source": "recovered_from_old_csv",
                }

                logger.info(f"    Recovered: {model_name} -> acc={best_acc:.4f}")

    return recovered


def recover_defense_from_csv() -> Dict[str, Dict[str, Any]]:
    """
    Extract defense performance from old CSV results.
    Picks the defense configuration with best accuracy under attack.
    """
    recovered = {}

    for csv_path in OLD_RESULTS_PATHS:
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df.columns = df.columns.str.strip()

        # Need: AGGREGATION_METHOD, DATASET, ATTACK_METHOD, FINAL_CLEAN_ACC
        required = ["AGGREGATION_METHOD", "DATASET"]
        if not all(c in df.columns for c in required):
            continue

        acc_col = next((c for c in ["FINAL_CLEAN_ACC", "FINAL_MAIN_ACC"] if c in df.columns), None)
        if acc_col is None:
            continue

        # Filter for attack runs
        if "ADV_RATE" in df.columns:
            attack_runs = df[df["ADV_RATE"] > 0]
        else:
            attack_runs = df

        for (defense, dataset), group in attack_runs.groupby(["AGGREGATION_METHOD", "DATASET"]):
            defense_norm = OLD_DEFENSE_MAP.get(str(defense).lower().strip())
            model_name = OLD_DATASET_MAP.get(str(dataset).strip())
            if defense_norm is None or model_name is None:
                continue

            attack_method = group["ATTACK_METHOD"].iloc[0] if "ATTACK_METHOD" in group.columns else "single"
            attack_norm = OLD_ATTACK_MAP.get(str(attack_method).strip(), "backdoor")

            key = f"{defense_norm}_{model_name}_{attack_norm}"

            best_row = group.loc[group[acc_col].idxmax()]
            best_acc = best_row[acc_col]

            if key not in recovered:
                recovered[key] = {
                    "aggregation.method": defense_norm,
                    "_best_acc": round(float(best_acc), 4),
                    "_source": "recovered_from_old_csv",
                }
                logger.info(f"    Recovered: {key} -> acc={best_acc:.4f}")

    return recovered


def run_recover():
    """Main recovery routine."""
    logger.info("\n" + "=" * 60)
    logger.info("  Recovering Parameters from Old Results")
    logger.info("=" * 60)

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Recover golden training params
    logger.info("\n--- Recovering Golden Training Params ---")
    recovered_golden = recover_golden_from_csv()

    if recovered_golden:
        # Merge with existing (don't overwrite better results)
        existing = load_json(GOLDEN_FILE)
        merged = {**existing}
        for model, params in recovered_golden.items():
            if model not in merged or params.get("_best_acc", 0) > merged[model].get("_best_acc", 0):
                merged[model] = params
                logger.info(f"  Updated: {model}")

        with open(GOLDEN_FILE, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        logger.info(f"  Saved {len(merged)} entries to {GOLDEN_FILE.name}")
    else:
        logger.info("  No old results found to recover golden params from.")

    # Recover defense params
    logger.info("\n--- Recovering Tuned Defense Params ---")
    recovered_defense = recover_defense_from_csv()

    if recovered_defense:
        existing = load_json(DEFENSE_FILE)
        merged = {**existing}
        for key, params in recovered_defense.items():
            if key not in merged:
                merged[key] = params

        with open(DEFENSE_FILE, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        logger.info(f"  Saved {len(merged)} entries to {DEFENSE_FILE.name}")
    else:
        logger.info("  No old results found to recover defense params from.")

    logger.info(f"\n{'=' * 60}")
    logger.info("  Recovery complete. Run --check to verify coverage.")
    logger.info("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate and recover experiment parameters.")
    parser.add_argument("--check", action="store_true", help="Validate parameter coverage")
    parser.add_argument("--recover", action="store_true", help="Recover params from old CSV results")
    args = parser.parse_args()

    if not args.check and not args.recover:
        args.check = True  # Default to check

    if args.recover:
        run_recover()

    if args.check:
        issues = run_check()
        sys.exit(1 if issues > 0 else 0)


if __name__ == "__main__":
    main()
