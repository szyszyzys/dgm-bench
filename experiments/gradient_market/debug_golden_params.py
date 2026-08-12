"""
Diagnostic for golden_training_params.json gaps.

Traces *why* a given model_config_name (e.g. 'femnist_cnn') is missing from
golden_training_params.json by walking the actual Step 1 results dirs and
applying exactly the same logic as auto_update_golden_params.analyze_step1.

For each expected model it reports one of:
  * OK              -> entry present, prints chosen HPs + acc
  * NO SCENARIO DIR -> step1_tune_fedavg_<...> directory does not exist
                       (Step 1 was never run for this model)
  * NO HP RUNS      -> scenario dir exists but contains no opt_*_lr_*_epochs_*
                       sub-directories (config generation failed silently)
  * NO METRICS      -> HP runs exist but no final_metrics.json files
                       (every run errored before completion)
  * INVALID METRICS -> final_metrics.json files exist but none have an 'acc'
                       field (training crashed mid-eval)
  * EXCLUDED        -> records were collected but the analyzer rejected them

For NO METRICS cases the script also dumps the last few lines of the most
recently modified training_log.csv / runner log it can find under that
scenario dir, so you see the actual error.

Usage:
    python experiments/gradient_market/debug_golden_params.py
    python experiments/gradient_market/debug_golden_params.py --model femnist_cnn
    python experiments/gradient_market/debug_golden_params.py --verbose
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = PROJECT_ROOT / "results"
ALT_RESULTS = PROJECT_ROOT / "new_results_nolocalclip"
GOLDEN_PARAMS_FILE = (
    PROJECT_ROOT / "experiments" / "gradient_market" / "configs_generation"
    / "tuned_params" / "golden_training_params.json"
)
CONFIGS_GENERATED_BASE = PROJECT_ROOT / "configs_generated_benchmark"

# Pull enabled-dataset scope from the central config so disabled datasets
# don't get reported as missing gaps.
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gradient_market" / "configs_generation"))
try:
    from config_common_utils import ENABLED_MODEL_CONFIGS  # noqa: E402
except Exception:
    ENABLED_MODEL_CONFIGS = {
        "cifar100_cnn", "mlp_texas100_baseline", "textcnn_trec_baseline",
    }

# Mirror auto_update_golden_params.py
STEP1_SCENARIO_RE = re.compile(
    r"step1_tune_fedavg_(?P<modality>\w+?)_(?P<dataset>\w+?)_(?P<model_arch>\w+?)_(?P<data_setting>iid|noniid)$"
)
HP_RE = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[0-9.eE\-+]+)_epochs_(?P<epochs>\d+)")

# (modality, dataset, model_arch) -> model_config_name
MODEL_CONFIG_MAP = {
    ("image", "FEMNIST", "flexiblecnn"): "femnist_cnn",
    ("image", "CIFAR100", "flexiblecnn"): "cifar100_cnn",
    ("tabular", "Texas100", "mlp"): "mlp_texas100_baseline",
    ("tabular", "Purchase100", "mlp"): "mlp_purchase100_baseline",
    ("text", "TREC", "textcnn"): "textcnn_trec_baseline",
}
INVERSE_MODEL_MAP = {v: k for k, v in MODEL_CONFIG_MAP.items()}


def search_dirs():
    out = []
    if RESULTS_BASE.exists():
        out.append(RESULTS_BASE)
    if ALT_RESULTS.exists():
        out.append(ALT_RESULTS)
    return out


def find_scenario_dirs_for(model_config_name: str):
    """Return list of (base_dir, scenario_dir, data_setting) tuples that match this model."""
    if model_config_name not in INVERSE_MODEL_MAP:
        return []
    target_modality, target_dataset, target_arch = INVERSE_MODEL_MAP[model_config_name]
    found = []
    for base in search_dirs():
        for scenario in sorted(base.iterdir()):
            if not scenario.is_dir():
                continue
            m = STEP1_SCENARIO_RE.match(scenario.name)
            if not m:
                continue
            if (m.group("modality") == target_modality
                and m.group("dataset") == target_dataset
                and m.group("model_arch") == target_arch):
                found.append((base, scenario, m.group("data_setting")))
    return found


def scan_hp_runs(scenario_dir: Path):
    """Returns list of (hp_dir, has_metrics, acc, last_log_lines)."""
    out = []
    # HP folders live under scenario_dir at variable depth — find anything matching HP_RE
    for hp_dir in sorted({p.parent for p in scenario_dir.rglob("config.yaml")
                          if HP_RE.search(str(p.parent))}):
        if not HP_RE.search(hp_dir.name):
            # Use the closest ancestor that matches HP_RE
            hp_dir = next((a for a in [hp_dir, *hp_dir.parents] if HP_RE.search(a.name)), hp_dir)
        metrics_files = list(hp_dir.rglob("final_metrics.json"))
        acc = None
        if metrics_files:
            try:
                data = json.loads(metrics_files[0].read_text())
                acc = data.get("acc")
            except Exception:
                pass
        # Look for the most recent log file
        log_lines = []
        log_candidates = (
            list(hp_dir.rglob("training_log.csv"))
            + list(hp_dir.rglob("*.log"))
            + list(hp_dir.rglob("error*.txt"))
        )
        if log_candidates:
            log_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                log_lines = log_candidates[0].read_text(errors="ignore").splitlines()[-15:]
            except Exception:
                pass
        out.append({
            "hp_dir": hp_dir,
            "has_metrics": bool(metrics_files),
            "acc": acc,
            "log_tail": log_lines,
        })
    return out


def diagnose_model(model_config_name: str, golden: dict, verbose: bool):
    print(f"\n{'=' * 70}")
    print(f"  MODEL: {model_config_name}")
    print(f"{'=' * 70}")

    if model_config_name in golden:
        gp = golden[model_config_name]
        print(f"  STATUS: OK")
        print(f"    optimizer    : {gp.get('training.optimizer')}")
        print(f"    lr           : {gp.get('training.learning_rate')}")
        print(f"    local_epochs : {gp.get('training.local_epochs')}")
        print(f"    best_acc     : {gp.get('_best_acc')}")
        print(f"    data_setting : {gp.get('_data_setting')}")
        return

    print(f"  STATUS: MISSING")

    if model_config_name not in INVERSE_MODEL_MAP:
        print(f"  REASON: '{model_config_name}' is not in MODEL_CONFIG_MAP at all.")
        print(f"          analyze_step1 will never produce an entry for it.")
        print(f"          Known model_config_names: {sorted(MODEL_CONFIG_MAP.values())}")
        return

    modality, dataset, arch = INVERSE_MODEL_MAP[model_config_name]
    print(f"  Looking for scenario dirs matching:")
    print(f"    pattern: step1_tune_fedavg_{modality}_{dataset}_{arch}_(iid|noniid)")

    scenario_matches = find_scenario_dirs_for(model_config_name)

    if not scenario_matches:
        print(f"\n  REASON: NO SCENARIO DIR")
        print(f"  No directory matches the expected pattern in any of:")
        for base in search_dirs():
            print(f"    - {base}")
        # Also check if config generation produced anything for this model
        cfg_dir = CONFIGS_GENERATED_BASE / "step1_iid_tuning"
        if cfg_dir.exists():
            cfg_matches = sorted(p.name for p in cfg_dir.iterdir()
                                 if p.is_dir() and dataset in p.name and arch in p.name)
            if cfg_matches:
                print(f"\n  But configs WERE generated under {cfg_dir}:")
                for c in cfg_matches[:5]:
                    print(f"    {c}")
                print(f"  -> This means Step 1 generation succeeded but the runner")
                print(f"     never executed these configs. Check:")
                print(f"     tail -200 {CONFIGS_GENERATED_BASE}/step1_iid_tuning_runner.log")
            else:
                print(f"\n  No configs found under {cfg_dir} for this dataset either.")
                print(f"  -> Step 1 generation was never run for {dataset}.")
                print(f"     Run: python experiments/gradient_market/run_full_benchmark.py --step 1")
        return

    print(f"\n  Found {len(scenario_matches)} scenario dir(s):")
    total_hp_runs = 0
    total_with_metrics = 0
    total_with_valid_acc = 0
    one_log_dump = None

    for base, scenario, data_setting in scenario_matches:
        print(f"    [{data_setting}] {scenario.relative_to(base)}")
        hp_runs = scan_hp_runs(scenario)
        n_with_metrics = sum(1 for r in hp_runs if r["has_metrics"])
        n_with_valid_acc = sum(1 for r in hp_runs if r["acc"] is not None)
        print(f"        HP runs found    : {len(hp_runs)}")
        print(f"        with metrics     : {n_with_metrics}")
        print(f"        with valid 'acc' : {n_with_valid_acc}")
        total_hp_runs += len(hp_runs)
        total_with_metrics += n_with_metrics
        total_with_valid_acc += n_with_valid_acc

        if verbose and hp_runs:
            for r in hp_runs[:3]:
                tag = "OK" if r["acc"] is not None else ("METRICS NO ACC" if r["has_metrics"] else "NO METRICS")
                print(f"          {tag:<14}  {r['hp_dir'].name}")

        # Capture one error log to display at the end
        if one_log_dump is None:
            for r in hp_runs:
                if r["acc"] is None and r["log_tail"]:
                    one_log_dump = (r["hp_dir"], r["log_tail"])
                    break

    # ----- Diagnose the failure -----
    print(f"\n  TOTAL: {total_hp_runs} HP runs, {total_with_metrics} with metrics, "
          f"{total_with_valid_acc} with valid acc")

    if total_hp_runs == 0:
        print(f"\n  REASON: NO HP RUNS")
        print(f"  Scenario dirs exist but contain no opt_*_lr_*_epochs_* sub-directories.")
        print(f"  -> Step 1 generation produced empty configs. Re-run --generate_only")
        print(f"     and inspect configs_generated_benchmark/step1_iid_tuning/")
    elif total_with_metrics == 0:
        print(f"\n  REASON: NO METRICS")
        print(f"  HP runs exist but every single one is missing final_metrics.json.")
        print(f"  -> All Step 1 runs for this model crashed before reaching evaluation.")
        if one_log_dump:
            hp_dir, lines = one_log_dump
            print(f"\n  Tail of latest log under {hp_dir.name}:")
            print(f"  {'-' * 60}")
            for ln in lines:
                print(f"    {ln}")
            print(f"  {'-' * 60}")
        print(f"\n  Common causes for FEMNIST specifically:")
        print(f"    - in_channels mismatch (FEMNIST is grayscale, model expects 3 channels)")
        print(f"    - dataset path missing under data_root")
        print(f"    - partition strategy='natural' requires writer_id field that may not exist")
    elif total_with_valid_acc == 0:
        print(f"\n  REASON: INVALID METRICS")
        print(f"  final_metrics.json files exist but none have an 'acc' field.")
        print(f"  -> Training likely produced NaN/Inf or eval was skipped.")
        print(f"     Inspect a sample: cat {scenario_matches[0][1]}/.../final_metrics.json")
    else:
        print(f"\n  REASON: EXCLUDED BY ANALYZER")
        print(f"  {total_with_valid_acc} runs have valid acc but the entry is still missing.")
        print(f"  This shouldn't happen with the current analyze_step1 logic.")
        print(f"  -> Re-run: python experiments/gradient_market/auto_update_golden_params.py --step 1")
        print(f"     and check if it picks them up.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None,
                   help="Diagnose just one model (e.g. femnist_cnn). Default: all.")
    p.add_argument("--verbose", action="store_true",
                   help="Print individual HP run statuses.")
    args = p.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Golden params: {GOLDEN_PARAMS_FILE}")

    if not GOLDEN_PARAMS_FILE.exists():
        print(f"\n[ERROR] {GOLDEN_PARAMS_FILE} does not exist.")
        print(f"        Step 1 has never been analyzed. Run:")
        print(f"        python experiments/gradient_market/auto_update_golden_params.py --step 1")
        golden = {}
    else:
        try:
            golden = json.loads(GOLDEN_PARAMS_FILE.read_text())
        except Exception as e:
            print(f"\n[ERROR] Failed to read {GOLDEN_PARAMS_FILE}: {e}")
            return

    print(f"Existing entries in golden_training_params: {len(golden)}")
    if golden:
        print(f"  -> {sorted(golden.keys())}")

    if args.model:
        targets = [args.model]
    else:
        targets = sorted(m for m in MODEL_CONFIG_MAP.values() if m in ENABLED_MODEL_CONFIGS)
        skipped = sorted(set(MODEL_CONFIG_MAP.values()) - set(targets))
        if skipped:
            print(f"\n[scope] ENABLED_MODEL_CONFIGS = {sorted(ENABLED_MODEL_CONFIGS)}")
            print(f"[scope] Skipping disabled models: {skipped}")
            print(f"[scope] (To re-enable, edit ENABLED_DATASETS in config_common_utils.py)")
    for m in targets:
        diagnose_model(m, golden, verbose=args.verbose)

    print(f"\n{'=' * 70}")
    print("  NEXT STEPS")
    print(f"{'=' * 70}")
    print("""
  - For NO SCENARIO DIR : run Step 1 to generate + execute configs:
        python experiments/gradient_market/run_full_benchmark.py --step 1 \\
            --gpu_ids 0,1,2,3 --num_processes 4

  - For NO METRICS      : Step 1 ran but every config crashed.
        Read the error tail printed above, then fix the underlying bug
        (FEMNIST in_channels, dataset path, etc.).

  - For EXCLUDED        : the analyzer logic disagrees with the data.
        Re-run the analyzer in dry-run mode to see what it picks up:
        python experiments/gradient_market/auto_update_golden_params.py --step 1 --dry_run

  - After fixing, regenerate the JSON:
        python experiments/gradient_market/auto_update_golden_params.py --step 1
""")


if __name__ == "__main__":
    main()
