"""
Diagnostic for Step 20 (sleeper agent) and Step 11 (free-rider).

For each step, computes:
  1. EXPECTED scenarios (per the generator's loops)
  2. EXPECTED tuned-defense entries needed by those scenarios
  3. ACTUAL entries present in tuned_defense_params.json
  4. ACTUAL config YAMLs found under configs_generated_benchmark/
  5. ACTUAL result dirs and .success markers

Then prints a per-section diff showing exactly which combinations are missing.

Usage:
    python experiments/gradient_market/debug_new_steps.py
    python experiments/gradient_market/debug_new_steps.py --step 20
    python experiments/gradient_market/debug_new_steps.py --verbose

Designed to be run from the project root with no special PYTHONPATH —
imports are pure stdlib (json, pathlib) so it works even if torch is broken.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


# ---------- Hardcoded sweep specs (mirrors the generator scripts) ----------

STEP20_DEFENSES = [
    "fedavg", "fltrust", "trimmed_mean", "multi_krum", "spmc", "daved",
    "martfl", "rflpa", "foolsgold",
]
STEP20_STRIKES = ["knock_out", "drowning", "alie"]
STEP20_DORMANT = [0, 25, 50, 100]

STEP11_DEFENSES = [
    "fedavg", "fltrust", "martfl", "skymask", "skymask_small",
    "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
    "flame", "deepsight", "bulyan", "foolsgold",
]
STEP11_STRATEGIES = ["norm_noise", "echo", "stale_local"]
STEP11_RATES = [0.1, 0.3, 0.5]   # 0.0 emitted once per defense as control

MODEL_CONFIG_NAME = "cifar100_cnn"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUNED_PARAMS_FILE = PROJECT_ROOT / "experiments" / "gradient_market" / \
                    "configs_generation" / "tuned_params" / "tuned_defense_params.json"
GOLDEN_PARAMS_FILE = PROJECT_ROOT / "experiments" / "gradient_market" / \
                     "configs_generation" / "tuned_params" / "golden_training_params.json"
CONFIGS_BASE = PROJECT_ROOT / "configs_generated_benchmark"
RESULTS_BASE = PROJECT_ROOT / "results"


def expected_step20_scenarios():
    out = []
    for d in STEP20_DEFENSES:
        for strike in STEP20_STRIKES:
            for dormant in STEP20_DORMANT:
                out.append(f"step5b_sleeper_{d}_strike_{strike}_dormant_{dormant}")
    return out


def expected_step11_scenarios():
    out = []
    for d in STEP11_DEFENSES:
        out.append(f"step11_freerider_{d}_control")
        for strat in STEP11_STRATEGIES:
            for rate in STEP11_RATES:
                out.append(f"step11_freerider_{d}_{strat}_rate_{rate}")
    return out


def expected_tuned_keys(defenses, attack_type="backdoor"):
    """Tuned param keys consumed by get_tuned_defense_params."""
    return [f"{d}_{MODEL_CONFIG_NAME}_{attack_type}" for d in defenses if d != "fedavg"]


def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  WARN: failed to read {path}: {e}")
        return None


def report_section(label, expected, present, verbose=False):
    expected_set = set(expected)
    present_set = set(present)
    missing = sorted(expected_set - present_set)
    extra = sorted(present_set - expected_set)
    print(f"  {label}:")
    print(f"    expected: {len(expected_set)}")
    print(f"    present : {len(expected_set & present_set)}")
    print(f"    missing : {len(missing)}")
    if missing:
        head = missing if verbose else missing[:8]
        for m in head:
            print(f"      - {m}")
        if not verbose and len(missing) > 8:
            print(f"      ... and {len(missing) - 8} more")
    if extra and verbose:
        print(f"    unexpected ({len(extra)}):")
        for e in extra[:8]:
            print(f"      + {e}")
    return missing


def scan_configs_dir(subdir_name, scenario_prefix):
    """Returns list of scenario names actually present under configs_generated_benchmark/<subdir_name>/."""
    base = CONFIGS_BASE / subdir_name
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and p.name.startswith(scenario_prefix))


def scan_results_dir(scenario_prefix):
    """Returns (all_dirs, completed_dirs) for results matching prefix."""
    if not RESULTS_BASE.exists():
        return [], []
    matches = [p for p in RESULTS_BASE.iterdir()
               if p.is_dir() and p.name.startswith(scenario_prefix)]
    completed = []
    for m in matches:
        # Look for .success marker recursively
        if any(m.rglob(".success")):
            completed.append(m.name)
    return sorted(p.name for p in matches), sorted(completed)


def diagnose_step(step_num, verbose=False):
    print(f"\n{'=' * 70}")
    print(f"  STEP {step_num} DIAGNOSTIC")
    print(f"{'=' * 70}")

    if step_num == 20:
        scenarios = expected_step20_scenarios()
        defenses = STEP20_DEFENSES
        cfg_subdir = "step20_sleeper_agent"
        cfg_prefix = "step5b_sleeper_"
        result_prefix = "step5b_sleeper_"
    else:  # 11
        scenarios = expected_step11_scenarios()
        defenses = STEP11_DEFENSES
        cfg_subdir = "step11_free_riding"
        cfg_prefix = "step11_freerider_"
        result_prefix = "step11_freerider_"

    print(f"\n[1/4] Expected scenarios:        {len(scenarios)}")
    print(f"[1/4] Expected defenses:         {len(defenses)} -> {defenses}")

    # --- 2. Tuned-defense param coverage ---
    print(f"\n[2/4] Tuned-defense param coverage")
    tuned = load_json_safe(TUNED_PARAMS_FILE)
    if tuned is None:
        print(f"  ERROR: {TUNED_PARAMS_FILE} not found")
    else:
        expected_keys = expected_tuned_keys(defenses)
        present_keys = [k for k in expected_keys if k in tuned]
        report_section("tuned defense entries", expected_keys, present_keys, verbose=verbose)

    # --- 2b. Golden training params for cifar100_cnn ---
    print(f"\n[2b] Golden training params")
    golden = load_json_safe(GOLDEN_PARAMS_FILE)
    if golden is None:
        print(f"  ERROR: {GOLDEN_PARAMS_FILE} not found")
    elif MODEL_CONFIG_NAME not in golden:
        print(f"  MISSING: '{MODEL_CONFIG_NAME}' not in golden_training_params")
    else:
        gp = golden[MODEL_CONFIG_NAME]
        print(f"  OK: {MODEL_CONFIG_NAME} -> {gp.get('training.optimizer')}, "
              f"lr={gp.get('training.learning_rate')}, "
              f"epochs={gp.get('training.local_epochs')}")

    # --- 3. Generated config YAMLs ---
    print(f"\n[3/4] Generated config directories under {CONFIGS_BASE / cfg_subdir}")
    if not (CONFIGS_BASE / cfg_subdir).exists():
        print(f"  ERROR: directory does not exist (run --generate_only first)")
    else:
        present_scenarios = scan_configs_dir(cfg_subdir, cfg_prefix)
        report_section("scenario dirs", scenarios, present_scenarios, verbose=verbose)

    # --- 4. Result dirs and .success markers ---
    print(f"\n[4/4] Result dirs and .success markers")
    all_dirs, completed = scan_results_dir(result_prefix)
    print(f"  result dirs found  : {len(all_dirs)}")
    print(f"  with .success      : {len(completed)}")
    print(f"  missing/failed     : {len(all_dirs) - len(completed)}")
    if all_dirs and not completed:
        print(f"  NOTE: 0 successes — either runs were killed mid-flight, or every run errored.")
        print(f"        Check the runner log for the actual error:")
        print(f"          tail -200 configs_generated_benchmark/{cfg_subdir}_runner.log")
    if all_dirs:
        # Show which scenarios are completely missing from results
        scenario_set = set(scenarios)
        result_set = set(all_dirs)
        not_yet_attempted = sorted(scenario_set - result_set)
        if not_yet_attempted:
            print(f"  scenarios never attempted: {len(not_yet_attempted)}")
            head = not_yet_attempted if verbose else not_yet_attempted[:5]
            for s in head:
                print(f"    - {s}")
            if not verbose and len(not_yet_attempted) > 5:
                print(f"    ... and {len(not_yet_attempted) - 5} more")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, choices=[11, 20], default=None,
                   help="Diagnose just one step (default: both)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full lists instead of head/tail summaries")
    args = p.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tuned params: {TUNED_PARAMS_FILE}")
    print(f"Golden params: {GOLDEN_PARAMS_FILE}")

    steps = [args.step] if args.step else [20, 11]
    for s in steps:
        diagnose_step(s, verbose=args.verbose)

    print(f"\n{'=' * 70}")
    print("  REMEDIATION HINTS")
    print(f"{'=' * 70}")
    print("""
  * Missing tuned-defense entries  -> run Step 3 with the missing defenses
                                       included, or accept dataclass defaults.
  * Missing config YAMLs           -> rerun:
        python experiments/gradient_market/run_full_benchmark.py --step <N> --generate_only
  * Result dirs without .success   -> a run errored. Inspect:
        tail -200 configs_generated_benchmark/<subdir>_runner.log
        find results/<prefix>* -name '.success' -prune -o -type f -name '*.log' -print
""")


if __name__ == "__main__":
    main()
