"""
Results Health Check Script
===========================
Scans experiment results and reports:
  1. Completion status per step (success / failed / in-progress / missing)
  2. Per-defense accuracy & ASR summary
  3. Flags results affected by the bugs we fixed (need rerun)
  4. Highlights suspicious patterns (NaN, zero accuracy, identical replicates)

Usage:
    python experiments/gradient_market/check_results.py --results_dir ./results
    python experiments/gradient_market/check_results.py --results_dir ./results --step 3
    python experiments/gradient_market/check_results.py --results_dir ./results --defense martfl
    python experiments/gradient_market/check_results.py --results_dir ./results --verbose
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ===========================================================================
# Expected output directories per CLI step (from STEP_REGISTRY in run_full_benchmark.py).
# Used for coverage check: did each step in rerun_priority.sh actually produce results?
# ===========================================================================

STEP_OUTPUT_DIRS = {
    # Map CLI step → list of scenario name prefixes that match its outputs
    # (results dirs use the scenario name, NOT the configs output_subdir)
    1:  ["step1_tune_"],
    2:  ["step2_validate_", "step2.5_find_hps_"],
    3:  ["step3_tune_"],
    4:  ["step5_atk_sens_"],
    5:  ["step6_adv_sybil_"],
    6:  ["step7_adaptive_", "step7_baseline_no_attack"],
    7:  ["step8_buyer_attack_", "step8b_combined_"],
    8:  ["step10_scalability_"],
    9:  ["step11_", "step9_comp_mimicry_"],
    10: ["step12_main_summary_"],
    11: ["step11_seller_only_", "step11_free_riding"],
    13: ["step13_drowning_"],
    14: ["step14_collusion_"],
    15: ["step15_pricing_"],
    16: ["step16_dp_"],
    17: ["step17_alie_"],
    # 18: DAVED comparison removed
    19: ["step19_new_defenses"],
    20: ["step20_sleeper_", "step5b_sleeper_"],
}

# Steps expected to have results after running rerun_priority.sh
EXPECTED_STEPS_AFTER_RERUN = [1, 3, 10, 4, 5, 6, 7, 9, 14, 8, 13, 15, 16, 17]

# Old/legacy scenario name patterns to ignore.
# The latest pipeline uses proper-case dataset names (CIFAR100, FEMNIST, Texas100, etc.)
# in scenario names. Legacy results don't use these.
#
# IMPORTANT: We CANNOT just match on lowercase dataset names like "_cifar100_"
# because model_config_name is always lowercase (e.g., "cifar100_cnn",
# "mlp_texas100_baseline") even in current scenarios.
#
# Strategy: A scenario is legacy if it contains NONE of the proper-case dataset
# markers AND its name structure looks like the old pre-fix format. The reliable
# signal is the absence of any proper-case marker, OR explicit obsolete prefixes.
PROPER_CASE_DATASETS = ["CIFAR100", "CIFAR10", "FEMNIST", "Texas100", "Purchase100", "TREC"]
OBSOLETE_PREFIXES = [
    "step3_bk",         # obsolete experiment naming
    "bk_step3",         # backup folder of old step 3 runs
    "step7_backup",     # obsolete
    "step9_comp_",      # unrelated to current step 9
]
OBSOLETE_SUFFIXES = [
    "_nolocalclip",     # obsolete experiment variant
    "_new",             # obsolete experiment variant
]

# Bug fingerprints: detect results produced by the OLD buggy code.
# Set to empty after a full rerun — the bugs are fixed, but this stays
# as a safety net if any pre-rerun results survived.
AFFECTED_DEFENSES = set()  # Was: {"skymask", "skymask_small", "martfl", "spmc"}
AFFECTED_DATASETS = set()  # Was: {"femnist"}


def is_legacy_scenario(scenario_name: str, full_path: str = "") -> bool:
    """Return True if scenario name matches an obsolete/old result pattern.

    Strategy:
      1. Explicit obsolete prefixes (step3_bk, step7_backup, etc.) → legacy
      2. Explicit obsolete suffixes (_nolocalclip, _new) → legacy
      3. If full_path contains an obsolete folder name → legacy
      4. If the scenario name contains NO proper-case dataset marker
         (CIFAR100, FEMNIST, Texas100, etc.) → legacy (old lowercase-only naming)
      5. Otherwise → current pipeline, keep it

    This handles the edge case where scenario names contain BOTH a proper-case
    dataset (e.g., CIFAR100) AND a lowercase model config (e.g., cifar100_cnn).
    Such names are CURRENT, not legacy.
    """
    if not scenario_name:
        return False

    # Rule 1: explicit obsolete prefixes (match on scenario name start)
    for prefix in OBSOLETE_PREFIXES:
        if scenario_name.startswith(prefix):
            return True

    # Rule 3 (moved up): if the full results path contains a backup folder
    if full_path:
        for prefix in OBSOLETE_PREFIXES:
            if f"/{prefix}/" in full_path.replace("\\", "/"):
                return True

    # Rule 2: explicit obsolete suffixes
    for suffix in OBSOLETE_SUFFIXES:
        if scenario_name.endswith(suffix) or f"{suffix}/" in scenario_name:
            return True

    # Rule 3: missing proper-case dataset marker → legacy lowercase-only naming
    has_proper_case_dataset = any(ds in scenario_name for ds in PROPER_CASE_DATASETS)
    if not has_proper_case_dataset:
        # Check if it has a lowercase dataset hint — if so, it's legacy
        lowercase_markers = ["cifar100", "cifar10", "femnist", "texas100", "purchase100", "trec"]
        if any(m in scenario_name for m in lowercase_markers):
            return True

    return False


def _extract_defense(path: Path) -> Optional[str]:
    """Try to extract defense name from the directory path."""
    path_str = str(path).lower()
    # Order matters: longer/more specific names first
    for d in ["skymask_small", "skymask", "martfl", "fltrust", "fedavg",
              "trimmed_mean", "multi_krum", "rflpa", "spmc", "daved",
              "deepsight", "foolsgold", "bulyan", "flame"]:
        if d in path_str:
            return d
    # Try from config.yaml if present
    config_path = _find_config_yaml(path)
    if config_path:
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("aggregation", {}).get("method", None)
        except Exception:
            pass
    return None


def _extract_dataset(path: Path) -> Optional[str]:
    path_str = str(path).lower()
    for ds in ["femnist", "cifar100", "cifar10", "texas100", "purchase100", "trec"]:
        if ds in path_str:
            return ds
    return None


def _extract_step(path: Path) -> Optional[str]:
    """Extract step identifier from path like results/step3_defense_tuning/..."""
    for part in path.parts:
        part_lower = part.lower()
        if part_lower.startswith("step"):
            return part
    return None


def _find_config_yaml(run_path: Path) -> Optional[Path]:
    """Walk up from run dir to find config.yaml."""
    current = run_path
    for _ in range(5):
        candidate = current / "config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


# ===========================================================================
# Result scanning
# ===========================================================================

def scan_run(run_path: Path) -> Dict:
    """Scan a single experiment run directory."""
    result = {
        "path": str(run_path),
        "status": "unknown",
        "defense": _extract_defense(run_path),
        "dataset": _extract_dataset(run_path),
        "step": _extract_step(run_path),
        "metrics": {},
        "issues": [],
        "needs_rerun": False,
    }

    # Status
    if (run_path / ".success").exists():
        result["status"] = "success"
    elif (run_path / ".failed").exists():
        result["status"] = "failed"
        try:
            result["fail_reason"] = (run_path / ".failed").read_text().strip()[:200]
        except Exception:
            pass
    elif (run_path / ".in_progress").exists():
        result["status"] = "in_progress"
    else:
        result["status"] = "incomplete"

    # Metrics
    metrics_file = run_path / "final_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            result["metrics"] = m

            # Sanity checks — match field names used in viz_utils.load_run_metrics
            acc = m.get("acc", m.get("test_accuracy", m.get("test_acc", m.get("accuracy", None))))
            asr = m.get("asr", m.get("test_asr", None))
            rounds = m.get("completed_rounds", None)
            # Normalize to percentage (some runs save as 0.0-1.0)
            if isinstance(acc, (int, float)) and 0 <= acc <= 1.0:
                acc = acc * 100
            if isinstance(asr, (int, float)) and 0 <= asr <= 1.0:
                asr = asr * 100

            if acc is not None:
                result["metrics"]["_acc"] = acc
                if acc == 0.0:
                    result["issues"].append("ZERO_ACCURACY")
                if acc < 5.0:
                    result["issues"].append("VERY_LOW_ACCURACY")

            if asr is not None:
                result["metrics"]["_asr"] = asr
                if asr > 95.0:
                    result["issues"].append("VERY_HIGH_ASR")

            if rounds is not None and rounds < 10:
                result["issues"].append(f"EARLY_STOP_round_{rounds}")

        except (json.JSONDecodeError, Exception) as e:
            result["issues"].append(f"CORRUPT_METRICS: {e}")
    elif result["status"] == "success":
        result["issues"].append("SUCCESS_BUT_NO_METRICS")

    # Check training log for NaN
    log_file = run_path / "training_log.csv"
    if log_file.exists():
        try:
            with open(log_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                result["metrics"]["_total_rounds_logged"] = len(rows)
                # Check last few rows for NaN
                for row in rows[-5:]:
                    for k, v in row.items():
                        if v and v.lower() == "nan":
                            result["issues"].append(f"NAN_in_{k}")
                            break

                # Check for identical replicate seeds (bug #6)
                # (can only detect if multiple runs exist in parent)
        except Exception:
            pass

    # Flag if affected by fixed bugs
    defense = result["defense"]
    dataset = result["dataset"]

    if defense in AFFECTED_DEFENSES:
        result["needs_rerun"] = True
        result["issues"].append(f"BUG_AFFECTED_DEFENSE:{defense}")

    if dataset in AFFECTED_DATASETS:
        result["needs_rerun"] = True
        result["issues"].append("BUG_AFFECTED_DATASET:femnist_in_channels")

    return result


def find_all_runs(results_dir: Path) -> List[Path]:
    """Find all experiment run directories (those containing run_* pattern)."""
    runs = []
    for p in results_dir.rglob("run_*_seed_*"):
        if p.is_dir():
            runs.append(p)
    if not runs:
        # Fallback: look for final_metrics.json
        for p in results_dir.rglob("final_metrics.json"):
            runs.append(p.parent)
    if not runs:
        # Fallback: look for .success markers
        for p in results_dir.rglob(".success"):
            runs.append(p.parent)
    return sorted(set(runs))


# ===========================================================================
# Replicate consistency check (detects seed bug)
# ===========================================================================

def check_replicate_consistency(runs: List[Dict]) -> List[str]:
    """Group runs by experiment (same parent) and flag if replicates are suspiciously identical."""
    issues = []
    by_parent = defaultdict(list)
    for r in runs:
        parent = str(Path(r["path"]).parent)
        by_parent[parent].append(r)

    for parent, group in by_parent.items():
        accs = [r["metrics"].get("_acc") for r in group if r["metrics"].get("_acc") is not None]
        if len(accs) >= 2:
            # Check if all replicates have identical accuracy (seed bug symptom)
            if len(set(round(a, 4) for a in accs)) == 1 and len(accs) >= 3:
                issues.append(f"IDENTICAL_REPLICATES ({len(accs)} runs, acc={accs[0]:.2f}): {parent}")
    return issues


# ===========================================================================
# Reporting
# ===========================================================================

def print_summary(runs: List[Dict], verbose: bool = False):
    """Print a comprehensive summary of all results."""

    if not runs:
        print("\n  No experiment runs found!")
        print("  Make sure --results_dir points to the correct location.")
        return

    print(f"\n{'='*70}")
    print(f"  RESULTS HEALTH CHECK — {len(runs)} experiment runs found")
    print(f"{'='*70}")

    # --- 1. Status overview ---
    status_counts = defaultdict(int)
    for r in runs:
        status_counts[r["status"]] += 1

    print(f"\n  STATUS OVERVIEW:")
    for status in ["success", "failed", "in_progress", "incomplete", "unknown"]:
        count = status_counts.get(status, 0)
        if count > 0:
            marker = {"success": "+", "failed": "X", "in_progress": "~", "incomplete": "?", "unknown": "?"}
            print(f"    [{marker.get(status, '?')}] {status:15s}: {count}")

    # --- 2. Per-step summary ---
    by_step = defaultdict(list)
    for r in runs:
        step = r["step"] or "unknown_step"
        by_step[step].append(r)

    print(f"\n  PER-STEP BREAKDOWN:")
    print(f"    {'Step':<30s} {'Total':>6s} {'OK':>6s} {'Fail':>6s} {'Inc':>6s}")
    print(f"    {'-'*60}")
    for step in sorted(by_step.keys()):
        group = by_step[step]
        ok = sum(1 for r in group if r["status"] == "success")
        fail = sum(1 for r in group if r["status"] == "failed")
        inc = sum(1 for r in group if r["status"] == "incomplete")
        print(f"    {step:<30s} {len(group):>6d} {ok:>6d} {fail:>6d} {inc:>6d}")

    # --- 2b. Coverage check: did rerun_priority.sh complete every expected step? ---
    print(f"\n  COVERAGE CHECK (vs rerun_priority.sh expected steps):")
    print(f"    {'CLI Step':<10s} {'Scenario Prefix(es)':<35s} {'Status':<18s} {'Runs':>6s}")
    print(f"    {'-'*72}")
    found_step_dirs = {step: by_step[step] for step in by_step.keys()}
    missing_steps = []
    for cli_step in EXPECTED_STEPS_AFTER_RERUN:
        prefixes = STEP_OUTPUT_DIRS.get(cli_step, [f"step{cli_step}_"])
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        # Find scenario dirs whose name starts with any of the expected prefixes
        matching = [s for s in found_step_dirs
                    if any(s.lower().startswith(p.lower()) for p in prefixes)]
        prefix_display = " | ".join(prefixes)[:34]
        if matching:
            total = sum(len(found_step_dirs[s]) for s in matching)
            ok = sum(1 for s in matching for r in found_step_dirs[s] if r["status"] == "success")
            if ok == total:
                status = f"OK ({ok}/{total})"
            elif ok == 0:
                status = f"ALL FAILED (0/{total})"
            else:
                status = f"PARTIAL ({ok}/{total})"
            print(f"    {cli_step:<10d} {prefix_display:<35s} {status:<18s} {total:>6d}")
        else:
            print(f"    {cli_step:<10d} {prefix_display:<35s} {'MISSING':<18s} {'0':>6s}")
            missing_steps.append(cli_step)

    if missing_steps:
        print(f"\n  WARNING: {len(missing_steps)} expected step(s) have NO results: {missing_steps}")

    # --- 3. Per-defense accuracy summary ---
    by_defense = defaultdict(list)
    for r in runs:
        if r["status"] == "success" and r["metrics"].get("_acc") is not None:
            defense = r["defense"] or "unknown"
            by_defense[defense].append(r)

    if by_defense:
        print(f"\n  PER-DEFENSE ACCURACY (completed runs only):")
        print(f"    {'Defense':<18s} {'Runs':>5s} {'Avg Acc':>8s} {'Min':>8s} {'Max':>8s} {'Avg ASR':>8s}")
        print(f"    {'-'*60}")
        for defense in sorted(by_defense.keys()):
            group = by_defense[defense]
            accs = [r["metrics"]["_acc"] for r in group]
            asrs = [r["metrics"].get("_asr", 0) for r in group if r["metrics"].get("_asr") is not None]
            avg_asr = f"{sum(asrs)/len(asrs):7.2f}%" if asrs else "    N/A"
            print(f"    {defense:<18s} {len(group):>5d} {sum(accs)/len(accs):>7.2f}% {min(accs):>7.2f}% {max(accs):>7.2f}% {avg_asr}")

    # --- 4. Per-dataset summary ---
    by_dataset = defaultdict(list)
    for r in runs:
        if r["status"] == "success" and r["metrics"].get("_acc") is not None:
            ds = r["dataset"] or "unknown"
            by_dataset[ds].append(r)

    if by_dataset:
        print(f"\n  PER-DATASET ACCURACY (completed runs only):")
        print(f"    {'Dataset':<15s} {'Runs':>5s} {'Avg Acc':>8s}")
        print(f"    {'-'*32}")
        for ds in sorted(by_dataset.keys()):
            group = by_dataset[ds]
            accs = [r["metrics"]["_acc"] for r in group]
            print(f"    {ds:<15s} {len(group):>5d} {sum(accs)/len(accs):>7.2f}%")

    # --- 5. Issues & warnings ---
    all_issues = []
    for r in runs:
        for issue in r["issues"]:
            all_issues.append((issue, r["path"]))

    issue_counts = defaultdict(int)
    for issue, _ in all_issues:
        tag = issue.split(":")[0]
        issue_counts[tag] += 1

    if issue_counts:
        print(f"\n  ISSUES DETECTED:")
        for tag, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    [{count:>4d}] {tag}")

    # --- 6. Replicate consistency ---
    replicate_issues = check_replicate_consistency(runs)
    if replicate_issues:
        print(f"\n  SEED BUG WARNING — Identical replicates detected ({len(replicate_issues)} groups):")
        for issue in replicate_issues[:5]:
            print(f"    {issue}")
        if len(replicate_issues) > 5:
            print(f"    ... and {len(replicate_issues) - 5} more")

    # --- 7. Rerun recommendation ---
    rerun_count = sum(1 for r in runs if r["needs_rerun"])
    if rerun_count > 0:
        print(f"\n  RERUN NEEDED: {rerun_count}/{len(runs)} runs affected by fixed bugs")
        print(f"  Affected defenses: {', '.join(sorted(AFFECTED_DEFENSES))}")
        print(f"  Affected datasets: {', '.join(sorted(AFFECTED_DATASETS))}")
        print(f"\n  To clear affected .success markers:")
        for defense in sorted(AFFECTED_DEFENSES):
            print(f"    find <results_dir> -path '*{defense}*' -name '.success' -delete")
        print(f"    find <results_dir> -path '*femnist*' -name '.success' -delete")

    # --- 8. Verbose: per-run details ---
    if verbose:
        print(f"\n  DETAILED RUN LIST:")
        for r in runs:
            acc = r["metrics"].get("_acc", "N/A")
            asr = r["metrics"].get("_asr", "N/A")
            acc_str = f"{acc:.2f}%" if isinstance(acc, (int, float)) else acc
            asr_str = f"{asr:.2f}%" if isinstance(asr, (int, float)) else asr
            issues_str = ", ".join(r["issues"]) if r["issues"] else "OK"
            status_icon = {"success": "+", "failed": "X", "in_progress": "~"}.get(r["status"], "?")
            print(f"    [{status_icon}] acc={acc_str:>8s}  asr={asr_str:>8s}  {issues_str}")
            print(f"        {r['path']}")

    # --- Final verdict ---
    ok_count = status_counts.get("success", 0)
    total = len(runs)
    failed = status_counts.get("failed", 0)
    incomplete = status_counts.get("incomplete", 0)

    print(f"\n{'='*70}")
    if failed == 0 and incomplete == 0 and not missing_steps and rerun_count == 0:
        print(f"  ALL CLEAR: {ok_count}/{total} runs completed successfully")
        print(f"  All expected steps from rerun_priority.sh have results")
    else:
        parts = []
        if missing_steps:
            parts.append(f"{len(missing_steps)} missing step(s): {missing_steps}")
        if failed > 0:
            parts.append(f"{failed} failed")
        if incomplete > 0:
            parts.append(f"{incomplete} incomplete")
        if rerun_count > 0:
            parts.append(f"{rerun_count} need rerun (bug-affected)")
        print(f"  ACTION NEEDED: {'; '.join(parts)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Check experiment results health")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the results directory")
    parser.add_argument("--step", type=str, default=None,
                        help="Filter to a specific step (e.g., 'step3' or '3')")
    parser.add_argument("--defense", type=str, default=None,
                        help="Filter to a specific defense (e.g., 'martfl')")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to a specific dataset (e.g., 'cifar100')")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-run details")
    parser.add_argument("--rerun_only", action="store_true",
                        help="Only show runs that need rerun")
    parser.add_argument("--include_legacy", action="store_true",
                        help="Include legacy/old results (default: filter out)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        sys.exit(1)

    print(f"Scanning: {results_dir}")
    run_paths = find_all_runs(results_dir)
    print(f"Found {len(run_paths)} experiment run directories")

    # Scan all runs
    runs = [scan_run(p) for p in run_paths]

    # Apply legacy filter (default: hide old results)
    if not args.include_legacy:
        before = len(runs)
        runs = [r for r in runs if not is_legacy_scenario(r.get("step", ""), r.get("path", ""))]
        skipped = before - len(runs)
        if skipped > 0:
            print(f"Filtered out {skipped} legacy/old runs (use --include_legacy to show)")

    # Apply filters
    if args.step:
        step_filter = args.step if args.step.startswith("step") else f"step{args.step}"
        runs = [r for r in runs if r["step"] and step_filter in r["step"].lower()]
        print(f"Filtered to step '{step_filter}': {len(runs)} runs")

    if args.defense:
        runs = [r for r in runs if r["defense"] and args.defense.lower() in r["defense"].lower()]
        print(f"Filtered to defense '{args.defense}': {len(runs)} runs")

    if args.dataset:
        runs = [r for r in runs if r["dataset"] and args.dataset.lower() in r["dataset"].lower()]
        print(f"Filtered to dataset '{args.dataset}': {len(runs)} runs")

    if args.rerun_only:
        runs = [r for r in runs if r["needs_rerun"]]
        print(f"Filtered to rerun-needed: {len(runs)} runs")

    print_summary(runs, verbose=args.verbose)


if __name__ == "__main__":
    main()
