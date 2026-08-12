"""
Extract partial results from the current benchmark runs.

Walks every completed step (based on .success markers), loads per-run metrics
via viz_utils.load_step_results, writes one CSV per step, and prints a compact
per-step insight summary (success count, per-defense means, per-dataset means,
best/worst defense, sanity flags).

Safe to run mid-rerun: load_step_results only picks up runs with .success
markers, so in-progress / failed / incomplete cells are ignored automatically.

Usage:
    python experiments/gradient_market/visualization/extract_partial_results.py
    python experiments/gradient_market/visualization/extract_partial_results.py \
        --results_dir ./results --output_dir ./analysis_partial
    python experiments/gradient_market/visualization/extract_partial_results.py --steps 4 5 7
    python experiments/gradient_market/visualization/extract_partial_results.py --json

The --json flag also dumps a combined machine-readable summary to
<output_dir>/summary.json so downstream scripts can consume the insights.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Make the project root importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.gradient_market.visualization.viz_utils import load_step_results  # noqa: E402


# ---------------------------------------------------------------------------
# Which steps to extract — aligned with rerun_priority.sh
# ---------------------------------------------------------------------------
# key = CLI step number (for --steps filter)
# value = (human label, scenario-name prefix(es) that load_step_results looks for)
#
# Intentionally omitted:
#   - step 1 (iid_tuning) — cached, not part of current rerun
#   - step 11 (free_riding) — deprioritized
#   - step 18 (daved_comparison) — DAVED removed
#   - step 20 (sleeper_agent) — deprioritized
STEP_PREFIXES: Dict[int, Dict] = {
    3:  {"label": "defense_tune",   "prefix": "step3_tune"},
    4:  {"label": "attack_sens",    "prefix": "step5_atk"},
    5:  {"label": "sybil",          "prefix": "step6_adv_sybil"},
    6:  {"label": "adaptive",       "prefix": "step7_adaptive"},
    7:  {"label": "buyer",          "prefix": "step8_buyer_attack"},
    8:  {"label": "scalability",    "prefix": "step10_scalability"},
    9:  {"label": "heterogeneity",  "prefix": "step11"},
    10: {"label": "main_summary",   "prefix": "step12_main_summary"},
    13: {"label": "drowning",       "prefix": "step13_drowning"},
    14: {"label": "collusion",      "prefix": "step14_collusion"},
    15: {"label": "pricing",        "prefix": "step15_pricing"},
    16: {"label": "dp_fairness",    "prefix": "step16_dp"},
    17: {"label": "alie",           "prefix": "step17_alie"},
    19: {"label": "new_defenses",   "prefix": "step19_new_defenses"},
}

# Excluded defense names (DAVED has been removed from the benchmark)
EXCLUDED_DEFENSES = {"daved", "DAVED", "Daved"}

# Datasets hidden from per-step output. Data stays on disk — this is a
# display-only filter. Override with --exclude-datasets / --no-exclude.
EXCLUDED_DATASETS_DEFAULT = {
    # Proper-case variants
    "FEMNIST", "Purchase100",
    # viz_utils.fmt() may normalize these; include common display forms
    "Femnist", "femnist", "FeMNIST",
    "purchase100", "Purchase-100",
}

# Sanity thresholds (percent, since load_step_results normalizes to 0-100)
ACC_COLLAPSE_THRESHOLD = 5.0      # accuracy ≤ 5% on a non-trivial task
ASR_VERY_HIGH_THRESHOLD = 80.0    # backdoor attack almost certainly succeeded


# ---------------------------------------------------------------------------
# Per-step insight extraction
# ---------------------------------------------------------------------------
def summarize_step(step_id: int, label: str, df: pd.DataFrame,
                   excluded_datasets: set) -> Dict:
    """Return a dict of per-step insights for printing and JSON export."""
    if df.empty:
        return {"step": step_id, "label": label, "n_runs": 0, "empty": True}

    # Drop excluded defense rows so they never show up
    if "defense" in df.columns:
        df = df[~df["defense"].isin(EXCLUDED_DEFENSES)].copy()
    # Drop excluded dataset rows (display-only filter)
    if "dataset" in df.columns and excluded_datasets:
        df = df[~df["dataset"].isin(excluded_datasets)].copy()

    n_runs = len(df)
    info = {
        "step": step_id,
        "label": label,
        "n_runs": n_runs,
        "defenses": {},
        "datasets": {},
        "flags": {},
    }

    # Per-defense rollup
    if "defense" in df.columns:
        for defense, group in df.groupby("defense"):
            row = {"n": len(group)}
            for metric in ("acc", "asr", "bsr", "msr"):
                if metric in group.columns and group[metric].notna().any():
                    row[f"{metric}_mean"] = float(group[metric].mean())
                    row[f"{metric}_std"] = float(group[metric].std()) if len(group) > 1 else 0.0
            info["defenses"][str(defense)] = row

    # Per-dataset rollup
    if "dataset" in df.columns:
        for dataset, group in df.groupby("dataset"):
            if pd.isna(dataset):
                continue
            row = {"n": len(group)}
            if "acc" in group.columns:
                row["acc_mean"] = float(group["acc"].mean())
            if "asr" in group.columns:
                row["asr_mean"] = float(group["asr"].mean())
            info["datasets"][str(dataset)] = row

    # Best / worst defense by accuracy
    if "acc" in df.columns and "defense" in df.columns and df["acc"].notna().any():
        by_def = df.groupby("defense")["acc"].mean().sort_values(ascending=False)
        if len(by_def) > 0:
            info["best_defense"] = {"name": str(by_def.index[0]),
                                     "acc_mean": float(by_def.iloc[0])}
            info["worst_defense"] = {"name": str(by_def.index[-1]),
                                      "acc_mean": float(by_def.iloc[-1])}

    # Sanity flags
    if "acc" in df.columns:
        collapse = df[df["acc"] <= ACC_COLLAPSE_THRESHOLD]
        info["flags"]["n_collapsed"] = int(len(collapse))
    if "asr" in df.columns:
        high_asr = df[df["asr"] >= ASR_VERY_HIGH_THRESHOLD]
        info["flags"]["n_very_high_asr"] = int(len(high_asr))

    return info


def format_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:5.1f}%"


def print_step_card(info: Dict, max_defenses: int = 10) -> None:
    """Human-readable per-step insight card."""
    step = info["step"]
    label = info["label"]
    n = info["n_runs"]

    border = "─" * 72
    print(border)
    print(f"Step {step:<3} {label:<18}  {n} successful run(s)")
    print(border)

    if info.get("empty") or n == 0:
        print("  (no successful runs yet)")
        return

    # Per-defense table
    if info["defenses"]:
        print(f"  {'Defense':<18}{'n':>4}  {'acc':>8}{'asr':>9}{'bsr':>9}{'msr':>9}")
        items = sorted(
            info["defenses"].items(),
            key=lambda kv: kv[1].get("acc_mean", -1),
            reverse=True,
        )[:max_defenses]
        for defense, row in items:
            acc = row.get("acc_mean")
            asr = row.get("asr_mean")
            bsr = row.get("bsr_mean")
            msr = row.get("msr_mean")
            print(f"  {defense:<18}{row['n']:>4}  "
                  f"{format_pct(acc):>8}{format_pct(asr):>9}"
                  f"{format_pct(bsr):>9}{format_pct(msr):>9}")

    # Per-dataset one-liner
    if info["datasets"]:
        ds_parts = []
        for ds, row in sorted(info["datasets"].items()):
            acc = row.get("acc_mean")
            ds_parts.append(f"{ds}:acc={format_pct(acc).strip()}(n={row['n']})")
        print(f"  datasets: {'  '.join(ds_parts)}")

    # Best / worst callouts
    if "best_defense" in info and "worst_defense" in info:
        b, w = info["best_defense"], info["worst_defense"]
        print(f"  best acc : {b['name']:<18} {b['acc_mean']:5.1f}%")
        print(f"  worst acc: {w['name']:<18} {w['acc_mean']:5.1f}%")

    # Sanity flags
    flags = info["flags"]
    if flags.get("n_collapsed", 0) > 0:
        print(f"  ⚠ {flags['n_collapsed']} run(s) with acc ≤ {ACC_COLLAPSE_THRESHOLD}% "
              f"(model collapse or degenerate config)")
    if flags.get("n_very_high_asr", 0) > 0:
        print(f"  ⚠ {flags['n_very_high_asr']} run(s) with ASR ≥ {ASR_VERY_HIGH_THRESHOLD}% "
              f"(attack effectively succeeded)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="./results",
                    help="Root results directory")
    ap.add_argument("--output_dir", default="./analysis_partial",
                    help="Directory to write per-step CSVs and summary.json")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="CLI step numbers to extract (default: all completed ones)")
    ap.add_argument("--json", action="store_true",
                    help="Also write a combined summary.json of all per-step insights")
    ap.add_argument("--max_defenses", type=int, default=12,
                    help="Max defenses to show in per-step table (default: 12)")
    ap.add_argument("--exclude-datasets", nargs="*", default=None,
                    metavar="DATASET",
                    help=f"Datasets to hide from the output (display-only; "
                         f"data on disk is untouched). Default: "
                         f"{sorted(EXCLUDED_DATASETS_DEFAULT)}")
    ap.add_argument("--no-exclude", action="store_true",
                    help="Clear the dataset exclusion list — include every "
                         "dataset found in the results")
    args = ap.parse_args()

    if args.no_exclude:
        excluded_datasets = set()
    elif args.exclude_datasets is not None:
        excluded_datasets = set(args.exclude_datasets)
    else:
        excluded_datasets = set(EXCLUDED_DATASETS_DEFAULT)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"❌ Results dir not found: {results_dir.resolve()}")
        sys.exit(1)

    steps_to_run = args.steps or sorted(STEP_PREFIXES.keys())
    unknown = [s for s in steps_to_run if s not in STEP_PREFIXES]
    if unknown:
        print(f"❌ Unknown steps: {unknown}")
        print(f"   Known steps: {sorted(STEP_PREFIXES.keys())}")
        sys.exit(1)

    print(f"📂 Scanning {results_dir.resolve()}")
    print(f"📁 Writing CSVs to {output_dir.resolve()}")
    print(f"🚫 Excluded datasets: {sorted(excluded_datasets) if excluded_datasets else '(none)'}")
    print(f"🔢 Steps: {steps_to_run}")
    print()

    all_summaries: List[Dict] = []
    grand_total = 0

    for step in steps_to_run:
        spec = STEP_PREFIXES[step]
        label = spec["label"]
        prefix = spec["prefix"]

        df = load_step_results(str(results_dir), prefix, require_success=True)

        # Drop DAVED rows before writing CSV so it's hidden end-to-end
        if not df.empty and "defense" in df.columns:
            df = df[~df["defense"].isin(EXCLUDED_DEFENSES)].copy()
        # Drop excluded dataset rows from the CSV too, so downstream tools
        # see the same scope as the printed summary.
        if not df.empty and "dataset" in df.columns and excluded_datasets:
            df = df[~df["dataset"].isin(excluded_datasets)].copy()

        # Write CSV even if empty — easier to spot "step has no results" as a
        # zero-byte file than as "nothing to see here"
        csv_path = output_dir / f"step{step}_{label}.csv"
        df.to_csv(csv_path, index=False)

        info = summarize_step(step, label, df, excluded_datasets)
        info["csv"] = str(csv_path)
        all_summaries.append(info)
        grand_total += info["n_runs"]

        print_step_card(info, max_defenses=args.max_defenses)
        print()

    # Grand totals
    print("═" * 72)
    print(f" GRAND TOTAL: {grand_total} successful runs across {len(steps_to_run)} step(s)")
    print("═" * 72)

    if args.json:
        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump({
                "results_dir": str(results_dir.resolve()),
                "steps": all_summaries,
                "grand_total": grand_total,
            }, f, indent=2)
        print(f"\n📝 Machine-readable summary: {summary_path}")


if __name__ == "__main__":
    main()
