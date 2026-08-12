"""
extract_step8_system_scalability.py — Extract ALL system + behavioral metrics
from Step 8 (scalability sweep) into a single CSV for analysis.

Produces: analysis_partial/step8_system_scalability.csv

Columns (per seed run):
  defense, dataset, n_sellers, seed,
  acc, asr, bsr, msr,
  agg_latency_sec, round_time_sec, wall_clock_sec,
  peak_vram_gb, peak_vram_reserved_gb,
  benign_rejection_pct, wasted_upload_mb_per_round,
  total_upload_mb, total_upload_mb_accepted,
  completed_rounds, gpu_name

Usage:
    python experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py
    python experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py --results_dir ./results
    python experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py --output analysis/scalability.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_KNOWN_DATASETS = ("CIFAR100", "CIFAR10", "FEMNIST", "TREC6", "Texas100", "Purchase100")
SCENARIO_RE = re.compile(
    r"step10_scalability_(?P<defense>.+)_(?P<dataset>" + "|".join(_KNOWN_DATASETS) + r")$"
)
N_SELLERS_RE = re.compile(r"n_sellers_(\d+)")
SEED_RE = re.compile(r"run_\d+_seed_(\d+)")


def extract_system_scalability(results_dir: Path) -> pd.DataFrame:
    """Walk step 8 results and extract every metric into a flat DataFrame."""
    rows = []

    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue

        defense = m.group("defense")
        dataset = m.group("dataset")

        for metrics_file in scenario_dir.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                continue

            # Extract n_sellers from path
            n_sellers = None
            seed = "?"
            for part in metrics_file.relative_to(scenario_dir).parts:
                sm = N_SELLERS_RE.match(part)
                if sm:
                    n_sellers = int(sm.group(1))
                sd = SEED_RE.match(part)
                if sd:
                    seed = sd.group(1)

            if n_sellers is None:
                continue

            try:
                with open(metrics_file) as f:
                    met = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            row = {
                "defense": defense,
                "dataset": dataset,
                "n_sellers": n_sellers,
                "seed": seed,
                # --- Behavioral metrics ---
                "acc": met.get("acc"),
                "asr": met.get("asr"),
                # --- Temporal metrics ---
                "agg_latency_sec": met.get("avg_aggregation_latency_sec"),
                "max_agg_latency_sec": met.get("max_aggregation_latency_sec"),
                "median_agg_latency_sec": met.get("median_aggregation_latency_sec"),
                "round_time_sec": met.get("avg_seconds_per_round"),
                "wall_clock_sec": met.get("wall_clock_seconds"),
                "throughput_rps": met.get("throughput_rounds_per_sec"),
                # --- Hardware metrics ---
                "peak_vram_mb": met.get("max_aggregation_peak_vram_mb",
                                        met.get("peak_gpu_memory_allocated_mb")),
                "peak_vram_reserved_mb": met.get("peak_gpu_memory_reserved_mb"),
                "peak_host_rss_mb": met.get("peak_host_rss_mb"),
                "gpu_name": met.get("gpu_device_name"),
                # --- Network / CoC metrics ---
                "total_upload_mb": met.get("total_upload_mb"),
                "total_upload_mb_accepted": met.get("total_upload_mb_accepted"),
                "total_wasted_upload_mb": met.get("total_wasted_upload_mb"),
                "completed_rounds": met.get("completed_rounds"),
            }

            # --- Selection metrics from marketplace_report.json ---
            report_file = run_dir / "marketplace_report.json"
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    sellers = report.get("seller_summaries", {})
                    ben = [s for s in sellers.values()
                           if s.get("type") == "benign" and s.get("selection_rate") is not None]
                    adv = [s for s in sellers.values()
                           if s.get("type") == "adversary" and s.get("selection_rate") is not None]
                    if ben:
                        bsr = float(np.mean([s["selection_rate"] for s in ben]))
                        row["bsr"] = bsr
                        row["benign_rejection_pct"] = (1.0 - bsr) * 100
                    if adv:
                        row["msr"] = float(np.mean([s["selection_rate"] for s in adv]))
                    row["n_benign"] = len(ben)
                    row["n_adversary"] = len(adv)
                except (json.JSONDecodeError, OSError):
                    pass

            # Derived: wasted upload per round
            if row.get("total_wasted_upload_mb") and row.get("completed_rounds"):
                row["wasted_mb_per_round"] = row["total_wasted_upload_mb"] / row["completed_rounds"]

            # Derived: peak VRAM in GB
            if row.get("peak_vram_mb"):
                row["peak_vram_gb"] = row["peak_vram_mb"] / 1024.0

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["defense", "n_sellers", "seed"]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame):
    """Print a grouped summary table."""
    if df.empty:
        print("No data.")
        return

    print(f"\n{'='*100}")
    print(f" Step 8 System Scalability — {len(df)} runs")
    print(f"{'='*100}")

    # Group by defense × n_sellers, show means
    grouped = df.groupby(["defense", "n_sellers"])

    cols = [
        ("acc", ".3f", "ACC"),
        ("asr", ".3f", "ASR"),
        ("bsr", ".3f", "BSR"),
        ("msr", ".3f", "MSR"),
        ("agg_latency_sec", ".2f", "AggLat(s)"),
        ("round_time_sec", ".1f", "Round(s)"),
        ("peak_vram_gb", ".2f", "VRAM(GB)"),
        ("benign_rejection_pct", ".1f", "BenRej%"),
        ("wasted_mb_per_round", ".1f", "Waste(MB)"),
        ("total_upload_mb", ".0f", "CoC(MB)"),
    ]

    hdr = f"{'Defense':<12} {'N':>5} {'seeds':>5}"
    for _, _, label in cols:
        hdr += f" {label:>10}"
    print(hdr)
    print("-" * len(hdr))

    for (defense, n_sellers), group in grouped:
        row_str = f"{defense:<12} {n_sellers:>5} {len(group):>5}"
        for col_name, fmt_spec, _ in cols:
            if col_name in group.columns:
                vals = group[col_name].dropna()
                if len(vals) > 0:
                    val = vals.mean()
                    formatted = f"{val:{fmt_spec}}"
                    row_str += f" {formatted:>10}"
                else:
                    row_str += f" {'—':>10}"
            else:
                row_str += f" {'—':>10}"
        print(row_str)

    # GPU info
    if "gpu_name" in df.columns:
        gpus = df["gpu_name"].dropna().unique()
        if len(gpus) > 0:
            print(f"\nHardware: {', '.join(gpus)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Step 8 system scalability metrics to CSV.")
    parser.add_argument("--results_dir", default="./results",
                        help="Path to results directory")
    parser.add_argument("--output", default="analysis_partial/step8_system_scalability.csv",
                        help="Output CSV path")
    # Whitelist filters: drop cells whose (defense, n_sellers) aren't in the
    # current step 8 spec. Stops legacy / out-of-scope rows (N=30, N=500,
    # un-configured defenses) from contaminating the scalability figure.
    parser.add_argument("--defenses", default="fedavg,fltrust,martfl,multi_krum,skymask",
                        help="Comma-separated whitelist of defenses to include "
                             "(default matches the broadened defense set in "
                             "run_step8_system_scalability.sh). Use 'all' to disable filter.")
    parser.add_argument("--n-sellers", default="10,25,50,75,100",
                        help="Comma-separated whitelist of n_sellers (default matches "
                             "run_step8_system_scalability.sh's runtime patch). "
                             "Use 'all' to disable filter.")
    parser.add_argument("--max-seeds-per-cell", type=int, default=None,
                        help="Cap seeds per (defense, n_sellers) cell — keeps the lowest seed numbers. "
                             "Default: keep all. Set to 2 to match NUM_SEEDS_PER_CONFIG.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting from: {results_dir}")
    df = extract_system_scalability(results_dir)

    if df.empty:
        print("No step 8 scalability results found.")
        sys.exit(1)

    # Apply whitelist filters
    n_before = len(df)
    if args.defenses != "all":
        keep_def = {d.strip() for d in args.defenses.split(",")}
        df = df[df["defense"].isin(keep_def)]
    if args.n_sellers != "all":
        keep_n = {int(n) for n in args.n_sellers.split(",")}
        df = df[df["n_sellers"].isin(keep_n)]
    if args.max_seeds_per_cell is not None:
        # Sort by seed within each (defense, n_sellers), keep first K
        df = (df.sort_values(["defense", "n_sellers", "seed"])
                .groupby(["defense", "n_sellers"], group_keys=False)
                .head(args.max_seeds_per_cell)
                .reset_index(drop=True))
    n_after = len(df)
    if n_after < n_before:
        print(f"Filtered: kept {n_after}/{n_before} rows "
              f"(defenses={args.defenses}, n_sellers={args.n_sellers}, "
              f"max_seeds={args.max_seeds_per_cell})")

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to: {output_path}")

    # Print summary
    print_summary(df)

    # Also print column list for LLM context
    print(f"\nCSV columns ({len(df.columns)}):")
    for c in df.columns:
        non_null = df[c].notna().sum()
        print(f"  {c:<30} ({non_null}/{len(df)} non-null)")


if __name__ == "__main__":
    main()
