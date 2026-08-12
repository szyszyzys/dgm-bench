"""
plot_system_scalability.py — System Scalability Figure for the paper.

Generates a 2x3 figure with six subplots covering all system metrics:
  (a) Auditing Latency      — aggregation-only seconds per round vs N
  (b) End-to-End Round Time  — full round duration vs N
  (c) Peak VRAM              — aggregation-phase GPU memory vs N
  (d) Benign Rejection Rate  — 1 - BSR vs N (participation tax)
  (e) Network Waste          — wasted upload MB per round vs N
  (f) Cost of Convergence    — total upload MB to reach target acc vs N

Also generates a compact 1x2 figure (latency + memory only) for the main
paper body, saving the full 2x3 as a supplementary figure.

Reads from step 8 results: results/step10_scalability_*/n_sellers_*/

Usage:
    python experiments/gradient_market/visualization/plot_system_scalability.py
    python experiments/gradient_market/visualization/plot_system_scalability.py --results_dir ./results
    python experiments/gradient_market/visualization/plot_system_scalability.py --compact   # 1x2 only
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Try to import the centralized viz_utils for consistent styling
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from viz_utils import (
        setup_style, save_figure, fmt,
        DEFENSE_COLORS, DEFENSE_NAMES, DEFENSE_ORDER,
    )
    HAS_VIZ_UTILS = True
except ImportError:
    HAS_VIZ_UTILS = False


# =============================================================================
# Configuration
# =============================================================================

# All defenses that may appear, in display order. Auto-filtered to data.
SCALABILITY_DEFENSES_ALL = [
    ("fedavg", "FedAvg"),
    ("fltrust", "FLTrust"),
    ("foolsgold", "FoolsGold"),
    ("martfl", "MartFL"),
    ("deepsight", "DeepSight"),
    ("flame", "FLAME"),
    ("trimmed_mean", "Trim-Mean"),
    ("multi_krum", "Multi-Krum"),
    ("bulyan", "Bulyan"),
    ("rflpa", "RFLPA"),
    ("spmc", "SPMC"),
    ("skymask", "SkyMask"),
]

_FALLBACK_COLORS = {
    "FedAvg": "#7f8c8d", "FLTrust": "#3498db", "MartFL": "#2ecc71",
    "SkyMask": "#e74c3c", "Trim-Mean": "#9b59b6", "Multi-Krum": "#1abc9c",
    "RFLPA": "#f39c12", "SPMC": "#2c3e50", "FLAME": "#fd79a8",
    "DeepSight": "#6c5ce7", "Bulyan": "#00b894", "FoolsGold": "#fdcb6e",
}

_FALLBACK_MARKERS = {
    "FedAvg": "s", "FLTrust": "o", "MartFL": "^", "SkyMask": "D",
    "Trim-Mean": "v", "Multi-Krum": "<", "RFLPA": ">", "SPMC": "p",
    "FLAME": "h", "DeepSight": "H", "Bulyan": "*", "FoolsGold": "P",
}


# =============================================================================
# Data collection
# =============================================================================

def collect_scalability_system_metrics(results_dir: Path):
    """
    Collect all system metrics from step 8 scalability results.

    Returns dict: (defense, n_sellers) -> {metric_name: [values_across_seeds]}
    """
    pattern = re.compile(r"step10_scalability_(\w+?)_(\w+)")
    n_sellers_re = re.compile(r"n_sellers_(\d+)")

    data = defaultdict(lambda: defaultdict(list))

    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = pattern.match(scenario_dir.name)
        if not m:
            continue
        defense = m.group(1)

        for metrics_file in scenario_dir.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                continue

            # Extract n_sellers from path
            n_sellers = None
            for part in metrics_file.relative_to(scenario_dir).parts:
                sm = n_sellers_re.match(part)
                if sm:
                    n_sellers = int(sm.group(1))
                    break
            if n_sellers is None:
                continue

            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            key = (defense, n_sellers)

            # --- Metric 1: Auditing Latency (aggregation only) ---
            agg_lat = metrics.get("avg_aggregation_latency_sec")
            if agg_lat is not None:
                data[key]["agg_latency_sec"].append(agg_lat)

            # --- Metric 2: End-to-End Round Time ---
            spr = metrics.get("avg_seconds_per_round")
            if spr is not None:
                data[key]["round_time_sec"].append(spr)

            # --- Metric 3: Peak VRAM (aggregation phase) ---
            agg_vram = metrics.get("max_aggregation_peak_vram_mb")
            if agg_vram is not None:
                data[key]["peak_vram_gb"].append(agg_vram / 1024.0)
            else:
                peak_mb = metrics.get("peak_gpu_memory_allocated_mb")
                if peak_mb is not None:
                    data[key]["peak_vram_gb"].append(peak_mb / 1024.0)

            # --- Metric 4: Benign Rejection Rate (1 - BSR) ---
            report_file = run_dir / "marketplace_report.json"
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    sellers = report.get("seller_summaries", {})
                    ben_rates = [s["selection_rate"] for s in sellers.values()
                                 if s.get("type") == "benign" and s.get("selection_rate") is not None]
                    if ben_rates:
                        bsr = float(np.mean(ben_rates))
                        data[key]["benign_rejection_pct"].append((1.0 - bsr) * 100)
                except (json.JSONDecodeError, OSError, KeyError):
                    pass

            # --- Metric 5: Network Waste (wasted upload per round) ---
            wasted = metrics.get("total_wasted_upload_mb")
            rounds = metrics.get("completed_rounds")
            if wasted is not None and rounds and rounds > 0:
                data[key]["wasted_mb_per_round"].append(wasted / rounds)

            # --- Metric 6: Cost of Convergence (total upload MB) ---
            total_upload = metrics.get("total_upload_mb")
            if total_upload is not None:
                data[key]["total_upload_mb"].append(total_upload)

    return data


# =============================================================================
# Plotting helpers
# =============================================================================

def _get_defense_style(def_label):
    color = DEFENSE_COLORS.get(def_label, _FALLBACK_COLORS.get(def_label, "#333"))
    marker = _FALLBACK_MARKERS.get(def_label, "o")
    return color, marker


def _plot_metric_line(ax, all_n, data, def_key, def_label, metric_key):
    """Plot one defense's line for a given metric on an axis."""
    n_vals, means, stds = [], [], []
    for n in all_n:
        vals = data.get((def_key, n), {}).get(metric_key, [])
        if vals:
            n_vals.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    if not n_vals:
        return False

    color, marker = _get_defense_style(def_label)
    n_arr = np.array(n_vals)
    m_arr = np.array(means)
    s_arr = np.array(stds)

    ax.plot(n_arr, m_arr, marker=marker, color=color, label=def_label,
            linewidth=2, markersize=7)
    ax.fill_between(n_arr, m_arr - s_arr, m_arr + s_arr, alpha=0.12, color=color)
    return True


def _style_ax(ax, all_n, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_n)


def _save(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_VIZ_UTILS:
        save_figure(fig, str(output_path))
    else:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Full 2x3 figure (all 6 system metrics)
# =============================================================================

PANEL_SPECS = [
    # (metric_key,           ylabel,                     title)
    ("agg_latency_sec",      "Seconds",                  "(a) Auditing Latency"),
    ("round_time_sec",       "Seconds",                  "(b) End-to-End Round Time"),
    ("peak_vram_gb",         "Peak VRAM (GB)",           "(c) Memory Overhead"),
    ("benign_rejection_pct", "Rejected Honest (%)",      "(d) Benign Rejection Rate"),
    ("wasted_mb_per_round",  "Wasted Upload (MB/round)", "(e) Network Waste"),
    ("total_upload_mb",      "Total Upload (MB)",        "(f) Cost of Convergence"),
]


def plot_full_scalability(data, active_defenses, all_n, output_path):
    """Generate 2x3 comprehensive system scalability figure."""
    if HAS_VIZ_UTILS:
        setup_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (metric_key, ylabel, title) in enumerate(PANEL_SPECS):
        ax = axes_flat[idx]
        has_data = False
        for def_key, def_label in active_defenses:
            if _plot_metric_line(ax, all_n, data, def_key, def_label, metric_key):
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#999")

        _style_ax(ax, all_n, "Number of Sellers ($N$)", ylabel, title)
        if metric_key in ("agg_latency_sec", "round_time_sec"):
            ax.set_yscale("log")

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.05), ncol=min(len(handles), 6),
                   frameon=False, fontsize=11)

    _save(fig, output_path)


# =============================================================================
# Compact 1x2 figure (latency + memory only, for main paper body)
# =============================================================================

def plot_compact_scalability(data, active_defenses, all_n, output_path):
    """Generate 1x2 figure with just auditing latency and memory.

    Sized for single-column \\columnwidth in a two-column venue (SIGMOD/VLDB).
    """
    if HAS_VIZ_UTILS:
        setup_style()

    # Single-column figure: ~3.5in wide × 2in tall + legend
    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                          constrained_layout=True)

    # Use agg_latency if available, fall back to round_time
    lat_key = "agg_latency_sec"
    has_agg = any(data.get((d, n), {}).get(lat_key) for d, _ in active_defenses for n in all_n)
    if not has_agg:
        lat_key = "round_time_sec"

    for def_key, def_label in active_defenses:
        _plot_metric_line(ax_lat, all_n, data, def_key, def_label, lat_key)
        _plot_metric_line(ax_mem, all_n, data, def_key, def_label, "peak_vram_gb")

    lat_ylabel = "Aggregation Time (s)" if lat_key == "agg_latency_sec" else "Seconds per Round"
    _style_ax(ax_lat, all_n, "Number of Sellers ($N$)", lat_ylabel, "(a) Auditing Latency")
    ax_lat.set_yscale("log")
    _style_ax(ax_mem, all_n, "Number of Sellers ($N$)", "Peak VRAM (GB)", "(b) Memory Overhead")

    # Compact legend below the title area
    handles, labels = ax_lat.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.12),
                   ncol=min(len(handles), 5),
                   frameon=False, fontsize=8,
                   handlelength=1.5, columnspacing=1.0)

    _save(fig, output_path)


# =============================================================================
# Main
# =============================================================================

def load_data_from_csv(csv_path: Path):
    """Load pre-extracted CSV and convert to the same dict format as collect_scalability_system_metrics."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    data = defaultdict(lambda: defaultdict(list))

    # Map CSV column names to internal metric keys
    col_map = {
        "agg_latency_sec": "agg_latency_sec",
        "round_time_sec": "round_time_sec",
        "peak_vram_gb": "peak_vram_gb",
        "benign_rejection_pct": "benign_rejection_pct",
        "wasted_mb_per_round": "wasted_mb_per_round",
        "total_upload_mb": "total_upload_mb",
    }

    for _, row in df.iterrows():
        defense = row.get("defense")
        n_sellers = row.get("n_sellers")
        if pd.isna(defense) or pd.isna(n_sellers):
            continue
        key = (str(defense), int(n_sellers))
        for csv_col, metric_key in col_map.items():
            val = row.get(csv_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                data[key][metric_key].append(float(val))

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate system scalability figures (latency, memory, BSR, waste, CoC vs N).")
    parser.add_argument("--results_dir", default="./results",
                        help="Path to results directory")
    parser.add_argument("--csv", default=None,
                        help="Read from pre-extracted CSV instead of scanning results "
                             "(e.g. analysis_partial/step8_system_scalability.csv)")
    parser.add_argument("--output_dir", default="figures",
                        help="Output directory for figures")
    parser.add_argument("--compact", action="store_true",
                        help="Only generate compact 1x2 (latency + memory)")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading from CSV: {csv_path}")
        data = load_data_from_csv(csv_path)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Collecting system metrics from: {results_dir}")
        data = collect_scalability_system_metrics(results_dir)

    if not data:
        print("ERROR: No scalability results found.", file=sys.stderr)
        print("  Expected: results/step10_scalability_<defense>_<dataset>/n_sellers_<N>/")
        sys.exit(1)

    # Discover defenses and N values
    defenses_in_data = {d for (d, _) in data.keys()}
    active_defenses = [(k, v) for k, v in SCALABILITY_DEFENSES_ALL if k in defenses_in_data]
    all_n = sorted({n for (_, n) in data.keys()})

    print(f"Defenses: {[d[1] for d in active_defenses]}")
    print(f"N_sellers: {all_n}")

    # Print data summary table
    print(f"\n{'Defense':<12} {'N':>5} {'Agg Lat':>10} {'Round':>10} {'VRAM':>10} "
          f"{'Rej%':>8} {'Waste':>10} {'CoC MB':>10} {'seeds':>6}")
    print("-" * 90)
    for def_key, def_label in active_defenses:
        for n in all_n:
            cell = data.get((def_key, n), {})
            if not cell:
                continue

            def _fmt(vals, unit=""):
                if not vals:
                    return "—"
                return f"{np.mean(vals):.2f}{unit}"

            seeds = max(len(v) for v in cell.values()) if cell else 0
            print(f"{def_label:<12} {n:>5} "
                  f"{_fmt(cell.get('agg_latency_sec'), 's'):>10} "
                  f"{_fmt(cell.get('round_time_sec'), 's'):>10} "
                  f"{_fmt(cell.get('peak_vram_gb'), 'G'):>10} "
                  f"{_fmt(cell.get('benign_rejection_pct'), '%'):>8} "
                  f"{_fmt(cell.get('wasted_mb_per_round'), 'M'):>10} "
                  f"{_fmt(cell.get('total_upload_mb'), 'M'):>10} "
                  f"{seeds:>6}")

    out_dir = Path(args.output_dir)

    # Always generate compact (main paper figure)
    print("\nGenerating compact figure (1x2: latency + memory)...")
    plot_compact_scalability(data, active_defenses, all_n,
                             out_dir / "fig_system_scalability.pdf")

    # Full 2x3 unless --compact
    if not args.compact:
        print("Generating full figure (2x3: all 6 system metrics)...")
        plot_full_scalability(data, active_defenses, all_n,
                              out_dir / "fig_system_scalability_full.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
