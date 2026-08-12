"""
Pareto Frontier Visualization
===============================

Generates publication-quality plots showing the trade-offs between:
  - Model Utility (Accuracy or 1/Perplexity)
  - Security (1 - MSR, i.e., adversary rejection rate)
  - Economic Cost (Cost of Convergence in MB)

Produces (per dataset x attack type):
  1. 2D Pareto frontier: Utility vs Security (bubble size = CoC)
  2. 3D scatter: Utility vs Security vs CoC
  3. Radar/spider chart: multi-metric defense comparison
  4. CoC breakdown bar chart: per-defense total transmission cost

When multiple attack types are present, generates a multi-panel figure
faceted by attack so that different threat models are not conflated.

Usage:
    python experiments/gradient_market/visualization/pareto_frontier.py \
        --results_dir ./results --output_dir ./figures/pareto

    # For LLM experiments (uses perplexity instead of accuracy):
    python experiments/gradient_market/visualization/pareto_frontier.py \
        --results_dir ./results --output_dir ./figures/pareto --llm
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# Styling — use centralized registry from viz_utils
# ==========================================
from viz_utils import (
    DEFENSE_COLORS, DEFENSE_MARKERS, DEFENSE_NAMES as PRETTY_NAMES,
    ATTACK_NAMES, fmt, get_color, get_marker
)

PRETTY_ATTACK_NAMES = {
    **ATTACK_NAMES,
    "clean": "Clean",
    "gap": "GAP",
    "gap20": "GAP",
    "alignment_degradation": "Safety Degradation",
    "safedeg20": "Safety Degradation",
    "alignment_refusal": "Targeted Refusal",
    "refusal20": "Targeted Refusal",
}


def set_publication_style():
    from viz_utils import setup_style
    setup_style()


# ==========================================
# Data Collection
# ==========================================

def collect_pareto_data(base_dir: str, is_llm: bool = False) -> pd.DataFrame:
    """
    Walk result directories and extract (defense, attack, utility, security, cost) tuples.

    Searches for final_metrics.json, marketplace_report.json, and training_log.csv.
    """
    records = []
    base_path = Path(base_dir)

    for metrics_file in base_path.rglob("final_metrics.json"):
        run_dir = metrics_file.parent

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract defense and attack names from directory structure
        defense = _infer_defense(run_dir, base_path)
        dataset = _infer_dataset(run_dir, base_path)
        attack = _infer_attack(run_dir, base_path)

        # Utility metric
        if is_llm:
            ppl = metrics.get("perplexity", metrics.get("loss", None))
            if ppl is None:
                continue
            # Convert perplexity to utility: lower ppl = better
            utility = 1.0 / max(ppl, 1.0)
            utility_label = "1/Perplexity"
        else:
            utility = metrics.get("acc", 0.0)
            utility_label = "Accuracy"

        # Security: MSR (adversary selection rate) — lower is better
        msr = _get_msr(run_dir)
        security = 1.0 - msr  # Higher = more secure

        # Cost: cumulative upload in MB (prefer accepted-only if available)
        coc_mb = metrics.get("total_upload_mb_accepted",
                             metrics.get("total_upload_mb", 0.0))
        if coc_mb == 0.0:
            # Fallback: try to compute from training log
            coc_mb = _estimate_coc_from_log(run_dir)

        records.append({
            "defense": PRETTY_NAMES.get(defense.lower(), defense),
            "dataset": dataset,
            "attack": PRETTY_ATTACK_NAMES.get(attack.lower(), attack),
            "utility": utility,
            "utility_label": utility_label,
            "security": security,
            "msr": msr,
            "coc_mb": coc_mb,
            "acc": metrics.get("acc", None),
            "asr": metrics.get("asr", None),
            "perplexity": metrics.get("perplexity", None),
            "safety_score": metrics.get("safety_score", None),
            "refusal_bias": metrics.get("refusal_bias", None),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"Collected {len(df)} experiment results across "
              f"{df['defense'].nunique()} defenses, {df['attack'].nunique()} attack types")
    else:
        print("No results found.")
    return df


def _infer_defense(run_dir: Path, base_path: Path) -> str:
    """Infer defense name from directory path."""
    rel = str(run_dir.relative_to(base_path)).lower()
    for defense in ["spmc", "rflpa", "multi_krum", "trimmed_mean",
                     "skymask_small", "skymask", "martfl", "fltrust", "fedavg"]:
        if defense in rel:
            return defense
    return "unknown"


def _infer_dataset(run_dir: Path, base_path: Path) -> str:
    """Infer dataset name from directory path."""
    rel = str(run_dir.relative_to(base_path)).lower()
    for ds in ["cifar10", "cifar100", "trec", "texas100", "purchase100",
               "fed_chatbot_it", "fed_wildchat", "fed_chatbot_pa"]:
        if ds in rel:
            return ds
    return "unknown"


def _infer_attack(run_dir: Path, base_path: Path) -> str:
    """Infer attack type from directory path or config snapshot."""
    # First try config snapshot
    config_path = run_dir / "config_snapshot.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            adv_rate = cfg.get("experiment", {}).get("adv_rate", 0.0)
            if adv_rate == 0.0:
                return "clean"
            atk = cfg.get("data", {}).get("llm", {}).get("llm_attack", {}).get("attack_type", "none")
            if atk and atk != "none":
                return atk
        except Exception:
            pass

    # Fallback: infer from directory name
    rel = str(run_dir.relative_to(base_path)).lower()
    for atk_key in ["alignment_refusal", "refusal20", "alignment_degradation",
                     "safedeg20", "gap20", "gap"]:
        if atk_key in rel:
            return atk_key
    if "clean" in rel:
        return "clean"
    return "clean"


def _get_msr(run_dir: Path) -> float:
    """Extract adversary selection rate from marketplace report."""
    report_path = run_dir / "marketplace_report.json"
    if not report_path.exists():
        return 0.0
    try:
        with open(report_path) as f:
            report = json.load(f)
        sellers = report.get("seller_summaries", {})
        adv_rates = [
            s["selection_rate"] for s in sellers.values()
            if s.get("type") == "adversary"
        ]
        return np.mean(adv_rates) if adv_rates else 0.0
    except Exception:
        return 0.0


def _estimate_coc_from_log(run_dir: Path) -> float:
    """Fallback: estimate CoC from training_log.csv if available."""
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return 0.0
    try:
        df = pd.read_csv(log_path)
        # Prefer accepted-only CoC
        if "cumulative_upload_mb_accepted" in df.columns:
            return df["cumulative_upload_mb_accepted"].iloc[-1]
        if "cumulative_upload_mb" in df.columns:
            return df["cumulative_upload_mb"].iloc[-1]
        if "round_upload_mb" in df.columns:
            return df["round_upload_mb"].sum()
    except Exception:
        pass
    return 0.0


# ==========================================
# Pareto Frontier Computation
# ==========================================

def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Compute 2D Pareto frontier indices.
    Assumes BOTH dimensions should be MAXIMIZED.

    Args:
        points: (N, 2) array of (utility, security) values

    Returns:
        Boolean mask of Pareto-optimal points
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= on both and > on at least one
            if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                is_pareto[i] = False
                break

    return is_pareto


# ==========================================
# Plot 1: 2D Pareto Frontier (Utility vs Security, bubble=CoC)
#   Faceted by attack type when multiple attacks present
# ==========================================

def plot_pareto_2d(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Scatter plot: X = Utility, Y = Security (1-MSR), Bubble size = CoC (MB).
    When multiple attack types exist, creates a multi-panel figure.
    """
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return

    attacks = sorted(subset["attack"].unique())
    n_attacks = len(attacks)

    if n_attacks <= 1:
        # Single-panel (original behavior)
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_pareto_2d_panel(ax, subset, attacks[0] if attacks else "All")
    else:
        # Multi-panel faceted by attack
        fig, axes = plt.subplots(1, n_attacks, figsize=(8 * n_attacks, 8),
                                  sharey=True)
        if n_attacks == 1:
            axes = [axes]
        for ax, attack in zip(axes, attacks):
            atk_data = subset[subset["attack"] == attack]
            _plot_pareto_2d_panel(ax, atk_data, attack)
            if ax != axes[0]:
                ax.set_ylabel("")

    fig.suptitle(f"Pareto Frontier: {dataset}", fontsize=22, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"pareto_2d_{dataset}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_pareto_2d_panel(ax, subset: pd.DataFrame, attack_label: str):
    """Draw a single 2D Pareto panel for one attack type."""
    # Aggregate across seeds
    agg = subset.groupby("defense").agg({
        "utility": "mean", "security": "mean", "coc_mb": "mean",
        "utility_label": "first",
    }).reset_index()

    # Bubble sizes: normalize CoC to reasonable visual range
    max_coc = agg["coc_mb"].max()
    if max_coc > 0:
        sizes = 100 + 600 * (agg["coc_mb"] / max_coc)
    else:
        sizes = np.full(len(agg), 300)

    # Plot each defense
    for idx, row in agg.iterrows():
        defense = row["defense"]
        color = DEFENSE_COLORS.get(defense, "#333333")
        marker = DEFENSE_MARKERS.get(defense, "o")
        size = sizes[idx]

        ax.scatter(
            row["utility"], row["security"],
            s=size, c=color, marker=marker, edgecolors="black",
            linewidths=1.5, zorder=5, label=defense, alpha=0.85,
        )

    # Draw Pareto frontier
    points = agg[["utility", "security"]].values
    pareto_mask = compute_pareto_front(points)
    if pareto_mask.sum() > 1:
        pareto_pts = points[pareto_mask]
        sort_idx = np.argsort(pareto_pts[:, 0])
        pareto_sorted = pareto_pts[sort_idx]
        ax.plot(
            pareto_sorted[:, 0], pareto_sorted[:, 1],
            "--", color="#555555", linewidth=2.0, alpha=0.6,
            label="Pareto Frontier", zorder=3,
        )

    # Labels
    utility_label = agg["utility_label"].iloc[0]
    ax.set_xlabel(f"Model Utility ({utility_label})", fontsize=16, fontweight="bold")
    ax.set_ylabel("Security (1 - MSR)", fontsize=16, fontweight="bold")
    ax.set_title(f"Attack: {attack_label}", fontsize=18, fontweight="bold", pad=12)

    # CoC annotation
    for idx, row in agg.iterrows():
        ax.annotate(
            f'{row["coc_mb"]:.0f} MB',
            (row["utility"], row["security"]),
            textcoords="offset points", xytext=(8, -12),
            fontsize=9, color="#555555",
        )

    ax.legend(
        loc="lower left", framealpha=0.9, edgecolor="#cccccc",
        title="Defense (bubble = CoC)",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)


# ==========================================
# Plot 2: 3D Scatter (Utility vs Security vs CoC)
#   Faceted by attack type
# ==========================================

def plot_pareto_3d(df: pd.DataFrame, dataset: str, output_dir: Path):
    """3D scatter: Utility x Security x CoC, faceted by attack type."""
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return

    attacks = sorted(subset["attack"].unique())
    n_attacks = len(attacks)

    fig = plt.figure(figsize=(11 * max(n_attacks, 1), 8))

    for panel_idx, attack in enumerate(attacks):
        atk_data = subset[subset["attack"] == attack]
        agg = atk_data.groupby("defense").agg({
            "utility": "mean", "security": "mean", "coc_mb": "mean",
            "utility_label": "first",
        }).reset_index()

        ax = fig.add_subplot(1, n_attacks, panel_idx + 1, projection="3d")

        for _, row in agg.iterrows():
            defense = row["defense"]
            color = DEFENSE_COLORS.get(defense, "#333333")
            marker = DEFENSE_MARKERS.get(defense, "o")
            ax.scatter(
                row["utility"], row["security"], row["coc_mb"],
                s=200, c=color, marker=marker, edgecolors="black",
                linewidths=1.0, label=defense, alpha=0.9,
            )

        utility_label = agg["utility_label"].iloc[0]
        ax.set_xlabel(utility_label, fontsize=12, labelpad=10)
        ax.set_ylabel("Security (1-MSR)", fontsize=12, labelpad=10)
        ax.set_zlabel("CoC (MB)", fontsize=12, labelpad=10)
        ax.set_title(f"Attack: {attack}", fontsize=16, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    fig.suptitle(f"3D Trade-off: {dataset}", fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"pareto_3d_{dataset}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==========================================
# Plot 3: Radar Chart (Multi-metric comparison)
#   Faceted by attack type
# ==========================================

def plot_radar(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Radar/spider chart comparing defenses across:
    Utility, Security (1-MSR), Efficiency (1/CoC normalized), BSR
    """
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return

    attacks = sorted(subset["attack"].unique())
    n_attacks = len(attacks)

    fig, axes_flat = plt.subplots(1, n_attacks, figsize=(8 * max(n_attacks, 1), 8),
                                   subplot_kw=dict(polar=True))
    if n_attacks == 1:
        axes_flat = [axes_flat]

    for ax, attack in zip(axes_flat, attacks):
        atk_data = subset[subset["attack"] == attack]
        agg = atk_data.groupby("defense").agg({
            "utility": "mean", "security": "mean", "coc_mb": "mean", "msr": "mean",
        }).reset_index()

        # Normalize all metrics to [0, 1] for radar
        metrics = ["utility", "security", "efficiency", "benign_preservation"]
        agg["efficiency"] = 1.0 - (agg["coc_mb"] / agg["coc_mb"].max()) if agg["coc_mb"].max() > 0 else 1.0
        agg["benign_preservation"] = 1.0 - agg["msr"]

        u_min, u_max = agg["utility"].min(), agg["utility"].max()
        if u_max > u_min:
            agg["utility"] = (agg["utility"] - u_min) / (u_max - u_min)

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        for _, row in agg.iterrows():
            defense = row["defense"]
            color = DEFENSE_COLORS.get(defense, "#333333")
            values = [row[m] for m in metrics]
            values += values[:1]

            ax.plot(angles, values, "o-", color=color, linewidth=2.0, label=defense, markersize=6)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Utility", "Security", "Efficiency", "Benign\nPreservation"],
                            fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Attack: {attack}", fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)

    fig.suptitle(f"Defense Comparison: {dataset}", fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"radar_{dataset}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==========================================
# Plot 4: CoC Breakdown Bar Chart
# ==========================================

def plot_coc_breakdown(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Bar chart: total transmission cost per defense."""
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return

    agg = subset.groupby("defense")["coc_mb"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [DEFENSE_COLORS.get(d, "#333333") for d in agg.index]

    bars = ax.barh(agg.index, agg.values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, agg.values):
        ax.text(
            bar.get_width() + agg.max() * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} MB", va="center", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Total Transmission Cost (MB)", fontsize=14, fontweight="bold")
    ax.set_title(f"Cost of Convergence: {dataset}", fontsize=18, fontweight="bold")
    ax.set_xlim(0, agg.max() * 1.2)

    plt.tight_layout()
    out_path = output_dir / f"coc_breakdown_{dataset}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate Pareto frontier visualizations.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Base directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="./figures/pareto",
                        help="Directory to save figures")
    parser.add_argument("--llm", action="store_true",
                        help="Use perplexity (1/PPL) as utility instead of accuracy")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific datasets to plot (default: all found)")
    parser.add_argument("--attacks", type=str, nargs="*", default=None,
                        help="Specific attack types to plot (default: all found)")
    args = parser.parse_args()

    set_publication_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {args.results_dir}")
    df = collect_pareto_data(args.results_dir, is_llm=args.llm)

    if df.empty:
        print("No data found. Exiting.")
        return

    # Filter attacks if specified
    if args.attacks:
        pretty_attacks = {PRETTY_ATTACK_NAMES.get(a, a) for a in args.attacks}
        df = df[df["attack"].isin(pretty_attacks)]

    datasets = args.datasets or df["dataset"].unique().tolist()
    datasets = [d for d in datasets if d != "unknown"]

    print(f"\nDatasets found: {datasets}")
    print(f"Defenses found: {sorted(df['defense'].unique())}")
    print(f"Attack types found: {sorted(df['attack'].unique())}")

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"  Generating plots for: {dataset}")
        print(f"{'='*50}")

        plot_pareto_2d(df, dataset, output_dir)
        plot_pareto_3d(df, dataset, output_dir)
        plot_radar(df, dataset, output_dir)
        plot_coc_breakdown(df, dataset, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
