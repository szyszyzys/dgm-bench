"""
Unified Figure Generator
========================
Generates all paper figures from experiment results using viz_utils.

Usage:
    python experiments/gradient_market/visualization/plot_all.py --results_dir ./results
    python experiments/gradient_market/visualization/plot_all.py --results_dir ./results --steps 10 4 5
    python experiments/gradient_market/visualization/plot_all.py --results_dir ./results --dataset CIFAR-100
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.gradient_market.visualization.viz_utils import (
    setup_style, load_step_results, save_figure, fmt,
    order_defenses, get_palette, get_color, get_marker,
    create_metric_row, add_shared_legend, plot_grouped_metrics,
    plot_line_sweep, plot_dual_stack, annotate_bars,
    METRIC_COLORS, VALUATION_COLORS,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json


# =============================================================================
# STEP 10: Main Summary — Grouped benchmark (Figure 3 in paper)
# =============================================================================

def plot_step10_main_summary(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """All defenses × all datasets — the main comparison table."""
    print("\n=== Step 10: Main Summary ===")
    df = load_step_results(results_dir, "step12")
    if df.empty:
        print("  No step 12 (main summary) results found.")
        return

    datasets = [dataset_filter] if dataset_filter else sorted(df["dataset"].dropna().unique())

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue

        # Grouped 4-metric bar chart
        fig = plot_grouped_metrics(sub, title=f"Main Benchmark — {ds}")
        save_figure(fig, str(output_dir / f"fig_main_benchmark_{ds}.pdf"))

        # Metric row (individual panels)
        fig, axes = create_metric_row(sub, metrics=["acc", "asr", "bsr", "msr"],
                                       titles=["Accuracy", "Attack Success Rate",
                                                "Benign Selection Rate", "Adversary Selection Rate"])
        add_shared_legend(fig, axes)
        save_figure(fig, str(output_dir / f"fig_main_metric_row_{ds}.pdf"))


# =============================================================================
# STEP 4: Attack Sensitivity — Line sweeps (Figure 4)
# =============================================================================

def plot_step4_attack_sensitivity(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Accuracy & ASR vs adv_rate and poison_rate."""
    print("\n=== Step 4: Attack Sensitivity ===")
    df = load_step_results(results_dir, "step5_atk")
    if df.empty:
        print("  No step 5 (attack sensitivity) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    for attack in df.get("attack", pd.Series()).dropna().unique():
        attack_df = df[df["attack"] == attack]

        # --- adv_rate sweep ---
        if "adv_rate" in attack_df.columns:
            sweep = attack_df.dropna(subset=["adv_rate"])
            if not sweep.empty:
                metrics = ["acc", "asr", "bsr", "msr"]
                fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5),
                                          constrained_layout=True)
                for i, m in enumerate(metrics):
                    if m not in sweep.columns:
                        continue
                    plot_line_sweep(sweep, x_col="adv_rate", y_col=m,
                                    xlabel="Adversary Rate", ylabel=f"{m.upper()} (%)",
                                    title=m.upper(), ax=axes[i])
                add_shared_legend(fig, axes)
                save_figure(fig, str(output_dir / f"fig_atk_sens_adv_rate_{attack}.pdf"))

        # --- poison_rate sweep ---
        if "poison_rate" in attack_df.columns:
            sweep = attack_df.dropna(subset=["poison_rate"])
            if not sweep.empty:
                metrics = ["acc", "asr", "bsr", "msr"]
                fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5),
                                          constrained_layout=True)
                for i, m in enumerate(metrics):
                    if m not in sweep.columns:
                        continue
                    plot_line_sweep(sweep, x_col="poison_rate", y_col=m,
                                    xlabel="Poison Rate", ylabel=f"{m.upper()} (%)",
                                    title=m.upper(), ax=axes[i])
                add_shared_legend(fig, axes)
                save_figure(fig, str(output_dir / f"fig_atk_sens_poison_rate_{attack}.pdf"))


# =============================================================================
# STEP 5: Sybil Strategies (Figure 5)
# =============================================================================

def plot_step5_sybil(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Per-defense comparison across sybil strategies."""
    print("\n=== Step 5: Sybil Strategies ===")
    df = load_step_results(results_dir, "step6_adv_sybil")
    if df.empty:
        print("  No step 6 (sybil) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    if "sybil_strategy" not in df.columns:
        print("  No sybil_strategy column found.")
        return

    metrics = ["acc", "asr", "bsr", "msr"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5),
                              constrained_layout=True)
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        ax = axes[i]
        defenses = order_defenses(df["defense"].dropna().unique().tolist())
        sns.barplot(data=df, x="sybil_strategy", y=m, hue="defense",
                     hue_order=defenses, palette=get_palette(defenses),
                     edgecolor="white", linewidth=1.2, ax=ax, errorbar="sd", capsize=0.08)
        ax.set_title(m.upper(), fontsize=22, fontweight="bold")
        ax.set_xlabel("Sybil Strategy", fontsize=18)
        ax.set_ylabel("(%)", fontsize=18)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax, right=True, top=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    add_shared_legend(fig, axes)
    save_figure(fig, str(output_dir / "fig_sybil_strategies.pdf"))


# =============================================================================
# STEP 6: Adaptive Attacks (Figure supplement)
# =============================================================================

def plot_step6_adaptive(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Adaptive attack robustness per defense."""
    print("\n=== Step 6: Adaptive Attacks ===")
    df = load_step_results(results_dir, "step7_adaptive")
    if df.empty:
        print("  No step 7 (adaptive) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    fig = plot_grouped_metrics(df, title="Adaptive Attack Robustness")
    save_figure(fig, str(output_dir / "fig_adaptive_attack.pdf"))


# =============================================================================
# STEP 7: Buyer Attacks / Demand-Side Bias (Figure 6)
# =============================================================================

def plot_step7_buyer_attacks(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Buyer-side attack impact per defense."""
    print("\n=== Step 7: Buyer Attacks ===")
    df = load_step_results(results_dir, "step8_buyer")
    if df.empty:
        print("  No step 8 (buyer attack) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    fig = plot_grouped_metrics(df, title="Demand-Side Bias")
    save_figure(fig, str(output_dir / "fig_buyer_attacks.pdf"))

    # Per-metric row
    fig, axes = create_metric_row(df, metrics=["acc", "asr", "bsr", "msr"])
    add_shared_legend(fig, axes)
    save_figure(fig, str(output_dir / "fig_buyer_attacks_row.pdf"))


# =============================================================================
# STEP 8: Scalability (Figure 7)
# =============================================================================

def plot_step8_scalability(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Metrics vs number of sellers."""
    print("\n=== Step 8: Scalability ===")
    df = load_step_results(results_dir, "step10_scalability")
    if df.empty:
        print("  No step 10 (scalability) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    if "n_sellers" not in df.columns:
        print("  No n_sellers column found.")
        return

    metrics = ["acc", "asr", "bsr", "msr"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5),
                              constrained_layout=True)
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        plot_line_sweep(df, x_col="n_sellers", y_col=m,
                         xlabel="Number of Sellers", ylabel=f"{m.upper()} (%)",
                         title=m.upper(), ax=axes[i])

    add_shared_legend(fig, axes)
    save_figure(fig, str(output_dir / "fig_scalability.pdf"))


# =============================================================================
# STEP 8 (system): System Scalability — latency + memory vs N
# =============================================================================

def plot_step8_system_scalability(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Latency and memory overhead vs number of sellers."""
    print("\n=== Step 8 (system): System Scalability ===")
    from plot_system_scalability import (
        collect_scalability_system_metrics, plot_system_scalability,
    )
    data = collect_scalability_system_metrics(Path(results_dir))
    if not data:
        print("  No scalability system metrics found.")
        return
    plot_system_scalability(data, output_dir / "fig_system_scalability.pdf")


# =============================================================================
# STEP 9: Heterogeneity (Figure supplement)
# =============================================================================

def plot_step9_heterogeneity(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Metrics vs dirichlet alpha."""
    print("\n=== Step 9: Heterogeneity ===")
    df = load_step_results(results_dir, "step11_")
    if df.empty:
        print("  No step 11 (heterogeneity) results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    if "dirichlet_alpha" not in df.columns:
        print("  No dirichlet_alpha column found.")
        return

    metrics = ["acc", "asr", "bsr", "msr"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5),
                              constrained_layout=True)
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        plot_line_sweep(df, x_col="dirichlet_alpha", y_col=m,
                         xlabel="Dirichlet α (→ more IID)", ylabel=f"{m.upper()} (%)",
                         title=m.upper(), ax=axes[i])

    add_shared_legend(fig, axes)
    save_figure(fig, str(output_dir / "fig_heterogeneity.pdf"))


# =============================================================================
# STEP 10 Valuation: Dual-Stack KernelSHAP (Figure 8)
# =============================================================================

def plot_step10_valuation(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Dual-stack valuation breakdown (KernelSHAP, LOO, Influence, Selection Rate)."""
    print("\n=== Step 10: Valuation Dual Stack ===")

    base = Path(results_dir)
    target_ds = (dataset_filter or "CIFAR-100").lower().replace("-", "")

    metrics_to_plot = ["selection_rate", "marginal_contrib_loo", "kernelshap_score", "influence_score"]
    metric_titles = {
        "selection_rate": "Selection Rate",
        "marginal_contrib_loo": "LOO Marginal Contribution",
        "kernelshap_score": "KernelSHAP Value",
        "influence_score": "Influence Score",
    }

    for metric in metrics_to_plot:
        records = []
        for folder in sorted(base.glob("step12_*")):
            # Check dataset match
            if target_ds not in folder.name.lower().replace("-", ""):
                continue

            # Extract defense from folder name
            defense = None
            for part in folder.name.split("_"):
                if part.lower() in ("fedavg", "fltrust", "martfl", "skymask"):
                    defense = fmt(part)
                    break
            if not defense:
                continue

            # Scan valuations.jsonl
            for jfile in folder.rglob("valuations.jsonl"):
                benign_paid, benign_disc, adv_paid, adv_disc = 0, 0, 0, 0
                count = 0
                try:
                    with open(jfile) as f:
                        lines = f.readlines()
                    start = max(0, len(lines) // 2)  # Use second half (converged)
                    for line in lines[start:]:
                        rec = json.loads(line)
                        selected = set(rec.get("selected_ids", []))
                        vals = rec.get("seller_valuations", {})

                        for sid, data in vals.items():
                            is_adv = str(sid).startswith("adv")
                            if metric == "selection_rate":
                                val = 1.0
                            elif metric in data and data[metric] is not None:
                                val = max(0, float(data[metric]))
                            else:
                                continue

                            if is_adv:
                                if sid in selected: adv_paid += val
                                else: adv_disc += val
                            else:
                                if sid in selected: benign_paid += val
                                else: benign_disc += val
                        count += 1
                except Exception:
                    continue

                if count > 0:
                    records.append({
                        "defense": defense,
                        "Benign_Paid": benign_paid,
                        "Benign_Discarded": benign_disc,
                        "Adv_Paid": adv_paid,
                        "Adv_Discarded": adv_disc,
                    })

        if records:
            val_df = pd.DataFrame(records)
            fig = plot_dual_stack(val_df, title=metric_titles.get(metric, metric))
            save_figure(fig, str(output_dir / f"fig_valuation_dual_stack_{metric}.pdf"))
        else:
            print(f"  No valuation data for {metric}")


# =============================================================================
# STEP 13: Drowning Attack
# =============================================================================

def plot_step13_drowning(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """Drowning attack effectiveness."""
    print("\n=== Step 13: Drowning Attack ===")
    df = load_step_results(results_dir, "step13")
    if df.empty:
        print("  No step 13 results found.")
        return

    fig = plot_grouped_metrics(df, title="Drowning Attack")
    save_figure(fig, str(output_dir / "fig_drowning_attack.pdf"))


# =============================================================================
# STEP 14: MartFL Collusion
# =============================================================================

def plot_step14_collusion(results_dir: str, output_dir: Path, dataset_filter: str = None):
    """MartFL collusion analysis."""
    print("\n=== Step 14: MartFL Collusion ===")
    df = load_step_results(results_dir, "step14")
    if df.empty:
        print("  No step 14 results found.")
        return

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]

    fig = plot_grouped_metrics(df, title="MartFL Collusion")
    save_figure(fig, str(output_dir / "fig_martfl_collusion.pdf"))


# =============================================================================
# MAIN
# =============================================================================

STEP_PLOTTERS = {
    10:  ("Main Summary (all defenses × datasets)", plot_step10_main_summary),
    4:   ("Attack Sensitivity (adv_rate, poison_rate sweeps)", plot_step4_attack_sensitivity),
    5:   ("Sybil Strategies", plot_step5_sybil),
    6:   ("Adaptive Attacks", plot_step6_adaptive),
    7:   ("Buyer Attacks / Demand-Side Bias", plot_step7_buyer_attacks),
    8:   ("Scalability (n_sellers)", plot_step8_scalability),
    80:  ("System Scalability (latency + memory)", plot_step8_system_scalability),
    9:   ("Data Heterogeneity (dirichlet alpha)", plot_step9_heterogeneity),
    100: ("Valuation Dual Stack (KernelSHAP)", plot_step10_valuation),
    13:  ("Drowning Attack", plot_step13_drowning),
    14:  ("MartFL Collusion", plot_step14_collusion),
}


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Steps to plot (default: all). Use 100 for valuation.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to specific dataset (e.g., CIFAR-100)")
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = args.steps or sorted(STEP_PLOTTERS.keys())
    ds = fmt(args.dataset) if args.dataset else None

    print(f"Results: {args.results_dir}")
    print(f"Output:  {output_dir}")
    print(f"Dataset: {ds or 'all'}")
    print(f"Steps:   {steps}")

    for step in steps:
        if step not in STEP_PLOTTERS:
            print(f"\nUnknown step {step}, skipping.")
            continue
        desc, plotter = STEP_PLOTTERS[step]
        print(f"\n{'='*60}")
        print(f"  Step {step}: {desc}")
        print(f"{'='*60}")
        try:
            plotter(args.results_dir, output_dir, dataset_filter=ds)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
