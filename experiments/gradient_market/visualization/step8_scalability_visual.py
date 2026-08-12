import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step10_figures"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",  # Map both to same display name
}

# --- Color Standards (Consistent with your paper) ---
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",  # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",  # Green
    "SkyMask": "#e74c3c",  # Red (Highlighted)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]


def format_label(label: str) -> str:
    """Standardizes names."""
    if not isinstance(label, str): return str(label)
    # 1. Lowercase lookup
    if label.lower() in PRETTY_NAMES:
        return PRETTY_NAMES[label.lower()]
    # 2. Fallback formatting
    return label.replace("_", " ").title()


def set_publication_style():
    """Sets the 'Compact & Bold' professional style globally."""
    from viz_utils import setup_style
    setup_style()


# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name for Step 10."""
    try:
        # Regex: step10_scalability_<defense>_<dataset>
        pattern = r'step10_scalability_(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2)
            }
        else:
            return {}
    except Exception:
        return {}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'n_sellers_10' etc."""
    hps = {}
    pattern = r'n_sellers_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match: hps['n_sellers'] = int(match.group(1))
    return hps


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key data from final_metrics.json"""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            run_data['adv_selection_rate'] = np.mean(
                [s['selection_rate'] for s in adv_sellers]) if adv_sellers else np.nan
            run_data['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan

            # Handle cases where adv exists but list is empty or similar edge cases
            if not adv_sellers and ben_sellers:
                run_data['adv_selection_rate'] = 0.0

        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step10_scalability_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if not run_scenario: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics
                    })
            except Exception:
                continue

    df = pd.DataFrame(all_runs)

    # --- Standardization ---
    if not df.empty:
        # Apply standard naming
        df['defense'] = df['defense'].apply(format_label)

    return df


# ==========================================
# 3. PLOTTING FUNCTIONS (COMPACT)
# ==========================================

def plot_scalability_composite_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a COMPACT LOW-HEIGHT figure (1x4) for Scalability.
    Dimensions: 28 x 4.8
    """
    print(f"\n--- Plotting Composite Scalability Row: {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Convert to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        if col in subset.columns: subset[col] = subset[col] * 100

    # Determine Active Defenses
    active_defenses = [d for d in DEFENSE_ORDER if d in subset['defense'].unique()]

    # --- STYLE ---
    set_publication_style()

    # --- LAYOUT: Low Height (4.8) ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 4.8), constrained_layout=True)

    # Metric Configurations
    # UPDATED: (metric_col, title_text, y_label) to use BSR and MSR formal acronyms
    plot_configs = [
        ('acc', '(a) Accuracy vs Sellers', 'Accuracy (%)'),
        ('asr', '(b) ASR vs Sellers', 'ASR (%)'),
        ('benign_selection_rate', '(c) BSR vs Sellers', 'BSR (%)'),
        ('adv_selection_rate', '(d) MSR vs Sellers', 'MSR (%)')
    ]

    for i, (metric, title, ylabel) in enumerate(plot_configs):
        ax = axes[i]

        if metric in subset.columns and not subset[metric].isna().all():
            sns.lineplot(
                ax=ax,
                data=subset,
                x='n_sellers',
                y=metric,
                hue='defense',
                style='defense',
                hue_order=active_defenses,
                style_order=active_defenses,
                palette=DEFENSE_COLORS,  # <--- ENFORCES COLOR CONSISTENCY
                markers=True,
                dashes=False,
            )

            # Bold Titles & Labels (Using dynamic sizes set in rcParams)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel("Number of Sellers", fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_ylim(-5, 105)

            # Force Integer Ticks for Sellers (10, 20, 30...)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Bold Ticks
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            ax.grid(True, alpha=0.5, linewidth=1.5)

            # Remove individual legends
            if ax.get_legend(): ax.get_legend().remove()

        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontweight='bold')

    # --- LEGEND: TOP CENTER (Saving Vertical Space) ---
    # Use handles from the first plot
    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        # UPDATED: Bbox anchor shifted slightly to 1.05 to accommodate larger titles
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(active_defenses),
            frameon=False,
        )

    # Save
    safe_dataset = re.sub(r'[^\w]', '', dataset)
    filename = output_dir / f"plot_scalability_composite_{safe_dataset}.pdf"

    # bbox_inches='tight' crops perfectly around the short figure
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  -> Saved plot to: {filename}")
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    print(f"Loaded {len(df)} rows across datasets: {df['dataset'].unique()}")

    # 2. Plot for every dataset found
    for dataset in df['dataset'].unique():
        if pd.notna(dataset):
            plot_scalability_composite_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()