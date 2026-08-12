import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step5_figures"

# --- Naming Standards ---
PRETTY_NAMES = {
    # Defenses
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",  # Map small version to main name

    # Attacks
    "min_max": "Min-Max",
    "min_sum": "Min-Sum",
    "labelflip": "Label Flip",
    "label_flip": "Label Flip",
    "fang_krum": "Fang-Krum",
    "fang_trim": "Fang-Trim",
    "scaling": "Scaling Attack",
    "dba": "DBA",
    "backdoor": "Backdoor",
    "badnet": "BadNet",
    "pivot": "Targeted Pivot",
    "0. Baseline": "No Attack (Baseline)"
}

# --- Color Standards ---
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",   # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",   # Green
    "SkyMask": "#e74c3c",  # Red (Highlighted)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    """Standardizes names using the dictionary or title-casing."""
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the 'Big & Bold' professional style for all figures."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)

    # Force bold fonts globally
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'legend.title_fontsize': 20,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.5,
        'lines.markersize': 11,
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses the HP suffix folder name (e.g., 'adv_0.1_poison_0.5')"""
    hps = {}
    pattern = r'adv_([0-9\.]+)_poison_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['adv_rate'] = float(match.group(1))
        hps['poison_rate'] = float(match.group(2))
    return hps


def parse_scenario_name(scenario_name: str) -> Optional[Dict[str, str]]:
    """Parses the base scenario name for step5."""
    try:
        pattern = r'step5_atk_sens_(adv|poison)_(fedavg|fltrust|martfl|skymask|skymask_small)_(backdoor|labelflip)_(image|text|tabular)$'
        match = re.search(pattern, scenario_name)

        if match:
            modality = match.group(4)
            # Map modality to dataset name
            if modality == 'image': dataset_name = 'CIFAR100'
            elif modality == 'text': dataset_name = 'TREC'
            elif modality == 'tabular': dataset_name = 'Texas100'
            else: dataset_name = 'unknown'

            return {
                "scenario": scenario_name,
                "sweep_type": match.group(1), # 'adv' or 'poison'
                "defense": match.group(2),
                "attack": match.group(3),
                "modality": modality,
                "dataset": dataset_name,
            }
        else:
            return None
    except Exception as e:
        return None


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key data from final_metrics.json and marketplace_report.json"""
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

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0
        else:
            run_data['adv_selection_rate'] = 0.0
            run_data['benign_selection_rate'] = 0.0

        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step5_atk_sens_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario is None: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                if not run_hps: continue

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

    # --- STANDARDIZATION STEP ---
    if not df.empty:
        # Apply standard names immediately so all plots use "SkyMask", "Min-Max", etc.
        df['defense'] = df['defense'].apply(format_label)
        df['attack'] = df['attack'].apply(format_label)

    return df

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_sensitivity_composite_row(df: pd.DataFrame, dataset: str, attack: str, output_dir: Path):
    """
    Generates a SINGLE wide figure (1x4) for the Sensitivity Analysis.
    Uses standard colors and order.
    """
    print(f"\n--- Plotting Composite Sensitivity Row: {dataset} ({attack}) ---")

    subset = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()
    if subset.empty: return

    # Convert to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        if col in subset.columns: subset[col] = subset[col] * 100

    df_adv_sweep = subset[subset['sweep_type'] == 'adv']
    df_poison_sweep = subset[subset['sweep_type'] == 'poison']

    if df_adv_sweep.empty and df_poison_sweep.empty: return

    # Initialize Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    # Ensure we only include defenses that actually exist in this data
    active_defenses = [d for d in DEFENSE_ORDER if d in subset['defense'].unique()]

    # --- Helper for consistent line plots ---
    def draw_lineplot(ax, data, x_col, y_col, title, xlabel, ylabel):
        if data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            return

        sns.lineplot(
            ax=ax, data=data, x=x_col, y=y_col,
            hue='defense', style='defense',
            markers=True, dashes=False,
            hue_order=active_defenses, style_order=active_defenses,
            palette=DEFENSE_COLORS  # <--- USES STANDARD COLORS
        )
        ax.set_title(title, pad=15)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_ylim(-5, 105)
        if ax.get_legend(): ax.get_legend().remove()

    # 1. ASR vs Adv Rate
    draw_lineplot(axes[0], df_adv_sweep, 'adv_rate', 'asr',
                  "(a) ASR vs. Adversary Rate", "Adversary Rate", "ASR (%)")

    # 2. Benign Selection vs Adv Rate
    draw_lineplot(axes[1], df_adv_sweep, 'adv_rate', 'benign_selection_rate',
                  "(b) Benign Select vs. Adv Rate", "Adversary Rate", "Selection Rate (%)")

    # 3. ASR vs Poison Rate
    draw_lineplot(axes[2], df_poison_sweep, 'poison_rate', 'asr',
                  "(c) ASR vs. Poison Rate", "Poison Rate", "ASR (%)")

    # 4. Accuracy vs Poison Rate
    draw_lineplot(axes[3], df_poison_sweep, 'poison_rate', 'acc',
                  "(d) Accuracy vs. Poison Rate", "Poison Rate", "Accuracy (%)")

    # --- Global Legend ---
    # We create a dummy legend from the lines in the last plot to ensure order
    handles, labels = axes[3].get_legend_handles_labels()
    if not handles and not df_adv_sweep.empty:
        handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(active_defenses),
            frameon=True,
            title="Defense Methods"
        )

    # Save
    safe_dataset = re.sub(r'[^\w]', '', dataset)
    safe_attack = re.sub(r'[^\w]', '', attack)
    filename = output_dir / f"plot_sensitivity_composite_{safe_dataset}_{safe_attack}.pdf"
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  -> Saved plot to: {filename}")
    plt.close('all')


def main():
    # 1. Apply Global Style
    set_publication_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 2. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # 3. Generate Plots
    combinations = df[['dataset', 'attack']].drop_duplicates().values
    for dataset, attack in combinations:
        if dataset == 'unknown' or pd.isna(attack): continue
        plot_sensitivity_composite_row(df, dataset, attack, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()