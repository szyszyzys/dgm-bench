import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step11_figures_visuals"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

# --- Color Standards ---
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",  # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",  # Green
    "SkyMask": "#e74c3c",  # Red (Highlighted)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

# --- Markers for Line Plots ---
CUSTOM_MARKERS = {
    "FedAvg": "o",
    "FLTrust": "X",
    "MARTFL": "s",
    "SkyMask": "P"
}

# --- Constants for Heterogeneity Plot ---
HET_VAL_MAP = {100.0: 0, 1.0: 1, 0.5: 2, 0.1: 3}
HET_LABELS = ["IID", "1.0", "0.5", "0.1"]

# --- Constants for Scarcity Plot ---
RATIOS_IN_TEST = [0.01, 0.05, 0.1, 0.2]


def format_label(label: str) -> str:
    """Standardizes names."""
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())


def set_publication_style():
    """Sets the 'Big & Bold' professional style globally."""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # UPDATED: Enforce the same bolding and sizes used in Figures 4-8
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 24,  # Increased from 20
        'axes.labelsize': 20,  # Increased from 18
        'xtick.labelsize': 18,  # Increased from 14
        'ytick.labelsize': 18,  # Increased from 14
        'legend.fontsize': 20,  # Increased from 16
        'axes.titlepad': 8,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.5,
        'lines.markersize': 11,
        # Font Type for PDF editing
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses folder suffixes into numerical values."""
    hps = {}
    if hp_folder_name == "iid":
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = 100.0
        return hps

    match_alpha = re.search(r'alpha_([0-9\.]+)', hp_folder_name)
    if match_alpha:
        hps['experiment_type'] = 'heterogeneity'
        hps['x_val'] = float(match_alpha.group(1))
        return hps

    match_ratio = re.search(r'ratio_([0-9\.]+)', hp_folder_name)
    if match_ratio:
        hps['experiment_type'] = 'scarcity'
        hps['x_val'] = float(match_ratio.group(1))
        return hps

    return {}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads metrics from JSONs."""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0
        run_data['acc'] = acc
        run_data['asr'] = metrics.get('asr', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = list(report.get('seller_summaries', {}).values())
            adv = [s for s in sellers if s.get('type') == 'adversary']
            ben = [s for s in sellers if s.get('type') == 'benign']
            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv]) if adv else 0.0
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben]) if ben else 0.0
        else:
            run_data['adv_selection_rate'] = 0.0
            run_data['benign_selection_rate'] = 0.0

        return run_data
    except:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} step11 folders.")

    for folder in scenario_folders:
        folder_name = folder.name
        match = re.match(r'step11_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', folder_name)
        if not match: continue

        defense = match.group(1)
        dataset = match.group(2)

        for metrics_file in folder.rglob("final_metrics.json"):
            try:
                parts = metrics_file.parent.relative_to(folder).parts
                if len(parts) < 2: continue
                subdir_type = parts[0]
                hp_folder = parts[1]

                if subdir_type == "vary_buyer":
                    bias_source = "Buyer-Only Bias"
                elif subdir_type == "vary_seller":
                    bias_source = "Seller-Only Bias"
                elif subdir_type == "scarcity":
                    bias_source = "Data Scarcity"
                else:
                    continue

                run_hps = parse_hp_suffix(hp_folder)
                if 'experiment_type' not in run_hps: continue

                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({
                        "defense": defense,
                        "dataset": dataset,
                        "bias_source": bias_source,
                        **run_hps,
                        **metrics
                    })
            except Exception:
                continue

    df = pd.DataFrame(all_runs)
    if not df.empty:
        df['defense'] = df['defense'].apply(format_label)
    return df


# ==========================================
# 3. COMPACT PLOTTING FUNCTIONS
# ==========================================

def plot_heterogeneity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Heterogeneity Plots (4-Col) for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'heterogeneity')].copy()
    if dataset_df.empty: return

    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    dataset_df['x_mapped'] = dataset_df['x_val'].map(HET_VAL_MAP)
    dataset_df = dataset_df.dropna(subset=['x_mapped'])

    valid_biases = ['Buyer-Only Bias', 'Seller-Only Bias']

    # UPDATED: Mapping to BSR and MSR to match the rest of the paper
    metrics_order = [('acc', '(a) Accuracy'), ('asr', '(b) ASR'),
                     ('benign_selection_rate', '(c) BSR'), ('adv_selection_rate', '(d) MSR')]

    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]

    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        # UPDATED: figsize to (28, 4.8) to perfectly match Figures 4, 5, and 6
        fig, axes = plt.subplots(1, 4, figsize=(28, 4.8), constrained_layout=True)

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]
            sns.lineplot(
                ax=ax, data=bias_df, x='x_mapped', y=col_name,
                hue='defense', style='defense',
                hue_order=active_defenses, style_order=active_defenses,
                palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
                dashes=False, errorbar=('ci', 95)
            )
            # Enforce title size explicitly
            ax.set_title(display_name, fontweight='bold', fontsize=24)
            # Standardize y-limits for percentages
            ax.set_ylim(-5, 105)

            if i == 0:
                ax.set_ylabel("Rate (%)", fontweight='bold', fontsize=20)
            else:
                ax.set_ylabel("")

            # Integer Ticks mapping
            ax.set_xticks(sorted(HET_VAL_MAP.values()))
            ax.set_xticklabels(HET_LABELS)
            ax.set_xlabel(r"Heterogeneity ($\alpha$)", fontweight='bold', fontsize=20)
            ax.set_xlim(-0.2, 3.2)

            # Bold Ticks
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(18)

            if ax.get_legend(): ax.get_legend().remove()

        # TOP LEGEND - shifted up slightly to accommodate larger titles
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       ncol=len(active_defenses), frameon=False, fontsize=20)

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Heterogeneity_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


def plot_scarcity_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Scarcity Plots (4-Col) for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'scarcity')].copy()
    if dataset_df.empty: return

    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    # UPDATED: Mapping to BSR and MSR
    metrics_order = [('acc', '(a) Accuracy'), ('asr', '(b) ASR'),
                     ('benign_selection_rate', '(c) BSR'), ('adv_selection_rate', '(d) MSR')]
    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]

    # UPDATED: figsize to (28, 4.8)
    fig, axes = plt.subplots(1, 4, figsize=(28, 4.8), constrained_layout=True)

    for i, (col_name, display_name) in enumerate(metrics_order):
        ax = axes[i]
        sns.lineplot(
            ax=ax, data=dataset_df, x='x_val', y=col_name,
            hue='defense', style='defense',
            hue_order=active_defenses, style_order=active_defenses,
            palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
            dashes=False, errorbar=('ci', 95)
        )
        ax.set_title(display_name, fontweight='bold', fontsize=24)
        ax.set_ylim(-5, 105)

        if i == 0:
            ax.set_ylabel("Rate (%)", fontweight='bold', fontsize=20)
        else:
            ax.set_ylabel("")

        ax.set_xscale('log')
        ax.set_xticks(RATIOS_IN_TEST)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("Data Ratio", fontweight='bold', fontsize=20)

        # Bold Ticks
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)

        if ax.get_legend(): ax.get_legend().remove()

    # TOP LEGEND
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                   ncol=len(active_defenses), frameon=False, fontsize=20)

    fname = output_dir / f"Step11_Scarcity_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


def plot_heterogeneity_selection_only(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Heterogeneity Selection Only (2-Col) for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'heterogeneity')].copy()
    if dataset_df.empty: return

    for col in ['benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    dataset_df['x_mapped'] = dataset_df['x_val'].map(HET_VAL_MAP)
    dataset_df = dataset_df.dropna(subset=['x_mapped'])

    metrics_order = [('benign_selection_rate', '(a) BSR'),
                     ('adv_selection_rate', '(b) MSR')]

    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]
    valid_biases = ['Buyer-Only Bias', 'Seller-Only Bias']

    for bias in valid_biases:
        bias_df = dataset_df[dataset_df['bias_source'] == bias]
        if bias_df.empty: continue

        # UPDATED: Adjusted height to match 4.8
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

        for i, (col_name, display_name) in enumerate(metrics_order):
            ax = axes[i]
            sns.lineplot(
                ax=ax, data=bias_df, x='x_mapped', y=col_name,
                hue='defense', style='defense',
                hue_order=active_defenses, style_order=active_defenses,
                palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
                dashes=False, errorbar=('ci', 95)
            )
            ax.set_title(display_name, fontweight='bold', fontsize=24)
            ax.set_ylim(-5, 105)

            if i == 0:
                ax.set_ylabel("Selection Rate (%)", fontweight='bold', fontsize=20)
            else:
                ax.set_ylabel("")

            ax.set_xticks(sorted(HET_VAL_MAP.values()))
            ax.set_xticklabels(HET_LABELS)
            ax.set_xlabel(r"Heterogeneity ($\alpha$)", fontweight='bold', fontsize=20)
            ax.set_xlim(-0.2, 3.2)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(18)

            if ax.get_legend(): ax.get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       ncol=len(active_defenses), frameon=False, fontsize=20)

        safe_bias = bias.replace(' ', '').replace('-', '')
        fname = output_dir / f"Step11_Heterogeneity_SelectionOnly_{dataset}_{safe_bias}.pdf"
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        print(f"  Saved: {fname.name}")
        plt.close()


def plot_scarcity_selection_only(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Scarcity Selection Only (2-Col) for {dataset} ---")

    dataset_df = df[(df['dataset'] == dataset) & (df['experiment_type'] == 'scarcity')].copy()
    if dataset_df.empty: return

    for col in ['benign_selection_rate', 'adv_selection_rate']:
        dataset_df[col] = dataset_df[col] * 100

    metrics_order = [('benign_selection_rate', '(a) BSR'),
                     ('adv_selection_rate', '(b) MSR')]

    active_defenses = [d for d in DEFENSE_ORDER if d in dataset_df['defense'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    for i, (col_name, display_name) in enumerate(metrics_order):
        ax = axes[i]
        sns.lineplot(
            ax=ax, data=dataset_df, x='x_val', y=col_name,
            hue='defense', style='defense',
            hue_order=active_defenses, style_order=active_defenses,
            palette=DEFENSE_COLORS, markers=CUSTOM_MARKERS,
            dashes=False, errorbar=('ci', 95)
        )
        ax.set_title(display_name, fontweight='bold', fontsize=24)
        ax.set_ylim(-5, 105)

        if i == 0:
            ax.set_ylabel("Selection Rate (%)", fontweight='bold', fontsize=20)
        else:
            ax.set_ylabel("")

        ax.set_xscale('log')
        ax.set_xticks(RATIOS_IN_TEST)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("Data Ratio", fontweight='bold', fontsize=20)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)

        if ax.get_legend(): ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                   ncol=len(active_defenses), frameon=False, fontsize=20)

    fname = output_dir / f"Step11_Scarcity_SelectionOnly_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


def main():
    set_publication_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No valid data found.")
        return

    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_heterogeneity_row(df, dataset, output_dir)
            plot_scarcity_row(df, dataset, output_dir)
            # 2-Col Special Versions
            plot_heterogeneity_selection_only(df, dataset, output_dir)
            plot_scarcity_selection_only(df, dataset, output_dir)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()