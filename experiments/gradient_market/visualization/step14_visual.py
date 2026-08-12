import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step14_collusion_figures"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",

    "random": "Random Noise",
    "inverse": "Inverse Gradient",

    # Datasets
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "fashion_mnist": "Fashion-MNIST",
    "trec": "TREC",
    "texas100": "Texas100",
    "purchase100": "Purchase100"
}

# --- Color Standards ---
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",   # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",   # Green
    "SkyMask": "#e74c3c",  # Red
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)
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
        'lines.linewidth': 3.0,
        'figure.figsize': (14, 8),
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses folder names. Handles two formats:
    1. step14_collusion_{MODE}_{DEFENSE} (Implicitly CIFAR100/Default)
    2. step14_collusion_{MODE}_{DEFENSE}_{DATASET} (Explicit)
    """
    try:
        # Regex to capture Mode, Defense, and Optional Dataset
        pattern = r'step14_collusion_(random|inverse)_(fedavg|martfl|fltrust|skymask|skymask_small)(?:_(.*))?'
        match = re.search(pattern, scenario_name)
        if match:
            data = {
                "attack_mode": match.group(1),
                "defense": match.group(2),
            }
            if match.group(3):
                data["dataset"] = match.group(3)
            else:
                data["dataset"] = "cifar100" # Default assumption for initial experiments
            return data
    except Exception:
        pass
    return {"defense": "unknown"}

def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        run_data['acc'] = metrics.get('acc', 0)
        if run_data['acc'] > 1.0: run_data['acc'] /= 100.0

        run_data['asr'] = metrics.get('asr', 0)
        if run_data['asr'] > 1.0: run_data['asr'] /= 100.0

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = report.get('seller_summaries', {}).values()
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
    print(f"Searching for Step 14 results in {base_path}...")

    scenario_folders = [f for f in base_path.glob("step14_collusion_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("defense") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_metrics = load_run_data(metrics_file)
            if run_metrics:
                all_runs.append({**run_scenario, **run_metrics})

    df = pd.DataFrame(all_runs)
    if not df.empty:
        df['defense'] = df['defense'].apply(format_label)
        df['attack_mode'] = df['attack_mode'].apply(format_label)
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].apply(format_label)

    return df

# ==========================================
# 3. PLOTTING FUNCTION 1: ALL DEFENSES
# ==========================================

def plot_collusion_impact_all_defenses(df: pd.DataFrame, output_dir: Path):
    """
    Visual 1: Compare ALL defenses on the hardest dataset (CIFAR-100).
    Shows how FLTrust/MartFL fail while SkyMask survives.
    """
    print(f"\n--- Plotting Visual 1: Collusion Impact (All Defenses) ---")

    # Filter for CIFAR-100 (Target Dataset for main comparison)
    # Adjust this string if your dataset parsing result differs
    target_dataset = "CIFAR-100"
    subset = df[df['dataset'] == target_dataset].copy()

    if subset.empty:
        # Fallback: Use the dataset with the most defenses present
        target_dataset = df['dataset'].mode()[0]
        subset = df[df['dataset'] == target_dataset].copy()
        print(f"  Target dataset not found. Falling back to: {target_dataset}")

    # Convert to %
    for col in ['acc', 'adv_selection_rate', 'benign_selection_rate']:
        if col in subset.columns: subset[col] *= 100

    # Filter for Random Noise mode
    subset = subset[subset['attack_mode'] == "Random Noise"]
    if subset.empty: return

    # Aggregate
    df_agg = subset.groupby('defense').mean(numeric_only=True).reset_index()

    active_defenses = [d for d in DEFENSE_ORDER if d in df_agg['defense'].unique()]

    # Create Figure (3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

    # 1. Hijack Rate
    sns.barplot(ax=axes[0], data=df_agg, x='defense', y='adv_selection_rate',
                order=active_defenses, palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
    axes[0].set_title("Baseline Hijack Rate\n(Adv. Selection)", pad=15)
    axes[0].set_ylabel("Selection Rate (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 105)

    # Annotate Failure
    for p in axes[0].patches:
        if p.get_height() > 80:
            axes[0].annotate('HIJACKED', (p.get_x()+p.get_width()/2., p.get_height()-10),
                             ha='center', va='top', color='white', fontweight='bold', fontsize=13)

    # 2. Benign Survival
    sns.barplot(ax=axes[1], data=df_agg, x='defense', y='benign_selection_rate',
                order=active_defenses, palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
    axes[1].set_title("Benign Seller Survival", pad=15)
    axes[1].set_ylabel("Selection Rate (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 105)

    # 3. Accuracy
    sns.barplot(ax=axes[2], data=df_agg, x='defense', y='acc',
                order=active_defenses, palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
    axes[2].set_title("Final Model Accuracy", pad=15)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_xlabel("")
    axes[2].set_ylim(0, 105)

    fname = output_dir / f"Step14_Visual1_AllDefenses_{target_dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"  Saved: {fname}")
    plt.close()

# ==========================================
# 4. PLOTTING FUNCTION 2: MARTFL DEEP DIVE
# ==========================================

def plot_martfl_multidataset_hijack(df: pd.DataFrame, output_dir: Path):
    """
    Visual 2: Analyze ONLY MartFL across MULTIPLE datasets.
    Shows that the 'Election' logic fails fundamentally regardless of data complexity.
    """
    print(f"\n--- Plotting Visual 2: MartFL Deep Dive (Multi-Dataset) ---")

    subset = df[(df['defense'] == "MARTFL") & (df['attack_mode'] == "Random Noise")].copy()

    if subset.empty:
        print("  No MartFL data found.")
        return

    subset['hijack_rate'] = subset['adv_selection_rate'] * 100

    # Dataset Order (Complexity)
    dataset_order = ["Fashion-MNIST", "CIFAR-10", "CIFAR-100", "TREC", "Texas100", "Purchase100"]
    # Filter for datasets we actually have
    active_datasets = [d for d in dataset_order if d in subset['dataset'].unique()]

    # If we have datasets not in the list, add them
    others = [d for d in subset['dataset'].unique() if d not in active_datasets]
    active_datasets.extend(others)

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data=subset,
        x='dataset',
        y='hijack_rate',
        order=active_datasets,
        color="#e74c3c", # Use Red to indicate high vulnerability
        edgecolor='black',
        linewidth=2.5,
        errorbar='sd',
        capsize=0.1
    )

    ax.set_title("MartFL Vulnerability: Baseline Hijack Rate", pad=20)
    ax.set_ylabel("Hijack Success Rate (%)", labelpad=15)
    ax.set_xlabel("Dataset", labelpad=15)
    ax.set_ylim(0, 105)

    # "Compromised" Line
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(len(active_datasets)-0.6, 52, "System Compromised (>50%)",
            fontsize=14, fontweight='bold', color='black', ha='right')

    # Annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 5:
            ax.annotate(f'{height:.0f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=16, fontweight='bold')

    fname = output_dir / "Step14_Visual2_MartFL_DeepDive.pdf"
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"  Saved: {fname}")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    set_publication_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # 2. Save Aggregated CSV
    df.to_csv(output_dir / "step14_full_summary.csv", index=False)

    # 3. Generate Both Visuals
    plot_collusion_impact_all_defenses(df, output_dir)
    plot_martfl_multidataset_hijack(df, output_dir)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()