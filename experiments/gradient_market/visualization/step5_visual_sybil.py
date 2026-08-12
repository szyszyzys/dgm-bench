import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step6_figures"

# --- COLOR STANDARDS (Matches Figure 3) ---
METRIC_PALETTE = {
    "Accuracy": "#2ca02c",  # Green
    "ASR": "#d62728",  # Red
    "BSR": "#1f77b4",  # Blue
    "MSR": "#ff7f0e"  # Orange
}


def set_publication_style():
    """
    Sets the 'Big & Bold' professional style globally.
    """
    sns.set_theme(style="whitegrid")
    # Global Scaling
    sns.set_context("paper", font_scale=2.0)

    # Specific Parameter Overrides
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',  # Bold all text
        'axes.labelweight': 'bold',  # Bold axis labels
        'axes.titleweight': 'bold',  # Bold titles

        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,

        'legend.fontsize': 18,
        'legend.title_fontsize': 20,

        'axes.linewidth': 2.5,  # Thicker axis borders
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0,
        'figure.figsize': (20, 9)  # Default large size (Overridden in compact plot)
    })


# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses the HP suffix folder name."""
    hps = {}
    adv_match = re.search(r'adv_([0-9\.]+)', hp_folder_name)
    poison_match = re.search(r'poison_([0-9\.]+)', hp_folder_name)
    if adv_match: hps['adv_rate'] = float(adv_match.group(1))
    if poison_match: hps['poison_rate'] = float(poison_match.group(1))

    blend_match = re.search(r'blend_alpha_([0-9\.]+)', hp_folder_name)
    if blend_match: hps['blend_alpha'] = float(blend_match.group(1))

    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name."""
    try:
        match = re.search(r'step6_adv_sybil_(.+)_(fedavg|martfl|fltrust|skymask|skymask_small)', scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "strategy": match.group(1),
                "defense": match.group(2),
            }
        else:
            return {"scenario": scenario_name, "defense": "unknown"}
    except Exception:
        return {"scenario": scenario_name}


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
            run_data['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step6_adv_sybil_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("defense") == "unknown": continue

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
                        **run_metrics,
                        "hp_suffix": hp_folder_name
                    })
            except Exception:
                continue

    df = pd.DataFrame(all_runs)

    if not df.empty and 'defense' in df.columns:
        # Basic standardization for defense names
        name_map = {'skymask_small': 'SkyMask', 'skymask': 'SkyMask',
                    'fedavg': 'FedAvg', 'fltrust': 'FLTrust', 'martfl': 'MARTFL'}
        df['defense'] = df['defense'].map(lambda x: name_map.get(x, x.title()))

    return df


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

def plot_sybil_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- LOGIC: Standardize Alpha ---
    def get_standardized_alpha_and_order(label):
        if label == 'baseline_no_sybil': return (0, 0.0)
        if label == 'mimic': return (1, 0.10)  # Set baseline mimicry alpha here
        if label == 'knock_out': return (1, 0.20)
        if label == 'pivot': return (1, 1.00)
        if label.startswith('oracle_blend'):
            try:
                attack_weight = float(label.split('_')[-1])
                alpha_centroid = 1.0 - attack_weight
                return (2, round(alpha_centroid, 2))
            except:
                return (2, 0.5)
        return (3, 0.0)

    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_standardized_alpha_and_order)

    # --- DATA PREP ---
    # UPDATED: Maps correctly to the formal metrics
    metric_map = {
        'acc': 'Accuracy',
        'asr': 'ASR',
        'adv_selection_rate': 'MSR',
        'benign_selection_rate': 'BSR'
    }
    metrics_to_plot = [m for m in metric_map.keys() if m in defense_df.columns]
    plot_df = defense_df.copy()
    for m in metrics_to_plot:
        plot_df[m] = plot_df[m] * 100

    plot_df = plot_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- PLOTTING ---
    plt.figure(figsize=(22, 9))

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette=METRIC_PALETTE,  # Enforce color consistency
        edgecolor="black",
        linewidth=2.0
    )

    plt.ylabel('Rate (%)', fontweight='bold', fontsize=20)
    plt.xlabel('Sybil Strategy (Mimicry Factor)', fontweight='bold', fontsize=20)
    plt.title(f"Sybil Attack Effectiveness vs {defense}", pad=20, fontweight='bold', fontsize=24)

    # --- FORMATTING TICKS ---
    def format_tick_label(l):
        sort_key = get_standardized_alpha_and_order(l)
        family_id = sort_key[0]
        alpha_val = sort_key[1]

        if family_id == 0:
            return "Baseline"
        elif family_id == 1:
            return f"Mimic\n($\\alpha={alpha_val}$)"
        elif family_id == 2:
            return f"Oracle\n($\\alpha={alpha_val}$)"
        else:
            return l.replace('_', '\n').title()

    formatted_labels = [format_tick_label(l) for l in sorted_labels]

    ax.set_xticklabels(formatted_labels, rotation=0, fontweight='bold', fontsize=18)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(18)

    ax.set_ylim(0, 105)

    plt.legend(
        title=None,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    safe_defense = re.sub(r'[^\w]', '', defense)
    plot_file = output_dir / f"plot_sybil_standardized_{safe_defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def plot_compact_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        return

    # Ensure strategy is accessible as a column
    if 'strategy' not in defense_df.columns:
        defense_df = defense_df.reset_index()

    print(f"Plotting Compact Sybil for: {defense}")

    # --- 1. FILTERING FOR SPECIFIC X-AXIS ---

    # Baseline
    mask_baseline = defense_df['strategy'].str.contains('baseline', case=False, na=False)

    # Mimic
    mask_mimic = defense_df['strategy'] == 'mimic'

    # Oracle: Filter for alpha = 0.9 (Data), but Label as 0.1 (Visual)
    mask_oracle = (defense_df['strategy'].str.contains('oracle', case=False, na=False)) & \
                  (np.isclose(defense_df['blend_alpha'], 0.9))

    # Create slices
    df_base = defense_df[mask_baseline].copy()
    df_base['DisplayLabel'] = 'Baseline'

    df_mimic = defense_df[mask_mimic].copy()
    df_mimic['DisplayLabel'] = 'Mimic\n($\\alpha=0.1$)'

    df_oracle = defense_df[mask_oracle].copy()
    df_oracle['DisplayLabel'] = 'Oracle\n($\\alpha=0.1$)'  # Visual label

    # Combine
    plot_df_source = pd.concat([df_base, df_mimic, df_oracle])

    if plot_df_source.empty:
        print(f"  -> Skipping {defense}: Strategies not found.")
        return

    # --- 2. PREPARE METRICS ---
    # UPDATED: Maps correctly to the formal metrics BSR and MSR
    metric_map = {
        'acc': 'Accuracy',
        'asr': 'ASR',
        'adv_selection_rate': 'MSR',
        'benign_selection_rate': 'BSR'
    }
    metrics_to_plot = [m for m in metric_map.keys() if m in plot_df_source.columns]

    for m in metrics_to_plot:
        plot_df_source[m] = plot_df_source[m] * 100

    plot_df = plot_df_source.melt(
        id_vars=['DisplayLabel'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- 3. PLOTTING ---
    plt.figure(figsize=(10, 6))  # Maintained exactly as requested

    x_order = ['Baseline', 'Mimic\n($\\alpha=0.1$)', 'Oracle\n($\\alpha=0.1$)']

    ax = sns.barplot(
        data=plot_df,
        x='DisplayLabel',
        y='Value',
        hue='Metric',
        order=[x for x in x_order if x in plot_df['DisplayLabel'].unique()],
        palette=METRIC_PALETTE,  # UPDATED: Consistent colors
        edgecolor="black",
        linewidth=2.0
    )

    # UPDATED: Enforcing bolding and sizing directly on the axis elements
    plt.ylabel('Rate (%)', fontweight='bold', fontsize=20)
    plt.xlabel(None)
    plt.title(f"Sybil Effectiveness: {defense}", pad=15, fontweight='bold', fontsize=24)

    ax.set_ylim(0, 105)

    # Bold Ticks
    ax.set_xticklabels([x for x in x_order if x in plot_df['DisplayLabel'].unique()], fontweight='bold', fontsize=18)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(18)

    plt.legend(
        title=None,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        fontsize=18
    )

    plt.tight_layout()

    safe_defense = re.sub(r'[^\w]', '', defense)
    plot_file = output_dir / f"compact_sybil_{safe_defense}.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  -> Saved: {plot_file}")
    plt.close('all')


def main():
    set_publication_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No data loaded.")
        return

    df = df[df['strategy'] != 'systematic_probe'].copy()

    metrics = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate', 'rounds']
    metrics = [m for m in metrics if m in df.columns]

    group_cols = ['defense', 'strategy', 'blend_alpha']
    df['blend_alpha'] = df['blend_alpha'].fillna(-1)

    df_agg = df.groupby(group_cols, as_index=False)[metrics].mean()

    if 'strategy' not in df_agg.columns:
        df_agg = df_agg.reset_index()

    # Generate Plots
    defenses = df_agg['defense'].unique()
    for defense in defenses:
        plot_compact_comparison(df_agg[df_agg['defense'] == defense].copy(), defense, output_dir)

    print("\nCompact analysis complete.")


if __name__ == "__main__":
    main()