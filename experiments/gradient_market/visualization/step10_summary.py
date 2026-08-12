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
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"

# --- Use centralized naming, colors, and styling from viz_utils ---
from viz_utils import (
    DEFENSE_NAMES as PRETTY_NAMES, METRIC_COLORS as METRIC_PALETTE,
    DEFENSE_ORDER, fmt as format_label, setup_style as set_publication_style
)

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            # Expected format: step12_main_summary_{defense}_on_{dataset}
            # Index of 'summary':
            idx = parts.index('summary')
            return {
                "defense": parts[idx + 1],
                "dataset": parts[idx + 3] # Skip 'on'
            }
    except Exception:
        pass
    return {"defense": "unknown", "dataset": "unknown"}


def load_metrics_from_csv(run_dir: Path) -> pd.DataFrame:
    """Loads selection details from CSV if available."""
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty or 'seller_id' not in df.columns: return pd.DataFrame()

        df['type'] = df['seller_id'].apply(
            lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign'
        )

        if 'selected' in df.columns:
            # 'selected' column might be boolean or int
            df['selected'] = df['selected'].astype(int)
            summary = df.groupby('type')[['selected']].mean().reset_index()
            return summary
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks directories to find Step 12 results."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for Step 12 results in {base_path}...")

    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("defense") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                asr = metrics.get('asr', 0)
            except:
                acc = 0; asr = 0

            # Get selection rates from CSV
            df_val = load_metrics_from_csv(run_dir)

            flat_record = {**run_scenario, "acc": acc, "asr": asr}

            if not df_val.empty:
                for _, row in df_val.iterrows():
                    s_type = row['type']
                    if 'selected' in row:
                        flat_record[f"{s_type}_selected"] = row['selected']

            all_runs.append(flat_record)

    df = pd.DataFrame(all_runs)
    if not df.empty:
        df['defense'] = df['defense'].apply(format_label)

    return df

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_grouped_benchmark(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Generates the Grouped Bar Chart (4 bars per defense)."""
    print(f"\n--- Generating Grouped Benchmark for {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # 1. Normalize to Percentages
    if subset['acc'].max() <= 1.0: subset['acc'] *= 100
    if subset['asr'].max() <= 1.0: subset['asr'] *= 100
    if 'Benign_selected' in subset.columns: subset['Benign_selected'] *= 100
    if 'Adversary_selected' in subset.columns: subset['Adversary_selected'] *= 100

    # 2. Aggregate
    # We take the mean over multiple seeds/runs
    agg_df = subset.groupby('defense').mean(numeric_only=True).reset_index()

    # 3. Melt
    # UPDATED: Map the old dataframe columns to the new BSR and MSR acronyms
    rename_map = {
        'acc': 'Accuracy',
        'asr': 'ASR',
        'Benign_selected': 'BSR',
        'Adversary_selected': 'MSR'
    }
    cols_to_melt = [c for c in rename_map.keys() if c in agg_df.columns]

    melted = agg_df.melt(
        id_vars=['defense'],
        value_vars=cols_to_melt,
        var_name='Metric Type',
        value_name='Percentage'
    )
    melted['Metric Type'] = melted['Metric Type'].map(rename_map)

    # 4. Plot Setup
    # Use global style implicitly. Maintained exact figsize constraint.
    fig, ax = plt.subplots(figsize=(16, 8))

    # Ensure only present defenses are plotted
    defense_order = [d for d in DEFENSE_ORDER if d in melted['defense'].unique()]

    sns.barplot(
        data=melted,
        x='defense',
        y='Percentage',
        hue='Metric Type',
        order=defense_order,
        palette=METRIC_PALETTE,
        edgecolor='white',
        linewidth=1.5,
        ax=ax
    )

    # 5. Labels & Ticks
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)", labelpad=15)

    ax.set_ylim(0, 119) # Extra headroom for legend/labels

    # 6. Bar Annotations
    # UPDATED: Increased fontsize from 14 to 16 so the numbers pop out more
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=16, fontweight='bold')

    # 7. Legend (Top Center)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        title=None
    )

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 8. Save
    fname = output_dir / f"Step12_Grouped_Benchmark_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  Saved plot to: {fname}")
    plt.close()


def main():
    # 1. Apply Global Style
    set_publication_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir.resolve()}")

    # 2. Load Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found.")
        return

    # 3. Plot
    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_grouped_benchmark(df, dataset, output_dir)

    print("\n✅ Grouped Benchmark Figure Generated.")


if __name__ == "__main__":
    main()