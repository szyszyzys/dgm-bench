import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/skymask_comparison"
# ---------------------

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    return hps

def parse_scenario_name_universal(scenario_name: str) -> Dict[str, str]:
    try:
        pattern = r'step2\.5_find_hps_(?P<defense>.+?)_(?P<modality>image|text|tabular)_(?P<dataset>.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return match.groupdict()
        else:
            return {}
    except Exception as e:
        return {}

def collect_all_results_debug(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"\n--- SCANNING DIRECTORY: {base_path.resolve()} ---")

    if not base_path.exists():
        print(f"ERROR: The directory {base_path} does not exist!")
        return pd.DataFrame()

    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} potential folders.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name

        # Explicitly skip nolocalclip if needed, or comment out to include
        # if scenario_name.endswith("_nolocalclip"): continue

        run_scenario = parse_scenario_name_universal(scenario_name)
        if not run_scenario: continue

        metrics_files = list(scenario_path.rglob("final_metrics.json"))

        for metrics_file in metrics_files:
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)

                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                report_file = metrics_file.parent / "marketplace_report.json"
                adv_rate, ben_rate = 0.0, 0.0

                # Check for marketplace report
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        rep = json.load(f)
                    sellers = rep.get('seller_summaries', {}).values()
                    adv_s = [s for s in sellers if s.get('type') == 'adversary']
                    ben_s = [s for s in sellers if s.get('type') == 'benign']

                    if adv_s: adv_rate = np.mean([s['selection_rate'] for s in adv_s])
                    if ben_s: ben_rate = np.mean([s['selection_rate'] for s in ben_s])

                all_runs.append({
                    **run_scenario,
                    **run_hps,
                    'acc': metrics.get('acc', 0),
                    'rounds': metrics.get('completed_rounds', 0),
                    'adv_selection_rate': adv_rate,
                    'benign_selection_rate': ben_rate,
                    'path': str(metrics_file)
                })
            except Exception as e:
                continue

    df = pd.DataFrame(all_runs)
    return df

def add_labels(ax):
    """Helper to add value labels on bar charts"""
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: Path):
    # 1. Filter for skymask related items
    df_comp = df[df['defense'].str.contains("skymask", case=False)].copy()

    if df_comp.empty:
        print("\nNo 'skymask' related data found to plot.")
        return

    print(f"\n--- PROCESSING PLOTS ---")

    # 2. Get Best Run per Defense/Dataset (Maximize Accuracy)
    # We only want to compare the converged/best version of SkyMask vs SkyMask_Small
    # rather than averaging all the random hyperparameter sweep attempts.
    best_runs = df_comp.loc[df_comp.groupby(['dataset', 'defense'])['acc'].idxmax()].copy()

    datasets = best_runs['dataset'].unique()

    from viz_utils import setup_style
    setup_style()

    for ds in datasets:
        ds_data = best_runs[best_runs['dataset'] == ds].sort_values(by='defense')
        print(f"Plotting for dataset: {ds} | Defenses: {ds_data['defense'].tolist()}")

        # ==========================================
        # PLOT 1: ACCURACY COMPARISON
        # ==========================================
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=ds_data,
            x='defense',
            y='acc',
            palette='Blues_d',
            edgecolor='black'
        )
        add_labels(ax)
        plt.title(f"Best Accuracy Comparison - {ds}")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds}_comparison_accuracy.pdf")
        plt.close()

        # ==========================================
        # PLOT 2: SELECTION RATE COMPARISON
        # ==========================================
        # We need to melt the dataframe to plot Adv vs Benign side-by-side
        sel_cols = ['defense', 'adv_selection_rate', 'benign_selection_rate']

        # Check if we actually have selection rate data
        if ds_data['adv_selection_rate'].sum() > 0 or ds_data['benign_selection_rate'].sum() > 0:
            melted = ds_data[sel_cols].melt(
                id_vars='defense',
                var_name='metric',
                value_name='rate'
            )

            # Rename for cleaner legend
            melted['metric'] = melted['metric'].replace({
                'adv_selection_rate': 'Adversary',
                'benign_selection_rate': 'Benign'
            })

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=melted,
                x='defense',
                y='rate',
                hue='metric',
                palette={'Adversary': '#e74c3c', 'Benign': '#2ecc71'}, # Red for Adv, Green for Benign
                edgecolor='black'
            )
            add_labels(ax)
            plt.title(f"Selection Rate Comparison (SkyMask vs Small) - {ds}")
            plt.ylabel("Selection Rate (0-1)")
            plt.ylim(0, 1.1)
            plt.legend(title="Client Type", loc='upper right')
            plt.tight_layout()
            plt.savefig(output_dir / f"{ds}_comparison_selection_rates.pdf")
            plt.close()
            print(f"   -> Saved Selection Rate plot for {ds}")
        else:
            print(f"   -> Skipping selection rate plot for {ds} (All zeros or missing)")

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Use the collector
    df = collect_all_results_debug(BASE_RESULTS_DIR)

    if not df.empty:
        # Print a quick text summary of what we found before plotting
        print("\n--- Summary of Best Runs Found ---")
        best_runs_idx = df.groupby(['dataset', 'defense'])['acc'].idxmax()
        summary = df.loc[best_runs_idx, ['dataset', 'defense', 'acc', 'adv_selection_rate', 'benign_selection_rate']]
        print(summary.to_string(index=False))

        plot_comprehensive_comparison(df, output_dir)
        print(f"\nPlots saved to: {output_dir.resolve()}")
    else:
        print("No data found.")

if __name__ == "__main__":
    main()