import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step3_figures"

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90
# Fairness filter threshold
REASONABLE_BSR_THRESHOLD = 0.50


# --- Styling Helper ---
def set_plot_style():
    """Sets a consistent professional style for all plots."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.5


# --- Functions from Step 2.5 (for lookup) ---

def parse_scenario_name_step2_5(scenario_name: str) -> Dict[str, str]:
    try:
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return {"dataset": match.group(3), "defense": match.group(1)}
        else:
            return {}
    except Exception:
        return {}


def load_run_data_step2_5(metrics_file: Path) -> Dict[str, Any]:
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return {'acc': metrics.get('acc', 0)}
    except Exception:
        return {}


def get_step2_5_max_acc_lookup(base_dir: str) -> Dict[str, float]:
    print("--- Loading Step 2.5 data to find 'best' accuracy... ---")
    all_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]

    if not scenario_folders:
        print("Warning: No 'step2.5_find_hps_*' directories found.")
        return {}

    for scenario_path in scenario_folders:
        scenario_info = parse_scenario_name_step2_5(scenario_path.name)
        if not scenario_info: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_metrics = load_run_data_step2_5(metrics_file)
            if run_metrics:
                all_runs.append({
                    "dataset": scenario_info['dataset'],
                    "defense": scenario_info['defense'],
                    "acc": run_metrics['acc']
                })

    if not all_runs:
        return {}

    df_step2_5 = pd.DataFrame(all_runs)
    dataset_max_acc = df_step2_5.groupby('dataset')['acc'].max().to_dict()
    return dataset_max_acc


# --- Step 3 Parsing Functions ---

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    patterns = {
        'martfl.max_k': r'martfl\.max_k_([0-9]+)',
        'clip_norm': r'clip_norm_([0-9\.]+|None)',
        'mask_epochs': r'mask_epochs_([0-9]+)',
        'mask_lr': r'mask_lr_([0-9e\.\+]+)',
        'mask_threshold': r'mask_threshold_([0-9\.]+)',
        'mask_clip': r'mask_clip_([0-9e\.\-]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, hp_folder_name)
        if match:
            value_str = match.group(1)
            clean_key = key.split('.')[-1]
            if value_str == "None":
                hps[clean_key] = 'None'
            else:
                try:
                    val_float = float(value_str)
                    if val_float == int(val_float):
                        hps[clean_key] = int(val_float)
                    else:
                        hps[clean_key] = val_float
                except ValueError:
                    hps[clean_key] = value_str
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        # Explicitly handle the _new suffix in the regex
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+?)_(.+?)(_new|_old)?$'
        # Note: Your folder structure seemed to have defense repeated or different order,
        # adjusting regex based on your previous valid runs.
        # Standard: step3_tune_<DEFENSE>_<ATTACK>_<MODALITY>_<DATASET>_<MODEL>_new
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+?)_(.+?)(_new|_old)?$'

        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "attack": match.group(2),
                "modality": match.group(3),
                "dataset": match.group(4),
                "model": match.group(5),
            }
        else:
            return {"scenario": scenario_name}
    except Exception as e:
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
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

            # Extract mean selection rates
            if adv_sellers:
                run_data['adv_selection_rate'] = pd.Series([s['selection_rate'] for s in adv_sellers]).mean()
            else:
                run_data['adv_selection_rate'] = 0.0

            if ben_sellers:
                run_data['benign_selection_rate'] = pd.Series([s['selection_rate'] for s in ben_sellers]).mean()
            else:
                run_data['benign_selection_rate'] = 0.0

        return run_data
    except Exception as e:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)

    print(f"Searching for 'step3_tune_*_new' results in {base_path.resolve()}...")
    scenario_folders = [f for f in base_path.glob("step3_tune_*_new") if f.is_dir()]

    if not scenario_folders:
        print(f"Error: No directories matching 'step3_tune_*_new' found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} valid '_new' scenario directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue
            hp_folder_name = hp_path.name
            run_hps = parse_hp_suffix(hp_folder_name)

            for metrics_file in hp_path.rglob("final_metrics.json"):
                try:
                    run_metrics = load_run_data(metrics_file)
                    if run_metrics:
                        all_runs.append({
                            **run_scenario,
                            **run_hps,
                            **run_metrics,
                            "hp_suffix": hp_folder_name
                        })
                except Exception as e:
                    pass

    if not all_runs:
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    df['defense_score'] = df['acc'] - df['asr']
    if 'mask_clip' in df.columns:
        df['mask_clip'] = df['mask_clip'].fillna(1.0)

    # Add flags
    if 'benign_selection_rate' in df.columns:
        df['is_fair'] = df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
    else:
        df['is_fair'] = True

    return df


def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path):
    print("\n--- Generating Trade-off Scatter Plots ---")
    set_plot_style()

    combinations = df[['dataset', 'attack']].drop_duplicates().values

    for dataset, attack in combinations:
        subset = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()
        if subset.empty: continue

        subset['acc_pct'] = subset['acc'] * 100
        subset['asr_pct'] = subset['asr'] * 100
        subset['status'] = 'Unfair/Filtered'
        subset.loc[subset['is_fair'], 'status'] = 'Valid Candidate'

        plt.figure(figsize=(10, 8))

        sns.scatterplot(
            data=subset, x="acc_pct", y="asr_pct",
            hue="defense",
            style="status",
            markers={"Valid Candidate": "o", "Unfair/Filtered": "X"},
            palette="deep", s=120, alpha=0.8, edgecolor="black"
        )

        ax = plt.gca()
        rect = patches.Rectangle((85, 0), 20, 10, linewidth=2,
                                 edgecolor='green', facecolor='green', alpha=0.1, linestyle='--')
        ax.add_patch(rect)

        plt.text(92.5, 5, "Target Zone", color='green',
                 fontsize=12, fontweight='bold', ha='center', va='center')

        plt.title(f"Defense Trade-off: {dataset} ({attack})", fontsize=16, fontweight='bold')
        plt.xlabel("Model Utility (Accuracy %)", fontsize=14)
        plt.ylabel("Attack Success Rate (ASR %)", fontsize=14)
        plt.xlim(0, 105)
        plt.ylim(-5, 105)
        plt.axhline(y=90, color='red', linestyle=':', alpha=0.5, label="Attack Goal")
        plt.legend(title="Defense & Status", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()

        filename = output_dir / f"plot_tradeoff_scatter_{dataset}_{attack}.pdf"
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        plt.close('all')
        print(f"Saved scatter plot: {filename}")


# --- MAIN FUNCTION ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    dataset_max_acc_lookup = get_step2_5_max_acc_lookup(BASE_RESULTS_DIR)

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No 'step3_tune_*_new' results data was loaded. Exiting.")
        return

    df.to_csv(output_dir / "step3_full_results_summary.csv", index=False, float_format="%.4f")

    # Calculate Usability
    if dataset_max_acc_lookup:
        df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc_lookup)
        if df['dataset_max_acc'].isnull().any():
            step3_max_acc = df.groupby('dataset')['acc'].max().to_dict()
            df['dataset_max_acc'] = df['dataset_max_acc'].fillna(df['dataset'].map(step3_max_acc))
        df['usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
        df['platform_usable'] = df['acc'] >= df['usable_threshold']
    else:
        df['platform_usable'] = True

    # Use only usable runs for the tables
    reasonable_acc_df = df[df['platform_usable']].copy()

    # =========================================================
    # TABLE 1: TUNING SUMMARY (Added Selection Rates)
    # =========================================================
    agg_metrics = {
        'hp_suffix': 'nunique',
        'acc': ['min', 'max', 'mean'],
        'asr': ['min', 'max', 'mean']
    }

    # Add selection rate aggregation if columns exist
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        agg_metrics['benign_selection_rate'] = 'mean'
    if 'adv_selection_rate' in reasonable_acc_df.columns:
        agg_metrics['adv_selection_rate'] = 'mean'

    if not reasonable_acc_df.empty:
        df_summary = reasonable_acc_df.groupby(['defense', 'dataset', 'attack']).agg(agg_metrics)
        # Flatten MultiIndex columns
        df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
        df_summary = df_summary.reset_index()

        def rng(row, m):
            return f"{row[f'{m}_min'] * 100:.1f}-{row[f'{m}_max'] * 100:.1f}" if f'{m}_min' in row else "N/A"

        df_summary['ACC Range %'] = df_summary.apply(lambda r: rng(r, 'acc'), axis=1)
        df_summary['ASR Range %'] = df_summary.apply(lambda r: rng(r, 'asr'), axis=1)
        df_summary = df_summary.rename(columns={'hp_suffix_nunique': 'N_HPs'})

        # Format Selection Rates
        if 'benign_selection_rate_mean' in df_summary.columns:
            df_summary['Benign Select %'] = (df_summary['benign_selection_rate_mean'] * 100).map('{:.1f}'.format)
        else:
            df_summary['Benign Select %'] = "N/A"

        if 'adv_selection_rate_mean' in df_summary.columns:
            df_summary['Adv. Select %'] = (df_summary['adv_selection_rate_mean'] * 100).map('{:.1f}'.format)
        else:
            df_summary['Adv. Select %'] = "N/A"

        cols = [
            'defense', 'dataset', 'attack', 'N_HPs',
            'ACC Range %', 'ASR Range %',
            'Benign Select %', 'Adv. Select %'
        ]
        final_cols = [c for c in cols if c in df_summary.columns]

        table_path_1 = output_dir / "step3_tuning_summary_range.tex"
        df_summary[final_cols].to_latex(table_path_1, index=False)
        print(f"Saved Table 1 (Summary) to {table_path_1}")

    # =========================================================
    # TABLE 2: BEST HPs (Added Selection Rates)
    # =========================================================
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        df_final = reasonable_acc_df[reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD].copy()
    else:
        df_final = reasonable_acc_df.copy()

    if not df_final.empty:
        df_final['sort_metric'] = np.where(
            df_final['attack'] == 'backdoor',
            df_final['asr'], 1.0 - df_final['acc']
        )
        best_rows = df_final.sort_values(['scenario', 'sort_metric']).groupby('scenario').head(1)

        # Format Selection Rates
        if 'adv_selection_rate' in best_rows.columns:
            best_rows['Adv. Select %'] = (best_rows['adv_selection_rate'] * 100).map('{:.1f}'.format)
        else:
            best_rows['Adv. Select %'] = "N/A"

        if 'benign_selection_rate' in best_rows.columns:
            best_rows['Benign Select %'] = (best_rows['benign_selection_rate'] * 100).map('{:.1f}'.format)
        else:
            best_rows['Benign Select %'] = "N/A"

        # Format Acc/ASR
        best_rows['Acc %'] = (best_rows['acc'] * 100).map('{:.1f}'.format)
        best_rows['ASR %'] = (best_rows['asr'] * 100).map('{:.1f}'.format)

        # Format Parameters string
        hp_cols = [c for c in best_rows.columns if c in ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold']]
        def format_hps(row):
            params = []
            for col in hp_cols:
                val = row[col]
                if pd.notna(val) and val != 'None':
                    short_name = col.replace('mask_', '').replace('clip_norm', 'clip')
                    params.append(f"{short_name}={val}")
            return ", ".join(params)

        best_rows['Parameters'] = best_rows.apply(format_hps, axis=1)

        display_cols = [
            'defense', 'dataset', 'attack',
            'Parameters',
            'Acc %', 'ASR %',
            'Benign Select %', 'Adv. Select %'
        ]
        final_cols = [c for c in display_cols if c in best_rows.columns]

        table_path_2 = output_dir / "step3_best_hps_table.tex"
        best_rows[final_cols].to_latex(
            table_path_2,
            index=False,
            caption="Optimal Hyperparameters and Selection Rates",
            label="tab:step3_best_hps"
        )
        print(f"Saved Table 2 (Best HPs) to {table_path_2}")

    # --- PLOTTING ---
    plot_tradeoff_scatter(df, output_dir)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()