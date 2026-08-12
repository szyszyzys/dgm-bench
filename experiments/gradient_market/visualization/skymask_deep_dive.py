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
# Point this to the directory where your 'results' folder is
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./skymask_deep_dive_figures"


# ---------------------


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses 'step5_atk_sens_adv_skymask_backdoor_image'"""
    try:
        # step5_atk_sens_adv_skymask_backdoor_image
        pattern = r'step5_atk_sens_(adv|poison)_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(.+)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "sweep_type": f"{match.group(1)}_rate",  # adv_rate or poison_rate
                "defense": match.group(2),
                "attack_type": match.group(3),
                "modality_dataset": match.group(4),
                "dataset": "CIFAR100"  # Hardcoded from your config
            }
        raise ValueError("Pattern not matched")
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'adv_0.1_poison_1.0' into a dict."""
    hps = {}
    try:
        adv_match = re.search(r'adv_([0-9\.]+)', hp_folder_name)
        poison_match = re.search(r'poison_([0-9\.]+)', hp_folder_name)

        if adv_match:
            hps['adv_rate'] = float(adv_match.group(1))
        if poison_match:
            hps['poison_rate'] = float(poison_match.group(1))

    except Exception as e:
        print(f"Error parsing HPs from '{hp_folder_name}': {e}")
    return hps


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key metrics from output files."""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)

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

        # Handle the 0,0 attack case
        if run_data.get('adv_selection_rate') is np.nan and run_data.get('acc', 0) > 0:
            run_data['adv_selection_rate'] = 0.0  # No adversaries, so 0% selection

        return run_data
    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the Step 5 results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step5_atk_sens_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step5_atk_sens_*' directories found in {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        base_scenario_info = parse_scenario_name(scenario_path.name)

        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue
            hp_info = parse_hp_suffix(hp_path.name)

            for metrics_file in hp_path.rglob("final_metrics.json"):
                run_metrics = load_run_data(metrics_file)
                if run_metrics:
                    all_runs.append({
                        **base_scenario_info,
                        **hp_info,
                        **run_metrics
                    })

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_skymask_sensitivity(df_skymask: pd.DataFrame, output_dir: Path):
    """
    Plots the 2x2 grid showing SkyMask's performance vs. attack strength.
    This is the visual proof of failure.
    """
    print("\n--- Plotting SkyMask Sensitivity Analysis ---")
    if df_skymask.empty:
        print("No SkyMask data found to plot.")
        return

    # === Plot 1: Sweeping Adversary Rate ===
    df_sweep_adv = df_skymask[df_skymask['sweep_type'] == 'adv_rate'].copy()
    df_sweep_adv = df_sweep_adv.melt(
        id_vars=['adv_rate', 'attack_type'],
        value_vars=['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate'],
        var_name='Metric',
        value_name='Value'
    )

    g = sns.relplot(
        data=df_sweep_adv,
        x='adv_rate',
        y='Value',
        hue='attack_type',
        style='attack_type',
        col='Metric',
        kind='line',
        col_wrap=2,
        height=4,
        aspect=1.2,
        facet_kws={'sharey': False},
        markers=True
    )
    g.fig.suptitle("SkyMask Performance vs. Adversary Rate (Tuned HPs, CIFAR-100)", y=1.03)
    g.set_axis_labels("Adversary Rate (adv_rate)", "Value")
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / "plot_skymask_sensitivity_vs_ADV_RATE.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()

    # === Plot 2: Sweeping Poison Rate ===
    df_sweep_poison = df_skymask[df_skymask['sweep_type'] == 'poison_rate'].copy()
    df_sweep_poison = df_sweep_poison.melt(
        id_vars=['poison_rate', 'attack_type'],
        value_vars=['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate'],
        var_name='Metric',
        value_name='Value'
    )

    g = sns.relplot(
        data=df_sweep_poison,
        x='poison_rate',
        y='Value',
        hue='attack_type',
        style='attack_type',
        col='Metric',
        kind='line',
        col_wrap=2,
        height=4,
        aspect=1.2,
        facet_kws={'sharey': False},
        markers=True
    )
    g.fig.suptitle("SkyMask Performance vs. Poison Rate (Tuned HPs, CIFAR-100)", y=1.03)
    g.set_axis_labels("Poison Rate (poison_rate)", "Value")
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / "plot_skymask_sensitivity_vs_POISON_RATE.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots and CSV will be saved to: {output_dir.resolve()}")

    # Collect all data from Step 5
    df_all_defenses = collect_all_results(BASE_RESULTS_DIR)

    if df_all_defenses.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Create the requested CSV for SkyMask ---
    df_skymask_only = df_all_defenses[df_all_defenses['defense'] == 'skymask'].copy()

    csv_output_path = output_dir / "skymask_step5_sensitivity_results.csv"
    if not df_skymask_only.empty:
        df_skymask_only.to_csv(csv_output_path, index=False, float_format="%.4f")
        print(f"\n✅ Successfully saved SkyMask deep-dive data to: {csv_output_path}\n")
    else:
        print("\nWarning: No data found for SkyMask.\n")

    # --- Generate the plots for SkyMask ---
    plot_skymask_sensitivity(df_skymask_only, output_dir)

    print("\nAnalysis complete. Check 'skymask_deep_dive_figures' folder.")


if __name__ == "__main__":
    main()