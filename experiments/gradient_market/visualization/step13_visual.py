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
FIGURE_OUTPUT_DIR = "./figures/step13_figures"
VICTIM_SELLER_ID = "bn_3"  # The seller being targeted


# --- End Configuration ---

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    FIXED: Parses the base scenario name, accommodating the actual naming pattern
    like 'step13_drowning_drowning_martfl' or 'step13_drowning_martfl_bn_3'.
    """
    try:
        # Adjusted pattern: Uses a non-greedy wildcard followed by the defense name,
        # allowing an optional suffix.
        pattern = r'step13_.*_(fedavg|martfl|fltrust|skymask)(?:_bn_[0-9]+)?'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                # Note: victim_id cannot be reliably extracted from the top-level folder name alone.
            }
        else:
            # Setting defense to None ensures it's skipped cleanly in the main loop
            return {"scenario": scenario_name, "defense": None}

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name, "defense": None}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the results directory and aggregates all seller_metrics.csv data.
    """
    all_seller_rows = []
    base_path = Path(base_dir)
    print(f"Searching for 'seller_metrics.csv' in {base_path.resolve()}...")

    # Look for folders starting with the step prefix
    scenario_folders = [f for f in base_path.glob("step13_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step13_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)
        defense = run_scenario.get('defense')

        if defense is None:
            print(f"Skipping unparsed scenario: {scenario_name}")
            continue

        processed_seeds = 0

        # rglob finds the seller_metrics.csv file, which is deep in the seed folder
        seller_metric_files = list(scenario_path.rglob("seller_metrics.csv"))

        for metrics_csv in seller_metric_files:
            try:
                seller_df = pd.read_csv(metrics_csv)

                # Extract seed from file path (e.g., .../run_0_seed_42/seller_metrics.csv)
                seed_match = re.search(r'seed_([0-9]+)', str(metrics_csv.parent.name))
                seed_val = int(seed_match.group(1)) if seed_match else processed_seeds

                # Add scenario info
                seller_df['defense'] = defense
                seller_df['seed'] = seed_val

                # Group sellers into the three categories
                def get_group(seller_id):
                    # We rely on the global VICTIM_SELLER_ID
                    if seller_id == VICTIM_SELLER_ID:
                        return f"Victim ({VICTIM_SELLER_ID})"
                    elif seller_id.startswith('adv'):
                        return "Attackers"
                    else:
                        return "Other Benign"

                seller_df['seller_group'] = seller_df['seller_id'].apply(get_group)
                all_seller_rows.append(seller_df)
                processed_seeds += 1

            except Exception as e:
                print(f"Error reading {metrics_csv}: {e}")

    if not all_seller_rows:
        print("Error: No 'seller_metrics.csv' files were successfully processed.")
        return pd.DataFrame()

    return pd.concat(all_seller_rows, ignore_index=True)


def plot_drowning_attack(df: pd.DataFrame, output_dir: Path):
    """
    Generates the multi-panel line plot showing selection rate over time.
    """
    print(f"\n--- Plotting Targeted Drowning Attack Effectiveness ---")
    if df.empty:
        print("No data to plot.")
        return

    # We want the average selection rate (True=1, False=0) per group, per round, per defense
    plot_df = df.groupby(['defense', 'round', 'seller_group'])['selected'].mean().reset_index()

    # Rename 'selected' to 'selection_rate' for clarity
    plot_df.rename(columns={'selected': 'selection_rate'}, inplace=True)

    # FIXED: Ensure 'fedavg' is included and all found defenses are processed.
    defense_order = ['fedavg', 'martfl', 'fltrust', 'skymask']
    defense_order = [d for d in defense_order if d in plot_df['defense'].unique()]

    g = sns.relplot(
        data=plot_df,
        x='round',
        y='selection_rate',
        hue='seller_group',  # Lines for Victim, Attacker, Other
        style='seller_group',
        col='defense',  # Panel for each defense
        kind='line',
        col_order=defense_order,
        height=4,
        aspect=1.2,
        facet_kws={'sharey': True},
        markers=True,
        dashes=False
    )

    g.fig.suptitle(f'Effectiveness of Targeted Drowning Attack (Victim: {VICTIM_SELLER_ID})', y=1.05)
    g.set_axis_labels("Round", "Selection Rate")
    g.set_titles(col_template="{col_name}")
    g.legend.set_title("Seller Group")

    plot_file = output_dir / "plot_drowning_attack.png"
    g.fig.savefig(plot_file, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close(g.fig)


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if not df.empty:
        plot_drowning_attack(df, output_dir)

    print("\nAnalysis complete. Check 'step13_figures' folder for plots. 📊")


if __name__ == "__main__":
    main()