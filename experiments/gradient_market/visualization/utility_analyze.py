import json
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_RESULTS_DIR = "./results"
OUTPUT_DIR = "./figures/utility_analysis"
TARGET_DEFENSE = "fedavg"  # <--- Change this to 'martfl', 'skymask', etc. to analyze others
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    return hps

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    # Universal parser
    pattern = r'step2\.5_find_hps_(?P<defense>.+?)_(?P<modality>image|text|tabular)_(?P<dataset>.+)'
    match = re.search(pattern, scenario_name)
    if match:
        return match.groupdict()
    return {}

def collect_data(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"🔍 Scanning {base_path} for '{TARGET_DEFENSE}' results...")

    # Find folders that look like "step2.5_find_hps_..."
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        # 1. Parse Scenario
        info = parse_scenario_name(scenario_path.name)
        if not info or info.get('defense') != TARGET_DEFENSE:
            continue

        # 2. Find Runs inside
        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                # Get Hyperparameters from folder name
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hps = parse_hp_suffix(relative_parts[0])

                # Load Metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                all_runs.append({
                    **info,
                    **hps,
                    'acc': metrics.get('acc', 0) * 100, # Convert to %
                    'rounds': metrics.get('completed_rounds', 0),
                    'path': str(metrics_file.parent)
                })
            except Exception:
                continue

    return pd.DataFrame(all_runs)

def analyze_utility(df: pd.DataFrame):
    if df.empty:
        print(f"❌ No results found for defense: {TARGET_DEFENSE}")
        return

    datasets = df['dataset'].unique()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n📊 UTILITY ANALYSIS FOR: {TARGET_DEFENSE.upper()}")
    print("="*60)

    for ds in datasets:
        ds_data = df[df['dataset'] == ds].copy()

        print(f"\n👉 Dataset: {ds}")

        # 1. Top 5 Configurations
        top_runs = ds_data.sort_values(by='acc', ascending=False).head(5)
        print("\n🏆 Top 5 Best Configurations:")
        print(top_runs[['learning_rate', 'local_epochs', 'acc', 'rounds']].to_string(index=False))

        # 2. Pivot Table (Heatmap style text)
        print("\nheatmap (Avg Accuracy per LR/Epochs):")
        pivot = ds_data.pivot_table(index='learning_rate', columns='local_epochs', values='acc', aggfunc='mean')
        print(pivot.round(2))

        # 3. Plotting
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Line plot: Accuracy vs LR, colored by Epochs
        sns.lineplot(
            data=ds_data,
            x='learning_rate',
            y='acc',
            hue='local_epochs',
            style='local_epochs',
            markers=True,
            dashes=False,
            palette="viridis",
            linewidth=2.5,
            err_style="bars" # Shows variance if you have multiple seeds
        )

        plt.title(f"Utility Sensitivity: {TARGET_DEFENSE.title()} on {ds}", fontsize=16, fontweight='bold')
        plt.xlabel("Learning Rate", fontsize=14)
        plt.ylabel("Test Accuracy (%)", fontsize=14)
        plt.xscale('log') # Log scale is usually better for LR sweeps
        plt.legend(title="Local Epochs")
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Save
        out_path = Path(OUTPUT_DIR) / f"utility_{TARGET_DEFENSE}_{ds}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"   [Saved plot to {out_path}]")

if __name__ == "__main__":
    df = collect_data(BASE_RESULTS_DIR)
    analyze_utility(df)