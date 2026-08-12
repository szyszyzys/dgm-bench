import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_figures"
TARGET_VICTIM_ID = "bn_5"

# --- Naming Standards ---
PRETTY_NAMES = {
    # Defenses
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",  # Map the small version to the main name

    # Attacks
    "min_max": "Min-Max",
    "min_sum": "Min-Sum",
    "labelflip": "Label Flip",
    "label_flip": "Label Flip",
    "fang_krum": "Fang-Krum",
    "fang_trim": "Fang-Trim",
    "scaling": "Scaling Attack",
    "dba": "DBA",
    "badnet": "BadNet",
    "pivot": "Targeted Pivot",
    "0. Baseline": "No Attack (Baseline)"
}

# --- Color Standards ---
# Consistent colors across all papers figures
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",  # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",  # Green
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
    sns.set_context("talk", font_scale=1.2)

    # Force bold fonts globally
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 22,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'legend.title_fontsize': 17,
        'axes.linewidth': 2.5,
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'figure.figsize': (9, 6),
    })


# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses folder names. Handles BOTH Step 8 (Attacks) and Step 7 (Baseline).
    """
    try:
        # 1. Check for Step 8 (Buyer Attacks)
        pattern_step8 = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
        match8 = re.search(pattern_step8, scenario_name)
        if match8:
            return {
                "scenario": scenario_name,
                "type": "attack",
                "attack": match8.group(1),
                "defense": match8.group(2),
                "dataset": match8.group(3),
            }

        # 2. Check for Step 7 (Baseline - No Attack)
        pattern_step7 = r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
        match7 = re.search(pattern_step7, scenario_name)
        if match7:
            return {
                "scenario": scenario_name,
                "type": "baseline",
                "attack": "0. Baseline",
                "defense": match7.group(1),
                "dataset": match7.group(2),
            }

        return {"scenario": scenario_name, "type": "unknown"}
    except Exception as e:
        return {"scenario": scenario_name, "type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    """Loads performance metrics and per-seller selection rates."""
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        base_metrics['acc'] = acc
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})

        found_sellers = False
        for seller_id, seller_data in sellers.items():
            if seller_data.get('type') == 'benign':
                found_sellers = True
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        if not found_sellers:
            return [base_metrics]

        return run_records
    except Exception:
        return [base_metrics]


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)

    # Look for BOTH Step 8 and Step 7 folders
    folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]
    folders += [f for f in base_path.glob("step7_baseline_no_attack_*") if f.is_dir()]

    print(f"Found {len(folders)} scenario directories (Step 8 + Step 7).")

    for scenario_path in folders:
        run_info = parse_scenario_name(scenario_path.name)
        if run_info.get("type") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            records = load_run_data(metrics_file)
            for r in records:
                all_records.append({**run_info, **r})

    df = pd.DataFrame(all_records)

    # --- STANDARDIZATION STEP ---
    if not df.empty:
        # Apply standard names immediately
        df['defense'] = df['defense'].apply(format_label)
        df['attack'] = df['attack'].apply(format_label)

        # If we have 'skymask_small' data that was renamed to 'SkyMask',
        # ensure we don't have duplicates if old runs exist.
        # (Optional logic: drop duplicates if needed)

    return df


def get_baseline_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Computes the average selection rate for the 'No Attack' baseline."""
    baseline_label = format_label("0. Baseline")
    baseline_df = df[df['attack'] == baseline_label]

    if baseline_df.empty:
        print("⚠️ No Step 7 Baseline data found. Plots will rely only on Step 8 data.")
        return {}

    lookup = baseline_df.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()
    print(f"✅ Baseline calculated for {len(lookup)} configs.")
    return lookup


# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_buyer_attack_distribution(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    print("\n--- Plotting Selection Distributions (Fig 1) ---")

    baseline_label = format_label("0. Baseline")
    step8_attacks = [a for a in df['attack'].unique()
                     if 'Pivot' not in str(a) and a != baseline_label]

    for attack in step8_attacks:
        attack_df = df[df['attack'] == attack]
        if attack_df.empty: continue

        dataset = attack_df['dataset'].iloc[0] if 'dataset' in attack_df.columns else "Unknown"

        fig, ax = plt.subplots(figsize=(9, 6))

        # Boxplot with consistent palette
        sns.boxplot(
            data=attack_df,
            x='defense',
            y='selection_rate',
            order=DEFENSE_ORDER,
            palette=DEFENSE_COLORS,
            hue='defense',
            legend=False,
            ax=ax
        )

        # Draw Baseline Lines
        # We draw a red dashed line indicating the "Healthy" selection rate
        for i, defense in enumerate(DEFENSE_ORDER):
            base_val = baseline_lookup.get((defense, dataset))
            if base_val is not None:
                ax.hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4,
                          color='#c0392b', linestyle='--', lw=3,
                          label='Healthy Baseline' if i == 0 else "")

        # Add Custom Legend for Baseline
        if any(baseline_lookup.get((d, dataset)) for d in DEFENSE_ORDER):
            handles, labels = ax.get_legend_handles_labels()
            # Filter for the baseline line only
            baseline_handles = [h for h, l in zip(handles, labels) if "Baseline" in l]
            if baseline_handles:
                ax.legend(baseline_handles[:1], ["Healthy Baseline"], loc='lower right', frameon=True)

        ax.set_title(f'Selection Rate Distribution\nAttack: {attack}', pad=15)
        ax.set_ylabel("Selection Rate")
        ax.set_xlabel("Defense Strategy")
        ax.set_ylim(-0.05, 1.05)

        # Save
        safe_name = re.sub(r'[^\w]', '', attack)
        fname = output_dir / f"Step8_SELECTION_{safe_name}.pdf"
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {fname.name}")


def plot_targeted_attack_breakdown(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Targeted Breakdown (Fig 1.5) ---")
    # Filter for Pivot attacks
    pivot_df = df[df['attack'].str.contains("Pivot", case=False)].copy()

    if pivot_df.empty: return

    # Determine if seller is the specific victim or just another benign seller
    pivot_df['Status'] = pivot_df['seller_id'].apply(
        lambda x: 'Victim (Target)' if str(x) == TARGET_VICTIM_ID else 'Other Sellers'
    )

    plt.figure(figsize=(10, 7))

    # Grouped Bar Plot
    sns.barplot(
        data=pivot_df, x='defense', y='selection_rate', hue='Status',
        order=DEFENSE_ORDER,
        palette={'Victim (Target)': '#e74c3c', 'Other Sellers': '#95a5a6'},
        edgecolor='black',
        linewidth=1.5,
        errorbar='sd'
    )

    plt.title("Targeted Exclusion: Victim vs Others", pad=15)
    plt.ylabel("Selection Rate")
    plt.xlabel("")
    plt.ylim(0, 1.05)

    # Legend positioning
    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    fname = output_dir / "Step8_SELECTION_TARGETED_PIVOT.pdf"
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_buyer_attack_performance(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Performance Metrics (Fig 2) ---")

    # Deduplicate (we only need one entry per run, not per seller)
    run_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])

    baseline_label = format_label("0. Baseline")
    attacks = [a for a in run_df['attack'].unique() if a != baseline_label]
    attacks.sort()

    baseline_df = run_df[run_df['attack'] == baseline_label]

    for attack in attacks:
        current_attack_df = run_df[run_df['attack'] == attack]

        # Combine Baseline + Current Attack for comparison
        combined_df = pd.concat([baseline_df, current_attack_df], ignore_index=True)
        if combined_df.empty: continue

        # Melt for Faceted Plotting (Acc vs Rounds)
        melted = combined_df.melt(
            id_vars=['attack', 'defense'],
            value_vars=['acc', 'rounds'],
            var_name='MetricKey', value_name='Value'
        )
        melted['Metric'] = melted['MetricKey'].map({'acc': 'Accuracy', 'rounds': 'Rounds'})

        # Create FacetGrid
        g = sns.catplot(
            data=melted, x='defense', y='Value',
            hue='attack',
            col='Metric', kind='bar',
            order=DEFENSE_ORDER,
            height=5, aspect=1.2, sharey=False,
            palette={baseline_label: '#95a5a6', attack: '#c0392b'},  # Grey vs Red
            edgecolor='black', linewidth=1.2,
            legend_out=False
        )

        # Titles & Labels
        g.fig.suptitle(f'Marketplace Damage Assessment\nAttack: {attack}', y=1.05, fontweight='bold', fontsize=22)
        g.set_axis_labels("", "Metric Value")
        g.set_titles("{col_name}", fontweight='bold', size=18)

        safe_name = re.sub(r'[^\w]', '', attack)
        fname = output_dir / f"Step8_PERFORMANCE_{safe_name}.pdf"
        g.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def main():
    # 1. Apply Global Style
    set_publication_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots saved to: {output_dir.resolve()}")

    # 2. Collect Data
    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # 3. Calculate Baselines
    baseline_lookup = get_baseline_lookup(df)

    # 4. Generate Plots
    plot_buyer_attack_distribution(df, baseline_lookup, output_dir)
    plot_targeted_attack_breakdown(df, output_dir)
    plot_buyer_attack_performance(df, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
