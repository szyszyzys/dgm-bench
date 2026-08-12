import json
import os
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_filtered_view"
TARGET_VICTIM_ID = "bn_5"

# --- ONLY SHOW THESE DEFENSES ---
DEFENSE_ORDER = ["FLTrust", "MARTFL"]

# Consistent Colors for these two
DEFENSE_COLORS = {
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71"  # Green
}

ATTACK_CATEGORIES = {
    "disruption": [
        "DoS",
        "Trust Erosion",
        "Oscillating (Binary)", "Oscillating (Random)", "Oscillating (Drift)"
    ],
    "manipulation": [
        "Starvation",
        "Class Exclusion (Neg)",
        "Class Exclusion (Pos)",
        "Oscillating (Binary)",
        "Oscillating (Random)",
        "Oscillating (Drift)"
    ],
    "isolation": [
        "Pivot"
    ]
}


def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2.0)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',

        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,

        'legend.fontsize': 16, # Optimized for horizontal fit
        'legend.title_fontsize': 18,

        'axes.linewidth': 2.5,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0,
    })


# ==========================================
# 2. DATA PROCESSING
# ==========================================

def format_label(label: str) -> str:
    label = label.lower()
    mapping = {
        "fedavg": "FedAvg", "fltrust": "FLTrust",
        "martfl": "MARTFL", "skymask": "SkyMask",
        "dos": "DoS", "erosion": "Trust Erosion",
        "oscillating_binary": "Oscillating (Binary)",
        "oscillating_random": "Oscillating (Random)",
        "oscillating_drift": "Oscillating (Drift)",
        "starvation": "Starvation",
        "class_exclusion_neg": "Class Exclusion (Neg)",
        "class_exclusion_pos": "Class Exclusion (Pos)",
        "orthogonal_pivot_legacy": "Pivot", "pivot": "Pivot",
        "0. baseline": "Healthy Baseline"
    }
    return mapping.get(label, label.replace("_", " ").title())


def parse_scenario(scenario_name: str):
    pattern = r'(step[78])_(baseline_no_attack|buyer_attack)_(?:(.+?)_)?(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
    match = re.search(pattern, scenario_name)
    if match:
        _, mode, attack_raw, defense, dataset = match.groups()
        attack = "0. Baseline" if "baseline" in mode else attack_raw
        return {
            "attack": format_label(attack),
            "defense": format_label(defense),
            "dataset": dataset
        }
    return None


def load_data(base_dir: str):
    records = []
    base_path = Path(base_dir)
    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))

    print(f"Scanning {len(folders)} folders...")

    for path in folders:
        meta = parse_scenario(path.name)
        if not meta: continue

        if meta['defense'] not in DEFENSE_ORDER:
            continue

        for mfile in path.rglob("final_metrics.json"):
            try:
                with open(mfile) as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                if acc > 1.0: acc /= 100.0

                report_file = mfile.parent / "marketplace_report.json"
                if report_file.exists():
                    with open(report_file) as rf:
                        report = json.load(rf)
                    for sid, sdata in report.get('seller_summaries', {}).items():
                        if sdata.get('type') == 'benign':
                            records.append({
                                **meta,
                                "acc": acc,
                                "seller_id": sid,
                                "selection_rate": sdata.get('selection_rate', 0.0)
                            })
            except Exception:
                pass

    return pd.DataFrame(records)


# ==========================================
# 3. VISUALIZATION LOGIC
# ==========================================

def plot_disruption_impact(df, output_dir):
    attacks = [a for a in ATTACK_CATEGORIES["disruption"] if a in df['attack'].unique()]
    if not attacks: return
    print(f"--- Plotting Disruption (Accuracy) ---")

    subset = df[df['attack'].isin(attacks + ["Healthy Baseline"])].copy()
    subset = subset.drop_duplicates(subset=['attack', 'defense', 'acc'])

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        data=subset, x="defense", y="acc", hue="attack",
        order=DEFENSE_ORDER, palette="magma",
        edgecolor="black", linewidth=2.0
    )

    plt.ylabel("Test Accuracy", fontweight='bold')
    plt.xlabel("")
    plt.ylim(0, 1.0)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Attack Type", frameon=False)
    plt.savefig(output_dir / "1_Disruption_Accuracy.pdf", bbox_inches='tight')
    plt.close()


def plot_manipulation_fairness(df, output_dir):
    manipulation_attacks = [a for a in ATTACK_CATEGORIES["manipulation"] if a in df['attack'].unique()]
    if not manipulation_attacks: return
    print(f"--- Plotting Manipulation (Selection Rates) ---")

    attacks_to_plot = manipulation_attacks
    if "Healthy Baseline" in df['attack'].unique():
        attacks_to_plot = ["Healthy Baseline"] + manipulation_attacks

    subset = df[df['attack'].isin(attacks_to_plot)].copy()

    for attack in manipulation_attacks:
        comparison_subset = subset[subset['attack'].isin(["Healthy Baseline", attack])].copy()
        if comparison_subset.empty: continue

        # 1. SHORTER ASPECT RATIO (6 width, 3.8 height)
        plt.figure(figsize=(6, 3.8))

        my_palette = {"Healthy Baseline": "#95a5a6", attack: "#e74c3c"}

        ax = sns.stripplot(
            data=comparison_subset, x="defense", y="selection_rate", hue="attack",
            order=DEFENSE_ORDER, palette=my_palette,
            alpha=0.6, jitter=0.25, size=10, edgecolor='black', linewidth=1.5,
            dodge=True
        )

        sns.boxplot(
            data=comparison_subset, x="defense", y="selection_rate", hue="attack",
            order=DEFENSE_ORDER, palette=my_palette,
            boxprops={'facecolor': 'none', 'edgecolor': 'gray'},
            linewidth=2.5, fliersize=0, zorder=10,
            dodge=True
        )

        sns.despine()

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l in [attack, "Healthy Baseline"] and l not in unique_labels:
                unique_labels[l] = h

        # 2. HORIZONTAL LEGEND (ncol=2) pushed slightly lower to clear labels
        plt.legend(
            handles=unique_labels.values(),
            labels=unique_labels.keys(),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fontsize=16,
            title=None,
            frameon=False,
            columnspacing=1.0 # Pulls the two legend items closer together
        )

        plt.ylabel("Seller Selection Rate", fontweight='bold', fontsize=18)
        plt.xlabel("Defense Mechanism", fontweight='bold', fontsize=18)

        plt.ylim(-0.05, 1.05)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)

        safe_name = attack.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(output_dir / f"2_Manipulation_{safe_name}_vs_Baseline.pdf", bbox_inches='tight')
        plt.close()


def plot_victim_isolation(df, output_dir):
    attacks = [a for a in ATTACK_CATEGORIES["isolation"] if a in df['attack'].unique()]
    if not attacks: return
    print(f"--- Plotting Isolation (Target Focus) ---")

    isolation_df = df[df['attack'].isin(attacks)].copy()

    isolation_df['is_target'] = isolation_df['seller_id'].apply(lambda x: str(x) == str(TARGET_VICTIM_ID))
    isolation_df['Status'] = isolation_df['is_target'].map({True: 'Target', False: 'Others'})

    try:
        isolation_df['id_num'] = isolation_df['seller_id'].apply(lambda x: int(str(x).split('_')[-1]))
    except ValueError:
        isolation_df['id_num'] = isolation_df['seller_id']

    isolation_df = isolation_df.sort_values(by=['id_num'])
    my_palette = {'Target': '#e74c3c', 'Others': '#95a5a6'}

    # 1. SHORTER ASPECT RATIO (height=3.8, aspect=0.79 -> ~6 wide total)
    g = sns.catplot(
        data=isolation_df, x="id_num", y="selection_rate",
        col="defense", col_order=DEFENSE_ORDER,
        hue="Status",
        palette=my_palette,
        kind="bar", height=3.8, aspect=0.79,
        dodge=False, edgecolor="black", linewidth=2.0,
        legend=False
    )

    g.set_axis_labels("Seller ID", "Seller Selection Rate", fontweight='bold', fontsize=18)
    g.set(ylim=(-0.05, 1.05), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for ax in g.axes.flatten():
        title = ax.get_title().replace("defense = ", "")
        ax.set_title(title, fontweight='bold', fontsize=24)

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(14)
            # label.set_rotation(45)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(18)

    target_patch = mpatches.Patch(color=my_palette['Target'], label='Target', ec='black', lw=1.5)
    others_patch = mpatches.Patch(color=my_palette['Others'], label='Others', ec='black', lw=1.5)

    # 2. HORIZONTAL LEGEND (ncol=2) aligned with the right plot
    g.figure.legend(
        handles=[target_patch, others_patch],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=16,
        columnspacing=2.0
    )

    plt.savefig(output_dir / "3_Isolation_VictimCheck.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():
    set_publication_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found or no matching defenses.")
        return

    plot_disruption_impact(df, output_dir)
    plot_manipulation_fairness(df, output_dir)
    plot_victim_isolation(df, output_dir)

    print(f"\nFigures saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()