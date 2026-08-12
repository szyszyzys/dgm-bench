import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_dual_stack_polished"

TARGET_DATASET = "CIFAR-100"

METRICS_TO_PLOT = [
    "selection_rate",             # Participation
    "marginal_contrib_loo",       # Economic Value
    "kernelshap_score",           # Economic Value
    "influence_score"             # Economic Value
]

# --- Use centralized naming, colors, and styling from viz_utils ---
from viz_utils import (
    VALUATION_COLORS, DEFENSE_ORDER,
    fmt as format_label, setup_style as set_publication_style
)

COLOR_BENIGN_PAID = VALUATION_COLORS["benign_paid"]
COLOR_BENIGN_LOST = VALUATION_COLORS["benign_discarded"]
COLOR_ADV_PAID = VALUATION_COLORS["adv_paid"]
COLOR_ADV_CAUGHT = VALUATION_COLORS["adv_blocked"]

# ==========================================
# 2. DATA LOADING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_dual_breakdown(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for '{target_metric}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        defense_name = format_label(info['defense'])
        jsonl_files = list(folder.rglob("valuations.jsonl"))

        benign_paid = 0.0
        benign_discarded = 0.0
        adv_paid = 0.0
        adv_discarded = 0.0
        round_count = 0

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # Mode Selection
                    if target_metric == "selection_rate":
                        has_metric = True
                    else:
                        if not valuations: continue
                        first_val = next(iter(valuations.values()))
                        has_metric = target_metric in first_val

                    if not has_metric: continue

                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        if target_metric == "selection_rate":
                            val = 1.0
                        else:
                            if target_metric in data and data[target_metric] is not None:
                                val = max(0, float(data[target_metric]))
                            else:
                                continue

                        if is_adv:
                            if sid in selected_ids: adv_paid += val
                            else: adv_discarded += val
                        else:
                            if sid in selected_ids: benign_paid += val
                            else: benign_discarded += val

                    round_count += 1
            except: pass

        if round_count > 0:
            records.append({
                "defense": defense_name,
                "Benign_Paid": benign_paid,
                "Benign_Discarded": benign_discarded,
                "Adv_Paid": adv_paid,
                "Adv_Discarded": adv_discarded
            })

    return pd.DataFrame(records)

# ==========================================
# 3. DUAL STACK PLOTTING
# ==========================================

def plot_dual_stack(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty: return

    # Normalize
    df['Total_Benign'] = df['Benign_Paid'] + df['Benign_Discarded']
    df['Total_Adv'] = df['Adv_Paid'] + df['Adv_Discarded']

    df['Total_Benign'] = df['Total_Benign'].replace(0, 1)
    df['Total_Adv'] = df['Total_Adv'].replace(0, 1)

    df['Pct_Benign_Paid'] = (df['Benign_Paid'] / df['Total_Benign']) * 100
    df['Pct_Benign_Discarded'] = (df['Benign_Discarded'] / df['Total_Benign']) * 100
    df['Pct_Adv_Paid'] = (df['Adv_Paid'] / df['Total_Adv']) * 100
    df['Pct_Adv_Discarded'] = (df['Adv_Discarded'] / df['Total_Adv']) * 100

    df = df.set_index("defense")
    existing_order = [d for d in DEFENSE_ORDER if d in df.index]
    df = df.loc[existing_order]

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    # Group 1: Benign
    p1 = ax.bar(x - width/2, df['Pct_Benign_Paid'], width, label='Benign: Paid',
                color=COLOR_BENIGN_PAID, edgecolor='black', linewidth=1.5)
    p2 = ax.bar(x - width/2, df['Pct_Benign_Discarded'], width, bottom=df['Pct_Benign_Paid'],
                label='Benign: Discarded', color=COLOR_BENIGN_LOST, edgecolor='black', linewidth=1.5, hatch='//')

    # Group 2: Adversary
    p3 = ax.bar(x + width/2, df['Pct_Adv_Paid'], width, label='Adversary: Paid',
                color=COLOR_ADV_PAID, edgecolor='black', linewidth=1.5)
    p4 = ax.bar(x + width/2, df['Pct_Adv_Discarded'], width, bottom=df['Pct_Adv_Paid'],
                label='Adversary: Blocked', color=COLOR_ADV_CAUGHT, edgecolor='black', linewidth=1.5, hatch='..')

    # Data Labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 5:
                ax.annotate(f'{height:.0f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha='center', va='center', color='white',
                            fontweight='bold', fontsize=16) # Larger, bold font

    add_labels(p1)
    add_labels(p2)
    add_labels(p3)
    add_labels(p4)

    # Clean Layout
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    # REMOVED TITLE AS REQUESTED
    # ax.set_title(...)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontweight='bold')
    ax.set_ylim(0, 115) # Headroom for 1-row legend

    # 1-ROW LEGEND
    ax.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc='lower center',
        ncol=4,             # Forces 1 row
        frameon=False,
        fontsize=16,
        columnspacing=1.0,  # Tighten spacing to fit 1 row
        handletextpad=0.5
    )

    plt.tight_layout()
    filename = f"Fig_DualStack_{metric_name}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Starting Polished Dual Stack Analysis ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        df = load_dual_breakdown(base_dir, TARGET_DATASET, metric)
        if not df.empty:
            plot_dual_stack(df, metric, output_dir)
        else:
            print(f"  -> Skipping {metric} (No data)")

if __name__ == "__main__":
    main()