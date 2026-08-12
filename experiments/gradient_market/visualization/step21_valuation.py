"""
step21_valuation.py — Dedicated marketplace-fairness figures for the Step 21
valuation analysis (results/step21_valuation_*/).

Complements step10_valuation.py (which targets step12_main_summary_*).
This script differs from that one by:

  1. Reading from step21_valuation_* scenario folders.
  2. Aggregating across seeds and drawing std error bars on the Paid bars.
  3. Tracking negative valuations instead of silently clipping them. Default
     behavior still plots gross-positive value (to preserve the visual); the
     console logs how much negative value was present, and --use-signed
     switches to signed sums.
  4. Printing absolute benign/adv totals under each bar pair so reviewers
     can see that the two group percentages are not commensurable.
  5. --warmup-frac replaces the hard-coded 0.5.
  6. Bare `except` replaced with named exceptions so parsing bugs surface.

Usage:
    python experiments/gradient_market/visualization/step21_valuation.py
    python experiments/gradient_market/visualization/step21_valuation.py --dataset CIFAR-100
    python experiments/gradient_market/visualization/step21_valuation.py --use-signed
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step21_valuation"

TARGET_DATASET = "CIFAR-100"
SCENARIO_PREFIX = "step21_valuation"

METRICS_TO_PLOT = [
    "selection_rate",             # Participation
    "marginal_contrib_loo",       # Economic Value
    "kernelshap_score",           # Economic Value
    "influence_score",            # Economic Value
]

from viz_utils import (
    VALUATION_COLORS, DEFENSE_ORDER,
    fmt as format_label, setup_style as set_publication_style,
)

COLOR_BENIGN_PAID = VALUATION_COLORS["benign_paid"]
COLOR_BENIGN_LOST = VALUATION_COLORS["benign_discarded"]
COLOR_ADV_PAID = VALUATION_COLORS["adv_paid"]
COLOR_ADV_CAUGHT = VALUATION_COLORS["adv_blocked"]

SEED_RE = re.compile(r"run_\d+_seed_(\d+)")


# ==========================================
# 2. PARSING HELPERS
# ==========================================
def parse_scenario_name(scenario_name: str, prefix: str) -> Dict[str, str]:
    """Extract (defense, dataset) from 'step21_valuation_<defense>_<dataset>'.

    Defense names can contain underscores (multi_krum, trimmed_mean), so we
    rpartition on the final underscore: dataset is the trailing token.
    """
    if not scenario_name.startswith(prefix):
        return {"defense": "unknown", "dataset": "unknown"}
    tail = scenario_name[len(prefix):].lstrip("_")
    if "_" not in tail:
        return {"defense": tail, "dataset": "unknown"}
    defense, _, dataset = tail.rpartition("_")
    return {"defense": defense, "dataset": dataset}


def extract_seed(path: Path) -> str:
    for part in path.parts:
        m = SEED_RE.match(part)
        if m:
            return m.group(1)
    return "?"


# ==========================================
# 3. DATA LOADING
# ==========================================
def load_dual_breakdown(
    base_dir: Path,
    dataset: str,
    target_metric: str,
    prefix: str,
    warmup_frac: float,
    use_signed: bool,
) -> pd.DataFrame:
    """One row per (defense, seed). Plotter averages across seeds."""
    per_seed: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(
        lambda: {"benign_paid": 0.0, "benign_discarded": 0.0,
                 "adv_paid": 0.0, "adv_discarded": 0.0,
                 "neg_value_clipped": 0.0, "rounds": 0}
    ))

    scenario_folders = list(base_dir.glob(f"{prefix}*"))
    print(f"[{target_metric}] scanning {len(scenario_folders)} scenario(s) "
          f"(dataset={dataset}, warmup_frac={warmup_frac}, signed={use_signed})")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name, prefix)
        if dataset.lower().replace("-", "") not in info["dataset"].lower().replace("-", ""):
            continue
        defense_name = format_label(info["defense"])

        for j_file in folder.rglob("valuations.jsonl"):
            seed = extract_seed(j_file)
            try:
                with open(j_file) as f:
                    lines = f.readlines()
            except OSError as e:
                print(f"  [warn] cannot read {j_file}: {e}")
                continue

            start_idx = max(0, int(len(lines) * warmup_frac))
            acc = per_seed[defense_name][seed]

            for line in lines[start_idx:]:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [warn] bad JSONL line in {j_file}: {e}")
                    continue

                selected_ids = set(rec.get("selected_ids", []))
                valuations = rec.get("seller_valuations", {})

                if target_metric != "selection_rate":
                    if not valuations:
                        continue
                    first_val = next(iter(valuations.values()))
                    if target_metric not in first_val:
                        continue

                for sid, data in valuations.items():
                    is_adv = str(sid).startswith("adv")
                    if target_metric == "selection_rate":
                        val = 1.0
                    else:
                        raw = data.get(target_metric)
                        if raw is None:
                            continue
                        try:
                            raw = float(raw)
                        except (TypeError, ValueError):
                            continue
                        if use_signed:
                            val = raw
                        else:
                            if raw < 0:
                                acc["neg_value_clipped"] += -raw
                            val = max(0.0, raw)

                    if is_adv:
                        if sid in selected_ids:
                            acc["adv_paid"] += val
                        else:
                            acc["adv_discarded"] += val
                    else:
                        if sid in selected_ids:
                            acc["benign_paid"] += val
                        else:
                            acc["benign_discarded"] += val
                acc["rounds"] += 1

    records = []
    for defense_name, seeds in per_seed.items():
        for seed, acc in seeds.items():
            if acc["rounds"] == 0:
                continue
            records.append({
                "defense": defense_name,
                "seed": seed,
                "Benign_Paid": acc["benign_paid"],
                "Benign_Discarded": acc["benign_discarded"],
                "Adv_Paid": acc["adv_paid"],
                "Adv_Discarded": acc["adv_discarded"],
                "neg_value_clipped": acc["neg_value_clipped"],
                "rounds": acc["rounds"],
            })
    return pd.DataFrame(records)


# ==========================================
# 4. PLOTTING
# ==========================================
def plot_dual_stack(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty:
        return

    print(f"\n--- {metric_name} ---")
    n_seeds = df.groupby("defense")["seed"].nunique()
    clipped_total = df.groupby("defense")["neg_value_clipped"].sum()
    for d in n_seeds.index:
        clip = clipped_total.get(d, 0.0)
        msg = f"  | clipped negative value: {clip:.2f}" if clip > 0 else ""
        print(f"  {d:<20} seeds={n_seeds[d]}{msg}")

    sum_cols = ["Benign_Paid", "Benign_Discarded", "Adv_Paid", "Adv_Discarded"]
    agg_mean = df.groupby("defense")[sum_cols].mean()
    agg_std = df.groupby("defense")[sum_cols].std().fillna(0.0)

    total_benign = (agg_mean["Benign_Paid"] + agg_mean["Benign_Discarded"]).replace(0, 1)
    total_adv = (agg_mean["Adv_Paid"] + agg_mean["Adv_Discarded"]).replace(0, 1)

    plot_df = pd.DataFrame({
        "Pct_Benign_Paid": agg_mean["Benign_Paid"] / total_benign * 100,
        "Pct_Benign_Discarded": agg_mean["Benign_Discarded"] / total_benign * 100,
        "Pct_Adv_Paid": agg_mean["Adv_Paid"] / total_adv * 100,
        "Pct_Adv_Discarded": agg_mean["Adv_Discarded"] / total_adv * 100,
        "Err_Benign_Paid": agg_std["Benign_Paid"] / total_benign * 100,
        "Err_Adv_Paid": agg_std["Adv_Paid"] / total_adv * 100,
        "Total_Benign": total_benign,
        "Total_Adv": total_adv,
    })

    existing_order = [d for d in DEFENSE_ORDER if d in plot_df.index]
    plot_df = plot_df.loc[existing_order]
    if plot_df.empty:
        print(f"  -> no defenses in DEFENSE_ORDER overlap with data for {metric_name}")
        return

    print("  absolute group totals (mean across seeds):")
    for d in plot_df.index:
        b = plot_df.loc[d, "Total_Benign"]
        a = plot_df.loc[d, "Total_Adv"]
        ratio = a / b if b > 0 else float("inf")
        print(f"    {d:<20} benign={b:>10.2f}  adv={a:>10.2f}  adv/benign={ratio:.3f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.35

    p1 = ax.bar(x - width / 2, plot_df["Pct_Benign_Paid"], width,
                label="Benign: Paid",
                color=COLOR_BENIGN_PAID, edgecolor="black", linewidth=1.5,
                yerr=plot_df["Err_Benign_Paid"], capsize=3,
                error_kw={"elinewidth": 1, "ecolor": "black"})
    p2 = ax.bar(x - width / 2, plot_df["Pct_Benign_Discarded"], width,
                bottom=plot_df["Pct_Benign_Paid"],
                label="Benign: Discarded",
                color=COLOR_BENIGN_LOST, edgecolor="black", linewidth=1.5, hatch="//")
    p3 = ax.bar(x + width / 2, plot_df["Pct_Adv_Paid"], width,
                label="Adversary: Paid",
                color=COLOR_ADV_PAID, edgecolor="black", linewidth=1.5,
                yerr=plot_df["Err_Adv_Paid"], capsize=3,
                error_kw={"elinewidth": 1, "ecolor": "black"})
    p4 = ax.bar(x + width / 2, plot_df["Pct_Adv_Discarded"], width,
                bottom=plot_df["Pct_Adv_Paid"],
                label="Adversary: Blocked",
                color=COLOR_ADV_CAUGHT, edgecolor="black", linewidth=1.5, hatch="..")

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 5:
                ax.annotate(
                    f"{height:.0f}%",
                    xy=(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2),
                    ha="center", va="center", color="white",
                    fontweight="bold", fontsize=16,
                )

    for rects in (p1, p2, p3, p4):
        add_labels(rects)

    for i, d in enumerate(plot_df.index):
        b = plot_df.loc[d, "Total_Benign"]
        a = plot_df.loc[d, "Total_Adv"]
        ax.annotate(f"Σb={b:.1f}", xy=(i - width / 2, 102),
                    ha="center", va="bottom", fontsize=9, color="dimgray")
        ax.annotate(f"Σa={a:.1f}", xy=(i + width / 2, 102),
                    ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(
        bbox_to_anchor=(0.5, 1.06), loc="lower center", ncol=4,
        frameon=False, fontsize=14, columnspacing=1.0, handletextpad=0.5,
    )

    plt.tight_layout()
    save_path = output_dir / f"Fig_DualStack_{metric_name}.pdf"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  saved: {save_path}")
    plt.close()


# ==========================================
# 5. MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=BASE_RESULTS_DIR)
    ap.add_argument("--output-dir", default=FIGURE_OUTPUT_DIR)
    ap.add_argument("--dataset", default=TARGET_DATASET)
    ap.add_argument("--prefix", default=SCENARIO_PREFIX,
                    help="Scenario folder prefix (default: step21_valuation)")
    ap.add_argument("--warmup-frac", type=float, default=0.5,
                    help="Fraction of leading rounds to skip as warmup (default 0.5)")
    ap.add_argument("--use-signed", action="store_true",
                    help="Use signed valuations (negative SHAP/IF subtract). "
                         "Default keeps gross-positive view but logs clipped magnitude.")
    args = ap.parse_args()

    set_publication_style()
    base_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        df = load_dual_breakdown(
            base_dir, args.dataset, metric,
            prefix=args.prefix,
            warmup_frac=args.warmup_frac,
            use_signed=args.use_signed,
        )
        if not df.empty:
            plot_dual_stack(df, metric, output_dir)
        else:
            print(f"  -> Skipping {metric} (no data)")


if __name__ == "__main__":
    main()
