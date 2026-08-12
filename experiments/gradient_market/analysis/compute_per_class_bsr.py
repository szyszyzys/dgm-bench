"""
compute_per_class_bsr.py — Post-hoc per-class fairness analysis for Step 7
                          (buyer-side / demand-side attacks).

Buyer attacks like `class_exclusion_neg` and `starvation` work by
under-selecting honest sellers that hold a target class. The aggregate
benign-selection-rate (BSR) hides this — even if the buyer attacks one
class, the overall average can look fine.

This script joins each benign seller's selection_rate with the *dominant
class* of the seller's data shard, then groups by (target_class, attack_type)
to surface the per-class fairness gap. The output CSV is the data backing
for the buyer-attack story in the paper.

No rerun needed — walks marketplace_report.json files that already exist
under results/step8_buyer_attack_*/.

Usage:
    python experiments/gradient_market/analysis/compute_per_class_bsr.py
    python experiments/gradient_market/analysis/compute_per_class_bsr.py \
        --results_dir ./results --output ./analysis_partial/step7_per_class_bsr.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario name parsing — same defense whitelist as the extractors
# ---------------------------------------------------------------------------
KNOWN_DEFENSES = [
    "skymask_small", "trimmed_mean", "multi_krum",
    "fedavg", "fltrust", "martfl", "skymask", "rflpa", "spmc",
    "flame", "deepsight", "bulyan", "foolsgold",
]
_DEFENSES_RE = "|".join(KNOWN_DEFENSES)

# step8_buyer_attack_<attack>_<defense>_<dataset>
RE_BUYER_ONLY = re.compile(
    r"step8_buyer_attack_(.+?)_(" + _DEFENSES_RE +
    r")_(?:CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)"
)
# step8b_combined_backdoor_<attack>_<defense>_<dataset>
RE_COMBINED = re.compile(
    r"step8b_combined_backdoor_(.+?)_(" + _DEFENSES_RE +
    r")_(?:CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)"
)


def parse_scenario(report_path: Path) -> Optional[Dict]:
    """Pull (family, attack, defense, dataset) from the result path."""
    s = str(report_path).replace("\\", "/")
    m = RE_COMBINED.search(s)
    if m:
        family, attack, defense = "combined", m.group(1), m.group(2)
    else:
        m = RE_BUYER_ONLY.search(s)
        if not m:
            return None
        family, attack, defense = "buyer_only", m.group(1), m.group(2)

    dataset = None
    for ds in ("CIFAR100", "CIFAR10", "Texas100", "Purchase100", "FEMNIST", "TREC"):
        if ds in s:
            dataset = ds
            break

    return {
        "family":  family,
        "attack":  attack,
        "defense": defense,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# Per-seller record extraction
# ---------------------------------------------------------------------------
def extract_per_seller_records(report_path: Path) -> List[Dict]:
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        print(f"  ⚠ {report_path}: {exc}")
        return []

    meta = parse_scenario(report_path)
    if meta is None:
        return []

    sellers = report.get("seller_summaries", {})
    records: List[Dict] = []
    for sid, summary in sellers.items():
        if summary.get("type") != "benign":
            continue

        # The dominant class field may not exist yet — fall back gracefully.
        dominant_class = (
            summary.get("dominant_class")
            or summary.get("majority_class")
            or summary.get("primary_class")
        )

        # If the seller summary has a per-class distribution, infer the
        # dominant class from it (more reliable than a single field).
        class_dist = (summary.get("class_distribution") or
                      summary.get("label_distribution") or
                      summary.get("class_counts"))
        if dominant_class is None and isinstance(class_dist, dict) and class_dist:
            try:
                dominant_class = max(class_dist.items(),
                                       key=lambda kv: float(kv[1]))[0]
            except Exception:
                dominant_class = None

        records.append({
            "run_dir":         str(report_path.parent),
            "family":          meta["family"],
            "attack":          meta["attack"],
            "defense":         meta["defense"],
            "dataset":         meta["dataset"],
            "seller_id":       sid,
            "selection_rate":  float(summary.get("selection_rate", 0)),
            "n_samples":       int(summary.get("n_samples", 0)) if summary.get("n_samples") is not None else None,
            "dominant_class":  dominant_class,
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="./results",
                    help="Root experiment results directory")
    ap.add_argument("--output", default="./analysis_partial/step7_per_class_bsr.csv",
                    help="Path to write the per-seller CSV")
    ap.add_argument("--summary", default="./analysis_partial/step7_per_class_bsr_summary.csv",
                    help="Path to write the per-(attack, defense, target_class) aggregate CSV")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results dir not found: {results_dir.resolve()}")
        sys.exit(1)

    print(f"📂 Scanning {results_dir.resolve()}")
    paths = sorted(
        list(results_dir.glob("step8_buyer_attack_*/**/marketplace_report.json"))
        + list(results_dir.glob("step8b_combined_*/**/marketplace_report.json"))
    )
    print(f"Found {len(paths)} step 7 marketplace reports")

    if not paths:
        print("  (no Step 7 marketplace reports found)")
        return

    all_records: List[Dict] = []
    for p in paths:
        all_records.extend(extract_per_seller_records(p))

    if not all_records:
        print("  ⚠ No per-seller records produced.")
        return

    df = pd.DataFrame(all_records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"📝 Wrote {len(df)} per-seller records → {args.output}")

    # How many records actually have a dominant class?
    with_class = df[df["dominant_class"].notna()]
    n_total = len(df)
    n_with = len(with_class)
    print(f"   {n_with}/{n_total} records ({100*n_with/max(n_total,1):.0f}%) "
          f"have a usable dominant_class field.")

    if n_with == 0:
        print()
        print("  ⚠ marketplace_report.json does not yet record per-seller class labels.")
        print("  The per-class BSR aggregate cannot be computed yet. To enable it,")
        print("  modify the seller-init code in run_exp.py:_init_sellers_for_clf to")
        print("  record each seller's dominant_class (or full class_distribution) in")
        print("  the seller_summaries dict that gets serialized into")
        print("  marketplace_report.json.")
        print()
        print("  In the meantime, the per-seller CSV at the path above is still")
        print("  useful: it gives you per-(attack, defense, seller_id) selection")
        print("  rates that you can join with seller→class info from the configs.")
        return

    # Aggregate: average benign-selection-rate per (attack, defense, dominant_class)
    agg = (with_class
           .groupby(["family", "attack", "defense", "dataset", "dominant_class"],
                    as_index=False)
           .agg(n_sellers=("seller_id", "count"),
                bsr_mean=("selection_rate", "mean"),
                bsr_min=("selection_rate", "min"),
                bsr_max=("selection_rate", "max")))
    agg.to_csv(args.summary, index=False)
    print(f"📝 Wrote {len(agg)} aggregates → {args.summary}")

    # Sanity print: which (attack, defense) cells show the largest gap
    # between the most-favored and least-favored class?
    if len(agg) > 0:
        gaps = (agg.groupby(["family", "attack", "defense"], as_index=False)
                  .agg(min_bsr=("bsr_mean", "min"),
                       max_bsr=("bsr_mean", "max")))
        gaps["gap"] = gaps["max_bsr"] - gaps["min_bsr"]
        top_gaps = gaps.nlargest(10, "gap")

        print()
        print("=" * 72)
        print(" LARGEST PER-CLASS BSR GAPS (top 10)")
        print("=" * 72)
        print(f"  {'family':<12}{'attack':<22}{'defense':<14}"
              f"{'min':>8}{'max':>8}{'gap':>8}")
        print("  " + "-" * 70)
        for _, row in top_gaps.iterrows():
            print(f"  {str(row['family']):<12}{str(row['attack']):<22}"
                  f"{str(row['defense']):<14}"
                  f"{row['min_bsr']:>8.3f}{row['max_bsr']:>8.3f}"
                  f"{row['gap']:>8.3f}")


if __name__ == "__main__":
    main()
