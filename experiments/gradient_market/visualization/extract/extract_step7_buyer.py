"""
extract_step7_buyer.py — Step 7 (buyer-side / demand-side attacks).

What step 7 produces (verified against generate_step7_buyer_attacks.py):
  CIFAR-100 / cifar100_cnn / image. Two scenario families:

  Family A — buyer-only attacks (no seller poisoning):
    For each defense in FOCUSED_DEFENSES = {fltrust, martfl}, four buyer attacks:
      - dos
      - starvation
      - class_exclusion_neg
      - oscillating_binary
    → 2 × 4 = 8 cells

  Family B — combined seller + buyer attacks:
    For each defense in IMAGE_DEFENSES (the FULL list, not focused), three
    representative buyer attacks combined with image_backdoor seller attack:
      - dos, starvation, oscillating_binary
    → 12 defenses × 3 = 36 cells (when all enabled)
    Currently only fltrust + martfl are tuned in step 3, so only those land.

Cell key: (defense, dataset, family, buyer_attack)
  family ∈ {"buyer_only", "combined"}

Total expected: 8 (family A) + 36 (family B, full IMAGE_DEFENSES) = 44 cells.
With FOCUSED_DEFENSES only: 8 + 6 = 14 cells.

Usage:
    python extract_step7_buyer.py
"""

import re
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gradient_market.visualization.viz_utils import load_step_results
from experiments.gradient_market.visualization.extract._common import (
    write_step_csv_and_report, make_parser, canonical_name,
)


# ---------------------------------------------------------------------------
# Declared expected grid (matches generate_step7_buyer_attacks.py)
# ---------------------------------------------------------------------------
# fedavg is included because the buyer-only family runs against it as a
# baseline (visible in `find results/step8_buyer_attack_*_fedavg_*`).
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl", "fedavg"]
ENABLED_DATASETS_RAW = ["CIFAR100"]

BUYER_ONLY_ATTACKS = ["dos", "starvation", "class_exclusion_neg", "oscillating_binary"]

# Combined family iterates over IMAGE_DEFENSES, which now includes flame /
# deepsight / bulyan / foolsgold. List explicitly here so the extractor can
# detect drift if config_common_utils.py changes.
COMBINED_DEFENSES_RAW = [
    "fltrust", "martfl", "skymask", "trimmed_mean", "multi_krum",
    "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
]
COMBINED_BUYER_ATTACKS = ["dos", "starvation", "oscillating_binary"]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "family", "buyer_attack"]


def expected_cells() -> Set[Tuple[str, str, str, str]]:
    cells = set()
    # Family A: buyer-only
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for atk in BUYER_ONLY_ATTACKS:
                cells.add((
                    canonical_name(defense),
                    canonical_name(dataset),
                    "buyeronly",
                    canonical_name(atk),
                ))
    # Family B: combined seller + buyer
    for defense in COMBINED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for atk in COMBINED_BUYER_ATTACKS:
                cells.add((
                    canonical_name(defense),
                    canonical_name(dataset),
                    "combined",
                    canonical_name(atk),
                ))
    return cells


# Scenario name patterns. The defense name is at the END of the scenario
# directory (just before the dataset), so we anchor on the known defense
# names — that way multi-word attack tags like "class_exclusion_neg" don't
# get truncated by a non-greedy match.
KNOWN_BUYER_DEFENSES = [
    "fltrust", "martfl", "fedavg",
    "skymask", "skymask_small",
    "trimmed_mean", "multi_krum", "rflpa", "spmc",
    "flame", "deepsight", "bulyan", "foolsgold",
]
_DEFENSES_GROUP = "(" + "|".join(KNOWN_BUYER_DEFENSES) + ")"

# step8_buyer_attack_<attack>_<defense>_<dataset>
RE_BUYER_ONLY = re.compile(
    r"step8_buyer_attack_(.+?)_" + _DEFENSES_GROUP +
    r"_(?:CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)"
)
# step8b_combined_backdoor_<attack>_<defense>_<dataset>
RE_COMBINED = re.compile(
    r"step8b_combined_backdoor_(.+?)_" + _DEFENSES_GROUP +
    r"_(?:CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)"
)


def key_from_row(row) -> Optional[Tuple[str, str, str, str]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", ""))

    m = RE_COMBINED.search(run_dir)
    if m:
        attack_tag = m.group(1)
        return (canonical_name(defense), canonical_name(dataset),
                "combined", canonical_name(attack_tag))

    m = RE_BUYER_ONLY.search(run_dir)
    if m:
        attack_tag = m.group(1)
        return (canonical_name(defense), canonical_name(dataset),
                "buyeronly", canonical_name(attack_tag))

    return None


def main():
    ap = make_parser(7, "buyer", description=__doc__)
    args = ap.parse_args()

    # Both family A and family B sit under different prefixes; load them all.
    df_a = load_step_results(args.results_dir, "step8_buyer_attack",
                              require_success=True)
    df_b = load_step_results(args.results_dir, "step8b_combined",
                              require_success=True)

    import pandas as pd
    df = pd.concat([df_a, df_b], ignore_index=True) if not df_b.empty else df_a

    write_step_csv_and_report(
        step_id=7, label="buyer",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
