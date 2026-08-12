"""
extract_step15_pricing.py — Step 15 (proportional pricing under three payment models).

What step 15 produces (verified against generate_step15_proportional_pricing.py):
  CIFAR-100 / cifar100_cnn / image. For each defense in IMAGE_DEFENSES, three
  payment models are tested: binary, proportional, quality_based. The same
  experiment is run with each payment model so that attacker revenue capture
  can be compared across pricing schemes.

Cell key: (defense, dataset, payment_model)
Expected: 11 IMAGE_DEFENSES × 1 dataset × 3 payment_models = 33 cells.

Note: in practice the runs only land for fltrust + martfl right now because
the other defenses skip when their tuned params are missing. If your
verify_all shows 18 successes (2 defenses × 3 payment models × ~3 seeds),
that's the current realistic floor.

Usage:
    python extract_step15_pricing.py
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
# Declared expected grid (matches generate_step15_proportional_pricing.py)
# ---------------------------------------------------------------------------
# Same two-tier scoping as step 17 / 7: declared = generator's IMAGE_DEFENSES,
# realistic = only those with tuned params. Default to realistic.
DECLARED_PRICING_DEFENSES_RAW = [
    "fltrust", "martfl", "skymask", "trimmed_mean", "multi_krum",
    "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
]
REALISTIC_PRICING_DEFENSES_RAW = ["fltrust", "martfl"]
PRICING_DEFENSES_RAW = REALISTIC_PRICING_DEFENSES_RAW

ENABLED_DATASETS_RAW = ["CIFAR100"]
PAYMENT_MODELS = ["binary", "proportional", "quality_based"]

# Bumped from 2 to 3 to match the seed override in
# generate_step15_proportional_pricing.py.
NUM_SEEDS_PER_CONFIG = 3

CELL_COLUMNS = ["defense", "dataset", "payment_model"]


def expected_cells() -> Set[Tuple[str, str, str]]:
    cells = set()
    for defense in PRICING_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for pm in PAYMENT_MODELS:
                cells.add((canonical_name(defense),
                           canonical_name(dataset),
                           canonical_name(pm)))
    return cells


# Path encoding (verified by listing the cell dirs):
#   results/step15_pricing_<defense>_<dataset>/payment_<model>/<cell>
# where <model> ∈ {binary, proportional, quality_based}. The non-greedy
# capture stops at the next "/" so multi-word model names like "quality_based"
# don't get truncated.
RE_PAYMENT = re.compile(r"/payment_([a-z_]+?)(?:/|$)")


def key_from_row(row) -> Optional[Tuple[str, str, str]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", "")).replace("\\", "/")
    m = RE_PAYMENT.search(run_dir)
    if not m:
        return None
    payment = m.group(1)

    return (canonical_name(defense), canonical_name(dataset),
            canonical_name(payment))


def main():
    ap = make_parser(15, "pricing", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step15_pricing",
                            require_success=True)

    write_step_csv_and_report(
        step_id=15, label="pricing",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
