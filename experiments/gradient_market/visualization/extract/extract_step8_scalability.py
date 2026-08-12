"""
extract_step8_scalability.py — Step 8 (marketplace size sweep).

What step 8 produces (verified against generate_step8_scalability.py):
  CIFAR-100 / cifar100_cnn / image / backdoor (adv_rate=0.3, poison_rate=0.5).
  For each defense in FOCUSED_DEFENSES = {fltrust, martfl}, sweep n_sellers ∈
  {10, 50, 100} (reduced from {10, 30, 50, 100, 500}).

Cell key: (defense, dataset, n_sellers)
Expected: 2 defenses × 3 sizes = 6 cells.

Usage:
    python extract_step8_scalability.py
"""

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
# Declared expected grid (matches generate_step8_scalability.py)
# ---------------------------------------------------------------------------
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100"]
MARKETPLACE_SIZES = [10, 50, 100]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "n_sellers"]


def expected_cells() -> Set[Tuple[str, str, int]]:
    cells = set()
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for n in MARKETPLACE_SIZES:
                cells.add((canonical_name(defense), canonical_name(dataset), int(n)))
    return cells


def key_from_row(row) -> Optional[Tuple[str, str, int]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    n_sellers = row.get("n_sellers")
    if any(v is None for v in (defense, dataset, n_sellers)):
        return None
    try:
        return (canonical_name(defense), canonical_name(dataset), int(n_sellers))
    except (TypeError, ValueError):
        return None


def main():
    ap = make_parser(8, "scalability", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step10_scalability",
                            require_success=True)

    write_step_csv_and_report(
        step_id=8, label="scalability",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
