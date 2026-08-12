"""
extract_step14_collusion.py — Step 14 (MartFL collusion stress test).

What step 14 produces (verified against generate_step14_martfl_collusion.py
AND the actual on-disk directory listing):

  Current format:
    step14_collusion_random_martfl_<dataset>/
      For each dataset in ENABLED_DATASETS, martfl with the "random"
      collusion mode. Defense is hardcoded to martfl in the generator.
      Currently on disk for 6 datasets (CIFAR10, CIFAR100, FEMNIST,
      Purchase100, Texas100, TREC) — older runs from before the dataset
      filter was added.

  Legacy format (kept for backward-compat detection):
    step14_collusion_collusion_<defense>/adv_0.3_mode_<mode>/
      Older naming where the dataset was implicit (CIFAR100 only) and
      multiple defenses were tested. These are leftovers — the current
      generator no longer emits them. They show up as 'unexpected' cells.

Cell key: (defense, dataset, collusion_mode)

Expected (current scope, after ENABLED_DATASETS filter):
  martfl × {CIFAR100, Texas100, TREC} × {random} = 3 cells.

Usage:
    python extract_step14_collusion.py
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
# Declared expected grid (matches generate_step14_martfl_collusion.py
# AFTER the ENABLED_DATASETS filter is applied)
# ---------------------------------------------------------------------------
COLLUSION_DEFENSES_RAW = ["martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100", "Texas100", "TREC"]
COLLUSION_MODES = ["random"]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "collusion_mode"]


def expected_cells() -> Set[Tuple[str, str, str]]:
    cells = set()
    for defense in COLLUSION_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for mode in COLLUSION_MODES:
                cells.add((canonical_name(defense),
                           canonical_name(dataset),
                           canonical_name(mode)))
    return cells


# Current format: step14_collusion_<mode>_<defense>_<dataset>
RE_CURRENT = re.compile(
    r"step14_collusion_(random|inverse)_(martfl|fltrust|fedavg|skymask)_"
    r"(CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)"
)
# Legacy format: step14_collusion_collusion_<defense>/adv_0.3_mode_<mode>
RE_LEGACY = re.compile(
    r"step14_collusion_collusion_(martfl|fltrust|fedavg|skymask).*?mode_(random|inverse)"
)


def key_from_row(row) -> Optional[Tuple[str, str, str]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", "")).replace("\\", "/")

    # Try current format first
    m = RE_CURRENT.search(run_dir)
    if m:
        mode = m.group(1)
        return (canonical_name(defense), canonical_name(dataset),
                canonical_name(mode))

    # Fall back to legacy format (will likely be flagged as "unexpected"
    # because the current expected grid only includes the current format)
    m = RE_LEGACY.search(run_dir)
    if m:
        mode = m.group(2)
        return (canonical_name(defense), canonical_name(dataset),
                canonical_name(mode))

    return None


def main():
    ap = make_parser(14, "collusion", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step14_collusion",
                            require_success=True)

    write_step_csv_and_report(
        step_id=14, label="collusion",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
