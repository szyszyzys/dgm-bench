"""
extract_step10_main_summary.py — Step 10 (headline multi-modal benchmark).

What step 10 produces (verified against generate_step10_main_summary.py):
  For each (modality, dataset, model) target in MAIN_SUMMARY_TARGETS ∩
  ENABLED_DATASETS, every defense in IMAGE_DEFENSES (image) or
  TEXT_TABULAR_DEFENSES (text/tabular). Fixed backdoor attack
  (adv_rate=0.3, poison_rate=0.5).

Cell key: (defense, dataset)

Skymask is image-only (it's not in TEXT_TABULAR_DEFENSES).

Expected (current ENABLED_DATASETS = {CIFAR100, Texas100, TREC}):
  CIFAR100 (image):       11 IMAGE_DEFENSES
  Texas100 (tabular):     10 TEXT_TABULAR_DEFENSES (no skymask)
  TREC     (text):        10 TEXT_TABULAR_DEFENSES (no skymask)
Total: 31 cells.

This is the headline table of the paper. Strict-mode 31/31 = paper-ready.

Usage:
    python extract_step10_main_summary.py
    python extract_step10_main_summary.py --strict
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
# Declared expected grid (matches generate_step10_main_summary.py)
# ---------------------------------------------------------------------------
ENABLED_DATASETS_RAW = ["CIFAR100", "Texas100", "TREC"]

# Match config_common_utils.py post-filter
IMAGE_DEFENSES_RAW = [
    "fltrust", "martfl", "skymask", "trimmed_mean", "multi_krum",
    "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
]
TEXT_TABULAR_DEFENSES_RAW = [
    "fltrust", "martfl", "trimmed_mean", "multi_krum",
    "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
]

DATASET_MODALITY = {
    "CIFAR100":    "image",
    "Texas100":    "tabular",
    "TREC":        "text",
}

# Bumped from 2 to 3 to match the seed override in
# generate_step10_main_summary.py. Existing seed_42 + seed_43 cells stay;
# the rerun adds seed_44 to every (defense, dataset) pair.
NUM_SEEDS_PER_CONFIG = 3

CELL_COLUMNS = ["defense", "dataset"]


def expected_cells() -> Set[Tuple[str, str]]:
    cells = set()
    for dataset in ENABLED_DATASETS_RAW:
        modality = DATASET_MODALITY[dataset]
        defenses = (IMAGE_DEFENSES_RAW if modality == "image"
                    else TEXT_TABULAR_DEFENSES_RAW)
        for defense in defenses:
            cells.add((canonical_name(defense), canonical_name(dataset)))
    return cells


def key_from_row(row) -> Optional[Tuple[str, str]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None
    if not isinstance(defense, str) or not isinstance(dataset, str):
        return None
    return (canonical_name(defense), canonical_name(dataset))


def main():
    ap = make_parser(10, "main_summary", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step12_main_summary",
                            require_success=True)

    write_step_csv_and_report(
        step_id=10, label="main_summary",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
