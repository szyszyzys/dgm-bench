"""
extract_step3_defense_tune.py — Step 3 (defense hyperparameter tuning).

What step 3 produces (verified against generate_step3_defense_tuning.py):
  For every (defense, modality, dataset, model) combination defined by
  TUNING_TARGETS_STEP3 ∩ ENABLED_DATASETS, swept across the defense's HP grid
  (TUNING_GRIDS) under each ENABLED_ATTACK_TYPE, with NUM_SEEDS_PER_CONFIG
  seeds per HP cell.

  - fedavg is excluded (it has no defense to tune).
  - skymask / skymask_small are image-only.
  - Currently only "backdoor" is in ENABLED_ATTACK_TYPES (labelflip disabled).

Readiness criterion (loose):
  Every (defense, dataset) combination in the planned grid has at least one
  successful run.

Readiness criterion (strict):
  Every (defense, dataset) combination has at least NUM_SEEDS_PER_CONFIG (=2)
  successful runs.

Usage:
    python extract_step3_defense_tune.py
    python extract_step3_defense_tune.py --strict
    python extract_step3_defense_tune.py --json
"""

import sys
from pathlib import Path
from typing import Optional, Set, Tuple

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gradient_market.visualization.viz_utils import load_step_results
from experiments.gradient_market.visualization.extract._common import (
    write_step_csv_and_report, make_parser, canonical_name,
)


# ---------------------------------------------------------------------------
# Declared expected grid (matches generate_step3_defense_tuning.py)
# ---------------------------------------------------------------------------
# Currently enabled datasets (subset of all 5 supported by the generator).
# FEMNIST and Purchase100 are filtered out at generation time via
# config_common_utils.ENABLED_DATASETS.
ENABLED_DATASETS_RAW = ["CIFAR100", "Texas100", "TREC"]

# Defenses that step 3 actually tunes (fedavg is excluded by the generator).
# These match TUNING_GRIDS in generate_step3_defense_tuning.py and are filtered
# down to ENABLED_DEFENSES in config_common_utils.py.
TUNED_DEFENSES_RAW = [
    "fltrust", "martfl",
    "skymask",                       # image-only
    "trimmed_mean", "multi_krum",
    "rflpa", "spmc",
    "flame", "deepsight",            # new: image + text/tabular
    "bulyan", "foolsgold",
    # "skymask_small" is in TUNING_GRIDS but disabled in ENABLED_DEFENSES.
]

# Modality scoping per defense. skymask is image-only.
IMAGE_ONLY_DEFENSES = {"skymask", "skymask_small"}

# Dataset → modality lookup (so we can skip image-only defenses on text/tabular)
DATASET_MODALITY = {
    "CIFAR100":    "image",
    "Texas100":    "tabular",
    "Purchase100": "tabular",
    "FEMNIST":     "image",
    "TREC":        "text",
}

NUM_SEEDS_PER_CONFIG = 2


# ---------------------------------------------------------------------------
# Cell shape: (defense, dataset)  — both fmt-normalized
# ---------------------------------------------------------------------------
CELL_COLUMNS = ["defense", "dataset", "model", "attack"]


def expected_cells() -> Set[Tuple[str, str]]:
    """Build the set of (defense, dataset) cells that step 3 should produce.

    Keys are canonical-form (lowercase, alphanumeric only) so they survive
    inconsistent name formatting between the config files and viz_utils.fmt().
    """
    cells = set()
    for defense in TUNED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            modality = DATASET_MODALITY[dataset]
            if defense in IMAGE_ONLY_DEFENSES and modality != "image":
                continue
            cells.add((canonical_name(defense), canonical_name(dataset)))
    return cells


def key_from_row(row) -> Optional[Tuple[str, str]]:
    """Extract a (defense, dataset) cell-key from a DataFrame row.

    Returns the same canonical form as expected_cells() so matching works
    regardless of how the underlying viz_utils.fmt() rendered the strings.
    """
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None
    if not isinstance(defense, str) or not isinstance(dataset, str):
        return None
    return (canonical_name(defense), canonical_name(dataset))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = make_parser(3, "defense_tune", description=__doc__)
    args = ap.parse_args()

    # load_step_results filters on .success markers, so partial-state runs
    # are correctly ignored without any extra work here.
    df = load_step_results(args.results_dir, "step3_tune", require_success=True)

    write_step_csv_and_report(
        step_id=3,
        label="defense_tune",
        df=df,
        output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict,
        write_json=args.json,
    )


if __name__ == "__main__":
    main()
