"""
extract_step6_adaptive.py — Step 6 (adaptive attack threat models).

What step 6 produces (verified against generate_step6_adaptive_attack.py):
  CIFAR-100 / cifar100_cnn / image.
  For each defense in FOCUSED_DEFENSES = {fltrust, martfl}:
    - One baseline_no_attack scenario (adv_rate=0.0)
    - For each (threat_model, adaptive_mode) pair:
        threat_models = {black_box, oracle}
        adaptive_modes = {gradient_manipulation}
      → 2 adaptive scenarios per defense

Cell key: (defense, dataset, scenario_kind)
  scenario_kind ∈ {"baseline_no_attack",
                   "adaptive_black_box_gradient_manipulation",
                   "adaptive_oracle_gradient_manipulation"}

Expected per defense: 1 baseline + 2 adaptive = 3 cells. Total: 2 × 3 = 6 cells.

Note: scenario_kind is parsed from the run_dir path, since viz_utils does not
populate a dedicated "threat_model" column.

Usage:
    python extract_step6_adaptive.py
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
# Declared expected grid (matches generate_step6_adaptive_attack.py)
# ---------------------------------------------------------------------------
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100"]

THREAT_MODELS = ["black_box", "oracle"]
# Generator's docstring claims only gradient_manipulation, but data_poisoning
# runs are present on disk. Include both so they're not flagged as unexpected.
ADAPTIVE_MODES = ["gradient_manipulation", "data_poisoning"]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "scenario_kind"]


def expected_cells() -> Set[Tuple[str, str, str]]:
    cells = set()
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            d, ds = canonical_name(defense), canonical_name(dataset)
            # Baseline (no attack)
            cells.add((d, ds, "baselinenoattack"))
            # Adaptive (threat × mode). Generator explicitly skips
            #   `oracle × data_poisoning` and `gradient_inversion × data_poisoning`
            # via:
            #   if threat_model != "black_box" and adaptive_mode == "data_poisoning":
            #       continue
            for tm in THREAT_MODELS:
                for am in ADAPTIVE_MODES:
                    if am == "data_poisoning" and tm != "black_box":
                        continue
                    cells.add((d, ds, canonical_name(f"adaptive_{tm}_{am}")))
    return cells


# Match scenario directory name to figure out which kind of step 6 run it is.
# step7_baseline_no_attack_<defense>_<dataset>
# step7_adaptive_<threat_model>_<adaptive_mode>_<defense>_<dataset>
RE_BASELINE = re.compile(r"step7_baseline_no_attack")
RE_ADAPTIVE = re.compile(r"step7_adaptive_(black_box|oracle|gradient_inversion)_"
                          r"(gradient_manipulation|data_poisoning)")


def key_from_row(row) -> Optional[Tuple[str, str, str]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", ""))

    if RE_BASELINE.search(run_dir):
        kind = "baselinenoattack"
    else:
        m = RE_ADAPTIVE.search(run_dir)
        if not m:
            return None
        kind = canonical_name(f"adaptive_{m.group(1)}_{m.group(2)}")

    return (canonical_name(defense), canonical_name(dataset), kind)


def main():
    ap = make_parser(6, "adaptive", description=__doc__)
    args = ap.parse_args()

    # Two prefixes: adaptive scenarios + baseline scenarios.
    # load_step_results matches by prefix, so use the shared root "step7_".
    df = load_step_results(args.results_dir, "step7_", require_success=True)

    write_step_csv_and_report(
        step_id=6, label="adaptive",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
