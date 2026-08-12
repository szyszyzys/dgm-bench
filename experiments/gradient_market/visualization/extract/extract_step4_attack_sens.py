"""
extract_step4_attack_sens.py — Step 4 (attack sensitivity sweep).

What step 4 produces (verified against generate_step4_attack_sensitivity.py):
  CIFAR-100 / cifar100_cnn / image / backdoor.
  For each defense in FOCUSED_DEFENSES = {fltrust, martfl}, two sweeps:
    - vary adv_rate ∈ {0.1, 0.3, 0.5} with poison_rate fixed at 0.5
    - vary poison_rate ∈ {0.1, 0.5, 1.0} with adv_rate fixed at 0.3
  The (adv=0.3, poison=0.5) point appears in both sweeps but in separate
  scenario directories.

Cell key: (defense, dataset, adv_rate, poison_rate)
Expected: 2 defenses × 5 unique (adv, poison) combinations = 10 cells
          (3 from adv-sweep + 3 from poison-sweep, with (0.3, 0.5) overlap)

Usage:
    python extract_step4_attack_sens.py
    python extract_step4_attack_sens.py --strict
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
# Declared expected grid (matches generate_step4_attack_sensitivity.py)
# ---------------------------------------------------------------------------
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100"]

# Actual on-disk sweeps (verified by listing the cell directories):
#   adv-rate sweep:    {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
#                      (0.0 only present for martfl backdoor)
#   poison-rate sweep: {0.0, 0.1, 0.3, 0.5, 1.0}  (with adv_rate fixed at 0.3)
ADV_RATES_TO_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
POISON_RATES_TO_SWEEP = [0.0, 0.1, 0.3, 0.5, 1.0]
DEFAULT_ADV_RATE = 0.3
DEFAULT_POISON_RATE = 0.5

# The current generator only declares "backdoor", but legacy "labelflip" cells
# also exist on disk under step5_atk_sens_*_labelflip_*. Those are leftovers
# and will appear as "unexpected" rows — that's intentional.
EXPECTED_ATTACK = "backdoor"

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "attack", "adv_rate", "poison_rate"]


def expected_cells() -> Set[Tuple[str, str, str, float, float]]:
    """Build the (defense, dataset, attack, adv_rate, poison_rate) cells.

    Both sweeps are unioned: the (0.3, 0.5) point counts as one cell, not two,
    since it has the same hyperparameters even though it appears in both
    scenario directories.

    `attack` is included in the key so leftover labelflip cells on disk are
    flagged as "unexpected" rather than being silently merged with backdoor.
    """
    cells = set()
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            d, ds, atk = (canonical_name(defense),
                            canonical_name(dataset),
                            canonical_name(EXPECTED_ATTACK))
            # Adv-rate sweep
            for ar in ADV_RATES_TO_SWEEP:
                cells.add((d, ds, atk, float(ar), float(DEFAULT_POISON_RATE)))
            # Poison-rate sweep
            for pr in POISON_RATES_TO_SWEEP:
                cells.add((d, ds, atk, float(DEFAULT_ADV_RATE), float(pr)))
    return cells


def key_from_row(row) -> Optional[Tuple[str, str, str, float, float]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    attack = row.get("attack")
    adv_rate = row.get("adv_rate")
    poison_rate = row.get("poison_rate")
    if any(v is None for v in (defense, dataset, adv_rate, poison_rate)):
        return None

    # Attack may be missing from the column. Fall back to parsing from path.
    if attack is None or not isinstance(attack, str):
        run_dir = str(row.get("run_dir", "")).lower()
        if "_labelflip_" in run_dir:
            attack = "labelflip"
        else:
            attack = "backdoor"

    try:
        return (canonical_name(defense), canonical_name(dataset),
                canonical_name(attack),
                float(adv_rate), float(poison_rate))
    except (TypeError, ValueError):
        return None


def main():
    ap = make_parser(4, "attack_sens", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step5_atk", require_success=True)

    write_step_csv_and_report(
        step_id=4, label="attack_sens",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
