"""
extract_step13_drowning.py — Step 13 (targeted drowning attack).

What step 13 produces (verified against generate_step13_drowning_attack.py):
  CIFAR-100 / cifar100_cnn / image. Tests both fltrust and martfl against a
  targeted drowning sybil attack. The attack_strength is swept over 2 values
  (was 4 in the original).

  Note: even though the generator declares both fltrust and martfl, the
  current rerun_priority.sh / verify_all output shows only fltrust has runs.
  martfl drowning runs are absent — possibly intentionally dropped from the
  current pipeline.

Cell key: (defense, dataset, attack_strength)
Expected per defense: 2 strength values. Total: 2 × 2 = 4 cells.

Usage:
    python extract_step13_drowning.py
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
# Declared expected grid (matches generate_step13_drowning_attack.py)
# ---------------------------------------------------------------------------
DROWNING_DEFENSES_RAW = ["fedavg", "fltrust", "martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100"]
# All four strengths exist on disk under
#   step13_drowning_drowning_<defense>/adv_0.3_attack_strength_<value>/...
# fedavg is included as a no-defense baseline.
ATTACK_STRENGTHS = [0.5, 1.0, 1.5, 2.0]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "attack_strength"]


def expected_cells() -> Set[Tuple[str, str, float]]:
    cells = set()
    for defense in DROWNING_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for s in ATTACK_STRENGTHS:
                cells.add((canonical_name(defense),
                           canonical_name(dataset), float(s)))
    return cells


# Path patterns: drowning runs encode strength as
#   adv_0.3_attack_strength_<value>
# in the second path component.
RE_STRENGTH = re.compile(r"attack_strength_([\d.]+)")


def key_from_row(row) -> Optional[Tuple[str, str, float]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", ""))
    m = RE_STRENGTH.search(run_dir)
    if m:
        try:
            strength = float(m.group(1))
        except ValueError:
            return None
    else:
        # Fallback: if a numeric attack_strength column exists, use it
        s = row.get("attack_strength")
        if s is None:
            return None
        try:
            strength = float(s)
        except (TypeError, ValueError):
            return None

    return (canonical_name(defense), canonical_name(dataset), strength)


def main():
    ap = make_parser(13, "drowning", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step13_drowning",
                            require_success=True)

    write_step_csv_and_report(
        step_id=13, label="drowning",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
