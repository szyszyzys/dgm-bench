"""
extract_step17_alie.py — Step 17 (ALIE — A Little Is Enough — attack).

What step 17 produces (verified against generate_step17_alie_attack.py):
  CIFAR-100 / cifar100_cnn / image / backdoor (adv_rate=0.3, poison_rate=0.5).
  For each defense in IMAGE_DEFENSES, ALIE is enabled as the sybil strategy.
  Z-scores ∈ {1.0, 2.0} (reduced from {0.5, 1.0, 1.5, 2.0}).

Cell key: (defense, dataset, z_score)
Expected: 11 IMAGE_DEFENSES × 1 dataset × 2 z-scores = 22 cells.

Note: like step 15, the realistic floor is currently 2 defenses (fltrust +
martfl) because other defenses skip when their tuned params are missing.

Usage:
    python extract_step17_alie.py
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
# Declared expected grid (matches generate_step17_alie_attack.py)
# ---------------------------------------------------------------------------
# Generator iterates over IMAGE_DEFENSES (11 entries), but in practice the
# generator skips any defense without tuned params from step 3, and only
# fltrust + martfl currently have those. The realistic grid is therefore
# narrower than the declared one. Listed in two tiers so the difference is
# auditable: anything in REALISTIC_DEFENSES_RAW is paper-ready scope; the
# difference vs DECLARED is "would tune more defenses if we could".
DECLARED_ALIE_DEFENSES_RAW = [
    "fltrust", "martfl", "skymask", "trimmed_mean", "multi_krum",
    "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
]
REALISTIC_ALIE_DEFENSES_RAW = ["fltrust", "martfl"]
# Set ALIE_DEFENSES_RAW to whichever scope you want the readiness check to
# compare against. Default = realistic so ✅ status reflects paper-readiness.
ALIE_DEFENSES_RAW = REALISTIC_ALIE_DEFENSES_RAW

ENABLED_DATASETS_RAW = ["CIFAR100"]
# All four z-scores actually exist on disk (z_0.5, z_1.0, z_1.5, z_2.0).
# The "reduced from 4" comment in the generator is stale.
Z_SCORES = [0.5, 1.0, 1.5, 2.0]

NUM_SEEDS_PER_CONFIG = 2

CELL_COLUMNS = ["defense", "dataset", "z_score"]


def expected_cells() -> Set[Tuple[str, str, float]]:
    cells = set()
    for defense in ALIE_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for z in Z_SCORES:
                cells.add((canonical_name(defense),
                           canonical_name(dataset), float(z)))
    return cells


# Path patterns: z-score lives in its own path component as "z_X.X"
# e.g.  results/step17_alie_fltrust_CIFAR100/z_1.5/ds-cifar100_...
RE_ZSCORE = re.compile(r"/z_([\d.]+)/")


def key_from_row(row) -> Optional[Tuple[str, str, float]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    # z-score is encoded only in the path, not in any column. Use forward-
    # slashes so the regex works on both Linux and Windows-style paths.
    run_dir = str(row.get("run_dir", "")).replace("\\", "/")
    m = RE_ZSCORE.search(run_dir)
    if not m:
        return None
    try:
        return (canonical_name(defense), canonical_name(dataset), float(m.group(1)))
    except ValueError:
        return None


def main():
    ap = make_parser(17, "alie", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step17_alie",
                            require_success=True)

    write_step_csv_and_report(
        step_id=17, label="alie",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
