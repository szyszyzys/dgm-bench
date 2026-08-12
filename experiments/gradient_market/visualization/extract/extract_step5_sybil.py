"""
extract_step5_sybil.py — Step 5 (sybil strategy comparison).

What step 5 produces (verified against generate_step5_advanced_sybil.py):
  CIFAR-100 / cifar100_cnn / image / backdoor.
  For each defense in FOCUSED_DEFENSES = {fltrust, martfl}, four sybil strategies:
    - baseline_no_sybil  (no parameter sweep, 1 cell)
    - mimic              (no parameter sweep, 1 cell)
    - pivot              (no parameter sweep, 1 cell)
    - oracle_blend       (sweep blend_alpha ∈ {0.1, 0.5, 0.9}, 3 cells)

Cell key: (defense, dataset, sybil_strategy, blend_alpha)
Expected per defense: 1+1+1+3 = 6 cells. Total: 2 × 6 = 12 cells.

The blend_alpha is None for non-oracle_blend strategies, encoded as -1.0 in
the cell key (since None is not hashable in some contexts and we need a stable
sort key).

Usage:
    python extract_step5_sybil.py
    python extract_step5_sybil.py --strict
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
# Declared expected grid (matches SYBIL_FOCUS_DEFENSES in
# generate_step5_advanced_sybil.py).
# Includes the canonical clustering / similarity-based sybil defenses
# (FoolsGold, DeepSight) in addition to the trust-based ones.
# ---------------------------------------------------------------------------
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl", "foolsgold", "deepsight"]
ENABLED_DATASETS_RAW = ["CIFAR100"]

# Strategy → list of blend_alpha values to sweep (None = no sweep, single cell)
SYBIL_STRATEGIES = {
    "baseline_no_sybil": [None],
    "mimic":             [None],
    "pivot":             [None],
    "oracle_blend":      [0.1, 0.5, 0.9],
}

NUM_SEEDS_PER_CONFIG = 2

# Sentinel for "no blend_alpha" so the cell key is always a tuple of comparable
# values (None breaks set ordering and dict keys in some pandas operations).
NO_BLEND = -1.0

CELL_COLUMNS = ["defense", "dataset", "sybil_strategy", "blend_alpha"]


def expected_cells() -> Set[Tuple[str, str, str, float]]:
    cells = set()
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            for strategy, alphas in SYBIL_STRATEGIES.items():
                for alpha in alphas:
                    a = NO_BLEND if alpha is None else float(alpha)
                    cells.add((
                        canonical_name(defense),
                        canonical_name(dataset),
                        canonical_name(strategy),
                        a,
                    ))
    return cells


import re
# Scenario name format: step6_adv_sybil_<strategy>_<defense>
# e.g.  step6_adv_sybil_baseline_no_sybil_fltrust
#       step6_adv_sybil_oracle_blend_martfl
#       step6_adv_sybil_mimic_fltrust
# blend_alpha is then encoded in the HP cell suffix: ..._blend_alpha_0.5
# Defenses are matched against a fixed whitelist so multi-word strategy names
# (baseline_no_sybil, oracle_blend) aren't truncated.
KNOWN_SYBIL_DEFENSES = ["fltrust", "martfl", "fedavg",
                         "skymask", "skymask_small",
                         "trimmed_mean", "multi_krum", "rflpa", "spmc",
                         "flame", "deepsight", "bulyan", "foolsgold"]
RE_SYBIL_SCENARIO = re.compile(
    r"step6_adv_sybil_(.+?)_(" + "|".join(KNOWN_SYBIL_DEFENSES) + r")(?:/|$)"
)
RE_BLEND_ALPHA = re.compile(r"blend_alpha[_/]?([\d.]+)")


def key_from_row(row) -> Optional[Tuple[str, str, str, float]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    # Parse sybil strategy from the scenario directory name (it's not a column
    # — viz_utils never reads it because the configs don't expose it).
    run_dir = str(row.get("run_dir", "")).replace("\\", "/")
    m = RE_SYBIL_SCENARIO.search(run_dir)
    if not m:
        return None
    strategy_raw = m.group(1)

    # blend_alpha is encoded in the HP cell suffix for oracle_blend; absent
    # for the other strategies.
    a = NO_BLEND
    am = RE_BLEND_ALPHA.search(run_dir)
    if am:
        try:
            a = float(am.group(1))
        except ValueError:
            a = NO_BLEND

    return (canonical_name(defense), canonical_name(dataset),
            canonical_name(strategy_raw), a)


def main():
    ap = make_parser(5, "sybil", description=__doc__)
    args = ap.parse_args()

    df = load_step_results(args.results_dir, "step6_adv_sybil",
                            require_success=True)

    write_step_csv_and_report(
        step_id=5, label="sybil",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
