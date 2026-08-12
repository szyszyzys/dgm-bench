"""
extract_step9_heterogeneity.py — Step 9 (data heterogeneity sweep).

What step 9 produces (verified against generate_step9_heterogeneity.py):
  CIFAR-100 / cifar100_cnn / image / backdoor.
  For each defense in FOCUSED_DEFENSES = {fltrust, martfl}, three experiment
  groups:

    Group A — Vary Seller (Buyer fixed IID):
      For each α ∈ {1.0, 0.5, 0.1}, set seller dirichlet_alpha=α
      → 3 cells per defense

    Group B — Vary Buyer (Seller fixed Dirichlet 0.5):
      For each α ∈ {1.0, 0.5, 0.1}, set buyer dirichlet_alpha=α
      → 3 cells per defense

    Group C — Scarcity (Buyer ratio sweep):
      For each ratio ∈ {0.1}, set buyer_ratio
      → 1 cell per defense

Cell key: (defense, dataset, group, sweep_value)
  group ∈ {"varyseller", "varybuyer", "scarcity"}
  sweep_value = the dirichlet_alpha (groups A, B) or buyer_ratio (group C)

Expected per defense: 3 + 3 + 1 = 7 cells. Total: 2 × 7 = 14 cells.

Note: heterogeneity scenarios get an asymmetric prefix list because the bare
"step11_" prefix would also match step11_freerider_*.

Usage:
    python extract_step9_heterogeneity.py
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
# Declared expected grid (matches generate_step9_heterogeneity.py)
# ---------------------------------------------------------------------------
FOCUSED_DEFENSES_RAW = ["fltrust", "martfl"]
ENABLED_DATASETS_RAW = ["CIFAR100"]

HETEROGENEITY_ALPHAS = [1.0, 0.5, 0.1]
# Root-size sweep (matches generate_step9_heterogeneity.py).
SCARCITY_RATIOS = [0.05, 0.1, 0.2]

NUM_SEEDS_PER_CONFIG = 5

CELL_COLUMNS = ["defense", "dataset", "group", "sweep_value"]


def expected_cells() -> Set[Tuple[str, str, str, float]]:
    cells = set()
    for defense in FOCUSED_DEFENSES_RAW:
        for dataset in ENABLED_DATASETS_RAW:
            d, ds = canonical_name(defense), canonical_name(dataset)
            # Group A: vary_seller
            for alpha in HETEROGENEITY_ALPHAS:
                cells.add((d, ds, "varyseller", float(alpha)))
            # Group B: vary_buyer
            for alpha in HETEROGENEITY_ALPHAS:
                cells.add((d, ds, "varybuyer", float(alpha)))
            # Group C: scarcity
            for ratio in SCARCITY_RATIOS:
                cells.add((d, ds, "scarcity", float(ratio)))
    return cells


# Path patterns (from save_path):
#   results/step11_<defense>_<dataset>/vary_seller/alpha_<x>
#   results/step11_<defense>_<dataset>/vary_buyer/alpha_<x>
#   results/step11_<defense>_<dataset>/scarcity/ratio_<x>
RE_VARY_SELLER = re.compile(r"vary_seller/alpha_([\d.]+)")
RE_VARY_BUYER  = re.compile(r"vary_buyer/alpha_([\d.]+)")
RE_SCARCITY    = re.compile(r"scarcity/ratio_([\d.]+)")


def key_from_row(row) -> Optional[Tuple[str, str, str, float]]:
    defense = row.get("defense")
    dataset = row.get("dataset")
    if defense is None or dataset is None:
        return None

    run_dir = str(row.get("run_dir", "")).replace("\\", "/")

    m = RE_VARY_SELLER.search(run_dir)
    if m:
        return (canonical_name(defense), canonical_name(dataset),
                "varyseller", float(m.group(1)))

    m = RE_VARY_BUYER.search(run_dir)
    if m:
        return (canonical_name(defense), canonical_name(dataset),
                "varybuyer", float(m.group(1)))

    m = RE_SCARCITY.search(run_dir)
    if m:
        return (canonical_name(defense), canonical_name(dataset),
                "scarcity", float(m.group(1)))

    return None


def main():
    ap = make_parser(9, "heterogeneity", description=__doc__)
    args = ap.parse_args()

    # Load all per-defense step11 prefixes (filter out free-rider drift)
    import pandas as pd
    dfs = []
    for defense in FOCUSED_DEFENSES_RAW + [
        "fedavg", "skymask", "trimmed_mean", "multi_krum",
        "rflpa", "spmc", "flame", "deepsight", "bulyan", "foolsgold",
    ]:
        prefix = f"step11_{defense}_"
        d = load_step_results(args.results_dir, prefix, require_success=True)
        if not d.empty:
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    write_step_csv_and_report(
        step_id=9, label="heterogeneity",
        df=df, output_dir=Path(args.output_dir),
        expected_cells_fn=expected_cells,
        key_from_row=key_from_row,
        expected_seeds=NUM_SEEDS_PER_CONFIG,
        cell_columns=CELL_COLUMNS,
        strict=args.strict, write_json=args.json,
    )


if __name__ == "__main__":
    main()
