"""
check_coverage_matrix.py — per-step (dataset × defense) coverage matrix.

For each step in your current rerun, reads every generated config.yaml under
configs_generated_benchmark/<step_subdir>/, groups them by (dataset,
aggregation_method), and checks whether each combination has completed runs
under the path given by each config's experiment.save_path.

Output per step:
    - a matrix with rows = defenses, columns = datasets
    - each cell shows  done/total  where
        total = number of generated configs in this (dataset, defense) cell
        done  = number of those configs with at least one .success marker
                underneath their save_path
    - status icon:
        ✓  all configs in this cell have succeeded at least once
        ◐  some configs have succeeded, some haven't
        ✗  zero configs have any success
        ·  cell is not part of this step (defense × dataset combo not generated)

Usage:
    python check_coverage_matrix.py
    python check_coverage_matrix.py --steps 4 5 10
    python check_coverage_matrix.py --strict     # require .success for every
                                                  # expected seed (n_samples),
                                                  # not just "at least one"
    python check_coverage_matrix.py --csv        # also dump a flat CSV

What it does NOT do:
    - Run experiments
    - Delete anything
    - Parse generate_*.py scripts (too fragile); the generated configs on
      disk are the source of truth
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Steps included in the current rerun → configs_generated_benchmark subdir
# (matches STEP_REGISTRY in run_full_benchmark.py)
# ---------------------------------------------------------------------------
STEP_CONFIGS_SUBDIR: Dict[int, Dict] = {
    # CLI step : {
    #   label             : human label
    #   configs_subdir    : directory under configs_generated_benchmark/
    #   results_prefixes  : scenario-name prefixes under ./results/ used for
    #                       the fallback when the configs subdir has been
    #                       deleted mid-rerun (rerun_priority.sh cleans each
    #                       step's configs at the start of every phase).
    # }
    3:  {"label": "defense_tune",   "configs_subdir": "step3_defense_tuning",
         "results_prefixes": ["step3_tune_"]},
    4:  {"label": "attack_sens",    "configs_subdir": "step5_attack_sensitivity",
         "results_prefixes": ["step5_atk_sens_"]},
    5:  {"label": "sybil",          "configs_subdir": "step6_advanced_sybil",
         "results_prefixes": ["step6_adv_sybil_"]},
    6:  {"label": "adaptive",       "configs_subdir": "step7_adaptive_attack",
         "results_prefixes": ["step7_adaptive_", "step7_baseline_no_attack"]},
    7:  {"label": "buyer",          "configs_subdir": "step8_buyer_attacks",
         "results_prefixes": ["step8_buyer_attack_"]},
    8:  {"label": "scalability",    "configs_subdir": "step10_scalability",
         "results_prefixes": ["step10_scalability_"]},
    9:  {"label": "heterogeneity",  "configs_subdir": "step11_heterogeneity",
         # bare "step11_" would also match step11_freerider_* — enumerate the
         # per-defense prefixes instead, same trick as in rerun_priority.sh.
         "results_prefixes": [
             "step11_fedavg_", "step11_fltrust_", "step11_martfl_",
             "step11_skymask_", "step11_trimmed_mean_", "step11_multi_krum_",
             "step11_rflpa_", "step11_spmc_",
         ]},
    10: {"label": "main_summary",   "configs_subdir": "step12_main_summary",
         "results_prefixes": ["step12_main_summary_"]},
    13: {"label": "drowning",       "configs_subdir": "step13_drowning_attack",
         "results_prefixes": ["step13_drowning_"]},
    14: {"label": "collusion",      "configs_subdir": "step14_martfl_collusion",
         "results_prefixes": ["step14_collusion_"]},
    15: {"label": "pricing",        "configs_subdir": "step15_proportional_pricing",
         "results_prefixes": ["step15_pricing_"]},
    16: {"label": "dp_fairness",    "configs_subdir": "step16_dp_fairness",
         "results_prefixes": ["step16_dp_"]},
    17: {"label": "alie",           "configs_subdir": "step17_alie_attack",
         "results_prefixes": ["step17_alie_"]},
    19: {"label": "new_defenses",   "configs_subdir": "step19_new_defenses",
         "results_prefixes": ["step19_new_defenses"]},
}

# Intentionally not checked — not part of the current rerun
EXCLUDED_DEFENSES = {"daved"}

# Datasets intentionally hidden from the coverage matrix. The data stays on
# disk (so it can be re-enabled later) — it just doesn't appear in the output.
# Override on the command line with:  --exclude-datasets X Y   or  --no-exclude
EXCLUDED_DATASETS_DEFAULT = {"FEMNIST", "Purchase100"}

# Lowercase dataset tag → proper-case canonical name
DS_TAG_TO_PROPER = {
    "cifar100": "CIFAR100", "cifar10": "CIFAR10", "femnist": "FEMNIST",
    "texas100": "Texas100", "purchase100": "Purchase100", "trec": "TREC",
}
# Defenses we know about (longest first so skymask_small matches before skymask)
KNOWN_DEFENSES = [
    "skymask_small", "trimmed_mean", "multi_krum",
    "fedavg", "fltrust", "martfl", "skymask", "rflpa", "spmc",
    "flame", "deepsight", "bulyan", "foolsgold",
]
RE_DS_TAG = re.compile(r"ds-([a-z0-9]+)")
RE_AGG_TAG = re.compile(r"agg-(" + "|".join(KNOWN_DEFENSES) + r")(?:[_-]|$)")


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------
def read_cell_from_config(config_path: Path) -> Optional[Dict]:
    """Return {dataset, defense, save_path, n_samples} from a generated config.
    Returns None if the file is unreadable or missing required fields.
    """
    try:
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None

    exp = cfg.get("experiment") or {}
    agg = cfg.get("aggregation") or {}

    dataset = exp.get("dataset_name")
    defense = agg.get("method")
    save_path = exp.get("save_path")
    # n_samples lives either at top-level or under experiment depending on the
    # scenario generator; check both.
    n_samples = (
        exp.get("n_samples")
        or cfg.get("n_samples")
        or 1
    )

    if not dataset or not defense or not save_path:
        return None
    return {
        "dataset": str(dataset),
        "defense": str(defense).lower(),
        "save_path": str(save_path),
        "n_samples": int(n_samples),
    }


def count_success_under(save_path: Path) -> int:
    """Count .success markers recursively under a given save_path."""
    if not save_path.exists():
        return 0
    try:
        return sum(1 for _ in save_path.rglob(".success"))
    except Exception:
        return 0


def parse_cell_from_path(path_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (dataset, defense) from a result path using ds-XXX and agg-XXX
    tags that appear in every HP cell directory name."""
    lower = path_str.lower().replace("\\", "/")
    dataset = None
    defense = None

    m_ds = RE_DS_TAG.search(lower)
    if m_ds and m_ds.group(1) in DS_TAG_TO_PROPER:
        dataset = DS_TAG_TO_PROPER[m_ds.group(1)]

    m_agg = RE_AGG_TAG.search(lower)
    if m_agg:
        defense = m_agg.group(1)

    return dataset, defense


def build_matrix_from_results(
    results_root: Path,
    results_prefixes: List[str],
    excluded_datasets: set,
) -> Dict[Tuple[str, str], Dict]:
    """Fallback scanner used when the configs subdir has been deleted.

    Walks ./results/<prefix>* for each prefix, finds every leaf run directory
    (by looking for .success / .failed / .in_progress markers OR for
    final_metrics.json), parses (dataset, defense) from the path, and builds
    the same cell dict shape that build_matrix_from_configs emits.

    Rows with a dataset in excluded_datasets are dropped (display-only filter,
    data on disk is untouched).
    """
    cells: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
        "configs": 0,
        "configs_with_success": 0,
        "success_markers": 0,
        "expected_seeds": 0,
    })

    if not results_root.exists():
        return cells

    seen_leaf_dirs = set()
    for prefix in results_prefixes:
        for scenario_dir in results_root.glob(f"{prefix}*"):
            if not scenario_dir.is_dir():
                continue
            # Each leaf run dir holds exactly one .success or .failed or
            # .in_progress marker. Use .success + .failed + .in_progress as
            # the "this was an intended run" signal.
            markers = list(scenario_dir.rglob(".success")) \
                    + list(scenario_dir.rglob(".failed")) \
                    + list(scenario_dir.rglob(".in_progress"))
            for marker in markers:
                leaf = marker.parent
                if leaf in seen_leaf_dirs:
                    continue
                seen_leaf_dirs.add(leaf)

                dataset, defense = parse_cell_from_path(str(leaf))
                if not dataset or not defense:
                    continue
                if defense in EXCLUDED_DEFENSES:
                    continue
                if dataset in excluded_datasets:
                    continue

                key = (dataset, defense)
                cell = cells[key]
                # Treat each leaf run dir as one "config" for matrix purposes.
                # (In reality it's one seed × HP cell, but for a dataset ×
                # defense coverage check that distinction is irrelevant — we
                # just want to know whether this combo has any working runs.)
                cell["configs"] += 1
                cell["expected_seeds"] += 1
                if (leaf / ".success").exists():
                    cell["configs_with_success"] += 1
                    cell["success_markers"] += 1

    return cells


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------
def build_matrix(
    configs_root: Path,
    results_root: Path,
    strict: bool,
    excluded_datasets: set,
) -> Dict:
    """For each step, build a (dataset, defense) coverage matrix.

    Preferred source: generated configs under configs_root/<step_subdir>/.
    Fallback: walk ./results/<prefix>* when the configs subdir has been
    deleted (rerun_priority.sh cleans each phase's configs at the start of
    the phase, so finished steps lose their configs while keeping results).

    Returns a nested dict:
        result[step_id] = {
            "label": str,
            "source": "configs" | "results" | None,
            "cells": {(dataset, defense): {...}},
            "datasets": sorted list,
            "defenses": sorted list,
            "missing": bool,
        }
    """
    out: Dict[int, Dict] = {}

    for step_id, spec in STEP_CONFIGS_SUBDIR.items():
        label = spec["label"]
        subdir_name = spec["configs_subdir"]
        step_configs_root = configs_root / subdir_name

        entry = {
            "label": label,
            "subdir": subdir_name,
            "source": None,
            "cells": defaultdict(lambda: {
                "configs": 0,
                "configs_with_success": 0,
                "success_markers": 0,
                "expected_seeds": 0,
            }),
            "datasets": set(),
            "defenses": set(),
            "missing": False,
            "total_configs": 0,
        }

        # --- Source 1: generated configs ---
        if step_configs_root.exists():
            for cfg_file in step_configs_root.rglob("config.yaml"):
                info = read_cell_from_config(cfg_file)
                if info is None:
                    continue
                if info["defense"] in EXCLUDED_DEFENSES:
                    continue
                if info["dataset"] in excluded_datasets:
                    continue

                ds = info["dataset"]
                df = info["defense"]
                cell = entry["cells"][(ds, df)]
                cell["configs"] += 1
                cell["expected_seeds"] += info["n_samples"]
                n_success = count_success_under(Path(info["save_path"]))
                cell["success_markers"] += n_success

                if strict:
                    if n_success >= info["n_samples"]:
                        cell["configs_with_success"] += 1
                else:
                    if n_success >= 1:
                        cell["configs_with_success"] += 1

                entry["datasets"].add(ds)
                entry["defenses"].add(df)
                entry["total_configs"] += 1

            if entry["total_configs"] > 0:
                entry["source"] = "configs"

        # --- Source 2: fallback — scan results for this step's prefixes ---
        if entry["source"] is None:
            fallback_cells = build_matrix_from_results(
                results_root, spec["results_prefixes"], excluded_datasets
            )
            if fallback_cells:
                entry["cells"] = fallback_cells
                entry["source"] = "results"
                entry["datasets"] = {k[0] for k in fallback_cells}
                entry["defenses"] = {k[1] for k in fallback_cells}
                entry["total_configs"] = sum(
                    c["configs"] for c in fallback_cells.values()
                )

        if entry["source"] is None:
            entry["missing"] = True

        entry["datasets"] = sorted(entry["datasets"])
        entry["defenses"] = sorted(entry["defenses"])
        out[step_id] = entry

    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def cell_status(cell: Dict) -> str:
    if cell["configs"] == 0:
        return "·"
    if cell["configs_with_success"] == cell["configs"]:
        return "✓"
    if cell["configs_with_success"] == 0:
        return "✗"
    return "◐"


def print_step_matrix(step_id: int, entry: Dict) -> None:
    label = entry["label"]
    source = entry.get("source")
    source_tag = {
        "configs": "(source: generated configs)",
        "results": "(source: results/ fallback — configs were cleaned)",
        None:       "",
    }[source]

    border = "─" * 80
    print(border)
    print(f" Step {step_id:<3} {label:<16}  {source_tag}")
    print(border)

    if entry["missing"]:
        print("  (no configs and no results found — step not started yet)")
        return
    if entry["total_configs"] == 0:
        print("  (no valid entries found for this step)")
        return

    datasets = entry["datasets"]
    defenses = entry["defenses"]

    # Column widths — reserve enough room that cells never touch.
    # Each cell prints as "{icon} {frac} " (icon + space + frac + trailing space).
    cell_w = max(12, max((len(d) for d in datasets), default=0) + 2)
    def_w = max(14, max((len(d) for d in defenses), default=0) + 2)

    # Header: right-justify dataset names in each cell column
    header = " " * def_w + "".join(f"{d:>{cell_w}}" for d in datasets) + "   status"
    print(header)

    for defense in defenses:
        row = f"{defense:<{def_w}}"
        total_configs = 0
        total_done = 0
        for ds in datasets:
            cell = entry["cells"].get((ds, defense))
            if cell is None:
                # Not in this step's plan
                row += f"{'·':>{cell_w}}"
                continue
            total_configs += cell["configs"]
            total_done += cell["configs_with_success"]
            icon = cell_status(cell)
            frac = f"{cell['configs_with_success']}/{cell['configs']}"
            # Reserve 2 chars for "icon space", rest for right-justified frac.
            # Subtract 1 more for a trailing space so cells never touch.
            inner = cell_w - 3
            row += f" {icon} {frac:>{inner}}"
        if total_configs == 0:
            row += "   ·"
        elif total_done == total_configs:
            row += f"   ✓ all done ({total_done}/{total_configs})"
        elif total_done == 0:
            row += f"   ✗ none done (0/{total_configs})"
        else:
            row += f"   ◐ {total_done}/{total_configs} done"
        print(row)

    # Overall step coverage line
    overall_configs = sum(c["configs"] for c in entry["cells"].values())
    overall_done = sum(c["configs_with_success"] for c in entry["cells"].values())
    pct = 100.0 * overall_done / overall_configs if overall_configs else 0
    print(f"  → overall: {overall_done}/{overall_configs} ready ({pct:.1f}%)")


def dump_csv(matrix: Dict, csv_path: Path) -> None:
    """Write a flat CSV: step, label, dataset, defense, configs,
    configs_with_success, success_markers, expected_seeds, status."""
    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "label", "dataset", "defense",
            "configs", "configs_with_success", "success_markers",
            "expected_seeds", "status",
        ])
        for step_id, entry in matrix.items():
            for (ds, defense), cell in entry["cells"].items():
                writer.writerow([
                    step_id, entry["label"], ds, defense,
                    cell["configs"], cell["configs_with_success"],
                    cell["success_markers"], cell["expected_seeds"],
                    cell_status(cell),
                ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs_dir", default="./configs_generated_benchmark",
                    help="Root of generated configs (primary source)")
    ap.add_argument("--results_dir", default="./results",
                    help="Root of experiment results (fallback source)")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="CLI step numbers to check (default: all current steps)")
    ap.add_argument("--strict", action="store_true",
                    help="A config counts as 'done' only when every seed in "
                         "n_samples has a .success (default: at least one)")
    ap.add_argument("--csv", action="store_true",
                    help="Also dump coverage_matrix.csv next to this script")
    ap.add_argument("--exclude-datasets", nargs="*", default=None,
                    metavar="DATASET",
                    help=f"Datasets to hide from the matrix (default: "
                         f"{sorted(EXCLUDED_DATASETS_DEFAULT)}). Pass zero "
                         f"arguments to explicitly clear the list.")
    ap.add_argument("--no-exclude", action="store_true",
                    help="Clear the dataset exclusion list entirely — show "
                         "every dataset found on disk.")
    args = ap.parse_args()

    # Resolve final dataset exclusion set
    if args.no_exclude:
        excluded_datasets = set()
    elif args.exclude_datasets is not None:
        excluded_datasets = set(args.exclude_datasets)
    else:
        excluded_datasets = set(EXCLUDED_DATASETS_DEFAULT)

    configs_root = Path(args.configs_dir)
    results_root = Path(args.results_dir)
    if not configs_root.exists() and not results_root.exists():
        print(f"❌ Neither configs nor results roots exist:")
        print(f"   configs: {configs_root.resolve()}")
        print(f"   results: {results_root.resolve()}")
        sys.exit(1)

    steps = args.steps or sorted(STEP_CONFIGS_SUBDIR.keys())
    unknown = [s for s in steps if s not in STEP_CONFIGS_SUBDIR]
    if unknown:
        print(f"❌ Unknown steps: {unknown}")
        print(f"   Known: {sorted(STEP_CONFIGS_SUBDIR.keys())}")
        sys.exit(1)

    print(f"📂 Configs root : {configs_root.resolve()}")
    print(f"📂 Results root : {results_root.resolve()}")
    print(f"🎯 Mode         : {'STRICT (all seeds)' if args.strict else 'LOOSE (≥1 seed)'}")
    print(f"🚫 Excluded     : defenses={sorted(EXCLUDED_DEFENSES)}  "
          f"datasets={sorted(excluded_datasets) if excluded_datasets else '(none)'}")
    print(f"🔢 Steps        : {steps}")
    print()

    matrix = build_matrix(configs_root, results_root, args.strict, excluded_datasets)

    grand_configs = 0
    grand_done = 0
    for step_id in steps:
        entry = matrix[step_id]
        print_step_matrix(step_id, entry)
        print()
        grand_configs += sum(c["configs"] for c in entry["cells"].values())
        grand_done += sum(c["configs_with_success"] for c in entry["cells"].values())

    print("═" * 72)
    pct = 100.0 * grand_done / grand_configs if grand_configs else 0
    print(f" GRAND TOTAL: {grand_done}/{grand_configs} configs ready ({pct:.1f}%) "
          f"across {len(steps)} step(s)")
    print("═" * 72)

    if args.csv:
        out_path = Path("coverage_matrix.csv")
        dump_csv(matrix, out_path)
        print(f"\n📝 Flat CSV written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
