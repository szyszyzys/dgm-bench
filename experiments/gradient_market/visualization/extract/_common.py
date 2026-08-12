"""
Shared framework for per-step result extractors.

Each per-step extractor (extract_stepN_*.py) declares its expected grid as a
set of cell-keys, then calls write_step_csv_and_report() with the loaded
DataFrame. This module handles:

  - filtering DAVED / FEMNIST / Purchase100 (display-only, not destructive)
  - normalizing column names from viz_utils.load_step_results
  - writing the step's CSV
  - computing readiness against the expected grid (loose vs strict)
  - printing a per-step status card
  - optionally writing a JSON summary for downstream tooling

Per-step extractors must:
  1. Declare the EXPECTED_* constants at module top
  2. Implement expected_cells() returning a set of tuples
  3. Implement key_from_row(row) returning the matching tuple from a DataFrame row
  4. Call write_step_csv_and_report() with the loaded df + the above
"""

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical name normalization
# ---------------------------------------------------------------------------
# viz_utils.fmt() is inconsistent: fmt("CIFAR100") returns "CIFAR-100" via the
# DATASET_NAMES dict, but fmt("CIFAR-100") falls through to .title() and
# returns "Cifar-100" because "cifar-100" (with hyphen) is not in the dict.
# Step 3 configs happen to write "CIFAR-100" into experiment.dataset_name, so
# load_step_results' DataFrame has "Cifar-100", while expected_cells() built
# from raw "CIFAR100" gets "CIFAR-100". The two never match.
#
# Fix: collapse names to a case-and-separator-insensitive form for matching.
# This is used ONLY for the readiness comparison; the CSV still contains
# whatever the data loader produced, so plots see the original strings.
# Aliases that map any display variant of a defense / dataset / attack to a
# single canonical raw form. The key reason this exists is that
# viz_utils.fmt() applies "paper-style" renames (e.g. "trimmed_mean" →
# "Trim-Mean", "skymask_small" → "SkyMask-S") which break naive character
# normalization — "trimmedmean" and "trimmean" don't compare equal even after
# stripping case and punctuation.
#
# Add a row here whenever you discover the matching is missing a synonym.
# Format: any variant (after canonical_name() character-normalization) →
# the canonical raw name (also in canonical_name() form).
NAME_ALIASES = {
    # defenses
    "trimmean":   "trimmedmean",
    "trim":       "trimmedmean",
    "skymasks":   "skymasksmall",
    # datasets
    "cifar":      "cifar100",
    "cifar10":    "cifar10",       # explicit so it doesn't collapse to cifar100
    "texas":      "texas100",
    "purchase":   "purchase100",
    # leave attacks/sybil names alone for now — extend on demand
}


def canonical_name(s: Any) -> str:
    """Normalize a name for matching: lowercase, alphanumeric only, then
    apply NAME_ALIASES so paper-style renames map back to a single form.

    Examples:
        canonical_name("CIFAR100")     == "cifar100"
        canonical_name("CIFAR-100")    == "cifar100"
        canonical_name("Cifar-100")    == "cifar100"
        canonical_name("FLTrust")      == "fltrust"
        canonical_name("multi_krum")   == "multikrum"
        canonical_name("Multi-Krum")   == "multikrum"
        canonical_name("trimmed_mean") == "trimmedmean"
        canonical_name("Trim-Mean")    == "trimmedmean"   # via NAME_ALIASES
    """
    if not isinstance(s, str):
        return str(s).lower()
    raw = "".join(ch.lower() for ch in s if ch.isalnum())
    return NAME_ALIASES.get(raw, raw)


def canonical_tuple(t: Tuple[Any, ...]) -> Tuple[str, ...]:
    """Apply canonical_name() element-wise to a tuple. Used for cell-keys."""
    return tuple(canonical_name(x) for x in t)

# ---------------------------------------------------------------------------
# Display-only filters. Data on disk is untouched.
# ---------------------------------------------------------------------------
EXCLUDED_DEFENSES = {"daved", "DAVED", "Daved"}
EXCLUDED_DATASETS_DEFAULT = {
    "FEMNIST", "Femnist", "femnist", "FeMNIST",
    "Purchase100", "purchase100", "Purchase-100",
}


def apply_global_filters(df: pd.DataFrame,
                          excluded_datasets: Optional[Set[str]] = None) -> pd.DataFrame:
    """Drop rows for excluded defenses and datasets. Returns a new DataFrame."""
    if df.empty:
        return df
    out = df.copy()
    if "defense" in out.columns:
        out = out[~out["defense"].isin(EXCLUDED_DEFENSES)]
    if excluded_datasets is None:
        excluded_datasets = EXCLUDED_DATASETS_DEFAULT
    if excluded_datasets and "dataset" in out.columns:
        out = out[~out["dataset"].isin(excluded_datasets)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cell-level readiness computation
# ---------------------------------------------------------------------------
CellKey = Tuple[Any, ...]
KeyFromRowFn = Callable[[pd.Series], Optional[CellKey]]


def compute_readiness(
    df: pd.DataFrame,
    expected: Set[CellKey],
    key_from_row: KeyFromRowFn,
    expected_seeds: int,
) -> Dict[str, Any]:
    """Walk df rows, group by cell-key, compare to the expected set.

    Returns a dict with:
      n_expected           : len(expected)
      n_present            : how many expected cells have ≥1 seed
      n_complete_seeds     : how many expected cells have ≥expected_seeds seeds
      missing              : sorted list of expected cells with 0 seeds
      extra                : sorted list of cells in df that aren't in expected
      incomplete_seeds     : dict of (cell -> n_seeds) for cells with <expected
      seed_counts          : dict of (cell -> n_seeds) for all present cells
    """
    seed_counts: Dict[CellKey, int] = {}
    extra_keys: Set[CellKey] = set()

    if not df.empty:
        for _, row in df.iterrows():
            key = key_from_row(row)
            if key is None:
                continue
            if key not in expected:
                extra_keys.add(key)
                continue
            seed_counts[key] = seed_counts.get(key, 0) + 1

    present = set(seed_counts.keys())
    complete = {k for k, n in seed_counts.items() if n >= expected_seeds}
    missing = expected - present
    incomplete = {k: seed_counts[k] for k in present if seed_counts[k] < expected_seeds}

    return {
        "n_expected":       len(expected),
        "n_present":        len(present),
        "n_complete_seeds": len(complete),
        "missing":          sorted(missing, key=lambda t: tuple(str(x) for x in t)),
        "extra":            sorted(extra_keys, key=lambda t: tuple(str(x) for x in t)),
        "incomplete_seeds": incomplete,
        "seed_counts":      seed_counts,
    }


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------
def status_label(report: Dict[str, Any], strict: bool) -> Tuple[str, str]:
    """Return (icon, human_label) for the readiness report."""
    n_expected = report["n_expected"]
    n_present = report["n_present"]
    n_complete = report["n_complete_seeds"]

    if n_expected == 0:
        return ("∅", "NO PLAN — expected_cells() returned empty set")
    if n_present == 0:
        return ("✗", "NOT STARTED — zero expected cells have any seeds")

    if strict:
        if n_complete == n_expected:
            return ("✅", f"COMPREHENSIVE — every expected cell has full seeds")
        if n_complete > 0:
            return ("◐", f"PARTIAL — {n_complete}/{n_expected} cells with full seeds")
        return ("◐", f"PARTIAL — cells exist but none have full seeds")
    else:
        if n_present == n_expected:
            return ("✅", f"COMPREHENSIVE — every expected cell has ≥1 seed")
        return ("◐", f"PARTIAL — {n_present}/{n_expected} cells have ≥1 seed")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_step_card(
    step_id: int,
    label: str,
    csv_path: Path,
    n_rows: int,
    report: Dict[str, Any],
    strict: bool,
    max_missing: int = 10,
) -> None:
    border = "=" * 72
    print()
    print(border)
    print(f" Step {step_id}: {label}")
    print(border)
    print(f"  CSV path           : {csv_path}")
    print(f"  Rows in CSV        : {n_rows}")
    print(f"  Expected cells     : {report['n_expected']}")
    print(f"  Cells with ≥1 seed : {report['n_present']}/{report['n_expected']}"
          f"  ({pct(report['n_present'], report['n_expected'])})")
    print(f"  Cells with full   : {report['n_complete_seeds']}/{report['n_expected']}"
          f"  ({pct(report['n_complete_seeds'], report['n_expected'])})")

    icon, msg = status_label(report, strict)
    print(f"  Status            : {icon}  {msg}")

    if report["missing"]:
        print()
        print(f"  Missing cells (showing first {min(max_missing, len(report['missing']))}):")
        for cell in report["missing"][:max_missing]:
            print(f"    - {cell}")
        if len(report["missing"]) > max_missing:
            print(f"    ... and {len(report['missing']) - max_missing} more")

    if report["extra"]:
        print()
        print(f"  ⚠ Unexpected cells in results (not in declared plan):")
        for cell in report["extra"][:5]:
            print(f"    - {cell}")
        if len(report["extra"]) > 5:
            print(f"    ... and {len(report['extra']) - 5} more")

    if strict and report["incomplete_seeds"]:
        print()
        print(f"  Cells missing seeds (showing first 10):")
        for k, n in list(report["incomplete_seeds"].items())[:10]:
            print(f"    - {k}: {n} seed(s)")


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "—"


# ---------------------------------------------------------------------------
# Main entry point used by every per-step extractor
# ---------------------------------------------------------------------------
def write_step_csv_and_report(
    *,
    step_id: int,
    label: str,
    df: pd.DataFrame,
    output_dir: Path,
    expected_cells_fn: Callable[[], Set[CellKey]],
    key_from_row: KeyFromRowFn,
    expected_seeds: int,
    cell_columns: List[str],
    strict: bool = False,
    write_json: bool = False,
    excluded_datasets: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Filter, write CSV, compute readiness, print card, optionally write JSON.

    Returns the readiness report dict (so an extract_all.py wrapper can build
    a unified summary across all steps).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = apply_global_filters(df, excluded_datasets)

    # Sort columns: identity columns first, then metrics, then everything else
    if not df.empty:
        identity_cols = [c for c in cell_columns if c in df.columns]
        metric_cols   = [c for c in ("acc", "asr", "bsr", "msr",
                                       "completed_rounds") if c in df.columns]
        other_cols    = [c for c in df.columns
                          if c not in identity_cols + metric_cols]
        df = df[identity_cols + metric_cols + other_cols]

    csv_path = output_dir / f"step{step_id}_{label}.csv"
    df.to_csv(csv_path, index=False)

    expected = expected_cells_fn()
    report = compute_readiness(df, expected, key_from_row, expected_seeds)
    report["step"] = step_id
    report["label"] = label
    report["csv"] = str(csv_path)
    report["n_rows"] = len(df)
    icon, msg = status_label(report, strict)
    report["status_icon"] = icon
    report["status_msg"] = msg

    print_step_card(step_id, label, csv_path, len(df), report, strict)

    if write_json:
        json_path = output_dir / f"step{step_id}_{label}.json"
        with json_path.open("w") as f:
            json.dump({
                "step": step_id,
                "label": label,
                "csv": str(csv_path),
                "n_rows": len(df),
                "n_expected": report["n_expected"],
                "n_present": report["n_present"],
                "n_complete_seeds": report["n_complete_seeds"],
                "missing": [list(c) for c in report["missing"]],
                "extra":   [list(c) for c in report["extra"]],
                "incomplete_seeds": {
                    str(list(k)): v for k, v in report["incomplete_seeds"].items()
                },
                "comprehensive": (report["n_complete_seeds"] == report["n_expected"]
                                   if strict
                                   else report["n_present"] == report["n_expected"]),
                "status_icon": icon,
                "status_msg": msg,
            }, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Argparse helper used by every per-step extractor
# ---------------------------------------------------------------------------
def make_parser(step_id: int, label: str, description: str = "") -> Any:
    import argparse
    ap = argparse.ArgumentParser(
        description=description or f"Extract step {step_id} ({label}) results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results_dir", default="./results",
                    help="Root experiment results directory")
    ap.add_argument("--output_dir", default="./analysis_partial",
                    help="Where to write the CSV (and JSON if --json)")
    ap.add_argument("--strict", action="store_true",
                    help="Require expected_seeds per cell, not just ≥1 seed")
    ap.add_argument("--json", action="store_true",
                    help="Also write a JSON readiness summary alongside the CSV")
    return ap
