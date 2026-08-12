"""
generate_main_table.py — Generate the headline LaTeX table from Step 10 results.

Produces a multi-panel table organized by three evaluation dimensions:
  Panel A: Model Performance  (ACC ↑)
  Panel B: Attack Performance (ASR ↓)
  Panel C: Selection Patterns (BSR ↑, MSR ↓)

Each panel shows all defenses × all datasets, grouped by defense family.

Usage:
    python experiments/gradient_market/visualization/generate_main_table.py
    python experiments/gradient_market/visualization/generate_main_table.py --output figures/main_table.tex
    python experiments/gradient_market/visualization/generate_main_table.py --show_std
    python experiments/gradient_market/visualization/generate_main_table.py --compact  # single-panel 4-metric layout
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

DEFENSE_GROUPS = [
    ("No Defense", [
        ("fedavg", "FedAvg"),
    ]),
    ("Similarity-based", [
        ("fltrust", "FLTrust"),
        ("foolsgold", "FoolsGold"),
    ]),
    ("Clustering", [
        ("martfl", "MartFL"),
        ("deepsight", "DeepSight"),
        ("flame", "FLAME"),
    ]),
    ("Statistical", [
        ("trimmed_mean", "Trim-Mean"),
        ("multi_krum", "Multi-Krum"),
        ("rflpa", "RFLPA"),
        ("spmc", "SPMC"),
    ]),
    ("Mask-based", [
        ("skymask", "SkyMask"),
    ]),
]

# Defenses shown below the main table with a footnote (partial dataset coverage)
DEFENSE_FOOTNOTES = [
    ("Requires $N \\geq 4f{+}3$", [
        ("bulyan", "Bulyan"),
    ]),
]

DATASET_ORDER = [
    ("CIFAR100", "CIFAR-100"),
    ("FEMNIST", "FEMNIST"),
    ("Texas100", "Texas-100"),
    ("Purchase100", "Purchase-100"),
    ("TREC", "TREC"),
]

SCENARIO_RE = re.compile(
    r"step12_main_summary_(?P<defense>[a-z_]+?)_(?P<modality>image|text|tabular)_"
    r"(?P<dataset>\w+?)_(?P<model>\w+)$"
)


# =============================================================================
# Data collection
# =============================================================================

def collect_step10_results(results_dir: Path):
    """
    Collect results from step 10 directories.

    Returns:
        data: dict (defense, dataset) -> {metric_name: [values_across_seeds]}
        failures: dict (defense, dataset) -> failure reason string ("OOM", "T.O.", "ERR")
    """
    data = defaultdict(lambda: defaultdict(list))
    failures: Dict[Tuple[str, str], str] = {}

    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue

        defense = m.group("defense")
        dataset = m.group("dataset")
        key = (defense, dataset)

        has_success = False
        for metrics_file in scenario_dir.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                continue

            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            acc = metrics.get("acc")
            if acc is None:
                continue

            has_success = True
            data[key]["acc"].append(acc)
            asr = metrics.get("asr")
            if asr is not None:
                data[key]["asr"].append(asr)

            # BSR / MSR from marketplace_report.json
            report_file = run_dir / "marketplace_report.json"
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    sellers = report.get("seller_summaries", {})
                    ben_rates, adv_rates = [], []
                    for s in sellers.values():
                        sr = s.get("selection_rate")
                        if sr is None:
                            continue
                        if s.get("type") == "adversary":
                            adv_rates.append(sr)
                        else:
                            ben_rates.append(sr)
                    if ben_rates:
                        data[key]["bsr"].append(np.mean(ben_rates))
                    if adv_rates:
                        data[key]["msr"].append(np.mean(adv_rates))
                except (json.JSONDecodeError, OSError):
                    pass

        # Check for failures if no successful runs found
        if not has_success and key not in data:
            for failed_file in scenario_dir.rglob(".failed"):
                try:
                    err_text = failed_file.read_text(encoding="utf-8", errors="replace").lower()
                    if "out of memory" in err_text or "oom" in err_text or "cuda" in err_text:
                        failures[key] = "OOM"
                    elif "timeout" in err_text or "time" in err_text:
                        failures[key] = "T.O."
                    else:
                        failures[key] = "ERR"
                    break  # one failure marker is enough
                except OSError:
                    failures[key] = "ERR"
                    break

    return data, failures


def agg(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std)."""
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.mean(values)), float(np.std(values)))


# =============================================================================
# LaTeX formatting helpers
# =============================================================================

def _pct(val: float, bold: bool = False, color: Optional[str] = None) -> str:
    """Format a float as a percentage string for LaTeX."""
    if np.isnan(val):
        return "—"
    s = f"{val * 100:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if color:
        s = f"\\textcolor{{{color}}}{{{s}}}"
    return s


def _pct_std(mean: float, std: float, bold: bool = False) -> str:
    """Format mean ± std as a percentage string for LaTeX."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std) or std < 0.005:
        s = f"{mean * 100:.1f}"
    else:
        s = f"{mean * 100:.1f}{{\\scriptsize$\\pm${std * 100:.1f}}}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def _defense_row_label(def_label: str) -> str:
    return f"\\quad {def_label}"


# =============================================================================
# Identify best/worst per column for highlighting
# =============================================================================

def _compute_highlights(data, datasets, metric: str, higher_is_better: bool):
    """
    For each dataset column, find the best non-FedAvg defense value.
    Returns dict: (defense, dataset) -> 'best' | 'worst' | None
    """
    highlights = {}
    for ds_key, _ in datasets:
        values = {}
        for _, defenses in DEFENSE_GROUPS:
            for def_key, _ in defenses:
                if def_key == "fedavg":
                    continue
                vals = data.get((def_key, ds_key), {}).get(metric, [])
                if vals:
                    values[def_key] = float(np.mean(vals))

        if not values:
            continue

        best_def = max(values, key=values.get) if higher_is_better else min(values, key=values.get)
        highlights[(best_def, ds_key)] = "best"

    return highlights


# =============================================================================
# Panel generators
# =============================================================================

def _failure_cell(reason: str, n_metrics: int) -> str:
    """Generate a LaTeX cell spanning n_metrics columns for a failure reason."""
    label = {
        "OOM": r"\textsc{oom}",
        "T.O.": r"\textsc{t.o.}",
        "ERR": r"\textsc{err}",
    }.get(reason, "—")
    if n_metrics > 1:
        return f" & \\multicolumn{{{n_metrics}}}{{c}}{{{label}}}"
    return f" & {label}"


def _generate_panel_rows(data, datasets, metrics_spec, show_std, highlights_map,
                         failures=None):
    """
    Generate LaTeX rows for one panel.

    metrics_spec: list of (metric_key, higher_is_better)
    highlights_map: dict from _compute_highlights for bolding best values
    failures: dict (defense, dataset) -> "OOM" | "T.O." | "ERR"
    """
    if failures is None:
        failures = {}
    lines = []
    n_mcols = len(metrics_spec)

    for gi, (_, defenses) in enumerate(DEFENSE_GROUPS):
        if gi > 0:
            lines.append(r"\hdashline")

        for def_key, def_label in defenses:
            row = _defense_row_label(def_label)

            for ds_key, _ in datasets:
                key = (def_key, ds_key)

                # Check for failure first
                if key in failures and key not in data:
                    row += _failure_cell(failures[key], n_mcols)
                    continue

                cell_data = data.get(key, {})

                for metric_key, _ in metrics_spec:
                    vals = cell_data.get(metric_key, [])
                    if vals:
                        m, s = agg(vals)
                        is_best = highlights_map.get((metric_key, def_key, ds_key)) == "best"
                        if show_std:
                            row += f" & {_pct_std(m, s, bold=is_best)}"
                        else:
                            row += f" & {_pct(m, bold=is_best)}"
                    else:
                        row += " & —"

            row += " \\\\"
            lines.append(row)

    # Footnote defenses (e.g. Bulyan) — rendered below a midrule
    for fn_label, fn_defenses in DEFENSE_FOOTNOTES:
        has_any = any(
            data.get((dk, dsk), {}).get(metrics_spec[0][0], [])
            for dk, _ in fn_defenses
            for dsk, _ in datasets
        )
        if not has_any:
            continue

        lines.append(r"\midrule")
        for def_key, def_label in fn_defenses:
            row = f"{_defense_row_label(def_label)}$^\\dagger$"
            for ds_key, _ in datasets:
                key = (def_key, ds_key)
                if key in failures and key not in data:
                    row += _failure_cell(failures[key], n_mcols)
                    continue
                cell_data = data.get(key, {})
                for metric_key, _ in metrics_spec:
                    vals = cell_data.get(metric_key, [])
                    if vals:
                        m, s = agg(vals)
                        if show_std:
                            row += f" & {_pct_std(m, s)}"
                        else:
                            row += f" & {_pct(m)}"
                    else:
                        row += " & —"
            row += " \\\\"
            lines.append(row)

    return lines


def _build_highlights_map(data, datasets, metrics_spec):
    """Build combined highlights map across all metrics."""
    hmap = {}
    for metric_key, higher_is_better in metrics_spec:
        per_metric = _compute_highlights(data, datasets, metric_key, higher_is_better)
        for (def_key, ds_key), label in per_metric.items():
            hmap[(metric_key, def_key, ds_key)] = label
    return hmap


# =============================================================================
# Multi-panel table (default): three sub-tables stacked
# =============================================================================

def generate_latex_multipanel(data, datasets, output_path=None, show_std=False,
                              failures=None):
    """
    Generate a three-panel LaTeX table:
      Panel A — Model Performance (ACC)
      Panel B — Attack Effectiveness (ASR)
      Panel C — Selection Patterns (BSR, MSR)
    """
    n_ds = len(datasets)

    panels = [
        ("(a) Model Performance",
         [("acc", True)],
         "Higher accuracy = better model utility."),
        ("(b) Attack Effectiveness",
         [("asr", False)],
         "Lower ASR = better defense against backdoor."),
        ("(c) Selection Patterns",
         [("bsr", True), ("msr", False)],
         "Higher BSR = honest sellers included; lower MSR = adversaries excluded."),
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Marketplace Integrity Benchmark.} "
                 r"12 defenses evaluated across 5 datasets under backdoor attack "
                 r"(30\% adversarial sellers, 50\% poison rate). "
                 r"\textbf{Bold} = best among non-baseline defenses per column. "
                 r"$^\dagger$Partial coverage: Bulyan requires $N \geq 4f{+}3$ active "
                 r"sellers per round, infeasible on small-shard datasets.}")
    lines.append(r"\label{tab:main_benchmark}")
    lines.append(r"\small")

    for pi, (panel_title, metrics_spec, _) in enumerate(panels):
        n_mcols = len(metrics_spec)
        col_spec = "l" + "".join(f"|{'c' * n_mcols}" for _ in datasets)

        if pi > 0:
            lines.append(r"\vspace{0.5em}")

        lines.append("")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Panel title spanning all columns
        total_cols = 1 + n_ds * n_mcols
        lines.append(f"\\multicolumn{{{total_cols}}}{{l}}"
                     f"{{\\textbf{{{panel_title}}}}} \\\\")

        # Dataset header row
        hdr = ""
        for i, (_, ds_label) in enumerate(datasets):
            sep = "c" if i == 0 else "|c"
            if n_mcols > 1:
                hdr += f" & \\multicolumn{{{n_mcols}}}{{{sep}}}{{{ds_label}}}"
            else:
                hdr += f" & {ds_label}"
        lines.append(f"Defense{hdr} \\\\")

        # Metric sub-header (only if panel has >1 metric)
        if n_mcols > 1:
            metric_labels = {"acc": "ACC", "asr": "ASR", "bsr": "BSR", "msr": "MSR"}
            subhdr = ""
            for _ in datasets:
                for mk, _ in metrics_spec:
                    subhdr += f" & {metric_labels[mk]}"
            lines.append(f"{subhdr} \\\\")

        lines.append(r"\midrule")

        # Build highlights and rows
        hmap = _build_highlights_map(data, datasets, metrics_spec)
        row_lines = _generate_panel_rows(data, datasets, metrics_spec, show_std, hmap,
                                         failures=failures)
        lines.extend(row_lines)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")

    lines.append(r"\end{table*}")
    latex = "\n".join(lines)

    print("\n" + "=" * 70)
    print(" LaTeX Table (multi-panel)")
    print("=" * 70)
    print(latex)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding="utf-8")
        print(f"\nSaved to: {output_path}")

    return latex


# =============================================================================
# Compact single-panel table (--compact): all 4 metrics per dataset
# =============================================================================

def generate_latex_compact(data, datasets, output_path=None, show_std=False,
                           failures=None):
    """
    Single wide table: Defense | ACC ASR BSR MSR | ACC ASR BSR MSR | ...
    """
    n_ds = len(datasets)
    all_metrics = [("acc", True), ("asr", False), ("bsr", True), ("msr", False)]
    metric_labels = {"acc": "ACC", "asr": "ASR", "bsr": "BSR", "msr": "MSR"}
    col_spec = "l" + "|cccc" * n_ds

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Marketplace Integrity Benchmark.} "
                 r"12 defenses across 5 datasets under backdoor attack "
                 r"(30\% adversary, 50\% poison). "
                 r"\textbf{Bold} = best non-baseline defense per column. "
                 r"$^\dagger$Partial coverage: requires $N \geq 4f{+}3$.}")
    lines.append(r"\label{tab:main_benchmark}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Dataset header
    hdr = ""
    for i, (_, ds_label) in enumerate(datasets):
        sep = "|" if i > 0 else ""
        hdr += f" & \\multicolumn{{4}}{{{sep}c}}{{\\textbf{{{ds_label}}}}}"
    lines.append(f" {hdr} \\\\")

    # Metric header
    mhdr = "\\textbf{Defense}"
    for _ in datasets:
        for mk, _ in all_metrics:
            mhdr += f" & {metric_labels[mk]}"
    lines.append(f"{mhdr} \\\\")
    lines.append(r"\midrule")

    hmap = _build_highlights_map(data, datasets, all_metrics)

    for gi, (group_name, defenses) in enumerate(DEFENSE_GROUPS):
        if gi > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{1 + 4 * n_ds}}}{{l}}"
                     f"{{\\textit{{{group_name}}}}} \\\\")

        for def_key, def_label in defenses:
            row = _defense_row_label(def_label)
            for ds_key, _ in datasets:
                key = (def_key, ds_key)
                # Check for failure
                if failures and key in failures and key not in data:
                    row += _failure_cell(failures[key], 4)
                    continue
                cell_data = data.get(key, {})
                for mk, _ in all_metrics:
                    vals = cell_data.get(mk, [])
                    if vals:
                        m, s = agg(vals)
                        is_best = hmap.get((mk, def_key, ds_key)) == "best"
                        if show_std:
                            row += f" & {_pct_std(m, s, bold=is_best)}"
                        else:
                            row += f" & {_pct(m, bold=is_best)}"
                    else:
                        row += " & —"
            row += " \\\\"
            lines.append(row)

    # Footnote defenses (e.g. Bulyan) below the main body
    for fn_label, fn_defenses in DEFENSE_FOOTNOTES:
        has_any = any(
            data.get((dk, dsk), {}).get("acc", [])
            for dk, _ in fn_defenses for dsk, _ in datasets
        )
        if not has_any:
            continue
        lines.append(r"\midrule")
        for def_key, def_label in fn_defenses:
            row = f"{_defense_row_label(def_label)}$^\\dagger$"
            for ds_key, _ in datasets:
                key = (def_key, ds_key)
                if failures and key in failures and key not in data:
                    row += _failure_cell(failures[key], 4)
                    continue
                cell_data = data.get(key, {})
                for mk, _ in all_metrics:
                    vals = cell_data.get(mk, [])
                    if vals:
                        m, s = agg(vals)
                        if show_std:
                            row += f" & {_pct_std(m, s)}"
                        else:
                            row += f" & {_pct(m)}"
                    else:
                        row += " & —"
                row += " \\\\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)

    print("\n" + "=" * 70)
    print(" LaTeX Table (compact)")
    print("=" * 70)
    print(latex)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding="utf-8")
        print(f"\nSaved to: {output_path}")

    return latex


# =============================================================================
# ASCII preview
# =============================================================================

def print_ascii_table(data, datasets):
    """Print a readable ASCII summary."""
    metric_order = ["acc", "asr", "bsr", "msr"]
    header = f"{'Defense':<14}"
    for _, ds_label in datasets:
        header += f" | {'ACC':>5} {'ASR':>5} {'BSR':>5} {'MSR':>5}"
    print(header)
    print("-" * len(header))

    for _, defenses in DEFENSE_GROUPS:
        for def_key, def_label in defenses:
            row = f"{def_label:<14}"
            for ds_key, _ in datasets:
                key = (def_key, ds_key)
                vals = data.get(key, {})
                parts = []
                for mk in metric_order:
                    m, _ = agg(vals.get(mk, []))
                    parts.append(f"{m * 100:5.1f}" if not np.isnan(m) else "    —")
                row += " | " + " ".join(parts)
            print(row)
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate headline LaTeX table from Step 10 results.")
    parser.add_argument("--results_dir", default="./results",
                        help="Path to results directory")
    parser.add_argument("--output", default=None,
                        help="Output .tex file (default: figures/main_table.tex)")
    parser.add_argument("--show_std", action="store_true",
                        help="Show ± std in cells")
    parser.add_argument("--compact", action="store_true",
                        help="Single wide table instead of 3-panel layout")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting Step 10 results from: {results_dir}")
    data, failures = collect_step10_results(results_dir)

    if not data and not failures:
        print("ERROR: No step 10 results found.", file=sys.stderr)
        sys.exit(1)

    # Filter to datasets that have at least one result or failure
    available = set()
    for (_, ds), v in data.items():
        if v["acc"]:
            available.add(ds)
    for (_, ds) in failures:
        available.add(ds)
    datasets = [(k, v) for k, v in DATASET_ORDER if k in available]
    if not datasets:
        print("ERROR: No datasets with results.", file=sys.stderr)
        sys.exit(1)

    defenses_with_data = sorted({d for (d, _), v in data.items() if v["acc"]})
    print(f"Datasets: {[d[1] for d in datasets]}")
    print(f"Defenses: {defenses_with_data}")
    if failures:
        print(f"Failures: {dict(failures)}")

    # Coverage matrix
    print(f"\nCoverage (seeds per cell):")
    header = f"{'Defense':<14}"
    for _, ds_label in datasets:
        header += f" {ds_label:>12}"
    print(header)
    print("-" * len(header))
    for _, defenses in DEFENSE_GROUPS:
        for def_key, def_label in defenses:
            row = f"{def_label:<14}"
            for ds_key, _ in datasets:
                key = (def_key, ds_key)
                n = len(data.get(key, {}).get("acc", []))
                if n > 0:
                    row += f" {n:>12}"
                elif key in failures:
                    row += f" {failures[key]:>12}"
                else:
                    row += f" {'—':>12}"
            print(row)

    # ASCII preview
    print(f"\n{'=' * 60}")
    print(" ASCII Preview (mean %)")
    print("=" * 60)
    print_ascii_table(data, datasets)

    # LaTeX
    output_path = Path(args.output) if args.output else Path("figures/main_table.tex")
    if args.compact:
        generate_latex_compact(data, datasets, output_path, show_std=args.show_std,
                               failures=failures)
    else:
        generate_latex_multipanel(data, datasets, output_path, show_std=args.show_std,
                                  failures=failures)


if __name__ == "__main__":
    main()
