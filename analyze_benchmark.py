"""
Benchmark Results Analyzer
==========================
Comprehensive status and performance analysis for the rerun pipeline.
Reports:
  1. Completion status per step/phase
  2. Defense filtering performance comparison (acc, asr, bsr, msr)
  3. Per-dataset breakdown
  4. Issues and warnings
  5. Key takeaways for paper

Usage:
    python analyze_benchmark.py
    python analyze_benchmark.py --results_dir ./results
    python analyze_benchmark.py --step 10
    python analyze_benchmark.py --csv results_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFENSE_NAMES = {
    "fedavg": "FedAvg", "fltrust": "FLTrust", "martfl": "MartFL",
    "skymask": "SkyMask", "skymask_small": "SkyMask-S",
    "trimmed_mean": "Trim-Mean", "multi_krum": "Multi-Krum",
    "rflpa": "RFLPA", "spmc": "SPMC", "daved": "DAVED",
}
DEFENSE_ORDER = [
    "FedAvg", "FLTrust", "MartFL", "SkyMask", "SkyMask-S",
    "Trim-Mean", "Multi-Krum", "RFLPA", "SPMC", "DAVED",
]
DATASET_NAMES = {
    "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "femnist": "FEMNIST",
    "texas100": "Texas-100", "purchase100": "Purchase-100", "trec": "TREC",
}

# Map Python --step numbers to results directory prefixes
STEP_INFO = {
    "step1":  {"phase": 1, "label": "IID Baseline Tuning"},
    "step3":  {"phase": 1, "label": "Defense HP Tuning"},
    "step5":  {"phase": 3, "label": "Attack Sensitivity"},
    "step6":  {"phase": 3, "label": "Sybil Strategies"},
    "step7":  {"phase": 3, "label": "Adaptive Attacks"},
    "step8":  {"phase": 3, "label": "Buyer Attacks"},
    "step10": {"phase": 4, "label": "Scalability"},
    "step11": {"phase": 3, "label": "Heterogeneity"},
    "step12": {"phase": 2, "label": "Main Summary"},
    "step13": {"phase": 4, "label": "Drowning Attack"},
    "step14": {"phase": 3, "label": "MartFL Collusion"},
    "step15": {"phase": 4, "label": "Proportional Pricing"},
    "step16": {"phase": 4, "label": "DP Fairness"},
    "step17": {"phase": 4, "label": "ALIE Attack"},
    "step18": {"phase": 4, "label": "DAVED Comparison"},
}

PHASE_NAMES = {
    1: "Foundation (Steps 1+3)",
    2: "Main Summary (Step 10)",
    3: "Deep-Dive Analysis",
    4: "Remaining Steps",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt(name: str) -> str:
    if not name:
        return "Unknown"
    low = name.lower().replace("-", "").replace(" ", "")
    for code, pretty in {**DEFENSE_NAMES, **DATASET_NAMES}.items():
        if low == code.lower().replace("_", ""):
            return pretty
    return name


def extract_defense(path_str: str) -> Optional[str]:
    path_lower = path_str.lower().replace("\\", "/")
    for code in sorted(DEFENSE_NAMES.keys(), key=len, reverse=True):
        if f"_{code}_" in path_lower or f"/{code}/" in path_lower or f"_{code}/" in path_lower:
            return DEFENSE_NAMES[code]
    # Fallback: substring match
    for code in sorted(DEFENSE_NAMES.keys(), key=len, reverse=True):
        if code in path_lower:
            return DEFENSE_NAMES[code]
    return None


def extract_dataset(path_str: str) -> Optional[str]:
    path_lower = path_str.lower().replace("\\", "/")
    for code in sorted(DATASET_NAMES.keys(), key=len, reverse=True):
        if code in path_lower:
            return DATASET_NAMES[code]
    return None


def extract_step(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.lower().startswith("step"):
            # Return just the stepN prefix (e.g., "step12" from "step12_main_summary_...")
            for key in sorted(STEP_INFO.keys(), key=lambda k: -len(k)):
                if part.lower().startswith(key):
                    return key
            # Fallback: extract stepN
            import re
            m = re.match(r"(step\d+)", part.lower())
            if m:
                return m.group(1)
    return None


def extract_attack(path_str: str) -> Optional[str]:
    path_lower = path_str.lower()
    attacks = {
        "backdoor": "Backdoor", "labelflip": "Label Flip", "label_flip": "Label Flip",
        "min_max": "Min-Max", "min_sum": "Min-Sum", "alie": "ALIE",
        "drowning": "Drowning", "scaling": "Scaling", "dba": "DBA",
    }
    for code in sorted(attacks.keys(), key=len, reverse=True):
        if code in path_lower:
            return attacks[code]
    return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> Dict[str, Any]:
    """Load metrics from a single experiment run."""
    rec = {
        "run_dir": str(run_dir),
        "status": "incomplete",
        "defense": extract_defense(str(run_dir)),
        "dataset": extract_dataset(str(run_dir)),
        "step": extract_step(run_dir),
        "attack": extract_attack(str(run_dir)),
        "acc": None, "asr": None, "bsr": None, "msr": None,
        "completed_rounds": None,
        "issues": [],
    }

    # Status markers
    if (run_dir / ".success").exists():
        rec["status"] = "success"
    elif (run_dir / ".failed").exists():
        rec["status"] = "failed"
    elif (run_dir / ".in_progress").exists():
        rec["status"] = "in_progress"

    # final_metrics.json
    mf = run_dir / "final_metrics.json"
    if mf.exists():
        try:
            with open(mf) as f:
                m = json.load(f)
            acc = m.get("acc", m.get("test_accuracy", None))
            asr = m.get("asr", m.get("test_asr", None))
            if acc is not None:
                rec["acc"] = acc * 100 if acc <= 1.0 else acc
            if asr is not None:
                rec["asr"] = asr * 100 if asr <= 1.0 else asr
            rec["completed_rounds"] = m.get("completed_rounds", None)
        except Exception:
            rec["issues"].append("CORRUPT_METRICS")

    # marketplace_report.json
    rf = run_dir / "marketplace_report.json"
    if rf.exists():
        try:
            with open(rf) as f:
                report = json.load(f)
            sellers = list(report.get("seller_summaries", {}).values())
            ben = [s for s in sellers if s.get("type") == "benign"]
            adv = [s for s in sellers if s.get("type") == "adversary"]
            if ben:
                bsr = np.mean([s["selection_rate"] for s in ben])
                rec["bsr"] = bsr * 100 if bsr <= 1.0 else bsr
            if adv:
                msr = np.mean([s["selection_rate"] for s in adv])
                rec["msr"] = msr * 100 if msr <= 1.0 else msr
        except Exception:
            pass

    # Issue flags
    if rec["acc"] is not None and rec["acc"] < 5.0:
        rec["issues"].append("VERY_LOW_ACC")
    if rec["asr"] is not None and rec["asr"] > 95.0:
        rec["issues"].append("VERY_HIGH_ASR")
    if rec["status"] == "success" and rec["acc"] is None:
        rec["issues"].append("SUCCESS_NO_METRICS")

    return rec


def find_all_runs(results_dir: Path) -> List[Path]:
    runs = set()
    for p in results_dir.rglob("run_*_seed_*"):
        if p.is_dir():
            runs.add(p)
    if not runs:
        for p in results_dir.rglob("final_metrics.json"):
            runs.add(p.parent)
    return sorted(runs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _avg(values):
    return np.mean(values) if values else float("nan")


def _std(values):
    return np.std(values) if len(values) > 1 else 0.0


def print_completion_status(runs: List[Dict]):
    """Section 1: What's done, what's pending."""
    print(f"\n{'='*72}")
    print(f"  1. COMPLETION STATUS  ({len(runs)} total runs)")
    print(f"{'='*72}")

    # Overall
    by_status = defaultdict(int)
    for r in runs:
        by_status[r["status"]] += 1
    icons = {"success": "+", "failed": "X", "in_progress": "~", "incomplete": "?"}
    for s in ["success", "failed", "in_progress", "incomplete"]:
        if by_status[s]:
            print(f"    [{icons.get(s,'?')}] {s:<14s} {by_status[s]:>5d}")

    # Per-phase
    by_phase = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0})
    for r in runs:
        step = r["step"]
        info = STEP_INFO.get(step, {"phase": 0})
        phase = info["phase"]
        by_phase[phase]["total"] += 1
        if r["status"] == "success":
            by_phase[phase]["ok"] += 1
        elif r["status"] == "failed":
            by_phase[phase]["fail"] += 1

    print(f"\n    {'Phase':<35s} {'Done':>6s} {'Fail':>6s} {'Total':>6s} {'%':>6s}")
    print(f"    {'-'*65}")
    for phase in sorted(by_phase.keys()):
        if phase == 0:
            continue
        d = by_phase[phase]
        pct = 100.0 * d["ok"] / d["total"] if d["total"] else 0
        name = PHASE_NAMES.get(phase, f"Phase {phase}")
        print(f"    {name:<35s} {d['ok']:>6d} {d['fail']:>6d} {d['total']:>6d} {pct:>5.1f}%")

    # Per-step
    by_step = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0})
    for r in runs:
        step = r["step"] or "unknown"
        by_step[step]["total"] += 1
        if r["status"] == "success":
            by_step[step]["ok"] += 1
        elif r["status"] == "failed":
            by_step[step]["fail"] += 1

    print(f"\n    {'Step':<20s} {'Label':<25s} {'Done':>5s} {'Fail':>5s} {'Total':>5s}")
    print(f"    {'-'*65}")
    for step in sorted(by_step.keys()):
        d = by_step[step]
        label = STEP_INFO.get(step, {}).get("label", "")
        print(f"    {step:<20s} {label:<25s} {d['ok']:>5d} {d['fail']:>5d} {d['total']:>5d}")


def print_defense_comparison(runs: List[Dict]):
    """Section 2: Defense filtering performance comparison."""
    print(f"\n{'='*72}")
    print(f"  2. DEFENSE PERFORMANCE COMPARISON")
    print(f"{'='*72}")

    ok_runs = [r for r in runs if r["status"] == "success" and r["acc"] is not None]
    if not ok_runs:
        print("    No completed runs with metrics found.")
        return

    by_defense = defaultdict(list)
    for r in ok_runs:
        d = r["defense"] or "Unknown"
        by_defense[d].append(r)

    # Sort by DEFENSE_ORDER
    ordered = sorted(by_defense.keys(),
                     key=lambda x: DEFENSE_ORDER.index(x) if x in DEFENSE_ORDER else 999)

    print(f"\n    Overall (all steps, all datasets):")
    print(f"    {'Defense':<12s} {'Runs':>5s} {'Acc%':>8s} {'ASR%':>8s} {'BSR%':>8s} {'MSR%':>8s}  Verdict")
    print(f"    {'-'*72}")
    for d in ordered:
        group = by_defense[d]
        accs = [r["acc"] for r in group if r["acc"] is not None]
        asrs = [r["asr"] for r in group if r["asr"] is not None]
        bsrs = [r["bsr"] for r in group if r["bsr"] is not None]
        msrs = [r["msr"] for r in group if r["msr"] is not None]

        avg_acc = _avg(accs)
        avg_asr = _avg(asrs)
        avg_bsr = _avg(bsrs)
        avg_msr = _avg(msrs)

        # Verdict
        verdict = ""
        if not np.isnan(avg_asr) and not np.isnan(avg_acc):
            if avg_asr < 10 and avg_acc > 50:
                verdict = "Strong"
            elif avg_asr < 20 and avg_acc > 40:
                verdict = "Good"
            elif avg_asr > 50:
                verdict = "Weak (high ASR)"
            elif avg_acc < 20:
                verdict = "Weak (low acc)"

        def f(v):
            return f"{v:>7.1f}%" if not np.isnan(v) else "     N/A"

        print(f"    {d:<12s} {len(group):>5d} {f(avg_acc)} {f(avg_asr)} {f(avg_bsr)} {f(avg_msr)}  {verdict}")


def print_main_summary_table(runs: List[Dict]):
    """Section 3: Main benchmark table (Step 10 / step12 results) — the key paper figure."""
    step12_runs = [r for r in runs if r["step"] == "step12" and r["status"] == "success" and r["acc"] is not None]
    if not step12_runs:
        print(f"\n{'='*72}")
        print(f"  3. MAIN BENCHMARK TABLE (Step 10)")
        print(f"{'='*72}")
        print("    No step12 (main summary) results available yet.")
        return

    print(f"\n{'='*72}")
    print(f"  3. MAIN BENCHMARK TABLE (Step 10 — Figure 3)")
    print(f"{'='*72}")

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in step12_runs if r["dataset"]))

    for ds in datasets:
        ds_runs = [r for r in step12_runs if r["dataset"] == ds]
        by_defense = defaultdict(list)
        for r in ds_runs:
            by_defense[r["defense"] or "Unknown"].append(r)

        ordered = sorted(by_defense.keys(),
                         key=lambda x: DEFENSE_ORDER.index(x) if x in DEFENSE_ORDER else 999)

        print(f"\n    {ds}:")
        print(f"    {'Defense':<12s} {'Runs':>4s} {'Acc':>9s} {'ASR':>9s} {'BSR':>9s} {'MSR':>9s}")
        print(f"    {'-'*52}")

        for d in ordered:
            group = by_defense[d]
            accs = [r["acc"] for r in group if r["acc"] is not None]
            asrs = [r["asr"] for r in group if r["asr"] is not None]
            bsrs = [r["bsr"] for r in group if r["bsr"] is not None]
            msrs = [r["msr"] for r in group if r["msr"] is not None]

            def fs(vals):
                if not vals:
                    return "      N/A"
                return f"{_avg(vals):>5.1f}±{_std(vals):.1f}"

            print(f"    {d:<12s} {len(group):>4d} {fs(accs):>9s} {fs(asrs):>9s} {fs(bsrs):>9s} {fs(msrs):>9s}")


def print_dataset_breakdown(runs: List[Dict]):
    """Section 4: Per-dataset performance."""
    ok_runs = [r for r in runs if r["status"] == "success" and r["acc"] is not None]
    if not ok_runs:
        return

    print(f"\n{'='*72}")
    print(f"  4. PER-DATASET BREAKDOWN")
    print(f"{'='*72}")

    by_ds = defaultdict(list)
    for r in ok_runs:
        by_ds[r["dataset"] or "Unknown"].append(r)

    print(f"\n    {'Dataset':<14s} {'Runs':>5s} {'Avg Acc':>8s} {'Avg ASR':>8s} {'Defenses':>8s} {'Steps':>6s}")
    print(f"    {'-'*55}")
    for ds in sorted(by_ds.keys()):
        group = by_ds[ds]
        accs = [r["acc"] for r in group if r["acc"] is not None]
        asrs = [r["asr"] for r in group if r["asr"] is not None]
        n_defenses = len(set(r["defense"] for r in group if r["defense"]))
        n_steps = len(set(r["step"] for r in group if r["step"]))
        avg_acc = _avg(accs)
        avg_asr = _avg(asrs)

        def f(v):
            return f"{v:>7.1f}%" if not np.isnan(v) else "     N/A"

        print(f"    {ds:<14s} {len(group):>5d} {f(avg_acc)} {f(avg_asr)} {n_defenses:>8d} {n_steps:>6d}")


def print_deep_dive_summary(runs: List[Dict]):
    """Section 5: Deep-dive steps summary."""
    deep_dive_steps = {"step5", "step6", "step7", "step8", "step10", "step11", "step14"}
    dd_runs = [r for r in runs if r["step"] in deep_dive_steps and r["status"] == "success" and r["acc"] is not None]
    if not dd_runs:
        return

    print(f"\n{'='*72}")
    print(f"  5. DEEP-DIVE ANALYSIS HIGHLIGHTS")
    print(f"{'='*72}")

    for step in sorted(deep_dive_steps):
        step_runs = [r for r in dd_runs if r["step"] == step]
        if not step_runs:
            continue

        label = STEP_INFO.get(step, {}).get("label", step)
        n_runs = len(step_runs)
        n_defenses = len(set(r["defense"] for r in step_runs if r["defense"]))
        accs = [r["acc"] for r in step_runs if r["acc"] is not None]
        asrs = [r["asr"] for r in step_runs if r["asr"] is not None]

        print(f"\n    {step} — {label} ({n_runs} runs, {n_defenses} defenses)")

        # Per-defense for this step
        by_def = defaultdict(list)
        for r in step_runs:
            by_def[r["defense"] or "Unknown"].append(r)

        ordered = sorted(by_def.keys(),
                         key=lambda x: DEFENSE_ORDER.index(x) if x in DEFENSE_ORDER else 999)

        print(f"      {'Defense':<12s} {'Runs':>4s} {'Acc%':>7s} {'ASR%':>7s} {'BSR%':>7s} {'MSR%':>7s}")
        print(f"      {'-'*48}")
        for d in ordered:
            group = by_def[d]
            accs_d = [r["acc"] for r in group if r["acc"] is not None]
            asrs_d = [r["asr"] for r in group if r["asr"] is not None]
            bsrs_d = [r["bsr"] for r in group if r["bsr"] is not None]
            msrs_d = [r["msr"] for r in group if r["msr"] is not None]

            def f(vals):
                return f"{_avg(vals):>6.1f}" if vals else "   N/A"

            print(f"      {d:<12s} {len(group):>4d} {f(accs_d)} {f(asrs_d)} {f(bsrs_d)} {f(msrs_d)}")


def print_issues(runs: List[Dict]):
    """Section 6: Warnings and issues."""
    issues = []
    for r in runs:
        for iss in r["issues"]:
            issues.append((iss, r["defense"], r["dataset"], r["step"]))

    if not issues:
        return

    print(f"\n{'='*72}")
    print(f"  6. ISSUES & WARNINGS ({len(issues)} total)")
    print(f"{'='*72}")

    by_type = defaultdict(int)
    for iss, _, _, _ in issues:
        by_type[iss.split(":")[0]] += 1

    for tag, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    [{count:>4d}] {tag}")

    # Show a few examples of failures
    failed = [r for r in runs if r["status"] == "failed"]
    if failed:
        print(f"\n    Failed runs ({len(failed)}):")
        for r in failed[:5]:
            print(f"      {r['step'] or '?':<10s} {r['defense'] or '?':<12s} {r['dataset'] or '?':<12s}  {r['run_dir']}")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")


def print_key_findings(runs: List[Dict]):
    """Section 7: Auto-generated key takeaways."""
    ok = [r for r in runs if r["status"] == "success" and r["acc"] is not None]
    if not ok:
        return

    print(f"\n{'='*72}")
    print(f"  7. KEY FINDINGS")
    print(f"{'='*72}")

    # Best defense by acc-asr tradeoff (on step12 if available, else all)
    step12 = [r for r in ok if r["step"] == "step12"]
    pool = step12 if step12 else ok

    by_defense = defaultdict(list)
    for r in pool:
        if r["defense"] and r["asr"] is not None:
            by_defense[r["defense"]].append(r)

    if by_defense:
        scores = {}
        for d, group in by_defense.items():
            avg_acc = _avg([r["acc"] for r in group if r["acc"] is not None])
            avg_asr = _avg([r["asr"] for r in group if r["asr"] is not None])
            avg_msr = _avg([r["msr"] for r in group if r["msr"] is not None])
            if not np.isnan(avg_acc) and not np.isnan(avg_asr):
                scores[d] = {"acc": avg_acc, "asr": avg_asr, "msr": avg_msr, "score": avg_acc - avg_asr}

        if scores:
            ranked = sorted(scores.items(), key=lambda x: -x[1]["score"])
            source = "Main Summary (step12)" if step12 else "All completed runs"
            print(f"\n    Defense ranking by (Acc - ASR) [{source}]:")
            print(f"    {'Rank':<5s} {'Defense':<12s} {'Acc%':>7s} {'ASR%':>7s} {'MSR%':>7s} {'Score':>7s}")
            print(f"    {'-'*48}")
            for i, (d, s) in enumerate(ranked, 1):
                msr_str = f"{s['msr']:>6.1f}" if not np.isnan(s["msr"]) else "   N/A"
                print(f"    {i:<5d} {d:<12s} {s['acc']:>6.1f} {s['asr']:>6.1f} {msr_str} {s['score']:>6.1f}")

            best = ranked[0]
            worst = ranked[-1]
            print(f"\n    Best:  {best[0]} (acc={best[1]['acc']:.1f}%, asr={best[1]['asr']:.1f}%)")
            print(f"    Worst: {worst[0]} (acc={worst[1]['acc']:.1f}%, asr={worst[1]['asr']:.1f}%)")

    # FedAvg as baseline comparison
    fedavg_runs = [r for r in pool if r["defense"] == "FedAvg"]
    non_fedavg = [r for r in pool if r["defense"] and r["defense"] != "FedAvg"]
    if fedavg_runs and non_fedavg:
        fa_acc = _avg([r["acc"] for r in fedavg_runs if r["acc"] is not None])
        fa_asr = _avg([r["asr"] for r in fedavg_runs if r["asr"] is not None])
        def_accs = [r["acc"] for r in non_fedavg if r["acc"] is not None]
        def_asrs = [r["asr"] for r in non_fedavg if r["asr"] is not None]
        if not np.isnan(fa_asr) and def_asrs:
            print(f"\n    FedAvg baseline:  acc={fa_acc:.1f}%, asr={fa_asr:.1f}%")
            print(f"    Avg w/ defenses:  acc={_avg(def_accs):.1f}%, asr={_avg(def_asrs):.1f}%")
            if fa_asr > 0:
                reduction = ((fa_asr - _avg(def_asrs)) / fa_asr) * 100
                print(f"    ASR reduction:    {reduction:.1f}% relative to no defense")


def export_csv(runs: List[Dict], csv_path: str):
    """Export all results to CSV for further analysis."""
    import csv
    fields = ["step", "defense", "dataset", "attack", "status",
              "acc", "asr", "bsr", "msr", "completed_rounds", "run_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r.get(k) for k in fields})
    print(f"\n  Exported {len(runs)} runs to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--step", type=str, default=None,
                        help="Filter to step prefix (e.g., 'step12', '12')")
    parser.add_argument("--defense", type=str, default=None,
                        help="Filter to defense (e.g., 'martfl')")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to dataset (e.g., 'cifar100')")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    print(f"Scanning: {results_dir}")
    run_paths = find_all_runs(results_dir)
    print(f"Found {len(run_paths)} experiment runs")

    if not run_paths:
        print("\nNo results found. Run experiments first:")
        print("  bash rerun_priority.sh")
        sys.exit(0)

    runs = [load_run(p) for p in run_paths]

    # Filters
    if args.step:
        sf = args.step if args.step.startswith("step") else f"step{args.step}"
        runs = [r for r in runs if r["step"] and sf in r["step"]]
        print(f"Filtered to step '{sf}': {len(runs)} runs")

    if args.defense:
        df = args.defense.lower()
        runs = [r for r in runs if r["defense"] and df in r["defense"].lower()]
        print(f"Filtered to defense '{args.defense}': {len(runs)} runs")

    if args.dataset:
        dsf = args.dataset.lower()
        runs = [r for r in runs if r["dataset"] and dsf in r["dataset"].lower()]
        print(f"Filtered to dataset '{args.dataset}': {len(runs)} runs")

    # Reports
    print_completion_status(runs)
    print_defense_comparison(runs)
    print_main_summary_table(runs)
    print_dataset_breakdown(runs)
    print_deep_dive_summary(runs)
    print_issues(runs)
    print_key_findings(runs)

    if args.csv:
        export_csv(runs, args.csv)

    total = len(runs)
    ok = sum(1 for r in runs if r["status"] == "success")
    print(f"\n{'='*72}")
    print(f"  {ok}/{total} runs completed ({100*ok/total:.0f}%)" if total else "  No runs found")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
