"""
extract_system_perf.py — Build a CSV of system-performance metrics across all
benchmark cells.

For each completed cell (one final_metrics.json), produces one CSV row with:

  scenario, dataset, defense, attack, modality, model, seed,
  acc, asr,
  completed_rounds,
  wall_clock_seconds,           # from final_metrics.json (NEW cells) or
                                  # derived from runner log timestamps (OLD)
  throughput_rounds_per_sec,    # derived
  avg_seconds_per_round,        # derived
  total_upload_mb,              # already in final_metrics.json
  avg_upload_mb_per_round,      # already
  peak_gpu_memory_allocated_mb, # NEW cells only (else NaN)
  peak_host_rss_mb,             # NEW cells only (else NaN)
  source                        # 'instrumented' | 'log_derived' | 'none'

Usage:
    python extract_system_perf.py                                # all step10
    python extract_system_perf.py --steps 3 10                   # step3 + step10
    python extract_system_perf.py --output tables/sys_perf.csv
    python extract_system_perf.py --dataset CIFAR100 --defense fltrust
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

RESULTS_DIR = Path("./results")
CONFIGS_DIR = Path("./configs_generated_benchmark")

# Step prefix → result-dir prefix mapping
STEP_PREFIX_MAP = {
    1:  "step1_tune_",
    3:  "step3_tune_",
    4:  "step5_atk_sens_",
    5:  "step6_adv_sybil_",
    6:  "step7_adaptive_",
    7:  "step8_buyer_attack_",
    8:  "step10_scalability_",
    9:  "step11_",  # heterogeneity uses per-defense prefixes
    10: "step12_main_summary_",
    13: "step13_drowning_",
    14: "step14_collusion_",
    15: "step15_pricing_",
    17: "step17_alie_",
    21: "step21_valuation_",
}

# Scenario name regexes — try the most general one first.
# Step 10 example:  step12_main_summary_fltrust_image_CIFAR100_cnn
# Step 3 example:   step3_tune_fltrust_backdoor_image_CIFAR100_cifar100_cnn
SCENARIO_RES = [
    re.compile(
        r"^step12_main_summary_(?P<defense>.+?)_"
        r"(?P<modality>image|tabular|text)_"
        r"(?P<dataset>CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)_"
        r"(?P<model>.+)$"
    ),
    re.compile(
        r"^step16_bagdasaryan_(?P<defense>.+?)_"
        r"(?P<modality>image|tabular|text)_"
        r"(?P<dataset>CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)_"
        r"(?P<model>.+)$"
    ),
    re.compile(
        r"^step3_tune_(?P<defense>.+?)_"
        r"(?P<attack>backdoor|labelflip)_"
        r"(?P<modality>image|tabular|text)_"
        r"(?P<dataset>CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)_"
        r"(?P<model>.+)$"
    ),
]

# Match runner log timestamps like "2026-04-09 22:26:09,541"
LOG_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def parse_scenario_name(name: str) -> Optional[dict]:
    for rx in SCENARIO_RES:
        m = rx.match(name)
        if m:
            d = m.groupdict()
            d.setdefault("attack", "backdoor")
            return d
    return None


def parse_log_timestamp(line: str) -> Optional[float]:
    """Return unix epoch seconds for the timestamp at the start of a log line."""
    m = LOG_TS_RE.match(line)
    if not m:
        return None
    import datetime as _dt
    try:
        dt = _dt.datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)),
            int(m.group(7)) * 1000,
        )
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def derive_wall_clock_from_log(run_dir: Path) -> Optional[Tuple[float, float, float]]:
    """For old cells with no instrumented start_timestamp, derive (start, end,
    duration_seconds) from the cell's per-cell runner log if it exists.

    Strategy:
      1. look for a *.log file inside run_dir or its parents
      2. read the first and last log lines that have a recognized timestamp
      3. return (start, end, end - start)

    If we can't find or parse a log, return None.
    """
    # Look in run_dir then walk up to scenario root
    candidates = []
    cur = run_dir
    for _ in range(8):  # walk up at most 8 levels
        candidates.extend(sorted(cur.glob("*.log")))
        if cur.parent == cur:
            break
        cur = cur.parent

    for log_path in candidates:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue
        if not lines:
            continue

        # First line with a timestamp
        start_ts = None
        for line in lines[:200]:
            t = parse_log_timestamp(line)
            if t:
                start_ts = t
                break
        if start_ts is None:
            continue

        # Last line with a timestamp
        end_ts = None
        for line in reversed(lines[-1000:]):
            t = parse_log_timestamp(line)
            if t:
                end_ts = t
                break
        if end_ts is None or end_ts <= start_ts:
            continue

        return (start_ts, end_ts, end_ts - start_ts)

    return None


def collect_rows(steps: list, dataset_filter: Optional[str],
                 defense_filter: Optional[str]) -> list:
    """Walk results/ and yield one dict per completed cell."""
    rows = []
    prefixes = [STEP_PREFIX_MAP.get(s) for s in steps]
    prefixes = [p for p in prefixes if p]
    if not prefixes:
        return rows

    for scenario_dir in sorted(RESULTS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if not any(scenario_dir.name.startswith(p) for p in prefixes):
            continue

        meta = parse_scenario_name(scenario_dir.name)
        if not meta:
            continue

        if dataset_filter and meta.get("dataset") != dataset_filter:
            continue
        if defense_filter and meta.get("defense") != defense_filter:
            continue

        for metrics_path in scenario_dir.rglob("final_metrics.json"):
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            if metrics.get("acc") is None:
                continue

            run_dir = metrics_path.parent

            # Extract seed from path
            seed = None
            for part in run_dir.parts[::-1]:
                m = re.match(r"run_\d+_seed_(\d+)", part)
                if m:
                    seed = int(m.group(1))
                    break

            # Wall clock — three sources, in priority order
            wall_clock = metrics.get("wall_clock_seconds")
            source = "instrumented" if wall_clock is not None else None

            if wall_clock is None:
                derived = derive_wall_clock_from_log(run_dir)
                if derived is not None:
                    _, _, wall_clock = derived
                    source = "log_derived"

            if source is None:
                source = "none"

            completed_rounds = metrics.get("completed_rounds")
            try:
                cr_int = int(completed_rounds) if completed_rounds is not None else None
            except (TypeError, ValueError):
                cr_int = None

            throughput = metrics.get("throughput_rounds_per_sec")
            avg_sec_per_round = metrics.get("avg_seconds_per_round")
            if throughput is None and wall_clock and cr_int and wall_clock > 0:
                throughput = cr_int / wall_clock
                avg_sec_per_round = wall_clock / cr_int

            rows.append({
                "scenario":                     scenario_dir.name,
                "dataset":                      meta.get("dataset"),
                "defense":                      meta.get("defense"),
                "attack":                       meta.get("attack", "backdoor"),
                "modality":                     meta.get("modality"),
                "model":                        meta.get("model"),
                "seed":                         seed if seed is not None else "",
                "acc":                          metrics.get("acc"),
                "asr":                          metrics.get("asr"),
                "completed_rounds":             cr_int if cr_int is not None else "",
                "wall_clock_seconds":           round(wall_clock, 3) if wall_clock else "",
                "throughput_rounds_per_sec":    round(throughput, 4) if throughput else "",
                "avg_seconds_per_round":        round(avg_sec_per_round, 4) if avg_sec_per_round else "",
                "total_upload_mb":              metrics.get("total_upload_mb", ""),
                "avg_upload_mb_per_round":      metrics.get("avg_upload_mb_per_round", ""),
                "total_upload_mb_accepted":     metrics.get("total_upload_mb_accepted", ""),
                "peak_gpu_memory_allocated_mb": metrics.get("peak_gpu_memory_allocated_mb", ""),
                "peak_gpu_memory_reserved_mb":  metrics.get("peak_gpu_memory_reserved_mb", ""),
                "peak_host_rss_mb":             metrics.get("peak_host_rss_mb", ""),
                "gpu_device_name":              metrics.get("gpu_device_name", ""),
                "source":                       source,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", type=int, default=[10],
                    help="Step numbers to scan (default: 10)")
    ap.add_argument("--dataset", default=None, help="Filter to one dataset")
    ap.add_argument("--defense", default=None, help="Filter to one defense")
    ap.add_argument("--output", default="system_perf.csv",
                    help="Output CSV path (default: system_perf.csv)")
    args = ap.parse_args()

    rows = collect_rows(args.steps, args.dataset, args.defense)
    if not rows:
        print("No completed cells found matching the filters.")
        return

    fieldnames = list(rows[0].keys())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {out_path}")

    # Quick summary
    by_source = defaultdict(int)
    for r in rows:
        by_source[r["source"]] += 1
    print(f"  Source breakdown:")
    for src, n in sorted(by_source.items()):
        print(f"    {src:<14}: {n}")

    # Mean/median throughput per dataset
    by_dataset = defaultdict(list)
    for r in rows:
        wc = r.get("wall_clock_seconds")
        if isinstance(wc, (int, float)) and wc > 0:
            by_dataset[r["dataset"]].append(wc)
    if by_dataset:
        print(f"  Wall-clock distribution per dataset (seconds):")
        print(f"    {'dataset':<14}{'n':>5}{'min':>10}{'median':>10}{'mean':>10}{'max':>10}")
        for ds in sorted(by_dataset):
            vals = by_dataset[ds]
            vals_sorted = sorted(vals)
            mid = vals_sorted[len(vals) // 2]
            print(f"    {ds:<14}{len(vals):>5}{min(vals):>10.1f}{mid:>10.1f}"
                  f"{sum(vals)/len(vals):>10.1f}{max(vals):>10.1f}")


if __name__ == "__main__":
    main()
