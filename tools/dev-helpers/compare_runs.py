"""Compare two run directories for bit-for-bit metric equality.

Used to confirm the E1-E4 extensions leave default behavior unchanged:
all model/governance metrics must match exactly; only wall-clock/system
fields (timestamps, durations, latency, memory) are excluded.

    python tools/dev-helpers/compare_runs.py <before_run_dir> <after_run_dir>
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

TIME_COLS = {
    'timestamp', 'duration_sec', 'aggregation_latency_sec',
    'aggregation_peak_vram_mb',
}
TIME_JSON_KEYS = {
    'timestamp', 'start_timestamp', 'wall_clock_seconds',
    'throughput_rounds_per_sec', 'avg_seconds_per_round',
    'peak_gpu_memory_allocated_mb', 'peak_gpu_memory_reserved_mb',
    'gpu_device_index', 'gpu_device_name', 'peak_host_rss_mb',
    'avg_aggregation_latency_sec', 'max_aggregation_latency_sec',
    'median_aggregation_latency_sec', 'max_aggregation_peak_vram_mb',
}

failures = []


def check(name, ok, detail=""):
    status = "OK " if ok else "DIFF"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def compare_csv(a: Path, b: Path, name: str):
    da, db = pd.read_csv(a), pd.read_csv(b)
    cols_a = [c for c in da.columns if c not in TIME_COLS]
    cols_b = [c for c in db.columns if c not in TIME_COLS]
    if cols_a != cols_b:
        check(f"{name} columns", False, f"{cols_a} vs {cols_b}")
        return
    check(f"{name} columns", True)
    da, db = da[cols_a], db[cols_b]
    if len(da) != len(db):
        check(f"{name} rows", False, f"{len(da)} vs {len(db)}")
        return
    same = True
    for c in cols_a:
        va, vb = da[c], db[c]
        if va.dtype.kind in 'fc' or vb.dtype.kind in 'fc':
            eq = ((va.isna() & vb.isna()) | (va == vb)).all()
        else:
            eq = (va.fillna('<na>') == vb.fillna('<na>')).all()
        if not eq:
            same = False
            check(f"{name}[{c}]", False)
    if same:
        check(f"{name} values ({len(da)} rows)", True)


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items() if k not in TIME_JSON_KEYS}
    if isinstance(obj, list):
        return [_strip_times(x) for x in obj]
    return obj


def compare_json(a: Path, b: Path, name: str):
    ja = _strip_times(json.load(open(a)))
    jb = _strip_times(json.load(open(b)))
    check(name, ja == jb, "" if ja == jb else f"keys a-only={set(ja) - set(jb)}, b-only={set(jb) - set(ja)}")
    if ja != jb and set(ja) == set(jb):
        for k in ja:
            if ja[k] != jb[k]:
                print(f"      {k}: {ja[k]!r} vs {jb[k]!r}")


def compare_jsonl(a: Path, b: Path, name: str):
    la = [json.loads(x) for x in open(a) if x.strip()]
    lb = [json.loads(x) for x in open(b) if x.strip()]
    if len(la) != len(lb):
        check(name, False, f"{len(la)} vs {len(lb)} entries")
        return
    same = all(_strip_times(x) == _strip_times(y) for x, y in zip(la, lb))
    check(f"{name} ({len(la)} entries)", same)


def main():
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"BEFORE: {before}\nAFTER : {after}\n")
    compare_csv(before / "training_log.csv", after / "training_log.csv", "training_log.csv")
    compare_csv(before / "seller_metrics.csv", after / "seller_metrics.csv", "seller_metrics.csv")
    compare_csv(before / "round_aggregates.csv", after / "round_aggregates.csv", "round_aggregates.csv")
    compare_json(before / "final_metrics.json", after / "final_metrics.json", "final_metrics.json")
    compare_json(before / "marketplace_report.json", after / "marketplace_report.json", "marketplace_report.json")
    compare_jsonl(before / "valuations.jsonl", after / "valuations.jsonl", "valuations.jsonl")

    print()
    if failures:
        print(f"RESULT: {len(failures)} difference(s): {failures}")
        sys.exit(1)
    print("RESULT: all compared metrics identical (bit-for-bit reproduction confirmed).")


if __name__ == "__main__":
    main()
