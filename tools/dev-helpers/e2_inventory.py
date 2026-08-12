#!/usr/bin/env python3
"""Phase-0 inventory for E2 multi-task market persistence.

Read-only, pure stdlib. For each filter scenario: seeds present, #tasks,
exit_k value(s) used, and which artifacts exist (multi_task_summary.csv,
multi_task_rounds.csv, market_state.json). Critically reports the SET of
distinct exit_k values — the k-independence claim needs >1.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def find_exit_k(seed_dir):
    """exit_k from a config_snapshot.json under this run (repr-string tolerant)."""
    for cs in glob.glob(os.path.join(seed_dir, "**", "config_snapshot.json"), recursive=True):
        raw = open(cs, "r", errors="ignore").read()
        m = re.search(r"exit_k[=\"':\s]+(\d+)", raw)
        if m:
            return int(m.group(1))
    # fall back: search parent scenario dir
    return None


def num_tasks(summary_path):
    try:
        with open(summary_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        tasks = {r.get("task") for r in rows}
        return len(tasks), sorted(tasks, key=lambda x: int(x) if str(x).isdigit() else 99)
    except Exception:
        return 0, []


print("E2 multi-task inventory under", RES)
scenarios = sorted(glob.glob(os.path.join(RES, "stepE2_multitask_*")))
if not scenarios:
    print("  NO stepE2_multitask_* scenarios found.")
    sys.exit(0)

all_k = defaultdict(set)
for scen in scenarios:
    name = os.path.basename(scen)
    filt = "martfl" if "martfl" in name else "fltrust" if "fltrust" in name else "?"
    seed_dirs = []
    for summ in glob.glob(os.path.join(scen, "**", "multi_task_summary.csv"), recursive=True):
        seed_dirs.append(os.path.dirname(summ))
    print(f"\n=== {name}  (filter={filt}) ===")
    print(f"  seed dirs with multi_task_summary.csv: {len(seed_dirs)}")
    if not seed_dirs:
        # maybe runs exist but summary missing
        rd = glob.glob(os.path.join(scen, "**", "run_*"), recursive=True)
        print(f"  (run_* dirs present: {len(rd)}; summaries missing -> PARTIAL/broken)")
        continue
    for sd in sorted(seed_dirs):
        sm = re.search(r"seed_(\d+)", sd)
        seed = sm.group(1) if sm else "?"
        nt, tasks = num_tasks(os.path.join(sd, "multi_task_summary.csv"))
        k = find_exit_k(sd)
        has_rounds = os.path.exists(os.path.join(sd, "multi_task_rounds.csv"))
        has_state = os.path.exists(os.path.join(sd, "market_state.json"))
        if k is not None:
            all_k[filt].add(k)
        print(f"    seed {seed:>3}: tasks={nt} {tasks}  exit_k={k}  "
              f"rounds_csv={'Y' if has_rounds else 'n'}  market_state={'Y' if has_state else 'n'}")

print("\n" + "=" * 60)
print("DISTINCT exit_k VALUES PER FILTER:")
for filt in ("fltrust", "martfl"):
    ks = sorted(all_k.get(filt, []))
    verdict = "OK for k-independence" if len(ks) >= 2 else "SINGLE k -> k-independence UNVERIFIABLE"
    print(f"  {filt:8}: {ks or 'none'}   [{verdict}]")
