#!/usr/bin/env python3
"""Recompute the Fig-7 valuation gap from RAW valuations.jsonl, across every
solution concept (selection / kernelshap / banzhaf / leastcore / loo / influence).

Mirrors extract_stepE4_valuation_gap.py's definition (drop first 50% of rounds as
warmup; clamp negative values to 0) but reads raw files directly.

Usage: python3 recompute_e4_gap.py <results_dir>
"""
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
WARMUP = 0.5

METHODS = {
    "selection": "selection_score",
    "kernelshap": "kernelshap_score",
    "banzhaf": "banzhaf_score",
    "leastcore": "leastcore_score",
    "loo": "marginal_contrib_loo",
    "influence": "influence_score",
}
ORDER = ["selection", "kernelshap", "banzhaf", "leastcore", "loo", "influence"]

per = defaultdict(lambda: defaultdict(list))
rounds_with = defaultdict(lambda: defaultdict(int))

pattern = os.path.join(RES, "stepE4*", "**", "valuations.jsonl")
for path in sorted(glob.glob(pattern, recursive=True)):
    scen = os.path.relpath(path, RES).split(os.sep)[0]
    scen = scen.replace("stepE4_valuation_", "").replace("_CIFAR100", "")
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not entries:
        continue
    entries = entries[int(len(entries) * WARMUP):] or entries[-1:]

    sums = defaultdict(lambda: defaultdict(float))
    for e in entries:
        selected = set(e.get("selected_ids") or [])
        for sid, v in (e.get("seller_valuations") or {}).items():
            group = "adv" if str(sid).startswith("adv") else "benign"
            state = "paid" if sid in selected else "disc"
            bucket = group + "_" + state
            for m, key in METHODS.items():
                x = v.get(key)
                if x is None:
                    continue
                sums[m][bucket] += max(0.0, float(x))
                rounds_with[scen][m] += 1

    for m, b in sums.items():
        b_tot = b.get("benign_paid", 0.0) + b.get("benign_disc", 0.0)
        a_tot = b.get("adv_paid", 0.0) + b.get("adv_disc", 0.0)
        if b_tot:
            per[scen][m + "|benign_paid"].append(100.0 * b.get("benign_paid", 0.0) / b_tot)
        if a_tot:
            per[scen][m + "|adv_paid"].append(100.0 * b.get("adv_paid", 0.0) / a_tot)

hdr = "{:<26}{:<11}{:>16}{:>16}{:>7}".format("scenario", "method", "benign_paid%", "adv_paid%", "seeds")
print(hdr)
print("-" * len(hdr))
for scen in sorted(per):
    for m in ORDER:
        bp = per[scen].get(m + "|benign_paid")
        ap = per[scen].get(m + "|adv_paid")
        if not bp and not ap:
            continue
        bp = bp or [float("nan")]
        ap = ap or [float("nan")]
        bs = pstdev(bp) if len(bp) > 1 else 0.0
        asd = pstdev(ap) if len(ap) > 1 else 0.0
        print("{:<26}{:<11}{:>10.1f} +-{:<4.1f}{:>10.1f} +-{:<4.1f}{:>6}".format(
            scen, m, mean(bp), bs, mean(ap), asd, len(bp)))
    print()

print("\n=== per-scenario method coverage (nonnull seller-entries) ===")
for scen in sorted(rounds_with):
    miss = [m for m in ORDER if not rounds_with[scen].get(m)]
    print("  {:<26} missing={}".format(scen, miss or "none"))
