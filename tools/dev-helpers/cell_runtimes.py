#!/usr/bin/env python3
"""Per-scenario wall-clock stats from final_metrics.json, to ground Phase-2 estimates."""
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import median

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
by = defaultdict(list)
rounds = defaultdict(list)

for p in glob.glob(os.path.join(RES, "*", "**", "final_metrics.json"), recursive=True):
    scen = os.path.relpath(p, RES).split(os.sep)[0]
    try:
        d = json.load(open(p))
    except Exception:
        continue
    w = d.get("wall_clock_seconds")
    r = d.get("completed_rounds")
    if w:
        by[scen].append(w / 60.0)
    if r:
        rounds[scen].append(r)

print("{:<46}{:>7}{:>10}{:>10}{:>10}{:>9}".format(
    "scenario", "cells", "med_min", "min", "max", "med_rnds"))
print("-" * 92)
grand = []
for s in sorted(by):
    v = by[s]
    grand += v
    print("{:<46}{:>7}{:>10.1f}{:>10.1f}{:>10.1f}{:>9.0f}".format(
        s[:45], len(v), median(v), min(v), max(v),
        median(rounds[s]) if rounds[s] else 0))
if grand:
    print("-" * 92)
    print("{:<46}{:>7}{:>10.1f}".format("ALL", len(grand), median(grand)))
    print("\ntotal GPU-hours already spent: {:.1f}".format(sum(grand) / 60.0))
