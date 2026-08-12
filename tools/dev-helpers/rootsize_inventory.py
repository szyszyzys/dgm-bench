#!/usr/bin/env python3
"""Phase-0 inventory for R3.O4 root-size + biased-D_B sweeps. Read-only, stdlib.

Scans step11_<filter>_CIFAR100. Families: scarcity/ratio_* (root SIZE sweep),
vary_buyer/alpha_* (biased D_B), vary_seller/alpha_* (seller heterogeneity).
Reports per (filter,family,level): seeds present, rounds, seller_metrics presence,
newest mtime (regeneration marker), config_snapshot sha256.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def cfg_sha(run_dir):
    for d in (run_dir, os.path.dirname(run_dir), os.path.dirname(os.path.dirname(run_dir))):
        p = os.path.join(d, "config_snapshot.json")
        if os.path.exists(p):
            return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    return None


cells = defaultdict(lambda: defaultdict(dict))
also = defaultdict(int)
for scen in sorted(glob.glob(os.path.join(RES, "step11_*_CIFAR100"))):
    m = re.match(r"step11_(fltrust|martfl|\w+)_CIFAR100", os.path.basename(scen))
    filt = m.group(1) if m else "?"
    for metrics in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
        rd = os.path.dirname(metrics)
        rel = os.path.relpath(rd, scen).split(os.sep)
        if len(rel) < 2:
            continue
        family, level = rel[0], rel[1]
        also[family] += 1
        sm = re.search(r"seed_(\d+)$", os.path.basename(rd))
        seed = int(sm.group(1)) if sm else None
        try:
            fm = json.load(open(metrics))
        except Exception:
            fm = {}
        key = (filt, family, level)
        cells[key][seed] = {
            "rounds": fm.get("completed_rounds"),
            "has_sm": os.path.exists(os.path.join(rd, "seller_metrics.csv")),
            "mtime": os.path.getmtime(metrics),
            "sha": cfg_sha(rd),
        }

FAM_LABEL = {"scarcity": "root SIZE (Claim 1)", "vary_buyer": "biased D_B (Claim 2)",
             "vary_seller": "seller heterogeneity (aux)"}
print("R3.O4 root-size / biased-D_B inventory under", RES)
print("families found:", dict(also))
print()
print("{:8} {:12} {:>10} {:>7} {:>26} {:>10} {:>8}".format(
    "filter", "family", "level", "seeds", "seeds(rounds)", "newest", "cfg_sha"))
print("-" * 90)
for key in sorted(cells):
    filt, family, level = key
    d = cells[key]
    seeds = sorted(s for s in d if s is not None)
    seed_str = ",".join(f"{s}({d[s]['rounds']})" for s in seeds)
    newest = time.strftime("%Y-%m-%d", time.localtime(max(v["mtime"] for v in d.values())))
    shas = ",".join(sorted({str(v["sha"]) for v in d.values()}))[:8]
    allsm = all(v["has_sm"] for v in d.values())
    print("{:8} {:12} {:>10} {:>7} {:>26} {:>10} {:>8} {}".format(
        filt, family, level, len(seeds), seed_str[:26], newest, shas,
        "" if allsm else "[SM MISSING]"))

print("\n=== claim coverage ===")
for claim, fam in (("Claim 1 root-size", "scarcity"), ("Claim 2 biased-D_B", "vary_buyer")):
    filt_levels = defaultdict(set)
    for (filt, family, level) in cells:
        if family == fam:
            filt_levels[filt].add(level)
    print(f"  {claim} ({fam}):")
    for filt in sorted(filt_levels):
        print(f"    {filt}: levels {sorted(filt_levels[filt])}")
