#!/usr/bin/env python3
"""Compare PUBLISHED Table 5 (trusted) vs CURRENT regenerated numbers. Read-only.

Published values hardcoded from the paper LaTeX. Current recomputed from raw:
Acc/ASR from final_metrics.json, BSR/MSR from seller_metrics.csv per-round
selection. Flags any cell moving beyond THRESH pp.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
THRESH = 3.0  # pp; flag divergences beyond this

DS = ["CIFAR100", "FEMNIST", "Texas100", "Purchase100", "TREC"]

# published (acc, asr, bsr, msr); None where '-' in the paper
PUB = {
    "fedavg":     {"CIFAR100": (46.3,79.8,100,100), "FEMNIST": (89.0,100,100,100), "Texas100": (61.4,100,100,100), "Purchase100": (49.4,89.4,100,100), "TREC": (70.8,70.3,100,100)},
    "multi_krum": {"CIFAR100": (19.1,90.7,17.7,58.6),"FEMNIST": (88.0,5.3,42.9,0.0),"Texas100": (49.5,0.4,42.9,0.0),"Purchase100": (20.6,98.4,22.4,47.7),"TREC": (51.3,49.5,23.8,44.4)},
    "bulyan":     {"CIFAR100": (10.7,33.0,72.4,97.8),"TREC": (64.5,18.8,81.0,77.8)},
    "fltrust":    {"CIFAR100": (48.4,79.8,98.6,96.1), "FEMNIST": (89.0,99.9,100,99.9),"Texas100": (47.2,99.7,100,100),"Purchase100": (4.2,5.3,100,100),"TREC": (69.7,73.0,100,100)},
    "martfl":     {"CIFAR100": (46.9,74.9,62.5,42.3), "FEMNIST": (88.9,100,68.8,39.1), "Texas100": (60.6,100,52.5,57.2),"Purchase100": (21.2,10.8,92.1,5.2),"TREC": (70.0,56.3,81.0,55.0)},
    "deepsight":  {"CIFAR100": (46.9,78.9,95.3,92.8), "FEMNIST": (88.9,99.4,90.4,82.6),"Texas100": (60.4,100,90.6,85.1),"Purchase100": (49.1,79.7,99.7,77.5),"TREC": (70.8,66.4,87.9,98.7)},
    "trimmed_mean":{"CIFAR100": (46.4,80.1,None,None),"FEMNIST": (88.9,100,None,None),"Texas100": (61.1,100,None,None),"Purchase100": (50.5,78.3,None,None),"TREC": (73.1,79.5,None,None)},
    "flame":      {"CIFAR100": (43.5,76.0,None,None), "FEMNIST": (88.8,100,None,None), "Texas100": (61.2,100,None,None),"Purchase100": (44.6,84.5,None,None),"TREC": (71.1,84.2,None,None)},
    "foolsgold":  {"CIFAR100": (46.3,79.8,None,None), "FEMNIST": (89.0,100,None,None), "Texas100": (61.6,100,None,None),"Purchase100": (51.4,48.1,None,None),"TREC": (73.3,81.9,None,None)},
    "rflpa":      {"CIFAR100": (6.5,71.3,None,None),  "FEMNIST": (86.6,33.5,None,None),"Texas100": (23.5,100,None,None),"Purchase100": (21.6,37.6,None,None),"TREC": (71.4,75.7,None,None)},
    "spmc":       {"CIFAR100": (1.0,98.9,None,None),  "FEMNIST": (75.4,9.0,None,None), "Texas100": (12.1,98.3,None,None),"Purchase100": (3.1,7.8,None,None),"TREC": (44.0,3.8,None,None)},
    "skymask":    {"CIFAR100": (47.0,80.5,None,None), "FEMNIST": (89.0,100,None,None)},
}


def rates(sm):
    per = defaultdict(lambda: {"a": [0, 0], "b": [0, 0]})
    with open(sm, newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("seller_id") or "").strip()
            k = "a" if s.startswith("adv") else "b" if s.startswith("bn") else None
            if not k:
                continue
            sl = per[r.get("round")][k]
            sl[0] += int((r.get("selected") or "").strip().lower() == "true")
            sl[1] += 1
    msr = [d["a"][0]/d["a"][1] for d in per.values() if d["a"][1]]
    bsr = [d["b"][0]/d["b"][1] for d in per.values() if d["b"][1]]
    return (mean(msr)*100 if msr else None, mean(bsr)*100 if bsr else None)


def current(defense, dataset):
    pat = os.path.join(RES, f"step12_main_summary_{defense}_*_{dataset}_*")
    accs, asrs, msrs, bsrs = [], [], [], []
    for scen in glob.glob(pat):
        for mp in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
            rd = os.path.dirname(mp)
            if not os.path.exists(os.path.join(rd, ".success")):
                continue
            try:
                fm = json.load(open(mp))
            except Exception:
                continue
            if fm.get("acc") is not None:
                accs.append(fm["acc"]*100)
            if fm.get("asr") is not None:
                asrs.append(fm["asr"]*100)
            smp = os.path.join(rd, "seller_metrics.csv")
            if os.path.exists(smp):
                m, b = rates(smp)
                if m is not None:
                    msrs.append(m)
                if b is not None:
                    bsrs.append(b)
    def av(x):
        return round(mean(x), 1) if x else None
    return av(accs), av(asrs), av(bsrs), av(msrs), len(accs)


METRICS = ["Acc", "ASR", "BSR", "MSR"]
flags = []
print("PUBLISHED vs CURRENT  (flag |Δ| > %.0f pp).  cur n=seeds in ().\n" % THRESH)
for defense in PUB:
    for ds in DS:
        if ds not in PUB[defense]:
            continue
        pub = PUB[defense][ds]
        cur = current(defense, ds)
        n = cur[4]
        if n == 0:
            print(f"  {defense:12} {ds:12} CURRENT MISSING (0 seeds)")
            flags.append((defense, ds, "MISSING"))
            continue
        diffs = []
        for i, mname in enumerate(METRICS):
            pv, cv = pub[i], cur[i]
            if pv is None or cv is None:
                continue
            d = cv - pv
            mark = "  <<<" if abs(d) > THRESH else ""
            diffs.append(f"{mname} {pv:5.1f}->{cv:5.1f} ({d:+.1f}){mark}")
            if abs(d) > THRESH:
                flags.append((defense, ds, mname, pv, cv, round(d, 1)))
        print(f"  {defense:12} {ds:12} n={n}  " + " | ".join(diffs))

print("\n=== DIVERGENT CELLS (|Δ|>%.0f pp) ===" % THRESH)
for f in flags:
    print("  ", f)
