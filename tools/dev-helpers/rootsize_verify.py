#!/usr/bin/env python3
"""R3.O4 Phase-1 verification from RAW per-seller outputs. Read-only, stdlib.

Per (filter, family, level): MSR/BSR/Acc mean+std + PER-SEED acc (bimodal check).
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, pstdev

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


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


cells = defaultdict(lambda: defaultdict(dict))
for scen in sorted(glob.glob(os.path.join(RES, "step11_*_CIFAR100"))):
    m = re.match(r"step11_(\w+?)_CIFAR100", os.path.basename(scen))
    filt = m.group(1)
    for metrics in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
        rd = os.path.dirname(metrics)
        rel = os.path.relpath(rd, scen).split(os.sep)
        if len(rel) < 2:
            continue
        fam, lvl = rel[0], rel[1]
        sd = re.search(r"seed_(\d+)$", os.path.basename(rd))
        seed = int(sd.group(1)) if sd else -1
        try:
            fm = json.load(open(metrics))
        except Exception:
            fm = {}
        smp = os.path.join(rd, "seller_metrics.csv")
        msr = bsr = None
        if os.path.exists(smp):
            msr, bsr = rates(smp)
        cells[(filt, fam)][lvl][seed] = {
            "acc": (fm.get("acc") or 0)*100, "asr": (fm.get("asr") or 0)*100,
            "msr": msr, "bsr": bsr, "rounds": fm.get("completed_rounds"),
        }


def agg(vals):
    vals = [v for v in vals if v is not None]
    return (round(mean(vals), 2), round(pstdev(vals) if len(vals) > 1 else 0.0, 2)) if vals else (None, None)


def lvlnum(l):
    m = re.search(r"([0-9.]+)$", l)
    return float(m.group(1)) if m else 0


for fam, label in (("scarcity", "CLAIM 1 — root SIZE"), ("vary_buyer", "CLAIM 2 — biased D_B")):
    print("=" * 78)
    print(label, f"({fam})")
    print("=" * 78)
    for filt in ("fltrust", "martfl"):
        key = (filt, fam)
        if key not in cells:
            continue
        print(f"\n-- {filt} --")
        print("  {:>12} {:>13} {:>13} {:>13}  per-seed acc".format("level", "MSR", "BSR", "Acc"))
        for lvl in sorted(cells[key], key=lvlnum):
            d = cells[key][lvl]
            seeds = sorted(d)
            msr = agg([d[s]["msr"] for s in seeds])
            bsr = agg([d[s]["bsr"] for s in seeds])
            acc = agg([d[s]["acc"] for s in seeds])
            per_acc = [round(d[s]["acc"], 1) for s in seeds]
            collapsed = sum(1 for a in per_acc if a < 5.0)
            flag = f"  <<{collapsed}/{len(per_acc)} COLLAPSED" if collapsed else ""
            print("  {:>12} {:>6.1f}±{:<5.1f} {:>6.1f}±{:<5.1f} {:>6.1f}±{:<5.1f}  {}{}".format(
                lvl, msr[0], msr[1], bsr[0], bsr[1], acc[0], acc[1], per_acc, flag))
    print()
