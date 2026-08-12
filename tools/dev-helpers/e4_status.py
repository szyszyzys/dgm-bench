#!/usr/bin/env python3
"""E4 Phase-0 inventory: per-method per-cell valuation completion grid + runtime.

Read-only, stdlib. A method counts as PRESENT in a run if any valuations.jsonl
line has a non-null score for its key. Also reports Least-Core computable-round
fraction (intractability) and per-cell wall-clock from final_metrics.json.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
METHOD_KEYS = {
    "kernelshap": "kernelshap_score", "loo": "marginal_contrib_loo",
    "influence": "influence_score", "banzhaf": "banzhaf_score",
    "leastcore": "leastcore_score",
}
SCEN_RE = re.compile(r"stepE4_valuation_(?P<defense>.+?)_(?P<attack>no_attack|backdoor)_(?P<ds>.+)$")


def scan_run(val_path):
    present = set()
    lc_rounds = tot = 0
    with open(val_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tot += 1
            any_lc = False
            for sid, sc in (rec.get("seller_valuations") or {}).items():
                for m, k in METHOD_KEYS.items():
                    if sc.get(k) is not None:
                        present.add(m)
                        if m == "leastcore":
                            any_lc = True
            lc_rounds += int(any_lc)
    return present, lc_rounds, tot


grid = defaultdict(lambda: defaultdict(lambda: {"seeds": 0, "methods": defaultdict(int),
                                                "lc_frac": [], "walls": []}))
for scen in sorted(glob.glob(os.path.join(RES, "stepE4_valuation_*"))):
    m = SCEN_RE.match(os.path.basename(scen))
    if not m:
        continue
    cell = (m.group("defense"), m.group("attack"))
    for vp in glob.glob(os.path.join(scen, "**", "valuations.jsonl"), recursive=True):
        rd = os.path.dirname(vp)
        present, lc, tot = scan_run(vp)
        g = grid[m.group("ds")][cell]
        g["seeds"] += 1
        for mm in present:
            g["methods"][mm] += 1
        if tot:
            g["lc_frac"].append(lc / tot)
        fmp = os.path.join(rd, "final_metrics.json")
        if os.path.exists(fmp):
            try:
                w = json.load(open(fmp)).get("wall_clock_seconds")
                if w:
                    g["walls"].append(w / 60.0)
            except Exception:
                pass

METH_ORDER = ["kernelshap", "loo", "influence", "banzhaf", "leastcore"]
for ds in sorted(grid):
    print(f"\n==== dataset: {ds} ====")
    print("{:<26}{:>6}  {}".format("cell(defense,attack)", "seeds",
                                   "  ".join(f"{m[:5]:>5}" for m in METH_ORDER)))
    print("-" * 78)
    for cell in sorted(grid[ds]):
        g = grid[ds][cell]
        cov = "  ".join(f"{g['methods'].get(m,0):>5}" for m in METH_ORDER)
        print("{:<26}{:>6}  {}".format(f"{cell[0]}/{cell[1]}", g["seeds"], cov))
    # least-core tractability + runtime
    print("  Least-Core computable-round fraction & median wall (min):")
    for cell in sorted(grid[ds]):
        g = grid[ds][cell]
        lc = mean(g["lc_frac"]) * 100 if g["lc_frac"] else 0.0
        wall = sorted(g["walls"])[len(g["walls"]) // 2] if g["walls"] else 0
        print(f"    {cell[0]}/{cell[1]:<10} LC_rounds={lc:5.1f}%   med_wall={wall:6.1f} min   n_wall={len(g['walls'])}")

# totals
allwalls = [w for ds in grid for c in grid[ds] for w in grid[ds][c]["walls"]]
print("\n---- overall ----")
print("cells:", sum(len(grid[ds]) for ds in grid),
      " runs:", sum(grid[ds][c]["seeds"] for ds in grid for c in grid[ds]))
if allwalls:
    print("median per-run wall: %.1f min ; total GPU-h already spent on E4: %.1f"
          % (sorted(allwalls)[len(allwalls)//2], sum(allwalls)/60.0))
