#!/usr/bin/env python3
"""E3 Phase-1 full verification from RAW per-seller outputs. Read-only, stdlib.

BSR/MSR recomputed from seller_metrics.csv per-round selection (NOT cached
summaries). Emits JSON: per-(filter,eps) BSR/MSR/Acc mean+std+per-seed;
monotonicity test loose->tight; eps=none anchor (MSR + BSR).
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
EPS_ORDER = ["none", "8", "4", "1"]  # loose -> tight


def rates(sm_path):
    per = defaultdict(lambda: {"adv": [0, 0], "bn": [0, 0]})
    with open(sm_path, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("seller_id") or "").strip()
            kind = "adv" if sid.startswith("adv") else "bn" if sid.startswith("bn") else None
            if kind is None:
                continue
            slot = per[row.get("round")][kind]
            slot[0] += int((row.get("selected") or "").strip().lower() == "true")
            slot[1] += 1
    msr = [d["adv"][0] / d["adv"][1] for d in per.values() if d["adv"][1]]
    bsr = [d["bn"][0] / d["bn"][1] for d in per.values() if d["bn"][1]]
    return (mean(msr) * 100 if msr else None, mean(bsr) * 100 if bsr else None)


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(mean(vals), 3),
            "std": round(pstdev(vals) if len(vals) > 1 else 0.0, 3),
            "n": len(vals), "per_seed": [round(v, 3) for v in sorted(vals)]}


cells = defaultdict(lambda: defaultdict(list))
for scen in sorted(glob.glob(os.path.join(RES, "stepE3_dp_*"))):
    m = re.match(r"stepE3_dp_(fltrust|martfl)_eps_(none|\d+)_", os.path.basename(scen))
    if not m:
        continue
    filt, eps = m.group(1), m.group(2)
    for metrics in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
        rd = os.path.dirname(metrics)
        try:
            fm = json.load(open(metrics))
        except Exception:
            fm = {}
        smp = os.path.join(rd, "seller_metrics.csv")
        msr = bsr = None
        if os.path.exists(smp):
            msr, bsr = rates(smp)
        cells[filt][eps].append({"msr": msr, "bsr": bsr,
                                 "acc": fm.get("acc") * 100 if fm.get("acc") is not None else None})

out = {"per_cell": {}, "monotonicity": {}, "anchor": {}}
for filt in ("fltrust", "martfl"):
    out["per_cell"][filt] = {}
    seq = []
    for eps in EPS_ORDER:
        recs = cells[filt].get(eps, [])
        bsr = agg([r["bsr"] for r in recs])
        msr = agg([r["msr"] for r in recs])
        acc = agg([r["acc"] for r in recs])
        out["per_cell"][filt][eps] = {"bsr": bsr, "msr": msr, "acc": acc}
        if bsr:
            seq.append((eps, bsr["mean"], bsr["std"]))
    # monotonic non-increasing loose->tight, beyond combined seed std?
    reversals = []
    for (e1, m1, s1), (e2, m2, s2) in zip(seq, seq[1:]):
        comb = (s1 ** 2 + s2 ** 2) ** 0.5
        if m2 > m1 + comb:   # rose beyond noise at a tightening step
            reversals.append({"step": f"{e1}->{e2}", "from": m1, "to": m2, "combined_std": round(comb, 3)})
    drop = (seq[0][1] - seq[-1][1]) if len(seq) >= 2 else None
    end_comb = ((seq[0][2] ** 2 + seq[-1][2] ** 2) ** 0.5) if len(seq) >= 2 else None
    out["monotonicity"][filt] = {
        "sequence_loose_to_tight": [{"eps": e, "bsr_mean": m, "bsr_std": s} for e, m, s in seq],
        "monotonic_nonincreasing": len(reversals) == 0,
        "reversals": reversals,
        "total_drop_pp": round(drop, 3) if drop is not None else None,
        "endpoint_combined_std": round(end_comb, 3) if end_comb is not None else None,
        "drop_exceeds_noise": (drop is not None and end_comb is not None and drop > end_comb),
    }

# anchor: eps=none FLTrust MSR vs Table5 96.1; eps=none BSR both filters
out["anchor"] = {
    "fltrust_eps_none_MSR": out["per_cell"]["fltrust"]["none"]["msr"],
    "fltrust_eps_none_BSR": out["per_cell"]["fltrust"]["none"]["bsr"],
    "martfl_eps_none_BSR": out["per_cell"]["martfl"]["none"]["bsr"],
    "table5_binary_fltrust_MSR": 96.1,
}
print(json.dumps(out, indent=1))
