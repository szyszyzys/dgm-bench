#!/usr/bin/env python3
"""E1 Phase-1 full verification from RAW per-round / per-seller outputs.

Read-only, pure stdlib. Emits JSON with:
  - per-tau frontier: MSR/BSR/Acc/ASR mean+std over seeds (MSR/BSR from
    seller_metrics.csv per-round selection; Acc/ASR from final_metrics.json)
  - claim #1 test: any tau with MSR<=20 AND Acc>=40 ?
  - claim #2 test: adversarial capture share at tau=0 from valuations.jsonl price_paid
  - anchor: tau=0 MSR
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
ROOT = os.path.join(RES, "stepE1_tau_sweep_CIFAR100")


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
    return (mean(msr) * 100 if msr else None,
            mean(bsr) * 100 if bsr else None, len(per))


def capture_share(val_path):
    adv = tot = 0.0
    with open(val_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sid, v in (r.get("seller_valuations") or {}).items():
                p = v.get("price_paid")
                if p is None:
                    continue
                tot += p
                if sid.startswith("adv"):
                    adv += p
    return (adv / tot * 100 if tot else None)


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(mean(vals), 3),
            "std": round(pstdev(vals) if len(vals) > 1 else 0.0, 3),
            "n": len(vals),
            "min": round(min(vals), 3), "max": round(max(vals), 3),
            "per_seed": [round(v, 3) for v in vals]}


by_tau = defaultdict(lambda: defaultdict(list))
cap_by_tau = defaultdict(list)

for metrics in glob.glob(os.path.join(ROOT, "**", "final_metrics.json"), recursive=True):
    run_dir = os.path.dirname(metrics)
    m = re.search(r"tau-(0p\d+|\d+)", run_dir)
    tau = float(m.group(1).replace("p", ".")) if m else None
    sm = re.search(r"seed_(\d+)$", os.path.basename(run_dir))
    seed = int(sm.group(1)) if sm else None
    try:
        fm = json.load(open(metrics))
    except Exception:
        fm = {}
    smp = os.path.join(run_dir, "seller_metrics.csv")
    msr = bsr = rnds = None
    if os.path.exists(smp):
        msr, bsr, rnds = rates(smp)
    by_tau[tau]["msr"].append(msr)
    by_tau[tau]["bsr"].append(bsr)
    by_tau[tau]["acc"].append(fm.get("acc") * 100 if fm.get("acc") is not None else None)
    by_tau[tau]["asr"].append(fm.get("asr") * 100 if fm.get("asr") is not None else None)
    by_tau[tau]["rounds"].append(rnds)
    by_tau[tau]["seeds"].append(seed)
    vp = os.path.join(run_dir, "valuations.jsonl")
    if os.path.exists(vp):
        cs = capture_share(vp)
        if cs is not None:
            cap_by_tau[tau].append((seed, cs))

frontier = {}
for tau in sorted(by_tau):
    d = by_tau[tau]
    frontier[f"{tau:g}"] = {
        "msr": agg(d["msr"]), "bsr": agg(d["bsr"]),
        "acc": agg(d["acc"]), "asr": agg(d["asr"]),
        "rounds": agg(d["rounds"]),
        "seeds": sorted(s for s in d["seeds"] if s is not None),
    }

# Claim #1: any tau with MSR<=20 AND Acc>=40 (using per-tau means)?
violators = []
for tau_s, v in frontier.items():
    if v["msr"] and v["acc"] and v["msr"]["mean"] <= 20 and v["acc"]["mean"] >= 40:
        violators.append(tau_s)

# Claim #2: capture share at tau=0
cap0 = cap_by_tau.get(0.0, [])
cap0_vals = [c for _, c in cap0]
claim2 = {
    "tau0_capture_share": agg(cap0_vals),
    "per_seed": sorted(cap0),
    "population_share": 30.0,
}

out = {
    "frontier": frontier,
    "claim1_joint_target_MSR<=20_AND_Acc>=40": {
        "violating_taus": violators,
        "result": "NO tau reaches joint target (claim holds)" if not violators
                  else "VIOLATION — some tau reaches joint target",
    },
    "claim2_capture_share": claim2,
    "anchor_tau0_MSR": frontier.get("0", {}).get("msr"),
    # also every-tau capture share for context
    "capture_share_all_tau": {f"{t:g}": agg([c for _, c in cap_by_tau[t]])
                              for t in sorted(cap_by_tau)},
}
print(json.dumps(out, indent=1))
