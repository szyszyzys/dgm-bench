#!/usr/bin/env python3
"""Phase-0 inventory for E1 continuous-payment / thresholded-FLTrust tau sweep.

Read-only. Pure stdlib. For every tau x seed run dir it records: presence of
final_metrics.json / seller_metrics.csv / valuations.jsonl, completed rounds,
payment_model, and a sha256 of config_snapshot.json (old runs predate
run_provenance.json, so the config snapshot is the only per-cell fingerprint).
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
ROOT = os.path.join(RES, "stepE1_tau_sweep_CIFAR100")


def cfg_hash_and_payment(run_dir):
    # config_snapshot.json lives at the run dir or a parent scenario dir.
    for d in (run_dir, os.path.dirname(run_dir), os.path.dirname(os.path.dirname(run_dir))):
        p = os.path.join(d, "config_snapshot.json")
        if os.path.exists(p):
            raw = open(p, "rb").read()
            h = hashlib.sha256(raw).hexdigest()[:12]
            try:
                c = json.loads(raw)
            except Exception:
                return h, "?"
            pm = None
            val = c.get("valuation") if isinstance(c, dict) else None
            if isinstance(val, dict):
                pm = val.get("payment_model")
            return h, pm or "?"
    return None, None


def has_msr_source(run_dir):
    sm = os.path.join(run_dir, "seller_metrics.csv")
    if not os.path.exists(sm):
        return False
    with open(sm, newline="") as fh:
        head = fh.readline()
    return "selected" in head and "seller_id" in head


rows = []
for metrics in glob.glob(os.path.join(ROOT, "**", "final_metrics.json"), recursive=True):
    run_dir = os.path.dirname(metrics)
    m = re.search(r"tau-(0p\d+|\d+)", run_dir)
    tau = float(m.group(1).replace("p", ".")) if m else None
    sm = re.search(r"seed_(\d+)$", os.path.basename(run_dir))
    seed = int(sm.group(1)) if sm else None
    success = os.path.exists(os.path.join(run_dir, ".success"))
    try:
        fm = json.load(open(metrics))
    except Exception:
        fm = {}
    h, pm = cfg_hash_and_payment(run_dir)
    rows.append({
        "tau": tau, "seed": seed,
        "rounds": fm.get("completed_rounds"),
        "acc": fm.get("acc"),
        "success": success,
        "has_seller_metrics": has_msr_source(run_dir),
        "has_valuations": os.path.exists(os.path.join(run_dir, "valuations.jsonl")),
        "cfg_sha": h, "payment_model": pm,
    })

taus = sorted({r["tau"] for r in rows if r["tau"] is not None})
seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
print("E1 tau-sweep inventory:", ROOT)
print("tau values found :", taus)
print("seeds found      :", seeds)
print("total run dirs   :", len(rows))
print()

by = defaultdict(dict)
for r in rows:
    by[r["tau"]][r["seed"]] = r

print("{:>5} {:>7} {:>26} {:>16} {:>8}".format("tau", "n_seeds", "seeds(rounds)", "payment", "cfg_sha"))
print("-" * 70)
for tau in taus:
    d = by[tau]
    seed_str = ",".join(f"{s}({d[s]['rounds']})" for s in sorted(d))
    pms = {d[s]["payment_model"] for s in d}
    shas = {d[s]["cfg_sha"] for s in d}
    print("{:>5g} {:>7} {:>26} {:>16} {:>8}".format(
        tau, len(d), seed_str[:26], ",".join(sorted(str(x) for x in pms)),
        ",".join(sorted(str(x) for x in shas))[:8]))

# completeness matrix
print("\ncompleteness (X=success+seller_metrics, o=metrics only, .=absent):")
hdr = "tau\\seed " + " ".join(f"{s:>4}" for s in seeds)
print(hdr)
for tau in taus:
    line = f"{tau:>7g} "
    for s in seeds:
        r = by[tau].get(s)
        if r is None:
            c = " . "
        elif r["success"] and r["has_seller_metrics"]:
            c = " X "
        else:
            c = " o "
        line += f"{c:>4}"
    print(line)

# payment-model summary
pm_all = defaultdict(int)
for r in rows:
    pm_all[r["payment_model"]] += 1
print("\npayment_model across all runs:", dict(pm_all))
