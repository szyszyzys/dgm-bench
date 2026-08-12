#!/usr/bin/env python3
"""Cross-check the two BSR/MSR estimators used in this repo.

  A) seller_metrics.csv : per-round (#selected / #present), averaged over rounds
  B) training_log.csv   : 1 - false_positive_rate  /  1 - adversary_detection_rate

They should agree. Any systematic gap means the headline table and the revision
analysis are reporting different quantities under the same name.
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import mean

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def est_a(path):
    per = defaultdict(lambda: {"adv": [0, 0], "bn": [0, 0]})
    with open(path, newline="") as fh:
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
    return (mean(msr) if msr else None, mean(bsr) if bsr else None)


def est_b(path):
    fpr, adr = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            for key, acc in (("false_positive_rate", fpr), ("adversary_detection_rate", adr)):
                v = row.get(key)
                if v not in (None, ""):
                    try:
                        acc.append(float(v))
                    except ValueError:
                        pass
    return (1.0 - mean(adr) if adr else None, 1.0 - mean(fpr) if fpr else None)


rows = []
for sm in sorted(glob.glob(os.path.join(RES, "step11_*", "**", "seller_metrics.csv"), recursive=True)):
    tl = os.path.join(os.path.dirname(sm), "training_log.csv")
    if not os.path.exists(tl):
        continue
    try:
        a_msr, a_bsr = est_a(sm)
        b_msr, b_bsr = est_b(tl)
    except Exception:
        continue
    if None in (a_msr, a_bsr, b_msr, b_bsr):
        continue
    rows.append((a_msr, a_bsr, b_msr, b_bsr))

if not rows:
    print("no comparable cells found")
    raise SystemExit(0)

dm = [abs(r[0] - r[2]) for r in rows]
db = [abs(r[1] - r[3]) for r in rows]
print("cells compared: {}".format(len(rows)))
print("  MSR: seller_metrics {:.4f} vs training_log {:.4f}   mean|diff|={:.4f}  max|diff|={:.4f}".format(
    mean([r[0] for r in rows]), mean([r[2] for r in rows]), mean(dm), max(dm)))
print("  BSR: seller_metrics {:.4f} vs training_log {:.4f}   mean|diff|={:.4f}  max|diff|={:.4f}".format(
    mean([r[1] for r in rows]), mean([r[3] for r in rows]), mean(db), max(db)))
print("\nVERDICT:", "AGREE (<1pp)" if max(max(dm), max(db)) < 0.01 else "DISAGREE — estimators are not interchangeable")
