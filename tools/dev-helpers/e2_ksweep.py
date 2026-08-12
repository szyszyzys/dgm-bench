#!/usr/bin/env python3
"""E2 k-independence + composition across k in {2,3,5}. Read-only, stdlib.

Unpaid-streak gap (benign - adversary) per (filter, k) from market_state.json;
task-5 composition from multi_task_summary.csv. Tests whether the
MartFL-starves / FLTrust-doesn't separation holds at EVERY k.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

RES = sys.argv[1] if len(sys.argv) > 1 else "results"

# (filter, k) -> list of scenario dirs
DIRS = {
    ("fltrust", 2): ["stepE2_multitask_fltrust_plain_CIFAR100_k2"],
    ("fltrust", 3): ["stepE2_multitask_fltrust_plain_CIFAR100",
                     "stepE2_multitask_fltrust_plain_CIFAR100_seedfill"],
    ("fltrust", 5): ["stepE2_multitask_fltrust_plain_CIFAR100_k5"],
    ("martfl", 2): ["stepE2_multitask_martfl_plain_CIFAR100_k2"],
    ("martfl", 3): ["stepE2_multitask_martfl_plain_CIFAR100"],
    ("martfl", 5): ["stepE2_multitask_martfl_plain_CIFAR100_k5"],
}


def streak_gap(state_path):
    """-> (benign_mean_streak, adv_mean_streak) from one market_state.json."""
    d = json.load(open(state_path))
    vals = d.values() if isinstance(d, dict) else d
    b = [v.get("unpaid_streak") for v in vals if isinstance(v, dict) and v.get("type") == "benign"]
    a = [v.get("unpaid_streak") for v in vals if isinstance(v, dict) and v.get("type") == "adversarial"]
    b = [x for x in b if x is not None]
    a = [x for x in a if x is not None]
    return (mean(b) if b else None, mean(a) if a else None)


def task5(summary_path):
    """-> (n_active_benign, adv_fraction) at the last task."""
    rows = list(csv.DictReader(open(summary_path)))
    if not rows:
        return None, None
    last = rows[-1]
    def f(k):
        try:
            return float(last.get(k))
        except (TypeError, ValueError):
            return None
    return f("n_active_benign"), f("adversarial_fraction_active")


def agg(vals):
    vals = [v for v in vals if v is not None]
    return (round(mean(vals), 3), round(pstdev(vals) if len(vals) > 1 else 0.0, 3), len(vals)) if vals else (None, None, 0)


print("E2 k-independence: unpaid-streak gap (benign - adv) + task-5 composition\n")
print("{:8} {:>3} {:>6} {:>16} {:>16} {:>14} {:>14} {:>10}".format(
    "filter", "k", "seeds", "benign_streak", "adv_streak", "GAP(b-a)", "adv_frac_t5", "benign_t5"))
print("-" * 96)

summary = {}
for (filt, k), subs in DIRS.items():
    bstreaks, astreaks, gaps, advfracs, benactive = [], [], [], [], []
    n = 0
    for sub in subs:
        for ms_path in glob.glob(os.path.join(RES, sub, "**", "market_state.json"), recursive=True):
            n += 1
            bm, am = streak_gap(ms_path)
            if bm is not None:
                bstreaks.append(bm)
            if am is not None:
                astreaks.append(am)
            if bm is not None and am is not None:
                gaps.append(bm - am)
            sm = os.path.join(os.path.dirname(ms_path), "multi_task_summary.csv")
            if os.path.exists(sm):
                nb, af = task5(sm)
                if af is not None:
                    advfracs.append(af)
                if nb is not None:
                    benactive.append(nb)
    bs, adv_s, gp = agg(bstreaks), agg(astreaks), agg(gaps)
    af5, nb5 = agg(advfracs), agg(benactive)
    summary[(filt, k)] = {"gap": gp, "adv_frac_t5": af5}
    print("{:8} {:>3} {:>6} {:>10.2f}±{:<4.2f} {:>10.2f}±{:<4.2f} {:>9.2f}±{:<4.2f} {:>9}  {:>9}".format(
        filt, k, n,
        bs[0] if bs[0] is not None else float("nan"), bs[1] or 0,
        adv_s[0] if adv_s[0] is not None else float("nan"), adv_s[1] or 0,
        gp[0] if gp[0] is not None else float("nan"), gp[1] or 0,
        f"{af5[0]}" if af5[0] is not None else "n/a",
        f"{nb5[0]}" if nb5[0] is not None else "n/a"))

print("\n=== k-independence verdict ===")
for filt in ("fltrust", "martfl"):
    gaps = {k: summary[(filt, k)]["gap"][0] for k in (2, 3, 5) if summary.get((filt, k), {}).get("gap")}
    print(f"  {filt}: unpaid-streak gap by k = {gaps}")
