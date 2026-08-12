#!/usr/bin/env python3
"""IF sparsity resolution on the MartFL valuation gap. READ-ONLY, stdlib.

For martfl/{backdoor,no_attack} CIFAR-100, seeds 42-46, recompute the Fig-7 gap
metric for Influence Functions across warmup fractions {0,25,40,50}% and diagnose
why the adversary bucket is empty in most seeds. Also emits the sign breakdown of
adversary IF scores (the suspected cause) and the other methods at matched window.

Gap metric (same as recompute_e4_gap.py): per post-warmup round, bucket each
seller into {benign,adv}_{paid,discarded} (paid = in selected_ids); accumulate
value = max(0, score) [clamped] — the bucket is 'empty' when its group's value
sum is 0. We additionally compute SIGNED sums to expose the clamp effect.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, pstdev

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
CELLS = ["martfl_backdoor", "martfl_no_attack"]
WARMUPS = [0.0, 0.25, 0.40, 0.50]
METHOD_KEYS = {"kernelshap": "kernelshap_score", "banzhaf": "banzhaf_score",
               "loo": "marginal_contrib_loo", "influence": "influence_score"}


def run_dirs(cell):
    scen = os.path.join(RES, f"stepE4_valuation_{cell}_CIFAR100")
    for vp in sorted(glob.glob(os.path.join(scen, "**", "valuations.jsonl"), recursive=True)):
        m = re.search(r"seed_(\d+)", vp)
        yield (int(m.group(1)) if m else -1, vp)


def load(vp):
    out = []
    with open(vp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def gap_for(entries, key, warmup, signed=False):
    """Return dict with clamped/signed group sums + adversary sign counts."""
    start = int(len(entries) * warmup)
    use = entries[start:] or entries[-1:]
    b = defaultdict(float)         # bucket -> value (clamped or signed)
    sign = defaultdict(lambda: [0, 0])   # grp -> [pos, total]
    for e in use:
        sel = set(e.get("selected_ids") or [])
        for sid, sc in (e.get("seller_valuations") or {}).items():
            v = sc.get(key)
            if v is None:
                continue
            v = float(v)
            grp = "adv" if str(sid).startswith("adv") else "benign"
            state = "paid" if sid in sel else "disc"
            b[f"{grp}_{state}"] += (v if signed else max(0.0, v))
            sign[grp][0] += int(v > 0)
            sign[grp][1] += 1
    adv_scored, adv_pos = sign["adv"][1], sign["adv"][0]
    adv_nonpos = adv_scored - adv_pos
    ben_scored, ben_pos = sign["benign"][1], sign["benign"][0]
    b_tot = b["benign_paid"] + b["benign_disc"]
    a_tot = b["adv_paid"] + b["adv_disc"]
    return {
        "benign_paid_pct": (100 * b["benign_paid"] / b_tot) if b_tot else None,
        "adv_paid_pct": (100 * b["adv_paid"] / a_tot) if a_tot else None,
        "benign_mass": round(b_tot, 4), "adv_mass": round(a_tot, 4),
        "adv_scored": adv_scored, "adv_pos": adv_pos, "adv_nonpos": adv_nonpos,
        "ben_scored": ben_scored, "ben_pos": ben_pos,
        "ben_frac_positive": round(ben_pos / ben_scored, 3) if ben_scored else None,
        "n_rounds": len(use),
    }


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (round(mean(vals), 2), round(pstdev(vals) if len(vals) > 1 else 0.0, 2), len(vals))


report = {"T1_warmup_sweep": {}, "T2_no_warmup": {}, "T3_per_seed": {},
          "T4_sign_breakdown": {}, "cross_check": {}}

# preload
data = {cell: {seed: load(vp) for seed, vp in run_dirs(cell)} for cell in CELLS}

# T1 + T2: IF gap across warmups (0.0 == no-warmup baseline)
for cell in CELLS:
    report["T1_warmup_sweep"][cell] = {}
    for w in WARMUPS:
        per_seed = {s: gap_for(ent, "influence_score", w) for s, ent in data[cell].items()}
        populated = [s for s, g in per_seed.items() if g["adv_paid_pct"] is not None]
        report["T1_warmup_sweep"][cell][f"{int(w*100)}%"] = {
            "seeds_populated": len(populated), "populated_seeds": sorted(populated),
            "benign_pct": agg([g["benign_paid_pct"] for g in per_seed.values()]),
            "adv_pct": agg([per_seed[s]["adv_paid_pct"] for s in populated]),
        }
report["T2_no_warmup"] = {c: report["T1_warmup_sweep"][c]["0%"] for c in CELLS}

# T3: per-seed direction (clamped) at 0% and 50%
for cell in CELLS:
    report["T3_per_seed"][cell] = {}
    for w in (0.0, 0.5):
        rows = {}
        for s, ent in data[cell].items():
            g = gap_for(ent, "influence_score", w)
            rows[s] = {"benign_pct": None if g["benign_paid_pct"] is None else round(g["benign_paid_pct"], 1),
                       "adv_pct": None if g["adv_paid_pct"] is None else round(g["adv_paid_pct"], 1),
                       "adv_mass": g["adv_mass"]}
        report["T3_per_seed"][cell][f"{int(w*100)}%"] = rows

# T4: adversary IF sign breakdown (post-warmup 50%) + signed-gap contrast
for cell in CELLS:
    report["T4_sign_breakdown"][cell] = {}
    for s, ent in data[cell].items():
        cl = gap_for(ent, "influence_score", 0.5, signed=False)
        sg = gap_for(ent, "influence_score", 0.5, signed=True)
        report["T4_sign_breakdown"][cell][s] = {
            "adv_scored": cl["adv_scored"], "adv_frac_positive": round(cl["adv_pos"] / cl["adv_scored"], 3) if cl["adv_scored"] else None,
            "ben_frac_positive": cl["ben_frac_positive"],
            "adv_mass_clamped": cl["adv_mass"], "adv_mass_signed": sg["adv_mass"],
            "benign_mass_signed": sg["benign_mass"],
        }

# cross-check: other methods at 50% warmup (matched)
for cell in CELLS:
    report["cross_check"][cell] = {}
    for meth, key in METHOD_KEYS.items():
        per_seed = [gap_for(ent, key, 0.5) for ent in data[cell].values()]
        pop = [g for g in per_seed if g["adv_paid_pct"] is not None]
        report["cross_check"][cell][meth] = {
            "seeds_populated": len(pop),
            "benign_pct": agg([g["benign_paid_pct"] for g in per_seed]),
            "adv_pct": agg([g["adv_paid_pct"] for g in pop]),
        }

print(json.dumps(report, indent=1, default=str))
