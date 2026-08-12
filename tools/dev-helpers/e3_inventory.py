#!/usr/bin/env python3
"""Phase-0 inventory for E3 DP privacy-tax sweep. Read-only, pure stdlib.

For each stepE3_dp_<filter>_eps_<eps>_CIFAR100 scenario: seeds present, rounds,
whether seller_metrics.csv exists (raw BSR source), newest mtime (to confirm the
REGENERATED run), and a sha256 of config_snapshot.json + the DP calibration/eps it
records.
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


def cfg_fingerprint(run_dir):
    for d in (run_dir, os.path.dirname(run_dir), os.path.dirname(os.path.dirname(run_dir))):
        p = os.path.join(d, "config_snapshot.json")
        if os.path.exists(p):
            raw = open(p, "rb").read()
            h = hashlib.sha256(raw).hexdigest()[:12]
            txt = raw.decode("utf-8", "ignore")
            eps = re.search(r"epsilon[=\"':\s]+([0-9.]+|None|none)", txt)
            cal = re.search(r"calibration[=\"':\s]+'?([a-z]+)'?", txt)
            enabled = re.search(r"enabled[=\"':\s]+(True|False)", txt)
            return h, (eps.group(1) if eps else "?"), (cal.group(1) if cal else "?"), \
                   (enabled.group(1) if enabled else "?")
    return None, None, None, None


rows = defaultdict(lambda: defaultdict(dict))
scenarios = sorted(glob.glob(os.path.join(RES, "stepE3_dp_*")))
print("E3 DP-sweep inventory under", RES)
print("scenarios found:", len(scenarios))

for scen in scenarios:
    name = os.path.basename(scen)
    m = re.match(r"stepE3_dp_(?P<filt>fltrust|martfl)_eps_(?P<eps>none|\d+)_", name)
    if not m:
        print("  (unmatched)", name)
        continue
    filt, eps = m.group("filt"), m.group("eps")
    for metrics in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
        rd = os.path.dirname(metrics)
        sm = re.search(r"seed_(\d+)$", os.path.basename(rd))
        seed = int(sm.group(1)) if sm else None
        try:
            fm = json.load(open(metrics))
        except Exception:
            fm = {}
        h, ceps, cal, en = cfg_fingerprint(rd)
        rows[(filt, eps)][seed] = {
            "rounds": fm.get("completed_rounds"),
            "acc": fm.get("acc"),
            "has_sm": os.path.exists(os.path.join(rd, "seller_metrics.csv")),
            "mtime": os.path.getmtime(metrics),
            "cfg": h, "cfg_eps": ceps, "cal": cal, "dp_enabled": en,
        }

EPS_ORDER = ["none", "8", "4", "1"]  # loose -> tight
print("\n{:8} {:>5} {:>7} {:>26} {:>6} {:>10} {:>9} {:>7}".format(
    "filter", "eps", "seeds", "seeds(rounds)", "rounds", "newest", "cal", "cfg_sha"))
print("-" * 92)
for filt in ("fltrust", "martfl"):
    for eps in EPS_ORDER:
        d = rows.get((filt, eps))
        if not d:
            print(f"{filt:8} {eps:>5}   MISSING")
            continue
        seeds = sorted(s for s in d if s is not None)
        seed_str = ",".join(f"{s}({d[s]['rounds']})" for s in seeds)
        newest = max(v["mtime"] for v in d.values())
        newest_s = time.strftime("%Y-%m-%d", time.localtime(newest))
        cals = ",".join(sorted({str(v["cal"]) for v in d.values()}))
        shas = ",".join(sorted({str(v["cfg"]) for v in d.values()}))[:7]
        allsm = all(v["has_sm"] for v in d.values())
        print("{:8} {:>5} {:>7} {:>26} {:>6} {:>10} {:>9} {:>7} {}".format(
            filt, eps, len(seeds), seed_str[:26],
            str(sorted({v["rounds"] for v in d.values()}))[:6],
            newest_s, cals, shas, "" if allsm else "  [seller_metrics MISSING]"))

print("\ncompleteness matrix (X = has seller_metrics; . = absent):")
seeds_all = sorted({s for d in rows.values() for s in d if s is not None})
print("filter   eps  " + " ".join(f"{s:>4}" for s in seeds_all))
for filt in ("fltrust", "martfl"):
    for eps in EPS_ORDER:
        d = rows.get((filt, eps), {})
        line = f"{filt:8} {eps:>3}  "
        for s in seeds_all:
            line += f"{('  X ' if d.get(s, {}).get('has_sm') else '  . '):>4}"
        print(line)
