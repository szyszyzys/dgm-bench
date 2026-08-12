#!/usr/bin/env python3
"""F3 — E3 Privacy Tax: BSR vs epsilon (loose->tight) for FLTrust and MartFL.

Recomputes BSR from RAW seller_metrics.csv (single source of truth). x-axis is
the epsilon grid ordered loose->tight (none, 8, 4, 1); the no-noise baseline is
marked. Two lines (one per filter), error bars = seed std.

Usage: python plot_e3_privacy_tax.py --results_dir RES --out_dir OUT
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

EPS_ORDER = ["none", "8", "4", "1"]
EPS_LABEL = {"none": "none\n(no DP)", "8": "8", "4": "4", "1": "1\n(tightest)"}


def bsr_of(sm_path):
    per = defaultdict(lambda: [0, 0])
    with open(sm_path, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("seller_id") or "").strip()
            if not sid.startswith("bn"):
                continue
            per[row.get("round")][0] += int((row.get("selected") or "").strip().lower() == "true")
            per[row.get("round")][1] += 1
    vals = [a / b for a, b in per.values() if b]
    return mean(vals) * 100 if vals else None


def collect(results_dir):
    cells = defaultdict(lambda: defaultdict(list))
    for scen in sorted(glob.glob(os.path.join(results_dir, "stepE3_dp_*"))):
        m = re.match(r"stepE3_dp_(fltrust|martfl)_eps_(none|\d+)_", os.path.basename(scen))
        if not m:
            continue
        filt, eps = m.group(1), m.group(2)
        for smp in glob.glob(os.path.join(scen, "**", "seller_metrics.csv"), recursive=True):
            b = bsr_of(smp)
            if b is not None:
                cells[filt][eps].append(b)
    return cells


def plot(cells, path, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sparse + large: small canvas, big fonts, few ticks, faint grid, no title;
    # caption carries the detail. eps is a real continuum so lines stay.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 19,
                         "xtick.labelsize": 16, "ytick.labelsize": 16})
    rows = [("filter", "eps", "bsr_mean", "bsr_std", "n_seeds")]
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    styles = {"fltrust": ("#1565c0", "o", "FLTrust"), "martfl": ("#c62828", "s", "MartFL")}
    x = list(range(len(EPS_ORDER)))
    for filt in ("fltrust", "martfl"):
        means, stds = [], []
        for eps in EPS_ORDER:
            vals = cells[filt].get(eps, [])
            mu = mean(vals) if vals else float("nan")
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            means.append(mu); stds.append(sd)
            rows.append((filt, eps, round(mu, 3), round(sd, 3), len(vals)))
        color, marker, label = styles[filt]
        ax.errorbar(x, means, yerr=stds, marker=marker, color=color, capsize=4,
                    lw=2.6, ms=9, label=label)

    ax.set_xticks(x, ["∞", "8", "4", "1"])   # eps=none shown as ∞ (no DP)
    ax.set_xlabel(r"$\epsilon$  (loose $\rightarrow$ tight)")
    ax.set_ylabel(r"BSR (%)  $\uparrow$")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="y", alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, handlelength=1.1, handletextpad=0.4,
              labelspacing=0.25, borderpad=0.1, fontsize=12, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")

    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cells = collect(a.results_dir)
    plot(cells, os.path.join(a.out_dir, "e3_privacy_tax.png"),
         os.path.join(a.out_dir, "e3_privacy_tax.csv"))
    print("wrote e3_privacy_tax.png/.pdf and e3_privacy_tax.csv to", a.out_dir)


if __name__ == "__main__":
    main()
