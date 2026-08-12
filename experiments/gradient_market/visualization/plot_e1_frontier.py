#!/usr/bin/env python3
"""F1 — E1 continuous-payment frontier: Accuracy vs MSR across thresholded-FLTrust tau.

Recomputes (tau, MSR, Acc) from RAW per-round / per-seller outputs (single source
of truth — no cached summaries), writes e1_frontier.csv, and renders F1:
  * one point per tau, line-connected in tau order
  * shaded UNREACHABLE target region {MSR<=20 AND Acc>=40}
  * tau labels; error bars = seed std

Usage:
  python plot_e1_frontier.py --results_dir RESULTS --out_dir OUTDIR
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

TARGET_MSR = 20.0
TARGET_ACC = 40.0


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


def collect(results_dir):
    root = os.path.join(results_dir, "stepE1_tau_sweep_CIFAR100")
    by = defaultdict(lambda: defaultdict(list))
    for metrics in glob.glob(os.path.join(root, "**", "final_metrics.json"), recursive=True):
        run_dir = os.path.dirname(metrics)
        m = re.search(r"tau-(0p\d+|\d+)", run_dir)
        tau = float(m.group(1).replace("p", ".")) if m else None
        try:
            fm = json.load(open(metrics))
        except Exception:
            continue
        smp = os.path.join(run_dir, "seller_metrics.csv")
        msr = bsr = None
        if os.path.exists(smp):
            msr, bsr = rates(smp)
        by[tau]["msr"].append(msr)
        by[tau]["acc"].append(fm.get("acc") * 100 if fm.get("acc") is not None else None)
        by[tau]["asr"].append(fm.get("asr") * 100 if fm.get("asr") is not None else None)
    rows = []
    for tau in sorted(by):
        def ms(key):
            v = [x for x in by[tau][key] if x is not None]
            return (mean(v), pstdev(v) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)
        msr_m, msr_s, n = ms("msr")
        acc_m, acc_s, _ = ms("acc")
        asr_m, asr_s, _ = ms("asr")
        rows.append({"tau": tau, "n_seeds": n,
                     "msr_mean": msr_m, "msr_std": msr_s,
                     "acc_mean": acc_m, "acc_std": acc_s,
                     "asr_mean": asr_m, "asr_std": asr_s})
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["msr_mean"] is not None and r["acc_mean"] is not None]
    rows.sort(key=lambda r: r["tau"])
    msr = [r["msr_mean"] for r in rows]
    acc = [r["acc_mean"] for r in rows]
    msr_e = [r["msr_std"] for r in rows]
    acc_e = [r["acc_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Sparse + large: small canvas, big fonts, few ticks, faint grid, bold marks,
    # so the figure survives heavy downscaling. Caption carries all detail.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 19,
                         "xtick.labelsize": 16, "ytick.labelsize": 16})

    # unreachable target region: MSR <= 20 AND Acc >= 40 (top-left, empty)
    ax.axhspan(TARGET_ACC, 100, xmin=0, xmax=TARGET_MSR / 100.0,
               facecolor="#2e7d32", alpha=0.14, zorder=0)
    ax.text(TARGET_MSR / 2 + 1, 50, "target\n(empty)", ha="center", va="center",
            color="#2e7d32", fontsize=17, fontweight="bold", zorder=2)

    # NO connecting line: a line would assert a continuous frontier a reader
    # could slide along (operating points at MSR 50 / Acc 25), which is exactly
    # what the "cliff, not a trade-off" claim denies. Nothing exists between
    # tau=0 and tau>=0.1 — so plot markers ONLY; the gap is the message.
    ax.errorbar(msr, acc, xerr=msr_e, yerr=acc_e, fmt="o", linestyle="none",
                color="#1565c0", ecolor="#90caf9", capsize=4, ms=12, zorder=4)

    # two labels, placed directly (no leaders): the lone high-accuracy point
    # (tau=0) and the pile of nine collapsed thresholds at bottom-left.
    for r in rows:
        if r["tau"] == 0.0:
            ax.annotate(r"$\tau$=0", (r["msr_mean"], r["acc_mean"]),
                        textcoords="offset points", xytext=(-12, 6), ha="right",
                        fontsize=17, color="#333")
    ax.annotate(r"$\tau\geq$0.1" "\n(9 thresholds)", (4, 1.5),
                textcoords="offset points", xytext=(14, 10), ha="left", va="bottom",
                fontsize=15, color="#333")

    ax.set_xlabel(r"MSR (%)  $\downarrow$")
    ax.set_ylabel(r"Accuracy (%)  $\uparrow$")
    ax.set_xlim(-5, 108)
    ax.set_ylim(-4, 62)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 20, 40, 60])
    ax.grid(axis="y", alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.set_size_inches(4.3, 3.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = collect(a.results_dir)
    write_csv(rows, os.path.join(a.out_dir, "e1_frontier.csv"))
    plot(rows, os.path.join(a.out_dir, "e1_frontier.png"))
    print("wrote", os.path.join(a.out_dir, "e1_frontier.csv"),
          "and e1_frontier.png/.pdf")
    for r in rows:
        print("  tau={tau:<4g} MSR={msr_mean:6.2f} Acc={acc_mean:6.2f} (n={n_seeds})".format(**r))


if __name__ == "__main__":
    main()
