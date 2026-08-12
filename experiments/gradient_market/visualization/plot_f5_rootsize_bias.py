#!/usr/bin/env python3
"""F5 — Root-size + biased-D_B (R3.O4). Recompute from RAW; plot HONESTLY.

Two rows:
  Row A — root SIZE sweep (scarcity/ratio_*): MSR, BSR, Acc vs ratio, both filters.
          Shows the failure is flat across sizes (Claim 1).
  Row B — biased-D_B sweep (vary_buyer/alpha_*): MSR, BSR, and Acc vs alpha.
          Acc panel plots PER-SEED POINTS (no mean line) so the bimodal collapse
          is visible; a faint mean is drawn only for reference. MSR/BSR panels
          show selection metrics do NOT worsen under bias (Claim 2 rewording).

Read-only. Usage: python plot_f5_rootsize_bias.py --results_dir RES --out_dir OUT
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

FILTERS = [("fltrust", "FLTrust", "#1565c0"), ("martfl", "MartFL", "#c62828")]


def rates(sm):
    per = defaultdict(lambda: {"a": [0, 0], "b": [0, 0]})
    with open(sm, newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("seller_id") or "").strip()
            k = "a" if s.startswith("adv") else "b" if s.startswith("bn") else None
            if not k:
                continue
            sl = per[r.get("round")][k]
            sl[0] += int((r.get("selected") or "").strip().lower() == "true")
            sl[1] += 1
    msr = [d["a"][0]/d["a"][1] for d in per.values() if d["a"][1]]
    bsr = [d["b"][0]/d["b"][1] for d in per.values() if d["b"][1]]
    return (mean(msr)*100 if msr else None, mean(bsr)*100 if bsr else None)


def lvlnum(l):
    m = re.search(r"([0-9.]+)$", l)
    return float(m.group(1)) if m else 0


def collect(results_dir):
    # data[family][filter][level] = list of {seed,msr,bsr,acc}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for filt, _, _ in FILTERS:
        scen = os.path.join(results_dir, f"step11_{filt}_CIFAR100")
        for metrics in glob.glob(os.path.join(scen, "**", "final_metrics.json"), recursive=True):
            rd = os.path.dirname(metrics)
            rel = os.path.relpath(rd, scen).split(os.sep)
            if len(rel) < 2:
                continue
            fam, lvl = rel[0], rel[1]
            if fam not in ("scarcity", "vary_buyer"):
                continue
            try:
                fm = json.load(open(metrics))
            except Exception:
                continue
            msr = bsr = None
            smp = os.path.join(rd, "seller_metrics.csv")
            if os.path.exists(smp):
                msr, bsr = rates(smp)
            data[fam][filt][lvl].append({"msr": msr, "bsr": bsr,
                                         "acc": (fm.get("acc") or 0)*100})
    return data


def ms(vals):
    vals = [v for v in vals if v is not None]
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)


def plot(data, png, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_csv = [("family", "filter", "level", "level_val", "metric", "mean", "std", "per_seed")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    row_spec = [("scarcity", "A — root size  (Claim 1: failure flat across sizes)", "root ratio |D_B|"),
                ("vary_buyer", "B — biased D_B  (Claim 2: bias collapses accuracy, not selection)",
                 "buyer-root Dirichlet α  (← more biased    representative →)")]

    for ri, (fam, rtitle, xlabel) in enumerate(row_spec):
        for ci, (metric, ylab) in enumerate([("msr", "MSR (%)  ↓"), ("bsr", "BSR (%)  ↑"),
                                              ("acc", "Acc (%)  ↑")]):
            ax = axes[ri][ci]
            for filt, flabel, color in FILTERS:
                levels = sorted(data[fam][filt], key=lvlnum)
                xs = [lvlnum(l) for l in levels]
                means, stds = [], []
                for l in levels:
                    vals = [r[metric] for r in data[fam][filt][l]]
                    m, s = ms(vals)
                    means.append(m); stds.append(s)
                    rows_csv.append((fam, filt, l, lvlnum(l), metric, round(m, 2), round(s, 2),
                                     [round(v, 1) for v in vals if v is not None]))
                    # per-seed points on the ACC panels (both rows) — this is the finding
                    if metric == "acc":
                        ax.scatter([lvlnum(l)] * len(vals), [v for v in vals], s=26,
                                   color=color, alpha=0.55, zorder=3,
                                   edgecolors="white", linewidths=0.4)
                if metric == "acc":
                    # faint mean line for reference only (finding is the scatter)
                    ax.plot(xs, means, color=color, lw=1.0, ls="--", alpha=0.5, label=f"{flabel} (mean)")
                else:
                    ax.errorbar(xs, means, yerr=stds, marker="o", color=color, capsize=3,
                                lw=1.7, ms=6, label=flabel)
            if metric == "acc" and fam == "vary_buyer":
                ax.axhspan(0, 5, color="#888", alpha=0.12)
                ax.text(ax.get_xlim()[0], 6.5, "chance (collapse)", fontsize=7.5, color="#555")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlabel, fontsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="center left")
            if ci == 0:
                ax.text(-0.22, 1.06, rtitle, transform=ax.transAxes, fontsize=10,
                        fontweight="bold", va="bottom")
    fig.suptitle("F5 — Root-set size and bias (R3.O4): failure is size-invariant; "
                 "bias collapses accuracy (per-seed) while selection metrics do not move",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = collect(a.results_dir)
    plot(data, os.path.join(a.out_dir, "f5_rootsize_bias.png"),
         os.path.join(a.out_dir, "f5_rootsize_bias.csv"))
    print("wrote f5_rootsize_bias.png/.pdf and .csv to", a.out_dir)


if __name__ == "__main__":
    main()
