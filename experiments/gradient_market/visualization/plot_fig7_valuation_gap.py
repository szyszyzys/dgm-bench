#!/usr/bin/env python3
"""Fig. 7 — valuation-gap misalignment across solution concepts (MartFL cells).

Recomputes the benign/adversary value-share from RAW valuations.jsonl (warmup =
drop first 50% of rounds; negatives clamped to 0, matching the Fig-7 metric) for
KernelSHAP, Banzhaf, LOO on the MartFL cells, and plots them with seed-std error
bars. Influence and Least Core are annotated as omitted (see IF_RESOLUTION.md and
E4_STATUS.md) — NOT plotted, because IF is a signed measure incompatible with a
non-negative value-share and Least Core's Core is empty on MartFL.

Read-only. Usage: python plot_fig7_valuation_gap.py --results_dir RES --out_dir OUT
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

WARMUP = 0.5
PLOT_METHODS = [("kernelshap", "KernelSHAP", "kernelshap_score"),
                ("banzhaf", "Banzhaf", "banzhaf_score"),
                ("loo", "LOO", "marginal_contrib_loo")]
CELLS = [("martfl_backdoor", "MartFL — backdoor"),
         ("martfl_no_attack", "MartFL — no attack")]


def gap(entries, key):
    start = int(len(entries) * WARMUP)
    use = entries[start:] or entries[-1:]
    b = defaultdict(float)
    for e in use:
        sel = set(e.get("selected_ids") or [])
        for sid, sc in (e.get("seller_valuations") or {}).items():
            v = sc.get(key)
            if v is None:
                continue
            grp = "adv" if str(sid).startswith("adv") else "benign"
            st = "paid" if sid in sel else "disc"
            b[f"{grp}_{st}"] += max(0.0, float(v))
    bt = b["benign_paid"] + b["benign_disc"]
    at = b["adv_paid"] + b["adv_disc"]
    return (100 * b["benign_paid"] / bt if bt else None,
            100 * b["adv_paid"] / at if at else None)


def collect(results_dir):
    out = {}
    for cell, _ in CELLS:
        scen = os.path.join(results_dir, f"stepE4_valuation_{cell}_CIFAR100")
        per = defaultdict(lambda: {"benign": [], "adv": []})
        for vp in glob.glob(os.path.join(scen, "**", "valuations.jsonl"), recursive=True):
            ent = []
            with open(vp) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            ent.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if not ent:
                continue
            for mkey, _, jkey in PLOT_METHODS:
                bp, ap = gap(ent, jkey)
                if bp is not None:
                    per[mkey]["benign"].append(bp)
                if ap is not None:
                    per[mkey]["adv"].append(ap)
        out[cell] = per
    return out


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), 0.0, 0)
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def plot(data, png, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Record BOTH cells to the CSV (the no-attack numbers back the one-line text
    # mention), but PLOT only the backdoor cell: the no-attack cell has no
    # adversaries, so it cannot illustrate the separation story and would make
    # the "under every method" claim look contradicted.
    rows = [("cell", "method", "group", "mean_pct", "std_pct", "n_seeds")]
    for cell, _ in CELLS:
        for grp in ("benign", "adv"):
            for mkey, _, _ in PLOT_METHODS:
                m, s, n = ms(data[cell][mkey][grp])
                rows.append((cell, mkey, grp, round(m, 2) if m == m else "nan", round(s, 2), n))

    # Sparse + large: small canvas, big fonts, no title, no in-plot omission
    # note (that belongs in the caption). Caption carries all detail.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 18,
                         "xtick.labelsize": 16, "ytick.labelsize": 16})
    cell = "martfl_backdoor"
    per = data[cell]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bar_w = 0.38
    colors = {"benign": "#1565c0", "adv": "#c62828"}
    xpos = np.arange(len(PLOT_METHODS))

    for gi, grp in enumerate(("benign", "adv")):
        means = [ms(per[mkey][grp])[0] for mkey, _, _ in PLOT_METHODS]
        stds = [ms(per[mkey][grp])[1] for mkey, _, _ in PLOT_METHODS]
        bars = ax.bar(xpos + (gi - 0.5) * bar_w, means, bar_w, yerr=stds, capsize=4,
                      color=colors[grp], alpha=0.9,
                      label=("honest" if grp == "benign" else "adversary"))
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 3, f"{m:.0f}", ha="center",
                    va="bottom", fontsize=13, color=colors[grp])

    ax.set_xticks(xpos, [lbl for _, lbl, _ in PLOT_METHODS])
    ax.set_ylim(0, 119)
    ax.set_yticks([0, 50, 100])
    ax.axhline(50, color="#888", ls=":", lw=1.2)
    ax.grid(axis="y", alpha=0.18)
    ax.set_ylabel("value paid (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=13,
              handlelength=1.1, handletextpad=0.4, columnspacing=1.0,
              borderpad=0.1)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = collect(a.results_dir)
    plot(data, os.path.join(a.out_dir, "fig7_valuation_gap.png"),
         os.path.join(a.out_dir, "fig7_valuation_gap.csv"))
    print("wrote fig7_valuation_gap.png/.pdf and .csv to", a.out_dir)


if __name__ == "__main__":
    main()
