#!/usr/bin/env python3
"""Break-even analysis for the adaptive (bandit) attacker — R2.O3 / C7.

Reads the black-box (UCB-bandit) adaptive-attack training_log.csv (step 6 /
step7_adaptive_black_box_*), extracts the two empirical inputs, and derives the
break-even point: rounds for the attacker to recoup its entry cost as a function
of the per-round payment.

Two empirical inputs (measured from the log, NOT assumed):
  c  = ENTRY COST in rounds = number of exploration rounds before the attacker is
       reliably accepted. Measured as the first round r after which the per-round
       MSR (= 1 - adversary_detection_rate) stays >= HALF the sustained level.
  s  = SUSTAINED MSR = mean per-round MSR over the post-entry (exploitation) phase.

Economic model (stated explicitly so the paper can cite the right form):
  During the c entry rounds the attacker is rejected -> earns 0 but still pays a
  per-round participation cost kappa (compute + upload). Once established it earns
  s * p per round, where p = total per-round payment to the market and s is its
  captured share. Net revenue after T rounds:
        R(T) = (T - c) * s * p   -   T * kappa
  Break-even T* solves R(T*) = 0:
        T*(p) = c / (1 - kappa/(s*p))          [participation-cost form]
  This is payment-DEPENDENT: as p grows, T* -> c (entry rounds dominate); as
  s*p -> kappa, T* -> infinity (attack never pays off).

  Special case kappa -> 0 (or entry cost measured purely as forgone revenue): the
  payment p cancels and break-even is payment-INDEPENDENT at T* = 2c rounds. Both
  forms are reported; the paper picks the cost model it wants to commit to.

Read-only. Usage:
  python break_even_analysis.py --results_dir RES --out_dir OUT [--kappa_frac 0.25]
"""
import argparse
import csv
import glob
import math
import os
from statistics import mean


def load_msr_series(training_log):
    """-> list of (round, msr) with msr = 1 - adversary_detection_rate."""
    out = []
    with open(training_log, newline="") as fh:
        for row in csv.DictReader(fh):
            adr = row.get("adversary_detection_rate")
            rnd = row.get("round")
            if adr in (None, "") or rnd in (None, ""):
                continue
            try:
                out.append((int(float(rnd)), 1.0 - float(adr)))
            except ValueError:
                continue
    out.sort()
    return out


def entry_and_sustained(series, tail_frac=0.5, window=10):
    """Measure (c, s): entry-cost rounds and sustained MSR.

    MSR from a bandit with few adversaries is coarse and noisy (multiples of
    1/n_adv, with single-round dips to 0), so a raw "stays above threshold
    forever" rule is dominated by late-round noise. We smooth with a `window`-
    round trailing moving average and define:
      s = sustained MSR = mean of raw MSR over the last tail_frac of rounds
      c = entry cost = first round whose MA reaches 90% of s AND stays >= 80% of
          s for the remainder (on the smoothed series). This is robust to the
          coarse 1/n_adv granularity; report c as order-of-magnitude, not exact.
    """
    if not series:
        return None, None
    rounds = [r for r, _ in series]
    msr = [m for _, m in series]
    n = len(series)
    s = mean(msr[int(n * (1 - tail_frac)):] or msr[-1:])
    if s <= 0:
        return None, s
    ma = [mean(msr[max(0, i - window + 1):i + 1]) for i in range(n)]
    c = rounds[-1]
    for i in range(n):
        if ma[i] >= 0.9 * s and all(ma[j] >= 0.8 * s for j in range(i, n)):
            c = rounds[i]
            break
    return c, s


def break_even_curve(c, s, kappa_frac, p_grid):
    """T*(p) with entry cost = forgone revenue, plus optional participation cost.

    Model: the attacker forgoes revenue during the c entry rounds (opportunity
    cost c*s*p) and additionally pays a per-round compute cost kappa on every
    round. Net at round T:  (T-c)*s*p  -  c*s*p  -  T*kappa , so break-even is
        T*(p) = 2c / (1 - kappa/(s*p)),   with  kappa = kappa_frac * s.
    Consistent limits: kappa->0 gives the payment-INDEPENDENT headline T* = 2c
    (the ~30-round anchor); with kappa>0 the curve is always >= 2c and lengthens
    as p falls (participation cost dominates), approaching 2c from above as p->inf.
    """
    kappa = kappa_frac * s          # per-round participation cost (payment units)
    curve = []
    for p in p_grid:
        denom = 1.0 - kappa / (s * p) if s * p > 0 else 0.0
        if denom <= 0:
            curve.append((p, math.inf))
        else:
            curve.append((p, 2 * c / denom))
    return curve, kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--kappa_frac", type=float, default=0.25,
                    help="per-round attacker cost as a fraction of sustained per-round revenue")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # locate black-box (bandit) adaptive logs; fall back to any adaptive log
    logs = sorted(glob.glob(os.path.join(
        a.results_dir, "*step7_adaptive_black_box*", "**", "training_log.csv"), recursive=True))
    if not logs:
        logs = sorted(glob.glob(os.path.join(
            a.results_dir, "*adaptive*", "**", "training_log.csv"), recursive=True))
    if not logs:
        print("NO bandit/adaptive training_log.csv found under", a.results_dir)
        print("Run step 6 first: run_parallel_experiment.py --configs_dir "
              "configs_generated_benchmark/step7_adaptive_attack")
        return

    rows = [("filter", "seed", "entry_cost_c", "sustained_msr_s", "n_rounds", "log")]
    per_filter = {}
    for lg in logs:
        filt = "martfl" if "martfl" in lg else "fltrust" if "fltrust" in lg else "?"
        seed = next((p for p in lg.split(os.sep) if p.startswith("run_")), "?")
        series = load_msr_series(lg)
        c, s = entry_and_sustained(series)
        if c is None:
            continue
        rows.append((filt, seed, c, round(s, 4), len(series), lg))
        per_filter.setdefault(filt, []).append((c, s))
        print(f"  {filt:8} {seed:14} entry_cost c={c:3d} rounds   sustained MSR s={s:.3f}  "
              f"({len(series)} rounds)")

    with open(os.path.join(a.out_dir, "break_even_inputs.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    # aggregate per filter and emit the break-even curve + formula
    p_grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    summary_lines = []
    curves = {}
    for filt, vals in per_filter.items():
        c = mean(v[0] for v in vals)
        s = mean(v[1] for v in vals)
        curve, kappa = break_even_curve(c, s, a.kappa_frac, p_grid)
        curves[filt] = curve
        payind = 2 * c  # kappa->0 payment-independent break-even
        summary_lines.append(
            f"{filt}: entry cost c={c:.1f} rounds, sustained MSR s={s:.3f}; "
            f"payment-independent break-even = 2c = {payind:.0f} rounds; "
            f"participation-cost T*(p) with kappa={a.kappa_frac}*s: "
            + ", ".join(f"p={p:g}->{('inf' if math.isinf(t) else f'{t:.0f}')}" for p, t in curve))
    print("\n".join(summary_lines))

    with open(os.path.join(a.out_dir, "break_even_summary.txt"), "w") as fh:
        fh.write("Break-even analysis (R2.O3 / C7)\n")
        fh.write("model: R(T)=(T-c)*s*p - T*kappa ; T*(p)=c/(1-kappa/(s*p))\n")
        fh.write("kappa_frac=%s (kappa = kappa_frac * s)\n\n" % a.kappa_frac)
        fh.write("\n".join(summary_lines) + "\n")

    # figure: break-even rounds vs per-round payment, per filter.
    # The kappa->0 payment-independent asymptote (2c) is drawn as a labeled
    # horizontal reference so the ~30-round headline is the visual anchor; the
    # participation-cost curve then reads as "lengthens only if participation is
    # costly (at low payment)".
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Minimal text, large fonts: the figure will be printed small, so keep a
        # high font-to-figure ratio and let the caption carry the explanation.
        plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                             "xtick.labelsize": 12, "ytick.labelsize": 12,
                             "legend.fontsize": 13})
        fig, ax = plt.subplots(figsize=(5.0, 3.7))
        colors = {"fltrust": "#1565c0", "martfl": "#c62828"}
        labels = {"fltrust": "FLTrust", "martfl": "MartFL"}
        for filt, curve in curves.items():
            c = mean(v[0] for v in per_filter[filt])
            asymptote = 2 * c
            xs = [p for p, t in curve if not math.isinf(t)]
            ys = [t for p, t in curve if not math.isinf(t)]
            col = colors.get(filt, "#555")
            ax.plot(xs, ys, marker="o", color=col, lw=2.2, ms=6, label=labels.get(filt, filt))
            ax.axhline(asymptote, color=col, ls="--", lw=1.6, alpha=0.9)
        # single prominent headline number, on the MartFL asymptote
        c_m = mean(v[0] for v in per_filter["martfl"]) if "martfl" in per_filter else 0
        ax.annotate(r"$\approx$%d rounds" % round(2 * c_m),
                    xy=(p_grid[-1], 2 * c_m), xytext=(0, 5), textcoords="offset points",
                    ha="right", fontsize=13, fontweight="bold", color=colors["martfl"])
        ax.set_xlabel(r"per-round payment  $p$")
        ax.set_ylabel(r"rounds to break even")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, handlelength=1.4)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "break_even.png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(a.out_dir, "break_even.pdf"), bbox_inches="tight")
        print("wrote break_even.png/.pdf, break_even_inputs.csv, break_even_summary.txt to", a.out_dir)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
