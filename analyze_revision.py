#!/usr/bin/env python
# =============================================================================
# analyze_revision.py — Reviewer-ready analysis of the E1-E4 revision results.
#
# Computes, for each experiment, the pre-registered decision quantity with seed
# variance, compares against the paper's baseline numbers, and emits one
# rebuttal-ready VERDICT line per claim.
#
#   python analyze_revision.py [--results_dir results/revision]
#                              [--baselines paper_baselines.csv]
#                              [--out analysis]
#
# Input schemes (auto-detected, confirmed against what is actually present):
#   A) flat per-seed JSON:  results/revision/e1_<agg>_seed<S>.json,
#      e1_threshold_tau<TAU>_seed<S>.json, e3_<agg>_eps<EPS>_seed<S>.json,
#      e2_<agg>[_<variant>]_seed<S>.json, e4_<val>_seed<S>.json
#   B) the stepE runner layout: results/stepE1_tau_sweep_*/<run>/run_i_seed_s/,
#      results/stepE3_dp_*/, results/stepE2_multitask_*/, results/stepE4_*/
#      (training_log.csv / final_metrics.json / multi_task_summary.csv /
#       valuations.jsonl), as written by the E-step generators + runners.
#
# Conventions (applied everywhere):
#   - mean ± sample std over seeds; 95% CI = mean ± 1.96*std/sqrt(n)
#   - a difference is MEANINGFUL only if |mean1-mean2| > sqrt(std1^2+std2^2);
#     anything smaller is flagged as within seed noise
#   - all rates reported in PERCENT to match the paper's table style
#   - only conditions present in the data are analyzed; none added or dropped
# =============================================================================

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

PCT_METRICS = {"bsr", "msr", "asr", "acc", "revenue_share", "bandit_msr",
               "benign_paid_pct", "benign_discarded_pct",
               "adv_paid_pct", "adv_blocked_pct", "adv_fraction"}


def to_pct(v):
    """Normalize a rate to percent. Heuristic: values <= 1.5 are fractions."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    v = float(v)
    return v * 100.0 if abs(v) <= 1.5 else v


def first_key(d, names):
    """Fetch the first present alias from a dict (case-insensitive)."""
    lower = {str(k).lower(): k for k in d.keys()}
    for n in names:
        if n.lower() in lower:
            return d[lower[n.lower()]]
    return None


def mstd(values):
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def ci95(mean, std, n):
    return float("nan") if n == 0 else 1.96 * std / math.sqrt(n)


def fmt(mean, std, digits=1):
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def meaningful(m1, s1, m2, s2):
    """True if the means differ by more than the combined seed std."""
    if math.isnan(m1) or math.isnan(m2):
        return False
    return abs(m1 - m2) > math.sqrt(s1 ** 2 + s2 ** 2)


# ---------------------------------------------------------------------------
# Step 0 — input discovery
# ---------------------------------------------------------------------------

FLAT_PATTERNS = {
    "e1_threshold": re.compile(r"^e1_threshold_tau(?P<tau>[\d]+(?:[.p][\d]+)?)_seed(?P<seed>\d+)\.json$"),
    "e1": re.compile(r"^e1_(?P<agg>[a-z_]+?)_seed(?P<seed>\d+)\.json$"),
    "e3": re.compile(r"^e3_(?P<agg>[a-z_]+?)_eps(?P<eps>none|[\d.]+)_seed(?P<seed>\d+)\.json$"),
    "e2": re.compile(r"^e2_(?P<agg>[a-z_]+?)(?:_(?P<variant>bandit_persist|bandit|plain))?_seed(?P<seed>\d+)\.json$"),
    "e4": re.compile(r"^e4_(?P<val>[a-z_]+?)_seed(?P<seed>\d+)\.json$"),
}

STEP_GLOBS = ["stepE1_tau_sweep_*", "stepE3_dp_*", "stepE2_multitask_*", "stepE4_valuation_*"]


def discover(results_dir: Path, alt_root: Path):
    """Return ("flat", files) or ("step", scenario_dirs) or (None, [])."""
    print("=" * 72)
    print("STEP 0 — INPUT DISCOVERY")
    print("=" * 72)
    if results_dir.is_dir():
        files = sorted(p for p in results_dir.iterdir() if p.is_file())
        print(f"results dir {results_dir}: {len(files)} files")
        flat = [p for p in files
                if any(rx.match(p.name) for rx in FLAT_PATTERNS.values())]
        if flat:
            by_exp = defaultdict(int)
            for p in flat:
                for tag, rx in FLAT_PATTERNS.items():
                    if rx.match(p.name):
                        by_exp[tag.split("_")[0]] += 1
                        break
            print(f"  scheme A (flat per-seed JSON) detected: {dict(by_exp)}")
            unmatched = [p.name for p in files if p not in flat]
            if unmatched:
                print(f"  NOTE: {len(unmatched)} file(s) match no known pattern: {unmatched[:8]}")
            return "flat", flat
    for root in (results_dir, alt_root):
        if not root.is_dir():
            continue
        dirs = []
        for g in STEP_GLOBS:
            dirs.extend(sorted(root.glob(g)))
        if dirs:
            print(f"  scheme B (stepE runner layout) detected under {root}: "
                  f"{[d.name for d in dirs]}")
            return "step", dirs
    print(f"  NO revision results found in {results_dir} (or stepE* under "
          f"{alt_root}). Nothing to analyze.")
    return None, []


def print_structure(label, obj, max_keys=40):
    print(f"\n  sample structure [{label}]:")
    if isinstance(obj, dict):
        for k in list(obj.keys())[:max_keys]:
            v = obj[k]
            tag = f"list[{len(v)}]" if isinstance(v, list) else type(v).__name__
            print(f"    {k}: {tag}")
    elif isinstance(obj, pd.DataFrame):
        print(f"    columns: {list(obj.columns)}")


# ---------------------------------------------------------------------------
# Ingestion — normalize either scheme into per-seed record DataFrames
# ---------------------------------------------------------------------------

ALIASES = {
    "bsr": ["bsr", "benign_selection_rate"],
    "msr": ["msr", "adv_selection_rate", "malicious_selection_rate"],
    "asr": ["asr", "test_asr", "attack_success_rate"],
    "acc": ["acc", "accuracy", "test_acc", "B-Acc", "clean_accuracy"],
    "revenue_share": ["adversary_revenue_share", "adv_revenue_share"],
}

E2_ALIASES = {
    "active_benign": ["active_benign_per_task", "n_active_benign", "active_benign"],
    "adv_fraction": ["adversarial_fraction_per_task", "adversarial_fraction_active", "adv_fraction"],
    "cum_adv_revenue": ["cumulative_adversary_revenue_per_task", "cumulative_adversary_revenue"],
    "bsr": ["bsr_per_task", "bsr"],
    "msr": ["msr_per_task", "msr"],
    "acc": ["acc_per_task", "acc"],
    "bandit_msr": ["bandit_msr_per_task", "bandit_msr"],
    "unpaid_streaks": ["unpaid_streaks", "final_unpaid_streaks", "unpaid_streak_distribution"],
    "payment_undefined": ["payment_undefined_rounds", "payment_undefined"],
}

E4_ALIASES = {
    "benign_paid_pct": ["benign_paid_pct", "benign_paid"],
    "benign_discarded_pct": ["benign_discarded_pct", "benign_discarded"],
    "adv_paid_pct": ["adv_paid_pct", "adversary_paid_pct", "adv_paid"],
    "adv_blocked_pct": ["adv_blocked_pct", "adversary_blocked_pct", "adv_blocked"],
}


def _metrics_from_json(d):
    return {m: to_pct(first_key(d, names)) for m, names in ALIASES.items()}


def ingest_flat(files):
    """Scheme A: one JSON per (experiment, condition, seed)."""
    e1a, e1b, e3, e2, e4 = [], [], [], [], []
    shown = set()
    for p in files:
        with open(p) as f:
            d = json.load(f)
        name = p.name
        m = FLAT_PATTERNS["e1_threshold"].match(name)
        if m:
            if "e1b" not in shown:
                print_structure(name, d); shown.add("e1b")
            e1b.append({"tau": float(m.group("tau").replace("p", ".")),
                        "seed": int(m.group("seed")), **_metrics_from_json(d)})
            continue
        m = FLAT_PATTERNS["e3"].match(name)
        if m:
            if "e3" not in shown:
                print_structure(name, d); shown.add("e3")
            eps = m.group("eps")
            e3.append({"agg": m.group("agg"), "eps": eps if eps == "none" else float(eps),
                       "seed": int(m.group("seed")), **_metrics_from_json(d)})
            continue
        m = FLAT_PATTERNS["e2"].match(name)
        if m and name.startswith("e2_"):
            if "e2" not in shown:
                print_structure(name, d); shown.add("e2")
            rec = {"agg": m.group("agg"), "variant": m.group("variant") or "plain",
                   "seed": int(m.group("seed"))}
            for key, names in E2_ALIASES.items():
                rec[key] = first_key(d, names)
            e2.append(rec)
            continue
        m = FLAT_PATTERNS["e4"].match(name)
        if m and name.startswith("e4_"):
            if "e4" not in shown:
                print_structure(name, d); shown.add("e4")
            rows = d if isinstance(d, list) else [d]
            for r in rows:
                e4.append({
                    "method": m.group("val"), "seed": int(m.group("seed")),
                    "defense": first_key(r, ["defense", "aggregator"]) or "all",
                    "attack": first_key(r, ["attack", "attack_condition"]) or "all",
                    **{k: to_pct(first_key(r, names)) for k, names in E4_ALIASES.items()},
                    "leastcore_computed": first_key(r, ["leastcore_computed_rounds", "leastcore_computed"]),
                    "leastcore_total": first_key(r, ["leastcore_total_rounds", "total_rounds"]),
                })
            continue
        m = FLAT_PATTERNS["e1"].match(name)
        if m and name.startswith("e1_"):
            if "e1a" not in shown:
                print_structure(name, d); shown.add("e1a")
            e1a.append({"agg": m.group("agg"), "seed": int(m.group("seed")),
                        **_metrics_from_json(d)})
            continue
    return (pd.DataFrame(e1a), pd.DataFrame(e1b), pd.DataFrame(e3),
            pd.DataFrame(e2), pd.DataFrame(e4))


# --- Scheme B: the stepE runner layout (same parsing as the extract_stepE* scripts)

def _run_metrics_step(run_dir: Path):
    out = {}
    lp = run_dir / "training_log.csv"
    if lp.exists():
        log = pd.read_csv(lp)
        if "false_positive_rate" in log:
            out["bsr"] = to_pct(1.0 - log["false_positive_rate"].dropna().mean())
        if "adversary_detection_rate" in log:
            out["msr"] = to_pct(1.0 - log["adversary_detection_rate"].dropna().mean())
        if "adversary_revenue_share" in log:
            out["revenue_share"] = to_pct(log["adversary_revenue_share"].dropna().mean())
    fp = run_dir / "final_metrics.json"
    if fp.exists():
        fm = json.load(open(fp))
        out["acc"] = to_pct(fm.get("acc", fm.get("B-Acc")))
        out["asr"] = to_pct(fm.get("asr"))
    return out


E4_METHOD_KEYS = {
    "kernelshap": "kernelshap_score", "banzhaf": "banzhaf_score",
    "leastcore": "leastcore_score", "loo": "marginal_contrib_loo",
    "influence": "influence_score",
}


def _e4_gap_from_jsonl(val_path: Path, warmup_frac=0.5):
    entries = [json.loads(x) for x in open(val_path) if x.strip()]
    if not entries:
        return {}, (0, 0)
    entries = entries[int(len(entries) * warmup_frac):] or entries[-1:]
    sums = defaultdict(lambda: defaultdict(float))
    lc_rounds, total_rounds = 0, len(entries)
    for e in entries:
        selected = set(e.get("selected_ids") or [])
        any_lc = False
        for sid, scores in (e.get("seller_valuations") or {}).items():
            adv = str(sid).startswith("adv")
            paid = sid in selected
            for method, key in E4_METHOD_KEYS.items():
                v = scores.get(key)
                if v is None:
                    continue
                if method == "leastcore":
                    any_lc = True
                sums[method][f"{'adv' if adv else 'benign'}_{'paid' if paid else 'discarded'}"] += max(0.0, float(v))
        lc_rounds += int(any_lc)
    out = {}
    for method, b in sums.items():
        bt = b["benign_paid"] + b["benign_discarded"]
        at = b["adv_paid"] + b["adv_discarded"]
        out[method] = {
            "benign_paid_pct": 100 * b["benign_paid"] / bt if bt else float("nan"),
            "benign_discarded_pct": 100 * b["benign_discarded"] / bt if bt else float("nan"),
            "adv_paid_pct": 100 * b["adv_paid"] / at if at else float("nan"),
            "adv_blocked_pct": 100 * b["adv_discarded"] / at if at else float("nan"),
        }
    return out, (lc_rounds, total_rounds)


def ingest_step(scenario_dirs):
    e1a, e1b, e3, e2, e4 = [], [], [], [], []
    shown = set()
    for sdir in scenario_dirs:
        name = sdir.name
        if name.startswith("stepE1_"):
            # Seed dirs can sit several levels below the scenario dir (the
            # parallel runner nests <runname>/default_hps/<runname>/run_i_seed_s);
            # recurse and read tau from the path.
            for seed_dir in sorted(sdir.glob("**/run_*_seed_*")):
                if not seed_dir.is_dir():
                    continue
                m = re.search(r"tau-([0-9]+(?:p[0-9]+)?)", str(seed_dir))
                if not m:
                    continue
                tau = float(m.group(1).replace("p", "."))
                rec = {"tau": tau, "seed": int(seed_dir.name.split("_seed_")[-1]),
                       **_run_metrics_step(seed_dir)}
                e1b.append(rec)
                if "e1b" not in shown:
                    print_structure(str(seed_dir), rec); shown.add("e1b")
        elif name.startswith("stepE3_"):
            m = re.match(r"stepE3_dp_(?P<agg>.+)_eps_(?P<eps>[^_]+)_", name)
            if not m:
                continue
            eps = m.group("eps")
            for seed_dir in sorted(sdir.glob("**/run_*_seed_*")):
                if not seed_dir.is_dir():
                    continue
                rec = {"agg": m.group("agg"),
                       "eps": eps if eps == "none" else float(eps),
                       "seed": int(seed_dir.name.split("_seed_")[-1]),
                       **_run_metrics_step(seed_dir)}
                e3.append(rec)
                if "e3" not in shown:
                    print_structure(str(seed_dir), rec); shown.add("e3")
        elif name.startswith("stepE2_"):
            m = re.match(r"stepE2_multitask_(?P<agg>[a-z_]+?)_(?P<variant>plain|bandit)_", name)
            variant = m.group("variant") if m else "plain"
            agg = m.group("agg") if m else name
            for seed_dir in sorted(sdir.glob("**/run_*_seed_*")):
                sp = seed_dir / "multi_task_summary.csv"
                if not sp.exists():
                    continue
                s = pd.read_csv(sp)
                rec = {
                    "agg": agg, "variant": variant,
                    "seed": int(seed_dir.name.split("_seed_")[-1]),
                    "active_benign": s["n_active_benign"].tolist(),
                    "adv_fraction": s["adversarial_fraction_active"].tolist(),
                    "cum_adv_revenue": s["cumulative_adversary_revenue"].tolist(),
                    "bsr": s["bsr"].tolist(), "msr": s["msr"].tolist(),
                    "acc": s["acc"].tolist(),
                    "bandit_msr": s["bandit_msr"].tolist() if "bandit_msr" in s else None,
                    "payment_undefined": s.get("payment_undefined_rounds", pd.Series(dtype=float)).sum(),
                    "unpaid_streaks": None,
                }
                ms = seed_dir / "market_state.json"
                if ms.exists():
                    st = json.load(open(ms))
                    rec["unpaid_streaks"] = [v.get("unpaid_streak") for v in st.values()
                                             if v.get("type") == "benign"]
                e2.append(rec)
                if "e2" not in shown:
                    print_structure(str(seed_dir), rec); shown.add("e2")
        elif name.startswith("stepE4_"):
            m = re.match(r"stepE4_valuation_(?P<defense>.+)_(?P<attack>no_attack|backdoor)_", name)
            if not m:
                continue
            for seed_dir in sorted(sdir.glob("**/run_*_seed_*")):
                vp = seed_dir / "valuations.jsonl"
                if not vp.exists():
                    continue
                gaps, (lc, tot) = _e4_gap_from_jsonl(vp)
                seed = int(seed_dir.name.split("_seed_")[-1])
                for method, g in gaps.items():
                    e4.append({"method": method, "seed": seed,
                               "defense": m.group("defense"), "attack": m.group("attack"),
                               **g, "leastcore_computed": lc, "leastcore_total": tot})
                if "e4" not in shown and gaps:
                    print_structure(str(seed_dir), e4[-1]); shown.add("e4")
    return (pd.DataFrame(e1a), pd.DataFrame(e1b), pd.DataFrame(e3),
            pd.DataFrame(e2), pd.DataFrame(e4))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = """\
# paper_baselines.csv — the paper's existing reported numbers (percent).
# source: table5 (CIFAR-100 main results) or fig7 (valuation gap, MartFL).
source,dataset,aggregator,metric,value
table5,CIFAR100,fedavg,acc,
table5,CIFAR100,fedavg,asr,
table5,CIFAR100,fedavg,bsr,
table5,CIFAR100,fedavg,msr,
table5,CIFAR100,fltrust,acc,
table5,CIFAR100,fltrust,asr,
table5,CIFAR100,fltrust,bsr,
table5,CIFAR100,fltrust,msr,
table5,CIFAR100,martfl,acc,
table5,CIFAR100,martfl,asr,
table5,CIFAR100,martfl,bsr,
table5,CIFAR100,martfl,msr,
table5,CIFAR100,skymask,acc,
table5,CIFAR100,skymask,asr,
table5,CIFAR100,skymask,bsr,
table5,CIFAR100,skymask,msr,
fig7,CIFAR100,martfl,benign_paid_pct,
fig7,CIFAR100,martfl,benign_discarded_pct,
fig7,CIFAR100,martfl,adv_paid_pct,
fig7,CIFAR100,martfl,adv_blocked_pct,
"""


def load_baselines(path: Path, out_dir: Path):
    if not path.exists():
        print(f"\nWARNING: baselines file '{path}' not found.")
        print("A template has been written to "
              f"{out_dir / 'paper_baselines.TEMPLATE.csv'} — fill it with the "
              "paper's Table 5 / Fig. 7 numbers and re-run for sanity anchors.")
        (out_dir / "paper_baselines.TEMPLATE.csv").write_text(BASELINE_TEMPLATE)
        return None
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["value"])
    print(f"\nLoaded {len(df)} baseline values from {path}.")
    return df


def baseline_value(baselines, source, agg, metric):
    if baselines is None:
        return None
    rows = baselines[(baselines["source"] == source)
                     & (baselines["aggregator"].str.lower() == agg.lower())
                     & (baselines["metric"].str.lower() == metric.lower())]
    return float(rows["value"].iloc[0]) if len(rows) else None


# ---------------------------------------------------------------------------
# Aggregation + verdicts
# ---------------------------------------------------------------------------

def agg_table(df, group_cols, metrics):
    rows = []
    grouper = group_cols if len(group_cols) > 1 else group_cols[0]
    for key, g in df.groupby(grouper, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        for m in metrics:
            if m not in g:
                continue
            mean, std, n = mstd(g[m])
            row[f"{m}_mean"], row[f"{m}_std"] = round(mean, 3), round(std, 3)
            row[f"{m}_ci95"] = round(ci95(mean, std, n), 3)
            row[f"{m}"] = fmt(mean, std)
            row["n_seeds"] = n
        rows.append(row)
    return pd.DataFrame(rows)


class Insights:
    def __init__(self):
        self.lines = defaultdict(list)
        self.sanity = []

    def add(self, reviewer_pts, exp, claim, verdict, evidence):
        line = f"[{exp}][{claim}] VERDICT: {verdict} Evidence: {evidence}"
        for rp in reviewer_pts:
            self.lines[rp].append(line)
        print(f"  -> {line}")

    def write(self, out_dir: Path):
        order = ["R2.O5", "R3.O3", "R1.O2", "R2.O6", "R2.O3", "R1.O3", "R3.O4"]
        parts = ["# INSIGHTS — rebuttal-ready verdicts (E1-E4)\n",
                 "All numbers are mean ± seed std over the seeds present (95% CI in the CSVs).",
                 "Differences are called meaningful only when they exceed the combined seed std.\n"]
        for rp in order:
            parts.append(f"\n## {rp}\n")
            if self.lines.get(rp):
                parts.extend(f"- {ln}" for ln in dict.fromkeys(self.lines[rp]))
            elif rp == "R3.O4":
                parts.append("- No step11 (root-size / biased-D_B) results were found. Run "
                             "`--step 11` and re-run this script; see E1's thresholded-FLTrust "
                             "baseline under R3.O3 if the point concerns explicit selection "
                             "objectives.")
            else:
                parts.append("- (no data present for this point)")
        (out_dir / "INSIGHTS.md").write_text("\n".join(parts), encoding="utf-8")

    def write_sanity(self, out_dir: Path):
        parts = ["# SANITY — anchors against paper_baselines.csv\n"]
        if not self.sanity:
            parts.append("No sanity anchors were evaluated (baselines file missing "
                         "or no matching conditions in the data).")
        parts.extend(self.sanity)
        (out_dir / "SANITY.md").write_text("\n".join(parts), encoding="utf-8")


def try_figure(fn, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fn(plt)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  wrote {path}")
    except Exception as e:
        print(f"  WARNING: figure {path} skipped ({e})")


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------

def analyze_e1(e1a, e1b, out_dir, ins):
    print("\n" + "=" * 72 + "\nE1 — Untenable Dilemma: binary-payment artifact? (R2.O5, R3.O3)\n" + "=" * 72)
    rpts = ["R2.O5", "R3.O3"]

    # (A) continuous payment vs binary MSR
    src = e1a if not e1a.empty else (e1b if (not e1b.empty and "revenue_share" in e1b) else pd.DataFrame())
    if not src.empty and "revenue_share" in src and src["revenue_share"].notna().any():
        group = "agg" if "agg" in src else "tau"
        t = agg_table(src.dropna(subset=["revenue_share"]), [group], ["revenue_share", "msr", "bsr"])
        t.to_csv(out_dir / "e1_continuous_payment.csv", index=False)
        for _, r in t.iterrows():
            gap = abs(r["revenue_share_mean"] - r["msr_mean"])
            comb = math.sqrt(r["revenue_share_std"] ** 2 + r["msr_std"] ** 2)
            cond = r[group]
            ev = (f"adversary_revenue_share = {fmt(r['revenue_share_mean'], r['revenue_share_std'])}% vs "
                  f"binary MSR = {fmt(r['msr_mean'], r['msr_std'])}% ({group}={cond}); gap {gap:.1f}pp, "
                  f"combined seed std {comb:.1f}pp.")
            if gap <= comb:
                v = ("The dilemma is NOT an artifact of the binary reward model: under "
                     "continuous weight-proportional payment, adversaries capture a revenue "
                     "share statistically indistinguishable from their binary selection rate — "
                     "the filter's own aggregation weights leak the value.")
            elif r["revenue_share_mean"] < r["msr_mean"]:
                v = ("Conditioned: continuous payment reduces the adversary's captured value "
                     "below the binary MSR by more than seed noise, so binarization amplifies "
                     "the dilemma's magnitude — though adversaries still capture a "
                     f"{r['revenue_share_mean']:.1f}% revenue share.")
            else:
                v = ("Strengthened: under continuous payment adversaries capture MORE value "
                     "than the binary MSR suggests — graded weights are worse, not better.")
            ins.add(rpts, "E1", f"A1:{cond}", v, ev)
    else:
        print("  (A) no adversary_revenue_share data present — skipping claim A1.")

    # (B) tau frontier
    if e1b.empty:
        print("  (B) no thresholded-FLTrust runs present — skipping claim A2.")
        return
    t = agg_table(e1b, ["tau"], ["msr", "bsr", "asr", "acc", "revenue_share"]).sort_values("tau")
    t.to_csv(out_dir / "e1_tau_frontier.csv", index=False)
    print(f"  wrote e1_tau_frontier.csv ({len(t)} tau values)")

    ok = t[(t["msr_mean"] <= 20) & (t["acc_mean"] >= 40)]
    i_min_msr = t["msr_mean"].idxmin()
    i_max_acc = t["acc_mean"].idxmax()
    lo, hi = t.loc[i_min_msr], t.loc[i_max_acc]
    trade = (f"min-MSR point: tau={lo['tau']:g} (MSR {fmt(lo['msr_mean'], lo['msr_std'])}%, "
             f"Acc {fmt(lo['acc_mean'], lo['acc_std'])}%); max-Acc point: tau={hi['tau']:g} "
             f"(Acc {fmt(hi['acc_mean'], hi['acc_std'])}%, MSR {fmt(hi['msr_mean'], hi['msr_std'])}%).")
    if ok.empty:
        v = ("The dilemma survives an explicit selection-integrity objective: across the "
             "entire tau grid, no thresholded-FLTrust operating point achieves MSR ≤ 20% and "
             "Acc ≥ 40% simultaneously — tightening tau cuts MSR only at the cost of accuracy.")
        ins.add(rpts, "E1", "A2", v, trade)
    else:
        b = ok.sort_values("msr_mean").iloc[0]
        near = (abs(b["msr_mean"] - 20) < b["msr_std"]) or (abs(b["acc_mean"] - 40) < b["acc_std"])
        v = (f"Constructive exception at tau={b['tau']:g}: thresholded FLTrust jointly achieves "
             f"MSR {fmt(b['msr_mean'], b['msr_std'])}% and Acc {fmt(b['acc_mean'], b['acc_std'])}% — "
             "an honest positive design finding: with selection integrity as an explicit objective, "
             "the joint criterion is attainable."
             + (" (Note: clears a threshold only within seed noise — borderline.)" if near else ""))
        ins.add(rpts, "E1", "A2", v, trade)

    def fig(plt):
        fig_, ax = plt.subplots(figsize=(6, 4.5))
        ax.errorbar(t["msr_mean"], t["acc_mean"], xerr=t["msr_std"], yerr=t["acc_std"],
                    marker="o", capsize=3)
        for _, r in t.iterrows():
            ax.annotate(f"{r['tau']:g}", (r["msr_mean"], r["acc_mean"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.axvspan(0, 20, ymin=0, ymax=1, alpha=0.0)
        import matplotlib.patches as mp
        ylim = max(45, ax.get_ylim()[1])
        ax.add_patch(mp.Rectangle((0, 40), 20, ylim - 40, alpha=0.15, color="green",
                                  label="target: MSR≤20, Acc≥40"))
        ax.set_xlabel("MSR (%)"); ax.set_ylabel("Acc (%)")
        ax.set_title("E1b — thresholded-FLTrust frontier over tau")
        ax.legend()
    try_figure(fig, out_dir / "e1_frontier.png")


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------

EPS_ORDER = ["none", 8.0, 4.0, 1.0]


def analyze_e3(e3, out_dir, ins, baselines):
    print("\n" + "=" * 72 + "\nE3 — Privacy Tax (R1.O2)\n" + "=" * 72)
    if e3.empty:
        print("  no E3 data present.")
        return
    t = agg_table(e3, ["agg", "eps"], ["bsr", "msr", "asr", "acc"])
    t["eps_order"] = t["eps"].apply(lambda e: EPS_ORDER.index(e) if e in EPS_ORDER else 99)
    t = t.sort_values(["agg", "eps_order"]).drop(columns="eps_order")
    t.to_csv(out_dir / "e3_dp_sweep.csv", index=False)
    print(f"  wrote e3_dp_sweep.csv ({len(t)} cells)")

    for agg_name, g in t.groupby("agg"):
        g = g.set_index("eps")
        if "none" not in g.index or 1.0 not in g.index:
            print(f"  {agg_name}: missing eps=none or eps=1 — cannot test the claim.")
            continue
        none_r, one_r = g.loc["none"], g.loc[1.0]
        drop = none_r["bsr_mean"] - one_r["bsr_mean"]
        comb = math.sqrt(none_r["bsr_std"] ** 2 + one_r["bsr_std"] ** 2)
        seq = [g.loc[e]["bsr_mean"] for e in EPS_ORDER if e in g.index]
        mono = all(a >= b for a, b in zip(seq, seq[1:]))
        ev = (f"BSR: none {fmt(none_r['bsr_mean'], none_r['bsr_std'])}% -> eps=1 "
              f"{fmt(one_r['bsr_mean'], one_r['bsr_std'])}% (drop {drop:.1f}pp, combined std {comb:.1f}pp); "
              f"sequence over eps none->8->4->1: {[round(x,1) for x in seq]}"
              f" ({'monotone' if mono else 'NOT monotone'}).")
        if drop > comb:
            v = (f"Privacy Tax confirmed for {agg_name}: DP noise on honest sellers alone "
                 f"depresses BSR by {drop:.1f}pp at eps=1, beyond seed noise — the filter "
                 "misclassifies privacy-induced variance as malice.")
        elif drop > 0:
            v = (f"Privacy Tax NOT confirmed beyond noise for {agg_name}: BSR drops only "
                 f"{drop:.1f}pp (combined std {comb:.1f}pp) — within seed noise; check whether "
                 "the clip bound dominates the noise scale before concluding.")
        else:
            v = (f"Privacy Tax NOT observed for {agg_name}: BSR did not decrease at eps=1; "
                 "the claim is not substantiated at these budgets.")
        ins.add(["R1.O2"], "E3", f"PrivacyTax:{agg_name}", v, ev)

        msr_up = one_r["msr_mean"] - none_r["msr_mean"]
        comb_m = math.sqrt(none_r["msr_std"] ** 2 + one_r["msr_std"] ** 2)
        if msr_up > comb_m:
            ins.add(["R1.O2"], "E3", f"Secondary:{agg_name}",
                    f"Secondary effect present for {agg_name}: MSR rises {msr_up:.1f}pp at eps=1 "
                    "(beyond noise) — the noisier honest cluster eases adversarial camouflage.",
                    f"MSR none {fmt(none_r['msr_mean'], none_r['msr_std'])}% -> eps=1 "
                    f"{fmt(one_r['msr_mean'], one_r['msr_std'])}%.")

        # sanity anchor at eps=none
        for metric in ("bsr", "msr", "acc"):
            base = baseline_value(baselines, "table5", agg_name, metric)
            if base is None:
                continue
            mean, std = none_r[f"{metric}_mean"], none_r[f"{metric}_std"]
            if abs(mean - base) <= max(std, 1e-9):
                ins.sanity.append(f"- OK: E3 eps=none {agg_name} {metric.upper()} "
                                  f"{fmt(mean, std)}% matches Table 5 ({base}%) within seed std.")
            else:
                w = (f"- **WARNING**: E3 eps=none {agg_name} {metric.upper()} = {fmt(mean, std)}% "
                     f"vs Table 5 = {base}% — outside seed std; something drifted "
                     "(config, seeds, or code) — investigate before using these numbers.")
                ins.sanity.append(w)
                print(f"  {w}")

    def fig(plt):
        fig_, ax = plt.subplots(figsize=(6, 4))
        for agg_name, g in t.groupby("agg"):
            g = g.set_index("eps").reindex([e for e in EPS_ORDER if e in g["eps"].values])
            x = range(len(g))
            ax.errorbar(x, g["bsr_mean"], yerr=g["bsr_std"], marker="o", capsize=3, label=agg_name)
            ax.set_xticks(list(x), [str(e) for e in g.index])
        ax.set_xlabel("epsilon (left = no DP, right = most private)")
        ax.set_ylabel("BSR (%)"); ax.set_title("E3 — Privacy Tax"); ax.legend()
    try_figure(fig, out_dir / "e3_privacy_tax.png")


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------

def _per_task(e2_rows, field):
    """rows -> DataFrame(task x stats) for a list-valued field."""
    arrays = [r for r in e2_rows[field] if isinstance(r, (list, tuple))]
    if not arrays:
        return None
    T = max(len(a) for a in arrays)
    rows = []
    for ti in range(T):
        vals = [a[ti] for a in arrays if len(a) > ti and a[ti] is not None
                and not (isinstance(a[ti], float) and math.isnan(a[ti]))]
        mean, std, n = mstd(vals)
        rows.append({"task": ti + 1, "mean": mean, "std": std, "n": n})
    return pd.DataFrame(rows)


def analyze_e2(e2, out_dir, ins):
    print("\n" + "=" * 72 + "\nE2 — Persistent market (R2.O6, R2.O3)\n" + "=" * 72)
    rpts = ["R2.O6", "R2.O3"]
    if e2.empty:
        print("  no E2 data present.")
        return

    all_rows = []
    for (agg_name, variant), g in e2.groupby(["agg", "variant"]):
        per = {}
        for field, label in [("active_benign", "active_benign"),
                             ("adv_fraction", "adv_fraction"),
                             ("cum_adv_revenue", "cum_adv_revenue"),
                             ("bsr", "bsr"), ("msr", "msr"), ("acc", "acc"),
                             ("bandit_msr", "bandit_msr")]:
            if field in g:
                d = _per_task(g, field)
                if d is not None:
                    if label in ("adv_fraction", "bsr", "msr", "acc", "bandit_msr"):
                        for c in ("mean", "std"):
                            d[c] = d[c].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
                    per[label] = d
                    for _, r in d.iterrows():
                        all_rows.append({"agg": agg_name, "variant": variant, "task": r["task"],
                                         "metric": label, "mean": round(r["mean"], 3),
                                         "std": round(r["std"], 3), "n_seeds": r["n"]})
        if "active_benign" not in per or "adv_fraction" not in per:
            continue

        ab, af = per["active_benign"], per["adv_fraction"]
        b0, bT = ab.iloc[0], ab.iloc[-1]
        f0, fT = af.iloc[0], af.iloc[-1]
        decline = meaningful(b0["mean"], b0["std"], bT["mean"], bT["std"]) and bT["mean"] < b0["mean"]
        rise = meaningful(f0["mean"], f0["std"], fT["mean"], fT["std"]) and fT["mean"] > f0["mean"]
        rev = per.get("cum_adv_revenue")
        rev_ev = (f"; cumulative adversary revenue {rev.iloc[0]['mean']:.2f} -> "
                  f"{rev.iloc[-1]['mean']:.2f}" if rev is not None else "")
        ev = (f"active benign sellers {fmt(b0['mean'], b0['std'])} (task 1) -> "
              f"{fmt(bT['mean'], bT['std'])} (task {int(bT['task'])}); adversarial fraction "
              f"{fmt(f0['mean'], f0['std'])}% -> {fmt(fT['mean'], fT['std'])}%{rev_ev} "
              f"[{agg_name}/{variant}].")
        if decline and rise:
            v = (f"Persistent-market premise confirmed for {agg_name}: under repeated queries the "
                 "honest side of the market attrits and the active population tilts adversarial — "
                 "single-task metrics understate the long-run damage of adverse selection.")
        elif not decline and abs(b0["mean"] - bT["mean"]) <= math.sqrt(b0["std"]**2 + bT["std"]**2):
            streaks = [s for r in g["unpaid_streaks"] if isinstance(r, (list, tuple)) for s in r]
            sm, ss, _ = mstd(streaks)
            extra = (f" Final benign unpaid-streak distribution: mean {sm:.2f} ± {ss:.2f} — "
                     "compare against exit_k to judge how close attrition came; tighten exit_k "
                     "to probe the margin." if streaks else "")
            v = (f"Attrition did not trigger for {agg_name}: active benign count is flat across "
                 "tasks — the finding is that this filter pays honest sellers often enough to "
                 "retain them under repeated queries." + extra)
        else:
            v = (f"Mixed for {agg_name}: benign decline={'yes' if decline else 'no'}, "
                 f"adversarial-fraction rise={'yes' if rise else 'no'} (beyond noise); "
                 "premise only partially supported — report both components.")
        ins.add(rpts, "E2", f"Persistence:{agg_name}/{variant}", v, ev)

    pd.DataFrame(all_rows).to_csv(out_dir / "e2_per_task.csv", index=False)
    print(f"  wrote e2_per_task.csv ({len(all_rows)} rows)")

    # Bandit amortization (persist vs reset, if both exist)
    bandit = e2[e2["variant"].str.startswith("bandit")] if "variant" in e2 else pd.DataFrame()
    if not bandit.empty and bandit["bandit_msr"].apply(lambda x: isinstance(x, (list, tuple))).any():
        for variant, g in bandit.groupby("variant"):
            d = _per_task(g, "bandit_msr")
            if d is None or len(d) < 2:
                continue
            for c in ("mean", "std"):
                d[c] = d[c].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
            t1 = d.iloc[0]
            rest_m, rest_s, _ = mstd(d.iloc[1:]["mean"])
            amort = rest_m - t1["mean"] > math.sqrt(t1["std"] ** 2 + rest_s ** 2)
            ev = (f"bandit MSR task 1 = {fmt(t1['mean'], t1['std'])}%, mean tasks 2..T = "
                  f"{rest_m:.1f}% [{variant}].")
            if variant == "bandit_persist" or "persist" in variant:
                v = ("Exploration cost amortizes across tasks: the persistent bandit pays its "
                     "probing cost in task 1 and exploits thereafter."
                     if amort else
                     "Amortization not demonstrated beyond noise: the persistent bandit's "
                     "task-1 vs later-task MSR difference is within seed std.")
                ins.add(["R2.O3"], "E2", "BanditAmortization", v, ev)
            else:
                ins.add(["R2.O3"], "E2", f"BanditBaseline:{variant}",
                        "Reset-bandit baseline recorded for comparison against the persistent run.", ev)

    def fig_attr(plt):
        fig_, ax = plt.subplots(figsize=(6, 4))
        for (a, vnt), g in e2.groupby(["agg", "variant"]):
            d = _per_task(g, "active_benign")
            if d is not None:
                ax.errorbar(d["task"], d["mean"], yerr=d["std"], marker="o", capsize=3,
                            label=f"{a}/{vnt}")
        ax.set_xlabel("task"); ax.set_ylabel("active benign sellers")
        ax.set_title("E2 — attrition"); ax.legend()
    try_figure(fig_attr, out_dir / "e2_attrition.png")

    def fig_rev(plt):
        fig_, ax = plt.subplots(figsize=(6, 4))
        for (a, vnt), g in e2.groupby(["agg", "variant"]):
            d = _per_task(g, "cum_adv_revenue")
            if d is not None:
                ax.errorbar(d["task"], d["mean"], yerr=d["std"], marker="o", capsize=3,
                            label=f"{a}/{vnt}")
        ax.set_xlabel("task"); ax.set_ylabel("cumulative adversary revenue")
        ax.set_title("E2 — cumulative adversary revenue"); ax.legend()
    try_figure(fig_rev, out_dir / "e2_cum_revenue.png")

    def fig_comp(plt):
        fig_, ax = plt.subplots(figsize=(6, 4))
        for (a, vnt), g in e2.groupby(["agg", "variant"]):
            d = _per_task(g, "adv_fraction")
            if d is not None:
                m = d["mean"].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
                s = d["std"].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
                ax.errorbar(d["task"], m, yerr=s, marker="o", capsize=3, label=f"{a}/{vnt}")
        ax.set_xlabel("task"); ax.set_ylabel("adversarial fraction of active sellers (%)")
        ax.set_title("E2 — market composition"); ax.legend()
    try_figure(fig_comp, out_dir / "e2_composition.png")

    if not bandit.empty:
        def fig_bandit(plt):
            fig_, ax = plt.subplots(figsize=(6, 4))
            for vnt, g in bandit.groupby("variant"):
                d = _per_task(g, "bandit_msr")
                if d is not None:
                    m = d["mean"].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
                    s = d["std"].apply(lambda v: v * 100 if abs(v) <= 1.5 else v)
                    ax.errorbar(d["task"], m, yerr=s, marker="o", capsize=3, label=vnt)
            ax.set_xlabel("task"); ax.set_ylabel("bandit MSR (%)")
            ax.set_title("E2 — bandit exploitation across tasks"); ax.legend()
        try_figure(fig_bandit, out_dir / "e2_bandit.png")


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------

def analyze_e4(e4, out_dir, ins, baselines):
    print("\n" + "=" * 72 + "\nE4 — Valuation gap across solution concepts (R1.O3)\n" + "=" * 72)
    if e4.empty:
        print("  no E4 data present.")
        return
    metrics = ["benign_paid_pct", "benign_discarded_pct", "adv_paid_pct", "adv_blocked_pct"]
    t = agg_table(e4, ["defense", "attack", "method"], metrics)
    t.to_csv(out_dir / "e4_valuation_gap.csv", index=False)
    print(f"  wrote e4_valuation_gap.csv ({len(t)} cells)")

    # Least-core coverage — count only rows that actually carry coverage info
    # (in the flat scheme that is the leastcore method's own rows).
    cov_txt = ""
    if "leastcore_computed" in e4 and e4["leastcore_computed"].notna().any():
        lc = (e4[e4["leastcore_computed"].notna()]
              .drop_duplicates(subset=["defense", "attack", "seed"]))
        comp = lc["leastcore_computed"].fillna(0).sum()
        tot = lc["leastcore_total"].fillna(0).sum()
        frac = 100.0 * comp / tot if tot else 0.0
        cov_txt = f"Least Core computable on {frac:.0f}% of valuation rounds."
        print(f"  {cov_txt}")
        if frac < 50:
            cov_txt += (" Below 50% — Least Core is reported only 'where computable' and "
                        "Banzhaf serves as the primary robustness check.")

    primary = "kernelshap"
    for (defense, attack), g in t.groupby(["defense", "attack"]):
        g = g.set_index("method")
        if primary not in g.index:
            continue
        k = g.loc[primary]
        consistent, divergent = [], []
        for method in g.index:
            if method in (primary, "selection"):
                continue
            r = g.loc[method]
            if math.isnan(r["adv_paid_pct_mean"]):
                continue
            same = (not meaningful(r["adv_paid_pct_mean"], r["adv_paid_pct_std"],
                                   k["adv_paid_pct_mean"], k["adv_paid_pct_std"])
                    and not meaningful(r["benign_discarded_pct_mean"], r["benign_discarded_pct_std"],
                                       k["benign_discarded_pct_mean"], k["benign_discarded_pct_std"]))
            (consistent if same else divergent).append(method)
        ev = (f"[{defense}/{attack}] KernelSHAP: adv-paid {fmt(k['adv_paid_pct_mean'], k['adv_paid_pct_std'])}%, "
              f"benign-discarded {fmt(k['benign_discarded_pct_mean'], k['benign_discarded_pct_std'])}%. "
              f"Within combined seed std of KernelSHAP: {consistent or 'none'}; divergent: {divergent or 'none'}. "
              "Banzhaf/Least Core reuse the SAME coalition samples as KernelSHAP, so differences are "
              "solution-concept, not sampling noise. " + cov_txt)
        if not divergent and consistent:
            v = (f"The valuation gap is robust across solution concepts for {defense} ({attack}): "
                 "every computable method reproduces KernelSHAP's benign-discarded / adversary-paid "
                 "pattern within seed noise — the Fig. 7 finding is not a KernelSHAP artifact.")
        elif divergent == ["leastcore"]:
            v = (f"Robust across semivalues for {defense} ({attack}); Least Core (a stability concept, "
                 "not a semivalue) diverges — an interesting nuance: the gap is about "
                 "marginal-contribution accounting rather than coalitional stability.")
        elif divergent:
            v = (f"Partially robust for {defense} ({attack}): {', '.join(divergent)} diverge beyond "
                 "seed noise from KernelSHAP — report the gap as method-dependent for these.")
        else:
            v = (f"Only KernelSHAP computable for {defense} ({attack}) — robustness cannot be "
                 "assessed from this cell.")
        ins.add(["R1.O3"], "E4", f"Robustness:{defense}/{attack}", v, ev)

        # Fig. 7 sanity anchor (MartFL)
        if defense.lower() == "martfl" and baselines is not None:
            for metric in metrics:
                base = baseline_value(baselines, "fig7", "martfl", metric)
                if base is None:
                    continue
                mean, std = k[f"{metric}_mean"], k[f"{metric}_std"]
                if abs(mean - base) <= max(std, 1e-9):
                    ins.sanity.append(f"- OK: E4 martfl/{attack} kernelshap {metric} "
                                      f"{fmt(mean, std)}% matches Fig. 7 ({base}%) within seed std.")
                else:
                    w = (f"- **WARNING**: E4 martfl/{attack} kernelshap {metric} = {fmt(mean, std)}% "
                         f"vs Fig. 7 = {base}% — outside seed std; investigate drift.")
                    ins.sanity.append(w)
                    print(f"  {w}")

    def fig(plt):
        sub = t[t["attack"] == "backdoor"] if (t["attack"] == "backdoor").any() else t
        defenses = sub["defense"].unique()
        fig_, axes = plt.subplots(1, len(defenses), figsize=(4 * len(defenses), 4), squeeze=False)
        for ax, defense in zip(axes[0], defenses):
            g = sub[sub["defense"] == defense].set_index("method")
            methods = [m for m in ["kernelshap", "banzhaf", "leastcore", "loo", "influence"]
                       if m in g.index]
            x = np.arange(len(methods))
            ax.bar(x - 0.2, [g.loc[m, "adv_paid_pct_mean"] for m in methods], width=0.4,
                   yerr=[g.loc[m, "adv_paid_pct_std"] for m in methods], capsize=3,
                   label="adversary paid %")
            ax.bar(x + 0.2, [g.loc[m, "benign_discarded_pct_mean"] for m in methods], width=0.4,
                   yerr=[g.loc[m, "benign_discarded_pct_std"] for m in methods], capsize=3,
                   label="benign discarded %")
            ax.set_xticks(x, methods, rotation=30, ha="right")
            ax.set_title(defense); ax.set_ylabel("%")
        axes[0][0].legend()
        fig_.suptitle("E4 — valuation gap across solution concepts")
    try_figure(fig, out_dir / "e4_gap.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# step11 — root-size scarcity + biased D_B  (R3.O4)
# ---------------------------------------------------------------------------

STEP11_GLOBS = ["step11_*"]

STEP11_RE = re.compile(r"^step11_(?P<agg>[a-z_]+?)_(?P<dataset>\w+)$")

# family -> (x-axis label, human name, "which direction is more adverse")
STEP11_FAMILIES = {
    "scarcity":    ("buyer root-set ratio |D_B|", "root size", "smaller"),
    "vary_buyer":  ("buyer/root Dirichlet alpha", "biased D_B", "smaller"),
    "vary_seller": ("seller Dirichlet alpha", "seller heterogeneity", "smaller"),
}


def _level_value(level: str):
    """'ratio_0.05' -> 0.05 ; 'alpha_0.1' -> 0.1 ; else None."""
    m = re.search(r"([0-9]*\.?[0-9]+)$", level or "")
    return float(m.group(1)) if m else None


def ingest_step11(scenario_dirs):
    """-> DataFrame(agg, dataset, family, level, level_val, seed, bsr, msr, acc, asr)."""
    rows = []
    for sdir in scenario_dirs:
        m = STEP11_RE.match(sdir.name)
        agg = m.group("agg") if m else sdir.name
        dataset = m.group("dataset") if m else "?"
        for metrics_file in sdir.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            try:
                rel = run_dir.relative_to(sdir).parts
            except ValueError:
                continue
            if len(rel) < 2:
                continue
            family, level = rel[0], rel[1]
            if family not in STEP11_FAMILIES:
                continue
            seed_m = re.search(r"seed_(\d+)$", run_dir.name)
            met = _run_metrics_step(run_dir)
            if not met:
                continue
            rows.append({
                "agg": agg, "dataset": dataset, "family": family,
                "level": level, "level_val": _level_value(level),
                "seed": int(seed_m.group(1)) if seed_m else -1,
                **met,
            })
    return pd.DataFrame(rows)


def analyze_step11(s11, out_dir, ins, baselines):
    """R3.O4 — do the core failures survive every root size, and does a biased
    (non-representative) D_B degrade reference-based filters further?"""
    print("\n" + "=" * 72 + "\nstep11 — Root size + biased D_B (R3.O4)\n" + "=" * 72)
    if s11 is None or s11.empty:
        print("  no step11 data present.")
        return

    t = agg_table(s11, ["agg", "family", "level"], ["bsr", "msr", "acc", "asr"])
    t["level_val"] = t["level"].apply(_level_value)
    t = t.sort_values(["family", "agg", "level_val"])
    t.to_csv(out_dir / "step11_root_and_bias.csv", index=False)
    print(f"  wrote step11_root_and_bias.csv ({len(t)} cells)")

    # ---- (a) do the core failures hold at EVERY root size? -----------------
    scar = t[t["family"] == "scarcity"]
    for agg_name, g in scar.groupby("agg"):
        g = g.sort_values("level_val")
        msr_seq = [round(v, 1) for v in g["msr_mean"]]
        acc_seq = [round(v, 1) for v in g["acc_mean"]]
        levels = list(g["level"])
        if not msr_seq:
            continue
        lo, hi = min(msr_seq), max(msr_seq)
        ev = (f"MSR across root sizes {levels}: {msr_seq} (min {lo}%, max {hi}%, "
              f"spread {hi - lo:.1f}pp); ACC {acc_seq}.")
        if lo >= 30.0:
            v = (f"Core failure is root-size invariant for {agg_name}: MSR stays >= {lo:.1f}% at "
                 f"every root size tested — enlarging the buyer's reference set does NOT buy "
                 "filtering integrity, so the failure is structural rather than a data-budget artifact.")
        else:
            v = (f"Core failure is NOT uniform across root sizes for {agg_name}: MSR falls to "
                 f"{lo:.1f}% at some root size — the claim 'holds at every root size' is too strong "
                 "as stated; report the per-size numbers instead.")
        ins.add(["R3.O4"], "step11", f"RootSizeInvariance:{agg_name}", v, ev)

    # ---- (b) does a biased D_B degrade reference-based filters further? ----
    # NOTE: a mean +- std summary is actively misleading here. On a biased root
    # set the seeds are bimodal — some runs train normally and others collapse
    # to chance — so we report the COLLAPSE RATE first and treat the mean drop
    # as secondary. A "30pp average degradation" would describe a gradual
    # decline that does not occur in the data.
    bias = t[t["family"] == "vary_buyer"]
    raw_bias = s11[s11["family"] == "vary_buyer"] if "family" in s11 else pd.DataFrame()
    for agg_name, g in bias.groupby("agg"):
        g = g.sort_values("level_val")
        if len(g) < 2:
            continue
        worst, best = g.iloc[0], g.iloc[-1]     # lowest alpha = most biased
        d_acc = best["acc_mean"] - worst["acc_mean"]
        comb = math.sqrt(best["acc_std"] ** 2 + worst["acc_std"] ** 2)

        # per-seed accuracies at the most-biased level; "collapsed" = the run
        # never learned (well under a quarter of the representative-root mean).
        seed_accs, collapsed, n_seeds = [], 0, 0
        thresh = 0.25 * best["acc_mean"]
        if not raw_bias.empty:
            sel = raw_bias[(raw_bias["agg"] == agg_name) & (raw_bias["level"] == worst["level"])]
            seed_accs = sorted(round(float(a), 1) for a in sel["acc"].dropna())
            n_seeds = len(seed_accs)
            collapsed = sum(1 for a in seed_accs if a < thresh)

        ev = (f"alpha {worst['level']} (most biased) vs {best['level']} (most representative): "
              f"ACC {fmt(worst['acc_mean'], worst['acc_std'])}% vs "
              f"{fmt(best['acc_mean'], best['acc_std'])}% (drop {d_acc:.1f}pp, combined std {comb:.1f}pp); "
              f"per-seed ACC at {worst['level']}: {seed_accs} "
              f"({collapsed}/{n_seeds} below {thresh:.1f}% = collapsed); "
              f"MSR {fmt(worst['msr_mean'], worst['msr_std'])}% vs "
              f"{fmt(best['msr_mean'], best['msr_std'])}%; "
              f"ASR {fmt(worst['asr_mean'], worst['asr_std'])}% vs "
              f"{fmt(best['asr_mean'], best['asr_std'])}%.")

        if n_seeds and 0 < collapsed < n_seeds:
            v = (f"Biased D_B makes {agg_name} UNSTABLE, not merely worse: {collapsed}/{n_seeds} seeds "
                 f"collapse to chance while the rest train normally ({seed_accs}). Report this as a "
                 "bimodal failure / collapse rate — the mean+-std is not a meaningful summary here, and "
                 f"quoting an average '{d_acc:.1f}pp degradation' would imply a gradual decline that "
                 "does not occur. A non-representative root set is a correctness risk, not a tuning knob.")
        elif n_seeds and collapsed == n_seeds:
            v = (f"Biased D_B breaks {agg_name} outright: all {n_seeds} seeds collapse to chance at "
                 f"{worst['level']} — the reference gradient carries no usable signal.")
        elif d_acc > comb:
            v = (f"Biased D_B materially degrades {agg_name}: a non-representative root set costs "
                 f"{d_acc:.1f}pp of accuracy beyond seed noise, with no seed collapsing outright.")
        else:
            v = (f"Biased D_B does NOT degrade {agg_name} beyond seed noise: ACC moves "
                 f"{d_acc:.1f}pp against a combined std of {comb:.1f}pp — do not claim a bias "
                 "penalty for this filter on this evidence.")
        ins.add(["R3.O4"], "step11", f"BiasedRoot:{agg_name}", v, ev)

    # ---- figure: 2 rows (MSR, ACC) x families ------------------------------
    families = [f for f in STEP11_FAMILIES if f in set(t["family"])]

    def fig(plt):
        fig_, axes = plt.subplots(2, len(families), figsize=(4.2 * len(families), 6.4),
                                  sharex="col", squeeze=False)
        for col, fam in enumerate(families):
            xlabel, nice, _ = STEP11_FAMILIES[fam]
            sub = t[t["family"] == fam]
            raw_fam = s11[s11["family"] == fam] if "family" in s11 else pd.DataFrame()
            for row, metric in enumerate(("msr", "acc")):
                ax = axes[row][col]
                for agg_name, g in sub.groupby("agg"):
                    g = g.sort_values("level_val")
                    line = ax.errorbar(g["level_val"], g[f"{metric}_mean"], yerr=g[f"{metric}_std"],
                                       marker="o", capsize=3, label=agg_name)
                    # Overlay per-seed points: where seeds are bimodal (biased
                    # D_B collapses some runs to chance) the mean+-std alone
                    # would imply a gradual spread that is not what happened.
                    if not raw_fam.empty:
                        r = raw_fam[raw_fam["agg"] == agg_name]
                        if metric in r:
                            ax.scatter(r["level_val"], r[metric], s=12, alpha=0.45,
                                       color=line.lines[0].get_color(), zorder=3)
                ax.set_ylim(0, 100)
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(nice)
                    ax.set_ylabel("MSR (%)  lower = better" if col == 0 else "")
                else:
                    ax.set_ylabel("ACC (%)  higher = better" if col == 0 else "")
                    ax.set_xlabel(xlabel)
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
        fig_.suptitle("F5 — Root size and buyer-set bias (R3.O4)")
        fig_.tight_layout()
    try_figure(fig, out_dir / "step11_root_and_bias.png")


def main():
    ap = argparse.ArgumentParser(description="Analyze E1-E4 revision results into rebuttal-ready insights")
    ap.add_argument("--results_dir", default="results/revision")
    ap.add_argument("--baselines", default="paper_baselines.csv")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scheme, items = discover(Path(args.results_dir), Path("results"))
    if scheme is None:
        sys.exit(1)
    if scheme == "flat":
        e1a, e1b, e3, e2, e4 = ingest_flat(items)
    else:
        e1a, e1b, e3, e2, e4 = ingest_step(items)

    # step11 (root size + biased D_B) lives in its own scenario dirs and is
    # discovered independently of the stepE* scheme above.
    s11_dirs = []
    for root in (Path(args.results_dir), Path("results")):
        if not root.is_dir():
            continue
        for g in STEP11_GLOBS:
            s11_dirs.extend(sorted(root.glob(g)))
        if s11_dirs:
            break
    s11 = ingest_step11(s11_dirs) if s11_dirs else pd.DataFrame()

    print("\nINPUT MAP:")
    for label, df in [("E1a (continuous payment)", e1a), ("E1b (tau frontier)", e1b),
                      ("E3 (DP sweep)", e3), ("E2 (multi-task)", e2), ("E4 (valuation gap)", e4),
                      ("step11 (root size / biased D_B)", s11)]:
        if df.empty:
            print(f"  {label}: ABSENT")
        else:
            seeds = sorted(df["seed"].unique()) if "seed" in df else "?"
            print(f"  {label}: {len(df)} records, seeds {seeds}")

    baselines = load_baselines(Path(args.baselines), out_dir)
    ins = Insights()

    analyze_e1(e1a, e1b, out_dir, ins)
    analyze_e3(e3, out_dir, ins, baselines)
    analyze_e2(e2, out_dir, ins)
    analyze_e4(e4, out_dir, ins, baselines)
    analyze_step11(s11, out_dir, ins, baselines)

    ins.write(out_dir)
    ins.write_sanity(out_dir)
    print(f"\n✅ Analysis complete. See {out_dir}/INSIGHTS.md and {out_dir}/SANITY.md")


if __name__ == "__main__":
    main()
