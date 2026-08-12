"""
compute_gini.py — Post-hoc wealth-concentration analysis for Step 15 (pricing).

For every Step 15 result directory, computes:
  - Gini coefficient over ALL sellers' total payment (includes adversaries)
  - Gini coefficient over BENIGN sellers only (fairness among honest sellers)
  - Adversary revenue share (fraction of total payments captured by attackers)
  - Per-(defense, payment_model) aggregates

The output CSV is the data backing for any "marketplace fairness" or
"revenue theft" claim in the paper. It does NOT require rerunning any
experiment — it just walks marketplace_report.json files that already exist.

If `payment_received` isn't yet stored per seller in marketplace_report.json,
this script will fall back to using the seller's selection_rate × an assumed
flat payment, which produces a less interesting Gini but still works as a
proof-of-concept until PaymentSimulator outputs are wired into the report.

Usage:
    python experiments/gradient_market/analysis/compute_gini.py
    python experiments/gradient_market/analysis/compute_gini.py \
        --results_dir ./results --output ./analysis_partial/step15_gini.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------
def gini(values: List[float]) -> float:
    """Standard Gini coefficient.

    0.0  = perfect equality (every seller earns the same amount)
    1.0  = perfect inequality (one seller earns everything)

    Returns 0.0 for empty / all-zero inputs to avoid NaN propagation.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return 0.0
    if v.sum() <= 0:
        return 0.0
    cumv = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cumv) / cumv[-1]) / n)


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------
RE_PAYMENT = re.compile(r"/payment_([a-z_]+?)(?:/|$)")
RE_DEFENSE = re.compile(r"step15_pricing_([a-z_]+)_(?:CIFAR|TREC|Texas|FEMNIST|Purchase)")


def extract_run_metadata(report_path: Path) -> Dict:
    """Pull defense + payment_model + dataset out of the report path."""
    s = str(report_path).replace("\\", "/")
    info: Dict[str, Optional[str]] = {
        "defense": None, "payment_model": None, "dataset": None,
    }
    m = RE_DEFENSE.search(s)
    if m:
        info["defense"] = m.group(1)
    m = RE_PAYMENT.search(s)
    if m:
        info["payment_model"] = m.group(1)
    # Dataset from the scenario dir
    for ds in ("CIFAR100", "CIFAR10", "Texas100", "Purchase100", "FEMNIST", "TREC"):
        if ds in s:
            info["dataset"] = ds
            break
    return info


def gini_for_one_run(report_path: Path) -> Optional[Dict]:
    """Read one marketplace_report.json and return a flat record."""
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        print(f"  ⚠ {report_path}: {exc}")
        return None

    sellers = report.get("seller_summaries", {})
    if not sellers:
        return None

    benign: List[float] = []
    adversary: List[float] = []
    for sid, summary in sellers.items():
        # Prefer explicit payment_received if present (PaymentSimulator output);
        # otherwise fall back to selection_rate as a proxy.
        if "payment_received" in summary:
            value = float(summary["payment_received"])
        elif "total_payment" in summary:
            value = float(summary["total_payment"])
        else:
            value = float(summary.get("selection_rate", 0))

        if summary.get("type") == "adversary":
            adversary.append(value)
        else:
            benign.append(value)

    all_values = benign + adversary
    total = sum(all_values)

    meta = extract_run_metadata(report_path)

    return {
        "run_dir":            str(report_path.parent),
        "defense":            meta["defense"],
        "payment_model":      meta["payment_model"],
        "dataset":            meta["dataset"],
        "n_benign":           len(benign),
        "n_adversary":        len(adversary),
        "total_payment":      total,
        "gini_all":           gini(all_values),
        "gini_benign_only":   gini(benign),
        "adv_revenue_share":  sum(adversary) / total if total > 0 else 0.0,
        "benign_mean_payment": float(np.mean(benign)) if benign else 0.0,
        "adv_mean_payment":    float(np.mean(adversary)) if adversary else 0.0,
        "value_source": (
            "payment_received" if any("payment_received" in s for s in sellers.values())
            else "selection_rate_proxy"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="./results",
                    help="Root experiment results directory")
    ap.add_argument("--output", default="./analysis_partial/step15_gini.csv",
                    help="Path to write the per-run Gini CSV")
    ap.add_argument("--summary", default="./analysis_partial/step15_gini_summary.csv",
                    help="Path to write the per-(defense, payment_model) aggregate CSV")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results dir not found: {results_dir.resolve()}")
        sys.exit(1)

    print(f"📂 Scanning {results_dir.resolve()}")
    report_paths = sorted(results_dir.glob("step15_pricing_*/**/marketplace_report.json"))
    print(f"Found {len(report_paths)} step 15 marketplace reports")

    if not report_paths:
        print("  (no Step 15 marketplace reports — has Step 15 produced any successes yet?)")
        return

    records = []
    for p in report_paths:
        rec = gini_for_one_run(p)
        if rec:
            records.append(rec)

    if not records:
        print("  ⚠ No usable records produced.")
        return

    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"📝 Wrote {len(df)} per-run records → {args.output}")

    # Aggregate by (defense, payment_model)
    if "defense" in df.columns and "payment_model" in df.columns:
        agg = (df.dropna(subset=["defense", "payment_model"])
                 .groupby(["defense", "payment_model"], as_index=False)
                 .agg(n_runs=("run_dir", "count"),
                      gini_all_mean=("gini_all", "mean"),
                      gini_all_std=("gini_all", "std"),
                      gini_benign_mean=("gini_benign_only", "mean"),
                      adv_revenue_share_mean=("adv_revenue_share", "mean"),
                      adv_revenue_share_std=("adv_revenue_share", "std")))
        agg.to_csv(args.summary, index=False)
        print(f"📝 Wrote {len(agg)} per-cell aggregates → {args.summary}")

        print("\n" + "=" * 72)
        print(" PER-CELL SUMMARY (defense × payment_model)")
        print("=" * 72)
        print(f"  {'defense':<14}{'payment':<14}{'n':>4}"
              f"  {'gini_all':>10}{'adv_share':>12}")
        print("  " + "-" * 56)
        for _, row in agg.iterrows():
            print(f"  {str(row['defense']):<14}{str(row['payment_model']):<14}"
                  f"{int(row['n_runs']):>4}  "
                  f"{row['gini_all_mean']:>10.3f}"
                  f"{row['adv_revenue_share_mean']:>12.1%}")

    # Note about the value source so the user knows whether the numbers are
    # backed by real PaymentSimulator output or the proxy fallback.
    sources = df["value_source"].unique() if "value_source" in df else []
    print()
    if "payment_received" in sources:
        print("✅ Numbers backed by real PaymentSimulator `payment_received` values.")
    elif "selection_rate_proxy" in sources:
        print("⚠ marketplace_report.json does not yet contain `payment_received`.")
        print("  Numbers above use selection_rate as a proxy. To get the real Gini,")
        print("  modify markplace_gradient.py's _create_round_record() to call")
        print("  PaymentSimulator and store the per-seller payment_received in")
        print("  the seller_summaries dict.")


if __name__ == "__main__":
    main()
