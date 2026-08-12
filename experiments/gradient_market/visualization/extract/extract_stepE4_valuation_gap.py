# =============================================================================
# Step E4 extractor — valuation-gap comparison across solution concepts.
#
# Recomputes the Fig-7 quantities (% benign value paid / discarded, % adversary
# value paid / blocked) from valuations.jsonl for EVERY valuation method —
# kernelshap, banzhaf, leastcore, loo, influence, plus the raw selection rate —
# and writes one combined table so the valuation gap can be compared across
# methods (is the gap an artifact of KernelSHAP, or robust?).
#
# Mirrors the accounting of visualization/step21_valuation.py
# (load_dual_breakdown): a seller's per-round value is attributed to
# paid/discarded by membership in selected_ids; the default gross-positive
# view clips negative scores at 0 (pass --use-signed to keep signs).
#
#   python extract_stepE4_valuation_gap.py [--results_dir ./results]
# =============================================================================

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

METHOD_KEYS = {
    "selection": "selection_score",
    "kernelshap": "kernelshap_score",
    "banzhaf": "banzhaf_score",
    "leastcore": "leastcore_score",
    "loo": "marginal_contrib_loo",
    "influence": "influence_score",
}

SCENARIO_RE = re.compile(r"stepE4_valuation_(?P<defense>.+)_(?P<attack>no_attack|backdoor)_(?P<dataset>.+)$")


def gap_for_run(val_path: Path, warmup_frac: float, use_signed: bool):
    """Per-method sums of value over (benign/adv) x (paid/discarded)."""
    entries = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        return {}

    start = int(len(entries) * warmup_frac)
    entries = entries[start:] or entries[-1:]

    sums = defaultdict(lambda: defaultdict(float))  # method -> bucket -> value
    for entry in entries:
        selected = set(entry.get("selected_ids") or [])
        for sid, scores in (entry.get("seller_valuations") or {}).items():
            is_adv = str(sid).startswith("adv")
            paid = sid in selected
            for method, key in METHOD_KEYS.items():
                v = scores.get(key)
                if v is None:
                    continue
                v = float(v)
                if not use_signed:
                    v = max(0.0, v)
                bucket = f"{'adv' if is_adv else 'benign'}_{'paid' if paid else 'discarded'}"
                sums[method][bucket] += v

    out = {}
    for method, buckets in sums.items():
        b_paid = buckets.get('benign_paid', 0.0)
        b_disc = buckets.get('benign_discarded', 0.0)
        a_paid = buckets.get('adv_paid', 0.0)
        a_disc = buckets.get('adv_discarded', 0.0)
        b_tot, a_tot = b_paid + b_disc, a_paid + a_disc
        out[method] = {
            'benign_paid_pct': 100.0 * b_paid / b_tot if b_tot else None,
            'benign_discarded_pct': 100.0 * b_disc / b_tot if b_tot else None,
            'adv_paid_pct': 100.0 * a_paid / a_tot if a_tot else None,
            'adv_blocked_pct': 100.0 * a_disc / a_tot if a_tot else None,
            'benign_total_value': b_tot,
            'adv_total_value': a_tot,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract Step E4 valuation-gap comparison")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--output_dir", default="./analysis_partial")
    ap.add_argument("--warmup-frac", type=float, default=0.5,
                    help="Fraction of leading rounds to skip as warmup (matches step21).")
    ap.add_argument("--use-signed", action="store_true",
                    help="Keep negative valuations instead of clipping at 0.")
    args = ap.parse_args()

    rows = []
    for scenario_dir in sorted(Path(args.results_dir).glob("stepE4_valuation_*")):
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue
        for seed_dir in sorted(scenario_dir.glob("*/run_*_seed_*")):
            val_path = seed_dir / "valuations.jsonl"
            if not val_path.exists():
                continue
            seed = int(seed_dir.name.split("_seed_")[-1])
            for method, gap in gap_for_run(val_path, args.warmup_frac, args.use_signed).items():
                rows.append({
                    'defense': m.group('defense'),
                    'attack': m.group('attack'),
                    'method': method,
                    'seed': seed,
                    **gap,
                })

    if not rows:
        print("No Step E4 results found — run the stepE4 configs first.")
        return

    df = pd.DataFrame(rows).sort_values(['defense', 'attack', 'method', 'seed'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "stepE4_valuation_gap.csv", index=False)

    agg = (df.groupby(['defense', 'attack', 'method'])
             .agg(['mean', 'std'])
             .drop(columns=['seed']))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "stepE4_valuation_gap_agg.csv", index=False)

    print(f"Wrote {len(df)} per-seed rows -> {out_dir / 'stepE4_valuation_gap.csv'}")
    print(f"Wrote {len(agg)} cell rows    -> {out_dir / 'stepE4_valuation_gap_agg.csv'}")
    cols = [c for c in ['defense', 'attack', 'method', 'benign_paid_pct_mean',
                        'adv_paid_pct_mean', 'adv_blocked_pct_mean'] if c in agg.columns]
    print("\nValuation-gap comparison (mean over seeds):")
    print(agg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
