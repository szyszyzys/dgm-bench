# =============================================================================
# Step E3 extractor — Privacy-Tax table: BSR (primary), MSR, ASR, Acc per
# (aggregator, dp_epsilon).
#
# Walks results/stepE3_dp_{defense}_eps_{tag}_CIFAR100/.../run_*_seed_*/ and
# emits per-seed + aggregated CSVs. Expected: BSR decreases as epsilon
# decreases (more noise -> more honest-seller rejection).
#
#   python extract_stepE3_dp_sweep.py [--results_dir ./results]
# =============================================================================

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCENARIO_RE = re.compile(r"stepE3_dp_(?P<defense>.+)_eps_(?P<eps>[^_]+)_(?P<dataset>.+)$")


def per_run_metrics(run_dir: Path):
    out = {}
    log_path = run_dir / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        if 'false_positive_rate' in log:
            out['bsr'] = 1.0 - log['false_positive_rate'].dropna().mean()
        if 'adversary_detection_rate' in log:
            out['msr'] = 1.0 - log['adversary_detection_rate'].dropna().mean()
        out['rounds'] = len(log)
    fm_path = run_dir / "final_metrics.json"
    if fm_path.exists():
        with open(fm_path) as f:
            fm = json.load(f)
        out['acc'] = fm.get('acc', fm.get('B-Acc'))
        out['asr'] = fm.get('asr')
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract Step E3 DP-sweep results")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--output_dir", default="./analysis_partial")
    args = ap.parse_args()

    rows = []
    for scenario_dir in sorted(Path(args.results_dir).glob("stepE3_dp_*")):
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue
        defense = m.group('defense')
        eps_tag = m.group('eps')
        epsilon = None if eps_tag == "none" else float(eps_tag)
        for seed_dir in sorted(scenario_dir.glob("*/run_*_seed_*")):
            seed = int(seed_dir.name.split("_seed_")[-1])
            metrics = per_run_metrics(seed_dir)
            if not metrics:
                continue
            rows.append({
                'aggregator': defense,
                'dp_epsilon': 'none' if epsilon is None else epsilon,
                'seed': seed,
                **metrics,
            })

    if not rows:
        print("No Step E3 results found — run the stepE3 configs first.")
        return

    df = pd.DataFrame(rows).sort_values(['aggregator', 'dp_epsilon', 'seed'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "stepE3_dp_sweep.csv", index=False)

    agg = (df.groupby(['aggregator', 'dp_epsilon'])
             .agg(['mean', 'std'])
             .drop(columns=['seed']))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "stepE3_dp_sweep_agg.csv", index=False)

    print(f"Wrote {len(df)} per-seed rows -> {out_dir / 'stepE3_dp_sweep.csv'}")
    print(f"Wrote {len(agg)} cell rows    -> {out_dir / 'stepE3_dp_sweep_agg.csv'}")
    cols = [c for c in ['aggregator', 'dp_epsilon', 'bsr_mean', 'msr_mean',
                        'asr_mean', 'acc_mean'] if c in agg.columns]
    print("\nPrivacy-Tax table (BSR is the primary metric):")
    print(agg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
