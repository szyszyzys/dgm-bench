# =============================================================================
# Step E1 extractor — (tau, MSR, BSR, ASR, Acc) frontier table.
#
# Walks results/stepE1_tau_sweep_CIFAR100/<run_name>/run_*_seed_*/ and emits
#   analysis_partial/stepE1_tau_frontier.csv          (per-seed rows)
#   analysis_partial/stepE1_tau_frontier_agg.csv      (mean/std per tau)
# so the MSR-vs-Acc frontier can be plotted and the joint objective
# (low MSR AND high Acc) checked for ANY tau.
#
#   python extract_stepE1_tau_frontier.py [--results_dir ./results]
# =============================================================================

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def per_run_metrics(run_dir: Path):
    """Selection metrics averaged over rounds + final model metrics."""
    out = {}
    log_path = run_dir / "training_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        # BSR = 1 - false_positive_rate; MSR = 1 - adversary_detection_rate
        # (the unchanged per-round defense metrics).
        if 'false_positive_rate' in log:
            out['bsr'] = 1.0 - log['false_positive_rate'].dropna().mean()
        if 'adversary_detection_rate' in log:
            out['msr'] = 1.0 - log['adversary_detection_rate'].dropna().mean()
        if 'adversary_revenue_share' in log:
            out['adversary_revenue_share'] = log['adversary_revenue_share'].dropna().mean()
        out['rounds'] = len(log)
    fm_path = run_dir / "final_metrics.json"
    if fm_path.exists():
        with open(fm_path) as f:
            fm = json.load(f)
        out['acc'] = fm.get('acc', fm.get('B-Acc'))
        out['asr'] = fm.get('asr')
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract Step E1 tau-frontier results")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--output_dir", default="./analysis_partial")
    ap.add_argument("--scenario_glob", default="stepE1_tau_sweep_*")
    args = ap.parse_args()

    rows = []
    for scenario_dir in sorted(Path(args.results_dir).glob(args.scenario_glob)):
        for run_name_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            m = re.search(r"tau-([0-9]+(?:p[0-9]+)?)", run_name_dir.name)
            if not m:
                print(f"  WARN: no tau in run name '{run_name_dir.name}', skipping.")
                continue
            tau = float(m.group(1).replace('p', '.'))
            for seed_dir in sorted(run_name_dir.glob("run_*_seed_*")):
                seed = int(seed_dir.name.split("_seed_")[-1])
                metrics = per_run_metrics(seed_dir)
                if not metrics:
                    continue
                rows.append({'tau': tau, 'seed': seed, **metrics})

    if not rows:
        print("No Step E1 results found — run the stepE1 configs first.")
        return

    df = pd.DataFrame(rows).sort_values(['tau', 'seed'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "stepE1_tau_frontier.csv", index=False)

    agg = (df.groupby('tau')
             .agg(['mean', 'std'])
             .drop(columns=['seed']))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "stepE1_tau_frontier_agg.csv", index=False)

    print(f"Wrote {len(df)} per-seed rows -> {out_dir / 'stepE1_tau_frontier.csv'}")
    print(f"Wrote {len(agg)} tau rows     -> {out_dir / 'stepE1_tau_frontier_agg.csv'}")
    cols = [c for c in ['tau', 'msr_mean', 'bsr_mean', 'asr_mean', 'acc_mean',
                        'adversary_revenue_share_mean'] if c in agg.columns]
    print("\nMSR-vs-Acc frontier:")
    print(agg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
