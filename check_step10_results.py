"""
check_step10_results.py — Inspect partial step 10 main-summary results.

Walks results/step12_main_summary_* and prints (clean acc, ASR) for every
completed cell, grouped by dataset and defense. The key sanity signal is
the FedAvg (no-defense) row per dataset: if FedAvg ASR is high, the
backdoor is working; then you can compare other defenses against it.

Usage:
    python check_step10_results.py                    # full summary table
    python check_step10_results.py --dataset FEMNIST  # one dataset only
    python check_step10_results.py --defense fltrust  # one defense only
    python check_step10_results.py --raw              # one row per seed run
    python check_step10_results.py --verify           # just check "is backdoor working"
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("./results")

# step12_main_summary_<defense>_<modality>_<dataset>_<model_tag>
# e.g. step12_main_summary_fltrust_image_CIFAR100_cnn
#      step12_main_summary_deepsight_tabular_Purchase100_baseline
SCENARIO_RE = re.compile(
    r"^step12_main_summary_(?P<defense>.+?)_"
    r"(?P<modality>image|tabular|text)_"
    r"(?P<dataset>CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)_"
    r"(?P<model_tag>.+)$"
)


def collect_runs():
    """Yield one record per completed seed run (final_metrics.json present)."""
    for scenario_dir in sorted(RESULTS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue

        defense = m.group("defense")
        dataset = m.group("dataset")
        modality = m.group("modality")

        for metrics_file in scenario_dir.rglob("final_metrics.json"):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            acc = metrics.get("acc")
            if acc is None:
                continue
            asr = metrics.get("asr", 0.0) or 0.0

            run_dir = metrics_file.parent
            done = (run_dir / ".success").exists()
            # Extract seed from path
            seed = "?"
            for part in metrics_file.relative_to(scenario_dir).parts:
                sm = re.match(r"run_\d+_seed_(\d+)", part)
                if sm:
                    seed = sm.group(1)
                    break

            yield {
                "defense": defense,
                "dataset": dataset,
                "modality": modality,
                "acc": acc,
                "asr": asr,
                "score": acc - asr,
                "seed": seed,
                "done": done,
                "path": str(run_dir.relative_to(RESULTS_DIR)),
            }


# Random-baseline ASR = 1/num_classes. If ASR ≤ 3x this, attack is effectively
# not working. Used only for the verdict printout.
RANDOM_ASR = {
    "CIFAR100":    1 / 100,
    "CIFAR10":     1 / 10,
    "FEMNIST":     1 / 62,
    "Texas100":    1 / 100,
    "Purchase100": 1 / 100,
    "TREC":        1 / 6,
}


def verdict_for(defense: str, acc: float, asr: float, dataset: str) -> str:
    """One-word verdict for a (defense, dataset) combo."""
    rand = RANDOM_ASR.get(dataset, 0.01)
    if defense == "fedavg":
        if asr >= 0.70:
            return "✅ attack works"
        elif asr >= 0.30:
            return "⚠ attack weak"
        elif asr <= 3 * rand:
            return "❌ attack broken"
        else:
            return "⚠ attack partial"
    else:
        if acc < 0.30:
            return "⚠ model collapsed"
        if asr >= 0.70:
            return "❌ defense fails"
        if asr <= 3 * rand:
            return "✅ defense works"
        return "⚠ partial defense"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default=None, help="Filter to one dataset")
    ap.add_argument("--defense", default=None, help="Filter to one defense")
    ap.add_argument("--raw", action="store_true", help="One row per seed run")
    ap.add_argument("--verify", action="store_true",
                    help="Only print whether the backdoor is working per dataset (fedavg rows)")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR.resolve()}")
        return

    records = list(collect_runs())
    if args.dataset:
        records = [r for r in records if r["dataset"] == args.dataset]
    if args.defense:
        records = [r for r in records if r["defense"] == args.defense]

    if not records:
        print("No step 10 results found yet (or no cells match your filters).")
        return

    n_done = sum(r["done"] for r in records)
    print(f"\nFound {len(records)} step 10 run(s) ({n_done} marked .success).\n")

    # --------------------------------------------------------------------
    # Mode: --verify — just tell me if the attack is working per dataset
    # --------------------------------------------------------------------
    if args.verify:
        fedavg = [r for r in records if r["defense"] == "fedavg"]
        if not fedavg:
            print("No FedAvg (no-defense) results yet — can't verify attack strength.")
            print("Wait for at least one fedavg cell per dataset to complete.")
            return

        by_dataset = defaultdict(list)
        for r in fedavg:
            by_dataset[r["dataset"]].append(r)

        print("Backdoor attack sanity check (FedAvg = no defense):")
        print(f"{'dataset':<14}{'seeds':>6}{'acc':>8}{'asr':>8}  verdict")
        print("-" * 70)
        for ds in sorted(by_dataset):
            runs = by_dataset[ds]
            avg_acc = sum(r["acc"] for r in runs) / len(runs)
            avg_asr = sum(r["asr"] for r in runs) / len(runs)
            v = verdict_for("fedavg", avg_acc, avg_asr, ds)
            print(f"{ds:<14}{len(runs):>6}{avg_acc:>8.4f}{avg_asr:>8.4f}  {v}")
        return

    # --------------------------------------------------------------------
    # Mode: --raw — one row per seed run
    # --------------------------------------------------------------------
    if args.raw:
        print(f"{'dataset':<14}{'defense':<14}{'seed':<6}{'acc':>8}{'asr':>8}{'score':>8}  status")
        print("-" * 80)
        for r in sorted(records, key=lambda x: (x["dataset"], x["defense"], x["seed"])):
            status = "✓" if r["done"] else "partial"
            print(f"{r['dataset']:<14}{r['defense']:<14}{r['seed']:<6}"
                  f"{r['acc']:>8.4f}{r['asr']:>8.4f}{r['score']:>+8.4f}  {status}")
        return

    # --------------------------------------------------------------------
    # Default mode: averaged per (dataset, defense), grouped by dataset
    # --------------------------------------------------------------------
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["dataset"], r["defense"])].append(r)

    rows = []
    for (dataset, defense), runs in grouped.items():
        rows.append({
            "dataset": dataset,
            "defense": defense,
            "n_seeds": len(runs),
            "n_done": sum(r["done"] for r in runs),
            "acc": sum(r["acc"] for r in runs) / len(runs),
            "asr": sum(r["asr"] for r in runs) / len(runs),
        })

    # Group output by dataset; within each dataset, fedavg first, then alphabetical
    def sort_key(row):
        defense_order = 0 if row["defense"] == "fedavg" else 1
        return (row["dataset"], defense_order, row["defense"])

    rows.sort(key=sort_key)
    current_dataset = None
    for row in rows:
        if row["dataset"] != current_dataset:
            current_dataset = row["dataset"]
            print()
            print(f"=== {current_dataset} ===")
            print(f"  {'defense':<14}{'seeds':>7}{'acc':>8}{'asr':>8}{'score':>8}  verdict")
            print(f"  {'-' * 75}")
        v = verdict_for(row["defense"], row["acc"], row["asr"], row["dataset"])
        seeds_str = f"{row['n_done']}/{row['n_seeds']}"
        print(f"  {row['defense']:<14}{seeds_str:>7}"
              f"{row['acc']:>8.4f}{row['asr']:>8.4f}"
              f"{row['acc'] - row['asr']:>+8.4f}  {v}")

    # Coverage footer
    print()
    datasets_seen = sorted(set(r["dataset"] for r in records))
    defenses_seen = sorted(set(r["defense"] for r in records))
    print(f"Coverage: {len(datasets_seen)} dataset(s) × {len(defenses_seen)} defense(s)")
    print(f"  datasets: {', '.join(datasets_seen)}")
    print(f"  defenses: {', '.join(defenses_seen)}")

    # Quick headline: is the attack working?
    print()
    fedavg_rows = [r for r in rows if r["defense"] == "fedavg"]
    if fedavg_rows:
        good_datasets = [r["dataset"] for r in fedavg_rows if r["asr"] >= 0.70]
        broken_datasets = [r["dataset"] for r in fedavg_rows
                           if r["asr"] <= 3 * RANDOM_ASR.get(r["dataset"], 0.01)]
        if good_datasets:
            print(f"✅ Attack working (FedAvg ASR ≥ 0.70): {', '.join(good_datasets)}")
        if broken_datasets:
            print(f"❌ Attack broken on:                    {', '.join(broken_datasets)}")
        if not good_datasets and not broken_datasets:
            print("⚠  Attack in between — see the per-dataset fedavg row for details.")
    else:
        print("⚠  No FedAvg cells completed yet — can't confirm attack strength.")
        print("   Run `check_step10_results.py --verify` once any FedAvg cell finishes.")


if __name__ == "__main__":
    main()
