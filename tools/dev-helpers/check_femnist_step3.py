"""
check_femnist_step3.py — Inspect partial FEMNIST defense-tuning results.

Walks results/step3_tune_*FEMNIST* and prints, for each defense, the best
(acc, asr) seen so far across all HP combinations and seeds — even if the
sweep is still in progress. Use this to peek at FEMNIST results before step
3 fully finishes.

Usage:
    python check_femnist_step3.py                    # summary table
    python check_femnist_step3.py --all              # show every HP combo
    python check_femnist_step3.py --defense flame    # filter to one defense
    python check_femnist_step3.py --raw              # one row per seed run
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("./results")

# Matches: step3_tune_<defense>_<attack>_<modality>_FEMNIST_<model_config>
SCENARIO_RE = re.compile(
    r"^step3_tune_(?P<defense>.+?)_(?P<attack>backdoor|labelflip)_"
    r"(?P<modality>image|tabular|text)_FEMNIST_(?P<model_config>.+)$"
)


def collect_runs():
    """Yield one record per (defense, hp_folder, seed_run)."""
    for scenario_dir in sorted(RESULTS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        m = SCENARIO_RE.match(scenario_dir.name)
        if not m:
            continue

        defense = m.group("defense")
        attack = m.group("attack")

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

            rel = metrics_file.relative_to(scenario_dir)
            hp_folder = rel.parts[0] if rel.parts else "default"
            # Try to find the seed from the path (last "run_*_seed_*" part)
            seed = "?"
            for part in rel.parts:
                sm = re.match(r"run_\d+_seed_(\d+)", part)
                if sm:
                    seed = sm.group(1)
                    break

            # Check if this run has a .success marker (otherwise it's partial)
            run_dir = metrics_file.parent
            done = (run_dir / ".success").exists()

            yield {
                "defense": defense,
                "attack": attack,
                "hp_folder": hp_folder,
                "seed": seed,
                "acc": acc,
                "asr": asr,
                "score": acc - asr,  # for backdoor: high acc, low asr
                "done": done,
                "path": str(run_dir.relative_to(RESULTS_DIR)),
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="Show every HP combo instead of just the best per defense")
    ap.add_argument("--defense", default=None,
                    help="Filter to a specific defense (e.g. 'flame')")
    ap.add_argument("--raw", action="store_true",
                    help="One row per seed run (not averaged)")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR.resolve()}")
        return

    records = list(collect_runs())
    if args.defense:
        records = [r for r in records if r["defense"] == args.defense]

    if not records:
        print("No FEMNIST step 3 results found yet.")
        return

    print(f"\nFound {len(records)} FEMNIST step 3 run(s) "
          f"({sum(r['done'] for r in records)} marked .success).\n")

    # ---- Mode 1: --raw — one row per seed run ----
    if args.raw:
        print(f"{'defense':<14}{'hp_folder':<60}{'seed':<6}"
              f"{'acc':>8}{'asr':>8}{'score':>8}  status")
        print("-" * 110)
        for r in sorted(records, key=lambda x: (x["defense"], x["hp_folder"], x["seed"])):
            status = "✓" if r["done"] else "partial"
            hp_short = r["hp_folder"][:58]
            print(f"{r['defense']:<14}{hp_short:<60}{r['seed']:<6}"
                  f"{r['acc']:>8.4f}{r['asr']:>8.4f}{r['score']:>+8.4f}  {status}")
        return

    # ---- Mode 2: average per (defense, hp_folder) across seeds ----
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["defense"], r["hp_folder"])].append(r)

    avg_rows = []
    for (defense, hp_folder), runs in grouped.items():
        avg_acc = sum(r["acc"] for r in runs) / len(runs)
        avg_asr = sum(r["asr"] for r in runs) / len(runs)
        avg_rows.append({
            "defense": defense,
            "hp_folder": hp_folder,
            "n_seeds": len(runs),
            "n_done": sum(r["done"] for r in runs),
            "acc": avg_acc,
            "asr": avg_asr,
            "score": avg_acc - avg_asr,
        })

    # ---- Mode 2a: --all — show every HP combo ----
    if args.all:
        print(f"{'defense':<14}{'hp_folder':<60}{'seeds':>7}"
              f"{'acc':>8}{'asr':>8}{'score':>8}")
        print("-" * 105)
        for row in sorted(avg_rows, key=lambda x: (x["defense"], -x["score"])):
            seeds_str = f"{row['n_done']}/{row['n_seeds']}"
            hp_short = row["hp_folder"][:58]
            print(f"{row['defense']:<14}{hp_short:<60}{seeds_str:>7}"
                  f"{row['acc']:>8.4f}{row['asr']:>8.4f}{row['score']:>+8.4f}")
        return

    # ---- Mode 2b: default — best HP combo per defense ----
    best_by_defense = {}
    for row in avg_rows:
        d = row["defense"]
        if d not in best_by_defense or row["score"] > best_by_defense[d]["score"]:
            best_by_defense[d] = row

    print("Best HP combo per defense (score = acc − asr; higher is better):\n")
    print(f"{'defense':<14}{'seeds':>7}{'acc':>8}{'asr':>8}{'score':>8}  best HP folder")
    print("-" * 100)
    for defense in sorted(best_by_defense):
        row = best_by_defense[defense]
        seeds_str = f"{row['n_done']}/{row['n_seeds']}"
        verdict = ""
        if row["asr"] >= 0.7:
            verdict = "  ⚠ defense fails (high ASR)"
        elif row["acc"] < 0.3:
            verdict = "  ⚠ model collapsed"
        elif row["asr"] < 0.2 and row["acc"] >= 0.4:
            verdict = "  ✓ defense works"
        print(f"{row['defense']:<14}{seeds_str:>7}"
              f"{row['acc']:>8.4f}{row['asr']:>8.4f}{row['score']:>+8.4f}  "
              f"{row['hp_folder'][:40]}{verdict}")

    # Coverage summary
    print()
    all_defenses = sorted(set(r["defense"] for r in records))
    print(f"Defenses with FEMNIST results so far: {len(all_defenses)}")
    print(f"  {', '.join(all_defenses)}")
    expected = ["bulyan", "deepsight", "flame", "fltrust", "foolsgold", "martfl",
                "multi_krum", "rflpa", "skymask", "spmc", "trimmed_mean"]
    missing = [d for d in expected if d not in all_defenses]
    if missing:
        print(f"Defenses NOT yet in FEMNIST step 3: {', '.join(missing)}")


if __name__ == "__main__":
    main()
