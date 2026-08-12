"""
find_missing_agg_stats.py — List step 8 cells that lack agg_stats/round_*.json
and therefore need a rerun (backfill cannot help them).

Usage:
    python find_missing_agg_stats.py                          # print summary
    python find_missing_agg_stats.py --prefix step8_scalability
    python find_missing_agg_stats.py --emit-yaml-list         # print YAML config paths
"""

import argparse
from pathlib import Path

RESULTS_DIR = Path("./results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="step8_scalability")
    ap.add_argument("--emit-yaml-list", action="store_true",
                    help="Print only the YAML config paths (one per line) for rerun")
    args = ap.parse_args()

    scenarios = sorted(d for d in RESULTS_DIR.iterdir()
                       if d.is_dir() and d.name.startswith(args.prefix))

    by_scenario = {}
    needs_rerun = []  # (scenario, cell_dir)

    for scen in scenarios:
        for fm in scen.rglob("final_metrics.json"):
            cell = fm.parent
            agg = cell / "agg_stats"
            has_rounds = agg.is_dir() and any(agg.glob("round_*.json"))
            key = scen.name
            by_scenario.setdefault(key, {"total": 0, "missing": 0})
            by_scenario[key]["total"] += 1
            if not has_rounds:
                by_scenario[key]["missing"] += 1
                needs_rerun.append((scen, cell))

    if args.emit_yaml_list:
        # Try to find the corresponding YAML config for each missing cell.
        # Step 8 layout: results/<scenario>/n_sellers_<N>/.../run_<R>_seed_<S>
        # Configs:       experiments/gradient_market/configs/<scenario>/...
        for scen, cell in needs_rerun:
            rel = cell.relative_to(RESULTS_DIR)
            # Drop the run_*_seed_* leaf — the YAML is at the cell's parent
            # if your generator emits one yaml per (scenario, n_sellers, ...).
            print(rel)
        return

    print(f"Scanning {len(scenarios)} scenario(s) under {RESULTS_DIR}/")
    print()
    print(f"{'Scenario':<60} {'cells':>6} {'missing_agg':>12}")
    print("-" * 82)
    for k in sorted(by_scenario):
        v = by_scenario[k]
        flag = " <-- needs rerun" if v["missing"] else ""
        print(f"{k:<60} {v['total']:>6} {v['missing']:>12}{flag}")

    print()
    print(f"Total cells needing rerun: {len(needs_rerun)}")
    if needs_rerun:
        print()
        print("First 20 cells needing rerun:")
        for scen, cell in needs_rerun[:20]:
            print(f"  {cell.relative_to(RESULTS_DIR)}")
        print()
        print("Re-run with: python find_missing_agg_stats.py --emit-yaml-list")
        print("to get the full list of cell paths.")


if __name__ == "__main__":
    main()
