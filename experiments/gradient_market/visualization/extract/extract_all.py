"""
extract_all.py — Run every per-step extractor and print a unified summary.

Walks each per-step extractor in canonical pipeline order, calls its main(),
collects its readiness report, and at the end prints a one-line-per-step
table plus a grand total.

Usage:
    python extract_all.py
    python extract_all.py --strict
    python extract_all.py --strict --json
    python extract_all.py --steps 3 4 10
"""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# (cli_step, module_name, label) — order is the rerun_priority.sh execution
# order so reading the table top-to-bottom mirrors what the pipeline does.
STEP_MODULES = [
    (3,  "extract_step3_defense_tune",  "defense_tune"),
    (10, "extract_step10_main_summary", "main_summary"),
    (4,  "extract_step4_attack_sens",   "attack_sens"),
    (5,  "extract_step5_sybil",         "sybil"),
    (6,  "extract_step6_adaptive",      "adaptive"),
    (7,  "extract_step7_buyer",         "buyer"),
    (9,  "extract_step9_heterogeneity", "heterogeneity"),
    (14, "extract_step14_collusion",    "collusion"),
    (8,  "extract_step8_scalability",   "scalability"),
    (13, "extract_step13_drowning",     "drowning"),
    (15, "extract_step15_pricing",      "pricing"),
    (17, "extract_step17_alie",         "alie"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--output_dir", default="./analysis_partial")
    ap.add_argument("--strict", action="store_true",
                    help="Require expected_seeds per cell, not just ≥1")
    ap.add_argument("--json", action="store_true",
                    help="Also write per-step summary.json files")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="CLI step numbers to run (default: all)")
    args = ap.parse_args()

    selected = set(args.steps) if args.steps else None
    runs = [(s, m, l) for s, m, l in STEP_MODULES if selected is None or s in selected]

    if not runs:
        print(f"No matching steps. Known: {[s for s, _, _ in STEP_MODULES]}")
        return

    print(f"📂 Results dir : {Path(args.results_dir).resolve()}")
    print(f"📁 Output dir  : {Path(args.output_dir).resolve()}")
    print(f"🎯 Mode        : {'STRICT (all seeds)' if args.strict else 'LOOSE (≥1 seed)'}")
    print(f"🔢 Steps       : {[s for s, _, _ in runs]}")

    # Collect per-step reports for the final summary
    summaries = []

    # Build the argv each per-step main() will see
    forwarded_argv = ["--results_dir", args.results_dir,
                       "--output_dir", args.output_dir]
    if args.strict:
        forwarded_argv.append("--strict")
    if args.json:
        forwarded_argv.append("--json")

    saved_argv = sys.argv
    try:
        for step_id, module_name, label in runs:
            full_module = f"experiments.gradient_market.visualization.extract.{module_name}"
            try:
                mod = importlib.import_module(full_module)
            except ImportError as e:
                print(f"\n❌ Could not import {full_module}: {e}")
                continue

            # Stub argv so each main()'s argparse sees the forwarded flags only
            sys.argv = [module_name + ".py"] + forwarded_argv
            try:
                mod.main()
            except SystemExit:
                pass
            except Exception as e:
                print(f"\n❌ Step {step_id} ({label}) extractor crashed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # We don't have a return-value channel from main() to here, so the
            # per-step card is the user-visible output. The unified summary
            # below is built fresh from the JSON files (if --json was passed).
            summaries.append((step_id, label))
    finally:
        sys.argv = saved_argv

    # Final compact summary table from JSON files (only if --json was set)
    if args.json:
        import json
        print()
        print("=" * 72)
        print(" UNIFIED SUMMARY")
        print("=" * 72)
        header = f"  {'Step':>4}  {'Label':<16}  {'Status':<6}  {'Cells':>10}  {'Seeded':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        grand_expected = 0
        grand_present = 0
        grand_complete = 0
        for step_id, label in summaries:
            json_path = Path(args.output_dir) / f"step{step_id}_{label}.json"
            if not json_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                continue
            n_exp = data.get("n_expected", 0)
            n_pre = data.get("n_present", 0)
            n_cmp = data.get("n_complete_seeds", 0)
            icon = data.get("status_icon", "?")
            grand_expected += n_exp
            grand_present += n_pre
            grand_complete += n_cmp
            print(f"  {step_id:>4}  {label:<16}  {icon:<6}  "
                  f"{n_pre:>4}/{n_exp:<5}  {n_cmp:>4}/{n_exp:<5}")

        print("  " + "-" * (len(header) - 2))
        print(f"  {'TOT':>4}  {'(grand total)':<16}  {'':<6}  "
              f"{grand_present:>4}/{grand_expected:<5}  "
              f"{grand_complete:>4}/{grand_expected:<5}")
        print()


if __name__ == "__main__":
    main()
