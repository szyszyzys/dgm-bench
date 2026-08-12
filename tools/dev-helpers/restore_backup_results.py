"""
restore_backup_results.py — Restore valid cells from a _backup_* results dir
to the live results/ tree, backfill aggregation metrics, and mark .success.

Problem: you had results in e.g.
    results/_backup_step10_scalability_martfl_CIFAR100/...
that you want to reuse instead of re-running. The live tree expects the
same scenario named WITHOUT the _backup_ prefix. This script:

  1. Walks every run dir under the backup tree.
  2. Validates that each run has the artifacts we need:
       - final_metrics.json with at least 'acc' present
       - agg_stats/round_*.json (for backfill)
  3. For each valid run, mirrors it to the corresponding live path
     (translating _backup_<name> -> <name> on the scenario dir).
  4. Backfills aggregation latency/VRAM into final_metrics.json from
     the per-round agg_stats files (same logic as backfill_aggregation_metrics.py).
  5. Writes a .success marker so the dispatcher's skip logic recognizes
     the cell as complete and won't re-run it.

Runs that fail validation are NOT moved — they're logged and left in the
backup so you can inspect them. Nothing is deleted from the backup.

Usage:
    python restore_backup_results.py                                # dry-run preview first
    python restore_backup_results.py --apply                        # actually move + backfill
    python restore_backup_results.py --backup-prefix _backup_step10_scalability --apply
    python restore_backup_results.py --copy                         # copy instead of move
    python restore_backup_results.py --apply --force-success        # overwrite existing .success

Safety:
  - Default is DRY-RUN (prints what would happen without changing disk).
  - Atomic writes for final_metrics.json.
  - Won't overwrite an existing live cell that already has .success unless
    --force-success is passed. This protects completed cells from being
    clobbered by older backup data.
"""

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean, median


RESULTS_DIR = Path("./results")

# Keys we aggregate from agg_stats/round_*.json into final_metrics.json
AGG_KEYS = [
    "avg_aggregation_latency_sec",
    "max_aggregation_latency_sec",
    "median_aggregation_latency_sec",
    "max_aggregation_peak_vram_mb",
]


# ----------------------------------------------------------------------------
# Validation: what counts as a "valid" backup run worth restoring?
# ----------------------------------------------------------------------------
def validate_run(run_dir: Path) -> tuple[bool, str, dict]:
    """Return (is_valid, reason, stats) for a single run dir."""
    stats = {"has_final_metrics": False, "has_acc": False,
             "n_agg_rounds": 0, "has_latency": False,
             "has_success": False}

    fm = run_dir / "final_metrics.json"
    if not fm.exists():
        return False, "no final_metrics.json", stats
    stats["has_final_metrics"] = True

    try:
        with open(fm) as f:
            fm_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"unreadable final_metrics.json ({e})", stats

    # Accept either 'acc' (classification) or 'perplexity' (LLM) as a signal
    # that evaluation ran.
    if fm_data.get("acc") is None and fm_data.get("perplexity") is None:
        return False, "final_metrics.json has no acc/perplexity — eval did not complete", stats
    stats["has_acc"] = fm_data.get("acc") is not None

    # agg_stats/ is optional — a run without it is still valid, just won't
    # backfill. We distinguish these cases for the summary.
    agg_dir = run_dir / "agg_stats"
    if agg_dir.is_dir():
        round_files = list(agg_dir.glob("round_*.json"))
        stats["n_agg_rounds"] = len(round_files)
        if round_files:
            try:
                with open(round_files[0]) as f:
                    first = json.load(f)
                stats["has_latency"] = "aggregation_latency_sec" in first
            except Exception:
                pass

    stats["has_success"] = (run_dir / ".success").exists()

    return True, "ok", stats


# ----------------------------------------------------------------------------
# Backfill: same logic as backfill_aggregation_metrics.py, applied in-place
# to the (now restored) run dir.
# ----------------------------------------------------------------------------
def backfill_final_metrics(run_dir: Path, force: bool = False) -> dict:
    """Read agg_stats/round_*.json, aggregate, patch final_metrics.json.
    Returns a dict describing what was changed.
    """
    fm = run_dir / "final_metrics.json"
    agg_dir = run_dir / "agg_stats"
    result = {"patched_keys": [], "n_rounds": 0, "skipped_reason": None}

    if not agg_dir.is_dir():
        result["skipped_reason"] = "no agg_stats/"
        return result
    if not fm.exists():
        result["skipped_reason"] = "no final_metrics.json"
        return result

    try:
        with open(fm) as f:
            fm_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["skipped_reason"] = f"unreadable ({e})"
        return result

    latencies, vrams = [], []
    for rjson in sorted(agg_dir.glob("round_*.json")):
        try:
            with open(rjson) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        result["n_rounds"] += 1
        if (v := d.get("aggregation_latency_sec")) is not None:
            try:
                latencies.append(float(v))
            except (TypeError, ValueError):
                pass
        if (v := d.get("aggregation_peak_vram_mb")) is not None:
            try:
                vrams.append(float(v))
            except (TypeError, ValueError):
                pass

    patch = {}
    if latencies:
        patch["avg_aggregation_latency_sec"] = float(mean(latencies))
        patch["max_aggregation_latency_sec"] = float(max(latencies))
        patch["median_aggregation_latency_sec"] = float(median(latencies))
    if vrams:
        patch["max_aggregation_peak_vram_mb"] = float(max(vrams))

    if not patch:
        result["skipped_reason"] = f"no latency/vram in {result['n_rounds']} agg_stats files"
        return result

    for k, v in patch.items():
        if force or k not in fm_data:
            fm_data[k] = v
            result["patched_keys"].append(k)

    if result["patched_keys"]:
        tmp = fm.with_suffix(fm.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(fm_data, f, indent=2)
        tmp.replace(fm)

    return result


# ----------------------------------------------------------------------------
# Path translation: backup → live
# ----------------------------------------------------------------------------
def backup_to_live_path(backup_run: Path, backup_prefix: str) -> Path:
    """Translate a backup run dir path to its live counterpart.

    Example:
      backup_run = results/_backup_step10_scalability_martfl_CIFAR100/n_sellers_10/.../run_0_seed_42
      returns      results/step10_scalability_martfl_CIFAR100/n_sellers_10/.../run_0_seed_42
    """
    parts = list(backup_run.parts)
    # Find the component that starts with the backup prefix (e.g. '_backup_...')
    # and strip the '_backup_' part.
    for i, p in enumerate(parts):
        if p.startswith("_backup_"):
            parts[i] = p[len("_backup_"):]
            break
    return Path(*parts)


def find_run_dirs(scenario_dir: Path):
    """Yield every dir that looks like a leaf run (contains final_metrics.json
    OR has no subdirs that could be deeper runs). We use final_metrics.json
    presence as the discriminator."""
    for fm in scenario_dir.rglob("final_metrics.json"):
        yield fm.parent


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup-prefix", default="_backup_",
                    help="Process backup dirs whose name starts with this "
                         "prefix (default: '_backup_')")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move files. Default is dry-run preview.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of move (leaves backup untouched)")
    ap.add_argument("--force-success", action="store_true",
                    help="Overwrite existing .success markers in live tree "
                         "(protects you from clobbering newer results by default)")
    ap.add_argument("--only-scenario", default=None,
                    help="Restrict to a single scenario (substring match "
                         "against the backup dir name)")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR.resolve()}")
        return

    # Find backup scenario dirs
    backup_scenarios = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith(args.backup_prefix)
    ]
    if args.only_scenario:
        backup_scenarios = [d for d in backup_scenarios if args.only_scenario in d.name]

    if not backup_scenarios:
        print(f"No backup scenarios found matching prefix '{args.backup_prefix}'"
              f"{' and --only-scenario=' + args.only_scenario if args.only_scenario else ''}.")
        return

    print(f"{'DRY RUN — no files will be changed' if not args.apply else 'APPLYING CHANGES'}")
    print(f"Backup prefix    : {args.backup_prefix}")
    print(f"Mode             : {'copy' if args.copy else 'move'}")
    print(f"Force .success   : {args.force_success}")
    print(f"Scenario dirs    : {len(backup_scenarios)}")
    print()

    total_runs = 0
    valid_runs = 0
    restored = 0
    backfilled = 0
    skipped_live_exists = 0
    skipped_invalid = 0
    skipped_other = 0

    for scenario in sorted(backup_scenarios):
        scen_rel = scenario.relative_to(RESULTS_DIR)
        print(f"=== {scen_rel} ===")
        runs_in_scenario = 0
        for run_dir in find_run_dirs(scenario):
            total_runs += 1
            runs_in_scenario += 1
            rel = run_dir.relative_to(scenario)

            is_valid, reason, stats = validate_run(run_dir)
            if not is_valid:
                print(f"  [invalid] {rel}  ({reason})")
                skipped_invalid += 1
                continue
            valid_runs += 1

            live_run = backup_to_live_path(run_dir, args.backup_prefix)
            live_fm = live_run / "final_metrics.json"
            live_success = live_run / ".success"

            # Skip if live cell already complete (protect newer results)
            if live_success.exists() and not args.force_success:
                print(f"  [skip   ] {rel}  (live cell already has .success)")
                skipped_live_exists += 1
                continue

            # Preview what we'd do
            summary = (f"rounds={stats['n_agg_rounds']} "
                       f"latency={'yes' if stats['has_latency'] else 'no'} "
                       f"had_success={'yes' if stats['has_success'] else 'no'}")
            action_label = "would " if not args.apply else ""

            if not args.apply:
                print(f"  [preview] {rel}  ({summary})")
                restored += 1
                if stats['has_latency']:
                    backfilled += 1
                continue

            # Execute: ensure target parent exists, move/copy the run dir
            live_run.parent.mkdir(parents=True, exist_ok=True)

            if live_run.exists():
                # Partial / stale live dir present. Remove it to avoid merge
                # ambiguity. Its .success check already protected us above.
                shutil.rmtree(live_run)

            try:
                if args.copy:
                    shutil.copytree(run_dir, live_run)
                else:
                    shutil.move(str(run_dir), str(live_run))
            except (OSError, shutil.Error) as e:
                print(f"  [ERROR  ] {rel}  ({e})")
                skipped_other += 1
                continue

            # Backfill aggregation metrics from the now-restored dir
            bf = backfill_final_metrics(live_run, force=args.force_success)
            if bf["patched_keys"]:
                backfilled += 1

            # Ensure .success marker exists
            if not live_success.exists():
                live_success.touch()
            restored += 1

            print(f"  [restored] {rel}  "
                  f"backfilled={len(bf['patched_keys'])} from {bf['n_rounds']} rounds")

        if runs_in_scenario == 0:
            print(f"  (no run dirs found under {scen_rel})")

    print()
    print("=" * 70)
    print(f"Total runs scanned        : {total_runs}")
    print(f"Valid                     : {valid_runs}")
    print(f"Restored{' (preview)' if not args.apply else ''}               : {restored}")
    print(f"Backfilled{' (preview)' if not args.apply else ''}             : {backfilled}")
    print(f"Skipped (invalid)         : {skipped_invalid}")
    print(f"Skipped (live .success)   : {skipped_live_exists}")
    print(f"Skipped (other errors)    : {skipped_other}")
    if not args.apply:
        print()
        print("This was a dry run. Re-run with --apply to actually move files.")


if __name__ == "__main__":
    main()
