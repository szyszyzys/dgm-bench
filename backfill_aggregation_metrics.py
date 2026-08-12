"""
backfill_aggregation_metrics.py — Backfill aggregation latency / VRAM metrics
into existing final_metrics.json files.

Context: prior to the run_exp.py:822 fix, save_marketplace_analysis_data_incremental
was commented out, so round_aggregates.csv was never written. That meant
final_metrics.json files produced by old experiments lack these fields:

    avg_aggregation_latency_sec
    max_aggregation_latency_sec
    median_aggregation_latency_sec
    max_aggregation_peak_vram_mb
    total_wasted_upload_mb

However, the per-round data IS persisted in <cell>/agg_stats/round_N.json
files. This script walks every cell, reads the per-round JSONs, computes
the aggregates, and patches them into final_metrics.json in place.

Safe to re-run: it only writes keys that are missing (unless --force is set).
Atomic writes via tmp+rename so a crash mid-run can't corrupt final_metrics.

Usage:
    python backfill_aggregation_metrics.py                          # all results
    python backfill_aggregation_metrics.py --prefix step10_scalability
    python backfill_aggregation_metrics.py --dry-run                # preview only
    python backfill_aggregation_metrics.py --force                  # overwrite existing
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median


RESULTS_DIR = Path("./results")

TARGET_KEYS = [
    "avg_aggregation_latency_sec",
    "max_aggregation_latency_sec",
    "median_aggregation_latency_sec",
    "max_aggregation_peak_vram_mb",
]


def atomic_write_json(obj, path: Path):
    """Write JSON atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def backfill_cell(final_metrics_path: Path, force: bool, dry_run: bool) -> str:
    """Backfill one cell. Returns a status string."""
    cell_dir = final_metrics_path.parent
    agg_dir = cell_dir / "agg_stats"

    if not agg_dir.is_dir():
        return "skip: no agg_stats/"

    # Read existing final_metrics
    try:
        with open(final_metrics_path) as f:
            fm = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return f"skip: unreadable final_metrics.json ({e})"

    # Skip if all keys already present with non-null values and --force is not set.
    # A key present with value=None counts as missing — earlier code paths wrote
    # null placeholders that we want to refill from agg_stats/round_*.json.
    already = [k for k in TARGET_KEYS if fm.get(k) is not None]
    if len(already) == len(TARGET_KEYS) and not force:
        return "skip: already has all keys (use --force to overwrite)"

    # Collect per-round latencies + vram
    latencies = []
    vrams = []
    n_rounds = 0
    for round_json in sorted(agg_dir.glob("round_*.json")):
        try:
            with open(round_json) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        n_rounds += 1
        lat = d.get("aggregation_latency_sec")
        if lat is not None:
            try:
                latencies.append(float(lat))
            except (TypeError, ValueError):
                pass
        vram = d.get("aggregation_peak_vram_mb")
        if vram is not None:
            try:
                vrams.append(float(vram))
            except (TypeError, ValueError):
                pass

    if n_rounds == 0:
        return "skip: no agg_stats/round_*.json files"
    if not latencies and not vrams:
        return f"skip: {n_rounds} round files present but no latency/vram keys"

    patch = {}
    if latencies:
        patch["avg_aggregation_latency_sec"] = float(mean(latencies))
        patch["max_aggregation_latency_sec"] = float(max(latencies))
        patch["median_aggregation_latency_sec"] = float(median(latencies))
    if vrams:
        patch["max_aggregation_peak_vram_mb"] = float(max(vrams))

    if dry_run:
        return f"dry-run: would patch {len(patch)} key(s) from {n_rounds} rounds"

    # Merge (force overrides existing; else fill keys that are missing OR null).
    # Treating null the same as missing makes the backfill idempotent against
    # final_metrics.json files written with placeholder None values.
    if force:
        fm.update(patch)
    else:
        for k, v in patch.items():
            if fm.get(k) is None:
                fm[k] = v

    atomic_write_json(fm, final_metrics_path)
    return f"patched ({len(patch)} keys from {n_rounds} rounds)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="",
                    help="Only process result dirs starting with this prefix "
                         "(e.g. step10_scalability, step12_main_summary)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview changes without writing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing values in final_metrics.json")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR.resolve()}")
        return

    # Find every final_metrics.json under results/
    glob_pattern = f"**/final_metrics.json"
    if args.prefix:
        scenario_dirs = list(RESULTS_DIR.glob(f"{args.prefix}*"))
    else:
        scenario_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]

    total = 0
    patched = 0
    skipped_existing = 0
    skipped_other = 0

    for sdir in sorted(scenario_dirs):
        if not sdir.is_dir():
            continue
        for fm_path in sdir.rglob("final_metrics.json"):
            total += 1
            status = backfill_cell(fm_path, force=args.force, dry_run=args.dry_run)
            rel = fm_path.relative_to(RESULTS_DIR).parent
            if status.startswith("patched") or status.startswith("dry-run"):
                patched += 1
                print(f"  [{status}] {rel}")
            elif "already has all keys" in status:
                skipped_existing += 1
            else:
                skipped_other += 1
                # Print reason for non-existing-key skips so they're debuggable
                if args.dry_run or "no agg_stats" in status or "unreadable" in status:
                    print(f"  [{status}] {rel}")

    print()
    print(f"Scanned  : {total} final_metrics.json files")
    print(f"Patched  : {patched}")
    print(f"Skipped (already populated)   : {skipped_existing}")
    print(f"Skipped (no agg_stats / other): {skipped_other}")
    if args.dry_run:
        print("\n(Dry run — no files written. Re-run without --dry-run to apply.)")


if __name__ == "__main__":
    main()
