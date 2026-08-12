"""
backfill_and_mark_success.py — Reconstruct final_metrics.json + mark .success
for cells that completed training but never finished the final evaluation
step.

Context: some cells have per-round data in evaluations/ and agg_stats/ but
no final_metrics.json (and therefore no .success marker) because
run_final_evaluation_and_logging() crashed or the dispatcher was killed
before it ran. This script reconstructs the final metrics from per-round
artifacts and writes a .success marker, but ONLY if the reconstruction
yields a valid acc (or perplexity) value.

Data sources, in priority order:
  1. evaluations/round_<N>.json (last round = final eval)   → acc, asr, ...
  2. training_log.csv (last row)                             → fallback for acc/asr
  3. agg_stats/round_<N>.json                                → aggregation_latency/VRAM
  4. seller_metrics.csv                                       → per-seller upload bytes (optional)

A cell is considered successfully reconstructible only if step 1 or step 2
yields a numeric acc or perplexity. Otherwise the cell is left untouched.

Safe to re-run: does not overwrite an existing final_metrics.json or
.success unless --force is passed.

Usage:
    python backfill_and_mark_success.py                                  # dry-run everything
    python backfill_and_mark_success.py --apply                          # actually write
    python backfill_and_mark_success.py --prefix step10_scalability      # just step 8
    python backfill_and_mark_success.py --prefix step12_main_summary     # just step 10
    python backfill_and_mark_success.py --apply --force                  # overwrite existing
"""

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median

RESULTS_DIR = Path("./results")

RUN_DIR_RE = re.compile(r"^run_\d+_seed_\d+$")


# ---------------------------------------------------------------------------
# Finders
# ---------------------------------------------------------------------------
def find_run_dirs(root: Path, prefix: str):
    """Yield every directory that matches 'run_<N>_seed_<M>' under root."""
    if prefix:
        scenarios = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    else:
        scenarios = [d for d in root.iterdir() if d.is_dir()]
    for scen in scenarios:
        for d in scen.rglob("run_*_seed_*"):
            if d.is_dir() and RUN_DIR_RE.match(d.name):
                yield d


# ---------------------------------------------------------------------------
# Readers: extract metrics from different per-round artifacts
# ---------------------------------------------------------------------------
def read_last_evaluation(run_dir: Path) -> dict | None:
    """Return the eval_metrics dict from the highest-numbered
    evaluations/round_<N>.json, or None if nothing valid."""
    ev_dir = run_dir / "evaluations"
    if not ev_dir.is_dir():
        return None

    round_files = []
    for f in ev_dir.glob("round_*.json"):
        m = re.match(r"round_(\d+)\.json$", f.name)
        if m:
            round_files.append((int(m.group(1)), f))
    if not round_files:
        return None
    round_files.sort()
    last_n, last_path = round_files[-1]
    try:
        with open(last_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    data["_source_round"] = last_n
    data["_n_eval_rounds"] = len(round_files)
    return data


def read_training_log_tail(run_dir: Path) -> dict | None:
    """Return the last row of training_log.csv as a dict, or None."""
    tl = run_dir / "training_log.csv"
    if not tl.exists():
        return None
    try:
        with open(tl) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return None
    return rows[-1]


def read_agg_stats_aggregates(run_dir: Path) -> dict:
    """Aggregate aggregation_latency / VRAM across agg_stats/round_*.json."""
    out = {}
    agg_dir = run_dir / "agg_stats"
    if not agg_dir.is_dir():
        return out

    latencies, vrams = [], []
    n_rounds = 0
    for f in agg_dir.glob("round_*.json"):
        try:
            with open(f) as fp:
                d = json.load(fp)
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

    if latencies:
        out["avg_aggregation_latency_sec"] = float(mean(latencies))
        out["max_aggregation_latency_sec"] = float(max(latencies))
        out["median_aggregation_latency_sec"] = float(median(latencies))
    if vrams:
        out["max_aggregation_peak_vram_mb"] = float(max(vrams))
    if n_rounds:
        out["_agg_stats_n_rounds"] = n_rounds
    return out


def read_upload_bytes(run_dir: Path) -> dict:
    """Try to total upload bytes from seller_metrics.csv if present."""
    out = {}
    sm = run_dir / "seller_metrics.csv"
    if not sm.exists():
        return out
    try:
        with open(sm) as f:
            reader = csv.DictReader(f)
            total = 0
            total_accepted = 0
            for row in reader:
                try:
                    b = int(row.get("upload_bytes") or 0)
                    total += b
                    # If there's a 'selected' column and it's True, count accepted
                    sel = (row.get("selected") or "").lower()
                    if sel in ("true", "1", "yes"):
                        total_accepted += b
                except (TypeError, ValueError):
                    pass
        if total:
            out["total_upload_bytes"] = total
            out["total_upload_mb"] = total / (1024 * 1024)
        if total_accepted:
            out["total_upload_bytes_accepted"] = total_accepted
            out["total_upload_mb_accepted"] = total_accepted / (1024 * 1024)
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Core: attempt reconstruction of a single run dir
# ---------------------------------------------------------------------------
def reconstruct_final_metrics(run_dir: Path) -> tuple[dict | None, str]:
    """Return (metrics_dict, reason). metrics_dict is None when
    reconstruction can't produce a valid acc/perplexity."""
    metrics = {}

    # Primary signal: last evaluation
    ev = read_last_evaluation(run_dir)
    if ev:
        # Copy the most common keys if present
        for k in ("acc", "loss", "asr", "B-Acc", "B-F1", "perplexity",
                  "safety_score", "refusal_bias"):
            if k in ev and ev[k] is not None:
                metrics[k] = ev[k]
        if ev.get("_source_round") is not None:
            metrics["completed_rounds"] = ev["_source_round"]
        if ev.get("_n_eval_rounds") is not None:
            metrics["_n_eval_rounds"] = ev["_n_eval_rounds"]

    # Fallback: training_log tail (only if evaluation didn't give us acc)
    if "acc" not in metrics and "perplexity" not in metrics:
        tl = read_training_log_tail(run_dir)
        if tl:
            # Try various column names commonly seen
            for src_key in ("test_acc", "val_acc", "acc"):
                v = tl.get(src_key)
                if v not in (None, ""):
                    try:
                        metrics["acc"] = float(v)
                        metrics["_acc_source"] = f"training_log.csv:{src_key}"
                        break
                    except (TypeError, ValueError):
                        pass
            for src_key in ("test_asr", "asr"):
                v = tl.get(src_key)
                if v not in (None, ""):
                    try:
                        metrics["asr"] = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
            if "round" in tl and tl["round"] not in (None, ""):
                try:
                    metrics["completed_rounds"] = int(float(tl["round"]))
                except (TypeError, ValueError):
                    pass

    # Reject if we still have no acc/perplexity
    if "acc" not in metrics and "perplexity" not in metrics:
        return None, "no acc/perplexity derivable from evaluations/ or training_log.csv"

    # Enrich with aggregation metrics (per-round agg_stats)
    agg = read_agg_stats_aggregates(run_dir)
    metrics.update(agg)

    # Enrich with communication cost if seller_metrics has it
    metrics.update(read_upload_bytes(run_dir))

    # Metadata about how we reconstructed this
    metrics["_reconstructed"] = True

    return metrics, "ok"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------
def atomic_write_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Process one run dir
# ---------------------------------------------------------------------------
def process_run(run_dir: Path, apply: bool, force: bool,
                skip_in_progress: bool = True) -> str:
    fm_path = run_dir / "final_metrics.json"
    success_path = run_dir / ".success"
    in_progress_path = run_dir / ".in_progress"

    fm_exists = fm_path.exists()
    success_exists = success_path.exists()
    in_progress_exists = in_progress_path.exists()

    # Skip cells that are still being trained — even if they happen to have
    # partial evaluations already on disk. The user will re-run these cleanly
    # so we must not write a premature .success marker.
    if in_progress_exists and skip_in_progress:
        return "skip: .in_progress present (cell is still running or was killed mid-run)"

    # If everything's already there, only touch with --force.
    if fm_exists and success_exists and not force:
        return "skip: already complete"

    metrics, reason = reconstruct_final_metrics(run_dir)
    if metrics is None:
        return f"skip: {reason}"

    # If final_metrics.json exists, merge our backfill into it rather than
    # overwriting (unless --force).
    if fm_exists:
        try:
            with open(fm_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
        if force:
            existing.update(metrics)
        else:
            for k, v in metrics.items():
                if k not in existing:
                    existing[k] = v
        merged = existing
    else:
        merged = metrics

    n_keys = len(metrics)
    n_rounds = metrics.get("_agg_stats_n_rounds", 0)
    has_asr = "asr" in metrics
    acc_val = metrics.get("acc", metrics.get("perplexity"))

    action_parts = []
    if not fm_exists:
        action_parts.append(f"write final_metrics.json (acc={acc_val}, asr={metrics.get('asr','—')}, "
                            f"{n_rounds} agg rounds)")
    else:
        action_parts.append(f"merge into existing final_metrics.json")
    if not success_exists:
        action_parts.append("create .success")
    if force and success_exists:
        action_parts.append("touch .success")

    if not apply:
        return "preview: " + " + ".join(action_parts)

    atomic_write_json(merged, fm_path)
    if not success_exists or force:
        success_path.touch()

    return "done: " + " + ".join(action_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prefix", default="",
                    help="Only process results/<prefix>* (e.g. step10_scalability, "
                         "step12_main_summary, step3_tune). Default: all scenarios.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write files. Default is dry-run.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing final_metrics.json fields and refresh .success")
    ap.add_argument("--include-in-progress", action="store_true",
                    help="Also process cells with a .in_progress marker. "
                         "Default is to skip them so the user can re-run the cell cleanly "
                         "and avoid partial results being marked as .success.")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR.resolve()}")
        return

    print(f"{'DRY RUN — no files will be changed' if not args.apply else 'APPLYING CHANGES'}")
    print(f"Prefix           : {args.prefix or '(all)'}")
    print(f"Force            : {args.force}")
    print(f"Skip in_progress : {'no (will process)' if args.include_in_progress else 'yes (default)'}")
    print()

    total = 0
    done = 0
    already = 0
    skipped = 0
    skipped_in_progress = 0
    reasons = {}

    for run_dir in find_run_dirs(RESULTS_DIR, args.prefix):
        total += 1
        status = process_run(
            run_dir,
            apply=args.apply,
            force=args.force,
            skip_in_progress=not args.include_in_progress,
        )

        rel = run_dir.relative_to(RESULTS_DIR)
        if status.startswith("done") or status.startswith("preview"):
            done += 1
            print(f"  [{status.split(':')[0]:<8}] {rel}")
        elif "already complete" in status:
            already += 1
        elif ".in_progress" in status:
            skipped_in_progress += 1
            print(f"  [in_prog] {rel}  (will not be marked .success)")
        else:
            skipped += 1
            reasons[status] = reasons.get(status, 0) + 1

    print()
    print("=" * 72)
    print(f"Run dirs scanned                : {total}")
    print(f"{'Would write' if not args.apply else 'Written':<32}: {done}")
    print(f"Already complete (skipped)      : {already}")
    print(f"Skipped (.in_progress)          : {skipped_in_progress}")
    print(f"Skipped (cannot reconstruct)    : {skipped}")
    if reasons:
        print()
        print("Skip reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>5}x  {reason}")
    print()
    if not args.apply:
        print("This was a dry run. Re-run with --apply to actually write files.")


if __name__ == "__main__":
    main()
