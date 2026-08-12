"""
plot_system_perf.py — Generate SIGMOD-style system performance figures from
the CSV produced by extract_system_perf.py.

Produces (in --out-dir, default ./figures/system_perf/):
  - throughput_by_defense.pdf      # rounds/sec per defense, grouped by dataset
  - wall_clock_by_defense.pdf      # total seconds per defense, grouped by dataset
  - peak_gpu_memory_by_defense.pdf # peak GPU MB per defense, grouped by dataset
  - communication_cost.pdf         # avg upload MB/round per defense
  - cost_vs_quality.pdf            # ACC vs wall-clock scatter

Usage:
    python extract_system_perf.py --output system_perf.csv
    python plot_system_perf.py --csv system_perf.csv

Designed to fail gracefully: any plot whose required column is missing or
all-NaN gets a "no data" placeholder so the script never crashes mid-run.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _try_float(x) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python extract_system_perf.py --output {path}` first."
        )
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate_by(rows: List[dict], group_keys: tuple, value_key: str) -> Dict[tuple, dict]:
    """Group rows by `group_keys` and aggregate `value_key` to (mean, n, min, max)."""
    bucket = defaultdict(list)
    for r in rows:
        v = _try_float(r.get(value_key))
        if v is None:
            continue
        key = tuple(r.get(k, "") for k in group_keys)
        bucket[key].append(v)

    out = {}
    for key, vals in bucket.items():
        if not vals:
            continue
        out[key] = {
            "mean": sum(vals) / len(vals),
            "min":  min(vals),
            "max":  max(vals),
            "n":    len(vals),
        }
    return out


def plot_grouped_bar(rows: List[dict], value_key: str, ylabel: str, title: str,
                     out_path: Path):
    """Grouped bar chart: x = defense, color = dataset, y = mean(value_key)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [skip] matplotlib not installed — {out_path.name}")
        return

    agg = aggregate_by(rows, ("defense", "dataset"), value_key)
    if not agg:
        print(f"  [skip] no data for {value_key} — {out_path.name}")
        return

    defenses = sorted(set(k[0] for k in agg))
    datasets = sorted(set(k[1] for k in agg))
    if not defenses or not datasets:
        print(f"  [skip] empty groups for {value_key} — {out_path.name}")
        return

    n_defenses = len(defenses)
    n_datasets = len(datasets)
    bar_w = 0.8 / max(n_datasets, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_defenses * 1.2), 5))
    cmap = plt.get_cmap("tab10")

    for j, ds in enumerate(datasets):
        xs = []
        ys = []
        for i, d in enumerate(defenses):
            stat = agg.get((d, ds))
            if stat is None:
                continue
            xs.append(i + (j - n_datasets / 2) * bar_w + bar_w / 2)
            ys.append(stat["mean"])
        if xs:
            ax.bar(xs, ys, width=bar_w, label=ds, color=cmap(j % 10))

    ax.set_xticks(range(n_defenses))
    ax.set_xticklabels(defenses, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=min(n_datasets, 3))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_cost_vs_quality(rows: List[dict], out_path: Path):
    """Scatter: x = wall_clock_seconds, y = acc, color = dataset, marker = defense."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [skip] matplotlib not installed — {out_path.name}")
        return

    points = []
    for r in rows:
        wc = _try_float(r.get("wall_clock_seconds"))
        acc = _try_float(r.get("acc"))
        if wc is None or acc is None:
            continue
        points.append({
            "wc": wc,
            "acc": acc,
            "dataset": r.get("dataset", ""),
            "defense": r.get("defense", ""),
        })

    if not points:
        print(f"  [skip] no (wall_clock, acc) pairs — {out_path.name}")
        return

    datasets = sorted(set(p["dataset"] for p in points))
    cmap = plt.get_cmap("tab10")
    color_map = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in datasets:
        xs = [p["wc"] for p in points if p["dataset"] == ds]
        ys = [p["acc"] for p in points if p["dataset"] == ds]
        ax.scatter(xs, ys, label=ds, color=color_map[ds], alpha=0.7, s=40)

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock seconds (log scale)")
    ax.set_ylabel("Clean Accuracy")
    ax.set_title("Cost vs. Quality — wall clock vs. clean accuracy per cell")
    ax.legend(loc="best", fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="system_perf.csv",
                    help="Input CSV from extract_system_perf.py (default: system_perf.csv)")
    ap.add_argument("--out-dir", default="figures/system_perf",
                    help="Output directory for figures (default: figures/system_perf/)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                    help="Image format (default: pdf)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} row(s) from {csv_path}")

    ext = args.format

    print("\nGenerating figures...")
    plot_grouped_bar(
        rows, "throughput_rounds_per_sec",
        "Throughput (rounds / sec)",
        "Training Throughput per Defense",
        out_dir / f"throughput_by_defense.{ext}",
    )
    plot_grouped_bar(
        rows, "wall_clock_seconds",
        "Wall-clock seconds",
        "Total Training Wall Clock per Defense",
        out_dir / f"wall_clock_by_defense.{ext}",
    )
    plot_grouped_bar(
        rows, "avg_seconds_per_round",
        "Seconds / round",
        "Per-Round Latency per Defense",
        out_dir / f"latency_by_defense.{ext}",
    )
    plot_grouped_bar(
        rows, "peak_gpu_memory_allocated_mb",
        "Peak GPU memory (MB)",
        "Peak GPU Memory per Defense",
        out_dir / f"peak_gpu_memory_by_defense.{ext}",
    )
    plot_grouped_bar(
        rows, "peak_host_rss_mb",
        "Peak host RSS (MB)",
        "Peak Host RAM per Defense",
        out_dir / f"peak_host_rss_by_defense.{ext}",
    )
    plot_grouped_bar(
        rows, "avg_upload_mb_per_round",
        "Avg upload (MB / round)",
        "Communication Cost per Defense",
        out_dir / f"communication_cost.{ext}",
    )
    plot_cost_vs_quality(rows, out_dir / f"cost_vs_quality.{ext}")

    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
