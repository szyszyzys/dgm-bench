"""
Scan all per-step runner logs and report seller-dropout statistics.

For each `<step>_runner.log` under configs_generated_benchmark/, this script:
  1. Parses every "📊 Seller Initialization Summary" block.
  2. Counts how many init blocks had created < requested (= partition-time loss).
  3. Counts mid-run failures: "Empty dataset! Skipping",
     "is inactive, skipping", "returned None gradient", "NaN in gradient".
  4. Prints a per-step table so you can see which steps are affected.

Usage:
    python check_seller_dropout.py
    python check_seller_dropout.py --logs-dir configs_generated_benchmark
    python check_seller_dropout.py --show-examples 3   # also print 3 example lines per category
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

# "  - Total created: 7/9"
RE_TOTAL = re.compile(r"Total created:\s*(\d+)\s*/\s*(\d+)")
# "  - Adversaries: 2/2"
RE_ADV = re.compile(r"Adversaries:\s*(\d+)\s*/\s*(\d+)")
# "  - Benign: 5/7"
RE_BENIGN = re.compile(r"Benign:\s*(\d+)\s*/\s*(\d+)")

DROPOUT_PATTERNS = {
    "empty_shard":      re.compile(r"Empty dataset! Skipping"),
    "inactive":         re.compile(r"is inactive, skipping"),
    "none_gradient":    re.compile(r"returned None gradient"),
    "nan_in_gradient":  re.compile(r"NaN in gradient"),
    "compute_failed":   re.compile(r"_compute_local_grad returned None"),
}


def scan_log(path: Path, show_examples: int = 0):
    """Return a dict of stats for one log file."""
    stats = {
        "init_blocks": 0,
        "init_complete": 0,           # X == Y
        "init_partial": 0,            # X < Y
        "min_ratio": (None, None),    # (X, Y) of worst run
        "totals": Counter(),          # for each (X, Y) pair, how many runs
        "dropout_counts": Counter(),
        "dropout_examples": defaultdict(list),
    }
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return stats

    # Init summaries
    for m in RE_TOTAL.finditer(text):
        x, y = int(m.group(1)), int(m.group(2))
        stats["init_blocks"] += 1
        stats["totals"][(x, y)] += 1
        if x == y:
            stats["init_complete"] += 1
        else:
            stats["init_partial"] += 1
            ratio = x / y if y else 1.0
            cur_x, cur_y = stats["min_ratio"]
            if cur_y is None or ratio < (cur_x / cur_y if cur_y else 1.0):
                stats["min_ratio"] = (x, y)

    # Mid-run dropout markers
    for name, rx in DROPOUT_PATTERNS.items():
        for m in rx.finditer(text):
            stats["dropout_counts"][name] += 1
            if show_examples and len(stats["dropout_examples"][name]) < show_examples:
                start = text.rfind("\n", 0, m.start()) + 1
                end = text.find("\n", m.end())
                stats["dropout_examples"][name].append(text[start:end].strip())

    return stats


def fmt_pair(x, y):
    if x is None or y is None:
        return "—"
    return f"{x}/{y} ({x/y:.0%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="configs_generated_benchmark",
                    help="Directory containing *_runner.log files")
    ap.add_argument("--show-examples", type=int, default=0,
                    help="Print N example log lines per dropout category")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"❌ Directory not found: {logs_dir.resolve()}")
        return

    log_files = sorted(logs_dir.glob("*_runner.log"))
    if not log_files:
        print(f"❌ No *_runner.log files in {logs_dir.resolve()}")
        return

    print(f"📂 Scanning {len(log_files)} runner logs in {logs_dir}\n")

    header = (
        f"{'Step (log file)':<48}"
        f"{'#runs':>7}"
        f"{'  complete':>11}"
        f"{'  partial':>10}"
        f"{'  worst X/Y':>14}"
        f"{'  empty':>8}"
        f"{'  inactive':>11}"
        f"{'  None grad':>12}"
        f"{'  NaN':>7}"
    )
    print(header)
    print("-" * len(header))

    grand = Counter()
    flagged = []

    for log_path in log_files:
        s = scan_log(log_path, args.show_examples)
        step_name = log_path.stem.replace("_runner", "")
        grand["init_blocks"] += s["init_blocks"]
        grand["init_partial"] += s["init_partial"]
        for k, v in s["dropout_counts"].items():
            grand[k] += v

        worst = fmt_pair(*s["min_ratio"]) if s["min_ratio"][0] is not None else "—"

        print(
            f"{step_name[:48]:<48}"
            f"{s['init_blocks']:>7}"
            f"{s['init_complete']:>11}"
            f"{s['init_partial']:>10}"
            f"{worst:>14}"
            f"{s['dropout_counts']['empty_shard']:>8}"
            f"{s['dropout_counts']['inactive']:>11}"
            f"{s['dropout_counts']['none_gradient']:>12}"
            f"{s['dropout_counts']['nan_in_gradient']:>7}"
        )

        if s["init_partial"] > 0 or s["dropout_counts"]:
            flagged.append((step_name, s))

    print("-" * len(header))
    print(
        f"{'TOTAL':<48}"
        f"{grand['init_blocks']:>7}"
        f"{'':>11}"
        f"{grand['init_partial']:>10}"
        f"{'':>14}"
        f"{grand['empty_shard']:>8}"
        f"{grand['inactive']:>11}"
        f"{grand['none_gradient']:>12}"
        f"{grand['nan_in_gradient']:>7}"
    )

    print("\nLegend:")
    print("  complete  = runs where every requested seller was registered (X == Y)")
    print("  partial   = runs where some sellers were dropped at init (X <  Y)")
    print("  worst X/Y = the run with the lowest registration ratio in this step")
    print("  empty     = 'Empty dataset! Skipping'   (partitioner gave 0 samples)")
    print("  inactive  = seller had is_active=False at gradient time")
    print("  None grad = _compute_local_grad returned None mid-run")
    print("  NaN       = NaN detected in gradient")

    if not flagged:
        print("\n✅ No seller dropout detected in any step.")
        return

    print(f"\n⚠️  {len(flagged)} step(s) had dropout. Distribution of (X, Y) pairs:")
    for step_name, s in flagged:
        if not s["totals"]:
            continue
        partials = {k: v for k, v in s["totals"].items() if k[0] < k[1]}
        if not partials:
            continue
        print(f"\n  {step_name}:")
        for (x, y), n in sorted(partials.items()):
            print(f"    {n:4d} run(s) created {x}/{y} sellers ({x/y:.0%})")

    if args.show_examples:
        print("\n📋 Example log lines per dropout category:")
        for step_name, s in flagged:
            if not any(s["dropout_examples"].values()):
                continue
            print(f"\n  {step_name}:")
            for cat, examples in s["dropout_examples"].items():
                for ex in examples:
                    print(f"    [{cat}] {ex}")


if __name__ == "__main__":
    main()
