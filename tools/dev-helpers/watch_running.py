"""
watch_running.py — Show which experiment runner logs are currently growing,
and what scenario / dataset / step / round each one is on.

Usage:
    python watch_running.py                       # one-shot snapshot
    python watch_running.py --watch               # refresh every 5s
    python watch_running.py --watch --interval 10 # refresh every 10s
    python watch_running.py --window 30           # treat <30s mtime as GROWING
    python watch_running.py --tail 8              # show last 8 log lines per growing log
    python watch_running.py --logs-dir configs_generated_benchmark

What it does:
  1. Lists every *_runner.log under --logs-dir, sorted by recency.
  2. For each, prints size, age (seconds since last write), and status:
       GROWING — modified within --window seconds (likely active worker)
       IDLE    — modified within the last hour but quiet right now
       STALE   — older than 1h
  3. For GROWING logs, parses the tail to extract:
       - the most recent config_path (→ step + dataset + defense)
       - the most recent round / epoch
       - the most recent GPU id
  4. Optionally prints the last N lines of each growing log so you can see
     exactly what each worker is doing.

This is read-only; it does not touch any results or configs.
"""

import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Regexes for parsing log content
# ---------------------------------------------------------------------------
# "[Run 12 | Smp 1 | Seed 42 | GPU 1] ..." prefix used by every per-worker line.
# Extracted independently from Config: lines because the prefix appears on every
# log line, but Config: appears only once per cell and scrolls out of any
# reasonable tail window quickly on verbose logs.
RE_RUN_PREFIX = re.compile(
    r"\[Run\s+(\d+)\s*\|\s*Smp\s+(\d+)\s*\|\s*Seed\s+(\d+)\s*\|\s*GPU\s+(\S+?)(?:\s*\(.*?\))?\]"
)
# "Config: configs_generated_benchmark/.../config.yaml"
RE_CONFIG_ANY = re.compile(r"Config:\s*(\S+\.yaml)")
# Save path inside the config — written as 'save_path: results/step12_main_summary_...'
# This appears every time a config is loaded and is a robust fallback when the
# Config: line itself has scrolled past.
RE_SAVE_PATH = re.compile(r"results[\\/]([a-zA-Z0-9._]+)")
# "Starting aggregation for epoch 5"  or  "Round 12 / 200"  or just "epoch 47"
RE_ROUND = re.compile(r"(?:Starting aggregation for epoch|epoch|Round)\s+(\d+)", re.IGNORECASE)
# Scenario directory inside the config path: anything between "configs_generated_benchmark/" and the next "/"
RE_SCENARIO_DIR = re.compile(r"configs_generated_benchmark[\\/]([^/\\]+)[\\/]")
# "ds-cifar100" / "ds-trec" / etc. — appears in every HP cell directory name
# and is the most reliable dataset signal for steps whose scenario name doesn't
# include a dataset (e.g. step5 attack sensitivity, which only encodes modality).
RE_DS_TAG = re.compile(r"ds-([a-z0-9]+)", re.IGNORECASE)
DS_TAG_TO_PROPER = {
    "cifar100": "CIFAR100", "cifar10": "CIFAR10", "femnist": "FEMNIST",
    "texas100": "Texas100", "purchase100": "Purchase100", "trec": "TREC",
}
# "seed-42" / "seed_42" — appears in every HP cell + run dir, far more reliable
# than the [Run … | Seed N] worker prefix which only appears once per cell.
RE_SEED_TAG = re.compile(r"seed[-_](\d+)")
# Datasets we care about (proper-case as used in current scenario names)
DATASETS = ["CIFAR100", "CIFAR10", "FEMNIST", "Texas100", "Purchase100", "TREC"]
# Defenses (longer names first so skymask_small matches before skymask)
DEFENSES = [
    "skymask_small", "trimmed_mean", "multi_krum",
    "fedavg", "fltrust", "martfl", "skymask", "rflpa", "spmc",
    "flame", "deepsight", "bulyan", "foolsgold",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human_size(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:6.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def human_age(sec):
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"
    if sec < 86400:
        return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"
    return f"{int(sec // 86400)}d"


def tail(path: Path, n: int = 20000, max_bytes: int = 16 * 1024 * 1024):
    """Return the last n lines of a file, reading at most max_bytes from the end.

    Defaults are tuned for very verbose training logs (per-batch debug output
    from gradient_seller, markplace_gradient, etc., where worker prefixes are
    rare). A 16 MB tail is roughly the last 5-15 minutes of one cell, large
    enough to almost always contain the most recent [Run … | GPU N] prefix
    that gets logged once per cell startup."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def parse_recent_state(lines, log_filename=""):
    """Walk lines bottom-up; pull the most recent worker prefix + config + round.

    The Run-prefix and Config-line are extracted independently because they
    only co-occur once per cell. log_filename is used as a fallback for the
    step name (e.g. step5_attack_sensitivity_runner.log → step5).
    """
    state = {"config": None, "scenario": None, "step": None, "dataset": None,
             "defense": None, "round": None, "gpu": None, "seed": None}

    # 1. Most recent round (any of the round/epoch markers)
    for line in reversed(lines):
        m = RE_ROUND.search(line)
        if m:
            state["round"] = int(m.group(1))
            break

    # 2. Most recent worker prefix → seed + gpu (only appears on lines that
    #    pass through run_parallel_experiment.py logger; most per-batch debug
    #    lines lack it, so this can fail on busy logs even with a big tail).
    for line in reversed(lines):
        m = RE_RUN_PREFIX.search(line)
        if m:
            state["seed"] = m.group(3)
            state["gpu"] = m.group(4)
            break

    # 3. Most recent config path (logged once per cell)
    for line in reversed(lines):
        m = RE_CONFIG_ANY.search(line)
        if m:
            state["config"] = m.group(1)
            break

    # 4. Fallback: any 'results/<scenario>' path mentioned in the tail
    #    (e.g. from save_path init logs or .success/.failed marker writes)
    if state["config"] is None:
        for line in reversed(lines):
            m = RE_SAVE_PATH.search(line)
            if m and m.group(1).startswith("step"):
                state["scenario"] = m.group(1)
                break

    # 4b. Dataset fallback: scan tail for "ds-cifar100" etc. cell-dir tags.
    #     This is more reliable than parsing the scenario name because
    #     several steps (e.g. step5 attack_sens) encode only modality in the
    #     scenario name, with the dataset showing up only in the cell path.
    if not state["dataset"]:
        for line in reversed(lines):
            m = RE_DS_TAG.search(line)
            if m and m.group(1).lower() in DS_TAG_TO_PROPER:
                state["dataset"] = DS_TAG_TO_PROPER[m.group(1).lower()]
                break

    # 4c. Seed fallback: scan tail for "seed-42" / "seed_42" cell-dir tags.
    #     Cell paths embed the seed and are logged on every save_path init,
    #     so this catches the seed even when the [Run … | Seed N] prefix is
    #     out of the tail window.
    if not state["seed"]:
        for line in reversed(lines):
            m = RE_SEED_TAG.search(line)
            if m:
                state["seed"] = m.group(1)
                break

    # Derive scenario / step / dataset / defense from the config path
    if state["config"] and not state["scenario"]:
        m = RE_SCENARIO_DIR.search(state["config"])
        if m:
            state["scenario"] = m.group(1)

    if state["scenario"]:
        sm = re.match(r"(step[\d.]+)", state["scenario"])
        if sm:
            state["step"] = sm.group(1)
        for ds in DATASETS:
            if ds in state["scenario"]:
                state["dataset"] = ds
                break
        for d in DEFENSES:
            if f"_{d}_" in f"_{state['scenario']}_" or state["scenario"].endswith(f"_{d}"):
                state["defense"] = d
                break

    # 5. Final fallback: derive step from log filename. The runner log is named
    #    after the configs subdir (e.g. step5_attack_sensitivity_runner.log),
    #    which always begins with the canonical stepN_ token.
    if not state["step"] and log_filename:
        sm = re.match(r"(step[\d.]+)", log_filename)
        if sm:
            state["step"] = sm.group(1)

    return state


def classify(age_sec, growing_window):
    if age_sec < growing_window:
        return "GROWING"
    if age_sec < 3600:
        return "IDLE"
    return "STALE"


def render(logs_dir: Path, growing_window: int, tail_lines: int):
    log_files = sorted(
        logs_dir.glob("*_runner.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not log_files:
        print(f"❌ No *_runner.log files found in {logs_dir.resolve()}")
        return

    now = time.time()
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    "
          f"watching {logs_dir}    growing-window={growing_window}s")
    print()
    header = f"{'STATUS':<8}{'AGE':>8}  {'SIZE':>8}  {'STEP':<10}{'DATASET':<12}{'DEFENSE':<14}{'ROUND':>6}  {'GPU':>4}  {'SEED':>5}  LOG"
    print(header)
    print("-" * len(header))

    growing_logs = []
    for path in log_files:
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        age = now - st.st_mtime
        status = classify(age, growing_window)

        # Always parse step from filename (even for STALE/IDLE) so the table
        # has a meaningful step column. Only do the expensive tail+parse for
        # GROWING logs to keep the script fast.
        if status == "GROWING":
            lines = tail(path)
            state = parse_recent_state(lines, log_filename=path.name)
        else:
            lines = []
            state = parse_recent_state([], log_filename=path.name)

        step = state["step"] or "—"
        dataset = state["dataset"] or "—"
        defense = state["defense"] or "—"
        rnd = str(state["round"]) if state["round"] is not None else "—"
        gpu = state["gpu"] or "—"
        seed = state["seed"] or "—"

        marker = {"GROWING": "🟢", "IDLE": "🟡", "STALE": "⚫"}[status]
        print(f"{marker} {status:<6}{human_age(age):>7}  {human_size(st.st_size):>8}  "
              f"{step:<10}{dataset:<12}{defense:<14}{rnd:>6}  {gpu:>4}  {seed:>5}  "
              f"{path.name}")

        if status == "GROWING":
            growing_logs.append((path, lines))

    if not growing_logs:
        print("\n(no logs are currently growing — nothing appears to be running)")
        return

    if tail_lines > 0:
        print()
        for path, lines in growing_logs:
            print(f"━━━ tail -{tail_lines} {path.name} ━━━")
            for ln in lines[-tail_lines:]:
                print(f"  {ln}")
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="configs_generated_benchmark",
                    help="Directory containing *_runner.log files")
    ap.add_argument("--window", type=int, default=120,
                    help="Seconds since last write to count as GROWING (default: 120)")
    ap.add_argument("--tail", type=int, default=4,
                    help="How many trailing lines to print per growing log (default: 4, 0 to disable)")
    ap.add_argument("--watch", action="store_true",
                    help="Re-render every --interval seconds until Ctrl+C")
    ap.add_argument("--interval", type=int, default=5,
                    help="Watch refresh interval in seconds (default: 5)")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"❌ Directory not found: {logs_dir.resolve()}")
        sys.exit(1)

    if not args.watch:
        render(logs_dir, args.window, args.tail)
        return

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            render(logs_dir, args.window, args.tail)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
