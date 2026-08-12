"""
watch_health.py — Confirm the benchmark is *healthily* running.

Companion to watch_step10.sh (coarse progress) and watch_running.py (what each
cell is doing). This script answers: "Is each GPU actually busy, is each worker
making forward progress, and if not, why not?"

Usage:
    python watch_health.py                         # one-shot snapshot
    python watch_health.py --watch                 # refresh every 10s
    python watch_health.py --watch --interval 5
    python watch_health.py --logs-dir configs_generated_benchmark

Output columns:
    GPU   util%  mem      pwr    pid     cpu%  rss    step/dataset/defense   round  r/min  verdict

Verdicts:
    OK          — GPU util >= 50% and log is advancing
    LIGHT       — GPU util 20-50%: running but underutilized
    CPU-BOUND   — GPU < 30% and CPU > 300% (dataloader/preprocessing bottleneck)
    IO-BOUND    — GPU < 30% and CPU < 100% and disk I/O active (dataset read bound)
    HUNG        — PID alive but log has not grown for > 120s
    IDLE        — no worker process on this GPU
    NO-LOG      — worker process present but no growing runner log matches

Read-only: does not touch any results, configs, or processes.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Reuse log parsing from watch_running.py so there's one source of truth.
from watch_running import tail, parse_recent_state, human_size

try:
    import psutil
except ImportError:
    print("❌ psutil is required: pip install psutil", file=sys.stderr)
    sys.exit(1)


# A process is counted as "our worker" if its exe path or cmdline contains ANY
# of these tokens. Covers both the dispatcher (cmdline mentions the script) and
# spawned workers (exe path lives inside the project's conda env). Tune with
# --worker-token / --exclude-token if your env name differs.
# Tokens that count a GPU-using process as ours. Env name alone is OK here
# because we restrict this matching to processes that actually hold GPU memory.
DEFAULT_WORKER_TOKENS = (
    "run_parallel_experiment.py",
    "run_full_benchmark.py",
    "poison_data_valuation",
    "ddm_fix",
)
# Stricter set used for the no-GPU dispatcher scan: must mention a real script
# path, not just the conda env, otherwise we'd catch every shell/jupyter in the
# env (saw workers=113 in practice).
DEFAULT_SCRIPT_TOKENS = (
    "run_parallel_experiment.py",
    "run_full_benchmark.py",
    "poison_data_valuation",
)
# Any process matching these is explicitly NOT ours (other users / other projects
# sharing the same box). Checked before the worker-token allowlist.
DEFAULT_EXCLUDE_TOKENS = (
    "vllm",
)

HUNG_AFTER_SEC = 120          # log silent this long => HUNG
GROWING_WINDOW_SEC = 120      # log must have been written within this to count


# ---------------------------------------------------------------------------
# GPU sampling
# ---------------------------------------------------------------------------
@dataclass
class GpuStat:
    idx: int
    util: int          # SM utilization %
    mem_used_mb: int
    mem_total_mb: int
    power_w: float


@dataclass
class GpuProc:
    pid: int
    gpu_idx: int
    used_mem_mb: int


def query_gpus() -> list[GpuStat]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(GpuStat(
                idx=int(parts[0]),
                util=int(parts[1]),
                mem_used_mb=int(float(parts[2])),
                mem_total_mb=int(float(parts[3])),
                power_w=float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else 0.0,
            ))
        except ValueError:
            continue
    return gpus


def query_gpu_procs() -> list[GpuProc]:
    """Map pid -> gpu_idx via nvidia-smi compute-apps."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,gpu_bus_id,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    # Need bus_id -> index mapping
    try:
        idx_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    bus_to_idx = {}
    for line in idx_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                bus_to_idx[parts[1]] = int(parts[0])
            except ValueError:
                pass

    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            gpu_idx = bus_to_idx.get(parts[1], -1)
            mem_mb = int(float(parts[2]))
            procs.append(GpuProc(pid=pid, gpu_idx=gpu_idx, used_mem_mb=mem_mb))
        except ValueError:
            continue
    return procs


# ---------------------------------------------------------------------------
# Worker process sampling
# ---------------------------------------------------------------------------
@dataclass
class WorkerProc:
    pid: int
    cpu_percent: float
    rss_bytes: int
    num_threads: int
    cmdline: str


def _proc_haystack(p: psutil.Process) -> str:
    """Lowercase string of exe+cmdline for token matching. Empty on error."""
    try:
        cmd = " ".join(p.cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ""
    try:
        exe = p.exe() or ""
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        exe = ""
    return f"{exe} {cmd}".lower()


def is_worker(haystack: str, worker_tokens: tuple, exclude_tokens: tuple) -> bool:
    if not haystack:
        return False
    if any(t.lower() in haystack for t in exclude_tokens):
        return False
    return any(t.lower() in haystack for t in worker_tokens)


# Persistent cache of psutil.Process instances across ticks. psutil tracks
# cpu_percent() deltas on the instance, so reusing the same object is REQUIRED
# for accurate CPU readings — creating a new Process(pid) each tick resets the
# baseline to 0. Garbage-collected lazily on NoSuchProcess.
_PROC_CACHE: dict[int, psutil.Process] = {}


def _get_proc(pid: int) -> Optional[psutil.Process]:
    p = _PROC_CACHE.get(pid)
    if p is not None:
        try:
            if p.is_running() and p.pid == pid:
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        _PROC_CACHE.pop(pid, None)
    try:
        p = psutil.Process(pid)
        _PROC_CACHE[pid] = p
        return p
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _sample(p: psutil.Process, haystack: str) -> Optional[WorkerProc]:
    try:
        with p.oneshot():
            return WorkerProc(
                pid=p.pid,
                cpu_percent=p.cpu_percent(interval=None),
                rss_bytes=p.memory_info().rss,
                num_threads=p.num_threads(),
                cmdline=haystack,
            )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def query_workers(pids_on_gpu: set[int],
                  worker_tokens: tuple,
                  exclude_tokens: tuple,
                  script_tokens: tuple) -> dict[int, WorkerProc]:
    """Return pid -> WorkerProc for our workers.

    Seed 1 (authoritative): every pid currently holding GPU memory per
    nvidia-smi, filtered by worker_tokens. This catches all GPU workers
    regardless of invocation style.

    Seed 2 (dispatcher only): scan psutil for processes whose cmdline contains
    a *script* token (not the env name alone — that would catch shells and
    notebooks in the same conda env). Used to find the dispatcher itself when
    it's CPU-only.
    """
    workers: dict[int, WorkerProc] = {}

    for pid in pids_on_gpu:
        p = _get_proc(pid)
        if p is None:
            continue
        haystack = _proc_haystack(p)
        if not is_worker(haystack, worker_tokens, exclude_tokens):
            continue
        w = _sample(p, haystack)
        if w is not None:
            workers[pid] = w

    for p in psutil.process_iter(["pid"]):
        pid = p.info["pid"]
        if pid in workers:
            continue
        cached = _get_proc(pid)
        if cached is None:
            continue
        haystack = _proc_haystack(cached)
        if not haystack:
            continue
        if any(t.lower() in haystack for t in exclude_tokens):
            continue
        if not any(t.lower() in haystack for t in script_tokens):
            continue
        w = _sample(cached, haystack)
        if w is not None:
            workers[pid] = w

    return workers


# ---------------------------------------------------------------------------
# Log state per GPU
# ---------------------------------------------------------------------------
@dataclass
class LogState:
    path: Path
    size: int
    mtime: float
    step: str
    dataset: str
    defense: str
    round: Optional[int]
    gpu: Optional[str]


def scan_growing_logs(logs_dir: Path) -> list[LogState]:
    now = time.time()
    out = []
    for path in logs_dir.glob("*_runner.log"):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if now - st.st_mtime > GROWING_WINDOW_SEC:
            continue
        lines = tail(path)
        state = parse_recent_state(lines, log_filename=path.name)
        out.append(LogState(
            path=path,
            size=st.st_size,
            mtime=st.st_mtime,
            step=state["step"] or "—",
            dataset=state["dataset"] or "—",
            defense=state["defense"] or "—",
            round=state["round"],
            gpu=state["gpu"],
        ))
    return out


# ---------------------------------------------------------------------------
# Cross-tick state (rounds/min, log_bps, hung detection)
# ---------------------------------------------------------------------------
@dataclass
class PrevSample:
    t: float
    round: Optional[int]
    size: int
    last_grow_t: float    # last time size changed
    last_grow_size: int


class Tracker:
    def __init__(self):
        self.prev: dict[str, PrevSample] = {}  # key = log path str

    def update(self, logs: list[LogState]) -> dict[str, tuple[float, float, float]]:
        """Return log_path -> (rounds_per_min, log_bps, seconds_since_grow)."""
        now = time.time()
        out = {}
        for ls in logs:
            key = str(ls.path)
            prev = self.prev.get(key)
            if prev is None:
                self.prev[key] = PrevSample(
                    t=now, round=ls.round, size=ls.size,
                    last_grow_t=now, last_grow_size=ls.size,
                )
                out[key] = (0.0, 0.0, 0.0)
                continue

            dt = max(now - prev.t, 1e-6)
            r_per_min = 0.0
            if prev.round is not None and ls.round is not None and ls.round >= prev.round:
                r_per_min = (ls.round - prev.round) * 60.0 / dt

            log_bps = max(ls.size - prev.size, 0) / dt

            if ls.size > prev.last_grow_size:
                last_grow_t = now
                last_grow_size = ls.size
            else:
                last_grow_t = prev.last_grow_t
                last_grow_size = prev.last_grow_size

            since_grow = now - last_grow_t
            out[key] = (r_per_min, log_bps, since_grow)

            self.prev[key] = PrevSample(
                t=now, round=ls.round, size=ls.size,
                last_grow_t=last_grow_t, last_grow_size=last_grow_size,
            )
        return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def verdict(gpu_util: int, cpu_pct: float, r_per_min: float,
            since_grow: float, has_worker: bool, has_log: bool) -> str:
    if not has_worker:
        return "IDLE"
    if not has_log:
        return "NO-LOG"
    if since_grow > HUNG_AFTER_SEC:
        return "HUNG"
    if gpu_util >= 50:
        return "OK"
    if gpu_util >= 20:
        return "LIGHT"
    # GPU starved. Diagnose what's holding it up.
    # cpu_percent is summed across cores by psutil, so 100 ≈ 1 core pegged,
    # which on a Python/GIL workload IS the CPU bottleneck.
    if cpu_pct >= 80:
        return "CPU-BOUND"
    if cpu_pct < 30:
        return "IO-BOUND?"
    return "LIGHT"


VERDICT_MARK = {
    "OK":        "🟢",
    "LIGHT":     "🟡",
    "CPU-BOUND": "🟠",
    "IO-BOUND?": "🟠",
    "HUNG":      "🔴",
    "IDLE":      "⚫",
    "NO-LOG":    "⚪",
    "OTHER":     "⬛",   # GPU in use by a different user/project
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(logs_dir: Path, tracker: Tracker,
           worker_tokens: tuple, exclude_tokens: tuple, script_tokens: tuple):
    gpus = query_gpus()
    gpu_procs = query_gpu_procs()
    pids_on_gpu = {gp.pid for gp in gpu_procs}
    workers = query_workers(pids_on_gpu, worker_tokens, exclude_tokens, script_tokens)
    logs = scan_growing_logs(logs_dir)
    rates = tracker.update(logs)
    # Shared-log fallback: if a worker GPU has no per-GPU log match (single
    # dispatcher writes one log for all workers), pick the most-recently-grown
    # log so the verdict still reflects the shared dispatcher's progress.
    shared_log: Optional[LogState] = None
    if logs:
        shared_log = max(logs, key=lambda ls: ls.mtime)

    # Classify every GPU-using pid that is NOT ours as "foreign" so we can
    # mark those GPUs as OTHER instead of counting them as our problem.
    foreign_pids_by_gpu: dict[int, int] = {}
    for gp in gpu_procs:
        if gp.pid not in workers and gp.gpu_idx >= 0:
            foreign_pids_by_gpu.setdefault(gp.gpu_idx, gp.pid)

    # Index growing logs by GPU field (from [Run … | GPU N] prefix)
    logs_by_gpu: dict[str, LogState] = {}
    for ls in logs:
        if ls.gpu is not None:
            logs_by_gpu.setdefault(str(ls.gpu), ls)
    # Fallback: unassigned logs, handed out in order to GPUs with a worker
    unassigned = [ls for ls in logs if ls.gpu is None]

    # Build pid -> gpu_idx
    pid_to_gpu = {p.pid: p.gpu_idx for p in gpu_procs}
    # Build gpu -> worker pid (first worker found on that GPU)
    gpu_to_pid: dict[int, int] = {}
    for pid, gpu_idx in pid_to_gpu.items():
        if pid in workers and gpu_idx not in gpu_to_pid:
            gpu_to_pid[gpu_idx] = pid

    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   "
          f"workers={len(workers)}  growing_logs={len(logs)}  gpus={len(gpus)}")
    print()
    header = (f"{'GPU':>3}  {'util':>5}  {'mem':>11}  {'pwr':>5}  "
              f"{'pid':>6}  {'cpu%':>6}  {'rss':>7}  "
              f"{'step':<8}{'dataset':<11}{'defense':<13}{'rnd':>4}  {'r/min':>6}  "
              f"{'logbps':>8}  verdict")
    print(header)
    print("-" * len(header))

    healthy = 0
    total_active = 0
    # Cache per-GPU result so the Problems summary uses the exact same
    # verdict/log that the main table showed (avoids NO-LOG drift when the
    # shared-log fallback kicks in).
    gpu_results: list[tuple[int, str, Optional[WorkerProc], float, float]] = []

    for g in gpus:
        pid = gpu_to_pid.get(g.idx)
        w = workers.get(pid) if pid else None
        ls = logs_by_gpu.get(str(g.idx))
        if ls is None and unassigned and w is not None:
            ls = unassigned.pop(0)
        # Shared dispatcher log fallback
        if ls is None and w is not None and shared_log is not None:
            ls = shared_log

        is_foreign = w is None and g.idx in foreign_pids_by_gpu

        if w is not None:
            total_active += 1

        r_per_min = log_bps = since_grow = 0.0
        if ls is not None:
            r_per_min, log_bps, since_grow = rates.get(str(ls.path), (0.0, 0.0, 0.0))

        if is_foreign:
            v = "OTHER"
        else:
            v = verdict(
                gpu_util=g.util,
                cpu_pct=w.cpu_percent if w else 0.0,
                r_per_min=r_per_min,
                since_grow=since_grow,
                has_worker=w is not None,
                has_log=ls is not None,
            )
        if v == "OK":
            healthy += 1
        gpu_results.append((g.idx, v, w, since_grow, g.util))

        mark = VERDICT_MARK.get(v, " ")
        mem_str = f"{g.mem_used_mb//1024}/{g.mem_total_mb//1024}G"
        pid_str = str(pid) if pid else "—"
        cpu_str = f"{w.cpu_percent:.0f}" if w else "—"
        rss_str = human_size(w.rss_bytes).strip() if w else "—"
        step = ls.step if ls else "—"
        ds = ls.dataset if ls else "—"
        defe = ls.defense if ls else "—"
        rnd = str(ls.round) if (ls and ls.round is not None) else "—"
        rpm = f"{r_per_min:.2f}" if ls else "—"
        bps = f"{log_bps/1024:.1f}K" if ls else "—"

        print(f"{g.idx:>3}  {g.util:>4}%  {mem_str:>11}  "
              f"{g.power_w:>4.0f}W  {pid_str:>6}  {cpu_str:>6}  {rss_str:>7}  "
              f"{step:<8}{ds:<11}{defe:<13}{rnd:>4}  {rpm:>6}  "
              f"{bps:>8}  {mark} {v}")

    print()
    if total_active == 0:
        print("⚠  No worker processes found. Is the dispatcher running?")
    else:
        pct = 100 * healthy / total_active
        tag = "🟢 HEALTHY" if pct >= 80 else ("🟡 DEGRADED" if pct >= 40 else "🔴 UNHEALTHY")
        print(f"{tag}  {healthy}/{total_active} active GPUs reporting OK ({pct:.0f}%)")
        # List any non-OK lanes explicitly so you can't miss them
        problems = []
        for gpu_idx, v, w, sg, util in gpu_results:
            if v in ("OK", "IDLE", "OTHER"):
                continue
            cpu_str = f"{w.cpu_percent:.0f}" if w else "—"
            problems.append(f"  GPU{gpu_idx}: {v}  util={util}%  "
                            f"cpu={cpu_str}%  since_grow={sg:.0f}s")
        if problems:
            print("Problems:")
            for p in problems:
                print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="configs_generated_benchmark",
                    help="Directory containing *_runner.log files")
    ap.add_argument("--watch", action="store_true",
                    help="Re-render every --interval seconds until Ctrl+C")
    ap.add_argument("--interval", type=int, default=10,
                    help="Watch refresh interval in seconds (default: 10)")
    ap.add_argument("--worker-token", action="append", default=[],
                    help="Extra substring to count a process as 'ours' "
                         "(repeatable). Merged with built-in defaults.")
    ap.add_argument("--exclude-token", action="append", default=[],
                    help="Substring that marks a process as foreign "
                         "(repeatable). Merged with built-in defaults.")
    args = ap.parse_args()

    worker_tokens = tuple(DEFAULT_WORKER_TOKENS) + tuple(args.worker_token)
    exclude_tokens = tuple(DEFAULT_EXCLUDE_TOKENS) + tuple(args.exclude_token)
    script_tokens = tuple(DEFAULT_SCRIPT_TOKENS)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"❌ Directory not found: {logs_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    tracker = Tracker()

    # Prime psutil cpu_percent so the first rendered number is real. The
    # _PROC_CACHE persists the Process instances, so the second call will
    # return a real delta.
    pids_on_gpu = {gp.pid for gp in query_gpu_procs()}
    query_workers(pids_on_gpu, worker_tokens, exclude_tokens, script_tokens)
    time.sleep(1.0)

    if not args.watch:
        # Discard the first render (cpu% still zero, no rate history) and
        # only print the second sample, which has real numbers.
        render(logs_dir, tracker, worker_tokens, exclude_tokens, script_tokens)
        time.sleep(args.interval)
        os.system("clear" if os.name != "nt" else "cls")
        render(logs_dir, tracker, worker_tokens, exclude_tokens, script_tokens)
        return

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            render(logs_dir, tracker, worker_tokens, exclude_tokens, script_tokens)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
