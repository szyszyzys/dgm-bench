#!/bin/bash
# =============================================================================
# Sequential runner: Step 10 (headline) first, then Step 8 (scalability).
# =============================================================================
# Runs run_all_step10.sh to completion, then run_step8_system_scalability.sh.
# This avoids the "concurrent kill" bug where one dispatcher's pkill could
# take out the other dispatcher's workers.
#
# Why sequential instead of parallel:
#   - Guaranteed no cross-dispatcher interference
#   - Full GPU bandwidth available to each step (no contention)
#   - Simpler debugging — logs don't interleave two steps
#
# Total wall-clock cost: ~run_all_step10 time + ~run_step8 time.  Roughly
# 6-12 hours depending on how much step 10 was already complete.
#
# Usage:
#   bash run_step10_then_step8.sh
#   GPU_IDS=0,1,2,3,4,5 NUM_PROCS=12 bash run_step10_then_step8.sh
#
# Env vars propagate to BOTH sub-scripts — they use the same GPU_IDS.
# If you want different GPUs per step, launch them directly:
#   GPU_IDS=1,2,3 NUM_PROCS=9 bash run_all_step10.sh
#   GPU_IDS=0,4,5 NUM_PROCS=6 bash run_step8_system_scalability.sh
# =============================================================================

set -u  # fail on unset variables, but NOT on command failures (we want to
        # continue to step 8 even if step 10 had some failed cells)

GPU_IDS="${GPU_IDS:-1,2,3}"
NUM_PROCS="${NUM_PROCS:-9}"
export GPU_IDS NUM_PROCS

mkdir -p logs

START_TS=$(date +%Y%m%d_%H%M)
STEP10_LOG="logs/step10_${START_TS}.log"
STEP8_LOG="logs/step8_${START_TS}.log"

echo ""
echo "=============================================================="
echo " Sequential pipeline: step 10 (headline) -> step 8 (scalability)"
echo "=============================================================="
echo "  GPU_IDS:    $GPU_IDS"
echo "  NUM_PROCS:  $NUM_PROCS"
echo "  Logs:       $STEP10_LOG , $STEP8_LOG"
echo "  Started at: $(date)"
echo ""

# =============================================================================
# PRE-FLIGHT: make sure nothing is already running that would conflict.
# =============================================================================
EXISTING=$(pgrep -f "run_full_benchmark.py\|run_parallel_experiment.py\|run_single_experiment" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "[ABORT] Another dispatcher/worker is already running:"
  ps -o pid,etime,cmd -p $EXISTING 2>/dev/null | head
  echo ""
  echo "  Kill it first (or wait for it), then re-run."
  echo "  To force-continue anyway: SKIP_CONFLICT_CHECK=1 bash $0"
  if [ "${SKIP_CONFLICT_CHECK:-0}" != "1" ]; then
    exit 1
  fi
  echo "  [WARN] SKIP_CONFLICT_CHECK=1 — continuing despite conflict."
fi

# =============================================================================
# PHASE A: Step 10 headline benchmark.
# =============================================================================
echo ""
echo "=============================================================="
echo " PHASE A: Step 10 (headline benchmark) — $(date)"
echo "=============================================================="
bash run_all_step10.sh 2>&1 | tee "$STEP10_LOG"
STEP10_EXIT=${PIPESTATUS[0]}
echo ""
echo "[A] Step 10 exit code: $STEP10_EXIT"

if [ "$STEP10_EXIT" -ne 0 ]; then
  echo "[WARN] Step 10 exited non-zero. Some cells may have failed."
  echo "       Proceeding to step 8 anyway (failures are expected for some defenses)."
fi

# Sanity check: nothing should still be running from step 10.  Wait a bit.
sleep 5
LEFTOVER=$(pgrep -f "run_parallel_experiment.py\|run_single_experiment" 2>/dev/null || true)
if [ -n "$LEFTOVER" ]; then
  echo "[WARN] Still-alive processes after step 10:"
  ps -o pid,etime,cmd -p $LEFTOVER 2>/dev/null | head
  echo "       Waiting 30 s for them to exit naturally..."
  sleep 30
  LEFTOVER=$(pgrep -f "run_parallel_experiment.py\|run_single_experiment" 2>/dev/null || true)
  if [ -n "$LEFTOVER" ]; then
    echo "[ABORT] Step 10 leftovers still alive. Aborting before step 8 to avoid GPU contention:"
    ps -o pid,etime,cmd -p $LEFTOVER 2>/dev/null | head
    echo "       Kill them manually, then re-run bash run_step8_system_scalability.sh."
    exit 2
  fi
fi

# =============================================================================
# PHASE B: Step 8 system scalability.
# =============================================================================
echo ""
echo "=============================================================="
echo " PHASE B: Step 8 (system scalability) — $(date)"
echo "=============================================================="
bash run_step8_system_scalability.sh 2>&1 | tee "$STEP8_LOG"
STEP8_EXIT=${PIPESTATUS[0]}
echo ""
echo "[B] Step 8 exit code: $STEP8_EXIT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================================="
echo " Sequential pipeline COMPLETE"
echo "=============================================================="
SUCC10=$(find results/step12_main_summary_* -name ".success" 2>/dev/null | wc -l)
FAIL10=$(find results/step12_main_summary_* -name ".failed" 2>/dev/null | wc -l)
SUCC8=$(find results/step10_scalability_* -name ".success" 2>/dev/null | wc -l)
FAIL8=$(find results/step10_scalability_* -name ".failed" 2>/dev/null | wc -l)
echo "  Step 10 (headline)    : $SUCC10 success, $FAIL10 failed"
echo "  Step 8  (scalability) : $SUCC8 success, $FAIL8 failed"
echo ""
echo "  Started: $START_TS  Finished: $(date +%Y%m%d_%H%M)"
echo ""
echo "  Logs:   $STEP10_LOG"
echo "          $STEP8_LOG"
echo ""
echo "  Extract tables:"
echo "    python check_step10_results.py"
echo "    python extract_system_perf.py --steps 10 8 --output tables/system_perf.csv"
echo "    python plot_system_perf.py"
echo ""

exit $STEP8_EXIT
