#!/bin/bash
# =============================================================================
# Step 10 Runner — stops any running step-10 dispatchers, clears stale FEMNIST
# step 10 markers, and runs step 10 with the new FEMNIST-friendly trigger.
# =============================================================================
# Workflow:
#   Phase 1: Kill any running step-10-specific dispatchers ONLY.
#            Does NOT touch step 8 (scalability) or other step dispatchers —
#            identified by matching the step 10 config-dir name in their
#            command line rather than by process name alone.
#   Phase 2: Delete stale .in_progress markers for step 10 scenarios.
#   Phase 3: Generate step 10 configs and run all (FEMNIST + non-FEMNIST).
#            Non-FEMNIST .success markers are preserved -> those cells skip.
#            FEMNIST cells re-run from scratch with the new trigger.
#
# Background: the previous BLENDED_PATCH/alpha=0.2 trigger was a white patch
# that vanished against FEMNIST's white background, dropping ASR to ~5%
# (random baseline). Defaults are now CHECKERBOARD/alpha=0.8/4x4, visible
# on both FEMNIST and CIFAR.
#
# Safe to run CONCURRENTLY with run_step8_system_scalability.sh as long as
# GPU sets don't overlap. Step 8 dispatcher has 'step10_scalability' in its
# command line; this script's kill logic ignores those by matching only on
# 'step12_main_summary'.
# =============================================================================

RESULTS_DIR="./results"
CONFIGS_DIR="./configs_generated_benchmark"
# Defaults: step 10 on GPUs 1,2,3 (leave 0, 4, 5 free for step 8 if needed).
GPU_IDS="${GPU_IDS:-1,2,3}"
NUM_PROCS="${NUM_PROCS:-9}"

# Tag strings used to disambiguate step-10 processes from other dispatchers.
# Any run_parallel_experiment.py or run_single_experiment whose command line
# contains ALL of these substrings is considered "our step 10" and eligible
# to be killed. Step 8 workers have step10_scalability in their path instead.
STEP10_CONFIG_TAG="step12_main_summary"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "[Setup] FD limit: $(ulimit -n)"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# --- Helper: kill ONLY step-10-specific dispatchers / workers. -----------
#   Matches process cmdlines containing the STEP10_CONFIG_TAG so we don't
#   accidentally kill a concurrently-running step 8 (scalability) dispatcher
#   that uses the same run_parallel_experiment.py binary but a different
#   configs_dir.
kill_step10_processes() {
  local LABEL="$1"
  local PATTERN="$2"
  local PIDS
  PIDS=$(pgrep -f "$PATTERN" 2>/dev/null | while read -r pid; do
    # Only keep pids whose full cmdline references step 12 main summary
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$STEP10_CONFIG_TAG"; then
      echo "$pid"
    fi
  done)
  if [ -n "$PIDS" ]; then
    echo "[Kill] $LABEL (step-10 only): $PIDS"
    kill $PIDS 2>/dev/null
  else
    echo "[Kill] $LABEL (step-10 only): none"
  fi
}

# =============================================================================
echo ""
echo "============================================================"
echo " Phase 1: Stop any step-10 dispatchers (step 8 is left alone)"
echo "============================================================"

# (1) run_full_benchmark.py --step 10 dispatchers.  The '--step 10' is
# unique enough that we can match on it directly.
RFB_PIDS=$(pgrep -f "run_full_benchmark.py --step 10" 2>/dev/null || true)
if [ -n "$RFB_PIDS" ]; then
  echo "[Kill] run_full_benchmark.py --step 10: $RFB_PIDS"
  kill $RFB_PIDS 2>/dev/null
else
  echo "[Kill] run_full_benchmark.py --step 10: none"
fi

sleep 2

# (2) run_parallel_experiment.py dispatchers that target step 10 configs.
kill_step10_processes "run_parallel_experiment.py" "run_parallel_experiment.py"

sleep 2

# (3) run_single_experiment worker children running a step-10 config.
kill_step10_processes "run_single_experiment" "run_single_experiment"

sleep 3

# (4) Force-kill any step-10 stragglers that ignored SIGTERM.
STILL_ALIVE=$(
  for pid in $(pgrep -f "run_full_benchmark.py --step 10\|run_parallel_experiment.py\|run_single_experiment" 2>/dev/null); do
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE "run_full_benchmark.py --step 10|$STEP10_CONFIG_TAG"; then
      echo "$pid"
    fi
  done
)
if [ -n "$STILL_ALIVE" ]; then
  echo "[Kill] Force-killing step-10 stragglers: $STILL_ALIVE"
  kill -9 $STILL_ALIVE 2>/dev/null
fi

echo "[Phase 1] Done (step 8 / other dispatchers untouched)."


# =============================================================================
echo ""
echo "============================================================"
echo " Phase 2: Clear stale step-10 .in_progress markers"
echo "============================================================"
find "$RESULTS_DIR"/step12_main_summary_* -name ".in_progress" -delete 2>/dev/null || true
echo "[Phase 2] Done."

# =============================================================================
echo ""
echo "============================================================"
echo " Phase 3: Run step 10 (FEMNIST with new trigger + others)"
echo "============================================================"

# Regenerate step 10 configs from scratch
rm -rf "$CONFIGS_DIR/step12_main_summary" 2>/dev/null
python experiments/gradient_market/run_full_benchmark.py --step 10 --generate_only

GEN_COUNT=$(ls -d "$CONFIGS_DIR"/step12_main_summary/*/ 2>/dev/null | wc -l)
echo "[Phase 3] Generated $GEN_COUNT config dir(s)."
echo "[Phase 3] Existing non-FEMNIST .success markers will skip those cells."
echo "[Phase 3] FEMNIST cells will re-run with CHECKERBOARD/alpha=0.8/4x4 trigger."

if [ "$GEN_COUNT" -gt 0 ]; then
  python experiments/gradient_market/run_parallel_experiment.py \
    --configs_dir "$CONFIGS_DIR/step12_main_summary" \
    --gpu_ids $GPU_IDS --num_processes $NUM_PROCS
  echo "[Phase 3] Done."
else
  echo "[Phase 3] ERROR: No configs generated. Aborting."
  exit 1
fi

# =============================================================================
echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
SUCC10=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".success" 2>/dev/null | wc -l)
FAIL10=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".failed" 2>/dev/null | wc -l)
INPR10=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".in_progress" 2>/dev/null | wc -l)
SUCC10_FEM=$(find "$RESULTS_DIR"/step12_main_summary_*FEMNIST* -name ".success" 2>/dev/null | wc -l)
echo "  Step 10 totals: success=$SUCC10  failed=$FAIL10  in_progress=$INPR10"
echo "  Step 10 FEMNIST success: $SUCC10_FEM"
echo ""
echo "Done. Verify with: bash verify_all.sh"
echo "Spot-check FEMNIST results with: python check_femnist_step3.py  (or step10 equivalent)"
