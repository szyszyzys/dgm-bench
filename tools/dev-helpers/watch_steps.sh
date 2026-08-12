#!/bin/bash
# =============================================================================
# Multi-step watchdog
# =============================================================================
# Monitors one or more benchmark steps simultaneously. Prints a heartbeat per
# step every poll interval, detects stalls, and auto-exits when all dispatchers
# are gone.
#
# Usage:
#   bash watch_steps.sh                         # watches step 3 + step 10
#   STEPS="10" bash watch_steps.sh              # step 10 only
#   STEPS="3 10 21" bash watch_steps.sh         # three steps
#   STALL_THRESHOLD=1800 bash watch_steps.sh    # custom stall (30 min)
#
# Each step needs two things configured in the arrays below:
#   RESULT_PREFIX  — glob prefix under ./results/ for .success/.failed markers
#   LOG_FILE       — the runner log file path
#   PGREP_PATTERN  — pattern to detect the dispatcher process
# =============================================================================

STEPS="${STEPS:-3 10}"
STALL_THRESHOLD="${STALL_THRESHOLD:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOGS_DIR="${LOGS_DIR:-configs_generated_benchmark}"

# ---- Step registry ----
# Add new steps here. The key is the step number.
declare -A RESULT_PREFIX LOG_FILE PGREP_PATTERN STEP_LABEL

# Step 1
RESULT_PREFIX[1]="step1_tune_"
LOG_FILE[1]="$LOGS_DIR/step1_iid_tuning_runner.log"
PGREP_PATTERN[1]="run_full_benchmark.py --step 1"
STEP_LABEL[1]="Step 1  (HP tuning)"

# Step 3
RESULT_PREFIX[3]="step3_tune_"
LOG_FILE[3]="$LOGS_DIR/step3_defense_tuning_runner.log"
PGREP_PATTERN[3]="run_full_benchmark.py --step 3"
STEP_LABEL[3]="Step 3  (defense tuning)"

# Step 10
RESULT_PREFIX[10]="step12_main_summary_"
LOG_FILE[10]="$LOGS_DIR/step12_main_summary_runner.log"
PGREP_PATTERN[10]="run_full_benchmark.py --step 10"
STEP_LABEL[10]="Step 10 (main summary)"

# Step 21
RESULT_PREFIX[21]="step21_valuation_"
LOG_FILE[21]="$LOGS_DIR/step21_valuation_analysis_runner.log"
PGREP_PATTERN[21]="run_full_benchmark.py --step 21"
STEP_LABEL[21]="Step 21 (valuation)"

# ---- State arrays (keyed by step number) ----
declare -A LAST_SUCC LAST_CHANGE START_SUCC

# ---- Init ----
now=$(date +%s)
echo "=============================================="
echo " Multi-step watchdog"
echo "=============================================="
echo "  Started at      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Watching steps  : $STEPS"
echo "  Stall threshold : ${STALL_THRESHOLD}s ($(( STALL_THRESHOLD / 60 )) min)"
echo "  Poll interval   : ${CHECK_INTERVAL}s"
echo "  Results dir     : $RESULTS_DIR"
echo ""

# Header
printf '%-22s %6s %5s %6s %7s %8s %10s  %s\n' \
    "step" "succ" "fail" "inprog" "log_age" "log_MB" "idle" "status"
printf '%s\n' "$(printf '%.0s-' {1..90})"

# Initialize per-step state
for s in $STEPS; do
    if [ -z "${RESULT_PREFIX[$s]}" ]; then
        echo "  [WARN] Step $s not in registry — skipping."
        continue
    fi
    cnt=$(find "$RESULTS_DIR"/${RESULT_PREFIX[$s]}* -name ".success" 2>/dev/null | wc -l)
    LAST_SUCC[$s]=$cnt
    LAST_CHANGE[$s]=$now
    START_SUCC[$s]=$cnt
done
start_time=$now

echo ""
echo "Press Ctrl-C to stop."
echo ""

# ---- Main loop ----
while true; do
    now=$(date +%s)
    any_alive=0

    for s in $STEPS; do
        [ -z "${RESULT_PREFIX[$s]}" ] && continue

        prefix="${RESULT_PREFIX[$s]}"
        log="${LOG_FILE[$s]}"
        label="${STEP_LABEL[$s]:-Step $s}"

        succ=$(find "$RESULTS_DIR"/${prefix}* -name ".success"     2>/dev/null | wc -l)
        fail=$(find "$RESULTS_DIR"/${prefix}* -name ".failed"      2>/dev/null | wc -l)
        inpr=$(find "$RESULTS_DIR"/${prefix}* -name ".in_progress" 2>/dev/null | wc -l)

        # Log freshness
        if [ -f "$log" ]; then
            log_age=$((now - $(stat -c %Y "$log" 2>/dev/null || echo "$now")))
            log_size_mb=$(stat -c %s "$log" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
        else
            log_age="—"
            log_size_mb="—"
        fi

        # Success tracking
        if [ "$succ" -gt "${LAST_SUCC[$s]}" ]; then
            delta=$((succ - ${LAST_SUCC[$s]}))
            gap=$((now - ${LAST_CHANGE[$s]}))
            LAST_SUCC[$s]=$succ
            LAST_CHANGE[$s]=$now
            status="$(printf '  +%d (gap %ds)' "$delta" "$gap")"
            mark="+"
        else
            idle=$((now - ${LAST_CHANGE[$s]}))
            idle_min=$(awk -v i=$idle 'BEGIN{printf "%.1f", i/60}')
            if [ "$idle" -ge "$STALL_THRESHOLD" ]; then
                status="STALL ${idle_min}min"
                mark="!"
            else
                status="idle ${idle_min}min"
                mark="."
            fi
        fi

        # Dispatcher alive?
        alive="  "
        if pgrep -f "${PGREP_PATTERN[$s]}" > /dev/null 2>&1; then
            any_alive=1
            alive="R"
        else
            alive="x"
        fi

        printf '%s [%s] %-22s %5s %5s %6s %6ss %7sMB  %s\n' \
            "$(date +%H:%M:%S)" "$alive" "$label" \
            "$succ" "$fail" "$inpr" \
            "$log_age" "$log_size_mb" "$status"
    done

    echo ""

    # Exit if ALL dispatchers are dead
    if [ "$any_alive" -eq 0 ]; then
        echo "$(date +%H:%M:%S)  All dispatchers exited. Watchdog done."
        elapsed=$((now - start_time))
        elapsed_min=$(awk -v e=$elapsed 'BEGIN{printf "%.1f", e/60}')
        echo "  Watched for: ${elapsed}s = ${elapsed_min} min"
        for s in $STEPS; do
            [ -z "${RESULT_PREFIX[$s]}" ] && continue
            gained=$((${LAST_SUCC[$s]} - ${START_SUCC[$s]}))
            echo "  ${STEP_LABEL[$s]:-Step $s}: +${gained} successes (final: ${LAST_SUCC[$s]})"
        done
        break
    fi

    sleep "$CHECK_INTERVAL"
done
