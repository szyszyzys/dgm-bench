#!/bin/bash
# =============================================================================
# Step 10 watchdog
# =============================================================================
# Background monitor for the step 10 rerun. Prints a heartbeat every minute,
# logs every new .success marker as it appears (with the gap since the previous
# success), and shouts STALL if no new success appears within STALL_THRESHOLD.
#
# Usage:
#   bash watch_step10.sh                       # default stall = 15 min
#   STALL_THRESHOLD=600 bash watch_step10.sh   # custom stall threshold (10 min)
#
# Exits cleanly when the step 10 dispatcher process is no longer running.
# =============================================================================

STALL_THRESHOLD="${STALL_THRESHOLD:-900}"   # alert if no new success in 15 min
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"      # poll every minute
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOG_PATH="${LOG_PATH:-configs_generated_benchmark/step12_main_summary_runner.log}"

last_succ=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".success" 2>/dev/null | wc -l)
last_change=$(date +%s)
start_succ=$last_succ
start_time=$last_change

echo "=============================================="
echo " Step 10 watchdog"
echo "=============================================="
echo "  Started at      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Initial succ    : $last_succ"
echo "  Stall threshold : ${STALL_THRESHOLD}s ($(( STALL_THRESHOLD / 60 )) min)"
echo "  Poll interval   : ${CHECK_INTERVAL}s"
echo "  Results dir     : $RESULTS_DIR"
echo "  Runner log      : $LOG_PATH"
echo ""
echo "Press Ctrl-C to stop. Watchdog auto-exits when the step 10 process dies."
echo ""

while true; do
    succ=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".success"     2>/dev/null | wc -l)
    fail=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".failed"      2>/dev/null | wc -l)
    inpr=$(find "$RESULTS_DIR"/step12_main_summary_* -name ".in_progress" 2>/dev/null | wc -l)
    now=$(date +%s)

    if [ -f "$LOG_PATH" ]; then
        log_age=$((now - $(stat -c %Y "$LOG_PATH" 2>/dev/null || echo "$now")))
        log_size_mb=$(stat -c %s "$LOG_PATH" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
    else
        log_age="?"
        log_size_mb="?"
    fi

    if [ "$succ" -gt "$last_succ" ]; then
        delta=$((succ - last_succ))
        gap=$((now - last_change))
        gap_min=$(awk -v g=$gap 'BEGIN{printf "%.1f", g/60}')
        printf '%s  ✓ +%d  succ=%-3s fail=%-2s inprog=%-2s log_age=%4ss log=%sMB  (gap=%ds=%smin)\n' \
            "$(date +%H:%M:%S)" "$delta" "$succ" "$fail" "$inpr" \
            "$log_age" "$log_size_mb" "$gap" "$gap_min"
        last_succ=$succ
        last_change=$now
    else
        idle=$((now - last_change))
        idle_min=$(awk -v i=$idle 'BEGIN{printf "%.1f", i/60}')
        if [ "$idle" -ge "$STALL_THRESHOLD" ]; then
            printf '%s  ⚠ STALL  succ=%-3s fail=%-2s inprog=%-2s log_age=%4ss log=%sMB  no new success in %smin\n' \
                "$(date +%H:%M:%S)" "$succ" "$fail" "$inpr" \
                "$log_age" "$log_size_mb" "$idle_min"
        else
            printf '%s  …       succ=%-3s fail=%-2s inprog=%-2s log_age=%4ss log=%sMB  idle %smin\n' \
                "$(date +%H:%M:%S)" "$succ" "$fail" "$inpr" \
                "$log_age" "$log_size_mb" "$idle_min"
        fi
    fi

    # Exit if the dispatcher is no longer alive
    if ! pgrep -f "run_full_benchmark.py --step 10" > /dev/null; then
        echo ""
        echo "$(date +%H:%M:%S)  ⚠ step 10 dispatcher is no longer running. Exiting watchdog."
        elapsed=$((now - start_time))
        gained=$((succ - start_succ))
        elapsed_min=$(awk -v e=$elapsed 'BEGIN{printf "%.1f", e/60}')
        echo "  Watched for       : ${elapsed}s = ${elapsed_min} min"
        echo "  New successes     : $gained"
        if [ "$gained" -gt 0 ]; then
            avg_gap=$((elapsed / gained))
            echo "  Avg gap/success   : ${avg_gap}s"
        fi
        echo "  Final succ count  : $succ"
        echo "  Final fail count  : $fail"
        break
    fi

    sleep "$CHECK_INTERVAL"
done
