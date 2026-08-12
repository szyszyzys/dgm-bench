#!/bin/bash
# =============================================================================
# Post-rerun Verification Script
# =============================================================================
# Run this after rerun_priority.sh completes to check experiment health.
# Reports per-step health (success/failed/incomplete/zombie counts, top
# failure reasons, per-defense breakdown), with all dropped/legacy noise
# filtered out so the output reflects only the experiments you are currently
# running.
#
# Aligned with rerun_priority.sh as of cleanup branch.
# =============================================================================

RESULTS_DIR="./results"
PARAMS_DIR="./experiments/gradient_market/configs_generation/tuned_params"

# =============================================================================
# What is INCLUDED in this verification (matches rerun_priority.sh exactly).
# =============================================================================
# CLI step → human label → scenario name prefix(es), space-separated.
# These prefixes are matched against top-level dirs under $RESULTS_DIR.
# Order matters: this is the order rerun_priority.sh executes them in.
#
# Steps 16 (dp_fairness) and 19 (new_defenses) are intentionally absent from
# STEPS_ORDER because they are commented out in rerun_priority.sh:
#   - Step 16: tangential to the marketplace thesis (dropped from current paper)
#   - Step 19: subsumed by Step 10 after FLAME/DeepSight/Bulyan/FoolsGold were
#              merged into IMAGE_DEFENSES / TEXT_TABULAR_DEFENSES
# Their LABEL/PREFIX entries are kept below as a registry so re-enabling them
# is a one-line edit (just add the step number back to STEPS_ORDER).
STEPS_ORDER=(3 10 4 5 6 7 9 14 8 13 15 17)
DISABLED_STEPS=(16 19)
declare -A STEP_LABEL=(
  [3]="defense_tune"        [10]="main_summary"
  [4]="attack_sens"         [5]="sybil"
  [6]="adaptive"            [7]="buyer"
  [9]="heterogeneity"       [14]="collusion"
  [8]="scalability"         [13]="drowning"
  [15]="pricing"            [16]="dp_fairness"
  [17]="alie"               [19]="new_defenses"
)
declare -A STEP_PREFIXES=(
  [3]="step3_tune_"
  [10]="step12_main_summary_"
  [4]="step5_atk_sens_"
  [5]="step6_adv_sybil_"
  [6]="step7_adaptive_ step7_baseline_no_attack"
  [7]="step8_buyer_attack_"
  # Step 9 prefix list MUST stay in sync with IMAGE_DEFENSES in
  # config_common_utils.py. The 4 new defenses (flame/deepsight/bulyan/
  # foolsgold) were promoted into IMAGE_DEFENSES, so step 9 now produces
  # per-defense scenario dirs for them too. Without the new prefixes here,
  # verify_all.sh would silently under-count step 9's progress.
  [9]="step11_fedavg_ step11_fltrust_ step11_martfl_ step11_skymask_ step11_trimmed_mean_ step11_multi_krum_ step11_rflpa_ step11_spmc_ step11_flame_ step11_deepsight_ step11_bulyan_ step11_foolsgold_"
  [14]="step14_collusion_"
  [8]="step10_scalability_"
  [13]="step13_drowning_"
  [15]="step15_pricing_"
  [16]="step16_dp_"
  [17]="step17_alie_"
  [19]="step19_new_defenses"
)

# =============================================================================
# What is EXCLUDED globally (intentionally not part of the current rerun).
# =============================================================================
# These appear nowhere in any section of this report.
#
#   - Step 1  (iid_tuning)        — cached, golden_training_params.json reused
#   - Step 11 (free_riding)       — deprioritized
#   - Step 18 (daved_comparison)  — DAVED removed from benchmark
#   - Step 20 (sleeper_agent)     — deprioritized
#   - DAVED defense itself        — removed from ENABLED_DEFENSES
#   - Old backup dirs             — step3_bk, step7_backup, step9_comp_, etc.
EXCLUDED_DEFENSES_REGEX='daved'
SKIPPED_STEP_PATTERN='step3_bk|step7_backup|step9_comp_|step11_freerider_|step11_free_riding|step18_|step20_|step5b_sleeper_'

# Datasets hidden from verification output. Data on disk is untouched —
# this is display-only. FEMNIST and Purchase100 are temporarily disabled
# in config_common_utils.py:ENABLED_DATASETS and we don't want their stale
# results from older runs cluttering the per-step health view.
# Matches both cell-dir tags (ds-femnist) and scenario-name tokens
# (_FEMNIST_). Edit or clear this regex when re-enabling.
EXCLUDED_DATASET_REGEX='(ds-femnist|ds-purchase100|_FEMNIST_|_Purchase100_|_femnist_|_purchase100_)'

# Legacy filter — a dir is legacy ONLY if it has a lowercase dataset token
# AND no proper-case dataset token anywhere in its name. This avoids the bug
# where step3_tune_*_image_CIFAR100_cifar100_cnn was wrongly dropped because
# the model_config "cifar100_cnn" is always lowercase.
LEGACY_LOWER='_(cifar100|cifar10|femnist|texas100|purchase100|trec)_'
CURRENT_PROPER='(CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)'

# Reusable pipe filter that drops:
#   - legacy-only paths
#   - skipped/deprecated step paths
#   - paths containing the excluded DAVED defense
#   - paths containing the excluded FEMNIST / Purchase100 datasets
filter_excluded() {
  awk -v lower="$LEGACY_LOWER" \
      -v proper="$CURRENT_PROPER" \
      -v skipped="$SKIPPED_STEP_PATTERN" \
      -v excluded_def="$EXCLUDED_DEFENSES_REGEX" \
      -v excluded_ds="$EXCLUDED_DATASET_REGEX" '
    {
      if ($0 ~ lower && $0 !~ proper) next
      if ($0 ~ skipped) next
      if ($0 ~ excluded_def) next
      if (excluded_ds != "" && $0 ~ excluded_ds) next
      print
    }'
}

# =============================================================================
# 1. Parameter file check (cached step 1 + freshly tuned step 3 outputs)
# =============================================================================
echo "=============================================="
echo " 1. PARAMETER FILES (cached Step 1 + freshly-tuned Step 3)"
echo "=============================================="
for f in golden_training_params.json tuned_defense_params.json; do
  if [ -f "$PARAMS_DIR/$f" ]; then
    size=$(wc -c < "$PARAMS_DIR/$f")
    printf "  [OK]      %-32s  (%s bytes)\n" "$f" "$size"
  else
    printf "  [MISSING] %s\n" "$f"
  fi
done

# =============================================================================
# 2. Per-step deep analysis
# =============================================================================
# For each step in CURRENT_RERUN order, walk its scenario dirs and report:
#   - scenarios:   number of top-level scenario dirs that match this step
#   - runs:        number of leaf run dirs found (one per seed × HP cell)
#   - success:     leaf dirs containing .success
#   - failed:      leaf dirs containing .failed
#   - in_progress: leaf dirs containing .in_progress (active or zombie)
#   - missing:     scenarios - success - failed - in_progress (incomplete leaves)
#   - top defense rollup (success counts) within this step
#   - top failure reason (truncated) within this step
# =============================================================================
echo ""
echo "=============================================="
echo " 2. PER-STEP HEALTH"
echo "=============================================="
echo ""
echo "  Legend: ✓ all OK · ! some failures · ? incomplete · ∅ no results yet"
echo ""

# Report disabled steps once, up front, so they don't silently disappear from
# the verification output. They are NOT counted in the grand total or the
# per-step health loop below.
if [ ${#DISABLED_STEPS[@]} -gt 0 ]; then
  echo "  Disabled (intentionally not run by current rerun_priority.sh):"
  for d in "${DISABLED_STEPS[@]}"; do
    printf "    — Step %-3s %s\n" "$d" "${STEP_LABEL[$d]}"
  done
  echo ""
fi

# Track grand totals for the final summary
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_INPROG=0
TOTAL_SCENARIOS=0
PROBLEM_STEPS=()

for step in "${STEPS_ORDER[@]}"; do
  label="${STEP_LABEL[$step]}"
  prefixes="${STEP_PREFIXES[$step]}"

  # Collect all scenario dirs across (possibly multiple) prefixes
  scenarios=""
  for p in $prefixes; do
    found=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "${p}*" 2>/dev/null \
            | filter_excluded)
    if [ -n "$found" ]; then
      scenarios+="${found}"$'\n'
    fi
  done
  scenarios=$(echo "$scenarios" | sed '/^$/d')   # drop blank lines

  if [ -z "$scenarios" ]; then
    printf "  ∅ Step %-3s %-15s  no results\n" "$step" "$label"
    continue
  fi

  scenario_count=$(echo "$scenarios" | wc -l)

  # Collect markers across these scenarios in one find pass
  marker_list=$(echo "$scenarios" | xargs -I{} find {} \
                  \( -name ".success" -o -name ".failed" -o -name ".in_progress" \) \
                  2>/dev/null | filter_excluded)

  success_count=$(echo "$marker_list" | grep -c "\.success$" || true)
  failed_count=$(echo "$marker_list" | grep -c "\.failed$" || true)
  inprog_count=$(echo "$marker_list" | grep -c "\.in_progress$" || true)
  total_runs=$((success_count + failed_count + inprog_count))

  # Pick a status icon
  if [ "$failed_count" -gt 0 ]; then
    icon="!"
    PROBLEM_STEPS+=("$step")
  elif [ "$inprog_count" -gt 0 ] || [ "$success_count" -eq 0 ]; then
    icon="?"
    PROBLEM_STEPS+=("$step")
  else
    icon="✓"
  fi

  TOTAL_SUCCESS=$((TOTAL_SUCCESS + success_count))
  TOTAL_FAILED=$((TOTAL_FAILED + failed_count))
  TOTAL_INPROG=$((TOTAL_INPROG + inprog_count))
  TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + scenario_count))

  printf "  %s Step %-3s %-15s  scenarios=%-3s  runs=%-4s  ok=%-4s  fail=%-3s  in_progress=%-3s\n" \
    "$icon" "$step" "$label" "$scenario_count" "$total_runs" "$success_count" "$failed_count" "$inprog_count"

  # Per-defense rollup of successes within this step (top 8, daved excluded).
  # Use a whitelist so we don't accidentally include HP suffixes (e.g.
  # agg-martfl_k-3 → "martfl", not "martfl_k"). Longer names listed first
  # so skymask_small matches before skymask.
  if [ "$success_count" -gt 0 ]; then
    defense_rollup=$(echo "$marker_list" | grep "\.success$" | \
      grep -oE 'agg-(skymask_small|trimmed_mean|multi_krum|fedavg|fltrust|martfl|skymask|rflpa|spmc|flame|deepsight|bulyan|foolsgold)' | \
      sed 's/^agg-//' | \
      sort | uniq -c | sort -rn | head -8 | \
      awk '{printf "%s(%d) ", $2, $1}')
    if [ -n "$defense_rollup" ]; then
      printf "      defenses ok: %s\n" "$defense_rollup"
    fi
  fi

  # Top failure reason within this step (truncated to 80 chars)
  if [ "$failed_count" -gt 0 ]; then
    top_reason=$(echo "$marker_list" | grep "\.failed$" | \
      xargs -I{} cat {} 2>/dev/null | \
      grep -v "^$" | sort | uniq -c | sort -rn | head -1 | \
      sed 's/^ *[0-9]* //' | cut -c1-80)
    if [ -n "$top_reason" ]; then
      printf "      top failure: %s\n" "$top_reason"
    fi
  fi
done

# =============================================================================
# 3. Aggregate failure & zombie report (current pipeline only)
# =============================================================================
echo ""
echo "=============================================="
echo " 3. FAILURE REASONS (across current steps)"
echo "=============================================="
all_failures=$(find $RESULTS_DIR -name ".failed" 2>/dev/null | filter_excluded)
fail_count=$(echo "$all_failures" | sed '/^$/d' | wc -l)

if [ "$fail_count" -eq 0 ]; then
  echo "  No failures detected."
else
  echo "  Found $fail_count failed runs. Unique failure reasons (top 10):"
  echo "$all_failures" | xargs -I{} cat {} 2>/dev/null | \
    grep -v "^$" | sort | uniq -c | sort -rn | head -10 | \
    awk '{count=$1; $1=""; sub(/^ /,""); printf "    [%4d] %s\n", count, substr($0, 1, 100)}'
  echo ""
  echo "  Failed run paths (first 10):"
  echo "$all_failures" | head -10 | sed 's|^|    |'
fi

echo ""
echo "=============================================="
echo " 4. ZOMBIE RUNS (.in_progress with no .success)"
echo "=============================================="
zombie_list=""
zombie_count=0
while IFS= read -r marker; do
  [ -z "$marker" ] && continue
  dir=$(dirname "$marker")
  if [ ! -f "$dir/.success" ]; then
    zombie_count=$((zombie_count + 1))
    zombie_list+="$dir"$'\n'
  fi
done < <(find $RESULTS_DIR -name ".in_progress" 2>/dev/null | filter_excluded)

if [ "$zombie_count" -eq 0 ]; then
  echo "  No zombie runs detected."
else
  echo "  Found $zombie_count zombie run(s). Distribution by step:"
  # Extract the top-level scenario dir name and strip the dataset suffix.
  # Use awk to avoid sed delimiter conflicts with the | alternation regex.
  echo "$zombie_list" | awk -F'/' '
    {
      # field 3 is the scenario dir under ./results/
      name = $3
      # strip everything from the first proper-case dataset marker onward
      sub(/_(CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC).*/, "", name)
      print name
    }' | sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "    [%4d] %s\n", $1, $2}'
  echo ""
  echo "  First 5 zombie paths:"
  echo "$zombie_list" | head -5 | sed 's|^|    |'
fi

# =============================================================================
# 5. Grand total
# =============================================================================
echo ""
echo "=============================================="
echo " GRAND TOTAL (current rerun scope only)"
echo "=============================================="
printf "  Scenarios:    %d\n" "$TOTAL_SCENARIOS"
printf "  Runs OK:      %d\n" "$TOTAL_SUCCESS"
printf "  Runs failed:  %d\n" "$TOTAL_FAILED"
printf "  In progress:  %d\n" "$TOTAL_INPROG"

if [ ${#PROBLEM_STEPS[@]} -eq 0 ] && [ "$TOTAL_FAILED" -eq 0 ]; then
  echo ""
  echo "  ✅ ALL CURRENT STEPS HEALTHY"
else
  echo ""
  echo "  ⚠️  Steps needing attention: ${PROBLEM_STEPS[*]}"
  echo ""
  echo "  Drill down with:"
  echo "    python experiments/gradient_market/check_results.py \\"
  echo "      --results_dir $RESULTS_DIR --step <N> --verbose"
fi

echo ""
echo "=============================================="
echo " VERIFICATION COMPLETE"
echo "=============================================="
