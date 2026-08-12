#!/bin/bash
# =============================================================================
# Main-Conclusion All-Datasets Runner — PRIORITY-RANKED, SINGLE-SHOT
# =============================================================================
# Runs every step needed for the paper, on ALL FIVE datasets where each step
# is intended to span them (CIFAR-100, Texas-100, Purchase-100, FEMNIST, TREC),
# grouped by priority tier so the most paper-critical work runs first.
#
# This is the canonical "do everything I need for the paper" script. After it
# finishes, you should have:
#
#   - Tuned training HPs    (golden_training_params.json)
#   - Tuned defense HPs     (tuned_defense_params.json)
#   - Headline benchmark    (Step 10, all 5 datasets × 11 defenses)
#   - In-depth analyses     (Steps 4, 5, 7, 8, 9 with filtering defenses)
#   - Threat-model attacks  (Steps 6, 13, 14, 17 with their scoped defenses)
#   - Pricing analysis      (Step 15, post-hoc replay)
#
# Step 2 ("find usable training HPs") is INTENTIONALLY EXCLUDED — it produces
# only an appendix tunability heatmap, costs ~2-4 days of GPU, and is not
# consumed by any other step. If a reviewer specifically asks for it, run
# rerun_step2.sh separately.
#
# PRIORITY TIERS
# --------------
# P0  PAPER-CRITICAL — must run; gates the headline table
#     Step 1  — Federated training HP tuning (writes golden_training_params.json)
#     Step 3  — Defense HP tuning            (writes tuned_defense_params.json)
#     Step 10 — Main summary, headline accuracy / ASR / BSR / MSR table
#
# P1  IN-DEPTH ANALYSIS — supports the paper's RQ figures
#     Step 4  — Attack sensitivity sweep (adv_rate × poison_rate)
#     Step 5  — Sybil strategies
#     Step 7  — Buyer-side attacks (demand-side bias)
#     Step 8  — Marketplace scalability (n_sellers sweep)
#     Step 9  — Data heterogeneity (Dirichlet α sweep)
#     All five use FILTERING_DEFENSES scope (~9 defenses) to support the
#     "filtering matters" narrative, not just FOCUSED_DEFENSES (fltrust + martfl).
#
# P2  THREAT-MODEL-SCOPED ATTACKS — narrow defense scope by design
#     Step 6  — Adaptive attacks (fltrust + martfl, by threat model)
#     Step 13 — Drowning attack  (similarity-based defenses only, by threat model)
#     Step 14 — MartFL collusion (martfl-only stress test, by design)
#     Step 17 — ALIE attack      (variance-based defenses, by threat model)
#
# P3  POST-HOC / APPENDIX — cheap, optional
#     Step 15 — Proportional pricing (post-hoc replay of Step 10 traces)
#
# P4  VALUATION ANALYSIS — dedicated per-seller value attribution
#     Step 21 — KernelSHAP / LOO / Influence on CIFAR-100, focused defense
#               subset (fedavg, fltrust, martfl, foolsgold), 2 attack
#               conditions (no-attack baseline + backdoor). Decoupled from
#               Step 10 because KernelSHAP was the dominant cost in step 10
#               cells and was hitting CPU-bound pathological paths on
#               clustering defenses. Step 10 now reports headline metrics
#               only; Step 21 provides the per-seller value-attribution data
#               for the paper's §5 valuation subsection.
#
# P5  ADAPTIVE-ATTACK SUPPLEMENT — Bagdasaryan model-replacement boosting
#     Re-runs Step 10 configs on CIFAR-100 ONLY, with the adversary's
#     post-training gradient multiplied by γ=10 before upload (enabled via
#     the BAGDASARYAN_SCALE_FACTOR env var). Implements the classic FL
#     backdoor boosting attack from Bagdasaryan et al. 2020. Produces a
#     supplementary table for the paper showing that existing defenses
#     get even worse under a simple adaptive attack, pre-empting the
#     expected reviewer objection "what about adaptive attacks?". Results
#     are written to results/step16_bagdasaryan_* so they do NOT shadow
#     the non-adaptive P0 step 10 results.
#     Cost: ~1-2 hours on 3 GPUs (12 CIFAR-100 cells, .success-skipped).
#
# DEPENDENCY ORDER (must be respected even when ranking by priority)
# ------------------------------------------------------------------
#   Step 1  → must finish before Step 3 (Step 3 reads its output)
#   Step 3  → must finish before EVERY downstream step (Steps 4-17 all read
#             tuned_defense_params.json)
#   Step 10 → reads tuned_defense_params.json
#   Steps 4-17 → read tuned_defense_params.json
#   P5      → depends on P0 Step 10 configs being generated (it copies them)
#
# Execution order chosen here:
#   P0:  Step 1 → Step 3 → Step 10                    (foundation + headline)
#   P1:  Step 4 → 5 → 7 → 8 → 9                       (in-depth filtering sweep)
#   P2:  Step 6 → 13 → 14 → 17                        (threat-model-scoped)
#   P3:  Step 15                                       (post-hoc pricing)
#   P4:  Step 21                                       (per-seller valuation)
#   P5:  Step 10 (CIFAR100, γ=10)                      (Bagdasaryan supplement)
#
# TIER OVERRIDES (set to 0 to skip a tier)
# ----------------------------------------
#   RUN_P0=1   default — paper-critical (foundation + headline)
#   RUN_P1=1   default — in-depth analyses
#   RUN_P2=1   default — threat-model-scoped attacks
#   RUN_P3=1   default — post-hoc pricing
#   RUN_P4=1   default — per-seller valuation analysis
#   RUN_P5=1   default — Bagdasaryan adaptive-attack supplement
#
# EXAMPLES
#   bash run_main_all_datasets.sh                         # everything
#   RUN_P2=0 RUN_P3=0 RUN_P4=0 RUN_P5=0 bash run_main_all_datasets.sh  # P0 + P1 only
#   RUN_P0=1 RUN_P1=0 RUN_P2=0 RUN_P3=0 RUN_P4=0 RUN_P5=0 bash run_main_all_datasets.sh
#                                                          # only foundation + headline
#   RUN_P0=0 RUN_P1=0 RUN_P2=0 RUN_P3=0 RUN_P4=0 bash run_main_all_datasets.sh
#                                                          # only Bagdasaryan supplement
#   DRY_RUN=1 bash run_main_all_datasets.sh                # print plan & exit
#   CLEAN=1 CLEAN_CONFIRM=1 bash run_main_all_datasets.sh  # wipe results/ for clean repro
#   bash run_main_all_datasets.sh 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
#
# AVOID-RERUN GUARANTEE
# ---------------------
# This script preserves any run that already has a .success marker. The
# clean_step helper:
#   - deletes generated configs (so they regenerate from the current grid)
#   - deletes scenario result dirs that have NO .success marker anywhere inside
#   - leaves dirs that contain at least one .success marker untouched and
#     only sweeps .failed / .in_progress markers out of them
# The parallel runner then skips per-config-dir based on .success +
# final_metrics.json, so re-running this script is idempotent.
#
# PREREQUISITES (one-time source edits already applied by the agent)
# -------------------------------------------------------------------
# 1. config_common_utils.py:ENABLED_DATASETS includes FEMNIST + Purchase100
# 2. config_common_utils.py:ENABLED_MODEL_CONFIGS includes femnist_cnn +
#    mlp_purchase100_baseline
# 3. config_common_utils.py:FILTERING_DEFENSES is defined (the in-depth
#    analysis defense scope — see config_common_utils.py for the rationale)
# 4. generate_step5_advanced_sybil.py:SYBIL_SETUPS_ALL exists (multi-dataset
#    refactor in place; currently pinned to CIFAR100 by SYBIL_SETUPS)
# 5. The Bulyan-on-small-marketplace skip is in generate_step10_main_summary.py
#    (skips Bulyan × {FEMNIST, Texas100, Purchase100} where N >= 4f+3 fails)
# 6. FEMNIST natural-partitioning fix: writer_ids is promoted by
#    UnifiedDatasetWrapper, and _partition_by_natural_id walks wrapper chains
#
# The script asserts these below and exits early if any are missing.
# =============================================================================

# Note: NOT using `set -e` so a failure in one step doesn't kill the whole pipeline.
RESULTS_DIR="./results"
CONFIGS_DIR="./configs_generated_benchmark"
# GPU_IDS and NUM_PROCS are env-overridable. Defaults match the dev-box layout
# (GPUs 2,3,4 × 1 cell each). Override for a 4-GPU × 3-cells-per-GPU sweep with:
#     GPU_IDS=1,2,3,4 NUM_PROCS=12 bash run_main_all_datasets.sh
GPU_IDS="${GPU_IDS:-1,2,3,4}"
NUM_PROCS="${NUM_PROCS:-8}"

# =============================================================================
# CONCURRENCY SAFETY CHECK — abort if another dispatcher is already running
# =============================================================================
# Phase 0 below will delete .in_progress markers GLOBALLY under results/.  If a
# live dispatcher is using those markers to track what it's currently training,
# deleting them + the subsequent clean_step wipe of "stale" scenario dirs will
# silently destroy in-flight work.  Better to hard-abort and make the user
# kill the other dispatcher explicitly.
# =============================================================================
EXISTING_PIDS=$(pgrep -f "run_full_benchmark.py\|run_parallel_experiment.py\|run_single_experiment" 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
  echo ""
  echo "=============================================="
  echo " ❌ ABORT: another dispatcher / worker is already running"
  echo "=============================================="
  ps -o pid,etime,cmd -p $EXISTING_PIDS 2>/dev/null | head -20
  echo ""
  echo "  Phase 0 below would delete .in_progress markers globally, which would"
  echo "  corrupt the active run above. Kill the existing dispatcher first:"
  echo ""
  echo "    pkill -f 'run_full_benchmark.py'"
  echo "    pkill -f 'run_parallel_experiment.py'"
  echo "    sleep 5"
  echo ""
  echo "  Then re-run this script. Re-runs are idempotent via .success markers."
  echo ""
  echo "  (If you're 100% sure no live dispatcher exists and these are zombies,"
  echo "   override with:  SKIP_CONCURRENCY_CHECK=1 bash run_main_all_datasets.sh)"
  echo "=============================================="
  if [ "${SKIP_CONCURRENCY_CHECK:-0}" != "1" ]; then
    exit 1
  fi
  echo "  [WARN] SKIP_CONCURRENCY_CHECK=1 — proceeding anyway."
fi

# =============================================================================
# ENVIRONMENT SMOKE TEST — fail fast if Python/torch/datasets are broken
# =============================================================================
# Without this, a broken env will surface as the first cell crashing after
# several minutes of setup.  A 2-second check up-front saves that time.
if ! python -c "import torch, numpy, datasets; print(f'  [env] torch {torch.__version__}, numpy {numpy.__version__}')" 2>&1; then
  echo "[ABORT] Python environment check failed — fix it before launching."
  exit 1
fi

# -----------------------------------------------------------------------------
# Tier flags — set any to 0 from the environment to skip that tier.
# Defaults: all tiers ON.
# -----------------------------------------------------------------------------
RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_P4="${RUN_P4:-1}"
RUN_P5="${RUN_P5:-1}"

# Bagdasaryan scaling factor for P5. Default γ=10 follows the original
# Bagdasaryan et al. 2020 paper for 10-client FedAvg. Override with e.g.
# BAGDASARYAN_GAMMA=50 for a stronger adaptive attack.
BAGDASARYAN_GAMMA="${BAGDASARYAN_GAMMA:-10}"

echo ""
echo "=============================================="
echo " Tier plan"
echo "=============================================="
printf "  P0  paper-critical    (Step 1 → Step 3 → Step 10)            : %s\n" \
       "$([ "$RUN_P0" = "1" ] && echo RUN || echo SKIP)"
printf "  P1  in-depth analysis (Steps 4, 5, 7, 8, 9 — filtering set)  : %s\n" \
       "$([ "$RUN_P1" = "1" ] && echo RUN || echo SKIP)"
printf "  P2  threat-model      (Steps 6, 13, 14, 17 — scoped narrow)  : %s\n" \
       "$([ "$RUN_P2" = "1" ] && echo RUN || echo SKIP)"
printf "  P3  post-hoc pricing  (Step 15)                              : %s\n" \
       "$([ "$RUN_P3" = "1" ] && echo RUN || echo SKIP)"
printf "  P4  valuation         (Step 21 — KernelSHAP/LOO/Influence)   : %s\n" \
       "$([ "$RUN_P4" = "1" ] && echo RUN || echo SKIP)"
printf "  P5  Bagdasaryan γ=%-2s  (CIFAR-100 only, adaptive supplement)  : %s\n" \
       "$BAGDASARYAN_GAMMA" \
       "$([ "$RUN_P5" = "1" ] && echo RUN || echo SKIP)"
if [ "$RUN_P0" != "1" ]; then
  echo ""
  echo "  ⚠ WARNING: P0 is OFF. The headline table will NOT be (re)generated."
  echo "    Set RUN_P0=1 unless you really mean to skip it."
fi
echo ""

# -----------------------------------------------------------------------------
# CLEAN + DRY_RUN — reproducibility flags
# -----------------------------------------------------------------------------
# CLEAN=1    wipe results/ and configs_generated_benchmark/ before starting
#            (produces a clean reproduction from nothing; use with care).
# DRY_RUN=1  print the tier plan and exit BEFORE any experiment runs or
#            file deletions — useful to confirm what would happen.
# -----------------------------------------------------------------------------
if [ "${CLEAN:-0}" = "1" ]; then
  echo "=============================================="
  echo " CLEAN=1 — wiping results/ and configs_generated_benchmark/"
  echo "=============================================="
  echo "  Directories to remove:"
  [ -d "$RESULTS_DIR" ] && echo "    $RESULTS_DIR   ($(du -sh $RESULTS_DIR 2>/dev/null | cut -f1))"
  [ -d "$CONFIGS_DIR" ] && echo "    $CONFIGS_DIR   ($(du -sh $CONFIGS_DIR 2>/dev/null | cut -f1))"
  if [ "${CLEAN_CONFIRM:-0}" != "1" ]; then
    echo ""
    echo "  Re-run with CLEAN=1 CLEAN_CONFIRM=1 to actually delete."
    exit 0
  fi
  rm -rf "$RESULTS_DIR" "$CONFIGS_DIR"
  echo "  [done] wiped."
  echo ""
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "=============================================="
  echo " DRY_RUN=1 — would run the tier plan above and exit here."
  echo "=============================================="
  echo "  GPU_IDS=$GPU_IDS  NUM_PROCS=$NUM_PROCS  BAGDASARYAN_GAMMA=$BAGDASARYAN_GAMMA"
  echo "  Re-run without DRY_RUN=1 to actually execute."
  exit 0
fi

# Reduce CUDA memory fragmentation (helps with OOM)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Bump file descriptor limit (Step 10 / main_summary needs it on long runs).
echo "[Setup] Current FD limit: $(ulimit -n)"
if ! ulimit -n 65536 2>/dev/null; then
  HARD_LIMIT=$(ulimit -Hn)
  echo "[WARN] Could not raise FD limit to 65536 (hard limit: $HARD_LIMIT)."
  ulimit -n "$HARD_LIMIT" 2>/dev/null || \
    echo "[WARN] Could not raise FD limit at all. Step 10 may fail with 'Too many open files'."
fi
echo "[Setup] FD limit now: $(ulimit -n)"

# =============================================================================
# Pre-flight checks: source edits must be in place
# =============================================================================
COMMON_UTILS="experiments/gradient_market/configs_generation/config_common_utils.py"
STEP5_GEN="experiments/gradient_market/configs_generation/generate_step5_advanced_sybil.py"
STEP10_GEN="experiments/gradient_market/configs_generation/generate_step10_main_summary.py"
DATASET_PY="src/common_utils/data_utils/dataset.py"
PARTITIONER_PY="src/common_utils/data_utils/data_partitioner.py"

echo ""
echo "=============================================="
echo " Pre-flight: verifying source edits"
echo "=============================================="

missing=0
if ! grep -q '^\s*"FEMNIST",' "$COMMON_UTILS"; then
  echo "[FAIL] $COMMON_UTILS does not list FEMNIST in ENABLED_DATASETS"
  missing=1
fi
if ! grep -q '^\s*"Purchase100",' "$COMMON_UTILS"; then
  echo "[FAIL] $COMMON_UTILS does not list Purchase100 in ENABLED_DATASETS"
  missing=1
fi
if ! grep -q '^\s*"femnist_cnn",' "$COMMON_UTILS"; then
  echo "[FAIL] $COMMON_UTILS does not list femnist_cnn in ENABLED_MODEL_CONFIGS"
  missing=1
fi
if ! grep -q '^\s*"mlp_purchase100_baseline",' "$COMMON_UTILS"; then
  echo "[FAIL] $COMMON_UTILS does not list mlp_purchase100_baseline in ENABLED_MODEL_CONFIGS"
  missing=1
fi
if ! grep -q 'FILTERING_DEFENSES' "$COMMON_UTILS"; then
  echo "[FAIL] $COMMON_UTILS does not define FILTERING_DEFENSES (in-depth analysis scope)"
  missing=1
fi
if ! grep -q 'SYBIL_SETUPS_ALL' "$STEP5_GEN"; then
  echo "[FAIL] $STEP5_GEN has not been refactored to multi-dataset (SYBIL_SETUPS_ALL missing)"
  missing=1
fi
if ! grep -q 'defense_name == "bulyan"' "$STEP10_GEN"; then
  echo "[FAIL] $STEP10_GEN does not have the Bulyan-on-small-marketplace skip"
  missing=1
fi
if ! grep -q 'self\.writer_ids = original_dataset' "$DATASET_PY"; then
  echo "[FAIL] $DATASET_PY does not promote writer_ids in UnifiedDatasetWrapper"
  echo "       (FEMNIST natural partitioning will crash without this)"
  missing=1
fi
if ! grep -q '_find_writer_ids' "$PARTITIONER_PY"; then
  echo "[FAIL] $PARTITIONER_PY does not have the recursive _find_writer_ids helper"
  missing=1
fi
if [ $missing -ne 0 ]; then
  echo ""
  echo "[ABORT] One or more prerequisite source edits are missing. See above."
  echo "        Run 'git pull' and re-check, or fix the listed files manually."
  exit 1
fi
echo "[OK] All source-edit prerequisites are in place."

# =============================================================================
# Helper: Clean stale results + configs for a given scenario prefix
#         (preserves any scenario dir that has at least one .success marker)
# =============================================================================
clean_step() {
  local config_subdir="$1"
  shift
  local scenario_prefixes=("$@")

  if [ -n "$config_subdir" ] && [ -d "$CONFIGS_DIR/$config_subdir" ]; then
    echo "  [Clean] Removing generated configs: $CONFIGS_DIR/$config_subdir"
    rm -rf "$CONFIGS_DIR/$config_subdir"
  fi

  for prefix in "${scenario_prefixes[@]}"; do
    for scenario_dir in "$RESULTS_DIR"/${prefix}*; do
      [ -d "$scenario_dir" ] || continue
      if ! find "$scenario_dir" -name ".success" -print -quit 2>/dev/null | grep -q .; then
        echo "  [Clean] Wiping stale scenario: $scenario_dir"
        rm -rf "$scenario_dir"
      else
        find "$scenario_dir" -name ".failed" -delete 2>/dev/null || true
        find "$scenario_dir" -name ".in_progress" -delete 2>/dev/null || true
      fi
    done
  done
}

# =============================================================================
# Phase 0: Global zombie cleanup (safety net)
# =============================================================================
echo ""
echo "=============================================="
echo " Phase 0: Global cleanup — zombies and failures"
echo "=============================================="
find "$RESULTS_DIR" -name ".in_progress" -delete 2>/dev/null || true
find "$RESULTS_DIR" -name ".failed" -delete 2>/dev/null || true
echo "[Cleanup] Done."

# =============================================================================
# TIER P0 — Paper-critical (Step 1 → Step 3 → Step 10)
# =============================================================================
# Dependency chain inside P0:
#
#   Step 1  produces  golden_training_params.json
#               │
#               ▼
#   Step 3  reads   golden_training_params.json
#           writes  tuned_defense_params.json
#               │
#               ▼
#   Step 10 reads   tuned_defense_params.json
#           writes  the headline main-summary table
#
# All three steps use the per-cell .success skip, so on a re-run the existing
# CIFAR-100 / Texas-100 / TREC cells from prior runs are preserved and only
# the new (FEMNIST, Purchase-100) cells actually execute.
# =============================================================================
if [ "$RUN_P0" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P0 — Paper-critical work starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P0 / Step 1: Federated training HP tuning (writes golden_training_params.json)
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P0 · 1/3] Step 1 — Federated training HP tuning"
  echo "=============================================="
  echo "  Cleaning step 1 (preserves existing .success runs)..."
  clean_step "step1_iid_tuning" "step1_tune_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 1 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # Sanity check: confirm golden_training_params.json now has entries for
  # every (model_config_name) referenced by the downstream steps.
  echo ""
  echo "  [Verify] Checking golden_training_params.json coverage..."
  python - <<'PY' || { echo "  ❌ Aborting P0 — golden_training_params.json is incomplete."; exit 1; }
import json, sys
p = "experiments/gradient_market/configs_generation/tuned_params/golden_training_params.json"
try:
    d = json.load(open(p))
except Exception as exc:
    print(f"  ❌ Could not read {p}: {exc}")
    sys.exit(1)
needed = [
    "cifar100_cnn",
    "mlp_texas100_baseline",
    "textcnn_trec_baseline",
    "femnist_cnn",
    "mlp_purchase100_baseline",
]
missing = [k for k in needed if k not in d]
print(f"  Total entries in golden_training_params.json: {len(d)}")
for k in needed:
    mark = "✓" if k in d else "✗ MISSING"
    print(f"    {mark}  {k}")
if missing:
    print()
    print(f"  ❌ {len(missing)} required entries missing after Step 1.")
    sys.exit(1)
print(f"  ✓ All {len(needed)} required entries present.")
PY

  # ---------------------------------------------------------------------------
  # P0 / Step 3: Defense HP tuning (all enabled datasets, all defenses)
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P0 · 2/3] Step 3 — Defense HP tuning"
  echo "=============================================="
  echo "  Cleaning step 3 (preserves existing .success runs)..."
  clean_step "step3_defense_tuning" "step3_tune_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 3 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # Sanity check: tuned_defense_params.json should now have entries for every
  # (defense × model × backdoor) triple expected by downstream steps. Missing
  # entries will cause the step 10 generator to fall back to defense defaults,
  # which silently degrades the headline table.
  echo ""
  echo "  [Verify] Checking tuned_defense_params.json coverage..."
  python - <<'PY' || echo "  ⚠ WARNING: tuned_defense_params.json has gaps — see above. Downstream steps will use defense defaults for missing combos."
import json, sys
p = "experiments/gradient_market/configs_generation/tuned_params/tuned_defense_params.json"
try:
    d = json.load(open(p))
except Exception as exc:
    print(f"  ❌ Could not read {p}: {exc}")
    sys.exit(1)

sys.path.insert(0, "experiments/gradient_market/configs_generation")
try:
    from config_common_utils import ENABLED_DEFENSES, ENABLED_DATASETS
except Exception as exc:
    print(f"  ❌ Could not import ENABLED_DEFENSES/ENABLED_DATASETS: {exc}")
    sys.exit(1)

dataset_to_model = {
    "CIFAR100":    "cifar100_cnn",
    "CIFAR10":     "cifar10_cnn",
    "FEMNIST":     "femnist_cnn",
    "Texas100":    "mlp_texas100_baseline",
    "Purchase100": "mlp_purchase100_baseline",
    "TREC":        "textcnn_trec_baseline",
}

image_defenses   = {"fedavg", "fltrust", "martfl", "skymask", "trimmed_mean",
                    "multi_krum", "rflpa", "spmc", "flame", "deepsight",
                    "bulyan", "foolsgold"}
nontext_defenses = image_defenses - {"skymask"}  # skymask is image-only

expected_combos = []
for ds in sorted(ENABLED_DATASETS):
    model = dataset_to_model.get(ds)
    if model is None:
        continue
    defenses = image_defenses if ds in ("CIFAR100", "CIFAR10", "FEMNIST") else nontext_defenses
    for defense in sorted(defenses):
        if defense in ENABLED_DEFENSES:
            expected_combos.append(f"{defense}_{model}_backdoor")

missing = [k for k in expected_combos if k not in d]
print(f"  Total entries in tuned_defense_params.json: {len(d)}")
print(f"  Expected (defense × dataset × backdoor) combos: {len(expected_combos)}")
print(f"  Present: {len(expected_combos) - len(missing)}")
if missing:
    print(f"  ⚠ MISSING ({len(missing)} combos):")
    for k in missing[:10]:
        print(f"      {k}")
    if len(missing) > 10:
        print(f"      ... and {len(missing) - 10} more")
    sys.exit(1)
print(f"  ✓ All {len(expected_combos)} expected entries present.")
PY

  # ---------------------------------------------------------------------------
  # P0 / Step 10: Main summary (headline accuracy / ASR / BSR / MSR table)
  # ---------------------------------------------------------------------------
  # Uses ALL 11 defenses (the full IMAGE_DEFENSES / TEXT_TABULAR_DEFENSES list)
  # because the headline table needs the full landscape including non-filtering
  # defenses (Trimmed-Mean, RFLPA, SPMC) for the "filtering matters" comparison
  # in the paper. Bulyan is auto-skipped on small-marketplace datasets via the
  # constraint check in generate_step10_main_summary.py.
  echo ""
  echo "=============================================="
  echo " [P0 · 3/3] Step 10 — Main summary (headline table)"
  echo "=============================================="
  echo "  Cleaning step 10 (preserves existing .success runs)..."
  clean_step "step12_main_summary" "step12_main_summary_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 10 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # Post-Step-10 sanity: verify the expected defense × dataset combos all
  # produced at least one .success result. Missing combos are flagged as
  # warnings (not fatal) so the rest of the pipeline can still proceed.
  echo ""
  echo "  [Verify] Checking step 10 coverage..."
  python - <<'PY' || echo "  ⚠ WARNING: step 10 is incomplete — see above. Downstream analyses may show gaps."
import os, re, sys
from pathlib import Path

sys.path.insert(0, "experiments/gradient_market/configs_generation")
try:
    from config_common_utils import ENABLED_DEFENSES, ENABLED_DATASETS
except Exception as exc:
    print(f"  ❌ Could not import ENABLED_DEFENSES/ENABLED_DATASETS: {exc}")
    sys.exit(1)

results_root = Path("./results")
if not results_root.exists():
    print("  ❌ results/ directory not found.")
    sys.exit(1)

# Find all step12_main_summary_* scenario dirs and classify by (defense, dataset)
SCENARIO_RE = re.compile(
    r"^step12_main_summary_(?P<defense>.+?)_"
    r"(?P<modality>image|tabular|text)_"
    r"(?P<dataset>CIFAR100|CIFAR10|FEMNIST|Texas100|Purchase100|TREC)_"
)

present_combos = {}
for scenario_dir in results_root.glob("step12_main_summary_*"):
    if not scenario_dir.is_dir():
        continue
    m = SCENARIO_RE.match(scenario_dir.name)
    if not m:
        continue
    key = (m.group("defense"), m.group("dataset"))
    n_success = sum(1 for _ in scenario_dir.rglob(".success"))
    present_combos[key] = present_combos.get(key, 0) + n_success

# Expected combos based on ENABLED_DEFENSES × ENABLED_DATASETS
image_datasets = {"CIFAR100", "CIFAR10", "FEMNIST"}
text_tabular_datasets = {"Texas100", "Purchase100", "TREC"}

expected_combos = []
for defense in sorted(ENABLED_DEFENSES):
    for ds in sorted(ENABLED_DATASETS):
        if ds in image_datasets or ds in text_tabular_datasets:
            # SkyMask is image-only
            if defense == "skymask" and ds not in image_datasets:
                continue
            # Bulyan requires N >= 4f+3 — auto-skipped on small-marketplace datasets
            if defense == "bulyan" and ds in {"FEMNIST", "Texas100", "Purchase100"}:
                continue
            expected_combos.append((defense, ds))

missing = [c for c in expected_combos if c not in present_combos or present_combos[c] == 0]
partial = [c for c in expected_combos if c in present_combos and 0 < present_combos[c] < 3]

print(f"  Expected (defense, dataset) combos: {len(expected_combos)}")
print(f"  Combos with ≥1 .success:            {len(expected_combos) - len(missing)}")
print(f"  Combos with all 3 seeds complete:   {len(expected_combos) - len(missing) - len(partial)}")
print(f"  Total .success markers found:       {sum(present_combos.values())}")

if missing:
    print()
    print(f"  ⚠ MISSING ({len(missing)} combos — no .success at all):")
    for defense, ds in missing[:15]:
        print(f"      {defense} × {ds}")
    if len(missing) > 15:
        print(f"      ... and {len(missing) - 15} more")
if partial:
    print()
    print(f"  ⚠ PARTIAL ({len(partial)} combos — fewer than 3 seeds):")
    for defense, ds in partial[:10]:
        n = present_combos[(defense, ds)]
        print(f"      {defense} × {ds}  ({n}/3 seeds)")
    if len(partial) > 10:
        print(f"      ... and {len(partial) - 10} more")

if missing or partial:
    sys.exit(1)
print(f"  ✓ All expected combos have ≥3 seeds complete.")
PY

  echo ""
  echo "✅ TIER P0 complete."
else
  echo ""
  echo "⏭  TIER P0 skipped (RUN_P0=0). Step 1 + Step 3 + Step 10 will NOT run."
fi

# =============================================================================
# TIER P1 — In-depth analysis (Steps 4, 5, 7, 8, 9)
# =============================================================================
# These steps test specific robustness axes (attack budget, sybil strategy,
# buyer-side bias, scalability, heterogeneity) using the FILTERING_DEFENSES
# scope rather than all 11 defenses. The scope is set in each generator via
# the FILTERING_DEFENSES import from config_common_utils.py.
#
# Why filtering defenses only here:
#   In a marketplace, simply tolerating malicious gradients in the aggregate is
#   not enough — honest sellers must be paid and adversarial sellers must not
#   be. Defenses that produce only a robust aggregate without per-seller
#   decisions (Trimmed-Mean, RFLPA, SPMC) cannot be evaluated against
#   marketplace fairness metrics (BSR, MSR) without imputing a per-seller
#   decision rule that the original defense does not specify. Including them
#   in the in-depth analysis would conflate apples and oranges.
#
# All 5 deep-dive steps share the same filtering-defense scope, so claims like
# "across all defense families..." in the paper are well-defined.
# =============================================================================
if [ "$RUN_P1" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P1 — In-depth analysis starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P1 / Step 4: Attack sensitivity sweep (adv_rate × poison_rate)
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P1 · 1/5] Step 4 — Attack sensitivity sweep"
  echo "=============================================="
  echo "  Cleaning step 4 (preserves existing .success runs)..."
  clean_step "step5_attack_sensitivity" "step5_atk_sens_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 4 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P1 / Step 5: Advanced sybil strategies
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P1 · 2/5] Step 5 — Advanced sybil strategies"
  echo "=============================================="
  echo "  Cleaning step 5 (preserves existing .success runs)..."
  clean_step "step6_advanced_sybil" "step6_adv_sybil_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 5 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P1 / Step 7: Buyer-side / demand-side attacks
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P1 · 3/5] Step 7 — Buyer-side attacks"
  echo "=============================================="
  echo "  Cleaning step 7 (preserves existing .success runs)..."
  clean_step "step8_buyer_attacks" "step8_buyer_attack_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 7 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P1 / Step 8: Marketplace scalability
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P1 · 4/5] Step 8 — Marketplace scalability"
  echo "=============================================="
  echo "  Cleaning step 8 (preserves existing .success runs)..."
  clean_step "step10_scalability" "step10_scalability_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 8 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P1 / Step 9: Data heterogeneity
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P1 · 5/5] Step 9 — Data heterogeneity"
  echo "=============================================="
  echo "  Cleaning step 9 (preserves existing .success runs)..."
  # IMPORTANT: enumerate per-defense prefixes to avoid wiping step11_freerider_*
  clean_step "step11_heterogeneity" \
    "step11_fedavg_" "step11_fltrust_" "step11_martfl_" "step11_skymask_" \
    "step11_trimmed_mean_" "step11_multi_krum_" "step11_rflpa_" "step11_spmc_" \
    "step11_flame_" "step11_deepsight_" "step11_bulyan_" "step11_foolsgold_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 9 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  echo ""
  echo "✅ TIER P1 complete."
else
  echo ""
  echo "⏭  TIER P1 skipped (RUN_P1=0). Steps 4, 5, 7, 8, 9 will NOT run."
fi

# =============================================================================
# TIER P2 — Threat-model-scoped attacks (Steps 6, 13, 14, 17)
# =============================================================================
# These attacks are by design targeted at specific defense families. The
# defense scope is intentionally narrower than the in-depth analysis tier
# because expanding them to all defenses would conflate "the targeted attack
# tested against the right defense" with "the targeted attack tested against
# defenses it was not designed to bypass," which weakens rather than
# strengthens the threat model story.
#
#   Step 6  — Adaptive attacks (gradient_manipulation against fltrust + martfl)
#             The "adaptive" framing requires the adversary to model the
#             defense; only similarity / trust-based defenses are meaningful
#             targets.
#
#   Step 13 — Drowning attack (similarity-based defenses only)
#             Drowning specifically exploits trust-score computation in
#             similarity-based aggregation. Distance-based and coordinate-wise
#             defenses are not its target.
#
#   Step 14 — MartFL collusion (martfl-only stress test)
#             Tests whether MartFL's clustering aggregation breaks under a
#             coordinated buyer-seller collusion attack. By design a single-
#             defense deep dive.
#
#   Step 17 — ALIE (A Little Is Enough) attack
#             Originally designed against Trimmed-Mean and Multi-Krum.
#             Currently scoped to fltrust + martfl in the generator.
#
# Each step's defense scope is set inside its own generator file, not by this
# script. This tier just orchestrates them.
# =============================================================================
if [ "$RUN_P2" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P2 — Threat-model-scoped attacks starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P2 / Step 6: Adaptive attacks
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P2 · 1/4] Step 6 — Adaptive attacks"
  echo "=============================================="
  echo "  Cleaning step 6 (preserves existing .success runs)..."
  clean_step "step7_adaptive_attack" "step7_adaptive_" "step7_baseline_no_attack"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 6 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P2 / Step 13: Drowning attack
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P2 · 2/4] Step 13 — Drowning attack"
  echo "=============================================="
  echo "  Cleaning step 13 (preserves existing .success runs)..."
  clean_step "step13_drowning_attack" "step13_drowning_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 13 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P2 / Step 14: MartFL collusion stress test
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P2 · 3/4] Step 14 — MartFL collusion"
  echo "=============================================="
  echo "  Cleaning step 14 (preserves existing .success runs)..."
  clean_step "step14_martfl_collusion" "step14_collusion_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 14 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  # ---------------------------------------------------------------------------
  # P2 / Step 17: ALIE attack
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P2 · 4/4] Step 17 — ALIE attack"
  echo "=============================================="
  echo "  Cleaning step 17 (preserves existing .success runs)..."
  clean_step "step17_alie_attack" "step17_alie_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 17 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  echo ""
  echo "✅ TIER P2 complete."
else
  echo ""
  echo "⏭  TIER P2 skipped (RUN_P2=0). Steps 6, 13, 14, 17 will NOT run."
fi

# =============================================================================
# TIER P3 — Post-hoc / appendix (Step 15)
# =============================================================================
# Step 15 (proportional pricing) is a post-hoc replay of Step 10 main-summary
# traces under three payment models (binary, proportional, quality-based). It
# is much cheaper than a fresh training run because it reuses the per-seller
# selection / quality data from Step 10. Belongs at the end of the pipeline
# both because it depends on Step 10 being done AND because it's cheap enough
# to defer.
# =============================================================================
if [ "$RUN_P3" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P3 — Post-hoc pricing starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P3 / Step 15: Proportional pricing analysis
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P3 · 1/1] Step 15 — Proportional pricing"
  echo "=============================================="
  echo "  Cleaning step 15 (preserves existing .success runs)..."
  clean_step "step15_proportional_pricing" "step15_pricing_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 15 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  echo ""
  echo "✅ TIER P3 complete."
else
  echo ""
  echo "⏭  TIER P3 skipped (RUN_P3=0). Step 15 will NOT run."
fi

# =============================================================================
# TIER P4 — Per-seller valuation analysis (Step 21)
# =============================================================================
# Step 21 runs the dedicated valuation pipeline (KernelSHAP / LOO / Influence)
# on a focused scope: CIFAR-100 only, 4 representative defenses (fedavg,
# fltrust, martfl, foolsgold), 2 attack conditions (no-attack baseline +
# backdoor), 2 seeds. Total expected cells: 4 × 2 × 2 = 16.
#
# Why it lives here as its own tier:
#   - It uses full-resolution KernelSHAP (kshap_samples=500), which is much
#     more expensive per cell than step 10 (which now has valuation disabled
#     entirely). Keeping it as a separate tier means you can skip it when
#     you only need the headline benchmark.
#   - It is independent of P1/P2/P3 — no dependency ordering — but DOES
#     depend on P0 (Step 3) for tuned defense HPs. Place it after P3 so all
#     defense-tuning work has settled before it consumes tuned_defense_params.
#   - Decoupling KernelSHAP from step 10 was the fix for the 30+ minute
#     stall that was blocking the headline benchmark. Step 21 is where the
#     valuation analysis lives now.
#
# Estimated cost: ~5-7 hours on 3 GPUs (16 cells × ~30-50 min/cell with full
# KernelSHAP, 3-way parallelism).
# =============================================================================
if [ "$RUN_P4" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P4 — Per-seller valuation starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P4 / Step 21: Per-seller valuation analysis
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P4 · 1/1] Step 21 — Per-seller valuation analysis"
  echo "=============================================="
  echo "  Cleaning step 21 (preserves existing .success runs)..."
  clean_step "step21_valuation_analysis" "step21_valuation_"
  python experiments/gradient_market/run_full_benchmark.py \
    --step 21 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS

  echo ""
  echo "✅ TIER P4 complete."
else
  echo ""
  echo "⏭  TIER P4 skipped (RUN_P4=0). Step 21 (valuation) will NOT run."
fi

# =============================================================================
# TIER P5 — Bagdasaryan adaptive-attack supplement (CIFAR-100 only)
# =============================================================================
# Re-runs the step-10 CIFAR-100 cells with the adversary's gradient scaled by
# γ (default 10) before upload. This is the classic FedAvg backdoor boosting
# attack from Bagdasaryan et al. 2020 ("How to Backdoor Federated Learning").
# It produces a supplementary table showing defenses get worse under a simple
# adaptive attack — addressing the "what about adaptive?" reviewer objection.
#
# Implementation approach (no new generator required):
#   1. Copy the step 10 CIFAR-100 configs to a new directory
#        configs_generated_benchmark/step16_bagdasaryan/
#   2. Rewrite `save_path` inside each config.yaml so results land in
#        results/step16_bagdasaryan_<defense>_image_CIFAR100_cnn/...
#      rather than shadowing the non-adaptive step 10 results.
#   3. Launch run_parallel_experiment.py against the new directory with
#      BAGDASARYAN_SCALE_FACTOR=γ set in the environment. The code hook in
#      AdvancedPoisoningAdversarySeller.get_gradient_for_upload reads the env
#      var and multiplies the uploaded gradient by γ — no code regeneration
#      needed.
#
# Why CIFAR-100 only:
#   - Adding a single representative dataset is enough to demonstrate the
#     trend "defenses get worse under adaptive attack" without doubling the
#     compute budget of the main sweep.
#   - CIFAR-100 already has .success results in the non-adaptive P0 run, so
#     you get a direct side-by-side comparison table for the paper.
# =============================================================================
if [ "$RUN_P5" = "1" ]; then
  echo ""
  echo "=============================================="
  echo " ▶ TIER P5 — Bagdasaryan adaptive supplement starts now"
  echo "=============================================="

  # ---------------------------------------------------------------------------
  # P5 / Step 10: Adaptive re-run of CIFAR-100 with γ=$BAGDASARYAN_GAMMA
  # ---------------------------------------------------------------------------
  echo ""
  echo "=============================================="
  echo " [P5 · 1/1] Bagdasaryan supplement — γ=$BAGDASARYAN_GAMMA, CIFAR100"
  echo "=============================================="

  P5_SRC_DIR="$CONFIGS_DIR/step12_main_summary"
  P5_DST_DIR="$CONFIGS_DIR/step16_bagdasaryan"

  # 1. Ensure the source step-10 configs exist. If P0 was skipped and we
  #    never generated step 10 configs in this run, generate them now
  #    (generate-only, does not launch training).
  if [ ! -d "$P5_SRC_DIR" ] || [ -z "$(ls -A "$P5_SRC_DIR" 2>/dev/null)" ]; then
    echo "  [P5] step12_main_summary configs not found — generating..."
    python experiments/gradient_market/run_full_benchmark.py \
      --step 10 --generate_only
  fi

  # 2. Fresh destination directory — wipe any previous P5 staging dir.
  rm -rf "$P5_DST_DIR"
  mkdir -p "$P5_DST_DIR"

  # 3. Copy only CIFAR-100 config dirs into the staging area, renaming
  #    each scenario dir from step12_main_summary_X → step16_bagdasaryan_X.
  COPIED=0
  for d in "$P5_SRC_DIR"/*CIFAR100*; do
    [ -d "$d" ] || continue
    src_name="$(basename "$d")"
    dst_name="${src_name/step12_main_summary_/step16_bagdasaryan_}"
    cp -r "$d" "$P5_DST_DIR/$dst_name" && COPIED=$((COPIED+1))
  done
  echo "  [P5] Copied $COPIED CIFAR-100 scenario dir(s) → $P5_DST_DIR"

  if [ "$COPIED" -eq 0 ]; then
    echo "  [P5] ⚠ No CIFAR-100 configs found in $P5_SRC_DIR. Skipping P5."
  else
    # 4. Rewrite save_path inside every config.yaml so results land in
    #    results/step16_bagdasaryan_* instead of shadowing the non-adaptive
    #    step12_main_summary_* results. The pattern appears both as
    #    `results/step12_main_summary_<defense>_...` (the save_path prefix)
    #    and, occasionally, as a `config_path` reference — this sed catches
    #    both forms.
    echo "  [P5] Rewriting save_path in config.yaml files..."
    REWRITTEN=0
    while IFS= read -r yaml; do
      sed -i 's|step12_main_summary_|step16_bagdasaryan_|g' "$yaml"
      REWRITTEN=$((REWRITTEN+1))
    done < <(find "$P5_DST_DIR" -name "config.yaml")
    echo "  [P5] Rewrote $REWRITTEN config.yaml file(s)."

    # 5. Clear any stale markers from prior P5 attempts.
    find "$RESULTS_DIR"/step16_bagdasaryan_* -name ".in_progress" -delete 2>/dev/null || true
    find "$RESULTS_DIR"/step16_bagdasaryan_* -name ".failed" -delete 2>/dev/null || true

    # 6. Launch the parallel runner with BAGDASARYAN_SCALE_FACTOR in the env.
    #    The code hook in gradient_seller.py reads this env var and applies
    #    the γ-scaling to adversary gradients after local training.
    echo "  [P5] Launching parallel runner with BAGDASARYAN_SCALE_FACTOR=$BAGDASARYAN_GAMMA"
    BAGDASARYAN_SCALE_FACTOR="$BAGDASARYAN_GAMMA" \
      python experiments/gradient_market/run_parallel_experiment.py \
        --configs_dir "$P5_DST_DIR" \
        --gpu_ids $GPU_IDS --num_processes $NUM_PROCS
    echo "  [P5] Done."
  fi

  echo ""
  echo "✅ TIER P5 complete."
else
  echo ""
  echo "⏭  TIER P5 skipped (RUN_P5=0). Bagdasaryan supplement will NOT run."
fi

# =============================================================================
# Intentionally NOT run by this script
# =============================================================================
# Step 2 (find usable training HPs)
#   Per-defense tunability sweep, ~1500 cells, 2-4 days of GPU time. NOT
#   consumed by any other step. Only useful for an appendix tunability heatmap
#   that no reviewer in this venue typically asks for. Run rerun_step2.sh
#   separately if/when a reviewer specifically asks for it.
#
# Step 11 (free riding)
#   Deprioritized — orthogonal to the main robustness story. Run separately
#   if you decide to include the free-riding analysis.
#
# Step 16 (DP fairness)
#   Dropped from current paper scope. The dp-fairness story is tangential to
#   the gradient-marketplace thesis (paper does not motivate privacy as a
#   constraint), and adding the no-attack control arm makes it expensive.
#
# Step 18 (DAVED comparison) — REMOVED. DAVED is no longer in ENABLED_DEFENSES.
#
# Step 19 (new defenses) — REDUNDANT with Step 10. After FLAME / DeepSight /
#   Bulyan / FoolsGold were merged into IMAGE_DEFENSES / TEXT_TABULAR_DEFENSES,
#   Step 10 already iterates over them.
#
# Step 20 (sleeper agent) — Deprioritized. Run separately if needed.
# =============================================================================

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================="
echo " DONE — tier summary"
echo "=============================================="
printf "  P0  paper-critical    (Step 1, Step 3, Step 10)              : %s\n" \
       "$([ "$RUN_P0" = "1" ] && echo "ran" || echo "skipped")"
printf "  P1  in-depth analysis (Steps 4, 5, 7, 8, 9)                  : %s\n" \
       "$([ "$RUN_P1" = "1" ] && echo "ran" || echo "skipped")"
printf "  P2  threat-model      (Steps 6, 13, 14, 17)                  : %s\n" \
       "$([ "$RUN_P2" = "1" ] && echo "ran" || echo "skipped")"
printf "  P3  post-hoc pricing  (Step 15)                              : %s\n" \
       "$([ "$RUN_P3" = "1" ] && echo "ran" || echo "skipped")"
printf "  P4  valuation         (Step 21)                              : %s\n" \
       "$([ "$RUN_P4" = "1" ] && echo "ran" || echo "skipped")"
printf "  P5  Bagdasaryan γ=%-2s  (CIFAR-100 adaptive supplement)         : %s\n" \
       "$BAGDASARYAN_GAMMA" \
       "$([ "$RUN_P5" = "1" ] && echo "ran" || echo "skipped")"
echo ""
echo " Verify with:"
echo "   bash verify_all.sh"
echo "   python experiments/gradient_market/visualization/extract/extract_all.py --strict"
echo ""
echo " Or inspect on disk:"
[ "$RUN_P0" = "1" ] && echo "   ls results/step1_tune_*           # federated training HP tuning"
[ "$RUN_P0" = "1" ] && echo "   ls results/step3_tune_*           # defense tuning sweep"
[ "$RUN_P0" = "1" ] && echo "   ls results/step12_main_summary_*  # headline table"
[ "$RUN_P1" = "1" ] && echo "   ls results/step5_atk_sens_*       # attack sensitivity sweep"
[ "$RUN_P1" = "1" ] && echo "   ls results/step6_adv_sybil_*      # sybil strategies"
[ "$RUN_P1" = "1" ] && echo "   ls results/step8_buyer_attack_*   # buyer-side attacks"
[ "$RUN_P1" = "1" ] && echo "   ls results/step10_scalability_*   # marketplace scalability"
[ "$RUN_P1" = "1" ] && echo "   ls results/step11_*               # data heterogeneity"
[ "$RUN_P2" = "1" ] && echo "   ls results/step7_adaptive_*       # adaptive attacks"
[ "$RUN_P2" = "1" ] && echo "   ls results/step13_drowning_*      # drowning attack"
[ "$RUN_P2" = "1" ] && echo "   ls results/step14_collusion_*     # MartFL collusion"
[ "$RUN_P2" = "1" ] && echo "   ls results/step17_alie_*          # ALIE attack"
[ "$RUN_P3" = "1" ] && echo "   ls results/step15_pricing_*       # proportional pricing"
[ "$RUN_P4" = "1" ] && echo "   ls results/step21_valuation_*     # per-seller valuation"
[ "$RUN_P5" = "1" ] && echo "   ls results/step16_bagdasaryan_*   # Bagdasaryan adaptive supplement"
echo ""
[ "$RUN_P0" = "1" ] && echo " HP files updated by Step 1 / Step 3:"
[ "$RUN_P0" = "1" ] && echo "   experiments/gradient_market/configs_generation/tuned_params/golden_training_params.json"
[ "$RUN_P0" = "1" ] && echo "   experiments/gradient_market/configs_generation/tuned_params/tuned_defense_params.json"
echo "=============================================="
