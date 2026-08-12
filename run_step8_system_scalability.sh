#!/bin/bash
# =============================================================================
# run_step8_system_scalability.sh — Run Step 8 with ALL defenses for the
# system scalability figure (latency + memory vs N).
#
# The default Step 8 only runs FOCUSED_DEFENSES (fltrust, martfl). For the
# paper's system scalability figure we need a representative set including:
#   - FedAvg (flat baseline)
#   - Similarity-based (fltrust, foolsgold)
#   - Clustering (martfl, deepsight, flame)
#   - Statistical (trimmed_mean, multi_krum)
#   - Mask-based (skymask)
#
# NOTE: Bulyan is excluded — at N=100 with adv_rate=0.3 (f=30), Bulyan
# requires N >= 4f+3 = 123, which fails. It silently clamps f and runs with
# reduced Byzantine tolerance, making the results misleading.
#
# MARKETPLACE SIZES: [10, 50, 100] (not 500)
# N=500 causes OOM in MartFL/Bulyan (5.3 GB monolithic torch.stack of all
# seller gradients), spawns 1000 DataLoader workers, and starves sellers
# to ~90 samples each with Dirichlet alpha=0.5. N=100 is the practical
# ceiling for the current aggregator implementations.
#
# This script temporarily patches FOCUSED_DEFENSES and MARKETPLACE_SIZES,
# runs Step 8, then restores the originals.
#
# Prerequisites:
#   - Step 1 must have completed (golden_training_params.json exists)
#   - Step 3 must have completed (tuned_defense_params.json exists)
#
# Usage:
#   bash run_step8_system_scalability.sh
#   GPU_IDS=0,1,2,3 NUM_PROCS=4 bash run_step8_system_scalability.sh
# =============================================================================

GPU_IDS="${GPU_IDS:-1,2,3,4}"
NUM_PROCS="${NUM_PROCS:-12}"
RESULTS_DIR="./results"

COMMON_UTILS="experiments/gradient_market/configs_generation/config_common_utils.py"
TUNED_PARAMS="experiments/gradient_market/configs_generation/tuned_params/tuned_defense_params.json"
GOLDEN_PARAMS="experiments/gradient_market/configs_generation/tuned_params/golden_training_params.json"

echo ""
echo "=============================================="
echo " Step 8 — System Scalability (all defenses)"
echo "=============================================="

# Pre-flight: check that Steps 1 and 3 have produced their output
if [ ! -f "$GOLDEN_PARAMS" ]; then
    echo "[ABORT] golden_training_params.json not found."
    echo "        Run Step 1 first:  python experiments/gradient_market/run_full_benchmark.py --step 1 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS"
    exit 1
fi
echo "[OK] golden_training_params.json exists."

if [ ! -f "$TUNED_PARAMS" ]; then
    echo "[ABORT] tuned_defense_params.json not found."
    echo "        Run Step 3 first:  python experiments/gradient_market/run_full_benchmark.py --step 3 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS"
    exit 1
fi
echo "[OK] tuned_defense_params.json exists."

# ---------------------------------------------------------------------------
# Temporarily widen FOCUSED_DEFENSES so Step 8 generates configs for all
# defenses. We back up the file and restore it after generation.
# ---------------------------------------------------------------------------
STEP8_GEN="experiments/gradient_market/configs_generation/generate_step8_scalability.py"
BACKUP="${COMMON_UTILS}.bak_step8"
BACKUP_S8="${STEP8_GEN}.bak_step8"
cp "$COMMON_UTILS" "$BACKUP"
cp "$STEP8_GEN" "$BACKUP_S8"

echo ""
echo "[Patch] Widening FOCUSED_DEFENSES for Step 8..."

# 1) Widen FOCUSED_DEFENSES (exclude bulyan — fails N>=4f+3 at N=100)
# 2) Cap MARKETPLACE_SIZES to [10, 25, 50, 75, 100] (N=500 causes OOM)
python3 - <<'PYEOF'
import re, sys

# --- Patch config_common_utils.py ---
path_cu = "experiments/gradient_market/configs_generation/config_common_utils.py"
with open(path_cu, "r", encoding="utf-8") as f:
    content = f.read()

# Use regex to be resilient to whitespace / line-ending differences
pat_focused = re.compile(r'_FOCUSED_DEFENSES_RAW\s*=\s*\[.*?\]', re.DOTALL)
match = pat_focused.search(content)
if match:
    new_line = '_FOCUSED_DEFENSES_RAW = ["fedavg", "fltrust", "martfl", "multi_krum", "skymask"]'
    content = content[:match.start()] + new_line + content[match.end():]
    with open(path_cu, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Patched FOCUSED_DEFENSES (was: {match.group().strip()}).")
else:
    print("  WARNING: Could not find _FOCUSED_DEFENSES_RAW to patch.")

# --- Patch generate_step8_scalability.py: marketplace sizes ---
path_s8 = "experiments/gradient_market/configs_generation/generate_step8_scalability.py"
with open(path_s8, "r", encoding="utf-8") as f:
    content_s8 = f.read()

pat_sizes = re.compile(r'MARKETPLACE_SIZES\s*=\s*\[.*?\]')
match_s8 = pat_sizes.search(content_s8)
if match_s8:
    new_sizes = "MARKETPLACE_SIZES = [10, 25, 50, 75, 100]"
    content_s8 = content_s8[:match_s8.start()] + new_sizes + content_s8[match_s8.end():]
    with open(path_s8, "w", encoding="utf-8") as f:
        f.write(content_s8)
    print(f"  Patched MARKETPLACE_SIZES (was: {match_s8.group().strip()}).")
else:
    print("  WARNING: Could not find MARKETPLACE_SIZES to patch.")
PYEOF

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Clean stale results (preserve .success)
echo ""
echo "[Clean] Removing stale step 8 results..."
find "$RESULTS_DIR" -path "*/step10_scalability_*/.in_progress" -delete 2>/dev/null || true
find "$RESULTS_DIR" -path "*/step10_scalability_*/.failed" -delete 2>/dev/null || true

# Remove generated configs so they regenerate with the widened defense list
rm -rf configs_generated_benchmark/step10_scalability 2>/dev/null || true

# ---------------------------------------------------------------------------
# Run Step 8
# ---------------------------------------------------------------------------
echo ""
echo "[Run] Starting Step 8 — scalability sweep (N = 10, 25, 50, 75, 100)"
echo "      Defenses: fedavg, fltrust, martfl, multi_krum, skymask"
echo "                (one per family: baseline, similarity, clustering, distance, mask)"
echo "      GPU IDs: $GPU_IDS | Parallel processes: $NUM_PROCS"
echo ""

python experiments/gradient_market/run_full_benchmark.py \
    --step 8 --gpu_ids $GPU_IDS --num_processes $NUM_PROCS
STEP8_EXIT=$?

# ---------------------------------------------------------------------------
# Restore original FOCUSED_DEFENSES
# ---------------------------------------------------------------------------
echo ""
echo "[Restore] Restoring original files..."
cp "$BACKUP" "$COMMON_UTILS"
cp "$BACKUP_S8" "$STEP8_GEN"
rm -f "$BACKUP" "$BACKUP_S8"
echo "[OK] config_common_utils.py and generate_step8_scalability.py restored."

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
if [ $STEP8_EXIT -eq 0 ]; then
    echo " Step 8 completed successfully."
else
    echo " Step 8 exited with code $STEP8_EXIT (some cells may have failed)."
fi
echo "=============================================="
echo ""
echo " Results:  ls results/step10_scalability_*/"
echo " Generate figure:"
echo "   python experiments/gradient_market/visualization/plot_system_scalability.py"
echo ""

# Quick summary of what completed
echo "Completed cells:"
for d in "$RESULTS_DIR"/step10_scalability_*/; do
    [ -d "$d" ] || continue
    defense=$(basename "$d" | sed 's/step10_scalability_//; s/_CIFAR100//')
    n_success=$(find "$d" -name ".success" 2>/dev/null | wc -l)
    n_failed=$(find "$d" -name ".failed" 2>/dev/null | wc -l)
    echo "  $defense: $n_success success, $n_failed failed"
done
