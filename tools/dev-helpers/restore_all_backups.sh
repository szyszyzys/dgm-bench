#!/bin/bash
# =============================================================================
# restore_all_backups.sh — One-shot wrapper to restore every _backup_* scenario
# under results/ and re-extract analysis CSVs afterward.
#
# Equivalent to running:
#   python restore_backup_results.py            # dry run
#   python restore_backup_results.py --apply    # actual restore
#   python experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py
#   python check_step10_results.py
#
# Usage:
#   bash restore_all_backups.sh                  # preview only (safe)
#   bash restore_all_backups.sh --apply          # actually move + backfill
#   bash restore_all_backups.sh --apply --copy   # copy instead of move (backup preserved)
#   bash restore_all_backups.sh --apply --force-success  # overwrite live .success markers
# =============================================================================

set -u

APPLY=0
COPY=""
FORCE=""

for arg in "$@"; do
  case "$arg" in
    --apply)          APPLY=1 ;;
    --copy)           COPY="--copy" ;;
    --force-success)  FORCE="--force-success" ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown option: $arg"; exit 1 ;;
  esac
done

echo "=============================================="
echo " Backup restore — all scenarios"
echo "=============================================="
echo ""

# --------------------------------------------------------------------------
# Step 0: show what backup dirs exist
# --------------------------------------------------------------------------
BACKUPS=$(ls -d results/_backup_*/ 2>/dev/null | wc -l)
if [ "$BACKUPS" -eq 0 ]; then
  echo "No results/_backup_* directories found. Nothing to restore."
  exit 0
fi

echo "Found $BACKUPS backup scenario dir(s):"
ls -d results/_backup_*/ 2>/dev/null | sed 's|^results/||; s|/$||' | head -20
echo ""

# --------------------------------------------------------------------------
# Step 1: always do a dry run first
# --------------------------------------------------------------------------
echo "=============================================="
echo " Step 1: Dry-run preview"
echo "=============================================="
python restore_backup_results.py $COPY $FORCE
DRY_EXIT=$?
if [ "$DRY_EXIT" -ne 0 ]; then
  echo "[ABORT] Dry run failed with exit code $DRY_EXIT"
  exit $DRY_EXIT
fi

# --------------------------------------------------------------------------
# Step 2: apply if requested
# --------------------------------------------------------------------------
if [ "$APPLY" -eq 0 ]; then
  echo ""
  echo "=============================================="
  echo " DRY RUN ONLY — no files changed"
  echo "=============================================="
  echo ""
  echo "To actually restore:  bash $0 --apply"
  echo "To copy (not move):   bash $0 --apply --copy"
  exit 0
fi

echo ""
echo "=============================================="
echo " Step 2: Applying restore"
echo "=============================================="
python restore_backup_results.py --apply $COPY $FORCE
APPLY_EXIT=$?
if [ "$APPLY_EXIT" -ne 0 ]; then
  echo "[ABORT] Apply failed with exit code $APPLY_EXIT"
  exit $APPLY_EXIT
fi

# --------------------------------------------------------------------------
# Step 3: re-extract analysis CSVs
# --------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Step 3: Re-extracting analysis CSVs"
echo "=============================================="

# Step 8 system scalability extractor
if [ -f "experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py" ]; then
  echo ""
  echo "[Extract] step 8 system scalability..."
  python experiments/gradient_market/visualization/extract/extract_step8_system_scalability.py || \
    echo "  (non-fatal; continuing)"
fi

# System perf extractor (covers step 10 etc.)
if [ -f "extract_system_perf.py" ]; then
  echo ""
  echo "[Extract] system perf across steps 8 + 10..."
  python extract_system_perf.py --steps 10 8 --output tables/system_perf.csv 2>/dev/null || \
    python extract_system_perf.py --steps 10 --output tables/system_perf.csv || \
    echo "  (non-fatal; continuing)"
fi

# Headline step 10 view
if [ -f "check_step10_results.py" ]; then
  echo ""
  echo "[Summary] step 10 headline (top of output only)"
  python check_step10_results.py 2>&1 | head -40 || echo "  (non-fatal)"
fi

echo ""
echo "=============================================="
echo " Done. Verify:"
echo "=============================================="
echo "  - Backup dirs remaining: $(ls -d results/_backup_*/ 2>/dev/null | wc -l)"
echo "  - Live .success markers under restored scenarios:"
for bak in results/_backup_*/; do
  [ -d "$bak" ] || continue
  live_name=$(basename "$bak" | sed 's/^_backup_//')
  succ=$(find "results/$live_name" -name ".success" 2>/dev/null | wc -l)
  echo "      $live_name : $succ .success markers"
done
echo ""
echo "Re-run extractors / plots as needed."
