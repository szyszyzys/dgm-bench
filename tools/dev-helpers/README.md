# Dev Helpers (not required for reproduction)

These scripts were used during development for monitoring, recovery, and
ad-hoc debugging. They are **not** part of the reproduction pipeline — see
the top-level `run_main_all_datasets.sh` for that.

Kept for reference / re-use:

| Script | Purpose |
|---|---|
| `run_all_step10.sh` | Earlier wrapper around step 10; superseded by `run_main_all_datasets.sh` (tier P0) |
| `run_step10_then_step8.sh` | Sequential step10 → step8 wrapper to avoid the dispatcher kill bug |
| `restore_all_backups.sh`, `restore_backup_results.py` | Move `results/_backup_*` cells back to the live tree |
| `backfill_and_mark_success.py` | Reconstruct `final_metrics.json` + `.success` from per-round artifacts |
| `find_missing_agg_stats.py` | List cells lacking `agg_stats/round_*.json` |
| `check_femnist_step3.py`, `check_seller_dropout.py`, `check_coverage_matrix.py` | One-off sanity checks |
| `watch_health.py`, `watch_running.py`, `watch_step10.sh`, `watch_steps.sh` | Live monitoring of GPU / dispatcher state |

If you encounter a half-finished sweep, the recovery flow is:
```bash
python tools/dev-helpers/find_missing_agg_stats.py --prefix step10_scalability
python tools/dev-helpers/backfill_and_mark_success.py --prefix step10_scalability --apply
```
