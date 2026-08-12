#!/usr/bin/env python3
"""Functional test for the seller_metrics.csv column fix.

Feeds REAL per-seller dicts (taken from a valuations.jsonl produced by a live
run) through the actual save_seller_metrics_incremental() and asserts that the
previously-empty columns are now populated and that price_paid survives.

Usage:
  python3 test_seller_metrics_columns.py <path/to/valuations.jsonl> [path/to/run_exp.py]

The optional second argument loads a run_exp.py from an arbitrary path, so the
patched version can be tested WITHOUT overwriting a copy that a live sweep is
currently importing.
"""
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

# Repo root: normally two levels up, but fall back to cwd so the test can be
# staged and run from a scratch directory.
_here = Path(__file__).resolve()
REPO = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "src").is_dir() else Path.cwd()
sys.path.insert(0, str(REPO))

if len(sys.argv) > 2:
    _spec = importlib.util.spec_from_file_location("run_exp_under_test", sys.argv[2])
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
else:
    import experiments.gradient_market.run_exp as _mod  # noqa: E402

SELLER_LOG_ALIASES = _mod.SELLER_LOG_ALIASES
SELLER_LOG_COLUMNS = _mod.SELLER_LOG_COLUMNS
save_seller_metrics_incremental = _mod.save_seller_metrics_incremental

val_path = sys.argv[1]
with open(val_path) as fh:
    entry = json.loads(next(l for l in fh if l.strip()))

# Rebuild detailed_seller_metrics exactly as markplace_gradient.py does.
rows = []
for sid, scores in entry["seller_valuations"].items():
    d = dict(scores)
    d["round"] = entry["round"]
    d["seller_id"] = sid
    rows.append(d)

print("input per-seller keys:", sorted(rows[0].keys()))

with tempfile.TemporaryDirectory() as td:
    save_seller_metrics_incremental(Path(td), {"detailed_seller_metrics": rows})
    out = Path(td) / "seller_metrics.csv"
    with open(out, newline="") as fh:
        got = list(csv.DictReader(fh))

print("\ncolumns written:", list(got[0].keys()))

failures = []
must_be_populated = list(SELLER_LOG_ALIASES) + ["price_paid", "selection_score"]
print("\n{:<22}{:<12}{}".format("column", "status", "sample"))
print("-" * 60)
for col in must_be_populated:
    vals = [r.get(col) for r in got]
    nonempty = [v for v in vals if v not in (None, "")]
    ok = len(nonempty) > 0
    if not ok:
        failures.append(col)
    print("{:<22}{:<12}{}".format(
        col, "OK" if ok else "EMPTY", (nonempty[0] if nonempty else "-")))

# Regression guard: appending must not widen an existing narrower file.
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "seller_metrics.csv"
    p.write_text("round,seller_id,selected\n1,adv_0,True\n")
    save_seller_metrics_incremental(Path(td), {"detailed_seller_metrics": rows})
    hdr = p.read_text().splitlines()[0]
    if hdr != "round,seller_id,selected":
        failures.append("legacy-append-widened-header")
    n_fields = {len(l.split(",")) for l in p.read_text().splitlines()}
    if len(n_fields) != 1:
        failures.append("legacy-append-misaligned-rows")
    print("\nlegacy append: header={!r} field-counts={}".format(hdr, n_fields))

print("\nRESULT:", "PASS" if not failures else "FAIL -> {}".format(failures))
sys.exit(1 if failures else 0)
