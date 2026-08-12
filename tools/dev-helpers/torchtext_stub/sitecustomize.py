"""Dev-machine shim (auto-imported via PYTHONPATH): Python 3.9 compatibility.

The pipeline targets Python 3.10 (`from types import NoneType` in
experiments/gradient_market/automate_exp/config_generator.py). On a 3.9
interpreter that import fails, so backfill the attribute here. Harmless on
3.10+ where the attribute already exists.
"""

import types

if not hasattr(types, "NoneType"):
    types.NoneType = type(None)
