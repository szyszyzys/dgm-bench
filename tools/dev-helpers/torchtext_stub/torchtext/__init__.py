"""Minimal import-time stub for torchtext (dev helper, NOT a runtime replacement).

The real pipeline pins torchtext==0.17.0+cpu (see requirements.txt), but
torchtext is discontinued and incompatible with newer torch builds on dev
machines. `src.common_utils.__init__` imports the text pipeline eagerly, so
even image-only runs fail to import without torchtext installed.

This stub provides just enough surface for those imports to succeed. Any
actual *use* of the stubbed APIs (i.e. running a text-dataset experiment)
raises immediately. Enable it by prepending this directory to PYTHONPATH:

    PYTHONPATH=tools/dev-helpers/torchtext_stub python -m experiments.gradient_market.run_exp <config>

Do NOT use this on machines where the real torchtext is installed.
"""
