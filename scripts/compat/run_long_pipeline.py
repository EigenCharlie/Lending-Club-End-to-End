"""Compatibility wrapper for the legacy long pipeline entrypoint."""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main

if __name__ == "__main__":
    raise SystemExit(
        _main(sys.argv[1:], compatibility_entrypoint="scripts/compat/run_long_pipeline.py")
    )
