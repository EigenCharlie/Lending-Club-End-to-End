"""Compatibility wrapper for the legacy end_to_end_pipeline entrypoint."""

from __future__ import annotations

from scripts.end_to_end_pipeline import main as _main

if __name__ == "__main__":
    raise SystemExit(_main())
