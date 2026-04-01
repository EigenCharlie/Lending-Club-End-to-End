"""Organized search entrypoint for PD/challenger runs."""

from __future__ import annotations

import sys

from scripts.run_champion_search import main as _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
