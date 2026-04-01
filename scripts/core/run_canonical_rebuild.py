"""Organized core entrypoint wrapper."""

from __future__ import annotations

import sys

from scripts.run_canonical_rebuild import main as _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
