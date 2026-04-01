"""Organized paper entrypoint for Paper 1 end-to-end runs."""

from __future__ import annotations

import sys

from scripts.run_paper_grade_final import main as _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
