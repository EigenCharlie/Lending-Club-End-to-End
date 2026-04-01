"""Historical compatibility wrapper for the archived pre-Quarto helper."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "history" / "update_cost_matrix_threshold.py"),
        run_name="__main__",
    )
