"""Build canonical pipeline_results artifact from stage outputs.

Usage:
    uv run python scripts/build_pipeline_results.py
"""

from __future__ import annotations

from scripts.end_to_end_pipeline import _persist_pipeline_results


def main() -> None:
    _persist_pipeline_results()


if __name__ == "__main__":
    main()
