"""Canonical champion-search entrypoint.

This entrypoint reuses the resumable long-run orchestration backend while
persisting champion-search semantics in metadata and documentation.
"""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        default_pipeline_family="champion_search",
        default_sampling_profile="mega64plus",
        default_include_rapids=True,
        default_include_notebooks=True,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
