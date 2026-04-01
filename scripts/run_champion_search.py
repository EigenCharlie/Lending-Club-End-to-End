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
        default_pipeline_family="search_pd",
        default_sampling_profile="mega64plus",
        default_include_rapids=False,
        default_include_notebooks=False,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
