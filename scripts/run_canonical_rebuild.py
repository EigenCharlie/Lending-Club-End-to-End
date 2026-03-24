"""Canonical operational rebuild entrypoint.

Runs the frozen operational pipeline: no heavy search by default, no RAPIDS or
notebook annexes by default, and metadata explicitly marks the run as
operationally frozen.
"""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        default_pipeline_family="canonical_rebuild",
        default_sampling_profile="champion64safe",
        default_include_rapids=False,
        default_include_notebooks=True,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
