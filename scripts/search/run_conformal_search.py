"""Organized search entrypoint for conformal benchmarking/tuning."""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        default_pipeline_family="search_conformal",
        default_sampling_profile="champion64safe",
        default_include_rapids=False,
        default_include_notebooks=False,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
