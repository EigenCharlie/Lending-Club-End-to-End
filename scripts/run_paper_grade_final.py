"""Paper-grade final heavy run entrypoint."""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main


def _inject_default_args(argv: list[str] | None = None) -> list[str]:
    args = list(argv or [])
    if "--pipeline-profile" not in args:
        args.extend(["--pipeline-profile", "paper_grade_final"])
    if "--comparison-baseline" not in args and "--comparison-baseline-run-tag" not in args:
        args.extend(
            [
                "--comparison-baseline-run-tag",
                "champion-2026-03-12-mega-definitive",
            ]
        )
    return args


def main(argv: list[str] | None = None) -> int:
    return _main(
        _inject_default_args(argv),
        default_pipeline_family="champion_search",
        default_sampling_profile="mega64plus",
        default_include_rapids=True,
        default_include_notebooks=True,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
