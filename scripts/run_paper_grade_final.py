"""Paper-grade final heavy run entrypoint."""

from __future__ import annotations

import sys

from scripts.run_long_pipeline import main as _main
from src.utils.baseline_registry import resolve_official_baseline_run_tag


def _inject_default_args(argv: list[str] | None = None) -> list[str]:
    args = list(argv or [])
    if "--pipeline-profile" not in args:
        args.extend(["--pipeline-profile", "paper_grade_final"])
    if "--comparison-baseline" not in args and "--comparison-baseline-run-tag" not in args:
        baseline_run_tag = resolve_official_baseline_run_tag()
        if baseline_run_tag:
            args.extend(["--comparison-baseline-run-tag", baseline_run_tag])
    return args


def main(argv: list[str] | None = None) -> int:
    return _main(
        _inject_default_args(argv),
        default_pipeline_family="paper1_e2e",
        default_sampling_profile="mega64plus",
        default_include_rapids=False,
        default_include_notebooks=False,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
