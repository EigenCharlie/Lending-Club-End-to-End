"""Compatibility shim for the old smoke/core entrypoint.

The previous implementation embedded an outdated mini end-to-end pipeline with
hardcoded sample sizes and research lanes. The active smoke path now delegates
to the pipeline-first orchestrator using the `core_canonical` family and the
`smoke` sampling profile.
"""

from __future__ import annotations

import argparse

from scripts.run_long_pipeline import main as _long_main


def _argv_from_legacy_api(
    *,
    run_name: str,
    continue_on_error: bool,
    skip_make_dataset: bool,
) -> list[str]:
    argv = [
        "--run-tag",
        str(run_name),
        "--pipeline-family",
        "core_canonical",
        "--sampling-profile",
        "smoke",
        "--no-rapids",
        "--no-notebooks",
    ]
    if not continue_on_error:
        argv.append("--stop-on-optional-failure")
    if skip_make_dataset:
        # No exact equivalent exists in the pipeline-first orchestrator. We keep
        # the flag for API compatibility and let the caller decide whether to
        # resume from a later step in the modern interface.
        argv.extend(["--resume"])
    return argv


def main(
    run_name: str = "smoke",
    continue_on_error: bool = False,
    skip_make_dataset: bool = False,
) -> int:
    argv = _argv_from_legacy_api(
        run_name=run_name,
        continue_on_error=continue_on_error,
        skip_make_dataset=skip_make_dataset,
    )
    return _long_main(
        argv,
        default_pipeline_family="core_canonical",
        default_sampling_profile="smoke",
        default_include_rapids=False,
        default_include_notebooks=False,
        compatibility_entrypoint="scripts/end_to_end_pipeline.py",
    )


if __name__ == "__main__":
    print(
        "[deprecated] scripts/end_to_end_pipeline.py is a compatibility entrypoint. "
        "Use scripts/run_smoke_pipeline.py instead."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="smoke")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-make-dataset", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        main(
            run_name=args.run_name,
            continue_on_error=args.continue_on_error,
            skip_make_dataset=args.skip_make_dataset,
        )
    )
