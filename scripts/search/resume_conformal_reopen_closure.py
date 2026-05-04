"""Derived resume entrypoint for conformal reopen closure."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.search.run_conformal_reopen_search import main as reopen_main  # noqa: E402


def _default_derived_tag(source_run_tag: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M")
    return f"{str(source_run_tag).strip()}__resume__{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-tag", required=True)
    parser.add_argument("--derived-run-tag", default=None)
    parser.add_argument("--upstream-canonical-run-tag", required=True)
    parser.add_argument("--pipeline-profile", default="search_conformal_reopen_exhaustive")
    parser.add_argument("--top-k-finalists", type=int, default=3)
    parser.add_argument(
        "--phase2-enabled",
        default="true",
        choices=["true", "false"],
    )
    args = parser.parse_args(argv)

    derived_run_tag = (
        str(args.derived_run_tag).strip()
        if args.derived_run_tag
        else _default_derived_tag(args.source_run_tag)
    )
    reopen_args = [
        "--run-tag",
        derived_run_tag,
        "--pipeline-profile",
        str(args.pipeline_profile),
        "--upstream-canonical-run-tag",
        str(args.upstream_canonical_run_tag),
        "--resume-from-run-tag",
        str(args.source_run_tag),
    ]
    # top-k is governed by the profile validation config; this script exposes it
    # for parity with the recovery protocol even though the current profile default is 3.
    if int(args.top_k_finalists) != 3:
        raise ValueError("top-k-finalists is fixed to 3 in the current recovery protocol.")
    if str(args.phase2_enabled).strip().lower() == "false":
        reopen_args.append("--phase1-only")
    return reopen_main(reopen_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
