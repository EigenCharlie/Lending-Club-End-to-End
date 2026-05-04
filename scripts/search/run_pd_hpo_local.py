"""Organized entrypoint for local PD HPO refinement on top of the best search winner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.build_pd_hpo_local_config import build_pd_hpo_local_config
from scripts.run_long_pipeline import main as _main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--base-search-run-tag", required=True)
    parser.add_argument("--base-config-path", default="configs/pd_model.champion.yaml")
    parser.add_argument("--output-config-path", default=None)
    parser.add_argument("--hpo-n-trials", type=int, default=120)
    args, passthrough = parser.parse_known_args(argv)

    config_path = build_pd_hpo_local_config(
        run_tag=args.run_tag,
        base_search_run_tag=args.base_search_run_tag,
        base_config_path=args.base_config_path,
        output_path=args.output_config_path,
        n_trials=args.hpo_n_trials,
    )
    repo_root = Path(__file__).resolve().parents[2]
    passthrough_args = list(passthrough)
    if "--pipeline-profile" not in passthrough_args:
        passthrough_args = [
            "--pipeline-profile",
            "search_pd_hpo_local_exhaustive",
            *passthrough_args,
        ]
    forwarded = [
        "--run-tag",
        args.run_tag,
        "--pd-config-override",
        str(Path(config_path).resolve().relative_to(repo_root)),
        *passthrough_args,
    ]
    return _main(
        forwarded,
        default_pipeline_family="search_pd",
        default_sampling_profile="mega64plus",
        default_include_rapids=False,
        default_include_notebooks=False,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
