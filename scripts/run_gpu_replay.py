"""Replay GPU-eligible pipeline stages against the current workspace artifacts.

This script is meant to run immediately after a completed CPU baseline so the
workspace still contains the exact upstream artifacts produced by that run.
It reruns only the heavy stages where GPU backends are meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGE_ORDER = ["pd", "lgd_ead", "portfolio", "tradeoff", "ab", "cate_portfolio"]

PROFILE_CONFIGS: dict[str, dict[str, int]] = {
    "mega64": {
        "portfolio_candidates": 100_000,
        "tradeoff_candidates": 60_000,
        "ab_candidates": 100_000,
        "cate_candidates": 100_000,
    },
    "mega64plus": {
        "portfolio_candidates": 150_000,
        "tradeoff_candidates": 80_000,
        "ab_candidates": 150_000,
        "cate_candidates": 150_000,
    },
}


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _normalize_stages(raw: str) -> list[str]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts or parts == ["all"]:
        return list(STAGE_ORDER)
    unknown = sorted(set(parts) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown GPU replay stages: {', '.join(unknown)}")
    return [stage for stage in STAGE_ORDER if stage in parts]


def build_stage_commands(
    *,
    run_tag: str,
    profile: str,
    pd_config: str,
    optimization_config: str,
) -> dict[str, str]:
    profile_cfg = PROFILE_CONFIGS[profile]
    return {
        "pd": (
            "uv run python -u scripts/train_pd_model.py "
            f"--config {shlex.quote(pd_config)} --sample_size 0"
        ),
        "lgd_ead": (
            "uv run python -u scripts/train_lgd_ead.py --sample_size 0 "
            f"--run-tag {shlex.quote(run_tag)} --catboost_backend gpu"
        ),
        "portfolio": (
            "uv run python -u scripts/optimize_portfolio.py "
            f"--config {shlex.quote(optimization_config)} "
            f"--max_candidates {profile_cfg['portfolio_candidates']} --solver_backend cuopt"
        ),
        "tradeoff": (
            "uv run python -u scripts/optimize_portfolio_tradeoff.py "
            f"--config {shlex.quote(optimization_config)} "
            f"--max_candidates {profile_cfg['tradeoff_candidates']} "
            "--grid-profile night --solver_backend cuopt"
        ),
        "ab": (
            "uv run python -u scripts/simulate_ab_test.py --max_portfolio_pd 0.18 "
            f"--max_candidates {profile_cfg['ab_candidates']} "
            "--n_boot 5000 --seed 42 --no_regression_tolerance_pct 0.05 "
            f"--run-tag {shlex.quote(run_tag)} --solver_backend cuopt"
        ),
        "cate_portfolio": (
            "uv run python -u scripts/optimize_cate_portfolio.py "
            f"--max_candidates {profile_cfg['cate_candidates']} --solver_backend cuopt"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_post_replay_commands(
    *,
    notebook_timeout: int,
    notebook_output_dir: str,
    notebook_inplace: bool,
    include_side_projects: bool,
    extract_images_after: bool,
) -> list[tuple[str, str]]:
    side_projects_flag = " --include-side-projects" if include_side_projects else ""
    inplace_value = "true" if notebook_inplace else "false"
    commands: list[tuple[str, str]] = [
        (
            "notebooks",
            "uv run python -u scripts/run_all_notebooks.py "
            f"--execute-all{side_projects_flag} --timeout {int(notebook_timeout)} "
            f"--inplace {inplace_value} --output-dir {shlex.quote(notebook_output_dir)}",
        )
    ]
    if extract_images_after:
        commands.append(
            (
                "extract_images",
                "uv run python -u scripts/extract_notebook_images.py "
                f"--notebook-dir {shlex.quote(str(Path(notebook_output_dir) / 'notebooks'))}",
            )
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay GPU-eligible stages after a CPU baseline run."
    )
    parser.add_argument("--baseline-run-tag", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="mega64plus")
    parser.add_argument("--stages", default="all")
    parser.add_argument("--pd-config", default="configs/pd_model.gpu.yaml")
    parser.add_argument("--optimization-config", default="configs/optimization.yaml")
    parser.add_argument("--run-notebooks-after", action="store_true")
    parser.add_argument("--notebook-timeout", type=int, default=3600)
    parser.add_argument("--notebook-output-dir", default="reports/notebook_exec")
    parser.add_argument("--notebook-inplace", action="store_true", default=True)
    parser.add_argument("--no-notebook-inplace", action="store_false", dest="notebook_inplace")
    parser.add_argument("--include-side-projects", action="store_true")
    parser.add_argument("--extract-images-after", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_stages = _normalize_stages(args.stages)
    commands = build_stage_commands(
        run_tag=args.run_tag,
        profile=args.profile,
        pd_config=args.pd_config,
        optimization_config=args.optimization_config,
    )

    run_dir = Path("reports/gpu_replay") / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run_info.json"
    summary_path = run_dir / "run_summary.json"

    payload = {
        "schema_version": "2026-03-08.1",
        "run_tag": args.run_tag,
        "baseline_run_tag": args.baseline_run_tag,
        "profile": args.profile,
        "selected_stages": selected_stages,
        "pd_config": args.pd_config,
        "optimization_config": args.optimization_config,
        "run_notebooks_after": bool(args.run_notebooks_after),
        "notebook_timeout": int(args.notebook_timeout),
        "notebook_output_dir": args.notebook_output_dir,
        "notebook_inplace": bool(args.notebook_inplace),
        "include_side_projects": bool(args.include_side_projects),
        "extract_images_after": bool(args.extract_images_after),
        "started_at_utc": _utc_now(),
        "state": "planned" if args.dry_run else "running",
        "note": (
            "This replay uses the current workspace artifacts. Run it immediately after the CPU "
            "baseline you want to compare against."
        ),
    }
    _write_json(info_path, payload)

    if args.dry_run:
        _write_json(
            summary_path,
            {
                **payload,
                "state": "dry_run",
                "commands": {stage: commands[stage] for stage in selected_stages},
                "post_replay_commands": build_post_replay_commands(
                    notebook_timeout=args.notebook_timeout,
                    notebook_output_dir=args.notebook_output_dir,
                    notebook_inplace=bool(args.notebook_inplace),
                    include_side_projects=bool(args.include_side_projects),
                    extract_images_after=bool(args.extract_images_after),
                )
                if args.run_notebooks_after
                else [],
                "ended_at_utc": _utc_now(),
            },
        )
        return 0

    env = os.environ.copy()
    env["PIPELINE_RUN_TAG"] = args.run_tag
    env["GPU_REPLAY_BASELINE_RUN_TAG"] = args.baseline_run_tag

    stage_results: list[dict[str, Any]] = []
    for stage in selected_stages:
        cmd = commands[stage]
        log_path = run_dir / f"{stage}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {cmd}\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
        duration = time.perf_counter() - started
        stage_results.append(
            {
                "stage": stage,
                "command": cmd,
                "exit_code": int(proc.returncode),
                "duration_seconds": round(duration, 3),
                "log_path": str(log_path),
            }
        )
        if proc.returncode != 0:
            _write_json(
                summary_path,
                {
                    **payload,
                    "state": "failed",
                    "ended_at_utc": _utc_now(),
                    "stage_results": stage_results,
                    "failed_stage": stage,
                    "final_exit_code": int(proc.returncode),
                },
            )
            return int(proc.returncode)

    post_results: list[dict[str, Any]] = []
    if args.run_notebooks_after:
        for stage, cmd in build_post_replay_commands(
            notebook_timeout=args.notebook_timeout,
            notebook_output_dir=args.notebook_output_dir,
            notebook_inplace=bool(args.notebook_inplace),
            include_side_projects=bool(args.include_side_projects),
            extract_images_after=bool(args.extract_images_after),
        ):
            log_path = run_dir / f"{stage}.log"
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {cmd}\n\n")
                log_file.flush()
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
            duration = time.perf_counter() - started
            post_results.append(
                {
                    "stage": stage,
                    "command": cmd,
                    "exit_code": int(proc.returncode),
                    "duration_seconds": round(duration, 3),
                    "log_path": str(log_path),
                }
            )
            if proc.returncode != 0:
                _write_json(
                    summary_path,
                    {
                        **payload,
                        "state": "failed",
                        "ended_at_utc": _utc_now(),
                        "stage_results": stage_results,
                        "post_replay_results": post_results,
                        "failed_stage": stage,
                        "final_exit_code": int(proc.returncode),
                    },
                )
                return int(proc.returncode)

    _write_json(
        summary_path,
        {
            **payload,
            "state": "completed",
            "ended_at_utc": _utc_now(),
            "stage_results": stage_results,
            "post_replay_results": post_results,
            "final_exit_code": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
