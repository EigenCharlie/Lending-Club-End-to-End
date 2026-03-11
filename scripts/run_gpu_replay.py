"""Replay GPU-eligible pipeline stages against the current workspace artifacts.

This script is meant to run immediately after a completed CPU baseline so the
workspace still contains the exact upstream artifacts produced by that run.
It reruns only the heavy stages where GPU backends are meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

STAGE_ORDER = [
    "pd",
    "lgd_ead",
    "portfolio",
    "tradeoff",
    "policy_selection",
    "ab",
    "cate_portfolio",
    "ifrs9_mc",
]
RAPIDS_STAGES = {"portfolio", "tradeoff", "policy_selection", "ab", "cate_portfolio", "ifrs9_mc"}

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
    "rapids_final": {
        "portfolio_candidates": 0,
        "tradeoff_candidates": 150_000,
        "ab_candidates": 150_000,
        "cate_candidates": 0,
        "ifrs9_mc_scenarios": 8192,
        "ifrs9_mc_chunk_size": 256,
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


def _resolve_rapids_python() -> Path:
    proc = subprocess.run(
        ["conda", "info", "--envs", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Unable to resolve RAPIDS env path: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout or "{}")
    envs = [Path(p) for p in payload.get("envs", [])]
    for env_path in envs:
        if env_path.name == "rapids":
            python_path = env_path / "bin" / "python"
            if python_path.exists():
                return python_path
    raise RuntimeError("Could not find a usable python binary for conda env 'rapids'.")


def build_stage_commands(
    *,
    run_tag: str,
    profile: str,
    pd_config: str,
    optimization_config: str,
    rapids_python: str | None = None,
) -> dict[str, str]:
    profile_cfg = PROFILE_CONFIGS[profile]
    rapids_python = rapids_python or "python"
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
            f"{shlex.quote(rapids_python)} -u -m scripts.optimize_portfolio "
            f"--config {shlex.quote(optimization_config)} "
            f"--max_candidates {profile_cfg['portfolio_candidates']} --solver_backend cuopt"
        ),
        "tradeoff": (
            f"{shlex.quote(rapids_python)} -u -m scripts.optimize_portfolio_tradeoff "
            f"--config {shlex.quote(optimization_config)} "
            f"--max_candidates {profile_cfg['tradeoff_candidates']} "
            "--grid-profile night --solver_backend cuopt"
        ),
        "policy_selection": (
            f"{shlex.quote(rapids_python)} -u -m scripts.select_economic_portfolio_policy "
            f"--config {shlex.quote(optimization_config)} "
            f"--run-tag {shlex.quote(run_tag)} --solver_backend cuopt"
        ),
        "ab": (
            f"{shlex.quote(rapids_python)} -u -m scripts.simulate_ab_test --max_portfolio_pd 0.18 "
            f"--max_candidates {profile_cfg['ab_candidates']} "
            "--n_boot 5000 --seed 42 --no_regression_tolerance_pct 0.05 "
            f"--run-tag {shlex.quote(run_tag)} --solver_backend cuopt "
            "--policy_selector explicit_champion_only"
        ),
        "cate_portfolio": (
            f"{shlex.quote(rapids_python)} -u -m scripts.optimize_cate_portfolio "
            f"--max_candidates {profile_cfg['cate_candidates']} --solver_backend cuopt"
        ),
        "ifrs9_mc": (
            f"{shlex.quote(rapids_python)} -u scripts/run_ifrs9_monte_carlo_gpu.py "
            f"--n-scenarios {int(profile_cfg.get('ifrs9_mc_scenarios', 8192))} "
            f"--chunk-size {int(profile_cfg.get('ifrs9_mc_chunk_size', 256))}"
        ),
    }


def validate_rapids_env(
    *,
    selected_stages: list[str],
    rapids_python: Path | None = None,
) -> dict[str, Any]:
    needs_rapids = any(stage in RAPIDS_STAGES for stage in selected_stages)
    if not needs_rapids:
        return {"checked": False, "needs_rapids": False}
    if rapids_python is None:
        rapids_python = _resolve_rapids_python()

    proc = subprocess.run(
        [
            str(rapids_python),
            "-c",
            "import json, cuopt, cudf, cupy, pyomo, loguru; "
            "print(json.dumps({'python': __import__('sys').executable, 'ok': True}))",
        ],
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "RAPIDS env validation failed for replay stages "
            f"{sorted(RAPIDS_STAGES & set(selected_stages))}: stderr={stderr}"
        )
    return {
        "checked": True,
        "needs_rapids": True,
        "missing": [],
        "python": str(rapids_python),
        "stdout": stdout,
        "stderr": stderr,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class _GpuSampler:
    def __init__(self, output_csv: Path, interval_seconds: float = 1.0) -> None:
        self.output_csv = output_csv
        self.interval_seconds = max(0.25, float(interval_seconds))
        self._stop = Event()
        self._thread: Thread | None = None
        self.samples: list[dict[str, float | str]] = []

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if not self.samples:
            return {"available": False, "sample_count": 0, "csv_path": str(self.output_csv)}

        with self.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "gpu_util",
                    "memory_used_mb",
                    "memory_total_mb",
                    "power_draw_w",
                ],
            )
            writer.writeheader()
            writer.writerows(self.samples)

        gpu_utils = [float(s["gpu_util"]) for s in self.samples]
        memory_used = [float(s["memory_used_mb"]) for s in self.samples]
        power = [float(s["power_draw_w"]) for s in self.samples]
        return {
            "available": True,
            "sample_count": len(self.samples),
            "csv_path": str(self.output_csv),
            "peak_gpu_util": max(gpu_utils),
            "avg_gpu_util": sum(gpu_utils) / len(gpu_utils),
            "peak_memory_used_mb": max(memory_used),
            "avg_memory_used_mb": sum(memory_used) / len(memory_used),
            "peak_power_draw_w": max(power),
            "avg_power_draw_w": sum(power) / len(power),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
                if lines:
                    first = [part.strip() for part in lines[0].split(",")]
                    if len(first) >= 5:
                        with suppress(ValueError):
                            self.samples.append(
                                {
                                    "timestamp": first[0],
                                    "gpu_util": float(first[1]),
                                    "memory_used_mb": float(first[2]),
                                    "memory_total_mb": float(first[3]),
                                    "power_draw_w": float(first[4]),
                                }
                            )
            self._stop.wait(self.interval_seconds)


def _build_stage_env(
    *,
    base_env: dict[str, str],
    run_tag: str,
    baseline_run_tag: str,
    artifact_root: Path,
    stage: str,
) -> dict[str, str]:
    env = dict(base_env)
    env["PIPELINE_RUN_TAG"] = run_tag
    env["GPU_REPLAY_BASELINE_RUN_TAG"] = baseline_run_tag
    env["GPU_REPLAY_ARTIFACT_ROOT"] = str(artifact_root)
    env["GPU_REPLAY_STAGE"] = stage
    return env


def _build_rapids_env(
    *,
    base_env: dict[str, str],
    run_tag: str,
    baseline_run_tag: str,
    artifact_root: Path,
    stage: str,
) -> dict[str, str]:
    env = _build_stage_env(
        base_env=base_env,
        run_tag=run_tag,
        baseline_run_tag=baseline_run_tag,
        artifact_root=artifact_root,
        stage=stage,
    )
    for key in [
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONNOUSERSITE",
    ]:
        env.pop(key, None)
    return env


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
    rapids_python = _resolve_rapids_python()
    commands = build_stage_commands(
        run_tag=args.run_tag,
        profile=args.profile,
        pd_config=args.pd_config,
        optimization_config=args.optimization_config,
        rapids_python=str(rapids_python),
    )
    rapids_validation = validate_rapids_env(
        selected_stages=selected_stages,
        rapids_python=rapids_python,
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
        "rapids_env_validation": rapids_validation,
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

    base_env = os.environ.copy()
    artifact_root = run_dir / "artifacts"

    stage_results: list[dict[str, Any]] = []
    for stage in selected_stages:
        cmd = commands[stage]
        log_path = run_dir / f"{stage}.log"
        stage_env = (
            _build_rapids_env(
                base_env=base_env,
                run_tag=args.run_tag,
                baseline_run_tag=args.baseline_run_tag,
                artifact_root=artifact_root,
                stage=stage,
            )
            if stage in RAPIDS_STAGES
            else _build_stage_env(
                base_env=base_env,
                run_tag=args.run_tag,
                baseline_run_tag=args.baseline_run_tag,
                artifact_root=artifact_root,
                stage=stage,
            )
        )
        started = time.perf_counter()
        sampler = _GpuSampler(run_dir / f"{stage}_gpu_metrics.csv")
        sampler.start()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {cmd}\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=stage_env,
                text=True,
            )
        duration = time.perf_counter() - started
        gpu_metrics = sampler.stop()
        stage_results.append(
            {
                "stage": stage,
                "command": cmd,
                "exit_code": int(proc.returncode),
                "duration_seconds": round(duration, 3),
                "log_path": str(log_path),
                "gpu_metrics": gpu_metrics,
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
            sampler = _GpuSampler(run_dir / f"{stage}_gpu_metrics.csv")
            sampler.start()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {cmd}\n\n")
                log_file.flush()
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=_build_stage_env(
                        base_env=base_env,
                        run_tag=args.run_tag,
                        baseline_run_tag=args.baseline_run_tag,
                        artifact_root=artifact_root,
                        stage=stage,
                    ),
                    text=True,
                )
            duration = time.perf_counter() - started
            gpu_metrics = sampler.stop()
            post_results.append(
                {
                    "stage": stage,
                    "command": cmd,
                    "exit_code": int(proc.returncode),
                    "duration_seconds": round(duration, 3),
                    "log_path": str(log_path),
                    "gpu_metrics": gpu_metrics,
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
