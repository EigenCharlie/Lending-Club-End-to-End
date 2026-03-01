"""Resumable long-run orchestrator for end-to-end full-data experiments.

Designed for multi-day runs on a single workstation/WSL instance. It writes:
- per-step logs
- per-step exit codes / JSON status
- a heartbeat JSON for terminal monitoring

The process itself can be launched with `nohup` and resumed after interruptions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_TAG = datetime.now(UTC).strftime("%Y-%m-%d-long-run")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StepStatus:
    step: str
    required: bool
    command: str
    started_at_utc: str
    ended_at_utc: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _bash_cmd(raw: str) -> list[str]:
    return ["bash", "-lc", raw]


def _run_dir(run_tag: str) -> Path:
    return REPO_ROOT / "reports" / "run_logs" / run_tag


def _status_dir(run_tag: str) -> Path:
    return _run_dir(run_tag) / "status"


def _comparison_dir(run_tag: str) -> Path:
    return REPO_ROOT / "reports" / "run_comparisons" / run_tag


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_master(run_tag: str, message: str) -> None:
    run_dir = _run_dir(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().astimezone().isoformat()}] {message}\n"
    with open(run_dir / "master.log", "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def write_heartbeat(
    run_tag: str,
    *,
    state: str,
    current_step: str | None,
    orchestrator_pid: int,
    active_child_pid: int | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "schema_version": "2026-02-26.1",
        "run_tag": run_tag,
        "last_update_utc": utc_now_iso(),
        "state": state,
        "current_step": current_step,
        "orchestrator_pid": orchestrator_pid,
        "active_child_pid": active_child_pid,
    }
    if extra:
        payload.update(extra)
    write_json(_run_dir(run_tag) / "heartbeat.json", payload)


def step_status_paths(run_tag: str, step: str) -> tuple[Path, Path]:
    status_dir = _status_dir(run_tag)
    return status_dir / f"{step}.json", status_dir / f"{step}.exit"


def load_completed_ok(run_tag: str, step: str) -> bool:
    json_path, exit_path = step_status_paths(run_tag, step)
    if not json_path.exists() or not exit_path.exists():
        return False
    try:
        status = json.loads(json_path.read_text(encoding="utf-8"))
        ec = int(exit_path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    return ec == 0 and bool(status.get("exit_code") == 0)


def write_step_status(run_tag: str, status: StepStatus) -> None:
    json_path, exit_path = step_status_paths(run_tag, status.step)
    write_json(json_path, asdict(status))
    if status.exit_code is not None:
        exit_path.write_text(f"{int(status.exit_code)}\n", encoding="utf-8")


def run_step(run_tag: str, step: str, command: str, *, required: bool) -> int:
    run_dir = _run_dir(run_tag)
    log_path = run_dir / f"{step}.log"
    status = StepStatus(
        step=step,
        required=required,
        command=command,
        started_at_utc=utc_now_iso(),
    )
    write_step_status(run_tag, status)
    append_master(run_tag, f"STEP_START name={step} required={int(required)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("UV_PROJECT_ENVIRONMENT", ".venv")

    proc = subprocess.Popen(
        _bash_cmd(command),
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_hb = 0.0
    with open(log_path, "a", encoding="utf-8") as logf:
        while True:
            line = proc.stdout.readline() if proc.stdout is not None else ""
            if line:
                logf.write(line)
                logf.flush()
            now = time.time()
            if now - last_hb >= 10:
                write_heartbeat(
                    run_tag,
                    state="running",
                    current_step=step,
                    orchestrator_pid=os.getpid(),
                    active_child_pid=proc.pid,
                )
                last_hb = now
            if not line and proc.poll() is not None:
                break

    ec = int(proc.wait())
    end = utc_now_iso()
    start_dt = datetime.fromisoformat(status.started_at_utc)
    end_dt = datetime.fromisoformat(end)
    status.ended_at_utc = end
    status.duration_seconds = (end_dt - start_dt).total_seconds()
    status.exit_code = ec
    write_step_status(run_tag, status)
    append_master(run_tag, f"STEP_END name={step} ec={ec} duration_s={status.duration_seconds:.1f}")
    return ec


def mark_skipped(run_tag: str, step: str, command: str, *, required: bool, reason: str) -> None:
    status = StepStatus(
        step=step,
        required=required,
        command=command,
        started_at_utc=utc_now_iso(),
        ended_at_utc=utc_now_iso(),
        duration_seconds=0.0,
        exit_code=0,
        skipped=True,
        skip_reason=reason,
    )
    write_step_status(run_tag, status)
    append_master(run_tag, f"STEP_SKIPPED name={step} reason={reason}")


def build_steps(
    run_tag: str,
    *,
    include_rapids: bool,
    include_notebooks: bool,
    sampling_profile: str = "full",
) -> list[tuple[str, bool, str]]:
    steps: list[tuple[str, bool, str]] = []

    # Sampling profile: "smart" uses reduced data for expensive stages (~5x faster).
    if sampling_profile == "smart":
        pd_sample = "--sample_size 500000"
        survival_args = "--sample_size 250000 --rsf_n_estimators 200"
        tradeoff_candidates = "--max_candidates 10000"
        tradeoff_profile = "custom"
        ab_candidates = "--max_candidates 10000"
        causal_sample = "--sample_size 200000"
        cate_candidates = "--max_candidates 10000"
    else:  # full
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        tradeoff_candidates = "--max_candidates 0"
        tradeoff_profile = "night"
        ab_candidates = "--max_candidates 0"
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 0"

    activate_main = (
        "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; "
        "elif [ -f lending-club-venv/bin/activate ]; then source lending-club-venv/bin/activate; fi"
    )
    activate_causal = (
        "if [ -f .venv-causal/bin/activate ]; then source .venv-causal/bin/activate; "
        "elif [ -f lending-club-venv/bin/activate ]; then source lending-club-venv/bin/activate; "
        "elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi"
    )
    preflight_cmd = (
        f"{activate_main} && "
        "python -m pytest -q tests/test_docs tests/test_streamlit/test_page_imports.py "
        "tests/test_config_consistency.py && "
        f"python scripts/run_comparison.py snapshot --run-tag {run_tag}"
    )
    steps.append(("preflight", True, preflight_cmd))

    main_pre_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/train_pd_model.py --config configs/pd_model.yaml {pd_sample} &&
        uv run python -u scripts/generate_conformal_intervals.py &&
        uv run python -u scripts/benchmark_conformal_variants.py &&
        uv run python -u scripts/backtest_conformal_coverage.py &&
        uv run python -u scripts/validate_conformal_policy.py &&
        uv run python -u scripts/forecast_default_rates.py --horizon 12
    """
    steps.append(("main_pre", True, main_pre_cmd))

    heavy_main_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/run_survival_analysis.py {survival_args} &&
        uv run python -u scripts/train_lgd_ead.py --sample_size 0 &&
        uv run python -u scripts/optimize_portfolio.py --config configs/optimization.yaml --max_candidates 0 --solver_backend highs &&
        uv run python -u scripts/optimize_portfolio_tradeoff.py --config configs/optimization.yaml {tradeoff_candidates} --grid-profile {tradeoff_profile} --solver_backend highs &&
        uv run python -u scripts/simulate_ab_test.py {ab_candidates} &&
        uv run python -u scripts/log_mlflow_experiment_suite.py
    """
    steps.append(("heavy_main", False, heavy_main_cmd))

    causal_cmd = f"""
        {activate_causal} &&
        python -u -c "import econml,dowhy; print('econml', econml.__version__, 'dowhy', dowhy.__version__)" &&
        python -u scripts/estimate_causal_effects.py --treatment int_rate {causal_sample} &&
        python -u scripts/simulate_causal_policy.py &&
        python -u scripts/backtest_causal_policy_oot.py
    """
    steps.append(("causal", False, causal_cmd))

    cate_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/optimize_cate_portfolio.py {cate_candidates}
    """
    steps.append(("cate_portfolio", False, cate_cmd))

    post_core_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/run_ifrs9_sensitivity.py &&
        uv run python -u scripts/build_pipeline_results.py &&
        uv run python -u scripts/build_pd_challenger_artifacts.py --config configs/pd_model.yaml &&
        uv run python -u scripts/run_fairness_audit.py &&
        uv run python -u scripts/validate_causal_policy.py &&
        uv run python -u scripts/generate_governance_status.py --config configs/mrm_policy.yaml &&
        uv run python -u scripts/generate_mrm_report.py &&
        uv run python -u scripts/export_streamlit_artifacts.py &&
        uv run python -u scripts/export_storytelling_snapshot.py &&
        uv run python -u scripts/export_dvc_metrics.py &&
        uv run python -u scripts/run_comparison.py compare --run-tag {run_tag}
    """
    steps.append(("post_core", False, post_core_cmd))

    if include_rapids:
        rapids_cmd = (
            "conda run --no-capture-output -n rapids "
            "bash scripts/side_projects/run_rapids_benchmarks.sh --profile full_data"
        )
        steps.append(("rapids", False, rapids_cmd))

    if include_notebooks:
        notebooks_cmd = f"""
            {activate_main} &&
            uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout 3600 --inplace false --output-dir reports/notebook_exec &&
            uv run python -u scripts/run_paper_notebook_suite.py &&
            uv run python -u scripts/extract_notebook_images.py
        """
        steps.append(("notebooks", False, notebooks_cmd))

    return [(name, req, " ".join(cmd.split())) for name, req, cmd in steps]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable long-run pipeline orchestrator")
    p.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    p.add_argument("--resume", action="store_true", help="Skip already successful steps")
    p.add_argument("--no-rapids", action="store_true")
    p.add_argument("--no-notebooks", action="store_true")
    p.add_argument(
        "--stop-on-optional-failure",
        action="store_true",
        help="Stop immediately if a non-required step fails",
    )
    p.add_argument(
        "--sampling-profile",
        choices=["full", "smart"],
        default="full",
        help="'smart' uses reduced sampling for survival/tradeoff/causal (~5x faster)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_tag = str(args.run_tag)
    run_dir = _run_dir(run_tag)
    status_dir = _status_dir(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "run_info.json",
        {
            "schema_version": "2026-02-27.1",
            "run_tag": run_tag,
            "started_at_utc": utc_now_iso(),
            "argv": sys.argv,
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "pid": os.getpid(),
            "resume": bool(args.resume),
            "include_rapids": not bool(args.no_rapids),
            "include_notebooks": not bool(args.no_notebooks),
            "sampling_profile": str(args.sampling_profile),
        },
    )

    append_master(run_tag, f"RUN_START pid={os.getpid()} resume={int(args.resume)}")
    write_heartbeat(
        run_tag,
        state="starting",
        current_step=None,
        orchestrator_pid=os.getpid(),
    )

    steps = build_steps(
        run_tag,
        include_rapids=not bool(args.no_rapids),
        include_notebooks=not bool(args.no_notebooks),
        sampling_profile=str(args.sampling_profile),
    )
    failed_required = False
    stopped_on_optional_failure = False
    failed_steps: list[str] = []

    for step, required, command in steps:
        if args.resume and load_completed_ok(run_tag, step):
            mark_skipped(run_tag, step, command, required=required, reason="resume_completed_ok")
            continue

        ec = run_step(run_tag, step, command, required=required)
        if ec != 0:
            failed_steps.append(step)
            if required:
                failed_required = True
                append_master(run_tag, f"RUN_ABORT required_step_failed={step}")
                break
            if args.stop_on_optional_failure:
                stopped_on_optional_failure = True
                append_master(run_tag, f"RUN_ABORT optional_step_failed={step}")
                break

    final_ec = 1 if (failed_required or stopped_on_optional_failure) else 0
    summary = {
        "schema_version": "2026-02-27.1",
        "run_tag": run_tag,
        "ended_at_utc": utc_now_iso(),
        "failed_required": failed_required,
        "stopped_on_optional_failure": stopped_on_optional_failure,
        "failed_steps": failed_steps,
        "final_exit_code": final_ec,
    }
    write_json(run_dir / "run_summary.json", summary)
    (status_dir / "overall.exit").write_text(f"{final_ec}\n", encoding="utf-8")
    write_heartbeat(
        run_tag,
        state="finished" if final_ec == 0 else "failed",
        current_step=None,
        orchestrator_pid=os.getpid(),
        extra={"failed_steps": failed_steps, "final_exit_code": final_ec},
    )
    append_master(run_tag, f"RUN_END final_ec={final_ec} failed_steps={failed_steps}")
    return final_ec


if __name__ == "__main__":
    raise SystemExit(main())
