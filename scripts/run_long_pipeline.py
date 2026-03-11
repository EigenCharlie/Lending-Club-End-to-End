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
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_TAG = datetime.now(UTC).strftime("%Y-%m-%d-long-run")
STATUS_SCHEMA_VERSION = "2026-03-01.1"
HEARTBEAT_SCHEMA_VERSION = "2026-03-01.1"
BASELINE_REGISTRY_PATH = REPO_ROOT / "configs" / "baselines" / "core_official_baseline.json"
DEFAULT_STALL_WINDOW_MINUTES = 15
STEP_DEFAULT_SECONDS = {
    "preflight": 5 * 60.0,
    "main_pre": 120 * 60.0,
    "heavy_main": 4.0 * 3600.0,
    "causal": 60 * 60.0,
    "cate_portfolio": 20 * 60.0,
    "post_core": 60 * 60.0,
    "rapids": 2.5 * 3600.0,
    "notebooks": 5.0 * 3600.0,
}
STEP_ORDER = [
    "preflight",
    "main_pre",
    "heavy_main",
    "causal",
    "cate_portfolio",
    "post_core",
    "rapids",
    "notebooks",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StepStatus:
    step: str
    required: bool
    command: str
    started_at_utc: str
    updated_at_utc: str | None = None
    phase: str | None = None
    subphase: str | None = None
    eta_seconds: float | None = None
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


def _resolve_comparison_baseline(
    *,
    baseline_path_arg: str | None,
    baseline_run_tag_arg: str | None,
) -> Path | None:
    if baseline_path_arg and baseline_run_tag_arg:
        raise ValueError(
            "Provide only one of --comparison-baseline or --comparison-baseline-run-tag."
        )
    if baseline_run_tag_arg:
        return (_comparison_dir(str(baseline_run_tag_arg)) / "baseline_snapshot.json").resolve()
    if baseline_path_arg:
        path = Path(str(baseline_path_arg)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    return None


def _resolve_registry_baseline_run_tag() -> str | None:
    if not BASELINE_REGISTRY_PATH.exists():
        return None
    try:
        payload = json.loads(BASELINE_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_tag = str(payload.get("official_run_tag", "")).strip()
    return run_tag or None


def _resolve_registry_baseline_path() -> Path | None:
    run_tag = _resolve_registry_baseline_run_tag()
    if not run_tag:
        return None
    candidate = (_comparison_dir(run_tag) / "baseline_snapshot.json").resolve()
    if not candidate.exists():
        return None
    return candidate


def _run_tag_requires_explicit_baseline(run_tag: str) -> bool:
    tag = str(run_tag).strip().lower()
    if not tag:
        return False
    return ("official" in tag) or ("-core-" in tag) or tag.endswith("-core")


def refresh_baseline_snapshot(run_tag: str) -> bool:
    """Refresh comparison baseline snapshot without rerunning full preflight tests."""
    preferred_python = REPO_ROOT / "lending-club-venv" / "bin" / "python"
    if not preferred_python.exists():
        preferred_python = REPO_ROOT / ".venv" / "bin" / "python"
    python_exec = str(preferred_python) if preferred_python.exists() else sys.executable
    cmd = [python_exec, "scripts/run_comparison.py", "snapshot", "--run-tag", run_tag]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        append_master(run_tag, "BASELINE_SNAPSHOT_REFRESHED on_resume=1")
        return True
    append_master(
        run_tag,
        f"BASELINE_SNAPSHOT_REFRESH_FAILED ec={proc.returncode} stderr={proc.stderr.strip()[:400]}",
    )
    return False


def _subphase_progress_path(run_tag: str, step: str) -> Path:
    return _status_dir(run_tag) / f"{step}.subphases.json"


def _split_step_command(command: str) -> tuple[str, list[str]]:
    """Split a step command into prelude + resumable subcommands."""
    parts = [p.strip() for p in command.split(" && ") if p.strip()]
    if len(parts) <= 1:
        return "", [command.strip()]
    return parts[0], parts[1:]


def _load_subphase_progress(run_tag: str, step: str, subcommands: list[str]) -> set[int]:
    path = _subphase_progress_path(run_tag, step)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    previous_subcommands = payload.get("subcommands", [])
    if previous_subcommands != subcommands:
        return set()
    completed = payload.get("completed_indices", [])
    out: set[int] = set()
    for idx in completed:
        if isinstance(idx, int) and 0 <= idx < len(subcommands):
            out.add(idx)
    return out


def _write_subphase_progress(
    run_tag: str,
    step: str,
    *,
    subcommands: list[str],
    completed_indices: set[int],
) -> None:
    payload = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_tag": run_tag,
        "step": step,
        "updated_at_utc": utc_now_iso(),
        "subcommands": subcommands,
        "completed_indices": sorted(int(i) for i in completed_indices),
        "completed_count": len(completed_indices),
        "total_count": len(subcommands),
    }
    write_json(_subphase_progress_path(run_tag, step), payload)


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
        "schema_version": HEARTBEAT_SCHEMA_VERSION,
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
    payload = asdict(status)
    payload["schema_version"] = STATUS_SCHEMA_VERSION
    payload["phase"] = status.phase or status.step
    payload["subphase"] = status.subphase
    payload["started_at"] = status.started_at_utc
    payload["updated_at"] = status.updated_at_utc or utc_now_iso()
    payload["eta"] = status.eta_seconds
    write_json(json_path, payload)
    if status.exit_code is not None:
        exit_path.write_text(f"{int(status.exit_code)}\n", encoding="utf-8")


def _load_env_file(env_file: str) -> dict[str, str]:
    out: dict[str, str] = {}
    path = Path(env_file).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            try:
                parsed = shlex.split(value)
                value = parsed[0] if parsed else ""
            except ValueError:
                value = value.strip("'").strip('"')
        out[key] = value.strip().strip("'").strip('"')
    return out


def _completed_step_durations(step: str, *, exclude_run_tag: str) -> list[float]:
    vals: list[float] = []
    for status_path in (REPO_ROOT / "reports" / "run_logs").glob(f"*/status/{step}.json"):
        run_tag = status_path.parents[1].name
        if run_tag == exclude_run_tag:
            continue
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dur = data.get("duration_seconds")
        raw_exit_code = data.get("exit_code", -1)
        try:
            exit_code = int(raw_exit_code)
        except (TypeError, ValueError):
            exit_code = -1
        if (
            isinstance(dur, int | float)
            and float(dur) > 0.0
            and exit_code == 0
            and not bool(data.get("skipped"))
        ):
            vals.append(float(dur))
    return vals


def _build_step_eta_defaults(run_tag: str, steps: list[tuple[str, bool, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for step, _required, _command in steps:
        hist = _completed_step_durations(step, exclude_run_tag=run_tag)
        if hist:
            out[step] = float(statistics.median(hist))
        else:
            out[step] = float(STEP_DEFAULT_SECONDS.get(step, 3600.0))
    return out


def _extract_subphase_from_line(step: str, line: str, fallback: str | None) -> str | None:
    if step in {"main_pre", "heavy_main", "causal", "post_core", "notebooks"}:
        m = re.search(r"scripts/[A-Za-z0-9_./-]+\.py", line)
        if m:
            return m.group(0)
    if step == "rapids":
        if "run_all_benchmarks.py" in line:
            return "reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py"
        if "run_rapids_benchmarks.sh" in line:
            return "scripts/side_projects/run_rapids_benchmarks.sh"
    return fallback


def run_step(
    run_tag: str,
    step: str,
    command: str,
    *,
    required: bool,
    step_eta_default_seconds: float | None,
    stall_window_seconds: int,
    resume_subphases: bool,
) -> int:
    run_dir = _run_dir(run_tag)
    log_path = run_dir / f"{step}.log"
    now_iso = utc_now_iso()
    status = StepStatus(
        step=step,
        required=required,
        command=command,
        started_at_utc=now_iso,
        updated_at_utc=now_iso,
        phase=step,
        subphase=None,
        eta_seconds=step_eta_default_seconds,
    )
    write_step_status(run_tag, status)
    append_master(run_tag, f"STEP_START name={step} required={int(required)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("UV_PROJECT_ENVIRONMENT", "lending-club-venv")
    env["PIPELINE_RUN_TAG"] = run_tag

    prelude, subcommands = _split_step_command(command)
    if not resume_subphases:
        _subphase_progress_path(run_tag, step).unlink(missing_ok=True)
        completed_subcommands: set[int] = set()
    else:
        completed_subcommands = _load_subphase_progress(run_tag, step, subcommands)
    if completed_subcommands:
        append_master(
            run_tag,
            f"STEP_SUBPHASE_RESUME name={step} completed={len(completed_subcommands)}/{len(subcommands)}",
        )

    last_hb = 0.0
    stall_warn_emitted = False
    step_start = time.time()
    last_output_at = step_start
    with open(log_path, "a", encoding="utf-8") as logf:
        for idx, raw_subcommand in enumerate(subcommands):
            subphase_label = (
                _extract_subphase_from_line(step=step, line=raw_subcommand, fallback=None)
                or raw_subcommand[:120]
            )
            status.subphase = f"{idx + 1}/{len(subcommands)} {subphase_label}"
            if idx in completed_subcommands:
                append_master(
                    run_tag,
                    f"STEP_SUBPHASE_SKIPPED name={step} idx={idx + 1}/{len(subcommands)} reason=resume_completed_ok",
                )
                continue

            subcommand = raw_subcommand if not prelude else f"{prelude} && {raw_subcommand}"
            append_master(
                run_tag,
                f"STEP_SUBPHASE_START name={step} idx={idx + 1}/{len(subcommands)} cmd={subphase_label}",
            )
            proc = subprocess.Popen(
                _bash_cmd(subcommand),
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
            )
            if proc.stdout is None:
                raise RuntimeError(f"Failed to capture stdout for step={step} subphase={idx + 1}")
            os.set_blocking(proc.stdout.fileno(), False)

            while True:
                line = ""
                try:
                    chunk = proc.stdout.read()
                    if chunk:
                        if isinstance(chunk, bytes):
                            line = chunk.decode("utf-8", errors="replace")
                        else:
                            line = str(chunk)
                except BlockingIOError:
                    line = ""

                if line:
                    for raw_line in line.splitlines(keepends=True):
                        logf.write(raw_line)
                        status.subphase = _extract_subphase_from_line(
                            step=step,
                            line=raw_line,
                            fallback=status.subphase,
                        )
                    logf.flush()
                    last_output_at = time.time()
                now = time.time()
                if now - last_hb >= 10:
                    elapsed = max(0.0, now - step_start)
                    eta_seconds = None
                    if isinstance(step_eta_default_seconds, int | float):
                        eta_seconds = max(0.0, float(step_eta_default_seconds) - elapsed)
                    stalled = bool(now - last_output_at >= float(stall_window_seconds))
                    status.updated_at_utc = utc_now_iso()
                    status.duration_seconds = elapsed
                    status.eta_seconds = eta_seconds
                    write_step_status(run_tag, status)
                    write_heartbeat(
                        run_tag,
                        state="running",
                        current_step=step,
                        orchestrator_pid=os.getpid(),
                        active_child_pid=proc.pid,
                        extra={
                            "current_subphase_index": int(idx + 1),
                            "subphase_count": int(len(subcommands)),
                            "step_elapsed_seconds": round(elapsed, 1),
                            "step_eta_seconds": eta_seconds,
                            "last_output_utc": datetime.fromtimestamp(
                                last_output_at, tz=UTC
                            ).isoformat(),
                            "stalled": stalled,
                            "stall_window_seconds": int(stall_window_seconds),
                        },
                    )
                    if stalled and not stall_warn_emitted:
                        append_master(
                            run_tag,
                            "STEP_STALL_WARNING "
                            f"name={step} no_log_output_for_s={int(now - last_output_at)}",
                        )
                        stall_warn_emitted = True
                    if not stalled:
                        stall_warn_emitted = False
                    last_hb = now
                if proc.poll() is not None:
                    try:
                        tail = proc.stdout.read()
                        if tail:
                            if isinstance(tail, bytes):
                                tail_text = tail.decode("utf-8", errors="replace")
                            else:
                                tail_text = str(tail)
                            logf.write(tail_text)
                            logf.flush()
                    except Exception:
                        pass
                    break
                time.sleep(0.5)

            ec = int(proc.wait())
            if ec != 0:
                append_master(
                    run_tag,
                    f"STEP_SUBPHASE_END name={step} idx={idx + 1}/{len(subcommands)} ec={ec}",
                )
                break
            completed_subcommands.add(idx)
            _write_subphase_progress(
                run_tag,
                step,
                subcommands=subcommands,
                completed_indices=completed_subcommands,
            )
            append_master(
                run_tag,
                f"STEP_SUBPHASE_END name={step} idx={idx + 1}/{len(subcommands)} ec=0",
            )
        else:
            ec = 0

    end = utc_now_iso()
    start_dt = datetime.fromisoformat(status.started_at_utc)
    end_dt = datetime.fromisoformat(end)
    status.updated_at_utc = end
    status.ended_at_utc = end
    status.duration_seconds = (end_dt - start_dt).total_seconds()
    status.eta_seconds = 0.0
    status.exit_code = ec
    write_step_status(run_tag, status)
    if ec == 0:
        _subphase_progress_path(run_tag, step).unlink(missing_ok=True)
    append_master(run_tag, f"STEP_END name={step} ec={ec} duration_s={status.duration_seconds:.1f}")
    return ec


def mark_skipped(run_tag: str, step: str, command: str, *, required: bool, reason: str) -> None:
    now_iso = utc_now_iso()
    status = StepStatus(
        step=step,
        required=required,
        command=command,
        started_at_utc=now_iso,
        updated_at_utc=now_iso,
        ended_at_utc=now_iso,
        phase=step,
        subphase="skipped",
        eta_seconds=0.0,
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
    comparison_baseline: str | None = None,
) -> list[tuple[str, bool, str]]:
    steps: list[tuple[str, bool, str]] = []
    smart_pd_config_exists = (REPO_ROOT / "configs" / "pd_model.smart.yaml").exists()
    champion_pd_config_exists = (REPO_ROOT / "configs" / "pd_model.champion.yaml").exists()
    if sampling_profile in {"champion64safe"} and champion_pd_config_exists:
        pd_config = "configs/pd_model.champion.yaml"
    elif sampling_profile in {"smart", "balanced"} and smart_pd_config_exists:
        pd_config = "configs/pd_model.smart.yaml"
    else:
        pd_config = "configs/pd_model.yaml"
    optimize_portfolio_script = REPO_ROOT / "scripts" / "optimize_portfolio.py"
    optimize_tradeoff_script = REPO_ROOT / "scripts" / "optimize_portfolio_tradeoff.py"
    optimize_portfolio_text = (
        optimize_portfolio_script.read_text(encoding="utf-8")
        if optimize_portfolio_script.exists()
        else ""
    )
    optimize_tradeoff_text = (
        optimize_tradeoff_script.read_text(encoding="utf-8")
        if optimize_tradeoff_script.exists()
        else ""
    )
    optimize_portfolio_has_max_candidates = "--max_candidates" in optimize_portfolio_text

    # Sampling profiles:
    # - smart: sampled data for expensive stages
    # - balanced: full where safe + large sampling where memory-sensitive
    # - full: full data across stages
    if sampling_profile == "smart":
        pd_sample = "--sample_size 500000"
        survival_args = "--sample_size 250000 --rsf_n_estimators 200"
        lgd_ead_sample = "--sample_size 500000"
        optimize_portfolio_candidates = (
            "--max_candidates 10000" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 10000"
        tradeoff_profile = "custom"
        ab_candidates = (
            "--max_candidates 10000 --n_boot 3000 --seed 42 --no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 200000"
        cate_candidates = "--max_candidates 10000"
        rapids_profile = "current"
    elif sampling_profile == "balanced":
        pd_sample = "--sample_size 0"
        survival_args = "--sample_size 250000 --rsf_n_estimators 200"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 20000" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 20000"
        tradeoff_profile = "balanced"
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 20000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 200000"
        cate_candidates = "--max_candidates 20000"
        rapids_profile = "current"
    elif sampling_profile == "mega":
        pd_sample = "--sample_size 0"  # FULL 1.35M
        survival_args = "--full-data --rsf_n_estimators 250"  # FULL + 250 trees (OOM-safe)
        lgd_ead_sample = "--sample_size 0"  # FULL ~50K defaults
        optimize_portfolio_candidates = (
            "--max_candidates 30000" if optimize_portfolio_has_max_candidates else ""
        )  # CAP: Pyomo LP OOM-safe
        tradeoff_candidates = "--max_candidates 30000"
        tradeoff_profile = "night"  # Full grid
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 30000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 500000"  # CAP: CausalForest DML OOM-safe
        cate_candidates = "--max_candidates 30000"
        rapids_profile = "full_data"
    elif sampling_profile == "mega64":
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 100000" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 60000"
        tradeoff_profile = "night"
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 100000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 100000"
        rapids_profile = "full_data"
    elif sampling_profile == "mega64plus":
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 150000" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 80000"
        tradeoff_profile = "night"
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 150000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 150000"
        rapids_profile = "full_data"
    elif sampling_profile == "mega64safe" or sampling_profile == "champion64safe":
        pd_sample = "--sample_size 0"
        survival_args = (
            "--full-data --rsf_n_estimators 200 --rsf_sample_size 500000 "
            "--rsf_max_samples 0.5 --rsf_n_jobs 12"
        )
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 150000" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 80000"
        tradeoff_profile = "night"
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 150000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 150000"
        rapids_profile = "full_data"
    else:  # full
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 0" if optimize_portfolio_has_max_candidates else ""
        )
        tradeoff_candidates = "--max_candidates 0"
        tradeoff_profile = "night"
        ab_candidates = (
            "--max_candidates 0 --n_boot 5000 --seed 42 --no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 0"
        rapids_profile = "full_data"
    optimize_tradeoff_grid = (
        f"--grid-profile {tradeoff_profile}" if "--grid-profile" in optimize_tradeoff_text else ""
    )
    compare_baseline_arg = (
        f" --baseline {shlex.quote(str(comparison_baseline))}" if comparison_baseline else ""
    )

    activate_main = (
        "if [ -f lending-club-venv/bin/activate ]; then source lending-club-venv/bin/activate; "
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
        (uv run python -u scripts/cleanup_optuna_stale_trials.py --db-path models/optuna_pd_catboost.db --min-age-hours 6 || true) &&
        uv run python -u scripts/train_pd_model.py --config {pd_config} {pd_sample} &&
        uv run python -u scripts/generate_conformal_intervals.py &&
        uv run python -u scripts/benchmark_conformal_variants.py &&
        uv run python -u scripts/backtest_conformal_coverage.py &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} &&
        uv run python -u scripts/forecast_default_rates.py --horizon 12
    """
    steps.append(("main_pre", True, main_pre_cmd))

    heavy_main_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/run_survival_analysis.py {survival_args} &&
        uv run python -u scripts/train_lgd_ead.py {lgd_ead_sample} --run-tag {run_tag} &&
        uv run python -u scripts/optimize_portfolio.py --config configs/optimization.yaml {optimize_portfolio_candidates} &&
        uv run python -u scripts/optimize_portfolio_tradeoff.py --config configs/optimization.yaml {tradeoff_candidates} {optimize_tradeoff_grid} &&
        uv run python -u scripts/select_economic_portfolio_policy.py --config configs/optimization.yaml --run-tag {run_tag} &&
        uv run python -u scripts/simulate_ab_test.py {ab_candidates} --run-tag {run_tag} --policy_selector explicit_champion_only &&
        (uv run python -u scripts/log_mlflow_experiment_suite.py || true)
    """
    steps.append(("heavy_main", False, heavy_main_cmd))

    causal_cmd = f"""
        bash scripts/causal/run_causal_pipeline.sh --treatment int_rate {causal_sample} --run_tag {run_tag}
    """
    steps.append(("causal", False, causal_cmd))

    cate_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/optimize_cate_portfolio.py {cate_candidates}
    """
    steps.append(("cate_portfolio", False, cate_cmd))

    post_core_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} &&
        uv run python -u scripts/run_ifrs9_sensitivity.py &&
        uv run python -u scripts/build_pipeline_results.py &&
        if [ -f scripts/build_pd_challenger_artifacts.py ]; then uv run python -u scripts/build_pd_challenger_artifacts.py --config {pd_config}; else true; fi &&
        uv run python -u scripts/run_fairness_audit.py --run-tag {run_tag} &&
        if [ -f scripts/generate_governance_status.py ]; then uv run python -u scripts/generate_governance_status.py --config configs/mrm_policy.yaml --run-tag {run_tag}; else true; fi &&
        if [ -f scripts/update_champion_registry.py ]; then uv run python -u scripts/update_champion_registry.py; else true; fi &&
        uv run python -u scripts/generate_mrm_report.py &&
        uv run python -u scripts/export_streamlit_artifacts.py &&
        uv run python -u scripts/export_storytelling_snapshot.py &&
        uv run python -u scripts/export_dvc_metrics.py --run-tag {run_tag} &&
        uv run python -u scripts/run_comparison.py compare --run-tag {run_tag}{compare_baseline_arg}
    """
    steps.append(("post_core", False, post_core_cmd))

    if include_rapids:
        rapids_headroom_guard = (
            "uv run python -u scripts/ensure_memory_headroom.py "
            "--label rapids --min-mem-gb 6 --min-swap-gb 4 "
            "--min-total-headroom-gb 12 --max-wait-seconds 1800 --poll-seconds 20"
        )
        rapids_cmd = (
            f"{activate_main} && "
            f"{rapids_headroom_guard} && "
            f"bash scripts/side_projects/run_rapids_benchmarks.sh --profile {rapids_profile}"
        )
        steps.append(("rapids", False, rapids_cmd))

    if include_notebooks:
        notebooks_headroom_guard = (
            "uv run python -u scripts/ensure_memory_headroom.py "
            "--label notebooks --min-mem-gb 5 --min-swap-gb 3 "
            "--min-total-headroom-gb 9 --max-wait-seconds 1800 --poll-seconds 20"
        )
        notebooks_cmd = f"""
            {activate_main} &&
            {notebooks_headroom_guard} &&
            uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout 3600 --inplace false --output-dir reports/notebook_exec &&
            uv run python -u scripts/extract_notebook_images.py
        """
        steps.append(("notebooks", False, notebooks_cmd))

    return [(name, req, " ".join(cmd.split())) for name, req, cmd in steps]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable long-run pipeline orchestrator")
    p.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    p.add_argument("--resume", action="store_true", help="Skip already successful steps")
    p.add_argument(
        "--refresh-baseline-on-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resume skips preflight, refresh baseline snapshot to prevent stale comparisons.",
    )
    p.add_argument(
        "--env-file",
        default=None,
        help="Path to .env style file to export before running steps.",
    )
    p.add_argument("--no-rapids", action="store_true")
    p.add_argument("--no-notebooks", action="store_true")
    p.add_argument(
        "--stop-on-optional-failure",
        action="store_true",
        help="Stop immediately if a non-required step fails",
    )
    p.add_argument(
        "--stall-window-minutes",
        type=int,
        default=DEFAULT_STALL_WINDOW_MINUTES,
        help="Emit stall warnings if a step has no log output for this many minutes.",
    )
    p.add_argument(
        "--from-step",
        choices=STEP_ORDER,
        default=None,
        help="Start execution from this step (inclusive).",
    )
    p.add_argument(
        "--until-step",
        choices=STEP_ORDER,
        default=None,
        help="Stop execution at this step (inclusive).",
    )
    p.add_argument(
        "--sampling-profile",
        choices=[
            "full",
            "smart",
            "balanced",
            "mega",
            "mega64",
            "mega64plus",
            "mega64safe",
            "champion64safe",
        ],
        default="full",
        help=(
            "Sampling profile: smart (lighter), balanced (mixed), full, "
            "mega (max data, OOM-safe caps), mega64 (24 threads / ~60GB WSL tuned), "
            "mega64plus (same hardware, more aggressive optimization caps), "
            "mega64safe (recovery profile with survival RSF memory guardrails), "
            "champion64safe (promotion-first rerun using pd_model.champion.yaml)"
        ),
    )
    p.add_argument(
        "--comparison-baseline",
        default=None,
        help=(
            "Optional path to a fixed run_comparison baseline_snapshot.json. "
            "If set, compare uses this baseline instead of run_tag-local snapshot."
        ),
    )
    p.add_argument(
        "--comparison-baseline-run-tag",
        default=None,
        help=(
            "Optional run tag whose baseline snapshot should be used for compare "
            "(reports/run_comparisons/<tag>/baseline_snapshot.json)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_tag = str(args.run_tag)
    comparison_baseline_path = _resolve_comparison_baseline(
        baseline_path_arg=args.comparison_baseline,
        baseline_run_tag_arg=args.comparison_baseline_run_tag,
    )
    comparison_baseline_source = "cli"
    if comparison_baseline_path is None and _run_tag_requires_explicit_baseline(run_tag):
        registry_baseline = _resolve_registry_baseline_path()
        if registry_baseline is not None:
            comparison_baseline_path = registry_baseline
            comparison_baseline_source = "registry_default"
    if _run_tag_requires_explicit_baseline(run_tag) and comparison_baseline_path is None:
        raise ValueError(
            "Core/official runs require explicit comparison baseline. "
            "Use --comparison-baseline or --comparison-baseline-run-tag."
        )
    if comparison_baseline_path is not None and not comparison_baseline_path.exists():
        raise FileNotFoundError(
            f"Comparison baseline snapshot not found: {comparison_baseline_path}"
        )
    comparison_baseline = (
        str(comparison_baseline_path) if comparison_baseline_path is not None else None
    )
    if args.env_file:
        env_kv = _load_env_file(str(args.env_file))
        os.environ.update(env_kv)
        append_master(run_tag, f"ENV_FILE_LOADED path={args.env_file} keys={len(env_kv)}")

    run_dir = _run_dir(run_tag)
    status_dir = _status_dir(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "run_info.json",
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_tag": run_tag,
            "started_at_utc": utc_now_iso(),
            "argv": sys.argv,
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "pid": os.getpid(),
            "resume": bool(args.resume),
            "refresh_baseline_on_resume": bool(args.refresh_baseline_on_resume),
            "include_rapids": not bool(args.no_rapids),
            "include_notebooks": not bool(args.no_notebooks),
            "sampling_profile": str(args.sampling_profile),
            "env_file": str(args.env_file) if args.env_file else None,
            "from_step": str(args.from_step) if args.from_step else None,
            "until_step": str(args.until_step) if args.until_step else None,
            "comparison_baseline_path": comparison_baseline,
            "comparison_baseline_run_tag": str(args.comparison_baseline_run_tag)
            if args.comparison_baseline_run_tag
            else None,
            "comparison_baseline_source": comparison_baseline_source,
            "stall_window_minutes": int(args.stall_window_minutes),
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
        comparison_baseline=comparison_baseline,
    )
    if args.from_step or args.until_step:
        names = [s for s, _req, _cmd in steps]
        start_idx = names.index(args.from_step) if args.from_step else 0
        end_idx = names.index(args.until_step) if args.until_step else len(steps) - 1
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid step range: from-step={args.from_step} occurs after until-step={args.until_step}"
            )
        steps = steps[start_idx : end_idx + 1]

    step_eta_defaults = _build_step_eta_defaults(run_tag, steps)
    stall_window_seconds = max(60, int(args.stall_window_minutes) * 60)
    failed_required = False
    stopped_on_optional_failure = False
    failed_steps: list[str] = []

    for step, required, command in steps:
        if args.resume and load_completed_ok(run_tag, step):
            if (
                step == "preflight"
                and bool(args.refresh_baseline_on_resume)
                and comparison_baseline is None
            ):
                if not refresh_baseline_snapshot(run_tag):
                    failed_required = True
                    failed_steps.append(step)
                    append_master(run_tag, "RUN_ABORT baseline_snapshot_refresh_failed=1")
                    break
                mark_skipped(
                    run_tag,
                    step,
                    command,
                    required=required,
                    reason="resume_completed_ok_refresh_snapshot",
                )
                continue
            if step == "preflight" and comparison_baseline is not None:
                mark_skipped(
                    run_tag,
                    step,
                    command,
                    required=required,
                    reason="resume_completed_ok_external_comparison_baseline",
                )
                continue
            mark_skipped(run_tag, step, command, required=required, reason="resume_completed_ok")
            continue

        ec = run_step(
            run_tag,
            step,
            command,
            required=required,
            step_eta_default_seconds=step_eta_defaults.get(step),
            stall_window_seconds=stall_window_seconds,
            resume_subphases=bool(args.resume),
        )
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
        "schema_version": STATUS_SCHEMA_VERSION,
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
