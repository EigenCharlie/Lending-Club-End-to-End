"""Compatibility long-run orchestrator.

This module now backs the new `run_champion_search.py` entrypoint and remains
available as a deprecated compatibility path for `run_long_pipeline.py`.

Designed for multi-day runs on a single workstation/WSL instance. It writes:
- per-step logs
- per-step exit codes / JSON status
- a heartbeat JSON for terminal monitoring

The process itself can be launched with `nohup` and resumed after interruptions.
"""

from __future__ import annotations

import argparse
import inspect
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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_TAG = datetime.now(UTC).strftime("%Y-%m-%d-long-run")
STATUS_SCHEMA_VERSION = "2026-03-01.1"
HEARTBEAT_SCHEMA_VERSION = "2026-03-01.1"
PRIMARY_BASELINE_REGISTRY_PATH = (
    REPO_ROOT / "configs" / "baselines" / "canonical_operational_baseline.json"
)
LEGACY_BASELINE_REGISTRY_PATH = REPO_ROOT / "configs" / "baselines" / "core_official_baseline.json"
BASELINE_REGISTRY_PATH = PRIMARY_BASELINE_REGISTRY_PATH
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

PIPELINE_PROFILE_DEFAULTS = {
    "champion_search": "champion_search_max",
    "canonical_rebuild": "canonical_operational",
}

PIPELINE_CONTRACT_DEFAULTS = {
    "champion_search": {
        "artifact_scope": "search",
        "promotion_state": "research_open",
        "writes_canonical_artifacts": False,
    },
    "canonical_rebuild": {
        "artifact_scope": "canonical",
        "promotion_state": "operationally_frozen",
        "writes_canonical_artifacts": True,
    },
}


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


def _baseline_registry_paths() -> list[Path]:
    paths = [BASELINE_REGISTRY_PATH, PRIMARY_BASELINE_REGISTRY_PATH, LEGACY_BASELINE_REGISTRY_PATH]
    deduped: list[Path] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


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


def _resolve_comparison_baseline_from_profile(profile_cfg: dict[str, object]) -> Path | None:
    defaults_cfg = dict(_profile_value(profile_cfg, "defaults", default={}) or {})
    baseline_path = str(defaults_cfg.get("comparison_baseline", "") or "").strip() or None
    baseline_run_tag = (
        str(defaults_cfg.get("comparison_baseline_run_tag", "") or "").strip() or None
    )
    return _resolve_comparison_baseline(
        baseline_path_arg=baseline_path,
        baseline_run_tag_arg=baseline_run_tag,
    )


def _extract_sqlite_db_path_from_pd_config(config_path: str | None) -> Path | None:
    if not config_path:
        return None
    path = Path(str(config_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    storage = str((payload.get("hpo", {}) or {}).get("study_storage", "")).strip()
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return None
    db_path = Path(storage[len(prefix) :].strip()).expanduser()
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    return db_path.resolve()


def _resolve_registry_baseline_run_tag() -> str | None:
    for path in _baseline_registry_paths():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_tag = str(payload.get("official_run_tag", "")).strip()
        if run_tag:
            return run_tag
    return None


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
    return (
        ("official" in tag)
        or ("-core-" in tag)
        or tag.endswith("-core")
        or tag.startswith("canonical-")
        or tag.startswith("champion-")
        or tag.startswith("insights-")
    )


def _resolve_upstream_canonical_run_tag(
    *,
    pipeline_family: str,
    upstream_run_tag_arg: str | None,
) -> str | None:
    explicit = str(upstream_run_tag_arg or "").strip()
    if explicit:
        return explicit
    if pipeline_family == "canonical_rebuild":
        return _resolve_registry_baseline_run_tag()
    return _resolve_registry_baseline_run_tag()


def _derive_pipeline_contract(
    *,
    pipeline_family: str,
    pipeline_profile_arg: str | None,
    sampling_profile: str,
    writes_canonical_artifacts_arg: bool | None,
    upstream_canonical_run_tag: str | None,
) -> dict[str, object]:
    family = str(pipeline_family).strip().lower() or "champion_search"
    defaults = dict(
        PIPELINE_CONTRACT_DEFAULTS.get(family, PIPELINE_CONTRACT_DEFAULTS["champion_search"])
    )
    pipeline_profile = str(pipeline_profile_arg or "").strip() or PIPELINE_PROFILE_DEFAULTS.get(
        family, sampling_profile
    )
    if family == "champion_search" and not pipeline_profile_arg:
        pipeline_profile = (
            "champion_search_max"
            if sampling_profile in {"mega", "mega64", "mega64plus", "mega64safe", "champion64safe"}
            else pipeline_profile
        )
    if writes_canonical_artifacts_arg is not None:
        defaults["writes_canonical_artifacts"] = bool(writes_canonical_artifacts_arg)
    return {
        "pipeline_family": family,
        "pipeline_profile": pipeline_profile,
        "artifact_scope": str(defaults["artifact_scope"]),
        "promotion_state": str(defaults["promotion_state"]),
        "writes_canonical_artifacts": bool(defaults["writes_canonical_artifacts"]),
        "upstream_canonical_run_tag": upstream_canonical_run_tag,
    }


def _load_pipeline_profile_config(profile_name: str | None) -> dict[str, object]:
    name = str(profile_name or "").strip()
    if not name:
        return {}
    path = REPO_ROOT / "configs" / "run_profiles" / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _profile_value(profile_cfg: dict[str, object], *path: str, default: object = None) -> object:
    current: object = profile_cfg
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def _csv_cli(values: object) -> str:
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values)
    return str(values)


def _resolve_conda_env_python(env_name: str) -> Path | None:
    proc = subprocess.run(
        ["conda", "info", "--envs", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return None
    envs = [Path(p) for p in payload.get("envs", [])]
    for env_path in envs:
        if env_path.name != env_name:
            continue
        python_path = env_path / "bin" / "python"
        if python_path.exists():
            return python_path
    return None


def _resolve_rapids_python_cmd(profile_cfg: dict[str, object]) -> str:
    env_name = str(
        _profile_value(profile_cfg, "python_envs", "rapids_conda", default="rapids")
    ).strip()
    env_name = env_name or "rapids"
    python_path = _resolve_conda_env_python(env_name)
    if python_path is not None:
        return shlex.quote(str(python_path))
    return f"conda run -n {shlex.quote(env_name)} python"


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
    pipeline_contract: dict[str, object] | None = None,
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
    if pipeline_contract:
        env["PIPELINE_FAMILY"] = str(pipeline_contract.get("pipeline_family", ""))
        env["PIPELINE_PROFILE"] = str(pipeline_contract.get("pipeline_profile", ""))
        env["PIPELINE_ARTIFACT_SCOPE"] = str(pipeline_contract.get("artifact_scope", ""))
        env["PIPELINE_PROMOTION_STATE"] = str(pipeline_contract.get("promotion_state", ""))
        env["WRITES_CANONICAL_ARTIFACTS"] = (
            "1" if bool(pipeline_contract.get("writes_canonical_artifacts", False)) else "0"
        )
        upstream = str(pipeline_contract.get("upstream_canonical_run_tag", "") or "").strip()
        if upstream:
            env["UPSTREAM_CANONICAL_RUN_TAG"] = upstream

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
    pipeline_family: str = "champion_search",
    profile_cfg: dict[str, object] | None = None,
) -> list[tuple[str, bool, str]]:
    steps: list[tuple[str, bool, str]] = []
    profile_cfg = dict(profile_cfg or {})
    optimization_cfg_path = REPO_ROOT / "configs" / "optimization.yaml"
    selection_cfg: dict[str, object] = {}
    if optimization_cfg_path.exists():
        try:
            selection_cfg = dict(
                (yaml.safe_load(optimization_cfg_path.read_text(encoding="utf-8")) or {}).get(
                    "portfolio_selection", {}
                )
                or {}
            )
        except Exception:
            selection_cfg = {}

    search_cfg = dict(_profile_value(profile_cfg, "search_space", default={}) or {})
    resource_policy = dict(_profile_value(profile_cfg, "resource_policy", default={}) or {})
    defaults_cfg = dict(_profile_value(profile_cfg, "defaults", default={}) or {})
    pd_search_cfg = dict(search_cfg.get("pd", {}) or {})
    conformal_search_cfg = dict(search_cfg.get("conformal", {}) or {})
    survival_search_cfg = dict(search_cfg.get("survival", {}) or {})
    portfolio_search_cfg = dict(search_cfg.get("portfolio", {}) or {})
    tradeoff_search_cfg = dict(search_cfg.get("tradeoff", {}) or {})
    ab_search_cfg = dict(search_cfg.get("ab", {}) or {})
    causal_search_cfg = dict(search_cfg.get("causal", {}) or {})
    cate_search_cfg = dict(search_cfg.get("cate_portfolio", {}) or {})
    rapids_search_cfg = dict(search_cfg.get("rapids", {}) or {})
    notebook_search_cfg = dict(search_cfg.get("notebooks", {}) or {})

    smart_pd_config_exists = (REPO_ROOT / "configs" / "pd_model.smart.yaml").exists()
    champion_pd_config_exists = (REPO_ROOT / "configs" / "pd_model.champion.yaml").exists()
    champion_search_pd_config = str(pd_search_cfg.get("config_path", "")).strip()
    if (
        pipeline_family == "champion_search"
        and champion_search_pd_config
        and (REPO_ROOT / champion_search_pd_config).exists()
    ):
        pd_config = champion_search_pd_config
    elif champion_pd_config_exists and (
        pipeline_family == "canonical_rebuild" or sampling_profile in {"champion64safe"}
    ):
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

    if sampling_profile == "smart":
        pd_sample = "--sample_size 500000"
        survival_args = "--sample_size 250000 --rsf_n_estimators 200"
        lgd_ead_sample = "--sample_size 500000"
        optimize_portfolio_candidates = (
            "--max_candidates 10000" if optimize_portfolio_has_max_candidates else ""
        )
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
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 20000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 200000"
        cate_candidates = "--max_candidates 20000"
        rapids_profile = "current"
    elif sampling_profile == "mega":
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 250"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 30000" if optimize_portfolio_has_max_candidates else ""
        )
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 30000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 500000"
        cate_candidates = "--max_candidates 30000"
        rapids_profile = "full_data"
    elif sampling_profile == "mega64":
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 100000" if optimize_portfolio_has_max_candidates else ""
        )
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
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 150000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 150000"
        rapids_profile = "full_data"
    elif sampling_profile in {"mega64safe", "champion64safe"}:
        pd_sample = "--sample_size 0"
        survival_args = (
            "--full-data --rsf_n_estimators 200 --rsf_sample_size 500000 "
            "--rsf_max_samples 0.5 --rsf_n_jobs 12"
        )
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 150000" if optimize_portfolio_has_max_candidates else ""
        )
        ab_candidates = (
            "--max_portfolio_pd 0.18 --max_candidates 150000 --n_boot 5000 --seed 42 "
            "--no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 150000"
        rapids_profile = "full_data"
    else:
        pd_sample = "--sample_size 0"
        survival_args = "--full-data --rsf_n_estimators 300"
        lgd_ead_sample = "--sample_size 0"
        optimize_portfolio_candidates = (
            "--max_candidates 0" if optimize_portfolio_has_max_candidates else ""
        )
        ab_candidates = (
            "--max_candidates 0 --n_boot 5000 --seed 42 --no_regression_tolerance_pct 0.05"
        )
        causal_sample = "--sample_size 0"
        cate_candidates = "--max_candidates 0"
        rapids_profile = "full_data"

    if survival_search_cfg:
        survival_parts = []
        if bool(survival_search_cfg.get("full_data", True)):
            survival_parts.append("--full-data")
        sample_size = survival_search_cfg.get("sample_size")
        if sample_size is not None:
            survival_parts.append(f"--sample_size {int(sample_size)}")
        for key in (
            "rsf_n_estimators",
            "rsf_max_depth",
            "rsf_min_samples_leaf",
            "rsf_sample_size",
            "rsf_n_jobs",
        ):
            value = survival_search_cfg.get(key)
            if value is not None:
                survival_parts.append(f"--{key} {value}")
        if survival_search_cfg.get("rsf_max_samples") is not None:
            survival_parts.append(f"--rsf_max_samples {survival_search_cfg['rsf_max_samples']}")
        survival_args = " ".join(survival_parts)

    if (
        portfolio_search_cfg.get("max_candidates") is not None
        and optimize_portfolio_has_max_candidates
    ):
        optimize_portfolio_candidates = (
            f"--max_candidates {int(portfolio_search_cfg['max_candidates'])}"
        )
    if ab_search_cfg:
        ab_candidates = (
            f"--max_portfolio_pd {float(ab_search_cfg.get('max_portfolio_pd', 0.18))} "
            f"--max_candidates {int(ab_search_cfg.get('max_candidates', 150000))} "
            f"--n_boot {int(ab_search_cfg.get('n_boot', 5000))} "
            f"--seed {int(ab_search_cfg.get('seed', 42))} "
            f"--no_regression_tolerance_pct {float(ab_search_cfg.get('no_regression_tolerance_pct', 0.05))}"
        )
    if cate_search_cfg.get("max_candidates") is not None:
        cate_candidates = f"--max_candidates {int(cate_search_cfg['max_candidates'])}"
    if causal_search_cfg.get("sample_size") is not None:
        causal_sample = f"--sample_size {int(causal_search_cfg['sample_size'])}"
    if defaults_cfg.get("rapids_profile"):
        rapids_profile = str(defaults_cfg["rapids_profile"]).strip() or rapids_profile

    canonical_execution_mode = (
        str(
            os.environ.get(
                "PORTFOLIO_SELECTION_EXECUTION_MODE",
                selection_cfg.get("canonical_execution_mode", "search"),
            )
        )
        .strip()
        .lower()
    )
    frozen_policy_path = str(
        selection_cfg.get("frozen_champion_policy_path", "models/champion_portfolio_policy.json")
    ).strip()
    use_frozen_policy = bool(
        (
            pipeline_family == "canonical_rebuild"
            or canonical_execution_mode == "freeze_if_available"
        )
        and (REPO_ROOT / frozen_policy_path).exists()
        and (
            pipeline_family == "canonical_rebuild"
            or sampling_profile in {"mega64safe", "champion64safe"}
        )
    )
    if pipeline_family == "champion_search":
        use_frozen_policy = False

    selection_grid_profile = str(
        tradeoff_search_cfg.get(
            "grid_profile",
            selection_cfg.get("selection_grid_profile", "quick"),
        )
    ).strip()
    selection_max_candidates = int(
        tradeoff_search_cfg.get(
            "max_candidates",
            selection_cfg.get("selection_max_candidates", 20000),
        )
    )
    selection_tradeoff_candidates = (
        f"--max_candidates {selection_max_candidates}"
        if selection_max_candidates > 0
        else "--max_candidates 0"
    )
    selection_tradeoff_grid_parts: list[str] = []
    if "--grid-profile" in optimize_tradeoff_text:
        selection_tradeoff_grid_parts.append(f"--grid-profile {selection_grid_profile}")
    if tradeoff_search_cfg.get("risk_grid") is not None and "--risk_grid" in optimize_tradeoff_text:
        selection_tradeoff_grid_parts.append(
            f"--risk_grid {shlex.quote(_csv_cli(tradeoff_search_cfg['risk_grid']))}"
        )
    if (
        tradeoff_search_cfg.get("aversion_grid") is not None
        and "--aversion_grid" in optimize_tradeoff_text
    ):
        selection_tradeoff_grid_parts.append(
            f"--aversion_grid {shlex.quote(_csv_cli(tradeoff_search_cfg['aversion_grid']))}"
        )
    selection_tradeoff_grid = " ".join(selection_tradeoff_grid_parts)
    compare_baseline_arg = (
        f" --baseline {shlex.quote(str(comparison_baseline))}" if comparison_baseline else ""
    )

    activate_main = (
        "if [ -f lending-club-venv/bin/activate ]; then source lending-club-venv/bin/activate; "
        "elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi"
    )
    rapids_required = bool(
        include_rapids
        or str(resource_policy.get("portfolio_backend", "")).strip().lower() == "cuopt"
        or str(resource_policy.get("tradeoff_backend", "")).strip().lower() == "cuopt"
        or str(resource_policy.get("selector_backend", "")).strip().lower() == "cuopt"
        or str(resource_policy.get("ab_backend", "")).strip().lower() == "cuopt"
        or str(resource_policy.get("cate_backend", "")).strip().lower() == "cuopt"
    )
    rapids_python_cmd = _resolve_rapids_python_cmd(profile_cfg) if rapids_required else "python"
    rapids_module_cmd = f"{rapids_python_cmd} -u -m"
    rapids_script_cmd = f"{rapids_python_cmd} -u"
    rapids_validate_cmd = ""
    if rapids_required:
        rapids_validate_cmd = (
            f"{rapids_python_cmd} -c "
            + shlex.quote("import cuopt, cuml, cupy; print('rapids_ok')")
            + " && "
        )

    pd_train_cmd = f"uv run python -u scripts/train_pd_model.py --config {pd_config} {pd_sample}"
    conformal_cmd_parts = ["uv run python -u scripts/generate_conformal_intervals.py"]
    if conformal_search_cfg:
        if conformal_search_cfg.get("alpha_target_90") is not None:
            conformal_cmd_parts.append(
                f"--alpha_target_90 {float(conformal_search_cfg['alpha_target_90'])}"
            )
        if conformal_search_cfg.get("alpha_95") is not None:
            conformal_cmd_parts.append(f"--alpha_95 {float(conformal_search_cfg['alpha_95'])}")
        if conformal_search_cfg.get("alpha_candidates_90") is not None:
            conformal_cmd_parts.append(
                f"--alpha_candidates_90 {shlex.quote(_csv_cli(conformal_search_cfg['alpha_candidates_90']))}"
            )
        if conformal_search_cfg.get("min_group_sizes") is not None:
            conformal_cmd_parts.append(
                f"--min_group_sizes {shlex.quote(_csv_cli(conformal_search_cfg['min_group_sizes']))}"
            )
        for key in (
            "min_group_coverage_target",
            "group_coverage_floor_target_90",
            "max_width_budget_90",
            "coverage_guardband_90",
            "min_group_guardband_90",
            "tuning_holdout_ratio",
            "tuning_random_state",
            "temporal_segment_min_size",
            "global_rebalance_min_factor",
            "global_rebalance_max_factor",
            "global_rebalance_step",
        ):
            if conformal_search_cfg.get(key) is not None:
                conformal_cmd_parts.append(f"--{key} {conformal_search_cfg[key]}")
        if conformal_search_cfg.get("temporal_segment_floor_enabled") is not None:
            conformal_cmd_parts.append(
                "--temporal_segment_floor_enabled "
                + ("1" if bool(conformal_search_cfg["temporal_segment_floor_enabled"]) else "0")
            )
        if conformal_search_cfg.get("temporal_segment_freq") is not None:
            conformal_cmd_parts.append(
                f"--temporal_segment_freq {shlex.quote(str(conformal_search_cfg['temporal_segment_freq']))}"
            )
        if conformal_search_cfg.get("global_rebalance_enabled") is not None:
            conformal_cmd_parts.append(
                "--global_rebalance_enabled "
                + ("1" if bool(conformal_search_cfg["global_rebalance_enabled"]) else "0")
            )
    conformal_cmd = " ".join(conformal_cmd_parts)

    benchmark_cmd = "uv run python -u scripts/benchmark_conformal_variants.py"
    min_group_size_default = conformal_search_cfg.get("benchmark_min_group_size_default")
    if min_group_size_default is None and conformal_search_cfg.get("min_group_sizes") is not None:
        min_group_size_default = min(conformal_search_cfg["min_group_sizes"])
    if min_group_size_default is not None:
        benchmark_cmd += f" --min_group_size_default {int(min_group_size_default)}"

    lgd_backend = str(resource_policy.get("lgd_ead_backend", "cpu")).strip().lower()
    lgd_backend_arg = " --catboost_backend gpu" if lgd_backend.startswith("gpu") else ""
    portfolio_backend = str(resource_policy.get("portfolio_backend", "highs")).strip().lower()
    tradeoff_backend = str(resource_policy.get("tradeoff_backend", "highs")).strip().lower()
    selector_backend = str(resource_policy.get("selector_backend", "highs")).strip().lower()
    ab_backend = str(resource_policy.get("ab_backend", "highs")).strip().lower()
    cate_backend = str(resource_policy.get("cate_backend", "highs")).strip().lower()
    portfolio_exec = rapids_module_cmd if portfolio_backend == "cuopt" else "uv run python -u -m"
    tradeoff_exec = rapids_module_cmd if tradeoff_backend == "cuopt" else "uv run python -u -m"
    selector_exec = rapids_module_cmd if selector_backend == "cuopt" else "uv run python -u -m"
    ab_exec = rapids_module_cmd if ab_backend == "cuopt" else "uv run python -u -m"
    cate_exec = rapids_module_cmd if cate_backend == "cuopt" else "uv run python -u -m"
    portfolio_solver_arg = (
        f" --solver_backend {portfolio_backend}" if portfolio_backend in {"highs", "cuopt"} else ""
    )
    tradeoff_solver_arg = (
        f" --solver_backend {tradeoff_backend}" if tradeoff_backend in {"highs", "cuopt"} else ""
    )
    selector_solver_arg = (
        f" --solver_backend {selector_backend}" if selector_backend in {"highs", "cuopt"} else ""
    )
    ab_solver_arg = f" --solver_backend {ab_backend}" if ab_backend in {"highs", "cuopt"} else ""
    cate_solver_arg = (
        f" --solver_backend {cate_backend}" if cate_backend in {"highs", "cuopt"} else ""
    )

    heavy_headroom_guard = (
        "uv run python -u scripts/ensure_memory_headroom.py "
        "--label heavy_main --min-mem-gb 8 --min-swap-gb 6 "
        "--min-total-headroom-gb 16 --max-wait-seconds 3600 --poll-seconds 20"
    )
    causal_headroom_guard = (
        "uv run python -u scripts/ensure_memory_headroom.py "
        "--label causal --min-mem-gb 8 --min-swap-gb 6 "
        "--min-total-headroom-gb 16 --max-wait-seconds 3600 --poll-seconds 20"
    )

    preflight_cmd = (
        f"{activate_main} && "
        f"{rapids_validate_cmd}"
        "python -m pytest -q tests/test_docs tests/test_streamlit/test_page_imports.py "
        "tests/test_config_consistency.py && "
        f"python scripts/run_comparison.py snapshot --run-tag {run_tag}"
    )
    steps.append(("preflight", True, preflight_cmd))

    optuna_db_cleanup_path = _extract_sqlite_db_path_from_pd_config(pd_config)
    optuna_cleanup_cmd = "true"
    if optuna_db_cleanup_path is not None:
        optuna_cleanup_cmd = (
            "uv run python -u scripts/cleanup_optuna_stale_trials.py "
            f"--db-path {shlex.quote(str(optuna_db_cleanup_path))} --min-age-hours 6 || true"
        )

    main_pre_cmd = f"""
        {activate_main} &&
        ({optuna_cleanup_cmd}) &&
        {pd_train_cmd} &&
        {conformal_cmd} &&
        {benchmark_cmd} &&
        uv run python -u scripts/backtest_conformal_coverage.py &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} --sensitivity-config configs/conformal_policy_sensitivity.yaml &&
        uv run python -u scripts/forecast_default_rates.py --horizon 12
    """
    steps.append(("main_pre", True, main_pre_cmd))

    portfolio_cmd = (
        f"{portfolio_exec} scripts.optimize_portfolio --config configs/optimization.yaml "
        f"{optimize_portfolio_candidates}{portfolio_solver_arg}"
    )
    tradeoff_cmd = (
        f"{tradeoff_exec} scripts.optimize_portfolio_tradeoff --config configs/optimization.yaml "
        f"{selection_tradeoff_candidates} {selection_tradeoff_grid}{tradeoff_solver_arg}"
    )
    selector_cmd = (
        f"{selector_exec} scripts.select_economic_portfolio_policy --config configs/optimization.yaml "
        f"--run-tag {run_tag}{selector_solver_arg}"
    )
    ab_cmd = (
        f"{ab_exec} scripts.simulate_ab_test {ab_candidates} --run-tag {run_tag} "
        f"--policy_selector explicit_champion_only{ab_solver_arg}"
    )
    heavy_main_prefix = f"{activate_main} && {heavy_headroom_guard} &&"
    if use_frozen_policy:
        heavy_main_cmd = f"""
            {heavy_main_prefix}
            uv run python -u scripts/run_survival_analysis.py {survival_args} &&
            uv run python -u scripts/train_lgd_ead.py {lgd_ead_sample} --run-tag {run_tag}{lgd_backend_arg} &&
            {portfolio_cmd} &&
            {ab_cmd} &&
            (uv run python -u scripts/log_mlflow_experiment_suite.py || true)
        """
    else:
        heavy_main_cmd = f"""
            {heavy_main_prefix}
            uv run python -u scripts/run_survival_analysis.py {survival_args} &&
            uv run python -u scripts/train_lgd_ead.py {lgd_ead_sample} --run-tag {run_tag}{lgd_backend_arg} &&
            {portfolio_cmd} &&
            {tradeoff_cmd} &&
            {selector_cmd} &&
            {ab_cmd} &&
            (uv run python -u scripts/log_mlflow_experiment_suite.py || true)
        """
    steps.append(("heavy_main", False, heavy_main_cmd))

    causal_args = ["--treatment int_rate", causal_sample, f"--run_tag {run_tag}"]
    if causal_search_cfg.get("cate_n_estimators") is not None:
        causal_args.append(f"--cate_n_estimators {int(causal_search_cfg['cate_n_estimators'])}")
    if causal_search_cfg.get("cate_cv") is not None:
        causal_args.append(f"--cate_cv {int(causal_search_cfg['cate_cv'])}")
    if causal_search_cfg.get("cate_mc_iters") is not None:
        causal_args.append(f"--cate_mc_iters {int(causal_search_cfg['cate_mc_iters'])}")
    if causal_search_cfg.get("cate_criterion") is not None:
        causal_args.append(f"--cate_criterion {causal_search_cfg['cate_criterion']}")
    if causal_search_cfg.get("cate_min_balancedness_tol") is not None:
        causal_args.append(
            f"--cate_min_balancedness_tol {float(causal_search_cfg['cate_min_balancedness_tol'])}"
        )
    if causal_search_cfg.get("cate_honest") is not None:
        causal_args.append(
            f"--cate_honest {'true' if bool(causal_search_cfg['cate_honest']) else 'false'}"
        )
    causal_cmd = f"""
        {activate_main} &&
        {causal_headroom_guard} &&
        bash scripts/causal/run_causal_pipeline.sh {" ".join(arg for arg in causal_args if arg.strip())}
    """
    steps.append(("causal", False, causal_cmd))

    cate_cmd = f"""
        {activate_main} &&
        {causal_headroom_guard} &&
        {cate_exec} scripts.optimize_cate_portfolio {cate_candidates}{cate_solver_arg}
    """
    steps.append(("cate_portfolio", False, cate_cmd))

    post_core_cmd = f"""
        {activate_main} &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} &&
        uv run python -u scripts/validate_conformal_policy.py --run-tag {run_tag} --sensitivity-config configs/conformal_policy_sensitivity.yaml &&
        uv run python -u scripts/run_ifrs9_sensitivity.py &&
        uv run python -u scripts/build_pipeline_results.py &&
        if [ -f scripts/build_pd_challenger_artifacts.py ]; then uv run python -u scripts/build_pd_challenger_artifacts.py --config {pd_config}; else true; fi &&
        uv run python -u scripts/run_fairness_audit.py --run-tag {run_tag} &&
        if [ -f scripts/generate_governance_status.py ]; then uv run python -u scripts/generate_governance_status.py --config configs/mrm_policy.yaml --run-tag {run_tag}; else true; fi &&
        if [ -f scripts/update_champion_registry.py ]; then uv run python -u scripts/update_champion_registry.py; else true; fi &&
        if [ -f scripts/build_champion_search_bundle.py ]; then uv run python -u scripts/build_champion_search_bundle.py; else true; fi &&
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
        rapids_extra_cmd = ""
        if str(resource_policy.get("ifrs9_mc_backend", "")).strip().lower() == "gpu_research":
            ifrs9_cfg = dict(rapids_search_cfg.get("ifrs9_mc", {}) or {})
            n_scenarios = int(ifrs9_cfg.get("n_scenarios", 8192))
            chunk_size = int(ifrs9_cfg.get("chunk_size", 256))
            rapids_extra_cmd = (
                " && "
                f"{rapids_script_cmd} scripts/run_ifrs9_monte_carlo_gpu.py "
                f"--n-scenarios {n_scenarios} --chunk-size {chunk_size}"
            )
        rapids_cmd = (
            f"{activate_main} && "
            f"{rapids_headroom_guard} && "
            f"bash scripts/side_projects/run_rapids_benchmarks.sh --profile {rapids_profile}"
            f"{rapids_extra_cmd}"
        )
        steps.append(("rapids", False, rapids_cmd))

    if include_notebooks:
        notebooks_headroom_guard = (
            "uv run python -u scripts/ensure_memory_headroom.py "
            "--label notebooks --min-mem-gb 5 --min-swap-gb 3 "
            "--min-total-headroom-gb 9 --max-wait-seconds 1800 --poll-seconds 20"
        )
        notebook_timeout = int(notebook_search_cfg.get("timeout", 5400))
        notebook_output_dir = str(
            notebook_search_cfg.get("output_dir", "reports/notebook_exec")
        ).strip()
        notebooks_cmd = f"""
            {activate_main} &&
            {notebooks_headroom_guard} &&
            uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout {notebook_timeout} --inplace false --output-dir {notebook_output_dir} &&
            uv run python -u scripts/extract_notebook_images.py
        """
        steps.append(("notebooks", False, notebooks_cmd))

    return [(name, req, " ".join(cmd.split())) for name, req, cmd in steps]


def parse_args(
    argv: list[str] | None = None,
    *,
    default_pipeline_family: str = "champion_search",
    default_sampling_profile: str = "full",
    default_include_rapids: bool = True,
    default_include_notebooks: bool = True,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable long-run pipeline orchestrator")
    p.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    p.add_argument(
        "--pipeline-family",
        choices=["champion_search", "canonical_rebuild"],
        default=default_pipeline_family,
        help="Semantic pipeline family used for metadata contracts and execution defaults.",
    )
    p.add_argument(
        "--pipeline-profile",
        default=None,
        help="Human-readable pipeline profile persisted in run metadata.",
    )
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
    p.add_argument(
        "--no-rapids",
        action="store_true",
        default=not default_include_rapids,
    )
    p.add_argument(
        "--no-notebooks",
        action="store_true",
        default=not default_include_notebooks,
    )
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
        default=default_sampling_profile,
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
    p.add_argument(
        "--upstream-canonical-run-tag",
        default=None,
        help="Canonical operational run tag used as upstream reference for search/insight runs.",
    )
    p.add_argument(
        "--writes-canonical-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the default write contract for the selected pipeline family.",
    )
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    default_pipeline_family: str = "champion_search",
    default_sampling_profile: str = "full",
    default_include_rapids: bool = True,
    default_include_notebooks: bool = True,
    compatibility_entrypoint: str | None = None,
) -> int:
    if (
        argv is None
        and default_pipeline_family == "champion_search"
        and default_sampling_profile == "full"
        and default_include_rapids is True
        and default_include_notebooks is True
    ):
        args = parse_args()
    else:
        args = parse_args(
            argv,
            default_pipeline_family=default_pipeline_family,
            default_sampling_profile=default_sampling_profile,
            default_include_rapids=default_include_rapids,
            default_include_notebooks=default_include_notebooks,
        )
    if compatibility_entrypoint:
        append_master(
            str(args.run_tag),
            f"ENTRYPOINT_COMPATIBILITY alias={compatibility_entrypoint} canonical=scripts/run_champion_search.py",
        )
    run_tag = str(args.run_tag)
    pipeline_family_arg = str(getattr(args, "pipeline_family", default_pipeline_family))
    pipeline_profile_arg = getattr(args, "pipeline_profile", None)
    writes_canonical_artifacts_arg = getattr(args, "writes_canonical_artifacts", None)
    upstream_canonical_run_tag = _resolve_upstream_canonical_run_tag(
        pipeline_family=pipeline_family_arg,
        upstream_run_tag_arg=getattr(args, "upstream_canonical_run_tag", None),
    )
    pipeline_contract = _derive_pipeline_contract(
        pipeline_family=pipeline_family_arg,
        pipeline_profile_arg=pipeline_profile_arg,
        sampling_profile=str(args.sampling_profile),
        writes_canonical_artifacts_arg=writes_canonical_artifacts_arg,
        upstream_canonical_run_tag=upstream_canonical_run_tag,
    )
    profile_cfg = _load_pipeline_profile_config(str(pipeline_contract["pipeline_profile"]))
    comparison_baseline_path = _resolve_comparison_baseline(
        baseline_path_arg=args.comparison_baseline,
        baseline_run_tag_arg=args.comparison_baseline_run_tag,
    )
    comparison_baseline_source = "cli"
    if comparison_baseline_path is None:
        profile_baseline = _resolve_comparison_baseline_from_profile(profile_cfg)
        if profile_baseline is not None:
            comparison_baseline_path = profile_baseline
            comparison_baseline_source = "profile_default"
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
            "pipeline_family": pipeline_contract["pipeline_family"],
            "pipeline_profile": pipeline_contract["pipeline_profile"],
            "artifact_scope": pipeline_contract["artifact_scope"],
            "promotion_state": pipeline_contract["promotion_state"],
            "upstream_canonical_run_tag": pipeline_contract["upstream_canonical_run_tag"],
            "writes_canonical_artifacts": pipeline_contract["writes_canonical_artifacts"],
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
        pipeline_family=str(pipeline_contract["pipeline_family"]),
        profile_cfg=profile_cfg,
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

        run_step_kwargs = {
            "required": required,
            "step_eta_default_seconds": step_eta_defaults.get(step),
            "stall_window_seconds": stall_window_seconds,
            "resume_subphases": bool(args.resume),
        }
        try:
            signature = inspect.signature(run_step)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "pipeline_contract" in signature.parameters:
            run_step_kwargs["pipeline_contract"] = pipeline_contract

        ec = run_step(run_tag, step, command, **run_step_kwargs)
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
        "pipeline_family": pipeline_contract["pipeline_family"],
        "pipeline_profile": pipeline_contract["pipeline_profile"],
        "artifact_scope": pipeline_contract["artifact_scope"],
        "promotion_state": pipeline_contract["promotion_state"],
        "upstream_canonical_run_tag": pipeline_contract["upstream_canonical_run_tag"],
        "writes_canonical_artifacts": pipeline_contract["writes_canonical_artifacts"],
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
    print(
        "[deprecated] scripts/run_long_pipeline.py is a compatibility entrypoint. "
        "Use scripts/run_champion_search.py instead.",
        file=sys.stderr,
    )
    raise SystemExit(main(compatibility_entrypoint="scripts/run_long_pipeline.py"))
