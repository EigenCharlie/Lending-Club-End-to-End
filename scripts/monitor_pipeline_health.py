#!/usr/bin/env python3
"""Operational monitor for long-run pipeline progress and silent stall detection.

This script samples heartbeat/log/resource signals and appends them to JSONL files:
- reports/run_logs/<run_tag>/monitoring/health_samples.jsonl
- reports/run_logs/<run_tag>/monitoring/incidents.jsonl

If no progress is detected for a configured window, it records a non-destructive
diagnostic snapshot under:
- reports/run_logs/<run_tag>/monitoring/diagnostics/<timestamp>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_LOGS = ROOT / "reports" / "run_logs"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _detect_run_tag(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates: list[tuple[datetime, str]] = []
    for hb_path in RUN_LOGS.glob("*/heartbeat.json"):
        hb = _safe_read_json(hb_path)
        if hb.get("state") != "running":
            continue
        pid = hb.get("orchestrator_pid")
        if not isinstance(pid, int) or not _safe_alive(pid):
            continue
        hb_ts = _parse_iso(hb.get("last_update_utc")) or datetime.fromtimestamp(
            hb_path.stat().st_mtime, tz=UTC
        )
        candidates.append((hb_ts, hb_path.parent.name))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    run_dirs = [p for p in RUN_LOGS.glob("*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No run directories found under reports/run_logs/")
    newest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return newest.name


def _step_log_path(run_dir: Path, step: str | None) -> Path | None:
    if not step:
        return None
    path = run_dir / f"{step}.log"
    return path if path.exists() else None


def _load_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            raw = value.strip().split()[0]
            out[key] = int(raw) * 1024
    except Exception:
        pass
    return out


def _disk_stats(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    total = float(usage.total)
    used = float(usage.used)
    free = float(usage.free)
    pct = (used / total * 100.0) if total > 0 else 0.0
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": int(total),
        "used_bytes": int(used),
        "free_bytes": int(free),
        "used_pct": round(pct, 2),
    }


def _ps_for_pid(pid: int | None) -> dict[str, Any]:
    if pid is None or pid <= 0:
        return {"pid": pid, "alive": False}
    if not _safe_alive(pid):
        return {"pid": pid, "alive": False}
    cmd = ["ps", "-p", str(pid), "-o", "pid=,ppid=,etimes=,rss=,%mem=,%cpu=,comm=,args="]
    proc = subprocess.run(cmd, cwd=ROOT, check=False, text=True, capture_output=True)
    line = proc.stdout.strip()
    return {"pid": pid, "alive": True, "raw": line}


def _tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return f"(missing) {path}\n"
    proc = subprocess.run(
        ["tail", "-n", str(n), str(path)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _top_processes() -> str:
    proc = subprocess.run(
        [
            "ps",
            "-eo",
            "pid,ppid,etimes,rss,%mem,%cpu,comm,args",
            "--sort=-%cpu",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    lines = proc.stdout.splitlines()
    return "\n".join(lines[:25]) + "\n"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sample(run_dir: Path) -> dict[str, Any]:
    hb = _safe_read_json(run_dir / "heartbeat.json")
    master = run_dir / "master.log"
    step = hb.get("current_step") if isinstance(hb.get("current_step"), str) else None
    step_log = _step_log_path(run_dir, step)

    hb_ts = _parse_iso(hb.get("last_update_utc"))
    master_mtime = (
        datetime.fromtimestamp(master.stat().st_mtime, tz=UTC) if master.exists() else None
    )
    step_mtime = (
        datetime.fromtimestamp(step_log.stat().st_mtime, tz=UTC) if step_log is not None else None
    )
    last_output = _parse_iso(hb.get("last_output_utc"))
    progress_candidates = [
        x for x in [hb_ts, master_mtime, step_mtime, last_output] if x is not None
    ]
    progress_ts = max(progress_candidates) if progress_candidates else None

    now = _utc_now()
    hb_age = (now - hb_ts).total_seconds() if hb_ts is not None else None
    progress_age = (now - progress_ts).total_seconds() if progress_ts is not None else None

    meminfo = _load_meminfo()
    mem_total = int(meminfo.get("MemTotal", 0))
    mem_available = int(meminfo.get("MemAvailable", 0))
    swap_total = int(meminfo.get("SwapTotal", 0))
    swap_free = int(meminfo.get("SwapFree", 0))
    mem_used = max(0, mem_total - mem_available)
    swap_used = max(0, swap_total - swap_free)

    orchestrator_pid = (
        hb.get("orchestrator_pid") if isinstance(hb.get("orchestrator_pid"), int) else None
    )
    child_pid = hb.get("active_child_pid") if isinstance(hb.get("active_child_pid"), int) else None

    return {
        "sampled_at_utc": _iso_now(),
        "run_tag": run_dir.name,
        "state": hb.get("state"),
        "current_step": step,
        "heartbeat_last_update_utc": hb.get("last_update_utc"),
        "heartbeat_age_seconds": hb_age,
        "progress_last_seen_utc": progress_ts.isoformat() if progress_ts else None,
        "progress_age_seconds": progress_age,
        "heartbeat_stalled_flag": bool(hb.get("stalled", False)),
        "orchestrator": _ps_for_pid(orchestrator_pid),
        "active_child": _ps_for_pid(child_pid),
        "master_log_size_bytes": int(master.stat().st_size) if master.exists() else 0,
        "step_log_path": str(step_log) if step_log else None,
        "step_log_size_bytes": int(step_log.stat().st_size) if step_log else 0,
        "memory": {
            "mem_total_bytes": mem_total,
            "mem_used_bytes": mem_used,
            "mem_available_bytes": mem_available,
            "mem_used_pct": round((mem_used / mem_total * 100.0), 2) if mem_total > 0 else None,
            "swap_total_bytes": swap_total,
            "swap_used_bytes": swap_used,
            "swap_used_pct": round((swap_used / swap_total * 100.0), 2) if swap_total > 0 else None,
        },
        "disks": {
            "root": _disk_stats(Path("/")),
            "mnt_c": _disk_stats(Path("/mnt/c")),
        },
    }


def _write_diagnostic_snapshot(
    run_dir: Path, sample: dict[str, Any], diagnostics_dir: Path
) -> Path:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = diagnostics_dir / f"incident_{stamp}.txt"
    step = sample.get("current_step")
    step_log = run_dir / f"{step}.log" if isinstance(step, str) else None
    lines = [
        f"timestamp_utc={sample.get('sampled_at_utc')}",
        f"run_tag={run_dir.name}",
        f"state={sample.get('state')}",
        f"current_step={step}",
        f"heartbeat_age_seconds={sample.get('heartbeat_age_seconds')}",
        f"progress_age_seconds={sample.get('progress_age_seconds')}",
        "",
        "== heartbeat.json ==",
        json.dumps(_safe_read_json(run_dir / "heartbeat.json"), indent=2, ensure_ascii=False),
        "",
        "== master.log tail ==",
        _tail(run_dir / "master.log", n=120),
        "",
        "== step.log tail ==",
        _tail(step_log, n=120) if step_log is not None else "(no current step)\n",
        "",
        "== ps top cpu ==",
        _top_processes(),
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor pipeline health and detect silent stalls."
    )
    parser.add_argument("--run-tag", default=None, help="Run tag under reports/run_logs/")
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--stall-window-seconds", type=int, default=15 * 60)
    parser.add_argument("--once", action="store_true", help="Collect one sample and exit.")
    args = parser.parse_args()

    run_tag = _detect_run_tag(args.run_tag)
    run_dir = RUN_LOGS / run_tag
    mon_dir = run_dir / "monitoring"
    sample_path = mon_dir / "health_samples.jsonl"
    incident_path = mon_dir / "incidents.jsonl"
    diagnostics_dir = mon_dir / "diagnostics"

    interval_seconds = max(10, int(args.interval_seconds))
    stall_window_seconds = max(60, int(args.stall_window_seconds))
    active_incident = False

    while True:
        sample = _sample(run_dir)
        progress_age = sample.get("progress_age_seconds")
        hb_age = sample.get("heartbeat_age_seconds")
        state = str(sample.get("state") or "unknown")
        stalled = bool(
            state == "running"
            and (
                (isinstance(progress_age, int | float) and progress_age >= stall_window_seconds)
                or (isinstance(hb_age, int | float) and hb_age >= stall_window_seconds)
            )
        )
        sample["stall_window_seconds"] = stall_window_seconds
        sample["silent_stall_detected"] = stalled
        _append_jsonl(sample_path, sample)

        step = sample.get("current_step")
        prog_age_txt = f"{int(progress_age)}s" if isinstance(progress_age, int | float) else "n/a"
        mem_pct = sample.get("memory", {}).get("mem_used_pct")
        mem_txt = f"{mem_pct}%" if isinstance(mem_pct, int | float) else "n/a"
        print(
            f"[{sample['sampled_at_utc']}] run={run_tag} step={step} "
            f"progress_age={prog_age_txt} mem_used={mem_txt} stalled={stalled}",
            flush=True,
        )

        if stalled and not active_incident:
            diag_path = _write_diagnostic_snapshot(run_dir, sample, diagnostics_dir)
            incident = {
                "detected_at_utc": _iso_now(),
                "run_tag": run_tag,
                "current_step": step,
                "state": state,
                "progress_age_seconds": progress_age,
                "heartbeat_age_seconds": hb_age,
                "diagnostic_path": str(diag_path),
                "action": "non_destructive_diagnostics_recorded",
            }
            _append_jsonl(incident_path, incident)
            print(
                f"INCIDENT silent stall detected. Diagnostics: {diag_path}",
                flush=True,
            )
            active_incident = True
        elif not stalled:
            active_incident = False

        if args.once:
            break
        time.sleep(interval_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
