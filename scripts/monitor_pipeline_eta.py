#!/usr/bin/env python3
"""Live pipeline monitor with subphase detection and ETA estimates.

Reads run logs produced by scripts/run_long_pipeline.py and prints:
- current phase and exact subphase (when detectable from running process tree)
- where it is inside heavy_main (k/N)
- ETA for current subphase and remaining phases

ETA is best-effort:
- prefers historical step durations from reports/run_logs/*/status/*.json
- falls back to heuristics when history is missing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_LOGS_DIR = ROOT / "reports" / "run_logs"

BASE_STEP_ORDER = [
    "preflight",
    "main_pre",
    "heavy_main",
    "causal",
    "cate_portfolio",
    "post_core",
    "rapids",
    "notebooks",
]

STEP_DEFAULTS_SECONDS = {
    "preflight": 300.0,
    "main_pre": 3600.0,
    "heavy_main": 4.5 * 3600.0,
    "causal": 5400.0,
    "cate_portfolio": 1800.0,
    "post_core": 7200.0,
    "rapids": 7200.0,
    "notebooks": 4.0 * 3600.0,
}

STEP_SUBPHASES = {
    "main_pre": [
        "scripts/train_pd_model.py",
        "scripts/generate_conformal_intervals.py",
        "scripts/benchmark_conformal_variants.py",
        "scripts/backtest_conformal_coverage.py",
        "scripts/validate_conformal_policy.py",
        "scripts/forecast_default_rates.py",
    ],
    "heavy_main": [
        "scripts/run_survival_analysis.py",
        "scripts/train_lgd_ead.py",
        "scripts/optimize_portfolio.py",
        "scripts/optimize_portfolio_tradeoff.py",
        "scripts/simulate_ab_test.py",
        "scripts/log_mlflow_experiment_suite.py",
    ],
    "causal": [
        "scripts/estimate_causal_effects.py",
        "scripts/simulate_causal_policy.py",
        "scripts/backtest_causal_policy_oot.py",
    ],
    "cate_portfolio": ["scripts/optimize_cate_portfolio.py"],
    "post_core": [
        "scripts/run_ifrs9_sensitivity.py",
        "scripts/build_pipeline_results.py",
        "scripts/build_pd_challenger_artifacts.py",
        "scripts/run_fairness_audit.py",
        "scripts/validate_causal_policy.py",
        "scripts/generate_governance_status.py",
        "scripts/generate_mrm_report.py",
        "scripts/export_streamlit_artifacts.py",
        "scripts/export_storytelling_snapshot.py",
        "scripts/export_dvc_metrics.py",
        "scripts/run_comparison.py",
    ],
    "rapids": ["scripts/side_projects/run_rapids_benchmarks.sh"],
    "notebooks": [
        "scripts/run_all_notebooks.py",
        "scripts/run_paper_notebook_suite.py",
        "scripts/extract_notebook_images.py",
    ],
}


@dataclass
class ProcRow:
    pid: int
    ppid: int
    etimes: int
    cmd: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_range(low_s: float, high_s: float) -> str:
    low = _fmt_duration(low_s)
    high = _fmt_duration(high_s)
    return f"{low} - {high}"


def _safe_kill0(pid: int | None) -> bool:
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
    for hb_path in RUN_LOGS_DIR.glob("*/heartbeat.json"):
        try:
            hb = _load_json(hb_path)
        except Exception:
            continue
        if hb.get("state") != "running":
            continue
        pid = hb.get("orchestrator_pid")
        if isinstance(pid, int) and _safe_kill0(pid):
            ts = _parse_iso_utc(hb.get("last_update_utc")) or datetime.fromtimestamp(
                hb_path.stat().st_mtime, tz=UTC
            )
            candidates.append((ts, hb_path.parent.name))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # Fallback to newest run dir.
    run_dirs = [p for p in RUN_LOGS_DIR.glob("*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No run directories found in reports/run_logs/")
    newest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return newest.name


def _load_status(run_dir: Path, step: str) -> dict | None:
    p = run_dir / "status" / f"{step}.json"
    if not p.exists():
        return None
    try:
        return _load_json(p)
    except Exception:
        return None


def _collect_completed_history(step: str, exclude_run_tag: str) -> list[float]:
    vals: list[float] = []
    for status_path in RUN_LOGS_DIR.glob(f"*/status/{step}.json"):
        run_tag = status_path.parents[1].name
        if run_tag == exclude_run_tag:
            continue
        try:
            data = _load_json(status_path)
        except Exception:
            continue
        dur = data.get("duration_seconds")
        if (
            isinstance(dur, int | float)
            and dur > 0
            and data.get("exit_code") == 0
            and not bool(data.get("skipped"))
        ):
            vals.append(float(dur))
    return vals


def _load_ps_table() -> dict[int, ProcRow]:
    out = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,etimes=,cmd="],
        text=True,
        cwd=ROOT,
    )
    table: dict[int, ProcRow] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            etimes = int(parts[2])
        except ValueError:
            continue
        table[pid] = ProcRow(pid=pid, ppid=ppid, etimes=etimes, cmd=parts[3])
    return table


def _descendants(table: dict[int, ProcRow], root_pid: int) -> set[int]:
    by_parent: dict[int, list[int]] = {}
    for pid, row in table.items():
        by_parent.setdefault(row.ppid, []).append(pid)
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(by_parent.get(cur, []))
    return seen


def _extract_matching_scripts(cmd: str, scripts: list[str]) -> list[str]:
    return [s for s in scripts if s in cmd]


def _detect_active_subphase(
    current_step: str | None,
    active_child_pid: int | None,
    ps_table: dict[int, ProcRow],
) -> tuple[str | None, int | None, int | None]:
    if not current_step or current_step not in STEP_SUBPHASES:
        return None, None, None
    if not isinstance(active_child_pid, int) or active_child_pid <= 0:
        return None, None, None
    if active_child_pid not in ps_table:
        return None, None, None

    descendants = _descendants(ps_table, active_child_pid)
    step_scripts = STEP_SUBPHASES[current_step]
    candidates: list[tuple[int, int, str, int]] = []
    for pid in descendants:
        row = ps_table.get(pid)
        if row is None:
            continue
        matches = _extract_matching_scripts(row.cmd, step_scripts)
        if len(matches) != 1:
            continue
        # Prefer process rows that look like the active runner (python/uv command with 1 script).
        score = 2 if ("python" in row.cmd or "uv run" in row.cmd) else 1
        # Smaller elapsed time is usually a deeper/active child in this process tree.
        candidates.append((score, -row.etimes, matches[0], pid))
    if not candidates:
        return None, None, None
    candidates.sort(reverse=True)
    _, _, script, pid = candidates[0]
    etimes = ps_table[pid].etimes
    return script, pid, etimes


def _read_optimization_time_limit(default: int = 300) -> int:
    cfg = ROOT / "configs" / "optimization.yaml"
    if not cfg.exists():
        return default
    for line in cfg.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*time_limit:\s*(\d+)\s*$", line)
        if m:
            return int(m.group(1))
    return default


def _read_latest_loaded_n_loans(heavy_log: Path) -> int | None:
    if not heavy_log.exists():
        return None
    text = heavy_log.read_text(encoding="utf-8", errors="ignore")
    # Example: Loaded 1,346,311 loans for survival analysis...
    matches = re.findall(r"Loaded\s+([\d,]+)\s+loans\s+for\s+survival\s+analysis", text)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _read_current_rsf_estimators(heavy_status: dict | None) -> int:
    if not heavy_status:
        return 300
    cmd = str(heavy_status.get("command", ""))
    m = re.search(r"--rsf_n_estimators\s+(\d+)", cmd)
    if not m:
        return 300
    return int(m.group(1))


def _estimate_survival_seconds(run_dir: Path, heavy_status: dict | None) -> float:
    default_est = 2.0 * 3600.0
    summary_path = ROOT / "models" / "survival_summary.pkl"
    if not summary_path.exists():
        return default_est
    try:
        import pickle

        summary = pickle.loads(summary_path.read_bytes())
    except Exception:
        return default_est

    rsf_prev = summary.get("rsf_training_time")
    n_prev = summary.get("n_loans")
    rsf_params = summary.get("rsf_params", {}) if isinstance(summary, dict) else {}
    est_prev = rsf_params.get("n_estimators") if isinstance(rsf_params, dict) else None

    if not isinstance(rsf_prev, int | float) or rsf_prev <= 0:
        return default_est

    n_cur = _read_latest_loaded_n_loans(run_dir / "heavy_main.log")
    n_ratio = 1.0
    if isinstance(n_prev, int) and n_prev > 0 and isinstance(n_cur, int) and n_cur > 0:
        n_ratio = max(0.25, min(20.0, float(n_cur) / float(n_prev)))

    est_cur = _read_current_rsf_estimators(heavy_status)
    est_ratio = 1.0
    if isinstance(est_prev, int) and est_prev > 0:
        est_ratio = max(0.5, min(4.0, float(est_cur) / float(est_prev)))

    # Add a fixed overhead for Cox fit + artifact IO.
    est = float(rsf_prev) * n_ratio * est_ratio + 10 * 60.0
    return max(20 * 60.0, min(8 * 3600.0, est))


def _estimate_heavy_subphases(run_dir: Path, heavy_status: dict | None) -> dict[str, float]:
    tlim = float(_read_optimization_time_limit(default=300))
    tradeoff_n_solves = 48.0  # grid-profile night: 6 risks * (1 baseline + 7 robust)

    return {
        "scripts/run_survival_analysis.py": _estimate_survival_seconds(run_dir, heavy_status),
        "scripts/train_lgd_ead.py": 30 * 60.0,
        "scripts/optimize_portfolio.py": max(120.0, min(tlim, tlim * 0.8)),
        "scripts/optimize_portfolio_tradeoff.py": max(45 * 60.0, tradeoff_n_solves * tlim * 0.55),
        "scripts/simulate_ab_test.py": 18 * 60.0,
        "scripts/log_mlflow_experiment_suite.py": 15 * 60.0,
    }


def _step_order(run_info: dict) -> list[str]:
    include_rapids = bool(run_info.get("include_rapids", True))
    include_notebooks = bool(run_info.get("include_notebooks", True))
    out = []
    for step in BASE_STEP_ORDER:
        if step == "rapids" and not include_rapids:
            continue
        if step == "notebooks" and not include_notebooks:
            continue
        out.append(step)
    return out


def _step_estimate_seconds(step: str, run_tag: str, run_dir: Path) -> tuple[float, str]:
    hist = _collect_completed_history(step, exclude_run_tag=run_tag)
    if hist:
        return float(statistics.median(hist)), f"hist n={len(hist)}"
    if step == "heavy_main":
        heavy_status = _load_status(run_dir, "heavy_main")
        sub = _estimate_heavy_subphases(run_dir, heavy_status)
        return float(sum(sub.values())), "heuristic(subphase)"
    return float(STEP_DEFAULTS_SECONDS.get(step, 3600.0)), "heuristic(default)"


def _remaining_current_step_seconds(
    *,
    step: str,
    run_tag: str,
    run_dir: Path,
    step_elapsed_s: float,
    active_subphase: str | None,
    subphase_elapsed_s: float | None,
) -> tuple[float, float, str]:
    # Returns low/high remaining seconds and source string.
    if step == "heavy_main":
        heavy_status = _load_status(run_dir, "heavy_main")
        sub = _estimate_heavy_subphases(run_dir, heavy_status)
        order = STEP_SUBPHASES["heavy_main"]
        if active_subphase and active_subphase in order:
            idx = order.index(active_subphase)
            rem_est = 0.0
            for i, name in enumerate(order):
                est = float(sub.get(name, 1800.0))
                if i < idx:
                    continue
                if i == idx and isinstance(subphase_elapsed_s, int | float):
                    est = max(0.0, est - float(subphase_elapsed_s))
                rem_est += est
            return rem_est * 0.65, rem_est * 1.45, "heuristic(subphase)"
        # Fallback: step-level only.
        step_est, source = _step_estimate_seconds(step, run_tag, run_dir)
        rem = max(0.0, step_est - step_elapsed_s)
        return rem * 0.7, rem * 1.4, source

    step_est, source = _step_estimate_seconds(step, run_tag, run_dir)
    rem = max(0.0, step_est - step_elapsed_s)
    return rem * 0.8, rem * 1.25, source


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor long pipeline with subphase and ETA")
    parser.add_argument("--run-tag", default=None, help="Run tag under reports/run_logs/")
    args = parser.parse_args()

    run_tag = _detect_run_tag(args.run_tag)
    run_dir = RUN_LOGS_DIR / run_tag
    hb_path = run_dir / "heartbeat.json"
    info_path = run_dir / "run_info.json"
    if not hb_path.exists():
        raise FileNotFoundError(f"Missing heartbeat: {hb_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing run_info: {info_path}")

    hb = _load_json(hb_path)
    run_info = _load_json(info_path)
    now = datetime.now(UTC)
    hb_time = _parse_iso_utc(hb.get("last_update_utc"))
    hb_age = (now - hb_time).total_seconds() if hb_time else None

    current_step = hb.get("current_step")
    orchestrator_pid = hb.get("orchestrator_pid")
    active_child_pid = hb.get("active_child_pid")
    state = hb.get("state", "unknown")

    step_order = _step_order(run_info)

    statuses: dict[str, dict | None] = {s: _load_status(run_dir, s) for s in step_order}

    # Current step elapsed.
    step_elapsed_s = 0.0
    if isinstance(current_step, str) and current_step in statuses:
        st = statuses.get(current_step) or {}
        start = _parse_iso_utc(st.get("started_at_utc")) if st else None
        if start:
            step_elapsed_s = max(0.0, (now - start).total_seconds())

    # Process tree based subphase detection.
    ps_table = _load_ps_table()
    active_subphase, active_pid, active_subphase_elapsed = _detect_active_subphase(
        current_step=current_step if isinstance(current_step, str) else None,
        active_child_pid=active_child_pid if isinstance(active_child_pid, int) else None,
        ps_table=ps_table,
    )

    # Remaining ETA for current step and full pipeline.
    rem_low_total = 0.0
    rem_high_total = 0.0
    rem_lines: list[str] = []

    # Step index.
    current_idx = step_order.index(current_step) if current_step in step_order else None

    if current_idx is not None and isinstance(current_step, str):
        cur_low, cur_high, cur_src = _remaining_current_step_seconds(
            step=current_step,
            run_tag=run_tag,
            run_dir=run_dir,
            step_elapsed_s=step_elapsed_s,
            active_subphase=active_subphase,
            subphase_elapsed_s=active_subphase_elapsed,
        )
        rem_low_total += cur_low
        rem_high_total += cur_high
        rem_lines.append(f"- {current_step}: {_fmt_range(cur_low, cur_high)} (restante, {cur_src})")
        for step in step_order[current_idx + 1 :]:
            st = statuses.get(step) or {}
            if st and st.get("exit_code") == 0:
                continue
            est, src = _step_estimate_seconds(step, run_tag, run_dir)
            low, high = est * 0.75, est * 1.35
            rem_low_total += low
            rem_high_total += high
            rem_lines.append(f"- {step}: {_fmt_range(low, high)} ({src})")

    # Subphase placement inside current step.
    subphase_part = None
    if isinstance(current_step, str) and current_step in STEP_SUBPHASES and active_subphase:
        order = STEP_SUBPHASES[current_step]
        if active_subphase in order:
            idx = order.index(active_subphase) + 1
            subphase_part = f"{idx}/{len(order)}"

    # Header.
    print(f"run_tag={run_tag}")
    print(f"state={state}")
    print(f"current_step={current_step}")
    print(
        f"orchestrator_pid={orchestrator_pid} alive={_safe_kill0(orchestrator_pid if isinstance(orchestrator_pid, int) else None)}"
    )
    print(
        f"active_child_pid={active_child_pid} alive={_safe_kill0(active_child_pid if isinstance(active_child_pid, int) else None)}"
    )
    if hb_age is not None:
        print(f"heartbeat_age={_fmt_duration(hb_age)}")

    # Current phase detail.
    print()
    print("fase_actual")
    print(f"- elapsed_fase={_fmt_duration(step_elapsed_s)}")
    if active_subphase:
        label = active_subphase.replace("scripts/", "")
        print(f"- subfase={label}")
        if subphase_part:
            print(f"- parte={subphase_part}")
        if active_pid is not None:
            print(f"- subfase_pid={active_pid}")
        if active_subphase_elapsed is not None:
            print(f"- elapsed_subfase={_fmt_duration(active_subphase_elapsed)}")
            if isinstance(current_step, str):
                cur_low, cur_high, _ = _remaining_current_step_seconds(
                    step=current_step,
                    run_tag=run_tag,
                    run_dir=run_dir,
                    step_elapsed_s=step_elapsed_s,
                    active_subphase=active_subphase,
                    subphase_elapsed_s=active_subphase_elapsed,
                )
                print(f"- eta_subfase_y_fase_actual={_fmt_range(cur_low, cur_high)}")
    else:
        print("- subfase=indetectable")

    # Remaining by phase.
    print()
    print("eta_fases_restantes")
    if rem_lines:
        for line in rem_lines:
            print(line)
        print(f"eta_total_restante={_fmt_range(rem_low_total, rem_high_total)}")
    else:
        print("- n/a")

    # Step completion summary.
    print()
    print("estado_etapas")
    for step in step_order:
        st = statuses.get(step)
        if not st:
            mark = "pending"
        elif st.get("exit_code") == 0 and st.get("ended_at_utc"):
            mark = "done"
        elif current_step == step:
            mark = "running"
        elif st.get("exit_code") not in (None, 0):
            mark = f"failed(ec={st.get('exit_code')})"
        else:
            mark = "pending"
        dur = st.get("duration_seconds") if st else None
        dur_txt = _fmt_duration(dur if isinstance(dur, int | float) else None)
        print(f"- {step}: {mark} duration={dur_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
