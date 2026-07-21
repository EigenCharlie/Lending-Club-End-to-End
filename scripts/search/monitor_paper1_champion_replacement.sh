#!/usr/bin/env bash
set -u

RUN_ROOT="${1:-paper1_champion_replacement_2026_05_23_r3}"
LOG_DIR="${LOG_DIR:-reports/run_logs}"
PID_FILE="${LOG_DIR}/${RUN_ROOT}.pid"

echo "RUN_ROOT=${RUN_ROOT}"
echo "now=$(date -Is)"
echo

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "process:"
  if ps -p "${PID}" -o pid,ppid,sid,etimes,%cpu,%mem,rss,cmd; then
    PROCESS_ALIVE=1
  else
    PROCESS_ALIVE=0
    echo "finished"
  fi
  echo
  echo "children:"
  ACTIVE_CHILDREN="$(pgrep -P "${PID}" -a || true)"
  printf '%s\n' "${ACTIVE_CHILDREN}"
else
  PROCESS_ALIVE=0
  ACTIVE_CHILDREN=""
  echo "process:"
  echo "missing pid file: ${PID_FILE}"
fi

echo
echo "events:"
tail -20 "${LOG_DIR}/${RUN_ROOT}_events.jsonl" 2>/dev/null || true

echo
echo "main log:"
tail -40 "${LOG_DIR}/${RUN_ROOT}.log" 2>/dev/null || true

echo
echo "latest stage logs:"
find "${LOG_DIR}" -maxdepth 1 -type f -name "${RUN_ROOT}*.log" \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null \
  | sort \
  | tail -4 \
  | while read -r _date _time path; do
      echo "--- ${path}"
      tail -20 "${path}" 2>/dev/null || true
    done

echo
echo "conformal statuses:"
python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = sys.argv[1]
for path in sorted(Path("models/conformal_gap").glob(f"*{run_root}*/conformal_reopen_status.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    phase1_ns = str(payload.get("phase1_oot_namespace", ""))
    final_ns = str(payload.get("final_namespace", ""))
    winner = payload.get("inner_search_winner", {}) or {}
    print(path)
    print(
        "  decision={decision} review_needed={review} winner={partition}/{source}/bins{bins}/alpha90={alpha}".format(
            decision=payload.get("promotion_decision", ""),
            review=payload.get("policy_review_needed", ""),
            partition=winner.get("partition", ""),
            source=winner.get("partition_probability_source", ""),
            bins=winner.get("n_score_bins", ""),
            alpha=winner.get("alpha_used_90", ""),
        )
    )
    for label, namespace in [("phase1", phase1_ns), ("final", final_ns)]:
        status_path = Path("models/conformal_gap") / namespace / "conformal_policy_status.json"
        if not namespace or not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        print(
            "  {label}: overall={overall} strict={strict} cov90={cov90:.4f} width90={width90:.4f} min_group90={min_group:.4f} alerts={alerts}".format(
                label=label,
                overall=status.get("overall_pass", ""),
                strict=status.get("strict_overall_pass", ""),
                cov90=float(status.get("coverage_90", 0.0)),
                width90=float(status.get("avg_width_90", 0.0)),
                min_group=float(status.get("min_group_coverage_90", 0.0)),
                alerts=status.get("total_alerts", ""),
            )
        )
PY

echo
echo "portfolio statuses:"
ACTIVE_CHILDREN="${ACTIVE_CHILDREN}" python - "${RUN_ROOT}" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

run_root = sys.argv[1]
active_children = os.environ.get("ACTIVE_CHILDREN", "")
stall_timeout_min = float(os.environ.get("CUOPT_STALL_TIMEOUT_MIN", "30") or "30")
now = datetime.now(UTC)
for path in sorted(Path("models/portfolio_bound_aware").glob(f"*{run_root}*/portfolio_bound_aware_runtime_status.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_tag = str(payload.get("run_tag", ""))
    active = bool(run_tag and run_tag in active_children)
    staleness_sec = max(0.0, (now - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)).total_seconds())
    stale_running = (
        payload.get("state") == "running"
        and staleness_sec > stall_timeout_min * 60.0
    )
    print(path)
    print(
        "  active={active} state={state} phase={phase} frontier={fc}/{ft} ({fp:.2%}) bound={bc}/{bt} ({bp:.2%}) elapsed={elapsed:.0f}s eta={eta} stale_sec={stale:.0f} stale_running={stale_running}".format(
            active=active,
            state=payload.get("state", ""),
            phase=payload.get("phase", ""),
            fc=int(payload.get("frontier_completed_units", 0)),
            ft=int(payload.get("frontier_total_units", 0)),
            fp=float(payload.get("frontier_pct_complete", 0.0)),
            bc=int(payload.get("bound_completed_checks", 0)),
            bt=int(payload.get("bound_total_checks", 0)),
            bp=float(payload.get("bound_pct_complete", 0.0)),
            elapsed=float(payload.get("elapsed_sec", 0.0)),
            eta=payload.get("eta_sec", None),
            stale=staleness_sec,
            stale_running=stale_running,
        )
    )
    extras = []
    for key in [
        "latest_policy_mode",
        "latest_risk_tolerance",
        "latest_gamma",
        "selection_reason",
        "selected_alpha01_exact_pass",
        "selected_realized_total_return",
    ]:
        if key in payload:
            extras.append(f"{key}={payload[key]}")
    if extras:
        print("  " + " ".join(extras))
PY

if [[ "${PROCESS_ALIVE}" == "0" ]]; then
  echo
  echo "note:"
  echo "No active driver process. Any runtime_status.json with state=running is stale from an interrupted run."
fi
