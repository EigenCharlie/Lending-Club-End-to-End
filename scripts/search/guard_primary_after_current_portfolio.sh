#!/usr/bin/env bash
set -uo pipefail

# Stop a long primary driver after the currently running portfolio finishes.
#
# This is intentionally conservative: it never interrupts the current portfolio
# run. It waits until the selected portfolio artifact exists and no process with
# the current run label is still alive, then terminates the primary driver
# process group so the smarter expansion/triage wave can take over.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PRIMARY_RUN_ROOT="${1:?primary run root required}"
CURRENT_PORTFOLIO_RUN="${2:?current portfolio run label required}"
FOLLOWUP_RUN_ROOT="${3:-paper1_bound_expansion_2026_05_24_r1}"

LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${PRIMARY_RUN_ROOT}__guard_after_current.log"
EVENT_LOG="${LOG_DIR}/${FOLLOWUP_RUN_ROOT}_events.jsonl"
PID_FILE="${LOG_DIR}/${PRIMARY_RUN_ROOT}.pid"
SELECTION_PATH="models/portfolio_bound_aware/${CURRENT_PORTFOLIO_RUN}/portfolio_bound_aware_selection.json"
STATUS_PATH="models/portfolio_bound_aware/${CURRENT_PORTFOLIO_RUN}/portfolio_bound_aware_runtime_status.json"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_PATH}"
}

emit_event() {
  local stage="$1"
  local state="$2"
  local detail="${3:-}"
  python - "$EVENT_LOG" "$FOLLOWUP_RUN_ROOT" "$stage" "$state" "$detail" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

event_log, run_root, stage, state, detail = sys.argv[1:6]
path = Path(event_log)
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "ts": datetime.now(UTC).isoformat(),
    "run_root": run_root,
    "stage": stage,
    "candidate_key": "primary_guard",
    "state": state,
    "detail": detail,
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

portfolio_process_alive() {
  pgrep -f "run_portfolio_bound_aware_search.py .*--run-label ${CURRENT_PORTFOLIO_RUN}" >/dev/null 2>&1
}

primary_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    cat "${PID_FILE}"
  fi
}

primary_alive() {
  local pid
  pid="$(primary_pid)"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

portfolio_completed() {
  [[ -f "${SELECTION_PATH}" ]] && return 0
  python - "${STATUS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("state") == "completed" or payload.get("phase") == "selection_complete":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

log "guard started for ${PRIMARY_RUN_ROOT}; current=${CURRENT_PORTFOLIO_RUN}"
emit_event "primary_guard" "running" "watching:${CURRENT_PORTFOLIO_RUN}"

while true; do
  if ! primary_alive; then
    log "primary driver already idle; guard exits"
    emit_event "primary_guard" "completed" "primary already idle"
    exit 0
  fi

  if portfolio_completed; then
    log "portfolio completion detected: ${CURRENT_PORTFOLIO_RUN}"
    while portfolio_process_alive; do
      log "waiting for current portfolio process to exit cleanly"
      sleep 30
    done

    pid="$(primary_pid)"
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      pgid="$(ps -o sid= -p "${pid}" | tr -d ' ')"
      log "terminating primary process group ${pgid} after current portfolio"
      emit_event "primary_guard" "stopping_primary" "pgid:${pgid}"
      kill -TERM "-${pgid}" >/dev/null 2>&1 || true
      sleep 20
      if ps -p "${pid}" >/dev/null 2>&1; then
        log "primary still alive after TERM; sending KILL to process group ${pgid}"
        kill -KILL "-${pgid}" >/dev/null 2>&1 || true
      fi
      emit_event "primary_guard" "completed" "stopped_after:${CURRENT_PORTFOLIO_RUN}"
    else
      log "primary exited before guard termination"
      emit_event "primary_guard" "completed" "primary exited naturally"
    fi
    exit 0
  fi

  sleep 60
done
