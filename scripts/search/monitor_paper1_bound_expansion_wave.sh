#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${1:-paper1_bound_expansion_2026_05_24_r1}"
PRIMARY_RUN_ROOT="${2:-paper1_champion_replacement_2026_05_23_r3}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"

echo "== expansion wave: ${RUN_ROOT} =="
if [[ -f "${LOG_DIR}/${RUN_ROOT}.pid" ]]; then
  pid="$(cat "${LOG_DIR}/${RUN_ROOT}.pid")"
  ps -p "${pid}" -o pid,ppid,sid,etime,%cpu,%mem,rss,cmd || echo "expansion_driver_dead pid=${pid}"
else
  echo "missing expansion pid: ${LOG_DIR}/${RUN_ROOT}.pid"
fi

echo
echo "== primary full pipeline: ${PRIMARY_RUN_ROOT} =="
if [[ -f "${LOG_DIR}/${PRIMARY_RUN_ROOT}.pid" ]]; then
  primary_pid="$(cat "${LOG_DIR}/${PRIMARY_RUN_ROOT}.pid")"
  ps -p "${primary_pid}" -o pid,ppid,sid,etime,%cpu,%mem,rss,cmd || echo "primary_driver_dead pid=${primary_pid}"
  pgrep -P "${primary_pid}" -a || true
else
  echo "missing primary pid: ${LOG_DIR}/${PRIMARY_RUN_ROOT}.pid"
fi

echo
echo "== primary after-current guard =="
guard_pid_file="${LOG_DIR}/${PRIMARY_RUN_ROOT}__guard_after_current.pid"
guard_log="${LOG_DIR}/${PRIMARY_RUN_ROOT}__guard_after_current.log"
if [[ -f "${guard_pid_file}" ]]; then
  guard_pid="$(cat "${guard_pid_file}")"
  ps -p "${guard_pid}" -o pid,ppid,sid,etime,%cpu,%mem,rss,cmd || echo "primary_guard_dead pid=${guard_pid}"
  tail -5 "${guard_log}" 2>/dev/null || true
else
  echo "missing guard pid: ${guard_pid_file}"
fi

echo
echo "== active related processes =="
pgrep -af "${RUN_ROOT}|${PRIMARY_RUN_ROOT}|run_conformal_reopen_search.py|run_portfolio_bound_aware_search.py|guard_primary_after_current_portfolio.sh" | grep -v "monitor_paper1_bound_expansion_wave" || true

echo
echo "== portfolio statuses =="
python - "${RUN_ROOT}" "${PRIMARY_RUN_ROOT}" <<'PY'
import json
import sys
import subprocess
from pathlib import Path

roots = [sys.argv[1], sys.argv[2]]
proc = subprocess.run(
    ["pgrep", "-af", "run_portfolio_bound_aware_search.py"],
    capture_output=True,
    text=True,
    check=False,
)
active_cmds = proc.stdout
def human_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return "0m"
    hours, rem = divmod(int(seconds), 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

for root in roots:
    for path in sorted(Path("models/portfolio_bound_aware").glob(f"{root}__*/portfolio_bound_aware_runtime_status.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(path, f"error={exc}")
            continue
        keys = [
            "run_tag",
            "phase",
            "state",
            "frontier_completed_units",
            "frontier_total_units",
            "frontier_pct_complete",
            "bound_completed_checks",
            "bound_total_checks",
            "global_pct_complete",
            "eta_sec",
            "selection_reason",
            "selected_realized_total_return",
            "selected_alpha01_exact_pass",
        ]
        compact = {key: payload.get(key) for key in keys if key in payload}
        compact["active_process"] = str(payload.get("run_tag", "")) in active_cmds
        if "eta_sec" in compact:
            compact["eta_human"] = human_seconds(compact["eta_sec"])
        print(path)
        print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== conformal statuses =="
python - "${PRIMARY_RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = sys.argv[1]
for path in sorted(Path("models/conformal_gap").glob(f"{root}__*__conformal/conformal_reopen_status.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(path, f"error={exc}")
        continue
    compact = {
        "run_tag": payload.get("run_tag"),
        "phase1_oot_namespace": payload.get("phase1_oot_namespace"),
        "final_namespace": payload.get("final_namespace"),
        "state": payload.get("state"),
        "selected": payload.get("selected"),
    }
    print(path)
    print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== diagnostics statuses =="
python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = sys.argv[1]
for path in sorted(Path("models/bound_diagnostics").glob(f"{root}__*/bound_decision_diagnostics_status.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(path, f"error={exc}")
        continue
    compact = {
        "run_label": payload.get("run_label"),
        "n_funded": payload.get("n_funded"),
        "total_allocated": payload.get("total_allocated"),
        "weighted_miscoverage_V": payload.get("weighted_miscoverage_V"),
        "gamma_cp": payload.get("gamma_cp"),
        "decision_loss_pd_excess": payload.get("decision_loss_pd_excess"),
        "all_bounds_hold": payload.get("all_bounds_hold"),
    }
    print(path)
    print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== recent expansion events =="
tail -30 "${LOG_DIR}/${RUN_ROOT}_events.jsonl" 2>/dev/null || true

echo
echo "== recent logs =="
for log in "${LOG_DIR}/${RUN_ROOT}.log" "${LOG_DIR}/${PRIMARY_RUN_ROOT}.log"; do
  if [[ -f "${log}" ]]; then
    echo "--- ${log}"
    tail -20 "${log}"
  fi
done
