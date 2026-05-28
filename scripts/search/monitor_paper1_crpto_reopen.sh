#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

RUN_ROOT="${1:-${RUN_ROOT:-paper1_crpto_reopen_ijds_2026_05_25}}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"

echo "== CRPTO/IJDS reopen: ${RUN_ROOT} =="
if [[ -f "${LOG_DIR}/${RUN_ROOT}.pid" ]]; then
  pid="$(cat "${LOG_DIR}/${RUN_ROOT}.pid")"
  ps -p "${pid}" -o pid,ppid,sid,etime,%cpu,%mem,rss,cmd || echo "driver_dead pid=${pid}"
else
  echo "missing driver pid: ${LOG_DIR}/${RUN_ROOT}.pid"
fi

echo
echo "== active related processes =="
pgrep -af "${RUN_ROOT}|run_paper1_crpto_ijds_reopen|run_pd_hpo_local.py|run_conformal_reopen_search.py|generate_conformal_intervals.py|run_portfolio_bound_aware_search.py|run_portfolio_bound_exact_eval.py" \
  | grep -v "monitor_paper1_crpto_reopen" || true

echo
echo "== resource snapshot =="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null || true
free -h 2>/dev/null | sed -n '1,2p' || true

echo
echo "== env gate =="
"${MAIN_PYTHON}" - "${LOG_DIR}/${RUN_ROOT}/env_audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"pending: {path}")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
rec = payload.get("environment_recommendation", {})
print(json.dumps(rec, indent=2, sort_keys=True))
PY

echo
echo "== phase statuses =="
"${MAIN_PYTHON}" - "${LOG_DIR}/${RUN_ROOT}/status" <<'PY'
import json
import sys
from pathlib import Path

status_dir = Path(sys.argv[1])
if not status_dir.exists():
    print(f"pending: {status_dir}")
    raise SystemExit(0)
for path in sorted(status_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{path}: {type(exc).__name__}: {exc}")
        continue
    compact = {
        key: payload.get(key)
        for key in [
            "phase",
            "state",
            "started_at_utc",
            "finished_at_utc",
            "exit_code",
            "skipped",
            "reason",
            "command",
        ]
        if key in payload
    }
    print(path.name)
    print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== portfolio runtime statuses =="
"${MAIN_PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = sys.argv[1]
paths = sorted(Path("models/portfolio_bound_aware").glob(f"{run_root}*/portfolio_bound_aware_runtime_status.json"))
if not paths:
    print("no portfolio runtime status yet")
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: payload.get(key)
        for key in [
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
        if key in payload
    }
    print(path)
    print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo
echo "== live summary =="
"${MAIN_PYTHON}" scripts/search/summarize_paper1_crpto_ijds_reopen.py \
  --run-root "${RUN_ROOT}" \
  --since-minutes "${SINCE_MINUTES:-30}" >/dev/null 2>&1 || true
summary="${LOG_DIR}/${RUN_ROOT}/summary_current.csv"
if [[ -f "${summary}" ]]; then
  "${MAIN_PYTHON}" - "${summary}" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
cols = [
    "artifact_kind",
    "tier",
    "run_label",
    "decision_read",
    "realized_total_return",
    "return_delta_vs_champion",
    "alpha01_weighted_miscoverage_V",
    "alpha01_gamma_cp",
]
cols = [c for c in cols if c in df.columns]
if df.empty:
    print("summary empty")
else:
    print(df[cols].tail(30).to_string(index=False))
PY
else
  echo "pending: ${summary}"
fi

echo
echo "== dirty manifest bucket counts =="
manifest="${LOG_DIR}/${RUN_ROOT}/dirty_manifest.txt"
if [[ -f "${manifest}" ]]; then
  sed -n '/## Bucket counts/,/## Raw git status/p' "${manifest}" | sed '$d'
else
  echo "pending: ${manifest}"
fi

echo
echo "== log tail =="
tail -80 "${LOG_DIR}/${RUN_ROOT}.log" 2>/dev/null || true
