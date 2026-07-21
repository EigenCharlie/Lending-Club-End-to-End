#!/usr/bin/env bash
set -uo pipefail

# Full CRPTO/IJDS champion tournament launcher.
#
# This is a supervisor around the existing champion-replacement pipeline. It:
# - reruns audit + tournament preflight;
# - launches all predeclared external PD finalists through conformal reopen;
# - runs cuOpt full-universe portfolio frontier where conformal gates allow it;
# - delegates exact rerank to .venv/HiGHS;
# - refreshes live summaries while the long job runs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

RUN_ROOT="${RUN_ROOT:-paper1_crpto_ijds_tournament_2026_05_25_r1}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
ART_ROOT="${ART_ROOT:-/mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned}"
SUMMARY_INTERVAL_SEC="${SUMMARY_INTERVAL_SEC:-900}"

mkdir -p "${LOG_DIR}/${RUN_ROOT}/status"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"
exec > >(tee -a "${LOG_DIR}/${RUN_ROOT}.log") 2>&1

log() {
  echo "[$(date -Is)] $*"
}

write_supervisor_status() {
  local state="$1"
  local detail="${2:-}"
  "${MAIN_PYTHON}" - "${LOG_DIR}/${RUN_ROOT}/supervisor_status.json" "${RUN_ROOT}" "${state}" "${detail}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

path = Path(sys.argv[1])
run_root, state, detail = sys.argv[2:5]
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
payload.update(
    {
        "run_root": run_root,
        "state": state,
        "detail": detail,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
)
if state == "running" and "started_at_utc" not in payload:
    payload["started_at_utc"] = payload["updated_at_utc"]
if state in {"completed", "failed"}:
    payload["finished_at_utc"] = payload["updated_at_utc"]
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

summary_loop() {
  while true; do
    "${MAIN_PYTHON}" scripts/search/summarize_paper1_crpto_ijds_reopen.py \
      --run-root "${RUN_ROOT}" \
      --since-minutes 60 >/dev/null 2>&1 || true
    sleep "${SUMMARY_INTERVAL_SEC}"
  done
}

log "START full CRPTO/IJDS tournament run_root=${RUN_ROOT}"
log "resources: RAPIDS_ENV=${RAPIDS_ENV} ART_ROOT=${ART_ROOT}"
write_supervisor_status running "starting preflight"

RUN_ROOT="${RUN_ROOT}" RAPIDS_ENV="${RAPIDS_ENV}" \
  bash scripts/search/run_paper1_crpto_ijds_reopen_2026_05_25.sh tournament-preflight
preflight_rc=$?
if [[ "${preflight_rc}" -ne 0 ]]; then
  write_supervisor_status failed "preflight failed rc=${preflight_rc}"
  log "preflight failed rc=${preflight_rc}"
  exit "${preflight_rc}"
fi

# The nested preflight driver writes its own short-lived PID. Restore the
# supervisor PID so the monitor tracks the long-running process.
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"

summary_loop &
summary_pid=$!
echo "${summary_pid}" > "${LOG_DIR}/${RUN_ROOT}/summary_loop.pid"
trap 'status=$?; kill "${summary_pid}" >/dev/null 2>&1 || true; wait "${summary_pid}" >/dev/null 2>&1 || true; log "EXIT full tournament status=${status}"; exit "${status}"' EXIT

write_supervisor_status running "champion replacement pipeline running"

export ART_ROOT
export RUN_ROOT
export LOG_DIR
export RAPIDS_ENV
export MAIN_PYTHON
export CONFORMAL_PROFILE="${CONFORMAL_PROFILE:-search_conformal_reopen_decision_wide}"
export CANDIDATE_KEYS="${CANDIDATE_KEYS:-bureau_behavior_15 canonical_4 affordability_rate_5}"
export PORTFOLIO_VARIANTS="${PORTFOLIO_VARIANTS:-phase1,final}"
export RESUME_EXISTING_CONFORMAL="${RESUME_EXISTING_CONFORMAL:-1}"
export RESUME_EXISTING_PORTFOLIO="${RESUME_EXISTING_PORTFOLIO:-1}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-0}"
export SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-560}"
export RANDOM_STATES="${RANDOM_STATES:-42}"
export CUOPT_BATCH_SCHEDULE="${CUOPT_BATCH_SCHEDULE:-16,8,4,1}"
export CUOPT_BATCH_SIZE="${CUOPT_BATCH_SIZE:-16}"
export CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
export CUOPT_METHOD_SCHEDULE="${CUOPT_METHOD_SCHEDULE:-Concurrent,PDLP,Barrier}"
export CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-18}"
export CHAMPION_PIPELINE_MODE="${CHAMPION_PIPELINE_MODE:-wavefront}"
export CONFORMAL_MAX_PARALLEL="${CONFORMAL_MAX_PARALLEL:-3}"
export CONFORMAL_NICE="${CONFORMAL_NICE:-5}"

log "launch champion replacement pipeline mode=${CHAMPION_PIPELINE_MODE}: candidates=${CANDIDATE_KEYS} max_candidates=${MAX_CANDIDATES} shortlist=${SHORTLIST_TOP_K} cuopt_batches=${CUOPT_BATCH_SCHEDULE} cuopt_methods=${CUOPT_METHOD_SCHEDULE} conformal_parallel=${CONFORMAL_MAX_PARALLEL}"
if [[ "${CHAMPION_PIPELINE_MODE}" == "wavefront" ]]; then
  bash scripts/search/run_paper1_champion_replacement_wavefront_2026_05_25.sh
else
  bash scripts/search/run_paper1_champion_replacement_pipeline_2026_05_23.sh
fi
pipeline_rc=$?

"${MAIN_PYTHON}" scripts/search/summarize_paper1_crpto_ijds_reopen.py \
  --run-root "${RUN_ROOT}" \
  --since-minutes 240 || true

if [[ "${pipeline_rc}" -eq 0 ]]; then
  write_supervisor_status completed "champion replacement pipeline completed"
  log "END full CRPTO/IJDS tournament status=completed"
else
  write_supervisor_status failed "champion replacement pipeline failed rc=${pipeline_rc}"
  log "END full CRPTO/IJDS tournament status=failed rc=${pipeline_rc}"
fi
exit "${pipeline_rc}"
