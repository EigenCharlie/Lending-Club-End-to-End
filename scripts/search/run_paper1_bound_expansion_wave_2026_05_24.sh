#!/usr/bin/env bash
set -uo pipefail

# Secondary expansion wave for Paper Estrella bound/champion reopen.
#
# It does not mutate canonical champion artifacts and does not interrupt the
# currently running full canonical_4 search. It reuses conformal results when
# available, runs small CPU segment-policy probes, then optionally queues medium
# cuOpt triage after the primary full pipeline is idle.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-paper1_bound_expansion_2026_05_24_r1}"
PRIMARY_RUN_ROOT="${PRIMARY_RUN_ROOT:-paper1_champion_replacement_2026_05_23_r3}"
CONFORMAL_RUN_ROOT="${CONFORMAL_RUN_ROOT:-paper1_champion_replacement_2026_05_23_r3}"
CANDIDATE_KEYS="${CANDIDATE_KEYS:-canonical_4 bureau_behavior_15 affordability_rate_5}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
mkdir -p "${LOG_DIR}"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"
EVENT_LOG="${EVENT_LOG:-${LOG_DIR}/${RUN_ROOT}_events.jsonl}"

SEGMENT_PROBE_MAX_CANDIDATES="${SEGMENT_PROBE_MAX_CANDIDATES:-25000}"
SEGMENT_PROBE_SHORTLIST_TOP_K="${SEGMENT_PROBE_SHORTLIST_TOP_K:-120}"
SEGMENT_PROBE_RANDOM_STATES="${SEGMENT_PROBE_RANDOM_STATES:-42}"
SEGMENT_PROBE_SOLVER="${SEGMENT_PROBE_SOLVER:-highs}"

RUN_MEDIUM_TRIAGE="${RUN_MEDIUM_TRIAGE:-1}"
WAIT_FOR_PRIMARY_BEFORE_MEDIUM="${WAIT_FOR_PRIMARY_BEFORE_MEDIUM:-1}"
DEFER_MEDIUM_UNTIL_AFTER_PROBES="${DEFER_MEDIUM_UNTIL_AFTER_PROBES:-1}"
MEDIUM_MAX_CANDIDATES="${MEDIUM_MAX_CANDIDATES:-75000}"
MEDIUM_SHORTLIST_TOP_K="${MEDIUM_SHORTLIST_TOP_K:-240}"
MEDIUM_RANDOM_STATES="${MEDIUM_RANDOM_STATES:-42}"
MEDIUM_SOLVER="${MEDIUM_SOLVER:-cuopt}"
MEDIUM_CUOPT_BATCH_SIZE="${MEDIUM_CUOPT_BATCH_SIZE:-8}"
MEDIUM_CUOPT_METHOD="${MEDIUM_CUOPT_METHOD:-Concurrent}"
MEDIUM_CUOPT_PDLP_SOLVER_MODE="${MEDIUM_CUOPT_PDLP_SOLVER_MODE:-}"
MEDIUM_CUOPT_NUM_CPU_THREADS="${MEDIUM_CUOPT_NUM_CPU_THREADS:-24}"

log() {
  echo "[$(date -Is)] $*" >&2
}

emit_event() {
  local stage="$1"
  local key="$2"
  local state="$3"
  local detail="${4:-}"
  "${MAIN_PYTHON}" - "$EVENT_LOG" "$RUN_ROOT" "$stage" "$key" "$state" "$detail" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

event_log, run_root, stage, key, state, detail = sys.argv[1:7]
path = Path(event_log)
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "ts": datetime.now(UTC).isoformat(),
    "run_root": run_root,
    "stage": stage,
    "candidate_key": key,
    "state": state,
    "detail": detail,
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

candidate_tag() {
  case "$1" in
    bureau_behavior_15) echo "regret_auditability_pd_bureau_behavior_15_2026_05_21" ;;
    canonical_4) echo "regret_auditability_pd_canonical_4_2026_05_23" ;;
    affordability_rate_5) echo "regret_auditability_pd_affordability_rate_5_2026_05_23" ;;
    *) return 2 ;;
  esac
}

status_value() {
  local path="$1"
  local key="$2"
  "${MAIN_PYTHON}" - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload
for part in key.split("."):
    value = value.get(part, "") if isinstance(value, dict) else ""
print(value if value is not None else "")
PY
}

wait_for_conformal_status() {
  local key="$1"
  local tag="$2"
  local conformal_run="${CONFORMAL_RUN_ROOT}__${key}__conformal"
  local status_path="models/conformal_gap/${conformal_run}/conformal_reopen_status.json"
  local prefetch_pid="${LOG_DIR}/${conformal_run}_prefetch.pid"
  local conformal_log="${LOG_DIR}/${conformal_run}_expansion.log"
  if [[ -f "${status_path}" ]]; then
    echo "${status_path}"
    return 0
  fi
  if [[ -f "${prefetch_pid}" ]] && ps -p "$(cat "${prefetch_pid}")" >/dev/null 2>&1; then
    log "waiting for conformal prefetch ${key} pid $(cat "${prefetch_pid}")"
  else
    log "starting conformal for ${key}: ${conformal_run}"
    emit_event "conformal" "${key}" "running" "${conformal_run}"
    uv run python scripts/search/run_conformal_reopen_search.py \
      --run-tag "${conformal_run}" \
      --pipeline-profile search_conformal_reopen_decision_wide \
      --upstream-canonical-run-tag "${tag}" \
      >"${conformal_log}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
      log "conformal failed ${key}: ${conformal_log}"
      emit_event "conformal" "${key}" "failed" "${conformal_log}"
      return 1
    fi
    emit_event "conformal" "${key}" "completed" "${conformal_run}"
  fi
  while [[ ! -f "${status_path}" ]]; do
    if [[ -f "${prefetch_pid}" ]] && ! ps -p "$(cat "${prefetch_pid}")" >/dev/null 2>&1; then
      if [[ ! -f "${status_path}" ]]; then
        log "prefetch exited without status for ${key}; see ${LOG_DIR}/${conformal_run}_prefetch.log"
        emit_event "conformal" "${key}" "failed" "${LOG_DIR}/${conformal_run}_prefetch.log"
        return 1
      fi
    fi
    emit_event "conformal_wait" "${key}" "running" "${status_path}"
    sleep 120
  done
  emit_event "conformal" "${key}" "available" "${conformal_run}"
  echo "${status_path}"
}

run_diagnostics_if_possible() {
  local key="$1"
  local portfolio_run="$2"
  local intervals="$3"
  local selection="models/portfolio_bound_aware/${portfolio_run}/portfolio_bound_aware_selection.json"
  if [[ ! -f "${selection}" ]]; then
    return 0
  fi
  local diagnostic_run="${portfolio_run}__diagnostics"
  local diagnostic_status="models/bound_diagnostics/${diagnostic_run}/bound_decision_diagnostics_status.json"
  local diagnostic_log="${LOG_DIR}/${diagnostic_run}.log"
  if [[ -f "${diagnostic_status}" ]]; then
    emit_event "diagnostics" "${key}" "reused" "${diagnostic_run}"
    return 0
  fi
  log "diagnostics ${key}: ${diagnostic_run}"
  emit_event "diagnostics" "${key}" "running" "${diagnostic_run}"
  if "${MAIN_PYTHON}" scripts/search/run_bound_decision_diagnostics.py \
    --selection-path "${selection}" \
    --conformal-intervals-path "${intervals}" \
    --run-label "${diagnostic_run}" \
    --alpha 0.01 \
    --max-candidates 0 \
    --random-state 42 \
    >"${diagnostic_log}" 2>&1; then
    emit_event "diagnostics" "${key}" "completed" "${diagnostic_run}"
  else
    emit_event "diagnostics" "${key}" "failed" "${diagnostic_log}"
    log "diagnostics failed ${key}; continuing"
  fi
}

run_portfolio() {
  local key="$1"
  local run_label="$2"
  local intervals="$3"
  local solver="$4"
  local max_candidates="$5"
  local shortlist_top_k="$6"
  local random_states="$7"
  local profile="$8"
  local log_path="${LOG_DIR}/${run_label}.log"
  local selection="models/portfolio_bound_aware/${run_label}/portfolio_bound_aware_selection.json"
  if [[ -f "${selection}" ]]; then
    log "portfolio ${profile} ${key}: reuse ${run_label}"
    emit_event "portfolio_${profile}" "${key}" "reused" "${run_label}"
    run_diagnostics_if_possible "${key}" "${run_label}" "${intervals}"
    return 0
  fi
  log "portfolio ${profile} ${key}: ${run_label} solver=${solver} max_candidates=${max_candidates}"
  emit_event "portfolio_${profile}" "${key}" "running" "${run_label}"
  local cmd_prefix=("${MAIN_PYTHON}")
  if [[ "${solver}" == "cuopt" ]]; then
    cmd_prefix=(conda run -n "${RAPIDS_ENV}" python)
  fi

  local risk_grid gamma_grid aversion_grid delta_grid tail_grid policy_modes alpha_grid
  if [[ "${profile}" == "segment_probe" ]]; then
    risk_grid="0.160,0.165,0.170,0.175,0.180,0.185,0.190"
    gamma_grid="0.05,0.10,0.20,0.35,0.50"
    aversion_grid="0,0.05,0.10,0.20"
    delta_grid="1.0"
    tail_grid="0.75,0.90,0.95"
    policy_modes="segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty"
    alpha_grid="0.01,0.03,0.10"
  else
    risk_grid="0.160,0.165,0.170,0.175,0.180,0.185,0.190"
    gamma_grid="0.325,0.375,0.400,0.425,0.450,0.475,0.500,0.550,0.600"
    aversion_grid="0,0.02,0.05,0.10,0.15,0.25,0.50"
    delta_grid="0.75,0.90,0.95,1.0"
    tail_grid="0.75,0.90,0.95,1.0"
    policy_modes="blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty"
    alpha_grid="0.01,0.03,0.05,0.10"
  fi

  if "${cmd_prefix[@]}" scripts/search/run_portfolio_bound_aware_search.py \
    --conformal-intervals-path "${intervals}" \
    --run-label "${run_label}" \
    --risk-grid "${risk_grid}" \
    --gamma-grid "${gamma_grid}" \
    --aversion-grid "${aversion_grid}" \
    --policy-modes "${policy_modes}" \
    --enable-segment-policy-grid \
    --delta-cap-grid "${delta_grid}" \
    --tail-focus-grid "${tail_grid}" \
    --budget-profiles free \
    --shortlist-top-k "${shortlist_top_k}" \
    --bucket-return-k 100 \
    --bucket-proxy-k 100 \
    --bucket-family-k 60 \
    --bucket-region-k 80 \
    --alpha-grid "${alpha_grid}" \
    --max-candidates "${max_candidates}" \
    --random-states "${random_states}" \
    --solver-backend "${solver}" \
    --exact-solver-backend highs \
    --exact-python-executable "${MAIN_PYTHON}" \
    --cuopt-batch-size "${MEDIUM_CUOPT_BATCH_SIZE}" \
    --cuopt-method "${MEDIUM_CUOPT_METHOD}" \
    --cuopt-pdlp-solver-mode "${MEDIUM_CUOPT_PDLP_SOLVER_MODE}" \
    --cuopt-num-cpu-threads "${MEDIUM_CUOPT_NUM_CPU_THREADS}" \
    --cuopt-dual-postsolve 0 \
    --incumbent-policy-path models/champion_portfolio_policy.json \
    --incumbent-risk-neighbors 0.165,0.170,0.175,0.180,0.185,0.190 \
    --incumbent-gamma-neighbors 0.375,0.400,0.425,0.450,0.475,0.500,0.550 \
    --incumbent-policy-modes "${policy_modes}" \
    >"${log_path}" 2>&1; then
    emit_event "portfolio_${profile}" "${key}" "completed" "${run_label}"
    run_diagnostics_if_possible "${key}" "${run_label}" "${intervals}"
  else
    emit_event "portfolio_${profile}" "${key}" "failed" "${log_path}"
    log "portfolio ${profile} failed ${key}; continuing"
    return 1
  fi
}

segment_gate_for_medium() {
  local run_label="$1"
  local selection="models/portfolio_bound_aware/${run_label}/portfolio_bound_aware_selection.json"
  "${MAIN_PYTHON}" - "${selection}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("skip:missing_selection")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
metrics = payload.get("selected_metrics", {})
passed = bool(metrics.get("alpha01_exact_pass", False))
n_funded = int(float(metrics.get("n_funded", 0) or 0))
allocated = float(metrics.get("total_allocated", 0) or 0)
if passed and n_funded >= 40 and allocated >= 500000:
    print("run")
else:
    print(f"skip:alpha01={passed},n_funded={n_funded},allocated={allocated:.0f}")
PY
}

wait_for_primary_idle() {
  if [[ "${WAIT_FOR_PRIMARY_BEFORE_MEDIUM}" != "1" ]]; then
    return 0
  fi
  local pid_file="${LOG_DIR}/${PRIMARY_RUN_ROOT}.pid"
  while [[ -f "${pid_file}" ]] && ps -p "$(cat "${pid_file}")" >/dev/null 2>&1; do
    emit_event "medium_wait_primary" "all" "running" "${PRIMARY_RUN_ROOT}"
    log "medium triage waiting for primary pipeline ${PRIMARY_RUN_ROOT} pid $(cat "${pid_file}")"
    sleep 600
  done
}

log "expansion wave start: ${RUN_ROOT}"
emit_event "wave" "all" "running" "start"
MEDIUM_QUEUE="${LOG_DIR}/${RUN_ROOT}_medium_queue.tsv"
: > "${MEDIUM_QUEUE}"

for key in ${CANDIDATE_KEYS}; do
  tag="$(candidate_tag "${key}")"
  if [[ -z "${tag}" ]]; then
    emit_event "candidate" "${key}" "failed" "unknown_candidate"
    continue
  fi
  emit_event "candidate" "${key}" "running" "${tag}"
  status_path="$(wait_for_conformal_status "${key}" "${tag}")"
  if [[ -z "${status_path}" || ! -f "${status_path}" ]]; then
    emit_event "candidate" "${key}" "failed" "missing_conformal_status"
    continue
  fi
  final_ns="$(status_value "${status_path}" "final_namespace")"
  if [[ -z "${final_ns}" ]]; then
    emit_event "candidate" "${key}" "failed" "missing_final_namespace"
    continue
  fi
  intervals="data/processed/conformal_gap/${final_ns}/conformal_intervals_mondrian.parquet"
  if [[ ! -f "${intervals}" ]]; then
    emit_event "candidate" "${key}" "failed" "missing_intervals:${intervals}"
    continue
  fi

  segment_run="${RUN_ROOT}__${key}__segment_probe_25k"
  run_portfolio \
    "${key}" \
    "${segment_run}" \
    "${intervals}" \
    "${SEGMENT_PROBE_SOLVER}" \
    "${SEGMENT_PROBE_MAX_CANDIDATES}" \
    "${SEGMENT_PROBE_SHORTLIST_TOP_K}" \
    "${SEGMENT_PROBE_RANDOM_STATES}" \
    "segment_probe"

  if [[ "${RUN_MEDIUM_TRIAGE}" == "1" ]]; then
    gate="$(segment_gate_for_medium "${segment_run}")"
    log "medium gate ${key}: ${gate}"
    emit_event "medium_gate" "${key}" "${gate%%:*}" "${gate}"
    if [[ "${gate}" == "run" ]]; then
      if [[ "${DEFER_MEDIUM_UNTIL_AFTER_PROBES}" == "1" ]]; then
        printf "%s\t%s\n" "${key}" "${intervals}" >> "${MEDIUM_QUEUE}"
        emit_event "medium_queue" "${key}" "queued" "${intervals}"
      else
        wait_for_primary_idle
        medium_run="${RUN_ROOT}__${key}__medium_triage_75k"
        run_portfolio \
          "${key}" \
          "${medium_run}" \
          "${intervals}" \
          "${MEDIUM_SOLVER}" \
          "${MEDIUM_MAX_CANDIDATES}" \
          "${MEDIUM_SHORTLIST_TOP_K}" \
          "${MEDIUM_RANDOM_STATES}" \
          "medium_triage"
      fi
    fi
  fi
  emit_event "candidate" "${key}" "completed" "${tag}"
done

if [[ "${RUN_MEDIUM_TRIAGE}" == "1" && "${DEFER_MEDIUM_UNTIL_AFTER_PROBES}" == "1" && -s "${MEDIUM_QUEUE}" ]]; then
  log "medium queue ready: ${MEDIUM_QUEUE}"
  wait_for_primary_idle
  while IFS=$'\t' read -r key intervals; do
    [[ -z "${key}" || -z "${intervals}" ]] && continue
    medium_run="${RUN_ROOT}__${key}__medium_triage_75k"
    run_portfolio \
      "${key}" \
      "${medium_run}" \
      "${intervals}" \
      "${MEDIUM_SOLVER}" \
      "${MEDIUM_MAX_CANDIDATES}" \
      "${MEDIUM_SHORTLIST_TOP_K}" \
      "${MEDIUM_RANDOM_STATES}" \
      "medium_triage"
  done < "${MEDIUM_QUEUE}"
fi

emit_event "wave" "all" "completed" "done"
log "expansion wave complete: ${RUN_ROOT}"
