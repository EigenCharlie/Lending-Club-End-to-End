#!/usr/bin/env bash
set -uo pipefail

# Wavefront champion-replacement runner:
# - stage all predeclared PD candidates first;
# - run conformal producers concurrently;
# - consume completed conformal namespaces through portfolio gates in a stable
#   predeclared order;
# - serialize GPU/cuOpt portfolio work with a lock and solver fallback schedule.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

export RUN_ROOT="${RUN_ROOT:-paper1_crpto_ijds_tournament_wavefront_2026_05_25}"
export LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
export EVENT_LOG="${EVENT_LOG:-${LOG_DIR}/${RUN_ROOT}_events.jsonl}"
export CANDIDATE_KEYS="${CANDIDATE_KEYS:-bureau_behavior_15 canonical_4 affordability_rate_5}"
export PORTFOLIO_VARIANTS="${PORTFOLIO_VARIANTS:-phase1,final}"
export RESUME_EXISTING_CONFORMAL="${RESUME_EXISTING_CONFORMAL:-1}"
export RESUME_EXISTING_PORTFOLIO="${RESUME_EXISTING_PORTFOLIO:-1}"
export CONFORMAL_MAX_PARALLEL="${CONFORMAL_MAX_PARALLEL:-3}"
export CONFORMAL_NICE="${CONFORMAL_NICE:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export EXECUTION_MODE="${EXECUTION_MODE:-search_tournament}"
export SOLVER_FRONTIER_BACKEND="${SOLVER_FRONTIER_BACKEND:-cuopt}"
export SOLVER_EXACT_BACKEND_PRIMARY="${SOLVER_EXACT_BACKEND_PRIMARY:-highs}"
export SOLVER_EXACT_BACKEND_SHADOW="${SOLVER_EXACT_BACKEND_SHADOW:-gurobi}"
export CUOPT_LADDER="${CUOPT_LADDER:-Concurrent:16,Concurrent:8,PDLP:16,Barrier:4,Concurrent:1}"
export CUOPT_STALL_TIMEOUT_MIN="${CUOPT_STALL_TIMEOUT_MIN:-30}"
export CUOPT_AUTO_DEMOTE_ON_STALL="${CUOPT_AUTO_DEMOTE_ON_STALL:-1}"

if [[ "${EXECUTION_MODE}" == "informs_lockdown" ]]; then
  echo "[$(date -Is)] execution_mode=informs_lockdown blocks champion search wavefront."
  echo "Use search_tournament for candidate discovery; informs_lockdown is replay/report only."
  exit 2
fi

mkdir -p "${LOG_DIR}/${RUN_ROOT}" "${LOG_DIR}/${RUN_ROOT}/status"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"

CHAMPION_PIPELINE_DEFINE_ONLY=1 source scripts/search/run_paper1_champion_replacement_pipeline_2026_05_23.sh
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"

gpu_lock="${LOG_DIR}/${RUN_ROOT}/gpu_portfolio.lock"

wait_for_slot() {
  local running
  while true; do
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if [[ "${running}" -lt "${CONFORMAL_MAX_PARALLEL}" ]]; then
      return 0
    fi
    sleep 15
  done
}

launch_conformal() {
  local key="$1"
  local tag conformal_run conformal_log status_path
  tag="$(candidate_tag "${key}")" || return 1
  conformal_run="${RUN_ROOT}__${key}__conformal"
  conformal_log="${LOG_DIR}/${conformal_run}.log"
  status_path="models/conformal_gap/${conformal_run}/conformal_reopen_status.json"

  if [[ "${RESUME_EXISTING_CONFORMAL}" == "1" && -f "${status_path}" ]]; then
    echo "[$(date -Is)] conformal ${key}: reuse ${conformal_run}"
    emit_event "conformal" "${key}" "reused" "${conformal_run}"
    return 0
  fi

  wait_for_slot
  echo "[$(date -Is)] conformal ${key}: launch ${conformal_run}"
  emit_event "conformal" "${key}" "running" "${conformal_run}"
  (
    set +e
    nice -n "${CONFORMAL_NICE}" uv run python scripts/search/run_conformal_reopen_search.py \
      --run-tag "${conformal_run}" \
      --pipeline-profile "${CONFORMAL_PROFILE}" \
      --upstream-canonical-run-tag "${tag}" \
      >"${conformal_log}" 2>&1
    rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      emit_event "conformal" "${key}" "completed" "${conformal_run}"
    else
      emit_event "conformal" "${key}" "failed" "${conformal_log}"
    fi
    exit "${rc}"
  ) &
  echo "$!" > "${LOG_DIR}/${RUN_ROOT}/status/conformal_${key}.pid"
}

wait_conformal_done() {
  local key="$1"
  local pid_path pid
  pid_path="${LOG_DIR}/${RUN_ROOT}/status/conformal_${key}.pid"
  if [[ -f "${pid_path}" ]]; then
    pid="$(cat "${pid_path}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[$(date -Is)] waiting conformal ${key} pid=${pid}"
      wait "${pid}"
      return $?
    fi
  fi
  return 0
}

run_portfolio_locked() {
  local intervals_path="$1"
  local run_label="$2"
  (
    flock 9
    echo "[$(date -Is)] gpu lock acquired ${run_label}"
    run_portfolio "${intervals_path}" "${run_label}"
  ) 9>"${gpu_lock}"
}

portfolio_for_variant() {
  local key="$1"
  local variant="$2"
  local namespace="$3"
  local intervals portfolio_run portfolio_log portfolio_selection gate event_stage
  [[ -z "${namespace}" ]] && return 0
  intervals="data/processed/conformal_gap/${namespace}/conformal_intervals_mondrian.parquet"
  portfolio_run="${RUN_ROOT}__${key}__${variant}__portfolio"
  portfolio_log="${LOG_DIR}/${portfolio_run}.log"
  portfolio_selection="models/portfolio_bound_aware/${portfolio_run}/portfolio_bound_aware_selection.json"
  event_stage="portfolio_${variant}"

  gate="$(portfolio_gate "${namespace}")"
  echo "[$(date -Is)] portfolio gate ${variant} ${key}: ${gate}"
  if [[ "${RESUME_EXISTING_PORTFOLIO}" == "1" && -f "${portfolio_selection}" ]]; then
    echo "[$(date -Is)] portfolio ${variant} ${key}: reuse ${portfolio_run}"
    emit_event "${event_stage}" "${key}" "reused" "${portfolio_run}"
  elif [[ "${gate}" == "run" ]]; then
    echo "[$(date -Is)] portfolio ${variant} ${key}: ${portfolio_run}"
    emit_event "${event_stage}" "${key}" "running" "${portfolio_run}"
    if run_portfolio_locked "${intervals}" "${portfolio_run}" >"${portfolio_log}" 2>&1; then
      emit_event "${event_stage}" "${key}" "completed" "${portfolio_run}"
    else
      echo "[$(date -Is)] portfolio ${variant} failed ${key}; continuing"
      emit_event "${event_stage}" "${key}" "failed" "${portfolio_log}"
    fi
  else
    emit_event "${event_stage}" "${key}" "skipped" "${gate}"
  fi
}

echo "[$(date -Is)] START wavefront run_root=${RUN_ROOT} candidates=${CANDIDATE_KEYS}"

for key in ${CANDIDATE_KEYS}; do
  tag="$(candidate_tag "${key}")"
  src="$(candidate_source_dir "${key}")"
  echo "[$(date -Is)] staging ${key} -> ${tag}"
  emit_event "staging" "${key}" "running" "${tag}"
  if stage_candidate "${key}" "${tag}" "${src}"; then
    emit_event "staging" "${key}" "completed" "${tag}"
  else
    echo "[$(date -Is)] staging failed ${key}; conformal not launched"
    emit_event "staging" "${key}" "failed" "${src}"
    continue
  fi
  launch_conformal "${key}"
done

for key in ${CANDIDATE_KEYS}; do
  conformal_run="${RUN_ROOT}__${key}__conformal"
  status_path="models/conformal_gap/${conformal_run}/conformal_reopen_status.json"
  wait_conformal_done "${key}" || true

  if [[ ! -f "${status_path}" ]]; then
    echo "[$(date -Is)] missing conformal status ${key}: ${status_path}; continuing"
    emit_event "conformal_status" "${key}" "failed" "${status_path}"
    continue
  fi

  phase1_ns="$(status_value "${status_path}" "phase1_oot_namespace")"
  final_ns="$(status_value "${status_path}" "final_namespace")"

  if [[ "${PORTFOLIO_VARIANTS}" == *"phase1"* ]]; then
    portfolio_for_variant "${key}" "phase1" "${phase1_ns}"
  fi
  if [[ "${PORTFOLIO_VARIANTS}" == *"final"* && "${final_ns}" != "${phase1_ns}" ]]; then
    portfolio_for_variant "${key}" "final" "${final_ns}"
  elif [[ "${PORTFOLIO_VARIANTS}" == *"final"* ]]; then
    emit_event "portfolio_final" "${key}" "skipped" "same_as_phase1"
  fi
done

wait || true
"${MAIN_PYTHON}" scripts/search/summarize_paper1_crpto_ijds_reopen.py \
  --run-root "${RUN_ROOT}" \
  --since-minutes 240 || true
echo "[$(date -Is)] END wavefront run_root=${RUN_ROOT}"
