#!/usr/bin/env bash
set -uo pipefail

# Governed CRPTO/IJDS champion-reopen driver.
#
# Default mode is audit-only. Run portfolio smoke explicitly:
#   RUN_ROOT=paper1_crpto_reopen_ijds_2026_05_25_r1 \
#   bash scripts/search/run_paper1_crpto_ijds_reopen_2026_05_25.sh canonical-smoke

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

MODE="${1:-${MODE:-audit}}"
RUN_ROOT="${RUN_ROOT:-paper1_crpto_reopen_ijds_2026_05_25}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${HOME}/scratch/lending-club-runs/${RUN_ROOT}}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
EXECUTION_MODE="${EXECUTION_MODE:-search_tournament}"

UPSTREAM_CANONICAL_RUN_TAG="${UPSTREAM_CANONICAL_RUN_TAG:-regret_auditability_pd_canonical_4_2026_05_23}"
PD_HPO_TRIALS="${PD_HPO_TRIALS:-80}"
PD_HPO_CARRIERS="${PD_HPO_CARRIERS:-canonical_4}"
SMOKE_VARIANTS="${SMOKE_VARIANTS:-score8_cal_none_a010}"

PORTFOLIO_SOLVER_BACKEND="${PORTFOLIO_SOLVER_BACKEND:-highs}"
PORTFOLIO_EXACT_SOLVER_BACKEND="${PORTFOLIO_EXACT_SOLVER_BACKEND:-highs}"
MAX_CANDIDATES="${MAX_CANDIDATES:-25000}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-180}"
RISK_GRID="${RISK_GRID:-0.165,0.170,0.175,0.180,0.185,0.190,0.195,0.200}"
GAMMA_GRID="${GAMMA_GRID:-0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600}"
AVERSION_GRID="${AVERSION_GRID:-0,0.05,0.10,0.15,0.20,0.25}"
POLICY_MODES="${POLICY_MODES:-blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty}"
DELTA_CAP_GRID="${DELTA_CAP_GRID:-0.60,0.70,0.75,0.80,0.90,1.0}"
TAIL_FOCUS_GRID="${TAIL_FOCUS_GRID:-0.90,0.95,1.0}"
ALPHA_GRID="${ALPHA_GRID:-0.01}"
RANDOM_STATES="${RANDOM_STATES:-42}"
CUOPT_BATCH_SIZE="${CUOPT_BATCH_SIZE:-1}"
CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-12}"
CUOPT_DUAL_POSTSOLVE="${CUOPT_DUAL_POSTSOLVE:-0}"
CUOPT_LADDER="${CUOPT_LADDER:-Concurrent:16,Concurrent:8,PDLP:16,Barrier:4,Concurrent:1}"
CUOPT_STALL_TIMEOUT_MIN="${CUOPT_STALL_TIMEOUT_MIN:-30}"
CUOPT_AUTO_DEMOTE_ON_STALL="${CUOPT_AUTO_DEMOTE_ON_STALL:-1}"
ALLOW_MIXED_RAPIDS="${ALLOW_MIXED_RAPIDS:-0}"
ENABLE_HIGHS_FALLBACK="${ENABLE_HIGHS_FALLBACK:-1}"
TOURNAMENT_PROFILE="${TOURNAMENT_PROFILE:-configs/run_profiles/paper1_crpto_ijds_champion_tournament_2026_05_25.yaml}"

mkdir -p "${LOG_DIR}/${RUN_ROOT}/status" "${SCRATCH_ROOT}"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"
exec > >(tee -a "${LOG_DIR}/${RUN_ROOT}.log") 2>&1

log() {
  echo "[$(date -Is)] $*"
}

write_status() {
  local phase="$1"
  local state="$2"
  local exit_code="${3:-}"
  local skipped="${4:-0}"
  local reason="${5:-}"
  local command="${6:-}"
  "${MAIN_PYTHON}" - "${LOG_DIR}/${RUN_ROOT}/status/${phase}.json" "${phase}" "${state}" "${exit_code}" "${skipped}" "${reason}" "${command}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

path = Path(sys.argv[1])
phase, state, exit_code, skipped, reason, command = sys.argv[2:8]
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
payload["phase"] = phase
payload["state"] = state
payload["updated_at_utc"] = datetime.now(UTC).isoformat()
if state == "running" and "started_at_utc" not in payload:
    payload["started_at_utc"] = payload["updated_at_utc"]
if state in {"completed", "failed", "skipped"}:
    payload["finished_at_utc"] = payload["updated_at_utc"]
if exit_code != "":
    payload["exit_code"] = int(exit_code)
payload["skipped"] = skipped == "1"
if reason:
    payload["reason"] = reason
if command:
    payload["command"] = command
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

run_phase() {
  local phase="$1"
  shift
  local command="$*"
  log "START phase=${phase} command=${command}"
  write_status "${phase}" "running" "" 0 "" "${command}"
  "$@"
  local rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    write_status "${phase}" "completed" "${rc}" 0 "" "${command}"
    log "END phase=${phase} status=completed"
  else
    write_status "${phase}" "failed" "${rc}" 0 "phase command exited nonzero" "${command}"
    log "END phase=${phase} status=failed rc=${rc}"
  fi
  return "${rc}"
}

summarize_live() {
  "${MAIN_PYTHON}" scripts/search/summarize_paper1_crpto_ijds_reopen.py \
    --run-root "${RUN_ROOT}" \
    --since-minutes 30 || log "live summary failed"
}

apply_execution_mode_policy() {
  if [[ "${EXECUTION_MODE}" == "informs_lockdown" ]]; then
    PORTFOLIO_SOLVER_BACKEND="highs"
    PORTFOLIO_EXACT_SOLVER_BACKEND="highs"
  fi
}

pd_base_tag() {
  case "$1" in
    canonical_4) echo "regret_auditability_pd_canonical_4_2026_05_23" ;;
    bureau_behavior_15) echo "regret_auditability_pd_bureau_behavior_15_2026_05_21" ;;
    affordability_rate_5) echo "regret_auditability_pd_affordability_rate_5_2026_05_23" ;;
    *) return 1 ;;
  esac
}

variant_spec() {
  case "$1" in
    score8_cal_none_a010)
      echo "${RUN_ROOT}__canonical4_score8_cal_none_a010|score_decile_mondrian|calibrated|8|grade_then_global|0.10|none|false|1000"
      ;;
    score8_raw_sqrt_a010)
      echo "${RUN_ROOT}__canonical4_score8_raw_sqrt_a010|score_decile_mondrian|raw|8|grade_then_global|0.10|bernoulli_sqrt|true|1000"
      ;;
    grade_cal_sqrt_a0090)
      echo "${RUN_ROOT}__canonical4_grade_cal_sqrt_a0090|grade|calibrated|5|grade_then_global|0.09|bernoulli_sqrt|true|1000"
      ;;
    *)
      return 1
      ;;
  esac
}

run_audit() {
  run_phase env_audit "${MAIN_PYTHON}" scripts/search/audit_crpto_reopen_env.py \
    --run-root "${RUN_ROOT}" \
    --log-dir "${LOG_DIR}" \
    --rapids-env "${RAPIDS_ENV}"
}

run_tournament_preflight() {
  run_phase tournament_preflight "${MAIN_PYTHON}" scripts/search/prepare_paper1_crpto_ijds_tournament.py \
    --run-root "${RUN_ROOT}" \
    --profile-path "${TOURNAMENT_PROFILE}" \
    --log-dir "${LOG_DIR}"
  summarize_live
}

run_pd_hpo() {
  local carrier base run_tag
  for carrier in ${PD_HPO_CARRIERS//,/ }; do
    base="$(pd_base_tag "${carrier}")" || {
      write_status "pd_hpo_${carrier}" "skipped" 0 1 "unknown PD carrier ${carrier}" ""
      continue
    }
    run_tag="${RUN_ROOT}__pd_hpo__${carrier}"
    run_phase "pd_hpo_${carrier}" env \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}" \
        "${MAIN_PYTHON}" scripts/search/run_pd_hpo_local.py \
        --run-tag "${run_tag}" \
        --base-search-run-tag "${base}" \
        --hpo-n-trials "${PD_HPO_TRIALS}"
    summarize_live
  done
}

run_conformal_variant() {
  local variant="$1"
  local spec namespace partition prob_source bins fallback alpha90 scale scaled mgs intervals
  spec="$(variant_spec "${variant}")" || {
    write_status "conformal_${variant}" "skipped" 0 1 "unknown smoke variant ${variant}" ""
    return 0
  }
  IFS='|' read -r namespace partition prob_source bins fallback alpha90 scale scaled mgs <<< "${spec}"
  intervals="data/processed/conformal_gap/${namespace}/conformal_intervals_mondrian.parquet"
  if [[ -f "${intervals}" ]]; then
    log "reuse conformal ${variant}: ${intervals}"
    write_status "conformal_${variant}" "completed" 0 0 "reused existing intervals" "reuse ${intervals}"
    return 0
  fi
  run_phase "conformal_${variant}" env \
      PIPELINE_RUN_TAG="${namespace}" \
      UPSTREAM_CANONICAL_RUN_TAG="${UPSTREAM_CANONICAL_RUN_TAG}" \
      "${MAIN_PYTHON}" scripts/generate_conformal_intervals.py \
      --artifact_namespace "${namespace}" \
      --evaluation_scope test \
      --partition "${partition}" \
      --partition_candidates "${partition}" \
      --partition_probability_sources "${prob_source}" \
      --n_score_bins_candidates "${bins}" \
      --fallback_modes "${fallback}" \
      --alpha_candidates_90 "${alpha90}" \
      --alpha_candidates_95 0.05 \
      --min_group_sizes "${mgs}" \
      --score_scale_families "${scale}" \
      --scaled_scores_options "${scaled}" \
      --calibration_fraction 1.0
  summarize_live
}

cuopt_gate_allows() {
  if [[ "${PORTFOLIO_SOLVER_BACKEND}" != "cuopt" ]]; then
    return 0
  fi
  if [[ "${ALLOW_MIXED_RAPIDS}" == "1" ]]; then
    return 0
  fi
  "${MAIN_PYTHON}" - "${LOG_DIR}/${RUN_ROOT}/env_audit.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(2)
payload = json.loads(path.read_text(encoding="utf-8"))
requires_clean = bool(
    payload.get("environment_recommendation", {}).get(
        "requires_clean_cuopt_env_before_serious_run", True
    )
)
raise SystemExit(1 if requires_clean else 0)
PY
}

portfolio_size_token() {
  if [[ "${MAX_CANDIDATES}" == "0" ]]; then
    echo "full"
  elif [[ "${MAX_CANDIDATES}" =~ 000$ ]]; then
    echo "$((MAX_CANDIDATES / 1000))k"
  else
    echo "${MAX_CANDIDATES}"
  fi
}

run_portfolio_once() {
  local variant="$1"
  local solver_backend="$2"
  local spec namespace intervals run_label selection size_token
  spec="$(variant_spec "${variant}")" || {
    write_status "portfolio_${variant}" "skipped" 0 1 "unknown smoke variant ${variant}" ""
    return 0
  }
  IFS='|' read -r namespace _partition _prob_source _bins _fallback _alpha90 _scale _scaled _mgs <<< "${spec}"
  intervals="data/processed/conformal_gap/${namespace}/conformal_intervals_mondrian.parquet"
  size_token="$(portfolio_size_token)"
  run_label="${RUN_ROOT}__${variant}__portfolio_smoke_${size_token}_${solver_backend}"
  selection="models/portfolio_bound_aware/${run_label}/portfolio_bound_aware_selection.json"
  if [[ ! -f "${intervals}" ]]; then
    write_status "portfolio_${variant}_${solver_backend}" "skipped" 0 1 "missing intervals ${intervals}" ""
    log "skip portfolio ${variant}: missing ${intervals}"
    return 0
  fi
  if [[ -f "${selection}" ]]; then
    log "reuse portfolio ${variant}: ${selection}"
    write_status "portfolio_${variant}_${solver_backend}" "completed" 0 0 "reused existing selection" "reuse ${selection}"
    return 0
  fi

  if [[ "${solver_backend}" == "cuopt" ]]; then
    if [[ "${EXECUTION_MODE}" == "informs_lockdown" ]]; then
      write_status "portfolio_${variant}_${solver_backend}" "skipped" 0 1 "execution_mode=informs_lockdown forbids cuOpt search" ""
      log "skip cuOpt portfolio ${variant}: execution_mode=informs_lockdown"
      return 0
    fi
    if ! cuopt_gate_allows; then
      write_status "portfolio_${variant}_${solver_backend}" "skipped" 0 1 "cuOpt env gate blocked; set ALLOW_MIXED_RAPIDS=1 only for explicit smoke" ""
      log "skip cuOpt portfolio ${variant}: env gate blocked"
      return 0
    fi
    run_phase "portfolio_${variant}_${solver_backend}" timeout --signal=TERM --kill-after=60 "${CUOPT_STALL_TIMEOUT_MIN}m" conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
      --conformal-intervals-path "${intervals}" \
      --run-label "${run_label}" \
      --risk-grid "${RISK_GRID}" \
      --gamma-grid "${GAMMA_GRID}" \
      --aversion-grid "${AVERSION_GRID}" \
      --policy-modes "${POLICY_MODES}" \
      --delta-cap-grid "${DELTA_CAP_GRID}" \
      --tail-focus-grid "${TAIL_FOCUS_GRID}" \
      --budget-profiles free \
      --shortlist-top-k "${SHORTLIST_TOP_K}" \
      --bucket-return-k 120 \
      --bucket-proxy-k 90 \
      --bucket-family-k 40 \
      --bucket-region-k 60 \
      --alpha-grid "${ALPHA_GRID}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --random-states "${RANDOM_STATES}" \
      --solver-backend cuopt \
      --exact-solver-backend "${PORTFOLIO_EXACT_SOLVER_BACKEND}" \
      --exact-shadow-backend gurobi \
      --exact-shadow-top-k 20 \
      --exact-solver-agreement-return-abs 250.0 \
      --exact-solver-agreement-v-abs 0.002 \
      --exact-solver-agreement-gamma-abs 0.005 \
      --exact-pass1-alpha 0.01 \
      --exact-pass2-bucket-min 8 \
      --exact-workers 4 \
      --exact-checkpoint-every 32 \
      --exact-python-executable "${MAIN_PYTHON}" \
      --cuopt-batch-size "${CUOPT_BATCH_SIZE}" \
      --cuopt-method "${CUOPT_METHOD}" \
      --cuopt-num-cpu-threads "${CUOPT_NUM_CPU_THREADS}" \
      --cuopt-dual-postsolve "${CUOPT_DUAL_POSTSOLVE}"
  else
    run_phase "portfolio_${variant}_${solver_backend}" "${MAIN_PYTHON}" scripts/search/run_portfolio_bound_aware_search.py \
      --conformal-intervals-path "${intervals}" \
      --run-label "${run_label}" \
      --risk-grid "${RISK_GRID}" \
      --gamma-grid "${GAMMA_GRID}" \
      --aversion-grid "${AVERSION_GRID}" \
      --policy-modes "${POLICY_MODES}" \
      --delta-cap-grid "${DELTA_CAP_GRID}" \
      --tail-focus-grid "${TAIL_FOCUS_GRID}" \
      --budget-profiles free \
      --shortlist-top-k "${SHORTLIST_TOP_K}" \
      --bucket-return-k 120 \
      --bucket-proxy-k 90 \
      --bucket-family-k 40 \
      --bucket-region-k 60 \
      --alpha-grid "${ALPHA_GRID}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --random-states "${RANDOM_STATES}" \
      --solver-backend highs \
      --exact-solver-backend highs \
      --exact-shadow-backend gurobi \
      --exact-shadow-top-k 20 \
      --exact-solver-agreement-return-abs 250.0 \
      --exact-solver-agreement-v-abs 0.002 \
      --exact-solver-agreement-gamma-abs 0.005 \
      --exact-pass1-alpha 0.01 \
      --exact-pass2-bucket-min 8 \
      --exact-workers 4 \
      --exact-checkpoint-every 32 \
      --exact-python-executable "${MAIN_PYTHON}"
  fi
  local rc=$?
  if [[ "${rc}" -ne 0 && "${solver_backend}" == "cuopt" && "${ENABLE_HIGHS_FALLBACK}" == "1" ]]; then
    log "cuOpt failed for ${variant}; running HiGHS fallback"
    PORTFOLIO_SOLVER_BACKEND=highs run_portfolio_once "${variant}" "highs"
  fi
  summarize_live
}

run_conformal_smoke() {
  local variant
  for variant in ${SMOKE_VARIANTS//,/ }; do
    run_conformal_variant "${variant}" || log "conformal failed ${variant}"
  done
}

run_portfolio_smoke() {
  local variant
  for variant in ${SMOKE_VARIANTS//,/ }; do
    run_portfolio_once "${variant}" "${PORTFOLIO_SOLVER_BACKEND}" || log "portfolio failed ${variant}"
  done
}

main() {
  apply_execution_mode_policy
  log "START ${RUN_ROOT} mode=${MODE}"
  log "execution_mode=${EXECUTION_MODE}"
  log "scratch=${SCRATCH_ROOT}"
  trap 'status=$?; log "EXIT ${RUN_ROOT} status=${status} line=${LINENO}"; summarize_live >/dev/null 2>&1 || true' EXIT

  run_audit || log "audit failed"
  summarize_live

  if [[ "${EXECUTION_MODE}" == "informs_lockdown" && "${MODE}" != "audit" && "${MODE}" != "tournament-preflight" ]]; then
    write_status "execution_mode_gate" "skipped" 0 1 "execution_mode=informs_lockdown allows only audit/tournament-preflight" ""
    log "execution_mode=informs_lockdown blocks search mode=${MODE}"
    summarize_live
    log "END ${RUN_ROOT} mode=${MODE}"
    return 0
  fi

  case "${MODE}" in
    audit)
      log "audit-only mode complete"
      ;;
    tournament-preflight)
      run_tournament_preflight
      log "tournament preflight complete"
      ;;
    canonical-smoke)
      run_conformal_smoke
      run_portfolio_smoke
      ;;
    pd-hpo)
      run_pd_hpo
      ;;
    conformal-smoke)
      run_conformal_smoke
      ;;
    portfolio-smoke)
      run_conformal_smoke
      run_portfolio_smoke
      ;;
    full-governed)
      write_status "full_governed" "skipped" 0 1 "full-governed mode intentionally blocked; run pd-hpo, conformal-smoke, and portfolio-smoke as separate governed phases" ""
      log "full-governed mode is blocked by design; execute phases separately."
      ;;
    *)
      write_status "mode" "failed" 2 0 "unknown mode ${MODE}" ""
      log "unknown mode: ${MODE}"
      return 2
      ;;
  esac

  summarize_live
  log "END ${RUN_ROOT} mode=${MODE}"
}

main "$@"
