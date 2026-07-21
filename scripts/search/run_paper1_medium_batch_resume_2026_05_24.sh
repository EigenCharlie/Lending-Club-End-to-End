#!/usr/bin/env bash
set -uo pipefail

# Resumable medium triage runner for the Paper Estrella champion reopen.
# cuOpt BatchSolve is fast on this problem family but cuOpt 26.02 can abort
# after repeated BatchSolve calls. This runner treats batch mode as an
# opportunistic accelerator and falls back through smaller batches to
# single-solve cuOpt while preserving per-risk frontier checkpoints.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-paper1_bound_expansion_2026_05_24_r1}"
CONFORMAL_RUN_ROOT="${CONFORMAL_RUN_ROOT:-paper1_champion_replacement_2026_05_23_r3}"
CANDIDATE_KEYS="${CANDIDATE_KEYS:-canonical_4 affordability_rate_5}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
mkdir -p "${LOG_DIR}"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}_medium_batch_resume.pid"

MAX_CANDIDATES="${MAX_CANDIDATES:-75000}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-240}"
RANDOM_STATES="${RANDOM_STATES:-42}"
BATCH_SIZES="${BATCH_SIZES:-128 64 32 16 1}"
MAX_ATTEMPTS_PER_BATCH="${MAX_ATTEMPTS_PER_BATCH:-10}"
MAX_STAGNANT_ATTEMPTS="${MAX_STAGNANT_ATTEMPTS:-2}"
CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
CUOPT_PDLP_SOLVER_MODE="${CUOPT_PDLP_SOLVER_MODE:-}"
CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-24}"

log() {
  echo "[$(date -Is)] $*" >&2
}

candidate_intervals() {
  case "$1" in
    canonical_4)
      echo "data/processed/conformal_gap/${CONFORMAL_RUN_ROOT}__canonical_4__conformal__phase2__final__rank-1/conformal_intervals_mondrian.parquet"
      ;;
    bureau_behavior_15)
      echo "data/processed/conformal_gap/${CONFORMAL_RUN_ROOT}__bureau_behavior_15__conformal__phase2__final__rank-1/conformal_intervals_mondrian.parquet"
      ;;
    affordability_rate_5)
      echo "data/processed/conformal_gap/${CONFORMAL_RUN_ROOT}__affordability_rate_5__conformal__phase2__final__rank-1/conformal_intervals_mondrian.parquet"
      ;;
    *)
      return 2
      ;;
  esac
}

partial_rows() {
  local run_label="$1"
  "${MAIN_PYTHON}" - "${run_label}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

run_label = sys.argv[1]
path = (
    Path("data")
    / "processed"
    / "portfolio_bound_aware"
    / run_label
    / "portfolio_bound_aware_frontier_raw_partial.parquet"
)
if not path.exists():
    print(0)
else:
    print(len(pd.read_parquet(path)))
PY
}

run_one_candidate() {
  local key="$1"
  local intervals
  intervals="$(candidate_intervals "${key}")" || {
    log "unknown candidate ${key}; skipping"
    return 1
  }
  if [[ ! -f "${intervals}" ]]; then
    log "missing intervals for ${key}: ${intervals}"
    return 1
  fi

  local run_label="${RUN_ROOT}__${key}__medium_triage_resume_75k"
  local selection="models/portfolio_bound_aware/${run_label}/portfolio_bound_aware_selection.json"
  local log_path="${LOG_DIR}/${run_label}.log"
  if [[ -f "${selection}" ]]; then
    log "reuse completed ${key}: ${run_label}"
    return 0
  fi

  local batch_size
  for batch_size in ${BATCH_SIZES}; do
    local attempts=0
    local stagnant=0
    local prev_rows=-1
    while [[ ! -f "${selection}" && ${attempts} -lt ${MAX_ATTEMPTS_PER_BATCH} ]]; do
      attempts=$((attempts + 1))
      local rows_before
      rows_before="$(partial_rows "${run_label}")"
      log "medium ${key}: batch=${batch_size} attempt=${attempts} partial_rows=${rows_before}"
      {
        echo "[$(date -Is)] run_label=${run_label} key=${key} batch=${batch_size} attempt=${attempts}"
        conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
          --conformal-intervals-path "${intervals}" \
          --run-label "${run_label}" \
          --risk-grid 0.160,0.165,0.170,0.175,0.180,0.185,0.190 \
          --gamma-grid 0.325,0.375,0.400,0.425,0.450,0.475,0.500,0.550,0.600 \
          --aversion-grid 0,0.02,0.05,0.10,0.15,0.25,0.50 \
          --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty \
          --enable-segment-policy-grid \
          --delta-cap-grid 0.75,0.90,0.95,1.0 \
          --tail-focus-grid 0.75,0.90,0.95,1.0 \
          --budget-profiles free \
          --shortlist-top-k "${SHORTLIST_TOP_K}" \
          --bucket-return-k 100 \
          --bucket-proxy-k 100 \
          --bucket-family-k 60 \
          --bucket-region-k 80 \
          --alpha-grid 0.01,0.03,0.05,0.10 \
          --max-candidates "${MAX_CANDIDATES}" \
          --random-states "${RANDOM_STATES}" \
          --solver-backend cuopt \
          --exact-solver-backend highs \
          --exact-python-executable "${MAIN_PYTHON}" \
          --cuopt-batch-size "${batch_size}" \
          --cuopt-method "${CUOPT_METHOD}" \
          --cuopt-pdlp-solver-mode "${CUOPT_PDLP_SOLVER_MODE}" \
          --cuopt-num-cpu-threads "${CUOPT_NUM_CPU_THREADS}" \
          --cuopt-dual-postsolve 0 \
          --incumbent-policy-path models/champion_portfolio_policy.json \
          --incumbent-risk-neighbors 0.165,0.170,0.175,0.180,0.185,0.190 \
          --incumbent-gamma-neighbors 0.375,0.400,0.425,0.450,0.475,0.500,0.550 \
          --incumbent-policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty
      } >> "${log_path}" 2>&1
      local rc=$?
      if [[ -f "${selection}" ]]; then
        log "medium ${key}: completed with batch=${batch_size}"
        return 0
      fi
      local rows_after
      rows_after="$(partial_rows "${run_label}")"
      log "medium ${key}: rc=${rc} partial_rows_after=${rows_after}"
      if [[ "${rows_after}" -le "${prev_rows}" ]]; then
        stagnant=$((stagnant + 1))
      else
        stagnant=0
      fi
      prev_rows="${rows_after}"
      if [[ "${stagnant}" -ge "${MAX_STAGNANT_ATTEMPTS}" ]]; then
        log "medium ${key}: no progress for ${stagnant} attempts at batch=${batch_size}; fallback"
        break
      fi
      sleep 20
    done
  done
  log "medium ${key}: failed after all batch fallbacks"
  return 1
}

log "medium batch resume start: candidates=${CANDIDATE_KEYS}"
for key in ${CANDIDATE_KEYS}; do
  run_one_candidate "${key}" || true
done
log "medium batch resume complete"
