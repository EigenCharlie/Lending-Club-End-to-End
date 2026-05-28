#!/usr/bin/env bash
set -uo pipefail

# Focused rescue for the score8_cal_none conformal variant.
#
# The first r4 attempt used cuOpt batch=64 and hit a cuOpt 26.02 native batch
# abort ("pure virtual method called"). This runner keeps the same scientific
# grid but uses a conservative batch fallback ladder and a fresh run label.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

RUN_ROOT="${RUN_ROOT:-paper1_bound_pareto_nextwave_2026_05_25_r4}"
VARIANT="${VARIANT:-score8_cal_none_a010}"
SOURCE_NAMESPACE="${SOURCE_NAMESPACE:-${RUN_ROOT}__canonical4_score8_cal_none_a010}"
RUN_LABEL="${RUN_LABEL:-${RUN_ROOT}__${VARIANT}__portfolio_probe_50k_fallback}"
INTERVALS="${INTERVALS:-data/processed/conformal_gap/${SOURCE_NAMESPACE}/conformal_intervals_mondrian.parquet}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
BATCH_SIZES="${BATCH_SIZES:-16 8 1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50000}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-240}"
CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-24}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${LOG_DIR}/${RUN_LABEL}.pid"

log() {
  echo "[$(date -Is)] $*" >&2
}

partial_rows() {
  "${MAIN_PYTHON}" - "${RUN_LABEL}" <<'PY'
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

run_search() {
  local solver_backend="$1"
  local batch_size="$2"
  conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
    --conformal-intervals-path "${INTERVALS}" \
    --run-label "${RUN_LABEL}" \
    --risk-grid 0.175,0.180,0.185,0.190,0.195 \
    --gamma-grid 0.275,0.300,0.325,0.350,0.375 \
    --aversion-grid 0,0.05,0.10,0.15,0.20 \
    --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty \
    --delta-cap-grid 0.60,0.70,0.75,0.80,0.90 \
    --tail-focus-grid 0.90,0.95 \
    --budget-profiles free \
    --shortlist-top-k "${SHORTLIST_TOP_K}" \
    --bucket-return-k 120 \
    --bucket-proxy-k 90 \
    --bucket-family-k 40 \
    --bucket-region-k 60 \
    --alpha-grid 0.01 \
    --max-candidates "${MAX_CANDIDATES}" \
    --random-states 42 \
    --solver-backend "${solver_backend}" \
    --exact-solver-backend highs \
    --exact-python-executable "${MAIN_PYTHON}" \
    --cuopt-batch-size "${batch_size}" \
    --cuopt-method "${CUOPT_METHOD}" \
    --cuopt-num-cpu-threads "${CUOPT_NUM_CPU_THREADS}" \
    --cuopt-dual-postsolve 0
}

main() {
  local log_path="${LOG_DIR}/${RUN_LABEL}.log"
  local selection="models/portfolio_bound_aware/${RUN_LABEL}/portfolio_bound_aware_selection.json"
  log "START fallback RUN_LABEL=${RUN_LABEL}"
  if [[ ! -f "${INTERVALS}" ]]; then
    log "missing intervals: ${INTERVALS}"
    return 1
  fi
  if [[ -f "${selection}" ]]; then
    log "already complete: ${selection}"
    return 0
  fi

  local batch_size
  for batch_size in ${BATCH_SIZES}; do
    local before
    before="$(partial_rows)"
    log "cuOpt fallback attempt batch=${batch_size} partial_rows_before=${before}"
    {
      echo "[$(date -Is)] RUN_LABEL=${RUN_LABEL} solver=cuopt batch=${batch_size}"
      run_search cuopt "${batch_size}"
    } >> "${log_path}" 2>&1
    local rc=$?
    local after
    after="$(partial_rows)"
    log "cuOpt batch=${batch_size} rc=${rc} partial_rows_after=${after}"
    if [[ -f "${selection}" ]]; then
      log "completed with cuOpt batch=${batch_size}"
      return 0
    fi
    sleep 10
  done

  log "cuOpt fallback exhausted; final CPU HiGHS fallback"
  {
    echo "[$(date -Is)] RUN_LABEL=${RUN_LABEL} solver=highs final_fallback"
    run_search highs 1
  } >> "${log_path}" 2>&1
  if [[ -f "${selection}" ]]; then
    log "completed with HiGHS fallback"
    return 0
  fi
  log "failed after all fallbacks"
  return 1
}

main "$@"
