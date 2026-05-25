#!/usr/bin/env bash
set -euo pipefail

# Paper Estrella champion-reopen portfolio search.
# Usage:
#   scripts/search/run_paper1_champion_reopen_portfolio_cuopt_2026_05_21.sh \
#     /path/to/conformal_intervals_mondrian.parquet [run_label] [max_candidates]
#
# Frontier search runs in the RAPIDS env with cuOpt. Exact bound rerank is
# delegated to the main project Python because the RAPIDS env intentionally
# does not carry highspy.

CONFORMAL_INTERVALS_PATH="${1:?conformal intervals parquet path is required}"
RUN_LABEL="${2:-paper1_champion_reopen_cuopt_2026_05_21}"
MAX_CANDIDATES="${3:-0}"

RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
MAIN_PYTHON="${MAIN_PYTHON:-$(pwd)/.venv/bin/python}"

conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
  --conformal-intervals-path "${CONFORMAL_INTERVALS_PATH}" \
  --run-label "${RUN_LABEL}" \
  --risk-grid 0.150,0.155,0.160,0.165,0.170,0.175,0.180,0.185,0.190 \
  --gamma-grid 0.350,0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,0.650 \
  --aversion-grid 0,0.02,0.05,0.10,0.15,0.25,0.35,0.50 \
  --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty \
  --delta-cap-grid 0.50,0.60,0.70,0.75,0.80,0.90,0.95,1.0 \
  --tail-focus-grid 0.50,0.60,0.70,0.75,0.80,0.90,0.95,1.0 \
  --budget-profiles free \
  --shortlist-top-k 420 \
  --bucket-return-k 120 \
  --bucket-proxy-k 120 \
  --bucket-family-k 40 \
  --bucket-region-k 80 \
  --alpha-grid 0.01,0.02,0.03,0.05,0.10,0.15,0.20 \
  --max-candidates "${MAX_CANDIDATES}" \
  --random-states 42 \
  --solver-backend cuopt \
  --exact-solver-backend highs \
  --exact-python-executable "${MAIN_PYTHON}" \
  --incumbent-policy-path models/champion_portfolio_policy.json \
  --incumbent-risk-neighbors 0.160,0.165,0.170,0.175,0.180,0.185 \
  --incumbent-gamma-neighbors 0.400,0.425,0.450,0.475,0.500,0.525,0.550 \
  --incumbent-policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty
