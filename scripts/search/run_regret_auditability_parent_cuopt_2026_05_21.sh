#!/usr/bin/env bash
set -euo pipefail

# Parent-project cuOpt command for the 2026-05-21 regret-auditability handoff.
# This script is intentionally not called by tests: run it only in an environment
# where cuOpt 26.02 is installed and the external sandbox artifact root is mounted.

uv run python scripts/search/run_portfolio_bound_aware_search.py \
  --conformal-intervals-path /mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned/conformal/regret_auditability_20260513_v3_resource_tuned/data/conformal_intervals_mondrian.parquet \
  --run-label regret_auditability_parent_cuopt_2026_05_21 \
  --risk-grid 0.155,0.160,0.165,0.170,0.175,0.180,0.185 \
  --gamma-grid 0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600 \
  --aversion-grid 0,0.02,0.05,0.10,0.15,0.25,0.50 \
  --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty \
  --delta-cap-grid 0.50,0.60,0.70,0.75,0.80,0.90,0.95,1.0 \
  --tail-focus-grid 0.50,0.60,0.70,0.75,0.80,0.90,0.95,1.0 \
  --budget-profiles free \
  --shortlist-top-k 220 \
  --bucket-return-k 60 \
  --bucket-proxy-k 60 \
  --bucket-family-k 24 \
  --bucket-region-k 40 \
  --alpha-grid 0.01,0.02,0.03,0.05,0.10,0.15,0.20 \
  --max-candidates 0 \
  --random-states 42 \
  --solver-backend cuopt \
  --exact-solver-backend highs \
  --incumbent-policy-path models/champion_portfolio_policy.json \
  --incumbent-risk-neighbors 0.165,0.170,0.175,0.180 \
  --incumbent-gamma-neighbors 0.425,0.450,0.500,0.550 \
  --incumbent-policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty,segment_tail_blended_uncertainty,segment_relative_tail_blended_uncertainty
