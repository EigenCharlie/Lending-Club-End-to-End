#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${1:?usage: scripts/recover_champion_search_run.sh <run_tag> [baseline_run_tag] [pipeline_profile] [sampling_profile]}"
BASELINE_RUN_TAG="${2:-2026-03-11-C-official-selector-v3-freeze}"
PIPELINE_PROFILE="${3:-champion_search_max}"
SAMPLING_PROFILE="${4:-mega64plus}"

RUN_DIR="reports/run_logs/${RUN_TAG}"
RECOVERY_LOG="${RUN_DIR}/recovery_main_pre.log"
RECOVERY_STATUS="${RUN_DIR}/recovery_status.txt"

mkdir -p "${RUN_DIR}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY_BIN="lending-club-venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
  else
    PY_BIN="$(command -v python3 || command -v python)"
  fi
fi

export PIPELINE_RUN_TAG="${RUN_TAG}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-lending-club-venv}"

ENV_ARGS=()
if [[ -f ".env" ]]; then
  ENV_ARGS+=(--env-file .env)
fi

{
  echo "[recovery] started_at=$(date --iso-8601=seconds)"
  echo "[recovery] run_tag=${RUN_TAG}"
  echo "[recovery] baseline_run_tag=${BASELINE_RUN_TAG}"
  echo "[recovery] pipeline_profile=${PIPELINE_PROFILE}"
  echo "[recovery] sampling_profile=${SAMPLING_PROFILE}"
  echo "[recovery] step=train_pd_reuse"
  uv run python -u scripts/train_pd_model.py \
    --config configs/pd_model.champion_search_max.yaml \
    --sample_size 0 \
    --hpo_n_trials 0

  echo "[recovery] step=generate_conformal_intervals"
  uv run python -u scripts/generate_conformal_intervals.py \
    --alpha_target_90 0.1 \
    --alpha_95 0.05 \
    --alpha_candidates_90 0.1,0.095,0.09,0.085,0.08,0.075,0.07 \
    --min_group_sizes 150,250,500,1000,2000,4000 \
    --min_group_coverage_target 0.88 \
    --group_coverage_floor_target_90 0.9 \
    --max_width_budget_90 0.9 \
    --coverage_guardband_90 0.02 \
    --min_group_guardband_90 0.0 \
    --tuning_holdout_ratio 0.25 \
    --tuning_random_state 42 \
    --temporal_segment_min_size 150 \
    --global_rebalance_min_factor 0.7 \
    --global_rebalance_max_factor 1.1 \
    --global_rebalance_step 0.005 \
    --temporal_segment_floor_enabled 1 \
    --temporal_segment_freq M \
    --global_rebalance_enabled 1

  echo "[recovery] step=benchmark_conformal_variants"
  uv run python -u scripts/benchmark_conformal_variants.py --min_group_size_default 150

  echo "[recovery] step=backtest_conformal_coverage"
  uv run python -u scripts/backtest_conformal_coverage.py

  echo "[recovery] step=validate_conformal_policy"
  uv run python -u scripts/validate_conformal_policy.py --run-tag "${RUN_TAG}"

  echo "[recovery] step=forecast_default_rates"
  uv run python -u scripts/forecast_default_rates.py --horizon 12

  echo "[recovery] step=resume_from_heavy_main"
  "${PY_BIN}" -u scripts/run_champion_search.py \
    --run-tag "${RUN_TAG}" \
    --resume \
    --from-step heavy_main \
    --pipeline-profile "${PIPELINE_PROFILE}" \
    --sampling-profile "${SAMPLING_PROFILE}" \
    --comparison-baseline-run-tag "${BASELINE_RUN_TAG}" \
    "${ENV_ARGS[@]}"
} 2>&1 | tee -a "${RECOVERY_LOG}"

echo "ok $(date --iso-8601=seconds)" > "${RECOVERY_STATUS}"
