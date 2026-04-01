#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${1:?usage: scripts/recover_champion_search_run.sh <run_tag> [baseline_run_tag] [pipeline_profile] [sampling_profile]}"
BASELINE_RUN_TAG="${2:-2026-03-11-C-official-selector-v3-freeze}"
PIPELINE_PROFILE="${3:-search_pd_default}"
SAMPLING_PROFILE="${4:-mega64plus}"

RUN_DIR="reports/run_logs/${RUN_TAG}"
RECOVERY_LOG="${RUN_DIR}/recovery_search_pd.log"
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
  echo "[recovery] step=resume_search_pd"
  ENTRYPOINT="scripts/search/run_pd_search.py"
  if [[ ! -f "${ENTRYPOINT}" ]]; then
    ENTRYPOINT="scripts/run_champion_search.py"
  fi
  "${PY_BIN}" -u "${ENTRYPOINT}" \
    --run-tag "${RUN_TAG}" \
    --resume \
    --pipeline-profile "${PIPELINE_PROFILE}" \
    --sampling-profile "${SAMPLING_PROFILE}" \
    --comparison-baseline-run-tag "${BASELINE_RUN_TAG}" \
    "${ENV_ARGS[@]}"
} 2>&1 | tee -a "${RECOVERY_LOG}"

echo "ok $(date --iso-8601=seconds)" > "${RECOVERY_STATUS}"
