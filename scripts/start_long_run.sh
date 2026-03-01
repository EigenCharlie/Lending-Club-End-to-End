#!/usr/bin/env bash
set -euo pipefail

cd /home/eigenlinux/projects/lending-club-risk-project

# Load local integration credentials/config (DAGSHUB_*, MLFLOW_*, etc.)
# so non-interactive runs inherit the same env as manual shells.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

RUN_TAG="${1:-$(date +%F-long-run)}"
shift || true
EXTRA_ARGS=("$@")

RUN_DIR="reports/run_logs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/status"

PID_FILE="${RUN_DIR}/orchestrator.pid"
LAUNCH_LOG="${RUN_DIR}/launcher.log"

PY_BIN="lending-club-venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
  else
    PY_BIN="$(command -v python3 || command -v python)"
  fi
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Orchestrator already running for ${RUN_TAG} (pid=${old_pid})"
    exit 1
  fi
fi

if [[ "${RUN_TAG}" == *official* ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to start official run '${RUN_TAG}' with dirty working tree."
    echo "Commit/stash changes or use a non-official run tag."
    exit 2
  fi
fi

has_resume=0
has_sampling=0
has_env_file=0
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --resume)
      has_resume=1
      ;;
    --sampling-profile|--sampling-profile=*)
      has_sampling=1
      ;;
    --env-file|--env-file=*)
      has_env_file=1
      ;;
  esac
done

DEFAULT_ARGS=()
if [[ "${has_resume}" -eq 0 ]]; then
  DEFAULT_ARGS+=(--resume)
fi
if [[ "${has_sampling}" -eq 0 ]]; then
  DEFAULT_ARGS+=(--sampling-profile full)
fi
if [[ "${has_env_file}" -eq 0 ]] && [[ -f ".env" ]]; then
  DEFAULT_ARGS+=(--env-file .env)
fi

nohup "${PY_BIN}" -u scripts/run_long_pipeline.py --run-tag "${RUN_TAG}" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  >"${LAUNCH_LOG}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Started run_tag=${RUN_TAG} pid=${pid}"
echo "Run dir: ${RUN_DIR}"
if [[ "${#DEFAULT_ARGS[@]}" -gt 0 ]]; then
  echo "Defaults applied: ${DEFAULT_ARGS[*]}"
fi
echo "Monitor:"
echo "  bash scripts/monitor_long_run.sh ${RUN_TAG}"
echo "  uv run python scripts/monitor_pipeline_health.py --run-tag ${RUN_TAG} --interval-seconds 900"
