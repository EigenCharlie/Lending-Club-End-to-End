#!/usr/bin/env bash
set -euo pipefail

cd /home/eigenlinux/projects/lending-club-risk-project

RUN_TAG="${1:-$(date +%F-long-run)}"
shift || true

RUN_DIR="reports/run_logs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/status"

PID_FILE="${RUN_DIR}/orchestrator.pid"
LAUNCH_LOG="${RUN_DIR}/launcher.log"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Orchestrator already running for ${RUN_TAG} (pid=${old_pid})"
    exit 1
  fi
fi

nohup lending-club-venv/bin/python -u scripts/run_long_pipeline.py --run-tag "${RUN_TAG}" "$@" \
  >"${LAUNCH_LOG}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Started run_tag=${RUN_TAG} pid=${pid}"
echo "Run dir: ${RUN_DIR}"
echo "Monitor:"
echo "  bash scripts/monitor_long_run.sh ${RUN_TAG}"
