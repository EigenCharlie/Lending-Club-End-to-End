#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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
  if ! blocked_dirty_paths="$("${PY_BIN}" scripts/git_dirty_guard.py --mode blocked-only)"; then
    echo "Refusing to start official run '${RUN_TAG}' with dirty working tree."
    echo "Blocked dirty paths:"
    while IFS= read -r dirty_path; do
      [[ -n "${dirty_path}" ]] && echo "  ${dirty_path}"
    done <<< "${blocked_dirty_paths}"
    echo "Commit/stash blocked paths or use a non-official run tag."
    exit 2
  fi
fi

has_resume=0
has_sampling=0
has_env_file=0
has_comparison_baseline=0
BASELINE_REGISTRY="configs/baselines/core_official_baseline.json"
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
    --comparison-baseline|--comparison-baseline=*|--comparison-baseline-run-tag|--comparison-baseline-run-tag=*)
      has_comparison_baseline=1
      ;;
  esac
done

if [[ "${RUN_TAG,,}" == *official* || "${RUN_TAG,,}" == *-core-* || "${RUN_TAG,,}" == *-core ]]; then
  if [[ "${has_comparison_baseline}" -eq 0 ]] && [[ -f "${BASELINE_REGISTRY}" ]]; then
    auto_baseline_tag="$("${PY_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path("configs/baselines/core_official_baseline.json")
if not p.exists():
    raise SystemExit("")
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit("")
tag = str(payload.get("official_run_tag", "")).strip()
print(tag)
PY
)"
    auto_baseline_tag="${auto_baseline_tag//$'\n'/}"
    if [[ -n "${auto_baseline_tag}" ]]; then
      EXTRA_ARGS+=(--comparison-baseline-run-tag "${auto_baseline_tag}")
      has_comparison_baseline=1
      echo "Auto baseline resolved from ${BASELINE_REGISTRY}: ${auto_baseline_tag}"
    fi
  fi
fi

if [[ "${RUN_TAG,,}" == *official* || "${RUN_TAG,,}" == *-core-* || "${RUN_TAG,,}" == *-core ]]; then
  if [[ "${has_comparison_baseline}" -eq 0 ]]; then
    echo "Run tag '${RUN_TAG}' requires explicit comparison baseline."
    echo "Add --comparison-baseline <path> or --comparison-baseline-run-tag <tag>,"
    echo "or define configs/baselines/core_official_baseline.json."
    exit 3
  fi
fi

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

if command -v setsid >/dev/null 2>&1; then
  setsid "${PY_BIN}" -u scripts/run_long_pipeline.py --run-tag "${RUN_TAG}" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    >"${LAUNCH_LOG}" 2>&1 < /dev/null &
  pid=$!
else
  nohup "${PY_BIN}" -u scripts/run_long_pipeline.py --run-tag "${RUN_TAG}" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    >"${LAUNCH_LOG}" 2>&1 < /dev/null &
  pid=$!
fi
echo "${pid}" > "${PID_FILE}"

echo "Started run_tag=${RUN_TAG} pid=${pid}"
echo "Run dir: ${RUN_DIR}"
if [[ "${#DEFAULT_ARGS[@]}" -gt 0 ]]; then
  echo "Defaults applied: ${DEFAULT_ARGS[*]}"
fi
echo "Monitor:"
echo "  bash scripts/monitor_long_run.sh ${RUN_TAG}"
echo "  uv run python scripts/monitor_pipeline_health.py --run-tag ${RUN_TAG} --interval-seconds 900"
