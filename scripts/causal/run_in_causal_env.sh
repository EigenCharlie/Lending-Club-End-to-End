#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CAUSAL_PYTHON="${REPO_ROOT}/.venv-causal/bin/python"

if [[ ! -x "${CAUSAL_PYTHON}" ]]; then
  echo "Missing ${CAUSAL_PYTHON}. Build the causal env first:" >&2
  echo "  bash scripts/causal/setup_causal_env.sh .venv-causal" >&2
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/causal/run_in_causal_env.sh <script.py> [args...]" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${CAUSAL_PYTHON}" "$@"
