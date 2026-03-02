#!/usr/bin/env bash
set -euo pipefail

# Create/update a dedicated causal environment that keeps EconML out of the main lock.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${1:-.venv-causal}"

echo "[1/2] Syncing project stack into ${ENV_DIR} (Python 3.12)..."
UV_PROJECT_ENVIRONMENT="${ENV_DIR}" uv sync --python 3.12 --extra dev --extra platform

echo "[2/2] Installing EconML overlay into ${ENV_DIR}..."
uv pip install --python "${ENV_DIR}/bin/python" -r requirements/causal-econml.txt

echo
echo "Causal env ready: ${ENV_DIR}"
echo "Note: this is a task-specific overlay env for causal workflows (EconML may downgrade sklearn/shap)."
echo "      Do not use it as a general-purpose replacement for the main project env."
echo "Run a causal script with:"
echo "  ${ENV_DIR}/bin/python scripts/estimate_causal_effects.py"
