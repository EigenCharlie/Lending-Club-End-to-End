#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUARTO_PYTHON="${ROOT}/.venv/bin/python"

cd "${ROOT}"
exec quarto "$@"
