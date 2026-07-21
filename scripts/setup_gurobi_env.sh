#!/usr/bin/env bash
# Configure this shell for the local Gurobi Optimizer install.
#
# Usage:
#   source scripts/setup_gurobi_env.sh
#   GUROBI_LICENSE_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx scripts/setup_gurobi_env.sh --activate

set -euo pipefail

GUROBI_HOME="${GUROBI_HOME:-$HOME/.local/opt/gurobi1302/linux64}"
export GUROBI_HOME
export PATH="$GUROBI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"

if [[ "${1:-}" == "--activate" ]]; then
  if [[ -z "${GUROBI_LICENSE_KEY:-}" ]]; then
    echo "Set GUROBI_LICENSE_KEY before running --activate." >&2
    exit 2
  fi
  printf '%s\n' "$HOME" | "$GUROBI_HOME/bin/grbgetkey" "$GUROBI_LICENSE_KEY"
fi

echo "GUROBI_HOME=$GUROBI_HOME"
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
command -v grbgetkey >/dev/null && grbgetkey --version
command -v gurobi_cl >/dev/null && gurobi_cl --version
