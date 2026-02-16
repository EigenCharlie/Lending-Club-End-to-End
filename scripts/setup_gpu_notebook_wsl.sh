#!/usr/bin/env bash
set -euo pipefail

# Run inside Ubuntu WSL as root:
#   bash /mnt/c/.../scripts/setup_gpu_notebook_wsl.sh

PROJECT_DIR="/mnt/c/Users/carlos/Documents/Claude Code/lending-club-risk-project"
VENV_DIR="${PROJECT_DIR}/.venv-gpu"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Project directory not found: ${PROJECT_DIR}"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  pkg-config

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"

cd "${PROJECT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv "${VENV_DIR}" --python 3.11
fi

source "${VENV_DIR}/bin/activate"

uv pip install -U pip
uv pip install \
  jupyter \
  pandas \
  polars \
  pyarrow \
  numpy \
  scipy \
  scikit-learn \
  networkx

# RAPIDS/cuOpt stack (CUDA 12 wheels; driver supports backward compatibility)
uv pip install --extra-index-url https://pypi.nvidia.com \
  cudf-cu12 \
  cuml-cu12 \
  cugraph-cu12 \
  nx-cugraph-cu12 \
  cudf-polars-cu12 \
  rmm-cu12 \
  cuopt-cu12 \
  cupy-cuda12x

python - <<'PY'
import importlib.util
mods = ["cudf", "cuml", "cugraph", "cuopt", "cupy", "rmm"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules after install: {missing}")
print("RAPIDS/cuOpt modules available:", ", ".join(mods))
PY

python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/10_rapids_gpu_benchmark_lending_club.ipynb \
  --ExecutePreprocessor.timeout=0

echo "Notebook execution finished."
echo "Benchmark outputs:"
echo "  ${PROJECT_DIR}/reports/gpu_benchmark/"
