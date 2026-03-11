#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Example:
#   bash scripts/side_projects/run_rapids_benchmarks.sh --profile full_data
# Profiles: current | full_data | stress_gpu
if [[ "${CONDA_DEFAULT_ENV:-}" == "rapids" ]]; then
  python reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py "$@"
  exit $?
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Activate env 'rapids' or install/initialize conda." >&2
  exit 1
fi

conda run --no-capture-output -n rapids python reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py "$@"
