#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install/initialize Conda and ensure env 'rapids' exists." >&2
  exit 1
fi

# Example:
#   bash scripts/side_projects/run_rapids_benchmarks.sh --profile full_data
# Profiles: current | full_data | stress_gpu
conda run -n rapids python reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py "$@"
