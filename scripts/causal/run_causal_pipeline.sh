#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_in_causal_env.sh"

treatment="int_rate"
sample_size=""
run_tag=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --treatment)
      treatment="${2:-}"
      shift 2
      ;;
    --sample_size)
      sample_size="${2:-}"
      shift 2
      ;;
    --run_tag)
      run_tag="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

estimate_args=(scripts/estimate_causal_effects.py --treatment "${treatment}")
if [[ -n "${sample_size}" ]]; then
  estimate_args+=(--sample_size "${sample_size}")
fi
if [[ -n "${run_tag}" ]]; then
  estimate_args+=(--run_tag "${run_tag}")
fi

cd "${REPO_ROOT}"
bash "${RUNNER}" "${estimate_args[@]}"
bash "${RUNNER}" scripts/simulate_causal_policy.py
bash "${RUNNER}" scripts/validate_causal_policy.py
bash "${RUNNER}" scripts/backtest_causal_policy_oot.py
