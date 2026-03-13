#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_in_causal_env.sh"

treatment="int_rate"
sample_size=""
run_tag=""
cate_n_estimators=""
cate_cv=""
cate_mc_iters=""
cate_criterion=""
cate_min_balancedness_tol=""
cate_honest=""

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
    --cate_n_estimators)
      cate_n_estimators="${2:-}"
      shift 2
      ;;
    --cate_cv)
      cate_cv="${2:-}"
      shift 2
      ;;
    --cate_mc_iters)
      cate_mc_iters="${2:-}"
      shift 2
      ;;
    --cate_criterion)
      cate_criterion="${2:-}"
      shift 2
      ;;
    --cate_min_balancedness_tol)
      cate_min_balancedness_tol="${2:-}"
      shift 2
      ;;
    --cate_honest)
      cate_honest="${2:-}"
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
if [[ -n "${cate_n_estimators}" ]]; then
  estimate_args+=(--cate_n_estimators "${cate_n_estimators}")
fi
if [[ -n "${cate_cv}" ]]; then
  estimate_args+=(--cate_cv "${cate_cv}")
fi
if [[ -n "${cate_mc_iters}" ]]; then
  estimate_args+=(--cate_mc_iters "${cate_mc_iters}")
fi
if [[ -n "${cate_criterion}" ]]; then
  estimate_args+=(--cate_criterion "${cate_criterion}")
fi
if [[ -n "${cate_min_balancedness_tol}" ]]; then
  estimate_args+=(--cate_min_balancedness_tol "${cate_min_balancedness_tol}")
fi
if [[ -n "${cate_honest}" ]]; then
  estimate_args+=(--cate_honest "${cate_honest}")
fi

cd "${REPO_ROOT}"
bash "${RUNNER}" "${estimate_args[@]}"
bash "${RUNNER}" scripts/simulate_causal_policy.py
bash "${RUNNER}" scripts/validate_causal_policy.py
bash "${RUNNER}" scripts/backtest_causal_policy_oot.py
