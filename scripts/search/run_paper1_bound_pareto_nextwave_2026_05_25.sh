#!/usr/bin/env bash
set -uo pipefail

# Paper Estrella bound Pareto next wave.
#
# Purpose:
#   Test whether canonical_4 can become a true champion replacement by pairing
#   it with tighter, decision-aware conformal variants before portfolio search.
#
# Scope control:
#   - no v### artifacts;
#   - no manuscript writes;
#   - three predefined conformal designs;
#   - one focused portfolio probe per design;
#   - each phase logs and continues after local failures.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}" || exit 1

RUN_ROOT="${RUN_ROOT:-paper1_bound_pareto_nextwave_2026_05_25_r1}"
LOG_DIR="${LOG_DIR:-${ROOT}/reports/run_logs}"
RAPIDS_ENV="${RAPIDS_ENV:-rapids}"
MAIN_PYTHON="${MAIN_PYTHON:-${ROOT}/.venv/bin/python}"
UPSTREAM_CANONICAL_RUN_TAG="${UPSTREAM_CANONICAL_RUN_TAG:-regret_auditability_pd_canonical_4_2026_05_23}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50000}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-240}"
CUOPT_BATCH_SIZE="${CUOPT_BATCH_SIZE:-64}"
CUOPT_METHOD="${CUOPT_METHOD:-Concurrent}"
CUOPT_NUM_CPU_THREADS="${CUOPT_NUM_CPU_THREADS:-24}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${LOG_DIR}/${RUN_ROOT}.pid"

log() {
  echo "[$(date -Is)] $*" >&2
}

trap 'status=$?; log "EXIT ${RUN_ROOT} status=${status} line=${LINENO}"' EXIT

run_conformal_variant() {
  local variant="$1"
  local namespace="$2"
  local partition="$3"
  local prob_source="$4"
  local bins="$5"
  local fallback="$6"
  local alpha90="$7"
  local scale="$8"
  local scaled="$9"
  local mgs="${10}"

  local intervals="data/processed/conformal_gap/${namespace}/conformal_intervals_mondrian.parquet"
  if [[ -f "${intervals}" ]]; then
    log "reuse conformal ${variant}: ${intervals}"
    return 0
  fi

  log "generate conformal ${variant}: namespace=${namespace}"
  {
    echo "[$(date -Is)] conformal variant=${variant} namespace=${namespace}"
    PIPELINE_RUN_TAG="${namespace}" UPSTREAM_CANONICAL_RUN_TAG="${UPSTREAM_CANONICAL_RUN_TAG}" "${MAIN_PYTHON}" scripts/generate_conformal_intervals.py \
      --artifact_namespace "${namespace}" \
      --evaluation_scope test \
      --partition "${partition}" \
      --partition_candidates "${partition}" \
      --partition_probability_sources "${prob_source}" \
      --n_score_bins_candidates "${bins}" \
      --fallback_modes "${fallback}" \
      --alpha_candidates_90 "${alpha90}" \
      --alpha_candidates_95 0.05 \
      --min_group_sizes "${mgs}" \
      --score_scale_families "${scale}" \
      --scaled_scores_options "${scaled}" \
      --calibration_fraction 1.0
  } >> "${LOG_DIR}/${RUN_ROOT}.log" 2>&1
}

run_portfolio_probe() {
  local variant="$1"
  local namespace="$2"
  local intervals="data/processed/conformal_gap/${namespace}/conformal_intervals_mondrian.parquet"
  local run_label="${RUN_ROOT}__${variant}__portfolio_probe_50k"
  local selection="models/portfolio_bound_aware/${run_label}/portfolio_bound_aware_selection.json"

  if [[ ! -f "${intervals}" ]]; then
    log "skip portfolio ${variant}: missing intervals ${intervals}"
    return 1
  fi
  if [[ -f "${selection}" ]]; then
    log "reuse portfolio ${variant}: ${selection}"
    return 0
  fi

  log "run portfolio probe ${variant}: run_label=${run_label}"
  {
    echo "[$(date -Is)] portfolio variant=${variant} run_label=${run_label}"
    conda run -n "${RAPIDS_ENV}" python scripts/search/run_portfolio_bound_aware_search.py \
      --conformal-intervals-path "${intervals}" \
      --run-label "${run_label}" \
      --risk-grid 0.175,0.180,0.185,0.190,0.195 \
      --gamma-grid 0.275,0.300,0.325,0.350,0.375 \
      --aversion-grid 0,0.05,0.10,0.15,0.20 \
      --policy-modes blended_uncertainty,capped_blended_uncertainty,tail_blended_uncertainty \
      --delta-cap-grid 0.60,0.70,0.75,0.80,0.90 \
      --tail-focus-grid 0.90,0.95 \
      --budget-profiles free \
      --shortlist-top-k "${SHORTLIST_TOP_K}" \
      --bucket-return-k 120 \
      --bucket-proxy-k 90 \
      --bucket-family-k 40 \
      --bucket-region-k 60 \
      --alpha-grid 0.01 \
      --max-candidates "${MAX_CANDIDATES}" \
      --random-states 42 \
      --solver-backend cuopt \
      --exact-solver-backend highs \
      --exact-python-executable "${MAIN_PYTHON}" \
      --cuopt-batch-size "${CUOPT_BATCH_SIZE}" \
      --cuopt-method "${CUOPT_METHOD}" \
      --cuopt-num-cpu-threads "${CUOPT_NUM_CPU_THREADS}"
  } >> "${LOG_DIR}/${RUN_ROOT}.log" 2>&1
}

summarize() {
  log "write nextwave summary"
  "${MAIN_PYTHON}" - "${RUN_ROOT}" <<'PY' >> "${LOG_DIR}/${RUN_ROOT}.log" 2>&1
from pathlib import Path
import json
import sys

import pandas as pd

run_root = sys.argv[1]
champ_return = 170464.5429284627
champ_v = 0.03645
champ_gamma = 0.18591
rows = []
for path in sorted(Path("data/processed/portfolio_bound_aware").glob(f"{run_root}__*__portfolio_probe_50k/portfolio_bound_aware_shortlist.parquet")):
    run_label = path.parent.name
    df = pd.read_parquet(path)
    if "alpha01_exact_pass" not in df.columns:
        continue
    ok = df[df["alpha01_exact_pass"].fillna(False)].copy()
    if ok.empty:
        rows.append({"run_label": run_label, "decision_read": "no_alpha01_pass"})
        continue
    ok["return_delta_vs_champion"] = ok["realized_total_return"] - champ_return
    best_return = ok.sort_values(["realized_total_return", "alpha01_weighted_miscoverage_V", "alpha01_gamma_cp"], ascending=[False, True, True]).iloc[0]
    best_v = ok.sort_values(["alpha01_weighted_miscoverage_V", "realized_total_return"], ascending=[True, False]).iloc[0]
    best_gamma = ok.sort_values(["alpha01_gamma_cp", "realized_total_return"], ascending=[True, False]).iloc[0]
    for label, row in [("best_return", best_return), ("best_V", best_v), ("best_Gamma", best_gamma)]:
        decision = "append_or_park"
        if (
            float(row["realized_total_return"]) >= champ_return
            and float(row["alpha01_weighted_miscoverage_V"]) <= champ_v
            and float(row["alpha01_gamma_cp"]) <= champ_gamma
            and float(row["alpha01_violation"]) <= 1e-12
        ):
            decision = "promote_if_confirmed"
        elif float(row["realized_total_return"]) >= champ_return:
            decision = "return_challenger_bound_worse"
        elif float(row["alpha01_weighted_miscoverage_V"]) <= champ_v or float(row["alpha01_gamma_cp"]) <= champ_gamma:
            decision = "bound_challenger_return_worse"
        rows.append(
            {
                "run_label": run_label,
                "tier": label,
                "decision_read": decision,
                "candidate_rank": int(row["candidate_rank"]),
                "shortlist_bucket": row.get("shortlist_bucket", ""),
                "policy_mode": row["policy_mode"],
                "risk_tolerance": row["risk_tolerance"],
                "gamma": row["gamma"],
                "delta_cap_quantile": row["delta_cap_quantile"],
                "tail_focus_quantile": row["tail_focus_quantile"],
                "uncertainty_aversion": row["uncertainty_aversion"],
                "realized_total_return": row["realized_total_return"],
                "return_delta_vs_champion": row["return_delta_vs_champion"],
                "alpha01_weighted_miscoverage_V": row["alpha01_weighted_miscoverage_V"],
                "alpha01_gamma_cp": row["alpha01_gamma_cp"],
                "alpha01_violation": row["alpha01_violation"],
                "alpha01_empirical_coverage_funded": row["alpha01_empirical_coverage_funded"],
                "n_funded": row.get("n_funded"),
            }
        )

out = pd.DataFrame(rows)
target = Path("reports/paper_material/paper1/tables/paper1_bound_pareto_nextwave_summary_2026-05-25.csv")
target.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(target, index=False)
print(target)
if not out.empty:
    print(out.to_string(index=False))
PY
}

main() {
  log "START ${RUN_ROOT}"

  # Two candidates are valid-width variants from the existing decision-wide
  # conformal report. The grade candidate is an explicit governance stress test:
  # it may fail group diagnostics, but can show whether score-bucket geometry is
  # the main reason Gamma_CP is inflated.
  local variants=(
    "score8_cal_none_a010|${RUN_ROOT}__canonical4_score8_cal_none_a010|score_decile_mondrian|calibrated|8|grade_then_global|0.10|none|false|1000"
    "score8_raw_sqrt_a010|${RUN_ROOT}__canonical4_score8_raw_sqrt_a010|score_decile_mondrian|raw|8|grade_then_global|0.10|bernoulli_sqrt|true|1000"
    "grade_cal_sqrt_a0090|${RUN_ROOT}__canonical4_grade_cal_sqrt_a0090|grade|calibrated|5|grade_then_global|0.09|bernoulli_sqrt|true|1000"
  )

  local spec
  for spec in "${variants[@]}"; do
    IFS='|' read -r variant namespace partition prob_source bins fallback alpha90 scale scaled mgs <<< "${spec}"
    run_conformal_variant "${variant}" "${namespace}" "${partition}" "${prob_source}" "${bins}" "${fallback}" "${alpha90}" "${scale}" "${scaled}" "${mgs}" || log "conformal failed ${variant}"
  done

  for spec in "${variants[@]}"; do
    IFS='|' read -r variant namespace partition prob_source bins fallback alpha90 scale scaled mgs <<< "${spec}"
    run_portfolio_probe "${variant}" "${namespace}" || log "portfolio failed ${variant}"
  done

  summarize || log "summary failed"
  log "END ${RUN_ROOT}"
}

main "$@"
