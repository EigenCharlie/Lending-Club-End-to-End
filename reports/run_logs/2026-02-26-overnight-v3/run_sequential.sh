#!/usr/bin/env bash
set -uo pipefail
cd /home/eigenlinux/projects/lending-club-risk-project
RUN_TAG="2026-02-26-overnight-v3"
RUN_DIR="reports/run_logs/$RUN_TAG"
STATUS_DIR="$RUN_DIR/status"
MASTER_LOG="$RUN_DIR/master.log"
mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.exit || true
: > "$MASTER_LOG"

log(){
  echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG"
}

run_step(){
  local name="$1"
  local required="$2"
  local cmd="$3"
  log "STEP_START name=$name required=$required"
  bash -lc "$cmd" 2>&1 | tee "$RUN_DIR/${name}.log"
  local ec=${PIPESTATUS[0]}
  echo "$ec" > "$STATUS_DIR/${name}.exit"
  log "STEP_END name=$name ec=$ec"
  if [[ "$required" == "1" && "$ec" -ne 0 ]]; then
    log "STEP_ABORT name=$name (required failed)"
    return "$ec"
  fi
  return 0
}

export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT=lending-club-venv

main_pre_cmd=$(cat <<'SH'
set -euo pipefail
source lending-club-venv/bin/activate
uv run python -u scripts/train_pd_model.py --config configs/pd_model.yaml --sample_size 0
uv run python -u scripts/generate_conformal_intervals.py
uv run python -u scripts/benchmark_conformal_variants.py
uv run python -u scripts/backtest_conformal_coverage.py
uv run python -u scripts/validate_conformal_policy.py
uv run python -u scripts/forecast_default_rates.py --horizon 12
SH
)

heavy_main_cmd=$(cat <<'SH'
set -euo pipefail
source lending-club-venv/bin/activate
uv run python -u scripts/run_survival_analysis.py --full-data --rsf_n_estimators 300
uv run python -u scripts/train_lgd_ead.py --sample_size 0
uv run python -u scripts/optimize_portfolio.py --config configs/optimization.yaml --max_candidates 0 --solver_backend highs
uv run python -u scripts/optimize_portfolio_tradeoff.py --config configs/optimization.yaml --max_candidates 0 --grid-profile night --solver_backend highs
uv run python -u scripts/simulate_ab_test.py --max_candidates 0
uv run python -u scripts/log_mlflow_experiment_suite.py
SH
)

causal_cmd=$(cat <<'SH'
set -euo pipefail
source .venv-causal/bin/activate
python -u -c "import econml,dowhy; print('econml', econml.__version__, 'dowhy', dowhy.__version__)"
python -u scripts/estimate_causal_effects.py --treatment int_rate --sample_size 0
python -u scripts/simulate_causal_policy.py
python -u scripts/backtest_causal_policy_oot.py
SH
)

cate_cmd=$(cat <<'SH'
set -euo pipefail
source lending-club-venv/bin/activate
uv run python -u scripts/optimize_cate_portfolio.py --max_candidates 0
SH
)

post_core_cmd=$(cat <<'SH'
set -euo pipefail
source lending-club-venv/bin/activate
uv run python -u scripts/run_ifrs9_sensitivity.py
uv run python -u scripts/build_pipeline_results.py
uv run python -u scripts/run_fairness_audit.py
uv run python -u scripts/validate_causal_policy.py
uv run python -u scripts/generate_mrm_report.py
uv run python -u scripts/export_streamlit_artifacts.py
uv run python -u scripts/export_storytelling_snapshot.py
uv run python -u scripts/export_dvc_metrics.py
uv run python -u scripts/run_comparison.py compare --run-tag 2026-02-26-overnight-v3
SH
)

rapids_cmd=$(cat <<'SH'
set -euo pipefail
conda run --no-capture-output -n rapids bash scripts/side_projects/run_rapids_benchmarks.sh --profile full_data
SH
)

notebooks_cmd=$(cat <<'SH'
set -euo pipefail
source lending-club-venv/bin/activate
uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout 3600 --inplace false --output-dir reports/notebook_exec
uv run python -u scripts/run_paper_notebook_suite.py
uv run python -u scripts/extract_notebook_images.py
SH
)

overall_ec=0
run_step main_pre 1 "$main_pre_cmd" || overall_ec=$?
if [[ "$overall_ec" -eq 0 ]]; then
  run_step heavy_main 0 "$heavy_main_cmd" || true
  run_step causal 0 "$causal_cmd" || true
  run_step cate_portfolio 0 "$cate_cmd" || true
  run_step post_core 0 "$post_core_cmd" || true
  run_step rapids 0 "$rapids_cmd" || true
  run_step notebooks 0 "$notebooks_cmd" || true
fi

# Derive summary status (required main_pre + post_core success if present)
main_ec=$(cat "$STATUS_DIR/main_pre.exit" 2>/dev/null || echo 99)
post_ec=$(cat "$STATUS_DIR/post_core.exit" 2>/dev/null || echo 99)
if [[ "$main_ec" -eq 0 && "$post_ec" -eq 0 ]]; then
  final_ec=0
else
  final_ec=1
fi
echo "$final_ec" > "$STATUS_DIR/overall.exit"
log "RUN_END final_ec=$final_ec main_pre=$main_ec post_core=$post_ec"
exit "$final_ec"
