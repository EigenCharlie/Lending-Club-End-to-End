#!/usr/bin/env bash
set -euo pipefail
cd /home/eigenlinux/projects/lending-club-risk-project
RUN_TAG="2026-02-26-overnight-v2"
RUN_DIR="reports/run_logs/$RUN_TAG"
STATUS_DIR="$RUN_DIR/status"
mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.exit || true

# Kill stale sessions from this run name if they exist
for s in lc_main_pre_v2 lc_heavy_chain_v2 lc_rapids_v2 lc_post_core_v2 lc_notebooks_v2; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

main_cmd=$(cat <<'SH'
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=lending-club-venv
export PYTHONUNBUFFERED=1
echo "[main_pre] start $(date -Iseconds)"
{
  source lending-club-venv/bin/activate
  uv run python -u scripts/train_pd_model.py --config configs/pd_model.yaml --sample_size 0
  uv run python -u scripts/generate_conformal_intervals.py
  uv run python -u scripts/benchmark_conformal_variants.py
  uv run python -u scripts/backtest_conformal_coverage.py
  uv run python -u scripts/validate_conformal_policy.py
  uv run python -u scripts/forecast_default_rates.py --horizon 12
} 2>&1 | tee reports/run_logs/2026-02-26-overnight-v2/main_pre.log
ec=${PIPESTATUS[0]}
echo "$ec" > reports/run_logs/2026-02-26-overnight-v2/status/main_pre.exit
echo "[main_pre] end $(date -Iseconds) ec=$ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/main_pre.log
exit "$ec"
SH
)

heavy_cmd=$(cat <<'SH'
set -euo pipefail
export PYTHONUNBUFFERED=1
while [ ! -f reports/run_logs/2026-02-26-overnight-v2/status/main_pre.exit ]; do sleep 30; done
main_ec=$(cat reports/run_logs/2026-02-26-overnight-v2/status/main_pre.exit || echo 1)
echo "[heavy_chain] dependency main_pre ec=$main_ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/heavy_chain.log
# Continue even if main_pre failed; post_core will capture degraded state.
{
  export UV_PROJECT_ENVIRONMENT=lending-club-venv
  source lending-club-venv/bin/activate
  echo "[heavy_chain] phase=survival_lgd_opt start $(date -Iseconds)"
  uv run python -u scripts/run_survival_analysis.py --full-data --rsf_n_estimators 300
  uv run python -u scripts/train_lgd_ead.py --sample_size 0
  uv run python -u scripts/optimize_portfolio.py --config configs/optimization.yaml --max_candidates 0 --solver_backend highs
  uv run python -u scripts/optimize_portfolio_tradeoff.py --config configs/optimization.yaml --max_candidates 0 --grid-profile night --solver_backend highs
  uv run python -u scripts/simulate_ab_test.py --max_candidates 0
  uv run python -u scripts/log_mlflow_experiment_suite.py
  echo "[heavy_chain] phase=causal start $(date -Iseconds)"
  deactivate || true
  source .venv-causal/bin/activate
  python -u -c "import econml,dowhy; print('econml', econml.__version__, 'dowhy', dowhy.__version__)"
  python -u scripts/estimate_causal_effects.py --treatment int_rate --sample_size 0
  python -u scripts/simulate_causal_policy.py
  python -u scripts/backtest_causal_policy_oot.py
  deactivate || true
  source lending-club-venv/bin/activate
  echo "[heavy_chain] phase=cate_portfolio start $(date -Iseconds)"
  uv run python -u scripts/optimize_cate_portfolio.py --max_candidates 0
} 2>&1 | tee reports/run_logs/2026-02-26-overnight-v2/heavy_chain.log
ec=${PIPESTATUS[0]}
echo "$ec" > reports/run_logs/2026-02-26-overnight-v2/status/heavy_chain.exit
echo "[heavy_chain] end $(date -Iseconds) ec=$ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/heavy_chain.log
exit "$ec"
SH
)

rapids_cmd=$(cat <<'SH'
set -euo pipefail
echo "[rapids] start $(date -Iseconds)"
{
  conda run --no-capture-output -n rapids bash scripts/side_projects/run_rapids_benchmarks.sh --profile full_data
} 2>&1 | tee reports/run_logs/2026-02-26-overnight-v2/rapids.log
ec=${PIPESTATUS[0]}
echo "$ec" > reports/run_logs/2026-02-26-overnight-v2/status/rapids.exit
echo "[rapids] end $(date -Iseconds) ec=$ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/rapids.log
exit "$ec"
SH
)

post_cmd=$(cat <<'SH'
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=lending-club-venv
export PYTHONUNBUFFERED=1
echo "[post_core] waiting for main_pre + heavy_chain"
while [ ! -f reports/run_logs/2026-02-26-overnight-v2/status/main_pre.exit ] || [ ! -f reports/run_logs/2026-02-26-overnight-v2/status/heavy_chain.exit ]; do sleep 30; done
main_ec=$(cat reports/run_logs/2026-02-26-overnight-v2/status/main_pre.exit || echo 1)
heavy_ec=$(cat reports/run_logs/2026-02-26-overnight-v2/status/heavy_chain.exit || echo 1)
echo "[post_core] deps main_pre=$main_ec heavy_chain=$heavy_ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/post_core.log
{
  source lending-club-venv/bin/activate
  uv run python -u scripts/run_ifrs9_sensitivity.py
  uv run python -u scripts/build_pipeline_results.py
  uv run python -u scripts/run_fairness_audit.py
  uv run python -u scripts/validate_causal_policy.py
  uv run python -u scripts/generate_mrm_report.py
  uv run python -u scripts/export_streamlit_artifacts.py
  uv run python -u scripts/export_storytelling_snapshot.py
  uv run python -u scripts/export_dvc_metrics.py
  uv run python -u scripts/run_comparison.py compare --run-tag 2026-02-26-overnight-v2
} 2>&1 | tee reports/run_logs/2026-02-26-overnight-v2/post_core.log
ec=${PIPESTATUS[0]}
echo "$ec" > reports/run_logs/2026-02-26-overnight-v2/status/post_core.exit
echo "[post_core] end $(date -Iseconds) ec=$ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/post_core.log
exit "$ec"
SH
)

notebooks_cmd=$(cat <<'SH'
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=lending-club-venv
export PYTHONUNBUFFERED=1
echo "[notebooks] waiting for post_core + rapids"
while [ ! -f reports/run_logs/2026-02-26-overnight-v2/status/post_core.exit ] || [ ! -f reports/run_logs/2026-02-26-overnight-v2/status/rapids.exit ]; do sleep 30; done
post_ec=$(cat reports/run_logs/2026-02-26-overnight-v2/status/post_core.exit || echo 1)
rapids_ec=$(cat reports/run_logs/2026-02-26-overnight-v2/status/rapids.exit || echo 1)
echo "[notebooks] deps post_core=$post_ec rapids=$rapids_ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/notebooks.log
{
  source lending-club-venv/bin/activate
  uv run python -u scripts/run_all_notebooks.py --execute-all --include-side-projects --timeout 3600 --inplace false --output-dir reports/notebook_exec
  uv run python -u scripts/run_paper_notebook_suite.py
  uv run python -u scripts/extract_notebook_images.py
} 2>&1 | tee reports/run_logs/2026-02-26-overnight-v2/notebooks.log
ec=${PIPESTATUS[0]}
echo "$ec" > reports/run_logs/2026-02-26-overnight-v2/status/notebooks.exit
echo "[notebooks] end $(date -Iseconds) ec=$ec" | tee -a reports/run_logs/2026-02-26-overnight-v2/notebooks.log
exit "$ec"
SH
)

# Start sessions (max 2 heavy jobs concurrently: main_pre + rapids; heavy_chain waits on main_pre)
tmux new-session -d -s lc_main_pre_v2  "bash -lc '$main_cmd'"
tmux new-session -d -s lc_heavy_chain_v2 "bash -lc '$heavy_cmd'"
tmux new-session -d -s lc_rapids_v2    "bash -lc '$rapids_cmd'"
tmux new-session -d -s lc_post_core_v2 "bash -lc '$post_cmd'"
tmux new-session -d -s lc_notebooks_v2 "bash -lc '$notebooks_cmd'"

echo "Launched tmux sessions:"
tmux ls | rg 'lc_(main_pre|heavy_chain|rapids|post_core|notebooks)_v2'
