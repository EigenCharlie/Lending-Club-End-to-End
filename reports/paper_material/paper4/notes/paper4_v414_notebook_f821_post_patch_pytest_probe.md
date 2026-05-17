# Paper 4 Post-F821 Validation-Target Pytest Probe v414

Generated: 2026-05-17T09:39:14.232806+00:00

v414 runs full repository pytest after v413 clears notebook F821.

## Result

- Command: `uv run pytest -q --tb=short`.
- Exit code: `0`.
- Pytest passed: `True`.
- Collected items: `1152`.
- Runtime seconds: `167.978`.
- Summary: `=========== 1152 passed, 2 skipped, 13 warnings in 155.93s (0:02:35) ===========`.
- Notebook diagnostics: `6`.
- Notebook F821 diagnostics: `0`.

## Stdout Tail

```text
tests/test_scripts/test_legacy_guardrails.py .                           [ 81%]
tests/test_scripts/test_mlflow_suite.py ...................              [ 83%]
tests/test_scripts/test_monitor_pipeline_eta.py ..                       [ 83%]
tests/test_scripts/test_optimize_cate_portfolio.py .                     [ 83%]
tests/test_scripts/test_optimize_portfolio_tradeoff.py .                 [ 83%]
tests/test_scripts/test_pipeline_entrypoints.py ......                   [ 84%]
tests/test_scripts/test_prepare_streamlit_deploy.py .....                [ 84%]
tests/test_scripts/test_run_all_notebooks.py ...                         [ 85%]
tests/test_scripts/test_run_bma_comparison.py .....                      [ 85%]
tests/test_scripts/test_run_calibration_mapping_shadow_validation.py ..  [ 85%]
tests/test_scripts/test_run_cif_ecl_impact.py .....                      [ 86%]
tests/test_scripts/test_run_comparison.py ............                   [ 87%]
tests/test_scripts/test_run_conformal_reopen_search.py ..                [ 87%]
tests/test_scripts/test_run_fairness_audit.py ..                         [ 87%]
tests/test_scripts/test_run_gpu_replay.py ....                           [ 87%]
tests/test_scripts/test_run_ifrs9_monte_carlo_gpu.py ......              [ 88%]
tests/test_scripts/test_run_long_pipeline.py .................           [ 89%]
tests/test_scripts/test_run_notebooks_inventory.py ...                   [ 90%]
tests/test_scripts/test_run_paper_notebook_suite.py .                    [ 90%]
tests/test_scripts/test_run_pd_rapids_benchmark.py ..                    [ 90%]
tests/test_scripts/test_run_portfolio_bound_aware_search.py ..           [ 90%]
tests/test_scripts/test_run_rapids_insight_factory.py ...                [ 90%]
tests/test_scripts/test_run_sicr_conformal.py .........                  [ 91%]
tests/test_scripts/test_run_stage_misclassification_cost.py ......       [ 92%]
tests/test_scripts/test_run_time_series_vnext.py .                       [ 92%]
tests/test_scripts/test_run_ts_ecl_intervals.py .....                    [ 92%]
tests/test_scripts/test_search_monotonic_competitor.py ..                [ 92%]
tests/test_scripts/test_search_monotonic_economic_promotion.py ..        [ 93%]
tests/test_scripts/test_select_economic_portfolio_policy.py .....        [ 93%]
tests/test_scripts/test_simulate_ab_test.py ........                     [ 94%]
tests/test_scripts/test_simulate_causal_policy.py .                      [ 94%]
tests/test_scripts/test_train_pd_model.py ..........                     [ 95%]
tests/test_scripts/test_update_champion_registry.py .                    [ 95%]
tests/test_scripts/test_validate_causal_policy.py .                      [ 95%]
tests/test_scripts/test_validate_conformal_experiment.py .               [ 95%]
tests/test_scripts/test_validate_conformal_policy.py .......             [ 96%]
tests/test_streamlit/test_app_shell_navigation.py .                      [ 96%]
tests/test_streamlit/test_companion_surface.py ...                       [ 96%]
tests/test_streamlit/test_dvc_kpi_components.py ....                     [ 96%]
tests/test_streamlit/test_dvc_metrics_fallback.py ..                     [ 96%]
tests/test_streamlit/test_page_imports.py .............                  [ 98%]
tests/test_streamlit/test_release_governance_utils.py ..                 [ 98%]
tests/test_streamlit/test_story_contracts.py .....                       [ 98%]
tests/test_streamlit/test_toboml_concept_coverage.py .......             [ 99%]
tests/test_streamlit/test_toboml_double_depth_contract.py ..             [ 99%]
tests/test_utils/test_mlflow_utils.py .......                            [100%]

=============================== warnings summary ===============================
tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_local_refine_best_params_are_materialized_for_catboost
  /home/eigenlinux/projects/lending-club-risk-project/.venv/lib/python3.12/site-packages/optuna/_experimental.py:33: ExperimentalWarning: Argument ``multivariate`` is an experimental feature. The interface can change in the future.
    optuna_warn(

tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_local_refine_best_params_are_materialized_for_catboost
  /home/eigenlinux/projects/lending-club-risk-project/.venv/lib/python3.12/site-packages/optuna/_experimental.py:33: ExperimentalWarning: Argument ``group`` is an experimental feature. The interface can change in the future.
    optuna_warn(

tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_catboost_tuned_and_default_predictions_differ
tests/test_models/test_pd_model.py::test_local_refine_best_params_are_materialized_for_catboost
tests/test_models/test_pd_model.py::test_local_refine_best_params_are_materialized_for_catboost
  /home/eigenlinux/projects/lending-club-risk-project/src/models/optuna_tuning.py:417: ExperimentalWarning: CatBoostPruningCallback is experimental (supported from v3.0.0). The interface can change in the future.
    pruning_callback = CatBoostPruningCallback(trial, "AUC")

tests/test_models/test_time_series.py::test_compute_forecastability_report_assigns_intermittent_routes
  /home/eigenlinux/projects/lending-club-risk-project/src/models/time_series.py:756: InterpolationWarning: The test statistic is outside of the range of p-values available in the
  look-up table. The actual p-value is greater than the p-value returned.
  
    kpss_res = kpss(y, nlags="auto")

tests/test_models/test_time_series.py::test_compute_forecastability_report_assigns_intermittent_routes
  /home/eigenlinux/projects/lending-club-risk-project/.venv/lib/python3.12/site-packages/statsmodels/regression/linear_model.py:1490: ValueWarning: Matrix is singular. Using pinv.
    warnings.warn("Matrix is singular. Using pinv.", ValueWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 1152 passed, 2 skipped, 13 warnings in 155.93s (0:02:35) ===========
```

## Stderr Tail

```text

```

## Required Caveat

v414 does not clear remaining style notebook lint and does not create Paper 4
final promotion.

## Next Executable Wave

Build `paper4_v415_notebook_remaining_style_lint_triage.md`.
