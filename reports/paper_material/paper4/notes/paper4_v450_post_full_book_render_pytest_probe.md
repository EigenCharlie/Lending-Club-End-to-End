# Paper 4 Post-Full-Book-Render Pytest Probe v450

Generated: 2026-05-17T14:38:20.929356+00:00

v450 reruns full repository pytest after v449's clean full-book Quarto render.

## Result

- Command: `uv run pytest -q --tb=short`.
- Exit code: `0`.
- Pytest passed: `True`.
- Collected items: `1188`.
- Runtime seconds: `181.506`.
- Summary: `=========== 1188 passed, 2 skipped, 13 warnings in 163.16s (0:02:43) ===========`.
- Repository Ruff diagnostics: `0`.
- Full-book render clean from v449: `True`.
- Final promotion created: `False`.

## Stdout Tail

```text
tests/test_scripts/test_legacy_guardrails.py .                           [ 82%]
tests/test_scripts/test_mlflow_suite.py ...................              [ 84%]
tests/test_scripts/test_monitor_pipeline_eta.py ..                       [ 84%]
tests/test_scripts/test_optimize_cate_portfolio.py .                     [ 84%]
tests/test_scripts/test_optimize_portfolio_tradeoff.py .                 [ 84%]
tests/test_scripts/test_pipeline_entrypoints.py ......                   [ 84%]
tests/test_scripts/test_prepare_streamlit_deploy.py .....                [ 85%]
tests/test_scripts/test_run_all_notebooks.py ...                         [ 85%]
tests/test_scripts/test_run_bma_comparison.py .....                      [ 86%]
tests/test_scripts/test_run_calibration_mapping_shadow_validation.py ..  [ 86%]
tests/test_scripts/test_run_cif_ecl_impact.py .....                      [ 86%]
tests/test_scripts/test_run_comparison.py ............                   [ 87%]
tests/test_scripts/test_run_conformal_reopen_search.py ..                [ 87%]
tests/test_scripts/test_run_fairness_audit.py ..                         [ 87%]
tests/test_scripts/test_run_gpu_replay.py ....                           [ 88%]
tests/test_scripts/test_run_ifrs9_monte_carlo_gpu.py ......              [ 88%]
tests/test_scripts/test_run_long_pipeline.py .................           [ 90%]
tests/test_scripts/test_run_notebooks_inventory.py ...                   [ 90%]
tests/test_scripts/test_run_paper_notebook_suite.py .                    [ 90%]
tests/test_scripts/test_run_pd_rapids_benchmark.py ..                    [ 90%]
tests/test_scripts/test_run_portfolio_bound_aware_search.py ..           [ 90%]
tests/test_scripts/test_run_rapids_insight_factory.py ...                [ 91%]
tests/test_scripts/test_run_sicr_conformal.py .........                  [ 91%]
tests/test_scripts/test_run_stage_misclassification_cost.py ......       [ 92%]
tests/test_scripts/test_run_time_series_vnext.py .                       [ 92%]
tests/test_scripts/test_run_ts_ecl_intervals.py .....                    [ 92%]
tests/test_scripts/test_search_monotonic_competitor.py ..                [ 93%]
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
=========== 1188 passed, 2 skipped, 13 warnings in 163.16s (0:02:43) ===========
```

## Stderr Tail

```text

```

## Required Caveat

v450 proves post-render pytest and Ruff cleanliness only. It does not create a
release readiness synthesis, champion replacement, Paper Estrella replacement,
or final Paper 4 promotion.

## Next Executable Wave

Build `paper4_v451_release_readiness_synthesis.md`.
