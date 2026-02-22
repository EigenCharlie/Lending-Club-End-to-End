# SESSION STATE - Lending Club Risk Project
Last Updated: 2026-02-21

---

## 1) Executive Status

Project is operational and artifact-consistent across the thesis pipeline.

- Serving strategy remains Streamlit-first (thesis showcase mode).
- PD architecture remains Logistic Regression baseline + CatBoost final (tuned + calibrated).
- Temporal validation and OOT evaluation remain mandatory.
- This file is only for current state. Historical logs are consolidated in `docs/DECISION_CHANGES_AND_LEARNINGS.md` (section "Session History (Consolidated)").

---

## 2) Serving Architecture Decision (Thesis Mode)

Given fixed historical data and showcase objective:

1. Streamlit is the primary delivery layer.
2. DuckDB is used for local analytical queries.
3. dbt provides governance/lineage/tests over analytical assets.
4. Feast is kept as a feature-store consistency layer for train/serve narrative.
5. FastAPI and MCP remain optional support services.

Design implication:
- Priority is narrative quality, reproducibility, and auditability over online serving complexity.

---

## 3) Pipeline Connection Map

```text
1. src/data/make_dataset.py              -> interim cleaned dataset
2. src/data/prepare_dataset.py           -> OOT train/calibration/test splits
3. src/data/build_datasets.py            -> loan_master, time_series, ead_dataset
4. scripts/train_pd_model.py             -> LR baseline + CatBoost default/tuned + calibration selection + contract
5. scripts/generate_conformal_intervals.py -> Mondrian conformal intervals
6. scripts/backtest_conformal_coverage.py -> temporal monitoring
7. scripts/validate_conformal_policy.py   -> formal policy gate
8. scripts/estimate_causal_effects.py     -> CATE estimation
9. scripts/simulate_causal_policy.py      -> policy simulation
10. scripts/validate_causal_policy.py     -> rule selection + bootstrap
11. scripts/backtest_causal_policy_oot.py -> OOT policy backtest
12. scripts/run_ifrs9_sensitivity.py      -> scenario + sensitivity ECL
13. scripts/optimize_portfolio.py         -> LP/MILP allocation
14. scripts/optimize_portfolio_tradeoff.py -> robustness frontier
15. scripts/run_survival_analysis.py      -> Cox PH + RSF lifetime PD
16. scripts/benchmark_conformal_variants.py -> variant comparison
17. scripts/run_fairness_audit.py         -> demographic parity, EO gap, DIR
18. scripts/optimize_cate_portfolio.py    -> CATE-adjusted portfolio comparison
19. scripts/simulate_ab_test.py           -> robust vs non-robust A/B simulation
20. scripts/generate_mrm_report.py        -> SR 11-7 consolidated report
21. scripts/build_pipeline_results.py     -> pipeline KPI aggregation
22. scripts/export_streamlit_artifacts.py -> Streamlit-ready data export
23. scripts/export_storytelling_snapshot.py -> storytelling JSON
24. scripts/end_to_end_pipeline.py        -> orchestration
```

---

## 4) Current Runtime Snapshot

Source artifacts:
- `data/processed/model_comparison.json`
- `models/conformal_policy_status.json`
- `models/causal_policy_rule.json`
- `data/processed/ifrs9_scenario_summary.parquet`
- `data/processed/portfolio_robustness_summary.parquet`
- `data/processed/pipeline_summary.json`
- `models/fairness_audit_status.json`
- `models/ab_simulation_status.json`
- `models/cate_portfolio_status.json`
- `reports/mrm/mrm_validation_report.json`

### 4.1 PD Model (OOT, calibrated final)
- Best model: `CatBoost (tuned + calibrated)`
- Calibration selected: `Isotonic Regression`
- AUC: `0.7172`
- Gini: `0.4344`
- KS: `0.3200`
- Brier: `0.1538`
- ECE: `0.0094`
- HPO trials executed: `400`
- Best validation AUC (Optuna): `0.7199`

### 4.2 Conformal (Mondrian)
- Coverage 90%: `0.8917`
- Coverage 95%: `0.9511`
- Avg width 90%: `0.7225`
- Policy checks passed: `2/7`
- Overall policy pass: `false`

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Selected action rate: `27.51%`
- Selected total net value: `43.67M`
- Selected bootstrap p05 net value: `43.52M`

### 4.4 IFRS9 Sensitivity
- Baseline total ECL: `0.968B`
- Severe total ECL: `1.779B`
- Uplift severe vs baseline: `+83.81%`

### 4.5 Optimization Robustness (risk tolerance 0.10)
- Baseline non-robust funded: `148`
- Best robust funded: `100`
- Baseline non-robust return: `98,235`
- Best robust return: `62,030`
- Price of robustness: `36.86%`

---

## 5) Delivery Layer Status (Current)

### Streamlit
- 25-page multi-page app in `streamlit_app/`, all registered in `app.py`.
- Model laboratory and thesis pages consume runtime artifacts for metrics.
- Includes A/B testing simulation, fairness audit, and CATE portfolio pages.

### FastAPI
- Endpoints implemented in `api/`:
  - `/health`, `/ready`
  - `/api/v1/predict`, `/api/v1/conformal`, `/api/v1/ecl`
  - `/api/v1/query`, `/api/v1/tables`, `/api/v1/summary/*`

### Docker
- `docker-compose.yml` includes `api` and `streamlit`.
- Streamlit can run standalone in thesis mode.

### dbt + Feast
- dbt project configured in `dbt_project/`.
- Feast repo configured in `feature_repo/`.

---

## 6) Environment Notes

- Python: `3.11` (`.python-version`)
- Environment manager: `uv`
- Local virtual environment: `.venv`
- Optional platform extras (`dbt`, `feast`) are under `pyproject.toml` extra `platform`

---

## 7) Test Suite

394 tests passing (7.88s) across 19 test files:

| Category | Tests | Files |
|----------|-------|-------|
| API normalization | 5 | `test_router_normalization` |
| Config consistency | 54 | `test_config_consistency` (7 config classes) |
| Data pipeline | 29 | `test_make_dataset`, `test_prepare_dataset`, `test_build_datasets` |
| Features | 5 | `test_features` |
| PD model | 9 | `test_pd_model` |
| Calibration | 10 | `test_calibration` |
| Conformal | 18 | `test_conformal` |
| Conformal tuning | 21 | `test_conformal_tuning` |
| PD contract | 13 | `test_pd_contract` |
| IFRS9 | 19 | `test_ifrs9` |
| Metrics | 11 | `test_metrics` |
| Fairness | 10 | `test_fairness` |
| A/B testing | 8 | `test_ab_testing` |
| Portfolio | 15 | `test_portfolio` |
| Portfolio model | 13 | `test_portfolio_model` |
| Causal portfolio | 8 | `test_causal_portfolio` |
| Robust opt | 13 | `test_robust_opt` |
| MLflow/utils | 18 | `test_mlflow_suite`, `test_mlflow_utils` |
| Streamlit | 55 | `test_page_imports` (25 pages + utils) |
| Integration | 8 | `test_integration` |

## 8) Current Priorities

1. Keep docs and Streamlit narratives strictly artifact-driven (no stale hardcoded claims).
2. Config files are templates — runtime calibration selection is artifact-driven.
3. Preserve reproducibility gates (`ruff`, `pytest`, `dvc`) in routine runs.
4. DVC pipeline has 24 stages; `dvc.lock` is authoritative for artifact hashes.

---

## 9) Source of Truth

| Reference | Purpose |
|-----------|---------|
| `SESSION_STATE.md` | Current official state |
| `docs/PROJECT_JUSTIFICATION.md` | Current official design rationale |
| `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Historical decisions, errors, learnings, and session history |
| `data/processed/model_comparison.json` | PD model comparison and final metrics |
| `models/conformal_policy_status.json` | Conformal policy gate snapshot |
| `models/causal_policy_rule.json` | Causal policy rule and selected metrics |
| `data/processed/pipeline_summary.json` | Cross-module pipeline KPI snapshot |
