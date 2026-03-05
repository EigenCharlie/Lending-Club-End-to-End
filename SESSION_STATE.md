# SESSION STATE - Lending Club Risk Project
Last Updated: 2026-03-05

---

## 1) Executive Status

Project is operational and artifact-consistent across the thesis pipeline.
Main branch finalized with full C smart promotion package.

Closure updates (2026-03-05):
- Official core baseline frozen at `2026-03-04-C-core-balanced-cert2`.
- Baseline registry enabled in `configs/baselines/core_official_baseline.json`.
- Core launcher resolves baseline automatically from registry when not explicitly passed.
- Canonical status artifacts are single-write (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`).

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
7. scripts/validate_conformal_policy.py   -> formal policy gate (Kupiec, Christoffersen, Winkler)
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
20. scripts/generate_governance_status.py -> per-feature drift diagnostics
21. scripts/generate_mrm_report.py        -> SR 11-7 consolidated report
22. scripts/build_pipeline_results.py     -> pipeline KPI aggregation
23. scripts/export_streamlit_artifacts.py -> Streamlit-ready data export
24. scripts/export_storytelling_snapshot.py -> storytelling JSON
25. scripts/end_to_end_pipeline.py        -> orchestration
26. scripts/export_dvc_metrics.py         -> DVC metrics + plot exports
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
- `models/governance_status.json`
- `reports/mrm/mrm_validation_report.json`

### 4.1 PD Model (OOT, calibrated final)
- Best model: `CatBoost (tuned + calibrated)`
- Calibration selected: `Isotonic Regression`
- AUC: `0.7117`
- Gini: `0.4234`
- KS: `0.3129`
- Brier: `0.1548`
- ECE: `0.0072`
- HPO: reused best trial 855 (val AUC 0.7201) from prior Optuna study

### 4.2 Conformal (Mondrian)
- Coverage 90%: `0.9167`
- Coverage 95%: `0.9559`
- Avg width 90%: `0.7442`
- Min group coverage 90%: `0.8840`
- Policy checks passed: `8/13`
- Overall policy pass: `false` (Kupiec/Christoffersen fail on 276K OOT sample — expected; promotion gate treats these as non-blocking diagnostics)
- Conformal promotion pass: `true`

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Selected action rate: `26.31%`
- Selected total net value: `5.86M`
- Selected bootstrap p05 net value: `5.82M`

### 4.4 IFRS9 Sensitivity
- Baseline total ECL: `0.977B`
- Conservative total ECL: `1.791B`
- ECL range: `0.814B`

### 4.5 Optimization Robustness
- Non-robust return: `$111,438` (155 loans funded)
- Robust return: `$67,871` (90 loans funded)
- Price of robustness (absolute): `$43,567`

### 4.6 Fairness Audit
- Overall pass: `false` (2/3 attributes pass)
- home_ownership: PASS (DPD=0.073, EO_gap=0.079, DIR=0.924)
- annual_inc_quartile: FAIL (DPD=0.116, EO_gap=0.133)
- verification_status: PASS (DPD=0.075, EO_gap=0.081, DIR=0.922)

### 4.7 A/B Simulation
- Strategy A (non-robust): $17,067 return, 153 loans
- Strategy B (robust): $17,918 return, 81 loans
- No-regression gate: PASS

---

## 5) Delivery Layer Status (Current)

### Streamlit
- 27-page multi-page app in `streamlit_app/`, all registered in `app.py`.
- Professional light theme with audience toggle (General/Negocio/Técnico).
- Model laboratory and thesis pages consume runtime artifacts for metrics.
- Includes A/B testing simulation, fairness audit, CATE portfolio, and 3 paper draft pages.

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

- Python: `3.12.12` (miniforge3)
- Environment manager: `uv`
- Local virtual environment: `.venv/` (uv-managed, backed by miniforge3 Python)
- Optional platform tooling: `dbt` under `pyproject.toml` extra `platform`; `econml` in `.venv-causal`

---

## 7) Test Suite

Local verification on 2026-03-01:

- `463/463` tests passing (`pytest -q`, Python 3.12.12)
- `49` test files (`tests/**/test_*.py`)
- Streamlit smoke/import coverage includes all `27` pages (`tests/test_streamlit/test_page_imports.py`)

Operational note:
- `data/processed/runtime_status.json` is a generated snapshot and may lag until `scripts/export_streamlit_artifacts.py` is re-run.

## 8) Current Priorities

1. Keep docs and Streamlit narratives strictly artifact-driven (no stale hardcoded claims).
2. Config files are templates — runtime calibration selection is artifact-driven.
3. Preserve reproducibility gates (`ruff`, `pytest`, `dvc`) in routine runs.
4. DVC pipeline has 26 stages; `dvc.lock` is authoritative for artifact hashes.

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
