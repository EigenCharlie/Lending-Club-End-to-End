# SESSION STATE — Lending Club Risk Project
Last Updated: 2026-02-17

---

## 1) Executive Status

Project is operational and artifact-consistent across the full thesis pipeline.

- Notebooks `01` to `09`: executed with zero errors.
- Core scripts: producing artifacts in `data/processed/` and `models/`.
- Policy gates: conformal policy gate activo (snapshot actual: 6/7 checks), causal policy selected, IFRS9 sensitivity produced.
- Tests: `195` passing (6.23s).

This session aligns docs with the actual repository and formalizes a **Streamlit-first Thesis Mode** serving strategy.

---

## 2) Serving Architecture Decision (Thesis Mode)

Given fixed historical dataset and portfolio/showcase objective:

1. Streamlit is the primary delivery layer.
2. DuckDB is the analytical engine for local queries and storytelling pages.
3. dbt is used as governance layer (lineage/docs/tests over parquet-derived sources).
4. Feast is used as feature store showcase for train/serve consistency narrative.
5. FastAPI and MCP remain optional support services.

Design implication:
- No heavy production inference stack is required for new external data streams.
- Priority is narrative depth, reproducibility, and visual explanation quality.

---

## 3) Pipeline Connection Map

```text
1. src/data/make_dataset.py         -> interim cleaned dataset
2. src/data/prepare_dataset.py      -> OOT train/calibration/test splits
3. src/data/build_datasets.py       -> loan_master, time_series, ead_dataset
4. scripts/train_pd_model.py        -> CatBoost PD + calibrator + contract
5. scripts/generate_conformal_intervals.py -> Mondrian conformal intervals
6. scripts/backtest_conformal_coverage.py  -> temporal monitoring
7. scripts/validate_conformal_policy.py    -> formal policy gate
8. scripts/estimate_causal_effects.py      -> CATE estimation
9. scripts/simulate_causal_policy.py       -> policy simulation
10. scripts/validate_causal_policy.py      -> rule selection + bootstrap
11. scripts/run_ifrs9_sensitivity.py       -> scenario + sensitivity ECL
12. scripts/optimize_portfolio.py          -> LP/MILP allocation
13. scripts/optimize_portfolio_tradeoff.py -> robustness frontier
14. scripts/end_to_end_pipeline.py         -> orchestration
```

---

## 4) Key Metrics Snapshot

Source: `reports/project_audit_snapshot.json`

### 4.1 PD Model
- AUC: `0.7187`
- Gini: `0.4373`
- KS: `0.3221`
- Brier: `0.1537`
- ECE: `0.0128`

### 4.2 Conformal (Mondrian)
- Coverage 90%: `0.9197`
- Coverage 95%: `0.9608`
- Avg width 90%: `0.7593`
- Min group coverage 90%: `0.8916`
- Policy checks: `6/7` pass (snapshot 2026-02-17)

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Action rate: `26.30%`
- Net value: `5.857M`
- Bootstrap p05: `5.800M`

### 4.4 IFRS9 Sensitivity
- Baseline ECL: `795.9M`
- Severe ECL: `1.358B`
- Uplift: `+70.65%`

### 4.5 Optimization Robustness
- Tolerance `0.10`: robust funded `54`, robust return `33,580`, robustness cost `51.76%`

---

## 5) Delivery Layer Status (Real Code State)

### Streamlit
- Multi-page app implemented in `streamlit_app/`.
- New thesis storytelling page: `streamlit_app/pages/thesis_end_to_end.py`.
- Chat page supports optional NL->SQL with Grok key via env var (`GROK_API_KEY`).

### FastAPI
- Endpoints implemented in `api/`:
  - `/health`, `/ready`
  - `/api/v1/predict`, `/api/v1/conformal`, `/api/v1/ecl`
  - `/api/v1/query`, `/api/v1/tables`, `/api/v1/summary/*`

### Docker
- `docker-compose.yml` has `api` and `streamlit`.
- Streamlit can run independently for thesis showcase mode.

### dbt + Feast
- dbt project configured in `dbt_project/`.
- Feast repo configured in `feature_repo/` with entity, feature views, and feature services.

---

## 6) Environment Notes

- Python: `3.11` (`.python-version`).
- Environment manager: `uv`.
- Local venv: `.venv`.
- Platform extras (`dbt`, `feast`) are optional dependencies under `pyproject.toml` extra `platform`.

---

## 7) Current Priorities

1. ~~Continue Streamlit storytelling depth~~ → Done (Thesis Contribution page added).
2. ~~Add richer visual lineage/governance story~~ → Done (dbt SQL viewer + Feast details).
3. Keep API as optional support (not primary UI dependency).
4. ~~Add CI checks~~ → Done (GitHub Actions: lint + test + Streamlit smoke).
5. ~~Strengthen reproducibility docs~~ → Done (`docs/RUNBOOK.md`).
6. Re-run pipeline with Platt calibration to regenerate canonical artifacts.
7. Finalize thesis defense narrative and bibliography alignment.

---

## 8) Audit Findings Resolved (2026-02-16)

### Critical Fixes Applied
1. **Calibration config/code mismatch**: `configs/pd_model.yaml` said `isotonic` but thesis uses Platt. Fixed config + script to use `calibrate_platt()`.
2. **Windows backslash paths**: `pd_contract.py` used `str()` producing `models\\pd_canonical.cbm`. Fixed to `.as_posix()`.
3. **mapie version constraint**: Changed `>=0.9` to `>=1.3.0,<2` in `pyproject.toml` to prevent silent breakage.
4. **Legacy conformal config**: Updated `configs/pd_model.yaml` conformal section from MAPIE <1.0 params to 1.3+ params.
5. **Calibration docstring**: Updated `src/models/calibration.py` to reflect Platt as canonical selection.

### Methodology Improvements
6. Added Grade A Mondrian coverage discussion (expandable note in Streamlit).
7. Added LightGBM time series under-coverage explanation (exchangeability violation).
8. Added Cox PH assumption violations discussion with RSF cross-validation.

### New Deliverables
9. **Thesis Contribution page** (`streamlit_app/pages/thesis_contribution.py`): pipeline diagram, comparison table, KPIs, trade-off chart.
10. **Data Architecture enhancements**: dbt SQL viewer, Feast feature view details.
11. **GitHub Actions CI** (`.github/workflows/ci.yml`): lint + test + Streamlit smoke.
12. **Config consistency test** (`tests/test_config_consistency.py`): 7 new tests.
13. **Reproducibility runbook** (`docs/RUNBOOK.md`).

---

## 9) Source of Truth

| Reference | Purpose |
|-----------|---------|
| `SESSION_STATE.md` | Current state and roadmap |
| `README.md` | Setup and architecture summary |
| `reports/project_audit_snapshot.json` | Artifact validation snapshot |
| `models/conformal_policy_status.json` | Conformal policy gate |
| `models/causal_policy_rule.json` | Causal policy selection |
