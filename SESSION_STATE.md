# SESSION STATE — Lending Club Risk Project
Last Updated: 2026-02-18

---

## 1) Executive Status

Project is operational and artifact-consistent across the full thesis pipeline.

- Notebooks `01` to `09`: executed with zero errors.
- Core scripts: producing artifacts in `data/processed/` and `models/`.
- Policy gates: conformal policy gate activo (snapshot actual: 1/7 checks tras hardening sin leakage), causal policy selected, IFRS9 sensitivity produced.
- Tests: `201` passing (4.88s).

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

Source: canonical runtime artifacts (`data/processed/model_comparison.json`,
`models/conformal_policy_status.json`, `data/processed/ifrs9_scenario_summary.parquet`,
`data/processed/portfolio_robustness_summary.parquet`).

### 4.1 PD Model
- AUC: `0.6990`
- Gini: `0.3980`
- KS: `0.2942`
- Brier: `0.1572`
- ECE: `0.0130`

### 4.2 Conformal (Mondrian)
- Coverage 90%: `0.8887`
- Coverage 95%: `0.9480`
- Avg width 90%: `0.7459`
- Min group coverage 90%: `0.8522`
- Policy checks: `1/7` pass (`overall_pass = false`, snapshot vigente tras remover adaptación con etiquetas de test)

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Action rate: `27.51%`
- Net value: `43.67M`
- Bootstrap p05: `43.52M`

### 4.4 IFRS9 Sensitivity
- Baseline ECL: `1.010B`
- Severe ECL: `1.851B`
- Uplift: `+83.18%`

### 4.5 Optimization Robustness
- Tolerance `0.10`: robust funded `9`, baseline non-robust funded `150`, price of robustez `97.05%`

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

---

## 10) Post-Reboot Recovery Log (2026-02-17)

Recovery and synchronization checks executed after reboot:

- Quality gates: `uv run ruff check src/ scripts/ tests/` and `uv run ruff format --check src/ scripts/ tests/` both green.
- Functional suite: `uv run pytest -q` -> `199 passed` (4.77s).
- DVC local consistency: `uv run dvc status --json` -> `{}`.
- DVC cloud consistency: `uv run dvc status -c --json` -> `{}`.
- DVC upload smoke: `uv run dvc push -r dagshub -v --jobs 1` -> `Everything is up to date`.
- DVC topology: `uv run dvc dag` includes governance/showcase stages (`backtest_conformal_coverage`, `validate_conformal_policy`, `export_streamlit_artifacts`, `export_storytelling_snapshot`).
- MLflow tracking (DagsHub): 8 experiments under `lending_club/*` with latest runs in `FINISHED` state (UTC timestamps from current session).
- Narrative consistency: no stale hardcoded policy-check ratio claims in docs/pages (guarded by `tests/test_docs/test_narrative_consistency.py`).

Git sync state:

- Local branch and remotos (`origin`, `dagshub`) deben mantenerse alineados en el mismo SHA de `HEAD`.
- Tokens y credenciales se gestionan vía `.env` + `scripts/configure_integrations.sh` (DagsHub-first).

---

## 11) Repro-Contract Closure Log (2026-02-18)

Implementation pass executed on branch `stabilization/repro-contract-v1`.

### 11.1 Quality + Functional Gates
- `uv run ruff check src/ scripts/ tests/` -> pass.
- `uv run ruff format --check src/ scripts/ tests/` -> pass.
- `uv run pytest -q` -> `199 passed`.

### 11.2 DVC Consistency + Cloud Sync
- `uv run dvc repro build_pipeline_results`
- `uv run dvc repro forecast_default_rates run_survival_analysis export_streamlit_artifacts`
- `uv run dvc repro export_storytelling_snapshot`
- `uv run dvc status --json` -> `{}`
- `uv run dvc push -r dagshub -v --jobs 1` -> `17 files pushed`
- `uv run dvc status -c --json` -> `{}`

### 11.3 Key Operational Fixes Applied
- Added missing DVC lock metadata for new stage/outs (repro contract restored).
- Hardened `forecast_default_rates.py` with baseline-only fallback when LightGBM native deps are unavailable (`libgomp` missing).
- Streamlit pages now degrade gracefully for optional artifacts and use runtime status snapshot for test/page counters.
- API model loading switched to canonical-first with legacy fallback.
- `end_to_end_pipeline.py` now supports raw-to-end default path plus `--skip-make-dataset`.
- DagsHub-only integrations retained; no operational Google Drive path remains.

### 11.4 MLflow Backfill (DagsHub)
Executed:

`uv run python scripts/log_mlflow_experiment_suite.py --repo-owner "$DAGSHUB_USER" --repo-name "$DAGSHUB_REPO"`

Latest run IDs (UTC 2026-02-18 02:07:18Z batch):
- `end_to_end`: `3e2e9d8b5c8f4c92b2f4c32f3fde1d9e`
- `pd_model`: `36f602f0836843ad8c01e8d542a1a051`
- `conformal`: `0032c4c02ad74f0ea789c70656abbdc9`
- `causal_policy`: `9026662a07074deab5beba4b1b8a1b38`
- `ifrs9`: `a93a6b7d5bf1452a982dec001e093754`
- `optimization`: `c17bdd4dc5064a0ebb60b1fa884378c5`
- `survival`: `68842353493b4495a92758c92bc896a2`
- `time_series`: `bb9b099783a54b94b5b86d8ca317bc60`

---

## 12) Validity Hardening Log (2026-02-18)

Execution pass completed on branch `publication/validity-hardening` for phases 0-5:

- **Phase 0 (baseline + reproducibilidad):** baseline congelado en `reports/baseline_2026-02-18.json`, `dvc status` local limpio.
- **Phase 1 (validity-critical):** conformal sin leakage por etiquetas de test, alineación `id` en optimización, y corrección `nonrobust_funded`.
- **Phase 2 (narrativa + literatura):** snapshots dinámicos en páginas Streamlit + actualización de panorama con papers recientes.
- **Phase 3 (benchmarks CP + OR):** nuevo benchmark de variantes conformales y comparación por subgrupos.
- **Phase 4 (survival + IFRS9):** `time_to_event` con fechas reales cuando existen e incertidumbre conformal operativa en SICR.
- **Phase 5 (causal temporal):** backtest OOT mensual de política causal con ventana mínima de historia.

Validation snapshot after rerun:
- `uv run dvc repro ... export_storytelling_snapshot` -> completed (exit 0).
- `uv run pytest -q` -> `201 passed`.
- `uv run dvc status --json` -> `{}`.
- `uv run dvc status -c --json` -> artifacts pending push to remote (expected until `dvc push`).

Detailed rationale, file-level changes, before/after deltas and pending risks:
- `reports/PHASES_0_5_EXECUTION_2026-02-18.md`
