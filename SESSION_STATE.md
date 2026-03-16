# SESSION STATE - Lending Club Risk Project
Last Updated: 2026-03-13

---

## 1) Executive Status

Project is operational and artifact-consistent across the thesis pipeline.
Current truth is baseline-registry-first.

Canonical update (2026-03-13):
- Official operational baseline: `champion-2026-03-12-mega-definitive`.
- Source of truth for baseline resolution: `configs/baselines/canonical_operational_baseline.json`.
- Legacy registry `configs/baselines/core_official_baseline.json` is compatibility fallback only.
- Canonical status artifacts remain single-write (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`).
- Threshold semantics are now explicit in `models/threshold_semantics.json`:
  internal PD screening/search threshold is separate from the operational fairness/approval threshold.

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
- Calibration selected: `Venn-Abers`
- AUC: `0.7128`
- Gini: `0.4256`
- KS: `0.3134`
- Brier: `0.1545`
- ECE: `0.0062`
- HPO: reused best trial 855 (val AUC 0.7201) from prior Optuna study

### 4.2 Conformal (Mondrian)
- Variant: `score_decile_mondrian`
- Coverage 90%: see `models/conformal_policy_status.json`
- Coverage 95%: see `models/conformal_policy_status.json`
- Min group coverage 90%: see `models/conformal_policy_status.json`
- Policy checks: Kupiec/Christoffersen are diagnostic only (expected to fail on 276K OOT — statistical power artifact)
- Methodological justification pass: `true` (paper-grade closure authoritative)
- `comparison.json` operational_overall_pass: `true` (paper-grade run 2026-03-13)

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Selected action rate: `26.31%`
- Selected total net value: `5.86M`
- Selected bootstrap p05 net value: `5.82M`

### 4.4 IFRS9 Sensitivity
- Baseline total ECL: `0.999B`
- Conservative total ECL: `1.802B`
- ECL range: `0.803B`

### 4.5 Optimization Robustness

- Non-robust return: `$82,483` (155 loans funded)
- Robust return: `$169,491` (300 loans funded)
- Price of robustness (absolute): `$-87,007`

### 4.5.1 Champion Portfolio Policy (Promoted 2026-03-16)

- Artifact: `models/champion_portfolio_policy.json` (`promoted: true`)
- `risk_tolerance`: `0.18`
- `policy_mode`: `segment_relative_tail_blended_uncertainty`
- `gamma`: `0.1`
- `uncertainty_aversion`: `0.1`
- Selected by: economic selector v3 + A/B baseline no-regression PASS
- Ambiguity-defer scenario: NOT promoted (diff=-$13.5K, outside $1.1K tolerance; gate is diagnostic only)

### 4.6 Fairness Audit
- Overall pass: `true` (`6/6` attributes pass)
- Threshold operativo oficial: `0.35`
- Threshold interno PD search/screening: ver `models/threshold_semantics.json`
- Fairness y decisión operativa deben leerse con el threshold oficial, no con el threshold interno PD.

### 4.7 A/B Simulation
- Strategy A (non-robust): mean return `1.4753`
- Strategy B (robust): mean return `1.5175`
- Diff B-A: `0.0422` with CI `[-0.5790, 0.6600]`
- No-regression gate: PASS

---

## 5) Delivery Layer Status (Current)

### Streamlit
- Historical note: page counts in this file are snapshots and may drift; use runtime exports and Streamlit utilities for live inventory.
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

Historical verification snapshot on 2026-03-01:

- Prior pass counts in this file are historical only and must not be read as live inventory.
- Current test/page counts are generated dynamically by runtime exports and Streamlit utilities.

Operational note:
- `data/processed/runtime_status.json` is a generated snapshot and may lag until `scripts/export_streamlit_artifacts.py` is re-run.

## 8) Current Priorities

1. Keep docs and Streamlit narratives strictly artifact-driven (no stale hardcoded claims).
2. Config files are templates — runtime calibration selection is artifact-driven.
3. Preserve reproducibility gates (`ruff`, `pytest`, `dvc`) in routine runs.
4. DVC stage counts in this file are historical snapshots; `dvc.lock` is authoritative for artifact hashes.

---

## 9) Source of Truth

| Reference | Purpose |
|-----------|---------|
| `SESSION_STATE.md` | Current official state |
| `docs/PROJECT_JUSTIFICATION.md` | Current official design rationale |
| `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Historical decisions, errors, learnings, and session history |
| `models/threshold_semantics.json` | Canonical split between internal PD threshold and operational fairness/approval threshold |
| `data/processed/model_comparison.json` | PD model comparison and final metrics |
| `models/conformal_policy_status.json` | Conformal policy gate snapshot |
| `models/causal_policy_rule.json` | Causal policy rule and selected metrics |
| `data/processed/pipeline_summary.json` | Cross-module pipeline KPI snapshot |
| `models/champion_portfolio_policy.json` | Promoted portfolio champion policy (risk_tolerance=0.18) |
| `models/champion_registry.json` | Full champion registry across all modules |
