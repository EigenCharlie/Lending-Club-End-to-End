# SESSION STATE - Lending Club Risk Project
Last Updated: 2026-03-31

---

## 1) Executive Status

Project is operational and artifact-consistent across the thesis pipeline.
Current truth is baseline-registry-first.

Repository hygiene update (2026-03-31):
- Pipeline-first orchestration is now the active execution contract.
- Active vs historical vs research documentation is explicitly separated under `docs/`, `docs/history/`, and `docs/research/`.
- `reports/` root has been reduced to live editorial/technical artifacts; historical snapshots moved to `reports/history/`.
- Research scratch under `models/*_runtime_checkpoints/` was purged; only the still-referenced `conformal_v3_grade_noshrink_2026_03_26` namespace remains under `conformal_gap/`.
- Historical pre-Quarto helpers were archived under `scripts/history/` with compatibility wrappers left in place.

Canonical update (2026-03-13):
- Official operational baseline: `champion-2026-03-12-mega-definitive`.
- Source of truth for baseline resolution: `configs/baselines/canonical_operational_baseline.json`.
- Legacy registry `configs/baselines/core_official_baseline.json` is compatibility fallback only.
- Canonical status artifacts remain single-write (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`).
- Threshold semantics are now explicit in `models/threshold_semantics.json`:
  internal PD screening/search threshold is separate from the operational fairness/approval threshold.

- Serving strategy is now Quarto-first with a reduced local Streamlit companion.
- PD architecture remains Logistic Regression baseline + CatBoost final (tuned + calibrated).
- Calibration candidates: Platt, Isotonic, Venn-Abers, Beta (4 candidates; runtime auto-selection via temporal policy).
- Temporal validation and OOT evaluation remain mandatory.
- Notebooks 10-12 executed with outputs (`include_notebooks=True` in both canonical and paper-grade profiles).
- 690 tests passing, 0 failures, 0 skips.
- All metadata run_tags fixed (MRM, pd_rare_event, mrm_report_status wrapper created).
- Conformal policy test fixed with methodological justification logic.
- This file is only for current state. Historical logs are consolidated in `docs/DECISION_CHANGES_AND_LEARNINGS.md` (section "Session History (Consolidated)").

---

## 2) Serving Architecture Decision (Quarto-First)

Given fixed historical data and showcase objective:

1. Quarto is the primary delivery layer and official source of truth.
2. Streamlit local is an optional companion lab with 5 pages.
3. The public Streamlit showcase is a historical frozen snapshot.
4. DuckDB is used for local analytical queries.
5. dbt provides governance/lineage/tests over analytical assets.
6. Feast is kept as a feature-store consistency layer for train/serve narrative.
7. FastAPI and MCP remain optional support services.

Design implication:
- Priority is narrative quality, reproducibility, and auditability over online serving complexity; Streamlit only keeps interaction that is stronger in app form than in Quarto.

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
25. scripts/end_to_end_pipeline.py        -> orchestration (legacy, see run_smoke_pipeline.py)
26. scripts/export_dvc_metrics.py         -> DVC metrics + plot exports
27. scripts/run_insights_factory.py       -> complementary insight generation (canonical/research profiles)
28. scripts/run_spo_comparison.py          -> SPO+ decision regret comparison (Paper Estrella)
29. scripts/run_spo_real.py               -> SPO+ v2: point-wise MLP, calibrated PD, multi-seed (Paper Estrella)
30. scripts/run_sicr_conformal.py         -> SICR width trigger + ECL alpha sensitivity (Paper 2)
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
- Calibration candidates: Platt, Isotonic, Venn-Abers, Beta (4 candidates)
- Calibration selected: `Venn-Abers` (runtime auto-selection via temporal multi-metric policy)
- AUC: `0.7145`
- Gini: `0.4260`
- KS: `0.3149`
- Brier: `0.1543`
- ECE: `0.0087`
- Log-loss: tracked per calibrator per temporal fold
- PR-AUC: `0.3998`
- Recall@0.35: `0.360`
- F1@0.35: `0.392`
- HPO: 320 Optuna trials (hpo_enabled=false in confirmatory run, uses prior best params)
- Temporal calibration monitoring: per-fold degradation rate + monthly log-loss tracking
- Murphy diagram: available via `src/utils/visualization.py::plot_murphy_diagram()`
- Confirmatory run tag: `paper-grade-pre-quarto` (2026-03-16)

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

### 4.8 SPO+ v2 (Paper Estrella)
- Architecture: point-wise permutation-equivariant MLP (n_features=10 input, not flat 500-dim)
- Costs: calibrated PD via Venn-Abers (continuous [-0.24, +0.17]), not binary default_flag
- Multi-seed: 5 seeds × 200 test instances = 1,000 paired observations
- Two-stage mean regret: `0.4259 ± 0.1173`
- SPO+ mean regret: `0.2168 ± 0.0721` (**49.1% improvement**)
- Conformal robust mean regret: `0.9474 ± 0.2007` (worst-case ≠ expected-cost opt)
- Wilcoxon p-value: `0.0000` (H1: two-stage > SPO+)
- Artifact: `models/spo_real_training_status.json` (SCHEMA_VERSION 2026-03-17.2)

### 4.9 SICR Conformal (Paper 2)
- Optimal width threshold: `t* = 0.30` (F1=0.2515, precision=15.1%, recall of missed=75.8%)
- ECL additional at t*: `$56.6M` (incremental Stage 2 provisioning)
- Alpha sensitivity: ECL goes from `$54.6M` (alpha=0.20) to `$66.4M` (alpha=0.01) — +22% regulatory cost for 99% vs 90% confidence
- Grid: 5 PD thresholds × 20 width thresholds = 100 rows
- Alpha sweep: 8 Mondrian alpha levels from pareto
- Artifacts: `models/sicr_conformal_status.json`, `data/processed/sicr_conformal_grid.parquet`, `data/processed/ecl_alpha_sensitivity.parquet`

---

## 5) Delivery Layer Status (Current)

### Streamlit
- Historical note: page counts in this file are snapshots and may drift; use runtime exports and Streamlit utilities for live inventory.
- Professional light theme with audience toggle (General/Negocio/Técnico).
- Model laboratory and thesis pages consume runtime artifacts for metrics.
- Includes A/B testing simulation, fairness audit, CATE portfolio, 3 paper draft pages, and Paper Estrella page.

### Insights Factory
- Entrypoint: `scripts/run_insights_factory.py`
- Two profiles: `canonical` (lightweight) and `research` (GPU + notebooks + SPO+).
- Consumes canonical artifacts without modifying or promoting.
- Canonical profile: conformal method registry, set prediction benchmark, rare-event calibration, paper notebooks, image extraction, storytelling snapshot.
- Research profile adds: all notebooks, PD RAPIDS benchmark, RAPIDS insight factory (cuDF/cuML/cuGraph), SPO+ comparison.

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

Current state (2026-03-23): **690 tests passing, 0 failures, 0 skips**.

Key changes since last snapshot:
- CRPTO test skip removed (test now runs normally)
- Conformal policy test fixed (methodological justification pass logic)
- Beta calibration tests added
- Classification set benchmark tests added
- Quarto book guardrail tests (3/3 passing)

Operational note:
- `data/processed/runtime_status.json` is a generated snapshot and may lag until `scripts/export_streamlit_artifacts.py` is re-run.
- For live inventory: `uv run pytest --collect-only -q`

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
