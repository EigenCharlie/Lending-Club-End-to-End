# SESSION STATE - Lending Club Risk Project
Last Updated: 2026-05-04

---

## 1) Executive Status

Project is operational and artifact-consistent across the thesis pipeline.
Current truth is **`models/final_project_promotion.json` + `models/champion_portfolio_policy.json`**.

### Current closure (Paper Estrella, 2026-05-04)

The Paper Estrella closure is now anchored on the **bound-aware 276K economic champion**:

- Run tag canonical paper: `paper-thesis-final-economic-2026-04-06`
- Promotion basis: `economic_champion_within_exact_robust_region`
- Champion policy: `risk_tolerance=0.175`, `gamma=0.45`, `policy_mode=blended_uncertainty`, `uncertainty_aversion=0.10`
- `theorem_tight_comparator` (gamma=0.55) and `balanced_comparator` (rt=0.17) remain as documented internal comparators, NOT as official champions
- Robust region: 45/45 policies pass `alpha=0.01` exactly on full 276K OOT
- Conformal winner: `score_decile_mondrian` (rank1 reopen), coverage 92.97%, Winkler 1.111
- PD upstream: HPO local trial 56 (val_AUC 0.722, OOT AUC 0.7124, Brier 0.1546, ECE 0.0064, Venn-Abers calibrator)
- Paper-facing tables regenerated via `scripts/export_paper1_canonical_tables.py` from canonical sources

### Pipeline freeze policy (2026-05-04)

Non-search pipelines (`paper1_e2e`, `paper2_e2e`, `core_canonical`, `canonical_rebuild`) now use `freeze_if_available` execution mode:

- `core_portfolio` runs in ~1 min (LP + AB on frozen champion) instead of ~3 hours (tradeoff + selector + AB rebuild)
- AB simulation uses `--policy_selector explicit_champion_only` to read directly from `models/champion_portfolio_policy.json`
- Search-only pipelines (`search_portfolio`, `search_pd`) still do full search when invoked explicitly
- See `configs/profiles/{paper1_e2e_default, core_canonical_cpu, canonical_operational, canonical_confirmatory_full, paper2_e2e_default}.yaml`

### Repository hygiene

- Pipeline-first orchestration is the active execution contract.
- Active/historical/research docs separated under `docs/`, `docs/history/`, `docs/research/`.
- 716 tests passing, 0 failures, 2 skips (as of 2026-05-04).
- All metadata run_tags fixed; conformal policy test fixed with methodological justification logic.
- This file is only for current state. Historical logs in `docs/DECISION_CHANGES_AND_LEARNINGS.md`.

### Canonical baselines

- Official operational PD baseline: `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`.
- Paper Estrella final closure: `paper-thesis-final-economic-2026-04-06`.
- Source of truth for baseline resolution: `configs/baselines/canonical_operational_baseline.json`.
- Threshold semantics in `models/threshold_semantics.json`: internal PD screening (0.05) vs operational approval (0.35).
- Serving: Quarto-first with reduced local Streamlit companion (5 pages).
- PD: LR baseline + CatBoost (tuned monotónico + Venn-Abers calibrator); auto-selection from 4 calibrators.
- Notebooks 10-12 executed with outputs preserved (`include_notebooks=True`).

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

- Best model: `CatBoost (tuned + calibrated)` with monotonic constraints `installment:1, annual_inc:-1, dti:1, loan_to_income:1`
- Calibration candidates: Platt, Isotonic, Venn-Abers, Beta (4 candidates)
- Calibration selected: `Venn-Abers` (runtime auto-selection via temporal multi-metric policy)
- AUC: `0.7124` (latest: `paper1-e2e-all-champions-2026-04-07` run)
- Brier: `0.1546`
- ECE: `0.0064`
- KS: `0.3115`
- Best historical PD-only: `pd-hpo-local-2026-04-03-1325` (AUC 0.7139, Brier 0.1544 — documented but not promoted as paper champion to avoid family-mixing)
- Confirmatory run tag (operational/regulatory): `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`
- Source of truth: `data/processed/pipeline_summary.json`, `reports/dvc/metrics_summary.json`

### 4.2 Conformal (Mondrian — Reopen Rank1 Winner)

- Variant: `score_decile_mondrian` (winner from `conformal-reopen-2026-04-03-2149__resume__2026-04-05-1612`)
- Config: `partition=grade, prob_source=calibrated, n_bins=10, fallback=global_only, score_scale=bernoulli_sqrt, min_group_size=100, calibration_fraction=0.5`
- Coverage 90%: `0.9293` (target 0.90)
- Coverage 95%: `0.9663` (target 0.95)
- Avg width 90%: `0.7642`
- Min group coverage 90%: `0.9004`
- Winkler 90: `1.1937` (raw pass, policy pass)
- Policy checks: Kupiec/Christoffersen are diagnostic only (expected to fail on 276K OOT — statistical power artifact)
- Methodological justification pass: `true` (paper-grade closure authoritative)
- Source of truth: `models/conformal_policy_status.json`

### 4.3 Causal Policy
- Selected rule: `high_plus_medium_positive`
- Selected action rate: `26.31%`
- Selected total net value: `5.86M`
- Selected bootstrap p05 net value: `5.82M`

### 4.4 IFRS9 Sensitivity
- Baseline total ECL: `0.999B`
- Conservative total ECL: `1.802B`
- ECL range: `0.803B`

### 4.5 Optimization Robustness — Bound-Aware 276K Closure

- **Champion run tag**: `paper-thesis-final-economic-2026-04-06`
- **Champion label**: `bound_aware_276k_economic_champion`
- **Realized total return**: `$170,464.54`
- **Price of robustness**: `-$14,465.69` (-10.56%)
- **Robust region cardinality**: 45 unique policies, 100% pass `alpha=0.01` exactly
- **Region span**: `risk_tolerance ∈ [0.155, 0.175]`, `gamma ∈ [0.45, 0.55]`, `uncertainty_aversion ∈ [0.0, 0.1]`
- **Bound-aware metrics** at champion: `V=0.03645`, `gamma_cp=0.18591`, `violation=0.0`

### 4.5.1 Champion Portfolio Policy (Promoted 2026-04-06, restored 2026-05-04)

- Artifact: `models/champion_portfolio_policy.json` (mirror of `models/final_project_promotion.json::final_champion`)
- `risk_tolerance`: `0.175`
- `policy_mode`: `blended_uncertainty`
- `gamma`: `0.45`
- `uncertainty_aversion`: `0.10`
- Selected by: bound-aware 276K full-OOT mini-grid + economic ranking inside the exact robust region
- Comparators (documented, not champions):
  - `theorem_tight_comparator`: `rt=0.175, gamma=0.55` — best `V`/`gamma_cp` tightness, return $166,270
  - `balanced_comparator`: `rt=0.17, gamma=0.45` — middle of region, return $169,390

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
