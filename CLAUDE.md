# CLAUDE.md — Project Context for Claude Code

## PROJECT OVERVIEW

Master's thesis repository on credit risk management using the **Lending Club
Loan Data** (Kaggle 2007-2020Q3 source; modeling uses a cleaned, resolved-only
temporal subset). It preserves a historical ML + Operations Research
**predict-then-optimize** pipeline and its subsequent identification audit.

**Owner**: Carlos Vergara | **Language**: Python | **Package manager**: `uv` (not pip)

**I prefer simple, working code over sophisticated abstractions. Get things done quickly and accurately.**

## DOCUMENT SCOPE POLICY

This file and `docs/PROJECT_JUSTIFICATION.md` contain current engineering
standards for this repository. Scientific claims about the active IJDS paper
belong to the independent `Paper_CRPTO` repository.

- Do not store historical decision changes, mistakes, or retrospective notes here.
- Store that history in `docs/DECISION_CHANGES_AND_LEARNINGS.md`.
- Keep runtime metrics dynamic and sourced from artifacts (no fixed snapshot numbers in these docs).

## THESIS CONTRIBUTION

The repository contains a historical **predict-then-optimize pipeline with
conformal prediction**:

```
CatBoost PD → Calibration Selection → binary-outcome conformal endpoints
  → diagnostic box-stress heuristics → Pyomo portfolio experiments
```

Why this matters:
- Point estimates ignore uncertainty → fragile portfolios
- Bootstrap intervals have no finite-sample guarantees
- Bayesian intervals require distributional assumptions
- Conformal coverage is finite-sample under the method's assumptions for the
  declared future outcome. With binary calibration labels here, it covers
  observed `Y`, not latent individual PD, ECL, SICR or a selected policy.

The current external CRPTO/IJDS surface is a retrospective identification
audit. It selects no learner, window, taxonomy, gamma, ruler, coordinate, cap,
comparator or portfolio policy. Never promote this repository's legacy
`pd_low`/`pd_high`, IFRS9 or `champion` artifacts into that paper by default.
The observed external authority is pinned by commit and content hash in
`docs/research/crpto_external_contract_2026-07-20.yml`; refresh that contract
explicitly instead of accepting cross-repository drift.

## CURRENT ENGINEERING DECISIONS AND HISTORICAL LOCAL FREEZE

- **Serving mode**: Quarto-first publication mode, with a reduced local Streamlit companion for optional interactive analysis and FastAPI/MCP as optional support services.
- **Historical score architecture**: `Logistic Regression` baseline + `CatBoost` default/tuned + a locally selected calibrator. The frozen artifact labelled `champion` uses monotonic constraints `installment:1, annual_inc:-1, dti:1, loan_to_income:1`; the label is runtime provenance, not a current scientific winner.
- **Historical evaluation scheme**: chronological `train/val/cal/test` partitions. The resolved-only `test OOT` is a retrospective internal evaluation that was reused downstream; it is not an untouched, prospective or external holdout.
- **Feature contract**: driven by `data/processed/feature_config.pkl` and persisted in `models/pd_model_contract.json`.
- **HPO runtime**: Optuna tuning for CatBoost when enabled by config. The frozen local artifact records trial 56 parameters; this does not select a current CRPTO learner.
- **Calibration runtime**: 4 candidates (Platt, Isotonic, Venn-Abers, Beta) are compared by a temporal multi-metric routine. Venn-Abers was retained in the historical local freeze. Config files are **templates** with defaults (`method: auto`); runtime artifacts reproduce software behavior, not paper-wide scientific authority.
- **Historical conformal freeze**: `score_decile_mondrian` (rank1 reopen,
  2026-04-05), retained for local reproducibility; not selected by current CRPTO.
- **Historical portfolio freeze**: `bound_aware_276k_economic_champion` from
  `paper-thesis-final-economic-2026-04-06`, retained for non-search pipeline
  compatibility; not a current scientific promotion or deployment policy.
- **Pipeline freeze policy**: Non-search pipelines (`paper1_e2e`, `paper2_e2e`, `core_canonical`, `canonical_rebuild`) use `freeze_if_available` execution mode — no portfolio re-search. Search pipelines (`search_portfolio`, `search_pd`) still do full search when invoked explicitly.

To reproduce the historical local runtime, read the following artifacts.
Fields such as `winner`, `champion`, `selected` or `promotion` are compatibility
and provenance labels; they do not override the claim contracts in
`SESSION_STATE.md`:

- `models/final_project_promotion.json` (legacy promotion package + comparators + local robust-region diagnostics)
- `models/champion_portfolio_policy.json` (frozen policy used by all non-search pipelines)
- `models/champion_registry.json` (broader registry incl. PD, conformal, threshold semantics)
- `data/processed/pipeline_summary.json` (historical score/conformal/portfolio snapshot)
- `data/processed/model_comparison.json` (PD model family comparison)
- `models/pd_training_record.pkl` (PD training audit log)

## DATASET

**Source**: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data

**Historical resolved-only splits** (chronological, not random; subject to
maturity selection and downstream reuse):
| Split | Rows | Default Rate | Date Range |
|-------|------|-------------|------------|
| Train | 1,346,311 | 18.52% | 2007-06 to 2017-03 |
| Calibration | 237,584 | 22.20% | 2017-03 to 2017-12 |
| Test (OOT) | 276,869 | 21.98% | 2018-01 to 2020-09 |

**Data Leakage (CRITICAL)**: Post-loan variables removed in `src/data/make_dataset.py`:
total_pymnt, total_rec_*, recoveries, collection_recovery_fee, out_prncp*, last_pymnt_*, settlement_*, hardship_*, funded_amnt*.

**Three Analytical Datasets**:
1. `loan_master.parquet` — One row per loan (binary score and historical severity/survival proxies)
2. `time_series.parquet` — Monthly aggregates (118 rows, Nixtla-ready: unique_id, ds, y)
3. `ead_dataset.parquet` — Defaults only (EAD modeling)

## TECH STACK

All dependencies in `pyproject.toml`. Key versions (as of 2026-02-09):

```bash
uv sync --extra dev     # Install all deps
uv run pytest -x        # Tests (stop on first failure)
uv run ruff check src/  # Lint
uv run ruff format src/ # Format
```

| Category | Libraries |
|----------|-----------|
| ML | catboost 1.2.8, scikit-learn 1.6.1, lightgbm 4.5+, optuna 4.7, shap 0.48, optbinning |
| Conformal | mapie 1.3.0 (SplitConformalRegressor, Mondrian), crepes |
| Time Series | statsforecast 2.0+, mlforecast 0.13+, hierarchicalforecast 1.0+ |
| Survival | lifelines 0.30+, scikit-survival 0.24+ |
| Causal | econml 0.16+, dowhy 0.12+ |
| Optimization | pyomo 6.8+, highspy 1.10+ (HiGHS solver), cvxpy 1.6+ |
| MLOps | dvc 3.56+, mlflow 3.9+, dagshub, pandera 0.22+ |
| Dev | uv, ruff, pytest, nbstripout, loguru, pre-commit |

### MAPIE 1.3.0 API used by the local runtime
- `SplitConformalRegressor` (not MapieRegressor)
- `SplitConformalClassifier` (not MapieClassifier)
- With the local `prefit=True` wrapper, the estimator is already fitted:
  `SplitConformalRegressor(...)` → `conformalize()` → `predict_interval()`;
  do **not** call `fit()` on the MAPIE wrapper.
- `confidence_level` at `__init__` (not alpha at predict)
- `prefit=True` (not cv="prefit")
- Binary-outcome endpoints: the legacy implementation wraps CatBoost in
  `ProbabilityRegressor` (`src/models/conformal.py`). Schema names such as
  `pd_low`/`pd_high` do not make those endpoints intervals for latent PD.

## PROJECT STRUCTURE

```
├── CLAUDE.md               # This file
├── SESSION_STATE.md         # Current project state snapshot + operating notes
├── README.md                # Project overview
├── pyproject.toml           # Dependencies
├── configs/                 # YAML configurations
│   ├── pd_model.yaml
│   ├── optimization.yaml
│   ├── conformal_policy.yaml
│   ├── fairness_policy.yaml
│   └── mrm_policy.yaml
├── data/
│   ├── raw/                 # Original CSV (manual download from Kaggle)
│   ├── interim/             # Cleaned parquet
│   └── processed/           # Final splits, features, intervals, scenarios
├── src/                     # Reusable source code
│   ├── data/                # make_dataset, prepare_dataset, build_datasets
│   ├── features/            # feature_engineering, schemas (Pandera)
│   ├── models/              # pd_model, calibration, conformal, lgd, ead,
│   │                        # time_series, survival, causal, pd_contract
│   ├── optimization/        # portfolio_model, robust_opt, sda, spo_integration
│   ├── evaluation/          # metrics, backtesting, ifrs9
│   └── utils/               # mlflow_utils, visualization
├── scripts/                 # Executable pipeline scripts
│   ├── train_pd_model.py
│   ├── generate_conformal_intervals.py
│   ├── optimize_portfolio.py
│   ├── end_to_end_pipeline.py
│   ├── ... (15 total core scripts)
│   └── side_projects/       # Non-core exploratory scripts
│       └── build_gpu_benchmark_notebook.py
├── notebooks/               # Analysis notebooks (01-09 = core thesis)
│   ├── 01_eda_lending_club.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── ... (09 total)
│   └── side_projects/       # Non-core exploratory notebooks
│       └── 10_rapids_gpu_benchmark_lending_club.ipynb
├── models/                  # Saved artifacts (DVC tracked)
├── tests/                   # pytest suite
├── reports/                 # Generated reports & figures
├── docs/                    # Technical documentation
├── api/                     # FastAPI services (optional in thesis mode)
└── streamlit_app/           # Reduced local interactive companion (5 labs)
```

### Pipeline Order (scripts)
1. `src/data/make_dataset.py` → interim cleaned dataset
2. `src/data/prepare_dataset.py` → OOT train/calibration/test splits
3. `src/data/build_datasets.py` + `src/features/feature_engineering.py` → analytical datasets
4. `scripts/train_pd_model.py` → LR baseline + CatBoost variants + locally retained calibrator + runtime contract
5. `scripts/generate_conformal_intervals.py` → legacy-named endpoints for binary outcome `Y`
6. `scripts/backtest_conformal_coverage.py` + `scripts/validate_conformal_policy.py` → historical coverage diagnostics + internal software gate
7. `scripts/estimate_causal_effects.py` → `simulate_causal_policy.py` → `validate_causal_policy.py` (observational diagnostic; no identified policy value)
8. `scripts/run_ifrs9_sensitivity.py` → IFRS9-inspired mechanical scenarios; no accounting measurement
9. `scripts/optimize_portfolio.py` + `scripts/optimize_portfolio_tradeoff.py` → historical allocation stress + retrospective frontier
10. `scripts/run_fairness_audit.py` → internal approval-proxy disparity diagnostics; no fair-lending determination
11. `scripts/optimize_cate_portfolio.py` → CATE-labelled stress comparison; no causal policy selection
12. `scripts/simulate_ab_test.py` → retroactive A/B simulation (robust vs non-robust)
13. `scripts/generate_mrm_report.py` → internal MRM diagnostic report; not independent SR 11-7 validation
14. `scripts/end_to_end_pipeline.py` → orchestrates all stages

### Canonical Model Contract
- `models/pd_model_contract.json` — feature names, types, thresholds
- `models/pd_canonical.cbm` — trained CatBoost model
- `models/pd_canonical_calibrator.pkl` — selected score calibrator artifact
Downstream scripts consume this contract to avoid feature drift.

## CODING STANDARDS

- **Type hints** on all function signatures
- **Google-style docstrings** on all public functions
- **loguru** for logging (not print): `from loguru import logger`
- **Pandera schemas** at data pipeline boundaries
- **Constants** in UPPER_SNAKE_CASE at module top
- **Config** via YAML files in configs/ (not hardcoded params)
- **No star imports** (`from x import *`)
- Files should be <400 lines. If longer, refactor.
- Notebooks call functions from `src/` — no duplicated logic.
- **Tool isolation**: Always use `uv run` for dev deps and `uvx` for one-off tools (never global installs).
- **Pre-commit**: ruff lint+format, nbstripout, trailing-whitespace, check-yaml/toml, large file guard.

## TESTING

```bash
uv run pytest -x              # Stop on first failure
uv run pytest -m "not slow"   # Skip slow tests
uv run pytest --cov=src       # Coverage report
```

Test inventory and counts evolve frequently. Use `uv run pytest --collect-only -q` (or `data/processed/runtime_status.json` if freshly exported) for current totals; the suite covers data pipeline, features, models, evaluation, optimization, config/DVC consistency, MLflow/utils/scripts, API, Streamlit smoke/imports, and integration.

Pytest config uses `--strict-markers --strict-config` to prevent typos in markers and invalid configs.
Ruff rules: `E, F, W, I, UP, B, SIM, C4` (includes flake8-comprehensions).

## IFRS9 / BASEL CONTEXT AND PROJECT BOUNDARY

A common modeling approximation is `PD x LGD x EAD x Discount Factor`. It is
not the complete IFRS 9 accounting measurement, which requires coherent
reporting-date information, horizons, probability-weighted cash shortfalls,
forward-looking scenarios and time value of money.

| Stage | Condition | Measurement posture |
|-------|-----------|---------------------|
| 1 | No significant increase since initial recognition | 12-month ECL |
| 2 | Significant increase in credit risk | Lifetime ECL |
| 3 | Credit-impaired asset | Lifetime ECL with the applicable interest treatment |

The historical prototype tested binary-endpoint width as a SICR-labelled proxy.
That rule is retired: width does not identify a change in PD, SICR, Stage 2 or
ECL. Likewise, 30/90 DPD thresholds are rebuttable backstops or default
presumptions in the relevant context, not automatic stage identities.

## KEY REFERENCES

1. Elmachtoub & Grigas (2022) — Smart Predict, then Optimize (SPO+ loss)
2. Romano et al. (2019) — Conformalized Quantile Regression
3. Chernozhukov et al. (2018) — Double/Debiased ML
4. Athey & Wager (2019) — Causal Forest
5. Taquet et al. (2025) — MAPIE library
6. Vovk et al. (2005) — Algorithmic Learning in a Random World

## IMPORTANT NOTES

- Kaggle dataset must be downloaded manually to `data/raw/`
- WOE features are computed in NB02 via OptBinning (not pre-existing in raw data)
- CatBoost handles NaN natively — no imputation needed. LogReg baseline uses fillna(0).
- LGD modeling only uses defaults (default_flag=1). ~88% null LGD values are expected.
- The local calibration routine can retain a different method across runs (Platt, Isotonic, Venn-Abers, Beta); `data/processed/model_comparison.json` records that run's selection, not an active scientific winner.
- Side projects (RAPIDS GPU benchmark) are in `*/side_projects/` — not part of core thesis.
- **Legacy Paper 1 tables** (`reports/paper_material/paper1/tables/*`) are regenerated only via `scripts/export_paper1_canonical_tables.py` for historical reproducibility. They are not the current external CRPTO claim surface.
- **Legacy Paper Estrella P1 evidence** remains in `models/paper1_p1_evidence_status.json`, `docs/research/paper_estrella_p1_evidence_2026-05-04.md`, and `paper1_tableA3`--`paper1_tableA6`; it records post-selection diagnostics and synthetic stresses, not a current selection or prospective validation.
- **Legacy policy naming**: the root freeze labels the economic point, rather
  than `theorem_tight`, as `champion` for pipeline compatibility. This is
  historical provenance only; current Paper_CRPTO selects no policy.
- **Pipeline freeze**: `paper1_e2e`, `paper2_e2e`, `core_canonical`, `canonical_rebuild` use `freeze_if_available` mode and the AB selector `explicit_champion_only` to read `models/champion_portfolio_policy.json` directly — no portfolio re-search.
- History of decision changes, errors, and learnings lives in `docs/DECISION_CHANGES_AND_LEARNINGS.md`.
