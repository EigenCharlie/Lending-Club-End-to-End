# CLAUDE.md — Project Context for Claude Code

## PROJECT OVERVIEW

Master's thesis on credit risk management using the **Lending Club Loan Data** (2.26M loans, 2007-2020). Implements a complete ML + Operations Research pipeline with a novel **predict-then-optimize** approach using conformal prediction.

**Owner**: Carlos Vergara | **Language**: Python | **Package manager**: `uv` (not pip)

**I prefer simple, working code over sophisticated abstractions. Get things done quickly and accurately.**

## DOCUMENT SCOPE POLICY

This file and `docs/PROJECT_JUSTIFICATION.md` contain only **current official decisions and standards**.

- Do not store historical decision changes, mistakes, or retrospective notes here.
- Store that history in `docs/DECISION_CHANGES_AND_LEARNINGS.md`.
- Keep runtime metrics dynamic and sourced from artifacts (no fixed snapshot numbers in these docs).

## THESIS CONTRIBUTION

The central innovation is a **predict-then-optimize pipeline with conformal prediction**:

```
CatBoost PD → Calibration Selection (Platt/Isotonic) → MAPIE Mondrian Conformal → [PD_low, PD_high]
  → Box Uncertainty Sets → Pyomo Robust Optimization (HiGHS) → Optimal Portfolio
```

Why this matters:
- Point estimates ignore uncertainty → fragile portfolios
- Bootstrap intervals have no finite-sample guarantees
- Bayesian intervals require distributional assumptions
- **Conformal prediction intervals are distribution-free with mathematical coverage guarantees**

## CURRENT OFFICIAL DECISIONS

- **Serving mode**: Streamlit-first Thesis Mode, with FastAPI/MCP as optional support services.
- **PD architecture**: `Logistic Regression` baseline + `CatBoost` default/tuned + calibrated final model.
- **Validation scheme**: temporal `train/val/cal/test` with strict OOT evaluation.
- **Feature contract**: driven by `data/processed/feature_config.pkl` and persisted in `models/pd_model_contract.json`.
- **HPO policy**: Optuna tuning for CatBoost in canonical training when enabled by config.
- **Calibration policy**: method selected by temporal multi-metric validation policy; resulting method persisted in artifacts.

Current runtime metrics and winners must be read from artifacts, primarily:
- `data/processed/model_comparison.json`
- `models/pd_training_record.pkl`
- `data/processed/pipeline_summary.json`

## DATASET

**Source**: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data

**Splits** (out-of-time, NOT random):
| Split | Rows | Default Rate | Date Range |
|-------|------|-------------|------------|
| Train | 1,346,311 | 18.52% | 2007-06 to 2017-03 |
| Calibration | 237,584 | 22.20% | 2017-03 to 2017-12 |
| Test (OOT) | 276,869 | 21.98% | 2018-01 to 2020-09 |

**Data Leakage (CRITICAL)**: Post-loan variables removed in `src/data/make_dataset.py`:
total_pymnt, total_rec_*, recoveries, collection_recovery_fee, out_prncp*, last_pymnt_*, settlement_*, hardship_*, funded_amnt*.

**Three Analytical Datasets**:
1. `loan_master.parquet` — One row per loan (PD, LGD, survival)
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

### MAPIE 1.3.0 API (current)
- `SplitConformalRegressor` (not MapieRegressor)
- `SplitConformalClassifier` (not MapieClassifier)
- Workflow: `fit()` → `conformalize()` → `predict_interval()`
- `confidence_level` at `__init__` (not alpha at predict)
- `prefit=True` (not cv="prefit")
- PD intervals: wrap CatBoost in `ProbabilityRegressor` (src/models/conformal.py)

## PROJECT STRUCTURE

```
├── CLAUDE.md               # This file
├── SESSION_STATE.md         # Current project state & metrics (source of truth)
├── README.md                # Project overview
├── pyproject.toml           # Dependencies
├── configs/                 # YAML configurations
│   ├── pd_model.yaml
│   ├── optimization.yaml
│   ├── conformal_policy.yaml
│   └── modeva_governance.yaml
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
│       ├── build_gpu_benchmark_notebook.py
│       └── run_modeva_governance_checks.py
├── notebooks/               # Analysis notebooks (01-09 = core thesis)
│   ├── 01_eda_lending_club.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── ... (09 total)
│   └── side_projects/       # Non-core exploratory notebooks
│       ├── 10_rapids_gpu_benchmark_lending_club.ipynb
│       └── 10_modeva_side_task_full_validation.ipynb
├── models/                  # Saved artifacts (DVC tracked)
├── tests/                   # pytest suite
├── reports/                 # Generated reports & figures
├── docs/                    # Technical documentation
├── api/                     # FastAPI services (optional in thesis mode)
└── streamlit_app/           # Primary storytelling dashboard
```

### Pipeline Order (scripts)
1. `src/data/make_dataset.py` → interim cleaned dataset
2. `src/data/prepare_dataset.py` → OOT train/calibration/test splits
3. `src/data/build_datasets.py` + `src/features/feature_engineering.py` → analytical datasets
4. `scripts/train_pd_model.py` → LR baseline + CatBoost default/tuned + selected calibrator + canonical contract
5. `scripts/generate_conformal_intervals.py` → Mondrian conformal intervals
6. `scripts/backtest_conformal_coverage.py` + `scripts/validate_conformal_policy.py` → monitoring + policy gate
7. `scripts/estimate_causal_effects.py` → `simulate_causal_policy.py` → `validate_causal_policy.py`
8. `scripts/run_ifrs9_sensitivity.py` → scenario and sensitivity ECL
9. `scripts/optimize_portfolio.py` + `scripts/optimize_portfolio_tradeoff.py` → allocation + robustness frontier
10. `scripts/end_to_end_pipeline.py` → orchestrates all stages

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

## TESTING

```bash
uv run pytest -x              # Stop on first failure
uv run pytest -m "not slow"   # Skip slow tests
uv run pytest --cov=src       # Coverage report
```

Current tests: 199 passing across data pipeline, features, models, evaluation, optimization, config/DVC consistency, MLflow/utils/scripts, API, Streamlit smoke, and integration.

## IFRS9 / BASEL CONTEXT

**ECL = PD x LGD x EAD x Discount Factor**

| Stage | Trigger | PD Used |
|-------|---------|---------|
| 1 | No SICR | 12-month PD |
| 2 | SICR detected | Lifetime PD |
| 3 | Credit-impaired (90+ DPD) | PD ~ 1.0 |

**Our enhancement**: Conformal interval width (PD_high - PD_point) as additional SICR signal.

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
- Calibration method can change across runs by temporal model-selection policy; check `data/processed/model_comparison.json` for the active winner.
- Side projects (RAPIDS GPU benchmark, Modeva governance) are in `*/side_projects/` — not part of core thesis.
- History of decision changes, errors, and learnings lives in `docs/DECISION_CHANGES_AND_LEARNINGS.md`.
