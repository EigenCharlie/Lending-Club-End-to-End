# Reproducibility Runbook

Step-by-step guide to reproduce the entire project from a fresh clone.

## Prerequisites

- **Python 3.11** (3.11.x, not 3.12+)
- **uv** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Kaggle dataset**: Download manually from https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data and place CSV in `data/raw/`

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd Lending-Club-End-to-End

# 2. Install dependencies
uv sync --extra dev

# 3. Place Kaggle data
# Download Loan_status_2007-2020Q3.csv to data/raw/

# 4. Run the full pipeline
uv run python scripts/end_to_end_pipeline.py

# 5. Verify tests pass
uv run pytest -x

# 6. Launch Streamlit dashboard
uv run streamlit run streamlit_app/app.py
```

## Step-by-Step Pipeline

If you want to run individual stages:

| Step | Command | Produces |
|------|---------|----------|
| 1 | `uv run python -c "from src.data.make_dataset import main; main()"` | `data/interim/lending_club_cleaned.parquet` |
| 2 | `uv run python -c "from src.data.prepare_dataset import main; main()"` | Train/calibration/test splits |
| 3 | `uv run python -c "from src.data.build_datasets import main; main()"` | loan_master, time_series, ead_dataset |
| 4 | `uv run python scripts/train_pd_model.py` | CatBoost model + Platt calibrator + contract |
| 5 | `uv run python scripts/generate_conformal_intervals.py` | Mondrian conformal intervals |
| 6 | `uv run python scripts/backtest_conformal_coverage.py` | Temporal monitoring |
| 7 | `uv run python scripts/validate_conformal_policy.py` | Policy gate (checks formales de conformal) |
| 8 | `uv run python scripts/estimate_causal_effects.py` | CATE estimates |
| 9 | `uv run python scripts/simulate_causal_policy.py` | Policy simulation |
| 10 | `uv run python scripts/validate_causal_policy.py` | Rule selection + bootstrap |
| 11 | `uv run python scripts/run_ifrs9_sensitivity.py` | ECL scenarios |
| 12 | `uv run python scripts/optimize_portfolio.py` | LP/MILP allocation |
| 13 | `uv run python scripts/optimize_portfolio_tradeoff.py` | Robustness frontier |

## Optional: Platform Layer (dbt + Feast)

```bash
# WARNING: dev and platform extras conflict (pyarrow versions).
# Use a separate venv or switch extras.
uv sync --extra platform

# dbt
cd dbt_project
uv run dbt run --target duckdb
uv run dbt test
uv run dbt docs generate
cd ..

# Feast
cd feature_repo
uv run feast apply
cd ..
```

## Optional: Docker Compose

```bash
# Prerequisite: pipeline must have run first (data/processed/ populated)
docker compose up --build
# API: http://localhost:8000
# Streamlit: http://localhost:8501
```

## Optional: API Only

```bash
uv run uvicorn api.main:app --reload --port 8000
```

## Optional: Free Public Streamlit Deploy Bundle

Build a lightweight folder for Streamlit Community Cloud:

```bash
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Then follow `docs/DEPLOY_STREAMLIT_FREE.md` to publish the generated `dist/streamlit_deploy/` bundle.

## Integrations (DVC + MLflow + DagsHub)

For full setup details, see `docs/INTEGRATIONS_SETUP.md`.

### Setup rápido de integraciones

```bash
# DagsHub-first (recomendado)
bash scripts/configure_integrations.sh
```

### DVC Pipeline

```bash
# Reproduce the full pipeline (incremental — only re-runs changed stages)
uv run dvc repro

# View the DAG
uv run dvc dag

# Push artifacts to DagsHub remote
uv run dvc push -r dagshub
```

`dvc repro` is equivalent to running `scripts/end_to_end_pipeline.py` but with automatic caching and incremental execution.

### MLflow Experiment Logging

```bash
# Log all 8 experiments from existing artifacts to DagsHub MLflow
uv run python scripts/log_mlflow_experiment_suite.py
```

Experiments logged: `end_to_end`, `pd_model`, `conformal`, `causal_policy`, `ifrs9`, `optimization`, `survival`, `time_series`.

### DagsHub

- **Git mirror**: `git remote add dagshub https://dagshub.com/<user>/<repo>.git`
- **DVC remote**: configured in `.dvc/config`
- **MLflow UI**: accessible at `https://dagshub.com/<user>/<repo>/experiments`
- **Environment**: copy `.env.example` → `.env` and fill in tokens

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv sync` recreates venv | Ensure `VIRTUAL_ENV` doesn't have Windows interop prefix |
| `mapie` import errors | Verify `mapie>=1.3.0` installed (not 0.9.x) |
| `feast` + `pyarrow` conflict | Use separate venv for platform extras |
| Missing parquet files | Run `scripts/end_to_end_pipeline.py` first |
| DuckDB file not found | Run dbt or let Streamlit create it on first access |
| Tests fail on import | Run `uv sync --extra dev` to install test dependencies |

## Environment Notes

- Python 3.11.x on WSL2 (tested)
- `uv` at `~/.local/bin/uv`
- Venv at `.venv/bin/python`
- Pre-commit hooks: `uv run pre-commit install`
