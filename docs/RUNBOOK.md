# Reproducibility Runbook

Step-by-step guide to reproduce the entire project from a fresh clone.

## Prerequisites

- **Python 3.12** recommended (project supports `>=3.11,<3.13`, current default `.python-version` is `3.12`)
- **uv** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Kaggle dataset**: Download manually from https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data and place CSV in `data/raw/`
- **GPU side-projects (RAPIDS)**: run in Conda env `rapids` (keep `lending-club-venv` for the core pipeline)
- **Causal ML with EconML**: run in a separate venv (kept out of main lock to avoid pinning `scikit-learn`/`shap`)

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd Lending-Club-End-to-End

# 2. Install dependencies in the canonical main env
export UV_PROJECT_ENVIRONMENT=lending-club-venv
uv sync --extra dev
test -e .venv || ln -s lending-club-venv .venv

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
| 4 | `uv run python scripts/train_pd_model.py` | CatBoost model + calibrator candidates (Platt/Isotonic/Venn-Abers) + `models/decision_threshold.json` |
| 5 | `uv run python scripts/generate_conformal_intervals.py` | Mondrian conformal intervals |
| 6 | `uv run python scripts/backtest_conformal_coverage.py` | Temporal monitoring |
| 7 | `uv run python scripts/validate_conformal_policy.py` | Policy gate + Winkler + Kupiec/Christoffersen (`conformal_policy_status.json`) |
| 8 | `uv run python scripts/build_pd_challenger_artifacts.py --config configs/pd_model.yaml` | Challenger feature selection + monotonic constraints spec |
| 9 | `uv run python scripts/run_fairness_audit.py --config configs/fairness_policy.yaml` | Fairness gate using threshold artifact (`fairness_audit_status.json`) |
| 10 | `uv run python scripts/estimate_causal_effects.py` | CATE estimates |
| 11 | `uv run python scripts/simulate_causal_policy.py` | Policy simulation |
| 12 | `uv run python scripts/validate_causal_policy.py` | Rule selection + bootstrap |
| 13 | `uv run python scripts/run_ifrs9_sensitivity.py` | ECL scenarios |
| 14 | `uv run python scripts/optimize_portfolio.py` | LP/MILP allocation |
| 15 | `uv run python scripts/optimize_portfolio_tradeoff.py` | Robustness frontier |

Compatibility note:
- Canonical status artifacts are single-write (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`).
- Conformal intervals use canonical output only: `data/processed/conformal_intervals_mondrian.parquet`.

## Conformal Promotion Gate (2026-02-27)

- `scripts/validate_conformal_policy.py` remains **strict** for model-risk policy (`overall_pass` still includes Kupiec/Christoffersen).
- `scripts/run_comparison.py` is the **promotion gate** and now treats Kupiec/Christoffersen as non-blocking diagnostics.
- `scripts/run_comparison.py` now includes a blocking `artifact_coherence` gate:
  critical status artifacts must carry consistent `schema_version`, `generated_at_utc`, and `run_tag`.
- Blocking conformal checks in promotion are: `coverage_90`, `coverage_95`, `min_group_coverage_90`, `winkler_90`, `critical_alerts`.
- Comparison artifacts now include `conformal_promotion_pass` and `conformal_statistical_warning` in `reports/run_comparisons/<run_tag>/comparison.json`.

## Official Rerun Profile (Core)

Use this profile for official reruns that should be stable and resumable on workstation resources:

```bash
bash scripts/start_long_run.sh <run_tag> \
  --comparison-baseline-run-tag <baseline_run_tag> \
  --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh <run_tag>
```

Notes:
- Official run tags (`*official*`) require a clean git working tree.
- Core/official tags require baseline. If no baseline flag is passed, launcher resolves default from `configs/baselines/core_official_baseline.json`.
- Launcher defaults are `--resume`, `--sampling-profile full`, and baseline snapshot refresh on resume.

Official baseline freeze workflow:

```bash
uv run python scripts/freeze_core_baseline.py \
  --run-tag 2026-03-04-C-core-balanced-cert2 \
  --refresh-snapshot \
  --set-current
```

To resume an interrupted run:

```bash
bash scripts/start_long_run.sh <run_tag> --resume \
  --comparison-baseline-run-tag <baseline_run_tag> \
  --no-rapids --no-notebooks --stop-on-optional-failure
```

## Optional: Platform Layer (dbt + Feast)

```bash
# dbt runs in the project venv (optional extra).
uv sync --extra platform

# dbt
cd dbt_project
uv run dbt run --target duckdb
uv run dbt test
uv run dbt docs generate
cd ..

# Feast runs in a separate venv to avoid pinning uvicorn in the main lock.
uv venv .venv-feast
uv pip install --python .venv-feast/bin/python -r requirements/feast-platform.txt

# Feast
cd feature_repo
../.venv-feast/bin/feast apply
cd ..

# EconML (causal workflows) in separate env to avoid blocking sklearn/shap upgrades
bash scripts/causal/setup_causal_env.sh .venv-causal
```

## Optional: Causal ML (DoWhy + EconML)

```bash
# Build/update dedicated causal env (project stack + EconML overlay)
bash scripts/causal/setup_causal_env.sh .venv-causal

# Example
./.venv-causal/bin/python scripts/estimate_causal_effects.py
```

Note:
- `.venv-causal` is a task-specific overlay env for causal workflows. `econml` may downgrade `scikit-learn`/`shap`, so keep using `lending-club-venv` for the rest of the project (PD/survival/API/Streamlit).

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
uv run python scripts/export_streamlit_artifacts.py
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Then follow `docs/DEPLOY_STREAMLIT_FREE.md` to publish the generated `dist/streamlit_deploy/` bundle.

## Integrations (DVC + MLflow + DagsHub)

For full setup details, see `docs/INTEGRATIONS_SETUP.md`.

### Setup rápido de integraciones

```bash
# DagsHub-first (recomendado, S3-compatible por defecto)
bash scripts/configure_integrations.sh
```

Notas rápidas:

- El script configura DVC en DagsHub con backend `s3` por defecto (evita `413` en artefactos grandes).
- Usa `DVC_REMOTE_BACKEND=http` si necesitas compatibilidad legacy temporal.
- Instala hooks `pre-commit` / `pre-push` automáticamente si encuentra `.pre-commit-config.yaml`.

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

### DVC Metrics / Plots (comparables por commit)

```bash
# Refresh canonical KPI summary + plot CSVs
uv run dvc repro export_dvc_metrics

# Show current KPI snapshot
uv run dvc metrics show

# Compare metrics vs previous commit/branch
uv run dvc metrics diff

# Visualize canonical plots (local browser)
uv run dvc plots show
```

### MLflow Experiment Logging

```bash
# Log all 8 experiments from existing artifacts to DagsHub MLflow
uv run python scripts/log_mlflow_experiment_suite.py
```

Experiments logged: `end_to_end`, `pd_model`, `conformal`, `causal_policy`, `ifrs9`, `optimization`, `survival`, `time_series`.

### DagsHub

- **Git mirror**: `git remote add dagshub https://dagshub.com/<user>/<repo>.git`
- **DVC remote**: configured in `.dvc/config` (project) + `.dvc/config.local` (credentials/local backend override)
- **MLflow UI**: accessible at `https://dagshub.com/<user>/<repo>/experiments`
- **Environment**: copy `.env.example` → `.env` and fill in tokens
- **Onboarding checkbox** ("Version your data with our client"): optional/cosmético; no bloquea DVC/MLflow

Optional onboarding/UI bootstrap (one-time):

```bash
DAGSHUB_CLIENT_BOOTSTRAP=1 bash scripts/configure_integrations.sh
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv sync` recreates venv | Ensure `VIRTUAL_ENV` doesn't have Windows interop prefix and export `UV_PROJECT_ENVIRONMENT=lending-club-venv` |
| `dvc push` fails with `413 Request Entity Too Large` | Use DagsHub S3-compatible remote (`DVC_REMOTE_BACKEND=s3`) and ensure `dvc[s3]` is installed |
| `mapie` import errors | Verify `mapie>=1.3.0` installed (not 0.9.x) |
| `feast` + `pyarrow` conflict | Use separate venv for platform extras |
| Missing parquet files | Run `scripts/end_to_end_pipeline.py` first |
| DuckDB file not found | Run dbt or let Streamlit create it on first access |
| Tests fail on import | Run `UV_PROJECT_ENVIRONMENT=lending-club-venv uv sync --extra dev` to install test dependencies |

## Environment Notes

- Python 3.12.x on WSL2 (tested)
- `uv` at `~/.local/bin/uv`
- Main venv at `lending-club-venv/bin/python` (uv-managed, backed by miniforge3 Python 3.12)
- Compatibility alias kept: `.venv -> lending-club-venv`
- Pre-commit hooks: `uv run pre-commit install`

## GitHub Governance (recommended minimal settings)

Set these in GitHub UI (Rulesets / Branch protection for `main`):

- Require pull request before merge
- Require status checks: `lint`, `config-contracts`, `test`, `streamlit-smoke`
- Require conversation resolution
- Block force pushes to `main`

Optional for collaboration:

- `CODEOWNERS` is available at `.github/CODEOWNERS`

## DVC Experiments (optional, useful for config iteration)

Use `dvc exp` when you want quick, reproducible comparisons of config changes without early commits:

```bash
uv run dvc exp run
uv run dvc exp show
uv run dvc exp diff
uv run dvc exp apply <exp-name>
```
