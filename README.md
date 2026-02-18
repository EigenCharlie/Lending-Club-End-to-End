# Lending Club End-to-End Risk Intelligence Platform

Credit risk thesis platform that integrates machine learning, conformal uncertainty, IFRS9 scenario analytics, causal policy evaluation, and robust portfolio optimization.

[![CI](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml/badge.svg)](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml)
[![Live Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://lending-club-showcase.streamlit.app/)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-00A86B)](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Live Demo

Public Streamlit showcase:

`https://lending-club-showcase.streamlit.app/`

## Project Scope

This repository is built as a reproducible, research-to-production-style workflow over Lending Club historical loans, with a Streamlit-first delivery layer for thesis defense and stakeholder communication.

Core methodological chain:

```text
CatBoost PD -> Platt Calibration -> Mondrian Conformal Intervals
-> Causal Policy Simulation -> IFRS9 Scenario Sensitivity
-> Robust Portfolio Optimization
```

## Why It Matters

1. Point estimates alone are insufficient for risk-sensitive decisions.
2. Conformal prediction introduces finite-sample uncertainty quantification.
3. Robust optimization converts uncertainty into actionable portfolio constraints.
4. IFRS9 sensitivity links predictive risk to accounting impact.
5. Causal policy analysis goes beyond correlation for intervention design.

## Architecture (Thesis Mode)

| Layer | Role |
|---|---|
| Streamlit | Primary UX and storytelling |
| DuckDB | Local analytical engine for queries and derived marts |
| dbt | Data lineage/tests/docs over analytical models |
| Feast | Feature-store consistency narrative |
| FastAPI | Optional service layer for API-style consumption |
| DVC + DagsHub | Artifact versioning and remote synchronization |
| MLflow (DagsHub) | Experiment tracking suite (`lending_club/*`) |

## Repository Map

```text
api/                 FastAPI endpoints
configs/             YAML runtime configuration
data/                Raw/interim/processed assets
dbt_project/         dbt models/tests/docs artifacts
docs/                Runbooks, architecture, and thesis notes
feature_repo/        Feast entities/views/services
models/              Serialized models and policy artifacts
notebooks/           Research notebooks (01-09 + side projects)
reports/             Audits and notebook image exports
scripts/             Pipeline stages and orchestration scripts
src/                 Reusable package modules
streamlit_app/       Multipage Streamlit application
tests/               Automated test suite
```

## Quick Start

```bash
# 1) Install dependencies
uv sync --extra dev

# 2) Place Kaggle CSV in data/raw/
# Loan_status_2007-2020Q3.csv

# 3) Run pipeline (raw -> artifacts)
uv run python scripts/end_to_end_pipeline.py

# 4) Export Streamlit-ready artifacts
uv run python scripts/export_streamlit_artifacts.py

# 5) Run app locally
uv run streamlit run streamlit_app/app.py
```

## Reproducibility and MLOps

```bash
# DVC pipeline graph
uv run dvc dag

# Check local/cloud consistency
uv run dvc status --json
uv run dvc status -c --json

# Push artifacts to DagsHub remote
uv run dvc push -r dagshub
```

One-shot integrations setup:

```bash
bash scripts/configure_integrations.sh
```

MLflow backfill from existing artifacts:

```bash
set -a && source .env && set +a
uv run python scripts/log_mlflow_experiment_suite.py
```

## Deploy Streamlit for Free

Build a deploy bundle optimized for Streamlit Community Cloud:

```bash
uv run python scripts/export_streamlit_artifacts.py
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Detailed guide:

`docs/DEPLOY_STREAMLIT_FREE.md`

## Quality Gates

```bash
uv run ruff check src/ scripts/ tests/
uv run ruff format --check src/ scripts/ tests/
uv run pytest -q
```

CI workflow:

`.github/workflows/ci.yml`

## Key Documents

1. `SESSION_STATE.md` - canonical status, snapshots, recovery logs
2. `docs/RUNBOOK.md` - end-to-end reproducibility runbook
3. `docs/INTEGRATIONS_SETUP.md` - GitHub/DagsHub/DVC/MLflow setup
4. `docs/PROJECT_JUSTIFICATION.md` - methodological rationale
5. `docs/THESIS_SHOWCASE_PLAN_ES.md` - showcase execution plan

## License

MIT
