<h1 align="center">Lending Club Risk Intelligence Platform</h1>
<p align="center">
  Credit risk thesis platform combining machine learning, conformal uncertainty, IFRS9 analytics, and robust portfolio optimization.
</p>

<p align="center">
  <a href="https://github.com/EigenCharlie/Lending-Club-End-to-End"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-0A0A0A?logo=github"></a>
  <a href="https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End"><img alt="DagsHub" src="https://img.shields.io/badge/DagsHub-MLops-00A86B"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Multipage-red">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

## Executive Summary

This repository delivers an end-to-end thesis workflow for uncertainty-aware credit decisions over the Lending Club historical portfolio.

Core thesis chain:

```text
CatBoost PD -> Platt Calibration -> MAPIE Mondrian Conformal
-> Uncertainty-aware constraints -> Pyomo (HiGHS) optimization
-> Robust portfolio and IFRS9 reserve analysis
```

The project is designed as a **Streamlit-first showcase platform** with strong reproducibility and governance.

## Why This Project Matters

- Point prediction alone is not enough for lending decisions.
- Conformal intervals add finite-sample uncertainty guarantees.
- Robust optimization quantifies the economic price of conservatism.
- IFRS9 scenarios connect model output to accounting impact.

## Platform Highlights

- Multi-page Streamlit storytelling app (`streamlit_app/`)
- Full analytical pipeline scripts (`scripts/`)
- Reusable production-style modules (`src/`)
- Optional service layer via FastAPI (`api/`)
- Data governance with DuckDB + dbt + Feast (`dbt_project/`, `feature_repo/`)
- CI checks for lint, tests, and Streamlit page smoke imports (`.github/workflows/ci.yml`)

## Verified Snapshot

- Notebooks `01`-`09` executed successfully.
- Pipeline outputs are artifact-consistent in `data/processed/` and `models/`.
- Conformal policy gate and causal policy selection available.
- Current metrics and roadmap: `SESSION_STATE.md`.

## Architecture (Thesis Mode)

| Layer | Role |
|---|---|
| Streamlit | Primary user experience and thesis narrative |
| DuckDB | Local analytical query engine |
| dbt | Data lineage, tests, docs |
| Feast | Feature-store consistency showcase |
| FastAPI + MCP | Optional integration services |

## Repository Layout

```text
api/                 FastAPI endpoints
configs/             YAML runtime configuration
data/                Raw/interim/processed assets (artifact-oriented)
dbt_project/         Governance SQL models and tests
docs/                Runbooks, architecture, and thesis plans
feature_repo/        Feast entities, views, services
models/              Serialized model artifacts and contracts
notebooks/           Thesis notebooks (01-09 + side projects)
reports/             Audits, figures, notebook image exports
scripts/             Executable stage-by-stage pipeline
src/                 Core reusable package modules
streamlit_app/       Main dashboard (multipage)
tests/               Automated validation suite
```

## Quick Start

```bash
# 1) Install dependencies
uv sync --extra dev

# 2) Place Kaggle CSV in data/raw/
# Loan_status_2007-2020Q3.csv

# 3) Run full pipeline
uv run python scripts/end_to_end_pipeline.py

# 4) Export Streamlit-ready artifacts
uv run python scripts/export_streamlit_artifacts.py

# 5) Launch app
uv run streamlit run streamlit_app/app.py
```

## Free Public Deploy (Streamlit Community Cloud)

```bash
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Deployment guide: `docs/DEPLOY_STREAMLIT_FREE.md`

## Integrations (GitHub + DVC + DagsHub)

One-shot setup script:

```bash
bash scripts/configure_integrations.sh
```

Integration guide: `docs/INTEGRATIONS_SETUP.md`

MLflow backfill for full thesis experiment suite:

```bash
set -a && source .env && set +a
uv run python scripts/log_mlflow_experiment_suite.py
```

## Optional Services

```bash
# API only
uv run uvicorn api.main:app --reload --port 8000

# Full docker stack
docker compose up --build
```

## Key Documents

- `SESSION_STATE.md` - canonical project status and metrics
- `docs/RUNBOOK.md` - reproducibility runbook
- `docs/PROJECT_JUSTIFICATION.md` - design and method rationale
- `docs/THESIS_SHOWCASE_PLAN_ES.md` - showcase execution plan
- `reports/project_audit_snapshot.json` - artifact validation snapshot

## License

MIT
