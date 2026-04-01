# Lending Club End-to-End Risk Intelligence Platform

Credit risk thesis platform organized around pipeline-first execution families:
`core_canonical`, `search_*`, `paper*_e2e`, `diagnostics_governance`, and `research_labs`.

[![CI](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml/badge.svg)](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml)
[![Historical Showcase](https://img.shields.io/badge/Historical%20Showcase-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://lending-club-showcase.streamlit.app/)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-00A86B)](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Official Surfaces

- Quarto book is the official, citable, up-to-date source of truth for the project.
- The local Streamlit app is an optional companion lab for interaction that is not worth duplicating in Quarto.
- The public Streamlit showcase is a historical snapshot and is no longer the primary product surface.

Historical public showcase:

`https://lending-club-showcase.streamlit.app/`

## Project Scope

This repository is built as a reproducible, research-to-production-style workflow over Lending Club historical loans, with Quarto as the official editorial layer and Streamlit as a reduced local companion for interactive analysis.

Core methodological chain:

```text
Monotonic CatBoost PD -> Auto Calibration (Platt/Isotonic/Venn-Abers) -> Fairness on Approval Decisions
-> Mondrian Conformal Intervals -> Governance / MRM Diagnostics
-> IFRS9 Scenario Sensitivity -> Robust Portfolio Optimization
```

## Why It Matters

1. Point estimates alone are insufficient for risk-sensitive decisions.
2. Conformal prediction introduces finite-sample uncertainty quantification.
3. Robust optimization converts uncertainty into actionable portfolio constraints.
4. IFRS9 sensitivity links predictive risk to accounting impact.
5. Governance layers like fairness, C2ST, monotonicity audits, and PD validation interpretation make the stack defendable rather than merely predictive.
6. Causal policy analysis remains a research-grade intervention lane beyond correlation.

## Architecture (Quarto-First)

| Layer | Role |
|---|---|
| Quarto | Official narrative, figures, tables, and defendable results |
| Streamlit | Local companion lab for interaction, simulation, and exploratory slicing |
| DuckDB | Local analytical engine for queries and derived marts |
| dbt | Data lineage/tests/docs over analytical models |
| Feast | Feature-store consistency narrative |
| FastAPI | Optional service layer for API-style consumption |
| DVC + DagsHub | Artifact versioning and remote synchronization |
| MLflow (DagsHub) | Experiment tracking suite (`lending_club/*`) |

![End-to-end credit risk architecture](docs/assets/architecture/lending-club-architecture-e2e.jpg)

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
# 1) Install dependencies in the canonical env
export UV_PROJECT_ENVIRONMENT=lending-club-venv
uv sync --extra dev
test -e .venv || ln -s lending-club-venv .venv

# 2) Build the dedicated causal env once
bash scripts/causal/setup_causal_env.sh .venv-causal

# 3) Place Kaggle CSV in data/raw/
# Loan_status_2007-2020Q3.csv

# 4) Run the canonical full pipeline (incremental, DVC-managed)
uv run dvc repro

# 5) Run the frozen operational rebuild
uv run python scripts/core/run_canonical_rebuild.py --run-tag canonical-local-smoke

# 6) Run the focused PD search lane when needed
uv run python scripts/search/run_pd_search.py --run-tag champion-local-max --sampling-profile mega64plus

# 7) Run research-only complementary insight generation
uv run python scripts/labs/run_research_labs.py --run-tag insights-local --profile canonical

# 8) Run local companion lab
uv run streamlit run streamlit_app/app.py
```

Notes:
- `uv run dvc repro` is the canonical thesis-grade rebuild path.
- `uv run python scripts/run_smoke_pipeline.py` is the lightweight smoke pipeline.
- `scripts/end_to_end_pipeline.py` and `scripts/run_long_pipeline.py` remain as compatibility entrypoints only.
- The organized wrappers under `scripts/core`, `scripts/search`, `scripts/papers`, `scripts/diagnostics`, and `scripts/labs` are the preferred public entrypoints.
- `bash scripts/causal/run_causal_pipeline.sh --treatment int_rate` is the canonical standalone causal runner when you only need the causal layer.

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

For a heavyweight champion search run with resumability and run tagging, use:

```bash
bash scripts/start_long_run.sh <run_tag> --comparison-baseline-run-tag <baseline_run_tag>
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

## Historical Streamlit Showcase

The public Streamlit showcase is intentionally frozen as a historical snapshot. The local Streamlit app should evolve independently from that deploy target.

If you need to rebuild the historical showcase bundle anyway:

```bash
uv run python scripts/export_streamlit_artifacts.py
uv run python scripts/prepare_streamlit_deploy.py --clean --strict
```

Detailed guide:

`docs/history/DEPLOY_STREAMLIT_FREE.md`

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
2. `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md` - canonical editorial ledger for live techniques, claims, and Quarto mapping
3. `docs/RUNBOOK.md` - end-to-end reproducibility runbook
4. `docs/INTEGRATIONS_SETUP.md` - GitHub/DagsHub/DVC/MLflow setup
5. `docs/PROJECT_JUSTIFICATION.md` - methodological rationale
6. `docs/QUARTO_BOOK_BLUEPRINT.md` - Quarto-first editorial contract for the book
7. `docs/DOCUMENTATION_MAP.md` - active vs historical documentation map

## License

MIT
