# Lending Club End-to-End Risk Intelligence Platform

Credit risk thesis platform organized around pipeline-first execution families:
`core_canonical`, `search_*`, `paper*_e2e`, `diagnostics_governance`, and `research_labs`.

[![CI](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml/badge.svg)](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml)
[![Historical Showcase](https://img.shields.io/badge/Historical%20Showcase-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://lending-club-showcase.streamlit.app/)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-00A86B)](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Official Surfaces

- Quarto is the editorial source of truth for this repository's historical
  pipeline, diagnostics, Paper 2 and Paper 4 claim boundaries.
- The current CRPTO/IJDS paper is maintained independently in
  `Paper_CRPTO`; it is a retrospective identification audit and selects no
  learner or portfolio policy. Local artifacts named `champion` are provenance
  from an earlier root-repository freeze, not current CRPTO claims. The observed
  external commit and content hashes are pinned in
  `docs/research/crpto_external_contract_2026-07-20.yml`.
- The local Streamlit app is an optional companion lab for interaction that is not worth duplicating in Quarto.
- The public Streamlit showcase is a historical snapshot and is no longer the primary product surface.

Historical public showcase:

`https://lending-club-showcase.streamlit.app/`

## Project Scope

This repository is built as a reproducible historical research workflow over
Lending Club loans, with Quarto as the official editorial layer and Streamlit
as a reduced local companion for interactive analysis. Executable services and
runtime labels do not establish production readiness.

Core methodological chain:

```text
Monotonic CatBoost score -> Local Calibration Selection -> Historical approval-proxy fairness checks
-> Binary-outcome conformal intervals -> Governance / MRM diagnostics
-> Historical IFRS9-inspired and portfolio stress experiments
```

When calibration labels are binary, the conformal endpoints cover the future
binary outcome under the stated assumptions. They are not confidence bounds
for latent individual PD, and their use as ECL inputs or PD uncertainty boxes
is a historical stress heuristic without a transported coverage guarantee.

## Why It Matters

1. Point estimates alone are insufficient for risk-sensitive decisions.
2. Conformal prediction supplies finite-sample coverage for a declared target
   under exchangeability; the target and transport assumptions must remain explicit.
3. Historical portfolio experiments show how endpoint geometry can affect
   decisions, but do not establish a selected policy or latent-PD uncertainty set.
4. The IFRS9-inspired lane is diagnostic and parked pending reporting-date data
   and an estimand-specific uncertainty method.
5. Governance layers like fairness, C2ST, monotonicity audits, and score-validation interpretation make internal limitations and provenance auditable; they are not independent validation.
6. The causal-methods lane remains an observational diagnostic and does not identify policy value.

## Historical Local Freeze (pre-identification CRPTO, 2026-04-06)

The following table records the root repository's frozen historical local state. It
must not be cited as the current Paper_CRPTO result or as a validated deployment
policy.

| Component | Status |
|---|---|
| PD model | Monotonic CatBoost (HPO local trial 56) + Venn-Abers calibrator — AUC 0.7124, Brier 0.1546, ECE 0.0064 |
| Conformal | `score_decile_mondrian` (rank1 reopen) — coverage 92.97%, Winkler 1.111 |
| Portfolio | `bound_aware_276k_economic_champion` — `rt=0.175, gamma=0.45, blended_uncertainty`, return $170,464.54 |
| Robust region | 45/45 policies pass `alpha=0.01` exactly on full 276K OOT |
| Run tag | `paper-thesis-final-economic-2026-04-06` |
| Runtime provenance | `models/final_project_promotion.json`, `models/champion_portfolio_policy.json` |
| Audit dossier | `docs/research/paper_estrella_audit_2026-05-04.md` |

The `theorem-tight` (gamma=0.55) and `balanced` (rt=0.17) policies are historical
internal comparators. The active external CRPTO audit does not promote any of
these policies.

## Architecture (Quarto-First)

| Layer | Role |
|---|---|
| Quarto | Official local narrative, figures, tables, and bounded claim status |
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

# 5) Reproduce the frozen historical local rebuild
# The default core profile loads clean_baseline_manifest.json and preserves
# its explicit Venn--Abers midpoint_legacy point rule.
uv run python scripts/run_canonical_rebuild.py --run-tag canonical-local-smoke

# 6) Run the focused PD search lane when needed
uv run python scripts/search/run_pd_search.py --run-tag champion-local-max --sampling-profile mega64plus

# 7) Run research-only complementary insight generation
uv run python scripts/labs/run_research_labs.py --run-tag insights-local --profile canonical

# 8) Run local companion lab
uv run streamlit run streamlit_app/app.py
```

Notes:
- `uv run dvc repro` is the fresh thesis-grade rebuild path. It refits current
  code and therefore uses the new Venn--Abers `log_loss_minimax` rule; it is not
  byte-for-byte replay of the April artifacts.
- `scripts/run_canonical_rebuild.py` is the historical replay path. Its default
  profile freezes `clean_baseline_manifest.json`, including
  `midpoint_legacy`, so old predictions do not change silently.
- `uv run python scripts/run_smoke_pipeline.py` is the lightweight smoke pipeline.
- `scripts/end_to_end_pipeline.py` and `scripts/run_long_pipeline.py` remain as compatibility entrypoints only.
- The preferred public entrypoints live under `scripts/`, `scripts/search`, `scripts/papers`, `scripts/diagnostics`, and `scripts/labs`.
- `bash scripts/causal/run_causal_pipeline.sh --treatment int_rate` is the canonical standalone causal runner when you only need the causal layer.
- `uv run python scripts/run_time_series_vnext.py --config configs/time_series_vnext.yaml` is the research-only TS/IFRS9 redesign lane; it must not overwrite canonical `time_series` artifacts.

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

## Time Series Status

The executable time-series lane is retained as a historical vintage diagnostic.
Its point and interval outputs are not validated as calendar-time forecasts or
as an IFRS9 overlay.

- Canonical lane: `scripts/forecast_default_rates.py` -> `models/time_series_status.json`
- Research redesign lane: `scripts/run_time_series_vnext.py` -> `models/time_series_vnext_status.json`
- Current scientific decision: keep all TS->IFRS9/ECL transformations as diagnostic research; Paper 2 remains `parked_ifrs9`, and neither the point nor interval layer is promoted

The latest decision package for this redesign lives in:

- `models/time_series_vnext_status.json`
- `models/time_series_policy_review.json`
- `docs/TIME_SERIES_VNEXT_DECISION_2026-04-02.md`

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
2. `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md` - superseded March--April runtime ledger retained for historical traceability, not current scientific claims
3. `docs/RUNBOOK.md` - end-to-end reproducibility runbook
4. `docs/INTEGRATIONS_SETUP.md` - GitHub/DagsHub/DVC/MLflow setup
5. `docs/PROJECT_JUSTIFICATION.md` - methodological rationale
6. `docs/QUARTO_BOOK_BLUEPRINT.md` - Quarto-first editorial contract for the book
7. `docs/DOCUMENTATION_MAP.md` - active vs historical documentation map

## License

MIT
