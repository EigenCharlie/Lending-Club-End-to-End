# Reproducibility Runbook

Step-by-step guide to reproduce the entire project from a fresh clone.

## Prerequisites

- **Python 3.12** recommended (project supports `>=3.11,<3.13`, current default `.python-version` is `3.12`)
- **uv** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Kaggle dataset**: Download manually from https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data and place CSV in `data/raw/`
- **GPU side-projects (RAPIDS)**: run in Conda env `rapids` (keep `lending-club-venv` for the core pipeline)
- **Causal ML with EconML/CausalPy**: run in a separate venv (kept out of main lock to avoid pinning `scikit-learn`/`shap` and heavy Bayesian dependencies)

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

# 4. Build/update dedicated causal env
bash scripts/causal/setup_causal_env.sh .venv-causal

# 5. Run the canonical full pipeline
uv run dvc repro

# 6. Verify tests pass
uv run pytest -x

# 7. Launch Streamlit dashboard
uv run streamlit run streamlit_app/app.py
```

Notes:
- Canonical full rebuild: `uv run dvc repro` or `scripts/run_canonical_rebuild.py` (`core_canonical`).
- PD/challenger search: `scripts/run_champion_search.py` (`search_pd` by default).
- Paper 2 IFRS9 search: `scripts/search/run_paper2_ifrs9_search.py` (`search_paper2_ifrs9`).
- Insight-factory complement: `scripts/run_insights_factory.py --profile canonical|research` (`research_labs`).
- `scripts/end_to_end_pipeline.py` and `scripts/run_long_pipeline.py` are compatibility entrypoints only.
- Canonical standalone causal runner: `bash scripts/causal/run_causal_pipeline.sh --treatment int_rate`.

## Final Paper / Thesis Closure (2026-04-05)

The live repo now carries two layers that must not be confused:

- **Canonical operational base**: the monotonic confirmatory stack `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`
- **Final paper/thesis promoted stack**: the same upstream PD/governance base plus the conformal reopen winner and the economic `portfolio_bound_aware` closure

Current final promoted portfolio policy:

- `risk_tolerance=0.175`
- `policy_mode=blended_uncertainty`
- `gamma=0.45`
- `uncertainty_aversion=0.10`

Source-of-truth artifacts:

- `models/final_project_promotion.json`
- `data/processed/final_project_summary.parquet`
- `models/champion_portfolio_policy.json`
- `models/champion_registry.json`

Interpretation rules:

- the monotonic canonical run remains the regulatory and operational upstream base;
- the conformal reopen winner remains `rank1_score_decile_raw_bins5_mgs100`, i.e. the project-level conformal winner is `score_decile_mondrian`;
- `grade Mondrian` remains necessary as the natural/interpretable baseline for business, IFRS9, and governance narratives, but it is **not** a co-winner;
- the final promoted paper/thesis portfolio champion is **not** the old monotonic economic policy;
- the `276k` full-OOT mini-grid established a **robust region** (`45/45` alpha01 passers), and the final promoted champion is now the economic selector inside that region, with the theorem-tight point retained only as a documented comparator.

## Step-by-Step Pipeline

If you want to run individual stages:

| Step | Command | Produces |
|------|---------|----------|
| 1 | `uv run python -c "from src.data.make_dataset import main; main()"` | `data/interim/lending_club_cleaned.parquet` |
| 2 | `uv run python -c "from src.data.prepare_dataset import main; main()"` | Train/calibration/test splits |
| 3 | `uv run python -c "from src.data.build_datasets import main; main()"` | loan_master, time_series, ead_dataset |
| 4 | `uv run python scripts/train_pd_model.py` | CatBoost model + calibrator candidates (Platt/Isotonic/Venn-Abers/Beta) + `models/decision_threshold.json` |
| 5 | `uv run python scripts/generate_conformal_intervals.py` | Mondrian conformal intervals |
| 6 | `uv run python scripts/backtest_conformal_coverage.py` | Temporal monitoring |
| 7 | `uv run python scripts/validate_conformal_policy.py` | Policy gate + Winkler + Kupiec/Christoffersen (`conformal_policy_status.json`) |
| 8 | `uv run python scripts/benchmark_conformal_variants.py` | Variant selector + `conformal_temporal_diagnostics.parquet` |
| 9 | `uv run python scripts/benchmark_pd_set_prediction.py` | Binary classification-set benchmark (`pd_set_prediction_status.json`) |
| 10 | `uv run python scripts/analyze_pd_rare_event_calibration.py` | Rare-event calibration audit (`pd_rare_event_calibration_status.json`) |
| 11 | `uv run python scripts/export_conformal_method_registry.py` | Canonical method/library registry (`conformal_method_registry.json`) |
| 12 | `uv run python scripts/build_pd_challenger_artifacts.py --config configs/pd_model.yaml` | Challenger feature selection + monotonic constraints spec |
| 13 | `uv run python scripts/run_fairness_audit.py --config configs/fairness_policy.yaml` | Fairness gate using threshold artifact (`fairness_audit_status.json`) |
| 14 | `bash scripts/causal/run_causal_pipeline.sh --treatment int_rate` | `estimate -> simulate -> validate -> backtest` in `.venv-causal` |
| 15 | `uv run python scripts/run_ifrs9_sensitivity.py` | ECL scenarios |
| 16 | `uv run python scripts/optimize_portfolio.py` | LP/MILP allocation |
| 17 | `uv run python scripts/optimize_portfolio_tradeoff.py` | Robustness frontier |

Compatibility note:
- Canonical status artifacts are single-write (`conformal_policy_status.json`, `fairness_audit_status.json`, `governance_status.json`).
- Conformal intervals use canonical output only: `data/processed/conformal_intervals_mondrian.parquet`.
- Method adoption is canonicalized in `models/conformal_method_registry.json`.
- Binary set prediction and rare-event calibration are sidecars, not champion gates:
  `models/pd_set_prediction_status.json`, `models/pd_rare_event_calibration_status.json`.
- Threshold semantics are canonicalized in `models/threshold_semantics.json`.
  `decision_threshold.json.selected_threshold` is threshold interno PD de screening/search;
  `fairness_audit_status.json.primary_threshold` and `fairness_decision_policy.json.global_threshold`
  are threshold operativo de aprobación/fairness.

## Conformal Promotion Gate (2026-02-27 / 2026-03-13)

- `scripts/validate_conformal_policy.py` remains **strict** for model-risk policy (`overall_pass` still includes Kupiec/Christoffersen).
- `models/conformal_policy_status.json` now also emits:
  `strict_overall_pass`, `non_statistical_checks_pass`, `methodological_justification_pass`,
  `failing_statistical_checks`, `failing_non_statistical_checks`, and `sample_size_context`.
- `methodological_justification_pass=true` is allowed only when:
  non-statistical checks pass, only statistical p-value checks fail, coverage deviations remain
  within configured materiality bounds, and Christoffersen independence subtests do not flag.
- `winkler_90` now uses a formal two-stage policy:
  strict target `<= 1.20`, or compensated-band pass `<= 1.22` only if
  `coverage_90 >= 0.92`, `min_group_coverage_90 >= 0.885`, `avg_width_90 <= 0.80`,
  and `critical_alerts == 0`.
- This compensated band is an explicit internal policy rule, not a library default and not a
  claim that `1.20` or `1.22` are universal literature cutoffs.
- `scripts/run_comparison.py` is the **promotion gate** and now treats Kupiec/Christoffersen as non-blocking diagnostics.
- `scripts/run_comparison.py` now includes a blocking `artifact_coherence` gate:
  critical status artifacts must carry consistent `schema_version`, `generated_at_utc`, and `run_tag`.
- Blocking conformal checks in promotion are: `coverage_90`, `coverage_95`, `min_group_coverage_90`, `winkler_90`, `critical_alerts`.
- Comparison artifacts now include `conformal_promotion_pass` and `conformal_statistical_warning` in `reports/run_comparisons/<run_tag>/comparison.json`.
- `comparison.json` now also includes `operational_overall_pass` (promotion gate using `methodological_justification_pass`) and `overall_pass` (strict, diagnostic only).
- Causal/CATE artifacts are exempted from `run_tag` coherence when they are the only mismatches and `insights_only=true` is in `champion_registry.json`.

## Threshold Semantics

Two separate thresholds coexist in this project. Confusing them leads to incorrect fairness or approval interpretations.

**`pd_internal_selected_threshold` = 0.05**
Artifact: `models/threshold_semantics.json`, `models/decision_threshold.json`.
Internal screening/search — selects candidates for portfolio optimization. NOT the approval threshold.

**`fairness_primary_threshold` / `decision_policy_global_threshold` = 0.35**
Artifact: `models/fairness_decision_policy.json`, `models/threshold_semantics.json`.
Operational approval/fairness threshold. Used in fairness audit and business narratives.

Rules:

- Fairness metrics (DPD, EOD) must always be computed at the **operational threshold (0.35)**, not the internal threshold.
- The internal threshold is only for candidate universe generation in portfolio optimization.
- `models/threshold_semantics.json` is the canonical source of truth for both values.

## LGD/EAD Conformal (Promoted 2026-03-16)

- Promoted variant: `direct_adaptive_grade_temporal`
- Coverage 90%: 0.9052, Min grade coverage: 0.9047, Avg width: 0.4951, Bias: -0.090
- Artifact: `models/conformal_lgd_ead_status.json` (`promoted: true`)
- Other variants: all fail `overall_pass` due to coverage_90 < target or min_grade_coverage < threshold.

## Official Rerun Profile (Core)

Use this profile for frozen operational reruns that should be stable and resumable on workstation resources:

```bash
bash scripts/start_long_run.sh canonical-<run_tag> \
  --comparison-baseline-run-tag <baseline_run_tag> \
  --no-rapids --no-notebooks --stop-on-optional-failure
bash scripts/monitor_long_run.sh <run_tag>
```

Notes:
- Official / canonical / champion tags require baseline. If no baseline flag is passed, launcher resolves default from `configs/baselines/canonical_operational_baseline.json` and then falls back to the legacy core registry.
- `reports/history/project_audit_snapshot.json` is historical context only; do not treat it as the live baseline snapshot.
- Launcher defaults are `--resume`, `--sampling-profile full`, and baseline snapshot refresh on resume.

Official baseline freeze workflow:

```bash
uv run python scripts/freeze_core_baseline.py \
  --run-tag 2026-03-04-C-core-balanced-cert2 \
  --refresh-snapshot \
  --set-current
```

Notebook execution policy (non-destructive):

- `scripts/run_all_notebooks.py` runs notebooks in reference mode.
- Writes targeting canonical outputs (`data/processed`, `models`, `reports/paper_material`, `reports/figures`) are redirected to `reports/notebook_exec/generated/`.
- Keep `--inplace false` for long runs; source notebooks remain as code references.
- `scripts/extract_notebook_images.py` reads executed notebooks from `reports/notebook_exec/notebooks` by default.

To resume an interrupted run:

```bash
bash scripts/start_long_run.sh <run_tag> --resume \
  --comparison-baseline-run-tag <baseline_run_tag> \
  --no-rapids --no-notebooks --stop-on-optional-failure
```

## Exhaustive Search Wave 2026-04

Pipeline-first exhaustive search lanes now split into:
- `search_pd` for blockwise / constrained-threshold monotonic challenger work
- `search_paper2_ifrs9` for survival / PoC / IFRS9 search
- `search_conformal` for strictness-oriented Mondrian / CQR tuning

Suggested launch commands:

```bash
uv run python scripts/search/run_pd_search.py \
  --run-tag pd-blockwise-exhaustive-2026-04 \
  --pipeline-profile search_pd_blockwise_exhaustive \
  --sampling-profile mega64plus \
  --upstream-canonical-run-tag canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129 \
  --env-file .env --no-rapids --no-notebooks

uv run python scripts/search/run_paper2_ifrs9_search.py \
  --run-tag paper2-ifrs9-exhaustive-2026-04 \
  --pipeline-profile search_paper2_ifrs9_exhaustive \
  --sampling-profile mega64safe \
  --upstream-canonical-run-tag canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129 \
  --env-file .env --no-rapids --no-notebooks

uv run python scripts/search/run_conformal_search.py \
  --run-tag conformal-strict-exhaustive-2026-04 \
  --pipeline-profile search_conformal_exhaustive \
  --sampling-profile champion64safe \
  --upstream-canonical-run-tag canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129 \
  --env-file .env --no-rapids --no-notebooks
```

Conformal reopen workflow over the refined PD candidate:

```bash
uv run python scripts/search/run_conformal_reopen_search.py \
  --run-tag conformal-reopen-2026-04 \
  --pipeline-profile search_conformal_reopen_exhaustive \
  --upstream-canonical-run-tag pd-hpo-local-2026-04-03-1325
```

This workflow performs:
- repeated inner search on calibration holdout only,
- single OOT confirmation under namespace,
- set-prediction sidecar benchmark,
- optional phase-2 calibrator search (`venn_abers`, `isotonic`, `platt`, `beta`) if phase 1 still fails policy.

Bound-aware portfolio closure on top of the conformal winner:

1. `5k` corrected shortlist run to prove an `alpha=0.01` exact passer exists;
2. `25k` hybrid run (GPU frontier + exact CPU) to confirm the region is not a subset artifact;
3. `276k` full-OOT mini-grid to certify the final region and promote the economic champion.

Important lesson:
- the failed wide `5k` refinement did **not** show that `alpha01-safe` was impossible;
- it exposed a shortlist bug where return/PoR ranking was done before exact bound ranking;
- current promotion artifacts already encode the corrected shortlist logic and the final economic closure.
- analogously, the conformal stack should not be narrated as having “two winners”:
  `grade` is the explanatory/regulatory baseline, while `score_decile_mondrian` is the single final winner promoted by objective benchmark/reopen criteria.

Monitoring:

```bash
uv run python scripts/monitor_pipeline_eta.py --run-tag <run_tag>
bash scripts/monitor_long_run.sh
```

Local HPO refinement on top of the best blockwise challenger:

```bash
uv run python scripts/search/run_pd_hpo_local.py \
  --run-tag pd-hpo-local-2026-04 \
  --base-search-run-tag pd-blockwise-exhaustive-2026-04-02-0940 \
  --hpo-n-trials 120 \
  --sampling-profile mega64plus \
  --upstream-canonical-run-tag canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129 \
  --env-file .env --no-rapids --no-notebooks
```

## Time Series Operational Semantics

The time-series lane is governed by a single status contract:
- canonical status artifact: `models/time_series_status.json`
- experimental / research status artifact: `models/time_series_research_status.json`
- vNext redesign research artifact: `models/time_series_vnext_status.json`
- vNext policy decision artifact: `models/time_series_policy_review.json`

End-to-end producer:
- `scripts/forecast_default_rates.py`
- `scripts/run_time_series_vnext.py` for the enriched redesign lane

Core outputs:
- `data/processed/ts_forecasts.parquet`
- `data/processed/ts_ifrs9_scenarios.parquet`
- `data/processed/ts_panel_forecasts.parquet`
- `models/time_series_status.json`

The canonical status separates:
- `point_champion`
- `interval_champion`
- `warnings`
- `final_interval_decision`

Operational meaning:
- `point_champion.promotable=true` means the point forecast can be treated as the official operational forecast.
- `interval_champion.promotable=true` means the interval layer is also officially validated.
- `final_interval_decision.status=research_only` means interval outputs still exist and are published, but they remain diagnostic/research evidence rather than a promoted interval policy.
- `final_interval_decision.status=promoted` means the interval layer passed the same governed selection logic and is part of the official forecast contract.

Artifact flow:

| Artifact | Produced by | Main consumers | If `research_only` | If `promoted` |
|----------|-------------|----------------|--------------------|---------------|
| `models/time_series_status.json` | `scripts/forecast_default_rates.py` | `export_storytelling_snapshot.py`, `build_champion_search_bundle.py`, `update_champion_registry.py`, Streamlit | Canonical point forecast stays official; interval layer is reported with warning | Point + interval layers become fully official |
| `data/processed/ts_forecasts.parquet` | `scripts/forecast_default_rates.py` | Streamlit, MLflow logging | Forecasts visible and usable, but interval interpretation stays diagnostic | Forecasts and their intervals are both official |
| `data/processed/ts_ifrs9_scenarios.parquet` | `scripts/forecast_default_rates.py` | `run_ifrs9_sensitivity.py`, Streamlit, MLflow logging | Scenarios remain useful, but interval-backed uncertainty is not promoted | Scenarios inherit an officially validated interval layer |
| `data/processed/ts_panel_forecasts.parquet` | `scripts/forecast_default_rates.py` | Streamlit deploy bundle, architecture views | Bottom-up/panel outputs remain analytical support | Bottom-up/panel outputs sit on top of a promoted TS contract |
| `models/time_series_research_status.json` | `scripts/forecast_default_rates.py --config configs/time_series_v2.yaml` | research review only | Experimental evidence without canonical overwrite | Not used once the canonical lane has already passed |

Current project interpretation:
- canonical point forecast is official
- canonical interval forecast is not promotable yet
- therefore the lane is operationally useful, but its interval layer remains `research_only`
- the `vNext` redesign lane was executed on `2026-04-02` and the decision was `maintain_canonical_keep_vnext_research`
- the enriched contract, `MAPIE_ENBPI` interval challenger, joint sample paths, and TS->ECL translation remain research evidence only unless a future rerun clears the same governed promotion logic

Promotion rule of thumb:
- do not promote because a challenger is more sophisticated
- promote only if the interval challenger beats the canonical lane under the governed policy and then updates `models/time_series_status.json`

Current keep / research / discard posture:

| Component | Current posture | Why |
|----------|------------------|-----|
| Canonical point forecast | keep | `AutoARIMA` remains operationally usable and the vNext redesign did not produce a material point improvement |
| Canonical IFRS9 temporal overlay | keep | still useful for scenario scaling even with interval caution |
| Canonical interval layer | keep as diagnostic only | still published, still not promotable |
| Enriched internal-only data contract | research only | useful for benchmarking, not yet justified as canonical replacement |
| `MAPIE_ENBPI` / adaptive interval challengers | research only | improved coverage gap but still failed governed promotion threshold |
| Joint sample paths (`gaussian_copula`, `schaake_shuffle`) | research only | add prudential insight for accumulated risk, but not yet an operational policy layer |
| vNext TS->ECL interval translation | keep as research support | numerically stable and useful for review, but not a canonical downstream dependency |

Reference closeout:
- `docs/TIME_SERIES_VNEXT_DECISION_2026-04-02.md`

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

# Canonical standalone causal chain
bash scripts/causal/run_causal_pipeline.sh --treatment int_rate

# Run a single causal script inside the dedicated env
bash scripts/causal/run_in_causal_env.sh scripts/estimate_causal_effects.py --treatment int_rate
```

Note:
- `.venv-causal` is a task-specific overlay env for causal workflows. `econml` may downgrade `scikit-learn`/`shap`, and `CausalPy` brings `pymc`-family dependencies, so keep using `lending-club-venv` for the rest of the project (PD/survival/API/Streamlit).

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

Then follow `docs/history/DEPLOY_STREAMLIT_FREE.md` only if you intentionally want to rebuild the frozen historical showcase bundle in `dist/streamlit_deploy/`.

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

`dvc repro` is the canonical full rebuild path with automatic caching and incremental execution.
`scripts/run_smoke_pipeline.py` is the core/minimal smoke pipeline, not the official orchestration path.

### DVC Metrics / Plots (comparables por commit)

```bash
# Refresh canonical KPI summary + plot CSVs
uv run dvc repro core.governance.export_dvc_metrics

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
| Missing parquet files | Run `scripts/run_canonical_rebuild.py` first |
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
