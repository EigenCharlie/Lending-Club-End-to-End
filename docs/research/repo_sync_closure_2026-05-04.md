# Repository Sync Closure - 2026-05-04

This note records the final cleanup decisions after the Paper Estrella audit and
the Claude follow-up commit on branch
`sync/paper-estrella-economic-champion-pipeline-freeze-2026-05-04`.

## Canonical Paper Estrella State

- Official run tag: `paper-thesis-final-economic-2026-04-06`.
- Official policy: `bound_aware_276k_economic_champion`.
- Canonical artifacts:
  - `models/final_project_promotion.json`
  - `models/champion_portfolio_policy.json`
  - `models/champion_registry.json`
  - `data/processed/final_project_summary.parquet`
  - `reports/paper_material/paper1/tables/*`
- Key closure metrics:
  - realized return: `170464.5429284627`
  - `alpha01_exact_pass=true`
  - `V=0.03645`
  - `gamma_cp=0.18591`
  - robust region: `45/45` policies pass at `alpha=0.01`

## Preserved Local Search Artifacts

The remaining dirty workspace entries were search-only PD artifacts from the
April 2026 HPO/blockwise runs. They were split by storage role:

- Small metadata and replay records stay in Git under `models/search_pd/` and
  `data/processed/search_pd/`.
- Large binary files and prediction tables are tracked by DVC pointers:
  - Optuna databases for `pd-hpo-local-2026-04-03-*`
  - CatBoost model binaries and calibrators for `pd-hpo-local-2026-04-03-1325`
  - `data/processed/search_pd/.../test_predictions.parquet`
  - `reports/figures/shap/shap_values_test.npz`
  - `reports/history/kaggle_faressayah_lendingclub.html`

This preserves historical reproducibility without adding large blobs to Git.

## Guardrails Added

- `tests/test_docs/test_paper_estrella_final_sync.py` checks that promotion,
  champion policy, registry, DVC metrics, and paper-facing tables agree on the
  economic champion.
- The same test prevents duplicate search names in
  `configs/pipeline_registry/search_registry.yaml`.
- `tests/test_scripts/test_mlflow_suite.py` now verifies that the
  `paper1_final` MLflow backfill logs the final champion metrics and canonical
  artifacts.

## Registry Cleanup

The duplicated `conformal_reopen_exhaustive` search entry was fused into one
registry row containing both validation scripts and all expected final artifacts.
`models/pipeline_registry/search_registry.json` was regenerated from the YAML
source.
