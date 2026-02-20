# Consistency Audit — 2026-02-20

## Scope
- External references in HPO/benchmark files
- Core PD artifacts after long run (400 trials)
- Streamlit narrative consistency with current artifacts
- Process reproducibility checks

## Findings

### 1) Wrong external reference (high) — FIXED
- Issue: `PMC9533764` was labeled as Lending Club benchmark but is a COVID-19 surgery paper.
- Evidence:
  - `PMC9533764` title: "COVID-19 Vaccination and the Timing of Surgery Following COVID-19 Infection"
  - `PMC9222552` title: "P2P Lending Default Prediction Based on AI and Statistical Models"
- Fixes applied:
  - `reports/hpo_research_notes_2026-02-19.md` updated to `PMC9222552`
  - `scripts/benchmark_kaggle_lendingclub.py` literature row updated
  - `data/processed/lendingclub_literature_benchmark.parquet` regenerated

### 2) Unsupported benchmark value/link pairing (high) — FIXED
- Issue: Literature row claimed AUC `0.729` tied to wrong URL.
- Fix: replaced with validated peer-reviewed Lending Club source (`PMC9222552`) and updated `reported_auc=0.7492` per paper text (74.92%).

### 3) Streamlit stale claims about calibration method (medium) — PARTIALLY FIXED
- Issue: multiple pages hardcoded "Platt selected" / fixed ECE values from old runs.
- Fixes applied (dynamic or neutralized text):
  - `streamlit_app/pages/glossary_fundamentals.py`
  - `streamlit_app/pages/thesis_contribution.py`
  - `streamlit_app/pages/paper_1_cp_robust_opt.py`
  - `streamlit_app/pages/research_landscape.py`
  - `streamlit_app/pages/thesis_defense.py`
  - `src/models/calibration.py` docstring de-hardcoded

### 4) Artifact/process consistency (high) — VERIFIED
- `models/pd_training_record.pkl`:
  - `hpo_trials_executed = 400`
  - `hpo_best_validation_auc = 0.719885...`
  - final test AUC calibrated = `0.717183...`
- `models/optuna_pd_catboost.db`:
  - trials state distribution: `PRUNED=296`, `COMPLETE=104`
  - confirms pruning callback was active and functional.
- `data/processed/model_comparison.json` consistent with training record (best model/calibration and final test metrics).

### 5) Remaining non-blocking inconsistencies (low)
- Historical/static docs still mention old Platt/ECE values:
  - `CLAUDE.md`
  - `docs/PROJECT_JUSTIFICATION.md`
  - historical notebook outputs (e.g., `notebooks/03_pd_modeling.ipynb` saved cells)
  - historical snapshots (e.g., `reports/project_audit_snapshot.json`)
- Note: these appear to be legacy/static narrative artifacts, not runtime model artifacts.

### 6) Config semantics mismatch (medium, open)
- `configs/pd_model.yaml` keeps `calibration.method: platt`, but training pipeline currently selects method by temporal multi-metric policy (`platt` vs `isotonic`) regardless of that field.
- Impact: potential confusion for maintainers/readers.
- Recommendation: rename config key to `calibration.policy: auto_select` or enforce field usage explicitly.

## Validation executed
- URL/title verification with live HTTP checks for corrected/incorrect PMCs.
- Runtime artifact checks (`model_comparison.json`, `pipeline_summary.json`, `conformal_policy_status.json`, Optuna DB).
- Tests:
  - `tests/test_streamlit/test_page_imports.py`
  - `tests/test_scripts/test_export_streamlit_artifacts.py`
  - `tests/test_models/test_pd_model.py`
  - `tests/test_config_consistency.py`
