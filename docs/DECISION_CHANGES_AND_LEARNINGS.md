# Decision Changes and Learnings Log
Version: 2026-02-20

This file stores project history: decision changes, mistakes, inconsistencies, and practical learnings.
Do not store this type of historical content in `CLAUDE.md` or `docs/PROJECT_JUSTIFICATION.md`.

---

## 1) Decision Change Log

| Date | Topic | Before | After | Why | Evidence |
|------|-------|--------|-------|-----|----------|
| 2026-02-20 | PD architecture comparison | `y_prob_cb_default` and `y_prob_cb_tuned` were effectively equal in export path | Default and tuned predictions are generated and stored independently | Avoid false "no improvement" conclusions | `scripts/train_pd_model.py`, `tests/test_models/test_pd_model.py` |
| 2026-02-20 | Feature contract source | Runtime relied on legacy static subset (11 features) | Runtime resolves feature sets from `feature_config.pkl` and persists contract | Recover predictive signal from full engineered feature set | `src/models/pd_model.py`, `configs/pd_model.yaml`, `models/pd_model_contract.json` |
| 2026-02-20 | Calibration policy | Fixed narrative said "Platt selected" | Temporal multi-metric selection policy between Platt/Isotonic | Better OOT probability quality and less narrative drift | `scripts/train_pd_model.py`, `data/processed/model_comparison.json` |
| 2026-02-20 | CatBoost tuning process | Tuning existed but was not consistently surfaced in artifacts | Long-run Optuna config enabled (400 trials, TPE multivariate, pruning, SQLite study) | Make HPO real, traceable, and reproducible | `configs/pd_model.yaml`, `src/models/pd_model.py`, `models/optuna_pd_catboost.db` |

---

## 2) Errors and Inconsistencies Found

| Date | Issue | Impact | Resolution | Evidence |
|------|-------|--------|------------|----------|
| 2026-02-20 | Peer-reviewed Lending Club link pointed to unrelated COVID paper (`PMC9533764`) | External benchmark credibility risk | Corrected to Lending Club paper `PMC9222552` and updated benchmark artifact | `reports/hpo_research_notes_2026-02-19.md`, `scripts/benchmark_kaggle_lendingclub.py` |
| 2026-02-20 | Hardcoded Streamlit claims for old calibration/method snapshots | UI could present stale or incorrect conclusions | Replaced with dynamic/neutral text tied to artifacts | `streamlit_app/pages/*.py`, `reports/consistency_audit_2026-02-20.md` |
| 2026-02-20 | Confusion between Optuna trial validation AUC and final OOT calibrated AUC | Misinterpretation of model selection quality | Documented split clearly: trial score is validation; final score is calibrated OOT | `models/pd_training_record.pkl`, `data/processed/model_comparison.json` |

---

## 3) Practical Learnings

1. Temporal validation is mandatory for Lending Club; random split benchmarks often overstate performance.
2. OOT calibrated AUC can be lower than best validation AUC without indicating a bug.
3. Calibration quality (Brier/ECE) can improve materially even when AUC changes little.
4. Narrative drift is a recurring risk; docs and UI must read artifact outputs, not fixed metric strings.
5. Feature-contract governance is as important as model hyperparameters in credit-risk pipelines.

---

## 4) Open Follow-Ups

1. Align `configs/pd_model.yaml` calibration wording with runtime policy (currently `method: platt` while runtime can auto-select).
2. Continue cleaning legacy historical snapshots in notebooks/reports so they are clearly marked as archival.
3. Add an automated "reference integrity" check for external URLs used in benchmark notes.

---

## 5) Related Audit Reports

- `reports/consistency_audit_2026-02-20.md`
- `reports/hpo_research_notes_2026-02-19.md`
- `reports/before_after_recompute_comparison_longrun.json`

---

## 6) Session History (Consolidated)

This section replaces the need for a separate `SESSION_HISTORY.md`.

| Date | Session | What was executed | Outcome | Evidence |
|------|---------|-------------------|---------|----------|
| 2026-02-17 | Post-reboot recovery | Quality gates (`ruff`, `pytest`), DVC local/cloud status, DVC push smoke, DAG verification | Environment recovered and synchronized; integrity checks green | `SESSION_STATE.md` (section "Post-Reboot Recovery Log") |
| 2026-02-18 | Repro-contract closure | `dvc repro` for pipeline/export stages, DVC push, MLflow suite backfill | Reproducibility contract restored; artifacts and tracking refreshed | `SESSION_STATE.md` (section "Repro-Contract Closure Log") |
| 2026-02-18 | Validity hardening phases 0-5 | Leakage hardening, optimization fixes, dynamic narrative updates, CP/OR benchmark updates, temporal causal backtest | `pytest` green and DVC status consistent after rerun | `SESSION_STATE.md` (section "Validity Hardening Log"), `reports/PHASES_0_5_EXECUTION_2026-02-18.md` |
| 2026-02-19 | HPO research + long-run setup | CatBoost/Optuna best practices review; HPO policy strengthened (multivariate TPE + pruning + persistent study) | Long-run search configured for 400 trials with reproducible tracking | `reports/hpo_research_notes_2026-02-19.md`, `configs/pd_model.yaml` |
| 2026-02-20 | Consistency and reference audit | External link verification, narrative consistency cleanup, artifact/process cross-check | Incorrect literature link fixed; stale claims reduced; open items documented | `reports/consistency_audit_2026-02-20.md` |
