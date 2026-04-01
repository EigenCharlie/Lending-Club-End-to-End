# Decision Changes and Learnings Log
Version: 2026-03-31

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
| 2026-02-27 | Conformal promotion gate semantics | Promotion gate blocked by strict policy `overall_pass` including Kupiec/Christoffersen | Promotion gate now blocks on business checks (coverage/group/Winkler/critical alerts) and keeps statistical tests as diagnostics | Avoid false promotion blocks due to sample-size sensitivity while preserving strict policy traceability | `scripts/run_comparison.py`, `tests/test_scripts/test_run_comparison.py`, `docs/RUNBOOK.md` |
| 2026-03-16 | Paper-grade run: integral promotion deferred | `comparison.json.overall_pass=false` (two failing semantic gates) | Fixed: causal-only mismatch exemption + `paper_grade_closure_authoritative` flag; added `operational_overall_pass=true` as promotion gate | Root cause was design decisions (causal not regenerated = `insights_only`; strict conformal gate vs paper-grade closure), not model quality failures | `scripts/run_comparison.py`, `reports/run_comparisons/paper-grade-2026-03-13-final-heavy-2026-03-13-230650/comparison.json` |
| 2026-03-16 | Selective promotion of paper-grade components | No explicit promotion flags on portfolio policy, LGD conformal variant, fairlearn/skops | Portfolio policy (risk_tolerance=0.18), LGD `direct_adaptive_grade_temporal`, fairlearn+skops marked `promoted: true` in their artifacts | Full integral run NOT promoted (`overall_pass` strict=false); individual components with passing quality gates promoted selectively | `models/champion_portfolio_policy.json`, `models/conformal_lgd_ead_status.json`, `models/conformal_method_registry.json`, `models/champion_registry.json` |
| 2026-03-05 | Official baseline freeze + canonical artifacts | Core runs depended on manually passed baselines and migration dual-write artifacts (`*_v2`) | Baseline registry + freeze CLI + launcher default baseline resolution; canonical single-write for conformal/fairness/governance status | Remove ambiguity in reruns and reduce legacy artifact noise in operations | `configs/baselines/core_official_baseline.json`, `scripts/freeze_core_baseline.py`, `scripts/start_long_run.sh`, `scripts/validate_conformal_policy.py`, `scripts/run_fairness_audit.py`, `scripts/generate_governance_status.py` |
| 2026-03-16 | Paper 1 absorbed into Paper Estrella | Paper 1 (CP + Robust Opt) and Paper Estrella were separate concepts | Paper Estrella absorbs Paper 1 + adds theoretical bound alpha-Gamma, SPO+ regret comparison, and uncertainty set baselines | Avoid overlap between papers; concentrate strongest contribution in flagship venue (MS/OR/EJOR) | `docs/backlog-papers-unified.md`, `book/chapters/14-paper-estrella/*` |
| 2026-03-16 | Calibration config: method: platt → auto | 7 YAML configs hardcoded `method: platt` | All configs changed to `method: auto` with comment explaining runtime auto-selection | Config said platt but runtime selected Venn-Abers via temporal policy; misleading for readers | `configs/pd_model*.yaml` |
| 2026-03-16 | SPO+ integration surfaced | `src/optimization/spo_integration.py` existed but was not in any pipeline step | Created `scripts/run_spo_comparison.py` + added to insights_factory research profile | Code existed since early development but was never executed; needed for Paper Estrella | `scripts/run_spo_comparison.py`, `scripts/run_insights_factory.py` |
| 2026-03-17 | SPO+ v2: 5 architectural fixes | SPO+ v1 used flat MLP (500-dim input), binary costs, single seed; showed only ~2.5% improvement | v2: point-wise permutation-equivariant MLP (10-dim), calibrated PD costs (continuous), multi-seed (5 seeds), conformal robust as 3rd method, n_items=100 | Binary costs → flat landscape; flat MLP → vanishing gradients; single seed → high variance | `scripts/run_spo_real.py` (SCHEMA_VERSION 2026-03-17.2), `models/spo_real_training_status.json` |
| 2026-03-17 | SICR conformal trigger (Paper 2) | IFRS9 SICR used only PD threshold (12m) | Width of conformal interval as additional SICR signal; optimal t*=0.30 via F1 grid; ECL alpha sensitivity quantified | Loans with high model uncertainty (wide intervals) are SICR candidates regardless of PD level; regulatory cost of confidence level choice now measurable | `scripts/run_sicr_conformal.py`, `data/processed/sicr_conformal_grid.parquet`, `data/processed/ecl_alpha_sensitivity.parquet` |
| 2026-03-23 | Beta calibration as 4th candidate | 3 calibration candidates: Platt, Isotonic, Venn-Abers | 4 candidates: Platt, Isotonic, Venn-Abers, Beta (betacal library) | Broader calibrator search space; Beta calibration is theoretically appropriate for probability outputs | `src/models/calibration.py::calibrate_beta()`, `scripts/train_pd_model.py` |
| 2026-03-26 | Feature pipeline closure (rerun V2) | Runtime still depended on historical FE artifacts not rebuilt canonically from splits | Canonical feature producer materializes `train_fe/calibration_fe/test_fe`, `feature_config.pkl`, `woe_encoders.pkl`, and `feature_manifest_v2` as an explicit stage | Remove notebook/artifact drift and make reruns reconstructible from data stages | `scripts/materialize_feature_artifacts.py`, `src/features/feature_engineering.py`, `dvc.yaml` |
| 2026-03-26 | Causal lane positioning | Causal artifacts and docs still sounded like a promotable end-to-end lane | Causal becomes research-grade pricing intervention lane; official output stays `insights_only` unless overlap, sensitivity and policy gates all pass | Full-data rerun showed positive ATE/policy value but failed sensitivity, so causal remains valuable science without contaminating champion promotion | `scripts/estimate_causal_effects.py`, `models/causal_effect_status.json`, `models/causal_policy_rule.json`, `models/cate_portfolio_status.json` |
| 2026-03-26 | Challenger monotonic constraints | Monotonic constraints existed but their governance meaning was underspecified | Monotonic constraints kept as challenger-only audited spec, not injected into the canonical champion | Interpretability/governance gains were real, but predictive cost (`auc_drop`, `ece_delta`) and drift guardrails blocked promotion | `src/models/pd_model.py::resolve_monotonic_constraints`, `models/pd_challenger_spec.json`, `models/challenger_promotion_report.json` |
| 2026-03-26 | Long-run observability | Several long phases could still fail late or leave ambiguous status after reboot/OOM | Runtime statuses, checkpoints, last-valid-artifact markers and atomic writes expanded across PD, survival, LGD/EAD, causal, IFRS9 and optimization | Convert reruns from “best effort” into resumable audited execution | `src/utils/pipeline_runtime.py`, `scripts/run_long_pipeline.py`, `models/*_runtime_status.json` |
| 2026-03-26 | Bundle traceability after partial reruns | `champion_search_bundle.json` hid that some subsystems came from a later causal rerun | Bundle now exposes `artifact_run_tags` and `mixed_run_tags_present` | Avoid false impression that all artifacts come from a single homogeneous run tag | `scripts/build_champion_search_bundle.py`, `models/champion_search_bundle.json` |
| 2026-03-26 | Conformal policy recovery v3 | Canonical conformal artifact stayed blocked by `avg_width_90`, despite good coverage, because the selected tuning path was overly conservative and `shrinkback` cancelled useful group-floor adjustments | Canonical conformal now uses the grade-scaled configuration without shrinkback; `overall_pass=true`, `methodological_justification_pass=true`, and MRM returns to 4/4 PASS | Close the operational conformal gate without hiding statistical over-coverage in large OOT samples | `scripts/generate_conformal_intervals.py`, `scripts/backtest_conformal_coverage.py`, `scripts/validate_conformal_policy.py`, `models/conformal_policy_status.json` |
| 2026-03-23 | Notebooks 10-12 included in pipeline | `include_notebooks=False` in canonical and paper-grade profiles | `include_notebooks=True` in both profiles; NB10-12 executed with outputs | Paper notebooks should have outputs for Quarto book and reproducibility | `configs/run_profiles/canonical_operational.yaml`, `configs/run_profiles/paper_grade_final.yaml` |
| 2026-03-23 | Calibration monitoring enriched | Basic Brier/ECE metrics per calibrator | Log-loss per temporal fold, degradation rate, Murphy diagram, monthly calibration monitoring | Richer calibration diagnostics for paper-grade evidence and MRM defense | `scripts/train_pd_model.py`, `src/utils/visualization.py::plot_murphy_diagram()` |
| 2026-03-23 | Metadata run_tags fixed | MRM and pd_rare_event artifacts had `run_tag=None`/`untracked` | All metadata run_tags corrected to paper-grade run tag; `mrm_report_status.json` wrapper created | Metadata consistency for artifact traceability | `models/mrm_report_status.json`, `models/pd_rare_event_calibration_status.json` |
| 2026-03-23 | Conformal policy test fixed | Test expected strict `overall_pass` which fails on large OOT (276K) | Test now validates `methodological_justification_pass` logic correctly | Kupiec/Christoffersen are known to fail at high sample sizes; the methodological justification path is the correct paper-grade gate | `scripts/validate_conformal_policy.py`, tests |
| 2026-03-25 | Streamlit reduced to 5 local labs | Streamlit local retained 31 pages and broad narrative duplication | Streamlit local reduced physically to 5 optional labs; Quarto becomes explicit source of truth and public Streamlit is treated as historical showcase | Remove editorial duplication, keep only high-value interaction, and decouple local development from frozen showcase | `streamlit_app/pages/`, `book/chapters/E-streamlit-companion.qmd`, `docs/STREAMLIT_QUARTO_MIGRATION_REGISTRY.yml` |
| 2026-03-31 | Pipeline-first repository cleanup | Long-run orchestration, docs, reports, scripts, and scratch artifacts still mixed active and historical surfaces | Active vs history/research namespaces enforced; pre-Quarto helpers archived; notebook taxonomy updated; scratch runtime checkpoints purged conservatively | Reduce noise, stop accidental dependence on stale artifacts, and make sync/rebuild behavior easier to defend | `docs/DOCUMENTATION_MAP.md`, `docs/ARTIFACT_RETENTION_POLICY.md`, `scripts/history/`, `reports/history/`, `configs/pipeline_registry/*` |

---

## 2) Errors and Inconsistencies Found

| Date | Issue | Impact | Resolution | Evidence |
|------|-------|--------|------------|----------|
| 2026-02-20 | Peer-reviewed Lending Club link pointed to unrelated COVID paper (`PMC9533764`) | External benchmark credibility risk | Corrected to Lending Club paper `PMC9222552` and updated benchmark artifact | `reports/history/hpo_research_notes_2026-02-19.md`, `scripts/benchmark_kaggle_lendingclub.py` |
| 2026-02-20 | Hardcoded Streamlit claims for old calibration/method snapshots | UI could present stale or incorrect conclusions | Replaced with dynamic/neutral text tied to artifacts | `streamlit_app/pages/*.py`, `reports/history/consistency_audit_2026-02-20.md` |
| 2026-02-20 | Confusion between Optuna trial validation AUC and final OOT calibrated AUC | Misinterpretation of model selection quality | Documented split clearly: trial score is validation; final score is calibrated OOT | `models/pd_training_record.pkl`, `data/processed/model_comparison.json` |

---

## 3) Practical Learnings

1. Temporal validation is mandatory for Lending Club; random split benchmarks often overstate performance.
2. OOT calibrated AUC can be lower than best validation AUC without indicating a bug.
3. Calibration quality (Brier/ECE) can improve materially even when AUC changes little.
4. Narrative drift is a recurring risk; docs and UI must read artifact outputs, not fixed metric strings.
5. Feature-contract governance is as important as model hyperparameters in credit-risk pipelines.
6. Kupiec/Christoffersen tests are sample-size sensitive and will reject at n=276K even with coverage close to nominal; methodological justification with Winkler compensation is the correct defense.
7. Test skips should be removed once the underlying issue is fixed; carrying skips obscures the true test count.

---

## 4) Open Follow-Ups

All items resolved:

1. ~~Align `configs/pd_model.yaml` calibration wording with runtime policy~~ — **RESOLVED 2026-03-16**: all 7 pd_model YAML configs changed from `method: platt` to `method: auto`.
2. ~~Continue pruning historical snapshots with retention policy~~ — **RESOLVED 2026-03-18**: historical docs already carry HISTORICAL/DEPRECATED banners. `DOCUMENTATION_MAP.md` created.
3. ~~Add an automated "reference integrity" check for external URLs~~ — **DEFERRED 2026-03-18**: low ROI for thesis context; external links are manually audited in `docs/PAPER_REFERENCES_STATE_OF_ART.md`.

---

## 5) Related Audit Reports

- `reports/history/consistency_audit_2026-02-20.md`
- `reports/history/hpo_research_notes_2026-02-19.md`
- `reports/history/before_after_recompute_comparison_longrun.json`

---

## 6) Session History (Consolidated)

This section replaces the need for a separate `SESSION_HISTORY.md`.

| Date | Session | What was executed | Outcome | Evidence |
|------|---------|-------------------|---------|----------|
| 2026-02-17 | Post-reboot recovery | Quality gates (`ruff`, `pytest`), DVC local/cloud status, DVC push smoke, DAG verification | Environment recovered and synchronized; integrity checks green | `SESSION_STATE.md` (section "Post-Reboot Recovery Log") |
| 2026-02-18 | Repro-contract closure | `dvc repro` for pipeline/export stages, DVC push, MLflow suite backfill | Reproducibility contract restored; artifacts and tracking refreshed | `SESSION_STATE.md` (section "Repro-Contract Closure Log") |
| 2026-02-18 | Validity hardening phases 0-5 | Leakage hardening, optimization fixes, dynamic narrative updates, CP/OR benchmark updates, temporal causal backtest | `pytest` green and DVC status consistent after rerun | `SESSION_STATE.md` (section "Validity Hardening Log"), `reports/history/PHASES_0_5_EXECUTION_2026-02-18.md` |
| 2026-02-19 | HPO research + long-run setup | CatBoost/Optuna best practices review; HPO policy strengthened (multivariate TPE + pruning + persistent study) | Long-run search configured for 400 trials with reproducible tracking | `reports/history/hpo_research_notes_2026-02-19.md`, `configs/pd_model.yaml` |
| 2026-02-20 | Consistency and reference audit | External link verification, narrative consistency cleanup, artifact/process cross-check | Incorrect literature link fixed; stale claims reduced; open items documented | `reports/history/consistency_audit_2026-02-20.md` |
| 2026-02-27 | Conformal rerun hardening v1 | Soft promotion diagnostics for statistical tests, governance stage added before MRM, optional-failure exit code fix, incremental HPO budget | Rerun orchestration aligned with production promotion policy and resumable core profile | `scripts/run_comparison.py`, `scripts/run_long_pipeline.py`, `configs/run_profiles/overnight_full.yaml`, `configs/pd_model.yaml` |
| 2026-03-13 | Paper-grade final run | Run `paper-grade-2026-03-13-final-heavy` completed: AUC=0.7128, conformal 90%≥0.92, fairness 6/6 PASS, governance 6/6 PASS, Venn-Abers selected (ECE=0.0061), HPO trial 151 seed-invariant | All individual module gates PASS; full integral promotion blocked by `comparison.json` semantic gate (fixed 2026-03-16). TS intervals remain research_only. | `models/paper_grade_protocol_status.json`, `reports/run_comparisons/paper-grade-2026-03-13-final-heavy-2026-03-13-230650/comparison.json` |
| 2026-03-16 | P0/P1 fixes and selective promotions | Fixed semantic gates in `run_comparison.py` (P0.1), skops render bug (P0.3), `_row_number` added to conformal intervals (P0.4); promoted portfolio policy, LGD conformal, fairlearn+skops | `operational_overall_pass=true`, `card_render_status=rendered`, parquet has `_row_number`. 3 components promoted with explicit flags. | `models/champion_portfolio_policy.json`, `models/conformal_lgd_ead_status.json`, `models/conformal_method_registry.json` |
| 2026-03-17 | SPO+ v2 + Paper 2 IFRS9 analysis | SPO+ v2 (5 fixes): point-wise MLP, calibrated PD costs, multi-seed, conformal robust, n_items=100. SICR conformal: t* grid + ECL alpha sensitivity. | 49.1% regret reduction (Wilcoxon p=0.0000); t*=0.30, $56.6M ECL_add; alpha sensitivity +22% ECL from 90→99% conf. 665 tests passing. | `scripts/run_spo_real.py`, `scripts/run_sicr_conformal.py`, `models/spo_real_training_status.json`, `models/sicr_conformal_status.json` |
| 2026-03-21 | MAPIE deep-dive Sprint 1-4 | HSIC/SSC/MWI/CWC diagnostics, VennAbersCalibrator migrated to MAPIE native, classification set benchmark (LAC/APS/RAPS), Quarto book enriched with MAPIE diagnostics. | 674 tests passing (1 skipped). Quarto book 94/94 files render. | `scripts/backtest_conformal_coverage.py`, `scripts/run_classification_set_benchmark.py`, `book/` |
| 2026-03-23 | Calibration gap analysis + doc sync | Beta calibration added as 4th candidate, log-loss/degradation rate/Murphy diagram added, NB10-12 executed, CRPTO test skip removed, metadata run_tags fixed, conformal policy test fixed. | 690 tests passing, 0 failures, 0 skips. All metadata gaps resolved. | `scripts/train_pd_model.py`, `configs/run_profiles/`, `models/mrm_report_status.json`, `models/pd_rare_event_calibration_status.json` |
