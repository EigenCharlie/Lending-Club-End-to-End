# Canonical Documentation and Quarto Traceability

Date: 2026-03-30

## Purpose

This document is the canonical editorial ledger for the current project state. It consolidates:

- the live champion and active diagnostic layers;
- the ADSFCR adoption already implemented in this repository;
- the backlog that remains methodological only;
- the Quarto chapters that must carry each claim;
- the primary references that should support those claims in the book;
- the legacy claims that should no longer be narrated as current state.

The book should use this document as an editorial source of truth together with live artifacts in `models/`, `data/processed/`, `reports/mrm/`, and `reports/run_comparisons/`.

## Current Canonical State

### Champion and promotion state

- Current champion run tag: `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`
- Champion family: CatBoost PD + Venn-Abers calibration + monotonic constraints in the promoted champion lane
- Operational fairness semantics: `outcome_mode=approval`
- Official decision threshold: `0.35`
- Internal PD screening/search threshold: `0.05`
- Promotion closeout: `reports/run_comparisons/canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129/comparison.json`
- Registry source: `models/champion_registry.json`

### Core live evidence

| Theme | Canonical artifact | Current read |
|---|---|---|
| PD champion | `data/processed/pipeline_summary.json` | `AUC=0.7127`, `Brier=0.1546`, `ECE=0.0067`, calibration `Venn-Abers` |
| Fairness | `models/fairness_audit_status.json` | `6/6` attributes pass; `approval` semantics; threshold `0.35` |
| Monotonicity | `models/monotonicity_audit_status.json` | `PASS`; `0` adjacent disruptions; `0` constrained-feature violations |
| Governance / C2ST | `models/governance_status.json` | overall governance `PASS`; C2ST remains severe but diagnostic |
| PD backtesting | `models/pd_backtesting_status.json` | statistical `diagnostic fail`, not a promotion rollback |
| Bootstrap validation | `models/bootstrap_validation_status.json` | diagnostic-only gap uncertainty layer for large-`N` calibration reading |
| PD interpretation | `models/pd_validation_interpretation_status.json` | `warning`; global gap small, but persistent quarter/cohort deviations |
| Calibration mapping | `models/calibration_mapping_status.json` | sidecar remap/intercept lane; does not overwrite canonical calibrator |
| IFRS9 diagnostics | `models/ifrs9_diagnostics_status.json` | `diagnostic fail`; recursive instability and weak near-unit-root power |
| Encoding stability | `models/encoding_stability_status.json` | `PASS`; no structural encoding/binning failures |
| Model-shift semantics | `models/model_shift_status.json` | distinguishes structural shift from predictive degradation |
| MRM wrapper | `reports/mrm/mrm_validation_report.json` | overall `PASS` with explicit diagnostic layers |

## Techniques Currently Active In The Project

The book should explicitly document the following as part of the live system, not as speculative add-ons:

| Technique | Why it was added | Evidence now | State |
|---|---|---|---|
| Monotonic champion constraints | Economic coherence, interpretability, structurally safer score behavior | `models/champion_registry.json`, `models/monotonicity_audit_status.json` | vigente |
| Official fairness on approval decisions | Align search, audit, and business semantics around approval rather than raw PD thresholding | `models/fairness_audit_status.json`, `models/threshold_semantics.json` | vigente |
| Venn-Abers calibration selection | Stronger probabilistic fidelity for downstream policy, IFRS9 and uncertainty | `data/processed/pipeline_summary.json`, `data/processed/model_comparison.json` | vigente |
| Mondrian conformal intervals | Group-conditional uncertainty for robust decisions and monitoring | `models/conformal_policy_status.json` | vigente |
| Representativeness C2ST with drivers | Drift/representativeness diagnosis beyond PSI only | `models/governance_status.json` | vigente con warning |
| Monotonicity audit | Post-promotion defense of structural monotone behavior | `models/monotonicity_audit_status.json` | vigente |
| PD backtesting suite | Stronger validation language than a single aggregate calibration metric | `models/pd_backtesting_status.json` | vigente con diagnostic fail |
| Bootstrap validation layer | Materiality-oriented uncertainty around aggregate and slice calibration gaps | `models/bootstrap_validation_status.json` | vigente |
| PD validation interpretation | Translate statistical failures into materiality-oriented language | `models/pd_validation_interpretation_status.json` | vigente con warning |
| Calibration mapping diagnostics | Shadow remap/intercept lane to test whether cohort persistence can be reduced without retraining | `models/calibration_mapping_status.json` | vigente |
| IFRS9 diagnostics | Stress-sign coherence, recursive stability, ADF power, scenario uncertainty | `models/ifrs9_diagnostics_status.json` | vigente con diagnostic fail |
| Encoding/binning stability audit | Structural robustness of WOE/bucketed transformations | `models/encoding_stability_status.json` | vigente |
| Model-shift and p-value interpretation | Separate structural representativeness shift from predictive degradation | `models/model_shift_status.json`, `models/governance_status.json` | vigente |

## ADSFCR Adoption Status

### Implemented from ADSFCR-inspired work

| ADSFCR family | What entered the repo | Live artifact | Implementation status |
|---|---|---|---|
| Representativeness via C2ST | classifier-based train/test representativeness with driver attribution | `models/governance_status.json` | implemented |
| Heterogeneity / monotonicity disruption | monotonicity structure audit for the promoted monotonic champion | `models/monotonicity_audit_status.json` | implemented |
| PD validation / backtesting | exact-binomial, Jeffreys-aware interpretation, HL/Z-style reading, material slice persistence | `models/pd_backtesting_status.json`, `models/pd_validation_interpretation_status.json` | implemented |
| Bootstrap hypothesis tests | bootstrap-based gap uncertainty for aggregate and slice calibration interpretation | `models/bootstrap_validation_status.json` | implemented |
| Calibration mapping diagnostics | intercept-shift and monotone remap sidecars on OOT windows | `models/calibration_mapping_status.json` | implemented |
| Calibration mapping shadow validation | consolidated wrapper to test whether a promising remap survives downstream Mondrian checks | `models/calibration_mapping_shadow_impact_status.json` | implemented and executed; closed as `keep_current_calibrator` |
| IFRS9 diagnostics | recursive regressions, ADF power, sign coherence, interval-width interpretation | `models/ifrs9_diagnostics_status.json` | implemented |
| Binning/encoding stability | WOE and bucket stability diagnostics | `models/encoding_stability_status.json` | implemented |
| Model-shift and p-value hardening | structural-vs-predictive shift semantics in governance and MRM | `models/model_shift_status.json`, `models/governance_status.json` | implemented |

### Still valuable but not implemented

| ADSFCR family | Why it remains interesting | Current decision |
|---|---|---|
| Blockwise / constrained threshold model design | Could inspire future challenger architectures or score-policy blends | methodological backlog |
| LGD survival / PoC calibration | Valuable for a future LGD/EAD lane hardening | backlog for LGD lane |
| LDP, Vasicek, concentration, EIR, repayment plans | Not central to the current Lending Club champion pipeline | do not integrate into core book narrative |

## Quarto Traceability Map

| Claim family | Main Quarto chapter(s) | Supporting artifact(s) | Primary reference direction |
|---|---|---|---|
| Modern project overview and active stack | `01-executive-map` | `pipeline_summary.json`, `champion_registry.json`, `mrm_validation_report.json` | CatBoost, conformal, SR 11-7, fairness |
| Monotonic champion rationale | `06b`, `06d`, `F-rerun-v2-refactor` | `champion_registry.json`, `comparison.json`, `monotonicity_audit_status.json` | CatBoost, monotone ML / interpretable risk modeling |
| Official fairness semantics on approval | `06d`, `10e`, `10f` | `fairness_audit_status.json`, `threshold_semantics.json` | Equalized odds / fairness in supervised learning |
| PD calibration and cohort diagnostics | `06d`, `07d`, `10e`, `10f` | `pipeline_summary.json`, `pd_validation_interpretation_status.json`, `pd_backtesting_status.json`, `bootstrap_validation_status.json`, `calibration_mapping_status.json` | calibration / binomial interval / backtesting references |
| Conformal operational closure | `07d`, `10e`, `10f` | `conformal_policy_status.json` | conformal prediction, ACI, coverage diagnostics |
| C2ST representativeness | `10e`, `10f` | `governance_status.json` | C2ST primary paper |
| Model shift semantics and warning posture | `10e`, `10f` | `model_shift_status.json`, `governance_status.json` | MRM / diagnostic interpretation references |
| IFRS9 diagnostics and open risks | `10d`, `10e`, `10f` | `ifrs9_diagnostics_status.json`, `ifrs9_sensitivity_grid.parquet` | recursive regressions, unit-root testing, scenario analysis |
| Encoding/binning stability | `05c`, `10e`, `10f` | `encoding_stability_status.json` | WOE/binning stability references |
| Research-lane separation | `01`, `06d`, `10e`, `14c`, GPU / paper chapters | `champion_registry.json`, `paper_grade_protocol_status.json` | project internal canons + paper references |

## Claim To Evidence Ledger

| Claim | Evidence artifact | Source type | Quarto location |
|---|---|---|---|
| The monotonic model is the current champion | `models/champion_registry.json`, `reports/run_comparisons/canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129/comparison.json` | internal canonical artifact | `06d`, `10e`, `F` |
| Fairness is audited on approval decisions, not on raw PD exceedance | `models/fairness_audit_status.json`, `models/threshold_semantics.json` | internal canonical artifact | `06d`, `10e`, `10f` |
| The champion remains operationally valid despite some diagnostic warnings | `reports/mrm/mrm_validation_report.json` | internal canonical artifact | `01`, `10e`, `10f` |
| C2ST is intentionally diagnostic and should be interpreted with PSI/performance context | `models/governance_status.json` | internal canonical artifact + primary methodology | `10e`, `10f` |
| PD validation is not globally broken, but some cohorts deviate materially | `models/pd_validation_interpretation_status.json` | internal canonical artifact | `07d`, `10e`, `10f` |
| Bootstrap diagnostics add a large-`N` materiality lens on top of classical calibration tests | `models/bootstrap_validation_status.json` | internal canonical artifact + bootstrap testing references | `07d`, `10e`, `10f` |
| Calibration mapping is now monitored as a shadow lane, not as a canonical calibrator replacement | `models/calibration_mapping_status.json` | internal canonical artifact | `06d`, `07d`, `10e`, `10f` |
| Calibration mapping shadow validation closed with no promotable remap candidate; the next PD work is analytical cohort interpretation, not a lightweight remap | `models/calibration_mapping_shadow_impact_status.json` | internal canonical artifact | `06d`, `07d`, `10e`, `10f` |
| IFRS9 currently needs stronger temporal defensibility | `models/ifrs9_diagnostics_status.json` | internal canonical artifact + primary time-series references | `10d`, `10e`, `10f` |
| Encoding/binning is structurally stable today | `models/encoding_stability_status.json` | internal canonical artifact | `05c`, `10e` |
| Governance separates structural shift from predictive degradation instead of collapsing both into raw p-values | `models/model_shift_status.json`, `models/governance_status.json` | internal canonical artifact + governance methodology | `10e`, `10f` |

## Legacy Claims That Must Not Be Narrated As Current State

| Legacy claim | Why it is now wrong or misleading | Editorial action |
|---|---|---|
| The monotonic challenger was useful but not promoted | the monotonic lane is now the promoted champion | rewrite everywhere |
| The active champion run is `paper-grade-2026-03-13-final-heavy-...` or `champion-2026-03-12-mega-definitive` | those are historical milestones, not the current canonical champion | replace with current run tag or downgrade to historical context |
| Fairness semantics should be read on the old PD-threshold logic | current audit/search/policy semantics are approval-based | rewrite |
| C2ST should be described as a pure failure signal | in the current governance stack it is an informative/severe diagnostic with driver context | rewrite |
| IFRS9 sensitivity is dominated by PD | current diagnostic surface says `lgd_mult` dominates the sensitivity slope | rewrite |
| The book is structurally complete and only needs minor polishing | the project evolved materially after the earlier Quarto closure; multiple chapters require live-state refresh | replace with maintenance reality |

## Internal Discovery Sources Used For This Refresh

These are discovery inputs only. They should not be cited in the book as scholarly references:

- `docs/ADSFCR_AUDIT_AND_MONOTONIC_CHALLENGER_PLAN_2026-03-29.md`
- `docs/MODEL_RISK_MANAGEMENT.md`
- `SESSION_STATE.md`
- `reports/mrm/mrm_validation_report.json`
- `.claude/projects/-home-eigenlinux-projects-lending-club-risk-project/memory/project_quarto_book_progress.md`
- `.claude/projects/-home-eigenlinux-projects-lending-club-risk-project/memory/project_mega_run_20260312.md`
- Codex thread registry metadata from `.codex/state_5.sqlite`

Use rule:

- discovery from chats and memory may suggest what changed;
- only repo artifacts and canonical docs may substantiate live claims;
- Quarto should cite primary literature plus canonical project artifacts.

## Primary Reference Targets For The Book

These references should anchor the methodology in Quarto:

- CatBoost and calibrated PD modeling
- fairness in supervised learning and threshold-based decision policies
- conformal prediction and adaptive conformal monitoring
- classifier two-sample tests for representativeness
- binomial interval / calibration backtesting interpretation
- recursive regression stability and unit-root testing
- SR 11-7 / model risk governance

## Editorial Defaults

- Narrate the current state, not the full promotion history.
- Keep historical promotions only when they help interpret why a technique exists.
- If a diagnostic layer is `warning` or `diagnostic fail`, explain it explicitly; do not hide it and do not escalate it into a false production failure.
- Prefer dynamic tables/figures sourced from canonical artifacts over hardcoded snapshots.
- Keep research lanes visible, but never present them as champion behavior when they are not part of the promoted stack.
