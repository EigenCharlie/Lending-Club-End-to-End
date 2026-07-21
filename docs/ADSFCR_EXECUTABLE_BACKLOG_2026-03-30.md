# ADSFCR Executable Backlog

Date: 2026-03-30

> **Historical backlog — superseded for Paper 2.** This file records ideas at
> the March cutoff. The current Paper 2 contract is `parked_ifrs9`: the
> terminal-outcome archive cannot identify reporting-date PD, ECL or SICR, and
> this backlog does not reopen those claims. See
> `reports/paper_material/paper2/paper2_claim_contract.yml` and
> `SESSION_STATE.md`.

This document isolates the remaining `adsfcr`-inspired work that still looks worth implementing after the monotonic promotion, confirmatory rebuild, and the first three ADSFCR tranches already integrated into the repo.

It is intentionally short and execution-oriented. For the full audit and rationale, see:

- `docs/ADSFCR_AUDIT_AND_MONOTONIC_CHALLENGER_PLAN_2026-03-29.md`
- `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md`

## Priority Summary

Update after implementation wave:

- `P1` bootstrap diagnostics: implemented
- `P1` calibration mapping diagnostics: implemented
- `P1` calibration mapping shadow validation: implemented and closed as `negative but valuable result`
- `P2` model-shift / p-value interpretation hardening: implemented
- remaining valuable backlog: `LGD survival / PoC calibration` and `blockwise / constrained-threshold challenger ideas`

| Priority | Workstream | Why it still matters | Retraining | Downstream rerun |
|---|---|---|---|---|
| P1 | Bootstrap hypothesis tests | Strengthens PD/conformal/MRM interpretation beyond asymptotic tests at large `N` | No | Yes |
| P1 | Calibration mapping diagnostics | The current bottleneck is cohort persistence, not AUC; worth testing intercept/remapping diagnostics | No initially | Yes |
| P2 | Model-shift and p-value interpretation hardening | Improves governance language, warning semantics, and MRM readability | No | Yes |
| P2 | LGD survival / PoC calibration | Best remaining ADSFCR value for the IFRS9/LGD lane | Yes | Yes |
| P3 | Blockwise / constrained-threshold challenger ideas | Interesting next-generation challenger lane, but too invasive for the current promoted stack | Yes | Yes |

## Implemented In This Wave

### Bootstrap hypothesis tests

- status: implemented
- live artifacts:
  - `models/bootstrap_validation_status.json`
  - `data/processed/bootstrap_validation_slices.parquet`
- repo integration:
  - `src/evaluation/backtesting.py`
  - `scripts/run_bootstrap_validation_diagnostics.py`
  - `scripts/run_long_pipeline.py`
  - `scripts/generate_mrm_report.py`
  - `configs/mrm_policy.yaml`

### Calibration mapping diagnostics

- status: implemented
- live artifacts:
  - `models/calibration_mapping_status.json`
  - `data/processed/calibration_mapping_candidates.parquet`
  - `models/calibration_mapping_shadow_impact_status.json`
- repo integration:
  - `src/evaluation/calibration_mapping.py`
  - `scripts/run_calibration_mapping_diagnostics.py`
  - `scripts/run_calibration_mapping_shadow_validation.py`
  - `scripts/run_long_pipeline.py`
  - `scripts/generate_mrm_report.py`
  - `configs/mrm_policy.yaml`

### Calibration mapping shadow validation closeout

- status: implemented and executed on the confirmatory monotonic run
- live artifacts:
  - `models/calibration_mapping_status.json`
  - `models/calibration_mapping_shadow_impact_status.json`
- operational result:
  - no shadow candidate passed `Gate PD`
  - `current_identity` remained the best observable candidate
  - neither `logit_intercept_shift` nor `isotonic_sidecar` improved cohort persistence
  - downstream Mondrian rerun was therefore not triggered
- implication:
  - this line should remain documented as a useful negative result for thesis / MRM
  - the next PD work is better spent on cohort-sensitive analytical interpretation than on more lightweight remapping

Suggested thesis/documentation reading: the shadow validation is worth preserving not because it changed the champion, but because it closed a plausible methodological alternative. A lightweight post-hoc remap was a reasonable hypothesis given the residual cohort persistence, yet the executed evidence showed that the simple sidecars worsened calibration quality instead of improving it. That makes the remaining issue better understood: what is left to explain is not a missing easy recalibration trick, but a cohort-level temporal pattern that deserves analytical interpretation rather than another quick remapping pass.

### Model-shift and p-value interpretation hardening

- status: implemented
- live artifacts:
  - `models/model_shift_status.json`
  - `models/governance_status.json`
  - `reports/mrm/mrm_validation_report.json`
- repo integration:
  - `src/evaluation/model_shift.py`
  - `scripts/generate_governance_status.py`
  - `scripts/generate_mrm_report.py`
  - `configs/mrm_policy.yaml`

## Archived Execution Specs

### P1. Bootstrap Hypothesis Tests

### Goal

Add bootstrap-based diagnostic tests where the current stack relies mostly on asymptotic p-values or threshold heuristics. The target is better interpretation, not a new promotion gate.

### Why now

- `models/pd_backtesting_status.json` is statistically strict.
- `models/pd_validation_interpretation_status.json` already improved the language, but it still leans on classical tests.
- Bootstrap gives a more stable “is this effect material?” layer in large-sample settings.

### Files to create or touch

- `src/evaluation/backtesting.py`
- new runner: `scripts/run_bootstrap_validation_diagnostics.py`
- `scripts/generate_mrm_report.py`
- `scripts/run_long_pipeline.py`
- `configs/mrm_policy.yaml`
- Quarto chapters `07d`, `10e`, `10f`

### New artifacts

- `models/bootstrap_validation_status.json`
- optionally `data/processed/bootstrap_validation_slices.parquet`

### Acceptance criteria

- bootstrap diagnostics run over PD validation and optionally conformal backtesting summaries;
- output is clearly labeled `diagnostic_only`;
- MRM report includes it without changing champion promotion status;
- Quarto explains why bootstrap helps when `N` is very large.

### P1. Calibration Mapping Diagnostics

### Goal

Add a diagnostic lane for calibration mapping inspired by:

- discrete PD rating-scale calibration;
- intercept optimization for discrete calibration.

This should begin as a sidecar diagnostic, not as a replacement for Venn-Abers.

### Why now

- current PD lane is strong in AUC/Brier/ECE;
- the open issue is `material_slice_deviation` by quarter/cohort;
- the best next question is whether a lightweight remapping/intercept correction reduces persistence without reopening model training.

### Files to create or touch

- new module: `src/evaluation/calibration_mapping.py`
- new runner: `scripts/run_calibration_mapping_diagnostics.py`
- `scripts/generate_mrm_report.py`
- `scripts/run_long_pipeline.py`
- possibly `src/evaluation/pd_validation_interpretation.py`
- Quarto chapters `06d`, `07d`, `10e`, `10f`

### New artifacts

- `models/calibration_mapping_status.json`
- `data/processed/calibration_mapping_candidates.parquet`

### Acceptance criteria

- compare current calibration against intercept/remap candidates on OOT cohorts;
- report effect on global gap, quarter persistence, and slice materiality;
- do not overwrite the canonical calibrator;
- document whether this lane is promising enough to justify a later calibration challenger.

### Execution result after implementation

- the lane was executed through `scripts/run_calibration_mapping_shadow_validation.py`;
- the final decision was `keep_current_calibrator`;
- `current_identity` remained the best candidate in `models/calibration_mapping_status.json`;
- the two lightweight challengers worsened absolute gap, quarter breaches, worst-quarter gap, and ECE;
- this means the remaining issue is not a simple post-hoc remap problem but a cohort-interpretation problem.

### P2. Model Shift and P-Value Interpretation Hardening

### Goal

Improve the governance layer so the repo speaks more clearly about:

- model shift;
- when p-values are informative vs misleading;
- why some diagnostics stay warnings instead of gates.

### Why now

- C2ST is strong and easy to misread;
- PD/conformal diagnostics already use nuanced semantics;
- this work improves explanation quality more than model quality, but that is still high-value for the thesis and Quarto book.

### Files to create or touch

- `scripts/generate_governance_status.py`
- `scripts/generate_mrm_report.py`
- possibly a small helper module under `src/evaluation/`
- Quarto chapters `10e`, `10f`
- `docs/MODEL_RISK_MANAGEMENT.md`

### New artifacts

- optional: `models/model_shift_status.json`

### Acceptance criteria

- governance/MRM can distinguish structural shift, predictive degradation, and purely statistical detection;
- Quarto gets a cleaner explanation of C2ST, p-values, and warning semantics;
- no new gate is introduced unless explicitly configured later.

## P2. LGD Survival / Probability-of-Cure Calibration

### Goal

Open the next useful ADSFCR lane for IFRS9 by strengthening LGD/EAD methodology rather than pushing more on the already-promoted PD champion.

### Why now

- `models/ifrs9_diagnostics_status.json` says the current open problem is temporal defensibility and uncertainty width;
- the sensitivity surface indicates `lgd_mult` dominates the current ECL slope;
- this makes LGD a more valuable next ADSFCR target than another PD challenger.

### Files to create or touch

- likely new modules under `src/models/` or `src/evaluation/`
- likely new scripts beside existing LGD/EAD runners
- `scripts/run_long_pipeline.py`
- IFRS9 reporting and Quarto chapters `10a`, `10d`, `10e`, `15b`, `15c`

### New artifacts

- to be defined with the design, but should not overwrite current canonical LGD/EAD artifacts on first pass

### Acceptance criteria

- initial implementation should be a research or shadow lane;
- must show whether LGD survival / PoC materially narrows uncertainty or improves IFRS9 defensibility;
- only after that should it be considered for canonical integration.

## P3. Blockwise / Constrained-Threshold Challenger Designs

### Goal

Use ADSFCR blockwise and constrained-threshold ideas as inspiration for a future challenger family.

### Why later

- the monotonic champion is already promoted and stable;
- this line implies a deeper redesign of model + policy rather than a monitoring improvement;
- it is better treated as a future research challenger than as an immediate next sprint.

### Files to create or touch

- future challenger search scripts
- future config families under `configs/`
- future Quarto discussion in PD modeling / research agenda

### Acceptance criteria

- only start this if the team explicitly wants a new challenger family;
- define it as `research lane` first, not as a hidden modification of the current champion.

## Explicitly Out Of Scope For Now

These were audited and are not worth near-term implementation for the current project state:

- LDP methods
- Vasicek / asset-correlation modeling
- concentration risk package
- effective interest rate utilities
- loan repayment plan material
- scorecard-scaling-specific material
- broad OLS assumption notes with no direct pipeline connection

## Recommended Execution Order

1. `LGD survival / PoC` shadow lane
2. blockwise challenger research

## Rule Of Thumb

If a pending ADSFCR idea:

- improves monitoring, interpretation, or MRM without retraining, it is a good near-term candidate;
- requires a new model family or a new feature contract, it belongs to a later research lane unless a concrete production-style gain is already visible.
