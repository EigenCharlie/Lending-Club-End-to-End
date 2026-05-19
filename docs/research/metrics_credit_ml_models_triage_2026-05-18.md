# Metrics for Credit and ML Models - triage 2026-05-18

## Purpose

This memo closes the intake of `C:\Users\carlos\Downloads\Metrics for Credit and ML Models.pdf`.
The PDF is a practitioner binder with five pieces on discrimination, calibration,
equity and trust. The goal here is not to open another experimental loop. The
goal is to decide what enters Paper Estrella, what enters Paper 4, and what
stays parked.

## Source status

| Source | Status | Verified source | Project use |
| --- | --- | --- | --- |
| Wuthrich, "Model Selection with Gini Indices under Auto-Calibration" | peer-reviewed journal article, European Actuarial Journal 2023 | https://doi.org/10.1007/s13385-022-00339-9 | Paper Estrella + Paper 4 |
| Albanesi and Vamossy, "Credit Scores: Performance and Equity" | NBER Working Paper 32917, 2024 | https://doi.org/10.3386/w32917 | Paper 4 strong context; Paper Estrella light context |
| Dinga et al., "Beyond accuracy" | bioRxiv preprint, 2019 | https://doi.org/10.1101/743138 | metric taxonomy context only |
| Somers' D / ordered association material | local binder material | local PDF | optional metric appendix |
| ReScorer | local binder material, GenAI/e-commerce reason scoring | local PDF | park |

## Paper 2 absorption check

Paper 2 no longer needs to be a near-term standalone target. Paper 4 is the live
lab and future full-paper container, so it should absorb all useful Paper 2
prudential evidence: ECL scenarios, conformal ECL ranges, SICR by conformal
width, CIF/prepayment, stage-cost governance and TS-to-ECL context. The
absorption is complete for project planning, but bounded in claim language:
Paper 4 keeps the material as an `IFRS9-inspired SICR/ECL proxy` lane, not as a
contractual IFRS 9 implementation.

| Paper 2 evidence | Paper 4 destination | Decision |
| --- | --- | --- |
| ECL by scenario and conformal range | IFRS9 evidence card | append |
| SICR conformal threshold `t*=0.30`, recall 75.8%, ECL +56.6M | IFRS9 evidence card | append strong |
| CIF vs Kaplan-Meier prepayment correction | IFRS9 evidence card | append |
| Stage misclassification cost | governance/threshold appendix | append if staging is discussed |
| TS to ECL intervals | stress context | context-only |
| Near-term standalone Paper 2 | no longer required | park/supersede |
| Contractual IFRS9 claim | no destination | keep false |

## What changes for Paper Estrella

The binder does not add a new contribution to Paper Estrella. It strengthens two
phrases that were already true:

1. **Wuthrich** supports the statement that AUC/Gini are not sufficient for
   model selection unless the score is calibration-gated. This fits the current
   narrative: CRPTO is not an AUC leaderboard; it uses calibrated PD plus
   conformal uncertainty plus a robust decision.
2. **Albanesi and Vamossy** support the credit-scoring/equity motivation: better
   ML scores can reduce misclassification of traditional score bands and change
   borrower standing. This is context, not legal fair-lending evidence.

Do not add Dinga, Somers' D, or ReScorer to the Paper Estrella bibliography
unless a future appendix explicitly needs them.

## What changes for Paper 4

| Source | Use in Paper 4 | Decision |
| --- | --- | --- |
| Wuthrich | Calibration-gated Gini/AUC: ranking is interpreted only after calibration | append methodological |
| Albanesi and Vamossy | FICO/score proxy vs champion ML, misclassification, ranking difference, observable vulnerable groups | append strong |
| Dinga et al. | Metric taxonomy: discrimination, calibration, utility and equity | context |
| Somers' D | Bootstrap/tie sensitivity for Gini/AUC/Somers' D | optional |
| ReScorer | Only if we audit LLM reasons in the research workflow | park |

## Executed bounded experiment for Paper 4

The compact experiment was executed on 2026-05-19 through the consolidated
Paper 4 frontier runner, without creating a new versioned wave.

**Question.** Does the project champion materially improve ranking and
misclassification relative to an origin-time FICO proxy, especially for
observable vulnerable or low-data-quality groups?

**Data.**

- `fico_range_low`, `fico_range_high` or `fico_score`
- `default_flag` or project default target
- champion calibrated PD
- `annual_inc`, `dti`, `home_ownership`, `mort_acc`, `total_acc`,
  `earliest_cr_line`, `addr_state`, `zip3`, `grade`, `sub_grade`, `purpose`

**Protocol.**

1. Build a FICO proxy score as midpoint of FICO range and risk bands by decile
   or conventional score bands.
2. Compare FICO proxy vs champion on AUC/Gini/Somers' D.
3. Calibrate FICO proxy with one simple monotone mapping before comparing Brier
   or ECE, because Wuthrich's gate makes raw rank metrics insufficient.
4. Compute ranking difference: `champion_percentile - fico_percentile`.
5. Measure misclassification by risk band: loans whose realized default rate is
   closer to a different model-implied band than their FICO band.
6. Slice by observable groups: income quintile, thin-file proxy, no-mortgage,
   delinquency/history proxy if available, state/zip3 support cells and grade.
7. Close with one decision: append, park, or delete.

**Result.** On the latest 40% of the OOT window (`n=103,865`), the calibrated
champion beats the origin-time FICO proxy on the intended metric stack:

| metric | champion | FICO proxy | delta champion - FICO |
| --- | ---: | ---: | ---: |
| AUC | 0.700477 | 0.592906 | 0.107570 |
| Gini/Somers' D | 0.400953 | 0.185813 | 0.215140 |
| Brier | 0.140380 | 0.151195 | -0.010815 |
| ECE 10-bin | 0.028995 | 0.051132 | -0.022137 |
| decile band MAE | 0.030545 | 0.051076 | -0.020532 |

`48.6526%` of loans move at least 20 percentile points between the FICO proxy
ranking and the champion ranking. The result also survives the intended
observable slices (income tails, high DTI, no-mortgage, thin-file proxy and
grades A-E) as an appendix diagnostic.

**Decision.** Append to Paper 4 as metric-governance evidence. Paper Estrella
can use this only as light context if needed. No fair-lending legal claim is
allowed.

## Editorial decision

- Paper Estrella gets a surgical calibration/equity-context reinforcement.
- Paper 4 gets an IFRS9 evidence card and a future metric-governance experiment
  design.
- No new `paper4_v###`, no new exploratory CSV wave, and no commit-per-iteration
  artifact should be created from this intake.
