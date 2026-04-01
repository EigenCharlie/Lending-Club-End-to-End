# Promotion Dossier - 2026-03-01

## Historical Status

This document is a historical snapshot of the March 1, 2026 promotion exercise.

Do not treat it as the live canonical policy state.

For current truth, use live artifacts under `models/`, `data/processed/`, `reports/storytelling_snapshot.json`,
and the active documentation map in `docs/DOCUMENTATION_MAP.md`.

## Scope
This dossier captures the smart-profile promotion decision executed on **March 1, 2026** for branches:
- A: `main` (`dbf5f18`)
- B: `research/toboml2-integration-v1` (`b8b44b1` base)
- C: `experiment/conformal-toboml-integration-rerun` (`623d53f` base + hardening updates)

## Official smart run tags
- A: `2026-03-01-A-smart-v2`
- B: `2026-03-01-B-smart-v2`
- C: `2026-03-01-C-smart-v2`

## Cross-branch blocking comparisons
Stored in `reports/run_comparisons/2026-03-01-promotion-dossier/`:
- `B_vs_A_final.json` / `B_vs_A_final.md`
- `C_vs_B_final.json` / `C_vs_B_final.md`
- `C_vs_A_final.json` / `C_vs_A_final.md`

## Result summary
All three cross-branch comparisons pass (`overall_pass=true`).

### B vs A
- overall_pass: true
- artifact_coherence: true
- pd_quality: true
- ab_no_regression: true
- fairness_relative: true
- conformal_promotion_pass: true
- survival_quality: true

### C vs B
- overall_pass: true
- artifact_coherence: true
- pd_quality: true
- ab_no_regression: true
- fairness_relative: true
- conformal_promotion_pass: true
- survival_quality: true

### C vs A
- overall_pass: true
- artifact_coherence: true
- pd_quality: true
- ab_no_regression: true
- fairness_relative: true
- conformal_promotion_pass: true
- survival_quality: true

## Important diagnostics
- Conformal strict statistical tests remain warnings (Kupiec/Christoffersen p-values), but promotion pass is true by policy (`conformal_promotion_pass=true`).
- B and C are effectively tied in PD/AB/fairness for smart run.
- C shows stronger survival metrics than A/B in this cycle.

## Operational run summaries
- `A_run_summary.json`
- `B_run_summary.json`
- `C_run_summary.json`

## Promotion decision
Promote **C** to `main` and continue iteration from `main`.
`full` profile remains optional and deferred due multi-day cost.
