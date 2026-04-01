# Conformal Gap Stage C — ARCHIVED 2026-03-16

## Summary

12 Stage C gap-correction variants were evaluated during the paper-grade run (2026-03-13/14).

**Decision: REJECTED — `score_decile_mondrian` retained as conformal champion.**

## Why Stage C was Not Promoted

All 12 candidates failed the `min_group_coverage_90` gate:

| Metric | Best Stage C | Champion (`score_decile_mondrian`) | Gate |
|--------|-------------|-------------------------------------|------|
| coverage_90 | 0.9038 | 0.9283 | ≥0.90 |
| min_group_coverage_90 | **0.8624** | **0.8873** | ≥0.88 |
| avg_width_90 | 0.7795 | 0.7569 | — |
| winkler_90 | 1.1908 | 1.2032 (justified) | ≤1.22 |
| methodological_justification_pass | **False** | **True** | — |

The gap correction improved Winkler (1.19 vs 1.20) but degraded min group coverage
(0.8624 < 0.88 target). The champion already passes the compensated Winkler band
(1.2032 ≤ 1.22 with coverage ≥0.92 compensation) so the Stage C tradeoff is not
favorable.

## Namespace Patterns Explored

- `gap_stage_c_{1,2,3}_grade_scaled0_mgs200_alpha009_floor092_t{0,1}_ts{250,500}`
  - Variants: 3 stage weights × 2 temporal flags × 2 temporal min sizes = 12 total

## Artifacts

- Per-namespace status: `models/conformal_gap/{namespace}/conformal_policy_status.json`
- Experiment orchestration: `models/pd_conformal_gap_experiment_status.json`
- Ranking summary: `models/conformal_gap_summary.json`

## Future Work

If min_group_coverage_90 must be improved, the correct approach is grouping-variant
selection (e.g., score_decile within-grade) rather than post-hoc gap correction.
See `docs/backlog-papers-unified.md` Paper 3 research items.
