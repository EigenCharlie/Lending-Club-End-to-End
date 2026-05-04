# Paper Estrella P1 Evidence - 2026-05-04

This dossier records the P1 evidence now materialized around the official
`paper-thesis-final-economic-2026-04-06` champion. It does not reopen the
champion search.

## Generated artifacts

- `reports/paper_material/paper1/tables/paper1_tableA3_nested_holdout.csv`
- `reports/paper_material/paper1/tables/paper1_tableA3_nested_holdout.tex`
- `reports/paper_material/paper1/tables/paper1_tableA4_segment_period_sensitivity.csv`
- `reports/paper_material/paper1/tables/paper1_tableA4_segment_period_sensitivity.tex`
- `reports/paper_material/paper1/tables/paper1_tableA5_decision_aware_selector.csv`
- `reports/paper_material/paper1/tables/paper1_tableA5_decision_aware_selector.tex`
- `reports/paper_material/paper1/tables/paper1_tableA6_synthetic_shift.csv`
- `reports/paper_material/paper1/tables/paper1_tableA6_synthetic_shift.tex`
- `models/paper1_p1_evidence_status.json`
- `docs/research/paper_estrella_p1_evidence_2026-05-04.md`

## Scope notes

- The nested-holdout evidence is an artifact-level staged confirmation
  chain: 5K screening, 25K refinement, and 276K full OOT confirmation. It
  is stronger than a single final table, but it is not a fresh strict
  disjoint funded-set split.
- The decision-aware conformal selector is a CROMS-style screen over the
  three conformal finalists plus the final exact bound-aware champion.
  Only rank 1 has final 276K exact bound-aware metrics because ranks 2 and
  3 failed the conformal policy gate.
- Synthetic shift checks are covariate-reweighting stress scenarios on OOT
  labels; they are not an external dataset replacement.

## Key status

- Nested final return: `170464.542928`.
- Nested final V: `0.036450`.
- Decision-aware selected rank: `1`.
- Worst segment coverage 90: `0.903203`.
- Worst synthetic coverage 90: `0.929714`.
