# Paper 4 Papers_tesis conformal candidate diagnostics - 2026-06-06

## Protocol

- **Claim target:** test whether group-weighted, utility-directed or localized
  conformal candidates can improve Paper 4 source/coverage diagnostics.
- **Split:** frozen v4 replay table, 2018 rows for calibration and 2019--2020
  rows for holdout.
- **Gate:** holdout coverage >= 0.90, defended-source coverage >= 0.80 and
  average interval width <= 0.98.
- **Stop rule:** append diagnostic evidence only. Do not modify the Paper
  Estrella champion and do not make online deployment, legal fairness or exact
  conditional-coverage claims.

## Summary

| variant | coverage | avg_width | worst defended source coverage | width delta vs Mondrian | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| source_aware_guarded_v4_context | 0.9713 | 0.9743 | 0.9013 | 0.1967 | append_context_reference |
| localized_score_replay | 0.9497 | 0.8643 | 0.9252 | 0.0867 | append_mixed_diagnostic_wider_than_mondrian |
| group_weighted_source_max_replay | 0.9387 | 0.9394 | 0.8602 | 0.1618 | append_mixed_diagnostic_source_not_better |
| mondrian_grade_temporal_replay | 0.9244 | 0.7775 | 0.8713 | 0.0000 | retain_baseline |
| utility_directed_loss_replay | 0.9954 | 0.9981 | 0.9499 | 0.2205 | park_gate_fail |

Best diagnostic row by the predeclared sort is `source_aware_guarded_v4_context`. The result
is appendix-scoped: it can support a Paper 4 source/shift conformal discussion,
but it does not replace the CRPTO champion protocol.

## Gate Register

| paper | candidate | decision | absolute gate pass |
| --- | --- | --- | --- |
| Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction | group_weighted_source_max_replay | append_mixed_diagnostic_source_not_better | True |
| Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction | utility_directed_loss_replay | park_gate_fail | False |
| Guan 2023 - Localized Conformal Prediction | localized_score_replay | append_mixed_diagnostic_wider_than_mondrian | True |
