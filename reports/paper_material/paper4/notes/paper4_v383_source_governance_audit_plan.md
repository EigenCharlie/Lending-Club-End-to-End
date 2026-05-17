# Paper 4 Source Governance Audit Plan v383

Generated: 2026-05-17T06:09:14.584644+00:00

v383 converts the v371-v373 source collapse into a targeted audit plan. It does
not relax source caps, apply a repair candidate, restart blind chunking or change
the Paper 4 solver claim scope.

## Audit Tasks

- P1 `audit_grade_a_cap_slack_boundary`: recompute cap slack, tolerance and binding status for grade=A
- P1 `audit_grade_a_flow_direction`: separate add-tight, drop-tight and neutral flows for grade=A
- P1 `audit_grade_a_relief_counterfactual`: quantify the return cost of any feasible relief-style move
- P2 `audit_score_decile_secondary_tightness`: rerun secondary source pass after grade explanation is isolated
- P2 `audit_sampled_chunk_representativeness`: compare sampled chunks by grade-A pressure and source-exact collapse
- P3 `audit_source_cap_contract`: document cap source, tolerance, rounding and family definitions
- P3 `audit_global_solver_implication`: map audit findings to future certificate requirements

## Evidence Summary

- Primary blocker family: `grade`.
- Secondary blocker family: `score_decile`.
- Grade family retention share: `0.0`.
- Grade-A relief return-improving rows: `0`.
- Sampled source-exact rows: `0`.

## Required Caveat

This is an audit plan, not a repair. It must not be used to claim source-cap
relaxation, global optimality, integer optimality, champion replacement or final
Paper 4 promotion.

## Next Executable Wave

Build `paper4_v384_formal_spo_dla_review_packet.md` while keeping formal SPO/DLA claims
blocked unless the review packet explicitly satisfies them.
