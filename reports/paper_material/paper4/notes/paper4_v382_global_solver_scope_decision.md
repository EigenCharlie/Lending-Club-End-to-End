# Paper 4 Global Solver Scope Decision v382

Generated: 2026-05-17T06:03:37.940547+00:00

## Decision

Paper 4 should report the solver lane as a bounded frontier plus negative/gap
evidence. It must keep full-v55 global optimality and full-universe integer
optimality prohibited.

## Why

- v71/v363 still record `5738`
  improving omitted columns.
- v363 has `5` open certificate
  requirements.
- v373 sampled `8` chunks and found
  `0` source-exact rows.

## Open Certificate Route

- `full_omitted_universe_priced_or_excluded`: v361 source-tight pool covers only a bounded subset of full omitted universe (next: `paper4_v364_v353_dual_bound_resource_plan.csv`).
- `direct_full_mip_guard_met`: direct full-v55 binary solve remains above guard (next: `paper4_v364_v353_dual_bound_resource_plan.csv`).
- `restricted_dual_screen_terminated_without_negative_rc`: v71 restricted-master dual pricing is not termination (next: `paper4_v364_v353_dual_bound_resource_plan.csv`).
- `all_column_pricing_terminated`: no full-v55 branch-price termination proof exists (next: `paper4_v364_v353_dual_bound_resource_plan.csv`).
- `integer_optimality_certificate_available`: continuous or bounded pricing evidence is not integer proof (next: `paper4_v364_v353_dual_bound_resource_plan.csv`).

## Required Caveat

This decision supports bounded/gap manuscript language only. It must not be used
to claim global optimality, integer optimality, live deployment, legal/IFRS9
compliance, Paper Estrella replacement or final Paper 4 promotion.

## Next Executable Wave

Build `paper4_v383_source_governance_audit_plan.csv` from the v373 sampled source-screen
evidence.
