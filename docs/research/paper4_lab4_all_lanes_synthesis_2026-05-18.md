# Paper 4 Lab 4 All-Lane Synthesis - 2026-05-18

## Decision

The literature-driven Lab 4 pass is closed as a Paper 4 living-lab synthesis.
The work uses retained Paper 4 artifacts and source-review evidence, but it does
not select a policy and does not create a Paper 4 final promotion artifact. It
does not extend the active external CRPTO IJDS identification contract.

- Append/governance lanes: `6`; negative-audit lanes: `1`.
- Parked lanes: `2`.
- Paper Estrella export: limited to possible reviewer-defense, caveat or
  model-risk context; no new champion claim.

## Lane Decisions

| lane | decision | final sink | Paper 1 export | key result |
| --- | --- | --- | --- | --- |
| lane1_crc_ltt_decision_loss_gates | append | paper4_appendix_or_governance | possible_limited_context_only | 22 policies pass operational CRC/LTT gates; 1 also pass source-hardening. |
| lane2_croms_lite_selection | append | paper4_appendix_or_governance | no_export_now | Source-defended selector chooses crpto_rt0p155_g0p45_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 with return delta -3842.90 vs official. |
| lane5_cvar_oce_tail_challenger | append | paper4_appendix_or_governance | possible_limited_context_only | CVaR challenger reduces loss p95 by 36960.02 but prob beats reference is 0.4766. |
| lane3_e2e_conformal_calibration | park | parked_with_boundary | no_export_now | No promotable E2E conformal training artifact exists in the current Lab 4 surface. |
| lane4_online_multisource_conformal | park_with_appendix_limitation | paper4_appendix_or_governance | no_export_now | Best bounded defended min coverage is 0.7788. |
| lane6_spo_dfl_comparator | park_integrated_dfl_append_oracle_regret | paper4_appendix_or_governance | no_export_now | Toy oracle gap remains 4329509.73. |
| lane7_ifrs9_proxy | diagnostic_negative_audit | paper4_negative_audit_only | no_export_now | Available terminal and snapshot fields do not identify report-date PD, ECL, SICR or staging. |
| lane8_governance_fairness_proxy | append | paper4_appendix_or_governance | possible_limited_context_only | Legal fair-lending claim allowed = False. |
| lane9_causal_cate_boundary | park | parked_with_boundary | no_export_now | Overlap share 10-90 is 0.2838. |

## Selector Table

The compact selector table is useful because it makes the CROMS-lite result
readable: source robustness, balanced decision risk and official economic
champion objectives choose different policies. This is exactly Paper 4
material: tradeoff evidence, not a promotion protocol.

| selector | selected policy | delta return | delta worst source | Paper 4 use |
| --- | --- | ---: | ---: | --- |
| official_paper1_champion | paper1_economic_champion | 0.00 | 0.0000 | reference champion |
| source_defended_return | crpto_rt0p155_g0p45_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | -3842.90 | 0.3636 | source-governance tradeoff |
| croms_lite_balanced_score | crpto_rt0p175_g0p50_u0p05_alpha0p01_conservative_proxy_rs42 | -1663.23 | 0.2500 | CROMS-lite tradeoff evidence |
| coverage_only_source | crpto_rt0p160_g0p55_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | -6616.55 | 0.3696 | coverage-only negative control |
| diagnostic_selector_rank | crpto_rt0p160_g0p55_u0p05_alpha0p01_incumbent_region_rs42 | -6191.19 | 0.1667 | diagnostic selector anchor |

## What Enters Paper 4

- Lane 1 enters only as governance: CRC/LTT-style gates document what passes
  risk, return, satisficing and source-hardening checks.
- Lane 2 enters as a selector tradeoff table, because it shows why objective
  choice matters and why no selector automatically promotes a new champion.
- Lane 5 enters as a strong tail-risk appendix: CVaR/OCE improves tail views
  but does not beat paired wealth robustly.
- Lane 7 enters only as a negative estimand/readiness audit. Historical lifts,
  thresholds and monetary values are non-citable and support no PD, ECL, SICR
  or staging claim.
- Lane 8 enters as source/proxy governance, explicitly not legal fair-lending
  evidence.

## What Stays Parked

- Lane 3 is parked because there is no end-to-end learned conformal calibration
  artifact and the main environment is not the isolated differentiable stack.
- Lane 4 is parked as a live/online claim because defended coverage is close
  but below gate and the available views are post-selection slices, not strict
  holdouts.
- Lane 6 is parked for integrated DFL/SPO+ because current evidence is
  oracle-regret/surrogate only.
- Lane 9 is parked because CATE/policy value remains blocked by accepted-loan
  selection, weak overlap and unresolved identification.

## Stop Rules

- Do not create `paper4_final_promotion.json`.
- Do not create `paper4_v###` follow-up waves from this synthesis.
- Reopen a parked lane only with new data, an isolated working prototype, a
  formal proof, or a reviewer request that changes a manuscript claim.
