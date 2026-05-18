# Paper 4 Lab 4 Lanes 1, 2 and 5 Execution Memo - 2026-05-18

## Scope

This execution treats Lab 4 as the full living-lab surface of Paper 4: all retained
Paper 4 artifacts, Paper 1 champion references and curated frontier diagnostics can
be used as inputs. The outputs remain Lab 4 evidence until a later promote/append/
park/delete decision is made.

No `paper4_v###` artifacts, promotion JSONs or per-iteration status packets were
created. The official Paper Estrella champion remains protected.

## Lane 1 - CRC/LTT Decision-Loss Gates

- Operational CRC/LTT gate passes: `22` policies.
- Source-hardened passes: `1` policies.
- Official champion lane decision: `protect_official_champion_source_caveat`.

Top source-hardened or near-source-hardened policies:

| policy | return | V | Gamma_CP | worst source | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| crpto_rt0p155_g0p45_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | 166621.64 | 0.0418 | 0.1575 | 0.8636 | append_source_hardened_gate_pass |
| paper1_economic_champion | 170464.54 | 0.0365 | 0.1859 | 0.5000 | protect_official_champion_source_caveat |
| crpto_rt0p170_g0p45_u0p10_alpha0p01_incumbent_region_rs42 | 169389.58 | 0.0355 | 0.1776 | 0.6667 | append_operational_pass_source_fragile |
| crpto_rt0p170_g0p45_u0p05_alpha0p01_incumbent_region_rs42 | 169337.16 | 0.0355 | 0.1791 | 0.6667 | append_operational_pass_source_fragile |
| crpto_rt0p170_g0p45_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | 169262.38 | 0.0442 | 0.1802 | 0.5000 | append_operational_pass_source_fragile |

Interpretation: the CRC/LTT framing is useful as governance. It creates explicit
pass/fail gates for risk control, operational return, satisficing and source
hardening. It does not by itself promote any challenger to Paper Estrella.

## Lane 2 - CROMS-Lite Selection Audit

| selector | selected policy | delta return vs official | delta worst source vs official | decision |
| --- | --- | ---: | ---: | --- |
| official_paper1_champion | paper1_economic_champion | 0.00 | 0.0000 | paper1_protected_reference |
| diagnostic_selector_rank | crpto_rt0p160_g0p55_u0p05_alpha0p01_incumbent_region_rs42 | -6191.19 | 0.1667 | append_tradeoff_evidence_not_promotion |
| return_subject_to_crc_ltt | paper1_economic_champion | 0.00 | 0.0000 | paper1_protected_reference |
| source_defended_return | crpto_rt0p155_g0p45_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | -3842.90 | 0.3636 | append_tradeoff_evidence_not_promotion |
| coverage_only_source | crpto_rt0p160_g0p55_u0p00_alpha0p01_forced_incumbent_neighbors_rs42 | -6616.55 | 0.3696 | append_tradeoff_evidence_not_promotion |
| min_gamma_cp_near_champion | crpto_rt0p155_g0p55_u0p05_alpha0p01_incumbent_region_rs42 | -7710.88 | 0.0000 | append_tradeoff_evidence_not_promotion |
| croms_lite_balanced_score | crpto_rt0p175_g0p50_u0p05_alpha0p01_conservative_proxy_rs42 | -1663.23 | 0.2500 | append_tradeoff_evidence_not_promotion |
| tail_return_source_proxy | paper1_economic_champion | 0.00 | 0.0000 | paper1_protected_reference |

Interpretation: CROMS-lite is valuable as a selector audit. Different objectives
select different policies, especially when source-family robustness is made hard.
That is Paper 4 evidence, not a full CROMS implementation and not a Paper 1
promotion protocol.

## Lane 5 - CVaR/OCE Tail Challenger

| evidence block | candidate | primary metric | primary value | decision |
| --- | --- | --- | ---: | --- |
| paper1_official_tail_profile_lgd45 | paper1_economic_champion | funded_set_repriced_return | 173329.6634 | paper1_reference_only |
| paired_common_path_champion_vs_cvar | v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000 | prob_challenger_beats_reference | 0.4766 | append_tail_challenger_retain_champion |
| raw_cashflow_tail_proxy | cashflow_cvar_tail_proxy | cvar95_loan_loss_reduction_vs_cashflow_economic | 21667.9567 | append_cashflow_challenger_not_promotion |
| restricted_master_cvar_frontier | v13_cvar_mdcp_colgen_relaxed_k32000_floor105000_cap300000 | tail_champion_score_v33 | 0.5833 | append_restricted_master_only |
| curated_local_frontier_probe | expanded_branch_price_candidate | strict_return_cvar_improvement_vs_predecessor_v467 | True | append_local_probe_boundary |

Interpretation: CVaR/OCE remains the strongest quantitative challenger lane.
The tail evidence is real, but paired wealth dominance is absent, so it belongs
in Paper 4 as a tail challenger appendix and, at most, in Paper Estrella as a
robustness caveat.

## Consolidated Decisions

| lane | decision | Paper 4 destination | Paper 1 destination |
| --- | --- | --- | --- |
| lane1_crc_ltt_decision_loss_gates | append | lab4_governance_appendix | possible_reviewer_defense_only |
| lane2_croms_lite_selection | append | lab4_selector_tradeoff_appendix | none_now |
| lane5_cvar_oce_tail_challenger | append | lab4_tail_challenger_appendix | possible_appendix_caveat_only |

## Stop Rules

- Do not reopen the official champion from these lanes.
- Do not treat source-holdout replay as live online validity.
- Do not call CROMS-lite a full implementation of CROMS.
- Do not call CVaR/OCE an economic champion unless paired wealth dominance is shown.
