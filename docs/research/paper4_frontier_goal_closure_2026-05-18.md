# Paper 4 Frontier Goal Closure - 2026-05-18

## Decision

The bounded Paper 4 frontier goal is closed as a governed research pass, not a
new versioned wave. Each lane has one semantic decision artifact. No
`paper4_final_promotion.json` is created, no Paper Estrella champion is reopened,
and no new `paper4_v###` wave is introduced.

The implementation used one consolidated runner,
`scripts/papers/run_paper4_frontier_lanes.py`, and scratch data under
`/tmp/lc-paper4-goal-runs/`. Optional SPO/DFL dependencies were tested only in
`/tmp/lc-paper4-envs/spo/`.

## Lane Outcomes

| lane | decision | destination | key metrics | caveat |
| --- | --- | --- | --- | --- |
| ifrs9_sicr | append | appendix_raw_enriched_sicr_diagnostic | combined_default_lift=73.6416, combined_triggered_share=0.2936, cashflow_fields_present=True | No contractual days-past-due history, monthly account panel or macro scenario path. |
| online_conformal | park | appendix_source_family_holdout | best_method=source_grade_shrinkback, defended_min_coverage=0.7788, avg_width=0.5968, n_test=36000 | Historical issue-month replay only; no external source distribution or live feedback. |
| cvar_oce | append | appendix_tail_challenger_cashflow_check | best_tail_policy=cashflow_cvar_tail_proxy, best_tail_cvar95=416.6667, economic_cvar95=22084.6233, return_gap_vs_official_champion=-68875.5515 | Uses retrospective realized cashflows for challenger diagnosis; no full-universe exact optimum. |
| cate_policy_value | park | lab_notebook_only | aipw_ate=0.007132, overlap_share_10_90=0.2838, placebo_aipw_ate=-0.002225 | No rejected applicants, randomized pricing instrument or policy counterfactual. |
| fair_lending_proxy | append | appendix_source_governance_only | top_dispersion_dimension=grade, top_interest_rate_range=20.8492, legal_claim_allowed=False | BISG-style race/ethnicity proxy is not feasible with zip3/state only and no surname. |
| dla_adp | append | appendix_rollout_only | best_existing_policy=v11_adp_return_recovery, best_delta_state_value_vs_static=214521.2201, state_features_profiled=5 | Last-payment and outstanding fields are snapshots/proxies, not a transition panel. |
| spo_dfl | park | lab_notebook_isolated_prototype | toy_oracle_gap=4329509.727, pyepo_version=1.1.1, cvxpylayers_version=1.1.0 | Toy top-k probe is not SPO+ theorem or production optimization training. |

## Source Log

The reviewed research and implementation references are consolidated in
`reports/paper_material/paper4/tables/paper4_frontier_source_log_2026-05-18.csv`.
The most important consequences are:

- IFRS9/SICR: survival/lifetime-ECL literature supports the direction, but the
  Lending Club data still lacks contractual monthly DPD and macro scenario paths.
- Online conformal: ACI and multi-source conformal work justify source-aware
  diagnostics, but the project has retrospective history rather than live
  feedback.
- CATE: DoWhy/EconML-style checks reaffirm that accepted-loan observational
  sensitivity is not policy value.
- Fair lending: CFPB BISG methodology requires surname and richer geocoding, so
  zip3/state governance cannot become a legal fair-lending claim.
- SPO/DFL: PyEPO/cvxpylayers installed in isolation, but the toy oracle gap and
  dependency surface argue against integrating the lane into CRPTO.

## What Moves Forward

- Paper 4 can cite the raw-enriched IFRS9/SICR diagnostic only as proxy ECL/SICR
  evidence, never as contractual IFRS9.
- Online conformal remains a retrospective source-family governance diagnostic.
- CVaR/OCE remains a tail challenger; official economic champion replacement is
  still blocked by paired-wealth evidence.
- CATE is observational sensitivity only.
- Fair lending is source governance only, because protected attributes, surname
  and tract-level geocoding are absent.
- DLA/ADP is rollout-only, not exact Bellman optimality.
- SPO/DFL remains isolated-prototype/oracle-regret evidence and is not integrated
  into the main CRPTO pipeline.

## Future Work Gate

Reopen a lane only with one of: a servicing panel with monthly DPD/state paths,
rejected-applicant or randomized pricing data, approved protected-attribute proxy
inputs, a reviewer request, or a venue-driven revision that changes the claim.
