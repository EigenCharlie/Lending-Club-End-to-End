# Paper 1 Regret-Auditability Sandbox Closure - 2026-05-28

> **HISTORICAL / SUPERSEDED — NO TRANSFER TO ACTIVE CRPTO (2026-07-20).** This memo preserves a past project decision or proposal. It does not govern the autonomous external CRPTO dossier, select a learner/comparator/policy, or reactivate Paper 2/Paper 4 claims. Current authority: `SESSION_STATE.md`, `docs/research/crpto_external_contract_2026-07-20.yml` and `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md`.

## Purpose

This memo closes the external CRPTO regret-auditability sandbox intake for the
parent Lending Club project. The sandbox was created outside this repository at:

`/mnt/d/crpto_experiments/regret_auditability/regret_auditability_20260513_v3_resource_tuned`

The question was whether a much broader CatBoost monotone + Optuna +
Venn-Abers + Mondrian conformal + robust portfolio search could improve the
Paper 1 CRPTO champion, or at least strengthen the paper's evidence around the
regret-auditability frontier.

## Closure Decision

The sandbox was useful, but it does not replace the frozen economic champion by
itself.

It produced credible PD and conformal challenger evidence, triggered a governed
parent-project champion tournament, and generated negative-result evidence that
is useful for IJDS-style documentation. However, the downstream portfolio and
bound evidence seen so far does not produce a clean new champion that dominates
the frozen policy on return, `V`, `Gamma_CP`, violation, and coverage at the
same time.

Frozen champion reference:

- policy artifact: `models/portfolio_bound_aware/rank1_alpha01_bound_aware_276k_full_2026-04-05-1734/portfolio_bound_aware_selection.json`
- realized return: `170464.5429284627`
- `V`: `0.03645`
- `Gamma_CP`: `0.18591`
- violation: `0`
- funded coverage: `0.9433`
- policy: `blended_uncertainty`, risk `0.175`, gamma `0.45`, uncertainty aversion `0.1`

## What Was Imported Into The Parent Project

The parent project already absorbed the useful sandbox material through these
paper-facing and research artifacts:

- `docs/research/paper1_bound_improvement_intake_2026-05-21.md`
- `docs/research/paper1_champion_reopen_plan_2026-05-21.md`
- `docs/research/paper1_crpto_ijds_champion_tournament_protocol_2026-05-25.md`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_pd_intake_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_bound_improvement_conformal_config_candidates_2026-05-21.csv`
- `reports/paper_material/paper1/tables/paper1_conformal_reopen_candidate_gap_diagnostics_2026-05-25.csv`
- `reports/paper_material/paper1/tables/paper1_bound_pareto_decision_summary_2026-05-25.csv`

The parent project also staged the main external PD challengers under:

- `models/search_pd/regret_auditability_pd_bureau_behavior_15_2026_05_21/`
- `models/search_pd/regret_auditability_pd_affordability_rate_5_2026_05_23/`
- `models/search_pd/regret_auditability_pd_canonical_4_2026_05_23/`

## PD Findings

The strongest sandbox contribution was the PD search. It showed that the frozen
PD stack was not the predictive ceiling.

| Role | Candidate | AUC | Brier | ECE | Parent use |
| --- | ---: | ---: | ---: | ---: | --- |
| incumbent | `incumbent__frozen_champion` | `0.712678` | `0.154591` | `0.006152` | frozen reference |
| challenger | `full_challenger_woe__bureau_behavior_15` | `0.720679` | `0.153161` | `0.007689` | main challenger |
| challenger | `full_challenger__canonical_4` | `0.720624` | `0.153182` | `0.005917` | sensitivity baseline |
| challenger | `full_challenger_woe__affordability_rate_5` | `0.720052` | `0.153276` | `0.007502` | sensitivity baseline |

Interpretation:

- `bureau_behavior_15` is the best pure discrimination/Brier signal, but its ECE
  is worse than the incumbent and it has a higher feature-governance burden.
- `canonical_4` is a cleaner governance sensitivity: nearly the same AUC lift as
  `bureau_behavior_15`, better Brier than incumbent, and better ECE than the
  incumbent replay.
- `affordability_rate_5` is useful as a business-story challenger because it
  tests whether affordability monotonicity and WOE transformations add a stable
  signal.

This evidence is strong enough for a challenger appendix and for a gated
champion-reopen protocol, but not enough by itself to change the Paper 1 claim.

## Conformal Findings

The sandbox selected a usable but non-final conformal configuration:

- partition: `grade`
- probability source: `raw`
- score bins: `5`
- fallback: `grade_then_global`
- alpha90: `0.075`
- alpha95: `0.06`
- minimum group size: `100`
- score scaling: `bernoulli_sqrt`

The parent conformal follow-up showed why this should remain a challenger rather
than a direct promotion:

| Candidate | coverage90 | min group coverage90 | avg width90 | worst group | Decision reading |
| --- | ---: | ---: | ---: | --- | --- |
| `affordability_rate_5` | `0.944317` | `0.916647` | `0.806270` | `score_q00` | viable but wider |
| `bureau_behavior_15` | `0.919951` | `0.870059` | `0.749615` | `E` | rare-grade weakness |
| `canonical_4` | `0.931878` | `0.917582` | `0.790729` | `score_q04` | viable sensitivity |
| `official_champion` | `0.929714` | `0.918983` | `0.784230` | `score_q03` | still balanced |

Interpretation:

- The conformal search was valuable because it revealed where PD improvements
  transfer cleanly and where they create rare-grade weaknesses.
- `bureau_behavior_15` is predictive, but the grade `E` weakness makes it hard
  to promote without further conformal repair.
- `canonical_4` and `affordability_rate_5` are better downstream sensitivity
  candidates than their AUC ranking alone would suggest.

## Portfolio And Bound Findings

The portfolio layer is where the frozen champion remains strongest as a balanced
paper-facing claim.

The parent decision table contains 73 exact/pass decision rows:

- `35` append-or-park rows with no champion case
- `20` Gamma-only challengers with worse V/return
- `9` V-only challengers with worse Gamma/return
- `7` bound-only challengers with worse return
- `1` return-only challenger with worse bounds
- `1` official baseline row

The best positive-return challenger found in the parent evidence was:

- candidate: `canonical_4_return_aware`
- run: `paper1_bound_expansion_2026_05_24_r1__canonical_4__medium_triage_resume_75k__return_aware_alpha01_2026_05_25_r5`
- return: `170611.34163424745`
- return delta vs champion: `+146.79870578474947`
- `V`: `0.058675`
- `Gamma_CP`: `0.270366`
- violation: `0`
- decision: `return_challenger_only_bound_worse`

The best V/Gamma challengers improved one bound dimension but paid too much in
return or worsened the other bound dimension. This is useful negative evidence:
the broad search did not find a free lunch.

## Scientific Value For IJDS

The sandbox improves the Paper 1 evidence package in four ways:

1. It supports an anti-cherry-pick story. The project did not stop at the first
   champion; it reopened PD, conformal, and portfolio under explicit gates.
2. It clarifies the regret-auditability frontier. Higher-return candidates
   exist, but they tend to weaken `V`, `Gamma_CP`, funded coverage, or group
   coverage.
3. It provides credible negative results. These are publishable as appendix or
   robustness evidence because they show why the frozen champion remains the
   main balanced policy.
4. It separates predictive improvement from decision improvement. Better AUC
   does not automatically imply a better robust portfolio under conformal
   guarantees.

Recommended paper framing:

- main text: keep the frozen economic champion as the primary CRPTO result.
- appendix: report the governed reopen/tournament as robustness and
  sensitivity evidence.
- Paper 4 or methods appendix: use the regret-auditability frontier and PyEPO
  regret suite to discuss the difference between decision efficiency and
  auditability.

## What Should Not Be Claimed

Do not claim that the sandbox produced a new champion unless a later sealed
full-universe confirmation passes all champion replacement gates.

Do not compare the child 25k quick portfolio return directly against the frozen
276k champion.

Do not promote `bureau_behavior_15` on AUC alone while the conformal rare-grade
weakness remains unresolved.

Do not treat high-return portfolio probes as paper champions when their
`V`/`Gamma_CP` trade-off is worse than the frozen policy.

## Remaining Optional Work

Only needed if the project still wants to pursue replacement rather than close
the lane:

- run a sealed full-universe cuOpt + HiGHS rerank for a small predeclared set;
- repair or explicitly park the `bureau_behavior_15` grade `E` conformal issue;
- decide whether `canonical_4` becomes the main appendix challenger because it
  has the cleanest PD/calibration/conformal balance;
- export a final negative-results registry for the IJDS appendix;
- avoid further open-ended search unless the protocol version is reopened before
  seeing downstream results.

## Final Closure Note

The sandbox should be considered successful as evidence generation, not as a
champion replacement. It gave the parent project stronger PD challengers, a
more rigorous conformal/portfolio tournament, and a clearer empirical case for
the CRPTO paper's central tension: robust auditability can be bought, but it is
not free in return/regret space.
