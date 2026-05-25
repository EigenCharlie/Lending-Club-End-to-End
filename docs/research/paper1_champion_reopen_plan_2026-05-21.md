# Paper Estrella Champion Reopen Plan 2026-05-21

## Objective

Reopen the Paper Estrella champion search only where the new evidence can plausibly replace
the frozen champion and improve the theoretical bound. The target is not another broad
artifact loop; the target is one governed challenger package with a promote / append / park
decision.

## Current Evidence

External PD search produced a credible replacement candidate:

- main challenger: `full_challenger_woe__bureau_behavior_15`
- AUC: `0.720679` vs incumbent replay `0.712678` (`+0.008001`)
- Brier: `0.153161` vs `0.154591` (`-0.001430`)
- ECE: `0.007689` vs `0.006152` (`+0.001537`)

Sensitivity candidates remain useful:

- `full_challenger__canonical_4`: AUC `0.720624`, Brier `0.153182`, ECE `0.005917`
- `full_challenger_woe__affordability_rate_5`: AUC `0.720052`, Brier `0.153276`, ECE `0.007502`

The external PD model expands the local champion contract from 42 to 106 features, adding
64 bureau / behavior / WOE features while retaining all current local features. This is
predictive upside with governance cost, so promotion requires downstream proof, not AUC alone.

The external conformal package is usable but not final. Grade E misses the strict 90% group
gate, and E/G are weak at 95%. Therefore the next search should focus conformal before
portfolio promotion.

## Non-Negotiable Gates

1. PD gate
   - AUC improvement at least `+0.005` vs frozen incumbent replay.
   - Brier improvement or non-inferiority.
   - ECE not worse by more than `+0.0025`, or a sensitivity candidate with better ECE must remain competitive downstream.
   - No monotonicity, threshold, or governance regression that invalidates the MRM story.

2. Conformal gate
   - Global 90/95 coverage passes.
   - Minimum grade coverage at 90% passes strict floor, with grade E explicitly resolved or caveated.
   - Grade E/F/G 95% weakness improves or is proven irrelevant to funded-set risk.
   - Width/Winkler does not inflate enough to destroy portfolio value.

3. Portfolio gate
   - Full-universe parent run, not 25k quick signal.
   - Exact `alpha01` and `alpha03` pass.
   - `violation = 0`.
   - `V <= sqrt(alpha)` at alpha01.
   - Robust return beats the frozen economic champion on the same universe, or improves bound metrics enough that it becomes an appendix challenger rather than a replacement.

4. Bound-improvement gate
   - `Gamma_CP` falls, or `V` falls, or the same return is achieved with cleaner proof surface.
   - Funded-set group weights do not hide subgroup failure.
   - A sealed/nested confirmation run remains possible after candidate selection.

## Reopen Sequence

### Phase 0: Stage PD Challenger Locally

Stage only the three external PD finalists into `models/search_pd/<run_tag>/` as candidate
artifacts. Do not overwrite canonical PD artifacts.

Expected main local tag:

`regret_auditability_pd_bureau_behavior_15_2026_05_21`

Required staged files:

- `pd_candidate_model.cbm`
- `pd_candidate_calibrator.pkl`
- `pd_model_contract.json`
- `pd_training_record.pkl`
- `pd_hpo_seed_replay_status.json`
- `test_predictions.parquet`

This makes existing conformal code work via `UPSTREAM_CANONICAL_RUN_TAG` without changing
canonical champion files.

### Phase 1: Focused Conformal Reopen

Run conformal reopen in the main env, using the staged PD candidate as upstream.

Primary profile:

`search_conformal_reopen_exhaustive`

Search intent:

- keep Venn-Abers / calibrated and raw sources;
- compare `grade`, `score_decile_mondrian`, `grade_x_scoreband_mondrian`;
- force attention to grade E/F/G and temporal slices;
- prefer configurations that reduce `Gamma_CP` and keep funded-set coverage, not only global coverage.

Promotion output:

- one winning conformal namespace;
- one comparison table vs external child conformal and frozen champion conformal;
- one compact memo explaining whether grade E/G got resolved.

### Phase 2: Portfolio Reopen With cuOpt Frontier + Exact Rerank

Use the RAPIDS env for frontier generation and delegate exact HiGHS rerank to the main env when
needed.

Reason: `rapids` has `cuOpt`, but not `highspy`; the main env has HiGHS and project-native exact
validation.

Run in waves:

1. Smoke compatibility
   - `max_candidates=25000`
   - shortlist 80-120
   - alpha grid `0.01,0.03,0.10`
   - objective: confirm new conformal artifact loads and alpha01 can pass.

2. Medium frontier
   - `max_candidates=100000` or `150000`
   - shortlist 220-320
   - wider grids around incumbent and promising quick region.
   - objective: find whether return/bound tradeoff is competitive.

3. Full universe
   - `max_candidates=0`
   - shortlist 320-500
   - full alpha grid `0.01,0.02,0.03,0.05,0.10,0.15,0.20`
   - objective: promotion decision against frozen `bound_aware_276k_economic_champion`.

### Phase 3: Bound Hardening

Only after Phase 2 selects a serious candidate:

- funded-set Mondrian bound: compute `sum_g W_g alpha_g` for selected loans;
- decision-aware selector audit: compare conformal finalists by `return`, `V`, `Gamma_CP`, coverage, width, group gates;
- dependency-aware diagnostic: cluster by `issue_month`, `grade`, and state/source proxy;
- nested/sealed confirmation: run final validation on a predeclared temporal/source holdout;
- direct CRC/LTT loss: prototype only if it can change the main theorem or appendix bound.

## Stop Rules

Stop and park the challenger if any of these happen:

- PD improvement fails to survive conformal/portfolio downstream.
- Conformal fixes E/G only by making intervals so wide that `Gamma_CP` or return becomes unusable.
- Full-universe portfolio cannot beat the frozen champion and does not improve `V`/`Gamma_CP` meaningfully.
- The only remaining improvement requires unsupported IFRS9/ECL, live deployment, or legal fair-lending claims.

## Promotion Outcomes

- Promote: replace champion in Paper Estrella only if full-universe return and bound metrics beat current champion under exact validation.
- Append: keep as stronger challenger appendix if PD/conformal improves but portfolio does not replace champion.
- Park: keep for Paper 4 if it teaches a method lesson but does not change Paper Estrella.
- Delete/archive: discard scratch runs and repeated variants that do not change the decision.
