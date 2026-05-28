# Paper CRPTO / IJDS Champion Tournament Protocol - 2026-05-25

## Purpose

This protocol reopens the Paper CRPTO champion search as a governed tournament,
not as another artifact loop. The target claim is:

`calibrated PD -> Mondrian conformal interval -> uncertainty set -> robust LP -> auditable policy`

The search may replace the frozen economic champion only if a predeclared
candidate wins the complete downstream trade-off. Better AUC alone, better
return alone, or better bound alone is not sufficient.

Frozen champion:

- artifact: `models/portfolio_bound_aware/rank1_alpha01_bound_aware_276k_full_2026-04-05-1734/portfolio_bound_aware_selection.json`
- return: `170464.5429284627`
- `V`: `0.03645`
- `Gamma_CP`: `0.18591`
- violation: `0`
- funded coverage: `0.9433`
- policy: `blended_uncertainty`, risk `0.175`, gamma `0.45`, uncertainty aversion `0.1`

## What This Is Testing

The question is not whether a new scorer can beat the old scorer. The question
is whether the full CRPTO chain can find a policy that is more valuable and at
least as defensible under IJDS standards:

1. PD is calibrated and auditable.
2. Conformal coverage is stable enough to become a decision input.
3. The robust LP converts uncertainty into a funded set with zero violation.
4. Exact alpha-grid validation confirms that the apparent frontier is real.
5. Nested or prospective confirmation shows the final winner was not selected
   by peeking across waves.
6. The paper can publish a claim-artifact-test map, negative-results registry,
   and final selection rule.

## Anti-Cherry-Pick Contract

Before any serious medium/full run, the run root must contain:

- `predeclared_candidate_registry.json`: every PD/conformal/portfolio lane that
  may compete for champion replacement.
- `phase_gate_status.json`: uniform gates for every lane.
- `selection_rule.json`: the final ranking rule and tie-breakers.
- `negative_results_registry.csv`: every failed, skipped, parked or appendix
  lane, with reason.

A late idea may be added only by opening a new protocol version before running
its downstream portfolio stage. Late ideas cannot enter the same tournament as
champion candidates after seeing portfolio results; they go to Paper 4,
appendix, or a future protocol.

## Candidate Lanes

### Lane A: Frozen Incumbent Replay

Purpose: anchor every comparison to the official claim.

Inputs:

- official PD/conformal/portfolio artifacts from `models/final_project_promotion.json`
- official champion policy and robust region

Role: comparator and sanity gate, not a new challenger.

### Lane B: External PD Finalists

Purpose: use the best recent PD evidence without pretending AUC is the final
objective.

Candidates:

- `canonical_4`
- `bureau_behavior_15`
- `affordability_rate_5`

Required gates:

- AUC, Brier, ECE, reliability and Gini reported separately.
- calibration deterioration bounded.
- monotonicity and feature-governance story remains auditable.
- downstream conformal/portfolio proxy must improve or preserve the IJDS claim.

### Lane C: Governance-Aware PD HPO

Purpose: search around the strongest finalists using modern CatBoost/Optuna
features while protecting auditability.

Allowed CatBoost features:

- `monotone_constraints` for mandatory economic monotonicity.
- `feature_weights` to prioritize economically interpretable affordability and
  capacity variables when defensible.
- `first_feature_use_penalties` and `penalties_coefficient` to discourage
  unstable or governance-costly features.
- `posterior_sampling` / Langevin only as an uncertainty or stability diagnostic
  unless a CPU reproducible path is demonstrated.

Optuna requirements:

- persistent storage per lane;
- no in-memory-only serious HPO;
- trial artifacts and failed trials retained in the run root;
- multi-objective or constrained objective that includes calibration and
  downstream proxies, not AUC alone;
- seed replay for top trials before conformal promotion.

Promotion from PD to conformal:

- top PD by constrained score;
- best calibration candidate;
- at least one governance/light-feature challenger if it is within tolerance;
- any PD that wins only AUC but fails calibration is appended or parked.

### Lane D: Calibration and Conformal Tournament

Purpose: reopen conformal before portfolio.

Primary candidates:

- `score_decile_mondrian`
- `grade`
- `grade_x_scoreband_mondrian`
- `vintage_x_scoreband` only if minimum cell size is sufficient

Calibration candidates:

- Venn-Abers as the main calibrated source when it remains competitive.
- Isotonic, Platt and beta as controlled alternatives.
- MAPIE risk-control APIs only as sidecar/prototype unless smoke tests show
  cleaner risk-control evidence without changing the paper's conformal contract.

Conformal selection metrics:

- global 90/95 coverage;
- minimum group coverage;
- E/F/G behavior;
- average width and Winkler;
- funded-set coverage proxy;
- `V` and `Gamma_CP` proxy;
- temporal coverage warnings.

Conformal phase champion:

The phase champion is not the highest coverage row. It is the best row passing
coverage, group, width, temporal and downstream-feasibility gates. The official
`score_decile_mondrian` lesson remains active: interpretability matters, but a
regulatory-looking grouping that fails group coverage cannot feed the robust LP.

### Lane E: Portfolio Frontier Tournament

Purpose: find a real downstream trade-off without exact-all CPU loops.

Cascade:

1. 25k smoke, uniform across all conformal finalists.
2. 50k/75k medium, only for smoke winners.
3. 100k/150k frontier, only for medium winners.
4. Full universe, only for predeclared finalists.
5. Exact alpha-grid rerank and sealed confirmation.

Solver policy:

- cuOpt runs the broad frontier in the RAPIDS env.
- batch size `1` is the default serious path until batch mode passes a separate
  smoke without instability.
- methods `Concurrent`, `PDLP` and `Barrier` can be tested as solver variants,
  but solver changes do not create new scientific candidates unless declared.
- exact rerank runs in `.venv` with HiGHS/highspy.
- HiGHS fallback is allowed for failure recovery, but fallback rows are labeled.

Portfolio grid:

- risk around champion: `0.165` to `0.200`;
- gamma around champion and return region: `0.275` to `0.600`;
- uncertainty aversion: `0` to `0.25`, with higher values only for tail-risk
  diagnostic lanes;
- policy families: `blended_uncertainty`, `capped_blended_uncertainty`,
  `tail_blended_uncertainty`, and segment-tail variants only if they preserve
  the main CRPTO claim.

### Lane F: Theory and Bound Hardening

Implement now:

- funded-set Mondrian refinement;
- decision-aware conformal selector audit;
- nested/prospective confirmation;
- regret and price-of-robustness table;
- bootstrap funded-set diagnostics if already supported.

Prototype or park:

- direct CRC/LTT on decision loss;
- dependence-aware concentration by cluster;
- online/shift-aware conformal;
- richer LGD/ECL targets;
- OCE/CVaR as a new objective.

## Phase Gates

### PD Gate

Advance only if:

- AUC improves or is within tolerance;
- Brier and ECE are non-inferior or better;
- calibration diagrams do not reveal a new material failure;
- monotonicity/governance constraints remain defensible;
- downstream conformal feasibility proxy is not degraded.

### Conformal Gate

Advance only if:

- global 90/95 coverage passes;
- min-group coverage passes the declared floor;
- rare grades are visible in diagnostics;
- width is not inflated enough to destroy portfolio value;
- temporal warnings are labeled and not hidden.

### Portfolio Gate

Advance only if:

- alpha01 exact pass is true;
- violation is zero;
- realized return is competitive;
- `V`, `Gamma_CP`, funded coverage and funded composition remain defensible;
- the result survives medium/full confirmation, not only 25k/50k probes.

### Champion Replacement Gate

Promote only if:

- full universe exact alpha01 pass is true;
- violation is `0`;
- return is at least `170464.5429284627`, preferably with a non-trivial margin;
- `V <= 0.03645`, or a clearly superior `Gamma_CP`/coverage trade-off is
  defended without materially worsening `V`;
- `Gamma_CP <= 0.18591`, or the bound trade-off is clearly superior;
- funded coverage is comparable or better;
- nested/prospective confirmation passes;
- the negative-results registry proves the selected winner was not cherry-picked.

## IJDS Evidence Packet If A New Champion Wins

Promote only these paper-facing artifacts:

- `reports/paper_material/paper1/tables/paper1_crpto_ijds_tournament_final_summary_YYYY-MM-DD.csv`
- `reports/paper_material/paper1/tables/paper1_crpto_ijds_negative_results_registry_YYYY-MM-DD.csv`
- `docs/research/paper1_crpto_ijds_champion_decision_memo_YYYY-MM-DD.md`
- final selection JSON/context only if the claim changes

The manuscript body may change only after the decision memo says `promote`.
Appendix receives high-value negative or theorem-tight evidence. Paper 4 receives
ideas that teach a method lesson but fail replacement gates.

## What Not To Re-run

- exact-all CPU grid;
- AUC-only HPO;
- `score8_raw_sqrt`, `grade_cal_sqrt`, or `score8_cal_none` as champion lanes
  after their prior downstream failures, except as diagnostic controls;
- online conformal without a serious temporal split;
- OCE/CVaR as replacement objective unless a new protocol explicitly targets
  a tail-risk paper, not IJDS CRPTO replacement.

## Immediate Execution Order

1. Freeze this protocol and run root.
2. Emit candidate registry and dirty-state audit.
3. Run tournament smoke across all declared PD/conformal lanes, not canonical
   alone.
4. Promote only conformal finalists with decision-aware evidence.
5. Run cuOpt frontier 25k/50k uniformly for finalists.
6. Run exact rerank only for finalists selected before seeing full-universe
   results.
7. Seal final selection, then run nested/prospective confirmation.
