# Paper Estrella Backlog - 2026-05-04

This backlog separates improvements that are already applied in the current
Paper Estrella from future work that requires new experiments, proofs, or
external validation. The current official champion remains
`bound_aware_276k_economic_champion`; this backlog must not be used to reopen the
champion unless a new search run is explicitly approved.

## P0 - Keep Current Paper Publishable

| Item | Why it matters | Artifact / owner | Acceptance criteria |
|---|---|---|---|
| Keep champion sync guardrails green | Prevents the economic champion from being overwritten by quick/search runs | `tests/test_docs/test_paper_estrella_final_sync.py` | promotion, policy, registry, DVC metrics and paper tables agree |
| Keep tables generated from canonical promotion | Avoids legacy 5,001-frontier drift | `scripts/export_paper1_canonical_tables.py` | table 0 reports `$170.5K`, `V=0.03645`, `gamma_cp=0.18591`, `45/45` region |
| Keep DVC/Dagshub ownership clean | Makes results reproducible without Git blobs | `dvc.lock`, `.dvc` pointers | `dvc status --no-updates` and `dvc status -c -r dagshub` are clean |
| Keep MLflow Paper Estrella final run discoverable | Preserves experiment tracking for the paper-facing closure | DagsHub MLflow run `6af4b95d152c47ec9420d5b1a2e78959` | run logs final champion metrics and canonical artifacts |

## P1 - Journal-Grade Evidence

| Item | Literature driver | Implementation sketch | Acceptance criteria |
|---|---|---|---|
| Nested holdout / post-selection validation | LTT, RCPS | Split the OOT or calibration/evaluation universe into selection and confirmation layers for bound-aware policy selection | final policy selected on one slice passes `alpha01` and reports `V`, `gamma_cp`, return on untouched confirmation slice |
| Decision-aware conformal selector | CROMS | Rank conformal variants by a joint objective: coverage, width, min group coverage, return, `V`, `gamma_cp`, violation | selector identifies a variant/policy pair without mixing conformal and portfolio metrics after the fact |
| Conditional tightening lemma | CRC + Hoeffding/Bernstein | Add a theorem/appendix result under explicit independence or conditional independence assumptions for weighted miscoverage indicators | Markov remains the main theorem; tighter bounds are clearly labeled conditional |
| External or synthetic shift replication | MDCP, online CP | Create stress scenarios or an external credit dataset validation of coverage, width, return, and `V` | coverage and funded-set risk are reported by period/source, not only globally |
| Segment-period sensitivity | model risk / governance | Expand the stability table by grade, period and funded-set composition | no hidden weak segment drives the champion result |

### P1 Implementation Snapshot - 2026-05-04

The P1 items above now have a first reproducible evidence layer around the
official champion. This layer strengthens the current paper without reopening
the champion search.

| Item | Implemented artifact | What it proves now | Remaining journal hardening |
|---|---|---|---|
| Nested holdout / post-selection validation | `paper1_tableA3_nested_holdout.csv`, `paper1_tableA9_strict_temporal_holdout.csv`, `models/paper1_p1_evidence_status.json` | the 5K -> 25K -> 276K chain is explicit, and the frozen champion also passes `alpha01` on strict temporal confirmation slices; both 2018 selection and 2019--2020 confirmation have zero violation | a fully prospective protocol where the strict split is declared before any policy search |
| Decision-aware conformal selector | `paper1_tableA5_decision_aware_selector.csv`, `paper1_tableA10_conformal_finalist_exact_bound_eval.csv` | a CROMS-style screen selects rank 1 after combining conformal gates, A/B pass, tradeoff return and exact 276K bound metrics for ranks 1, 2 and 3; ranks 2/3 pass exact portfolio eval but fail min-group conformal coverage | full prospective training where the conformal score itself is optimized for decision loss |
| Conditional tightening lemma | `book/chapters/14-paper-estrella/14b-theoretical-framework.qmd`, `docs/research/paper_estrella_conditional_tightening_appendix_2026-05-04.md` | Hoeffding/Bernstein tightening is stated as conditional on additional independence assumptions, while Markov remains the main distribution-free theorem | empirical or theoretical justification of conditional independence, or a weaker dependence-aware concentration result |
| External or synthetic shift replication | `paper1_tableA6_synthetic_shift.csv`, `paper1_tableA11_enhanced_synthetic_shift.csv` | OOT covariate-reweighting and adversarial label-flip stress tests keep coverage above 90% across high-PD, high-grade-risk, late-period and weakest-segment scenarios | true external credit dataset replication |
| Segment-period sensitivity | `paper1_tableA4_segment_period_sensitivity.csv`, `paper1_tableA7_funded_set_loans.csv`, `paper1_tableA8_funded_set_composition.csv` | all observed period-grade cuts stay above 90% coverage; the exact funded set is exported loan-by-loan and summarized by period/grade composition | external or prospective funded-set composition replication |

## P2 - Methodological Extensions

| Item | Literature driver | Implementation sketch | Acceptance criteria |
|---|---|---|---|
| OCE/CVaR funded-set conformal risk | Conformal Risk Training | Replace or augment expected weighted miscoverage with OCE/CVaR-style tail risk of funded-set loss | reports tail-risk metrics alongside return, `V`, `gamma_cp` and price of robustness |
| Multi-distribution robust conformal layer | MDCP | Calibrate for multiple possible sources/groups without assuming test-time group availability | reports worst-source coverage and robust set width |
| Online conformal recalibration | UP-OCP / ACI | Update conformal quantiles under streaming monthly originations | coverage regret or online miscoverage is tracked over time |
| Online DFL comparison | Online DFL | Compare CRPTO, SPO+ and online DFL under drift and repeated decisions | reports static/dynamic regret plus coverage and auditability metrics |
| SPO+ + conformal hybrid | SPO+, end-to-end conformal calibration | Train the predictor or calibration layer with decision loss while retaining conformal wrapper | shows whether regret improves without losing coverage traceability |
| Robust satisficing policy | Conformal Robust Optimization and Satisficing | Add a satisficing objective where policies meet risk/return thresholds instead of maximizing return alone | reports fragility/satisficing margin next to price of robustness |

## P3 - Broader Thesis / Product Track

| Item | Why it is future work | Acceptance criteria |
|---|---|---|
| Multi-period portfolio with rebalancing | Current CRPTO is one-period | state transition, transaction costs and repeated decisions are explicitly modeled |
| Multi-asset credit validation | Lending Club is one asset class | method tested on another loan/credit product |
| Intersectional fairness conformal audit | Current fairness is attribute-level | coverage and decision impact are evaluated on intersections |
| Production monitoring dashboard | Paper is artifact-backed but not live | champion metrics, DVC version, MLflow run and conformal drift visible in one operational view |

## Do Not Reopen Without Approval

- Do not replace `paper-thesis-final-economic-2026-04-06` as the Paper Estrella
  champion without a named search run, DVC/MLflow sync, and updated guardrails.
- Do not compare PD AUC, conformal coverage, portfolio return and bound-aware
  tightness in a single leaderboard.
- Do not promote theorem-tight as champion unless the editorial objective changes
  from economic champion to theoretical tightness champion.
