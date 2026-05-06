# Paper Estrella Backlog - 2026-05-04

This backlog separates improvements that are already applied in the current
Paper Estrella from future work that requires new experiments, proofs, or
external validation. The current official champion remains
`bound_aware_276k_economic_champion`; this backlog must not be used to reopen the
champion unless a new search run is explicitly approved.

## Consolidated Status Matrix

This matrix is the current operational backlog view. It separates maintenance,
implemented journal-facing evidence, and work that would require genuinely new
experiments or theory.

| Priority | Status | Work item | Affects | Current artifact | Needs new run/metric? | Paper impact | Next action |
|---|---|---|---|---|---|---|---|
| P0 | Maintenance | Champion sync guardrails | official Paper Estrella metrics | `tests/test_docs/test_paper_estrella_final_sync.py` | No | keeps paper aligned | keep tests green before any paper/table change |
| P0 | Maintenance | Canonical paper tables | body tables and DVC metrics | `scripts/export_paper1_canonical_tables.py` | No | prevents metric drift | regenerate only from canonical promotion |
| P0 | Maintenance | DVC/Dagshub ownership | reproducibility | `dvc.lock`, `.dvc` pointers | No | supports artifact-backed claims | keep local and remote status clean |
| P0 | Maintenance | MLflow final run discoverability | experiment lineage | DagsHub MLflow run `6af4b95d152c47ec9420d5b1a2e78959` | No | supports reproducibility appendix | keep final metrics and artifacts traceable |
| P0 | Maintenance | Pipeline freeze / explicit champion reruns | reproducible reruns | `run_long_pipeline.py`, `configs/profiles/*`, `explicit_champion_only` | No | prevents accidental re-search | only `search_*` families reopen searches; paper/core reruns consume frozen champion |
| P1 | Implemented | Nested/post-selection evidence | post-selection criticism | A3, A9, `paper1_p1_evidence_status.json` | No for current paper | strengthens current paper | only future hardening is a prospective pre-declared split |
| P1 | Implemented | Decision-aware conformal selector | CROMS-style selector narrative | A5, A10 | No for current paper | strengthens current paper | future work is training score by decision loss |
| P1 | Implemented | Conditional tightening lemma | theory appendix | `14b`, tightening appendix | No for current paper | strengthens theory with caveat | prove dependence-aware version only for journal extension |
| P1 | Implemented | Synthetic/period shift evidence | robustness | A4, A6, A7, A8, A11 | No for current paper | strengthens current paper | external dataset remains future work |
| P1/J | Implemented | Alpha sweep and alpha-Gamma validation | alpha/robustness narrative | `alpha_sweep_pareto_mondrian.parquet`, `alpha_sweep_pareto_both.parquet`, `alpha_gamma_bound/*` | No | strengthens current paper | use as supporting alpha policy evidence, not a new champion search |
| P1/J | Implemented | Uncertainty-set baselines | CP vs bootstrap/parametric/ellipsoidal evidence | `uncertainty_baselines_comparison.parquet`, `uncertainty_baselines_by_grade.parquet` | No | strengthens current paper | use to justify conformal robust set selection |
| P1/J | Implemented | CQR comparator evidence | conformal alternative | `cqr_comparison_status.json`, `cqr_mondrian_status.json`, `estrella_fig10` | No | complements paper | keep as comparator/appendix; do not replace official Mondrian winner |
| P1/J | Implemented | Manuscript blueprint | paper structure | `14g-manuscript-blueprint.qmd` | No | prepares manuscript | compress into actual paper draft when writing starts |
| P1/J | Pending | Standalone manuscript extraction | final submission artifact | `14g`, `14o`, A1--A18, figures | No | required for submission | write the short paper from the book package; no champion changes |
| P1/J | Implemented | Journal appendix A12--A18 | appendix evidence | `14h-journal-appendix-robustness.qmd` | No | complements paper | use as appendix package, not new champion evidence |
| P1/J | Implemented | Mondrian ablation page | conformal winner defense | `14i-mondrian-ablation.qmd` | No | strengthens method selection | use when reviewer asks why score-decile, not grade |
| P1/J | Implemented | SPO+ protocol page | DFL comparator | `14j-spo-protocol-and-regret.qmd` | No | strengthens comparator narrative | keep train-time 49.1% and temporal stability configs separate |
| P1/J | Implemented | Fair lending checkpoint | governance/funded set | `14k-fair-lending-checkpoint.qmd` | No | strengthens auditability | cite as proxy/intersectional audit, not legal protected-attribute proof |
| P1/J | Implemented | MRM/SR 11-7 approval page | model risk management | `14l-governance-mrm-approval.qmd` | No | strengthens deployment credibility | keep triggers and challenger criteria aligned with MRM artifacts |
| P1/J | Implemented | Funded-set composition page | portfolio evidence | `14m-funded-set-composition.qmd` | No | strengthens result audit | use in appendix to show no hidden segment drives champion |
| P1/J | Implemented | Artifact traceability runbook | reproducibility | `14n-artifact-traceability.qmd` | No | strengthens reviewer response | keep claim-script-test paths real and guarded |
| P1/J | Implemented | Paper/journal/thesis extraction map | editorial planning | `14-paper-estrella/index.qmd` | No | preserves rich book content | later compress, but do not delete useful thesis evidence now |
| P1/J | Implemented | Extraction/release manifest | paper/journal/thesis packaging | `14o-extraction-release-manifest.qmd` | No | strengthens editorial extraction | use before creating the standalone manuscript |
| P1/J | Implemented | Journal figures | visual explanation/results | `estrella_fig12`--`estrella_fig14` | No | improves paper readability | choose which figures go to body vs appendix |
| P1/J | Implemented | Tail risk diagnostics | funded-set risk | A12 | No | complements paper | do not cite repriced return as official return |
| P1/J | Implemented | Satisficing margins | OR framing | A13 | No | complements paper | justify thresholds if moved to body |
| P1/J | Implemented | Dependence diagnostics | conditional tightening | A14 | No | complements theory | do not claim independence from this table |
| P1/J | Implemented | Temporal stress and bootstrap | robustness | A15, A16 | No | complements paper | keep as descriptive appendix evidence |
| P1/J | Implemented | Budget/LGD/cap sensitivity | applied credit robustness | A17 | No | complements paper | cap checks are diagnostics, not solver constraints |
| P1/J | Implemented | Robust region family table | compatible leaderboard | A18 | No | strengthens results | report only inside bound-aware family |
| P2 | Pending | OCE/CVaR as optimization target | portfolio search objective | A12 is diagnostic only | Yes | can strengthen or redirect method | implement tail-risk-aware search if approved |
| P2 | Pending | Multi-distribution robust CP | conformal layer | none | Yes | new methodological direction | design source/group robust calibration |
| P2 | Pending | Online conformal recalibration | deployment/streaming | none | Yes | new sequential direction | simulate monthly recalibration and coverage regret |
| P2 | Pending | Online DFL comparison | DFL benchmark | SPO+ static evidence exists | Yes | new comparison direction | build repeated-decision experiment |
| P2 | Pending | SPO+ + conformal hybrid | model training/calibration | current SPO+ and CP are separate | Yes | could change method | train decision-loss-aware predictor/calibrator with CP wrapper |
| P2 | Pending | Robust satisficing policy | OR objective | A13 is diagnostic only | Yes | new OR variant | optimize thresholds/margins directly |
| P2 | Pending | Dependence-aware main bound | theory | conditional lemma exists | No metric, but new proof | journal theory direction | prove a valid concentration result under shared calibration dependence |
| P2 | Pending | Multi-dataset credit replication | external validity | none | Yes | top-venue validation | run Lending Club-like pipeline on another credit dataset |
| P2 | Pending | Pareto regret-auditability frontier | OR/DFL framing | SPO+ and CRPTO diagnostics exist | Maybe | strengthens MS/OR story | formalize tradeoff between regret reduction and coverage/auditability |
| P3 | Future | Multi-period portfolio | production realism | none | Yes | new paper/product track | model state transitions and rebalancing |
| P3 | Future | Conformal CATE / causal CRPTO | causal decision layer | CATE lane exists as `insights_only` | Yes | new paper direction | require identification, overlap, policy value and causal sensitivity checks |
| P3 | Future | Distribution shift / online conformal CRPTO | sequential uncertainty | `time_series_vnext` is `research_only` | Yes | new sequential direction | use ACI/online CP only after interval gate is promotable |
| P3 | Future | Open-source CRPTO package | adoption/software | current codebase modules | Yes | thesis/product differentiator | extract a clean library/API after paper method stabilizes |
| P3 | Future | Multi-asset credit validation | external validity | none | Yes | broader thesis validation | test another credit product |
| P3 | Future | Direct protected-attribute / temporal fairness validation | fairness/governance | proxy base + proxy-intersectional audit exists in `14k` | Yes | complements thesis | repeat with protected attributes if available and monitor disparity over time |
| P3 | Future | Field trial or counterfactual deployment study | real-world validation | offline evidence exists | Yes | product/doctoral differentiator | run partner pilot or rigorous replay study |
| P3 | Future | Production monitoring dashboard | productization | artifacts exist, dashboard not live | Yes | product track | expose champion/DVC/MLflow/drift in one view |

## Historical Backlog Reconciliation - 2026-05-06

The March/April backlog screenshots are now reconciled against the current
repository state. The key change is that the old backlog mixed four different
things: champion repair, artifact generation, paper writing, and doctoral/top
venue extensions. The current backlog keeps those lanes separate.

### Closed From The Older Backlogs

| Historical item | Current status | Evidence now in repo | Current decision |
|---|---|---|---|
| Restore real Apr 5--6 portfolio champion | Closed | `models/final_project_promotion.json`, `models/champion_portfolio_policy.json`, `models/champion_registry.json` | economic champion is official; theorem-tight remains comparator |
| Freeze paper/core reruns to avoid three-hour selector search | Closed | `run_long_pipeline.py`, `configs/profiles/*`, `freeze_if_available`, `explicit_champion_only` | paper/core reruns consume frozen policy; only search families reopen search |
| Promote HPO local trial 56 and conformal rank 1 chain | Closed | `README.md`, `SESSION_STATE.md`, `models/final_project_promotion.json` | PD and conformal upstream are frozen for Paper Estrella |
| Alpha sweep `{0.01..0.20}` | Closed as supporting evidence | `data/processed/alpha_sweep_pareto_both.parquet`, `data/processed/alpha_sweep_pareto_mondrian.parquet`, `models/alpha_sweep_status.json` | use to explain the alpha policy dial and why global intervals are too wide |
| Empirical alpha -> Gamma_CP bound validation | Closed for current paper | `data/processed/alpha_gamma_bound/*`, `scripts/validate_alpha_gamma_bound.py`, `estrella_fig_alpha_gamma_bound` | supports the Markov-based theorem and exact champion eval; does not replace post-selection caveats |
| Uncertainty-set baselines | Closed as comparator evidence | `data/processed/uncertainty_baselines_comparison.parquet`, `data/processed/uncertainty_baselines_by_grade.parquet`, `estrella_fig7_uncertainty_baselines` | conformal Mondrian is the defendable robust set; bootstrap/parametric are comparators |
| CQR alternative | Closed as appendix/comparator, not champion | `models/cqr_comparison_status.json`, `models/cqr_mondrian_status.json`, `estrella_fig10_cqr_per_grade` | keep as related evidence; no method replacement in current paper |
| SICR trigger optimization and ECL alpha sensitivity | Closed, but belongs to Paper 2 | `sicr_trigger_optimization.parquet`, `ecl_alpha_sensitivity.parquet`, Paper 2 figures | cite only as IFRS9/mega-extension context, not Paper Estrella champion evidence |
| SPO+ / DFL comparison | Closed for current comparator layer | `models/spo_comparison_status.json`, `models/spo_real_training_status.json`, `14j-spo-protocol-and-regret.qmd` | use as regret/auditability comparator, not as a promoted CRPTO policy |
| Quarto scaffolding and Paper Estrella pages | Closed | `14a`--`14o`, `_quarto.yml`, rendered/freeze cache | book remains rich; future manuscript extracts rather than deletes |
| Pipeline-first topology and notebook roles | Closed | `configs/pipeline_registry/*`, `scripts/run_long_pipeline.py`, `docs/RUNBOOK.md` | pipelines are producer/search/paper/research lanes; notebooks are executable documentation, not canonical producers |
| Insights factory / research labs | Closed as sidecar | `research_labs` profile, notebook atlas, governance docs | useful source of figures and ideas, but not a Paper Estrella champion dependency |
| Publication figures | Mostly closed for current evidence | `estrella_fig7`, `estrella_fig10`, `estrella_fig12`--`estrella_fig14`, alpha-gamma figures | remaining action is editorial selection for body vs appendix |
| A/B, fairness, MRM and governance concerns | Closed for current paper | `14j`, `14k`, `14l`, `14n`, guardrail tests | cite as auditability/governance, with proxy and diagnostic caveats |
| Time-series interval redesign | Closed as `research_only` | `docs/TIME_SERIES_VNEXT_DECISION_2026-04-02.md`, `models/time_series_vnext_status.json` | valuable for Paper 2/mega extension, not current Paper Estrella |
| Causal/CATE lane | Closed as `insights_only` for current paper | `14e`, Paper 2 mega-extension page, causal artifacts | future causal CRPTO, not current champion contribution |
| Streamlit companion | Partially deferred | `docs/STREAMLIT_QUARTO_MIGRATION_REGISTRY.yml`, book pages, Streamlit app | Quarto is the primary paper surface; a live dashboard is product/P3 unless a venue requests a companion URL |

### Immediate Work That Does Not Change The Paper Direction

These items use existing artifacts and are safe to do before submission. They
are about manuscript quality, not new champion selection.

| Item | Uses existing artifacts? | Needs new run? | Why it is immediate | Done when |
|---|---|---|---|---|
| Extract standalone manuscript draft | Yes: `14g`, `14o`, A1--A18, figures | No | converts book evidence into a journal-shaped paper | abstract, intro, related work, theory, method, results and appendix skeleton exist outside the book |
| Final body-vs-appendix table/figure selection | Yes | No | avoids overloading the paper body | final list maps each table/figure to body, appendix or thesis-only |
| Write alpha sweep / alpha-Gamma narrative | Yes | No | turns old "alpha sweep" task into a clear business-policy dial | body text explains alpha, width, Gamma_CP, funded set and robust region without overclaiming |
| Write uncertainty-baseline narrative | Yes | No | answers why conformal robust sets, not bootstrap/parametric sets | comparator table/figure is tied to the CRPTO contribution |
| Tighten related work around PtC, CROMS, CRC, SPO+, CQR and robust optimization | Yes | No | positions the paper in the right literatures | related-work paragraphs distinguish direct competitors from future extensions |
| Final obsolete-number sweep | Yes | No | prevents regression to old return, old tau or wrong champion-role claims | `rg` confirms no obsolete champion/metric language in Paper Estrella pages |
| Submission reproducibility checklist | Yes | No | keeps reviewer package reproducible | DVC, MLflow, paper table export, Quarto render and guardrail tests are listed with commands |
| Online companion URL decision | Yes | No | old backlogs asked for a companion; now the decision should be explicit | release manifest names the stable Quarto URL or documents that the companion is deferred |

### Keep As Future Paper / Journal Extension

These items are valuable, but they change the method, the target guarantee, the
data universe or the operational setting. They should not be hidden as "minor
cleanup" for the current Paper Estrella.

| Future item | Why it is not immediate | Natural placement |
|---|---|---|
| Hoeffding/Bernstein as main bound | the current lemma is conditional; a main theorem needs a dependence-aware proof or defensible independence design | journal version / theory appendix |
| Multi-dataset credit validation | requires new data, new leakage checks, new calibration and new DVC/MLflow evidence | journal version / thesis validation |
| Distribution shift and online conformal | changes offline CRPTO into sequential CRPTO; time-series intervals are still `research_only` | Paper 2 extension or new sequential CRPTO paper |
| Multi-period portfolio with rebalancing | changes one-shot LP into a dynamic decision problem | new OR paper / thesis chapter |
| Conformal CATE / causal CRPTO | requires identification, overlap, sensitivity and policy-value evidence | new causal CRPTO paper / mega extension |
| Fairness-constrained conformal optimization | current fairness is proxy audit; formal fairness constraints require new protected-attribute or proxy-governance design | governance/fairness paper or journal appendix if data permits |
| Field trial or production deployment | requires real operational adoption or a strong counterfactual deployment study | product/doctoral differentiator |
| Open-source CRPTO library | needs API extraction, tests, docs and external adoption | post-paper software track |

## Current Rule of Record

- The current paper is a **CRPTO post-hoc auditable** paper with a frozen
  economic champion.
- P0/P1/P1-J items strengthen the current paper without changing its direction.
- P2 items are real methodological extensions and should be opened only with a
  named run/protocol.
- P3 items belong to broader thesis/product work.
- If any diagnostic table contradicts `models/final_project_promotion.json`,
  the promotion artifact wins.

## P0 - Keep Current Paper Publishable

| Item | Why it matters | Artifact / owner | Acceptance criteria |
|---|---|---|---|
| Keep champion sync guardrails green | Prevents the economic champion from being overwritten by quick/search runs | `tests/test_docs/test_paper_estrella_final_sync.py` | promotion, policy, registry, DVC metrics and paper tables agree |
| Keep tables generated from canonical promotion | Avoids legacy 5,001-frontier drift | `scripts/export_paper1_canonical_tables.py` | table 0 reports `$170.5K`, `V=0.03645`, `gamma_cp=0.18591`, `45/45` region |
| Keep DVC/Dagshub ownership clean | Makes results reproducible without Git blobs | `dvc.lock`, `.dvc` pointers | `dvc status --no-updates` and `dvc status -c -r dagshub` are clean |
| Keep MLflow Paper Estrella final run discoverable | Preserves experiment tracking for the paper-facing closure | DagsHub MLflow run `6af4b95d152c47ec9420d5b1a2e78959` | run logs final champion metrics and canonical artifacts |
| Keep paper/core profiles frozen | Prevents `paper1_e2e`, `paper2_e2e`, `canonical_rebuild` and `core_canonical` from re-searching a known champion | `run_long_pipeline.py`, `configs/profiles/*` | non-search families use `freeze_if_available` and `explicit_champion_only`; search families remain explicit |
| Keep the Quarto book richer than the manuscript | Preserves reviewer-facing reasoning before paper compression | `book/chapters/14-paper-estrella/14f-editorial-claims-references.qmd` | claim ladder, reviewer Q&A, paper-placement table and numbered references stay rendered |

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

### Journal Package Implementation Snapshot - 2026-05-04

The immediate paper-to-journal packaging items are also implemented. This
package is deliberately diagnostic: it strengthens the current Paper Estrella
without changing the official champion or reopening the search.

| Item | Implemented artifact | What it adds | Scope caveat |
|---|---|---|---|
| Convert chapter 14 into paper blueprint | `book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd` | target venue, abstract, claims C1--C7, manuscript outline, final table/figure plan and notation | blueprint, not final manuscript |
| Appendix A12--A18 | `book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd` | renders tail risk, satisficing, dependency, stress, bootstrap, LGD/cap and robust-region evidence | appendix material unless a journal asks for more body evidence |
| Clean CRPTO figure | `estrella_fig12_crpto_conceptual_pipeline.png` | candidate Figure 1 | visual explanation only |
| Alpha -> Gamma_CP -> funded set figure | `estrella_fig13_alpha_gamma_funded_set.png` | connects conformal alpha to portfolio quantities | diagnostic curve from frozen artifacts |
| Robust region heatmap | `estrella_fig14_robust_region_heatmap.png` | visualizes the `45/45` robust region | summarizes final mini-grid, not a new search |
| OCE/CVaR funded-set risk | `paper1_tableA12_tail_risk_oce_cvar.csv` | reports mean loss, entropic OCE and CVaR under LGD 35/45/60 | return column is funded-set repricing diagnostic, not official champion return |
| Satisficing margin | `paper1_tableA13_satisficing_margins.csv` | expresses return, `V`, `Gamma_CP`, violation and robust-region pass as OR thresholds | editorial thresholds should be justified if used in paper body |
| Dependence diagnostics | `paper1_tableA14_dependency_cluster_diagnostics.csv` | documents concentration by period, grade and period-grade for the tightening appendix | does not prove independence |
| Leave-one-period-out stress | `paper1_tableA15_leave_one_period_stress.csv` | checks temporal sensitivity by dropping or overweighting OOT periods | reweights exported funded set, not re-optimized policies |
| Bootstrap funded-set metrics | `paper1_tableA16_bootstrap_funded_set_metrics.csv` | adds empirical intervals for return, default, `V` and miscoverage counts | descriptive bootstrap, not formal conformal guarantee |
| Budget / LGD / cap sensitivity | `paper1_tableA17_budget_cap_lgd_sensitivity.csv` | reprices under budgets, LGD alternatives and segment caps | cap check is diagnostic, not a constrained optimization |
| Robust region by policy family | `paper1_tableA18_robust_region_policy_family.csv` | groups final policies by `risk_tolerance x gamma` and confirms all pass | compatible leaderboard only within bound-aware family |
| Reproducible generator | `scripts/build_paper1_journal_package.py`, `models/paper1_journal_package_status.json` | regenerates A12--A18 and figures from frozen artifacts | no champion promotion logic |

### Quarto Expansion Snapshot - 2026-05-05

The Paper Estrella section is intentionally richer than the future manuscript.
The current book package now preserves the material needed to later extract a
short paper, a journal version and a thesis chapter without losing context.

| Page | Status | What it adds | Later placement |
|---|---|---|---|
| `index.qmd` | Implemented | curated navigation through `14a`--`14o` and extraction rule paper/journal/thesis | editorial hub |
| `14i-mondrian-ablation.qmd` | Implemented | rank 1/2/3 conformal ablation and winner configuration | appendix or method robustness |
| `14j-spo-protocol-and-regret.qmd` | Implemented | SPO+ train-time vs temporal protocol split | comparator appendix |
| `14k-fair-lending-checkpoint.qmd` | Implemented | 3 base + 3 proxy-intersectional fairness checks, all PASS | governance appendix |
| `14l-governance-mrm-approval.qmd` | Implemented | SR 11-7 gates, challenger criteria and retraining triggers | governance appendix / thesis |
| `14m-funded-set-composition.qmd` | Implemented | funded-set loan/period/grade composition | results appendix |
| `14n-artifact-traceability.qmd` | Implemented | claim -> artifact -> script -> test map and runbook | reproducibility appendix |
| `14o-extraction-release-manifest.qmd` | Implemented | direction filter, table/figure placement, venue response bank and release checklist | editorial/reproducibility supplement |

Remaining Quarto maintenance is not about reducing content. It is about keeping
paths, claims and caches synchronized as new evidence pages are added.

## P2 - Methodological Extensions

| Item | Literature driver | Implementation sketch | Acceptance criteria |
|---|---|---|---|
| OCE/CVaR funded-set conformal risk as optimization target | Conformal Risk Training | A12 now reports diagnostic OCE/CVaR; the P2 version would replace or augment expected weighted miscoverage with OCE/CVaR-style tail risk during search | reports tail-risk metrics as constraints/objectives alongside official return, `V`, `gamma_cp` and price of robustness |
| Multi-distribution robust conformal layer | MDCP | Calibrate for multiple possible sources/groups without assuming test-time group availability | reports worst-source coverage and robust set width |
| Online conformal recalibration | UP-OCP / ACI | Update conformal quantiles under streaming monthly originations | coverage regret or online miscoverage is tracked over time |
| Online DFL comparison | Online DFL | Compare CRPTO, SPO+ and online DFL under drift and repeated decisions | reports static/dynamic regret plus coverage and auditability metrics |
| SPO+ + conformal hybrid | SPO+, end-to-end conformal calibration | Train the predictor or calibration layer with decision loss while retaining conformal wrapper | shows whether regret improves without losing coverage traceability |
| Robust satisficing policy | Conformal Robust Optimization and Satisficing | Add a satisficing objective where policies meet risk/return thresholds instead of maximizing return alone | reports fragility/satisficing margin next to price of robustness |
| Dependence-aware main bound | CRC + concentration under dependence | Upgrade the conditional Hoeffding/Bernstein lemma into a main theorem that explicitly handles the shared calibration-set dependence structure | Markov remains valid; any tighter main claim has a proof that does not assume independence silently |
| Multi-dataset credit replication | external validity / credit risk | Repeat the frozen CRPTO protocol on a separate credit dataset with fresh leakage checks and artifact ownership | coverage, return, `V`, `gamma_cp` and price of robustness are reported on at least one independent dataset |
| Pareto regret-auditability frontier | DFL + OR decision theory | Formalize the tradeoff between regret-oriented training/optimization and auditable conformal risk control | reports a frontier or scalar objective where CRPTO, SPO+ and hybrids can be compared without mixing incompatible leaderboards |

## P3 - Broader Thesis / Product Track

| Item | Why it is future work | Acceptance criteria |
|---|---|---|
| Multi-period portfolio with rebalancing | Current CRPTO is one-period | state transition, transaction costs and repeated decisions are explicitly modeled |
| Conformal CATE / causal CRPTO | Current CATE lane is `insights_only`; making it central changes the estimand and standard of proof | identification, overlap, sensitivity bounds, policy value and causal-funded-set bound checks pass |
| Distribution shift / online conformal CRPTO | Current conformal guarantee is offline/OOT; online shift control changes the operational setting | monthly or rolling-origin coverage regret is reported and interval policy becomes promotable |
| Open-source CRPTO package | Current code is project-specific, not a stable research library | clean API, installation docs, unit tests and an example dataset exist |
| Multi-asset credit validation | Lending Club is one asset class | method tested on another loan/credit product |
| Direct protected-attribute / temporal fairness validation | Current fairness uses available proxy attributes and proxy intersections, not protected attributes directly | coverage and decision impact are evaluated on protected attributes if legally available, plus temporal disparity monitoring |
| Field trial or counterfactual deployment study | Current evidence is artifact-backed offline validation | either a partner deployment or a strong counterfactual policy replay is documented |
| Production monitoring dashboard | Paper is artifact-backed but not live | champion metrics, DVC version, MLflow run and conformal drift visible in one operational view |

## Documentation Layer

The Quarto book now includes an explicit editorial guide and two journal-facing
pages for Paper Estrella, plus the new support pages that make the book useful
as a staging area for paper, journal and thesis:

- `book/chapters/14-paper-estrella/14f-editorial-claims-references.qmd`
- `book/chapters/14-paper-estrella/14g-manuscript-blueprint.qmd`
- `book/chapters/14-paper-estrella/14h-journal-appendix-robustness.qmd`
- `book/chapters/14-paper-estrella/14i-mondrian-ablation.qmd`
- `book/chapters/14-paper-estrella/14j-spo-protocol-and-regret.qmd`
- `book/chapters/14-paper-estrella/14k-fair-lending-checkpoint.qmd`
- `book/chapters/14-paper-estrella/14l-governance-mrm-approval.qmd`
- `book/chapters/14-paper-estrella/14m-funded-set-composition.qmd`
- `book/chapters/14-paper-estrella/14n-artifact-traceability.qmd`
- `book/chapters/14-paper-estrella/14o-extraction-release-manifest.qmd`

These pages are intentionally more explanatory than a journal paper. They keep
the claim ladder, reviewer Q&A, artifact placement map, local numbered
references `[1]`, `[2]`, ... and the A12--A18 appendix package that can later be
compressed into the manuscript.

The extraction/release manifest explicitly classifies OCE/CVaR optimization,
online conformal/Online DFL, MDCP, direct protected-attribute fairness,
external-dataset validation, dependence-aware main bound, causal CRPTO,
multi-period portfolio and open-source packaging as backlog items
because they would introduce a new method, guarantee or dataset rather than
merely organizing existing evidence.

Because `book/_quarto.yml` uses `execute.freeze: true`, rendered cache updates
under `book/_freeze/chapters/14-paper-estrella/` should be treated as
intentional reproducibility artifacts when they correspond to a real Quarto page
update. Do not clean them blindly; review them with the page they freeze.

The companion research note is
`docs/research/paper_estrella_quarto_expansion_2026-05-04.md`.

## Do Not Reopen Without Approval

- Do not replace `paper-thesis-final-economic-2026-04-06` as the Paper Estrella
  champion without a named search run, DVC/MLflow sync, and updated guardrails.
- Do not compare PD AUC, conformal coverage, portfolio return and bound-aware
  tightness in a single leaderboard.
- Do not promote theorem-tight as champion unless the editorial objective changes
  from economic champion to theoretical tightness champion.
