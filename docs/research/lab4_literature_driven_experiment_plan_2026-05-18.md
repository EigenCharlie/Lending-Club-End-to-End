# Lab 4 Literature-Driven Experiment Plan - 2026-05-18

## Purpose

This plan converts the newly reviewed papers into bounded Lab 4 experiments for
the Lending Club CRPTO project. Lab 4 is the initial sink for all work. Results
can later be classified as promote, append, park, archive, or delete. The plan
does not reopen the official Paper Estrella champion unless a predeclared gate
is met.

The local PDF surface reviewed in this pass included:

- New downloads in `/mnt/c/Users/carlos/Downloads/papers_nuevos/`.
- Thesis papers in `/mnt/c/Users/carlos/Documents/Paper_CRPTO/Papers_tesis/`.
- Additional web-verified candidates for missing PDFs or recent companion work.

## Governance Rules

- No `paper4_v###` waves, no signoff CSVs, no per-iteration status packets.
- Scratch runs live outside the repo under `/tmp/lc-paper4-lab4-runs/`.
- Promote only one compact table/memo per accepted or parked lane.
- Maximum three waves per lane: baseline, variants, confirmation.
- A lane stops when it cannot change a claim, table, appendix, or reviewer
  response.
- Paper Estrella can receive only reviewer-defense material, not speculative Lab
  4 prototypes.

## Source Families And Implications

| family | key local papers | strongest implication |
| --- | --- | --- |
| Conformal risk control | RCPS, Conformal Risk Control, Learn-then-Test, Conformal Risk Training | Turn Paper 4 into a gated decision-risk lab; Paper Estrella can cite risk-control framing. |
| Conformal robust optimization | Conformal uncertainty sets, Predict-then-Calibrate, End-to-End Conformal Calibration, CROMS, CRO/CRS | Run ablations where uncertainty sets are selected by downstream decision risk, not coverage alone. |
| Online/multi-source conformal | ACI, Multi-Source Conformal Inference, MDCP, UP-OCP | Source holdouts remain retrospective unless a true external/live stream appears. |
| Robust optimization and tail risk | Price of Robustness, CVaR, OCE, Conformal Risk Training | CVaR/OCE is a serious challenger appendix, not a champion replacement unless paired wealth wins. |
| Decision-focused learning | SPO+, task-based learning, DFL survey, online DFL, cvxpylayers, PyEPO | Use as isolated comparators/prototypes; do not turn CRPTO into an end-to-end DFL paper. |
| Credit governance | FinRegLab, SR 11-7, Jagtiani/Lemieux, Lessmann | Strengthen the credit/model-risk framing and build governance diagnostics. |
| IFRS/ECL | Basel ECL guidance, IFRS 9 sources, survival/competing-risk ECL | Keep IFRS9/SICR as proxy ECL/SICR only. |
| Fair lending and causal | CFPB BISG, Zhang, DoWhy, DML, causal forests | Useful negative/diagnostic lanes; no legal fairness or policy-value claims with current data. |

## Priority Roadmap

### Lane 1: CRC/LTT Decision-Loss Gate

Claim target: Paper 4 can show that CRPTO-style selection is governed by
predeclared decision-risk gates rather than by ad hoc iteration.

Sources: RCPS, Conformal Risk Control, Learn-then-Test, Conformal Risk Training.

Experiment:

1. Define two or three bounded losses:
   - coverage violation loss at alpha 0.01;
   - funded-bad-tail loss for loans whose realized loss is outside the robust
     protection band;
   - return-shortfall loss versus official economic champion.
2. Calibrate thresholds on a fixed calibration split.
3. Test candidate selectors on OOT 276K and a Paper 4 replay split.
4. Apply LTT-style gate logic to decide whether a candidate is append-worthy.

Variants:

- global vs grade-Mondrian vs score-decile-Mondrian gates;
- loss by calendar vintage and grade;
- expected loss gate vs OCE/CVaR gate.

Evidence gate:

- alpha and loss definitions fixed before runs;
- no split leakage;
- pass/fail table by candidate, grade and vintage;
- current champion either passes or the failure is documented as a limitation.

Destination:

- Paper 4 appendix if it clarifies governance.
- Paper Estrella only as a reviewer-defense paragraph if it strengthens the
  existing champion without changing it.

Stop rule:

- Park if gates are too sensitive to split choice or cannot be expressed with
  current artifact columns.

What not to claim:

- No universal conditional guarantee.
- No individual borrower guarantee.
- No fairness or regulatory compliance claim.

### Lane 2: CROMS-Lite Model And Calibrator Selection

Claim target: Paper 4 can test whether downstream decision risk chooses a
different uncertainty model than coverage-only selection.

Sources: Optimal Model Selection for Conformalized Robust Optimization (CROMS),
Predict-then-Calibrate, Conformal uncertainty sets for robust optimization,
End-to-End Conformal Calibration.

Experiment:

1. Freeze the predictor family first.
2. Compare a compact set of uncertainty/calibration candidates:
   - official conformal robust region;
   - grade Mondrian;
   - score-decile Mondrian;
   - source-grade shrinkback;
   - box residual set;
   - Mahalanobis or ellipsoidal residual set if cheap.
3. Use a selection split to pick by decision risk, not by coverage alone.
4. Confirm selected candidate on untouched OOT.

Variants:

- selection objective: return subject to coverage, CVaR subject to return floor,
  or OCE subject to coverage;
- group-aware selection by grade or vintage;
- finite-sample conservative selection versus asymptotic/CROMS-lite selection.

Evidence gate:

- selected method improves decision risk or width materially without violating
  alpha coverage;
- improvement survives OOT and at least one temporal slice;
- table is small enough to be a single appendix table.

Destination:

- Paper 4 methods appendix.
- Paper Estrella only if it reinforces the current robust region choice.

Stop rule:

- Park if candidate selection is unstable or the official region remains best
  within noise.

What not to claim:

- No exact implementation of full CROMS unless its split/gate assumptions are
  faithfully implemented.
- No end-to-end learned uncertainty-set claim for CatBoost unless a differentiable
  surrogate is actually trained.

### Lane 3: E2E Conformal Calibration Prototype

Claim target: Test whether decision-aware uncertainty-set learning is worth a
future paper extension.

Sources: End-to-End Conformal Calibration for Optimization Under Uncertainty,
End-to-end Conditional Robust Optimization, Conformal Risk Training.

Experiment:

1. Build an isolated small PyTorch prototype on a reduced Lending Club sample.
2. Use a simple differentiable predictor for return/loss residuals.
3. Train uncertainty-set parameters with a downstream decision loss.
4. Compare against post-hoc conformal calibration and official CRPTO proxy.

Variants:

- simple interval scale network;
- convex set proxy with diagonal scale only;
- OCE/CVaR loss versus expected decision loss.

Evidence gate:

- OOT coverage preserved;
- decision risk improves over post-hoc conformal baseline;
- training is reproducible in an isolated environment;
- runtime is sane on a reduced sample.

Destination:

- Paper 4 lab notebook/prototype if positive.
- Park if only toy-level.

Stop rule:

- Stop immediately if gradients are unstable, calibration leaks, or runtime
  grows beyond prototype scope.

What not to claim:

- No production CRPTO replacement.
- No full TMLR/UAI method replication unless all assumptions and losses match.

### Lane 4: Online And Multi-Source Conformal Replay

Claim target: Decide whether source-aware conformal should remain a limitation
or become an appendix diagnostic.

Sources: ACI, Multi-Source Conformal Inference Under Distribution Shift,
Multi-Distribution Robust Conformal Prediction, Online Conformal Prediction via
Universal Portfolio Algorithms.

Experiment:

1. Define sources as issue-year, grade, state/zip3 bucket, purpose, or income/DTI
   bin.
2. Run rolling calibration/test windows by issue month.
3. Compare fixed split conformal, ACI-style update, source-shrinkback, MDCP-style
   max-p aggregation, and UP-OCP-inspired parameter-free update if feasible.
4. Evaluate worst-source coverage, average width, and decision return.

Variants:

- source known at test time versus hidden source;
- grade-only source versus geographic source;
- window sizes 6/12/24 months.

Evidence gate:

- worst-source coverage improves materially without useless interval inflation;
- replay is clearly historical, not live deployment;
- no group has too little sample support for a meaningful claim.

Destination:

- Paper 4 appendix/limitations.
- Paper Estrella only as caveat if it sharpens honesty around source holdouts.

Stop rule:

- Park if no variant improves the width/coverage frontier.

What not to claim:

- No live online validity.
- No external-distribution robustness.
- No fairness guarantee.

### Lane 5: CVaR/OCE Tail-Risk Challenger

Claim target: Determine whether tail-risk optimization adds a useful challenger
table beyond the official economic champion.

Sources: Rockafellar and Uryasev CVaR, Ben-Tal and Teboulle OCE, Conformal Risk
Training.

Experiment:

1. Build scenario losses from realized default/LGD/recovery/cashflow proxies.
2. Solve mean-CVaR portfolio variants at beta 0.90, 0.95 and 0.99.
3. Add OCE-style risk aversion variants if the loss function stays compact.
4. Compare with official economic champion on the same OOT universe and paths.

Variants:

- return floor with minimized CVaR;
- maximize return subject to CVaR cap;
- recovery-aware versus simple default-loss scenarios;
- grade/vintage stress slices.

Evidence gate:

- solver gap/certificate recorded;
- paired wealth, CVaR and drawdown all reported;
- challenger must either improve tail risk at acceptable return cost or be
  parked explicitly.

Destination:

- Paper 4 appendix tail challenger.
- Paper Estrella only as "we tested tail-risk alternatives; champion remains
  economic."

Stop rule:

- Park if it improves tail only by destroying wealth, or if scenario quality is
  too weak.

What not to claim:

- No champion replacement without paired wealth dominance.
- No true distributional tail forecast beyond the scenario construction.

### Lane 6: SPO+/DFL Comparator And Isolated Prototype

Claim target: Position CRPTO against decision-focused learning without making
Paper Estrella a DFL paper.

Sources: Smart Predict, then Optimize; task-based end-to-end learning; DFL
survey; Online DFL; cvxpylayers; PyEPO.

Experiment:

1. Encapsulate the existing SPO+ real-data scripts as a reproducible Lab 4
   comparator.
2. Compare two-stage, SPO+, and CRPTO on regret, feasibility, coverage, width
   and wealth.
3. Optional isolated PyEPO/cvxpylayers prototype on a small top-k/knapsack-like
   portfolio.
4. Optional online DFL replay if the batch comparator is stable.

Variants:

- 5 seeds for SPO+ training;
- 2018/2019/2020 temporal stability;
- small `n_items` PyEPO/cvxpylayers stress test;
- static regret and dynamic-regret proxy for online replay.

Evidence gate:

- SPO+ regret is measured on the same decision problem;
- conformal coverage is reported separately;
- dependency environment remains isolated;
- no claim is made from toy prototypes.

Destination:

- Paper 4 DFL comparator/prototype.
- Paper Estrella related work only, unless a clean comparator table already
  exists and strengthens positioning.

Stop rule:

- Park PyEPO/cvxpylayers if scaling, dependencies or gradients are fragile.

What not to claim:

- No end-to-end DFL implementation in the main pipeline.
- No guarantee that SPO+ provides conformal coverage.
- No online-deployment claim from historical replay.

### Lane 7: IFRS9-Inspired ECL/SICR Proxy

Claim target: Keep the IFRS lane honest: useful proxy diagnostic, not accounting
compliance.

Sources: Basel ECL guidance, IFRS 9 official material, survival/competing-risk
ECL literature.

Experiment:

1. Define proxy PD, LGD and EAD from current Lending Club fields.
2. Create Stage 1 versus lifetime proxy logic using term/vintage/default timing.
3. Add hardship, debt settlement, recovery and last-payment fields as diagnostic
   indicators where available.
4. Backtest proxy ECL against defaults and recoveries by vintage/grade.

Variants:

- simple lifetime PD from observed term horizon;
- survival/Cox or random survival forest if event timing is reliable enough;
- SICR proxy using FICO/risk-bucket migration if variables support it;
- macro/vintage stress only as scenario, not as official forecast.

Evidence gate:

- default definition explicit;
- EAD/LGD fields documented;
- backtest by vintage and grade;
- blocker table included.

Destination:

- Paper 4 appendix.
- Paper Estrella only as limitation/context if needed.

Stop rule:

- Park if monthly DPD, contractual schedule or macro scenarios are required for
  the next claim.

What not to claim:

- No IFRS9 compliance.
- No contractual lifetime ECL.
- No formal SICR.

### Lane 8: Explainability, Fairness Proxy And Governance Audit

Claim target: Turn governance into a useful audit appendix without legal claims.

Sources: FinRegLab ML credit underwriting report, SR 11-7, CFPB BISG, Zhang,
credit-scoring benchmarks.

Experiment:

1. Evaluate explanation stability for the champion model by vintage, grade, term
   and source group.
2. Compare SHAP/global importances against adverse-action-style reason ranks as
   a proxy diagnostic.
3. Run geography/source disparity diagnostics on approval/funding, interest rate,
   predicted risk, realized loss and portfolio selection.
4. If surname/tract data is absent, explicitly document why BISG is impossible.

Variants:

- SHAP stability by time slice;
- explanation drift stress;
- less-discriminatory-alternative screen by AUC/return/CVaR/proxy disparity;
- probability-weighted proxy metrics only if valid proxy probabilities exist.

Evidence gate:

- no protected-attribute claim;
- governance metric and data limitation table;
- explanation results stable enough to interpret.

Destination:

- Paper 4 governance appendix.
- Paper Estrella intro/model-risk paragraph if it improves reviewer trust.

Stop rule:

- Park if the result would read as legal fairness evidence.

What not to claim:

- No ECOA/fair-lending compliance.
- No individual race/ethnicity inference.
- No causal explanation from SHAP/LIME.

### Lane 9: Causal Diagnostics And CATE Boundary

Claim target: Use causal literature to define why the accepted-loan data cannot
support strong policy-value claims.

Sources: DoWhy, DML, causal forests, reject-inference literature.

Experiment:

1. Draw a minimal DAG for application, pricing, grade, funding, default and
   recovery.
2. Define one or two candidate estimands, such as high-rate-within-grade effect
   on default or loss.
3. Run DoWhy refuters: placebo treatment, random common cause, subset refuter and
   sensitivity.
4. Report overlap and stability by vintage.

Variants:

- treatment: high interest rate within grade/subgrade;
- treatment: 60m versus 36m term;
- outcome: default, realized LGD, net return.

Evidence gate:

- estimand and assumptions explicit;
- overlap acceptable;
- placebo/refuters do not obviously break the result;
- result is labelled observational sensitivity.

Destination:

- Paper 4 parked/diagnostic memo.

Stop rule:

- Park unless rejected applicants, randomized pricing, or a credible instrument
  appears.

What not to claim:

- No causal policy value.
- No treatment-effect recommendation.
- No claim that portfolio selection causes repayment outcomes.

## Missing Or Worth-Adding PDFs

These are worth collecting if possible, but none should block the Lab 4 plan:

| source | why useful | current action |
| --- | --- | --- |
| Jagtiani and Lemieux, "The Roles of Alternative Data and Machine Learning in Fintech Lending" | direct Lending Club / fintech-lending empirical context | add to Paper Estrella intro/governance context |
| Lessmann et al., "Benchmarking state-of-the-art classification algorithms for credit scoring" | standard credit-scoring benchmark | already in bibliography; useful for intro |
| Ben-Tal and Teboulle, "An Old-New Concept of Convex Risk Measures: The Optimized Certainty Equivalent" | OCE theory for tail-risk lane | add if OCE text stays |
| Zhang, "Assessing Fair Lending Risks Using Race/Ethnicity Proxies" | peer-reviewed BISG/proxy fair-lending method | add if fair-lending appendix stays |
| PyEPO paper | practical SPO+/predict-then-optimize implementation | add only if isolated prototype is discussed |
| End-to-end Conditional Robust Optimization | closest recent E2E CRO neighbor | add to Lab 4 source log if E2E prototype runs |

## Recommended Execution Order

1. Run Lane 1 and Lane 2 first. These most directly strengthen the CRPTO
   decision-risk story and can feed either Paper 4 or Paper Estrella.
2. Run Lane 5 next. CVaR/OCE is the strongest quantitative challenger appendix.
3. Run Lane 4 only after Lane 1/2 define the coverage/risk gates clearly.
4. Run Lane 8 and Lane 7 as governance/data-boundary appendices.
5. Run Lane 6 and Lane 3 only in isolated environments.
6. Run Lane 9 as a short diagnostic memo, not a long empirical loop.

## Output Contract

For each completed lane:

- one compact final table;
- one paragraph of interpretation;
- one explicit decision: promote, append, park, archive, or delete;
- one blocker/gate row;
- no versioned wave artifacts.

Final Lab 4 synthesis:

- lane decision matrix;
- source-to-claim map;
- Paper Estrella export memo only for results that strengthen the official
  manuscript without reopening the champion.
