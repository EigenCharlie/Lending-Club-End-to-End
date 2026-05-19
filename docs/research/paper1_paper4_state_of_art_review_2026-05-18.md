# Paper Estrella and Paper 4 State-of-the-Art Review - 2026-05-18

## Purpose

This memo consolidates a deeper literature pass for the official Paper Estrella
and the long-horizon Paper 4 living lab. It is intentionally not a new
experimental loop and not a Quarto rewrite. The output is a source map and a
claim-governance recommendation: what should strengthen the official manuscript,
what should remain in Paper 4 appendices, and what should stay parked until new
data or a reviewer request changes the decision.

Detailed source triage is in
`reports/paper_material/global/tables/paper1_paper4_state_of_art_sources_2026-05-18.csv`.

## Executive Decision

Paper Estrella should stay centered on the economic champion and the
conformal-robust portfolio contribution. The best literature additions are
positioning and reviewer defense, not new claims:

- Use conformal risk control, Learn-then-Test, conditional-coverage limits and
  conformal robust optimization to explain why the method controls decision risk
  without pretending to solve exact conditional validity.
- Use robust optimization and predict-then-optimize references to position CRPTO
  as prescriptive analytics with calibrated uncertainty, not merely prediction.
- Use credit scoring, fintech lending and model-risk references to explain why
  the Lending Club application is practically relevant and why auditability
  matters.
- Mention CVaR/OCE only as a tail-risk challenger appendix/sensitivity; do not
  reopen the champion.

Paper 4 should remain a governed living lab. Its strongest contribution is not a
near-term publication claim; it is a documented boundary map for seven ambitious
extensions. Four lanes are append-worthy as evidence appendices, while three
remain parked:

For a future full-paper route, Paper 4 also absorbs the useful prudential core
of Paper 2. The standalone Paper 2 route is parked unless stronger contractual
servicing and macro scenario data appear. The retained material enters Paper 4
as IFRS9-inspired proxy evidence: ECL scenarios, conformal ECL ranges, SICR
width signal, CIF/prepayment correction, stage-cost governance and TS-to-ECL
stress context.

| lane | decision | literature role | claim boundary |
| --- | --- | --- | --- |
| IFRS9/SICR | append | IFRS 9, ECL, survival and competing-risk sources justify a proxy diagnostic | not contractual IFRS 9 without monthly DPD, contractual terms and macro scenarios |
| CVaR/OCE | append | CVaR and OCE sources justify tail-risk challenger analysis | not champion replacement without paired wealth gains |
| fair-lending proxy | append | BISG, proxy-fairness and ML underwriting governance sources justify proxy-risk governance | not legal fair-lending evidence without surname, protected attributes or tract-level geography |
| DLA/ADP | append | SDAM/ADP sources justify sequential rollout framing | not exact Bellman optimality without decision logs and state transition panels |
| online conformal | park | ACI and multi-source conformal explain what would be needed | no live feedback or external source distribution |
| CATE/policy value | park | causal and double-ML sources support a sensitivity appendix only | no causal policy claim without rejected applicants or an instrument |
| SPO/DFL | park | SPO+, PyEPO and cvxpylayers explain the prototype direction | no main-pipeline integration while benefit is only toy/oracle-regret evidence |

## Paper Estrella: Best Literature Fit

### 1. Conformal Decision Guarantees

The official manuscript already has a solid conformal backbone. The highest
value additions are not more conformal breadth, but clearer hierarchy:

- Foundation: Vovk et al. and Romano et al. establish distribution-free
  conformal prediction and conformalized quantile regression.
- Risk control: Bates et al., Angelopoulos et al. on Conformal Risk Control, and
  Learn-then-Test support the paper's risk-control language.
- Boundary: Barber, Candes, Ramdas and Tibshirani on limits of distribution-free
  conditional predictive inference should be used to prevent overclaiming source
  or subgroup validity.
- Prescriptive bridge: Johnstone and Cox, Patel et al., and Sun et al.
  strengthen the conformal robust optimization / robust contextual LP framing.

Recommended use: one compact paragraph in the theory/literature section, plus a
reviewer-facing note that Paper Estrella controls an operational risk target and
does not claim exact conditional coverage for every borrower/source subgroup.

### 2. Robust Optimization and Prescriptive Analytics

The robust optimization literature should be used to position CRPTO as a
decision pipeline, not as a forecasting-only paper:

- Bertsimas and Sim is the core robust optimization reference for uncertainty
  budgets and the price of robustness.
- Conformal uncertainty sets for robust optimization and conformal contextual
  robust optimization are the closest methodological neighbors.
- Predict-then-Calibrate is especially useful because it frames robust contextual
  LP through calibration rather than pure point prediction.
- SPO+ and DFL should be cited only as adjacent decision-focused learning. Paper
  Estrella does not need to become a differentiable optimization paper.

Recommended use: add a "closest methods" contrast in Paper Estrella: CRPTO is
closer to calibrated robust prescriptive analytics than to end-to-end SPO/DFL.

### 3. Credit Scoring, Fintech Lending and Model Risk

The credit literature should support empirical relevance and governance:

- Lessmann et al. provide a widely cited credit-scoring benchmark and connect
  predictive accuracy with business value.
- Jagtiani and Lemieux are directly relevant because they study Lending Club and
  fintech lending with alternative data and loan grades.
- SR 11-7 and FinRegLab support model-risk governance, auditability,
  explainability and fair-lending sensitivity language.

Recommended use: strengthen the introduction and empirical setting. This is the
cleanest way to make the Lending Club application feel less like a dataset
exercise and more like a regulated credit decision problem.

### 4. Tail-Risk Challenger, Not Champion Reopening

CVaR and OCE references are valuable for reviewer defense if a reader asks why
the official champion is not tail-risk optimized:

- Rockafellar and Uryasev support the CVaR optimization formulation.
- Ben-Tal and Teboulle support OCE as a convex risk-measure family.
- The Paper 4 CVaR/OCE experiment is useful as a challenger appendix because it
  improved tail framing but did not beat paired wealth.

Recommended use: keep this as appendix/sensitivity. Do not reopen the economic
champion unless future paired replay beats it robustly.

## Paper 4: Best Literature Fit By Lane

### IFRS9/SICR

The literature supports a proxy diagnostic, not a full IFRS 9 implementation.
IFRS Foundation and Basel/EBA materials establish the accounting and supervision
context. Recent survival, competing-risk and term-structure ECL work supports
the idea that default timing and lifetime PD matter.

Paper 4 should keep this as an appendix because the project has useful cashflow,
hardship and recovery fields, but lacks monthly contractual days-past-due
history, original effective interest rate accounting infrastructure and macro
scenario paths. The honest title is "IFRS9-inspired SICR/ECL proxy diagnostic."

### Online Conformal / Source Holdouts

ACI, multi-source conformal and multi-distribution conformal references justify
the direction of the lane. The conditional-coverage impossibility literature is
also important: source-aware coverage is not free.

Paper 4 should keep the existing source holdout evidence as retrospective
governance only. It should not claim online/adaptive validity until there is
live feedback, a production-like stream, or a genuinely external source
distribution.

### CVaR/OCE

This lane is append-worthy. CVaR/OCE literature gives a clean mathematical
language for tail utility and risk aversion, and the existing Paper 4 result is
useful as a stress/challenger result. The blocker is not theory; it is empirical
dominance. The challenger did not replace the economic champion.

Paper 4 destination: appendix tail challenger and caveat for Paper Estrella.

### CATE / Policy Value

The causal literature is useful mainly because it tells us to stop. DoWhy,
double/debiased ML and causal-forest references require explicit identification,
overlap and refutation. Lending Club accepted-loan data can support an
observational sensitivity screen, but not a strong policy-value claim because
rejected applicants, randomized pricing or a credible instrument are absent.

Paper 4 destination: parked with a causal-identification memo, not a promoted
experiment.

### Fair-Lending Proxy

The fair-lending literature is valuable for claim boundaries. CFPB BISG and
Zhang's proxy-method literature show why surname plus fine geography matter. The
project has state and zip3, not surname or protected attributes.

Paper 4 destination: appendix source/proxy-governance risk. Legal fair-lending
claims remain false. This is a good example of a valuable negative result.

### DLA/ADP

Powell's SDAM/ADP framing fits Paper 4's role as a sequential decision lab, but
the data blocks exact dynamic programming. Lending Club snapshot fields can
support rollout-style diagnostics and state summaries, not Bellman optimality.

Paper 4 destination: appendix rollout/sequential analytics. Good framing, weak
optimality claim.

### SPO/DFL

SPO+, PyEPO and cvxpylayers establish a legitimate frontier, but the current
project only has isolated toy/oracle-regret evidence. Integrating this would
increase dependency and maintenance risk without improving the official
champion.

Paper 4 destination: parked prototype. Reopen only if there is a specific
reviewer request or a compact optimization benchmark that directly dominates a
CRPTO comparator.

## Candidate Additions To Bibliography Later

Do not add every source immediately. The next bibliography patch should be small
and tied to actual Quarto text. Highest priority candidates not yet fully
integrated into the official narrative are:

- Barber et al. on limits of distribution-free conditional predictive inference.
- Jagtiani and Lemieux on Lending Club fintech lending.
- FinRegLab's ML credit underwriting policy/empirical reports.
- Rockafellar and Uryasev on CVaR if the tail-risk appendix is cited in Paper
  Estrella.
- Ben-Tal and Teboulle on OCE if OCE remains in the appendix text.
- CFPB BISG and Zhang if Paper 4 keeps a fair-lending/proxy governance appendix.
- Agrawal et al. and Tang/Khalil only if the SPO/DFL prototype is referenced.

## Anti-Loop Reopen Gates

Reopen a parked or append lane only if one of these happens:

- A monthly servicing panel becomes available, including contractual DPD,
  payment states and macro scenario paths.
- Rejected-applicant data or randomized pricing/instrumental variation becomes
  available.
- Surname, tract-level geography or protected-attribute proxy inputs become
  available with an approved governance plan.
- A reviewer explicitly asks for a lane and the response can be answered with
  one compact table/memo.
- A future run can change the official Paper Estrella champion under the paired
  wealth gate.

Otherwise, the correct next move is not more experiments; it is citation
integration and manuscript extraction.

## Paper Estrella Integration Patch

The 2026-05-18 integration patch keeps the bibliography small and tied to
actual manuscript text. It adds only sources that now appear in Paper Estrella:

| source | destination | reason |
| --- | --- | --- |
| `barber2021_limits_conditional` | `14a`, `14b` | Bound source/subgroup claims and avoid overstating conditional validity. |
| `jagtiani2019` | `14a` | Establish Lending Club as a real fintech-lending empirical setting. |
| `rockafellar2000_cvar` | `14h` | Ground CVaR as a canonical tail-risk diagnostic. |
| `bental_teboulle2007_oce` | `14h` | Ground OCE as convex risk-measure framing. |
| `finreglab2023_ml_credit` | `14k`, `14l` | Support ML credit underwriting governance, explainability and fairness context. |
| `cfpb_bisg_proxy` | `14k` | Document why surname plus fine geography are needed for BISG-style proxy analysis. |
| `zhang2018_fair_proxy` | `14k` | Support the fair-lending proxy boundary and why current Lending Club fields are insufficient for legal claims. |

No IFRS9, SPO/DFL prototype or Paper 4-only sources were integrated into Paper
Estrella in this patch. Those remain Lab 4/Paper 4 material unless a future
appendix or reviewer request creates a concrete textual need.

## Metrics Binder Addendum

The local binder `Metrics for Credit and ML Models.pdf` was triaged on
2026-05-18 and logged in
`docs/research/metrics_credit_ml_models_triage_2026-05-18.md`.

The only source that should enter Paper Estrella as a methodological support is
Wuthrich's Gini/autocalibration result: it reinforces the current claim that
CRPTO is not an AUC leaderboard, and that rank metrics are meaningful only after
the PD layer is calibration-gated. Albanesi and Vamossy can support the credit
scoring/equity motivation, but only as context; it does not authorize a legal
fair-lending claim for Lending Club.

For Paper 4, the binder opens one bounded future appendix: FICO/score proxy vs
champion ML, with misclassification, ranking difference and observable-group
diagnostics. Dinga et al. is taxonomy-only, Somers' D is optional metric
sensitivity, and ReScorer stays parked unless the project later audits
LLM-generated research reasons.
