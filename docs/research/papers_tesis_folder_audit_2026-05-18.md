# Papers_tesis Folder Audit - 2026-05-18

## Scope

Folders audited:

- Paper Estrella pack:
  `/mnt/c/Users/carlos/Documents/Paper_CRPTO/Papers_tesis`
- Paper 4 pack:
  `/mnt/c/Users/carlos/Documents/Claude Code/lending-club-risk-project/Paper 4`

Inventory:

- 30 PDF files total.
- 18 core PDFs are in the Paper Estrella pack root.
- 5 future-work PDFs are in `Papers_tesis/shared_future_work`.
- 7 PDFs are in the Paper 4 pack.
- The two newly added Powell books are machine-readable with `pdftotext`
  (`45,976` and `105,095` extracted words respectively).
- `Decision-Focused Learning - Foundations, State of the Art, Benchmark and Future Opportunities.pdf` replaces the earlier non-text-extractable browser print of the Mandi et al. survey.
- `Conformal Contextual Robust Optimization.pdf` and `An_Old-New_Concept_of_Convex_Risk_Measures_The_Opt.pdf` were added after the first audit and are now covered below.
- `Conformal Risk Training`, end-to-end conformal calibration, online DFL,
  UP-OCP and the Mandi survey are now explicitly staged in
  `shared_future_work` rather than the core Paper Estrella pack.
- `Powell-Bridging-Vol-I-Framing-Kindle-Jan-7-2026-w_cover.pdf` and
  `Powell-Kindle-SDAM-2nd-ed-Feb-9-2026-w_cover.pdf` are now in the Paper 4
  pack.

This audit is claim-governed: the question is not "can we cite everything?",
but whether each paper changes Paper Estrella, Paper 4, or neither.

## Executive Decision

All 30 folder papers were reviewed or reconciled against the current
bibliography and Quarto surface.

Paper Estrella already uses the core papers it should use:

- conformal risk control / learn-then-test / RCPS;
- conformal robust optimization / PTC / CROMS / CRS;
- robust optimization and SPO/DFL comparators;
- Lending Club fintech context;
- CVaR/OCE tail-risk appendix;
- model-risk/fair-lending governance boundaries.

Paper 4 is the correct destination for the papers that are frontier, blocked,
or prototype-only:

- DoWhy and causal-assumption validation;
- Basel/ECL/IFRS9 guidance;
- multi-source/online conformal;
- cvxpylayers and differentiable optimization layers;
- end-to-end conditional robust optimization;
- online DFL and UP-OCP as future-work/prototype context;
- Powell SDAM / sequential-decision framing for Paper 4 and bounded Paper
  Estrella framing.

No paper in the folder currently justifies reopening the Paper Estrella
champion.

## Current Folder Split

The split is now coherent:

- `Papers_tesis` contains the sources that are central or directly useful for
  Paper Estrella: conformal risk control, robust optimization, conformal robust
  optimization, CVaR/OCE, Lending Club context, model-risk governance and
  proxy/fair-lending boundaries.
- `Papers_tesis/shared_future_work` contains future-work sources that Paper
  Estrella may cite only as frontier context: conformal risk training,
  end-to-end conformal calibration, online conformal, online DFL and the DFL
  survey.
- `Paper 4` contains sources that remain lab/frontier material: causal
  assumptions, IFRS9/ECL, multi-source conformal, cvxpylayers, end-to-end
  conditional robust optimization and the two Powell books for SDAM/sequential
  analytics framing.

Some `Paper 4` PDFs are still cited in Paper Estrella as future-work or
contrast references. That is acceptable if `Papers_tesis` is interpreted as the
central Paper Estrella pack rather than a complete copy of every cited source.

## Detailed Paper Matrix

| # | Folder file | Paper / source | Year / status | Core idea | Main result / conclusion | Limitations for this project | Current use | Decision |
|---:|---|---|---|---|---|---|---|---|
| 1 | `10_Conformal_Robust_Optimizati.pdf` | Zhao, Jiang, Qi, *Conformal Robust Optimization and Satisficing for Prescriptive Analytics with Black-Box Predictors* | 2026 workshop/preprint | Converts black-box forecasts into conformal uncertainty sets for robust optimization and conformal robust satisficing. | CRO and CRS are linked by an explicit parameter mapping; experiments show better robust objectives and feasibility in knapsack-style decisions. | General prescriptive analytics; not credit-specific; no funded-set PD bound or Lending Club evidence. | Paper Estrella cites `zhao2025robust`; Paper 4 uses it as selector/governance context. | Keep in Paper Estrella positioning, not as baseline requirement. |
| 2 | `32_dowhy_icml_causal_assumptions_workshop_2021_final.pdf` | Sharma et al., *DoWhy: Addressing Challenges in Expressing and Validating Causal Assumptions* | 2021 ICML NACI workshop | Causal estimation requires explicit assumptions and partial validation. | There is no global validator for causal estimates; graphs, refutations and assumption checks are first-class stages. | Supports stopping CATE policy-value claims, not promoting them. | Paper 4 Lane 9 / causal boundary. | Keep out of Paper Estrella; cite only if Paper 4 CATE appendix is written. |
| 3 | `Conformal Risk Training End-to-End Optimization of Conformal Risk Control.pdf` | Yeh et al., *Conformal Risk Training* | 2025 NeurIPS / arXiv | End-to-end optimization of conformal risk control, including OCE/CVaR. | Extends CRC to OCE risk control and differentiates through conformal risk during training. | Requires end-to-end training and exchangeability/monotone loss; current project is post-hoc CRPTO. | Paper Estrella cites `yeh2025training` as frontier, not method. | Keep as future-work contrast; do not implement now. |
| 4 | `Conformal Uncertainty Sets for Robust Optimization.pdf` | Johnstone and Cox, *Conformal Uncertainty Sets for Robust Optimization* | 2021 COPA / PMLR | Uses conformal prediction regions as robust optimization uncertainty sets. | Gives finite-sample valid conservative uncertainty sets; empirical performance needs more robust-optimization study. | Simulated setting; no credit/funded-set instantiation. | Paper Estrella cites `johnstone2021`. | Keep as closest-method foundation. |
| 5 | `Conformal_Risk_Control.pdf` | Angelopoulos et al., *Conformal Risk Control* | 2024 ICLR | Controls expected value of monotone bounded losses with finite-sample conformal calibration. | Generalizes split conformal to arbitrary monotone loss; includes shift/multiple/adversarial extensions. | Expected-risk control does not alone prove source-level or live deployment validity. | Paper Estrella cites `angelopoulos2024risk`. | Core Paper Estrella theory support. |
| 6 | `Distribution-Free, Risk-Controlling Prediction Sets.pdf` | Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets* | 2021 arXiv / risk-control foundation | Calibrates set-valued predictions to control expected loss. | Works post-hoc with any black-box predictor and finite-sample holdout calibration. | General prediction sets; not portfolio-specific. | Paper Estrella cites `bates2021rcps`. | Core Paper Estrella support. |
| 7 | `End-to-End Conformal Calibration for Optimization Under Uncertainty.pdf` | Yeh et al., *End-to-End Conformal Calibration for Optimization Under Uncertainty* | TMLR 2025, arXiv v2 2026 | Learns uncertainty sets for conditional robust optimization with downstream decision performance. | End-to-end calibration can pick uncertainty sets that are valid and decision-useful. | Requires neural/end-to-end setup; Paper Estrella is post-hoc and auditable. | Paper Estrella cites `yeh2026`; Paper 4 Lane 3 context. | Keep as frontier contrast, not implementation target. |
| 8 | `End-to-end Conditional Robust Optimization.pdf` | Chenreddy and Delage, *End-to-end Conditional Robust Optimization* | 2024 UAI/preprint | Differentiable end-to-end conditional robust optimization balancing decision risk and conditional coverage. | Empirically improves conditional coverage/objective tradeoffs with differentiable optimization. | The authors note conformal-theory success guarantees for conditional objective are impossible in that framing; project lacks integrated stack. | In `references.bib` as `chenreddy2024`, not currently Paper Estrella-cited. | Keep Paper 4-only unless writing end-to-end frontier section. |
| 9 | `FinRegLab_2023-12-07_Research-Report_Explainability-and-Fairness-in-Machine-Learning-for-Credit-Undewriting_Policy-Analysis.pdf` | FinRegLab, *Explainability and Fairness in ML for Credit Underwriting: Policy Analysis* | 2023 policy report | Governance, explainability and fairness controls for ML credit underwriting. | Supports treating fair lending and explainability as deployment governance controls, not pure model metrics. | Policy report, not theorem or empirical CRPTO evidence. | Paper Estrella cites `finreglab2023_ml_credit` in fair lending/MRM. | Keep in Paper Estrella governance. |
| 10 | `Guidance on credit risk and accounting for expected credit losses.pdf` | Basel Committee, *Guidance on credit risk and accounting for expected credit losses* | 2015 official guidance | Supervisory expectations for ECL credit-risk practices. | Strong governance source for IFRS9/ECL-style credit-risk measurement. | Current data lacks monthly contractual DPD, EAD paths and macro scenario infrastructure. | Paper 4 IFRS9/SICR boundary only. | Do not cite Paper Estrella now; add to Paper 4 IFRS appendix if drafted. |
| 11 | `Learn then test.pdf` | Angelopoulos et al., *Learn then Test* | 2025 Annals of Applied Statistics | Calibrates predictive algorithms to satisfy finite-sample risk guarantees. | Model-agnostic post-hoc tests can control risk without retraining. | Does not by itself solve downstream portfolio selection. | Paper Estrella cites `angelopoulos2025ltt`; Lane 1 governance context. | Core Paper Estrella theory support. |
| 12 | `Multi-Distribution Robust Conformal Prediction.pdf` | Yang and Jin, *Multi-Distribution Robust Conformal Prediction* | 2026 arXiv/preprint | Uniform conformal validity over multiple source distributions or mixtures. | Max-p aggregation gives finite-sample multi-distribution coverage. | Future-looking; project source groups are retrospective and not external/live. | Paper Estrella cites `yang2026multidistribution`; Paper 4 Lane 4 boundary. | Keep as future-work/source-robustness context. |
| 13 | `Multi-Source Conformal Inference Under Distribution Shift.pdf` | Liu et al., *Multi-Source Conformal Inference Under Distribution Shift* | 2024 arXiv/preprint | Prediction intervals under multiple sources with heterogeneous conditional outcome distributions. | Relevant for source-aware conformal inference under shift. | Needs a stronger source-shift setup than current Lending Club retrospective split. | Paper 4 Lane 4 source-holdout only; not in Paper Estrella bibliography. | Keep source-log only unless Paper 4 online/source appendix is written. |
| 14 | `NeurIPS-2019-differentiable-convex-optimization-layers-Paper.pdf` | Agrawal et al., *Differentiable Convex Optimization Layers* | 2019 NeurIPS | Differentiates through disciplined convex programs via cvxpylayers. | Makes convex optimization layers usable in neural architectures. | Dependency/DPP constraints and maintenance cost; not needed for official champion. | Paper 4 Lane 6/SPO-DFL prototype only. | Do not cite Paper Estrella; reopen only with working isolated prototype. |
| 15 | `ONLINE DECISION-FOCUSED LEARNING.pdf` | Capitaine et al., *Online Decision-Focused Learning* | 2026 ICLR | DFL in dynamic environments with changing objectives/distributions. | Regularization and online methods address non-smooth decision losses. | Orthogonal to post-hoc CRPTO; no coverage/auditability guarantee. | Paper Estrella cites `capitaine2026online` as future-work/contrast. | Keep as contrast; no implementation now. |
| 16 | `Online Conformal Prediction via Universal Portfolio Algorithms.pdf` | Liu, Dobriban, Orabona, *Online Conformal Prediction via Universal Portfolio Algorithms* | 2026 arXiv/preprint | Online conformal prediction with regret-to-coverage theory and parameter-free UP-OCP. | Long-run coverage for arbitrary streams via universal portfolio reduction. | Project lacks production feedback stream; current evidence is retrospective. | Paper Estrella future work; Paper 4 Lane 4 parked. | Keep as future-work citation, not current claim. |
| 17 | `Optimal Model Selection for Conformalized Robust Optimization.pdf` | Bao et al., *Optimal Model Selection for Conformalized Robust Optimization* | 2025 arXiv/preprint | CROMS selects models for conformalized robust optimization by downstream decision risk. | Shows model selection should account for robust decision loss, not just prediction. | Current project has only CROMS-lite selector over retained artifacts. | Paper Estrella cites `bao2025croms`; Paper 4 Lane 2 selector. | Keep; do not claim full CROMS implementation. |
| 18 | `Optimization of Conditional Value-at-Risk.pdf` | Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk* | 2000 Journal of Risk | CVaR optimization as tractable tail-risk minimization. | CVaR can be optimized directly and is more coherent/useful than VaR. | Tail-risk improvement does not replace wealth champion without paired dominance. | Paper Estrella appendix cites `rockafellar2000_cvar`; Paper 4 Lane 5. | Keep appendix citation. |
| 19 | `Predict-then-Calibrate.pdf` | Sun, Liu, Li, *Predict-then-Calibrate* | 2024 arXiv / NeurIPS-era robust contextual LP | Separates prediction and uncertainty calibration for robust contextual LP. | Produces box/ellipsoid uncertainty sets with population coverage for robust LP. | Coverage is for parameter/cost set, not funded-set PD violation; no credit-specific governance. | Paper Estrella cites `sun2024ptc`. | Core closest-neighbor contrast. |
| 20 | `Smart “Predict, then Optimize”.pdf` | Elmachtoub and Grigas, *Smart Predict, then Optimize* | 2022 Management Science | SPO/SPO+ trains prediction for downstream optimization quality. | SPO+ is convex surrogate and can beat prediction-error training under misspecification. | Lower regret does not imply conformal coverage, auditability or tail governance. | Paper Estrella cites `elmachtoub2022`. | Keep as primary DFL comparator. |
| 21 | `Task-based End-to-end Model Learning in Stochastic Optimization.pdf` | Donti, Amos, Kolter, *Task-based End-to-end Model Learning in Stochastic Optimization* | 2017 NeurIPS / arXiv | Learns probabilistic models through stochastic optimization objectives. | Improves downstream task performance in inventory, grid scheduling and energy arbitrage. | Domain/general end-to-end method; no credit/coverage claim. | Paper Estrella cites `donti2017`. | Keep as DFL lineage. |
| 22 | `The Price of Robustness.pdf` | Bertsimas and Sim, *The Price of Robustness* | 2004 Operations Research | Robust LP with tunable conservatism and probabilistic violation bounds. | Protection level trades objective value for robustness; tractable for LP/IP. | Robustness budget is chosen, not conformally calibrated. | Paper Estrella cites `bertsimas2004`. | Core robust-optimization foundation. |
| 23 | `The limits of distribution-free conditional predictive inference.pdf` | Barber, Candes, Ramdas, Tibshirani, *Limits of Distribution-Free Conditional Predictive Inference* | 2021 Information and Inference | Shows exact distribution-free conditional coverage is impossible without assumptions. | Defines what relaxations of conditional validity can be possible. | It is a boundary source, not a method that improves results. | Paper Estrella cites `barber2021_limits_conditional`. | Keep to prevent overclaiming. |
| 24 | `Using-Publicly-Available-Information-to-Proxy.pdf` | CFPB, *Using publicly available information to proxy for unidentified race and ethnicity* | 2014 official methodology | BISG-style proxy using surname and census geography. | Explains how race/ethnicity proxying is constructed and assessed when protected attributes are missing. | Project lacks surname and fine geography; zip3/state alone is insufficient. | Paper Estrella cites `cfpb_bisg_proxy`; Paper 4 Lane 8 boundary. | Keep as limitation/governance citation. |
| 25 | `Decision-Focused Learning - Foundations, State of the Art, Benchmark and Future Opportunities.pdf` | Mandi et al., *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities* | 2024 JAIR-accepted / arXiv v4 | Survey of DFL foundations, benchmarks and open problems. | Establishes SPO/DFL as a mature adjacent literature, separates gradient-based and gradient-free methods, benchmarks 11 methods over seven problems, and finds no single DFL method dominates everywhere. | DFL optimizes downstream regret but does not provide conformal coverage, source robustness or MRM auditability by itself. | Paper Estrella cites `mandi2024`; Paper 4 Lane 6 uses it as SPO/DFL context. | Keep as the main DFL survey and as support for not turning Paper Estrella into an end-to-end DFL paper. |
| 26 | `the-roles-of-alternative-data.pdf` | Jagtiani and Lemieux, *The Roles of Alternative Data and Machine Learning in Fintech Lending* | 2019 Financial Management / FRB WP | Studies LendingClub, alternative data, grades and loan performance. | LendingClub grades correlate with loan performance and increasingly differ from FICO, indicating alternative-data use. | Context source only; does not validate CRPTO or fair-lending legality. | Paper Estrella cites `jagtiani2019`. | Keep as empirical setting support. |
| 27 | `Conformal Contextual Robust Optimization.pdf` | Patel, Rayan and Tewari, *Conformal Contextual Robust Optimization* | 2023 arXiv / AISTATS-era preprint | Conformal-Predict-Then-Optimize (CPO) uses informative nonconvex conformal prediction regions from conditional generative models for robust decision-making. | Proposes CPO plus interpretable representative summaries of uncertainty regions; demonstrates on simulation-based inference benchmarks and weather-aware routing. | High-dimensional/generative uncertainty regions and black-box robust optimization; not credit-specific and not a funded-set PD bound. | Paper Estrella cites `patel2024` as a close conformal robust optimization neighbor. | Keep in Paper Estrella related work; no implementation required. |
| 28 | `An_Old-New_Concept_of_Convex_Risk_Measures_The_Opt.pdf` | Ben-Tal and Teboulle, *An Old-New Concept of Convex Risk Measures: The Optimized Certainty Equivalent* | 2007 Mathematical Finance | Re-examines optimized certainty equivalent (OCE) as a utility-based decision criterion and convex risk-measure family. | Shows negative OCE gives convex risk measures, connects OCE to phi-divergences and derives CVaR as a special case for a piecewise-linear utility. | Mathematical risk-measure foundation only; does not change the empirical champion without paired wealth dominance. | Paper Estrella cites `bental_teboulle2007_oce` in the tail-risk appendix; Paper 4 Lane 5 uses it for CVaR/OCE framing. | Keep as appendix foundation, not main-method claim. |
| 29 | `Powell-Bridging-Vol-I-Framing-Kindle-Jan-7-2026-w_cover.pdf` | Powell, *Bridging the Gap Between Stochastic Optimization and Sequential Decision Analytics, Vol. I: Framing* | 2026 book / monograph | Frames sequential decision analytics and the universal modeling elements behind state, decision, exogenous information, transition and contribution. | Gives vocabulary for separating one-period decision policies from genuinely sequential systems. | Book-level framing, not empirical evidence; does not convert CRPTO into DLA or online deployment. | Paper Estrella uses Powell only to classify CRPTO as a CFA-style one-period policy; Paper 4 can expand the SDAM framing. | Keep in Paper 4 pack; cite Paper Estrella only as bounded framing. |
| 30 | `Powell-Kindle-SDAM-2nd-ed-Feb-9-2026-w_cover.pdf` | Powell, *Sequential Decision Analytics and Modeling*, 2nd ed. | 2026 book / monograph | Systematizes PFA/CFA/VFA/DLA policy classes and sequential decision modeling. | Useful taxonomy for why CRPTO is an auditable CFA, while richer online/DLA versions are future work. | Does not provide a new result for Lending Club; overuse would make Paper Estrella sound sequential when it is currently uniperiod. | Paper Estrella cites `powell2026sdam`; Paper 4 stores the detailed SDAM/future-work context. | Keep in Paper 4 pack; do not expand Paper Estrella beyond bounded SDAM language. |

## What Is Already Applied To Paper Estrella

The Paper Estrella surface already cites the relevant folder papers for the
official claim:

- `bertsimas2004`: robust optimization foundation.
- `johnstone2021`, `sun2024ptc`, `zhao2025robust`, `bao2025croms`: conformal robust / prescriptive analytics neighbors.
- `bates2021rcps`, `angelopoulos2024risk`, `angelopoulos2025ltt`, `barber2021_limits_conditional`: risk-control and claim-boundary theory.
- `elmachtoub2022`, `donti2017`, `mandi2024`, `capitaine2026online`: DFL/SPO contrast.
- `yeh2025training`, `yeh2026`, `yang2026multidistribution`, `liu2026portfolio`: frontier/future-work context.
- `jagtiani2019`: Lending Club fintech empirical setting.
- `rockafellar2000_cvar`: tail-risk appendix.
- `finreglab2023_ml_credit`, `cfpb_bisg_proxy`: governance and fair-lending boundary.

## What Belongs In Paper 4, Not Paper Estrella

These folder papers are useful but should remain Paper 4 / Lab 4 material until
a real manuscript section needs them:

- DoWhy causal-assumption validation: Lane 9 CATE boundary.
- Basel ECL guidance: Lane 7 IFRS9/SICR proxy boundary.
- Multi-source conformal: Lane 4 online/source holdout boundary.
- cvxpylayers / differentiable convex layers: Lane 6 SPO/DFL prototype boundary.
- End-to-end conditional robust optimization: Lane 3 or 6 frontier context.
- Powell SDAM books: Paper 4 sequential-decision framing; Paper Estrella should
  only use the CFA/uniperiod boundary.

## Necessary Folder Gaps: Paper Estrella Citations Not Present In `Papers_tesis`

The following Paper Estrella citation is in the manuscript but not represented
as a PDF in the folder and is worth adding because it supports active Paper
Estrella text:

| key | role | action |
|---|---|---|
| `zhang2018_fair_proxy` | fair-lending proxy boundary, now cited | Worth adding PDF because Paper Estrella cites it. |

Optional but not necessary for the current Paper Estrella patch:

- `vovk2005`: foundational book; OK outside the folder.
- `angelopoulos2023`: gentle introduction; useful but not essential.
- `gibbs2021aci`: future-work online conformal context.
- `sr117`: official MRM guidance; web source is acceptable.
- `powell2026sdam`: no longer missing as a local source; the books now live in
  the Paper 4 pack and are used only as bounded SDAM framing.

`mandi2024` is no longer missing: a text-readable PDF is now present and has
been reviewed.

`patel2024` and `bental_teboulle2007_oce` are no longer missing: both readable
PDFs are now present and reviewed.

## Mandi et al. 2024 Detailed Read

`Decision-Focused Learning - Foundations, State of the Art, Benchmark and Future Opportunities.pdf`
is the correct readable copy of `mandi2024`.

- **Publication/date status:** arXiv v4 dated 2024-09-04; the PDF states that
  the article has been accepted for publication in JAIR.
- **Concepts:** decision-focused learning integrates ML and constrained
  optimization end-to-end so that the ML model is trained against downstream
  task loss, not only prediction error.
- **Method taxonomy:** the survey separates gradient-free DFL from
  gradient-based DFL, and organizes gradient-based methods into analytical
  differentiation of optimization mappings, analytical smoothing, random
  perturbation smoothing and surrogate-loss differentiation.
- **Benchmark:** the paper benchmarks 11 DFL methods across seven problems and
  releases code/data at `PredOpt/predopt-benchmarks`.
- **Results:** no method dominates across all tasks. SPO is robust across
  benchmark problems, MAP and learning-to-rank losses can be more scalable, and
  methods that rely on relaxations can fail when the relaxed LP differs
  materially from the integer problem.
- **Limitations/open directions:** robust risk-sensitive DFL remains sparse;
  many methods target expected regret rather than tail or worst-case risk;
  uncertainty in constraints is underexplored; scalability and theoretical
  guarantees remain open.
- **Paper Estrella implication:** keep Mandi as the main survey citation for
  DFL/SPO context, but use it to sharpen the contrast: CRPTO is not trying to
  beat DFL on every regret benchmark; it buys conformal coverage,
  auditability, robust-region traceability and MRM-friendly decision evidence.
- **Paper 4 implication:** Mandi supports Lane 6 as a legitimate long-horizon
  SPO/DFL appendix/prototype lane, but also justifies parking it until there is
  a compact prototype that changes regret evidence without exploding
  dependencies.

## What To Improve

1. Add the remaining missing PDF to `Papers_tesis` if available:
   `zhang2018_fair_proxy`.
2. Keep Basel/ECL, DoWhy, Multi-Source conformal and cvxpylayers out of Paper
   Estrella unless a reviewer asks for those extensions.
3. Do not add `basel_ecl_2015`, `multi_source_conformal_2024` or
   `agrawal2019_cvxpylayers` to the Paper Estrella bibliography now; they would
   inflate the references without changing the official claim.
4. If Paper 4 is ever drafted, write appendices in this order:
   (a) selector/governance, (b) tail-risk CVaR/OCE, (c) IFRS9 proxy,
   (d) source/fairness governance, (e) parked lanes with explicit blockers.

## Closed Decisions

- Paper Estrella does not need more experiments from these PDFs.
- Paper Estrella bibliography should stay selective; the current integration is
  enough for the official champion paper.
- Paper 4 remains the destination for ambitious but data-blocked or
  prototype-only ideas.
- No folder paper justifies creating `paper4_final_promotion.json` or reopening
  the Paper Estrella champion.

## Folder Papers That Do Not Go To Paper Estrella

These papers are useful to the project but should not be inserted into Paper
Estrella's bibliography/text now, because they do not support the official
CRPTO champion claim or they belong to blocked Paper 4 lanes:

| Folder paper | Why not Paper Estrella now | Correct sink |
|---|---|---|
| `32_dowhy_icml_causal_assumptions_workshop_2021_final.pdf` | Causal assumptions/refutations are relevant only if claiming treatment effects or policy value. Paper Estrella does not make a CATE claim. | Paper 4 Lane 9 CATE boundary. |
| `Guidance on credit risk and accounting for expected credit losses.pdf` | ECL/IFRS9 guidance is accounting-governance context; Paper Estrella is not an IFRS9 implementation. | Paper 4 Lane 7 IFRS9/SICR proxy appendix. |
| `Multi-Source Conformal Inference Under Distribution Shift.pdf` | Current source-holdout evidence is retrospective and below live-deployment gate. | Paper 4 Lane 4 online/source conformal. |
| `NeurIPS-2019-differentiable-convex-optimization-layers-Paper.pdf` | cvxpylayers is dependency/prototype infrastructure; not part of the official CRPTO pipeline. | Paper 4 Lane 6 SPO/DFL prototype. |
| `End-to-end Conditional Robust Optimization.pdf` | End-to-end conditional robust optimization is a frontier direction; Paper Estrella's claim is post-hoc, auditable CRPTO. | Paper 4 frontier note / Lane 3 or Lane 6. |
| `Conformal Risk Training End-to-End Optimization of Conformal Risk Control.pdf` | Useful as frontier contrast, but not part of the implemented method and should not become a new Paper Estrella claim. | Paper Estrella future-work citation only; Paper 4 frontier context. |
| `End-to-End Conformal Calibration for Optimization Under Uncertainty.pdf` | Similar frontier contrast; no current end-to-end calibration run in the official pipeline. | Paper Estrella related/future work; Paper 4 Lane 3 context. |
| `Online Conformal Prediction via Universal Portfolio Algorithms.pdf` | Online conformal requires stream feedback; the project has retrospective holdouts. | Paper Estrella future work; Paper 4 Lane 4 parked. |
| `ONLINE DECISION-FOCUSED LEARNING.pdf` | Online DFL is an adjacent future direction and does not supply conformal coverage/auditability for the current champion. | Paper Estrella contrast/future work only. |
| `Decision-Focused Learning - Foundations, State of the Art, Benchmark and Future Opportunities.pdf` | Keep as a survey citation, but do not expand Paper Estrella into a DFL benchmark paper. | Paper Estrella contrast; Paper 4 Lane 6 context. |
| `Powell-Bridging-Vol-I-Framing-Kindle-Jan-7-2026-w_cover.pdf` | Supports SDAM/sequential framing but does not change the current one-period CRPTO claim. | Paper 4 SDAM/sequential-decision context; bounded Paper Estrella framing. |
| `Powell-Kindle-SDAM-2nd-ed-Feb-9-2026-w_cover.pdf` | Same: useful taxonomy, not new empirical or theoretical evidence for the champion. | Paper 4 SDAM/sequential-decision context; bounded Paper Estrella framing. |

Everything else in the folder either is already part of Paper Estrella's core
positioning or is kept as bounded appendix/governance support.
