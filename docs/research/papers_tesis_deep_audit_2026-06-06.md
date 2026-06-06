# Papers_tesis Deep Audit - 2026-06-06

## Resumen ejecutivo

Esta auditoría cubre `61` PDFs locales en `/mnt/c/Users/carlos/Documents/Paper_CRPTO/Papers_tesis` y fue generada para el corte `2026-06-06` con `scripts/research/build_papers_tesis_deep_audit.py`.

La decisión central no cambia: **Paper CRPTO conserva el champion oficial** y la literatura nueva se usa para reforzar teoría, related work, appendices y límites de claim. Paper 4 absorbe el material que sí pertenece al laboratorio vivo: source/shift conformal, utility-directed conformal, tail risk, DFL, IFRS9 proxy, data/noise/equity y governance.

Artefactos generados:

- Matriz fuente: `reports/paper_material/global/tables/papers_tesis_source_matrix_2026-06-06.csv`
- Índice compacto de captions: `reports/paper_material/global/tables/papers_tesis_figure_caption_index_2026-06-06.csv`
- Curaduría de visual sinks: `reports/paper_material/global/tables/papers_tesis_curated_visual_sinks_2026-06-06.csv`

## Inventario

| folder | pdfs |
| --- | --- |
| paper | 21 |
| supplement | 28 |
| tesis | 12 |

## Decisiones por destino editorial

| decision | n |
| --- | --- |
| append_claim_boundary | 2 |
| append_comparator | 3 |
| append_conformal_variant | 1 |
| append_crpto_related_work | 1 |
| append_crpto_selector | 1 |
| append_future_work | 3 |
| append_governance | 1 |
| append_paper4_context | 1 |
| append_paper4_governance | 4 |
| append_paper4_ifrs9 | 1 |
| append_paper4_source_shift | 3 |
| append_tail_risk | 2 |
| context_credit_domain | 3 |
| context_crpto_related_work | 2 |
| context_decision_conformal | 1 |
| context_decision_support | 1 |
| context_finance_cp | 2 |
| context_governance | 1 |
| context_source_shift | 1 |
| append_mixed_diagnostic_source_not_better | 1 |
| append_mixed_diagnostic_wider_than_mondrian | 1 |
| park_future_dfl | 1 |
| park_future_work | 11 |
| park_width_fail | 1 |
| promote_crpto_body | 11 |
| thesis_foundation | 1 |

## Acciones requeridas

| action_required | n |
| --- | --- |
| completed_bib_quarto_patch | 7 |
| completed_quarto_patch | 1 |
| experiment_completed_appendix_diagnostic | 2 |
| experiment_completed_parked | 1 |
| none_now | 50 |

## Familias conceptuales

| primary_domain | n |
| --- | --- |
| AI/OR digital lending | 1 |
| CQR | 1 |
| CROMS / model selection | 1 |
| CVaR optimization | 1 |
| ECL / IFRS9 governance | 1 |
| ML credit governance | 1 |
| ML credit markets / equity | 1 |
| OCE / convex risk measures | 1 |
| P2P credit decision support | 1 |
| P2P credit portfolio optimization | 1 |
| P2P investment recommendation | 1 |
| SPO+ / predict-then-optimize | 1 |
| adaptive conformal inference | 1 |
| beyond exchangeability / weighted conformal | 1 |
| conditional conformal guarantees | 1 |
| conditional coverage limits | 1 |
| conformal contextual robust optimization | 1 |
| conformal decision theory | 1 |
| conformal foundations | 2 |
| conformal inverse optimization | 1 |
| conformal portfolio optimization | 1 |
| conformal portfolio selection | 1 |
| conformal risk control | 1 |
| conformal robust optimization | 1 |
| conformal robust optimization / satisficing | 1 |
| conformal robustness / decision risk | 1 |
| consumer credit ML | 1 |
| covariate shift conformal systems | 1 |
| credit data noise / disparities | 1 |
| credit invisibles / unscored consumers | 1 |
| credit scoring / equity | 1 |
| data-driven robust optimization | 1 |
| decision risk certificates | 1 |
| decision theory for conformal prediction | 1 |
| decision-aware conformal prediction | 1 |
| decision-focused learning | 1 |
| decision-focused learning survey | 1 |
| end-to-end conditional robust optimization | 1 |
| end-to-end conformal calibration for optimization | 1 |
| end-to-end conformal risk training | 1 |
| fair-lending proxy methodology | 1 |
| fintech lending / LendingClub | 1 |
| group-weighted conformal prediction | 1 |
| human decision support with CP | 1 |
| inverse conformal risk control | 1 |
| label noise / conformal robustness | 1 |
| localized conformal prediction | 1 |
| multi-distribution robust conformal prediction | 1 |
| multi-source conformal inference | 1 |
| non-monotonic conformal risk control | 1 |
| online conformal prediction | 1 |
| ordinal credit scoring / conformal prediction | 1 |
| predict-then-calibrate | 1 |
| prescriptive analytics | 1 |
| risk control / post-hoc testing | 1 |
| risk-controlling prediction sets | 1 |
| robust DFL | 1 |
| robust optimization | 1 |
| robust optimization foundations | 1 |
| weighted conformal / covariate shift | 1 |

## Lectura integrada para Paper CRPTO

El cuerpo de Paper CRPTO debe quedarse en cuatro pilares: conformal risk/control, robust optimization, conformal robust optimization / predict-then-calibrate y contexto Lending Club/DFL como comparador. Las fuentes promovidas al cuerpo son:

| relative_path | bib_key | core_concepts | crpto_value | limitations |
| --- | --- | --- | --- | --- |
| paper/Angelopoulos Bates 2023 - Gentle Introduction to Conformal Prediction.pdf | angelopoulos2023 | Exchangeability, split conformal, prediction sets, finite-sample marginal coverage. | Core theory support for the uncertainty layer. | Introductory; not a portfolio optimizer or credit-risk paper. |
| paper/Angelopoulos et al 2024 - Conformal Risk Control.pdf | angelopoulos2024risk | Risk control, monotone bounded losses, post-hoc calibration, user-defined loss. | Core theory support for mapping uncertainty to a decision-relevant loss. | Does not itself define a portfolio-funded-set bound. |
| paper/Angelopoulos et al 2024 - Theoretical Foundations of Conformal Prediction.pdf | angelopoulos2026_foundations | Exchangeability, exact finite-sample coverage, randomized quantiles, modern CP theory. | Canonical theory reference for the proof and notation. | Too broad for the paper body; cite selectively. |
| paper/Angelopoulos et al 2025 - Learn Then Test.pdf | angelopoulos2025ltt | Learn-then-test, finite-sample risk guarantees, post-hoc risk calibration. | Supports exact risk-gate language and post-selection caveats. | Does not solve downstream optimization by itself. |
| paper/Barber et al 2021 - Limits of Distribution-Free Conditional Predictive Inference.pdf | barber2021_limits_conditional | Conditional coverage impossibility, approximate conditional validity, subgroup limits. | Keeps Mondrian coverage language honest. | Boundary paper, not an implementation. |
| paper/Bates et al 2021 - Distribution-Free Risk-Controlling Prediction Sets.pdf | bates2021rcps | RCPS, set-valued prediction, expected loss control, black-box models. | Core CRPTO theory lineage. | General prediction sets; no credit portfolio objective. |
| paper/Bertsimas Sim 2004 - The Price of Robustness.pdf | bertsimas2004 | Budgeted uncertainty, protection level, tractable robust LP/IP, price of robustness. | Core RO foundation. | Budget is chosen, not conformally calibrated. |
| paper/Jagtiani Lemieux 2019 - Alternative Data and Machine Learning in Fintech Lending.pdf | jagtiani2019 | LendingClub, alternative data, credit grades, FICO relationship, fintech underwriting. | Core empirical-setting support. | Does not validate CRPTO or fairness claims. |
| paper/Johnstone Cox 2021 - Conformal Uncertainty Sets for Robust Optimization.pdf | johnstone2021 | Conformal regions as uncertainty sets, robust optimization, finite-sample validity. | Core method foundation. | No credit-specific funded-set weighted bound. |
| paper/Patel et al 2024 - Conformal Contextual Robust Optimization.pdf | patel2024 | Contextual robust optimization, conformal regions, generative uncertainty. | Direct neighbor in related work. | Not credit-specific and no funded-set PD constraint. |
| paper/Sun et al 2024 - Predict-then-Calibrate.pdf | sun2024ptc | Predict-then-calibrate, robust contextual LP, box/ellipsoid uncertainty sets. | Core closest-neighbor contrast. | No Lending Club and no Mondrian funded-set bound. |

Las fuentes de appendix o comparador CRPTO deben apoyar selectivamente el selector, SPO+/DFL, CVaR/OCE, CQR y límites de claim:

| relative_path | bib_key | decision | crpto_value | figures_tables_useful |
| --- | --- | --- | --- | --- |
| paper/Chi Ding Peng 2019 - Data-Driven Robust Credit Portfolio Optimization in P2P Lending.pdf | chi2019_p2p_robust_credit | append_crpto_related_work | Important applied-credit related work. | Use experimental design/tables as comparator template for Lending Club portfolio papers. |
| paper/Donti et al 2017 - Task-Based End-to-End Model Learning.pdf | donti2017 | append_comparator | Comparator lineage only. | Use as historical DFL lineage, not as CRPTO method figure. |
| paper/Elmachtoub Grigas 2022 - Smart Predict Then Optimize.pdf | elmachtoub2022 | append_comparator | Core comparator, not replacement. | Use benchmark/regret tables as comparator template. |
| supplement/Bao et al 2025 - CROMS Optimal Model Selection for Conformalized Robust Optimization.pdf | bao2025croms | append_crpto_selector | Appendix selector support. | Use selection diagrams/tables to structure A5/A10 appendix. |
| supplement/Ben-Tal Teboulle 2007 - Optimized Certainty Equivalent.pdf | bental_teboulle2007_oce | append_tail_risk | Appendix tail-risk foundation. | Use definitions, not figures. |
| supplement/Mandi et al 2024 - Decision-Focused Learning Survey.pdf | mandi2024 | append_comparator | Comparator framing. | Use taxonomy table for Paper 4 DFL suite framing. |
| supplement/Rockafellar Uryasev 2000 - Optimization of Conditional Value-at-Risk.pdf | rockafellar2000_cvar | append_tail_risk | Appendix tail-risk diagnostic. | Use formulation, not figures. |

## Lectura integrada para Paper 4

Paper 4 es el destino correcto para fuentes que fortalecen governance, source/shift robustness, fairness proxy, IFRS9/SICR proxy, DFL ampliado y data-quality/equity. Estas fuentes no reabren el champion CRPTO:

| relative_path | decision | paper4_value | evidence_gate | stop_rule |
| --- | --- | --- | --- | --- |
| paper/Albanesi Vamossy 2024 - Credit Scores Performance and Equity.pdf | append_paper4_governance | Strong appendix support for score-proxy vs champion ML and equity boundary. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| paper/Torkian Bamdad Sarfaraz 2025 - AI OR Investment Decisions in Digital Lending.pdf | append_paper4_context | Paper 4 digital-lending architecture context. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Barber Candes Ramdas Tibshirani 2023 - Conformal Prediction Beyond Exchangeability.pdf | append_paper4_source_shift | Strong source/shift conformal lane reference. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction.pdf | append_mixed_diagnostic_source_not_better | Potential source/shift experiment over grade/period/source families. | Executed on frozen v4 replay with 2018 calibration and 2019-2020 holdout; absolute gate passed, but worst-source coverage fell 0.0111 below Mondrian. | Append as source/shift diagnostic only; do not claim group-weighted source improvement over Mondrian. |
| supplement/CFPB 2014 - Public Information Proxy Race Ethnicity.pdf | append_paper4_governance | Strong claim-boundary source for proxy governance. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Guan 2023 - Localized Conformal Prediction.pdf | append_mixed_diagnostic_wider_than_mondrian | Potential source/shift experiment. | Executed on frozen v4 replay with 2018 calibration and 2019-2020 holdout; coverage and worst-source coverage improved, but width increased 0.0867 versus Mondrian. | Append only as reviewer-facing localized diagnostic; no localized guarantee or champion replacement claim. |
| supplement/Liu Levis Normand Han 2024 - Multi-Source Conformal Inference Under Distribution Shift.pdf | append_paper4_source_shift | Source-family holdout lane. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Tibshirani et al 2019 - Conformal Prediction Under Covariate Shift.pdf | append_paper4_source_shift | Source/shift conformal lane. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| tesis/Basel Committee 2015 - Guidance on Credit Risk and Expected Credit Losses.pdf | append_paper4_ifrs9 | Strong IFRS9/SICR proxy boundary source. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| tesis/Blattner Nelson 2021 - How Costly is Noise Data and Disparities in Consumer Credit.pdf | append_paper4_governance | Strong metric/data governance context. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |
| tesis/Fuster et al 2022 - Predictably Unequal Effects of Machine Learning on Credit Markets.pdf | append_paper4_governance | Strong governance/fairness boundary source. | No run: literature integration only. | Do not reopen a lane unless it changes a manuscript claim. |

## Experimentos evidence-gated

Los tres candidatos marcados por la matriz fueron ejecutados después de la auditoría inicial, con split temporal fijo sobre el replay congelado v4: 2018 como calibración y 2019--2020 como holdout. El sink principal queda en `reports/paper_material/paper4/tables/paper4_papers_tesis_conformal_candidate_gate_register_2026-06-06.csv`. La lectura es diagnóstica: Paper 4 gana appendix source/shift/localized, pero no hay cambio de champion ni claim de despliegue online.

| relative_path | implementation_or_experiment | evidence_gate | artifact_sink | stop_rule |
| --- | --- | --- | --- | --- |
| supplement/Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction.pdf | Group-weighted source-max replay: coverage 0.9387, avg width 0.9394, worst defended source 0.8602. | Absolute gate passed, but worst-source coverage fell 0.0111 below Mondrian. | `paper4_papers_tesis_conformal_candidate_gate_register_2026-06-06.csv` | Append as source/shift diagnostic only; do not claim group-weighted source improvement over Mondrian. |
| supplement/Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction.pdf | Utility-directed loss replay: coverage 0.9954, avg width +0.2205 versus Mondrian, worst defended source 0.9499. | Width gate failed: avg width 0.9981 exceeds 0.98 and does not beat Mondrian. | `paper4_papers_tesis_conformal_candidate_summary_2026-06-06.csv` | Park as negative width result; no utility-directed selector claim. |
| supplement/Guan 2023 - Localized Conformal Prediction.pdf | Localized score-bin replay: coverage 0.9497, avg width 0.8643, worst defended source 0.9252. | Coverage and worst-source coverage improved, but width increased 0.0867 versus Mondrian. | `paper4_papers_tesis_conformal_candidate_by_source_2026-06-06.csv` | Append only as reviewer-facing localized diagnostic; no localized guarantee or champion replacement claim. |

## Curaduría de figuras/tablas

El índice de captions no autoriza reproducir figuras ajenas ni convierte resultados externos en evidencia del proyecto. La curaduría siguiente solo define qué visuales pueden inspirar tablas, esquemas propios, appendices o respuestas a reviewers.

| relative_path | caption_type | caption_index | editorial_sink | why_useful | claim_boundary |
| --- | --- | --- | --- | --- | --- |
| paper/Hu et al 2026 - Conformal Robustness Control.pdf | figure | 1 | Paper 4 robustness-certificate appendix | Shows the conceptual contrast between conventional CRO and CRC-style robustness control. | Future gate over V/violation, not current CRPTO evidence. |
| supplement/Barber Candes Ramdas Tibshirani 2023 - Conformal Prediction Beyond Exchangeability.pdf | figure | 2 | Paper 4 source/shift conformal appendix | Coverage-width caption is useful for explaining why non-exchangeability gates must report both validity and efficiency. | Requires declared weighting or shift structure. |
| supplement/Tibshirani et al 2019 - Conformal Prediction Under Covariate Shift.pdf | figure | 1 | Paper 4 source/shift conformal appendix | Canonical weighted-conformal covariate-shift coverage caption for density-ratio caveats. | No source-shift deployment claim without credible weights. |
| supplement/Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction.pdf | figure | 3 | Paper 4 group/source governance appendix | Compares older weighted CP guarantees to the new group-weighted guarantee. | Needs target/source group weights before any run. |
| supplement/Liu Levis Normand Han 2024 - Multi-Source Conformal Inference Under Distribution Shift.pdf | figure | 1 | Paper 4 multi-source conformal appendix | Illustrates multi-source calibration structure and source-combination logic. | LC source families are retrospective, not validated external sources. |
| supplement/Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction.pdf | figure | 1 | Paper 4 utility-directed CP future gate | Contrasts standard CP with utility-directed CP in a decision-aware frame. | No selector change without fixed loss and coverage gate. |
| supplement/Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction.pdf | figure | 4 | Paper 4 utility-directed CP future gate | Connects base model accuracy to downstream optimization value. | Not evidence that CRPTO currently optimizes utility-directed sets. |
| supplement/Guan 2023 - Localized Conformal Prediction.pdf | figure | 1 | Paper 4 localized conformal candidate appendix | Visualizes global versus localized conformal bands. | No current localized calibration design. |
| paper/Chi Ding Peng 2019 - Data-Driven Robust Credit Portfolio Optimization in P2P Lending.pdf | figure | 2 | CRPTO/Paper 4 applied robust-credit related work | Performance-comparison caption anchors the P2P robust portfolio context. | Different data, objective and uncertainty construction. |
| paper/Albanesi Vamossy 2024 - Credit Scores Performance and Equity.pdf | figure | 4 | Paper 4 score/equity governance appendix | Gini-over-time caption is a useful template for score-vs-model governance reporting. | Not a legal fair-lending claim. |
| tesis/Fuster et al 2022 - Predictably Unequal Effects of Machine Learning on Credit Markets.pdf | figure | 1 | Paper 4 equity/noise governance appendix | Clarifies how better prediction technology can create group-specific effects. | No protected-attribute causal claim in LC. |
| tesis/Blattner Nelson 2021 - How Costly is Noise Data and Disparities in Consumer Credit.pdf | table | 4 | Paper 4 data-quality/equity governance appendix | Links predictive-performance gaps to disadvantaged consumers and data quality. | Project lacks the paper's protected-group and lender data. |
| supplement/CFPB 2014 - Public Information Proxy Race Ethnicity.pdf | table | 5 | Paper 4 fairness-proxy boundary appendix | Shows proxy-probability validation needs richer race/ethnicity methodology. | LC lacks surname/fine geography and protected labels. |
| tesis/Cresswell et al 2024 - Conformal Prediction Sets Improve Human Decision Making.pdf | figure | 2 | Paper 4 governance/committee communication appendix | Human-facing uncertainty display can inspire committee explanation design. | Different task, subjects and utility function. |

## Future work y stop rules

| relative_path | decision | crpto_value | paper4_value | stop_rule |
| --- | --- | --- | --- | --- |
| paper/Hu et al 2026 - Conformal Robustness Control.pdf | park_future_work | Future-work frontier; supports not overclaiming current method. | Candidate future gate for robustness certificates. | Park if it cannot be evaluated on frozen CRPTO artifacts without retraining a new method. |
| paper/Zhao et al 2026 - Conformal Robust Optimization and Satisficing.pdf | append_future_work | Related-work frontier. | Satisficing/governance candidate. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Angelopoulos et al 2026 - Conformal Risk Control for Non-Monotonic Losses.pdf | park_future_work | Future-work only. | Strong future gate for composite risk. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction.pdf | append_mixed_diagnostic_source_not_better | Future robustness/context only. | Potential source/shift experiment over grade/period/source families. | Append as source/shift diagnostic only; do not claim group-weighted source improvement over Mondrian. |
| supplement/Chenreddy Delage 2024 - End-to-End Conditional Robust Optimization.pdf | park_future_work | Future-work contrast. | End-to-end lane context. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction.pdf | park_width_fail | Future-work extension; supports A5 selector direction. | Candidate decision-loss gate or conformal utility experiment. | Park as negative width result; no utility-directed selector claim. |
| supplement/Gibbs Candes 2021 - Adaptive Conformal Inference Under Distribution Shift.pdf | park_future_work | Future-work caveat. | Online/source conformal lane context. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Guan 2023 - Localized Conformal Prediction.pdf | append_mixed_diagnostic_wider_than_mondrian | Future-work comparator. | Potential source/shift experiment. | Append only as reviewer-facing localized diagnostic; no localized guarantee or champion replacement claim. |
| supplement/Kiyani et al 2025 - Decision Theoretic Foundations for Conformal Prediction.pdf | append_future_work | Future/positioning support. | Decision-risk lab framing. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Lin Delage Chan 2024 - Conformal Inverse Optimization.pdf | park_future_work | Future-work context. | Decision audit/source governance context. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Liu et al 2026 - Online Conformal Prediction via Universal Portfolio Algorithms.pdf | park_future_work | Future-work only. | Online lane context and stop-rule support. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Schutte et al 2024 - Robust Losses for Decision-Focused Learning.pdf | park_future_dfl | Comparator/future work. | Potential DFL suite extension. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Yang Jin 2026 - Multi-Distribution Robust Conformal Prediction.pdf | append_future_work | Future-work context. | Source-governance lane and caveat. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Yeh et al 2025 - Conformal Risk Training.pdf | park_future_work | Future-work contrast. | Candidate composite decision-risk training lane. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Yeh et al 2026 - End-to-End Conformal Calibration for Optimization Under Uncertainty.pdf | park_future_work | Future-work frontier. | Lane 3/end-to-end context. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Zhou Orfanoudaki Zhu 2025 - Conformalized Decision Risk Assessment.pdf | park_future_work | Future auditability contrast. | Strong future lane for decision certificate rather than policy promotion. | Do not reopen a lane unless it changes a manuscript claim. |
| supplement/Zhou Zhu 2025 - Calibrating Decision Robustness via Inverse Conformal Risk Control.pdf | park_future_work | Future-work context. | Decision robustness gate candidate. | Do not reopen a lane unless it changes a manuscript claim. |
| tesis/Kawasumi Kato Duan 2026 - Conformal Prediction for Ordinal Credit Scoring.pdf | park_future_work | Future credit conformal context. | Ordinal score/rating future lane. | Do not reopen a lane unless it changes a manuscript claim. |

## Matriz paper-by-paper

La tabla siguiente es deliberadamente densa. Cada fila resume concepto, claim, método/evidencia, conclusión, figuras/tablas útiles, limitación y destino editorial. Para auditoría operativa usar el CSV completo.

| relative_path | title | status | core_concepts | key_claims | conclusions | figures_tables_useful | limitations | decision | action_required |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper/Albanesi Vamossy 2024 - Credit Scores Performance and Equity.pdf | Credit Scores: Performance and Equity | NBER working paper | Score performance, equity, misclassification, rank disagreement, observable vulnerable groups. | Better scores can improve accuracy while also changing error allocation across groups. | Useful for metric governance: report discrimination, calibration, utility and equity together. | Use tables on score performance/equity as template for FICO proxy vs champion diagnostics. | Not Lending Club CRPTO evidence and not a legal fair-lending protocol. | append_paper4_governance | completed_quarto_patch |
| paper/Angelopoulos Bates 2023 - Gentle Introduction to Conformal Prediction.pdf | Conformal Prediction: A Gentle Introduction | Foundations and Trends in Machine Learning | Exchangeability, split conformal, prediction sets, finite-sample marginal coverage. | Distribution-free uncertainty can be wrapped around black-box predictors. | Best pedagogical citation for explaining why CRPTO's intervals are valid before optimization. | Use tutorial figures only conceptually; the paper needs its own CRPTO diagram. | Introductory; not a portfolio optimizer or credit-risk paper. | promote_crpto_body | none_now |
| paper/Angelopoulos et al 2024 - Conformal Risk Control.pdf | Conformal Risk Control | ICLR 2024 | Risk control, monotone bounded losses, post-hoc calibration, user-defined loss. | Conformal calibration can control expected loss beyond ordinary set coverage. | Direct ancestor of CRPTO's funded-set weighted miscoverage framing. | Use conceptual loss-control diagrams as related-work inspiration. | Does not itself define a portfolio-funded-set bound. | promote_crpto_body | none_now |
| paper/Angelopoulos et al 2024 - Theoretical Foundations of Conformal Prediction.pdf | Theoretical Foundations of Conformal Prediction | Cambridge University Press pre-publication / monograph | Exchangeability, exact finite-sample coverage, randomized quantiles, modern CP theory. | Conformal validity is a theorem about a procedure and assumptions, not a model-quality claim. | Use as the canonical proof reference for split conformal language and limitations. | Use definitions/proof structure, not figures, in CRPTO. | Too broad for the paper body; cite selectively. | promote_crpto_body | none_now |
| paper/Angelopoulos et al 2025 - Learn Then Test.pdf | Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control | Annals of Applied Statistics | Learn-then-test, finite-sample risk guarantees, post-hoc risk calibration. | Predictive algorithms can be calibrated after learning to satisfy risk constraints. | Supports predeclared gates and reviewer-safe validation language. | Use algorithmic workflow conceptually for gate diagrams. | Does not solve downstream optimization by itself. | promote_crpto_body | none_now |
| paper/Barber et al 2021 - Limits of Distribution-Free Conditional Predictive Inference.pdf | The Limits of Distribution-Free Conditional Predictive Inference | Information and Inference | Conditional coverage impossibility, approximate conditional validity, subgroup limits. | Exact distribution-free conditional coverage is impossible without assumptions or restrictions. | This is a claim-boundary source: it prevents overclaiming subgroup guarantees. | Use theorem statements, not figures. | Boundary paper, not an implementation. | promote_crpto_body | none_now |
| paper/Bates et al 2021 - Distribution-Free Risk-Controlling Prediction Sets.pdf | Distribution-Free, Risk-Controlling Prediction Sets | arXiv / risk-control foundation | RCPS, set-valued prediction, expected loss control, black-box models. | Prediction sets can be calibrated to control a user-defined risk distribution-free. | Provides the bridge from coverage to risk-control vocabulary. | Use algorithmic schematic as conceptual support only. | General prediction sets; no credit portfolio objective. | promote_crpto_body | none_now |
| paper/Bertsimas Gupta Kallus 2018 - Data-Driven Robust Optimization.pdf | Data-Driven Robust Optimization | Mathematical Programming | Uncertainty sets learned from data, robustness guarantees, finite-sample feasibility. | Robust sets can be constructed data-driven rather than manually specified. | Supports the historical progression from ad-hoc RO to calibrated/data-driven uncertainty sets. | Useful as related-work table contrasting source of uncertainty sets. | Not conformal and not Lending Club-specific. | context_crpto_related_work | none_now |
| paper/Bertsimas Kallus 2020 - From Predictive to Prescriptive Analytics.pdf | From Predictive to Prescriptive Analytics | Management Science | Prescriptive analytics, decision quality, predictive-to-prescriptive bridge. | Prediction alone is insufficient; decision policies should be evaluated by downstream outcomes. | Useful framing for why CRPTO evaluates funded-set outcomes, not AUC only. | Related-work schematic only. | Not conformal and not a risk-control guarantee. | context_crpto_related_work | none_now |
| paper/Bertsimas Sim 2004 - The Price of Robustness.pdf | The Price of Robustness | Operations Research | Budgeted uncertainty, protection level, tractable robust LP/IP, price of robustness. | Robustness trades objective value for protection in a tunable, tractable way. | Use as the language ancestor for alpha -> Gamma_CP -> return trade-off. | Use price-of-robustness framing in CRPTO figures. | Budget is chosen, not conformally calibrated. | promote_crpto_body | none_now |
| paper/Chi Ding Peng 2019 - Data-Driven Robust Credit Portfolio Optimization in P2P Lending.pdf | Data-Driven Robust Credit Portfolio Optimization for Investment Decisions in P2P Lending | Applied Soft Computing / Elsevier article | P2P lending, robust credit portfolio, data-driven scoring, risk-return allocation. | Robust optimization can improve P2P investment decisions under model uncertainty. | Closest credit-domain precedent for robust P2P allocation. | Use experimental design/tables as comparator template for Lending Club portfolio papers. | No conformal guarantees and no funded-set miscoverage bound. | append_crpto_related_work | completed_bib_quarto_patch |
| paper/Donti et al 2017 - Task-Based End-to-End Model Learning.pdf | Task-based End-to-end Model Learning in Stochastic Optimization | NeurIPS 2017 | End-to-end learning, differentiable optimization, downstream task loss. | Training against task loss can outperform prediction-error training for decisions. | DFL lineage for SPO+/PyEPO comparator. | Use as historical DFL lineage, not as CRPTO method figure. | No conformal coverage or credit-specific auditability. | append_comparator | none_now |
| paper/Elmachtoub Grigas 2022 - Smart Predict Then Optimize.pdf | Smart Predict, then Optimize | Management Science | SPO loss, SPO+ surrogate, regret-oriented training, downstream decision quality. | Models can be trained to reduce decision regret rather than prediction error. | Primary DFL comparator: SPO+ may reduce regret but does not provide conformal auditability. | Use benchmark/regret tables as comparator template. | No coverage guarantee and no uncertainty-set audit trail. | append_comparator | none_now |
| paper/Guo et al 2016 - Instance-Based Credit Risk Assessment for Investment Decisions in P2P Lending.pdf | Instance-based Credit Risk Assessment for Investment Decisions in P2P Lending | European Journal of Operational Research | Instance-based learning, P2P loan investment, risk-return decision support. | Loan-level decision support can improve P2P investment performance. | Useful historical Lending Club/P2P context. | Portfolio decision tables can inspire thesis background. | No conformal or robust bound. | context_credit_domain | none_now |
| paper/Hu et al 2026 - Conformal Robustness Control.pdf | Conformal Robustness Control | ICLR 2026 | Robustness control, conformal calibration, robust decision criteria. | Conformal machinery can certify robustness-oriented decision properties. | Confirms CRPTO is timely, but should remain future-work unless reproduced. | Use as frontier map figure/table in related work, not main evidence. | Not yet implemented in the project; no Lending Club funded-set result. | park_future_work | completed_bib_quarto_patch |
| paper/Jagtiani Lemieux 2019 - Alternative Data and Machine Learning in Fintech Lending.pdf | The Roles of Alternative Data and Machine Learning in Fintech Lending | Financial Management | LendingClub, alternative data, credit grades, FICO relationship, fintech underwriting. | Fintech grades and alternative data provide empirical credit information beyond traditional scores. | Best setting citation for why Lending Club is a meaningful credit-risk lab. | Use tables about grade/FICO/performance as context, not CRPTO evidence. | Does not validate CRPTO or fairness claims. | promote_crpto_body | none_now |
| paper/Johnstone Cox 2021 - Conformal Uncertainty Sets for Robust Optimization.pdf | Conformal Uncertainty Sets for Robust Optimization | COPA / PMLR | Conformal regions as uncertainty sets, robust optimization, finite-sample validity. | Conformal prediction regions can feed robust optimization with validity. | Closest foundational CP -> RO bridge. | Use as conceptual predecessor for CRPTO pipeline figure. | No credit-specific funded-set weighted bound. | promote_crpto_body | none_now |
| paper/Patel et al 2024 - Conformal Contextual Robust Optimization.pdf | Conformal Contextual Robust Optimization | AISTATS 2024 | Contextual robust optimization, conformal regions, generative uncertainty. | Informative conformal uncertainty regions can improve robust contextual decisions. | Strong neighboring method but with different uncertainty geometry and domains. | Use related-work contrast table. | Not credit-specific and no funded-set PD constraint. | promote_crpto_body | none_now |
| paper/Sun et al 2024 - Predict-then-Calibrate.pdf | Predict-then-Calibrate: A New Perspective of Robust Contextual LP | arXiv / robust contextual LP | Predict-then-calibrate, robust contextual LP, box/ellipsoid uncertainty sets. | Post-hoc calibration can build valid uncertainty sets for robust contextual optimization. | Closest conceptual comparator to post-hoc CRPTO. | Use contrast table: PtC set membership vs CRPTO funded-set weighted risk. | No Lending Club and no Mondrian funded-set bound. | promote_crpto_body | none_now |
| paper/Torkian Bamdad Sarfaraz 2025 - AI OR Investment Decisions in Digital Lending.pdf | Integrating AI and OR for investment decision-making in emerging digital lending businesses | Journal of the Operational Research Society; online 2025 | Digital lending, multi-objective risk-return optimization, AI + OR pipeline. | Combining ML and OR improves investment recommendations in digital lending. | Useful applied competitor/context; not a conformal method. | Use risk-return frontier and experimental setup as Paper 4 comparator inspiration. | No conformal coverage, no CRPTO bound, no Lending Club champion evidence. | append_paper4_context | completed_bib_quarto_patch |
| paper/Zhao et al 2026 - Conformal Robust Optimization and Satisficing.pdf | Conformal Robust Optimization and Satisficing for Prescriptive Analytics with Black-Box Predictors | AISTATS workshop / SSRN working paper | Conformal robust optimization, satisficing, black-box predictors, parameter mapping. | CRO and conformal robust satisficing can certify robust prescriptive decisions. | Valuable frontier for satisficing margins and policy acceptance thresholds. | Use satisficing margin idea for appendix, not body. | Preprint/workshop and not credit-specific. | append_future_work | none_now |
| supplement/Angelopoulos et al 2026 - Conformal Risk Control for Non-Monotonic Losses.pdf | Conformal Risk Control for Non-Monotonic Losses | arXiv preprint | Non-monotonic loss, risk control, multi-objective loss families. | CRC ideas can extend beyond monotone losses under new machinery. | Useful for future compound loss: return + V + CVaR + source risk. | Use loss examples for future gate design. | Not implemented; would change current CRPTO claim. | park_future_work | none_now |
| supplement/Bao et al 2025 - CROMS Optimal Model Selection for Conformalized Robust Optimization.pdf | Optimal Model Selection for Conformalized Robust Optimization | arXiv preprint | Conformalized robust optimization, decision-aware model selection, downstream risk. | Selecting conformal models should account for robust decision performance. | Justifies CRPTO's CROMS-lite selector but not a full implementation claim. | Use selection diagrams/tables to structure A5/A10 appendix. | Project only has a selector screen over artifacts, not CROMS training. | append_crpto_selector | none_now |
| supplement/Barber Candes Ramdas Tibshirani 2023 - Conformal Prediction Beyond Exchangeability.pdf | Conformal Prediction Beyond Exchangeability | Annals of Statistics | Weighted exchangeability, distribution shift, non-exchangeable conformal validity. | Conformal validity can be extended with weights under structured departures from exchangeability. | Important boundary/future source for source/shift lanes. | Use assumptions table for source governance appendix. | Requires weight/shift structure not currently guaranteed. | append_paper4_source_shift | completed_bib_quarto_patch |
| supplement/Ben-Tal Teboulle 2007 - Optimized Certainty Equivalent.pdf | An Old-New Concept of Convex Risk Measures: The Optimized Certainty Equivalent | Mathematical Finance | Optimized certainty equivalent, convex risk measures, CVaR relation. | OCE provides a principled convex risk-measure family. | Grounds OCE/CVaR tail-risk appendix. | Use definitions, not figures. | Does not choose a credit champion by itself. | append_tail_risk | none_now |
| supplement/Bhattacharyya Barber 2026 - Group-Weighted Conformal Prediction.pdf | Group-Weighted Conformal Prediction | Electronic Journal of Statistics | Group-weighted CP, group-based shift, weighted conformal prediction. | When groups drive covariate shift, group-weighted calibration can improve guarantees. | Natural candidate for Paper 4 source/grade reweighting, but not current CRPTO. | Use group-weighting assumptions for source-governance table. | Needs target/source group weights and prospective design. | append_mixed_diagnostic_source_not_better | experiment_completed_appendix_diagnostic |
| supplement/CFPB 2014 - Public Information Proxy Race Ethnicity.pdf | Using Publicly Available Information to Proxy for Unidentified Race and Ethnicity | official methodology report | BISG-style proxy, surname/geography, proxy validation, protected-attribute limitation. | Race/ethnicity proxying requires richer inputs and careful validation. | Supports why Lending Club zip3/state is insufficient for legal fair-lending claims. | Use proxy-method workflow as governance appendix reference. | Project lacks surname and fine geography. | append_paper4_governance | none_now |
| supplement/Chenreddy Delage 2024 - End-to-End Conditional Robust Optimization.pdf | End-to-End Conditional Robust Optimization | UAI 2024 / PMLR | Conditional robust optimization, differentiable optimization, end-to-end training. | End-to-end conditional robust training can improve conditional coverage/objective tradeoffs. | Frontier contrast: CRPTO is post-hoc and auditable, not end-to-end. | Use as frontier comparison table. | Different stack and conditional guarantees; not current project method. | park_future_work | none_now |
| supplement/Cortes-Gomez et al 2025 - Utility-Directed Conformal Prediction.pdf | Utility-Directed Conformal Prediction | ICLR 2025 | Utility-directed CP, decision loss, actionable uncertainty, coverage preservation. | Prediction sets can incorporate downstream utility while retaining standard coverage. | Highly relevant future step: optimize CP usefulness without abandoning coverage. | Use framework diagram for future decision-loss conformal selector design. | Current CRPTO does not implement this training/calibration objective. | park_width_fail | experiment_completed_parked |
| supplement/FinRegLab 2023 - Explainability and Fairness in ML Credit Underwriting.pdf | Explainability and Fairness in Machine Learning for Credit Underwriting: Policy Analysis | policy report | Explainability, fairness, adverse action, model governance, underwriting controls. | ML credit systems require governance controls beyond predictive performance. | Supports MRM/fairness appendix and explains why claims must be bounded. | Use control-taxonomy tables for governance mapping. | Policy report, not CRPTO empirical evidence. | append_governance | none_now |
| supplement/Gibbs Candes 2021 - Adaptive Conformal Inference Under Distribution Shift.pdf | Adaptive Conformal Inference Under Distribution Shift | NeurIPS 2021 | ACI, online updating, distribution shift, adaptive quantiles. | Conformal thresholds can be adapted under shift to improve long-run coverage. | Future online/drift candidate, not current static CRPTO. | Use online update schematic for Paper 4 future work. | Needs streaming/prospective feedback to claim deployment validity. | park_future_work | none_now |
| supplement/Gibbs Cherian Candes 2025 - Conformal Prediction with Conditional Guarantees.pdf | Conformal Prediction with Conditional Guarantees | JRSS-B | Conditional guarantees, coverage spectrum, restricted conditional targets. | Relaxed conditional goals can be achieved where exact conditional coverage cannot. | Supports future conditional-tightening language and source-group caveats. | Use conceptual spectrum for appendix if conditional validity is discussed. | Not implemented in project; current Mondrian is finite-group conditional-ish, not exact individual conditional. | append_claim_boundary | none_now |
| supplement/Guan 2023 - Localized Conformal Prediction.pdf | Localized Conformal Prediction | Biometrika | Localized conformity scores, local calibration, approximate conditional validity. | Local weighting can improve adaptivity while retaining conformal inference structure. | Candidate comparator to Mondrian if future coverage sharpness matters. | Use local-vs-global schematic for appendix only. | Would require a new localized calibration design. | append_mixed_diagnostic_wider_than_mondrian | experiment_completed_appendix_diagnostic |
| supplement/Jonkers et al 2024 - Conformal Predictive Systems Under Covariate Shift.pdf | Conformal Predictive Systems Under Covariate Shift | arXiv / preprint | Conformal predictive systems, covariate shift, calibration under shift. | Conformal predictive systems can be adapted for shifted covariate distributions. | Useful for source/shift caveat, not current champion. | Use shift taxonomy only. | Requires shift assumptions and is not credit-specific. | context_source_shift | none_now |
| supplement/Kiyani et al 2025 - Decision Theoretic Foundations for Conformal Prediction.pdf | Decision Theoretic Foundations for Conformal Prediction | AISTATS / ICML-era preprint | Decision-theoretic CP, utility, actionability, uncertainty quantification. | Conformal sets can be understood through decisions and utilities, not only coverage. | Good conceptual support for CRPTO's decision-aware framing. | Use as theory bridge in future-work paragraph. | Does not implement CRPTO portfolio optimization. | append_future_work | none_now |
| supplement/Lekeufack et al 2023 - Conformal Decision Theory.pdf | Conformal Decision Theory | NeurIPS workshop / preprint | Conformal prediction for decisions, calibrated decision policies, set-valued actions. | Conformal uncertainty can be integrated with decision-making objectives. | Supports Paper 4 decision-risk framing but is not a CRPTO baseline. | Use only as conceptual context. | No credit portfolio evidence. | context_decision_conformal | none_now |
| supplement/Lin Delage Chan 2024 - Conformal Inverse Optimization.pdf | Conformal Inverse Optimization | NeurIPS 2024 | Inverse optimization, conformal uncertainty, decision ambiguity. | Conformal methods can quantify uncertainty in inverse optimization. | Valuable frontier for auditing human/legacy decisions, not current CRPTO. | Use CREDO/inverse-optimization family as future audit lane. | Different problem: inferring preferences/parameters from decisions. | park_future_work | none_now |
| supplement/Liu Levis Normand Han 2024 - Multi-Source Conformal Inference Under Distribution Shift.pdf | Multi-Source Conformal Inference Under Distribution Shift | arXiv preprint | Multi-source inference, heterogeneous source distributions, shifted test populations. | Combining calibration sources carefully can maintain valid intervals under source shift. | Directly relevant to Paper 4 source-family holdout but needs stronger data design. | Use source-combination taxonomy for source governance. | Current LC source groups are retrospective, not validated external sources. | append_paper4_source_shift | none_now |
| supplement/Liu et al 2026 - Online Conformal Prediction via Universal Portfolio Algorithms.pdf | Online Conformal Prediction via Universal Portfolio Algorithms | arXiv preprint | Online conformal prediction, universal portfolios, regret-to-coverage, parameter-free adaptation. | Online CP can achieve long-run coverage via universal portfolio algorithms. | Excellent future direction for deployment, but current project is retrospective. | Use online update process as future-work schematic. | No production feedback stream in current data. | park_future_work | none_now |
| supplement/Mandi et al 2024 - Decision-Focused Learning Survey.pdf | Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities | Journal of Artificial Intelligence Research | DFL taxonomy, SPO+, gradient-based/gradient-free methods, benchmarks. | DFL is mature but no single method dominates across tasks. | Use to defend treating SPO+/PyEPO as comparator, not CRPTO replacement. | Use taxonomy table for Paper 4 DFL suite framing. | Survey does not provide conformal guarantees. | append_comparator | none_now |
| supplement/Rockafellar Uryasev 2000 - Optimization of Conditional Value-at-Risk.pdf | Optimization of Conditional Value-at-Risk | The Journal of Risk | CVaR, tail risk, tractable convex optimization. | CVaR is optimizable and more useful than VaR for tail-risk control. | Grounds A12 and Paper 4 tail challenger. | Use formulation, not figures. | Tail-risk improvement does not imply wealth champion. | append_tail_risk | none_now |
| supplement/Romano Patterson Candes 2019 - Conformalized Quantile Regression.pdf | Conformalized Quantile Regression | NeurIPS 2019 | Conformalized quantile regression, adaptive intervals, split conformal. | Quantile regression plus conformalization yields valid adaptive intervals. | Baseline/variant reference for interval adaptivity. | Use interval-width/adaptivity examples in conformal chapter. | Current champion is Mondrian score-decile, not CQR. | append_conformal_variant | none_now |
| supplement/Schutte et al 2024 - Robust Losses for Decision-Focused Learning.pdf | Robust Losses for Decision-Focused Learning | NeurIPS / arXiv-era DFL paper | Robust decision-focused losses, decision regret, misspecification. | Robust losses can improve DFL under uncertainty or misspecification. | Paper 4 DFL challenger context; not a CRPTO body claim. | Use loss comparison table for DFL appendix only. | No conformal auditability or credit-specific bound. | park_future_dfl | none_now |
| supplement/Tibshirani et al 2019 - Conformal Prediction Under Covariate Shift.pdf | Conformal Prediction Under Covariate Shift | NeurIPS 2019 | Covariate shift, weighted conformal prediction, likelihood-ratio weights. | Conformal coverage can adapt under covariate shift if weights are known/estimated. | Core source for future source/shift gates. | Use assumption table for source-shift appendix. | Requires credible density-ratio/weight estimation. | append_paper4_source_shift | none_now |
| supplement/Yang Jin 2026 - Multi-Distribution Robust Conformal Prediction.pdf | Multi-Distribution Robust Conformal Prediction | arXiv preprint | Multiple source distributions, robust coverage, max-p aggregation. | Finite-sample coverage can be made robust over multiple distributions/mixtures. | Strong source-robustness future-work citation. | Use source distribution diagram if Paper 4 source appendix expands. | Current project has retrospective source proxies, not true multi-source deployment. | append_future_work | none_now |
| supplement/Yeh et al 2025 - Conformal Risk Training.pdf | Conformal Risk Training | NeurIPS 2025 | Conformal risk training, OCE/CVaR, differentiable conformal risk. | Conformal risk can be optimized end-to-end, including OCE-style risks. | Future direction for replacing diagnostic tail risk with optimized conformal risk. | Use risk-training diagram as future-work context only. | Current CRPTO is post-hoc; implementing CRT would change method. | park_future_work | none_now |
| supplement/Yeh et al 2026 - End-to-End Conformal Calibration for Optimization Under Uncertainty.pdf | End-to-End Conformal Calibration for Optimization Under Uncertainty | TMLR; arXiv v2 in 2026 | End-to-end conformal calibration, downstream optimization, learned uncertainty sets. | Calibration can be optimized for decision usefulness while retaining validity. | Confirms future direction but not current post-hoc CRPTO. | Use related-work contrast table only. | Different training stack and not credit-specific. | park_future_work | none_now |
| supplement/Zhou Orfanoudaki Zhu 2025 - Conformalized Decision Risk Assessment.pdf | Conformalized Decision Risk Assessment | ICLR 2026 / arXiv 2025 | CREDO, decision risk certificates, inverse optimization, conformalized risk estimation. | A candidate decision can receive a distribution-free upper bound on probability of suboptimality. | Very relevant for Paper 4 auditability; not part of current CRPTO champion. | Use risk-certificate diagram as Paper 4 future audit inspiration. | Different problem geometry; no Lending Club implementation. | park_future_work | completed_bib_quarto_patch |
| supplement/Zhou Zhu 2025 - Calibrating Decision Robustness via Inverse Conformal Risk Control.pdf | Calibrating Decision Robustness via Inverse Conformal Risk Control | arXiv preprint | Inverse CRC, decision robustness, robustness calibration. | Decision robustness can be calibrated through inverse conformal risk-control ideas. | Relevant future lane for robustness calibration. | Use only as future-work contrast. | Not implemented and not credit-specific. | park_future_work | none_now |
| tesis/Babaei Bamdad 2020 - Multi-Objective Investment Recommendation in P2P Lending.pdf | A multi-objective instance-based decision support system for investment recommendation in peer-to-peer lending | Journal article | P2P lending, multi-objective recommendation, risk-return tradeoff, NPV. | Decision-support systems can optimize investment recommendations over risk and return. | Useful thesis context for digital lending optimization. | Use risk-return recommendation setup as background only. | No conformal guarantees. | context_credit_domain | none_now |
| tesis/Basel Committee 2015 - Guidance on Credit Risk and Expected Credit Losses.pdf | Guidance on credit risk and accounting for expected credit losses | official supervisory guidance | Expected credit losses, credit risk governance, forward-looking information, controls. | ECL estimation needs governance, data quality, forward-looking information and controls. | Supports IFRS9-inspired boundary and why contractual IFRS9 is not claimed. | Use guidance checklist for Paper 4 IFRS9 proxy appendix. | Open Lending Club data lacks contractual monthly DPD/EAD/recovery infrastructure. | append_paper4_ifrs9 | none_now |
| tesis/Ben-Tal El Ghaoui Nemirovski 2009 - Robust Optimization.pdf | Robust Optimization | Princeton University Press book | Uncertainty sets, robust counterparts, tractability, convex robust optimization. | Robust optimization provides a systematic language for feasible decisions under uncertainty. | Use as deep thesis foundation; CRPTO body can rely on shorter RO citations. | Use formulations/definitions, not figures. | Very broad; not conformal or credit-specific. | thesis_foundation | none_now |
| tesis/Blattner Nelson 2021 - How Costly is Noise Data and Disparities in Consumer Credit.pdf | How Costly is Noise? Data and Disparities in Consumer Credit | working paper / credit economics | Data noise, credit scores, disparities, unequal information quality. | Noisy credit data can create unequal access and distort credit allocations. | Important governance caveat: model performance can be constrained by source data quality. | Use disparity/noise tables as governance appendix inspiration. | Not Lending Club CRPTO evidence. | append_paper4_governance | none_now |
| tesis/Brevoort Grimm Kambara 2016 - Credit Invisibles and the Unscored.pdf | Credit Invisibles and the Unscored | Cityscape / CFPB research | Credit invisibles, unscored consumers, data availability, inclusion. | Large consumer segments lack conventional scores or have weak scoring coverage. | Supports why open accepted-loan data cannot make broad credit-access claims. | Use population breakdowns as motivation only. | Not a CRPTO method source. | context_governance | none_now |
| tesis/Cresswell et al 2024 - Conformal Prediction Sets Improve Human Decision Making.pdf | Conformal Prediction Sets Improve Human Decision Making | preprint / empirical decision support | Human-AI decision making, prediction sets, uncertainty communication. | Conformal sets can improve human decisions when uncertainty is communicated well. | Useful for discussing auditability and committee-facing uncertainty. | Use decision-support figures as inspiration for MRM/committee communication. | Not credit portfolio optimization. | context_decision_support | none_now |
| tesis/Einbinder et al 2024 - Label Noise Robustness of Conformal Prediction.pdf | Label Noise Robustness of Conformal Prediction | Journal of Machine Learning Research | Label noise, conformal robustness, noisy outcomes, validity under imperfect labels. | Conformal prediction has robustness properties under forms of label noise. | Useful threat-to-validity note for default labels and hardened shift tests. | Use noise taxonomy for appendix caveat. | Does not validate all Lending Club label issues. | append_claim_boundary | none_now |
| tesis/Fuster et al 2022 - Predictably Unequal Effects of Machine Learning on Credit Markets.pdf | Predictably Unequal? The Effects of Machine Learning on Credit Markets | Journal of Finance | ML credit screening, distributional impacts, mortgage markets, disparities. | More flexible ML can change rate disparities and who benefits from credit-market technology. | Critical fairness/equity context: better predictive technology does not automatically imply equitable outcomes. | Use disparity mechanism figures/tables in governance discussion. | Mortgage market, not Lending Club; no protected attributes in current project. | append_paper4_governance | completed_bib_quarto_patch |
| tesis/Kato 2024 - Conformal Predictive Portfolio Selection.pdf | Conformal Predictive Portfolio Selection | arXiv preprint | Portfolio selection, conformal predictive sets, financial decisions. | Conformal prediction can support portfolio selection under uncertainty. | Relevant neighbor in finance; CRPTO differs by credit PD/funded-set risk. | Use as finance-related-work table. | Not credit lending and not Lending Club. | context_finance_cp | none_now |
| tesis/Kawasumi Kato Duan 2026 - Conformal Prediction for Ordinal Credit Scoring.pdf | Conformal Prediction for Ordinal Credit Scoring | arXiv preprint | Ordinal credit scoring, conformal prediction, rating categories. | Conformal methods can be adapted to ordinal credit-score outputs. | Useful for future grade/rating-set extension, not current PD interval CRPTO. | Use only as future-work note. | No portfolio optimization and very short current evidence. | park_future_work | none_now |
| tesis/Khandani Kim Lo 2010 - Consumer Credit-Risk Models via Machine-Learning Algorithms.pdf | Consumer Credit-Risk Models via Machine-Learning Algorithms | Journal of Banking & Finance | Consumer credit risk, nonlinear ML, transaction data, economic value of risk forecasts. | ML forecasts can materially improve consumer credit-risk management and economic decisions. | Classic support for economic evaluation of credit ML, not just classification metrics. | Use economic-benefit tables as background for decision-value framing. | Credit cards/bank data, not LC/CRPTO. | context_credit_domain | completed_bib_quarto_patch |
| tesis/Noguer i Alonso 2024 - Conformal Portfolio Optimization.pdf | Conformal Portfolio Optimization | SSRN preprint | Conformal prediction, portfolio optimization, finance uncertainty. | Conformal intervals can support portfolio decisions in financial assets. | Finance-adjacent context, but CRPTO's credit/funded-set bound is different. | Use only as related-work mention. | Short, non-credit, not a robust credit allocation benchmark. | context_finance_cp | none_now |

## Control bibliográfico

Regla aplicada: `book/references.bib` se modifica solo cuando una fuente queda citada o se prepara explícitamente para texto Quarto de Paper CRPTO/Paper 4. Las fuentes `needs_bib_if_cited` permanecen en la matriz sin inflar la bibliografía.

| bib_status | n |
| --- | --- |
| added_2026_06_06 | 10 |
| existing | 36 |
| needs_bib_if_cited | 15 |

## Fronteras que permanecen falsas

- CRPTO no reclama legal fair lending con atributos protegidos directos.
- CRPTO no implementa IFRS9 contractual.
- Paper 4 no reclama CATE policy value.
- Paper 4 no reclama online deployment.
- Paper 4 no reclama Bellman/DLA exacto.
- SPO+/DFL puede ganar regret, pero no reemplaza la garantía/auditabilidad CRPTO.

## Cierre

La auditoría agrega valor como integración bibliográfica y de claims. No crea un nuevo champion, no exige nuevas corridas y no transforma fuentes future-work en evidencia empírica del paper actual.
