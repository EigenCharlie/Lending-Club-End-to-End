# Andrija Djurovic LinkedIn Intake Decisions - 2026-05-25

## Status

This pack covers the public material captured from
<https://www.linkedin.com/in/andrija-djurovic/> plus public articles and
resolved external links. LinkedIn material is intake evidence only: it
can suggest concepts, implementation ideas, reviewer-defense language or
source-discovery paths, but no public paper/book claim is promoted from
LinkedIn alone.

- Indexed posts: 21
- Public articles: 4
- External link backlog rows: 58
- High-value readable source rows: 50
- Asset manifest status counts: {'image_ocr_attempted': 4, 'pdf_text_extracted_pdftotext': 9, 'pending_external_source_review': 58, 'post_text_captured_no_media_found': 8, 'transcript_captured': 9}
- External source reading status counts: {'pdf_text_extracted_pdftotext': 40, 'readable_http_200': 10}
- Backlog closure counts: {'closed': 23, 'closed_blocked_public_capture': 2}
- Decision counts: {'already_implemented_append_trace': 1, 'append_as_governed_intake': 1, 'append_as_source_map': 2, 'append_to_thesis_lane': 2, 'archive_blocked': 2, 'archive_source_discovery': 1, 'archive_source_trace': 1, 'archive_tooling_context': 3, 'park_for_thesis': 1, 'park_with_claim_gate': 1, 'park_with_executable_prototype_rule': 1, 'promote_as_caveat': 1, 'promote_to_crpto_caveat': 1, 'promote_to_crpto_defense': 1, 'promote_to_crpto_defense_and_park_prototype': 1, 'promote_to_crpto_language': 1, 'promote_to_defense_and_park_prototype': 1, 'promote_to_reviewer_defense': 2, 'promote_to_thesis_and_defense': 1}
- Visual read/triage counts: {'manual_visual_read_completed': 4, 'manual_visual_triaged_context_only': 48}

## Promote / Append / Park / Archive

| backlog_id | topic_family | handling_decision | project_destination | closure_status |
| --- | --- | --- | --- | --- |
| ANDRIJA-POST-001 | ADSFCR / IFRS9 / validation source bundle | append_as_source_map | Mini libro CRPTO como intake; tesis para FLI/IFRS9; Paper 4 solo por appendices gobernados. | closed |
| ANDRIJA-POST-002 | Applied Data Science for Credit Risk umbrella | append_as_governed_intake | Mini libro CRPTO; docs de investigacion; tesis. | closed |
| ANDRIJA-POST-003 | PD backtesting multi-period average testing | promote_to_crpto_defense_and_park_prototype | Paper CRPTO reviewer-defense; Paper 4 candidate lane; tesis validation chapter. | closed |
| ANDRIJA-POST-004 | Tree-based PD risk-factor interactions | park_with_executable_prototype_rule | Paper 4 prototype; tesis feature-engineering; libro Ch05/Ch06. | closed |
| ANDRIJA-POST-005 | Reliability limits of multi-period normal tests | promote_as_caveat | Mini libro CRPTO reviewer-defense; tesis validation chapter. | closed |
| ANDRIJA-POST-006 | Supervised Macroeconomic Index for IFRS9 FLI | append_to_thesis_lane | Tesis IFRS9 chapter; CRPTO limitations. | closed |
| ANDRIJA-POST-007 | LGD/EAD Somers D under conservatism | park_for_thesis | Tesis; Paper 4 si se abre LGD/EAD diagnostic lane. | closed |
| ANDRIJA-POST-008 | MoC Type C aggregate conservatism | park_with_claim_gate | Tesis governance; Paper 4 optional appendix. | closed |
| ANDRIJA-POST-009 | Two-sided exact binomial PD backtesting | already_implemented_append_trace | Mini libro CRPTO validation appendix; docs ADSFCR. | closed |
| ANDRIJA-POST-010 | Blocked public LinkedIn post | archive_blocked | Archive. | closed_blocked_public_capture |
| ANDRIJA-POST-011 | Blocked public LinkedIn post | archive_blocked | Archive. | closed_blocked_public_capture |
| ANDRIJA-POST-012 | Monotonic binning tooling for PD/LGD/EAD | archive_tooling_context | Libro Ch05; tesis feature-engineering; archive/tooling context. | closed |
| ANDRIJA-POST-013 | IFRS9 forward-looking modeling source bundle | append_to_thesis_lane | Tesis; CRPTO limitations; docs research. | closed |
| ANDRIJA-POST-014 | Model-based discrete PD rating scale calibration | promote_to_defense_and_park_prototype | Paper CRPTO reviewer-defense; Paper 4 candidate; tesis calibration. | closed |
| ANDRIJA-POST-015 | R IRB toolkit | archive_tooling_context | Archive/tooling context; tesis appendix if needed. | closed |
| ANDRIJA-POST-016 | Probability of Default Rating Modeling with R book release | archive_source_discovery | Tesis source-discovery; archive. | closed |
| ANDRIJA-POST-017 | Credit Risk Modeling Working Notes second update | append_as_source_map | Tesis; docs research; CRPTO limitations. | closed |
| ANDRIJA-POST-018 | PDtoolkit package | archive_tooling_context | Archive/tooling context; thesis appendix. | closed |
| ANDRIJA-POST-019 | ADSFCR repository source anchor | archive_source_trace | Archive/source trace. | closed |
| ANDRIJA-POST-020 | Model shift and scorecard model risk | promote_to_thesis_and_defense | Tesis MRM; CRPTO limitations; Paper 4 governance. | closed |
| ANDRIJA-POST-021 | WoE encoding instability | promote_to_crpto_caveat | Mini libro CRPTO; Paper 4 candidate; tesis feature governance. | closed |
| ANDRIJA-ARTICLE-001 | Conformal inference for IRB model uncertainty | promote_to_reviewer_defense | Paper CRPTO discussion/reviewer-defense; tesis. | closed |
| ANDRIJA-ARTICLE-002 | IRB calibration and risk quantification | promote_to_crpto_language | Mini libro CRPTO; IJDS discussion; tesis. | closed |
| ANDRIJA-ARTICLE-003 | Selective ML support for IRB models | promote_to_reviewer_defense | Paper CRPTO reviewer-defense; Paper 4 prototypes; tesis. | closed |
| ANDRIJA-ARTICLE-004 | Machine learning for IRB models | promote_to_crpto_defense | Mini libro CRPTO; IJDS discussion; tesis. | closed |

## High-Value Decisions

- **PD backtesting**: multi-period average testing, reliability limits of
  normal tests, and exact binomial testing strengthen reviewer-defense
  around calibration/backtesting. Exact binomial is already implemented;
  multi-period average testing is parked as an appendix/prototype only
  if it changes a validation claim.
- **PD calibration**: the model-based discrete PD rating-scale deck and
  IRB calibration article sharpen the distinction between risk
  differentiation, risk quantification and decision value.
- **Selective ML**: tree-based interactions and selective ML articles
  support a controlled Paper 4 lane: ML as risk-factor engineering,
  residual diagnostics or challenger support, not as uncontrolled
  champion replacement.
- **WOE/encoding stability**: WoE instability is promoted as a CRPTO
  caveat and thesis/Paper 4 candidate because it connects preprocessing,
  monitoring and self-labeled replication risk.
- **IFRS9/LGD/EAD/MRM**: SMI, PCA/ADF/recursive-regression, LGD/EAD
  Somers D and model shift are thesis/governance material. They should
  remain outside the IJDS body except as limitations or future work.

## Stop Rule

The Andrija intake is closed when each indexed post/article has a local
content status, a destination, a possible implementable, and a stop
condition. Reopen only if a newly visible logged-in comment/link, a
canonical paper, or a local experiment can change a claim, appendix
table, reviewer response or thesis chapter.
