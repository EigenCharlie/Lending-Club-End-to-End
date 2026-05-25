# Andrija Logged-In Review Findings - 2026-05-25

This memo summarizes the Opera GX / Windows Playwright pass over the Andrija Djurovic P0/P1 queue. It used the user's own visible logged-in browser state through a non-destructive profile copy. Comments remain private research intake and are not public evidence.

## Coverage

- Queue rows captured: 37
- Capture status: {'not_authenticated_or_checkpoint': 2, 'capture_error': 1, 'logged_in_rendered_capture_complete': 34}
- Visible comments captured: 121 across 18 posts
- External link rows after dedupe by post/source/url: 72
- Link priority counts: {'high': 15, 'medium': 35, 'low': 22}
- Project decision rows: 37

## Why Opera Worked

The previous Chrome path was blocked because the Windows browser did not expose a reachable DevTools endpoint from the WSL process. The working path was to launch Opera GX on Windows with remote debugging against a copied Opera profile that already held the authenticated LinkedIn state, then run Playwright from Windows so `127.0.0.1` referred to the Windows browser.

## High-Value Logged-In Deltas

- `ANDRIJA-LOGIN-008`: comments exposed an arXiv preprint on statistical hypothesis testing for Information Value. This directly strengthens CRPTO metric-governance language around IV thresholds, inherited heuristics, class imbalance and p-value-based feature screening.
- `ANDRIJA-LOGIN-011`: the WoE instability thread clarified that the issue is not only binning or drift; iterative replacement using model-predicted outcomes can force a new model even under perfect replication, with a possible KL-convergent limit as a Paper 4 prototype only.
- `ANDRIJA-LOGIN-012`: residual-tree validation is useful as an omitted-risk-factor and over/underestimation diagnostic, but monotonicity and splitting-node governance keep it in the prototype/thesis lane.
- `ANDRIJA-LOGIN-013`: multi-period PD testing needs dependence-aware language; autocorrelation changes effective sample size and can make default-rate tests reject for assumption failure rather than true miscalibration.
- `ANDRIJA-LOGIN-015`: model shift stays thesis/MRM material, useful for specification-risk governance but not as a new IJDS claim.

## Blocked Or Archived

- Three queue rows remained blocked/error after the logged-in Opera pass; they are closed without captcha/checkpoint bypass.
- The dense non-credit-risk thread, hybrid-threat link, and financial-crime-compliance AI preprint are archived for CRPTO because they do not change the credit-risk decision pipeline.

## Source Reading Queue

| source_id | queue_id | source_title | evidence_status | decision |
| --- | --- | --- | --- | --- |
| ANDRIJA-LI-SRC-001 | ANDRIJA-LOGIN-008 | Statistical Hypothesis Testing for Information Value (IV) | preprint_not_peer_reviewed | promote_as_caveat_and_source_reading |
| ANDRIJA-LI-SRC-002 | ANDRIJA-LOGIN-012 | PDtoolkit package manual: segment.vld | software_documentation | append_to_paper4_prototype_queue |
| ANDRIJA-LI-SRC-003 | ANDRIJA-LOGIN-011 | ADSFCR repository | code_or_tool_source | append_as_context_not_public_claim |
| ANDRIJA-LI-SRC-004 | ANDRIJA-LOGIN-015 | Model shift prospectus | working_paper_or_prospectus | append_to_thesis_mrm |
| ANDRIJA-LI-SRC-005 | ANDRIJA-LOGIN-013 | Probability of default validation: Basel score and order statistic methodology | peer_reviewed_source_discovery_not_full_text | park_for_thesis_source_retrieval |
| ANDRIJA-LI-SRC-006 | ANDRIJA-LOGIN-030 | Agentic AI for Financial Crime Compliance | preprint_or_forthcoming_conference_out_of_scope | archive_for_crpto |

## Stop Rule

The logged-in Andrija pass is closed for P0/P1 when every row has a capture status, comment/link count, decision, implementable path, evidence status and stop condition. Reopen only if a newly accessible independent source or local experiment can change a claim, appendix table, reviewer response or thesis chapter.
