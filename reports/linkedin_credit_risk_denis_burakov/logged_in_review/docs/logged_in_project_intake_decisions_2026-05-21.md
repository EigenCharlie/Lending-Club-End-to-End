# Logged-In Project Intake Decisions - 2026-05-21

This memo closes the logged-in review of the Denis Burakov LinkedIn corpus. The logged-in pass used the user's own visible browser session to re-open the already indexed backlog, capture visible comment threads, inspect comment-shared links, and recover independent sources where possible. LinkedIn comments remain private intake context; public-facing claims below are tied to independent sources or explicitly parked.

## Coverage

- Posts re-opened with authenticated rendering: 80/80.
- Capture status: all rows are `logged_in_rendered_capture_complete`.
- Visible comments captured: 503 comments across 67 posts.
- External link inventory after dedupe: 234 rows.
- Priority mix: 73 high, 56 medium, 105 low.
- High-priority unique source attempts: 42.
- Direct readable/thin/blocked: 26 readable, 1 thin, 15 blocked.
- Alternate recoveries: 6 readable, 1 blocked.

## Promoted To The Quarto Book

- Chapter 05 now treats WOE as auditable evidence rather than only encoding, adds IV caution, and links WOE/Naive Bayes analogies to model-governance limits.
- Chapter 06 now makes Brier/reliability-diagram caveats explicit and separates ranking, calibration, threshold metrics, probability quality, and policy decisions.
- Chapter 09 now clarifies that Gini has economic value only when it changes approval volume, loss, capital allocation, or net return, and parks full profit scoring as future work.
- Chapter 10 now warns against universal PSI thresholds and strengthens separation among score, calibrator, decision policy, and business rules.
- Chapter 11 now frames SHAP-scorecards, RuleFit, and GBDT leaf one-hot linearization as explainable prototypes requiring stability/calibration checks.

## Paper Estrella

Decision: use this pass for reviewer-defense language only. The logged-in material strengthens the narrative around calibration-gated metric choice, reliability plots, explainability cost, rare-event probability correction, and model governance. It does not reopen the official champion or add a new experiment because the evidence is better suited to framing and caveats than to changing the empirical core.

Stop condition: closed unless a reviewer asks directly about Brier decomposition, calibration-plot comparability, rare-event logistic correction, or explainability trade-offs.

## Paper 4

Decision: keep four bounded lanes in the backlog rather than promoting them into experiments now:

- WOE recalibration and Good/Bayesian WOE under drift.
- RuleFit or GBDT-leaf scorecard distillation as interpretable model compression.
- PSI uncertainty and threshold governance as monitoring caveat.
- Gini-to-economic-value bridge for acceptance-rate and portfolio-value language.

Stop condition: no new lane opens unless it can change an appendix table, a reviewer response, or a source-backed claim already present in the paper package.

## Sources Recovered From Comment Threads

- Brier/Yates decomposition critique: arXiv source recovered and extracted.
- Gini/Accuracy Ratio/Somers D: BIS validation paper and Engelmann-Hayden-Tasche paper recovered.
- Rare-events logistic regression: King and Zeng source recovered through an official Harvard page.
- RuleFit and GBDT leaf encoding: arXiv sources recovered and extracted.
- IV/metric divergence: Zeng 2013 recovered through a direct Semantic Scholar PDF.
- PSI statistical properties: PDF recovered and extracted.
- Boosted/generalized Naive Bayes: Elkan, Larsen, and Ridgeway sources recovered.

## Remaining Blockers

Blocked or thin items remain logged rather than silently ignored: DataScience StackExchange class-imbalance discussion, SAS community SHAP/WOE page, SSRN mirrors, Medium, IEEE, Sage, Baeldung, ResearchGate images/pages, and one RPubs login wall. These are not needed for current claims because stronger or equivalent independent sources were recovered for the promoted material.

## Closure

The logged-in pass is closed for the current corpus. The remaining unresolved items are parked with explicit blocker status, not pending hidden work. Future LinkedIn review should start from new posts or new comments after 2026-05-21 rather than reprocessing this same queue.
