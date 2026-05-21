# External High-Value Sources Memo

Generated: 2026-05-21

## Scope

This memo closes the 21 external links marked `read_as_potential_evidence` in
`data/external_link_backlog.csv`. The readable snapshots are recorded in
`data/high_value_external_source_reading.csv` and stored under
`external_sources/readable/`.

All 21 sources were fetched as readable HTTP 200 artifacts. GitHub links were
converted to raw files or README snapshots where possible, so their status is no
longer based on GitHub UI HTML.

## Source-Level Decisions

### Metric Governance: AUROC/AUPRC Under Imbalance

Source: POST-058 external preprint, `arxiv.org/html/2401.06091v1`.

Decision: append as preprint-labeled metric governance context. The source
supports a manuscript-safe warning already aligned with the project: PR/AUPRC is
useful in rare-event settings, but it is not an automatic replacement for
AUROC/Gini. Metric choice must follow the decision objective, prevalence, cost
structure, and operating region. This has been reflected in Chapter 06 as a
short caution on AUPRC.

Stop condition: closed after adding the cautionary book language. Do not cite as
peer-reviewed evidence unless its publication status is separately verified.

### GitHub Prototypes: FastWoe, xBooster, WoeBoost, Fisher Scoring

Sources: POST-020, POST-024, POST-028, POST-047, POST-049, POST-054, POST-056,
POST-059 external GitHub links.

Decision: park as prototype candidates, not dependencies. These repos are
valuable for concept discovery around WOE inference, multiclass WOE, boosted
scorecards, SHAP scorecards, interval scorecards, fine-tuning, and Fisher
scoring variants. They do not justify changing the official champion or adding a
runtime dependency without a bounded experiment and rejection rule.

Stop condition: closed as backlog-only. Reopen only if a future appendix table,
reviewer response, or teaching notebook explicitly needs one of these packages.

### AWS Credit Risk Modeling Repo

Sources: POST-016, POST-018, POST-021 external GitHub links.

Decision: append as governance context only. The repo reinforces a useful system
boundary: real-time scoring, batch scoring, MLOps, and LLM/document-processing
examples should remain implementation patterns, not empirical claims. Chapter 10
already received the score/calibrator/policy/rules separation.

Stop condition: closed after governance language. No AWS implementation lane is
opened for the current project.

### Calibration Visualization Code

Source: POST-035 external raw Python file.

Decision: keep as teaching/source-trail context. The file benchmarks calibration
visualization using simulated data and common model families. The project already
has its own calibration artifacts and champion evidence, so no code import is
needed.

Stop condition: closed after snapshot. Use only if a future teaching figure or
appendix needs a calibration-visualization source trail.

### Pearsonify

Source: POST-045 external GitHub README.

Decision: append as related-work context. Pearson-residual/conformal-style
classification intervals are useful framing for Paper Estrella, but they answer
a different uncertainty question than the existing conformal PD intervals.

Stop condition: closed as related-work contrast. Do not run a new interval
experiment unless it changes an already-promoted table.

### Out-of-Scope Generative AI Sources

Sources: POST-038 external arXiv preprint and PatchGPT repo.

Decision: archive for context. These are readable but not relevant enough to
credit-risk modeling, Quarto book claims, Paper 4, or Paper Estrella.

Stop condition: closed as archive. No implementation.

### Official Matplotlib Sources

Sources: POST-041 and POST-057 external official documentation.

Decision: keep only as plotting-tool provenance. They do not support credit-risk
claims.

Stop condition: closed after source classification.

## Final Gate

No public-facing claim was promoted from LinkedIn or GitHub alone. The only
project text change from this source batch is a metric-governance caution in
Chapter 06. Everything else is parked, archived, or retained as implementation
context with explicit stop conditions.
