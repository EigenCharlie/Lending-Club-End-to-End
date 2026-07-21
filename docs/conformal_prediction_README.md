# Conformal Prediction — Legacy Engineering Index

> **LEGACY ENGINEERING NOTE (superseded for scientific claims on
> 2026-07-20).** This page is an inventory of reproducible local code and
> artifacts, not a production approval, model-validation opinion, or current
> scientific result. Historical names such as `pd_low`, `pd_high`, `champion`,
> `policy`, and `overall_pass` are runtime labels. They do not identify latent
> individual PD, ECL, SICR, staging, a selected learner, or a selected decision
> policy.

## Current authority

Use these sources before interpreting any artifact listed here:

- `SESSION_STATE.md` — current cross-project status;
- `docs/research/crpto_external_contract_2026-07-20.yml` — machine-readable
  boundary imported from the synchronized CRPTO repository;
- `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md` — history,
  evidence audit, and transfer decisions;
- `book/includes/_scientific-scope-contract.qmd` — shared book-page warning.

The present target is the observed future binary outcome $Y$. The repository
contains retrospective diagnostics of scores and prediction sets for that
target. It does **not** currently support confidence intervals for an
unobserved borrower-level probability, IFRS 9 accounting quantities, a robust
decision region, or a deployment claim.

## Reproducible inventory

| Surface | Historical path | Current reading |
|---|---|---|
| Runtime implementation | `src/models/conformal.py` | Code path for binary-$Y$ prediction diagnostics; API names retain historical terminology |
| Mondrian result object | `models/conformal_results_mondrian.pkl` | Frozen provenance object, not a current theorem or policy |
| Row-level output | `data/processed/conformal_intervals_mondrian.parquet` | Retrospective values whose `pd_*` columns are legacy schema names |
| Automated checks | `models/conformal_policy_status.json` | Snapshot of code-defined checks; `PASS` is not independent validation |
| Calibration artifact | `models/pd_canonical_calibrator.pkl` | Frozen historical calibrator; do not rewrite it to impose a new semantic rule |
| Research notes | `docs/research/conformal_prediction_*.md` | Historical literature and implementation notes, non-authoritative for current claims |

Counts, coverage values, and check ratios must be read from the artifact and
its recorded run context. They should not be copied into static prose as if
they were timeless facts.

## What may be claimed

Subject to the exact split and artifact provenance, the local evidence can
support statements about:

- discrimination or probability-score diagnostics for observed binary $Y$;
- empirical coverage of a declared prediction-set procedure in the evaluated
  sample;
- interval/set width and group-level descriptive summaries;
- software behavior, artifact lineage, and reproducibility;
- sensitivity of a retrospective calculation to a declared candidate rule.

Any finite-sample conformal theorem additionally requires its own valid
protocol and assumptions, including exchangeability and a fixed or otherwise
valid calibration procedure. The labeled 2018–2020 data were subsequently
inspected and reused across diagnostic and selection work, so they are not an
untouched prospective confirmation set for the repository's current narrative.

## What may not be promoted

The following inferences are outside the evidence contract:

- interpreting endpoints as confidence bounds for latent individual PD;
- calling a historical probability-score artifact an IFRS 9 PD at reporting
  date;
- deriving validated ECL, lifetime PD, SICR, or Stage 2 status from interval
  width;
- treating row-wise endpoints as a jointly valid box uncertainty set;
- claiming robust portfolio performance from reused terminal outcomes;
- calling proxy-group diagnostics a legal or demographic fairness validation;
- naming a current champion, threshold, policy, or deployable system.

Historical notebooks and scripts may mechanically compute some of these
quantities. Their existence is implementation evidence, not identification or
validation evidence.

## Venn–Abers semantic correction

The July 2026 audit separated two objects that had been conflated:

1. the public `venn-abers` multiprobability pair $(p_0,p_1)$; and
2. ordinary two-class probabilities $[P(Y=0),P(Y=1)]$ returned by a classifier
   API.

They are not interchangeable. For a Venn–Abers pair, the log-loss minimax point
is

$$
\hat p = \frac{p_1}{1-p_0+p_1}.
$$

New `VennAbersScoreCalibrator` fits record
`point_rule="log_loss_minimax"`. A legacy pickle that lacks `point_rule`
continues to use its historical midpoint rule so loading the same frozen file
does not silently alter its predictions. The compatibility behavior is tested;
the frozen pickle itself was not rewritten. `create_pd_intervals_venn_abers`
now obtains both the point and multiprobability endpoints from the public
`venn-abers` interface rather than interpreting class-probability columns as
bounds.

These endpoints remain Venn–Abers multiprobabilities for observed binary $Y$;
they are not confidence intervals for a latent borrower parameter.

## Historical engineering flow

The repository can reconstruct this local sequence:

```text
binary-outcome score
  -> probability calibration
  -> conformal or Venn–Abers diagnostic
  -> retrospective group/coverage checks
  -> experimental downstream transformations
```

Use the producer scripts and registries to reproduce code behavior, then retain
the generated run tag, configuration, input hashes, split dates, package
versions, and output hashes. Do not infer a scientific promotion merely because
an automated status field is true.

## Research-note index

- `docs/research/conformal_prediction_research_2026.md`
- `docs/research/conformal_prediction_quick_reference.md`
- `docs/research/conformal_libraries_comparison.md`

Those documents preserve the project's exploratory reasoning and may contain
superseded recommendations. They are useful for provenance and hypothesis
generation only. Where they conflict with the July 2026 contract, the contract
and the audited code behavior prevail.

## Closeout

The durable contribution of this lane is a reproducible binary-outcome
uncertainty laboratory plus a record of why prediction guarantees cannot be
transported automatically to PD, accounting, fairness, or downstream decision
claims. Future positive promotion requires a newly declared estimand, a clean
evaluation protocol, and evidence generated for that purpose.
