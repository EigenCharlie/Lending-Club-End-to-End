# Historical Model-Risk Control Reconstruction

> **HISTORICAL INTERNAL CONTROL DOCUMENT (2026-07-20 boundary).** This file
> reconstructs the local model freeze and its automated controls. It is not an
> independent validation, an SR 11-7 compliance opinion, a deployment approval,
> an accounting opinion, or authority for CRPTO, Paper 2, or Paper 4. Historical
> terms such as `champion`, `PD`, `PASS`, `promotable`, and `policy` retain their
> runtime-schema meaning only.

## 1. Authority and purpose

Current scientific status is governed by `SESSION_STATE.md`,
`docs/research/crpto_external_contract_2026-07-20.yml`, and the relevant
surface-specific claim contract. This document answers the narrower question:
what did the local software freeze, check, and record?

The primary modeled target is the observed future binary outcome $Y$ in the
Lending Club archive. The runtime often calls its score `PD`; that label does
not by itself identify an IFRS 9 probability of default at reporting date or a
latent borrower-level parameter.

## 2. Frozen artifact inventory

| Field | Historical runtime value | Current interpretation |
|---|---|---|
| Model name | `CorePDCanonical` | Local binary-outcome score |
| Estimator | monotonic CatBoost plus calibration | Reproducible fitted artifact, not a current winner |
| Model artifact | `models/pd_canonical.cbm` | Frozen provenance object |
| Calibrator | `models/pd_canonical_calibrator.pkl` | Frozen Venn–Abers object with legacy point semantics |
| Feature contract | `models/pd_model_contract.json` | Runtime schema for 42 model features |
| Conformal output | `data/processed/conformal_intervals_mondrian.parquet` | Binary-$Y$ diagnostic with legacy `pd_*` names |
| Registry | `models/champion_registry.json` | Historical selection record, not current scientific authority |

Artifacts and status JSON files must be interpreted with their run tags,
configs, data hashes, split definitions, and package versions. A file called
`canonical` or `champion` is evidence of an engineering decision at that time,
not proof of external validity.

## 3. Data and evaluation history

The archived workflow used temporal train, calibration, and 2018–2020 labeled
evaluation partitions. Post-outcome and repayment variables were removed by the
feature pipeline. That separation is useful evidence against direct fitting
leakage in the original run.

It is nevertheless incorrect to call the 2018–2020 outcomes a currently
untouched holdout. They were later inspected and reused across calibration
diagnostics, conformal analyses, challenger searches, threshold work, and
economic-policy experiments. Accordingly:

- reported metrics are retrospective, in-sample-to-the-research-program
  diagnostics;
- comparisons affected by that reuse are post-selection summaries;
- the archive cannot provide prospective confirmation for a selected model or
  policy without new untouched data or a properly nested protocol.

Temporal ordering also does not establish exchangeability or eliminate
distribution shift. It changes the evaluation geometry; it does not repair the
assumption.

## 4. What the automation checked

The repository records automated checks for discrimination, calibration,
coverage, drift, monotonicity, proxy-group disparities, feature stability, and
downstream calculations. Examples include:

- `data/processed/pipeline_summary.json`;
- `models/conformal_policy_status.json`;
- `models/fairness_audit_status.json`;
- `models/governance_status.json`;
- `models/pd_backtesting_status.json`;
- `models/bootstrap_validation_status.json`;
- `models/calibration_mapping_status.json`;
- `models/encoding_stability_status.json`.

These files can demonstrate that code ran and that declared numerical rules
were met. They do not demonstrate independent validation, regulatory
acceptance, legal fairness, or continuing monitoring. Large-sample hypothesis
tests and policy booleans require substantive interpretation; `overall_pass`
must never be translated mechanically into model approval.

### Conformal and Venn–Abers boundary

Observed coverage can be reported for the declared binary-$Y$ procedure and
sample. A finite-sample theorem requires the protocol's assumptions and does
not turn score endpoints into confidence bounds for latent PD. Group summaries
over proxies are not demographic fairness validation.

The July 2026 audit also found that ordinary class-probability columns and the
Venn–Abers multiprobability pair had been conflated in one helper. New fits use
the log-loss minimax point rule; old pickles without a rule marker preserve the
historical midpoint behavior. No frozen model or result artifact was rewritten.

## 5. Permitted and blocked uses

Permitted uses of the freeze are limited to:

- software and artifact-lineage reproduction;
- retrospective diagnostics for the observed binary outcome;
- teaching and methodological reconstruction;
- hypothesis generation for a newly designed study;
- sensitivity calculations explicitly labeled as such.

The current evidence does not authorize:

- automated underwriting or borrower-level decisions;
- deployment or an operational forecasting contract;
- interpretation as regulatory capital or IFRS 9 PD;
- ECL, lifetime-PD, SICR, or staging conclusions;
- a current learner, threshold, comparator, or portfolio-policy winner;
- robust-decision guarantees from marginal row-wise endpoints;
- legal, protected-class, or demographic fairness conclusions;
- generalization to other populations without new evidence.

Historical portfolio and IFRS9-inspired scripts remain useful as executable
research components. Their outputs are mechanical transformations of inputs,
not validated accounting estimates or realized-decision evidence.

## 6. Historical governance design

The repository encoded a developer/validator/owner separation, Git and DVC
lineage, MLflow logging, challenger criteria, retraining triggers, and quarterly
monitoring thresholds. That design is evidence of intended controls. The
archive does not establish that an independent validator assumed those roles,
that a committee approved deployment, or that recurring monitoring occurred.

Likewise, the runtime distinction among `point_champion`,
`interval_champion`, and `promotable` records how scripts routed artifacts. It
does not create an official operational forecast or validated interval layer.
The time-series and IFRS9-inspired lanes remain retrospective analytical
support under the current claim contract.

## 7. Requirements for a future promotion

A defensible future model-risk package would need, at minimum:

1. a declared estimand and intended use before evaluation;
2. a target definition appropriate to that use, including horizon and
   observation process;
3. untouched prospective data or properly nested selection and evaluation;
4. independent validation with documented challenge and disposition;
5. calibration, discrimination, stability, and subgroup analyses tied to the
   intended population;
6. separate validation of every downstream accounting or decision mapping;
7. explicit owners, monitoring cadence, escalation paths, and evidence that
   the controls actually operated.

Until those conditions are met, the appropriate status is historical,
diagnostic, and non-promoted.
