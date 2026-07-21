# Project Justification — Current Scientific and Architectural Rationale

Version: 2026-07-20

This document explains why the repository keeps a broad executable credit-risk
stack while assigning different scientific authority to its outputs. It does
not make every runtime artifact a current claim.

For the detailed evidence and commit history, use
`docs/research/crpto_evolution_cross_project_audit_2026-07-20.md`. The
surface-specific authorities are:

| Surface | Scientific status | Authority |
|---|---|---|
| External CRPTO/IJDS | retrospective identification audit; no selected learner or policy | `docs/research/crpto_external_contract_2026-07-20.yml` |
| Local runtime | reproducible historical engineering freeze | runtime artifacts and baseline registry |
| Paper 2 | `parked_ifrs9` methodological audit | `reports/paper_material/paper2/paper2_claim_contract.yml` |
| Paper 4 | bounded living lab without promotion | current Paper 4 findings and claim-boundary CSVs |
| Quarto book | historical/diagnostic synthesis | `book/_quarto.yml` and the scientific scope include |

## 1. Why preserve the end-to-end stack

The repository connects data engineering, tabular risk scores, calibration,
binary-outcome conformal diagnostics, portfolio stress utilities, survival and
time-series prototypes, causal research lanes, IFRS9-inspired calculations and
governance reports. Keeping those components is useful because it makes the
historical experiments reproducible and exposes where an estimand changes.

The graph is therefore an **execution graph**, not a chain of automatically
transported guarantees:

```text
data -> score/calibration -> interval for observed binary Y
     -> historical stress/optimization utilities
     -> diagnostic survival, causal, IFRS9 and governance modules
```

Every arrow that changes target, horizon, population or decision requires a
separate identification and validation argument. A successful software run is
not evidence that the downstream statistical or accounting claim is valid.

## 2. Repository architecture

The separation of concerns remains justified:

- `src/` contains reusable analytical logic;
- `scripts/` contains executable entry points;
- `configs/` records parameters and historical baseline resolution;
- `data/`, `models/` and `reports/` contain reproducible local artifacts;
- `book/` is the principal editorial surface for this repository;
- `api/` and `streamlit_app/` are optional delivery layers, not scientific
  authorities.

Pipeline families permit focused reconstruction without forcing every research
lane to rerun. Names such as `canonical`, `champion`, `selected` and
`operational` inside those interfaces describe their historical runtime role;
they do not establish a current scientific winner, deployment approval or
external-paper result.

## 3. Why the modeling components remain useful

### Score estimation and calibration

Logistic regression remains a transparent baseline and CatBoost a useful
tabular challenger/freeze. Temporal partitions and calibration diagnostics are
preferable to random-split leaderboards. They support retrospective description
of the observed binary outcome, but the current archive does not supply an
untouched final evaluation after all later selections.

### Conformal layer

Global, Mondrian and related variants remain valuable for auditing candidate
coverage and subgroup behavior for the observed binary outcome $Y$. The
endpoints are **not** confidence limits for an individual latent PD. Candidate
coverage also does not prove coverage after selecting a learner, taxonomy,
window or downstream policy with the same evidence.

### Portfolio utilities

The deterministic and robust solvers make assumptions and sensitivities
executable. Their legacy upper-endpoint box is a numerical stress device, not a
probabilistically identified PD uncertainty set. Historical price-of-robustness
and allocation comparisons are conditional on their ruler, coordinate, support
and simulator; none is a current selected policy.

### Survival and time-series lanes

These lanes demonstrate useful methods and reveal the importance of censoring,
prepayment, horizons and calendar structure. The available Lending Club
snapshot does not turn vintage summaries into a reporting-date panel or a
prospective portfolio forecast.

### Causal lane

The causal modules preserve a research workflow for assumptions, overlap and
sensitivity. Accepted-loan observational data do not identify an underwriting
policy effect without stronger treatment assignment, rejected-applicant support
or an appropriate instrument.

### IFRS9-inspired calculations

Scenario, staging and ECL code is retained as a mechanical diagnostic. The
current data lack the loan--reporting-date panel, contractual cash-flow state,
point-in-time DPD/cure/forbearance history, comparable origination/current PD
horizons and governed macro scenarios required for an IFRS 9 claim. Binary-$Y$
interval width is not a validated SICR signal. Paper 2 therefore remains
`parked_ifrs9`.

### Governance and fairness reports

Automated checks are useful reproducibility and screening controls. Runtime
`PASS` values, threshold files and fairness summaries are not independent model
validation, legal fair-lending conclusions, accounting opinions or regulatory
approval.

## 4. Why the CRPTO pivot was necessary

The autonomous CRPTO paper moved from an economic-champion narrative to an
identification audit after examining outcome maturity, endpoint geometry,
selection, common support and comparator dependence. Its current contract
reports the complete candidate universe and sensitivity families without
selecting a learner, window, taxonomy, $\gamma$, ruler, coordinate, cap,
comparator or policy.

That pivot does not discard the engineering work. It gives the work the
strongest interpretation supported by the observed data: a retrospective audit
of binary geometry and transport, rather than a policy recommendation.

## 5. Roles of Paper 2, Paper 4 and the book

- **Paper 2** is a negative methodological audit: it documents why the current
  binary terminal-outcome archive does not identify reporting-date PD, ECL or
  SICR and specifies the data contract required to reopen the study.
- **Paper 4** is a governed living lab: internally simulated paths,
  post-selection slices, teacher-cost PyEPO exercises and proxy-score
  comparisons may be reported descriptively with their boundaries, but cannot
  be promoted into CRPTO or prudential claims.
- **The Quarto book** preserves technical provenance, explains failure modes
  and makes cross-surface contracts visible. It is not a second source of truth
  for either paper.

## 6. Current priorities and stop rules

1. Keep each claim tied to a declared target, population, horizon, ruler and
   evidence artifact.
2. Separate outcome-free construction, model/rule selection and final
   evaluation.
3. Preserve unresolved outcomes and maturity bounds instead of silently
   imputing terminal labels.
4. Do not infer scientific status from filenames or runtime registry labels.
5. Reopen Paper 2 only with the full longitudinal and accounting data contract.
6. Promote a Paper 4 lane only with a defensible estimand, independent units or
   grouped inference, and a final temporal evaluation unused for selection.
7. Import anything into CRPTO only through its own registered protocol and
   active claim ledger.

Detailed historical choices and mistakes remain in
`docs/DECISION_CHANGES_AND_LEARNINGS.md`; they are provenance, not current
authorization.
