# SESSION STATE — Lending Club Risk Project

Last updated: 2026-07-20

## 1. Executive status

There is no single artifact that is the scientific source of truth for every
surface. The current authority depends on the question:

| Surface | Current status | Authority |
|---|---|---|
| External CRPTO/IJDS | retrospective identification audit; no selected learner or policy | `docs/research/crpto_external_contract_2026-07-20.yml` |
| Root runtime | reproducible historical pipeline freeze | runtime artifacts and `configs/baselines/canonical_operational_baseline.json` |
| Paper 2 | `parked_ifrs9` diagnostic note | `reports/paper_material/paper2/paper2_claim_contract.yml` |
| Paper 4 | 12-page living lab, no final promotion | current findings, boundaries and page registry under `reports/paper_material/paper4/tables/` |
| Quarto book | historical/diagnostic synthesis governed by explicit scope boundaries | `book/_quarto.yml` and `book/includes/_scientific-scope-contract.qmd` |

The former rule that `models/final_project_promotion.json` plus
`models/champion_portfolio_policy.json` represented the whole project's current
truth is retired. Those files remain runtime compatibility/provenance for the
local April freeze; they do not define current CRPTO, Paper 2 or Paper 4 claims.

## 2. External CRPTO/IJDS contract

The autonomous `Paper_CRPTO` repository was observed on `main` at
`69095e05beae282701b4ea38aa69da26a209106f`. The portable contract pins 13
active/editorial surfaces by SHA-256.

Current scientific boundary:

- target: observed binary outcome $Y$;
- role: retrospective identification audit;
- no selected learner, residual window, taxonomy, gamma, ruler, coordinate,
  cap, comparator or policy;
- candidate coverage of $Y$ is not an interval for latent individual PD;
- coverage does not transport automatically to ECL, SICR, expected loss or
  selected-set validity;
- local artifacts named `champion`, `pool93`, `compact-v7` or selected-policy
  are historical provenance, not external-paper evidence.

See `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md` for the
commit-level evolution and transfer matrix.

## 3. Local runtime freeze — historical, still executable

The local pipeline deliberately retains its April artifacts and interface
names so that old runs can be reproduced. Two identifiers remain important:

- operational PD baseline:
  `canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129`;
- historical Paper Estrella freeze:
  `paper-thesis-final-economic-2026-04-06`.

The latter includes the legacy label
`bound_aware_276k_economic_champion`. “Champion” here means *selected by that
historical local procedure*, not current scientific winner or deployment
recommendation.

Baseline resolution remains governed by
`configs/baselines/canonical_operational_baseline.json`. Threshold semantics
remain separated:

- **threshold interno**: PD search/screening threshold used inside the legacy
  experimentation workflow;
- **threshold operativo**: distinct local application threshold recorded in
  `models/threshold_semantics.json`.

Neither threshold is a current CRPTO policy and neither is an IFRS9/SICR rule.

### Runtime pipeline map

```text
raw data
  -> temporal splits and feature artifacts
  -> local score/calibrator freeze
  -> binary-outcome conformal endpoints
  -> local stress/optimization utilities
  -> diagnostic survival, causal, IFRS9 and governance modules
  -> Quarto and optional Streamlit views
```

This is an execution graph, not a chain of transported guarantees. Every arrow
that changes target or decision requires its own estimand and validation.

### July calibration/conformal implementation audit

The frozen Venn--Abers pickle has no `point_rule` marker and therefore retains
the historical midpoint summary. Replay manifests now freeze
`selected_calibration_point_rule`; an older Venn--Abers manifest without that
field resolves to `midpoint_legacy`. New search fits record
`log_loss_minimax` and use $p_1/(1-p_0+p_1)$. The helper that once interpreted
ordinary MAPIE class probabilities as Venn--Abers endpoints was removed. No
frozen model, pickle, metric table or historical decision was rewritten.

This does not restore an active conformal theorem. The same calibration labels
were used to fit the probability calibrator and to conformalize, and the OOT
outcomes were later reused during comparisons and selection. Local coverage is
therefore retrospective empirical coverage of binary $Y$, not an independent
holdout result or a guarantee for latent PD. Exact impact and compatibility
rules are recorded in the cross-project audit.

### Runtime authorities

Use these to reproduce local software behavior, not to override paper claim
contracts:

- `data/processed/model_comparison.json`;
- `models/conformal_policy_status.json`;
- `models/champion_registry.json`;
- `models/champion_portfolio_policy.json`;
- `models/threshold_semantics.json`;
- `models/causal_effect_status.json` and `models/causal_policy_rule.json`;
- `models/ifrs9_diagnostics_status.json`;
- `models/fairness_audit_status.json`;
- `reports/mrm/mrm_validation_report.json`;
- `data/processed/pipeline_summary.json`.

The automated MRM/fairness reports are internal diagnostics. They are not an
independent model validation, accounting opinion, fair-lending determination or
regulatory approval.

## 4. Paper 2 — parked IFRS9-inspired audit

Paper 2 no longer promotes its historical ECL, alpha-sweep, BMA or SICR
numbers. The current contribution is diagnostic: it shows why a binary terminal
outcome pipeline does not identify reporting-date PD, lifetime ECL or SICR.

The blockers are structural:

- no loan–reporting-date panel;
- no point-in-time DPD/cure/forbearance state;
- no origination/current PD pair with the same horizon;
- threshold design and evaluation used the same OOT evidence;
- resolved-loan maturity selection;
- inconsistent central ECL transformations;
- no final untouched temporal test for the prudential rule.

Do not reestimate or promote those numbers with the present archive. Reopen
only after satisfying every requirement in the Paper 2 claim contract.

## 5. Paper 4 — bounded living lab

The active Quarto surface has 12 pages. Its allowed interpretation is:

- F03: execution/provenance fact under one internal simulator;
- F04: descriptive result conditional on 512 internally generated paths;
- F05: post-selection slices, not strict holdouts;
- F06/PyEPO: descriptive teacher-cost benchmark over overlapping menu/seed
  rows; the active view removes the invalid row-level Wilcoxon and manual
  auditability score;
- F08/F13: negative estimand/readiness audits only; Paper 4 does not absorb
  Paper 2's ECL, SICR, staging, threshold or monetary claims;
- F14: legacy-root versus origin-time FICO proxy diagnostic only.

Paper 4 has no `paper4_final_promotion.json`. It does not establish independent
instance-level inference, external forecast validity, selected-set conformal
coverage, causal policy value, legal fairness, Bellman/DLA optimality or a
current CRPTO result.

Authorities:

- `reports/paper_material/paper4/tables/paper4_current_official_findings.csv`;
- `reports/paper_material/paper4/tables/paper4_current_claim_boundaries.csv`;
- `reports/paper_material/paper4/tables/paper4_quarto_page_registry.csv`;
- `reports/paper_material/paper4/notes/paper4_living_lab_notebook.md`.

## 6. Delivery architecture

1. Quarto is the primary editorial surface for this repository's historical
   pipeline and bounded research notes.
2. The public Streamlit app is a historical showcase; the local app is an
   optional companion lab.
3. DuckDB/dbt/Feast, FastAPI and MCP are engineering layers, not evidence that
   a model is production-validated.
4. The external CRPTO paper remains autonomous and must not depend on this
   repository.

The obsolete secondary book entrypoints `_quarto-core.yml`, `index-core.qmd`
and `scripts/serve_book_core.py` are retired. The sole book navigation authority
is `book/_quarto.yml`. The tracked 322-page `book/_output_pdf` snapshot was also
removed as a stale binary authority; current HTML/PDF outputs are disposable
renders produced from the governed sources and freezes.

## 7. Verification snapshot

The 2026-07-20 reconciliation closes with:

- 826/826 pytest checks passing and 13 expected third-party warnings;
- Ruff and formatting checks passing for all 16 changed Python files;
- one 90-input Quarto book rendered jointly to HTML and a 402-page letter PDF;
- 11,304 local HTML references checked with zero broken targets;
- 62/62 dependency-summary rows reproduced exactly;
- the required DVC cache synchronized remotely, while the four scientific
  stages listed in the cross-project audit remain intentionally dirty rather
  than silently refrozen;
- final PDF visual verdict `APPROVE_RENDER` for SHA-256
  `065d863e6aaa70290bf6a6fae9724edc2ce84383ce8ee07e72f0a070eaf9107a` after
  rasterizing all 402 pages; no clipping, overflow, collisions or retired
  legacy visuals were found.

This snapshot verifies the bounded contracts and software surfaces. It is not
new evidence for PD, ECL, SICR, causal policy value or deployment readiness.

## 7. Update rules

- Never infer scientific status from an artifact filename.
- Keep outcome-free construction separate from evaluation outcomes.
- Keep candidate performance separate from post-selection validity.
- Do not relabel endpoints for $Y$ as PD, ECL or SICR intervals.
- A new external CRPTO commit requires an explicit refresh of commit and hashes
  in the portable contract.
- A Paper 2 unpark or Paper 4 promotion requires its own data/protocol gate and
  a final evaluation not used for selection.
- Record detailed historical decisions in
  `docs/DECISION_CHANGES_AND_LEARNINGS.md`; keep this file compact and current.
