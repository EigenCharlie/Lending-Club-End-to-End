# Documentation Map

Quick reference for the documentation stack after the 2026-07-20 cross-project
claim audit. Local runtime artifacts remain engineering authorities for the
historical pipeline; they are not authorities for the external CRPTO/IJDS
paper.

## Keep Closest To Hand

| Category | File | Purpose |
|---|---|---|
| **Historical editorial ledger** | `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md` | March--April local freeze map; superseded for current paper claims |
| **Cross-project scientific audit** | `docs/research/crpto_evolution_cross_project_audit_2026-07-20.md` | Commit-level CRPTO evolution, estimand audit, and transfer/rejection matrix for the book, Paper 2, and Paper 4 |
| **External CRPTO contract** | `docs/research/crpto_external_contract_2026-07-20.yml` | Pinned external commit, hashes, counts, and no-selection/$Y$-estimand boundary |
| **CRPTO retirement boundary** | `docs/research/crpto_retirement_and_paper4_role_2026-06-06.md` | Current ownership, active external surfaces, and cross-project stop rules |
| **Current state** | `SESSION_STATE.md` | Operational snapshot and runtime-facing source list |
| **MRM / governance (historical runtime)** | `docs/MODEL_RISK_MANAGEMENT.md` | Local SR 11-7-inspired control framing; not independent validation or current paper authority |
| **ADSFCR adoption** | `docs/ADSFCR_AUDIT_AND_MONOTONIC_CHALLENGER_PLAN_2026-03-29.md` | Detailed audit of the external repo, adoption decisions, and tranche-by-tranche implementation status |
| **ADSFCR next work** | `docs/ADSFCR_EXECUTABLE_BACKLOG_2026-03-30.md` | Execution-oriented backlog for the remaining ADSFCR items that still look worth implementing |
| **Quarto contract** | `docs/QUARTO_BOOK_BLUEPRINT.md` | Book architecture, editorial contract, and maintenance rules |
| **Project rationale** | `docs/PROJECT_JUSTIFICATION.md` | Current layered rationale and scientific boundaries for the runtime freeze, CRPTO, Paper 2, Paper 4 and the book |
| **Runbook** | `docs/RUNBOOK.md` | Reproducibility playbook |
| **Search wave 2026-04** | `docs/PIPELINE_FIRST_TOPOLOGY_2026-03-31.md` | Pipeline-first taxonomy including the new `search_paper2_ifrs9` lane for exhaustive April runs |
| **TS vNext decision** | `docs/TIME_SERIES_VNEXT_DECISION_2026-04-02.md` | Current keep / research / do-not-promote decision for the time-series redesign lane |
| **History / learnings** | `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Historical decisions, fixes, and practical learnings |
| **Paper references** | `docs/PAPER_REFERENCES_STATE_OF_ART.md` | Curated literature map for papers and thesis chapters |
| **Backlog** | `docs/backlog-papers-unified.md` | Unified backlog for papers, experiments, and documentation follow-ups |
| **Conformal note (legacy engineering)** | `docs/conformal_prediction_README.md` | Historical method/API entrypoint; endpoint names do not override the current $Y$-estimand contract |
| **Paper Estrella audit (historical)** | `docs/research/paper_estrella_audit_2026-05-04.md` | Pre-identification-pivot snapshot; use the 2026-07-20 audit for current claims |
| **Paper Estrella backlog (historical)** | `docs/research/paper_estrella_backlog_2026-05-04.md` | Historical backlog, not the current external CRPTO contract |
| **Paper 2 claim contract** | `reports/paper_material/paper2/paper2_claim_contract.yml` | `parked_ifrs9` status, prohibited inferences, and evidence required to reopen |
| **Paper 4 official chapter** | `book/chapters/19-paper-mega-extension/index.qmd` | Curated 12-page Paper 4 Quarto surface after the living-lab claim-boundary audit |
| **Paper 4 living notebook** | `reports/paper_material/paper4/notes/paper4_living_lab_notebook.md` | Canonical notebook for Paper 4 waves, blockers, goals, failures, and next implementables |
| **Paper 4 official findings** | `reports/paper_material/paper4/tables/paper4_current_official_findings.csv` | Current artifact-backed findings allowed in the official Paper 4 chapter |
| **Paper 4 claim boundaries** | `reports/paper_material/paper4/tables/paper4_current_claim_boundaries.csv` | Current claim boundary matrix; supersedes historical page-level catalogs such as `19u` |
| **Paper 4 page registry** | `reports/paper_material/paper4/tables/paper4_quarto_page_registry.csv` | Registry of rendered official pages vs historical archive pages |
| **Paper 4 v41-v44 lab wave** | `reports/paper_material/paper4/status/paper4_v41_status.json` through `reports/paper_material/paper4/status/paper4_v44_status.json` | Latest living-lab execution wave; lab-only outputs documented in the notebook, not promoted to Quarto |
| **Paper 4 publishability memo** | `reports/paper_material/paper4/notes/paper4_publishability_focus_memo.md` | Strategy split: Paper Estrella toward INFORMS JDS, Paper 4 as future Management Science candidate only after stronger sequential evidence |
| **Global v38 synthesis (historical)** | `reports/paper_material/global/status/global_v38_status.json` | Provenance of the v31-v38 synthesis wave; it is not a current cross-project source of truth |

## Runtime Sources Of Truth

Use these before trusting prose about the **local historical pipeline**. For
CRPTO/IJDS claims, use the external contract and its pinned claim ledger
instead.

- `models/champion_registry.json`
- `data/processed/pipeline_summary.json`
- `models/fairness_audit_status.json`
- `models/threshold_semantics.json`
- `models/governance_status.json`
- `models/model_shift_status.json`
- `models/monotonicity_audit_status.json`
- `models/pd_backtesting_status.json`
- `models/bootstrap_validation_status.json`
- `models/pd_validation_interpretation_status.json`
- `models/calibration_mapping_status.json`
- `models/ifrs9_diagnostics_status.json`
- `models/encoding_stability_status.json`
- `reports/mrm/mrm_validation_report.json`
- `reports/run_comparisons/canonical-monotonic-confirmatory-adsfcr-2026-03-30-1129/comparison.json`

## Directory Contract

- `docs/` root contains active technical and editorial surfaces only.
- `docs/history/` contains archived plans, audits, and historical snapshots retained for provenance.
- `docs/research/` contains literature notes, exploratory comparisons, and research-only reference material that is not part of the live operational contract.

## Historical but Still Useful

| File | Why it remains |
|---|---|
| `docs/history/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md` | Provenance of the earlier paper-grade rerun program |
| `docs/history/PROMOTION_DOSSIER_2026-03-01.md` | Historical promotion snapshot; not live policy state |
| `docs/history/ENGINEERING_ENVS_AND_UPGRADE_PLAN_2026-02-25.md` | Environment migration notes if tooling breaks |
| `docs/history/DEPLOY_STREAMLIT_FREE.md` | Historical showcase deployment only |

## Research-Only References

| File | Why it remains |
|---|---|
| `docs/research/conformal_prediction_research_2026.md` | Deep conformal theory and implementation notes |
| `docs/research/conformal_prediction_quick_reference.md` | Coding patterns and formula crib sheet |
| `docs/research/conformal_libraries_comparison.md` | Library comparison retained for justification and appendix work |
| `docs/research/CALIBRATION_METHOD_SELECTION.md` | Calibration writeup preserved as research/method note |
| `docs/research/portfolio_selector_literature_2020_2026.md` | Literature support for portfolio selection writing |

## Editorial Rule

If Quarto, docs, Streamlit, and runtime artifacts disagree:

1. For external CRPTO/IJDS, trust the pinned external claim ledger and evidence
   manifest; local `champion` artifacts have no authority over it.
2. For local engineering behavior, trust runtime artifacts and executable
   contracts.
3. Trust the current cross-project audit and claim-status CSV/YAML before older
   narrative pages.
4. Treat older markdown snapshots as historical unless they explicitly say
   they are live and pass their current guardrails.
