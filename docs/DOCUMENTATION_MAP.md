# Documentation Map

Quick reference for the current documentation stack after the pipeline-first refactor, the monotonic promotion work, and the ADSFCR-inspired documentation refresh.

## Keep Closest To Hand

| Category | File | Purpose |
|---|---|---|
| **Canonical editorial ledger** | `docs/CANONICAL_DOCUMENTATION_AND_QUARTO_TRACEABILITY_2026-03-30.md` | Master map between live techniques, artifacts, Quarto chapters, references, and legacy claims to retire |
| **Current state** | `SESSION_STATE.md` | Operational snapshot and runtime-facing source list |
| **MRM / governance** | `docs/MODEL_RISK_MANAGEMENT.md` | SR 11-7 style governance narrative and control framing |
| **ADSFCR adoption** | `docs/ADSFCR_AUDIT_AND_MONOTONIC_CHALLENGER_PLAN_2026-03-29.md` | Detailed audit of the external repo, adoption decisions, and tranche-by-tranche implementation status |
| **ADSFCR next work** | `docs/ADSFCR_EXECUTABLE_BACKLOG_2026-03-30.md` | Execution-oriented backlog for the remaining ADSFCR items that still look worth implementing |
| **Quarto contract** | `docs/QUARTO_BOOK_BLUEPRINT.md` | Book architecture, editorial contract, and maintenance rules |
| **Project rationale** | `docs/PROJECT_JUSTIFICATION.md` | Methodological and architectural why |
| **Runbook** | `docs/RUNBOOK.md` | Reproducibility playbook |
| **History / learnings** | `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Historical decisions, fixes, and practical learnings |
| **Paper references** | `docs/PAPER_REFERENCES_STATE_OF_ART.md` | Curated literature map for papers and thesis chapters |
| **Backlog** | `docs/backlog-papers-unified.md` | Unified backlog for papers, experiments, and documentation follow-ups |
| **Conformal note** | `docs/conformal_prediction_README.md` | Compact operational/research entrypoint for conformal material |

## Runtime Sources Of Truth

Use these before trusting any prose:

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

1. Trust runtime artifacts first.
2. Trust the canonical traceability doc second.
3. Treat older markdown snapshots as historical unless they explicitly say they are live.
