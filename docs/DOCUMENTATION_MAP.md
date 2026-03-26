# Documentation Map

Quick reference for navigating the project's documentation after the Quarto-first cleanup.

## Keep Close

| Category | File | Purpose |
|---|---|---|
| **Official Decisions** | `CLAUDE.md` | Coding standards, architecture decisions, operating rules |
| **Official Decisions** | `docs/PROJECT_JUSTIFICATION.md` | Design rationale for methods and components |
| **Current State** | `SESSION_STATE.md` | Authoritative current snapshot and serving stance |
| **History** | `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Decision log, errors found, practical learnings |
| **Backlog** | `docs/backlog-papers-unified.md` | Master backlog: pipeline + papers + Quarto |
| **Runbook** | `docs/RUNBOOK.md` | Step-by-step reproducibility guide |
| **MRM** | `docs/MODEL_RISK_MANAGEMENT.md` | SR 11-7 model governance documentation |
| **Papers** | `docs/PAPER_REFERENCES_STATE_OF_ART.md` | Curated papers with direct links |
| **Quarto** | `docs/QUARTO_BOOK_BLUEPRINT.md` | Quarto-first editorial contract |
| **Doc Hygiene** | `docs/DOC_RETENTION_AUDIT_2026-03-25.md` | Keep/archive/delete decisions for markdown docs |

## Historical but Still Useful

| File | Why it remains |
|---|---|
| `docs/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md` | Executed historical plan with provenance of the paper-grade rerun |
| `docs/PROMOTION_DOSSIER_2026-03-01.md` | Historical promotion snapshot explicitly marked as non-live |
| `docs/ENGINEERING_ENVS_AND_UPGRADE_PLAN_2026-02-25.md` | Environment migration notes that may still help if tooling breaks |
| `docs/DEPLOY_STREAMLIT_FREE.md` | Frozen public showcase deployment guide; no longer core, still useful if rebuilding the historical showcase |

## Removed in Cleanup

| File | Why removed |
|---|---|
| `docs/backlog-13-03.md` | Explicitly deprecated and superseded by `docs/backlog-papers-unified.md` |
| `docs/THESIS_SHOWCASE_PLAN_ES.md` | Obsolete Streamlit-first showcase plan superseded by Quarto-first architecture |
| `docs/STREAMLIT_STORYTELLING_GUIDE.md` | Legacy 31-page Streamlit editorial guide superseded by the reduced companion + Quarto-first contract |

**Runtime artifacts** (source of truth for metrics): `data/processed/model_comparison.json`, `models/conformal_policy_status.json`, `data/processed/pipeline_summary.json`, and other JSON/parquet files listed in `SESSION_STATE.md`.
