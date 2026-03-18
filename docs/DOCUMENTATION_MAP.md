# Documentation Map

Quick reference for navigating the project's documentation.

| Category | File | Purpose |
|---|---|---|
| **Official Decisions** | `CLAUDE.md` | Coding standards, tech stack, architecture decisions |
| **Official Decisions** | `docs/PROJECT_JUSTIFICATION.md` | Design rationale for methods and components |
| **Current State** | `SESSION_STATE.md` | Authoritative runtime snapshot and metrics |
| **History** | `docs/DECISION_CHANGES_AND_LEARNINGS.md` | Decision log, errors found, practical learnings |
| **Backlog** | `docs/backlog-papers-unified.md` | Master backlog: pipeline + papers + Quarto |
| **Runbook** | `docs/RUNBOOK.md` | Step-by-step reproducibility guide |
| **MRM** | `docs/MODEL_RISK_MANAGEMENT.md` | SR 11-7 model governance documentation |
| **Papers** | `docs/PAPER_REFERENCES_STATE_OF_ART.md` | ~80 curated papers with direct links |
| **Quarto** | `docs/QUARTO_BOOK_BLUEPRINT.md` | 16-chapter Quarto book structure |
| **Reference** | Everything else in `docs/` | Technical references, research notes, historical logs |

**Runtime artifacts** (source of truth for metrics): `data/processed/model_comparison.json`, `models/conformal_policy_status.json`, `data/processed/pipeline_summary.json`, and other JSON/parquet files listed in `SESSION_STATE.md` section 4.
