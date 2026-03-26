# Documentation Retention Audit
Date: 2026-03-25

This audit records which markdown documents should remain active, which should be treated as historical, and which were safe to remove after the Quarto-first / reduced-Streamlit transition.

## Removed

| File | Decision | Reason |
|---|---|---|
| `docs/backlog-13-03.md` | delete | Already marked deprecated and fully superseded by `docs/backlog-papers-unified.md`. |
| `docs/THESIS_SHOWCASE_PLAN_ES.md` | delete | Obsolete Streamlit-first plan that contradicts the current Quarto-first architecture. |
| `docs/STREAMLIT_STORYTELLING_GUIDE.md` | delete | Legacy UX/editorial guide for the removed 31-page Streamlit surface; no longer needed after the reduced companion redesign. |

## Keep as Historical Reference

| File | Decision | Reason |
|---|---|---|
| `docs/OFFICIAL_RERUN_MASTER_PLAN_2026-02-27.md` | keep-historical | Contains provenance of the official rerun and already carries a historical banner. |
| `docs/PROMOTION_DOSSIER_2026-03-01.md` | keep-historical | Historical promotion snapshot, still useful for audit trail. |
| `docs/DEPLOY_STREAMLIT_FREE.md` | keep-historical | Useful only if someone ever needs to rebuild the frozen public showcase. |
| `docs/ENGINEERING_ENVS_AND_UPGRADE_PLAN_2026-02-25.md` | keep-historical | Old environment migration notes that may still help if infra drifts. |

## Keep Active

| File | Decision | Reason |
|---|---|---|
| `docs/PROJECT_JUSTIFICATION.md` | keep-active | Current official design rationale. |
| `docs/RUNBOOK.md` | keep-active | Operational reproducibility guide. |
| `docs/backlog-papers-unified.md` | keep-active | Live backlog across Quarto, papers, and pipeline. |
| `docs/QUARTO_BOOK_BLUEPRINT.md` | keep-active | Editorial contract for the official source of truth. |
| `docs/MODEL_RISK_MANAGEMENT.md` | keep-active | Governance/MRM companion document. |
| `docs/DOCUMENTATION_MAP.md` | keep-active | Current navigation map for docs. |
| `docs/STREAMLIT_QUARTO_MIGRATION_REGISTRY.yml` | keep-active | Operational record of what moved to Quarto vs what stays in Streamlit. |

## Heuristic Going Forward

- If a markdown file describes the current architecture, runbook, backlog, or official rationale, keep it active.
- If a markdown file documents a completed plan or frozen snapshot, keep it only if it adds audit value and mark it historical.
- If a markdown file is fully superseded by another living document and adds no audit value, delete it.
- Quarto chapters should carry official content; markdown docs outside the book should avoid duplicating official narrative unless they serve operations or history.
