#!/usr/bin/env python3
"""Build Paper 4 v380 manuscript-section scaffold artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 380
PRIOR_WORK_ORDER_VERSION = 379
NEXT_VERSION = 381
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_verified_literature_source_log.csv"
SCAFFOLD = NOTEBOOK.parent / "paper4_v380_manuscript_section_scaffold.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v380 scaffolds Paper 4 manuscript sections from bounded evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v380_manuscript_section_scaffold.md"
                ),
                "boundary": "Scaffold only; not a full manuscript or Quarto promotion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v380 maps open manuscript TODOs to future evidence-closure tasks.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v380_manuscript_todo_register.csv"
                ),
                "boundary": "TODO register only; gaps remain open.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v380 creates a submission-ready Paper 4 manuscript.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v380_claim_blockers.csv"
                ),
                "boundary": "Scaffold still has open TODOs and no verified literature log.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v380 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v380_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Citations/Related Work",
                "executable_item": (
                    "v380 scaffolds manuscript sections from bounded evidence and routes "
                    "the next wave to verified literature/source logging."
                ),
                "status": "manuscript_section_scaffold_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v381 verifies external sources before any related-work or citation claim"
                ),
                "last_wave": "v380",
                "execution_result": "manuscript_scaffold_created_with_open_todos_and_claim_controls",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v380")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _scaffold_markdown(status: dict[str, Any], sections: pd.DataFrame) -> str:
    section_text = "\n\n".join(
        (
            f"## {row['paper_section_v380']}\n\n"
            f"**Purpose.** {row['section_purpose_v380']}\n\n"
            f"**Allowed draft text.** {row['bounded_draft_text_v380']}\n\n"
            f"**Still open.** {row['open_todo_v380']}"
        )
        for _, row in sections.iterrows()
    )
    return f"""# Paper 4 Manuscript Section Scaffold v380

Generated: {status["generated_at_utc"]}

This is a scaffold, not a submission manuscript. It keeps the future Paper 4
text aligned with v374-v379 evidence and keeps all prohibited claims blocked.

{section_text}

## Required Caveat

The scaffold may describe bounded living-lab and offline/proxy evidence. It
must not claim submission readiness, strict live deployment, contractual IFRS9,
legal fairness compliance, full-v55 global optimality, a new working champion,
Paper Estrella replacement or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v380"]}` before adding external related-work or
bibliography claims.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V380_MANUSCRIPT_SECTION_SCAFFOLD_START -->"
    end = "<!-- V380_MANUSCRIPT_SECTION_SCAFFOLD_END -->"
    block = f"""
{start}

## Wave v380: Manuscript Section Scaffold

Generated: {status["generated_at_utc"]}

### Objective

v380 executes the first v379 work order by drafting a bounded manuscript
scaffold from existing evidence, while keeping Quarto promotion and submission
readiness blocked.

### Results

- Section scaffold rows:
  `{status["section_scaffold_rows_v380"]}`.
- Manuscript TODO rows:
  `{status["manuscript_todo_rows_v380"]}`.
- Claim-control rows:
  `{status["claim_control_rows_v380"]}`.
- Open TODO rows:
  `{status["open_todo_rows_v380"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v380"]}`.
- Quarto promotion allowed:
  `{status["quarto_promotion_allowed_v380"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v380"]}`.

### Interpretation

Paper 4 now has a manuscript shell that is easier to edit, audit and cite. The
next weak point is external related-work verification, not solver promotion or
live deployment.

### Claim Impact

- Allowed: bounded manuscript scaffold and TODO mapping.
- Still prohibited: submission-ready, Quarto promotion, live/legal/global
  claims, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v380 in the living notebook. v381 should create a verified literature
source log before related-work language is expanded.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v379_status = json.loads((STATUS_DIR / "paper4_v379_status.json").read_text(encoding="utf-8"))
    if v379_status["next_artifact_v379"] != "paper4_v380_manuscript_section_scaffold.md":
        raise RuntimeError("v380 expects v379 to route to the manuscript section scaffold.")
    work_order = read_csv("paper4_v379_evidence_gap_closure_work_order.csv")
    if work_order.empty:
        raise RuntimeError("Missing v379 evidence-gap closure work order.")

    sections = pd.DataFrame(
        [
            {
                f"paper_section_v{VERSION}": "Abstract",
                f"section_purpose_v{VERSION}": "Frame Paper 4 as a bounded living-lab audit.",
                f"bounded_draft_text_v{VERSION}": (
                    "Paper 4 studies a reproducible living-lab protocol around the protected "
                    "Paper Estrella champion, using bounded offline/proxy evidence and explicit "
                    "claim controls."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v374_paper4_claim_language_section_draft.md;"
                    "paper4_v376_publication_integration_patch.md"
                ),
                f"open_todo_v{VERSION}": "shorten after venue target and verified citations exist",
                f"claim_boundary_v{VERSION}": "framing only",
            },
            {
                f"paper_section_v{VERSION}": "Introduction",
                f"section_purpose_v{VERSION}": "Explain why bounded auditability matters.",
                f"bounded_draft_text_v{VERSION}": (
                    "The contribution is not a replacement champion; it is an auditable "
                    "protocol for distinguishing usable offline evidence from blocked live, "
                    "legal, global and final-promotion claims."
                ),
                f"source_artifacts_v{VERSION}": "paper4_v378_submission_readiness_gap_register.csv",
                f"open_todo_v{VERSION}": "add verified related-work source log from v381",
                f"claim_boundary_v{VERSION}": "motivation only",
            },
            {
                f"paper_section_v{VERSION}": "Methods: Living-Lab Governance",
                f"section_purpose_v{VERSION}": "Describe the claim boundary mechanism.",
                f"bounded_draft_text_v{VERSION}": (
                    "Each wave emits status JSON, citable artifacts, claim blockers and guardrails "
                    "before any wording is promoted."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v377_reproducibility_bundle_manifest.csv;"
                    "paper4_current_claim_boundaries.csv"
                ),
                f"open_todo_v{VERSION}": "convert process into a compact figure after v381",
                f"claim_boundary_v{VERSION}": "governance method only",
            },
            {
                f"paper_section_v{VERSION}": "Results: Solver Frontier",
                f"section_purpose_v{VERSION}": "Report bounded solver evidence without global proof.",
                f"bounded_draft_text_v{VERSION}": (
                    "The v361/v363/v373 evidence supports bounded no-entry and gap statements; "
                    "full-v55 global optimality remains blocked."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v361_v353_fourth_order_or_full_dual_bound.csv;"
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv;"
                    "paper4_v373_sampled_chunk_source_screen.csv"
                ),
                f"open_todo_v{VERSION}": "decide whether to add v382 global scope decision",
                f"claim_boundary_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"paper_section_v{VERSION}": "Results: Source Governance",
                f"section_purpose_v{VERSION}": "Explain why blind chunking stopped.",
                f"bounded_draft_text_v{VERSION}": (
                    "The v371-v373 diagnostics identify grade/source constraints and zero "
                    "sampled source-exact rows as the active blocker."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v371_source_governance_blocker_diagnostic.csv;"
                    "paper4_v372_grade_a_source_relief_prefilter.csv;"
                    "paper4_v373_full_v55_chunk_002_or_stop_rule.csv"
                ),
                f"open_todo_v{VERSION}": "decide whether to add v383 targeted source audit",
                f"claim_boundary_v{VERSION}": "diagnostic only",
            },
            {
                f"paper_section_v{VERSION}": "Limitations",
                f"section_purpose_v{VERSION}": "State what cannot be claimed.",
                f"bounded_draft_text_v{VERSION}": (
                    "The current evidence does not authorize submission-ready, live deployment, "
                    "contractual/legal, global optimality, new champion or final-promotion claims."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v375_claim_permission_register.csv;"
                    "paper4_v378_submission_readiness_gap_register.csv"
                ),
                f"open_todo_v{VERSION}": "preserve v375/v378 blocker wording in final draft",
                f"claim_boundary_v{VERSION}": "limitations only",
            },
            {
                f"paper_section_v{VERSION}": "Appendix: Reproducibility",
                f"section_purpose_v{VERSION}": "Point to the artifact bundle and guardrails.",
                f"bounded_draft_text_v{VERSION}": (
                    "The appendix should cite the v377 bundle manifest, status files and guardrail "
                    "tests as reproducibility evidence."
                ),
                f"source_artifacts_v{VERSION}": (
                    "paper4_v377_reproducibility_bundle_manifest.csv;"
                    "paper4_v377_guardrail_manifest.csv"
                ),
                f"open_todo_v{VERSION}": "update hashes after future waves are committed",
                f"claim_boundary_v{VERSION}": "appendix provenance only",
            },
        ]
    )
    todo_register = work_order.copy()
    todo_register[f"todo_status_v{VERSION}"] = "open"
    todo_register[f"target_manuscript_section_v{VERSION}"] = [
        "Abstract/Introduction",
        "Methods",
        "Related Work",
        "Results: Solver Frontier",
        "Results: Source Governance",
        "Limitations",
        "Limitations",
        "Limitations",
        "Limitations",
        "Methods Appendix",
        "Appendix",
    ]
    todo_register[f"claim_boundary_v{VERSION}"] = todo_register[
        "claim_boundary_v379"
    ]
    claim_controls = pd.DataFrame(
        [
            {
                f"control_id_v{VERSION}": "bounded_living_lab_language",
                f"allowed_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v380_section_scaffold.csv",
                f"claim_boundary_v{VERSION}": "scaffold wording only",
            },
            {
                f"control_id_v{VERSION}": "offline_proxy_language",
                f"allowed_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"claim_boundary_v{VERSION}": "proxy-only statement",
            },
            {
                f"control_id_v{VERSION}": "submission_ready_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v378_claim_blockers.csv",
                f"claim_boundary_v{VERSION}": "open gaps remain",
            },
            {
                f"control_id_v{VERSION}": "strict_live_deployment_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"claim_boundary_v{VERSION}": "live gates unmet",
            },
            {
                f"control_id_v{VERSION}": "contractual_or_legal_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"claim_boundary_v{VERSION}": "legal/contractual gates unmet",
            },
            {
                f"control_id_v{VERSION}": "global_optimality_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
                f"claim_boundary_v{VERSION}": "full-v55 certificate missing",
            },
            {
                f"control_id_v{VERSION}": "working_champion_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_current_claim_boundaries.csv",
                f"claim_boundary_v{VERSION}": "Paper Estrella remains protected",
            },
            {
                f"control_id_v{VERSION}": "final_promotion_language",
                f"allowed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "final promotion forbidden",
            },
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "scaffold_is_not_full_manuscript",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(sections)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "section scaffold only",
            },
            {
                f"blocker_id_v{VERSION}": "open_todos_remain",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(todo_register)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "future waves must close TODOs",
            },
            {
                f"blocker_id_v{VERSION}": "verified_literature_log_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "related-work claims blocked",
            },
            {
                f"blocker_id_v{VERSION}": "quarto_pages_not_promoted",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "living notebook only",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v380_manuscript_section_scaffold_created",
                "allowed": True,
                "artifact": "paper4_v380_manuscript_section_scaffold.md",
                "boundary": "scaffold only",
            },
            {
                "claim_id": "v380_manuscript_todo_register_created",
                "allowed": True,
                "artifact": "paper4_v380_manuscript_todo_register.csv",
                "boundary": "TODO mapping only",
            },
            {
                "claim_id": "v380_submission_ready_manuscript",
                "allowed": False,
                "artifact": "paper4_v380_claim_blockers.csv",
                "boundary": "open TODOs and missing literature log",
            },
            {
                "claim_id": "v380_quarto_promotion_or_live_legal_global_claim",
                "allowed": False,
                "artifact": "paper4_v380_claim_control_checklist.csv",
                "boundary": "claim controls block stronger claims",
            },
            {
                "claim_id": "v380_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v380_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v380_section_scaffold.csv", sections)
    write_csv(TABLE_DIR / "paper4_v380_manuscript_todo_register.csv", todo_register)
    write_csv(TABLE_DIR / "paper4_v380_claim_control_checklist.csv", claim_controls)
    write_csv(TABLE_DIR / "paper4_v380_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v380_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v380_manuscript_section_scaffold",
        "schema_version": "2026-05-17.380",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_work_order_version_v380": PRIOR_WORK_ORDER_VERSION,
        "prior_v379_work_order_rows_v380": int(v379_status["work_order_rows_v379"]),
        "prior_v379_executable_now_rows_v380": int(v379_status["executable_now_rows_v379"]),
        "section_scaffold_rows_v380": int(len(sections)),
        "manuscript_todo_rows_v380": int(len(todo_register)),
        "claim_control_rows_v380": int(len(claim_controls)),
        "open_todo_rows_v380": int(todo_register[f"todo_status_v{VERSION}"].eq("open").sum()),
        "allowed_claim_control_rows_v380": int(claim_controls[f"allowed_v{VERSION}"].astype(bool).sum()),
        "blocked_claim_control_rows_v380": int(
            (~claim_controls[f"allowed_v{VERSION}"].astype(bool)).sum()
        ),
        "claim_blocker_rows_v380": int(len(blockers)),
        "claim_matrix_rows_v380": int(len(claim_matrix)),
        "submission_ready_claim_allowed_v380": False,
        "quarto_promotion_allowed_v380": False,
        "bounded_living_lab_language_allowed_v380": True,
        "offline_proxy_language_allowed_v380": True,
        "strict_live_deployment_language_allowed_v380": False,
        "contractual_or_legal_language_allowed_v380": False,
        "global_optimality_language_allowed_v380": False,
        "working_champion_claim_allowed_v380": False,
        "paper1_promotion_allowed_v380": False,
        "paper4_working_champion_changed_v380": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "scaffold_artifact_v380": (
            "reports/paper_material/paper4/notes/"
            "paper4_v380_manuscript_section_scaffold.md"
        ),
        "next_artifact_v380": NEXT_ARTIFACT,
        "claim_boundary": (
            "v380 creates a bounded manuscript scaffold; submission-ready, Quarto, "
            "live/legal/global/final claims remain blocked"
        ),
    }
    SCAFFOLD.write_text(_scaffold_markdown(status, sections), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v380_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v380": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
