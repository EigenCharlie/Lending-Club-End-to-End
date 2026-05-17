#!/usr/bin/env python3
"""Build Paper 4 v504 reviewer assignment packet artifacts."""

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
    write_csv,
    write_json,
)

VERSION = 504
PRIOR_ASSIGNMENT_GAP_VERSION = 503
NEXT_ARTIFACT = "paper4_v505_reviewer_eligibility_checklist.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v504_reviewer_assignment_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _reviewer_assignment_packet(gaps: pd.DataFrame) -> pd.DataFrame:
    role_by_domain = {
        "layout_surface": "layout_surface_reviewer",
        "caption_claim_safety": "caption_claim_safety_reviewer",
    }
    rows = []
    for _, row in gaps.iterrows():
        rows.append(
            {
                "assignment_packet_id_v504": row["assignment_gap_id_v503"],
                "priority_v504": int(row["priority_v503"]),
                "review_domain_v504": row["review_domain_v503"],
                "asset_id_v504": row["asset_id_v503"],
                "reviewer_role_required_v504": role_by_domain[row["review_domain_v503"]],
                "reviewer_slot_created_v504": True,
                "reviewer_candidate_prefilled_v504": False,
                "reviewer_assigned_v504": False,
                "assignment_signoff_recorded_v504": False,
                "outcome_capture_allowed_v504": False,
                "patch_allowed_v504": False,
                "claim_boundary_v504": "reviewer assignment packet only",
            }
        )
    return pd.DataFrame(rows)


def _reviewer_requirement_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id_v504": "layout_surface_reviewer",
                "review_domain_v504": "layout_surface",
                "requirement_declared_v504": True,
                "requirement_satisfied_v504": False,
                "required_resolution_v504": "select reviewer for layout surface packet rows",
            },
            {
                "requirement_id_v504": "caption_claim_safety_reviewer",
                "review_domain_v504": "caption_claim_safety",
                "requirement_declared_v504": True,
                "requirement_satisfied_v504": False,
                "required_resolution_v504": "select reviewer for caption claim-safety rows",
            },
            {
                "requirement_id_v504": "claim_boundary_reviewer",
                "review_domain_v504": "all",
                "requirement_declared_v504": True,
                "requirement_satisfied_v504": False,
                "required_resolution_v504": "confirm no outcome creates prohibited claims",
            },
            {
                "requirement_id_v504": "conflict_of_interest_check",
                "review_domain_v504": "all",
                "requirement_declared_v504": True,
                "requirement_satisfied_v504": False,
                "required_resolution_v504": "record conflict check before assignment signoff",
            },
        ]
    )


def _assignment_control_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assignment_control_id_v504": "no_reviewer_candidate_prefill",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": True,
                "control_result_v504": "reviewer candidates remain blank",
            },
            {
                "assignment_control_id_v504": "no_assignment_signoff",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": True,
                "control_result_v504": "assignment signoff remains absent",
            },
            {
                "assignment_control_id_v504": "no_outcome_capture",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": True,
                "control_result_v504": "outcome capture remains blocked",
            },
            {
                "assignment_control_id_v504": "no_caption_finalization",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": True,
                "control_result_v504": "caption finalization remains blocked",
            },
            {
                "assignment_control_id_v504": "no_patch_approval",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": False,
                "control_result_v504": "patch approval remains absent",
            },
            {
                "assignment_control_id_v504": "no_final_promotion",
                "control_active_v504": True,
                "blocks_outcome_capture_v504": False,
                "control_result_v504": "final promotion artifact remains absent",
            },
        ]
    )


def _assignment_next_action_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_action_id_v504": "select_reviewer_candidates",
                "priority_v504": 1,
                "recommended_next_v504": True,
                "blocks_outcome_capture_v504": True,
                "claim_boundary_v504": "candidate selection next only",
            },
            {
                "next_action_id_v504": "perform_conflict_check",
                "priority_v504": 2,
                "recommended_next_v504": True,
                "blocks_outcome_capture_v504": True,
                "claim_boundary_v504": "conflict check next only",
            },
            {
                "next_action_id_v504": "record_assignment_signoff",
                "priority_v504": 3,
                "recommended_next_v504": True,
                "blocks_outcome_capture_v504": True,
                "claim_boundary_v504": "assignment signoff next only",
            },
            {
                "next_action_id_v504": "start_outcome_capture_after_assignment",
                "priority_v504": 4,
                "recommended_next_v504": False,
                "blocks_outcome_capture_v504": False,
                "claim_boundary_v504": "future outcome capture only after assignment",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v504": "reviewer_assignment_packet_created",
                "ready_v504": True,
                "evidence_artifact_v504": "paper4_v504_reviewer_assignment_packet.csv",
                "claim_boundary_v504": "assignment packet only",
            },
            {
                "readiness_gate_v504": "reviewer_requirement_matrix_created",
                "ready_v504": True,
                "evidence_artifact_v504": "paper4_v504_reviewer_requirement_matrix.csv",
                "claim_boundary_v504": "requirement matrix only",
            },
            {
                "readiness_gate_v504": "assignment_control_checklist_created",
                "ready_v504": True,
                "evidence_artifact_v504": "paper4_v504_assignment_control_checklist.csv",
                "claim_boundary_v504": "assignment controls only",
            },
            {
                "readiness_gate_v504": "reviewer_eligibility_checklist_ready",
                "ready_v504": True,
                "evidence_artifact_v504": "paper4_v504_assignment_next_action_queue.csv",
                "claim_boundary_v504": "future eligibility checklist readiness only",
            },
            {
                "readiness_gate_v504": "reviewer_candidates_prefilled",
                "ready_v504": False,
                "evidence_artifact_v504": "reviewer_candidate_prefilled_v504 remains false",
                "claim_boundary_v504": "no reviewer candidates prefilled",
            },
            {
                "readiness_gate_v504": "reviewers_assigned",
                "ready_v504": False,
                "evidence_artifact_v504": "reviewer_assigned_v504 remains false",
                "claim_boundary_v504": "reviewers are not assigned",
            },
            {
                "readiness_gate_v504": "ready_for_quarto_patch",
                "ready_v504": False,
                "evidence_artifact_v504": "assignments, outcomes and approval absent",
                "claim_boundary_v504": "patch remains blocked",
            },
            {
                "readiness_gate_v504": "paper4_final_promotion_created",
                "ready_v504": False,
                "evidence_artifact_v504": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v504": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v504_reviewer_assignment_packet_created",
                "allowed": True,
                "artifact": "paper4_v504_reviewer_assignment_packet.csv",
                "boundary": "reviewer assignment packet only",
            },
            {
                "claim_id": "v504_reviewer_requirements_declared",
                "allowed": True,
                "artifact": "paper4_v504_reviewer_requirement_matrix.csv",
                "boundary": "requirements declaration only",
            },
            {
                "claim_id": "v504_reviewer_eligibility_checklist_ready",
                "allowed": True,
                "artifact": "paper4_v504_assignment_next_action_queue.csv",
                "boundary": "future eligibility checklist readiness only",
            },
            {
                "claim_id": "v504_reviewers_assigned_or_outcomes_captured",
                "allowed": False,
                "artifact": "paper4_v504_reviewer_assignment_packet.csv",
                "boundary": "no reviewers assigned or outcomes captured",
            },
            {
                "claim_id": "v504_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v504_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v504_final_promotion",
                "allowed": False,
                "artifact": "paper4_v504_manuscript_readiness_delta.csv",
                "boundary": "no final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v504 creates reviewer assignment slots for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_reviewer_assignment_packet.csv"
                ),
                "boundary": "Reviewer assignment packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v504 declares reviewer requirements and assignment controls.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_reviewer_requirement_matrix.csv"
                ),
                "boundary": "Reviewer requirements and controls only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v504 makes reviewer eligibility checklist executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_assignment_next_action_queue.csv"
                ),
                "boundary": "Future eligibility checklist readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v504 assigns reviewers or captures review outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_reviewer_assignment_packet.csv"
                ),
                "boundary": "Reviewers and outcomes remain absent.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v504 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v504 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v504_manuscript_readiness_delta.csv"
                ),
                "boundary": (
                    "No final promotion artifact, champion replacement or deployment gate "
                    "is created."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Manuscript",
                "executable_item": "v504 creates reviewer assignment packet.",
                "status": "reviewer_assignment_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v505 creates reviewer eligibility checklist",
                "last_wave": "v504",
                "execution_result": "reviewer_assignment_packet_created_without_assignments",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v504")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Reviewer Assignment Packet v504

Generated: {status["generated_at_utc"]}

## Result

v504 creates reviewer assignment slots for all 14 open assignment gaps and
declares reviewer requirements, assignment controls and next actions. It does
not prefill reviewer candidates, assign reviewers, record signoff or allow
outcome capture.

## Counts

- Assignment packet rows: `{status["assignment_packet_rows_v504"]}`.
- Reviewer slot rows: `{status["reviewer_slot_rows_v504"]}`.
- Reviewer candidate prefilled rows: `{status["reviewer_candidate_prefilled_rows_v504"]}`.
- Reviewer assigned rows: `{status["reviewer_assigned_rows_v504"]}`.
- Assignment signoff rows: `{status["assignment_signoff_recorded_rows_v504"]}`.
- Requirement rows: `{status["requirement_rows_v504"]}`.
- Requirement satisfied rows: `{status["requirement_satisfied_rows_v504"]}`.
- Assignment control rows: `{status["assignment_control_rows_v504"]}`.
- Active assignment control rows: `{status["active_assignment_control_rows_v504"]}`.
- Recommended next action rows: `{status["recommended_next_action_rows_v504"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v504"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v504"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v504"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v504 is a reviewer assignment packet only. It does not assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V504_REVIEWER_ASSIGNMENT_PACKET_START -->"
    end = "<!-- V504_REVIEWER_ASSIGNMENT_PACKET_END -->"
    block = f"""
{start}

## Wave v504: Reviewer Assignment Packet

Generated: {status["generated_at_utc"]}

### Objective

v504 turns the v503 open assignment gaps into reviewer assignment slots and
declares reviewer requirements without assigning reviewers.

### Results

- Assignment packet rows:
  `{status["assignment_packet_rows_v504"]}`.
- Reviewer slot rows:
  `{status["reviewer_slot_rows_v504"]}`.
- Reviewer candidate prefilled rows:
  `{status["reviewer_candidate_prefilled_rows_v504"]}`.
- Reviewer assigned rows:
  `{status["reviewer_assigned_rows_v504"]}`.
- Assignment signoff rows:
  `{status["assignment_signoff_recorded_rows_v504"]}`.
- Requirement rows:
  `{status["requirement_rows_v504"]}`.
- Requirement satisfied rows:
  `{status["requirement_satisfied_rows_v504"]}`.
- Assignment control rows:
  `{status["assignment_control_rows_v504"]}`.
- Active assignment control rows:
  `{status["active_assignment_control_rows_v504"]}`.
- Recommended next action rows:
  `{status["recommended_next_action_rows_v504"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v504"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v504"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v504"]}`.
- Book sources modified:
  `{status["book_sources_modified_v504"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v504"]}`.

### Interpretation

The reviewer assignment surface is now explicit, but every reviewer candidate,
assignment and signoff field remains empty. The next executable artifact should
check eligibility before any assignment is recorded.

### Claim Impact

- Allowed: reviewer assignment packet, requirements declaration, assignment
  controls and reviewer eligibility checklist readiness.
- Still prohibited: reviewer assignment, completed review/signoff claims, final
  captions, Quarto patch readiness/application, Quarto/book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v504 in the living notebook. v505 should create a reviewer eligibility
checklist without assigning reviewers or pre-recording outcomes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v503 = _read_status(PRIOR_ASSIGNMENT_GAP_VERSION)
    if v503["next_artifact_v503"] != "paper4_v504_reviewer_assignment_packet.md":
        raise RuntimeError("v504 expects v503 to route to reviewer assignment packet.")
    if not v503["reviewer_assignment_packet_ready_v503"]:
        raise RuntimeError("v504 requires v503 reviewer assignment packet readiness.")

    gaps = pd.read_csv(TABLE_DIR / "paper4_v503_assignment_gap_audit.csv")
    packet = _reviewer_assignment_packet(gaps)
    requirements = _reviewer_requirement_matrix()
    controls = _assignment_control_checklist()
    next_actions = _assignment_next_action_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v504_reviewer_assignment_packet.csv", packet)
    write_csv(TABLE_DIR / "paper4_v504_reviewer_requirement_matrix.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v504_assignment_control_checklist.csv", controls)
    write_csv(TABLE_DIR / "paper4_v504_assignment_next_action_queue.csv", next_actions)
    write_csv(TABLE_DIR / "paper4_v504_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v504_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v504_reviewer_assignment_packet",
        "schema_version": "2026-05-17.504",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_assignment_gap_version_v504": PRIOR_ASSIGNMENT_GAP_VERSION,
        "reviewer_assignment_packet_created_v504": True,
        "assignment_packet_rows_v504": len(packet),
        "reviewer_slot_rows_v504": int(
            packet["reviewer_slot_created_v504"].astype(bool).sum()
        ),
        "reviewer_candidate_prefilled_rows_v504": int(
            packet["reviewer_candidate_prefilled_v504"].astype(bool).sum()
        ),
        "reviewer_assigned_rows_v504": int(
            packet["reviewer_assigned_v504"].astype(bool).sum()
        ),
        "assignment_signoff_recorded_rows_v504": int(
            packet["assignment_signoff_recorded_v504"].astype(bool).sum()
        ),
        "requirement_rows_v504": len(requirements),
        "requirement_declared_rows_v504": int(
            requirements["requirement_declared_v504"].astype(bool).sum()
        ),
        "requirement_satisfied_rows_v504": int(
            requirements["requirement_satisfied_v504"].astype(bool).sum()
        ),
        "assignment_control_rows_v504": len(controls),
        "active_assignment_control_rows_v504": int(
            controls["control_active_v504"].astype(bool).sum()
        ),
        "recommended_next_action_rows_v504": int(
            next_actions["recommended_next_v504"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v504": int(
            packet["outcome_capture_allowed_v504"].astype(bool).sum()
        ),
        "patch_allowed_rows_v504": int(packet["patch_allowed_v504"].astype(bool).sum()),
        "readiness_delta_rows_v504": len(readiness),
        "reviewer_eligibility_checklist_ready_v504": True,
        "ready_for_quarto_patch_v504": False,
        "quarto_patch_applied_v504": False,
        "book_sources_modified_v504": False,
        "book_references_modified_v504": False,
        "submission_ready_claim_allowed_v504": False,
        "working_champion_claim_allowed_v504": False,
        "paper1_promotion_allowed_v504": False,
        "paper4_working_champion_changed_v504": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v504": NEXT_ARTIFACT,
        "claim_boundary": (
            "v504 creates reviewer assignment slots only; candidates, assignments, "
            "outcomes, captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v504 must not create final Paper 4 promotion.")
    if status["reviewer_candidate_prefilled_rows_v504"] != 0:
        raise RuntimeError("v504 must not prefill reviewer candidates.")
    if status["reviewer_assigned_rows_v504"] != 0:
        raise RuntimeError("v504 must not assign reviewers.")
    if status["outcome_capture_allowed_rows_v504"] != 0:
        raise RuntimeError("v504 must not allow outcome capture.")
    if status["patch_allowed_rows_v504"] != 0:
        raise RuntimeError("v504 must not approve a Quarto patch.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v504": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
