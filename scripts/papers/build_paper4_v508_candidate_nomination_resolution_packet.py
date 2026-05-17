#!/usr/bin/env python3
"""Build Paper 4 v508 candidate nomination resolution packet artifacts."""

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

VERSION = 508
PRIOR_NOMINATION_GAP_VERSION = 507
NEXT_ARTIFACT = "paper4_v509_candidate_resolution_gap_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v508_candidate_nomination_resolution_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_nomination_resolution_packet(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in gaps.iterrows():
        rows.append(
            {
                "resolution_packet_id_v508": row["candidate_nomination_gap_id_v507"],
                "priority_v508": int(row["priority_v507"]),
                "review_domain_v508": row["review_domain_v507"],
                "reviewer_role_required_v508": row["reviewer_role_required_v507"],
                "resolution_packet_ready_v508": True,
                "candidate_identifier_resolved_v508": False,
                "nomination_fields_completed_v508": False,
                "candidate_nomination_recorded_v508": False,
                "eligibility_review_allowed_v508": False,
                "reviewer_assignment_allowed_v508": False,
                "outcome_capture_allowed_v508": False,
                "patch_allowed_v508": False,
                "required_resolution_action_v508": (
                    "collect_candidate_identifier_and_nomination_fields"
                ),
                "claim_boundary_v508": "candidate nomination resolution packet only",
            }
        )
    return pd.DataFrame(rows)


def _resolution_field_completion_matrix(fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fields.iterrows():
        rows.append(
            {
                "resolution_packet_id_v508": row["candidate_nomination_packet_id_v506"],
                "nomination_field_v508": row["nomination_field_v506"],
                "field_resolution_required_v508": bool(row["field_required_v506"]),
                "field_prefilled_v508": bool(row["field_prefilled_v506"]),
                "field_completed_v508": False,
                "human_entry_required_v508": True,
                "claim_boundary_v508": "field completion matrix only",
            }
        )
    return pd.DataFrame(rows)


def _resolution_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resolution_control_id_v508": "no_candidate_identifier_resolution",
                "control_active_v508": True,
                "blocks_nomination_v508": True,
                "control_result_v508": "candidate identifiers remain unresolved",
            },
            {
                "resolution_control_id_v508": "no_nomination_field_completion",
                "control_active_v508": True,
                "blocks_nomination_v508": True,
                "control_result_v508": "nomination fields remain incomplete",
            },
            {
                "resolution_control_id_v508": "no_candidate_nomination_recorded",
                "control_active_v508": True,
                "blocks_nomination_v508": True,
                "control_result_v508": "candidate nominations remain absent",
            },
            {
                "resolution_control_id_v508": "no_eligibility_review_started",
                "control_active_v508": True,
                "blocks_nomination_v508": False,
                "control_result_v508": "eligibility review remains blocked",
            },
            {
                "resolution_control_id_v508": "no_reviewer_assignment",
                "control_active_v508": True,
                "blocks_nomination_v508": False,
                "control_result_v508": "reviewer assignment remains absent",
            },
            {
                "resolution_control_id_v508": "no_final_promotion",
                "control_active_v508": True,
                "blocks_nomination_v508": False,
                "control_result_v508": "final promotion artifact remains absent",
            },
        ]
    )


def _resolution_next_action_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_action_id_v508": "collect_candidate_identifiers",
                "priority_v508": 1,
                "recommended_next_v508": True,
                "blocks_nomination_v508": True,
                "claim_boundary_v508": "candidate identifier collection next only",
            },
            {
                "next_action_id_v508": "complete_nomination_fields",
                "priority_v508": 2,
                "recommended_next_v508": True,
                "blocks_nomination_v508": True,
                "claim_boundary_v508": "nomination field completion next only",
            },
            {
                "next_action_id_v508": "record_nomination_signoff",
                "priority_v508": 3,
                "recommended_next_v508": True,
                "blocks_nomination_v508": True,
                "claim_boundary_v508": "nomination signoff next only",
            },
            {
                "next_action_id_v508": "audit_resolution_after_manual_entry",
                "priority_v508": 4,
                "recommended_next_v508": False,
                "blocks_nomination_v508": False,
                "claim_boundary_v508": "future audit only after manual entry",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v508": "candidate_nomination_resolution_packet_created",
                "ready_v508": True,
                "evidence_artifact_v508": (
                    "paper4_v508_candidate_nomination_resolution_packet.csv"
                ),
                "claim_boundary_v508": "resolution packet only",
            },
            {
                "readiness_gate_v508": "resolution_field_completion_matrix_created",
                "ready_v508": True,
                "evidence_artifact_v508": (
                    "paper4_v508_resolution_field_completion_matrix.csv"
                ),
                "claim_boundary_v508": "field completion matrix only",
            },
            {
                "readiness_gate_v508": "resolution_control_register_created",
                "ready_v508": True,
                "evidence_artifact_v508": "paper4_v508_resolution_control_register.csv",
                "claim_boundary_v508": "resolution controls only",
            },
            {
                "readiness_gate_v508": "candidate_resolution_gap_audit_ready",
                "ready_v508": True,
                "evidence_artifact_v508": "paper4_v508_resolution_next_action_queue.csv",
                "claim_boundary_v508": "future resolution gap audit readiness only",
            },
            {
                "readiness_gate_v508": "candidate_identifiers_resolved",
                "ready_v508": False,
                "evidence_artifact_v508": "candidate identifiers remain unresolved",
                "claim_boundary_v508": "no candidate identifiers resolved",
            },
            {
                "readiness_gate_v508": "candidate_nominations_recorded",
                "ready_v508": False,
                "evidence_artifact_v508": "candidate nominations remain absent",
                "claim_boundary_v508": "no candidates nominated",
            },
            {
                "readiness_gate_v508": "ready_for_quarto_patch",
                "ready_v508": False,
                "evidence_artifact_v508": "candidates, assignments and outcomes absent",
                "claim_boundary_v508": "patch remains blocked",
            },
            {
                "readiness_gate_v508": "paper4_final_promotion_created",
                "ready_v508": False,
                "evidence_artifact_v508": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v508": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v508_candidate_nomination_resolution_packet_created",
                "allowed": True,
                "artifact": "paper4_v508_candidate_nomination_resolution_packet.csv",
                "boundary": "resolution packet only",
            },
            {
                "claim_id": "v508_resolution_fields_and_controls_declared",
                "allowed": True,
                "artifact": "paper4_v508_resolution_field_and_control_artifacts",
                "boundary": "field and control declaration only",
            },
            {
                "claim_id": "v508_candidate_resolution_gap_audit_ready",
                "allowed": True,
                "artifact": "paper4_v508_resolution_next_action_queue.csv",
                "boundary": "future resolution gap audit readiness only",
            },
            {
                "claim_id": "v508_candidates_resolved_or_nominated",
                "allowed": False,
                "artifact": "paper4_v508_candidate_nomination_resolution_packet.csv",
                "boundary": "no candidates resolved or nominated",
            },
            {
                "claim_id": "v508_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v508_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v508_final_promotion",
                "allowed": False,
                "artifact": "paper4_v508_manuscript_readiness_delta.csv",
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
                "claim": "v508 creates a candidate nomination resolution packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_candidate_nomination_resolution_packet.csv"
                ),
                "boundary": "Candidate nomination resolution packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v508 declares resolution fields and controls.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_resolution_field_completion_matrix.csv"
                ),
                "boundary": "Resolution field and control declaration only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v508 makes candidate resolution gap audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_resolution_next_action_queue.csv"
                ),
                "boundary": "Future resolution gap audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v508 resolves candidates, nominates candidates, or assigns reviewers.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_candidate_nomination_resolution_packet.csv"
                ),
                "boundary": "Candidates and reviewers remain unresolved.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v508 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v508 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v508_manuscript_readiness_delta.csv"
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
                "executable_item": "v508 creates candidate nomination resolution packet.",
                "status": "candidate_nomination_resolution_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v509 audits candidate resolution gaps",
                "last_wave": "v508",
                "execution_result": "candidate_resolution_packet_created_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v508")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Nomination Resolution Packet v508

Generated: {status["generated_at_utc"]}

## Result

v508 creates a candidate nomination resolution packet for the 14 open
nomination gaps and maps the 84 nomination fields into a completion matrix. It
does not resolve candidate identifiers, complete nomination fields, record
candidate nominations, start eligibility review, assign reviewers or allow
outcome capture.

## Counts

- Resolution packet rows: `{status["resolution_packet_rows_v508"]}`.
- Resolution packet ready rows: `{status["resolution_packet_ready_rows_v508"]}`.
- Candidate identifier resolved rows: `{status["candidate_identifier_resolved_rows_v508"]}`.
- Nomination fields completed rows: `{status["nomination_fields_completed_rows_v508"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v508"]}`.
- Resolution field rows: `{status["resolution_field_rows_v508"]}`.
- Field completed rows: `{status["field_completed_rows_v508"]}`.
- Field prefilled rows: `{status["field_prefilled_rows_v508"]}`.
- Resolution control rows: `{status["resolution_control_rows_v508"]}`.
- Active resolution control rows: `{status["active_resolution_control_rows_v508"]}`.
- Recommended next action rows: `{status["recommended_next_action_rows_v508"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v508"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v508"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v508"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v508"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v508"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v508 is a resolution packet only. It does not resolve or nominate candidates,
assign reviewers, capture completed review outcomes, finalize captions, approve
patch scope, edit Quarto, render the book, make Paper 4 submission-ready,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V508_CANDIDATE_NOMINATION_RESOLUTION_PACKET_START -->"
    end = "<!-- V508_CANDIDATE_NOMINATION_RESOLUTION_PACKET_END -->"
    block = f"""
{start}

## Wave v508: Candidate Nomination Resolution Packet

Generated: {status["generated_at_utc"]}

### Objective

v508 prepares the manual resolution process for the v507 candidate nomination
gaps without resolving candidates or starting eligibility review.

### Results

- Resolution packet rows:
  `{status["resolution_packet_rows_v508"]}`.
- Resolution packet ready rows:
  `{status["resolution_packet_ready_rows_v508"]}`.
- Candidate identifier resolved rows:
  `{status["candidate_identifier_resolved_rows_v508"]}`.
- Nomination fields completed rows:
  `{status["nomination_fields_completed_rows_v508"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v508"]}`.
- Resolution field rows:
  `{status["resolution_field_rows_v508"]}`.
- Field completed rows:
  `{status["field_completed_rows_v508"]}`.
- Field prefilled rows:
  `{status["field_prefilled_rows_v508"]}`.
- Resolution control rows:
  `{status["resolution_control_rows_v508"]}`.
- Active resolution control rows:
  `{status["active_resolution_control_rows_v508"]}`.
- Recommended next action rows:
  `{status["recommended_next_action_rows_v508"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v508"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v508"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v508"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v508"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v508"]}`.
- Book sources modified:
  `{status["book_sources_modified_v508"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v508"]}`.

### Interpretation

The nomination process now has a resolution packet and field completion matrix,
but every candidate identifier, field completion, nomination, eligibility
review, assignment and outcome-capture permission remains absent.

### Claim Impact

- Allowed: resolution packet creation, field completion matrix, active controls
  and future resolution gap audit readiness.
- Still prohibited: candidate resolution/nomination, reviewer assignment,
  completed review claims, final captions, Quarto patch readiness/application,
  Quarto/book mutation, submission readiness, Paper Estrella replacement and
  final Paper 4 promotion.

### Quarto Promotion Decision

Keep v508 in the living notebook. v509 should audit candidate resolution gaps
before any real nomination or eligibility-review claim.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v507 = _read_status(PRIOR_NOMINATION_GAP_VERSION)
    expected_next = "paper4_v508_candidate_nomination_resolution_packet.md"
    if v507["next_artifact_v507"] != expected_next:
        raise RuntimeError("v508 expects v507 to route to nomination resolution packet.")
    if not v507["candidate_nomination_resolution_packet_ready_v507"]:
        raise RuntimeError("v508 requires v507 resolution packet readiness.")

    gaps = pd.read_csv(TABLE_DIR / "paper4_v507_candidate_nomination_gap_audit.csv")
    fields = pd.read_csv(TABLE_DIR / "paper4_v506_candidate_nomination_field_requirements.csv")
    packet = _candidate_nomination_resolution_packet(gaps)
    field_matrix = _resolution_field_completion_matrix(fields)
    controls = _resolution_control_register()
    next_actions = _resolution_next_action_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v508_candidate_nomination_resolution_packet.csv", packet)
    write_csv(TABLE_DIR / "paper4_v508_resolution_field_completion_matrix.csv", field_matrix)
    write_csv(TABLE_DIR / "paper4_v508_resolution_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v508_resolution_next_action_queue.csv", next_actions)
    write_csv(TABLE_DIR / "paper4_v508_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v508_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v508_candidate_nomination_resolution_packet",
        "schema_version": "2026-05-17.508",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_nomination_gap_version_v508": PRIOR_NOMINATION_GAP_VERSION,
        "candidate_nomination_resolution_packet_created_v508": True,
        "resolution_packet_rows_v508": len(packet),
        "resolution_packet_ready_rows_v508": int(
            packet["resolution_packet_ready_v508"].astype(bool).sum()
        ),
        "candidate_identifier_resolved_rows_v508": int(
            packet["candidate_identifier_resolved_v508"].astype(bool).sum()
        ),
        "nomination_fields_completed_rows_v508": int(
            packet["nomination_fields_completed_v508"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v508": int(
            packet["candidate_nomination_recorded_v508"].astype(bool).sum()
        ),
        "resolution_field_rows_v508": len(field_matrix),
        "field_completed_rows_v508": int(
            field_matrix["field_completed_v508"].astype(bool).sum()
        ),
        "field_prefilled_rows_v508": int(
            field_matrix["field_prefilled_v508"].astype(bool).sum()
        ),
        "resolution_control_rows_v508": len(controls),
        "active_resolution_control_rows_v508": int(
            controls["control_active_v508"].astype(bool).sum()
        ),
        "recommended_next_action_rows_v508": int(
            next_actions["recommended_next_v508"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v508": int(
            packet["eligibility_review_allowed_v508"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v508": int(
            packet["reviewer_assignment_allowed_v508"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v508": int(
            packet["outcome_capture_allowed_v508"].astype(bool).sum()
        ),
        "patch_allowed_rows_v508": int(packet["patch_allowed_v508"].astype(bool).sum()),
        "readiness_delta_rows_v508": len(readiness),
        "candidate_resolution_gap_audit_ready_v508": True,
        "ready_for_quarto_patch_v508": False,
        "quarto_patch_applied_v508": False,
        "book_sources_modified_v508": False,
        "book_references_modified_v508": False,
        "submission_ready_claim_allowed_v508": False,
        "working_champion_claim_allowed_v508": False,
        "paper1_promotion_allowed_v508": False,
        "paper4_working_champion_changed_v508": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v508": NEXT_ARTIFACT,
        "claim_boundary": (
            "v508 creates a candidate nomination resolution packet only; candidate "
            "resolution, nominations, assignments, outcomes, captions, patching, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v508 must not create final Paper 4 promotion.")
    if status["candidate_identifier_resolved_rows_v508"] != 0:
        raise RuntimeError("v508 must not resolve candidate identifiers.")
    if status["candidate_nomination_recorded_rows_v508"] != 0:
        raise RuntimeError("v508 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v508"] != 0:
        raise RuntimeError("v508 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v508"] != 0:
        raise RuntimeError("v508 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v508"] != 0:
        raise RuntimeError("v508 must not allow outcome capture.")
    if status["patch_allowed_rows_v508"] != 0:
        raise RuntimeError("v508 must not approve a Quarto patch.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v508": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
