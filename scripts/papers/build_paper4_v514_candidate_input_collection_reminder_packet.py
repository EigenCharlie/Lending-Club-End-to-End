#!/usr/bin/env python3
"""Build Paper 4 v514 candidate input collection reminder packet artifacts."""

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

VERSION = 514
PRIOR_INPUT_RECEIPT_AUDIT_VERSION = 513
NEXT_ARTIFACT = "paper4_v515_collection_reminder_followup_audit.md"
REMINDER_MD = NOTEBOOK.parent / "paper4_v514_candidate_input_collection_reminder_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _collection_reminder_packet(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in audit.iterrows():
        rows.append(
            {
                "collection_reminder_id_v514": row["input_receipt_audit_id_v513"],
                "priority_v514": int(row["priority_v513"]),
                "review_domain_v514": row["review_domain_v513"],
                "reviewer_role_required_v514": row["reviewer_role_required_v513"],
                "reminder_created_v514": True,
                "candidate_identifier_received_v514": bool(
                    row["candidate_identifier_received_v513"]
                ),
                "nomination_fields_received_v514": bool(
                    row["nomination_fields_received_v513"]
                ),
                "nomination_signoff_received_v514": bool(
                    row["nomination_signoff_received_v513"]
                ),
                "evidence_received_v514": bool(row["evidence_received_v513"]),
                "candidate_nomination_recorded_v514": bool(
                    row["candidate_nomination_recorded_v513"]
                ),
                "eligibility_review_allowed_v514": False,
                "reviewer_assignment_allowed_v514": False,
                "outcome_capture_allowed_v514": False,
                "patch_allowed_v514": False,
                "reminder_status_v514": "pending_human_response",
                "required_next_step_v514": (
                    "follow_up_on_candidate_input_collection"
                ),
                "claim_boundary_v514": (
                    "candidate input collection reminder packet only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_collection_checklist(field_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in field_audit.iterrows():
        field_received = bool(row["field_value_received_v513"])
        evidence_received = bool(row["evidence_received_v513"])
        rows.append(
            {
                "collection_reminder_id_v514": row["input_receipt_audit_id_v513"],
                "nomination_field_v514": row["nomination_field_v513"],
                "field_request_created_v514": bool(
                    row["field_request_created_v513"]
                ),
                "field_reminder_created_v514": True,
                "evidence_required_v514": bool(row["evidence_required_v513"]),
                "evidence_reminder_created_v514": True,
                "field_value_received_v514": field_received,
                "field_evidence_received_v514": evidence_received,
                "completion_gap_open_v514": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v514": "field and evidence collection checklist only",
            }
        )
    return pd.DataFrame(rows)


def _collection_reminder_control_register(evidence_summary: pd.DataFrame) -> pd.DataFrame:
    open_evidence_gaps = int(evidence_summary["evidence_gap_open_v513"].astype(bool).sum())
    return pd.DataFrame(
        [
            {
                "collection_reminder_control_id_v514": (
                    "no_candidate_identifier_received"
                ),
                "control_active_v514": True,
                "blocks_collection_completion_v514": True,
                "control_result_v514": "candidate identifiers remain unreceived",
            },
            {
                "collection_reminder_control_id_v514": "no_nomination_fields_received",
                "control_active_v514": True,
                "blocks_collection_completion_v514": True,
                "control_result_v514": "nomination fields remain unreceived",
            },
            {
                "collection_reminder_control_id_v514": (
                    "no_nomination_signoff_received"
                ),
                "control_active_v514": True,
                "blocks_collection_completion_v514": True,
                "control_result_v514": "nomination signoff remains unreceived",
            },
            {
                "collection_reminder_control_id_v514": "no_evidence_received",
                "control_active_v514": True,
                "blocks_collection_completion_v514": True,
                "control_result_v514": (
                    f"{open_evidence_gaps} evidence requirements remain open"
                ),
            },
            {
                "collection_reminder_control_id_v514": "eligibility_review_blocked",
                "control_active_v514": True,
                "blocks_collection_completion_v514": False,
                "control_result_v514": (
                    "eligibility review remains blocked until inputs are received"
                ),
            },
            {
                "collection_reminder_control_id_v514": "no_final_promotion",
                "control_active_v514": True,
                "blocks_collection_completion_v514": False,
                "control_result_v514": "final promotion artifact remains absent",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v514": "collection_reminder_packet_created",
                "ready_v514": True,
                "evidence_artifact_v514": (
                    "paper4_v514_candidate_input_collection_reminder_packet.csv"
                ),
                "claim_boundary_v514": "collection reminder packet only",
            },
            {
                "readiness_gate_v514": (
                    "field_evidence_collection_checklist_created"
                ),
                "ready_v514": True,
                "evidence_artifact_v514": (
                    "paper4_v514_field_evidence_collection_checklist.csv"
                ),
                "claim_boundary_v514": "field evidence collection checklist only",
            },
            {
                "readiness_gate_v514": "collection_reminder_controls_created",
                "ready_v514": True,
                "evidence_artifact_v514": (
                    "paper4_v514_collection_reminder_control_register.csv"
                ),
                "claim_boundary_v514": "collection reminder controls only",
            },
            {
                "readiness_gate_v514": "reminder_followup_audit_ready",
                "ready_v514": True,
                "evidence_artifact_v514": (
                    "paper4_v514_collection_reminder_control_register.csv"
                ),
                "claim_boundary_v514": "future reminder followup audit readiness only",
            },
            {
                "readiness_gate_v514": "candidate_identifiers_received",
                "ready_v514": False,
                "evidence_artifact_v514": "candidate identifiers remain unreceived",
                "claim_boundary_v514": "no candidate identifiers received",
            },
            {
                "readiness_gate_v514": "candidate_nominations_recorded",
                "ready_v514": False,
                "evidence_artifact_v514": "candidate nominations remain absent",
                "claim_boundary_v514": "no candidates nominated",
            },
            {
                "readiness_gate_v514": "ready_for_quarto_patch",
                "ready_v514": False,
                "evidence_artifact_v514": "candidate inputs remain absent",
                "claim_boundary_v514": "patch remains blocked",
            },
            {
                "readiness_gate_v514": "paper4_final_promotion_created",
                "ready_v514": False,
                "evidence_artifact_v514": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v514": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v514_collection_reminder_packet_created",
                "allowed": True,
                "artifact": (
                    "paper4_v514_candidate_input_collection_reminder_packet.csv"
                ),
                "boundary": "collection reminder packet only",
            },
            {
                "claim_id": "v514_field_evidence_checklist_created",
                "allowed": True,
                "artifact": "paper4_v514_field_evidence_collection_checklist.csv",
                "boundary": "field evidence collection checklist only",
            },
            {
                "claim_id": "v514_reminder_followup_audit_ready",
                "allowed": True,
                "artifact": "paper4_v514_collection_reminder_control_register.csv",
                "boundary": "future reminder followup audit readiness only",
            },
            {
                "claim_id": "v514_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": (
                    "paper4_v514_candidate_input_collection_reminder_packet.csv"
                ),
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v514_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v514_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v514_final_promotion",
                "allowed": False,
                "artifact": "paper4_v514_manuscript_readiness_delta.csv",
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
                "claim": "v514 creates a candidate input collection reminder packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_candidate_input_collection_reminder_packet.csv"
                ),
                "boundary": "Candidate input collection reminder packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v514 creates a field and evidence collection checklist.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_field_evidence_collection_checklist.csv"
                ),
                "boundary": "Field and evidence collection checklist only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v514 makes reminder followup audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_collection_reminder_control_register.csv"
                ),
                "boundary": "Future reminder followup audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v514 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_candidate_input_collection_reminder_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v514 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v514 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v514_manuscript_readiness_delta.csv"
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
                "executable_item": (
                    "v514 creates candidate input collection reminders."
                ),
                "status": "candidate_input_collection_reminder_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v515 audits collection reminder followup",
                "last_wave": "v514",
                "execution_result": (
                    "candidate_input_collection_reminder_packet_created_without_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v514")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _reminder_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Input Collection Reminder Packet v514

Generated: {status["generated_at_utc"]}

## Result

v514 creates candidate input collection reminders and a field/evidence
collection checklist for the still-open v513 receipt gaps. It does not receive
candidate identifiers, nomination fields, signoff or evidence; all candidate
nomination, eligibility, reviewer assignment, outcome capture and patch gates
remain closed.

## Counts

- Collection reminder packet rows: `{status["reminder_packet_rows_v514"]}`.
- Reminder created rows: `{status["reminder_created_rows_v514"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v514"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v514"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v514"]}`.
- Evidence received rows: `{status["evidence_received_rows_v514"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v514"]}`.
- Field/evidence checklist rows: `{status["field_evidence_checklist_rows_v514"]}`.
- Field reminder created rows: `{status["field_reminder_created_rows_v514"]}`.
- Evidence reminder created rows: `{status["evidence_reminder_created_rows_v514"]}`.
- Field value received rows: `{status["field_value_received_rows_v514"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v514"]}`.
- Open collection gap rows: `{status["open_collection_gap_rows_v514"]}`.
- Collection reminder control rows: `{status["collection_reminder_control_rows_v514"]}`.
- Active collection reminder control rows: `{status["active_collection_reminder_control_rows_v514"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v514"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v514"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v514"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v514"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v514"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v514 is a collection-reminder packet only. It does not receive candidate inputs,
resolve or nominate candidates, assign reviewers, capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V514_CANDIDATE_INPUT_COLLECTION_REMINDER_PACKET_START -->"
    end = "<!-- V514_CANDIDATE_INPUT_COLLECTION_REMINDER_PACKET_END -->"
    block = f"""
{start}

## Wave v514: Candidate Input Collection Reminder Packet

Generated: {status["generated_at_utc"]}

### Objective

v514 turns the still-open v513 input receipt gaps into an executable reminder
packet and checklist. It creates follow-up structure only; it does not mark any
candidate input or evidence as received.

### Results

- Collection reminder packet rows:
  `{status["reminder_packet_rows_v514"]}`.
- Reminder created rows:
  `{status["reminder_created_rows_v514"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v514"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v514"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v514"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v514"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v514"]}`.
- Field/evidence checklist rows:
  `{status["field_evidence_checklist_rows_v514"]}`.
- Field reminder created rows:
  `{status["field_reminder_created_rows_v514"]}`.
- Evidence reminder created rows:
  `{status["evidence_reminder_created_rows_v514"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v514"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v514"]}`.
- Open collection gap rows:
  `{status["open_collection_gap_rows_v514"]}`.
- Collection reminder control rows:
  `{status["collection_reminder_control_rows_v514"]}`.
- Active collection reminder control rows:
  `{status["active_collection_reminder_control_rows_v514"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v514"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v514"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v514"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v514"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v514"]}`.
- Book sources modified:
  `{status["book_sources_modified_v514"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v514"]}`.

### Interpretation

The candidate path now has a reminder packet for collection follow-up. Because
the evidence counters remain zero, the next executable step is a follow-up
audit, not eligibility review or manuscript patching.

### Claim Impact

- Allowed: collection reminder packet, field/evidence checklist and future
  reminder follow-up audit readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v514 in the living notebook. v515 should audit reminder follow-up while
preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v513 = _read_status(PRIOR_INPUT_RECEIPT_AUDIT_VERSION)
    expected_next = "paper4_v514_candidate_input_collection_reminder_packet.md"
    if v513["next_artifact_v513"] != expected_next:
        raise RuntimeError("v514 expects v513 to route to collection reminders.")
    if not v513["input_collection_reminder_packet_ready_v513"]:
        raise RuntimeError("v514 requires v513 input collection reminder readiness.")

    audit = pd.read_csv(TABLE_DIR / "paper4_v513_candidate_input_receipt_audit.csv")
    field_audit = pd.read_csv(
        TABLE_DIR / "paper4_v513_field_and_evidence_receipt_audit.csv"
    )
    evidence_summary = pd.read_csv(TABLE_DIR / "paper4_v513_evidence_receipt_summary.csv")
    packet = _collection_reminder_packet(audit)
    checklist = _field_evidence_collection_checklist(field_audit)
    controls = _collection_reminder_control_register(evidence_summary)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v514_candidate_input_collection_reminder_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v514_field_evidence_collection_checklist.csv",
        checklist,
    )
    write_csv(
        TABLE_DIR / "paper4_v514_collection_reminder_control_register.csv",
        controls,
    )
    write_csv(TABLE_DIR / "paper4_v514_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v514_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v514_candidate_input_collection_reminder_packet",
        "schema_version": "2026-05-17.514",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_input_receipt_audit_version_v514": PRIOR_INPUT_RECEIPT_AUDIT_VERSION,
        "candidate_input_collection_reminder_packet_created_v514": True,
        "reminder_packet_rows_v514": len(packet),
        "reminder_created_rows_v514": int(
            packet["reminder_created_v514"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v514": int(
            packet["candidate_identifier_received_v514"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v514": int(
            packet["nomination_fields_received_v514"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v514": int(
            packet["nomination_signoff_received_v514"].astype(bool).sum()
        ),
        "evidence_received_rows_v514": int(
            packet["evidence_received_v514"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v514": int(
            packet["candidate_nomination_recorded_v514"].astype(bool).sum()
        ),
        "field_evidence_checklist_rows_v514": len(checklist),
        "field_reminder_created_rows_v514": int(
            checklist["field_reminder_created_v514"].astype(bool).sum()
        ),
        "evidence_reminder_created_rows_v514": int(
            checklist["evidence_reminder_created_v514"].astype(bool).sum()
        ),
        "field_value_received_rows_v514": int(
            checklist["field_value_received_v514"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v514": int(
            checklist["field_evidence_received_v514"].astype(bool).sum()
        ),
        "open_collection_gap_rows_v514": int(
            checklist["completion_gap_open_v514"].astype(bool).sum()
        ),
        "collection_reminder_control_rows_v514": len(controls),
        "active_collection_reminder_control_rows_v514": int(
            controls["control_active_v514"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v514": int(
            packet["eligibility_review_allowed_v514"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v514": int(
            packet["reviewer_assignment_allowed_v514"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v514": int(
            packet["outcome_capture_allowed_v514"].astype(bool).sum()
        ),
        "patch_allowed_rows_v514": int(packet["patch_allowed_v514"].astype(bool).sum()),
        "readiness_delta_rows_v514": len(readiness),
        "reminder_followup_audit_ready_v514": True,
        "ready_for_quarto_patch_v514": False,
        "quarto_patch_applied_v514": False,
        "book_sources_modified_v514": False,
        "book_references_modified_v514": False,
        "submission_ready_claim_allowed_v514": False,
        "working_champion_claim_allowed_v514": False,
        "paper1_promotion_allowed_v514": False,
        "paper4_working_champion_changed_v514": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v514": NEXT_ARTIFACT,
        "claim_boundary": (
            "v514 creates candidate input collection reminders only; input "
            "receipt, candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v514 must not create final Paper 4 promotion.")
    if status["candidate_identifier_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive nomination signoff.")
    if status["evidence_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive evidence.")
    if status["candidate_nomination_recorded_rows_v514"] != 0:
        raise RuntimeError("v514 must not record candidate nominations.")
    if status["field_value_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive field values.")
    if status["field_evidence_received_rows_v514"] != 0:
        raise RuntimeError("v514 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v514"] != 0:
        raise RuntimeError("v514 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v514"] != 0:
        raise RuntimeError("v514 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v514"] != 0:
        raise RuntimeError("v514 must not allow outcome capture.")
    if status["patch_allowed_rows_v514"] != 0:
        raise RuntimeError("v514 must not approve a Quarto patch.")

    REMINDER_MD.write_text(_reminder_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v514": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
