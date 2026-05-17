#!/usr/bin/env python3
"""Build Paper 4 v518 candidate input escalation decision artifacts."""

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

VERSION = 518
PRIOR_SECOND_REMINDER_FOLLOWUP_AUDIT_VERSION = 517
NEXT_ARTIFACT = "paper4_v519_manual_owner_escalation_request_packet.md"
DECISION_MD = NOTEBOOK.parent / "paper4_v518_candidate_input_escalation_decision_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_input_escalation_decision_packet(
    followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in followup.iterrows():
        rows.append(
            {
                "escalation_decision_id_v518": row[
                    "second_reminder_followup_audit_id_v517"
                ],
                "priority_v518": int(row["priority_v517"]),
                "review_domain_v518": row["review_domain_v517"],
                "reviewer_role_required_v518": row["reviewer_role_required_v517"],
                "second_reminder_followup_gap_open_v518": bool(
                    row["second_reminder_followup_gap_open_v517"]
                ),
                "human_response_received_v518": bool(
                    row["human_response_received_v517"]
                ),
                "candidate_identifier_received_v518": bool(
                    row["candidate_identifier_received_v517"]
                ),
                "nomination_fields_received_v518": bool(
                    row["nomination_fields_received_v517"]
                ),
                "nomination_signoff_received_v518": bool(
                    row["nomination_signoff_received_v517"]
                ),
                "evidence_received_v518": bool(row["evidence_received_v517"]),
                "escalation_decision_recorded_v518": True,
                "escalation_decision_v518": "escalate_to_manual_owner_review",
                "manual_owner_escalation_required_v518": True,
                "candidate_input_collection_closed_v518": False,
                "candidate_nomination_recorded_v518": bool(
                    row["candidate_nomination_recorded_v517"]
                ),
                "eligibility_review_allowed_v518": False,
                "reviewer_assignment_allowed_v518": False,
                "outcome_capture_allowed_v518": False,
                "patch_allowed_v518": False,
                "required_next_step_v518": "issue_manual_owner_escalation_request",
                "claim_boundary_v518": (
                    "candidate input escalation decision packet only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_escalation_decision_matrix(
    field_followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_followup.iterrows():
        field_received = bool(row["field_value_received_v517"])
        evidence_received = bool(row["field_evidence_received_v517"])
        rows.append(
            {
                "escalation_decision_id_v518": row[
                    "second_reminder_followup_audit_id_v517"
                ],
                "nomination_field_v518": row["nomination_field_v517"],
                "field_second_reminder_followup_gap_open_v518": bool(
                    row["field_second_reminder_followup_gap_open_v517"]
                ),
                "field_value_received_v518": field_received,
                "field_evidence_received_v518": evidence_received,
                "field_evidence_escalation_required_v518": not (
                    field_received and evidence_received
                ),
                "manual_owner_request_ready_v518": True,
                "claim_boundary_v518": (
                    "field and evidence escalation decision matrix only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _escalation_requirement_matrix(blockers: pd.DataFrame) -> pd.DataFrame:
    manual_owner_requirements = {
        "no_candidate_identifier_after_second_reminder",
        "no_nomination_field_after_second_reminder",
        "no_nomination_signoff_after_second_reminder",
        "no_evidence_after_second_reminder",
    }
    rows = []
    for _, row in blockers.iterrows():
        requirement_id = row["second_reminder_response_blocker_id_v517"]
        manual_owner_required = requirement_id in manual_owner_requirements
        rows.append(
            {
                "escalation_requirement_id_v518": requirement_id,
                "requirement_open_v518": bool(row["blocker_open_v517"]),
                "manual_owner_escalation_required_v518": manual_owner_required,
                "blocks_candidate_input_completion_v518": manual_owner_required,
                "requirement_satisfied_v518": False,
                "required_resolution_v518": row["required_resolution_v517"],
                "claim_boundary_v518": "escalation requirement matrix only",
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_escalation_request_queue(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        rows.append(
            {
                "manual_owner_request_id_v518": row["escalation_decision_id_v518"],
                "priority_v518": int(row["priority_v518"]),
                "review_domain_v518": row["review_domain_v518"],
                "reviewer_role_required_v518": row["reviewer_role_required_v518"],
                "manual_owner_request_ready_v518": True,
                "manual_owner_request_dispatched_v518": False,
                "human_response_received_v518": bool(
                    row["human_response_received_v518"]
                ),
                "expected_next_artifact_v518": NEXT_ARTIFACT,
                "claim_boundary_v518": (
                    "manual owner escalation request queue only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v518": (
                    "candidate_input_escalation_decision_packet_created"
                ),
                "ready_v518": True,
                "evidence_artifact_v518": (
                    "paper4_v518_candidate_input_escalation_decision_packet.csv"
                ),
                "claim_boundary_v518": "candidate input escalation decision only",
            },
            {
                "readiness_gate_v518": (
                    "field_evidence_escalation_decision_matrix_created"
                ),
                "ready_v518": True,
                "evidence_artifact_v518": (
                    "paper4_v518_field_evidence_escalation_decision_matrix.csv"
                ),
                "claim_boundary_v518": "field evidence escalation matrix only",
            },
            {
                "readiness_gate_v518": "escalation_requirement_matrix_created",
                "ready_v518": True,
                "evidence_artifact_v518": (
                    "paper4_v518_escalation_requirement_matrix.csv"
                ),
                "claim_boundary_v518": "escalation requirement matrix only",
            },
            {
                "readiness_gate_v518": (
                    "manual_owner_escalation_request_packet_ready"
                ),
                "ready_v518": True,
                "evidence_artifact_v518": (
                    "paper4_v518_manual_owner_escalation_request_queue.csv"
                ),
                "claim_boundary_v518": (
                    "future manual owner escalation request packet readiness only"
                ),
            },
            {
                "readiness_gate_v518": "candidate_identifiers_received",
                "ready_v518": False,
                "evidence_artifact_v518": "candidate identifiers remain unreceived",
                "claim_boundary_v518": "no candidate identifiers received",
            },
            {
                "readiness_gate_v518": "candidate_nominations_recorded",
                "ready_v518": False,
                "evidence_artifact_v518": "candidate nominations remain absent",
                "claim_boundary_v518": "no candidates nominated",
            },
            {
                "readiness_gate_v518": "ready_for_quarto_patch",
                "ready_v518": False,
                "evidence_artifact_v518": "candidate inputs remain absent",
                "claim_boundary_v518": "patch remains blocked",
            },
            {
                "readiness_gate_v518": "paper4_final_promotion_created",
                "ready_v518": False,
                "evidence_artifact_v518": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v518": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": (
                    "v518_candidate_input_escalation_decision_packet_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v518_candidate_input_escalation_decision_packet.csv"
                ),
                "boundary": "candidate input escalation decision only",
            },
            {
                "claim_id": (
                    "v518_field_evidence_escalation_decision_matrix_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v518_field_evidence_escalation_decision_matrix.csv"
                ),
                "boundary": "field evidence escalation matrix only",
            },
            {
                "claim_id": "v518_manual_owner_escalation_request_packet_ready",
                "allowed": True,
                "artifact": "paper4_v518_manual_owner_escalation_request_queue.csv",
                "boundary": "future manual owner request packet readiness only",
            },
            {
                "claim_id": "v518_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": (
                    "paper4_v518_candidate_input_escalation_decision_packet.csv"
                ),
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v518_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v518_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v518_final_promotion",
                "allowed": False,
                "artifact": "paper4_v518_manuscript_readiness_delta.csv",
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
                "claim": "v518 records a candidate input escalation decision packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_candidate_input_escalation_decision_packet.csv"
                ),
                "boundary": "Candidate input escalation decision only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v518 records field and evidence escalation decisions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_field_evidence_escalation_decision_matrix.csv"
                ),
                "boundary": "Field and evidence escalation decision matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v518 makes manual owner escalation request executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_manual_owner_escalation_request_queue.csv"
                ),
                "boundary": (
                    "Future manual owner escalation request packet readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v518 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_candidate_input_escalation_decision_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v518 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v518 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v518_manuscript_readiness_delta.csv"
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
                "executable_item": "v518 records candidate input escalation decision.",
                "status": "candidate_input_escalation_decision_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v519 packages manual owner escalation request",
                "last_wave": "v518",
                "execution_result": (
                    "candidate_input_escalation_decision_recorded_without_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v518")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Input Escalation Decision Packet v518

Generated: {status["generated_at_utc"]}

## Result

v518 records the escalation decision after the v517 second-reminder follow-up
audit found no candidate inputs. The decision is to route every unresolved
capture item to a manual owner escalation request packet. It does not dispatch
that request, receive inputs, close collection, nominate candidates or open any
eligibility, reviewer, outcome, patch or promotion gates.

## Counts

- Escalation decision packet rows: `{status["escalation_decision_packet_rows_v518"]}`.
- Escalation decision recorded rows: `{status["escalation_decision_recorded_rows_v518"]}`.
- Manual owner escalation required rows: `{status["manual_owner_escalation_required_rows_v518"]}`.
- Manual owner request queue rows: `{status["manual_owner_request_queue_rows_v518"]}`.
- Manual owner request ready rows: `{status["manual_owner_request_ready_rows_v518"]}`.
- Manual owner request dispatched rows: `{status["manual_owner_request_dispatched_rows_v518"]}`.
- Human response received rows: `{status["human_response_received_rows_v518"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v518"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v518"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v518"]}`.
- Evidence received rows: `{status["evidence_received_rows_v518"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v518"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v518"]}`.
- Field/evidence escalation decision rows: `{status["field_evidence_escalation_decision_rows_v518"]}`.
- Field/evidence escalation required rows: `{status["field_evidence_escalation_required_rows_v518"]}`.
- Field value received rows: `{status["field_value_received_rows_v518"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v518"]}`.
- Open field/evidence escalation gap rows: `{status["open_field_evidence_escalation_gap_rows_v518"]}`.
- Escalation requirement rows: `{status["escalation_requirement_rows_v518"]}`.
- Open escalation requirement rows: `{status["open_escalation_requirement_rows_v518"]}`.
- Candidate input completion blocker rows: `{status["candidate_input_completion_blocker_rows_v518"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v518"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v518"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v518"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v518"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v518"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v518 is a candidate-input escalation decision packet only. It does not receive
candidate inputs, resolve or nominate candidates, assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V518_CANDIDATE_INPUT_ESCALATION_DECISION_PACKET_START -->"
    end = "<!-- V518_CANDIDATE_INPUT_ESCALATION_DECISION_PACKET_END -->"
    block = f"""
{start}

## Wave v518: Candidate Input Escalation Decision Packet

Generated: {status["generated_at_utc"]}

### Objective

v518 records the bounded decision implied by the v517 second-reminder follow-up
audit: unresolved candidate inputs should move to a manual owner escalation
request packet. It records the route only; it does not dispatch requests or
fabricate candidate inputs.

### Results

- Escalation decision packet rows:
  `{status["escalation_decision_packet_rows_v518"]}`.
- Escalation decision recorded rows:
  `{status["escalation_decision_recorded_rows_v518"]}`.
- Manual owner escalation required rows:
  `{status["manual_owner_escalation_required_rows_v518"]}`.
- Manual owner request queue rows:
  `{status["manual_owner_request_queue_rows_v518"]}`.
- Manual owner request ready rows:
  `{status["manual_owner_request_ready_rows_v518"]}`.
- Manual owner request dispatched rows:
  `{status["manual_owner_request_dispatched_rows_v518"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v518"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v518"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v518"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v518"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v518"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v518"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v518"]}`.
- Field/evidence escalation decision rows:
  `{status["field_evidence_escalation_decision_rows_v518"]}`.
- Field/evidence escalation required rows:
  `{status["field_evidence_escalation_required_rows_v518"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v518"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v518"]}`.
- Open field/evidence escalation gap rows:
  `{status["open_field_evidence_escalation_gap_rows_v518"]}`.
- Escalation requirement rows:
  `{status["escalation_requirement_rows_v518"]}`.
- Open escalation requirement rows:
  `{status["open_escalation_requirement_rows_v518"]}`.
- Candidate input completion blocker rows:
  `{status["candidate_input_completion_blocker_rows_v518"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v518"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v518"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v518"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v518"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v518"]}`.
- Book sources modified:
  `{status["book_sources_modified_v518"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v518"]}`.

### Interpretation

The escalation decision is procedural. Because no candidate identifiers,
nomination fields, signoff or evidence exist, v518 keeps collection open and
routes the next executable work to a manual owner request packet rather than
eligibility review, candidate nomination or manuscript patching.

### Claim Impact

- Allowed: candidate-input escalation decision packet, field/evidence
  escalation decision matrix and future manual owner escalation request packet
  readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v518 in the living notebook. v519 should package the manual owner
escalation request while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v517 = _read_status(PRIOR_SECOND_REMINDER_FOLLOWUP_AUDIT_VERSION)
    expected_next = "paper4_v518_candidate_input_escalation_decision_packet.md"
    if v517["next_artifact_v517"] != expected_next:
        raise RuntimeError("v518 expects v517 to route to escalation decision.")
    if not v517["input_escalation_decision_packet_ready_v517"]:
        raise RuntimeError("v518 requires v517 escalation decision readiness.")

    followup = pd.read_csv(
        TABLE_DIR / "paper4_v517_second_reminder_followup_audit.csv"
    )
    field_followup = pd.read_csv(
        TABLE_DIR / "paper4_v517_field_evidence_second_reminder_followup_audit.csv"
    )
    blockers = pd.read_csv(
        TABLE_DIR / "paper4_v517_second_reminder_response_blocker_register.csv"
    )
    packet = _candidate_input_escalation_decision_packet(followup)
    field_matrix = _field_evidence_escalation_decision_matrix(field_followup)
    requirements = _escalation_requirement_matrix(blockers)
    request_queue = _manual_owner_escalation_request_queue(packet)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v518_candidate_input_escalation_decision_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v518_field_evidence_escalation_decision_matrix.csv",
        field_matrix,
    )
    write_csv(
        TABLE_DIR / "paper4_v518_escalation_requirement_matrix.csv",
        requirements,
    )
    write_csv(
        TABLE_DIR / "paper4_v518_manual_owner_escalation_request_queue.csv",
        request_queue,
    )
    write_csv(TABLE_DIR / "paper4_v518_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v518_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v518_candidate_input_escalation_decision_packet",
        "schema_version": "2026-05-17.518",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_second_reminder_followup_audit_version_v518": (
            PRIOR_SECOND_REMINDER_FOLLOWUP_AUDIT_VERSION
        ),
        "candidate_input_escalation_decision_packet_created_v518": True,
        "escalation_decision_packet_rows_v518": len(packet),
        "escalation_decision_recorded_rows_v518": int(
            packet["escalation_decision_recorded_v518"].astype(bool).sum()
        ),
        "manual_owner_escalation_required_rows_v518": int(
            packet["manual_owner_escalation_required_v518"].astype(bool).sum()
        ),
        "manual_owner_request_queue_rows_v518": len(request_queue),
        "manual_owner_request_ready_rows_v518": int(
            request_queue["manual_owner_request_ready_v518"].astype(bool).sum()
        ),
        "manual_owner_request_dispatched_rows_v518": int(
            request_queue["manual_owner_request_dispatched_v518"].astype(bool).sum()
        ),
        "human_response_received_rows_v518": int(
            packet["human_response_received_v518"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v518": int(
            packet["candidate_identifier_received_v518"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v518": int(
            packet["nomination_fields_received_v518"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v518": int(
            packet["nomination_signoff_received_v518"].astype(bool).sum()
        ),
        "evidence_received_rows_v518": int(
            packet["evidence_received_v518"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v518": int(
            packet["candidate_input_collection_closed_v518"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v518": int(
            packet["candidate_nomination_recorded_v518"].astype(bool).sum()
        ),
        "field_evidence_escalation_decision_rows_v518": len(field_matrix),
        "field_evidence_escalation_required_rows_v518": int(
            field_matrix[
                "field_evidence_escalation_required_v518"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v518": int(
            field_matrix["field_value_received_v518"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v518": int(
            field_matrix["field_evidence_received_v518"].astype(bool).sum()
        ),
        "open_field_evidence_escalation_gap_rows_v518": int(
            field_matrix[
                "field_second_reminder_followup_gap_open_v518"
            ].astype(bool).sum()
        ),
        "escalation_requirement_rows_v518": len(requirements),
        "open_escalation_requirement_rows_v518": int(
            requirements["requirement_open_v518"].astype(bool).sum()
        ),
        "candidate_input_completion_blocker_rows_v518": int(
            requirements[
                "blocks_candidate_input_completion_v518"
            ].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v518": int(
            packet["eligibility_review_allowed_v518"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v518": int(
            packet["reviewer_assignment_allowed_v518"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v518": int(
            packet["outcome_capture_allowed_v518"].astype(bool).sum()
        ),
        "patch_allowed_rows_v518": int(packet["patch_allowed_v518"].astype(bool).sum()),
        "readiness_delta_rows_v518": len(readiness),
        "manual_owner_escalation_request_packet_ready_v518": True,
        "ready_for_quarto_patch_v518": False,
        "quarto_patch_applied_v518": False,
        "book_sources_modified_v518": False,
        "book_references_modified_v518": False,
        "submission_ready_claim_allowed_v518": False,
        "working_champion_claim_allowed_v518": False,
        "paper1_promotion_allowed_v518": False,
        "paper4_working_champion_changed_v518": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v518": NEXT_ARTIFACT,
        "claim_boundary": (
            "v518 records candidate input escalation decision only; input "
            "receipt, candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v518 must not create final Paper 4 promotion.")
    if status["human_response_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive human responses.")
    if status["candidate_identifier_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive nomination signoff.")
    if status["evidence_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive evidence.")
    if status["manual_owner_request_dispatched_rows_v518"] != 0:
        raise RuntimeError("v518 must not dispatch manual owner requests.")
    if status["candidate_input_collection_closed_rows_v518"] != 0:
        raise RuntimeError("v518 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v518"] != 0:
        raise RuntimeError("v518 must not record candidate nominations.")
    if status["field_value_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive field values.")
    if status["field_evidence_received_rows_v518"] != 0:
        raise RuntimeError("v518 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v518"] != 0:
        raise RuntimeError("v518 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v518"] != 0:
        raise RuntimeError("v518 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v518"] != 0:
        raise RuntimeError("v518 must not allow outcome capture.")
    if status["patch_allowed_rows_v518"] != 0:
        raise RuntimeError("v518 must not approve a Quarto patch.")

    DECISION_MD.write_text(_decision_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v518": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
