#!/usr/bin/env python3
"""Build Paper 4 v519 manual owner escalation request artifacts."""

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

VERSION = 519
PRIOR_CANDIDATE_INPUT_ESCALATION_DECISION_VERSION = 518
NEXT_ARTIFACT = "paper4_v520_manual_owner_escalation_followup_audit.md"
REQUEST_MD = NOTEBOOK.parent / "paper4_v519_manual_owner_escalation_request_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_owner_escalation_request_packet(
    decision_packet: pd.DataFrame,
    request_queue: pd.DataFrame,
) -> pd.DataFrame:
    ready_map = dict(
        zip(
            request_queue["manual_owner_request_id_v518"],
            request_queue["manual_owner_request_ready_v518"],
            strict=False,
        )
    )
    rows = []
    for _, row in decision_packet.iterrows():
        request_id = row["escalation_decision_id_v518"]
        rows.append(
            {
                "manual_owner_request_id_v519": request_id,
                "priority_v519": int(row["priority_v518"]),
                "review_domain_v519": row["review_domain_v518"],
                "reviewer_role_required_v519": row["reviewer_role_required_v518"],
                "manual_owner_request_ready_v519": bool(ready_map[request_id]),
                "manual_owner_request_created_v519": True,
                "manual_owner_request_dispatched_v519": False,
                "manual_owner_escalation_required_v519": bool(
                    row["manual_owner_escalation_required_v518"]
                ),
                "human_response_received_v519": bool(
                    row["human_response_received_v518"]
                ),
                "candidate_identifier_received_v519": bool(
                    row["candidate_identifier_received_v518"]
                ),
                "nomination_fields_received_v519": bool(
                    row["nomination_fields_received_v518"]
                ),
                "nomination_signoff_received_v519": bool(
                    row["nomination_signoff_received_v518"]
                ),
                "evidence_received_v519": bool(row["evidence_received_v518"]),
                "candidate_input_collection_closed_v519": False,
                "candidate_nomination_recorded_v519": bool(
                    row["candidate_nomination_recorded_v518"]
                ),
                "eligibility_review_allowed_v519": False,
                "reviewer_assignment_allowed_v519": False,
                "outcome_capture_allowed_v519": False,
                "patch_allowed_v519": False,
                "manual_owner_request_status_v519": (
                    "pending_manual_owner_dispatch_or_response"
                ),
                "required_next_step_v519": "audit_manual_owner_escalation_followup",
                "claim_boundary_v519": (
                    "manual owner escalation request packet only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_field_evidence_request_matrix(
    field_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_matrix.iterrows():
        field_received = bool(row["field_value_received_v518"])
        evidence_received = bool(row["field_evidence_received_v518"])
        rows.append(
            {
                "manual_owner_request_id_v519": row["escalation_decision_id_v518"],
                "nomination_field_v519": row["nomination_field_v518"],
                "field_escalation_required_v519": bool(
                    row["field_evidence_escalation_required_v518"]
                ),
                "manual_owner_field_request_created_v519": True,
                "manual_owner_evidence_request_created_v519": True,
                "field_value_received_v519": field_received,
                "field_evidence_received_v519": evidence_received,
                "manual_owner_field_evidence_gap_open_v519": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v519": (
                    "manual owner field and evidence request matrix only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_escalation_control_register(
    requirements: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in requirements.iterrows():
        rows.append(
            {
                "manual_owner_escalation_control_id_v519": row[
                    "escalation_requirement_id_v518"
                ],
                "control_active_v519": bool(row["requirement_open_v518"]),
                "blocks_manual_owner_completion_v519": bool(
                    row["blocks_candidate_input_completion_v518"]
                ),
                "required_resolution_v519": row["required_resolution_v518"],
                "control_result_v519": "open_manual_owner_escalation_requirement",
                "claim_boundary_v519": "manual owner escalation control register only",
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_escalation_followup_queue(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        rows.append(
            {
                "manual_owner_followup_item_id_v519": row[
                    "manual_owner_request_id_v519"
                ],
                "priority_v519": int(row["priority_v519"]),
                "manual_owner_request_created_v519": bool(
                    row["manual_owner_request_created_v519"]
                ),
                "manual_owner_request_dispatched_v519": bool(
                    row["manual_owner_request_dispatched_v519"]
                ),
                "human_response_received_v519": bool(
                    row["human_response_received_v519"]
                ),
                "followup_audit_ready_v519": True,
                "expected_next_artifact_v519": NEXT_ARTIFACT,
                "claim_boundary_v519": "manual owner escalation follow-up queue only",
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v519": (
                    "manual_owner_escalation_request_packet_created"
                ),
                "ready_v519": True,
                "evidence_artifact_v519": (
                    "paper4_v519_manual_owner_escalation_request_packet.csv"
                ),
                "claim_boundary_v519": "manual owner escalation request packet only",
            },
            {
                "readiness_gate_v519": (
                    "manual_owner_field_evidence_request_matrix_created"
                ),
                "ready_v519": True,
                "evidence_artifact_v519": (
                    "paper4_v519_manual_owner_field_evidence_request_matrix.csv"
                ),
                "claim_boundary_v519": "manual owner field evidence matrix only",
            },
            {
                "readiness_gate_v519": (
                    "manual_owner_escalation_control_register_created"
                ),
                "ready_v519": True,
                "evidence_artifact_v519": (
                    "paper4_v519_manual_owner_escalation_control_register.csv"
                ),
                "claim_boundary_v519": "manual owner escalation controls only",
            },
            {
                "readiness_gate_v519": (
                    "manual_owner_escalation_followup_audit_ready"
                ),
                "ready_v519": True,
                "evidence_artifact_v519": (
                    "paper4_v519_manual_owner_escalation_followup_queue.csv"
                ),
                "claim_boundary_v519": (
                    "future manual owner escalation follow-up audit readiness only"
                ),
            },
            {
                "readiness_gate_v519": "candidate_identifiers_received",
                "ready_v519": False,
                "evidence_artifact_v519": "candidate identifiers remain unreceived",
                "claim_boundary_v519": "no candidate identifiers received",
            },
            {
                "readiness_gate_v519": "candidate_nominations_recorded",
                "ready_v519": False,
                "evidence_artifact_v519": "candidate nominations remain absent",
                "claim_boundary_v519": "no candidates nominated",
            },
            {
                "readiness_gate_v519": "ready_for_quarto_patch",
                "ready_v519": False,
                "evidence_artifact_v519": "candidate inputs remain absent",
                "claim_boundary_v519": "patch remains blocked",
            },
            {
                "readiness_gate_v519": "paper4_final_promotion_created",
                "ready_v519": False,
                "evidence_artifact_v519": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v519": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v519_manual_owner_escalation_request_packet_created",
                "allowed": True,
                "artifact": "paper4_v519_manual_owner_escalation_request_packet.csv",
                "boundary": "manual owner escalation request packet only",
            },
            {
                "claim_id": (
                    "v519_manual_owner_field_evidence_request_matrix_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v519_manual_owner_field_evidence_request_matrix.csv"
                ),
                "boundary": "manual owner field evidence matrix only",
            },
            {
                "claim_id": "v519_manual_owner_escalation_followup_audit_ready",
                "allowed": True,
                "artifact": "paper4_v519_manual_owner_escalation_followup_queue.csv",
                "boundary": "future manual owner follow-up audit readiness only",
            },
            {
                "claim_id": "v519_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v519_manual_owner_escalation_request_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v519_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v519_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v519_final_promotion",
                "allowed": False,
                "artifact": "paper4_v519_manuscript_readiness_delta.csv",
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
                "claim": "v519 creates a manual owner escalation request packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manual_owner_escalation_request_packet.csv"
                ),
                "boundary": "Manual owner escalation request packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v519 creates a manual owner field and evidence request matrix."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manual_owner_field_evidence_request_matrix.csv"
                ),
                "boundary": "Manual owner field and evidence request matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v519 makes manual owner escalation follow-up audit executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manual_owner_escalation_followup_queue.csv"
                ),
                "boundary": (
                    "Future manual owner escalation follow-up audit readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v519 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manual_owner_escalation_request_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v519 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v519 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v519_manuscript_readiness_delta.csv"
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
                "executable_item": "v519 creates manual owner escalation request packet.",
                "status": "manual_owner_escalation_request_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v520 audits manual owner escalation follow-up",
                "last_wave": "v519",
                "execution_result": (
                    "manual_owner_escalation_request_packet_created_without_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v519")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _request_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Owner Escalation Request Packet v519

Generated: {status["generated_at_utc"]}

## Result

v519 creates the manual owner escalation request packet from the v518 escalation
decision. The packet prepares 14 manual owner requests and 84 field/evidence
request rows, but it does not dispatch requests, receive human responses,
receive candidate inputs, close collection, nominate candidates or open any
eligibility, reviewer, outcome, patch or promotion gates.

## Counts

- Manual owner request packet rows: `{status["manual_owner_request_packet_rows_v519"]}`.
- Manual owner request created rows: `{status["manual_owner_request_created_rows_v519"]}`.
- Manual owner request dispatched rows: `{status["manual_owner_request_dispatched_rows_v519"]}`.
- Manual owner request follow-up ready rows: `{status["manual_owner_request_followup_ready_rows_v519"]}`.
- Manual owner escalation required rows: `{status["manual_owner_escalation_required_rows_v519"]}`.
- Human response received rows: `{status["human_response_received_rows_v519"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v519"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v519"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v519"]}`.
- Evidence received rows: `{status["evidence_received_rows_v519"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v519"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v519"]}`.
- Field/evidence manual owner request rows: `{status["field_evidence_manual_owner_request_rows_v519"]}`.
- Field request created rows: `{status["field_request_created_rows_v519"]}`.
- Evidence request created rows: `{status["evidence_request_created_rows_v519"]}`.
- Field value received rows: `{status["field_value_received_rows_v519"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v519"]}`.
- Open manual owner field/evidence gap rows: `{status["open_manual_owner_field_evidence_gap_rows_v519"]}`.
- Escalation control rows: `{status["escalation_control_rows_v519"]}`.
- Active escalation control rows: `{status["active_escalation_control_rows_v519"]}`.
- Candidate input completion blocker rows: `{status["candidate_input_completion_blocker_rows_v519"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v519"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v519"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v519"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v519"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v519"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v519 is a manual owner escalation request packet only. It does not receive
candidate inputs, resolve or nominate candidates, assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V519_MANUAL_OWNER_ESCALATION_REQUEST_PACKET_START -->"
    end = "<!-- V519_MANUAL_OWNER_ESCALATION_REQUEST_PACKET_END -->"
    block = f"""
{start}

## Wave v519: Manual Owner Escalation Request Packet

Generated: {status["generated_at_utc"]}

### Objective

v519 materializes the manual owner escalation request packet created by the v518
decision. It prepares request rows and field/evidence request rows for follow-up
audit without dispatching requests or fabricating candidate inputs.

### Results

- Manual owner request packet rows:
  `{status["manual_owner_request_packet_rows_v519"]}`.
- Manual owner request created rows:
  `{status["manual_owner_request_created_rows_v519"]}`.
- Manual owner request dispatched rows:
  `{status["manual_owner_request_dispatched_rows_v519"]}`.
- Manual owner request follow-up ready rows:
  `{status["manual_owner_request_followup_ready_rows_v519"]}`.
- Manual owner escalation required rows:
  `{status["manual_owner_escalation_required_rows_v519"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v519"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v519"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v519"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v519"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v519"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v519"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v519"]}`.
- Field/evidence manual owner request rows:
  `{status["field_evidence_manual_owner_request_rows_v519"]}`.
- Field request created rows:
  `{status["field_request_created_rows_v519"]}`.
- Evidence request created rows:
  `{status["evidence_request_created_rows_v519"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v519"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v519"]}`.
- Open manual owner field/evidence gap rows:
  `{status["open_manual_owner_field_evidence_gap_rows_v519"]}`.
- Escalation control rows:
  `{status["escalation_control_rows_v519"]}`.
- Active escalation control rows:
  `{status["active_escalation_control_rows_v519"]}`.
- Candidate input completion blocker rows:
  `{status["candidate_input_completion_blocker_rows_v519"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v519"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v519"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v519"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v519"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v519"]}`.
- Book sources modified:
  `{status["book_sources_modified_v519"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v519"]}`.

### Interpretation

The manual owner request packet is procedural evidence only. Because no human
response or candidate input has been received, the next executable step is a
manual owner escalation follow-up audit, not eligibility review, candidate
nomination or manuscript patching.

### Claim Impact

- Allowed: manual owner escalation request packet, manual owner field/evidence
  request matrix and future manual owner escalation follow-up audit readiness.
- Still prohibited: request dispatch claims, candidate input receipt, candidate
  resolution/nomination, reviewer assignment, completed review claims, final
  captions, Quarto patch readiness/application, Quarto/book mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v519 in the living notebook. v520 should audit manual owner escalation
follow-up while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v518 = _read_status(PRIOR_CANDIDATE_INPUT_ESCALATION_DECISION_VERSION)
    expected_next = "paper4_v519_manual_owner_escalation_request_packet.md"
    if v518["next_artifact_v518"] != expected_next:
        raise RuntimeError("v519 expects v518 to route to manual owner request.")
    if not v518["manual_owner_escalation_request_packet_ready_v518"]:
        raise RuntimeError("v519 requires v518 manual owner request readiness.")

    decision_packet = pd.read_csv(
        TABLE_DIR / "paper4_v518_candidate_input_escalation_decision_packet.csv"
    )
    field_matrix = pd.read_csv(
        TABLE_DIR / "paper4_v518_field_evidence_escalation_decision_matrix.csv"
    )
    requirements = pd.read_csv(
        TABLE_DIR / "paper4_v518_escalation_requirement_matrix.csv"
    )
    request_queue = pd.read_csv(
        TABLE_DIR / "paper4_v518_manual_owner_escalation_request_queue.csv"
    )
    packet = _manual_owner_escalation_request_packet(decision_packet, request_queue)
    field_requests = _manual_owner_field_evidence_request_matrix(field_matrix)
    controls = _manual_owner_escalation_control_register(requirements)
    followup_queue = _manual_owner_escalation_followup_queue(packet)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v519_manual_owner_escalation_request_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v519_manual_owner_field_evidence_request_matrix.csv",
        field_requests,
    )
    write_csv(
        TABLE_DIR / "paper4_v519_manual_owner_escalation_control_register.csv",
        controls,
    )
    write_csv(
        TABLE_DIR / "paper4_v519_manual_owner_escalation_followup_queue.csv",
        followup_queue,
    )
    write_csv(TABLE_DIR / "paper4_v519_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v519_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v519_manual_owner_escalation_request_packet",
        "schema_version": "2026-05-17.519",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_candidate_input_escalation_decision_version_v519": (
            PRIOR_CANDIDATE_INPUT_ESCALATION_DECISION_VERSION
        ),
        "manual_owner_escalation_request_packet_created_v519": True,
        "manual_owner_request_packet_rows_v519": len(packet),
        "manual_owner_request_created_rows_v519": int(
            packet["manual_owner_request_created_v519"].astype(bool).sum()
        ),
        "manual_owner_request_dispatched_rows_v519": int(
            packet["manual_owner_request_dispatched_v519"].astype(bool).sum()
        ),
        "manual_owner_request_followup_ready_rows_v519": int(
            followup_queue["followup_audit_ready_v519"].astype(bool).sum()
        ),
        "manual_owner_escalation_required_rows_v519": int(
            packet["manual_owner_escalation_required_v519"].astype(bool).sum()
        ),
        "human_response_received_rows_v519": int(
            packet["human_response_received_v519"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v519": int(
            packet["candidate_identifier_received_v519"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v519": int(
            packet["nomination_fields_received_v519"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v519": int(
            packet["nomination_signoff_received_v519"].astype(bool).sum()
        ),
        "evidence_received_rows_v519": int(
            packet["evidence_received_v519"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v519": int(
            packet["candidate_input_collection_closed_v519"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v519": int(
            packet["candidate_nomination_recorded_v519"].astype(bool).sum()
        ),
        "field_evidence_manual_owner_request_rows_v519": len(field_requests),
        "field_request_created_rows_v519": int(
            field_requests[
                "manual_owner_field_request_created_v519"
            ].astype(bool).sum()
        ),
        "evidence_request_created_rows_v519": int(
            field_requests[
                "manual_owner_evidence_request_created_v519"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v519": int(
            field_requests["field_value_received_v519"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v519": int(
            field_requests["field_evidence_received_v519"].astype(bool).sum()
        ),
        "open_manual_owner_field_evidence_gap_rows_v519": int(
            field_requests[
                "manual_owner_field_evidence_gap_open_v519"
            ].astype(bool).sum()
        ),
        "escalation_control_rows_v519": len(controls),
        "active_escalation_control_rows_v519": int(
            controls["control_active_v519"].astype(bool).sum()
        ),
        "candidate_input_completion_blocker_rows_v519": int(
            controls["blocks_manual_owner_completion_v519"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v519": int(
            packet["eligibility_review_allowed_v519"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v519": int(
            packet["reviewer_assignment_allowed_v519"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v519": int(
            packet["outcome_capture_allowed_v519"].astype(bool).sum()
        ),
        "patch_allowed_rows_v519": int(packet["patch_allowed_v519"].astype(bool).sum()),
        "readiness_delta_rows_v519": len(readiness),
        "manual_owner_escalation_followup_audit_ready_v519": True,
        "ready_for_quarto_patch_v519": False,
        "quarto_patch_applied_v519": False,
        "book_sources_modified_v519": False,
        "book_references_modified_v519": False,
        "submission_ready_claim_allowed_v519": False,
        "working_champion_claim_allowed_v519": False,
        "paper1_promotion_allowed_v519": False,
        "paper4_working_champion_changed_v519": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v519": NEXT_ARTIFACT,
        "claim_boundary": (
            "v519 creates manual owner escalation request packet only; input "
            "receipt, candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v519 must not create final Paper 4 promotion.")
    if status["manual_owner_request_dispatched_rows_v519"] != 0:
        raise RuntimeError("v519 must not dispatch manual owner requests.")
    if status["human_response_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive human responses.")
    if status["candidate_identifier_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive nomination signoff.")
    if status["evidence_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive evidence.")
    if status["candidate_input_collection_closed_rows_v519"] != 0:
        raise RuntimeError("v519 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v519"] != 0:
        raise RuntimeError("v519 must not record candidate nominations.")
    if status["field_value_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive field values.")
    if status["field_evidence_received_rows_v519"] != 0:
        raise RuntimeError("v519 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v519"] != 0:
        raise RuntimeError("v519 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v519"] != 0:
        raise RuntimeError("v519 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v519"] != 0:
        raise RuntimeError("v519 must not allow outcome capture.")
    if status["patch_allowed_rows_v519"] != 0:
        raise RuntimeError("v519 must not approve a Quarto patch.")

    REQUEST_MD.write_text(_request_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v519": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
