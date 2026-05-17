#!/usr/bin/env python3
"""Build Paper 4 v523 dispatch evidence request packet artifacts."""

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

VERSION = 523
PRIOR_DISPATCH_FOLLOWUP_AUDIT_VERSION = 522
NEXT_ARTIFACT = "paper4_v524_dispatch_evidence_followup_audit.md"
REQUEST_MD = NOTEBOOK.parent / "paper4_v523_dispatch_evidence_request_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _dispatch_evidence_request_packet(followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in followup.iterrows():
        rows.append(
            {
                "dispatch_evidence_request_id_v523": row[
                    "manual_owner_dispatch_followup_audit_id_v522"
                ],
                "priority_v523": int(row["priority_v522"]),
                "review_domain_v523": row["review_domain_v522"],
                "reviewer_role_required_v523": row["reviewer_role_required_v522"],
                "dispatch_packet_created_v523": bool(
                    row["dispatch_packet_created_v522"]
                ),
                "dispatch_ready_v523": bool(row["dispatch_ready_v522"]),
                "dispatch_evidence_request_created_v523": True,
                "dispatch_delivery_trace_request_created_v523": True,
                "dispatch_timestamp_request_created_v523": True,
                "dispatch_recipient_ack_request_created_v523": True,
                "external_dispatch_recorded_v523": bool(
                    row["external_dispatch_recorded_v522"]
                ),
                "dispatch_evidence_received_v523": False,
                "human_response_received_v523": bool(
                    row["human_response_received_v522"]
                ),
                "candidate_identifier_received_v523": bool(
                    row["candidate_identifier_received_v522"]
                ),
                "nomination_fields_received_v523": bool(
                    row["nomination_fields_received_v522"]
                ),
                "nomination_signoff_received_v523": bool(
                    row["nomination_signoff_received_v522"]
                ),
                "evidence_received_v523": bool(row["evidence_received_v522"]),
                "candidate_input_collection_closed_v523": False,
                "candidate_nomination_recorded_v523": bool(
                    row["candidate_nomination_recorded_v522"]
                ),
                "eligibility_review_allowed_v523": False,
                "reviewer_assignment_allowed_v523": False,
                "outcome_capture_allowed_v523": False,
                "patch_allowed_v523": False,
                "dispatch_evidence_request_status_v523": (
                    "pending_dispatch_evidence"
                ),
                "required_next_step_v523": (
                    "audit_dispatch_evidence_request_followup"
                ),
                "claim_boundary_v523": "dispatch evidence request packet only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_dispatch_request_matrix(
    field_followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_followup.iterrows():
        field_received = bool(row["field_value_received_v522"])
        evidence_received = bool(row["field_evidence_received_v522"])
        rows.append(
            {
                "dispatch_evidence_request_id_v523": row[
                    "manual_owner_dispatch_followup_audit_id_v522"
                ],
                "nomination_field_v523": row["nomination_field_v522"],
                "field_dispatch_checklist_created_v523": bool(
                    row["field_dispatch_checklist_created_v522"]
                ),
                "evidence_dispatch_checklist_created_v523": bool(
                    row["evidence_dispatch_checklist_created_v522"]
                ),
                "field_dispatch_evidence_request_created_v523": True,
                "field_value_received_v523": field_received,
                "field_evidence_received_v523": evidence_received,
                "field_evidence_dispatch_request_gap_open_v523": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v523": (
                    "field evidence dispatch request matrix only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_evidence_requirement_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_evidence_requirement_id_v523": (
                    "dispatch_delivery_trace"
                ),
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "externally verifiable dispatch delivery trace"
                ),
            },
            {
                "dispatch_evidence_requirement_id_v523": "dispatch_timestamp",
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "timestamp for manual owner request delivery"
                ),
            },
            {
                "dispatch_evidence_requirement_id_v523": (
                    "dispatch_recipient_owner"
                ),
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "manual owner recipient or ownership evidence"
                ),
            },
            {
                "dispatch_evidence_requirement_id_v523": "request_payload_snapshot",
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "snapshot or checksum of the dispatched request payload"
                ),
            },
            {
                "dispatch_evidence_requirement_id_v523": (
                    "manual_owner_acknowledgement"
                ),
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "manual owner acknowledgement or response trace"
                ),
            },
            {
                "dispatch_evidence_requirement_id_v523": "chain_of_custody_note",
                "requirement_active_v523": True,
                "dispatch_evidence_required_v523": True,
                "dispatch_evidence_received_v523": False,
                "required_evidence_v523": (
                    "chain-of-custody note for dispatch evidence"
                ),
            },
        ]
    )


def _dispatch_evidence_request_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_evidence_request_control_id_v523": (
                    "dispatch_evidence_absent"
                ),
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": True,
                "control_result_v523": "dispatch evidence remains unreceived",
            },
            {
                "dispatch_evidence_request_control_id_v523": (
                    "delivery_trace_absent"
                ),
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": True,
                "control_result_v523": "delivery trace remains absent",
            },
            {
                "dispatch_evidence_request_control_id_v523": "timestamp_absent",
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": True,
                "control_result_v523": "dispatch timestamp remains absent",
            },
            {
                "dispatch_evidence_request_control_id_v523": (
                    "manual_owner_ack_absent"
                ),
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": True,
                "control_result_v523": "manual owner acknowledgement remains absent",
            },
            {
                "dispatch_evidence_request_control_id_v523": (
                    "human_response_absent_after_evidence_request"
                ),
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": True,
                "control_result_v523": "human response remains unreceived",
            },
            {
                "dispatch_evidence_request_control_id_v523": "no_final_promotion",
                "control_active_v523": True,
                "blocks_dispatch_evidence_followup_v523": False,
                "control_result_v523": "final promotion artifact remains absent",
            },
        ]
    )


def _dispatch_evidence_followup_queue(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        rows.append(
            {
                "dispatch_evidence_followup_item_id_v523": row[
                    "dispatch_evidence_request_id_v523"
                ],
                "priority_v523": int(row["priority_v523"]),
                "dispatch_evidence_request_created_v523": bool(
                    row["dispatch_evidence_request_created_v523"]
                ),
                "dispatch_evidence_received_v523": bool(
                    row["dispatch_evidence_received_v523"]
                ),
                "external_dispatch_recorded_v523": bool(
                    row["external_dispatch_recorded_v523"]
                ),
                "followup_audit_ready_v523": True,
                "expected_next_artifact_v523": NEXT_ARTIFACT,
                "claim_boundary_v523": (
                    "future dispatch evidence follow-up audit queue only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v523": "dispatch_evidence_request_packet_created",
                "ready_v523": True,
                "evidence_artifact_v523": (
                    "paper4_v523_dispatch_evidence_request_packet.csv"
                ),
                "claim_boundary_v523": "dispatch evidence request packet only",
            },
            {
                "readiness_gate_v523": (
                    "field_evidence_dispatch_request_matrix_created"
                ),
                "ready_v523": True,
                "evidence_artifact_v523": (
                    "paper4_v523_field_evidence_dispatch_request_matrix.csv"
                ),
                "claim_boundary_v523": "field evidence dispatch request matrix only",
            },
            {
                "readiness_gate_v523": (
                    "dispatch_evidence_requirement_register_created"
                ),
                "ready_v523": True,
                "evidence_artifact_v523": (
                    "paper4_v523_dispatch_evidence_requirement_register.csv"
                ),
                "claim_boundary_v523": (
                    "dispatch evidence requirement register only"
                ),
            },
            {
                "readiness_gate_v523": "dispatch_evidence_followup_audit_ready",
                "ready_v523": True,
                "evidence_artifact_v523": (
                    "paper4_v523_dispatch_evidence_followup_queue.csv"
                ),
                "claim_boundary_v523": (
                    "future dispatch evidence follow-up audit readiness only"
                ),
            },
            {
                "readiness_gate_v523": "candidate_identifiers_received",
                "ready_v523": False,
                "evidence_artifact_v523": "candidate identifiers remain unreceived",
                "claim_boundary_v523": "no candidate identifiers received",
            },
            {
                "readiness_gate_v523": "candidate_nominations_recorded",
                "ready_v523": False,
                "evidence_artifact_v523": "candidate nominations remain absent",
                "claim_boundary_v523": "no candidates nominated",
            },
            {
                "readiness_gate_v523": "ready_for_quarto_patch",
                "ready_v523": False,
                "evidence_artifact_v523": "candidate inputs remain absent",
                "claim_boundary_v523": "patch remains blocked",
            },
            {
                "readiness_gate_v523": "paper4_final_promotion_created",
                "ready_v523": False,
                "evidence_artifact_v523": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v523": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v523_dispatch_evidence_request_packet_created",
                "allowed": True,
                "artifact": "paper4_v523_dispatch_evidence_request_packet.csv",
                "boundary": "dispatch evidence request packet only",
            },
            {
                "claim_id": (
                    "v523_field_evidence_dispatch_request_matrix_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v523_field_evidence_dispatch_request_matrix.csv"
                ),
                "boundary": "field evidence dispatch request matrix only",
            },
            {
                "claim_id": "v523_dispatch_evidence_requirements_declared",
                "allowed": True,
                "artifact": (
                    "paper4_v523_dispatch_evidence_requirement_register.csv"
                ),
                "boundary": "dispatch evidence requirements declared only",
            },
            {
                "claim_id": "v523_dispatch_evidence_followup_audit_ready",
                "allowed": True,
                "artifact": "paper4_v523_dispatch_evidence_followup_queue.csv",
                "boundary": (
                    "future dispatch evidence follow-up audit readiness only"
                ),
            },
            {
                "claim_id": "v523_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v523_dispatch_evidence_request_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v523_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v523_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v523_final_promotion",
                "allowed": False,
                "artifact": "paper4_v523_manuscript_readiness_delta.csv",
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
                "claim": "v523 creates a dispatch evidence request packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_dispatch_evidence_request_packet.csv"
                ),
                "boundary": "Dispatch evidence request packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v523 creates a field evidence dispatch request matrix."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_field_evidence_dispatch_request_matrix.csv"
                ),
                "boundary": "Field evidence dispatch request matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v523 declares dispatch evidence requirements.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_dispatch_evidence_requirement_register.csv"
                ),
                "boundary": "Dispatch evidence requirements only; none received.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v523 makes dispatch evidence follow-up audit executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_dispatch_evidence_followup_queue.csv"
                ),
                "boundary": (
                    "Future dispatch evidence follow-up audit readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v523 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_dispatch_evidence_request_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v523 records external dispatch evidence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_dispatch_evidence_request_packet.csv"
                ),
                "boundary": (
                    "Dispatch evidence is requested, not received or recorded."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v523 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v523 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v523_manuscript_readiness_delta.csv"
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
                "executable_item": "v523 creates dispatch evidence request packet.",
                "status": "dispatch_evidence_request_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v524 audits dispatch evidence request follow-up",
                "last_wave": "v523",
                "execution_result": (
                    "dispatch_evidence_request_packet_created_without_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v523")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _request_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Dispatch Evidence Request Packet v523

Generated: {status["generated_at_utc"]}

## Result

v523 creates the bounded dispatch evidence request packet after v522 confirmed
that no external dispatch evidence or candidate input had been recorded. The
packet requests delivery traces, timestamps, recipient ownership, payload
snapshots, acknowledgement traces and chain-of-custody notes. It does not record
external dispatch or receive any evidence.

## Counts

- Dispatch evidence request rows: `{status["dispatch_evidence_request_rows_v523"]}`.
- Dispatch delivery trace request rows: `{status["dispatch_delivery_trace_request_rows_v523"]}`.
- Dispatch timestamp request rows: `{status["dispatch_timestamp_request_rows_v523"]}`.
- Dispatch recipient acknowledgement request rows: `{status["dispatch_recipient_ack_request_rows_v523"]}`.
- External dispatch recorded rows: `{status["external_dispatch_recorded_rows_v523"]}`.
- Dispatch evidence received rows: `{status["dispatch_evidence_received_rows_v523"]}`.
- Human response received rows: `{status["human_response_received_rows_v523"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v523"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v523"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v523"]}`.
- Evidence received rows: `{status["evidence_received_rows_v523"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v523"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v523"]}`.
- Field/evidence dispatch request rows: `{status["field_evidence_dispatch_request_rows_v523"]}`.
- Field dispatch evidence request created rows: `{status["field_dispatch_evidence_request_created_rows_v523"]}`.
- Field value received rows: `{status["field_value_received_rows_v523"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v523"]}`.
- Open field/evidence dispatch request gap rows: `{status["open_field_evidence_dispatch_request_gap_rows_v523"]}`.
- Dispatch evidence requirement rows: `{status["dispatch_evidence_requirement_rows_v523"]}`.
- Active dispatch evidence requirement rows: `{status["active_dispatch_evidence_requirement_rows_v523"]}`.
- Dispatch evidence requirement received rows: `{status["dispatch_evidence_requirement_received_rows_v523"]}`.
- Dispatch evidence request control rows: `{status["dispatch_evidence_request_control_rows_v523"]}`.
- Active dispatch evidence request control rows: `{status["active_dispatch_evidence_request_control_rows_v523"]}`.
- Blocking dispatch evidence request control rows: `{status["blocking_dispatch_evidence_request_control_rows_v523"]}`.
- Dispatch evidence follow-up queue rows: `{status["dispatch_evidence_followup_queue_rows_v523"]}`.
- Dispatch evidence follow-up audit ready rows: `{status["dispatch_evidence_followup_audit_ready_rows_v523"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v523"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v523"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v523"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v523"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v523"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v523 is a dispatch evidence request packet only. It does not record external
dispatch, receive dispatch evidence, receive candidate inputs, resolve or
nominate candidates, assign reviewers, capture completed review outcomes,
finalize captions, approve patch scope, edit Quarto, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V523_DISPATCH_EVIDENCE_REQUEST_PACKET_START -->"
    end = "<!-- V523_DISPATCH_EVIDENCE_REQUEST_PACKET_END -->"
    block = f"""
{start}

## Wave v523: Dispatch Evidence Request Packet

Generated: {status["generated_at_utc"]}

### Objective

v523 creates the request surface for externally verifiable dispatch evidence.
It asks for traces and provenance needed to prove manual owner request delivery,
but does not convert those requests into received evidence or candidate inputs.

### Results

- Dispatch evidence request rows:
  `{status["dispatch_evidence_request_rows_v523"]}`.
- Dispatch delivery trace request rows:
  `{status["dispatch_delivery_trace_request_rows_v523"]}`.
- Dispatch timestamp request rows:
  `{status["dispatch_timestamp_request_rows_v523"]}`.
- Dispatch recipient acknowledgement request rows:
  `{status["dispatch_recipient_ack_request_rows_v523"]}`.
- External dispatch recorded rows:
  `{status["external_dispatch_recorded_rows_v523"]}`.
- Dispatch evidence received rows:
  `{status["dispatch_evidence_received_rows_v523"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v523"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v523"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v523"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v523"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v523"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v523"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v523"]}`.
- Field/evidence dispatch request rows:
  `{status["field_evidence_dispatch_request_rows_v523"]}`.
- Field dispatch evidence request created rows:
  `{status["field_dispatch_evidence_request_created_rows_v523"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v523"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v523"]}`.
- Open field/evidence dispatch request gap rows:
  `{status["open_field_evidence_dispatch_request_gap_rows_v523"]}`.
- Dispatch evidence requirement rows:
  `{status["dispatch_evidence_requirement_rows_v523"]}`.
- Active dispatch evidence requirement rows:
  `{status["active_dispatch_evidence_requirement_rows_v523"]}`.
- Dispatch evidence requirement received rows:
  `{status["dispatch_evidence_requirement_received_rows_v523"]}`.
- Dispatch evidence request control rows:
  `{status["dispatch_evidence_request_control_rows_v523"]}`.
- Active dispatch evidence request control rows:
  `{status["active_dispatch_evidence_request_control_rows_v523"]}`.
- Blocking dispatch evidence request control rows:
  `{status["blocking_dispatch_evidence_request_control_rows_v523"]}`.
- Dispatch evidence follow-up queue rows:
  `{status["dispatch_evidence_followup_queue_rows_v523"]}`.
- Dispatch evidence follow-up audit ready rows:
  `{status["dispatch_evidence_followup_audit_ready_rows_v523"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v523"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v523"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v523"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v523"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v523"]}`.
- Book sources modified:
  `{status["book_sources_modified_v523"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v523"]}`.

### Interpretation

The dispatch evidence request packet is procedural evidence only. Because
external dispatch evidence, human responses and candidate inputs remain zero,
the next executable step is a dispatch evidence follow-up audit, not candidate
nomination, eligibility review or manuscript patching.

### Claim Impact

- Allowed: dispatch evidence request packet, field evidence dispatch request
  matrix, dispatch evidence requirements and future dispatch evidence follow-up
  audit readiness.
- Still prohibited: external dispatch completion, dispatch evidence receipt,
  candidate input receipt, candidate resolution/nomination, reviewer assignment,
  completed review claims, final captions, Quarto patch readiness/application,
  Quarto/book mutation, submission readiness, Paper Estrella replacement and
  final Paper 4 promotion.

### Quarto Promotion Decision

Keep v523 in the living notebook. v524 should audit dispatch evidence request
follow-up while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v522 = _read_status(PRIOR_DISPATCH_FOLLOWUP_AUDIT_VERSION)
    expected_next = "paper4_v523_dispatch_evidence_request_packet.md"
    if v522["next_artifact_v522"] != expected_next:
        raise RuntimeError("v523 expects v522 to route to dispatch evidence request.")
    if not v522["dispatch_evidence_request_packet_ready_v522"]:
        raise RuntimeError("v523 requires v522 dispatch evidence request readiness.")

    followup = pd.read_csv(
        TABLE_DIR / "paper4_v522_manual_owner_dispatch_followup_audit.csv"
    )
    field_followup = pd.read_csv(
        TABLE_DIR / "paper4_v522_field_evidence_dispatch_followup_audit.csv"
    )
    packet = _dispatch_evidence_request_packet(followup)
    field_requests = _field_evidence_dispatch_request_matrix(field_followup)
    requirements = _dispatch_evidence_requirement_register()
    controls = _dispatch_evidence_request_control_register()
    followup_queue = _dispatch_evidence_followup_queue(packet)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v523_dispatch_evidence_request_packet.csv", packet)
    write_csv(
        TABLE_DIR / "paper4_v523_field_evidence_dispatch_request_matrix.csv",
        field_requests,
    )
    write_csv(
        TABLE_DIR / "paper4_v523_dispatch_evidence_requirement_register.csv",
        requirements,
    )
    write_csv(
        TABLE_DIR / "paper4_v523_dispatch_evidence_request_control_register.csv",
        controls,
    )
    write_csv(
        TABLE_DIR / "paper4_v523_dispatch_evidence_followup_queue.csv",
        followup_queue,
    )
    write_csv(TABLE_DIR / "paper4_v523_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v523_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v523_dispatch_evidence_request_packet",
        "schema_version": "2026-05-17.523",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dispatch_followup_audit_version_v523": (
            PRIOR_DISPATCH_FOLLOWUP_AUDIT_VERSION
        ),
        "dispatch_evidence_request_packet_created_v523": True,
        "dispatch_evidence_request_rows_v523": len(packet),
        "dispatch_delivery_trace_request_rows_v523": int(
            packet[
                "dispatch_delivery_trace_request_created_v523"
            ].astype(bool).sum()
        ),
        "dispatch_timestamp_request_rows_v523": int(
            packet["dispatch_timestamp_request_created_v523"].astype(bool).sum()
        ),
        "dispatch_recipient_ack_request_rows_v523": int(
            packet[
                "dispatch_recipient_ack_request_created_v523"
            ].astype(bool).sum()
        ),
        "external_dispatch_recorded_rows_v523": int(
            packet["external_dispatch_recorded_v523"].astype(bool).sum()
        ),
        "dispatch_evidence_received_rows_v523": int(
            packet["dispatch_evidence_received_v523"].astype(bool).sum()
        ),
        "human_response_received_rows_v523": int(
            packet["human_response_received_v523"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v523": int(
            packet["candidate_identifier_received_v523"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v523": int(
            packet["nomination_fields_received_v523"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v523": int(
            packet["nomination_signoff_received_v523"].astype(bool).sum()
        ),
        "evidence_received_rows_v523": int(
            packet["evidence_received_v523"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v523": int(
            packet["candidate_input_collection_closed_v523"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v523": int(
            packet["candidate_nomination_recorded_v523"].astype(bool).sum()
        ),
        "field_evidence_dispatch_request_rows_v523": len(field_requests),
        "field_dispatch_evidence_request_created_rows_v523": int(
            field_requests[
                "field_dispatch_evidence_request_created_v523"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v523": int(
            field_requests["field_value_received_v523"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v523": int(
            field_requests["field_evidence_received_v523"].astype(bool).sum()
        ),
        "open_field_evidence_dispatch_request_gap_rows_v523": int(
            field_requests[
                "field_evidence_dispatch_request_gap_open_v523"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_requirement_rows_v523": len(requirements),
        "active_dispatch_evidence_requirement_rows_v523": int(
            requirements["requirement_active_v523"].astype(bool).sum()
        ),
        "dispatch_evidence_requirement_received_rows_v523": int(
            requirements[
                "dispatch_evidence_received_v523"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_request_control_rows_v523": len(controls),
        "active_dispatch_evidence_request_control_rows_v523": int(
            controls["control_active_v523"].astype(bool).sum()
        ),
        "blocking_dispatch_evidence_request_control_rows_v523": int(
            controls[
                "blocks_dispatch_evidence_followup_v523"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_followup_queue_rows_v523": len(followup_queue),
        "dispatch_evidence_followup_audit_ready_rows_v523": int(
            followup_queue["followup_audit_ready_v523"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v523": int(
            packet["eligibility_review_allowed_v523"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v523": int(
            packet["reviewer_assignment_allowed_v523"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v523": int(
            packet["outcome_capture_allowed_v523"].astype(bool).sum()
        ),
        "patch_allowed_rows_v523": int(
            packet["patch_allowed_v523"].astype(bool).sum()
        ),
        "readiness_delta_rows_v523": len(readiness),
        "dispatch_evidence_followup_audit_ready_v523": True,
        "ready_for_quarto_patch_v523": False,
        "quarto_patch_applied_v523": False,
        "book_sources_modified_v523": False,
        "book_references_modified_v523": False,
        "submission_ready_claim_allowed_v523": False,
        "working_champion_claim_allowed_v523": False,
        "paper1_promotion_allowed_v523": False,
        "paper4_working_champion_changed_v523": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v523": NEXT_ARTIFACT,
        "claim_boundary": (
            "v523 creates dispatch evidence request packet only; external "
            "dispatch, dispatch evidence receipt, input receipt, candidate "
            "resolution, nominations, assignments, outcomes, captions, patching, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v523 must not create final Paper 4 promotion.")
    if status["external_dispatch_recorded_rows_v523"] != 0:
        raise RuntimeError("v523 must not record external dispatch.")
    if status["dispatch_evidence_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive dispatch evidence.")
    if status["human_response_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive human responses.")
    if status["candidate_identifier_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive nomination signoff.")
    if status["evidence_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive candidate evidence.")
    if status["candidate_input_collection_closed_rows_v523"] != 0:
        raise RuntimeError("v523 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v523"] != 0:
        raise RuntimeError("v523 must not record candidate nominations.")
    if status["field_value_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive field values.")
    if status["field_evidence_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive field evidence.")
    if status["dispatch_evidence_requirement_received_rows_v523"] != 0:
        raise RuntimeError("v523 must not receive dispatch requirements.")
    if status["eligibility_review_allowed_rows_v523"] != 0:
        raise RuntimeError("v523 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v523"] != 0:
        raise RuntimeError("v523 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v523"] != 0:
        raise RuntimeError("v523 must not allow outcome capture.")
    if status["patch_allowed_rows_v523"] != 0:
        raise RuntimeError("v523 must not approve a Quarto patch.")

    REQUEST_MD.write_text(_request_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v523": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
