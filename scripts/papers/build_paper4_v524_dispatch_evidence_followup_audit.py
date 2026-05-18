#!/usr/bin/env python3
"""Build Paper 4 v524 dispatch evidence follow-up audit artifacts."""

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

VERSION = 524
PRIOR_DISPATCH_EVIDENCE_REQUEST_VERSION = 523
NEXT_ARTIFACT = "paper4_v525_dispatch_evidence_escalation_packet.md"
FOLLOWUP_MD = NOTEBOOK.parent / "paper4_v524_dispatch_evidence_followup_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _dispatch_evidence_followup_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        external_dispatch_recorded = bool(row["external_dispatch_recorded_v523"])
        dispatch_evidence_received = bool(row["dispatch_evidence_received_v523"])
        human_response_received = bool(row["human_response_received_v523"])
        candidate_identifier_received = bool(
            row["candidate_identifier_received_v523"]
        )
        nomination_fields_received = bool(row["nomination_fields_received_v523"])
        nomination_signoff_received = bool(row["nomination_signoff_received_v523"])
        evidence_received = bool(row["evidence_received_v523"])
        complete = (
            external_dispatch_recorded
            and dispatch_evidence_received
            and human_response_received
            and candidate_identifier_received
            and nomination_fields_received
            and nomination_signoff_received
            and evidence_received
        )
        rows.append(
            {
                "dispatch_evidence_followup_audit_id_v524": row[
                    "dispatch_evidence_request_id_v523"
                ],
                "priority_v524": int(row["priority_v523"]),
                "review_domain_v524": row["review_domain_v523"],
                "reviewer_role_required_v524": row["reviewer_role_required_v523"],
                "dispatch_evidence_request_created_v524": bool(
                    row["dispatch_evidence_request_created_v523"]
                ),
                "dispatch_delivery_trace_request_created_v524": bool(
                    row["dispatch_delivery_trace_request_created_v523"]
                ),
                "dispatch_timestamp_request_created_v524": bool(
                    row["dispatch_timestamp_request_created_v523"]
                ),
                "dispatch_recipient_ack_request_created_v524": bool(
                    row["dispatch_recipient_ack_request_created_v523"]
                ),
                "external_dispatch_recorded_v524": external_dispatch_recorded,
                "dispatch_evidence_received_v524": dispatch_evidence_received,
                "dispatch_delivery_trace_received_v524": False,
                "dispatch_timestamp_received_v524": False,
                "dispatch_recipient_ack_received_v524": False,
                "human_response_received_v524": human_response_received,
                "candidate_identifier_received_v524": (
                    candidate_identifier_received
                ),
                "nomination_fields_received_v524": nomination_fields_received,
                "nomination_signoff_received_v524": nomination_signoff_received,
                "evidence_received_v524": evidence_received,
                "dispatch_evidence_followup_complete_v524": complete,
                "dispatch_evidence_followup_gap_open_v524": not complete,
                "candidate_input_collection_closed_v524": False,
                "candidate_nomination_recorded_v524": bool(
                    row["candidate_nomination_recorded_v523"]
                ),
                "eligibility_review_allowed_v524": False,
                "reviewer_assignment_allowed_v524": False,
                "outcome_capture_allowed_v524": False,
                "patch_allowed_v524": False,
                "required_next_step_v524": (
                    "prepare_dispatch_evidence_escalation_packet"
                ),
                "claim_boundary_v524": "dispatch evidence follow-up audit only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_dispatch_followup_audit(
    field_requests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_requests.iterrows():
        field_received = bool(row["field_value_received_v523"])
        evidence_received = bool(row["field_evidence_received_v523"])
        rows.append(
            {
                "dispatch_evidence_followup_audit_id_v524": row[
                    "dispatch_evidence_request_id_v523"
                ],
                "nomination_field_v524": row["nomination_field_v523"],
                "field_dispatch_evidence_request_created_v524": bool(
                    row["field_dispatch_evidence_request_created_v523"]
                ),
                "field_value_received_v524": field_received,
                "field_evidence_received_v524": evidence_received,
                "field_evidence_dispatch_followup_gap_open_v524": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v524": (
                    "field evidence dispatch follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_requirement_followup_audit(requirements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in requirements.iterrows():
        received = bool(row["dispatch_evidence_received_v523"])
        rows.append(
            {
                "dispatch_requirement_followup_id_v524": row[
                    "dispatch_evidence_requirement_id_v523"
                ],
                "requirement_active_v524": bool(row["requirement_active_v523"]),
                "dispatch_evidence_required_v524": bool(
                    row["dispatch_evidence_required_v523"]
                ),
                "dispatch_evidence_received_v524": received,
                "dispatch_requirement_gap_open_v524": not received,
                "required_evidence_v524": row["required_evidence_v523"],
                "claim_boundary_v524": (
                    "dispatch evidence requirement follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_evidence_followup_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_evidence_followup_blocker_id_v524": (
                    "dispatch_evidence_absent_after_request"
                ),
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": True,
                "required_resolution_v524": (
                    "receive externally verifiable dispatch evidence"
                ),
            },
            {
                "dispatch_evidence_followup_blocker_id_v524": (
                    "delivery_trace_absent_after_request"
                ),
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": True,
                "required_resolution_v524": "receive dispatch delivery trace",
            },
            {
                "dispatch_evidence_followup_blocker_id_v524": (
                    "timestamp_absent_after_request"
                ),
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": True,
                "required_resolution_v524": "receive dispatch timestamp",
            },
            {
                "dispatch_evidence_followup_blocker_id_v524": (
                    "manual_owner_ack_absent_after_request"
                ),
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": True,
                "required_resolution_v524": (
                    "receive manual owner acknowledgement"
                ),
            },
            {
                "dispatch_evidence_followup_blocker_id_v524": (
                    "human_response_absent_after_dispatch_evidence_request"
                ),
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": True,
                "required_resolution_v524": (
                    "receive manual owner or human response"
                ),
            },
            {
                "dispatch_evidence_followup_blocker_id_v524": "no_final_promotion",
                "blocker_open_v524": True,
                "blocks_dispatch_evidence_followup_completion_v524": False,
                "required_resolution_v524": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v524": "dispatch_evidence_followup_audit_created",
                "ready_v524": True,
                "evidence_artifact_v524": (
                    "paper4_v524_dispatch_evidence_followup_audit.csv"
                ),
                "claim_boundary_v524": "dispatch evidence follow-up audit only",
            },
            {
                "readiness_gate_v524": (
                    "field_evidence_dispatch_followup_audit_created"
                ),
                "ready_v524": True,
                "evidence_artifact_v524": (
                    "paper4_v524_field_evidence_dispatch_followup_audit.csv"
                ),
                "claim_boundary_v524": "field evidence dispatch follow-up audit only",
            },
            {
                "readiness_gate_v524": (
                    "dispatch_requirement_followup_audit_created"
                ),
                "ready_v524": True,
                "evidence_artifact_v524": (
                    "paper4_v524_dispatch_requirement_followup_audit.csv"
                ),
                "claim_boundary_v524": "dispatch requirement follow-up audit only",
            },
            {
                "readiness_gate_v524": (
                    "dispatch_evidence_escalation_packet_ready"
                ),
                "ready_v524": True,
                "evidence_artifact_v524": (
                    "paper4_v524_dispatch_evidence_followup_blocker_register.csv"
                ),
                "claim_boundary_v524": (
                    "future dispatch evidence escalation packet readiness only"
                ),
            },
            {
                "readiness_gate_v524": "candidate_identifiers_received",
                "ready_v524": False,
                "evidence_artifact_v524": "candidate identifiers remain unreceived",
                "claim_boundary_v524": "no candidate identifiers received",
            },
            {
                "readiness_gate_v524": "candidate_nominations_recorded",
                "ready_v524": False,
                "evidence_artifact_v524": "candidate nominations remain absent",
                "claim_boundary_v524": "no candidates nominated",
            },
            {
                "readiness_gate_v524": "ready_for_quarto_patch",
                "ready_v524": False,
                "evidence_artifact_v524": "candidate inputs remain absent",
                "claim_boundary_v524": "patch remains blocked",
            },
            {
                "readiness_gate_v524": "paper4_final_promotion_created",
                "ready_v524": False,
                "evidence_artifact_v524": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v524": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v524_dispatch_evidence_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v524_dispatch_evidence_followup_audit.csv",
                "boundary": "dispatch evidence follow-up audit only",
            },
            {
                "claim_id": (
                    "v524_field_evidence_dispatch_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v524_field_evidence_dispatch_followup_audit.csv"
                ),
                "boundary": "field evidence dispatch follow-up audit only",
            },
            {
                "claim_id": "v524_dispatch_requirement_followup_audit_created",
                "allowed": True,
                "artifact": (
                    "paper4_v524_dispatch_requirement_followup_audit.csv"
                ),
                "boundary": "dispatch requirement follow-up audit only",
            },
            {
                "claim_id": "v524_dispatch_evidence_escalation_packet_ready",
                "allowed": True,
                "artifact": (
                    "paper4_v524_dispatch_evidence_followup_blocker_register.csv"
                ),
                "boundary": (
                    "future dispatch evidence escalation packet readiness only"
                ),
            },
            {
                "claim_id": "v524_dispatch_evidence_received_or_recorded",
                "allowed": False,
                "artifact": "paper4_v524_dispatch_evidence_followup_audit.csv",
                "boundary": "no dispatch evidence received or recorded",
            },
            {
                "claim_id": "v524_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v524_dispatch_evidence_followup_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v524_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v524_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v524_final_promotion",
                "allowed": False,
                "artifact": "paper4_v524_manuscript_readiness_delta.csv",
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
                "claim": "v524 audits dispatch evidence follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_dispatch_evidence_followup_audit.csv"
                ),
                "boundary": "Dispatch evidence follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v524 audits field evidence dispatch follow-up."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_field_evidence_dispatch_followup_audit.csv"
                ),
                "boundary": "Field evidence dispatch follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v524 audits dispatch evidence requirements follow-up."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_dispatch_requirement_followup_audit.csv"
                ),
                "boundary": "Dispatch requirement follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v524 makes dispatch evidence escalation packet executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_dispatch_evidence_followup_blocker_register.csv"
                ),
                "boundary": (
                    "Future dispatch evidence escalation packet readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v524 records external dispatch evidence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_dispatch_evidence_followup_audit.csv"
                ),
                "boundary": "Dispatch evidence remains unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v524 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_dispatch_evidence_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v524 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v524 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v524_manuscript_readiness_delta.csv"
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
                "executable_item": "v524 audits dispatch evidence follow-up.",
                "status": "dispatch_evidence_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v525 creates dispatch evidence escalation packet"
                ),
                "last_wave": "v524",
                "execution_result": (
                    "dispatch_evidence_followup_audit_confirmed_no_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v524")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Dispatch Evidence Follow-up Audit v524

Generated: {status["generated_at_utc"]}

## Result

v524 audits the v523 dispatch evidence request packet. Dispatch evidence
requests, field evidence requests and dispatch requirements exist, but no
external dispatch record, delivery trace, timestamp, recipient acknowledgement,
human response, candidate input or supporting evidence has been received. The
next executable step is a bounded dispatch evidence escalation packet.

## Counts

- Dispatch evidence follow-up audit rows: `{status["dispatch_evidence_followup_audit_rows_v524"]}`.
- Dispatch evidence request created rows: `{status["dispatch_evidence_request_created_rows_v524"]}`.
- Dispatch delivery trace request created rows: `{status["dispatch_delivery_trace_request_created_rows_v524"]}`.
- Dispatch timestamp request created rows: `{status["dispatch_timestamp_request_created_rows_v524"]}`.
- Dispatch recipient acknowledgement request created rows: `{status["dispatch_recipient_ack_request_created_rows_v524"]}`.
- External dispatch recorded rows: `{status["external_dispatch_recorded_rows_v524"]}`.
- Dispatch evidence received rows: `{status["dispatch_evidence_received_rows_v524"]}`.
- Dispatch delivery trace received rows: `{status["dispatch_delivery_trace_received_rows_v524"]}`.
- Dispatch timestamp received rows: `{status["dispatch_timestamp_received_rows_v524"]}`.
- Dispatch recipient acknowledgement received rows: `{status["dispatch_recipient_ack_received_rows_v524"]}`.
- Human response received rows: `{status["human_response_received_rows_v524"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v524"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v524"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v524"]}`.
- Evidence received rows: `{status["evidence_received_rows_v524"]}`.
- Dispatch evidence follow-up complete rows: `{status["dispatch_evidence_followup_complete_rows_v524"]}`.
- Open dispatch evidence follow-up gap rows: `{status["open_dispatch_evidence_followup_gap_rows_v524"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v524"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v524"]}`.
- Field/evidence dispatch follow-up rows: `{status["field_evidence_dispatch_followup_rows_v524"]}`.
- Field dispatch evidence request created rows: `{status["field_dispatch_evidence_request_created_rows_v524"]}`.
- Field value received rows: `{status["field_value_received_rows_v524"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v524"]}`.
- Open field/evidence dispatch follow-up gap rows: `{status["open_field_evidence_dispatch_followup_gap_rows_v524"]}`.
- Dispatch requirement follow-up rows: `{status["dispatch_requirement_followup_rows_v524"]}`.
- Open dispatch requirement gap rows: `{status["open_dispatch_requirement_gap_rows_v524"]}`.
- Dispatch evidence follow-up blocker rows: `{status["dispatch_evidence_followup_blocker_rows_v524"]}`.
- Open dispatch evidence follow-up blocker rows: `{status["open_dispatch_evidence_followup_blocker_rows_v524"]}`.
- Blocking dispatch evidence follow-up rows: `{status["blocking_dispatch_evidence_followup_rows_v524"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v524"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v524"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v524"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v524"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v524"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v524 is a dispatch evidence follow-up audit only. It does not record external
dispatch, receive dispatch evidence, receive candidate inputs, resolve or
nominate candidates, assign reviewers, capture completed review outcomes,
finalize captions, approve patch scope, edit Quarto, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V524_DISPATCH_EVIDENCE_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V524_DISPATCH_EVIDENCE_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v524: Dispatch Evidence Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v524 audits whether the v523 dispatch evidence request packet has produced
externally verifiable dispatch evidence, delivery traces or human responses. It
keeps the audit bounded to absence/presence checks and does not convert open
requests into recorded evidence.

### Results

- Dispatch evidence follow-up audit rows:
  `{status["dispatch_evidence_followup_audit_rows_v524"]}`.
- Dispatch evidence request created rows:
  `{status["dispatch_evidence_request_created_rows_v524"]}`.
- Dispatch delivery trace request created rows:
  `{status["dispatch_delivery_trace_request_created_rows_v524"]}`.
- Dispatch timestamp request created rows:
  `{status["dispatch_timestamp_request_created_rows_v524"]}`.
- Dispatch recipient acknowledgement request created rows:
  `{status["dispatch_recipient_ack_request_created_rows_v524"]}`.
- External dispatch recorded rows:
  `{status["external_dispatch_recorded_rows_v524"]}`.
- Dispatch evidence received rows:
  `{status["dispatch_evidence_received_rows_v524"]}`.
- Dispatch delivery trace received rows:
  `{status["dispatch_delivery_trace_received_rows_v524"]}`.
- Dispatch timestamp received rows:
  `{status["dispatch_timestamp_received_rows_v524"]}`.
- Dispatch recipient acknowledgement received rows:
  `{status["dispatch_recipient_ack_received_rows_v524"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v524"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v524"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v524"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v524"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v524"]}`.
- Dispatch evidence follow-up complete rows:
  `{status["dispatch_evidence_followup_complete_rows_v524"]}`.
- Open dispatch evidence follow-up gap rows:
  `{status["open_dispatch_evidence_followup_gap_rows_v524"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v524"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v524"]}`.
- Field/evidence dispatch follow-up rows:
  `{status["field_evidence_dispatch_followup_rows_v524"]}`.
- Field dispatch evidence request created rows:
  `{status["field_dispatch_evidence_request_created_rows_v524"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v524"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v524"]}`.
- Open field/evidence dispatch follow-up gap rows:
  `{status["open_field_evidence_dispatch_followup_gap_rows_v524"]}`.
- Dispatch requirement follow-up rows:
  `{status["dispatch_requirement_followup_rows_v524"]}`.
- Open dispatch requirement gap rows:
  `{status["open_dispatch_requirement_gap_rows_v524"]}`.
- Dispatch evidence follow-up blocker rows:
  `{status["dispatch_evidence_followup_blocker_rows_v524"]}`.
- Open dispatch evidence follow-up blocker rows:
  `{status["open_dispatch_evidence_followup_blocker_rows_v524"]}`.
- Blocking dispatch evidence follow-up rows:
  `{status["blocking_dispatch_evidence_followup_rows_v524"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v524"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v524"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v524"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v524"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v524"]}`.
- Book sources modified:
  `{status["book_sources_modified_v524"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v524"]}`.

### Interpretation

The dispatch evidence request has not yielded verifiable evidence. Because
dispatch evidence, responses and candidate inputs remain zero, the next
executable step is a dispatch evidence escalation packet, not candidate
nomination, eligibility review or manuscript patching.

### Claim Impact

- Allowed: dispatch evidence follow-up audit, field evidence dispatch follow-up
  audit, dispatch requirement follow-up audit and future dispatch evidence
  escalation packet readiness.
- Still prohibited: external dispatch completion, dispatch evidence receipt,
  candidate input receipt, candidate resolution/nomination, reviewer assignment,
  completed review claims, final captions, Quarto patch readiness/application,
  Quarto/book mutation, submission readiness, Paper Estrella replacement and
  final Paper 4 promotion.

### Quarto Promotion Decision

Keep v524 in the living notebook. v525 should prepare a dispatch evidence
escalation packet while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v523 = _read_status(PRIOR_DISPATCH_EVIDENCE_REQUEST_VERSION)
    expected_next = "paper4_v524_dispatch_evidence_followup_audit.md"
    if v523["next_artifact_v523"] != expected_next:
        raise RuntimeError("v524 expects v523 to route to dispatch evidence audit.")
    if not v523["dispatch_evidence_followup_audit_ready_v523"]:
        raise RuntimeError("v524 requires v523 dispatch evidence audit readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v523_dispatch_evidence_request_packet.csv"
    )
    field_requests = pd.read_csv(
        TABLE_DIR / "paper4_v523_field_evidence_dispatch_request_matrix.csv"
    )
    requirements = pd.read_csv(
        TABLE_DIR / "paper4_v523_dispatch_evidence_requirement_register.csv"
    )
    followup = _dispatch_evidence_followup_audit(packet)
    field_followup = _field_evidence_dispatch_followup_audit(field_requests)
    requirement_followup = _dispatch_requirement_followup_audit(requirements)
    blockers = _dispatch_evidence_followup_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v524_dispatch_evidence_followup_audit.csv",
        followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v524_field_evidence_dispatch_followup_audit.csv",
        field_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v524_dispatch_requirement_followup_audit.csv",
        requirement_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v524_dispatch_evidence_followup_blocker_register.csv",
        blockers,
    )
    write_csv(TABLE_DIR / "paper4_v524_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v524_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v524_dispatch_evidence_followup_audit",
        "schema_version": "2026-05-17.524",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dispatch_evidence_request_version_v524": (
            PRIOR_DISPATCH_EVIDENCE_REQUEST_VERSION
        ),
        "dispatch_evidence_followup_audit_created_v524": True,
        "dispatch_evidence_followup_audit_rows_v524": len(followup),
        "dispatch_evidence_request_created_rows_v524": int(
            followup["dispatch_evidence_request_created_v524"].astype(bool).sum()
        ),
        "dispatch_delivery_trace_request_created_rows_v524": int(
            followup[
                "dispatch_delivery_trace_request_created_v524"
            ].astype(bool).sum()
        ),
        "dispatch_timestamp_request_created_rows_v524": int(
            followup[
                "dispatch_timestamp_request_created_v524"
            ].astype(bool).sum()
        ),
        "dispatch_recipient_ack_request_created_rows_v524": int(
            followup[
                "dispatch_recipient_ack_request_created_v524"
            ].astype(bool).sum()
        ),
        "external_dispatch_recorded_rows_v524": int(
            followup["external_dispatch_recorded_v524"].astype(bool).sum()
        ),
        "dispatch_evidence_received_rows_v524": int(
            followup["dispatch_evidence_received_v524"].astype(bool).sum()
        ),
        "dispatch_delivery_trace_received_rows_v524": int(
            followup["dispatch_delivery_trace_received_v524"].astype(bool).sum()
        ),
        "dispatch_timestamp_received_rows_v524": int(
            followup["dispatch_timestamp_received_v524"].astype(bool).sum()
        ),
        "dispatch_recipient_ack_received_rows_v524": int(
            followup["dispatch_recipient_ack_received_v524"].astype(bool).sum()
        ),
        "human_response_received_rows_v524": int(
            followup["human_response_received_v524"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v524": int(
            followup["candidate_identifier_received_v524"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v524": int(
            followup["nomination_fields_received_v524"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v524": int(
            followup["nomination_signoff_received_v524"].astype(bool).sum()
        ),
        "evidence_received_rows_v524": int(
            followup["evidence_received_v524"].astype(bool).sum()
        ),
        "dispatch_evidence_followup_complete_rows_v524": int(
            followup[
                "dispatch_evidence_followup_complete_v524"
            ].astype(bool).sum()
        ),
        "open_dispatch_evidence_followup_gap_rows_v524": int(
            followup[
                "dispatch_evidence_followup_gap_open_v524"
            ].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v524": int(
            followup["candidate_input_collection_closed_v524"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v524": int(
            followup["candidate_nomination_recorded_v524"].astype(bool).sum()
        ),
        "field_evidence_dispatch_followup_rows_v524": len(field_followup),
        "field_dispatch_evidence_request_created_rows_v524": int(
            field_followup[
                "field_dispatch_evidence_request_created_v524"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v524": int(
            field_followup["field_value_received_v524"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v524": int(
            field_followup["field_evidence_received_v524"].astype(bool).sum()
        ),
        "open_field_evidence_dispatch_followup_gap_rows_v524": int(
            field_followup[
                "field_evidence_dispatch_followup_gap_open_v524"
            ].astype(bool).sum()
        ),
        "dispatch_requirement_followup_rows_v524": len(requirement_followup),
        "open_dispatch_requirement_gap_rows_v524": int(
            requirement_followup[
                "dispatch_requirement_gap_open_v524"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_followup_blocker_rows_v524": len(blockers),
        "open_dispatch_evidence_followup_blocker_rows_v524": int(
            blockers["blocker_open_v524"].astype(bool).sum()
        ),
        "blocking_dispatch_evidence_followup_rows_v524": int(
            blockers[
                "blocks_dispatch_evidence_followup_completion_v524"
            ].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v524": int(
            followup["eligibility_review_allowed_v524"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v524": int(
            followup["reviewer_assignment_allowed_v524"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v524": int(
            followup["outcome_capture_allowed_v524"].astype(bool).sum()
        ),
        "patch_allowed_rows_v524": int(
            followup["patch_allowed_v524"].astype(bool).sum()
        ),
        "readiness_delta_rows_v524": len(readiness),
        "dispatch_evidence_escalation_packet_ready_v524": True,
        "ready_for_quarto_patch_v524": False,
        "quarto_patch_applied_v524": False,
        "book_sources_modified_v524": False,
        "book_references_modified_v524": False,
        "submission_ready_claim_allowed_v524": False,
        "working_champion_claim_allowed_v524": False,
        "paper1_promotion_allowed_v524": False,
        "paper4_working_champion_changed_v524": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v524": NEXT_ARTIFACT,
        "claim_boundary": (
            "v524 audits dispatch evidence follow-up only; external dispatch, "
            "dispatch evidence receipt, input receipt, candidate resolution, "
            "nominations, assignments, outcomes, captions, patching, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v524 must not create final Paper 4 promotion.")
    if status["external_dispatch_recorded_rows_v524"] != 0:
        raise RuntimeError("v524 must not record external dispatch.")
    if status["dispatch_evidence_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive dispatch evidence.")
    if status["dispatch_delivery_trace_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive delivery traces.")
    if status["dispatch_timestamp_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive dispatch timestamps.")
    if status["dispatch_recipient_ack_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive dispatch acknowledgements.")
    if status["human_response_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive human responses.")
    if status["candidate_identifier_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive nomination signoff.")
    if status["evidence_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive candidate evidence.")
    if status["dispatch_evidence_followup_complete_rows_v524"] != 0:
        raise RuntimeError("v524 must not complete dispatch evidence follow-up.")
    if status["candidate_input_collection_closed_rows_v524"] != 0:
        raise RuntimeError("v524 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v524"] != 0:
        raise RuntimeError("v524 must not record candidate nominations.")
    if status["field_value_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive field values.")
    if status["field_evidence_received_rows_v524"] != 0:
        raise RuntimeError("v524 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v524"] != 0:
        raise RuntimeError("v524 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v524"] != 0:
        raise RuntimeError("v524 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v524"] != 0:
        raise RuntimeError("v524 must not allow outcome capture.")
    if status["patch_allowed_rows_v524"] != 0:
        raise RuntimeError("v524 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v524": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
