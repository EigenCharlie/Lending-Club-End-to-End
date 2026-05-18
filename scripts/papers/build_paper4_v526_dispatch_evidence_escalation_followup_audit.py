#!/usr/bin/env python3
"""Build Paper 4 v526 dispatch evidence escalation follow-up audit artifacts."""

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

VERSION = 526
PRIOR_DISPATCH_EVIDENCE_ESCALATION_VERSION = 525
NEXT_ARTIFACT = "paper4_v527_dispatch_evidence_escalation_dispatch_packet.md"
FOLLOWUP_MD = (
    NOTEBOOK.parent / "paper4_v526_dispatch_evidence_escalation_followup_audit.md"
)


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _dispatch_evidence_escalation_followup_audit(
    packet: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        escalation_dispatched = bool(
            row["dispatch_evidence_escalation_dispatched_v525"]
        )
        external_dispatch_recorded = bool(row["external_dispatch_recorded_v525"])
        dispatch_evidence_received = bool(row["dispatch_evidence_received_v525"])
        human_response_received = bool(row["human_response_received_v525"])
        candidate_identifier_received = bool(
            row["candidate_identifier_received_v525"]
        )
        nomination_fields_received = bool(row["nomination_fields_received_v525"])
        nomination_signoff_received = bool(row["nomination_signoff_received_v525"])
        evidence_received = bool(row["evidence_received_v525"])
        complete = (
            escalation_dispatched
            and external_dispatch_recorded
            and dispatch_evidence_received
            and human_response_received
            and candidate_identifier_received
            and nomination_fields_received
            and nomination_signoff_received
            and evidence_received
        )
        rows.append(
            {
                "dispatch_evidence_escalation_followup_audit_id_v526": row[
                    "dispatch_evidence_escalation_id_v525"
                ],
                "priority_v526": int(row["priority_v525"]),
                "review_domain_v526": row["review_domain_v525"],
                "reviewer_role_required_v526": row["reviewer_role_required_v525"],
                "dispatch_evidence_escalation_packet_created_v526": bool(
                    row["dispatch_evidence_escalation_packet_created_v525"]
                ),
                "dispatch_evidence_escalation_ready_v526": bool(
                    row["dispatch_evidence_escalation_ready_v525"]
                ),
                "dispatch_evidence_escalation_dispatched_v526": (
                    escalation_dispatched
                ),
                "external_dispatch_recorded_v526": external_dispatch_recorded,
                "dispatch_evidence_received_v526": dispatch_evidence_received,
                "dispatch_delivery_trace_received_v526": bool(
                    row["dispatch_delivery_trace_received_v525"]
                ),
                "dispatch_timestamp_received_v526": bool(
                    row["dispatch_timestamp_received_v525"]
                ),
                "dispatch_recipient_ack_received_v526": bool(
                    row["dispatch_recipient_ack_received_v525"]
                ),
                "human_response_received_v526": human_response_received,
                "candidate_identifier_received_v526": (
                    candidate_identifier_received
                ),
                "nomination_fields_received_v526": nomination_fields_received,
                "nomination_signoff_received_v526": nomination_signoff_received,
                "evidence_received_v526": evidence_received,
                "dispatch_evidence_escalation_followup_complete_v526": complete,
                "dispatch_evidence_escalation_followup_gap_open_v526": not complete,
                "candidate_input_collection_closed_v526": False,
                "candidate_nomination_recorded_v526": bool(
                    row["candidate_nomination_recorded_v525"]
                ),
                "eligibility_review_allowed_v526": False,
                "reviewer_assignment_allowed_v526": False,
                "outcome_capture_allowed_v526": False,
                "patch_allowed_v526": False,
                "required_next_step_v526": (
                    "prepare_dispatch_evidence_escalation_dispatch_packet"
                ),
                "claim_boundary_v526": (
                    "dispatch evidence escalation follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_escalation_followup_audit(
    field_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_matrix.iterrows():
        field_received = bool(row["field_value_received_v525"])
        evidence_received = bool(row["field_evidence_received_v525"])
        rows.append(
            {
                "dispatch_evidence_escalation_followup_audit_id_v526": row[
                    "dispatch_evidence_escalation_id_v525"
                ],
                "nomination_field_v526": row["nomination_field_v525"],
                "field_evidence_escalation_created_v526": bool(
                    row["field_evidence_escalation_created_v525"]
                ),
                "field_value_received_v526": field_received,
                "field_evidence_received_v526": evidence_received,
                "field_evidence_escalation_followup_gap_open_v526": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v526": (
                    "field evidence escalation follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_requirement_escalation_followup_audit(
    requirement_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in requirement_matrix.iterrows():
        received = bool(row["dispatch_evidence_received_v525"])
        rows.append(
            {
                "dispatch_requirement_escalation_followup_id_v526": row[
                    "dispatch_requirement_escalation_id_v525"
                ],
                "requirement_active_v526": bool(row["requirement_active_v525"]),
                "dispatch_evidence_required_v526": bool(
                    row["dispatch_evidence_required_v525"]
                ),
                "dispatch_requirement_escalation_created_v526": bool(
                    row["dispatch_requirement_escalation_created_v525"]
                ),
                "dispatch_evidence_received_v526": received,
                "dispatch_requirement_escalation_followup_gap_open_v526": (
                    not received
                ),
                "required_evidence_v526": row["required_evidence_v525"],
                "claim_boundary_v526": (
                    "dispatch requirement escalation follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_evidence_escalation_followup_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "escalation_dispatch_absent"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": True,
                "required_resolution_v526": (
                    "record externally verified dispatch of the escalation packet"
                ),
            },
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "dispatch_evidence_absent_after_escalation"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": True,
                "required_resolution_v526": (
                    "receive externally verifiable dispatch evidence"
                ),
            },
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "delivery_trace_absent_after_escalation"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": True,
                "required_resolution_v526": "receive dispatch delivery trace",
            },
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "manual_owner_ack_absent_after_escalation"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": True,
                "required_resolution_v526": (
                    "receive manual owner acknowledgement"
                ),
            },
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "candidate_input_absent_after_escalation"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": True,
                "required_resolution_v526": (
                    "receive candidate identifiers, nomination fields and evidence"
                ),
            },
            {
                "dispatch_evidence_escalation_followup_blocker_id_v526": (
                    "no_final_promotion"
                ),
                "blocker_open_v526": True,
                "blocks_dispatch_evidence_escalation_followup_v526": False,
                "required_resolution_v526": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v526": (
                    "dispatch_evidence_escalation_followup_audit_created"
                ),
                "ready_v526": True,
                "evidence_artifact_v526": (
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "claim_boundary_v526": (
                    "dispatch evidence escalation follow-up audit only"
                ),
            },
            {
                "readiness_gate_v526": (
                    "field_evidence_escalation_followup_audit_created"
                ),
                "ready_v526": True,
                "evidence_artifact_v526": (
                    "paper4_v526_field_evidence_escalation_followup_audit.csv"
                ),
                "claim_boundary_v526": (
                    "field evidence escalation follow-up audit only"
                ),
            },
            {
                "readiness_gate_v526": (
                    "dispatch_requirement_escalation_followup_audit_created"
                ),
                "ready_v526": True,
                "evidence_artifact_v526": (
                    "paper4_v526_dispatch_requirement_escalation_followup_audit.csv"
                ),
                "claim_boundary_v526": (
                    "dispatch requirement escalation follow-up audit only"
                ),
            },
            {
                "readiness_gate_v526": (
                    "dispatch_evidence_escalation_dispatch_packet_ready"
                ),
                "ready_v526": True,
                "evidence_artifact_v526": (
                    "paper4_v526_dispatch_evidence_escalation_followup_blocker_register.csv"
                ),
                "claim_boundary_v526": (
                    "future dispatch evidence escalation dispatch packet readiness only"
                ),
            },
            {
                "readiness_gate_v526": "candidate_identifiers_received",
                "ready_v526": False,
                "evidence_artifact_v526": "candidate identifiers remain unreceived",
                "claim_boundary_v526": "no candidate identifiers received",
            },
            {
                "readiness_gate_v526": "candidate_nominations_recorded",
                "ready_v526": False,
                "evidence_artifact_v526": "candidate nominations remain absent",
                "claim_boundary_v526": "no candidates nominated",
            },
            {
                "readiness_gate_v526": "ready_for_quarto_patch",
                "ready_v526": False,
                "evidence_artifact_v526": "candidate inputs remain absent",
                "claim_boundary_v526": "patch remains blocked",
            },
            {
                "readiness_gate_v526": "paper4_final_promotion_created",
                "ready_v526": False,
                "evidence_artifact_v526": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v526": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": (
                    "v526_dispatch_evidence_escalation_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "dispatch evidence escalation follow-up audit only",
            },
            {
                "claim_id": (
                    "v526_field_evidence_escalation_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v526_field_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "field evidence escalation follow-up audit only",
            },
            {
                "claim_id": (
                    "v526_dispatch_requirement_escalation_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v526_dispatch_requirement_escalation_followup_audit.csv"
                ),
                "boundary": "dispatch requirement escalation follow-up audit only",
            },
            {
                "claim_id": (
                    "v526_dispatch_evidence_escalation_dispatch_packet_ready"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v526_dispatch_evidence_escalation_followup_blocker_register.csv"
                ),
                "boundary": (
                    "future dispatch evidence escalation dispatch packet readiness only"
                ),
            },
            {
                "claim_id": "v526_dispatch_evidence_received_or_recorded",
                "allowed": False,
                "artifact": (
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "no dispatch evidence received or recorded",
            },
            {
                "claim_id": "v526_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": (
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v526_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v526_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v526_final_promotion",
                "allowed": False,
                "artifact": "paper4_v526_manuscript_readiness_delta.csv",
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
                "claim": "v526 audits dispatch evidence escalation follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "Dispatch evidence escalation follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 audits field evidence escalation follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_field_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "Field evidence escalation follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 audits dispatch requirement escalation follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_dispatch_requirement_escalation_followup_audit.csv"
                ),
                "boundary": "Dispatch requirement escalation follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v526 makes dispatch evidence escalation dispatch packet executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_dispatch_evidence_escalation_followup_blocker_register.csv"
                ),
                "boundary": (
                    "Future dispatch evidence escalation dispatch packet readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 records external dispatch evidence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "Dispatch evidence remains unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_dispatch_evidence_escalation_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v526 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v526_manuscript_readiness_delta.csv"
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
                    "v526 audits dispatch evidence escalation follow-up."
                ),
                "status": "dispatch_evidence_escalation_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v527 creates dispatch evidence escalation dispatch packet"
                ),
                "last_wave": "v526",
                "execution_result": (
                    "dispatch_evidence_escalation_followup_audit_confirmed_no_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v526")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Dispatch Evidence Escalation Follow-up Audit v526

Generated: {status["generated_at_utc"]}

## Result

v526 audits the v525 dispatch evidence escalation packet. The escalation packet
exists and is ready, but it has not been dispatched and no external dispatch
record, delivery trace, timestamp, acknowledgement, human response, candidate
input or evidence has been received. The next executable step is a bounded
dispatch evidence escalation dispatch packet.

## Counts

- Dispatch evidence escalation follow-up audit rows: `{status["dispatch_evidence_escalation_followup_audit_rows_v526"]}`.
- Dispatch evidence escalation packet created rows: `{status["dispatch_evidence_escalation_packet_created_rows_v526"]}`.
- Dispatch evidence escalation ready rows: `{status["dispatch_evidence_escalation_ready_rows_v526"]}`.
- Dispatch evidence escalation dispatched rows: `{status["dispatch_evidence_escalation_dispatched_rows_v526"]}`.
- External dispatch recorded rows: `{status["external_dispatch_recorded_rows_v526"]}`.
- Dispatch evidence received rows: `{status["dispatch_evidence_received_rows_v526"]}`.
- Dispatch delivery trace received rows: `{status["dispatch_delivery_trace_received_rows_v526"]}`.
- Dispatch timestamp received rows: `{status["dispatch_timestamp_received_rows_v526"]}`.
- Dispatch recipient acknowledgement received rows: `{status["dispatch_recipient_ack_received_rows_v526"]}`.
- Human response received rows: `{status["human_response_received_rows_v526"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v526"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v526"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v526"]}`.
- Evidence received rows: `{status["evidence_received_rows_v526"]}`.
- Dispatch evidence escalation follow-up complete rows: `{status["dispatch_evidence_escalation_followup_complete_rows_v526"]}`.
- Open dispatch evidence escalation follow-up gap rows: `{status["open_dispatch_evidence_escalation_followup_gap_rows_v526"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v526"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v526"]}`.
- Field/evidence escalation follow-up rows: `{status["field_evidence_escalation_followup_rows_v526"]}`.
- Field evidence escalation created rows: `{status["field_evidence_escalation_created_rows_v526"]}`.
- Field value received rows: `{status["field_value_received_rows_v526"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v526"]}`.
- Open field evidence escalation follow-up gap rows: `{status["open_field_evidence_escalation_followup_gap_rows_v526"]}`.
- Dispatch requirement escalation follow-up rows: `{status["dispatch_requirement_escalation_followup_rows_v526"]}`.
- Open dispatch requirement escalation follow-up gap rows: `{status["open_dispatch_requirement_escalation_followup_gap_rows_v526"]}`.
- Dispatch evidence escalation follow-up blocker rows: `{status["dispatch_evidence_escalation_followup_blocker_rows_v526"]}`.
- Open dispatch evidence escalation follow-up blocker rows: `{status["open_dispatch_evidence_escalation_followup_blocker_rows_v526"]}`.
- Blocking dispatch evidence escalation follow-up rows: `{status["blocking_dispatch_evidence_escalation_followup_rows_v526"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v526"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v526"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v526"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v526"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v526"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v526 is a dispatch evidence escalation follow-up audit only. It does not
dispatch escalation packets, record external dispatch, receive dispatch
evidence, receive candidate inputs, resolve or nominate candidates, assign
reviewers, capture completed review outcomes, finalize captions, approve patch
scope, edit Quarto, render the book, make Paper 4 submission-ready, replace
Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V526_DISPATCH_EVIDENCE_ESCALATION_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V526_DISPATCH_EVIDENCE_ESCALATION_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v526: Dispatch Evidence Escalation Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v526 audits whether the v525 dispatch evidence escalation packet has been
dispatched or answered. It confirms the escalation packet exists, but does not
promote readiness into delivery or receipt claims.

### Results

- Dispatch evidence escalation follow-up audit rows:
  `{status["dispatch_evidence_escalation_followup_audit_rows_v526"]}`.
- Dispatch evidence escalation packet created rows:
  `{status["dispatch_evidence_escalation_packet_created_rows_v526"]}`.
- Dispatch evidence escalation ready rows:
  `{status["dispatch_evidence_escalation_ready_rows_v526"]}`.
- Dispatch evidence escalation dispatched rows:
  `{status["dispatch_evidence_escalation_dispatched_rows_v526"]}`.
- External dispatch recorded rows:
  `{status["external_dispatch_recorded_rows_v526"]}`.
- Dispatch evidence received rows:
  `{status["dispatch_evidence_received_rows_v526"]}`.
- Dispatch delivery trace received rows:
  `{status["dispatch_delivery_trace_received_rows_v526"]}`.
- Dispatch timestamp received rows:
  `{status["dispatch_timestamp_received_rows_v526"]}`.
- Dispatch recipient acknowledgement received rows:
  `{status["dispatch_recipient_ack_received_rows_v526"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v526"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v526"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v526"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v526"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v526"]}`.
- Dispatch evidence escalation follow-up complete rows:
  `{status["dispatch_evidence_escalation_followup_complete_rows_v526"]}`.
- Open dispatch evidence escalation follow-up gap rows:
  `{status["open_dispatch_evidence_escalation_followup_gap_rows_v526"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v526"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v526"]}`.
- Field/evidence escalation follow-up rows:
  `{status["field_evidence_escalation_followup_rows_v526"]}`.
- Field evidence escalation created rows:
  `{status["field_evidence_escalation_created_rows_v526"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v526"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v526"]}`.
- Open field evidence escalation follow-up gap rows:
  `{status["open_field_evidence_escalation_followup_gap_rows_v526"]}`.
- Dispatch requirement escalation follow-up rows:
  `{status["dispatch_requirement_escalation_followup_rows_v526"]}`.
- Open dispatch requirement escalation follow-up gap rows:
  `{status["open_dispatch_requirement_escalation_followup_gap_rows_v526"]}`.
- Dispatch evidence escalation follow-up blocker rows:
  `{status["dispatch_evidence_escalation_followup_blocker_rows_v526"]}`.
- Open dispatch evidence escalation follow-up blocker rows:
  `{status["open_dispatch_evidence_escalation_followup_blocker_rows_v526"]}`.
- Blocking dispatch evidence escalation follow-up rows:
  `{status["blocking_dispatch_evidence_escalation_followup_rows_v526"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v526"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v526"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v526"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v526"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v526"]}`.
- Book sources modified:
  `{status["book_sources_modified_v526"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v526"]}`.

### Interpretation

The dispatch evidence escalation has not moved beyond preparation. Because
escalation dispatch, dispatch evidence, responses and candidate inputs remain
zero, the next executable step is a dispatch packet for the escalation, not
candidate nomination, eligibility review or manuscript patching.

### Claim Impact

- Allowed: dispatch evidence escalation follow-up audit, field evidence
  escalation follow-up audit, dispatch requirement escalation follow-up audit
  and future escalation dispatch packet readiness.
- Still prohibited: escalation dispatch claims, external dispatch completion,
  dispatch evidence receipt, candidate input receipt, candidate
  resolution/nomination, reviewer assignment, completed review claims, final
  captions, Quarto patch readiness/application, Quarto/book mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v526 in the living notebook. v527 should prepare a bounded dispatch packet
for the escalation while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v525 = _read_status(PRIOR_DISPATCH_EVIDENCE_ESCALATION_VERSION)
    expected_next = "paper4_v526_dispatch_evidence_escalation_followup_audit.md"
    if v525["next_artifact_v525"] != expected_next:
        raise RuntimeError("v526 expects v525 to route to escalation audit.")
    if not v525["dispatch_evidence_escalation_followup_audit_ready_v525"]:
        raise RuntimeError("v526 requires v525 escalation audit readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v525_dispatch_evidence_escalation_packet.csv"
    )
    field_matrix = pd.read_csv(
        TABLE_DIR / "paper4_v525_field_evidence_dispatch_escalation_matrix.csv"
    )
    requirement_matrix = pd.read_csv(
        TABLE_DIR / "paper4_v525_dispatch_requirement_escalation_matrix.csv"
    )
    followup = _dispatch_evidence_escalation_followup_audit(packet)
    field_followup = _field_evidence_escalation_followup_audit(field_matrix)
    requirement_followup = _dispatch_requirement_escalation_followup_audit(
        requirement_matrix
    )
    blockers = _dispatch_evidence_escalation_followup_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v526_dispatch_evidence_escalation_followup_audit.csv",
        followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v526_field_evidence_escalation_followup_audit.csv",
        field_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v526_dispatch_requirement_escalation_followup_audit.csv",
        requirement_followup,
    )
    write_csv(
        TABLE_DIR
        / "paper4_v526_dispatch_evidence_escalation_followup_blocker_register.csv",
        blockers,
    )
    write_csv(TABLE_DIR / "paper4_v526_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v526_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v526_dispatch_evidence_escalation_followup_audit",
        "schema_version": "2026-05-17.526",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dispatch_evidence_escalation_version_v526": (
            PRIOR_DISPATCH_EVIDENCE_ESCALATION_VERSION
        ),
        "dispatch_evidence_escalation_followup_audit_created_v526": True,
        "dispatch_evidence_escalation_followup_audit_rows_v526": len(followup),
        "dispatch_evidence_escalation_packet_created_rows_v526": int(
            followup[
                "dispatch_evidence_escalation_packet_created_v526"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_ready_rows_v526": int(
            followup["dispatch_evidence_escalation_ready_v526"].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_dispatched_rows_v526": int(
            followup[
                "dispatch_evidence_escalation_dispatched_v526"
            ].astype(bool).sum()
        ),
        "external_dispatch_recorded_rows_v526": int(
            followup["external_dispatch_recorded_v526"].astype(bool).sum()
        ),
        "dispatch_evidence_received_rows_v526": int(
            followup["dispatch_evidence_received_v526"].astype(bool).sum()
        ),
        "dispatch_delivery_trace_received_rows_v526": int(
            followup["dispatch_delivery_trace_received_v526"].astype(bool).sum()
        ),
        "dispatch_timestamp_received_rows_v526": int(
            followup["dispatch_timestamp_received_v526"].astype(bool).sum()
        ),
        "dispatch_recipient_ack_received_rows_v526": int(
            followup["dispatch_recipient_ack_received_v526"].astype(bool).sum()
        ),
        "human_response_received_rows_v526": int(
            followup["human_response_received_v526"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v526": int(
            followup["candidate_identifier_received_v526"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v526": int(
            followup["nomination_fields_received_v526"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v526": int(
            followup["nomination_signoff_received_v526"].astype(bool).sum()
        ),
        "evidence_received_rows_v526": int(
            followup["evidence_received_v526"].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_followup_complete_rows_v526": int(
            followup[
                "dispatch_evidence_escalation_followup_complete_v526"
            ].astype(bool).sum()
        ),
        "open_dispatch_evidence_escalation_followup_gap_rows_v526": int(
            followup[
                "dispatch_evidence_escalation_followup_gap_open_v526"
            ].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v526": int(
            followup["candidate_input_collection_closed_v526"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v526": int(
            followup["candidate_nomination_recorded_v526"].astype(bool).sum()
        ),
        "field_evidence_escalation_followup_rows_v526": len(field_followup),
        "field_evidence_escalation_created_rows_v526": int(
            field_followup[
                "field_evidence_escalation_created_v526"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v526": int(
            field_followup["field_value_received_v526"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v526": int(
            field_followup["field_evidence_received_v526"].astype(bool).sum()
        ),
        "open_field_evidence_escalation_followup_gap_rows_v526": int(
            field_followup[
                "field_evidence_escalation_followup_gap_open_v526"
            ].astype(bool).sum()
        ),
        "dispatch_requirement_escalation_followup_rows_v526": len(
            requirement_followup
        ),
        "open_dispatch_requirement_escalation_followup_gap_rows_v526": int(
            requirement_followup[
                "dispatch_requirement_escalation_followup_gap_open_v526"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_followup_blocker_rows_v526": len(blockers),
        "open_dispatch_evidence_escalation_followup_blocker_rows_v526": int(
            blockers["blocker_open_v526"].astype(bool).sum()
        ),
        "blocking_dispatch_evidence_escalation_followup_rows_v526": int(
            blockers[
                "blocks_dispatch_evidence_escalation_followup_v526"
            ].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v526": int(
            followup["eligibility_review_allowed_v526"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v526": int(
            followup["reviewer_assignment_allowed_v526"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v526": int(
            followup["outcome_capture_allowed_v526"].astype(bool).sum()
        ),
        "patch_allowed_rows_v526": int(
            followup["patch_allowed_v526"].astype(bool).sum()
        ),
        "readiness_delta_rows_v526": len(readiness),
        "dispatch_evidence_escalation_dispatch_packet_ready_v526": True,
        "ready_for_quarto_patch_v526": False,
        "quarto_patch_applied_v526": False,
        "book_sources_modified_v526": False,
        "book_references_modified_v526": False,
        "submission_ready_claim_allowed_v526": False,
        "working_champion_claim_allowed_v526": False,
        "paper1_promotion_allowed_v526": False,
        "paper4_working_champion_changed_v526": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v526": NEXT_ARTIFACT,
        "claim_boundary": (
            "v526 audits dispatch evidence escalation follow-up only; escalation "
            "dispatch, external dispatch, dispatch evidence receipt, input receipt, "
            "candidate resolution, nominations, assignments, outcomes, captions, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v526 must not create final Paper 4 promotion.")
    if status["dispatch_evidence_escalation_dispatched_rows_v526"] != 0:
        raise RuntimeError("v526 must not dispatch escalation packets.")
    if status["external_dispatch_recorded_rows_v526"] != 0:
        raise RuntimeError("v526 must not record external dispatch.")
    if status["dispatch_evidence_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive dispatch evidence.")
    if status["dispatch_delivery_trace_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive delivery traces.")
    if status["dispatch_timestamp_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive dispatch timestamps.")
    if status["dispatch_recipient_ack_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive dispatch acknowledgements.")
    if status["human_response_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive human responses.")
    if status["candidate_identifier_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive nomination signoff.")
    if status["evidence_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive candidate evidence.")
    if status["dispatch_evidence_escalation_followup_complete_rows_v526"] != 0:
        raise RuntimeError("v526 must not complete escalation follow-up.")
    if status["candidate_input_collection_closed_rows_v526"] != 0:
        raise RuntimeError("v526 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v526"] != 0:
        raise RuntimeError("v526 must not record candidate nominations.")
    if status["field_value_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive field values.")
    if status["field_evidence_received_rows_v526"] != 0:
        raise RuntimeError("v526 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v526"] != 0:
        raise RuntimeError("v526 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v526"] != 0:
        raise RuntimeError("v526 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v526"] != 0:
        raise RuntimeError("v526 must not allow outcome capture.")
    if status["patch_allowed_rows_v526"] != 0:
        raise RuntimeError("v526 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v526": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
