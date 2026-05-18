#!/usr/bin/env python3
"""Build Paper 4 v525 dispatch evidence escalation packet artifacts."""

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

VERSION = 525
PRIOR_DISPATCH_EVIDENCE_FOLLOWUP_AUDIT_VERSION = 524
NEXT_ARTIFACT = "paper4_v526_dispatch_evidence_escalation_followup_audit.md"
ESCALATION_MD = NOTEBOOK.parent / "paper4_v525_dispatch_evidence_escalation_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _dispatch_evidence_escalation_packet(followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in followup.iterrows():
        rows.append(
            {
                "dispatch_evidence_escalation_id_v525": row[
                    "dispatch_evidence_followup_audit_id_v524"
                ],
                "priority_v525": int(row["priority_v524"]),
                "review_domain_v525": row["review_domain_v524"],
                "reviewer_role_required_v525": row["reviewer_role_required_v524"],
                "dispatch_evidence_request_created_v525": bool(
                    row["dispatch_evidence_request_created_v524"]
                ),
                "dispatch_evidence_followup_gap_open_v525": bool(
                    row["dispatch_evidence_followup_gap_open_v524"]
                ),
                "dispatch_evidence_escalation_required_v525": True,
                "dispatch_evidence_escalation_packet_created_v525": True,
                "dispatch_evidence_escalation_ready_v525": True,
                "dispatch_evidence_escalation_dispatched_v525": False,
                "external_dispatch_recorded_v525": bool(
                    row["external_dispatch_recorded_v524"]
                ),
                "dispatch_evidence_received_v525": bool(
                    row["dispatch_evidence_received_v524"]
                ),
                "dispatch_delivery_trace_received_v525": bool(
                    row["dispatch_delivery_trace_received_v524"]
                ),
                "dispatch_timestamp_received_v525": bool(
                    row["dispatch_timestamp_received_v524"]
                ),
                "dispatch_recipient_ack_received_v525": bool(
                    row["dispatch_recipient_ack_received_v524"]
                ),
                "human_response_received_v525": bool(
                    row["human_response_received_v524"]
                ),
                "candidate_identifier_received_v525": bool(
                    row["candidate_identifier_received_v524"]
                ),
                "nomination_fields_received_v525": bool(
                    row["nomination_fields_received_v524"]
                ),
                "nomination_signoff_received_v525": bool(
                    row["nomination_signoff_received_v524"]
                ),
                "evidence_received_v525": bool(row["evidence_received_v524"]),
                "candidate_input_collection_closed_v525": False,
                "candidate_nomination_recorded_v525": bool(
                    row["candidate_nomination_recorded_v524"]
                ),
                "eligibility_review_allowed_v525": False,
                "reviewer_assignment_allowed_v525": False,
                "outcome_capture_allowed_v525": False,
                "patch_allowed_v525": False,
                "dispatch_evidence_escalation_status_v525": (
                    "escalation_packet_ready_not_dispatched"
                ),
                "required_next_step_v525": (
                    "audit_dispatch_evidence_escalation_followup"
                ),
                "claim_boundary_v525": "dispatch evidence escalation packet only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_dispatch_escalation_matrix(
    field_followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_followup.iterrows():
        field_received = bool(row["field_value_received_v524"])
        evidence_received = bool(row["field_evidence_received_v524"])
        rows.append(
            {
                "dispatch_evidence_escalation_id_v525": row[
                    "dispatch_evidence_followup_audit_id_v524"
                ],
                "nomination_field_v525": row["nomination_field_v524"],
                "field_dispatch_evidence_request_created_v525": bool(
                    row["field_dispatch_evidence_request_created_v524"]
                ),
                "field_evidence_escalation_required_v525": True,
                "field_evidence_escalation_created_v525": True,
                "field_value_received_v525": field_received,
                "field_evidence_received_v525": evidence_received,
                "field_evidence_escalation_gap_open_v525": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v525": (
                    "field evidence dispatch escalation matrix only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_requirement_escalation_matrix(
    requirement_followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in requirement_followup.iterrows():
        received = bool(row["dispatch_evidence_received_v524"])
        rows.append(
            {
                "dispatch_requirement_escalation_id_v525": row[
                    "dispatch_requirement_followup_id_v524"
                ],
                "requirement_active_v525": bool(row["requirement_active_v524"]),
                "dispatch_evidence_required_v525": bool(
                    row["dispatch_evidence_required_v524"]
                ),
                "dispatch_requirement_gap_open_v525": bool(
                    row["dispatch_requirement_gap_open_v524"]
                ),
                "dispatch_requirement_escalation_created_v525": True,
                "dispatch_evidence_received_v525": received,
                "required_evidence_v525": row["required_evidence_v524"],
                "claim_boundary_v525": (
                    "dispatch requirement escalation matrix only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_evidence_escalation_control_register(
    blockers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in blockers.iterrows():
        control_id = row["dispatch_evidence_followup_blocker_id_v524"]
        blocks_completion = bool(
            row["blocks_dispatch_evidence_followup_completion_v524"]
        )
        rows.append(
            {
                "dispatch_evidence_escalation_control_id_v525": control_id,
                "control_active_v525": bool(row["blocker_open_v524"]),
                "blocks_dispatch_evidence_escalation_completion_v525": (
                    blocks_completion
                ),
                "required_resolution_v525": row["required_resolution_v524"],
                "control_result_v525": (
                    "open_dispatch_evidence_escalation_requirement"
                ),
                "claim_boundary_v525": (
                    "dispatch evidence escalation control register only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_evidence_escalation_followup_queue(
    packet: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        rows.append(
            {
                "dispatch_evidence_escalation_followup_item_id_v525": row[
                    "dispatch_evidence_escalation_id_v525"
                ],
                "priority_v525": int(row["priority_v525"]),
                "dispatch_evidence_escalation_packet_created_v525": bool(
                    row["dispatch_evidence_escalation_packet_created_v525"]
                ),
                "dispatch_evidence_escalation_dispatched_v525": bool(
                    row["dispatch_evidence_escalation_dispatched_v525"]
                ),
                "dispatch_evidence_received_v525": bool(
                    row["dispatch_evidence_received_v525"]
                ),
                "external_dispatch_recorded_v525": bool(
                    row["external_dispatch_recorded_v525"]
                ),
                "followup_audit_ready_v525": True,
                "expected_next_artifact_v525": NEXT_ARTIFACT,
                "claim_boundary_v525": (
                    "future dispatch evidence escalation follow-up audit queue only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v525": (
                    "dispatch_evidence_escalation_packet_created"
                ),
                "ready_v525": True,
                "evidence_artifact_v525": (
                    "paper4_v525_dispatch_evidence_escalation_packet.csv"
                ),
                "claim_boundary_v525": "dispatch evidence escalation packet only",
            },
            {
                "readiness_gate_v525": (
                    "field_evidence_dispatch_escalation_matrix_created"
                ),
                "ready_v525": True,
                "evidence_artifact_v525": (
                    "paper4_v525_field_evidence_dispatch_escalation_matrix.csv"
                ),
                "claim_boundary_v525": (
                    "field evidence dispatch escalation matrix only"
                ),
            },
            {
                "readiness_gate_v525": (
                    "dispatch_requirement_escalation_matrix_created"
                ),
                "ready_v525": True,
                "evidence_artifact_v525": (
                    "paper4_v525_dispatch_requirement_escalation_matrix.csv"
                ),
                "claim_boundary_v525": (
                    "dispatch requirement escalation matrix only"
                ),
            },
            {
                "readiness_gate_v525": (
                    "dispatch_evidence_escalation_followup_audit_ready"
                ),
                "ready_v525": True,
                "evidence_artifact_v525": (
                    "paper4_v525_dispatch_evidence_escalation_followup_queue.csv"
                ),
                "claim_boundary_v525": (
                    "future dispatch evidence escalation follow-up audit readiness only"
                ),
            },
            {
                "readiness_gate_v525": "candidate_identifiers_received",
                "ready_v525": False,
                "evidence_artifact_v525": "candidate identifiers remain unreceived",
                "claim_boundary_v525": "no candidate identifiers received",
            },
            {
                "readiness_gate_v525": "candidate_nominations_recorded",
                "ready_v525": False,
                "evidence_artifact_v525": "candidate nominations remain absent",
                "claim_boundary_v525": "no candidates nominated",
            },
            {
                "readiness_gate_v525": "ready_for_quarto_patch",
                "ready_v525": False,
                "evidence_artifact_v525": "candidate inputs remain absent",
                "claim_boundary_v525": "patch remains blocked",
            },
            {
                "readiness_gate_v525": "paper4_final_promotion_created",
                "ready_v525": False,
                "evidence_artifact_v525": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v525": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v525_dispatch_evidence_escalation_packet_created",
                "allowed": True,
                "artifact": "paper4_v525_dispatch_evidence_escalation_packet.csv",
                "boundary": "dispatch evidence escalation packet only",
            },
            {
                "claim_id": (
                    "v525_field_evidence_dispatch_escalation_matrix_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v525_field_evidence_dispatch_escalation_matrix.csv"
                ),
                "boundary": "field evidence dispatch escalation matrix only",
            },
            {
                "claim_id": "v525_dispatch_requirement_escalation_matrix_created",
                "allowed": True,
                "artifact": (
                    "paper4_v525_dispatch_requirement_escalation_matrix.csv"
                ),
                "boundary": "dispatch requirement escalation matrix only",
            },
            {
                "claim_id": (
                    "v525_dispatch_evidence_escalation_followup_audit_ready"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v525_dispatch_evidence_escalation_followup_queue.csv"
                ),
                "boundary": (
                    "future dispatch evidence escalation follow-up audit readiness only"
                ),
            },
            {
                "claim_id": "v525_dispatch_evidence_received_or_recorded",
                "allowed": False,
                "artifact": "paper4_v525_dispatch_evidence_escalation_packet.csv",
                "boundary": "no dispatch evidence received or recorded",
            },
            {
                "claim_id": "v525_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v525_dispatch_evidence_escalation_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v525_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v525_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v525_final_promotion",
                "allowed": False,
                "artifact": "paper4_v525_manuscript_readiness_delta.csv",
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
                "claim": "v525 creates a dispatch evidence escalation packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_dispatch_evidence_escalation_packet.csv"
                ),
                "boundary": "Dispatch evidence escalation packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v525 creates a field evidence dispatch escalation matrix."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_field_evidence_dispatch_escalation_matrix.csv"
                ),
                "boundary": "Field evidence dispatch escalation matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v525 creates a dispatch requirement escalation matrix."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_dispatch_requirement_escalation_matrix.csv"
                ),
                "boundary": "Dispatch requirement escalation matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v525 makes dispatch evidence escalation follow-up audit executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_dispatch_evidence_escalation_followup_queue.csv"
                ),
                "boundary": (
                    "Future dispatch evidence escalation follow-up audit readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v525 records external dispatch evidence.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_dispatch_evidence_escalation_packet.csv"
                ),
                "boundary": "Dispatch evidence remains unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v525 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_dispatch_evidence_escalation_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v525 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v525 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v525_manuscript_readiness_delta.csv"
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
                    "v525 creates dispatch evidence escalation packet."
                ),
                "status": "dispatch_evidence_escalation_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v526 audits dispatch evidence escalation follow-up"
                ),
                "last_wave": "v525",
                "execution_result": (
                    "dispatch_evidence_escalation_packet_created_without_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v525")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _escalation_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Dispatch Evidence Escalation Packet v525

Generated: {status["generated_at_utc"]}

## Result

v525 creates the bounded dispatch evidence escalation packet after v524 found no
external dispatch evidence, delivery traces, timestamps, acknowledgements, human
responses or candidate inputs. The escalation packet is ready for a future
follow-up audit, but it is not dispatched and it does not record new evidence.

## Counts

- Dispatch evidence escalation rows: `{status["dispatch_evidence_escalation_rows_v525"]}`.
- Dispatch evidence escalation required rows: `{status["dispatch_evidence_escalation_required_rows_v525"]}`.
- Dispatch evidence escalation packet created rows: `{status["dispatch_evidence_escalation_packet_created_rows_v525"]}`.
- Dispatch evidence escalation ready rows: `{status["dispatch_evidence_escalation_ready_rows_v525"]}`.
- Dispatch evidence escalation dispatched rows: `{status["dispatch_evidence_escalation_dispatched_rows_v525"]}`.
- External dispatch recorded rows: `{status["external_dispatch_recorded_rows_v525"]}`.
- Dispatch evidence received rows: `{status["dispatch_evidence_received_rows_v525"]}`.
- Dispatch delivery trace received rows: `{status["dispatch_delivery_trace_received_rows_v525"]}`.
- Dispatch timestamp received rows: `{status["dispatch_timestamp_received_rows_v525"]}`.
- Dispatch recipient acknowledgement received rows: `{status["dispatch_recipient_ack_received_rows_v525"]}`.
- Human response received rows: `{status["human_response_received_rows_v525"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v525"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v525"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v525"]}`.
- Evidence received rows: `{status["evidence_received_rows_v525"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v525"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v525"]}`.
- Field/evidence dispatch escalation rows: `{status["field_evidence_dispatch_escalation_rows_v525"]}`.
- Field evidence escalation required rows: `{status["field_evidence_escalation_required_rows_v525"]}`.
- Field evidence escalation created rows: `{status["field_evidence_escalation_created_rows_v525"]}`.
- Field value received rows: `{status["field_value_received_rows_v525"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v525"]}`.
- Open field evidence escalation gap rows: `{status["open_field_evidence_escalation_gap_rows_v525"]}`.
- Dispatch requirement escalation rows: `{status["dispatch_requirement_escalation_rows_v525"]}`.
- Dispatch requirement escalation created rows: `{status["dispatch_requirement_escalation_created_rows_v525"]}`.
- Open dispatch requirement escalation gap rows: `{status["open_dispatch_requirement_escalation_gap_rows_v525"]}`.
- Dispatch evidence escalation control rows: `{status["dispatch_evidence_escalation_control_rows_v525"]}`.
- Active dispatch evidence escalation control rows: `{status["active_dispatch_evidence_escalation_control_rows_v525"]}`.
- Blocking dispatch evidence escalation rows: `{status["blocking_dispatch_evidence_escalation_rows_v525"]}`.
- Dispatch evidence escalation follow-up queue rows: `{status["dispatch_evidence_escalation_followup_queue_rows_v525"]}`.
- Dispatch evidence escalation follow-up audit ready rows: `{status["dispatch_evidence_escalation_followup_audit_ready_rows_v525"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v525"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v525"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v525"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v525"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v525"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v525 is a dispatch evidence escalation packet only. It does not dispatch the
escalation, record external dispatch, receive dispatch evidence, receive
candidate inputs, resolve or nominate candidates, assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V525_DISPATCH_EVIDENCE_ESCALATION_PACKET_START -->"
    end = "<!-- V525_DISPATCH_EVIDENCE_ESCALATION_PACKET_END -->"
    block = f"""
{start}

## Wave v525: Dispatch Evidence Escalation Packet

Generated: {status["generated_at_utc"]}

### Objective

v525 creates the escalation surface for missing dispatch evidence after v524
confirmed that no delivery traces, timestamps, acknowledgements, human responses
or candidate inputs had been received. The escalation packet is a preparation
artifact only.

### Results

- Dispatch evidence escalation rows:
  `{status["dispatch_evidence_escalation_rows_v525"]}`.
- Dispatch evidence escalation required rows:
  `{status["dispatch_evidence_escalation_required_rows_v525"]}`.
- Dispatch evidence escalation packet created rows:
  `{status["dispatch_evidence_escalation_packet_created_rows_v525"]}`.
- Dispatch evidence escalation ready rows:
  `{status["dispatch_evidence_escalation_ready_rows_v525"]}`.
- Dispatch evidence escalation dispatched rows:
  `{status["dispatch_evidence_escalation_dispatched_rows_v525"]}`.
- External dispatch recorded rows:
  `{status["external_dispatch_recorded_rows_v525"]}`.
- Dispatch evidence received rows:
  `{status["dispatch_evidence_received_rows_v525"]}`.
- Dispatch delivery trace received rows:
  `{status["dispatch_delivery_trace_received_rows_v525"]}`.
- Dispatch timestamp received rows:
  `{status["dispatch_timestamp_received_rows_v525"]}`.
- Dispatch recipient acknowledgement received rows:
  `{status["dispatch_recipient_ack_received_rows_v525"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v525"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v525"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v525"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v525"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v525"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v525"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v525"]}`.
- Field/evidence dispatch escalation rows:
  `{status["field_evidence_dispatch_escalation_rows_v525"]}`.
- Field evidence escalation required rows:
  `{status["field_evidence_escalation_required_rows_v525"]}`.
- Field evidence escalation created rows:
  `{status["field_evidence_escalation_created_rows_v525"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v525"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v525"]}`.
- Open field evidence escalation gap rows:
  `{status["open_field_evidence_escalation_gap_rows_v525"]}`.
- Dispatch requirement escalation rows:
  `{status["dispatch_requirement_escalation_rows_v525"]}`.
- Dispatch requirement escalation created rows:
  `{status["dispatch_requirement_escalation_created_rows_v525"]}`.
- Open dispatch requirement escalation gap rows:
  `{status["open_dispatch_requirement_escalation_gap_rows_v525"]}`.
- Dispatch evidence escalation control rows:
  `{status["dispatch_evidence_escalation_control_rows_v525"]}`.
- Active dispatch evidence escalation control rows:
  `{status["active_dispatch_evidence_escalation_control_rows_v525"]}`.
- Blocking dispatch evidence escalation rows:
  `{status["blocking_dispatch_evidence_escalation_rows_v525"]}`.
- Dispatch evidence escalation follow-up queue rows:
  `{status["dispatch_evidence_escalation_followup_queue_rows_v525"]}`.
- Dispatch evidence escalation follow-up audit ready rows:
  `{status["dispatch_evidence_escalation_followup_audit_ready_rows_v525"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v525"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v525"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v525"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v525"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v525"]}`.
- Book sources modified:
  `{status["book_sources_modified_v525"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v525"]}`.

### Interpretation

The escalation packet is a procedural preparation artifact. Because escalation
dispatch, dispatch evidence, responses and candidate inputs remain zero, the
next executable step is an escalation follow-up audit, not candidate nomination,
eligibility review or manuscript patching.

### Claim Impact

- Allowed: dispatch evidence escalation packet, field evidence dispatch
  escalation matrix, dispatch requirement escalation matrix and future
  dispatch evidence escalation follow-up audit readiness.
- Still prohibited: escalation dispatch claims, external dispatch completion,
  dispatch evidence receipt, candidate input receipt, candidate
  resolution/nomination, reviewer assignment, completed review claims, final
  captions, Quarto patch readiness/application, Quarto/book mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v525 in the living notebook. v526 should audit dispatch evidence escalation
follow-up while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v524 = _read_status(PRIOR_DISPATCH_EVIDENCE_FOLLOWUP_AUDIT_VERSION)
    expected_next = "paper4_v525_dispatch_evidence_escalation_packet.md"
    if v524["next_artifact_v524"] != expected_next:
        raise RuntimeError("v525 expects v524 to route to dispatch escalation.")
    if not v524["dispatch_evidence_escalation_packet_ready_v524"]:
        raise RuntimeError("v525 requires v524 dispatch escalation readiness.")

    followup = pd.read_csv(
        TABLE_DIR / "paper4_v524_dispatch_evidence_followup_audit.csv"
    )
    field_followup = pd.read_csv(
        TABLE_DIR / "paper4_v524_field_evidence_dispatch_followup_audit.csv"
    )
    requirement_followup = pd.read_csv(
        TABLE_DIR / "paper4_v524_dispatch_requirement_followup_audit.csv"
    )
    blockers = pd.read_csv(
        TABLE_DIR / "paper4_v524_dispatch_evidence_followup_blocker_register.csv"
    )
    packet = _dispatch_evidence_escalation_packet(followup)
    field_matrix = _field_evidence_dispatch_escalation_matrix(field_followup)
    requirement_matrix = _dispatch_requirement_escalation_matrix(
        requirement_followup
    )
    controls = _dispatch_evidence_escalation_control_register(blockers)
    followup_queue = _dispatch_evidence_escalation_followup_queue(packet)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v525_dispatch_evidence_escalation_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v525_field_evidence_dispatch_escalation_matrix.csv",
        field_matrix,
    )
    write_csv(
        TABLE_DIR / "paper4_v525_dispatch_requirement_escalation_matrix.csv",
        requirement_matrix,
    )
    write_csv(
        TABLE_DIR / "paper4_v525_dispatch_evidence_escalation_control_register.csv",
        controls,
    )
    write_csv(
        TABLE_DIR / "paper4_v525_dispatch_evidence_escalation_followup_queue.csv",
        followup_queue,
    )
    write_csv(TABLE_DIR / "paper4_v525_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v525_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v525_dispatch_evidence_escalation_packet",
        "schema_version": "2026-05-17.525",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dispatch_evidence_followup_audit_version_v525": (
            PRIOR_DISPATCH_EVIDENCE_FOLLOWUP_AUDIT_VERSION
        ),
        "dispatch_evidence_escalation_packet_created_v525": True,
        "dispatch_evidence_escalation_rows_v525": len(packet),
        "dispatch_evidence_escalation_required_rows_v525": int(
            packet[
                "dispatch_evidence_escalation_required_v525"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_packet_created_rows_v525": int(
            packet[
                "dispatch_evidence_escalation_packet_created_v525"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_ready_rows_v525": int(
            packet["dispatch_evidence_escalation_ready_v525"].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_dispatched_rows_v525": int(
            packet[
                "dispatch_evidence_escalation_dispatched_v525"
            ].astype(bool).sum()
        ),
        "external_dispatch_recorded_rows_v525": int(
            packet["external_dispatch_recorded_v525"].astype(bool).sum()
        ),
        "dispatch_evidence_received_rows_v525": int(
            packet["dispatch_evidence_received_v525"].astype(bool).sum()
        ),
        "dispatch_delivery_trace_received_rows_v525": int(
            packet["dispatch_delivery_trace_received_v525"].astype(bool).sum()
        ),
        "dispatch_timestamp_received_rows_v525": int(
            packet["dispatch_timestamp_received_v525"].astype(bool).sum()
        ),
        "dispatch_recipient_ack_received_rows_v525": int(
            packet["dispatch_recipient_ack_received_v525"].astype(bool).sum()
        ),
        "human_response_received_rows_v525": int(
            packet["human_response_received_v525"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v525": int(
            packet["candidate_identifier_received_v525"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v525": int(
            packet["nomination_fields_received_v525"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v525": int(
            packet["nomination_signoff_received_v525"].astype(bool).sum()
        ),
        "evidence_received_rows_v525": int(
            packet["evidence_received_v525"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v525": int(
            packet["candidate_input_collection_closed_v525"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v525": int(
            packet["candidate_nomination_recorded_v525"].astype(bool).sum()
        ),
        "field_evidence_dispatch_escalation_rows_v525": len(field_matrix),
        "field_evidence_escalation_required_rows_v525": int(
            field_matrix[
                "field_evidence_escalation_required_v525"
            ].astype(bool).sum()
        ),
        "field_evidence_escalation_created_rows_v525": int(
            field_matrix[
                "field_evidence_escalation_created_v525"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v525": int(
            field_matrix["field_value_received_v525"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v525": int(
            field_matrix["field_evidence_received_v525"].astype(bool).sum()
        ),
        "open_field_evidence_escalation_gap_rows_v525": int(
            field_matrix[
                "field_evidence_escalation_gap_open_v525"
            ].astype(bool).sum()
        ),
        "dispatch_requirement_escalation_rows_v525": len(requirement_matrix),
        "dispatch_requirement_escalation_created_rows_v525": int(
            requirement_matrix[
                "dispatch_requirement_escalation_created_v525"
            ].astype(bool).sum()
        ),
        "open_dispatch_requirement_escalation_gap_rows_v525": int(
            requirement_matrix[
                "dispatch_requirement_gap_open_v525"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_control_rows_v525": len(controls),
        "active_dispatch_evidence_escalation_control_rows_v525": int(
            controls["control_active_v525"].astype(bool).sum()
        ),
        "blocking_dispatch_evidence_escalation_rows_v525": int(
            controls[
                "blocks_dispatch_evidence_escalation_completion_v525"
            ].astype(bool).sum()
        ),
        "dispatch_evidence_escalation_followup_queue_rows_v525": len(
            followup_queue
        ),
        "dispatch_evidence_escalation_followup_audit_ready_rows_v525": int(
            followup_queue["followup_audit_ready_v525"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v525": int(
            packet["eligibility_review_allowed_v525"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v525": int(
            packet["reviewer_assignment_allowed_v525"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v525": int(
            packet["outcome_capture_allowed_v525"].astype(bool).sum()
        ),
        "patch_allowed_rows_v525": int(
            packet["patch_allowed_v525"].astype(bool).sum()
        ),
        "readiness_delta_rows_v525": len(readiness),
        "dispatch_evidence_escalation_followup_audit_ready_v525": True,
        "ready_for_quarto_patch_v525": False,
        "quarto_patch_applied_v525": False,
        "book_sources_modified_v525": False,
        "book_references_modified_v525": False,
        "submission_ready_claim_allowed_v525": False,
        "working_champion_claim_allowed_v525": False,
        "paper1_promotion_allowed_v525": False,
        "paper4_working_champion_changed_v525": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v525": NEXT_ARTIFACT,
        "claim_boundary": (
            "v525 creates dispatch evidence escalation packet only; escalation "
            "dispatch, external dispatch, dispatch evidence receipt, input receipt, "
            "candidate resolution, nominations, assignments, outcomes, captions, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v525 must not create final Paper 4 promotion.")
    if status["dispatch_evidence_escalation_dispatched_rows_v525"] != 0:
        raise RuntimeError("v525 must not dispatch escalation packets.")
    if status["external_dispatch_recorded_rows_v525"] != 0:
        raise RuntimeError("v525 must not record external dispatch.")
    if status["dispatch_evidence_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive dispatch evidence.")
    if status["dispatch_delivery_trace_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive delivery traces.")
    if status["dispatch_timestamp_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive dispatch timestamps.")
    if status["dispatch_recipient_ack_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive dispatch acknowledgements.")
    if status["human_response_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive human responses.")
    if status["candidate_identifier_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive nomination signoff.")
    if status["evidence_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive candidate evidence.")
    if status["candidate_input_collection_closed_rows_v525"] != 0:
        raise RuntimeError("v525 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v525"] != 0:
        raise RuntimeError("v525 must not record candidate nominations.")
    if status["field_value_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive field values.")
    if status["field_evidence_received_rows_v525"] != 0:
        raise RuntimeError("v525 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v525"] != 0:
        raise RuntimeError("v525 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v525"] != 0:
        raise RuntimeError("v525 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v525"] != 0:
        raise RuntimeError("v525 must not allow outcome capture.")
    if status["patch_allowed_rows_v525"] != 0:
        raise RuntimeError("v525 must not approve a Quarto patch.")

    ESCALATION_MD.write_text(_escalation_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v525": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
