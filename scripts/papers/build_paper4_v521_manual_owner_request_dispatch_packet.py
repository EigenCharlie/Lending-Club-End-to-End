#!/usr/bin/env python3
"""Build Paper 4 v521 manual owner request dispatch packet artifacts."""

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

VERSION = 521
PRIOR_MANUAL_OWNER_FOLLOWUP_AUDIT_VERSION = 520
NEXT_ARTIFACT = "paper4_v522_manual_owner_dispatch_followup_audit.md"
DISPATCH_MD = NOTEBOOK.parent / "paper4_v521_manual_owner_request_dispatch_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_owner_request_dispatch_packet(followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in followup.iterrows():
        rows.append(
            {
                "manual_owner_dispatch_id_v521": row[
                    "manual_owner_followup_audit_id_v520"
                ],
                "priority_v521": int(row["priority_v520"]),
                "review_domain_v521": row["review_domain_v520"],
                "reviewer_role_required_v521": row["reviewer_role_required_v520"],
                "manual_owner_request_created_v521": bool(
                    row["manual_owner_request_created_v520"]
                ),
                "dispatch_packet_created_v521": True,
                "dispatch_ready_v521": True,
                "manual_owner_request_dispatched_v521": bool(
                    row["manual_owner_request_dispatched_v520"]
                ),
                "human_response_received_v521": bool(
                    row["human_response_received_v520"]
                ),
                "candidate_identifier_received_v521": bool(
                    row["candidate_identifier_received_v520"]
                ),
                "nomination_fields_received_v521": bool(
                    row["nomination_fields_received_v520"]
                ),
                "nomination_signoff_received_v521": bool(
                    row["nomination_signoff_received_v520"]
                ),
                "evidence_received_v521": bool(row["evidence_received_v520"]),
                "candidate_input_collection_closed_v521": False,
                "candidate_nomination_recorded_v521": bool(
                    row["candidate_nomination_recorded_v520"]
                ),
                "eligibility_review_allowed_v521": False,
                "reviewer_assignment_allowed_v521": False,
                "outcome_capture_allowed_v521": False,
                "patch_allowed_v521": False,
                "dispatch_packet_status_v521": (
                    "dispatch_packet_ready_external_dispatch_not_recorded"
                ),
                "required_next_step_v521": "audit_manual_owner_dispatch_followup",
                "claim_boundary_v521": (
                    "manual owner request dispatch packet only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_dispatch_checklist(
    field_followup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_followup.iterrows():
        field_received = bool(row["field_value_received_v520"])
        evidence_received = bool(row["field_evidence_received_v520"])
        rows.append(
            {
                "manual_owner_dispatch_id_v521": row[
                    "manual_owner_followup_audit_id_v520"
                ],
                "nomination_field_v521": row["nomination_field_v520"],
                "field_request_created_v521": bool(
                    row["manual_owner_field_request_created_v520"]
                ),
                "evidence_request_created_v521": bool(
                    row["manual_owner_evidence_request_created_v520"]
                ),
                "field_dispatch_checklist_created_v521": True,
                "evidence_dispatch_checklist_created_v521": True,
                "field_value_received_v521": field_received,
                "field_evidence_received_v521": evidence_received,
                "field_evidence_dispatch_gap_open_v521": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v521": (
                    "field and evidence dispatch checklist only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _dispatch_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_control_id_v521": "external_dispatch_not_recorded",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": True,
                "required_resolution_v521": (
                    "record externally verified manual owner request dispatch"
                ),
            },
            {
                "dispatch_control_id_v521": "manual_owner_response_absent",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": True,
                "required_resolution_v521": (
                    "receive manual owner response after verified dispatch"
                ),
            },
            {
                "dispatch_control_id_v521": "candidate_identifier_absent",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": True,
                "required_resolution_v521": "receive candidate identifier",
            },
            {
                "dispatch_control_id_v521": "nomination_payload_absent",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": True,
                "required_resolution_v521": (
                    "receive nomination fields and nomination signoff"
                ),
            },
            {
                "dispatch_control_id_v521": "evidence_absent",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": True,
                "required_resolution_v521": "receive supporting evidence",
            },
            {
                "dispatch_control_id_v521": "no_final_promotion",
                "control_active_v521": True,
                "blocks_dispatch_followup_completion_v521": False,
                "required_resolution_v521": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v521": "manual_owner_request_dispatch_packet_created",
                "ready_v521": True,
                "evidence_artifact_v521": (
                    "paper4_v521_manual_owner_request_dispatch_packet.csv"
                ),
                "claim_boundary_v521": "manual owner request dispatch packet only",
            },
            {
                "readiness_gate_v521": (
                    "field_evidence_dispatch_checklist_created"
                ),
                "ready_v521": True,
                "evidence_artifact_v521": (
                    "paper4_v521_field_evidence_dispatch_checklist.csv"
                ),
                "claim_boundary_v521": "field evidence dispatch checklist only",
            },
            {
                "readiness_gate_v521": "dispatch_control_register_created",
                "ready_v521": True,
                "evidence_artifact_v521": (
                    "paper4_v521_dispatch_control_register.csv"
                ),
                "claim_boundary_v521": "dispatch control register only",
            },
            {
                "readiness_gate_v521": "manual_owner_dispatch_followup_audit_ready",
                "ready_v521": True,
                "evidence_artifact_v521": (
                    "paper4_v521_dispatch_control_register.csv"
                ),
                "claim_boundary_v521": (
                    "future manual owner dispatch follow-up audit readiness only"
                ),
            },
            {
                "readiness_gate_v521": "candidate_identifiers_received",
                "ready_v521": False,
                "evidence_artifact_v521": "candidate identifiers remain unreceived",
                "claim_boundary_v521": "no candidate identifiers received",
            },
            {
                "readiness_gate_v521": "candidate_nominations_recorded",
                "ready_v521": False,
                "evidence_artifact_v521": "candidate nominations remain absent",
                "claim_boundary_v521": "no candidates nominated",
            },
            {
                "readiness_gate_v521": "ready_for_quarto_patch",
                "ready_v521": False,
                "evidence_artifact_v521": "candidate inputs remain absent",
                "claim_boundary_v521": "patch remains blocked",
            },
            {
                "readiness_gate_v521": "paper4_final_promotion_created",
                "ready_v521": False,
                "evidence_artifact_v521": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v521": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v521_manual_owner_request_dispatch_packet_created",
                "allowed": True,
                "artifact": "paper4_v521_manual_owner_request_dispatch_packet.csv",
                "boundary": "manual owner request dispatch packet only",
            },
            {
                "claim_id": "v521_field_evidence_dispatch_checklist_created",
                "allowed": True,
                "artifact": "paper4_v521_field_evidence_dispatch_checklist.csv",
                "boundary": "field evidence dispatch checklist only",
            },
            {
                "claim_id": "v521_manual_owner_dispatch_followup_audit_ready",
                "allowed": True,
                "artifact": "paper4_v521_dispatch_control_register.csv",
                "boundary": "future manual owner dispatch follow-up audit readiness only",
            },
            {
                "claim_id": "v521_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v521_manual_owner_request_dispatch_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v521_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v521_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v521_final_promotion",
                "allowed": False,
                "artifact": "paper4_v521_manuscript_readiness_delta.csv",
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
                "claim": "v521 creates a manual owner request dispatch packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_manual_owner_request_dispatch_packet.csv"
                ),
                "boundary": (
                    "Dispatch packet only; external dispatch is not recorded."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v521 creates a field and evidence dispatch checklist.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_field_evidence_dispatch_checklist.csv"
                ),
                "boundary": "Field and evidence dispatch checklist only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v521 makes manual owner dispatch follow-up audit executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_dispatch_control_register.csv"
                ),
                "boundary": (
                    "Future manual owner dispatch follow-up audit readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v521 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_manual_owner_request_dispatch_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v521 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v521 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v521_manuscript_readiness_delta.csv"
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
                "executable_item": "v521 creates manual owner request dispatch packet.",
                "status": "manual_owner_request_dispatch_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v522 audits manual owner dispatch follow-up",
                "last_wave": "v521",
                "execution_result": (
                    "manual_owner_request_dispatch_packet_created_without_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v521")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _dispatch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Owner Request Dispatch Packet v521

Generated: {status["generated_at_utc"]}

## Result

v521 creates the bounded manual owner request dispatch packet after v520 found
no recorded external dispatch. The packet is ready for external execution, but
no dispatch, response, candidate identifier, nomination payload or evidence is
recorded in v521.

## Counts

- Manual owner dispatch packet rows: `{status["manual_owner_dispatch_packet_rows_v521"]}`.
- Dispatch packet created rows: `{status["dispatch_packet_created_rows_v521"]}`.
- Dispatch ready rows: `{status["dispatch_ready_rows_v521"]}`.
- Manual owner request dispatched rows: `{status["manual_owner_request_dispatched_rows_v521"]}`.
- Human response received rows: `{status["human_response_received_rows_v521"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v521"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v521"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v521"]}`.
- Evidence received rows: `{status["evidence_received_rows_v521"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v521"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v521"]}`.
- Field/evidence dispatch checklist rows: `{status["field_evidence_dispatch_checklist_rows_v521"]}`.
- Field dispatch checklist created rows: `{status["field_dispatch_checklist_created_rows_v521"]}`.
- Evidence dispatch checklist created rows: `{status["evidence_dispatch_checklist_created_rows_v521"]}`.
- Field value received rows: `{status["field_value_received_rows_v521"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v521"]}`.
- Open field/evidence dispatch gap rows: `{status["open_field_evidence_dispatch_gap_rows_v521"]}`.
- Dispatch control rows: `{status["dispatch_control_rows_v521"]}`.
- Active dispatch control rows: `{status["active_dispatch_control_rows_v521"]}`.
- Blocking dispatch control rows: `{status["blocking_dispatch_control_rows_v521"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v521"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v521"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v521"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v521"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v521"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v521 is a dispatch packet only. It does not record external dispatch, receive
candidate inputs, resolve or nominate candidates, assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V521_MANUAL_OWNER_REQUEST_DISPATCH_PACKET_START -->"
    end = "<!-- V521_MANUAL_OWNER_REQUEST_DISPATCH_PACKET_END -->"
    block = f"""
{start}

## Wave v521: Manual Owner Request Dispatch Packet

Generated: {status["generated_at_utc"]}

### Objective

v521 prepares a bounded manual owner request dispatch packet after v520 found no
recorded dispatch. It creates reproducible dispatch artifacts only; it does not
record external delivery, human response or candidate input receipt.

### Results

- Manual owner dispatch packet rows:
  `{status["manual_owner_dispatch_packet_rows_v521"]}`.
- Dispatch packet created rows:
  `{status["dispatch_packet_created_rows_v521"]}`.
- Dispatch ready rows:
  `{status["dispatch_ready_rows_v521"]}`.
- Manual owner request dispatched rows:
  `{status["manual_owner_request_dispatched_rows_v521"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v521"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v521"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v521"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v521"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v521"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v521"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v521"]}`.
- Field/evidence dispatch checklist rows:
  `{status["field_evidence_dispatch_checklist_rows_v521"]}`.
- Field dispatch checklist created rows:
  `{status["field_dispatch_checklist_created_rows_v521"]}`.
- Evidence dispatch checklist created rows:
  `{status["evidence_dispatch_checklist_created_rows_v521"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v521"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v521"]}`.
- Open field/evidence dispatch gap rows:
  `{status["open_field_evidence_dispatch_gap_rows_v521"]}`.
- Dispatch control rows:
  `{status["dispatch_control_rows_v521"]}`.
- Active dispatch control rows:
  `{status["active_dispatch_control_rows_v521"]}`.
- Blocking dispatch control rows:
  `{status["blocking_dispatch_control_rows_v521"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v521"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v521"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v521"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v521"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v521"]}`.
- Book sources modified:
  `{status["book_sources_modified_v521"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v521"]}`.

### Interpretation

The dispatch packet is a reproducible preparation artifact. Because external
dispatch and response counters remain zero, the next executable step is a
dispatch follow-up audit, not candidate nomination, eligibility review or
manuscript patching.

### Claim Impact

- Allowed: manual owner request dispatch packet, field/evidence dispatch
  checklist and future manual owner dispatch follow-up audit readiness.
- Still prohibited: external dispatch completion, candidate input receipt,
  candidate resolution/nomination, reviewer assignment, completed review claims,
  final captions, Quarto patch readiness/application, Quarto/book mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v521 in the living notebook. v522 should audit manual owner dispatch
follow-up while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v520 = _read_status(PRIOR_MANUAL_OWNER_FOLLOWUP_AUDIT_VERSION)
    expected_next = "paper4_v521_manual_owner_request_dispatch_packet.md"
    if v520["next_artifact_v520"] != expected_next:
        raise RuntimeError("v521 expects v520 to route to dispatch packet.")
    if not v520["manual_owner_request_dispatch_packet_ready_v520"]:
        raise RuntimeError("v521 requires v520 dispatch packet readiness.")

    followup = pd.read_csv(
        TABLE_DIR / "paper4_v520_manual_owner_escalation_followup_audit.csv"
    )
    field_followup = pd.read_csv(
        TABLE_DIR / "paper4_v520_field_evidence_manual_owner_followup_audit.csv"
    )
    packet = _manual_owner_request_dispatch_packet(followup)
    checklist = _field_evidence_dispatch_checklist(field_followup)
    controls = _dispatch_control_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v521_manual_owner_request_dispatch_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v521_field_evidence_dispatch_checklist.csv",
        checklist,
    )
    write_csv(TABLE_DIR / "paper4_v521_dispatch_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v521_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v521_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v521_manual_owner_request_dispatch_packet",
        "schema_version": "2026-05-17.521",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_owner_followup_audit_version_v521": (
            PRIOR_MANUAL_OWNER_FOLLOWUP_AUDIT_VERSION
        ),
        "manual_owner_request_dispatch_packet_created_v521": True,
        "manual_owner_dispatch_packet_rows_v521": len(packet),
        "dispatch_packet_created_rows_v521": int(
            packet["dispatch_packet_created_v521"].astype(bool).sum()
        ),
        "dispatch_ready_rows_v521": int(
            packet["dispatch_ready_v521"].astype(bool).sum()
        ),
        "manual_owner_request_dispatched_rows_v521": int(
            packet["manual_owner_request_dispatched_v521"].astype(bool).sum()
        ),
        "human_response_received_rows_v521": int(
            packet["human_response_received_v521"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v521": int(
            packet["candidate_identifier_received_v521"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v521": int(
            packet["nomination_fields_received_v521"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v521": int(
            packet["nomination_signoff_received_v521"].astype(bool).sum()
        ),
        "evidence_received_rows_v521": int(
            packet["evidence_received_v521"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v521": int(
            packet["candidate_input_collection_closed_v521"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v521": int(
            packet["candidate_nomination_recorded_v521"].astype(bool).sum()
        ),
        "field_evidence_dispatch_checklist_rows_v521": len(checklist),
        "field_dispatch_checklist_created_rows_v521": int(
            checklist["field_dispatch_checklist_created_v521"].astype(bool).sum()
        ),
        "evidence_dispatch_checklist_created_rows_v521": int(
            checklist["evidence_dispatch_checklist_created_v521"].astype(bool).sum()
        ),
        "field_value_received_rows_v521": int(
            checklist["field_value_received_v521"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v521": int(
            checklist["field_evidence_received_v521"].astype(bool).sum()
        ),
        "open_field_evidence_dispatch_gap_rows_v521": int(
            checklist["field_evidence_dispatch_gap_open_v521"].astype(bool).sum()
        ),
        "dispatch_control_rows_v521": len(controls),
        "active_dispatch_control_rows_v521": int(
            controls["control_active_v521"].astype(bool).sum()
        ),
        "blocking_dispatch_control_rows_v521": int(
            controls["blocks_dispatch_followup_completion_v521"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v521": int(
            packet["eligibility_review_allowed_v521"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v521": int(
            packet["reviewer_assignment_allowed_v521"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v521": int(
            packet["outcome_capture_allowed_v521"].astype(bool).sum()
        ),
        "patch_allowed_rows_v521": int(packet["patch_allowed_v521"].astype(bool).sum()),
        "readiness_delta_rows_v521": len(readiness),
        "manual_owner_dispatch_followup_audit_ready_v521": True,
        "ready_for_quarto_patch_v521": False,
        "quarto_patch_applied_v521": False,
        "book_sources_modified_v521": False,
        "book_references_modified_v521": False,
        "submission_ready_claim_allowed_v521": False,
        "working_champion_claim_allowed_v521": False,
        "paper1_promotion_allowed_v521": False,
        "paper4_working_champion_changed_v521": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v521": NEXT_ARTIFACT,
        "claim_boundary": (
            "v521 creates manual owner request dispatch packet only; external "
            "dispatch, input receipt, candidate resolution, nominations, "
            "assignments, outcomes, captions, patching, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v521 must not create final Paper 4 promotion.")
    if status["manual_owner_request_dispatched_rows_v521"] != 0:
        raise RuntimeError("v521 must not record external dispatch.")
    if status["human_response_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive human responses.")
    if status["candidate_identifier_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive nomination signoff.")
    if status["evidence_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive evidence.")
    if status["candidate_input_collection_closed_rows_v521"] != 0:
        raise RuntimeError("v521 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v521"] != 0:
        raise RuntimeError("v521 must not record candidate nominations.")
    if status["field_value_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive field values.")
    if status["field_evidence_received_rows_v521"] != 0:
        raise RuntimeError("v521 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v521"] != 0:
        raise RuntimeError("v521 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v521"] != 0:
        raise RuntimeError("v521 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v521"] != 0:
        raise RuntimeError("v521 must not allow outcome capture.")
    if status["patch_allowed_rows_v521"] != 0:
        raise RuntimeError("v521 must not approve a Quarto patch.")

    DISPATCH_MD.write_text(_dispatch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v521": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
