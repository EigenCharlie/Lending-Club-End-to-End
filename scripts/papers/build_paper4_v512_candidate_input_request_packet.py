#!/usr/bin/env python3
"""Build Paper 4 v512 candidate input request packet artifacts."""

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

VERSION = 512
PRIOR_POST_ENTRY_AUDIT_VERSION = 511
NEXT_ARTIFACT = "paper4_v513_candidate_input_receipt_audit.md"
REQUEST_MD = NOTEBOOK.parent / "paper4_v512_candidate_input_request_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_input_request_packet(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in audit.iterrows():
        rows.append(
            {
                "candidate_input_request_id_v512": row["post_entry_audit_id_v511"],
                "priority_v512": int(row["priority_v511"]),
                "review_domain_v512": row["review_domain_v511"],
                "reviewer_role_required_v512": row["reviewer_role_required_v511"],
                "candidate_identifier_request_created_v512": True,
                "nomination_field_request_created_v512": True,
                "nomination_signoff_request_created_v512": True,
                "evidence_request_created_v512": True,
                "candidate_identifier_received_v512": False,
                "nomination_fields_received_v512": False,
                "nomination_signoff_received_v512": False,
                "candidate_nomination_recorded_v512": False,
                "eligibility_review_allowed_v512": False,
                "reviewer_assignment_allowed_v512": False,
                "outcome_capture_allowed_v512": False,
                "patch_allowed_v512": False,
                "input_request_status_v512": "pending_human_input",
                "claim_boundary_v512": "candidate input request packet only",
            }
        )
    return pd.DataFrame(rows)


def _candidate_input_field_request_matrix(field_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in field_audit.iterrows():
        rows.append(
            {
                "candidate_input_request_id_v512": row["post_entry_audit_id_v511"],
                "nomination_field_v512": row["nomination_field_v511"],
                "field_required_v512": bool(row["field_required_v511"]),
                "field_request_created_v512": True,
                "field_value_received_v512": False,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "completion_gap_open_v512": True,
                "claim_boundary_v512": "candidate input field request only",
            }
        )
    return pd.DataFrame(rows)


def _evidence_requirement_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_requirement_id_v512": "candidate_identifier_source",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "candidate identifier and source note",
            },
            {
                "evidence_requirement_id_v512": "reviewer_identity_source",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "reviewer identity provenance",
            },
            {
                "evidence_requirement_id_v512": "nomination_field_values",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "complete nomination field values",
            },
            {
                "evidence_requirement_id_v512": "nomination_signoff",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "timestamped nomination signoff",
            },
            {
                "evidence_requirement_id_v512": "conflict_screening_basis",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "conflict screening basis",
            },
            {
                "evidence_requirement_id_v512": "human_approval_trace",
                "requirement_active_v512": True,
                "evidence_required_v512": True,
                "evidence_received_v512": False,
                "required_evidence_v512": "human approval trace",
            },
        ]
    )


def _input_request_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_request_control_id_v512": "no_candidate_identifier_received",
                "control_active_v512": True,
                "blocks_input_receipt_v512": True,
                "control_result_v512": "candidate identifiers remain unreceived",
            },
            {
                "input_request_control_id_v512": "no_nomination_fields_received",
                "control_active_v512": True,
                "blocks_input_receipt_v512": True,
                "control_result_v512": "nomination fields remain unreceived",
            },
            {
                "input_request_control_id_v512": "no_nomination_signoff_received",
                "control_active_v512": True,
                "blocks_input_receipt_v512": True,
                "control_result_v512": "nomination signoff remains unreceived",
            },
            {
                "input_request_control_id_v512": "no_evidence_received",
                "control_active_v512": True,
                "blocks_input_receipt_v512": True,
                "control_result_v512": "supporting evidence remains unreceived",
            },
            {
                "input_request_control_id_v512": "eligibility_review_blocked",
                "control_active_v512": True,
                "blocks_input_receipt_v512": False,
                "control_result_v512": "eligibility review remains blocked",
            },
            {
                "input_request_control_id_v512": "no_final_promotion",
                "control_active_v512": True,
                "blocks_input_receipt_v512": False,
                "control_result_v512": "final promotion artifact remains absent",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v512": "candidate_input_request_packet_created",
                "ready_v512": True,
                "evidence_artifact_v512": (
                    "paper4_v512_candidate_input_request_packet.csv"
                ),
                "claim_boundary_v512": "candidate input request only",
            },
            {
                "readiness_gate_v512": "candidate_input_field_request_matrix_created",
                "ready_v512": True,
                "evidence_artifact_v512": (
                    "paper4_v512_candidate_input_field_request_matrix.csv"
                ),
                "claim_boundary_v512": "field request matrix only",
            },
            {
                "readiness_gate_v512": "evidence_requirement_register_created",
                "ready_v512": True,
                "evidence_artifact_v512": (
                    "paper4_v512_evidence_requirement_register.csv"
                ),
                "claim_boundary_v512": "evidence requirement register only",
            },
            {
                "readiness_gate_v512": "candidate_input_receipt_audit_ready",
                "ready_v512": True,
                "evidence_artifact_v512": (
                    "paper4_v512_input_request_control_register.csv"
                ),
                "claim_boundary_v512": "future input receipt audit readiness only",
            },
            {
                "readiness_gate_v512": "candidate_identifiers_received",
                "ready_v512": False,
                "evidence_artifact_v512": "candidate identifiers remain unreceived",
                "claim_boundary_v512": "no candidate identifiers received",
            },
            {
                "readiness_gate_v512": "candidate_nominations_recorded",
                "ready_v512": False,
                "evidence_artifact_v512": "candidate nominations remain absent",
                "claim_boundary_v512": "no candidates nominated",
            },
            {
                "readiness_gate_v512": "ready_for_quarto_patch",
                "ready_v512": False,
                "evidence_artifact_v512": "candidate inputs remain absent",
                "claim_boundary_v512": "patch remains blocked",
            },
            {
                "readiness_gate_v512": "paper4_final_promotion_created",
                "ready_v512": False,
                "evidence_artifact_v512": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v512": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v512_candidate_input_request_packet_created",
                "allowed": True,
                "artifact": "paper4_v512_candidate_input_request_packet.csv",
                "boundary": "candidate input request only",
            },
            {
                "claim_id": "v512_evidence_requirements_declared",
                "allowed": True,
                "artifact": "paper4_v512_evidence_requirement_register.csv",
                "boundary": "evidence requirements declared only",
            },
            {
                "claim_id": "v512_input_receipt_audit_ready",
                "allowed": True,
                "artifact": "paper4_v512_input_request_control_register.csv",
                "boundary": "future input receipt audit readiness only",
            },
            {
                "claim_id": "v512_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v512_candidate_input_request_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v512_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v512_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v512_final_promotion",
                "allowed": False,
                "artifact": "paper4_v512_manuscript_readiness_delta.csv",
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
                "claim": "v512 creates a candidate input request packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_candidate_input_request_packet.csv"
                ),
                "boundary": "Candidate input request only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v512 declares candidate evidence requirements.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_evidence_requirement_register.csv"
                ),
                "boundary": "Evidence requirements only; no evidence received.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v512 makes candidate input receipt audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_input_request_control_register.csv"
                ),
                "boundary": "Future input receipt audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v512 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_candidate_input_request_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v512 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v512 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v512_manuscript_readiness_delta.csv"
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
                "executable_item": "v512 creates candidate input request packet.",
                "status": "candidate_input_request_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v513 audits candidate input receipt",
                "last_wave": "v512",
                "execution_result": "candidate_input_request_created_without_inputs",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v512")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _request_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Input Request Packet v512

Generated: {status["generated_at_utc"]}

## Result

v512 creates a candidate input request packet, field request matrix and evidence
requirement register. It does not receive candidate identifiers, receive
nomination fields, receive signoff, record candidate nominations, start
eligibility review, assign reviewers or allow outcome capture.

## Counts

- Input request rows: `{status["input_request_rows_v512"]}`.
- Candidate identifier request rows: `{status["candidate_identifier_request_rows_v512"]}`.
- Nomination field request rows: `{status["nomination_field_request_rows_v512"]}`.
- Nomination signoff request rows: `{status["nomination_signoff_request_rows_v512"]}`.
- Evidence request rows: `{status["evidence_request_rows_v512"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v512"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v512"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v512"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v512"]}`.
- Input field request rows: `{status["input_field_request_rows_v512"]}`.
- Field value received rows: `{status["field_value_received_rows_v512"]}`.
- Evidence requirement rows: `{status["evidence_requirement_rows_v512"]}`.
- Active evidence requirement rows: `{status["active_evidence_requirement_rows_v512"]}`.
- Input request control rows: `{status["input_request_control_rows_v512"]}`.
- Active input request control rows: `{status["active_input_request_control_rows_v512"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v512"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v512"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v512"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v512"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v512"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v512 is an input-request packet only. It does not receive candidate inputs,
resolve or nominate candidates, assign reviewers, capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V512_CANDIDATE_INPUT_REQUEST_PACKET_START -->"
    end = "<!-- V512_CANDIDATE_INPUT_REQUEST_PACKET_END -->"
    block = f"""
{start}

## Wave v512: Candidate Input Request Packet

Generated: {status["generated_at_utc"]}

### Objective

v512 creates the request surface for candidate inputs and evidence. It asks for
candidate identifiers and nomination fields, but it does not receive or record
them.

### Results

- Input request rows:
  `{status["input_request_rows_v512"]}`.
- Candidate identifier request rows:
  `{status["candidate_identifier_request_rows_v512"]}`.
- Nomination field request rows:
  `{status["nomination_field_request_rows_v512"]}`.
- Nomination signoff request rows:
  `{status["nomination_signoff_request_rows_v512"]}`.
- Evidence request rows:
  `{status["evidence_request_rows_v512"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v512"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v512"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v512"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v512"]}`.
- Input field request rows:
  `{status["input_field_request_rows_v512"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v512"]}`.
- Evidence requirement rows:
  `{status["evidence_requirement_rows_v512"]}`.
- Active evidence requirement rows:
  `{status["active_evidence_requirement_rows_v512"]}`.
- Input request control rows:
  `{status["input_request_control_rows_v512"]}`.
- Active input request control rows:
  `{status["active_input_request_control_rows_v512"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v512"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v512"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v512"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v512"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v512"]}`.
- Book sources modified:
  `{status["book_sources_modified_v512"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v512"]}`.

### Interpretation

The candidate-resolution track now has an explicit input request and evidence
contract. The request itself is not evidence of receipt, nomination or reviewer
assignment.

### Claim Impact

- Allowed: candidate input request creation, field request matrix, evidence
  requirement register and future input receipt audit readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v512 in the living notebook. v513 should audit whether requested candidate
inputs were actually received, and must keep all candidate claims blocked if no
evidence is present.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v511 = _read_status(PRIOR_POST_ENTRY_AUDIT_VERSION)
    expected_next = "paper4_v512_candidate_input_request_packet.md"
    if v511["next_artifact_v511"] != expected_next:
        raise RuntimeError("v512 expects v511 to route to input request packet.")
    if not v511["candidate_input_request_packet_ready_v511"]:
        raise RuntimeError("v512 requires v511 input request readiness.")

    audit = pd.read_csv(
        TABLE_DIR / "paper4_v511_post_entry_candidate_resolution_audit.csv"
    )
    field_audit = pd.read_csv(
        TABLE_DIR / "paper4_v511_post_entry_field_completion_audit.csv"
    )
    packet = _candidate_input_request_packet(audit)
    field_requests = _candidate_input_field_request_matrix(field_audit)
    evidence = _evidence_requirement_register()
    controls = _input_request_control_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v512_candidate_input_request_packet.csv", packet)
    write_csv(
        TABLE_DIR / "paper4_v512_candidate_input_field_request_matrix.csv",
        field_requests,
    )
    write_csv(TABLE_DIR / "paper4_v512_evidence_requirement_register.csv", evidence)
    write_csv(TABLE_DIR / "paper4_v512_input_request_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v512_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v512_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v512_candidate_input_request_packet",
        "schema_version": "2026-05-17.512",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_entry_audit_version_v512": PRIOR_POST_ENTRY_AUDIT_VERSION,
        "candidate_input_request_packet_created_v512": True,
        "input_request_rows_v512": len(packet),
        "candidate_identifier_request_rows_v512": int(
            packet["candidate_identifier_request_created_v512"].astype(bool).sum()
        ),
        "nomination_field_request_rows_v512": int(
            packet["nomination_field_request_created_v512"].astype(bool).sum()
        ),
        "nomination_signoff_request_rows_v512": int(
            packet["nomination_signoff_request_created_v512"].astype(bool).sum()
        ),
        "evidence_request_rows_v512": int(
            packet["evidence_request_created_v512"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v512": int(
            packet["candidate_identifier_received_v512"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v512": int(
            packet["nomination_fields_received_v512"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v512": int(
            packet["nomination_signoff_received_v512"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v512": int(
            packet["candidate_nomination_recorded_v512"].astype(bool).sum()
        ),
        "input_field_request_rows_v512": len(field_requests),
        "field_value_received_rows_v512": int(
            field_requests["field_value_received_v512"].astype(bool).sum()
        ),
        "evidence_requirement_rows_v512": len(evidence),
        "active_evidence_requirement_rows_v512": int(
            evidence["requirement_active_v512"].astype(bool).sum()
        ),
        "input_request_control_rows_v512": len(controls),
        "active_input_request_control_rows_v512": int(
            controls["control_active_v512"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v512": int(
            packet["eligibility_review_allowed_v512"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v512": int(
            packet["reviewer_assignment_allowed_v512"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v512": int(
            packet["outcome_capture_allowed_v512"].astype(bool).sum()
        ),
        "patch_allowed_rows_v512": int(packet["patch_allowed_v512"].astype(bool).sum()),
        "readiness_delta_rows_v512": len(readiness),
        "candidate_input_receipt_audit_ready_v512": True,
        "ready_for_quarto_patch_v512": False,
        "quarto_patch_applied_v512": False,
        "book_sources_modified_v512": False,
        "book_references_modified_v512": False,
        "submission_ready_claim_allowed_v512": False,
        "working_champion_claim_allowed_v512": False,
        "paper1_promotion_allowed_v512": False,
        "paper4_working_champion_changed_v512": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v512": NEXT_ARTIFACT,
        "claim_boundary": (
            "v512 creates candidate input requests only; input receipt, "
            "candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v512 must not create final Paper 4 promotion.")
    if status["candidate_identifier_received_rows_v512"] != 0:
        raise RuntimeError("v512 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v512"] != 0:
        raise RuntimeError("v512 must not receive nomination fields.")
    if status["candidate_nomination_recorded_rows_v512"] != 0:
        raise RuntimeError("v512 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v512"] != 0:
        raise RuntimeError("v512 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v512"] != 0:
        raise RuntimeError("v512 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v512"] != 0:
        raise RuntimeError("v512 must not allow outcome capture.")
    if status["patch_allowed_rows_v512"] != 0:
        raise RuntimeError("v512 must not approve a Quarto patch.")

    REQUEST_MD.write_text(_request_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v512": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
