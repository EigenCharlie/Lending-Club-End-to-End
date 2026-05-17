#!/usr/bin/env python3
"""Build Paper 4 v520 manual owner escalation follow-up audit artifacts."""

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

VERSION = 520
PRIOR_MANUAL_OWNER_ESCALATION_REQUEST_VERSION = 519
NEXT_ARTIFACT = "paper4_v521_manual_owner_request_dispatch_packet.md"
FOLLOWUP_MD = NOTEBOOK.parent / "paper4_v520_manual_owner_escalation_followup_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_owner_escalation_followup_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        dispatched = bool(row["manual_owner_request_dispatched_v519"])
        response_received = bool(row["human_response_received_v519"])
        candidate_identifier_received = bool(
            row["candidate_identifier_received_v519"]
        )
        nomination_fields_received = bool(row["nomination_fields_received_v519"])
        nomination_signoff_received = bool(row["nomination_signoff_received_v519"])
        evidence_received = bool(row["evidence_received_v519"])
        complete = (
            dispatched
            and response_received
            and candidate_identifier_received
            and nomination_fields_received
            and nomination_signoff_received
            and evidence_received
        )
        rows.append(
            {
                "manual_owner_followup_audit_id_v520": row[
                    "manual_owner_request_id_v519"
                ],
                "priority_v520": int(row["priority_v519"]),
                "review_domain_v520": row["review_domain_v519"],
                "reviewer_role_required_v520": row["reviewer_role_required_v519"],
                "manual_owner_request_created_v520": bool(
                    row["manual_owner_request_created_v519"]
                ),
                "manual_owner_request_dispatched_v520": dispatched,
                "human_response_received_v520": response_received,
                "candidate_identifier_received_v520": candidate_identifier_received,
                "nomination_fields_received_v520": nomination_fields_received,
                "nomination_signoff_received_v520": nomination_signoff_received,
                "evidence_received_v520": evidence_received,
                "manual_owner_followup_complete_v520": complete,
                "manual_owner_followup_gap_open_v520": not complete,
                "candidate_input_collection_closed_v520": False,
                "candidate_nomination_recorded_v520": bool(
                    row["candidate_nomination_recorded_v519"]
                ),
                "eligibility_review_allowed_v520": False,
                "reviewer_assignment_allowed_v520": False,
                "outcome_capture_allowed_v520": False,
                "patch_allowed_v520": False,
                "required_next_step_v520": (
                    "prepare_manual_owner_request_dispatch_packet"
                ),
                "claim_boundary_v520": "manual owner escalation follow-up audit only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_manual_owner_followup_audit(
    field_requests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in field_requests.iterrows():
        field_received = bool(row["field_value_received_v519"])
        evidence_received = bool(row["field_evidence_received_v519"])
        rows.append(
            {
                "manual_owner_followup_audit_id_v520": row[
                    "manual_owner_request_id_v519"
                ],
                "nomination_field_v520": row["nomination_field_v519"],
                "manual_owner_field_request_created_v520": bool(
                    row["manual_owner_field_request_created_v519"]
                ),
                "manual_owner_evidence_request_created_v520": bool(
                    row["manual_owner_evidence_request_created_v519"]
                ),
                "field_value_received_v520": field_received,
                "field_evidence_received_v520": evidence_received,
                "manual_owner_field_evidence_followup_gap_open_v520": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v520": (
                    "field and evidence manual owner follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_followup_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "manual_owner_followup_blocker_id_v520": (
                    "no_manual_owner_request_dispatch"
                ),
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": True,
                "required_resolution_v520": (
                    "dispatch manual owner request before follow-up can close"
                ),
            },
            {
                "manual_owner_followup_blocker_id_v520": "no_manual_owner_response",
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": True,
                "required_resolution_v520": (
                    "receive manual owner response after dispatch"
                ),
            },
            {
                "manual_owner_followup_blocker_id_v520": (
                    "no_candidate_identifier_after_manual_owner_request"
                ),
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": True,
                "required_resolution_v520": (
                    "receive candidate identifier from manual owner path"
                ),
            },
            {
                "manual_owner_followup_blocker_id_v520": (
                    "no_nomination_payload_after_manual_owner_request"
                ),
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": True,
                "required_resolution_v520": (
                    "receive nomination fields and signoff from manual owner path"
                ),
            },
            {
                "manual_owner_followup_blocker_id_v520": (
                    "no_evidence_after_manual_owner_request"
                ),
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": True,
                "required_resolution_v520": (
                    "receive supporting evidence from manual owner path"
                ),
            },
            {
                "manual_owner_followup_blocker_id_v520": "no_final_promotion",
                "blocker_open_v520": True,
                "blocks_manual_owner_followup_completion_v520": False,
                "required_resolution_v520": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v520": (
                    "manual_owner_escalation_followup_audit_created"
                ),
                "ready_v520": True,
                "evidence_artifact_v520": (
                    "paper4_v520_manual_owner_escalation_followup_audit.csv"
                ),
                "claim_boundary_v520": "manual owner escalation follow-up audit only",
            },
            {
                "readiness_gate_v520": (
                    "field_evidence_manual_owner_followup_audit_created"
                ),
                "ready_v520": True,
                "evidence_artifact_v520": (
                    "paper4_v520_field_evidence_manual_owner_followup_audit.csv"
                ),
                "claim_boundary_v520": "field evidence manual owner audit only",
            },
            {
                "readiness_gate_v520": (
                    "manual_owner_followup_blocker_register_created"
                ),
                "ready_v520": True,
                "evidence_artifact_v520": (
                    "paper4_v520_manual_owner_followup_blocker_register.csv"
                ),
                "claim_boundary_v520": "manual owner follow-up blockers only",
            },
            {
                "readiness_gate_v520": "manual_owner_request_dispatch_packet_ready",
                "ready_v520": True,
                "evidence_artifact_v520": (
                    "paper4_v520_manual_owner_followup_blocker_register.csv"
                ),
                "claim_boundary_v520": (
                    "future manual owner request dispatch packet readiness only"
                ),
            },
            {
                "readiness_gate_v520": "candidate_identifiers_received",
                "ready_v520": False,
                "evidence_artifact_v520": "candidate identifiers remain unreceived",
                "claim_boundary_v520": "no candidate identifiers received",
            },
            {
                "readiness_gate_v520": "candidate_nominations_recorded",
                "ready_v520": False,
                "evidence_artifact_v520": "candidate nominations remain absent",
                "claim_boundary_v520": "no candidates nominated",
            },
            {
                "readiness_gate_v520": "ready_for_quarto_patch",
                "ready_v520": False,
                "evidence_artifact_v520": "candidate inputs remain absent",
                "claim_boundary_v520": "patch remains blocked",
            },
            {
                "readiness_gate_v520": "paper4_final_promotion_created",
                "ready_v520": False,
                "evidence_artifact_v520": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v520": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v520_manual_owner_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v520_manual_owner_escalation_followup_audit.csv",
                "boundary": "manual owner escalation follow-up audit only",
            },
            {
                "claim_id": (
                    "v520_field_evidence_manual_owner_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v520_field_evidence_manual_owner_followup_audit.csv"
                ),
                "boundary": "field evidence manual owner follow-up audit only",
            },
            {
                "claim_id": "v520_manual_owner_request_dispatch_packet_ready",
                "allowed": True,
                "artifact": (
                    "paper4_v520_manual_owner_followup_blocker_register.csv"
                ),
                "boundary": "future manual owner dispatch packet readiness only",
            },
            {
                "claim_id": "v520_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v520_manual_owner_escalation_followup_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v520_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v520_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v520_final_promotion",
                "allowed": False,
                "artifact": "paper4_v520_manuscript_readiness_delta.csv",
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
                "claim": "v520 audits manual owner escalation follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_manual_owner_escalation_followup_audit.csv"
                ),
                "boundary": "Manual owner escalation follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v520 audits manual owner field and evidence follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_field_evidence_manual_owner_followup_audit.csv"
                ),
                "boundary": "Manual owner field and evidence follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v520 makes manual owner request dispatch packet executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_manual_owner_followup_blocker_register.csv"
                ),
                "boundary": (
                    "Future manual owner request dispatch packet readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v520 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_manual_owner_escalation_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v520 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v520 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v520_manuscript_readiness_delta.csv"
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
                "executable_item": "v520 audits manual owner escalation follow-up.",
                "status": "manual_owner_escalation_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v521 creates manual owner request dispatch packet",
                "last_wave": "v520",
                "execution_result": (
                    "manual_owner_escalation_followup_audit_confirmed_no_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v520")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Owner Escalation Follow-up Audit v520

Generated: {status["generated_at_utc"]}

## Result

v520 audits the v519 manual owner escalation request packet. The manual owner
request rows exist, but none have been dispatched and no human responses,
candidate identifiers, nomination fields, signoffs or evidence have been
received. The follow-up remains open and the next executable step is a bounded
manual owner request dispatch packet.

## Counts

- Manual owner follow-up audit rows: `{status["manual_owner_followup_audit_rows_v520"]}`.
- Manual owner request created rows: `{status["manual_owner_request_created_rows_v520"]}`.
- Manual owner request dispatched rows: `{status["manual_owner_request_dispatched_rows_v520"]}`.
- Human response received rows: `{status["human_response_received_rows_v520"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v520"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v520"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v520"]}`.
- Evidence received rows: `{status["evidence_received_rows_v520"]}`.
- Manual owner follow-up complete rows: `{status["manual_owner_followup_complete_rows_v520"]}`.
- Open manual owner follow-up gap rows: `{status["open_manual_owner_followup_gap_rows_v520"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v520"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v520"]}`.
- Field/evidence manual owner follow-up audit rows: `{status["field_evidence_manual_owner_followup_audit_rows_v520"]}`.
- Field request created rows: `{status["field_request_created_rows_v520"]}`.
- Evidence request created rows: `{status["evidence_request_created_rows_v520"]}`.
- Field value received rows: `{status["field_value_received_rows_v520"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v520"]}`.
- Open field/evidence manual owner follow-up gap rows: `{status["open_field_evidence_manual_owner_followup_gap_rows_v520"]}`.
- Manual owner follow-up blocker rows: `{status["manual_owner_followup_blocker_rows_v520"]}`.
- Open manual owner follow-up blocker rows: `{status["open_manual_owner_followup_blocker_rows_v520"]}`.
- Blocking manual owner follow-up rows: `{status["blocking_manual_owner_followup_rows_v520"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v520"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v520"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v520"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v520"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v520"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v520 is a manual owner escalation follow-up audit only. It does not dispatch
requests, receive candidate inputs, resolve or nominate candidates, assign
reviewers, capture completed review outcomes, finalize captions, approve patch
scope, edit Quarto, render the book, make Paper 4 submission-ready, replace
Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V520_MANUAL_OWNER_ESCALATION_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V520_MANUAL_OWNER_ESCALATION_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v520: Manual Owner Escalation Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v520 audits whether the v519 manual owner escalation request packet has been
dispatched or answered. It confirms that request rows exist but no dispatch,
response or candidate input has been recorded.

### Results

- Manual owner follow-up audit rows:
  `{status["manual_owner_followup_audit_rows_v520"]}`.
- Manual owner request created rows:
  `{status["manual_owner_request_created_rows_v520"]}`.
- Manual owner request dispatched rows:
  `{status["manual_owner_request_dispatched_rows_v520"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v520"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v520"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v520"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v520"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v520"]}`.
- Manual owner follow-up complete rows:
  `{status["manual_owner_followup_complete_rows_v520"]}`.
- Open manual owner follow-up gap rows:
  `{status["open_manual_owner_followup_gap_rows_v520"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v520"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v520"]}`.
- Field/evidence manual owner follow-up audit rows:
  `{status["field_evidence_manual_owner_followup_audit_rows_v520"]}`.
- Field request created rows:
  `{status["field_request_created_rows_v520"]}`.
- Evidence request created rows:
  `{status["evidence_request_created_rows_v520"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v520"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v520"]}`.
- Open field/evidence manual owner follow-up gap rows:
  `{status["open_field_evidence_manual_owner_followup_gap_rows_v520"]}`.
- Manual owner follow-up blocker rows:
  `{status["manual_owner_followup_blocker_rows_v520"]}`.
- Open manual owner follow-up blocker rows:
  `{status["open_manual_owner_followup_blocker_rows_v520"]}`.
- Blocking manual owner follow-up rows:
  `{status["blocking_manual_owner_followup_rows_v520"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v520"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v520"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v520"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v520"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v520"]}`.
- Book sources modified:
  `{status["book_sources_modified_v520"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v520"]}`.

### Interpretation

The follow-up audit confirms that manual owner escalation has not moved beyond
request packet creation. The next executable step is a dispatch packet, not
candidate nomination, eligibility review or manuscript patching.

### Claim Impact

- Allowed: manual owner escalation follow-up audit, field/evidence follow-up
  audit and future manual owner request dispatch packet readiness.
- Still prohibited: request dispatch claims, candidate input receipt, candidate
  resolution/nomination, reviewer assignment, completed review claims, final
  captions, Quarto patch readiness/application, Quarto/book mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v520 in the living notebook. v521 should create a bounded manual owner
request dispatch packet while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v519 = _read_status(PRIOR_MANUAL_OWNER_ESCALATION_REQUEST_VERSION)
    expected_next = "paper4_v520_manual_owner_escalation_followup_audit.md"
    if v519["next_artifact_v519"] != expected_next:
        raise RuntimeError("v520 expects v519 to route to follow-up audit.")
    if not v519["manual_owner_escalation_followup_audit_ready_v519"]:
        raise RuntimeError("v520 requires v519 follow-up audit readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v519_manual_owner_escalation_request_packet.csv"
    )
    field_requests = pd.read_csv(
        TABLE_DIR / "paper4_v519_manual_owner_field_evidence_request_matrix.csv"
    )
    followup = _manual_owner_escalation_followup_audit(packet)
    field_followup = _field_evidence_manual_owner_followup_audit(field_requests)
    blockers = _manual_owner_followup_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v520_manual_owner_escalation_followup_audit.csv",
        followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v520_field_evidence_manual_owner_followup_audit.csv",
        field_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v520_manual_owner_followup_blocker_register.csv",
        blockers,
    )
    write_csv(TABLE_DIR / "paper4_v520_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v520_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v520_manual_owner_escalation_followup_audit",
        "schema_version": "2026-05-17.520",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_owner_escalation_request_version_v520": (
            PRIOR_MANUAL_OWNER_ESCALATION_REQUEST_VERSION
        ),
        "manual_owner_escalation_followup_audit_created_v520": True,
        "manual_owner_followup_audit_rows_v520": len(followup),
        "manual_owner_request_created_rows_v520": int(
            followup["manual_owner_request_created_v520"].astype(bool).sum()
        ),
        "manual_owner_request_dispatched_rows_v520": int(
            followup["manual_owner_request_dispatched_v520"].astype(bool).sum()
        ),
        "human_response_received_rows_v520": int(
            followup["human_response_received_v520"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v520": int(
            followup["candidate_identifier_received_v520"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v520": int(
            followup["nomination_fields_received_v520"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v520": int(
            followup["nomination_signoff_received_v520"].astype(bool).sum()
        ),
        "evidence_received_rows_v520": int(
            followup["evidence_received_v520"].astype(bool).sum()
        ),
        "manual_owner_followup_complete_rows_v520": int(
            followup["manual_owner_followup_complete_v520"].astype(bool).sum()
        ),
        "open_manual_owner_followup_gap_rows_v520": int(
            followup["manual_owner_followup_gap_open_v520"].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v520": int(
            followup["candidate_input_collection_closed_v520"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v520": int(
            followup["candidate_nomination_recorded_v520"].astype(bool).sum()
        ),
        "field_evidence_manual_owner_followup_audit_rows_v520": len(
            field_followup
        ),
        "field_request_created_rows_v520": int(
            field_followup[
                "manual_owner_field_request_created_v520"
            ].astype(bool).sum()
        ),
        "evidence_request_created_rows_v520": int(
            field_followup[
                "manual_owner_evidence_request_created_v520"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v520": int(
            field_followup["field_value_received_v520"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v520": int(
            field_followup["field_evidence_received_v520"].astype(bool).sum()
        ),
        "open_field_evidence_manual_owner_followup_gap_rows_v520": int(
            field_followup[
                "manual_owner_field_evidence_followup_gap_open_v520"
            ].astype(bool).sum()
        ),
        "manual_owner_followup_blocker_rows_v520": len(blockers),
        "open_manual_owner_followup_blocker_rows_v520": int(
            blockers["blocker_open_v520"].astype(bool).sum()
        ),
        "blocking_manual_owner_followup_rows_v520": int(
            blockers[
                "blocks_manual_owner_followup_completion_v520"
            ].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v520": int(
            followup["eligibility_review_allowed_v520"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v520": int(
            followup["reviewer_assignment_allowed_v520"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v520": int(
            followup["outcome_capture_allowed_v520"].astype(bool).sum()
        ),
        "patch_allowed_rows_v520": int(
            followup["patch_allowed_v520"].astype(bool).sum()
        ),
        "readiness_delta_rows_v520": len(readiness),
        "manual_owner_request_dispatch_packet_ready_v520": True,
        "ready_for_quarto_patch_v520": False,
        "quarto_patch_applied_v520": False,
        "book_sources_modified_v520": False,
        "book_references_modified_v520": False,
        "submission_ready_claim_allowed_v520": False,
        "working_champion_claim_allowed_v520": False,
        "paper1_promotion_allowed_v520": False,
        "paper4_working_champion_changed_v520": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v520": NEXT_ARTIFACT,
        "claim_boundary": (
            "v520 audits manual owner escalation follow-up only; dispatch, "
            "input receipt, candidate resolution, nominations, assignments, "
            "outcomes, captions, patching, submission and final promotion remain "
            "blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v520 must not create final Paper 4 promotion.")
    if status["manual_owner_request_dispatched_rows_v520"] != 0:
        raise RuntimeError("v520 must not dispatch manual owner requests.")
    if status["human_response_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive human responses.")
    if status["candidate_identifier_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive nomination signoff.")
    if status["evidence_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive evidence.")
    if status["manual_owner_followup_complete_rows_v520"] != 0:
        raise RuntimeError("v520 must not complete manual owner follow-up.")
    if status["candidate_input_collection_closed_rows_v520"] != 0:
        raise RuntimeError("v520 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v520"] != 0:
        raise RuntimeError("v520 must not record candidate nominations.")
    if status["field_value_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive field values.")
    if status["field_evidence_received_rows_v520"] != 0:
        raise RuntimeError("v520 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v520"] != 0:
        raise RuntimeError("v520 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v520"] != 0:
        raise RuntimeError("v520 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v520"] != 0:
        raise RuntimeError("v520 must not allow outcome capture.")
    if status["patch_allowed_rows_v520"] != 0:
        raise RuntimeError("v520 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v520": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
