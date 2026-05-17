#!/usr/bin/env python3
"""Build Paper 4 v522 manual owner dispatch follow-up audit artifacts."""

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

VERSION = 522
PRIOR_MANUAL_OWNER_DISPATCH_PACKET_VERSION = 521
NEXT_ARTIFACT = "paper4_v523_dispatch_evidence_request_packet.md"
FOLLOWUP_MD = NOTEBOOK.parent / "paper4_v522_manual_owner_dispatch_followup_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_owner_dispatch_followup_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        external_dispatch_recorded = bool(row["manual_owner_request_dispatched_v521"])
        response_received = bool(row["human_response_received_v521"])
        candidate_identifier_received = bool(
            row["candidate_identifier_received_v521"]
        )
        nomination_fields_received = bool(row["nomination_fields_received_v521"])
        nomination_signoff_received = bool(row["nomination_signoff_received_v521"])
        evidence_received = bool(row["evidence_received_v521"])
        complete = (
            external_dispatch_recorded
            and response_received
            and candidate_identifier_received
            and nomination_fields_received
            and nomination_signoff_received
            and evidence_received
        )
        rows.append(
            {
                "manual_owner_dispatch_followup_audit_id_v522": row[
                    "manual_owner_dispatch_id_v521"
                ],
                "priority_v522": int(row["priority_v521"]),
                "review_domain_v522": row["review_domain_v521"],
                "reviewer_role_required_v522": row["reviewer_role_required_v521"],
                "dispatch_packet_created_v522": bool(
                    row["dispatch_packet_created_v521"]
                ),
                "dispatch_ready_v522": bool(row["dispatch_ready_v521"]),
                "external_dispatch_recorded_v522": external_dispatch_recorded,
                "human_response_received_v522": response_received,
                "candidate_identifier_received_v522": (
                    candidate_identifier_received
                ),
                "nomination_fields_received_v522": nomination_fields_received,
                "nomination_signoff_received_v522": nomination_signoff_received,
                "evidence_received_v522": evidence_received,
                "manual_owner_dispatch_followup_complete_v522": complete,
                "manual_owner_dispatch_followup_gap_open_v522": not complete,
                "candidate_input_collection_closed_v522": False,
                "candidate_nomination_recorded_v522": bool(
                    row["candidate_nomination_recorded_v521"]
                ),
                "eligibility_review_allowed_v522": False,
                "reviewer_assignment_allowed_v522": False,
                "outcome_capture_allowed_v522": False,
                "patch_allowed_v522": False,
                "required_next_step_v522": (
                    "prepare_dispatch_evidence_request_packet"
                ),
                "claim_boundary_v522": (
                    "manual owner dispatch follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_dispatch_followup_audit(
    checklist: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in checklist.iterrows():
        field_received = bool(row["field_value_received_v521"])
        evidence_received = bool(row["field_evidence_received_v521"])
        rows.append(
            {
                "manual_owner_dispatch_followup_audit_id_v522": row[
                    "manual_owner_dispatch_id_v521"
                ],
                "nomination_field_v522": row["nomination_field_v521"],
                "field_dispatch_checklist_created_v522": bool(
                    row["field_dispatch_checklist_created_v521"]
                ),
                "evidence_dispatch_checklist_created_v522": bool(
                    row["evidence_dispatch_checklist_created_v521"]
                ),
                "field_value_received_v522": field_received,
                "field_evidence_received_v522": evidence_received,
                "field_evidence_dispatch_followup_gap_open_v522": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v522": (
                    "field and evidence dispatch follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _manual_owner_dispatch_followup_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dispatch_followup_blocker_id_v522": (
                    "external_dispatch_evidence_absent"
                ),
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": True,
                "required_resolution_v522": (
                    "record externally verified dispatch evidence"
                ),
            },
            {
                "dispatch_followup_blocker_id_v522": (
                    "manual_owner_response_absent_after_dispatch_packet"
                ),
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": True,
                "required_resolution_v522": (
                    "receive manual owner response after verified dispatch"
                ),
            },
            {
                "dispatch_followup_blocker_id_v522": (
                    "candidate_identifier_absent_after_dispatch_packet"
                ),
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": True,
                "required_resolution_v522": "receive candidate identifier",
            },
            {
                "dispatch_followup_blocker_id_v522": (
                    "nomination_payload_absent_after_dispatch_packet"
                ),
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": True,
                "required_resolution_v522": (
                    "receive nomination fields and nomination signoff"
                ),
            },
            {
                "dispatch_followup_blocker_id_v522": (
                    "evidence_absent_after_dispatch_packet"
                ),
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": True,
                "required_resolution_v522": "receive supporting evidence",
            },
            {
                "dispatch_followup_blocker_id_v522": "no_final_promotion",
                "blocker_open_v522": True,
                "blocks_dispatch_followup_completion_v522": False,
                "required_resolution_v522": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v522": (
                    "manual_owner_dispatch_followup_audit_created"
                ),
                "ready_v522": True,
                "evidence_artifact_v522": (
                    "paper4_v522_manual_owner_dispatch_followup_audit.csv"
                ),
                "claim_boundary_v522": (
                    "manual owner dispatch follow-up audit only"
                ),
            },
            {
                "readiness_gate_v522": (
                    "field_evidence_dispatch_followup_audit_created"
                ),
                "ready_v522": True,
                "evidence_artifact_v522": (
                    "paper4_v522_field_evidence_dispatch_followup_audit.csv"
                ),
                "claim_boundary_v522": (
                    "field evidence dispatch follow-up audit only"
                ),
            },
            {
                "readiness_gate_v522": (
                    "manual_owner_dispatch_followup_blocker_register_created"
                ),
                "ready_v522": True,
                "evidence_artifact_v522": (
                    "paper4_v522_manual_owner_dispatch_followup_blocker_register.csv"
                ),
                "claim_boundary_v522": "manual owner dispatch blockers only",
            },
            {
                "readiness_gate_v522": "dispatch_evidence_request_packet_ready",
                "ready_v522": True,
                "evidence_artifact_v522": (
                    "paper4_v522_manual_owner_dispatch_followup_blocker_register.csv"
                ),
                "claim_boundary_v522": (
                    "future dispatch evidence request packet readiness only"
                ),
            },
            {
                "readiness_gate_v522": "candidate_identifiers_received",
                "ready_v522": False,
                "evidence_artifact_v522": "candidate identifiers remain unreceived",
                "claim_boundary_v522": "no candidate identifiers received",
            },
            {
                "readiness_gate_v522": "candidate_nominations_recorded",
                "ready_v522": False,
                "evidence_artifact_v522": "candidate nominations remain absent",
                "claim_boundary_v522": "no candidates nominated",
            },
            {
                "readiness_gate_v522": "ready_for_quarto_patch",
                "ready_v522": False,
                "evidence_artifact_v522": "candidate inputs remain absent",
                "claim_boundary_v522": "patch remains blocked",
            },
            {
                "readiness_gate_v522": "paper4_final_promotion_created",
                "ready_v522": False,
                "evidence_artifact_v522": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v522": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v522_manual_owner_dispatch_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v522_manual_owner_dispatch_followup_audit.csv",
                "boundary": "manual owner dispatch follow-up audit only",
            },
            {
                "claim_id": (
                    "v522_field_evidence_dispatch_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v522_field_evidence_dispatch_followup_audit.csv"
                ),
                "boundary": "field evidence dispatch follow-up audit only",
            },
            {
                "claim_id": "v522_dispatch_evidence_request_packet_ready",
                "allowed": True,
                "artifact": (
                    "paper4_v522_manual_owner_dispatch_followup_blocker_register.csv"
                ),
                "boundary": (
                    "future dispatch evidence request packet readiness only"
                ),
            },
            {
                "claim_id": "v522_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v522_manual_owner_dispatch_followup_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v522_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v522_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v522_final_promotion",
                "allowed": False,
                "artifact": "paper4_v522_manuscript_readiness_delta.csv",
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
                "claim": "v522 audits manual owner dispatch follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_manual_owner_dispatch_followup_audit.csv"
                ),
                "boundary": "Manual owner dispatch follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v522 audits field and evidence dispatch follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_field_evidence_dispatch_followup_audit.csv"
                ),
                "boundary": "Field and evidence dispatch follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v522 makes dispatch evidence request packet executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_manual_owner_dispatch_followup_blocker_register.csv"
                ),
                "boundary": (
                    "Future dispatch evidence request packet readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v522 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_manual_owner_dispatch_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v522 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v522 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v522_manuscript_readiness_delta.csv"
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
                "executable_item": "v522 audits manual owner dispatch follow-up.",
                "status": "manual_owner_dispatch_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v523 creates dispatch evidence request packet",
                "last_wave": "v522",
                "execution_result": (
                    "manual_owner_dispatch_followup_audit_confirmed_no_external_dispatch_or_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v522")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Owner Dispatch Follow-up Audit v522

Generated: {status["generated_at_utc"]}

## Result

v522 audits the v521 manual owner request dispatch packet. The packet and
field/evidence checklist remain reproducible and dispatch-ready, but no
externally verified dispatch, manual owner response, candidate identifier,
nomination payload or evidence is recorded. The next executable step is a
bounded dispatch evidence request packet.

## Counts

- Manual owner dispatch follow-up audit rows: `{status["manual_owner_dispatch_followup_audit_rows_v522"]}`.
- Dispatch packet created rows: `{status["dispatch_packet_created_rows_v522"]}`.
- Dispatch ready rows: `{status["dispatch_ready_rows_v522"]}`.
- External dispatch recorded rows: `{status["external_dispatch_recorded_rows_v522"]}`.
- Human response received rows: `{status["human_response_received_rows_v522"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v522"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v522"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v522"]}`.
- Evidence received rows: `{status["evidence_received_rows_v522"]}`.
- Manual owner dispatch follow-up complete rows: `{status["manual_owner_dispatch_followup_complete_rows_v522"]}`.
- Open manual owner dispatch follow-up gap rows: `{status["open_manual_owner_dispatch_followup_gap_rows_v522"]}`.
- Candidate input collection closed rows: `{status["candidate_input_collection_closed_rows_v522"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v522"]}`.
- Field/evidence dispatch follow-up audit rows: `{status["field_evidence_dispatch_followup_audit_rows_v522"]}`.
- Field dispatch checklist created rows: `{status["field_dispatch_checklist_created_rows_v522"]}`.
- Evidence dispatch checklist created rows: `{status["evidence_dispatch_checklist_created_rows_v522"]}`.
- Field value received rows: `{status["field_value_received_rows_v522"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v522"]}`.
- Open field/evidence dispatch follow-up gap rows: `{status["open_field_evidence_dispatch_followup_gap_rows_v522"]}`.
- Dispatch follow-up blocker rows: `{status["dispatch_followup_blocker_rows_v522"]}`.
- Open dispatch follow-up blocker rows: `{status["open_dispatch_followup_blocker_rows_v522"]}`.
- Blocking dispatch follow-up rows: `{status["blocking_dispatch_followup_rows_v522"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v522"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v522"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v522"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v522"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v522"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v522 is a manual owner dispatch follow-up audit only. It does not record
external dispatch, receive candidate inputs, resolve or nominate candidates,
assign reviewers, capture completed review outcomes, finalize captions, approve
patch scope, edit Quarto, render the book, make Paper 4 submission-ready,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V522_MANUAL_OWNER_DISPATCH_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V522_MANUAL_OWNER_DISPATCH_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v522: Manual Owner Dispatch Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v522 audits whether the v521 manual owner request dispatch packet has gained
externally verified dispatch evidence or downstream human/candidate inputs. It
keeps the work bounded to reproducible follow-up evidence and does not convert
prepared requests into recorded nominations.

### Results

- Manual owner dispatch follow-up audit rows:
  `{status["manual_owner_dispatch_followup_audit_rows_v522"]}`.
- Dispatch packet created rows:
  `{status["dispatch_packet_created_rows_v522"]}`.
- Dispatch ready rows:
  `{status["dispatch_ready_rows_v522"]}`.
- External dispatch recorded rows:
  `{status["external_dispatch_recorded_rows_v522"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v522"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v522"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v522"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v522"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v522"]}`.
- Manual owner dispatch follow-up complete rows:
  `{status["manual_owner_dispatch_followup_complete_rows_v522"]}`.
- Open manual owner dispatch follow-up gap rows:
  `{status["open_manual_owner_dispatch_followup_gap_rows_v522"]}`.
- Candidate input collection closed rows:
  `{status["candidate_input_collection_closed_rows_v522"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v522"]}`.
- Field/evidence dispatch follow-up audit rows:
  `{status["field_evidence_dispatch_followup_audit_rows_v522"]}`.
- Field dispatch checklist created rows:
  `{status["field_dispatch_checklist_created_rows_v522"]}`.
- Evidence dispatch checklist created rows:
  `{status["evidence_dispatch_checklist_created_rows_v522"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v522"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v522"]}`.
- Open field/evidence dispatch follow-up gap rows:
  `{status["open_field_evidence_dispatch_followup_gap_rows_v522"]}`.
- Dispatch follow-up blocker rows:
  `{status["dispatch_followup_blocker_rows_v522"]}`.
- Open dispatch follow-up blocker rows:
  `{status["open_dispatch_followup_blocker_rows_v522"]}`.
- Blocking dispatch follow-up rows:
  `{status["blocking_dispatch_followup_rows_v522"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v522"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v522"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v522"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v522"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v522"]}`.
- Book sources modified:
  `{status["book_sources_modified_v522"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v522"]}`.

### Interpretation

The dispatch packet remains prepared rather than externally evidenced. Because
external dispatch, responses and candidate inputs remain zero, the next
executable step is a dispatch evidence request packet, not candidate
nomination, eligibility review or manuscript patching.

### Claim Impact

- Allowed: manual owner dispatch follow-up audit, field/evidence dispatch
  follow-up audit and future dispatch evidence request packet readiness.
- Still prohibited: external dispatch completion, candidate input receipt,
  candidate resolution/nomination, reviewer assignment, completed review
  claims, final captions, Quarto patch readiness/application, Quarto/book
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v522 in the living notebook. v523 should prepare a dispatch evidence
request packet while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v521 = _read_status(PRIOR_MANUAL_OWNER_DISPATCH_PACKET_VERSION)
    expected_next = "paper4_v522_manual_owner_dispatch_followup_audit.md"
    if v521["next_artifact_v521"] != expected_next:
        raise RuntimeError("v522 expects v521 to route to dispatch follow-up.")
    if not v521["manual_owner_dispatch_followup_audit_ready_v521"]:
        raise RuntimeError("v522 requires v521 dispatch follow-up readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v521_manual_owner_request_dispatch_packet.csv"
    )
    checklist = pd.read_csv(
        TABLE_DIR / "paper4_v521_field_evidence_dispatch_checklist.csv"
    )
    followup = _manual_owner_dispatch_followup_audit(packet)
    field_followup = _field_evidence_dispatch_followup_audit(checklist)
    blockers = _manual_owner_dispatch_followup_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v522_manual_owner_dispatch_followup_audit.csv",
        followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v522_field_evidence_dispatch_followup_audit.csv",
        field_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v522_manual_owner_dispatch_followup_blocker_register.csv",
        blockers,
    )
    write_csv(TABLE_DIR / "paper4_v522_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v522_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v522_manual_owner_dispatch_followup_audit",
        "schema_version": "2026-05-17.522",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_owner_dispatch_packet_version_v522": (
            PRIOR_MANUAL_OWNER_DISPATCH_PACKET_VERSION
        ),
        "manual_owner_dispatch_followup_audit_created_v522": True,
        "manual_owner_dispatch_followup_audit_rows_v522": len(followup),
        "dispatch_packet_created_rows_v522": int(
            followup["dispatch_packet_created_v522"].astype(bool).sum()
        ),
        "dispatch_ready_rows_v522": int(
            followup["dispatch_ready_v522"].astype(bool).sum()
        ),
        "external_dispatch_recorded_rows_v522": int(
            followup["external_dispatch_recorded_v522"].astype(bool).sum()
        ),
        "human_response_received_rows_v522": int(
            followup["human_response_received_v522"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v522": int(
            followup["candidate_identifier_received_v522"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v522": int(
            followup["nomination_fields_received_v522"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v522": int(
            followup["nomination_signoff_received_v522"].astype(bool).sum()
        ),
        "evidence_received_rows_v522": int(
            followup["evidence_received_v522"].astype(bool).sum()
        ),
        "manual_owner_dispatch_followup_complete_rows_v522": int(
            followup[
                "manual_owner_dispatch_followup_complete_v522"
            ].astype(bool).sum()
        ),
        "open_manual_owner_dispatch_followup_gap_rows_v522": int(
            followup[
                "manual_owner_dispatch_followup_gap_open_v522"
            ].astype(bool).sum()
        ),
        "candidate_input_collection_closed_rows_v522": int(
            followup["candidate_input_collection_closed_v522"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v522": int(
            followup["candidate_nomination_recorded_v522"].astype(bool).sum()
        ),
        "field_evidence_dispatch_followup_audit_rows_v522": len(field_followup),
        "field_dispatch_checklist_created_rows_v522": int(
            field_followup[
                "field_dispatch_checklist_created_v522"
            ].astype(bool).sum()
        ),
        "evidence_dispatch_checklist_created_rows_v522": int(
            field_followup[
                "evidence_dispatch_checklist_created_v522"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v522": int(
            field_followup["field_value_received_v522"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v522": int(
            field_followup["field_evidence_received_v522"].astype(bool).sum()
        ),
        "open_field_evidence_dispatch_followup_gap_rows_v522": int(
            field_followup[
                "field_evidence_dispatch_followup_gap_open_v522"
            ].astype(bool).sum()
        ),
        "dispatch_followup_blocker_rows_v522": len(blockers),
        "open_dispatch_followup_blocker_rows_v522": int(
            blockers["blocker_open_v522"].astype(bool).sum()
        ),
        "blocking_dispatch_followup_rows_v522": int(
            blockers[
                "blocks_dispatch_followup_completion_v522"
            ].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v522": int(
            followup["eligibility_review_allowed_v522"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v522": int(
            followup["reviewer_assignment_allowed_v522"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v522": int(
            followup["outcome_capture_allowed_v522"].astype(bool).sum()
        ),
        "patch_allowed_rows_v522": int(
            followup["patch_allowed_v522"].astype(bool).sum()
        ),
        "readiness_delta_rows_v522": len(readiness),
        "dispatch_evidence_request_packet_ready_v522": True,
        "ready_for_quarto_patch_v522": False,
        "quarto_patch_applied_v522": False,
        "book_sources_modified_v522": False,
        "book_references_modified_v522": False,
        "submission_ready_claim_allowed_v522": False,
        "working_champion_claim_allowed_v522": False,
        "paper1_promotion_allowed_v522": False,
        "paper4_working_champion_changed_v522": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v522": NEXT_ARTIFACT,
        "claim_boundary": (
            "v522 audits manual owner dispatch follow-up only; external "
            "dispatch, input receipt, candidate resolution, nominations, "
            "assignments, outcomes, captions, patching, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v522 must not create final Paper 4 promotion.")
    if status["external_dispatch_recorded_rows_v522"] != 0:
        raise RuntimeError("v522 must not record external dispatch.")
    if status["human_response_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive human responses.")
    if status["candidate_identifier_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive nomination signoff.")
    if status["evidence_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive evidence.")
    if status["manual_owner_dispatch_followup_complete_rows_v522"] != 0:
        raise RuntimeError("v522 must not complete dispatch follow-up.")
    if status["candidate_input_collection_closed_rows_v522"] != 0:
        raise RuntimeError("v522 must not close candidate input collection.")
    if status["candidate_nomination_recorded_rows_v522"] != 0:
        raise RuntimeError("v522 must not record candidate nominations.")
    if status["field_value_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive field values.")
    if status["field_evidence_received_rows_v522"] != 0:
        raise RuntimeError("v522 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v522"] != 0:
        raise RuntimeError("v522 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v522"] != 0:
        raise RuntimeError("v522 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v522"] != 0:
        raise RuntimeError("v522 must not allow outcome capture.")
    if status["patch_allowed_rows_v522"] != 0:
        raise RuntimeError("v522 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v522": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
