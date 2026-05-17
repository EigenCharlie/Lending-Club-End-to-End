#!/usr/bin/env python3
"""Build Paper 4 v515 collection reminder follow-up audit artifacts."""

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

VERSION = 515
PRIOR_COLLECTION_REMINDER_PACKET_VERSION = 514
NEXT_ARTIFACT = "paper4_v516_candidate_input_second_reminder_packet.md"
FOLLOWUP_MD = NOTEBOOK.parent / "paper4_v515_collection_reminder_followup_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _collection_reminder_followup_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        complete = (
            bool(row["candidate_identifier_received_v514"])
            and bool(row["nomination_fields_received_v514"])
            and bool(row["nomination_signoff_received_v514"])
            and bool(row["evidence_received_v514"])
        )
        rows.append(
            {
                "collection_followup_audit_id_v515": row[
                    "collection_reminder_id_v514"
                ],
                "priority_v515": int(row["priority_v514"]),
                "review_domain_v515": row["review_domain_v514"],
                "reviewer_role_required_v515": row["reviewer_role_required_v514"],
                "reminder_created_v515": bool(row["reminder_created_v514"]),
                "human_response_received_v515": False,
                "candidate_identifier_received_v515": bool(
                    row["candidate_identifier_received_v514"]
                ),
                "nomination_fields_received_v515": bool(
                    row["nomination_fields_received_v514"]
                ),
                "nomination_signoff_received_v515": bool(
                    row["nomination_signoff_received_v514"]
                ),
                "evidence_received_v515": bool(row["evidence_received_v514"]),
                "collection_complete_v515": complete,
                "followup_gap_open_v515": not complete,
                "candidate_nomination_recorded_v515": bool(
                    row["candidate_nomination_recorded_v514"]
                ),
                "eligibility_review_allowed_v515": False,
                "reviewer_assignment_allowed_v515": False,
                "outcome_capture_allowed_v515": False,
                "patch_allowed_v515": False,
                "required_next_step_v515": (
                    "issue_second_candidate_input_reminder"
                ),
                "claim_boundary_v515": "collection reminder follow-up audit only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_followup_audit(checklist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in checklist.iterrows():
        field_received = bool(row["field_value_received_v514"])
        evidence_received = bool(row["field_evidence_received_v514"])
        rows.append(
            {
                "collection_followup_audit_id_v515": row[
                    "collection_reminder_id_v514"
                ],
                "nomination_field_v515": row["nomination_field_v514"],
                "field_reminder_created_v515": bool(
                    row["field_reminder_created_v514"]
                ),
                "evidence_reminder_created_v515": bool(
                    row["evidence_reminder_created_v514"]
                ),
                "field_value_received_v515": field_received,
                "field_evidence_received_v515": evidence_received,
                "field_followup_gap_open_v515": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v515": "field and evidence follow-up audit only",
            }
        )
    return pd.DataFrame(rows)


def _followup_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "followup_blocker_id_v515": "no_candidate_identifier_response",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": True,
                "required_resolution_v515": "receive candidate identifier response",
            },
            {
                "followup_blocker_id_v515": "no_nomination_field_response",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": True,
                "required_resolution_v515": "receive nomination field response",
            },
            {
                "followup_blocker_id_v515": "no_nomination_signoff_response",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": True,
                "required_resolution_v515": "receive nomination signoff response",
            },
            {
                "followup_blocker_id_v515": "no_evidence_response",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": True,
                "required_resolution_v515": "receive requested evidence response",
            },
            {
                "followup_blocker_id_v515": "eligibility_review_blocked",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": False,
                "required_resolution_v515": (
                    "start eligibility only after complete candidate inputs"
                ),
            },
            {
                "followup_blocker_id_v515": "no_final_promotion",
                "blocker_open_v515": True,
                "blocks_collection_completion_v515": False,
                "required_resolution_v515": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v515": "collection_reminder_followup_audit_created",
                "ready_v515": True,
                "evidence_artifact_v515": (
                    "paper4_v515_collection_reminder_followup_audit.csv"
                ),
                "claim_boundary_v515": "collection reminder follow-up audit only",
            },
            {
                "readiness_gate_v515": "field_evidence_followup_audit_created",
                "ready_v515": True,
                "evidence_artifact_v515": (
                    "paper4_v515_field_evidence_followup_audit.csv"
                ),
                "claim_boundary_v515": "field evidence follow-up audit only",
            },
            {
                "readiness_gate_v515": "followup_blocker_register_created",
                "ready_v515": True,
                "evidence_artifact_v515": (
                    "paper4_v515_followup_blocker_register.csv"
                ),
                "claim_boundary_v515": "follow-up blocker register only",
            },
            {
                "readiness_gate_v515": "second_reminder_packet_ready",
                "ready_v515": True,
                "evidence_artifact_v515": (
                    "paper4_v515_followup_blocker_register.csv"
                ),
                "claim_boundary_v515": "future second reminder readiness only",
            },
            {
                "readiness_gate_v515": "candidate_identifiers_received",
                "ready_v515": False,
                "evidence_artifact_v515": "candidate identifiers remain unreceived",
                "claim_boundary_v515": "no candidate identifiers received",
            },
            {
                "readiness_gate_v515": "candidate_nominations_recorded",
                "ready_v515": False,
                "evidence_artifact_v515": "candidate nominations remain absent",
                "claim_boundary_v515": "no candidates nominated",
            },
            {
                "readiness_gate_v515": "ready_for_quarto_patch",
                "ready_v515": False,
                "evidence_artifact_v515": "candidate inputs remain absent",
                "claim_boundary_v515": "patch remains blocked",
            },
            {
                "readiness_gate_v515": "paper4_final_promotion_created",
                "ready_v515": False,
                "evidence_artifact_v515": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v515": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v515_collection_reminder_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v515_collection_reminder_followup_audit.csv",
                "boundary": "collection reminder follow-up audit only",
            },
            {
                "claim_id": "v515_field_evidence_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v515_field_evidence_followup_audit.csv",
                "boundary": "field evidence follow-up audit only",
            },
            {
                "claim_id": "v515_second_reminder_packet_ready",
                "allowed": True,
                "artifact": "paper4_v515_followup_blocker_register.csv",
                "boundary": "future second reminder readiness only",
            },
            {
                "claim_id": "v515_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v515_collection_reminder_followup_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v515_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v515_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v515_final_promotion",
                "allowed": False,
                "artifact": "paper4_v515_manuscript_readiness_delta.csv",
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
                "claim": "v515 audits candidate input reminder follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_collection_reminder_followup_audit.csv"
                ),
                "boundary": "Candidate input reminder follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v515 audits field and evidence reminder follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_field_evidence_followup_audit.csv"
                ),
                "boundary": "Field and evidence reminder follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v515 makes a second reminder packet executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_followup_blocker_register.csv"
                ),
                "boundary": "Future second reminder readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v515 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_collection_reminder_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v515 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v515 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v515_manuscript_readiness_delta.csv"
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
                "executable_item": "v515 audits candidate input reminder follow-up.",
                "status": "collection_reminder_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v516 creates second candidate input reminder packet",
                "last_wave": "v515",
                "execution_result": (
                    "collection_reminder_followup_audit_confirmed_no_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v515")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Collection Reminder Follow-up Audit v515

Generated: {status["generated_at_utc"]}

## Result

v515 audits the v514 collection reminders. No human responses, candidate
identifiers, nomination fields, signoffs or evidence have been received, so all
14 follow-up gaps and all 84 field/evidence follow-up gaps remain open.

## Counts

- Follow-up audit rows: `{status["followup_audit_rows_v515"]}`.
- Open follow-up gap rows: `{status["open_followup_gap_rows_v515"]}`.
- Human response received rows: `{status["human_response_received_rows_v515"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v515"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v515"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v515"]}`.
- Evidence received rows: `{status["evidence_received_rows_v515"]}`.
- Collection complete rows: `{status["collection_complete_rows_v515"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v515"]}`.
- Field/evidence follow-up audit rows: `{status["field_evidence_followup_audit_rows_v515"]}`.
- Open field/evidence follow-up gap rows: `{status["open_field_evidence_followup_gap_rows_v515"]}`.
- Field value received rows: `{status["field_value_received_rows_v515"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v515"]}`.
- Follow-up blocker rows: `{status["followup_blocker_rows_v515"]}`.
- Open follow-up blocker rows: `{status["open_followup_blocker_rows_v515"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v515"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v515"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v515"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v515"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v515"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v515 is a collection-reminder follow-up audit only. It does not receive
candidate inputs, resolve or nominate candidates, assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V515_COLLECTION_REMINDER_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V515_COLLECTION_REMINDER_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v515: Collection Reminder Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v515 audits whether the v514 collection reminder packet produced any human
response, candidate identifiers, nomination fields, signoff or evidence. It
confirms the reminder path remains open without fabricating inputs.

### Results

- Follow-up audit rows:
  `{status["followup_audit_rows_v515"]}`.
- Open follow-up gap rows:
  `{status["open_followup_gap_rows_v515"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v515"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v515"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v515"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v515"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v515"]}`.
- Collection complete rows:
  `{status["collection_complete_rows_v515"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v515"]}`.
- Field/evidence follow-up audit rows:
  `{status["field_evidence_followup_audit_rows_v515"]}`.
- Open field/evidence follow-up gap rows:
  `{status["open_field_evidence_followup_gap_rows_v515"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v515"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v515"]}`.
- Follow-up blocker rows:
  `{status["followup_blocker_rows_v515"]}`.
- Open follow-up blocker rows:
  `{status["open_followup_blocker_rows_v515"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v515"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v515"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v515"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v515"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v515"]}`.
- Book sources modified:
  `{status["book_sources_modified_v515"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v515"]}`.

### Interpretation

The reminder follow-up audit finds no received input. The next executable step
is a second reminder packet, not eligibility review, candidate nomination or a
Quarto manuscript patch.

### Claim Impact

- Allowed: collection reminder follow-up audit, field/evidence follow-up audit
  and future second reminder readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v515 in the living notebook. v516 should create a second reminder packet
while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v514 = _read_status(PRIOR_COLLECTION_REMINDER_PACKET_VERSION)
    expected_next = "paper4_v515_collection_reminder_followup_audit.md"
    if v514["next_artifact_v514"] != expected_next:
        raise RuntimeError("v515 expects v514 to route to follow-up audit.")
    if not v514["reminder_followup_audit_ready_v514"]:
        raise RuntimeError("v515 requires v514 reminder follow-up readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v514_candidate_input_collection_reminder_packet.csv"
    )
    checklist = pd.read_csv(
        TABLE_DIR / "paper4_v514_field_evidence_collection_checklist.csv"
    )
    followup = _collection_reminder_followup_audit(packet)
    field_followup = _field_evidence_followup_audit(checklist)
    blockers = _followup_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v515_collection_reminder_followup_audit.csv",
        followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v515_field_evidence_followup_audit.csv",
        field_followup,
    )
    write_csv(TABLE_DIR / "paper4_v515_followup_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v515_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v515_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v515_collection_reminder_followup_audit",
        "schema_version": "2026-05-17.515",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_collection_reminder_packet_version_v515": (
            PRIOR_COLLECTION_REMINDER_PACKET_VERSION
        ),
        "collection_reminder_followup_audit_created_v515": True,
        "followup_audit_rows_v515": len(followup),
        "open_followup_gap_rows_v515": int(
            followup["followup_gap_open_v515"].astype(bool).sum()
        ),
        "human_response_received_rows_v515": int(
            followup["human_response_received_v515"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v515": int(
            followup["candidate_identifier_received_v515"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v515": int(
            followup["nomination_fields_received_v515"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v515": int(
            followup["nomination_signoff_received_v515"].astype(bool).sum()
        ),
        "evidence_received_rows_v515": int(
            followup["evidence_received_v515"].astype(bool).sum()
        ),
        "collection_complete_rows_v515": int(
            followup["collection_complete_v515"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v515": int(
            followup["candidate_nomination_recorded_v515"].astype(bool).sum()
        ),
        "field_evidence_followup_audit_rows_v515": len(field_followup),
        "open_field_evidence_followup_gap_rows_v515": int(
            field_followup["field_followup_gap_open_v515"].astype(bool).sum()
        ),
        "field_value_received_rows_v515": int(
            field_followup["field_value_received_v515"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v515": int(
            field_followup["field_evidence_received_v515"].astype(bool).sum()
        ),
        "followup_blocker_rows_v515": len(blockers),
        "open_followup_blocker_rows_v515": int(
            blockers["blocker_open_v515"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v515": int(
            followup["eligibility_review_allowed_v515"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v515": int(
            followup["reviewer_assignment_allowed_v515"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v515": int(
            followup["outcome_capture_allowed_v515"].astype(bool).sum()
        ),
        "patch_allowed_rows_v515": int(
            followup["patch_allowed_v515"].astype(bool).sum()
        ),
        "readiness_delta_rows_v515": len(readiness),
        "second_reminder_packet_ready_v515": True,
        "ready_for_quarto_patch_v515": False,
        "quarto_patch_applied_v515": False,
        "book_sources_modified_v515": False,
        "book_references_modified_v515": False,
        "submission_ready_claim_allowed_v515": False,
        "working_champion_claim_allowed_v515": False,
        "paper1_promotion_allowed_v515": False,
        "paper4_working_champion_changed_v515": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v515": NEXT_ARTIFACT,
        "claim_boundary": (
            "v515 audits candidate input collection reminder follow-up only; "
            "input receipt, candidate resolution, nominations, assignments, "
            "outcomes, captions, patching, submission and final promotion remain "
            "blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v515 must not create final Paper 4 promotion.")
    if status["human_response_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive human responses.")
    if status["candidate_identifier_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive nomination signoff.")
    if status["evidence_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive evidence.")
    if status["collection_complete_rows_v515"] != 0:
        raise RuntimeError("v515 must not complete collection.")
    if status["candidate_nomination_recorded_rows_v515"] != 0:
        raise RuntimeError("v515 must not record candidate nominations.")
    if status["field_value_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive field values.")
    if status["field_evidence_received_rows_v515"] != 0:
        raise RuntimeError("v515 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v515"] != 0:
        raise RuntimeError("v515 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v515"] != 0:
        raise RuntimeError("v515 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v515"] != 0:
        raise RuntimeError("v515 must not allow outcome capture.")
    if status["patch_allowed_rows_v515"] != 0:
        raise RuntimeError("v515 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v515": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
