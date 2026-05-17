#!/usr/bin/env python3
"""Build Paper 4 v516 candidate input second reminder packet artifacts."""

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

VERSION = 516
PRIOR_COLLECTION_REMINDER_FOLLOWUP_AUDIT_VERSION = 515
NEXT_ARTIFACT = "paper4_v517_second_reminder_followup_audit.md"
SECOND_REMINDER_MD = NOTEBOOK.parent / "paper4_v516_candidate_input_second_reminder_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_input_second_reminder_packet(followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in followup.iterrows():
        rows.append(
            {
                "second_reminder_id_v516": row["collection_followup_audit_id_v515"],
                "priority_v516": int(row["priority_v515"]),
                "review_domain_v516": row["review_domain_v515"],
                "reviewer_role_required_v516": row["reviewer_role_required_v515"],
                "prior_reminder_created_v516": bool(row["reminder_created_v515"]),
                "second_reminder_created_v516": True,
                "human_response_received_v516": False,
                "candidate_identifier_received_v516": bool(
                    row["candidate_identifier_received_v515"]
                ),
                "nomination_fields_received_v516": bool(
                    row["nomination_fields_received_v515"]
                ),
                "nomination_signoff_received_v516": bool(
                    row["nomination_signoff_received_v515"]
                ),
                "evidence_received_v516": bool(row["evidence_received_v515"]),
                "candidate_nomination_recorded_v516": bool(
                    row["candidate_nomination_recorded_v515"]
                ),
                "eligibility_review_allowed_v516": False,
                "reviewer_assignment_allowed_v516": False,
                "outcome_capture_allowed_v516": False,
                "patch_allowed_v516": False,
                "second_reminder_status_v516": (
                    "pending_human_response_after_second_reminder"
                ),
                "required_next_step_v516": "audit_second_reminder_followup",
                "claim_boundary_v516": "candidate input second reminder packet only",
            }
        )
    return pd.DataFrame(rows)


def _second_reminder_field_evidence_checklist(field_followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in field_followup.iterrows():
        field_received = bool(row["field_value_received_v515"])
        evidence_received = bool(row["field_evidence_received_v515"])
        rows.append(
            {
                "second_reminder_id_v516": row["collection_followup_audit_id_v515"],
                "nomination_field_v516": row["nomination_field_v515"],
                "prior_field_reminder_created_v516": bool(
                    row["field_reminder_created_v515"]
                ),
                "prior_evidence_reminder_created_v516": bool(
                    row["evidence_reminder_created_v515"]
                ),
                "field_second_reminder_created_v516": True,
                "evidence_second_reminder_created_v516": True,
                "field_value_received_v516": field_received,
                "field_evidence_received_v516": evidence_received,
                "second_reminder_gap_open_v516": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v516": (
                    "field and evidence second reminder checklist only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _second_reminder_escalation_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "second_reminder_escalation_id_v516": (
                    "no_candidate_identifier_after_first_followup"
                ),
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": True,
                "required_resolution_v516": (
                    "receive candidate identifier after second reminder"
                ),
            },
            {
                "second_reminder_escalation_id_v516": (
                    "no_nomination_field_after_first_followup"
                ),
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": True,
                "required_resolution_v516": (
                    "receive nomination fields after second reminder"
                ),
            },
            {
                "second_reminder_escalation_id_v516": (
                    "no_nomination_signoff_after_first_followup"
                ),
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": True,
                "required_resolution_v516": (
                    "receive nomination signoff after second reminder"
                ),
            },
            {
                "second_reminder_escalation_id_v516": (
                    "no_evidence_after_first_followup"
                ),
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": True,
                "required_resolution_v516": "receive evidence after second reminder",
            },
            {
                "second_reminder_escalation_id_v516": "eligibility_review_blocked",
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": False,
                "required_resolution_v516": (
                    "start eligibility only after complete candidate inputs"
                ),
            },
            {
                "second_reminder_escalation_id_v516": "no_final_promotion",
                "escalation_active_v516": True,
                "blocks_second_reminder_completion_v516": False,
                "required_resolution_v516": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v516": "candidate_input_second_reminder_packet_created",
                "ready_v516": True,
                "evidence_artifact_v516": (
                    "paper4_v516_candidate_input_second_reminder_packet.csv"
                ),
                "claim_boundary_v516": "candidate input second reminder only",
            },
            {
                "readiness_gate_v516": (
                    "second_reminder_field_evidence_checklist_created"
                ),
                "ready_v516": True,
                "evidence_artifact_v516": (
                    "paper4_v516_second_reminder_field_evidence_checklist.csv"
                ),
                "claim_boundary_v516": "field evidence second reminder checklist only",
            },
            {
                "readiness_gate_v516": "second_reminder_escalation_register_created",
                "ready_v516": True,
                "evidence_artifact_v516": (
                    "paper4_v516_second_reminder_escalation_register.csv"
                ),
                "claim_boundary_v516": "second reminder escalation register only",
            },
            {
                "readiness_gate_v516": "second_reminder_followup_audit_ready",
                "ready_v516": True,
                "evidence_artifact_v516": (
                    "paper4_v516_second_reminder_escalation_register.csv"
                ),
                "claim_boundary_v516": "future second reminder follow-up audit only",
            },
            {
                "readiness_gate_v516": "candidate_identifiers_received",
                "ready_v516": False,
                "evidence_artifact_v516": "candidate identifiers remain unreceived",
                "claim_boundary_v516": "no candidate identifiers received",
            },
            {
                "readiness_gate_v516": "candidate_nominations_recorded",
                "ready_v516": False,
                "evidence_artifact_v516": "candidate nominations remain absent",
                "claim_boundary_v516": "no candidates nominated",
            },
            {
                "readiness_gate_v516": "ready_for_quarto_patch",
                "ready_v516": False,
                "evidence_artifact_v516": "candidate inputs remain absent",
                "claim_boundary_v516": "patch remains blocked",
            },
            {
                "readiness_gate_v516": "paper4_final_promotion_created",
                "ready_v516": False,
                "evidence_artifact_v516": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v516": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v516_second_reminder_packet_created",
                "allowed": True,
                "artifact": "paper4_v516_candidate_input_second_reminder_packet.csv",
                "boundary": "candidate input second reminder only",
            },
            {
                "claim_id": "v516_second_reminder_checklist_created",
                "allowed": True,
                "artifact": (
                    "paper4_v516_second_reminder_field_evidence_checklist.csv"
                ),
                "boundary": "field evidence second reminder checklist only",
            },
            {
                "claim_id": "v516_second_reminder_followup_audit_ready",
                "allowed": True,
                "artifact": "paper4_v516_second_reminder_escalation_register.csv",
                "boundary": "future second reminder follow-up audit only",
            },
            {
                "claim_id": "v516_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v516_candidate_input_second_reminder_packet.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v516_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v516_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v516_final_promotion",
                "allowed": False,
                "artifact": "paper4_v516_manuscript_readiness_delta.csv",
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
                "claim": "v516 creates a candidate input second reminder packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_candidate_input_second_reminder_packet.csv"
                ),
                "boundary": "Candidate input second reminder packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v516 creates a second reminder field and evidence checklist.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_second_reminder_field_evidence_checklist.csv"
                ),
                "boundary": "Second reminder field and evidence checklist only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v516 makes second reminder follow-up audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_second_reminder_escalation_register.csv"
                ),
                "boundary": "Future second reminder follow-up audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v516 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_candidate_input_second_reminder_packet.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v516 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v516 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v516_manuscript_readiness_delta.csv"
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
                "executable_item": "v516 creates a second candidate input reminder.",
                "status": "candidate_input_second_reminder_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v517 audits second reminder follow-up",
                "last_wave": "v516",
                "execution_result": (
                    "second_candidate_input_reminder_packet_created_without_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v516")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _second_reminder_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Input Second Reminder Packet v516

Generated: {status["generated_at_utc"]}

## Result

v516 creates a second candidate input reminder packet after the v515 follow-up
audit found no responses. It does not receive candidate identifiers, nomination
fields, signoff or evidence, and it does not open candidate nomination,
eligibility, reviewer assignment, outcome capture or patch gates.

## Counts

- Second reminder packet rows: `{status["second_reminder_packet_rows_v516"]}`.
- Second reminder created rows: `{status["second_reminder_created_rows_v516"]}`.
- Human response received rows: `{status["human_response_received_rows_v516"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v516"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v516"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v516"]}`.
- Evidence received rows: `{status["evidence_received_rows_v516"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v516"]}`.
- Field/evidence second reminder checklist rows: `{status["field_evidence_second_reminder_checklist_rows_v516"]}`.
- Field second reminder created rows: `{status["field_second_reminder_created_rows_v516"]}`.
- Evidence second reminder created rows: `{status["evidence_second_reminder_created_rows_v516"]}`.
- Field value received rows: `{status["field_value_received_rows_v516"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v516"]}`.
- Open second reminder gap rows: `{status["open_second_reminder_gap_rows_v516"]}`.
- Second reminder escalation rows: `{status["second_reminder_escalation_rows_v516"]}`.
- Active second reminder escalation rows: `{status["active_second_reminder_escalation_rows_v516"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v516"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v516"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v516"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v516"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v516"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v516 is a second-reminder packet only. It does not receive candidate inputs,
resolve or nominate candidates, assign reviewers, capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V516_CANDIDATE_INPUT_SECOND_REMINDER_PACKET_START -->"
    end = "<!-- V516_CANDIDATE_INPUT_SECOND_REMINDER_PACKET_END -->"
    block = f"""
{start}

## Wave v516: Candidate Input Second Reminder Packet

Generated: {status["generated_at_utc"]}

### Objective

v516 turns the still-open v515 follow-up gaps into a second reminder packet and
field/evidence checklist. It creates follow-up pressure only; it does not mark
any candidate input, response or evidence as received.

### Results

- Second reminder packet rows:
  `{status["second_reminder_packet_rows_v516"]}`.
- Second reminder created rows:
  `{status["second_reminder_created_rows_v516"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v516"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v516"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v516"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v516"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v516"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v516"]}`.
- Field/evidence second reminder checklist rows:
  `{status["field_evidence_second_reminder_checklist_rows_v516"]}`.
- Field second reminder created rows:
  `{status["field_second_reminder_created_rows_v516"]}`.
- Evidence second reminder created rows:
  `{status["evidence_second_reminder_created_rows_v516"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v516"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v516"]}`.
- Open second reminder gap rows:
  `{status["open_second_reminder_gap_rows_v516"]}`.
- Second reminder escalation rows:
  `{status["second_reminder_escalation_rows_v516"]}`.
- Active second reminder escalation rows:
  `{status["active_second_reminder_escalation_rows_v516"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v516"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v516"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v516"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v516"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v516"]}`.
- Book sources modified:
  `{status["book_sources_modified_v516"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v516"]}`.

### Interpretation

The second reminder packet is procedural evidence only. Because the response and
evidence counters remain zero, the next executable step is a second-reminder
follow-up audit, not eligibility review or manuscript patching.

### Claim Impact

- Allowed: second reminder packet, second reminder field/evidence checklist and
  future second-reminder follow-up audit readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v516 in the living notebook. v517 should audit the second reminder
follow-up while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v515 = _read_status(PRIOR_COLLECTION_REMINDER_FOLLOWUP_AUDIT_VERSION)
    expected_next = "paper4_v516_candidate_input_second_reminder_packet.md"
    if v515["next_artifact_v515"] != expected_next:
        raise RuntimeError("v516 expects v515 to route to second reminder packet.")
    if not v515["second_reminder_packet_ready_v515"]:
        raise RuntimeError("v516 requires v515 second reminder packet readiness.")

    followup = pd.read_csv(
        TABLE_DIR / "paper4_v515_collection_reminder_followup_audit.csv"
    )
    field_followup = pd.read_csv(
        TABLE_DIR / "paper4_v515_field_evidence_followup_audit.csv"
    )
    packet = _candidate_input_second_reminder_packet(followup)
    checklist = _second_reminder_field_evidence_checklist(field_followup)
    escalations = _second_reminder_escalation_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v516_candidate_input_second_reminder_packet.csv",
        packet,
    )
    write_csv(
        TABLE_DIR / "paper4_v516_second_reminder_field_evidence_checklist.csv",
        checklist,
    )
    write_csv(
        TABLE_DIR / "paper4_v516_second_reminder_escalation_register.csv",
        escalations,
    )
    write_csv(TABLE_DIR / "paper4_v516_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v516_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v516_candidate_input_second_reminder_packet",
        "schema_version": "2026-05-17.516",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_collection_reminder_followup_audit_version_v516": (
            PRIOR_COLLECTION_REMINDER_FOLLOWUP_AUDIT_VERSION
        ),
        "candidate_input_second_reminder_packet_created_v516": True,
        "second_reminder_packet_rows_v516": len(packet),
        "second_reminder_created_rows_v516": int(
            packet["second_reminder_created_v516"].astype(bool).sum()
        ),
        "human_response_received_rows_v516": int(
            packet["human_response_received_v516"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v516": int(
            packet["candidate_identifier_received_v516"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v516": int(
            packet["nomination_fields_received_v516"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v516": int(
            packet["nomination_signoff_received_v516"].astype(bool).sum()
        ),
        "evidence_received_rows_v516": int(
            packet["evidence_received_v516"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v516": int(
            packet["candidate_nomination_recorded_v516"].astype(bool).sum()
        ),
        "field_evidence_second_reminder_checklist_rows_v516": len(checklist),
        "field_second_reminder_created_rows_v516": int(
            checklist["field_second_reminder_created_v516"].astype(bool).sum()
        ),
        "evidence_second_reminder_created_rows_v516": int(
            checklist["evidence_second_reminder_created_v516"].astype(bool).sum()
        ),
        "field_value_received_rows_v516": int(
            checklist["field_value_received_v516"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v516": int(
            checklist["field_evidence_received_v516"].astype(bool).sum()
        ),
        "open_second_reminder_gap_rows_v516": int(
            checklist["second_reminder_gap_open_v516"].astype(bool).sum()
        ),
        "second_reminder_escalation_rows_v516": len(escalations),
        "active_second_reminder_escalation_rows_v516": int(
            escalations["escalation_active_v516"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v516": int(
            packet["eligibility_review_allowed_v516"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v516": int(
            packet["reviewer_assignment_allowed_v516"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v516": int(
            packet["outcome_capture_allowed_v516"].astype(bool).sum()
        ),
        "patch_allowed_rows_v516": int(packet["patch_allowed_v516"].astype(bool).sum()),
        "readiness_delta_rows_v516": len(readiness),
        "second_reminder_followup_audit_ready_v516": True,
        "ready_for_quarto_patch_v516": False,
        "quarto_patch_applied_v516": False,
        "book_sources_modified_v516": False,
        "book_references_modified_v516": False,
        "submission_ready_claim_allowed_v516": False,
        "working_champion_claim_allowed_v516": False,
        "paper1_promotion_allowed_v516": False,
        "paper4_working_champion_changed_v516": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v516": NEXT_ARTIFACT,
        "claim_boundary": (
            "v516 creates a candidate input second reminder packet only; input "
            "receipt, candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v516 must not create final Paper 4 promotion.")
    if status["human_response_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive human responses.")
    if status["candidate_identifier_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive nomination signoff.")
    if status["evidence_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive evidence.")
    if status["candidate_nomination_recorded_rows_v516"] != 0:
        raise RuntimeError("v516 must not record candidate nominations.")
    if status["field_value_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive field values.")
    if status["field_evidence_received_rows_v516"] != 0:
        raise RuntimeError("v516 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v516"] != 0:
        raise RuntimeError("v516 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v516"] != 0:
        raise RuntimeError("v516 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v516"] != 0:
        raise RuntimeError("v516 must not allow outcome capture.")
    if status["patch_allowed_rows_v516"] != 0:
        raise RuntimeError("v516 must not approve a Quarto patch.")

    SECOND_REMINDER_MD.write_text(_second_reminder_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v516": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
