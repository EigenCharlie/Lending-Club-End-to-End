#!/usr/bin/env python3
"""Build Paper 4 v517 second reminder follow-up audit artifacts."""

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

VERSION = 517
PRIOR_CANDIDATE_INPUT_SECOND_REMINDER_PACKET_VERSION = 516
NEXT_ARTIFACT = "paper4_v518_candidate_input_escalation_decision_packet.md"
FOLLOWUP_MD = NOTEBOOK.parent / "paper4_v517_second_reminder_followup_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _second_reminder_followup_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        human_response_received = bool(row["human_response_received_v516"])
        candidate_identifier_received = bool(
            row["candidate_identifier_received_v516"]
        )
        nomination_fields_received = bool(row["nomination_fields_received_v516"])
        nomination_signoff_received = bool(row["nomination_signoff_received_v516"])
        evidence_received = bool(row["evidence_received_v516"])
        complete = (
            human_response_received
            and candidate_identifier_received
            and nomination_fields_received
            and nomination_signoff_received
            and evidence_received
        )
        rows.append(
            {
                "second_reminder_followup_audit_id_v517": row[
                    "second_reminder_id_v516"
                ],
                "priority_v517": int(row["priority_v516"]),
                "review_domain_v517": row["review_domain_v516"],
                "reviewer_role_required_v517": row["reviewer_role_required_v516"],
                "second_reminder_created_v517": bool(
                    row["second_reminder_created_v516"]
                ),
                "human_response_received_v517": human_response_received,
                "candidate_identifier_received_v517": candidate_identifier_received,
                "nomination_fields_received_v517": nomination_fields_received,
                "nomination_signoff_received_v517": nomination_signoff_received,
                "evidence_received_v517": evidence_received,
                "second_reminder_complete_v517": complete,
                "second_reminder_followup_gap_open_v517": not complete,
                "candidate_nomination_recorded_v517": bool(
                    row["candidate_nomination_recorded_v516"]
                ),
                "eligibility_review_allowed_v517": False,
                "reviewer_assignment_allowed_v517": False,
                "outcome_capture_allowed_v517": False,
                "patch_allowed_v517": False,
                "required_next_step_v517": (
                    "prepare_candidate_input_escalation_decision"
                ),
                "claim_boundary_v517": "second reminder follow-up audit only",
            }
        )
    return pd.DataFrame(rows)


def _field_evidence_second_reminder_followup_audit(
    checklist: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in checklist.iterrows():
        field_received = bool(row["field_value_received_v516"])
        evidence_received = bool(row["field_evidence_received_v516"])
        rows.append(
            {
                "second_reminder_followup_audit_id_v517": row[
                    "second_reminder_id_v516"
                ],
                "nomination_field_v517": row["nomination_field_v516"],
                "field_second_reminder_created_v517": bool(
                    row["field_second_reminder_created_v516"]
                ),
                "evidence_second_reminder_created_v517": bool(
                    row["evidence_second_reminder_created_v516"]
                ),
                "field_value_received_v517": field_received,
                "field_evidence_received_v517": evidence_received,
                "field_second_reminder_followup_gap_open_v517": not (
                    field_received and evidence_received
                ),
                "claim_boundary_v517": (
                    "field and evidence second reminder follow-up audit only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _second_reminder_response_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "second_reminder_response_blocker_id_v517": (
                    "no_candidate_identifier_after_second_reminder"
                ),
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": True,
                "required_resolution_v517": (
                    "receive candidate identifier before closing input collection"
                ),
            },
            {
                "second_reminder_response_blocker_id_v517": (
                    "no_nomination_field_after_second_reminder"
                ),
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": True,
                "required_resolution_v517": (
                    "receive nomination fields before closing input collection"
                ),
            },
            {
                "second_reminder_response_blocker_id_v517": (
                    "no_nomination_signoff_after_second_reminder"
                ),
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": True,
                "required_resolution_v517": (
                    "receive nomination signoff before closing input collection"
                ),
            },
            {
                "second_reminder_response_blocker_id_v517": (
                    "no_evidence_after_second_reminder"
                ),
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": True,
                "required_resolution_v517": (
                    "receive evidence before closing input collection"
                ),
            },
            {
                "second_reminder_response_blocker_id_v517": (
                    "eligibility_review_blocked"
                ),
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": False,
                "required_resolution_v517": (
                    "start eligibility only after complete candidate inputs"
                ),
            },
            {
                "second_reminder_response_blocker_id_v517": "no_final_promotion",
                "blocker_open_v517": True,
                "blocks_escalation_decision_v517": False,
                "required_resolution_v517": "keep Paper Estrella protection active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v517": "second_reminder_followup_audit_created",
                "ready_v517": True,
                "evidence_artifact_v517": (
                    "paper4_v517_second_reminder_followup_audit.csv"
                ),
                "claim_boundary_v517": "second reminder follow-up audit only",
            },
            {
                "readiness_gate_v517": (
                    "field_evidence_second_reminder_followup_audit_created"
                ),
                "ready_v517": True,
                "evidence_artifact_v517": (
                    "paper4_v517_field_evidence_second_reminder_followup_audit.csv"
                ),
                "claim_boundary_v517": (
                    "field evidence second reminder follow-up audit only"
                ),
            },
            {
                "readiness_gate_v517": (
                    "second_reminder_response_blocker_register_created"
                ),
                "ready_v517": True,
                "evidence_artifact_v517": (
                    "paper4_v517_second_reminder_response_blocker_register.csv"
                ),
                "claim_boundary_v517": "second reminder response blockers only",
            },
            {
                "readiness_gate_v517": "input_escalation_decision_packet_ready",
                "ready_v517": True,
                "evidence_artifact_v517": (
                    "paper4_v517_second_reminder_response_blocker_register.csv"
                ),
                "claim_boundary_v517": (
                    "future candidate input escalation decision readiness only"
                ),
            },
            {
                "readiness_gate_v517": "candidate_identifiers_received",
                "ready_v517": False,
                "evidence_artifact_v517": "candidate identifiers remain unreceived",
                "claim_boundary_v517": "no candidate identifiers received",
            },
            {
                "readiness_gate_v517": "candidate_nominations_recorded",
                "ready_v517": False,
                "evidence_artifact_v517": "candidate nominations remain absent",
                "claim_boundary_v517": "no candidates nominated",
            },
            {
                "readiness_gate_v517": "ready_for_quarto_patch",
                "ready_v517": False,
                "evidence_artifact_v517": "candidate inputs remain absent",
                "claim_boundary_v517": "patch remains blocked",
            },
            {
                "readiness_gate_v517": "paper4_final_promotion_created",
                "ready_v517": False,
                "evidence_artifact_v517": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v517": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v517_second_reminder_followup_audit_created",
                "allowed": True,
                "artifact": "paper4_v517_second_reminder_followup_audit.csv",
                "boundary": "second reminder follow-up audit only",
            },
            {
                "claim_id": (
                    "v517_field_evidence_second_reminder_followup_audit_created"
                ),
                "allowed": True,
                "artifact": (
                    "paper4_v517_field_evidence_second_reminder_followup_audit.csv"
                ),
                "boundary": "field evidence second reminder follow-up audit only",
            },
            {
                "claim_id": "v517_input_escalation_decision_packet_ready",
                "allowed": True,
                "artifact": (
                    "paper4_v517_second_reminder_response_blocker_register.csv"
                ),
                "boundary": "future input escalation decision readiness only",
            },
            {
                "claim_id": "v517_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v517_second_reminder_followup_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v517_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v517_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v517_final_promotion",
                "allowed": False,
                "artifact": "paper4_v517_manuscript_readiness_delta.csv",
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
                "claim": "v517 audits candidate input second reminder follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_second_reminder_followup_audit.csv"
                ),
                "boundary": "Candidate input second reminder follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v517 audits field and evidence second reminder follow-up.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_field_evidence_second_reminder_followup_audit.csv"
                ),
                "boundary": "Field and evidence second reminder follow-up audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v517 makes candidate input escalation decision executable next."
                ),
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_second_reminder_response_blocker_register.csv"
                ),
                "boundary": (
                    "Future candidate input escalation decision readiness only."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v517 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_second_reminder_followup_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v517 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v517 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v517_manuscript_readiness_delta.csv"
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
                "executable_item": "v517 audits second reminder follow-up.",
                "status": "second_reminder_followup_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v518 records candidate input escalation decision"
                ),
                "last_wave": "v517",
                "execution_result": (
                    "second_reminder_followup_audit_confirmed_no_inputs"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v517")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _followup_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Second Reminder Follow-up Audit v517

Generated: {status["generated_at_utc"]}

## Result

v517 audits the v516 second reminder packet. No human responses, candidate
identifiers, nomination fields, signoffs or evidence have been received after
the second reminder, so all 14 second-reminder follow-up gaps and all 84
field/evidence second-reminder follow-up gaps remain open.

## Counts

- Second reminder follow-up audit rows: `{status["second_reminder_followup_audit_rows_v517"]}`.
- Open second reminder follow-up gap rows: `{status["open_second_reminder_followup_gap_rows_v517"]}`.
- Human response received rows: `{status["human_response_received_rows_v517"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v517"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v517"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v517"]}`.
- Evidence received rows: `{status["evidence_received_rows_v517"]}`.
- Second reminder complete rows: `{status["second_reminder_complete_rows_v517"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v517"]}`.
- Field/evidence second reminder follow-up audit rows: `{status["field_evidence_second_reminder_followup_audit_rows_v517"]}`.
- Open field/evidence second reminder follow-up gap rows: `{status["open_field_evidence_second_reminder_followup_gap_rows_v517"]}`.
- Field value received rows: `{status["field_value_received_rows_v517"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v517"]}`.
- Second reminder response blocker rows: `{status["second_reminder_response_blocker_rows_v517"]}`.
- Open second reminder response blocker rows: `{status["open_second_reminder_response_blocker_rows_v517"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v517"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v517"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v517"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v517"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v517"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v517 is a second-reminder follow-up audit only. It does not receive candidate
inputs, resolve or nominate candidates, assign reviewers, capture completed
review outcomes, finalize captions, approve patch scope, edit Quarto, render
the book, make Paper 4 submission-ready, replace Paper Estrella, or promote
Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V517_SECOND_REMINDER_FOLLOWUP_AUDIT_START -->"
    end = "<!-- V517_SECOND_REMINDER_FOLLOWUP_AUDIT_END -->"
    block = f"""
{start}

## Wave v517: Second Reminder Follow-up Audit

Generated: {status["generated_at_utc"]}

### Objective

v517 audits whether the v516 second reminder packet produced any human response,
candidate identifiers, nomination fields, signoff or evidence. It confirms the
second reminder path remains open without fabricating inputs.

### Results

- Second reminder follow-up audit rows:
  `{status["second_reminder_followup_audit_rows_v517"]}`.
- Open second reminder follow-up gap rows:
  `{status["open_second_reminder_followup_gap_rows_v517"]}`.
- Human response received rows:
  `{status["human_response_received_rows_v517"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v517"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v517"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v517"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v517"]}`.
- Second reminder complete rows:
  `{status["second_reminder_complete_rows_v517"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v517"]}`.
- Field/evidence second reminder follow-up audit rows:
  `{status["field_evidence_second_reminder_followup_audit_rows_v517"]}`.
- Open field/evidence second reminder follow-up gap rows:
  `{status["open_field_evidence_second_reminder_followup_gap_rows_v517"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v517"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v517"]}`.
- Second reminder response blocker rows:
  `{status["second_reminder_response_blocker_rows_v517"]}`.
- Open second reminder response blocker rows:
  `{status["open_second_reminder_response_blocker_rows_v517"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v517"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v517"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v517"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v517"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v517"]}`.
- Book sources modified:
  `{status["book_sources_modified_v517"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v517"]}`.

### Interpretation

The second-reminder follow-up audit finds no received input. The next executable
step is a candidate-input escalation decision packet, not eligibility review,
candidate nomination or a Quarto manuscript patch.

### Claim Impact

- Allowed: second-reminder follow-up audit, field/evidence second-reminder
  follow-up audit and future candidate-input escalation decision readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v517 in the living notebook. v518 should record the candidate-input
escalation decision while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v516 = _read_status(PRIOR_CANDIDATE_INPUT_SECOND_REMINDER_PACKET_VERSION)
    expected_next = "paper4_v517_second_reminder_followup_audit.md"
    if v516["next_artifact_v516"] != expected_next:
        raise RuntimeError("v517 expects v516 to route to follow-up audit.")
    if not v516["second_reminder_followup_audit_ready_v516"]:
        raise RuntimeError("v517 requires v516 follow-up audit readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v516_candidate_input_second_reminder_packet.csv"
    )
    checklist = pd.read_csv(
        TABLE_DIR / "paper4_v516_second_reminder_field_evidence_checklist.csv"
    )
    followup = _second_reminder_followup_audit(packet)
    field_followup = _field_evidence_second_reminder_followup_audit(checklist)
    blockers = _second_reminder_response_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v517_second_reminder_followup_audit.csv", followup)
    write_csv(
        TABLE_DIR / "paper4_v517_field_evidence_second_reminder_followup_audit.csv",
        field_followup,
    )
    write_csv(
        TABLE_DIR / "paper4_v517_second_reminder_response_blocker_register.csv",
        blockers,
    )
    write_csv(TABLE_DIR / "paper4_v517_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v517_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v517_second_reminder_followup_audit",
        "schema_version": "2026-05-17.517",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_candidate_input_second_reminder_packet_version_v517": (
            PRIOR_CANDIDATE_INPUT_SECOND_REMINDER_PACKET_VERSION
        ),
        "second_reminder_followup_audit_created_v517": True,
        "second_reminder_followup_audit_rows_v517": len(followup),
        "open_second_reminder_followup_gap_rows_v517": int(
            followup["second_reminder_followup_gap_open_v517"].astype(bool).sum()
        ),
        "human_response_received_rows_v517": int(
            followup["human_response_received_v517"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v517": int(
            followup["candidate_identifier_received_v517"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v517": int(
            followup["nomination_fields_received_v517"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v517": int(
            followup["nomination_signoff_received_v517"].astype(bool).sum()
        ),
        "evidence_received_rows_v517": int(
            followup["evidence_received_v517"].astype(bool).sum()
        ),
        "second_reminder_complete_rows_v517": int(
            followup["second_reminder_complete_v517"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v517": int(
            followup["candidate_nomination_recorded_v517"].astype(bool).sum()
        ),
        "field_evidence_second_reminder_followup_audit_rows_v517": len(
            field_followup
        ),
        "open_field_evidence_second_reminder_followup_gap_rows_v517": int(
            field_followup[
                "field_second_reminder_followup_gap_open_v517"
            ].astype(bool).sum()
        ),
        "field_value_received_rows_v517": int(
            field_followup["field_value_received_v517"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v517": int(
            field_followup["field_evidence_received_v517"].astype(bool).sum()
        ),
        "second_reminder_response_blocker_rows_v517": len(blockers),
        "open_second_reminder_response_blocker_rows_v517": int(
            blockers["blocker_open_v517"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v517": int(
            followup["eligibility_review_allowed_v517"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v517": int(
            followup["reviewer_assignment_allowed_v517"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v517": int(
            followup["outcome_capture_allowed_v517"].astype(bool).sum()
        ),
        "patch_allowed_rows_v517": int(
            followup["patch_allowed_v517"].astype(bool).sum()
        ),
        "readiness_delta_rows_v517": len(readiness),
        "input_escalation_decision_packet_ready_v517": True,
        "ready_for_quarto_patch_v517": False,
        "quarto_patch_applied_v517": False,
        "book_sources_modified_v517": False,
        "book_references_modified_v517": False,
        "submission_ready_claim_allowed_v517": False,
        "working_champion_claim_allowed_v517": False,
        "paper1_promotion_allowed_v517": False,
        "paper4_working_champion_changed_v517": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v517": NEXT_ARTIFACT,
        "claim_boundary": (
            "v517 audits second reminder follow-up only; input receipt, "
            "candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v517 must not create final Paper 4 promotion.")
    if status["human_response_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive human responses.")
    if status["candidate_identifier_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive candidate identifiers.")
    if status["nomination_fields_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive nomination fields.")
    if status["nomination_signoff_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive nomination signoff.")
    if status["evidence_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive evidence.")
    if status["second_reminder_complete_rows_v517"] != 0:
        raise RuntimeError("v517 must not complete second reminder rows.")
    if status["candidate_nomination_recorded_rows_v517"] != 0:
        raise RuntimeError("v517 must not record candidate nominations.")
    if status["field_value_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive field values.")
    if status["field_evidence_received_rows_v517"] != 0:
        raise RuntimeError("v517 must not receive field evidence.")
    if status["eligibility_review_allowed_rows_v517"] != 0:
        raise RuntimeError("v517 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v517"] != 0:
        raise RuntimeError("v517 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v517"] != 0:
        raise RuntimeError("v517 must not allow outcome capture.")
    if status["patch_allowed_rows_v517"] != 0:
        raise RuntimeError("v517 must not approve a Quarto patch.")

    FOLLOWUP_MD.write_text(_followup_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v517": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
