#!/usr/bin/env python3
"""Build Paper 4 v510 candidate resolution manual-entry packet artifacts."""

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

VERSION = 510
PRIOR_RESOLUTION_GAP_AUDIT_VERSION = 509
NEXT_ARTIFACT = "paper4_v511_post_entry_candidate_resolution_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v510_candidate_resolution_manual_entry_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_entry_packet(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in gaps.iterrows():
        rows.append(
            {
                "manual_entry_packet_id_v510": row[
                    "candidate_resolution_gap_id_v509"
                ],
                "priority_v510": int(row["priority_v509"]),
                "review_domain_v510": row["review_domain_v509"],
                "reviewer_role_required_v510": row["reviewer_role_required_v509"],
                "manual_entry_packet_ready_v510": True,
                "candidate_identifier_required_v510": True,
                "nomination_fields_required_v510": True,
                "candidate_identifier_prefilled_v510": False,
                "candidate_identifier_entered_v510": False,
                "nomination_fields_entered_v510": False,
                "nomination_signoff_recorded_v510": False,
                "candidate_nomination_recorded_v510": False,
                "eligibility_review_allowed_v510": False,
                "reviewer_assignment_allowed_v510": False,
                "outcome_capture_allowed_v510": False,
                "patch_allowed_v510": False,
                "required_human_action_v510": (
                    "enter_candidate_identifier_nomination_fields_and_signoff"
                ),
                "claim_boundary_v510": "candidate resolution manual-entry packet only",
            }
        )
    return pd.DataFrame(rows)


def _manual_entry_field_template(field_gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in field_gaps.iterrows():
        rows.append(
            {
                "manual_entry_packet_id_v510": row["resolution_packet_id_v509"],
                "nomination_field_v510": row["nomination_field_v509"],
                "field_required_v510": bool(row["field_resolution_required_v509"]),
                "field_value_prefilled_v510": False,
                "field_value_entered_v510": False,
                "human_entry_required_v510": True,
                "completion_blocker_v510": True,
                "claim_boundary_v510": "manual entry field template only",
            }
        )
    return pd.DataFrame(rows)


def _manual_entry_quality_gate_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "manual_entry_quality_gate_id_v510": (
                    "no_candidate_identifier_entered"
                ),
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": True,
                "quality_gate_result_v510": "candidate identifiers remain blank",
            },
            {
                "manual_entry_quality_gate_id_v510": (
                    "no_nomination_field_values_entered"
                ),
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": True,
                "quality_gate_result_v510": "nomination fields remain blank",
            },
            {
                "manual_entry_quality_gate_id_v510": "no_nomination_signoff",
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": True,
                "quality_gate_result_v510": "nomination signoff remains absent",
            },
            {
                "manual_entry_quality_gate_id_v510": (
                    "no_candidate_nomination_recorded"
                ),
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": True,
                "quality_gate_result_v510": "candidate nominations remain absent",
            },
            {
                "manual_entry_quality_gate_id_v510": "eligibility_review_blocked",
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": False,
                "quality_gate_result_v510": "eligibility review remains blocked",
            },
            {
                "manual_entry_quality_gate_id_v510": "reviewer_assignment_blocked",
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": False,
                "quality_gate_result_v510": "reviewer assignment remains blocked",
            },
            {
                "manual_entry_quality_gate_id_v510": "no_final_promotion",
                "quality_gate_active_v510": True,
                "blocks_resolution_v510": False,
                "quality_gate_result_v510": "final promotion artifact remains absent",
            },
        ]
    )


def _manual_entry_next_action_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_action_id_v510": "enter_candidate_identifiers",
                "priority_v510": 1,
                "recommended_next_v510": True,
                "blocks_resolution_v510": True,
                "claim_boundary_v510": "candidate identifier entry next only",
            },
            {
                "next_action_id_v510": "enter_nomination_field_values",
                "priority_v510": 2,
                "recommended_next_v510": True,
                "blocks_resolution_v510": True,
                "claim_boundary_v510": "nomination field entry next only",
            },
            {
                "next_action_id_v510": "record_nomination_signoff",
                "priority_v510": 3,
                "recommended_next_v510": True,
                "blocks_resolution_v510": True,
                "claim_boundary_v510": "nomination signoff next only",
            },
            {
                "next_action_id_v510": "run_post_entry_candidate_resolution_audit",
                "priority_v510": 4,
                "recommended_next_v510": False,
                "blocks_resolution_v510": False,
                "claim_boundary_v510": "future post-entry audit only",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v510": "candidate_resolution_manual_entry_packet_created",
                "ready_v510": True,
                "evidence_artifact_v510": (
                    "paper4_v510_candidate_resolution_manual_entry_packet.csv"
                ),
                "claim_boundary_v510": "manual entry packet only",
            },
            {
                "readiness_gate_v510": "manual_entry_field_template_created",
                "ready_v510": True,
                "evidence_artifact_v510": (
                    "paper4_v510_manual_entry_field_template.csv"
                ),
                "claim_boundary_v510": "manual entry field template only",
            },
            {
                "readiness_gate_v510": "manual_entry_quality_gates_created",
                "ready_v510": True,
                "evidence_artifact_v510": (
                    "paper4_v510_manual_entry_quality_gate_register.csv"
                ),
                "claim_boundary_v510": "manual entry quality gates only",
            },
            {
                "readiness_gate_v510": "post_entry_candidate_resolution_audit_ready",
                "ready_v510": True,
                "evidence_artifact_v510": (
                    "paper4_v510_manual_entry_next_action_queue.csv"
                ),
                "claim_boundary_v510": "future post-entry audit readiness only",
            },
            {
                "readiness_gate_v510": "candidate_identifiers_entered",
                "ready_v510": False,
                "evidence_artifact_v510": "candidate identifier fields remain blank",
                "claim_boundary_v510": "no candidate identifiers entered",
            },
            {
                "readiness_gate_v510": "candidate_nominations_recorded",
                "ready_v510": False,
                "evidence_artifact_v510": "candidate nominations remain absent",
                "claim_boundary_v510": "no candidates nominated",
            },
            {
                "readiness_gate_v510": "ready_for_quarto_patch",
                "ready_v510": False,
                "evidence_artifact_v510": "manual entry is incomplete",
                "claim_boundary_v510": "patch remains blocked",
            },
            {
                "readiness_gate_v510": "paper4_final_promotion_created",
                "ready_v510": False,
                "evidence_artifact_v510": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v510": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v510_manual_entry_packet_created",
                "allowed": True,
                "artifact": "paper4_v510_candidate_resolution_manual_entry_packet.csv",
                "boundary": "manual entry packet only",
            },
            {
                "claim_id": "v510_manual_entry_field_template_created",
                "allowed": True,
                "artifact": "paper4_v510_manual_entry_field_template.csv",
                "boundary": "manual entry field template only",
            },
            {
                "claim_id": "v510_post_entry_audit_ready",
                "allowed": True,
                "artifact": "paper4_v510_manual_entry_next_action_queue.csv",
                "boundary": "future post-entry audit readiness only",
            },
            {
                "claim_id": "v510_candidates_entered_or_nominated",
                "allowed": False,
                "artifact": "paper4_v510_candidate_resolution_manual_entry_packet.csv",
                "boundary": "no candidates entered or nominated",
            },
            {
                "claim_id": "v510_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v510_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v510_final_promotion",
                "allowed": False,
                "artifact": "paper4_v510_manuscript_readiness_delta.csv",
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
                "claim": "v510 creates a candidate resolution manual-entry packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_candidate_resolution_manual_entry_packet.csv"
                ),
                "boundary": "Candidate resolution manual-entry packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v510 creates manual-entry field templates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_manual_entry_field_template.csv"
                ),
                "boundary": "Field template only; no field values entered.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v510 makes post-entry candidate resolution audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_manual_entry_next_action_queue.csv"
                ),
                "boundary": "Future post-entry audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v510 enters candidate identifiers or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_candidate_resolution_manual_entry_packet.csv"
                ),
                "boundary": "Candidate identifiers and nominations remain blank.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v510 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v510 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v510_manuscript_readiness_delta.csv"
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
                "executable_item": "v510 creates manual entry packet.",
                "status": "candidate_resolution_manual_entry_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v511 audits post-entry candidate resolution",
                "last_wave": "v510",
                "execution_result": "manual_entry_packet_created_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v510")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Resolution Manual Entry Packet v510

Generated: {status["generated_at_utc"]}

## Result

v510 converts the v509 open candidate-resolution gaps into a manual-entry packet
and field template. It does not enter candidate identifiers, fill nomination
fields, record signoff, nominate candidates, start eligibility review, assign
reviewers or allow outcome capture.

## Counts

- Manual entry packet rows: `{status["manual_entry_packet_rows_v510"]}`.
- Manual entry packet ready rows: `{status["manual_entry_packet_ready_rows_v510"]}`.
- Candidate identifier required rows: `{status["candidate_identifier_required_rows_v510"]}`.
- Candidate identifier entered rows: `{status["candidate_identifier_entered_rows_v510"]}`.
- Candidate identifier prefilled rows: `{status["candidate_identifier_prefilled_rows_v510"]}`.
- Nomination fields entered rows: `{status["nomination_fields_entered_rows_v510"]}`.
- Nomination signoff recorded rows: `{status["nomination_signoff_recorded_rows_v510"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v510"]}`.
- Manual entry field rows: `{status["manual_entry_field_rows_v510"]}`.
- Field value entered rows: `{status["field_value_entered_rows_v510"]}`.
- Field value prefilled rows: `{status["field_value_prefilled_rows_v510"]}`.
- Manual entry quality gate rows: `{status["manual_entry_quality_gate_rows_v510"]}`.
- Active manual entry quality gate rows: `{status["active_manual_entry_quality_gate_rows_v510"]}`.
- Recommended next action rows: `{status["recommended_next_action_rows_v510"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v510"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v510"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v510"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v510"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v510"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v510 is a manual-entry packet only. It does not enter candidate identifiers,
resolve or nominate candidates, assign reviewers, capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V510_CANDIDATE_RESOLUTION_MANUAL_ENTRY_PACKET_START -->"
    end = "<!-- V510_CANDIDATE_RESOLUTION_MANUAL_ENTRY_PACKET_END -->"
    block = f"""
{start}

## Wave v510: Candidate Resolution Manual Entry Packet

Generated: {status["generated_at_utc"]}

### Objective

v510 transforms the v509 resolution gaps into a manual-entry packet and field
template that a human can complete later. It does not enter or invent candidate
identifiers.

### Results

- Manual entry packet rows:
  `{status["manual_entry_packet_rows_v510"]}`.
- Manual entry packet ready rows:
  `{status["manual_entry_packet_ready_rows_v510"]}`.
- Candidate identifier required rows:
  `{status["candidate_identifier_required_rows_v510"]}`.
- Candidate identifier entered rows:
  `{status["candidate_identifier_entered_rows_v510"]}`.
- Candidate identifier prefilled rows:
  `{status["candidate_identifier_prefilled_rows_v510"]}`.
- Nomination fields entered rows:
  `{status["nomination_fields_entered_rows_v510"]}`.
- Nomination signoff recorded rows:
  `{status["nomination_signoff_recorded_rows_v510"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v510"]}`.
- Manual entry field rows:
  `{status["manual_entry_field_rows_v510"]}`.
- Field value entered rows:
  `{status["field_value_entered_rows_v510"]}`.
- Field value prefilled rows:
  `{status["field_value_prefilled_rows_v510"]}`.
- Manual entry quality gate rows:
  `{status["manual_entry_quality_gate_rows_v510"]}`.
- Active manual entry quality gate rows:
  `{status["active_manual_entry_quality_gate_rows_v510"]}`.
- Recommended next action rows:
  `{status["recommended_next_action_rows_v510"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v510"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v510"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v510"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v510"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v510"]}`.
- Book sources modified:
  `{status["book_sources_modified_v510"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v510"]}`.

### Interpretation

Paper 4 now has an auditable manual-entry surface for candidate resolution, but
every candidate identifier, nomination field value, signoff, nomination,
eligibility review, assignment and outcome-capture permission remains absent.

### Claim Impact

- Allowed: manual-entry packet, field template, quality gates and future
  post-entry resolution audit readiness.
- Still prohibited: candidate identifier entry, candidate nomination, reviewer
  assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v510 in the living notebook. v511 should audit candidate resolution after
manual entry, and must still report zero resolved candidates unless real human
entries are present.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v509 = _read_status(PRIOR_RESOLUTION_GAP_AUDIT_VERSION)
    expected_next = "paper4_v510_candidate_resolution_manual_entry_packet.md"
    if v509["next_artifact_v509"] != expected_next:
        raise RuntimeError("v510 expects v509 to route to manual-entry packet.")
    if not v509["candidate_resolution_manual_entry_packet_ready_v509"]:
        raise RuntimeError("v510 requires v509 manual-entry packet readiness.")

    gaps = pd.read_csv(TABLE_DIR / "paper4_v509_candidate_resolution_gap_audit.csv")
    field_gaps = pd.read_csv(TABLE_DIR / "paper4_v509_resolution_field_gap_matrix.csv")
    packet = _manual_entry_packet(gaps)
    fields = _manual_entry_field_template(field_gaps)
    quality_gates = _manual_entry_quality_gate_register()
    next_actions = _manual_entry_next_action_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(
        TABLE_DIR / "paper4_v510_candidate_resolution_manual_entry_packet.csv",
        packet,
    )
    write_csv(TABLE_DIR / "paper4_v510_manual_entry_field_template.csv", fields)
    write_csv(
        TABLE_DIR / "paper4_v510_manual_entry_quality_gate_register.csv",
        quality_gates,
    )
    write_csv(TABLE_DIR / "paper4_v510_manual_entry_next_action_queue.csv", next_actions)
    write_csv(TABLE_DIR / "paper4_v510_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v510_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v510_candidate_resolution_manual_entry_packet",
        "schema_version": "2026-05-17.510",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_resolution_gap_audit_version_v510": (
            PRIOR_RESOLUTION_GAP_AUDIT_VERSION
        ),
        "candidate_resolution_manual_entry_packet_created_v510": True,
        "manual_entry_packet_rows_v510": len(packet),
        "manual_entry_packet_ready_rows_v510": int(
            packet["manual_entry_packet_ready_v510"].astype(bool).sum()
        ),
        "candidate_identifier_required_rows_v510": int(
            packet["candidate_identifier_required_v510"].astype(bool).sum()
        ),
        "candidate_identifier_entered_rows_v510": int(
            packet["candidate_identifier_entered_v510"].astype(bool).sum()
        ),
        "candidate_identifier_prefilled_rows_v510": int(
            packet["candidate_identifier_prefilled_v510"].astype(bool).sum()
        ),
        "nomination_fields_entered_rows_v510": int(
            packet["nomination_fields_entered_v510"].astype(bool).sum()
        ),
        "nomination_signoff_recorded_rows_v510": int(
            packet["nomination_signoff_recorded_v510"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v510": int(
            packet["candidate_nomination_recorded_v510"].astype(bool).sum()
        ),
        "manual_entry_field_rows_v510": len(fields),
        "field_value_entered_rows_v510": int(
            fields["field_value_entered_v510"].astype(bool).sum()
        ),
        "field_value_prefilled_rows_v510": int(
            fields["field_value_prefilled_v510"].astype(bool).sum()
        ),
        "manual_entry_quality_gate_rows_v510": len(quality_gates),
        "active_manual_entry_quality_gate_rows_v510": int(
            quality_gates["quality_gate_active_v510"].astype(bool).sum()
        ),
        "recommended_next_action_rows_v510": int(
            next_actions["recommended_next_v510"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v510": int(
            packet["eligibility_review_allowed_v510"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v510": int(
            packet["reviewer_assignment_allowed_v510"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v510": int(
            packet["outcome_capture_allowed_v510"].astype(bool).sum()
        ),
        "patch_allowed_rows_v510": int(packet["patch_allowed_v510"].astype(bool).sum()),
        "readiness_delta_rows_v510": len(readiness),
        "post_entry_candidate_resolution_audit_ready_v510": True,
        "ready_for_quarto_patch_v510": False,
        "quarto_patch_applied_v510": False,
        "book_sources_modified_v510": False,
        "book_references_modified_v510": False,
        "submission_ready_claim_allowed_v510": False,
        "working_champion_claim_allowed_v510": False,
        "paper1_promotion_allowed_v510": False,
        "paper4_working_champion_changed_v510": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v510": NEXT_ARTIFACT,
        "claim_boundary": (
            "v510 creates manual-entry packet only; candidate identifiers, "
            "field values, nominations, assignments, outcomes, captions, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v510 must not create final Paper 4 promotion.")
    if status["candidate_identifier_entered_rows_v510"] != 0:
        raise RuntimeError("v510 must not enter candidate identifiers.")
    if status["field_value_entered_rows_v510"] != 0:
        raise RuntimeError("v510 must not enter field values.")
    if status["candidate_nomination_recorded_rows_v510"] != 0:
        raise RuntimeError("v510 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v510"] != 0:
        raise RuntimeError("v510 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v510"] != 0:
        raise RuntimeError("v510 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v510"] != 0:
        raise RuntimeError("v510 must not allow outcome capture.")
    if status["patch_allowed_rows_v510"] != 0:
        raise RuntimeError("v510 must not approve a Quarto patch.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v510": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
