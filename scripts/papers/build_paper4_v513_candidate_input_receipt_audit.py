#!/usr/bin/env python3
"""Build Paper 4 v513 candidate input receipt audit artifacts."""

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

VERSION = 513
PRIOR_INPUT_REQUEST_PACKET_VERSION = 512
NEXT_ARTIFACT = "paper4_v514_candidate_input_collection_reminder_packet.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v513_candidate_input_receipt_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_input_receipt_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        complete = (
            bool(row["candidate_identifier_received_v512"])
            and bool(row["nomination_fields_received_v512"])
            and bool(row["nomination_signoff_received_v512"])
        )
        rows.append(
            {
                "input_receipt_audit_id_v513": row[
                    "candidate_input_request_id_v512"
                ],
                "priority_v513": int(row["priority_v512"]),
                "review_domain_v513": row["review_domain_v512"],
                "reviewer_role_required_v513": row["reviewer_role_required_v512"],
                "candidate_identifier_received_v513": bool(
                    row["candidate_identifier_received_v512"]
                ),
                "nomination_fields_received_v513": bool(
                    row["nomination_fields_received_v512"]
                ),
                "nomination_signoff_received_v513": bool(
                    row["nomination_signoff_received_v512"]
                ),
                "evidence_received_v513": False,
                "candidate_input_complete_v513": complete,
                "input_receipt_gap_open_v513": not complete,
                "candidate_nomination_recorded_v513": bool(
                    row["candidate_nomination_recorded_v512"]
                ),
                "eligibility_review_allowed_v513": False,
                "reviewer_assignment_allowed_v513": False,
                "outcome_capture_allowed_v513": False,
                "patch_allowed_v513": False,
                "required_next_step_v513": "collect_candidate_inputs_and_evidence",
                "claim_boundary_v513": "candidate input receipt audit only",
            }
        )
    return pd.DataFrame(rows)


def _field_and_evidence_receipt_audit(fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fields.iterrows():
        field_received = bool(row["field_value_received_v512"])
        evidence_received = bool(row["evidence_received_v512"])
        rows.append(
            {
                "input_receipt_audit_id_v513": row[
                    "candidate_input_request_id_v512"
                ],
                "nomination_field_v513": row["nomination_field_v512"],
                "field_request_created_v513": bool(
                    row["field_request_created_v512"]
                ),
                "field_value_received_v513": field_received,
                "evidence_required_v513": bool(row["evidence_required_v512"]),
                "evidence_received_v513": evidence_received,
                "receipt_gap_open_v513": not (field_received and evidence_received),
                "claim_boundary_v513": "field and evidence receipt audit only",
            }
        )
    return pd.DataFrame(rows)


def _evidence_receipt_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in evidence.iterrows():
        received = bool(row["evidence_received_v512"])
        rows.append(
            {
                "evidence_requirement_id_v513": row["evidence_requirement_id_v512"],
                "requirement_active_v513": bool(row["requirement_active_v512"]),
                "evidence_required_v513": bool(row["evidence_required_v512"]),
                "evidence_received_v513": received,
                "evidence_gap_open_v513": not received,
                "required_evidence_v513": row["required_evidence_v512"],
                "claim_boundary_v513": "evidence receipt summary only",
            }
        )
    return pd.DataFrame(rows)


def _receipt_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "receipt_blocker_id_v513": "no_candidate_identifiers_received",
                "blocker_open_v513": True,
                "blocks_nomination_v513": True,
                "required_resolution_v513": "receive candidate identifiers",
            },
            {
                "receipt_blocker_id_v513": "no_nomination_fields_received",
                "blocker_open_v513": True,
                "blocks_nomination_v513": True,
                "required_resolution_v513": "receive nomination fields",
            },
            {
                "receipt_blocker_id_v513": "no_nomination_signoff_received",
                "blocker_open_v513": True,
                "blocks_nomination_v513": True,
                "required_resolution_v513": "receive nomination signoff",
            },
            {
                "receipt_blocker_id_v513": "no_evidence_received",
                "blocker_open_v513": True,
                "blocks_nomination_v513": True,
                "required_resolution_v513": "receive evidence for requested inputs",
            },
            {
                "receipt_blocker_id_v513": "eligibility_review_blocked",
                "blocker_open_v513": True,
                "blocks_nomination_v513": False,
                "required_resolution_v513": "start eligibility only after nomination",
            },
            {
                "receipt_blocker_id_v513": "reviewer_assignment_blocked",
                "blocker_open_v513": True,
                "blocks_nomination_v513": False,
                "required_resolution_v513": "assign reviewers only after eligibility",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v513": "candidate_input_receipt_audit_created",
                "ready_v513": True,
                "evidence_artifact_v513": (
                    "paper4_v513_candidate_input_receipt_audit.csv"
                ),
                "claim_boundary_v513": "input receipt audit only",
            },
            {
                "readiness_gate_v513": "field_and_evidence_receipt_audit_created",
                "ready_v513": True,
                "evidence_artifact_v513": (
                    "paper4_v513_field_and_evidence_receipt_audit.csv"
                ),
                "claim_boundary_v513": "field and evidence receipt audit only",
            },
            {
                "readiness_gate_v513": "evidence_receipt_summary_created",
                "ready_v513": True,
                "evidence_artifact_v513": (
                    "paper4_v513_evidence_receipt_summary.csv"
                ),
                "claim_boundary_v513": "evidence receipt summary only",
            },
            {
                "readiness_gate_v513": "input_collection_reminder_packet_ready",
                "ready_v513": True,
                "evidence_artifact_v513": "paper4_v513_receipt_blocker_register.csv",
                "claim_boundary_v513": "future collection reminder readiness only",
            },
            {
                "readiness_gate_v513": "candidate_identifiers_received",
                "ready_v513": False,
                "evidence_artifact_v513": "candidate identifiers remain unreceived",
                "claim_boundary_v513": "no candidate identifiers received",
            },
            {
                "readiness_gate_v513": "candidate_nominations_recorded",
                "ready_v513": False,
                "evidence_artifact_v513": "candidate nominations remain absent",
                "claim_boundary_v513": "no candidates nominated",
            },
            {
                "readiness_gate_v513": "ready_for_quarto_patch",
                "ready_v513": False,
                "evidence_artifact_v513": "candidate inputs remain absent",
                "claim_boundary_v513": "patch remains blocked",
            },
            {
                "readiness_gate_v513": "paper4_final_promotion_created",
                "ready_v513": False,
                "evidence_artifact_v513": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v513": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v513_candidate_input_receipt_audit_created",
                "allowed": True,
                "artifact": "paper4_v513_candidate_input_receipt_audit.csv",
                "boundary": "candidate input receipt audit only",
            },
            {
                "claim_id": "v513_evidence_receipt_summary_created",
                "allowed": True,
                "artifact": "paper4_v513_evidence_receipt_summary.csv",
                "boundary": "evidence receipt summary only",
            },
            {
                "claim_id": "v513_input_collection_reminder_ready",
                "allowed": True,
                "artifact": "paper4_v513_manuscript_readiness_delta.csv",
                "boundary": "future collection reminder readiness only",
            },
            {
                "claim_id": "v513_candidate_inputs_received_or_nominated",
                "allowed": False,
                "artifact": "paper4_v513_candidate_input_receipt_audit.csv",
                "boundary": "no candidate inputs received or nominated",
            },
            {
                "claim_id": "v513_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v513_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v513_final_promotion",
                "allowed": False,
                "artifact": "paper4_v513_manuscript_readiness_delta.csv",
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
                "claim": "v513 audits candidate input receipt status.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_candidate_input_receipt_audit.csv"
                ),
                "boundary": "Candidate input receipt audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v513 summarizes evidence receipt gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_evidence_receipt_summary.csv"
                ),
                "boundary": "Evidence receipt summary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v513 makes input collection reminder executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future input collection reminder readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v513 receives candidate inputs or nominates candidates.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_candidate_input_receipt_audit.csv"
                ),
                "boundary": "Candidate inputs remain unreceived.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v513 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v513 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v513_manuscript_readiness_delta.csv"
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
                "executable_item": "v513 audits candidate input receipt.",
                "status": "candidate_input_receipt_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v514 creates input collection reminder packet",
                "last_wave": "v513",
                "execution_result": "candidate_input_receipt_audit_confirmed_no_inputs",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v513")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Input Receipt Audit v513

Generated: {status["generated_at_utc"]}

## Result

v513 audits the v512 candidate input requests. No candidate identifiers,
nomination fields, signoffs or evidence have been received, so all 14 input
receipt gaps and all 84 field/evidence receipt gaps remain open.

## Counts

- Input receipt audit rows: `{status["input_receipt_audit_rows_v513"]}`.
- Open input receipt gap rows: `{status["open_input_receipt_gap_rows_v513"]}`.
- Candidate input complete rows: `{status["candidate_input_complete_rows_v513"]}`.
- Candidate identifier received rows: `{status["candidate_identifier_received_rows_v513"]}`.
- Nomination fields received rows: `{status["nomination_fields_received_rows_v513"]}`.
- Nomination signoff received rows: `{status["nomination_signoff_received_rows_v513"]}`.
- Evidence received rows: `{status["evidence_received_rows_v513"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v513"]}`.
- Field and evidence receipt audit rows: `{status["field_and_evidence_receipt_audit_rows_v513"]}`.
- Open field/evidence receipt gap rows: `{status["open_field_evidence_receipt_gap_rows_v513"]}`.
- Field value received rows: `{status["field_value_received_rows_v513"]}`.
- Field evidence received rows: `{status["field_evidence_received_rows_v513"]}`.
- Evidence receipt summary rows: `{status["evidence_receipt_summary_rows_v513"]}`.
- Open evidence gap rows: `{status["open_evidence_gap_rows_v513"]}`.
- Receipt blocker rows: `{status["receipt_blocker_rows_v513"]}`.
- Open receipt blocker rows: `{status["open_receipt_blocker_rows_v513"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v513"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v513"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v513"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v513"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v513"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v513 is an input-receipt audit only. It does not receive candidate inputs,
resolve or nominate candidates, assign reviewers, capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V513_CANDIDATE_INPUT_RECEIPT_AUDIT_START -->"
    end = "<!-- V513_CANDIDATE_INPUT_RECEIPT_AUDIT_END -->"
    block = f"""
{start}

## Wave v513: Candidate Input Receipt Audit

Generated: {status["generated_at_utc"]}

### Objective

v513 audits whether the v512 candidate input requests produced received inputs
or evidence. None are present, so the candidate path remains blocked.

### Results

- Input receipt audit rows:
  `{status["input_receipt_audit_rows_v513"]}`.
- Open input receipt gap rows:
  `{status["open_input_receipt_gap_rows_v513"]}`.
- Candidate input complete rows:
  `{status["candidate_input_complete_rows_v513"]}`.
- Candidate identifier received rows:
  `{status["candidate_identifier_received_rows_v513"]}`.
- Nomination fields received rows:
  `{status["nomination_fields_received_rows_v513"]}`.
- Nomination signoff received rows:
  `{status["nomination_signoff_received_rows_v513"]}`.
- Evidence received rows:
  `{status["evidence_received_rows_v513"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v513"]}`.
- Field and evidence receipt audit rows:
  `{status["field_and_evidence_receipt_audit_rows_v513"]}`.
- Open field/evidence receipt gap rows:
  `{status["open_field_evidence_receipt_gap_rows_v513"]}`.
- Field value received rows:
  `{status["field_value_received_rows_v513"]}`.
- Field evidence received rows:
  `{status["field_evidence_received_rows_v513"]}`.
- Evidence receipt summary rows:
  `{status["evidence_receipt_summary_rows_v513"]}`.
- Open evidence gap rows:
  `{status["open_evidence_gap_rows_v513"]}`.
- Receipt blocker rows:
  `{status["receipt_blocker_rows_v513"]}`.
- Open receipt blocker rows:
  `{status["open_receipt_blocker_rows_v513"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v513"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v513"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v513"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v513"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v513"]}`.
- Book sources modified:
  `{status["book_sources_modified_v513"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v513"]}`.

### Interpretation

The candidate input request exists, but no requested input or evidence has been
received. The next executable artifact should remind or collect inputs, not
start eligibility review.

### Claim Impact

- Allowed: input receipt audit, field/evidence receipt audit, evidence receipt
  summary and future input collection reminder readiness.
- Still prohibited: candidate input receipt, candidate resolution/nomination,
  reviewer assignment, completed review claims, final captions, Quarto patch
  readiness/application, Quarto/book mutation, submission readiness, Paper
  Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v513 in the living notebook. v514 should create an input collection
reminder packet while preserving the no-fabricated-candidate boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v512 = _read_status(PRIOR_INPUT_REQUEST_PACKET_VERSION)
    expected_next = "paper4_v513_candidate_input_receipt_audit.md"
    if v512["next_artifact_v512"] != expected_next:
        raise RuntimeError("v513 expects v512 to route to input receipt audit.")
    if not v512["candidate_input_receipt_audit_ready_v512"]:
        raise RuntimeError("v513 requires v512 input receipt audit readiness.")

    packet = pd.read_csv(TABLE_DIR / "paper4_v512_candidate_input_request_packet.csv")
    fields = pd.read_csv(
        TABLE_DIR / "paper4_v512_candidate_input_field_request_matrix.csv"
    )
    evidence = pd.read_csv(TABLE_DIR / "paper4_v512_evidence_requirement_register.csv")
    audit = _candidate_input_receipt_audit(packet)
    field_audit = _field_and_evidence_receipt_audit(fields)
    evidence_summary = _evidence_receipt_summary(evidence)
    blockers = _receipt_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v513_candidate_input_receipt_audit.csv", audit)
    write_csv(TABLE_DIR / "paper4_v513_field_and_evidence_receipt_audit.csv", field_audit)
    write_csv(TABLE_DIR / "paper4_v513_evidence_receipt_summary.csv", evidence_summary)
    write_csv(TABLE_DIR / "paper4_v513_receipt_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v513_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v513_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v513_candidate_input_receipt_audit",
        "schema_version": "2026-05-17.513",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_input_request_packet_version_v513": PRIOR_INPUT_REQUEST_PACKET_VERSION,
        "candidate_input_receipt_audit_created_v513": True,
        "input_receipt_audit_rows_v513": len(audit),
        "open_input_receipt_gap_rows_v513": int(
            audit["input_receipt_gap_open_v513"].astype(bool).sum()
        ),
        "candidate_input_complete_rows_v513": int(
            audit["candidate_input_complete_v513"].astype(bool).sum()
        ),
        "candidate_identifier_received_rows_v513": int(
            audit["candidate_identifier_received_v513"].astype(bool).sum()
        ),
        "nomination_fields_received_rows_v513": int(
            audit["nomination_fields_received_v513"].astype(bool).sum()
        ),
        "nomination_signoff_received_rows_v513": int(
            audit["nomination_signoff_received_v513"].astype(bool).sum()
        ),
        "evidence_received_rows_v513": int(
            audit["evidence_received_v513"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v513": int(
            audit["candidate_nomination_recorded_v513"].astype(bool).sum()
        ),
        "field_and_evidence_receipt_audit_rows_v513": len(field_audit),
        "open_field_evidence_receipt_gap_rows_v513": int(
            field_audit["receipt_gap_open_v513"].astype(bool).sum()
        ),
        "field_value_received_rows_v513": int(
            field_audit["field_value_received_v513"].astype(bool).sum()
        ),
        "field_evidence_received_rows_v513": int(
            field_audit["evidence_received_v513"].astype(bool).sum()
        ),
        "evidence_receipt_summary_rows_v513": len(evidence_summary),
        "open_evidence_gap_rows_v513": int(
            evidence_summary["evidence_gap_open_v513"].astype(bool).sum()
        ),
        "receipt_blocker_rows_v513": len(blockers),
        "open_receipt_blocker_rows_v513": int(
            blockers["blocker_open_v513"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v513": int(
            audit["eligibility_review_allowed_v513"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v513": int(
            audit["reviewer_assignment_allowed_v513"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v513": int(
            audit["outcome_capture_allowed_v513"].astype(bool).sum()
        ),
        "patch_allowed_rows_v513": int(audit["patch_allowed_v513"].astype(bool).sum()),
        "readiness_delta_rows_v513": len(readiness),
        "input_collection_reminder_packet_ready_v513": True,
        "ready_for_quarto_patch_v513": False,
        "quarto_patch_applied_v513": False,
        "book_sources_modified_v513": False,
        "book_references_modified_v513": False,
        "submission_ready_claim_allowed_v513": False,
        "working_champion_claim_allowed_v513": False,
        "paper1_promotion_allowed_v513": False,
        "paper4_working_champion_changed_v513": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v513": NEXT_ARTIFACT,
        "claim_boundary": (
            "v513 audits candidate input receipt only; input receipt, "
            "candidate resolution, nominations, assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v513 must not create final Paper 4 promotion.")
    if status["candidate_input_complete_rows_v513"] != 0:
        raise RuntimeError("v513 must not complete candidate inputs.")
    if status["candidate_identifier_received_rows_v513"] != 0:
        raise RuntimeError("v513 must not receive candidate identifiers.")
    if status["candidate_nomination_recorded_rows_v513"] != 0:
        raise RuntimeError("v513 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v513"] != 0:
        raise RuntimeError("v513 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v513"] != 0:
        raise RuntimeError("v513 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v513"] != 0:
        raise RuntimeError("v513 must not allow outcome capture.")
    if status["patch_allowed_rows_v513"] != 0:
        raise RuntimeError("v513 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v513": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
