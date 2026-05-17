#!/usr/bin/env python3
"""Build Paper 4 v506 reviewer candidate nomination packet artifacts."""

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

VERSION = 506
PRIOR_ELIGIBILITY_VERSION = 505
NEXT_ARTIFACT = "paper4_v507_candidate_nomination_gap_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v506_reviewer_candidate_nomination_packet.md"
NOMINATION_FIELDS = [
    "candidate_identifier",
    "candidate_affiliation",
    "domain_expertise_evidence",
    "conflict_statement",
    "availability_window",
    "claim_boundary_attestation",
]


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_nomination_packet(checklist: pd.DataFrame) -> pd.DataFrame:
    assignments = (
        checklist[
            [
                "assignment_packet_id_v505",
                "priority_v505",
                "review_domain_v505",
                "reviewer_role_required_v505",
            ]
        ]
        .drop_duplicates()
        .sort_values("priority_v505")
    )
    rows = []
    for _, row in assignments.iterrows():
        rows.append(
            {
                "candidate_nomination_packet_id_v506": row["assignment_packet_id_v505"],
                "priority_v506": int(row["priority_v505"]),
                "review_domain_v506": row["review_domain_v505"],
                "reviewer_role_required_v506": row["reviewer_role_required_v505"],
                "candidate_slot_created_v506": True,
                "candidate_identifier_prefilled_v506": False,
                "candidate_nomination_recorded_v506": False,
                "eligibility_review_started_v506": False,
                "reviewer_assigned_v506": False,
                "outcome_capture_allowed_v506": False,
                "patch_allowed_v506": False,
                "claim_boundary_v506": "candidate nomination packet only",
            }
        )
    return pd.DataFrame(rows)


def _candidate_nomination_field_requirements(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, packet_row in packet.iterrows():
        for field in NOMINATION_FIELDS:
            rows.append(
                {
                    "candidate_nomination_packet_id_v506": (
                        packet_row["candidate_nomination_packet_id_v506"]
                    ),
                    "nomination_field_v506": field,
                    "field_required_v506": True,
                    "field_prefilled_v506": False,
                    "human_entry_required_v506": True,
                    "claim_boundary_v506": "candidate nomination field requirement only",
                }
            )
    return pd.DataFrame(rows)


def _domain_candidate_nomination_summary(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in packet.groupby("review_domain_v506", sort=True):
        rows.append(
            {
                "review_domain_v506": domain,
                "candidate_slot_rows_v506": len(group),
                "candidate_identifier_prefilled_rows_v506": int(
                    group["candidate_identifier_prefilled_v506"].astype(bool).sum()
                ),
                "candidate_nomination_recorded_rows_v506": int(
                    group["candidate_nomination_recorded_v506"].astype(bool).sum()
                ),
                "eligibility_review_started_rows_v506": int(
                    group["eligibility_review_started_v506"].astype(bool).sum()
                ),
                "domain_nomination_gap_open_v506": True,
                "claim_boundary_v506": "domain candidate nomination gap summary only",
            }
        )
    return pd.DataFrame(rows)


def _candidate_nomination_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nomination_control_id_v506": "no_candidate_identifier_prefill",
                "control_active_v506": True,
                "blocks_assignment_v506": True,
                "control_result_v506": "candidate identifiers remain blank",
            },
            {
                "nomination_control_id_v506": "no_candidate_nomination_recorded",
                "control_active_v506": True,
                "blocks_assignment_v506": True,
                "control_result_v506": "candidate nomination remains absent",
            },
            {
                "nomination_control_id_v506": "no_eligibility_review_started",
                "control_active_v506": True,
                "blocks_assignment_v506": True,
                "control_result_v506": "eligibility review remains not started",
            },
            {
                "nomination_control_id_v506": "no_reviewer_assignment",
                "control_active_v506": True,
                "blocks_assignment_v506": True,
                "control_result_v506": "reviewer assignment remains absent",
            },
            {
                "nomination_control_id_v506": "no_outcome_capture",
                "control_active_v506": True,
                "blocks_assignment_v506": False,
                "control_result_v506": "outcome capture remains blocked",
            },
            {
                "nomination_control_id_v506": "no_final_promotion",
                "control_active_v506": True,
                "blocks_assignment_v506": False,
                "control_result_v506": "final promotion artifact remains absent",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v506": "reviewer_candidate_nomination_packet_created",
                "ready_v506": True,
                "evidence_artifact_v506": (
                    "paper4_v506_reviewer_candidate_nomination_packet.csv"
                ),
                "claim_boundary_v506": "candidate nomination packet only",
            },
            {
                "readiness_gate_v506": "nomination_field_requirements_created",
                "ready_v506": True,
                "evidence_artifact_v506": (
                    "paper4_v506_candidate_nomination_field_requirements.csv"
                ),
                "claim_boundary_v506": "field requirements only",
            },
            {
                "readiness_gate_v506": "domain_nomination_summary_created",
                "ready_v506": True,
                "evidence_artifact_v506": (
                    "paper4_v506_domain_candidate_nomination_summary.csv"
                ),
                "claim_boundary_v506": "domain nomination summary only",
            },
            {
                "readiness_gate_v506": "candidate_nomination_gap_audit_ready",
                "ready_v506": True,
                "evidence_artifact_v506": "paper4_v506_candidate_nomination_control_register.csv",
                "claim_boundary_v506": "future nomination gap audit readiness only",
            },
            {
                "readiness_gate_v506": "candidate_identifiers_prefilled",
                "ready_v506": False,
                "evidence_artifact_v506": "candidate identifiers remain blank",
                "claim_boundary_v506": "no candidate identifiers provided",
            },
            {
                "readiness_gate_v506": "candidate_nominations_recorded",
                "ready_v506": False,
                "evidence_artifact_v506": "candidate nomination remains absent",
                "claim_boundary_v506": "no candidates nominated",
            },
            {
                "readiness_gate_v506": "ready_for_quarto_patch",
                "ready_v506": False,
                "evidence_artifact_v506": "candidates, assignments and outcomes absent",
                "claim_boundary_v506": "patch remains blocked",
            },
            {
                "readiness_gate_v506": "paper4_final_promotion_created",
                "ready_v506": False,
                "evidence_artifact_v506": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v506": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v506_candidate_nomination_packet_created",
                "allowed": True,
                "artifact": "paper4_v506_reviewer_candidate_nomination_packet.csv",
                "boundary": "candidate nomination packet only",
            },
            {
                "claim_id": "v506_nomination_fields_declared",
                "allowed": True,
                "artifact": "paper4_v506_candidate_nomination_field_requirements.csv",
                "boundary": "field declaration only",
            },
            {
                "claim_id": "v506_candidate_nomination_gap_audit_ready",
                "allowed": True,
                "artifact": "paper4_v506_candidate_nomination_control_register.csv",
                "boundary": "future nomination gap audit readiness only",
            },
            {
                "claim_id": "v506_candidates_nominated_or_reviewers_assigned",
                "allowed": False,
                "artifact": "paper4_v506_reviewer_candidate_nomination_packet.csv",
                "boundary": "no candidates nominated or reviewers assigned",
            },
            {
                "claim_id": "v506_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v506_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v506_final_promotion",
                "allowed": False,
                "artifact": "paper4_v506_manuscript_readiness_delta.csv",
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
                "claim": "v506 creates reviewer candidate nomination slots for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_reviewer_candidate_nomination_packet.csv"
                ),
                "boundary": "Candidate nomination packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v506 declares required candidate nomination fields.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_candidate_nomination_field_requirements.csv"
                ),
                "boundary": "Candidate nomination field requirements only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v506 makes candidate nomination gap audit executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_candidate_nomination_control_register.csv"
                ),
                "boundary": "Future nomination gap audit readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v506 nominates candidates, assigns reviewers, or captures outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_reviewer_candidate_nomination_packet.csv"
                ),
                "boundary": "Candidates, reviewers and outcomes remain absent.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v506 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v506 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v506_manuscript_readiness_delta.csv"
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
                "executable_item": "v506 creates reviewer candidate nomination packet.",
                "status": "reviewer_candidate_nomination_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v507 audits candidate nomination gaps",
                "last_wave": "v506",
                "execution_result": "candidate_nomination_packet_created_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v506")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Reviewer Candidate Nomination Packet v506

Generated: {status["generated_at_utc"]}

## Result

v506 creates candidate nomination slots for all 14 reviewer assignment packets
and declares six required nomination fields per slot. It does not prefill
candidate identifiers, record nominations, start eligibility review, assign
reviewers or allow outcome capture.

## Counts

- Nomination packet rows: `{status["nomination_packet_rows_v506"]}`.
- Candidate slot rows: `{status["candidate_slot_rows_v506"]}`.
- Candidate identifier prefilled rows: `{status["candidate_identifier_prefilled_rows_v506"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v506"]}`.
- Nomination field rows: `{status["nomination_field_rows_v506"]}`.
- Nomination field prefilled rows: `{status["nomination_field_prefilled_rows_v506"]}`.
- Domain summary rows: `{status["domain_summary_rows_v506"]}`.
- Domains with nomination gaps: `{status["domains_with_nomination_gap_rows_v506"]}`.
- Nomination control rows: `{status["nomination_control_rows_v506"]}`.
- Active nomination control rows: `{status["active_nomination_control_rows_v506"]}`.
- Eligibility review started rows: `{status["eligibility_review_started_rows_v506"]}`.
- Reviewer assigned rows: `{status["reviewer_assigned_rows_v506"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v506"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v506"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v506"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v506 is a candidate nomination packet only. It does not provide candidates,
assign reviewers, capture completed review outcomes, finalize captions, approve
patch scope, edit Quarto, render the book, make Paper 4 submission-ready,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V506_REVIEWER_CANDIDATE_NOMINATION_PACKET_START -->"
    end = "<!-- V506_REVIEWER_CANDIDATE_NOMINATION_PACKET_END -->"
    block = f"""
{start}

## Wave v506: Reviewer Candidate Nomination Packet

Generated: {status["generated_at_utc"]}

### Objective

v506 creates candidate nomination slots and required nomination fields without
providing actual reviewer candidates or assignments.

### Results

- Nomination packet rows:
  `{status["nomination_packet_rows_v506"]}`.
- Candidate slot rows:
  `{status["candidate_slot_rows_v506"]}`.
- Candidate identifier prefilled rows:
  `{status["candidate_identifier_prefilled_rows_v506"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v506"]}`.
- Nomination field rows:
  `{status["nomination_field_rows_v506"]}`.
- Nomination field prefilled rows:
  `{status["nomination_field_prefilled_rows_v506"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v506"]}`.
- Domains with nomination gaps:
  `{status["domains_with_nomination_gap_rows_v506"]}`.
- Nomination control rows:
  `{status["nomination_control_rows_v506"]}`.
- Active nomination control rows:
  `{status["active_nomination_control_rows_v506"]}`.
- Eligibility review started rows:
  `{status["eligibility_review_started_rows_v506"]}`.
- Reviewer assigned rows:
  `{status["reviewer_assigned_rows_v506"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v506"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v506"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v506"]}`.
- Book sources modified:
  `{status["book_sources_modified_v506"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v506"]}`.

### Interpretation

The reviewer workflow now has nomination slots and required fields. Every
candidate identifier, nomination, eligibility review, reviewer assignment and
outcome-capture permission remains absent.

### Claim Impact

- Allowed: candidate nomination packet creation, required field declaration,
  domain nomination gap summary and future gap audit readiness.
- Still prohibited: candidate nomination, reviewer assignment, completed review
  claims, final captions, Quarto patch readiness/application, Quarto/book
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v506 in the living notebook. v507 should audit candidate nomination gaps
before any real candidate nomination is recorded.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v505 = _read_status(PRIOR_ELIGIBILITY_VERSION)
    if v505["next_artifact_v505"] != "paper4_v506_reviewer_candidate_nomination_packet.md":
        raise RuntimeError("v506 expects v505 to route to candidate nomination packet.")
    if not v505["candidate_nomination_packet_ready_v505"]:
        raise RuntimeError("v506 requires v505 candidate nomination packet readiness.")

    checklist = pd.read_csv(TABLE_DIR / "paper4_v505_reviewer_eligibility_checklist.csv")
    packet = _candidate_nomination_packet(checklist)
    fields = _candidate_nomination_field_requirements(packet)
    domain_summary = _domain_candidate_nomination_summary(packet)
    controls = _candidate_nomination_control_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v506_reviewer_candidate_nomination_packet.csv", packet)
    write_csv(TABLE_DIR / "paper4_v506_candidate_nomination_field_requirements.csv", fields)
    write_csv(TABLE_DIR / "paper4_v506_domain_candidate_nomination_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v506_candidate_nomination_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v506_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v506_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v506_reviewer_candidate_nomination_packet",
        "schema_version": "2026-05-17.506",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_eligibility_checklist_version_v506": PRIOR_ELIGIBILITY_VERSION,
        "reviewer_candidate_nomination_packet_created_v506": True,
        "nomination_packet_rows_v506": len(packet),
        "candidate_slot_rows_v506": int(
            packet["candidate_slot_created_v506"].astype(bool).sum()
        ),
        "candidate_identifier_prefilled_rows_v506": int(
            packet["candidate_identifier_prefilled_v506"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v506": int(
            packet["candidate_nomination_recorded_v506"].astype(bool).sum()
        ),
        "nomination_field_rows_v506": len(fields),
        "nomination_field_prefilled_rows_v506": int(
            fields["field_prefilled_v506"].astype(bool).sum()
        ),
        "domain_summary_rows_v506": len(domain_summary),
        "domains_with_nomination_gap_rows_v506": int(
            domain_summary["domain_nomination_gap_open_v506"].astype(bool).sum()
        ),
        "nomination_control_rows_v506": len(controls),
        "active_nomination_control_rows_v506": int(
            controls["control_active_v506"].astype(bool).sum()
        ),
        "eligibility_review_started_rows_v506": int(
            packet["eligibility_review_started_v506"].astype(bool).sum()
        ),
        "reviewer_assigned_rows_v506": int(
            packet["reviewer_assigned_v506"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v506": int(
            packet["outcome_capture_allowed_v506"].astype(bool).sum()
        ),
        "patch_allowed_rows_v506": int(packet["patch_allowed_v506"].astype(bool).sum()),
        "readiness_delta_rows_v506": len(readiness),
        "candidate_nomination_gap_audit_ready_v506": True,
        "ready_for_quarto_patch_v506": False,
        "quarto_patch_applied_v506": False,
        "book_sources_modified_v506": False,
        "book_references_modified_v506": False,
        "submission_ready_claim_allowed_v506": False,
        "working_champion_claim_allowed_v506": False,
        "paper1_promotion_allowed_v506": False,
        "paper4_working_champion_changed_v506": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v506": NEXT_ARTIFACT,
        "claim_boundary": (
            "v506 creates reviewer candidate nomination slots only; candidates, "
            "assignments, outcomes, captions, patching, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v506 must not create final Paper 4 promotion.")
    if status["candidate_identifier_prefilled_rows_v506"] != 0:
        raise RuntimeError("v506 must not prefill candidate identifiers.")
    if status["candidate_nomination_recorded_rows_v506"] != 0:
        raise RuntimeError("v506 must not record candidate nominations.")
    if status["eligibility_review_started_rows_v506"] != 0:
        raise RuntimeError("v506 must not start eligibility review.")
    if status["reviewer_assigned_rows_v506"] != 0:
        raise RuntimeError("v506 must not assign reviewers.")
    if status["outcome_capture_allowed_rows_v506"] != 0:
        raise RuntimeError("v506 must not allow outcome capture.")
    if status["patch_allowed_rows_v506"] != 0:
        raise RuntimeError("v506 must not approve a Quarto patch.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v506": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
