#!/usr/bin/env python3
"""Build Paper 4 v509 candidate resolution gap audit artifacts."""

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

VERSION = 509
PRIOR_RESOLUTION_PACKET_VERSION = 508
NEXT_ARTIFACT = "paper4_v510_candidate_resolution_manual_entry_packet.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v509_candidate_resolution_gap_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_resolution_gap_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        resolved = bool(row["candidate_identifier_resolved_v508"]) and bool(
            row["nomination_fields_completed_v508"]
        )
        rows.append(
            {
                "candidate_resolution_gap_id_v509": row["resolution_packet_id_v508"],
                "priority_v509": int(row["priority_v508"]),
                "review_domain_v509": row["review_domain_v508"],
                "reviewer_role_required_v509": row["reviewer_role_required_v508"],
                "resolution_packet_ready_v509": bool(
                    row["resolution_packet_ready_v508"]
                ),
                "candidate_identifier_resolved_v509": bool(
                    row["candidate_identifier_resolved_v508"]
                ),
                "nomination_fields_completed_v509": bool(
                    row["nomination_fields_completed_v508"]
                ),
                "candidate_nomination_recorded_v509": bool(
                    row["candidate_nomination_recorded_v508"]
                ),
                "candidate_resolution_gap_open_v509": not resolved,
                "eligibility_review_allowed_v509": False,
                "reviewer_assignment_allowed_v509": False,
                "outcome_capture_allowed_v509": False,
                "patch_allowed_v509": False,
                "claim_boundary_v509": "candidate resolution gap audit only",
            }
        )
    return pd.DataFrame(rows)


def _resolution_field_gap_matrix(fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fields.iterrows():
        completed = bool(row["field_completed_v508"])
        rows.append(
            {
                "resolution_packet_id_v509": row["resolution_packet_id_v508"],
                "nomination_field_v509": row["nomination_field_v508"],
                "field_resolution_required_v509": bool(
                    row["field_resolution_required_v508"]
                ),
                "field_completed_v509": completed,
                "field_gap_open_v509": not completed,
                "human_entry_required_v509": bool(row["human_entry_required_v508"]),
                "claim_boundary_v509": "resolution field gap matrix only",
            }
        )
    return pd.DataFrame(rows)


def _resolution_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "resolution_blocker_id_v509": "candidate_identifier_unresolved",
                "blocker_open_v509": True,
                "blocks_nomination_v509": True,
                "required_resolution_v509": "enter candidate identifiers",
            },
            {
                "resolution_blocker_id_v509": "nomination_fields_incomplete",
                "blocker_open_v509": True,
                "blocks_nomination_v509": True,
                "required_resolution_v509": "complete all nomination fields",
            },
            {
                "resolution_blocker_id_v509": "nomination_signoff_missing",
                "blocker_open_v509": True,
                "blocks_nomination_v509": True,
                "required_resolution_v509": "record candidate nominations after field entry",
            },
            {
                "resolution_blocker_id_v509": "eligibility_review_blocked",
                "blocker_open_v509": True,
                "blocks_nomination_v509": False,
                "required_resolution_v509": "start eligibility only after nomination",
            },
            {
                "resolution_blocker_id_v509": "reviewer_assignment_blocked",
                "blocker_open_v509": True,
                "blocks_nomination_v509": False,
                "required_resolution_v509": "assign reviewers only after eligibility",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v509": "candidate_resolution_gap_audit_created",
                "ready_v509": True,
                "evidence_artifact_v509": "paper4_v509_candidate_resolution_gap_audit.csv",
                "claim_boundary_v509": "resolution gap audit only",
            },
            {
                "readiness_gate_v509": "resolution_field_gap_matrix_created",
                "ready_v509": True,
                "evidence_artifact_v509": "paper4_v509_resolution_field_gap_matrix.csv",
                "claim_boundary_v509": "field gap matrix only",
            },
            {
                "readiness_gate_v509": "resolution_blocker_register_created",
                "ready_v509": True,
                "evidence_artifact_v509": "paper4_v509_resolution_blocker_register.csv",
                "claim_boundary_v509": "resolution blocker register only",
            },
            {
                "readiness_gate_v509": "candidate_resolution_manual_entry_packet_ready",
                "ready_v509": True,
                "evidence_artifact_v509": "paper4_v509_candidate_resolution_gap_audit.csv",
                "claim_boundary_v509": "future manual entry packet readiness only",
            },
            {
                "readiness_gate_v509": "candidate_identifiers_resolved",
                "ready_v509": False,
                "evidence_artifact_v509": "all candidate resolution gaps remain open",
                "claim_boundary_v509": "no candidate identifiers resolved",
            },
            {
                "readiness_gate_v509": "candidate_nominations_recorded",
                "ready_v509": False,
                "evidence_artifact_v509": "candidate nominations remain absent",
                "claim_boundary_v509": "no candidates nominated",
            },
            {
                "readiness_gate_v509": "ready_for_quarto_patch",
                "ready_v509": False,
                "evidence_artifact_v509": "candidates, assignments and outcomes absent",
                "claim_boundary_v509": "patch remains blocked",
            },
            {
                "readiness_gate_v509": "paper4_final_promotion_created",
                "ready_v509": False,
                "evidence_artifact_v509": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v509": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v509_candidate_resolution_gap_audit_created",
                "allowed": True,
                "artifact": "paper4_v509_candidate_resolution_gap_audit.csv",
                "boundary": "resolution gap audit only",
            },
            {
                "claim_id": "v509_resolution_field_gaps_mapped",
                "allowed": True,
                "artifact": "paper4_v509_resolution_field_gap_matrix.csv",
                "boundary": "field gap mapping only",
            },
            {
                "claim_id": "v509_candidate_resolution_manual_entry_packet_ready",
                "allowed": True,
                "artifact": "paper4_v509_manuscript_readiness_delta.csv",
                "boundary": "future manual entry packet readiness only",
            },
            {
                "claim_id": "v509_candidates_resolved_or_nominated",
                "allowed": False,
                "artifact": "paper4_v509_candidate_resolution_gap_audit.csv",
                "boundary": "no candidates resolved or nominated",
            },
            {
                "claim_id": "v509_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v509_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v509_final_promotion",
                "allowed": False,
                "artifact": "paper4_v509_manuscript_readiness_delta.csv",
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
                "claim": "v509 audits candidate resolution gaps for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_candidate_resolution_gap_audit.csv"
                ),
                "boundary": "Candidate resolution gap audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v509 maps unresolved candidate nomination fields.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_resolution_field_gap_matrix.csv"
                ),
                "boundary": "Resolution field gap mapping only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v509 makes candidate resolution manual entry executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future manual entry readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v509 resolves candidates, nominates candidates, or assigns reviewers.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_candidate_resolution_gap_audit.csv"
                ),
                "boundary": "Candidates and reviewers remain unresolved.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v509 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v509 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v509_manuscript_readiness_delta.csv"
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
                "executable_item": "v509 audits candidate resolution gaps.",
                "status": "candidate_resolution_gap_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v510 creates candidate resolution manual entry packet",
                "last_wave": "v509",
                "execution_result": "candidate_resolution_gaps_audited_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v509")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Resolution Gap Audit v509

Generated: {status["generated_at_utc"]}

## Result

v509 audits the v508 candidate nomination resolution packet. All 14 candidate
resolution gaps and all 84 field completion gaps remain open, so nomination,
eligibility review, reviewer assignment and outcome capture remain blocked.

## Counts

- Candidate resolution gap rows: `{status["candidate_resolution_gap_rows_v509"]}`.
- Open candidate resolution gap rows: `{status["open_candidate_resolution_gap_rows_v509"]}`.
- Candidate identifier resolved rows: `{status["candidate_identifier_resolved_rows_v509"]}`.
- Nomination fields completed rows: `{status["nomination_fields_completed_rows_v509"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v509"]}`.
- Resolution field gap rows: `{status["resolution_field_gap_rows_v509"]}`.
- Open resolution field gap rows: `{status["open_resolution_field_gap_rows_v509"]}`.
- Field completed rows: `{status["field_completed_rows_v509"]}`.
- Resolution blocker rows: `{status["resolution_blocker_rows_v509"]}`.
- Open resolution blocker rows: `{status["open_resolution_blocker_rows_v509"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v509"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v509"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v509"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v509"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v509"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v509 is a resolution-gap audit only. It does not resolve or nominate candidates,
assign reviewers, capture completed review outcomes, finalize captions, approve
patch scope, edit Quarto, render the book, make Paper 4 submission-ready,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V509_CANDIDATE_RESOLUTION_GAP_AUDIT_START -->"
    end = "<!-- V509_CANDIDATE_RESOLUTION_GAP_AUDIT_END -->"
    block = f"""
{start}

## Wave v509: Candidate Resolution Gap Audit

Generated: {status["generated_at_utc"]}

### Objective

v509 audits whether the v508 resolution packet closed any candidate or field
completion gaps. It does not resolve candidates or nominate reviewers.

### Results

- Candidate resolution gap rows:
  `{status["candidate_resolution_gap_rows_v509"]}`.
- Open candidate resolution gap rows:
  `{status["open_candidate_resolution_gap_rows_v509"]}`.
- Candidate identifier resolved rows:
  `{status["candidate_identifier_resolved_rows_v509"]}`.
- Nomination fields completed rows:
  `{status["nomination_fields_completed_rows_v509"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v509"]}`.
- Resolution field gap rows:
  `{status["resolution_field_gap_rows_v509"]}`.
- Open resolution field gap rows:
  `{status["open_resolution_field_gap_rows_v509"]}`.
- Field completed rows:
  `{status["field_completed_rows_v509"]}`.
- Resolution blocker rows:
  `{status["resolution_blocker_rows_v509"]}`.
- Open resolution blocker rows:
  `{status["open_resolution_blocker_rows_v509"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v509"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v509"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v509"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v509"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v509"]}`.
- Book sources modified:
  `{status["book_sources_modified_v509"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v509"]}`.

### Interpretation

The resolution packet is prepared, but none of the candidate identifier or
nomination-field gaps have been closed. The next executable artifact should
prepare manual entry, not eligibility review or reviewer assignment.

### Claim Impact

- Allowed: resolution-gap audit, field-gap matrix, blocker register and future
  manual-entry packet readiness.
- Still prohibited: candidate resolution/nomination, reviewer assignment,
  completed review claims, final captions, Quarto patch readiness/application,
  Quarto/book mutation, submission readiness, Paper Estrella replacement and
  final Paper 4 promotion.

### Quarto Promotion Decision

Keep v509 in the living notebook. v510 should create a candidate resolution
manual-entry packet without fabricating candidate identifiers.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v508 = _read_status(PRIOR_RESOLUTION_PACKET_VERSION)
    if v508["next_artifact_v508"] != "paper4_v509_candidate_resolution_gap_audit.md":
        raise RuntimeError("v509 expects v508 to route to candidate resolution gap audit.")
    if not v508["candidate_resolution_gap_audit_ready_v508"]:
        raise RuntimeError("v509 requires v508 candidate resolution gap audit readiness.")

    packet = pd.read_csv(TABLE_DIR / "paper4_v508_candidate_nomination_resolution_packet.csv")
    fields = pd.read_csv(TABLE_DIR / "paper4_v508_resolution_field_completion_matrix.csv")
    gaps = _candidate_resolution_gap_audit(packet)
    field_gaps = _resolution_field_gap_matrix(fields)
    blockers = _resolution_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v509_candidate_resolution_gap_audit.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v509_resolution_field_gap_matrix.csv", field_gaps)
    write_csv(TABLE_DIR / "paper4_v509_resolution_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v509_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v509_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v509_candidate_resolution_gap_audit",
        "schema_version": "2026-05-17.509",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_resolution_packet_version_v509": PRIOR_RESOLUTION_PACKET_VERSION,
        "candidate_resolution_gap_audit_created_v509": True,
        "candidate_resolution_gap_rows_v509": len(gaps),
        "open_candidate_resolution_gap_rows_v509": int(
            gaps["candidate_resolution_gap_open_v509"].astype(bool).sum()
        ),
        "candidate_identifier_resolved_rows_v509": int(
            gaps["candidate_identifier_resolved_v509"].astype(bool).sum()
        ),
        "nomination_fields_completed_rows_v509": int(
            gaps["nomination_fields_completed_v509"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v509": int(
            gaps["candidate_nomination_recorded_v509"].astype(bool).sum()
        ),
        "resolution_field_gap_rows_v509": len(field_gaps),
        "open_resolution_field_gap_rows_v509": int(
            field_gaps["field_gap_open_v509"].astype(bool).sum()
        ),
        "field_completed_rows_v509": int(
            field_gaps["field_completed_v509"].astype(bool).sum()
        ),
        "resolution_blocker_rows_v509": len(blockers),
        "open_resolution_blocker_rows_v509": int(
            blockers["blocker_open_v509"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v509": int(
            gaps["eligibility_review_allowed_v509"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v509": int(
            gaps["reviewer_assignment_allowed_v509"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v509": int(
            gaps["outcome_capture_allowed_v509"].astype(bool).sum()
        ),
        "patch_allowed_rows_v509": int(gaps["patch_allowed_v509"].astype(bool).sum()),
        "readiness_delta_rows_v509": len(readiness),
        "candidate_resolution_manual_entry_packet_ready_v509": True,
        "ready_for_quarto_patch_v509": False,
        "quarto_patch_applied_v509": False,
        "book_sources_modified_v509": False,
        "book_references_modified_v509": False,
        "submission_ready_claim_allowed_v509": False,
        "working_champion_claim_allowed_v509": False,
        "paper1_promotion_allowed_v509": False,
        "paper4_working_champion_changed_v509": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v509": NEXT_ARTIFACT,
        "claim_boundary": (
            "v509 audits candidate resolution gaps only; candidate resolution, "
            "nominations, assignments, outcomes, captions, patching, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v509 must not create final Paper 4 promotion.")
    if status["candidate_identifier_resolved_rows_v509"] != 0:
        raise RuntimeError("v509 must not resolve candidate identifiers.")
    if status["candidate_nomination_recorded_rows_v509"] != 0:
        raise RuntimeError("v509 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v509"] != 0:
        raise RuntimeError("v509 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v509"] != 0:
        raise RuntimeError("v509 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v509"] != 0:
        raise RuntimeError("v509 must not allow outcome capture.")
    if status["patch_allowed_rows_v509"] != 0:
        raise RuntimeError("v509 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v509": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
