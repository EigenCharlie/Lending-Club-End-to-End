#!/usr/bin/env python3
"""Build Paper 4 v507 candidate nomination gap audit artifacts."""

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

VERSION = 507
PRIOR_NOMINATION_PACKET_VERSION = 506
NEXT_ARTIFACT = "paper4_v508_candidate_nomination_resolution_packet.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v507_candidate_nomination_gap_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _candidate_nomination_gap_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        nominated = bool(row["candidate_nomination_recorded_v506"])
        rows.append(
            {
                "candidate_nomination_gap_id_v507": (
                    row["candidate_nomination_packet_id_v506"]
                ),
                "priority_v507": int(row["priority_v506"]),
                "review_domain_v507": row["review_domain_v506"],
                "reviewer_role_required_v507": row["reviewer_role_required_v506"],
                "candidate_slot_created_v507": bool(row["candidate_slot_created_v506"]),
                "candidate_identifier_prefilled_v507": bool(
                    row["candidate_identifier_prefilled_v506"]
                ),
                "candidate_nomination_recorded_v507": nominated,
                "candidate_nomination_gap_open_v507": not nominated,
                "eligibility_review_allowed_v507": False,
                "reviewer_assignment_allowed_v507": False,
                "outcome_capture_allowed_v507": False,
                "patch_allowed_v507": False,
                "claim_boundary_v507": "candidate nomination gap audit only",
            }
        )
    return pd.DataFrame(rows)


def _domain_nomination_gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in gaps.groupby("review_domain_v507", sort=True):
        rows.append(
            {
                "review_domain_v507": domain,
                "candidate_slot_rows_v507": len(group),
                "open_candidate_nomination_gap_rows_v507": int(
                    group["candidate_nomination_gap_open_v507"].astype(bool).sum()
                ),
                "candidate_nomination_recorded_rows_v507": int(
                    group["candidate_nomination_recorded_v507"].astype(bool).sum()
                ),
                "domain_nomination_gap_open_v507": True,
                "claim_boundary_v507": "domain nomination gap summary only",
            }
        )
    return pd.DataFrame(rows)


def _nomination_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nomination_blocker_id_v507": "candidate_identifier_missing",
                "blocker_open_v507": True,
                "blocks_eligibility_review_v507": True,
                "required_resolution_v507": "record candidate identifiers for all slots",
            },
            {
                "nomination_blocker_id_v507": "candidate_nomination_missing",
                "blocker_open_v507": True,
                "blocks_eligibility_review_v507": True,
                "required_resolution_v507": "record candidate nominations",
            },
            {
                "nomination_blocker_id_v507": "nomination_field_values_missing",
                "blocker_open_v507": True,
                "blocks_eligibility_review_v507": True,
                "required_resolution_v507": "complete all six nomination fields",
            },
            {
                "nomination_blocker_id_v507": "eligibility_review_not_started",
                "blocker_open_v507": True,
                "blocks_eligibility_review_v507": False,
                "required_resolution_v507": "start eligibility only after nomination",
            },
            {
                "nomination_blocker_id_v507": "reviewer_assignment_blocked",
                "blocker_open_v507": True,
                "blocks_eligibility_review_v507": False,
                "required_resolution_v507": "assign reviewers only after eligibility",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v507": "candidate_nomination_gap_audit_created",
                "ready_v507": True,
                "evidence_artifact_v507": "paper4_v507_candidate_nomination_gap_audit.csv",
                "claim_boundary_v507": "nomination gap audit only",
            },
            {
                "readiness_gate_v507": "domain_nomination_gap_summary_created",
                "ready_v507": True,
                "evidence_artifact_v507": "paper4_v507_domain_nomination_gap_summary.csv",
                "claim_boundary_v507": "domain gap summary only",
            },
            {
                "readiness_gate_v507": "nomination_blocker_register_created",
                "ready_v507": True,
                "evidence_artifact_v507": "paper4_v507_nomination_blocker_register.csv",
                "claim_boundary_v507": "nomination blocker register only",
            },
            {
                "readiness_gate_v507": "candidate_nomination_resolution_packet_ready",
                "ready_v507": True,
                "evidence_artifact_v507": "paper4_v507_candidate_nomination_gap_audit.csv",
                "claim_boundary_v507": "future resolution packet readiness only",
            },
            {
                "readiness_gate_v507": "candidate_nominations_recorded",
                "ready_v507": False,
                "evidence_artifact_v507": "all nomination gaps remain open",
                "claim_boundary_v507": "no candidates nominated",
            },
            {
                "readiness_gate_v507": "eligibility_review_started",
                "ready_v507": False,
                "evidence_artifact_v507": "eligibility review remains blocked",
                "claim_boundary_v507": "no eligibility review started",
            },
            {
                "readiness_gate_v507": "ready_for_quarto_patch",
                "ready_v507": False,
                "evidence_artifact_v507": "candidates, assignments and outcomes absent",
                "claim_boundary_v507": "patch remains blocked",
            },
            {
                "readiness_gate_v507": "paper4_final_promotion_created",
                "ready_v507": False,
                "evidence_artifact_v507": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v507": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v507_candidate_nomination_gap_audit_created",
                "allowed": True,
                "artifact": "paper4_v507_candidate_nomination_gap_audit.csv",
                "boundary": "nomination gap audit only",
            },
            {
                "claim_id": "v507_nomination_blockers_identified",
                "allowed": True,
                "artifact": "paper4_v507_nomination_blocker_register.csv",
                "boundary": "blocker register only",
            },
            {
                "claim_id": "v507_candidate_nomination_resolution_packet_ready",
                "allowed": True,
                "artifact": "paper4_v507_manuscript_readiness_delta.csv",
                "boundary": "future resolution packet readiness only",
            },
            {
                "claim_id": "v507_candidates_nominated_or_reviewers_assigned",
                "allowed": False,
                "artifact": "paper4_v507_candidate_nomination_gap_audit.csv",
                "boundary": "no candidates nominated or reviewers assigned",
            },
            {
                "claim_id": "v507_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v507_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v507_final_promotion",
                "allowed": False,
                "artifact": "paper4_v507_manuscript_readiness_delta.csv",
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
                "claim": "v507 audits open reviewer candidate nomination gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_candidate_nomination_gap_audit.csv"
                ),
                "boundary": "Candidate nomination gap audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v507 identifies candidate nomination blockers.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_nomination_blocker_register.csv"
                ),
                "boundary": "Nomination blocker register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v507 makes candidate nomination resolution executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future resolution packet readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v507 nominates candidates, assigns reviewers, or captures outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_candidate_nomination_gap_audit.csv"
                ),
                "boundary": "Candidates, reviewers and outcomes remain absent.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v507 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v507 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v507_manuscript_readiness_delta.csv"
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
                "executable_item": "v507 audits candidate nomination gaps.",
                "status": "candidate_nomination_gap_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v508 creates nomination resolution packet",
                "last_wave": "v507",
                "execution_result": "candidate_nomination_gaps_audited_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v507")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Candidate Nomination Gap Audit v507

Generated: {status["generated_at_utc"]}

## Result

v507 audits the v506 candidate nomination packet. All 14 candidate nomination
gaps remain open, so eligibility review, reviewer assignment and outcome capture
remain blocked.

## Counts

- Candidate nomination gap rows: `{status["candidate_nomination_gap_rows_v507"]}`.
- Open candidate nomination gap rows: `{status["open_candidate_nomination_gap_rows_v507"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v507"]}`.
- Domain summary rows: `{status["domain_summary_rows_v507"]}`.
- Domains with nomination gaps: `{status["domains_with_nomination_gap_rows_v507"]}`.
- Nomination blocker rows: `{status["nomination_blocker_rows_v507"]}`.
- Open nomination blocker rows: `{status["open_nomination_blocker_rows_v507"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v507"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v507"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v507"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v507"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v507"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v507 is a nomination-gap audit only. It does not nominate candidates, assign
reviewers, capture completed review outcomes, finalize captions, approve patch
scope, edit Quarto, render the book, make Paper 4 submission-ready, replace
Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V507_CANDIDATE_NOMINATION_GAP_AUDIT_START -->"
    end = "<!-- V507_CANDIDATE_NOMINATION_GAP_AUDIT_END -->"
    block = f"""
{start}

## Wave v507: Candidate Nomination Gap Audit

Generated: {status["generated_at_utc"]}

### Objective

v507 audits whether the v506 candidate nomination packet contains actual
candidate nominations. It does not nominate candidates or start eligibility
review.

### Results

- Candidate nomination gap rows:
  `{status["candidate_nomination_gap_rows_v507"]}`.
- Open candidate nomination gap rows:
  `{status["open_candidate_nomination_gap_rows_v507"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v507"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v507"]}`.
- Domains with nomination gaps:
  `{status["domains_with_nomination_gap_rows_v507"]}`.
- Nomination blocker rows:
  `{status["nomination_blocker_rows_v507"]}`.
- Open nomination blocker rows:
  `{status["open_nomination_blocker_rows_v507"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v507"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v507"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v507"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v507"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v507"]}`.
- Book sources modified:
  `{status["book_sources_modified_v507"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v507"]}`.

### Interpretation

The candidate nomination packet is structurally ready, but every candidate
nomination gap remains open. The next executable artifact should resolve the
nomination process before eligibility review or reviewer assignment.

### Claim Impact

- Allowed: nomination-gap audit, domain gap summary, blocker register and future
  nomination resolution packet readiness.
- Still prohibited: candidate nomination, reviewer assignment, completed review
  claims, final captions, Quarto patch readiness/application, Quarto/book
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v507 in the living notebook. v508 should create a nomination resolution
packet without fabricating candidates or assigning reviewers.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v506 = _read_status(PRIOR_NOMINATION_PACKET_VERSION)
    if v506["next_artifact_v506"] != "paper4_v507_candidate_nomination_gap_audit.md":
        raise RuntimeError("v507 expects v506 to route to candidate nomination gap audit.")
    if not v506["candidate_nomination_gap_audit_ready_v506"]:
        raise RuntimeError("v507 requires v506 candidate nomination gap audit readiness.")

    packet = pd.read_csv(TABLE_DIR / "paper4_v506_reviewer_candidate_nomination_packet.csv")
    gaps = _candidate_nomination_gap_audit(packet)
    domain_summary = _domain_nomination_gap_summary(gaps)
    blockers = _nomination_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v507_candidate_nomination_gap_audit.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v507_domain_nomination_gap_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v507_nomination_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v507_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v507_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v507_candidate_nomination_gap_audit",
        "schema_version": "2026-05-17.507",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_nomination_packet_version_v507": PRIOR_NOMINATION_PACKET_VERSION,
        "candidate_nomination_gap_audit_created_v507": True,
        "candidate_nomination_gap_rows_v507": len(gaps),
        "open_candidate_nomination_gap_rows_v507": int(
            gaps["candidate_nomination_gap_open_v507"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v507": int(
            gaps["candidate_nomination_recorded_v507"].astype(bool).sum()
        ),
        "domain_summary_rows_v507": len(domain_summary),
        "domains_with_nomination_gap_rows_v507": int(
            domain_summary["domain_nomination_gap_open_v507"].astype(bool).sum()
        ),
        "nomination_blocker_rows_v507": len(blockers),
        "open_nomination_blocker_rows_v507": int(
            blockers["blocker_open_v507"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v507": int(
            gaps["eligibility_review_allowed_v507"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v507": int(
            gaps["reviewer_assignment_allowed_v507"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v507": int(
            gaps["outcome_capture_allowed_v507"].astype(bool).sum()
        ),
        "patch_allowed_rows_v507": int(gaps["patch_allowed_v507"].astype(bool).sum()),
        "readiness_delta_rows_v507": len(readiness),
        "candidate_nomination_resolution_packet_ready_v507": True,
        "ready_for_quarto_patch_v507": False,
        "quarto_patch_applied_v507": False,
        "book_sources_modified_v507": False,
        "book_references_modified_v507": False,
        "submission_ready_claim_allowed_v507": False,
        "working_champion_claim_allowed_v507": False,
        "paper1_promotion_allowed_v507": False,
        "paper4_working_champion_changed_v507": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v507": NEXT_ARTIFACT,
        "claim_boundary": (
            "v507 audits candidate nomination gaps only; nominations, assignments, "
            "outcomes, captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v507 must not create final Paper 4 promotion.")
    if status["candidate_nomination_recorded_rows_v507"] != 0:
        raise RuntimeError("v507 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v507"] != 0:
        raise RuntimeError("v507 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v507"] != 0:
        raise RuntimeError("v507 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v507"] != 0:
        raise RuntimeError("v507 must not allow outcome capture.")
    if status["patch_allowed_rows_v507"] != 0:
        raise RuntimeError("v507 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v507": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
