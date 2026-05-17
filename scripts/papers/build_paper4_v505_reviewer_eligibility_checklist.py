#!/usr/bin/env python3
"""Build Paper 4 v505 reviewer eligibility checklist artifacts."""

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

VERSION = 505
PRIOR_REVIEWER_PACKET_VERSION = 504
NEXT_ARTIFACT = "paper4_v506_reviewer_candidate_nomination_packet.md"
CHECKLIST_MD = NOTEBOOK.parent / "paper4_v505_reviewer_eligibility_checklist.md"
ELIGIBILITY_CRITERIA = [
    "domain_expertise_confirmed",
    "conflict_check_completed",
    "claim_boundary_training_confirmed",
    "availability_confirmed",
]


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _reviewer_eligibility_checklist(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        for criterion in ELIGIBILITY_CRITERIA:
            rows.append(
                {
                    "eligibility_check_id_v505": (
                        f"{row['assignment_packet_id_v504']}::{criterion}"
                    ),
                    "assignment_packet_id_v505": row["assignment_packet_id_v504"],
                    "priority_v505": int(row["priority_v504"]),
                    "review_domain_v505": row["review_domain_v504"],
                    "reviewer_role_required_v505": row["reviewer_role_required_v504"],
                    "eligibility_criterion_v505": criterion,
                    "criterion_declared_v505": True,
                    "candidate_provided_v505": False,
                    "criterion_satisfied_v505": False,
                    "reviewer_eligible_v505": False,
                    "assignment_allowed_v505": False,
                    "outcome_capture_allowed_v505": False,
                    "patch_allowed_v505": False,
                    "claim_boundary_v505": "reviewer eligibility checklist only",
                }
            )
    return pd.DataFrame(rows)


def _domain_eligibility_summary(checklist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in checklist.groupby("review_domain_v505", sort=True):
        rows.append(
            {
                "review_domain_v505": domain,
                "assignment_packet_rows_v505": group["assignment_packet_id_v505"].nunique(),
                "eligibility_check_rows_v505": len(group),
                "candidate_provided_rows_v505": int(
                    group["candidate_provided_v505"].astype(bool).sum()
                ),
                "criterion_satisfied_rows_v505": int(
                    group["criterion_satisfied_v505"].astype(bool).sum()
                ),
                "eligible_reviewer_rows_v505": int(
                    group["reviewer_eligible_v505"].astype(bool).sum()
                ),
                "domain_eligibility_gap_open_v505": True,
                "claim_boundary_v505": "domain eligibility gap summary only",
            }
        )
    return pd.DataFrame(rows)


def _eligibility_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eligibility_blocker_id_v505": "candidate_nomination_missing",
                "blocker_open_v505": True,
                "blocks_assignment_v505": True,
                "required_resolution_v505": "nominate reviewer candidates for all slots",
            },
            {
                "eligibility_blocker_id_v505": "domain_expertise_unverified",
                "blocker_open_v505": True,
                "blocks_assignment_v505": True,
                "required_resolution_v505": "confirm reviewer domain expertise",
            },
            {
                "eligibility_blocker_id_v505": "conflict_check_missing",
                "blocker_open_v505": True,
                "blocks_assignment_v505": True,
                "required_resolution_v505": "complete conflict of interest checks",
            },
            {
                "eligibility_blocker_id_v505": "availability_unconfirmed",
                "blocker_open_v505": True,
                "blocks_assignment_v505": True,
                "required_resolution_v505": "confirm reviewer availability",
            },
            {
                "eligibility_blocker_id_v505": "assignment_signoff_missing",
                "blocker_open_v505": True,
                "blocks_assignment_v505": True,
                "required_resolution_v505": "record assignment signoff after eligibility",
            },
        ]
    )


def _candidate_nomination_next_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_action_id_v505": "nominate_reviewer_candidates",
                "priority_v505": 1,
                "recommended_next_v505": True,
                "blocks_assignment_v505": True,
                "claim_boundary_v505": "candidate nomination next only",
            },
            {
                "next_action_id_v505": "verify_candidate_eligibility",
                "priority_v505": 2,
                "recommended_next_v505": True,
                "blocks_assignment_v505": True,
                "claim_boundary_v505": "eligibility verification next only",
            },
            {
                "next_action_id_v505": "record_candidate_conflict_check",
                "priority_v505": 3,
                "recommended_next_v505": True,
                "blocks_assignment_v505": True,
                "claim_boundary_v505": "conflict check next only",
            },
            {
                "next_action_id_v505": "assign_reviewers_after_eligibility",
                "priority_v505": 4,
                "recommended_next_v505": False,
                "blocks_assignment_v505": False,
                "claim_boundary_v505": "future assignment only after eligibility",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v505": "reviewer_eligibility_checklist_created",
                "ready_v505": True,
                "evidence_artifact_v505": "paper4_v505_reviewer_eligibility_checklist.csv",
                "claim_boundary_v505": "eligibility checklist only",
            },
            {
                "readiness_gate_v505": "domain_eligibility_summary_created",
                "ready_v505": True,
                "evidence_artifact_v505": "paper4_v505_domain_eligibility_summary.csv",
                "claim_boundary_v505": "domain eligibility summary only",
            },
            {
                "readiness_gate_v505": "eligibility_blocker_register_created",
                "ready_v505": True,
                "evidence_artifact_v505": "paper4_v505_eligibility_blocker_register.csv",
                "claim_boundary_v505": "eligibility blockers only",
            },
            {
                "readiness_gate_v505": "candidate_nomination_packet_ready",
                "ready_v505": True,
                "evidence_artifact_v505": "paper4_v505_candidate_nomination_next_queue.csv",
                "claim_boundary_v505": "future nomination packet readiness only",
            },
            {
                "readiness_gate_v505": "reviewer_candidates_provided",
                "ready_v505": False,
                "evidence_artifact_v505": "candidate_provided_v505 remains false",
                "claim_boundary_v505": "no reviewer candidates provided",
            },
            {
                "readiness_gate_v505": "reviewers_eligible_or_assigned",
                "ready_v505": False,
                "evidence_artifact_v505": "eligibility and assignment remain false",
                "claim_boundary_v505": "reviewers are not eligible or assigned",
            },
            {
                "readiness_gate_v505": "ready_for_quarto_patch",
                "ready_v505": False,
                "evidence_artifact_v505": "candidates, assignments and outcomes absent",
                "claim_boundary_v505": "patch remains blocked",
            },
            {
                "readiness_gate_v505": "paper4_final_promotion_created",
                "ready_v505": False,
                "evidence_artifact_v505": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v505": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v505_reviewer_eligibility_checklist_created",
                "allowed": True,
                "artifact": "paper4_v505_reviewer_eligibility_checklist.csv",
                "boundary": "eligibility checklist only",
            },
            {
                "claim_id": "v505_eligibility_gaps_summarized",
                "allowed": True,
                "artifact": "paper4_v505_domain_eligibility_summary.csv",
                "boundary": "eligibility gap summary only",
            },
            {
                "claim_id": "v505_candidate_nomination_packet_ready",
                "allowed": True,
                "artifact": "paper4_v505_candidate_nomination_next_queue.csv",
                "boundary": "future nomination readiness only",
            },
            {
                "claim_id": "v505_candidates_or_reviewers_assigned",
                "allowed": False,
                "artifact": "paper4_v505_reviewer_eligibility_checklist.csv",
                "boundary": "no candidates or reviewer assignments",
            },
            {
                "claim_id": "v505_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v505_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v505_final_promotion",
                "allowed": False,
                "artifact": "paper4_v505_manuscript_readiness_delta.csv",
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
                "claim": "v505 creates reviewer eligibility checks for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_reviewer_eligibility_checklist.csv"
                ),
                "boundary": "Eligibility checklist only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v505 summarizes reviewer eligibility gaps by review domain.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_domain_eligibility_summary.csv"
                ),
                "boundary": "Eligibility gap summary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v505 makes reviewer candidate nomination executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_candidate_nomination_next_queue.csv"
                ),
                "boundary": "Future nomination readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v505 provides candidates, assigns reviewers, or captures outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_reviewer_eligibility_checklist.csv"
                ),
                "boundary": "Candidates, reviewers and outcomes remain absent.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v505 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v505 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v505_manuscript_readiness_delta.csv"
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
                "executable_item": "v505 creates reviewer eligibility checklist.",
                "status": "reviewer_eligibility_checklist_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v506 creates candidate nomination packet",
                "last_wave": "v505",
                "execution_result": "reviewer_eligibility_checklist_created_without_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v505")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _checklist_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Reviewer Eligibility Checklist v505

Generated: {status["generated_at_utc"]}

## Result

v505 creates eligibility checks for the 14 reviewer assignment slots. It
declares four criteria per slot, summarizes eligibility gaps and queues the
candidate nomination packet. It does not provide candidates, mark reviewers
eligible, assign reviewers or allow outcome capture.

## Counts

- Eligibility checklist rows: `{status["eligibility_checklist_rows_v505"]}`.
- Eligibility criteria rows: `{status["eligibility_criteria_rows_v505"]}`.
- Candidate provided rows: `{status["candidate_provided_rows_v505"]}`.
- Criterion satisfied rows: `{status["criterion_satisfied_rows_v505"]}`.
- Eligible reviewer rows: `{status["eligible_reviewer_rows_v505"]}`.
- Domain summary rows: `{status["domain_summary_rows_v505"]}`.
- Domains with eligibility gaps: `{status["domains_with_eligibility_gap_rows_v505"]}`.
- Eligibility blocker rows: `{status["eligibility_blocker_rows_v505"]}`.
- Open eligibility blocker rows: `{status["open_eligibility_blocker_rows_v505"]}`.
- Assignment allowed rows: `{status["assignment_allowed_rows_v505"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v505"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v505"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v505"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v505 is an eligibility checklist only. It does not provide candidates, assign
reviewers, capture completed review outcomes, finalize captions, approve patch
scope, edit Quarto, render the book, make Paper 4 submission-ready, replace
Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V505_REVIEWER_ELIGIBILITY_CHECKLIST_START -->"
    end = "<!-- V505_REVIEWER_ELIGIBILITY_CHECKLIST_END -->"
    block = f"""
{start}

## Wave v505: Reviewer Eligibility Checklist

Generated: {status["generated_at_utc"]}

### Objective

v505 adds eligibility checks to the v504 reviewer assignment slots without
providing candidates or assigning reviewers.

### Results

- Eligibility checklist rows:
  `{status["eligibility_checklist_rows_v505"]}`.
- Eligibility criteria rows:
  `{status["eligibility_criteria_rows_v505"]}`.
- Candidate provided rows:
  `{status["candidate_provided_rows_v505"]}`.
- Criterion satisfied rows:
  `{status["criterion_satisfied_rows_v505"]}`.
- Eligible reviewer rows:
  `{status["eligible_reviewer_rows_v505"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v505"]}`.
- Domains with eligibility gaps:
  `{status["domains_with_eligibility_gap_rows_v505"]}`.
- Eligibility blocker rows:
  `{status["eligibility_blocker_rows_v505"]}`.
- Open eligibility blocker rows:
  `{status["open_eligibility_blocker_rows_v505"]}`.
- Assignment allowed rows:
  `{status["assignment_allowed_rows_v505"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v505"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v505"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v505"]}`.
- Book sources modified:
  `{status["book_sources_modified_v505"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v505"]}`.

### Interpretation

The assignment workflow now has explicit eligibility gates, but every candidate,
criterion satisfaction, eligibility result and assignment remains open.

### Claim Impact

- Allowed: eligibility checklist creation, eligibility gap summary, blocker
  register and candidate nomination packet readiness.
- Still prohibited: candidate provision, reviewer assignment, completed review
  claims, final captions, Quarto patch readiness/application, Quarto/book
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v505 in the living notebook. v506 should create a candidate nomination
packet without nominating actual reviewers or pre-recording outcomes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v504 = _read_status(PRIOR_REVIEWER_PACKET_VERSION)
    if v504["next_artifact_v504"] != "paper4_v505_reviewer_eligibility_checklist.md":
        raise RuntimeError("v505 expects v504 to route to reviewer eligibility checklist.")
    if not v504["reviewer_eligibility_checklist_ready_v504"]:
        raise RuntimeError("v505 requires v504 reviewer eligibility checklist readiness.")

    packet = pd.read_csv(TABLE_DIR / "paper4_v504_reviewer_assignment_packet.csv")
    checklist = _reviewer_eligibility_checklist(packet)
    domain_summary = _domain_eligibility_summary(checklist)
    blockers = _eligibility_blocker_register()
    next_queue = _candidate_nomination_next_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v505_reviewer_eligibility_checklist.csv", checklist)
    write_csv(TABLE_DIR / "paper4_v505_domain_eligibility_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v505_eligibility_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v505_candidate_nomination_next_queue.csv", next_queue)
    write_csv(TABLE_DIR / "paper4_v505_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v505_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v505_reviewer_eligibility_checklist",
        "schema_version": "2026-05-17.505",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_reviewer_packet_version_v505": PRIOR_REVIEWER_PACKET_VERSION,
        "reviewer_eligibility_checklist_created_v505": True,
        "eligibility_checklist_rows_v505": len(checklist),
        "eligibility_criteria_rows_v505": len(ELIGIBILITY_CRITERIA),
        "candidate_provided_rows_v505": int(
            checklist["candidate_provided_v505"].astype(bool).sum()
        ),
        "criterion_satisfied_rows_v505": int(
            checklist["criterion_satisfied_v505"].astype(bool).sum()
        ),
        "eligible_reviewer_rows_v505": int(
            checklist["reviewer_eligible_v505"].astype(bool).sum()
        ),
        "domain_summary_rows_v505": len(domain_summary),
        "domains_with_eligibility_gap_rows_v505": int(
            domain_summary["domain_eligibility_gap_open_v505"].astype(bool).sum()
        ),
        "eligibility_blocker_rows_v505": len(blockers),
        "open_eligibility_blocker_rows_v505": int(
            blockers["blocker_open_v505"].astype(bool).sum()
        ),
        "assignment_allowed_rows_v505": int(
            checklist["assignment_allowed_v505"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v505": int(
            checklist["outcome_capture_allowed_v505"].astype(bool).sum()
        ),
        "patch_allowed_rows_v505": int(checklist["patch_allowed_v505"].astype(bool).sum()),
        "readiness_delta_rows_v505": len(readiness),
        "candidate_nomination_packet_ready_v505": True,
        "ready_for_quarto_patch_v505": False,
        "quarto_patch_applied_v505": False,
        "book_sources_modified_v505": False,
        "book_references_modified_v505": False,
        "submission_ready_claim_allowed_v505": False,
        "working_champion_claim_allowed_v505": False,
        "paper1_promotion_allowed_v505": False,
        "paper4_working_champion_changed_v505": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v505": NEXT_ARTIFACT,
        "claim_boundary": (
            "v505 creates reviewer eligibility checks only; candidates, assignments, "
            "outcomes, captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v505 must not create final Paper 4 promotion.")
    if status["candidate_provided_rows_v505"] != 0:
        raise RuntimeError("v505 must not provide reviewer candidates.")
    if status["eligible_reviewer_rows_v505"] != 0:
        raise RuntimeError("v505 must not mark reviewers eligible.")
    if status["assignment_allowed_rows_v505"] != 0:
        raise RuntimeError("v505 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v505"] != 0:
        raise RuntimeError("v505 must not allow outcome capture.")
    if status["patch_allowed_rows_v505"] != 0:
        raise RuntimeError("v505 must not approve a Quarto patch.")

    CHECKLIST_MD.write_text(_checklist_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v505": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
