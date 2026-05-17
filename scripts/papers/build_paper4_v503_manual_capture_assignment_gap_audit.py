#!/usr/bin/env python3
"""Build Paper 4 v503 manual capture assignment gap audit artifacts."""

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

VERSION = 503
PRIOR_MANUAL_PACKET_VERSION = 502
NEXT_ARTIFACT = "paper4_v504_reviewer_assignment_packet.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v503_manual_capture_assignment_gap_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _assignment_gap_audit(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in assignments.iterrows():
        assigned = bool(row["reviewer_assigned_v502"])
        rows.append(
            {
                "assignment_gap_id_v503": row["assignment_stub_id_v502"],
                "priority_v503": int(row["priority_v502"]),
                "review_domain_v503": row["review_domain_v502"],
                "asset_id_v503": row["asset_id_v502"],
                "assignment_required_v503": bool(row["assignment_required_v502"]),
                "reviewer_assigned_v503": assigned,
                "assignment_gap_open_v503": not assigned,
                "outcome_capture_allowed_v503": False,
                "patch_allowed_v503": False,
                "claim_boundary_v503": "assignment gap audit only",
            }
        )
    return pd.DataFrame(rows)


def _domain_assignment_gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in gaps.groupby("review_domain_v503", sort=True):
        open_gaps = int(group["assignment_gap_open_v503"].astype(bool).sum())
        rows.append(
            {
                "review_domain_v503": domain,
                "assignment_rows_v503": len(group),
                "open_assignment_gap_rows_v503": open_gaps,
                "reviewer_assigned_rows_v503": int(
                    group["reviewer_assigned_v503"].astype(bool).sum()
                ),
                "domain_assignment_gap_open_v503": open_gaps > 0,
                "claim_boundary_v503": "domain assignment gap summary only",
            }
        )
    return pd.DataFrame(rows)


def _assignment_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assignment_blocker_id_v503": "reviewer_assignment_missing",
                "blocker_open_v503": True,
                "blocks_outcome_capture_v503": True,
                "required_resolution_v503": "assign reviewers for all 14 capture packets",
            },
            {
                "assignment_blocker_id_v503": "assignment_signoff_missing",
                "blocker_open_v503": True,
                "blocks_outcome_capture_v503": True,
                "required_resolution_v503": "record assignment signoff before outcomes",
            },
            {
                "assignment_blocker_id_v503": "outcome_capture_not_started",
                "blocker_open_v503": True,
                "blocks_outcome_capture_v503": True,
                "required_resolution_v503": "start capture only after assignments",
            },
            {
                "assignment_blocker_id_v503": "patch_approval_blocked",
                "blocker_open_v503": True,
                "blocks_outcome_capture_v503": False,
                "required_resolution_v503": "keep patch blocked until review signoff",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v503": "assignment_gap_audit_created",
                "ready_v503": True,
                "evidence_artifact_v503": "paper4_v503_assignment_gap_audit.csv",
                "claim_boundary_v503": "assignment gap audit only",
            },
            {
                "readiness_gate_v503": "domain_assignment_gap_summary_created",
                "ready_v503": True,
                "evidence_artifact_v503": "paper4_v503_domain_assignment_gap_summary.csv",
                "claim_boundary_v503": "domain gap summary only",
            },
            {
                "readiness_gate_v503": "assignment_blocker_register_created",
                "ready_v503": True,
                "evidence_artifact_v503": "paper4_v503_assignment_blocker_register.csv",
                "claim_boundary_v503": "blocker register only",
            },
            {
                "readiness_gate_v503": "reviewer_assignment_packet_ready",
                "ready_v503": True,
                "evidence_artifact_v503": "paper4_v503_assignment_gap_audit.csv",
                "claim_boundary_v503": "future assignment packet readiness only",
            },
            {
                "readiness_gate_v503": "reviewers_assigned",
                "ready_v503": False,
                "evidence_artifact_v503": "all assignment gaps remain open",
                "claim_boundary_v503": "reviewers are not assigned",
            },
            {
                "readiness_gate_v503": "real_review_outcomes_captured",
                "ready_v503": False,
                "evidence_artifact_v503": "outcome capture remains blocked",
                "claim_boundary_v503": "no review evidence captured",
            },
            {
                "readiness_gate_v503": "ready_for_quarto_patch",
                "ready_v503": False,
                "evidence_artifact_v503": "assignments, outcomes and approval absent",
                "claim_boundary_v503": "patch remains blocked",
            },
            {
                "readiness_gate_v503": "paper4_final_promotion_created",
                "ready_v503": False,
                "evidence_artifact_v503": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v503": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v503_assignment_gap_audit_created",
                "allowed": True,
                "artifact": "paper4_v503_assignment_gap_audit.csv",
                "boundary": "assignment gap audit only",
            },
            {
                "claim_id": "v503_domain_assignment_gap_summary_created",
                "allowed": True,
                "artifact": "paper4_v503_domain_assignment_gap_summary.csv",
                "boundary": "domain gap summary only",
            },
            {
                "claim_id": "v503_reviewer_assignment_packet_ready",
                "allowed": True,
                "artifact": "paper4_v503_manuscript_readiness_delta.csv",
                "boundary": "future assignment packet readiness only",
            },
            {
                "claim_id": "v503_reviewers_assigned_or_outcomes_captured",
                "allowed": False,
                "artifact": "paper4_v503_assignment_gap_audit.csv",
                "boundary": "reviewers and outcomes remain absent",
            },
            {
                "claim_id": "v503_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v503_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v503_final_promotion",
                "allowed": False,
                "artifact": "paper4_v503_manuscript_readiness_delta.csv",
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
                "claim": "v503 audits open reviewer assignment gaps for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_assignment_gap_audit.csv"
                ),
                "boundary": "Assignment gap audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v503 summarizes assignment gaps by review domain.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_domain_assignment_gap_summary.csv"
                ),
                "boundary": "Domain assignment gap summary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v503 makes a reviewer assignment packet executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future assignment packet readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v503 assigns reviewers or captures review outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_assignment_gap_audit.csv"
                ),
                "boundary": "Reviewers and outcomes remain absent.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v503 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v503 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v503_manuscript_readiness_delta.csv"
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
                "executable_item": "v503 audits manual capture assignment gaps.",
                "status": "manual_capture_assignment_gap_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v504 creates reviewer assignment packet",
                "last_wave": "v503",
                "execution_result": "assignment_gaps_audited_without_assignments",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v503")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Capture Assignment Gap Audit v503

Generated: {status["generated_at_utc"]}

## Result

v503 audits the v502 assignment stubs. All 14 reviewer assignment gaps remain
open, so outcome capture stays blocked. The next executable step is a reviewer
assignment packet, not review outcome capture.

## Counts

- Assignment gap rows: `{status["assignment_gap_rows_v503"]}`.
- Open assignment gap rows: `{status["open_assignment_gap_rows_v503"]}`.
- Reviewer assigned rows: `{status["reviewer_assigned_rows_v503"]}`.
- Domain summary rows: `{status["domain_summary_rows_v503"]}`.
- Domains with open gaps: `{status["domains_with_open_gap_rows_v503"]}`.
- Assignment blocker rows: `{status["assignment_blocker_rows_v503"]}`.
- Open assignment blocker rows: `{status["open_assignment_blocker_rows_v503"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v503"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v503"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v503"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v503 is an assignment-gap audit only. It does not assign reviewers, capture
completed review outcomes, finalize captions, approve patch scope, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V503_MANUAL_CAPTURE_ASSIGNMENT_GAP_AUDIT_START -->"
    end = "<!-- V503_MANUAL_CAPTURE_ASSIGNMENT_GAP_AUDIT_END -->"
    block = f"""
{start}

## Wave v503: Manual Capture Assignment Gap Audit

Generated: {status["generated_at_utc"]}

### Objective

v503 audits whether the v502 manual capture packet has reviewer assignments.
It does not assign reviewers or capture outcomes.

### Results

- Assignment gap rows:
  `{status["assignment_gap_rows_v503"]}`.
- Open assignment gap rows:
  `{status["open_assignment_gap_rows_v503"]}`.
- Reviewer assigned rows:
  `{status["reviewer_assigned_rows_v503"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v503"]}`.
- Domains with open gaps:
  `{status["domains_with_open_gap_rows_v503"]}`.
- Assignment blocker rows:
  `{status["assignment_blocker_rows_v503"]}`.
- Open assignment blocker rows:
  `{status["open_assignment_blocker_rows_v503"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v503"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v503"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v503"]}`.
- Book sources modified:
  `{status["book_sources_modified_v503"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v503"]}`.

### Interpretation

The manual capture packet is prepared, but reviewer assignment remains a hard
blocking gap. The next executable artifact should target assignment, not outcome
capture or Quarto mutation.

### Claim Impact

- Allowed: assignment-gap audit, domain gap summary, blocker register and next
  assignment packet readiness.
- Still prohibited: reviewer assignment, completed review/signoff claims, final
  captions, Quarto patch readiness/application, Quarto/book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v503 in the living notebook. v504 should create a reviewer assignment
packet without assigning reviewers or pre-recording outcomes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v502 = _read_status(PRIOR_MANUAL_PACKET_VERSION)
    if v502["next_artifact_v502"] != "paper4_v503_manual_capture_assignment_gap_audit.md":
        raise RuntimeError("v503 expects v502 to route to assignment gap audit.")
    if not v502["reviewer_assignment_gap_audit_ready_v502"]:
        raise RuntimeError("v503 requires v502 assignment gap audit readiness.")

    assignments = pd.read_csv(TABLE_DIR / "paper4_v502_review_assignment_stub.csv")
    gaps = _assignment_gap_audit(assignments)
    domain_summary = _domain_assignment_gap_summary(gaps)
    blockers = _assignment_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v503_assignment_gap_audit.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v503_domain_assignment_gap_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v503_assignment_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v503_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v503_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v503_manual_capture_assignment_gap_audit",
        "schema_version": "2026-05-17.503",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_packet_version_v503": PRIOR_MANUAL_PACKET_VERSION,
        "manual_capture_assignment_gap_audit_created_v503": True,
        "assignment_gap_rows_v503": len(gaps),
        "open_assignment_gap_rows_v503": int(
            gaps["assignment_gap_open_v503"].astype(bool).sum()
        ),
        "reviewer_assigned_rows_v503": int(
            gaps["reviewer_assigned_v503"].astype(bool).sum()
        ),
        "domain_summary_rows_v503": len(domain_summary),
        "domains_with_open_gap_rows_v503": int(
            domain_summary["domain_assignment_gap_open_v503"].astype(bool).sum()
        ),
        "assignment_blocker_rows_v503": len(blockers),
        "open_assignment_blocker_rows_v503": int(
            blockers["blocker_open_v503"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v503": int(
            gaps["outcome_capture_allowed_v503"].astype(bool).sum()
        ),
        "patch_allowed_rows_v503": int(gaps["patch_allowed_v503"].astype(bool).sum()),
        "readiness_delta_rows_v503": len(readiness),
        "reviewer_assignment_packet_ready_v503": True,
        "ready_for_quarto_patch_v503": False,
        "quarto_patch_applied_v503": False,
        "book_sources_modified_v503": False,
        "book_references_modified_v503": False,
        "submission_ready_claim_allowed_v503": False,
        "working_champion_claim_allowed_v503": False,
        "paper1_promotion_allowed_v503": False,
        "paper4_working_champion_changed_v503": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v503": NEXT_ARTIFACT,
        "claim_boundary": (
            "v503 audits reviewer assignment gaps only; assignments, outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v503 must not create final Paper 4 promotion.")
    if status["reviewer_assigned_rows_v503"] != 0:
        raise RuntimeError("v503 must not assign reviewers.")
    if status["outcome_capture_allowed_rows_v503"] != 0:
        raise RuntimeError("v503 must not allow outcome capture.")
    if status["patch_allowed_rows_v503"] != 0:
        raise RuntimeError("v503 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v503": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
