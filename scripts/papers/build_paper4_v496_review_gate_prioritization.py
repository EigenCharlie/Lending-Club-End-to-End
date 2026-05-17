#!/usr/bin/env python3
"""Build Paper 4 v496 review gate prioritization artifacts."""

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

VERSION = 496
PRIOR_NO_PATCH_RELEASE_VERSION = 495
NEXT_ARTIFACT = "paper4_v497_review_gate_execution_packet.md"
PRIORITIZATION_MD = NOTEBOOK.parent / "paper4_v496_review_gate_prioritization.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _review_gate_prioritization() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority_v496": 1,
                "review_gate_id_v496": "manual_layout_surface_review",
                "source_artifact_v496": "paper4_v492_manual_layout_review_packet.csv",
                "gate_status_v496": "pending",
                "recommended_now_v496": True,
                "blocks_patch_v496": True,
                "execution_mode_v496": "manual_review_no_source_mutation",
                "required_next_action_v496": "review four target surfaces",
            },
            {
                "priority_v496": 2,
                "review_gate_id_v496": "caption_claim_safety_review",
                "source_artifact_v496": "paper4_v493_caption_claim_safety_matrix.csv",
                "gate_status_v496": "pending",
                "recommended_now_v496": True,
                "blocks_patch_v496": True,
                "execution_mode_v496": "manual_review_no_source_mutation",
                "required_next_action_v496": "review ten caption claim boundaries",
            },
            {
                "priority_v496": 3,
                "review_gate_id_v496": "final_caption_signoff",
                "source_artifact_v496": "paper4_v493_signoff_action_register.csv",
                "gate_status_v496": "blocked_by_prior_review",
                "recommended_now_v496": True,
                "blocks_patch_v496": True,
                "execution_mode_v496": "signoff_after_claim_safety_review",
                "required_next_action_v496": "capture final caption approval after review",
            },
            {
                "priority_v496": 4,
                "review_gate_id_v496": "explicit_patch_approval_request",
                "source_artifact_v496": "paper4_v494_approval_request_packet.csv",
                "gate_status_v496": "blocked_by_caption_signoff",
                "recommended_now_v496": True,
                "blocks_patch_v496": True,
                "execution_mode_v496": "approval_request_only",
                "required_next_action_v496": "request explicit patch approval after signoff",
            },
            {
                "priority_v496": 5,
                "review_gate_id_v496": "rollback_render_acceptance",
                "source_artifact_v496": "paper4_v494_patch_approval_gap_packet.csv",
                "gate_status_v496": "blocked_by_patch_approval",
                "recommended_now_v496": True,
                "blocks_patch_v496": True,
                "execution_mode_v496": "pre_patch_risk_acceptance",
                "required_next_action_v496": "accept rollback and post-patch render expectations",
            },
            {
                "priority_v496": 6,
                "review_gate_id_v496": "paper_estrella_boundary_guard",
                "source_artifact_v496": "paper4_final_promotion_gate_not_created",
                "gate_status_v496": "active",
                "recommended_now_v496": True,
                "blocks_patch_v496": False,
                "execution_mode_v496": "guardrail_preservation",
                "required_next_action_v496": "keep final promotion absent",
            },
        ]
    )


def _gate_dependency_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_gate_id_v496": "manual_layout_surface_review",
                "depends_on_v496": "v492_manual_layout_review_packet_created",
                "dependency_satisfied_v496": True,
                "executable_next_v496": True,
                "dependency_boundary_v496": "packet exists; manual review pending",
            },
            {
                "review_gate_id_v496": "caption_claim_safety_review",
                "depends_on_v496": "v493_draft_captions_and_boundaries_exist",
                "dependency_satisfied_v496": True,
                "executable_next_v496": True,
                "dependency_boundary_v496": "draft captions exist; safety review pending",
            },
            {
                "review_gate_id_v496": "final_caption_signoff",
                "depends_on_v496": "caption_claim_safety_review_completed",
                "dependency_satisfied_v496": False,
                "executable_next_v496": False,
                "dependency_boundary_v496": "cannot sign off before claim-safety review",
            },
            {
                "review_gate_id_v496": "explicit_patch_approval_request",
                "depends_on_v496": "final_caption_signoff_completed",
                "dependency_satisfied_v496": False,
                "executable_next_v496": False,
                "dependency_boundary_v496": "approval request remains bounded until signoff",
            },
            {
                "review_gate_id_v496": "rollback_render_acceptance",
                "depends_on_v496": "explicit_patch_approval_documented",
                "dependency_satisfied_v496": False,
                "executable_next_v496": False,
                "dependency_boundary_v496": "rollback/render acceptance follows approval",
            },
            {
                "review_gate_id_v496": "paper_estrella_boundary_guard",
                "depends_on_v496": "paper4_final_promotion_absent",
                "dependency_satisfied_v496": True,
                "executable_next_v496": True,
                "dependency_boundary_v496": "guard remains active",
            },
        ]
    )


def _execution_priority_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "queue_item_id_v496": "execute_manual_layout_surface_review",
                "priority_v496": 1,
                "review_gate_id_v496": "manual_layout_surface_review",
                "execution_ready_v496": True,
                "patch_allowed_v496": False,
                "next_packet_v496": NEXT_ARTIFACT,
            },
            {
                "queue_item_id_v496": "execute_caption_claim_safety_review",
                "priority_v496": 2,
                "review_gate_id_v496": "caption_claim_safety_review",
                "execution_ready_v496": True,
                "patch_allowed_v496": False,
                "next_packet_v496": NEXT_ARTIFACT,
            },
            {
                "queue_item_id_v496": "defer_final_caption_signoff",
                "priority_v496": 3,
                "review_gate_id_v496": "final_caption_signoff",
                "execution_ready_v496": False,
                "patch_allowed_v496": False,
                "next_packet_v496": "blocked_until_caption_safety_review",
            },
            {
                "queue_item_id_v496": "defer_explicit_patch_approval_request",
                "priority_v496": 4,
                "review_gate_id_v496": "explicit_patch_approval_request",
                "execution_ready_v496": False,
                "patch_allowed_v496": False,
                "next_packet_v496": "blocked_until_final_caption_signoff",
            },
            {
                "queue_item_id_v496": "defer_rollback_render_acceptance",
                "priority_v496": 5,
                "review_gate_id_v496": "rollback_render_acceptance",
                "execution_ready_v496": False,
                "patch_allowed_v496": False,
                "next_packet_v496": "blocked_until_explicit_patch_approval",
            },
        ]
    )


def _claim_boundary_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id_v496": "v496_review_gate_prioritization_created",
                "allowed_v496": True,
                "evidence_artifact_v496": "paper4_v496_review_gate_prioritization.csv",
                "claim_boundary_v496": "prioritization only",
            },
            {
                "claim_id_v496": "v496_executable_review_gates_identified",
                "allowed_v496": True,
                "evidence_artifact_v496": "paper4_v496_execution_priority_queue.csv",
                "claim_boundary_v496": "review execution queue only",
            },
            {
                "claim_id_v496": "v496_no_patch_constraints_preserved",
                "allowed_v496": True,
                "evidence_artifact_v496": "paper4_v496_gate_dependency_matrix.csv",
                "claim_boundary_v496": "no-patch guardrail only",
            },
            {
                "claim_id_v496": "v496_reviews_or_captions_completed",
                "allowed_v496": False,
                "evidence_artifact_v496": "paper4_v496_execution_priority_queue.csv",
                "claim_boundary_v496": "reviews are prioritized, not completed",
            },
            {
                "claim_id_v496": "v496_patch_ready_or_applied",
                "allowed_v496": False,
                "evidence_artifact_v496": "paper4_v496_manuscript_readiness_delta.csv",
                "claim_boundary_v496": "patch remains blocked",
            },
            {
                "claim_id_v496": "v496_final_promotion",
                "allowed_v496": False,
                "evidence_artifact_v496": "paper4_v496_manuscript_readiness_delta.csv",
                "claim_boundary_v496": "no final promotion claim",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v496": "review_gate_prioritization_created",
                "ready_v496": True,
                "evidence_artifact_v496": "paper4_v496_review_gate_prioritization.csv",
                "claim_boundary_v496": "prioritization only",
            },
            {
                "readiness_gate_v496": "gate_dependency_matrix_created",
                "ready_v496": True,
                "evidence_artifact_v496": "paper4_v496_gate_dependency_matrix.csv",
                "claim_boundary_v496": "dependency matrix only",
            },
            {
                "readiness_gate_v496": "execution_priority_queue_created",
                "ready_v496": True,
                "evidence_artifact_v496": "paper4_v496_execution_priority_queue.csv",
                "claim_boundary_v496": "execution queue only",
            },
            {
                "readiness_gate_v496": "claim_boundary_matrix_created",
                "ready_v496": True,
                "evidence_artifact_v496": "paper4_v496_claim_boundary_matrix.csv",
                "claim_boundary_v496": "claim boundary matrix only",
            },
            {
                "readiness_gate_v496": "ready_for_quarto_patch",
                "ready_v496": False,
                "evidence_artifact_v496": "review, caption signoff and approval gates open",
                "claim_boundary_v496": "patch remains blocked",
            },
            {
                "readiness_gate_v496": "book_sources_or_references_modified",
                "ready_v496": False,
                "evidence_artifact_v496": "book sources unchanged",
                "claim_boundary_v496": "no Quarto/book mutation in v496",
            },
            {
                "readiness_gate_v496": "submission_ready",
                "ready_v496": False,
                "evidence_artifact_v496": "future approval, patch, render and venue gates",
                "claim_boundary_v496": "not a submission package",
            },
            {
                "readiness_gate_v496": "paper4_final_promotion_created",
                "ready_v496": False,
                "evidence_artifact_v496": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v496": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    boundaries = _claim_boundary_matrix()
    return pd.DataFrame(
        [
            {
                "claim_id": row["claim_id_v496"],
                "allowed": bool(row["allowed_v496"]),
                "artifact": row["evidence_artifact_v496"],
                "boundary": row["claim_boundary_v496"],
            }
            for _, row in boundaries.iterrows()
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v496 prioritizes Paper 4 review gates for execution.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_review_gate_prioritization.csv"
                ),
                "boundary": "Review-gate prioritization only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v496 identifies immediately executable review gates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_execution_priority_queue.csv"
                ),
                "boundary": "Execution queue only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v496 preserves no-patch constraints while prioritizing review.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_gate_dependency_matrix.csv"
                ),
                "boundary": "No-patch dependency matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v496 completes manual reviews or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_execution_priority_queue.csv"
                ),
                "boundary": "Reviews are prioritized, not completed.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v496 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v496 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v496_manuscript_readiness_delta.csv"
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
                "executable_item": "v496 prioritizes review gates for execution.",
                "status": "review_gate_prioritization_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v497 packages executable review gates without mutation",
                "last_wave": "v496",
                "execution_result": "review_gate_prioritization_created_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v496")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _prioritization_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Gate Prioritization v496

Generated: {status["generated_at_utc"]}

## Result

v496 prioritizes review gates after the v495 no-patch synthesis. It identifies
two immediately executable review gates and keeps caption signoff, explicit
patch approval, rollback/render acceptance and patching blocked.

## Counts

- Review gate rows: `{status["review_gate_rows_v496"]}`.
- Recommended gate rows: `{status["recommended_gate_rows_v496"]}`.
- Blocking gate rows: `{status["blocking_gate_rows_v496"]}`.
- Dependency rows: `{status["dependency_rows_v496"]}`.
- Dependency satisfied rows: `{status["dependency_satisfied_rows_v496"]}`.
- Execution queue rows: `{status["execution_queue_rows_v496"]}`.
- Executable now rows: `{status["executable_now_rows_v496"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v496"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v496 is a prioritization packet only. It does not complete reviews, finalize
captions, obtain approval, edit Quarto, apply a patch, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V496_REVIEW_GATE_PRIORITIZATION_START -->"
    end = "<!-- V496_REVIEW_GATE_PRIORITIZATION_END -->"
    block = f"""
{start}

## Wave v496: Review Gate Prioritization

Generated: {status["generated_at_utc"]}

### Objective

v496 prioritizes the review gates that emerged from the v495 no-patch release
synthesis without executing a patch or editing book sources.

### Results

- Review gate rows:
  `{status["review_gate_rows_v496"]}`.
- Recommended gate rows:
  `{status["recommended_gate_rows_v496"]}`.
- Blocking gate rows:
  `{status["blocking_gate_rows_v496"]}`.
- Dependency rows:
  `{status["dependency_rows_v496"]}`.
- Dependency satisfied rows:
  `{status["dependency_satisfied_rows_v496"]}`.
- Execution queue rows:
  `{status["execution_queue_rows_v496"]}`.
- Executable now rows:
  `{status["executable_now_rows_v496"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v496"]}`.
- Book sources modified:
  `{status["book_sources_modified_v496"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v496"]}`.

### Interpretation

The next executable work is manual layout review and caption claim-safety
review. Final caption signoff, explicit patch approval and rollback/render
acceptance remain downstream blockers.

### Claim Impact

- Allowed: review-gate prioritization, executable review-gate queue and
  no-patch dependency matrix.
- Still prohibited: completed review/signoff claims, Quarto patch
  readiness/application, Quarto/book-reference mutation, submission readiness,
  Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v496 in the living notebook. v497 should package the executable review
gates without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v495 = _read_status(PRIOR_NO_PATCH_RELEASE_VERSION)
    if v495["next_artifact_v495"] != "paper4_v496_review_gate_prioritization.md":
        raise RuntimeError("v496 expects v495 to route to review gate prioritization.")

    gates = _review_gate_prioritization()
    dependencies = _gate_dependency_matrix()
    queue = _execution_priority_queue()
    boundaries = _claim_boundary_matrix()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v496_review_gate_prioritization.csv", gates)
    write_csv(TABLE_DIR / "paper4_v496_gate_dependency_matrix.csv", dependencies)
    write_csv(TABLE_DIR / "paper4_v496_execution_priority_queue.csv", queue)
    write_csv(TABLE_DIR / "paper4_v496_claim_boundary_matrix.csv", boundaries)
    write_csv(TABLE_DIR / "paper4_v496_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v496_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v496_review_gate_prioritization",
        "schema_version": "2026-05-17.496",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_no_patch_release_version_v496": PRIOR_NO_PATCH_RELEASE_VERSION,
        "review_gate_prioritization_created_v496": True,
        "review_gate_rows_v496": len(gates),
        "recommended_gate_rows_v496": int(gates["recommended_now_v496"].astype(bool).sum()),
        "blocking_gate_rows_v496": int(gates["blocks_patch_v496"].astype(bool).sum()),
        "dependency_rows_v496": len(dependencies),
        "dependency_satisfied_rows_v496": int(
            dependencies["dependency_satisfied_v496"].astype(bool).sum()
        ),
        "dependency_executable_next_rows_v496": int(
            dependencies["executable_next_v496"].astype(bool).sum()
        ),
        "execution_queue_rows_v496": len(queue),
        "executable_now_rows_v496": int(queue["execution_ready_v496"].astype(bool).sum()),
        "blocked_execution_rows_v496": int((~queue["execution_ready_v496"].astype(bool)).sum()),
        "claim_boundary_rows_v496": len(boundaries),
        "allowed_claim_rows_v496": int(boundaries["allowed_v496"].astype(bool).sum()),
        "readiness_delta_rows_v496": len(readiness),
        "ready_for_quarto_patch_v496": False,
        "quarto_patch_applied_v496": False,
        "book_sources_modified_v496": False,
        "book_references_modified_v496": False,
        "submission_ready_claim_allowed_v496": False,
        "working_champion_claim_allowed_v496": False,
        "paper1_promotion_allowed_v496": False,
        "paper4_working_champion_changed_v496": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v496": NEXT_ARTIFACT,
        "claim_boundary": (
            "v496 prioritizes review gates only; reviews, caption signoff, approval, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v496 must not create final Paper 4 promotion.")

    PRIORITIZATION_MD.write_text(_prioritization_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v496": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
