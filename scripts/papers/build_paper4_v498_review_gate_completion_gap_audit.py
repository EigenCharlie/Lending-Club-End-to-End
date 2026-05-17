#!/usr/bin/env python3
"""Build Paper 4 v498 review gate completion gap audit artifacts."""

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

VERSION = 498
PRIOR_REVIEW_GATE_PACKET_VERSION = 497
NEXT_ARTIFACT = "paper4_v499_review_outcome_capture_template.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v498_review_gate_completion_gap_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _review_gate_completion_gap_audit() -> pd.DataFrame:
    packet = pd.read_csv(TABLE_DIR / "paper4_v497_review_gate_execution_packet.csv")
    rows = []
    action_by_gate = {
        "manual_layout_surface_review": "capture layout surface review outcomes",
        "caption_claim_safety_review": "capture caption claim-safety review outcomes",
    }
    for _, row in packet.sort_values("priority_v497").iterrows():
        completed = bool(row["execution_completed_v497"])
        rows.append(
            {
                "completion_gap_id_v498": row["execution_gate_id_v497"],
                "priority_v498": int(row["priority_v497"]),
                "review_item_count_v498": int(row["review_item_count_v497"]),
                "packet_ready_v498": row["execution_status_v497"]
                == "packet_ready_review_pending",
                "execution_started_v498": bool(row["execution_started_v497"]),
                "execution_completed_v498": completed,
                "completion_gap_open_v498": not completed,
                "required_completion_action_v498": action_by_gate[row["execution_gate_id_v497"]],
                "patch_allowed_v498": False,
            }
        )
    return pd.DataFrame(rows)


def _layout_completion_gap_matrix() -> pd.DataFrame:
    layout = pd.read_csv(TABLE_DIR / "paper4_v497_layout_surface_review_inputs.csv")
    rows = []
    for _, row in layout.iterrows():
        accepted = bool(row["accepted_for_patch_v497"])
        rows.append(
            {
                "layout_completion_gap_id_v498": row["layout_review_input_id_v497"],
                "target_file_v498": row["target_file_v497"],
                "target_block_v498": row["target_block_v497"],
                "asset_sequence_v498": row["asset_sequence_v497"],
                "layout_item_count_v498": int(row["layout_item_count_v497"]),
                "review_status_v498": row["review_status_v497"],
                "review_completed_v498": row["review_status_v497"] == "completed",
                "accepted_for_patch_v498": accepted,
                "completion_gap_open_v498": not accepted,
                "patch_allowed_v498": False,
            }
        )
    return pd.DataFrame(rows)


def _caption_completion_gap_matrix() -> pd.DataFrame:
    captions = pd.read_csv(TABLE_DIR / "paper4_v497_caption_claim_safety_review_inputs.csv")
    rows = []
    for _, row in captions.iterrows():
        accepted = bool(row["accepted_for_final_caption_v497"])
        rows.append(
            {
                "caption_completion_gap_id_v498": row["caption_review_input_id_v497"],
                "asset_id_v498": row["asset_id_v497"],
                "target_block_v498": row["target_block_v497"],
                "draft_caption_exists_v498": bool(row["draft_caption_exists_v497"]),
                "caption_final_v498": bool(row["caption_final_v497"]),
                "overclaim_review_required_v498": bool(row["overclaim_review_required_v497"]),
                "review_status_v498": row["review_status_v497"],
                "review_completed_v498": row["review_status_v497"] == "completed",
                "accepted_for_final_caption_v498": accepted,
                "completion_gap_open_v498": not accepted,
                "patch_allowed_v498": False,
            }
        )
    return pd.DataFrame(rows)


def _completion_blocker_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v498": "manual_layout_surface_review_completion",
                "blocker_open_v498": True,
                "blocks_patch_v498": True,
                "required_resolution_v498": "complete four layout surface reviews",
            },
            {
                "blocker_id_v498": "caption_claim_safety_review_completion",
                "blocker_open_v498": True,
                "blocks_patch_v498": True,
                "required_resolution_v498": "complete ten caption safety reviews",
            },
            {
                "blocker_id_v498": "final_caption_signoff",
                "blocker_open_v498": True,
                "blocks_patch_v498": True,
                "required_resolution_v498": "finalize captions after review",
            },
            {
                "blocker_id_v498": "explicit_patch_approval",
                "blocker_open_v498": True,
                "blocks_patch_v498": True,
                "required_resolution_v498": "obtain explicit approval after signoff",
            },
            {
                "blocker_id_v498": "post_patch_render",
                "blocker_open_v498": True,
                "blocks_patch_v498": True,
                "required_resolution_v498": "render only after a future patch exists",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v498": "review_gate_completion_gap_audit_created",
                "ready_v498": True,
                "evidence_artifact_v498": "paper4_v498_review_gate_completion_gap_audit.csv",
                "claim_boundary_v498": "completion gap audit only",
            },
            {
                "readiness_gate_v498": "layout_completion_gap_matrix_created",
                "ready_v498": True,
                "evidence_artifact_v498": "paper4_v498_layout_completion_gap_matrix.csv",
                "claim_boundary_v498": "layout completion gaps only",
            },
            {
                "readiness_gate_v498": "caption_completion_gap_matrix_created",
                "ready_v498": True,
                "evidence_artifact_v498": "paper4_v498_caption_completion_gap_matrix.csv",
                "claim_boundary_v498": "caption completion gaps only",
            },
            {
                "readiness_gate_v498": "completion_blocker_summary_created",
                "ready_v498": True,
                "evidence_artifact_v498": "paper4_v498_completion_blocker_summary.csv",
                "claim_boundary_v498": "blocker summary only",
            },
            {
                "readiness_gate_v498": "ready_for_quarto_patch",
                "ready_v498": False,
                "evidence_artifact_v498": "review completion gaps remain",
                "claim_boundary_v498": "patch remains blocked",
            },
            {
                "readiness_gate_v498": "book_sources_or_references_modified",
                "ready_v498": False,
                "evidence_artifact_v498": "book sources unchanged",
                "claim_boundary_v498": "no Quarto/book mutation in v498",
            },
            {
                "readiness_gate_v498": "submission_ready",
                "ready_v498": False,
                "evidence_artifact_v498": "future approval, patch, render and venue gates",
                "claim_boundary_v498": "not a submission package",
            },
            {
                "readiness_gate_v498": "paper4_final_promotion_created",
                "ready_v498": False,
                "evidence_artifact_v498": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v498": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v498_review_gate_completion_gap_audit_created",
                "allowed": True,
                "artifact": "paper4_v498_review_gate_completion_gap_audit.csv",
                "boundary": "completion gap audit only",
            },
            {
                "claim_id": "v498_layout_and_caption_completion_gaps_mapped",
                "allowed": True,
                "artifact": "paper4_v498_layout_and_caption_gap_matrices",
                "boundary": "completion gap matrices only",
            },
            {
                "claim_id": "v498_completion_blockers_identified",
                "allowed": True,
                "artifact": "paper4_v498_completion_blocker_summary.csv",
                "boundary": "blocker summary only",
            },
            {
                "claim_id": "v498_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v498_manuscript_readiness_delta.csv",
                "boundary": "completion gaps remain open",
            },
            {
                "claim_id": "v498_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v498_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v498_final_promotion",
                "allowed": False,
                "artifact": "paper4_v498_manuscript_readiness_delta.csv",
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
                "claim": "v498 audits Paper 4 review gate completion gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_review_gate_completion_gap_audit.csv"
                ),
                "boundary": "Completion gap audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v498 maps layout and caption review completion gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_layout_completion_gap_matrix.csv"
                ),
                "boundary": "Completion gap mapping only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v498 identifies open completion blockers before patching.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_completion_blocker_summary.csv"
                ),
                "boundary": "Open blocker summary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v498 completes manual reviews or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_manuscript_readiness_delta.csv"
                ),
                "boundary": "Completion gaps remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v498 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v498 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v498_manuscript_readiness_delta.csv"
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
                "executable_item": "v498 audits review gate completion gaps.",
                "status": "review_gate_completion_gap_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v499 creates review outcome capture template",
                "last_wave": "v498",
                "execution_result": "review_completion_gaps_audited_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v498")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Gate Completion Gap Audit v498

Generated: {status["generated_at_utc"]}

## Result

v498 audits the gap between packaged review inputs and completed review
outcomes. Both executable review gates remain open, all layout and caption
review inputs remain pending, and patching remains blocked.

## Counts

- Execution gate rows: `{status["execution_gate_rows_v498"]}`.
- Execution completion gap rows: `{status["execution_completion_gap_rows_v498"]}`.
- Layout completion gap rows: `{status["layout_completion_gap_rows_v498"]}`.
- Caption completion gap rows: `{status["caption_completion_gap_rows_v498"]}`.
- Completion blocker rows: `{status["completion_blocker_rows_v498"]}`.
- Open completion blocker rows: `{status["open_completion_blocker_rows_v498"]}`.
- Review completed rows: `{status["review_completed_rows_v498"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v498"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v498 is a completion-gap audit only. It does not complete reviews, finalize
captions, obtain approval, edit Quarto, apply a patch, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V498_REVIEW_GATE_COMPLETION_GAP_AUDIT_START -->"
    end = "<!-- V498_REVIEW_GATE_COMPLETION_GAP_AUDIT_END -->"
    block = f"""
{start}

## Wave v498: Review Gate Completion Gap Audit

Generated: {status["generated_at_utc"]}

### Objective

v498 audits whether the v497 review packets have produced completed review
outcomes. They have not; the completion gaps remain open.

### Results

- Execution gate rows:
  `{status["execution_gate_rows_v498"]}`.
- Execution completion gap rows:
  `{status["execution_completion_gap_rows_v498"]}`.
- Layout completion gap rows:
  `{status["layout_completion_gap_rows_v498"]}`.
- Caption completion gap rows:
  `{status["caption_completion_gap_rows_v498"]}`.
- Completion blocker rows:
  `{status["completion_blocker_rows_v498"]}`.
- Open completion blocker rows:
  `{status["open_completion_blocker_rows_v498"]}`.
- Review completed rows:
  `{status["review_completed_rows_v498"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v498"]}`.
- Book sources modified:
  `{status["book_sources_modified_v498"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v498"]}`.

### Interpretation

The paper now has review packets and an explicit completion-gap audit. The next
useful step is to create a capture template for future review outcomes.

### Claim Impact

- Allowed: completion-gap audit, layout/caption gap matrices and open blocker
  summary.
- Still prohibited: completed review/signoff claims, Quarto patch
  readiness/application, Quarto/book-reference mutation, submission readiness,
  Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v498 in the living notebook. v499 should create a review-outcome capture
template without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v497 = _read_status(PRIOR_REVIEW_GATE_PACKET_VERSION)
    if v497["next_artifact_v497"] != "paper4_v498_review_gate_completion_gap_audit.md":
        raise RuntimeError("v498 expects v497 to route to completion gap audit.")

    audit = _review_gate_completion_gap_audit()
    layout_gaps = _layout_completion_gap_matrix()
    caption_gaps = _caption_completion_gap_matrix()
    blockers = _completion_blocker_summary()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v498_review_gate_completion_gap_audit.csv", audit)
    write_csv(TABLE_DIR / "paper4_v498_layout_completion_gap_matrix.csv", layout_gaps)
    write_csv(TABLE_DIR / "paper4_v498_caption_completion_gap_matrix.csv", caption_gaps)
    write_csv(TABLE_DIR / "paper4_v498_completion_blocker_summary.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v498_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v498_claim_matrix_delta.csv", claim_matrix)

    review_completed_rows = int(audit["execution_completed_v498"].astype(bool).sum())
    status = {
        "phase": "v498_review_gate_completion_gap_audit",
        "schema_version": "2026-05-17.498",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_review_gate_packet_version_v498": PRIOR_REVIEW_GATE_PACKET_VERSION,
        "review_gate_completion_gap_audit_created_v498": True,
        "execution_gate_rows_v498": len(audit),
        "execution_completion_gap_rows_v498": int(
            audit["completion_gap_open_v498"].astype(bool).sum()
        ),
        "layout_completion_gap_rows_v498": int(
            layout_gaps["completion_gap_open_v498"].astype(bool).sum()
        ),
        "layout_review_completed_rows_v498": int(
            layout_gaps["review_completed_v498"].astype(bool).sum()
        ),
        "caption_completion_gap_rows_v498": int(
            caption_gaps["completion_gap_open_v498"].astype(bool).sum()
        ),
        "caption_review_completed_rows_v498": int(
            caption_gaps["review_completed_v498"].astype(bool).sum()
        ),
        "completion_blocker_rows_v498": len(blockers),
        "open_completion_blocker_rows_v498": int(blockers["blocker_open_v498"].astype(bool).sum()),
        "review_completed_rows_v498": review_completed_rows,
        "readiness_delta_rows_v498": len(readiness),
        "ready_for_quarto_patch_v498": False,
        "quarto_patch_applied_v498": False,
        "book_sources_modified_v498": False,
        "book_references_modified_v498": False,
        "submission_ready_claim_allowed_v498": False,
        "working_champion_claim_allowed_v498": False,
        "paper1_promotion_allowed_v498": False,
        "paper4_working_champion_changed_v498": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v498": NEXT_ARTIFACT,
        "claim_boundary": (
            "v498 audits review completion gaps only; review completion, captions, "
            "approval, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v498 must not create final Paper 4 promotion.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v498": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
