#!/usr/bin/env python3
"""Build Paper 4 v492 manual layout review packet artifacts."""

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

VERSION = 492
PRIOR_PATCH_PREFLIGHT_VERSION = 491
NEXT_ARTIFACT = "paper4_v493_caption_signoff_gap_packet.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v492_manual_layout_review_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _manual_layout_review_packet() -> pd.DataFrame:
    checklist = pd.read_csv(TABLE_DIR / "paper4_v491_manual_review_surface_checklist.csv")
    focus_by_block = {
        "methods_protocol": "verify methods placement before any patch",
        "results_evidence_cvar": "verify CVaR evidence placement and caveat fit",
        "results_evidence_governance_online": "verify governance/online evidence grouping",
        "discussion_limitations": "verify limitations placement and no overclaiming",
    }
    rows = []
    for _, row in checklist.iterrows():
        rows.append(
            {
                "review_item_id_v492": row["surface_check_id_v491"],
                "target_file_v492": row["target_file_v491"],
                "target_block_v492": row["target_block_v491"],
                "asset_sequence_v492": row["asset_sequence_v491"],
                "layout_item_count_v492": int(row["layout_item_count_v491"]),
                "review_focus_v492": focus_by_block[row["target_block_v491"]],
                "decision_status_v492": "pending_manual_review",
                "manual_review_required_v492": True,
                "patch_allowed_v492": False,
            }
        )
    return pd.DataFrame(rows)


def _asset_surface_review_detail() -> pd.DataFrame:
    layout = pd.read_csv(TABLE_DIR / "paper4_v488_layout_dry_run_packet.csv")
    rows = []
    for _, row in layout.sort_values("layout_order_v488").iterrows():
        rows.append(
            {
                "asset_review_id_v492": f"asset_{row['asset_id_v488']}",
                "asset_id_v492": row["asset_id_v488"],
                "asset_type_v492": row["asset_type_v488"],
                "source_asset_v492": row["source_asset_v488"],
                "target_file_v492": row["target_file_v488"],
                "target_block_v492": row["target_block_v488"],
                "layout_order_v492": int(row["layout_order_v488"]),
                "caption_final_v492": bool(row["caption_final_v488"]),
                "inserted_into_quarto_v492": bool(row["inserted_into_quarto_v488"]),
                "review_required_v492": True,
                "patch_allowed_v492": False,
                "claim_boundary_v492": row["claim_boundary_v488"],
            }
        )
    return pd.DataFrame(rows)


def _manual_review_acceptance_criteria() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "criterion_id_v492": "target_surfaces_identified",
                "criteria_ready_v492": True,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "paper4_v490_manual_review_queue.csv",
            },
            {
                "criterion_id_v492": "assets_mapped_to_surfaces",
                "criteria_ready_v492": True,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "paper4_v492_asset_surface_review_detail.csv",
            },
            {
                "criterion_id_v492": "claim_boundaries_preserved",
                "criteria_ready_v492": True,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "paper4_current_claim_boundaries.csv",
            },
            {
                "criterion_id_v492": "final_promotion_absent",
                "criteria_ready_v492": True,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "paper4_final_promotion_gate_not_created",
            },
            {
                "criterion_id_v492": "final_caption_signoff_present",
                "criteria_ready_v492": False,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "paper4_v488_layout_dry_run_packet.csv",
            },
            {
                "criterion_id_v492": "explicit_patch_approval_present",
                "criteria_ready_v492": False,
                "required_before_patch_v492": True,
                "evidence_artifact_v492": "manual approval missing",
            },
        ]
    )


def _review_decision_options() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_option_id_v492": "keep_manual_review_pending",
                "recommended_v492": True,
                "patch_allowed_v492": False,
                "decision_boundary_v492": "manual review not completed",
            },
            {
                "decision_option_id_v492": "prepare_caption_signoff_gap_packet",
                "recommended_v492": True,
                "patch_allowed_v492": False,
                "decision_boundary_v492": "caption signoff still missing",
            },
            {
                "decision_option_id_v492": "request_explicit_patch_approval",
                "recommended_v492": True,
                "patch_allowed_v492": False,
                "decision_boundary_v492": "approval still missing",
            },
            {
                "decision_option_id_v492": "apply_quarto_patch_now",
                "recommended_v492": False,
                "patch_allowed_v492": False,
                "decision_boundary_v492": "v492 is review packaging only",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v492": "manual_layout_review_packet_created",
                "ready_v492": True,
                "evidence_artifact_v492": "paper4_v492_manual_layout_review_packet.csv",
                "claim_boundary_v492": "review packet only",
            },
            {
                "readiness_gate_v492": "asset_surface_review_detail_created",
                "ready_v492": True,
                "evidence_artifact_v492": "paper4_v492_asset_surface_review_detail.csv",
                "claim_boundary_v492": "asset review detail only",
            },
            {
                "readiness_gate_v492": "acceptance_criteria_created",
                "ready_v492": True,
                "evidence_artifact_v492": "paper4_v492_manual_review_acceptance_criteria.csv",
                "claim_boundary_v492": "acceptance criteria only",
            },
            {
                "readiness_gate_v492": "review_decision_options_created",
                "ready_v492": True,
                "evidence_artifact_v492": "paper4_v492_review_decision_options.csv",
                "claim_boundary_v492": "decision options only",
            },
            {
                "readiness_gate_v492": "ready_for_quarto_patch",
                "ready_v492": False,
                "evidence_artifact_v492": "manual review, approval and final captions missing",
                "claim_boundary_v492": "patch remains blocked",
            },
            {
                "readiness_gate_v492": "book_sources_or_references_modified",
                "ready_v492": False,
                "evidence_artifact_v492": "book sources unchanged",
                "claim_boundary_v492": "no Quarto/book mutation in v492",
            },
            {
                "readiness_gate_v492": "submission_ready",
                "ready_v492": False,
                "evidence_artifact_v492": "future approval, patch, render and venue gates",
                "claim_boundary_v492": "not a submission package",
            },
            {
                "readiness_gate_v492": "paper4_final_promotion_created",
                "ready_v492": False,
                "evidence_artifact_v492": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v492": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v492_manual_layout_review_packet_created",
                "allowed": True,
                "artifact": "paper4_v492_manual_layout_review_packet.csv",
                "boundary": "manual review packet only",
            },
            {
                "claim_id": "v492_asset_surface_review_detail_created",
                "allowed": True,
                "artifact": "paper4_v492_asset_surface_review_detail.csv",
                "boundary": "asset review detail only",
            },
            {
                "claim_id": "v492_acceptance_criteria_created",
                "allowed": True,
                "artifact": "paper4_v492_manual_review_acceptance_criteria.csv",
                "boundary": "criteria only",
            },
            {
                "claim_id": "v492_quarto_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v492_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v492_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v492_manuscript_readiness_delta.csv",
                "boundary": "no submission or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v492 packages manual layout review for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_manual_layout_review_packet.csv"
                ),
                "boundary": "Manual review packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v492 maps assets to manual layout review surfaces.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_asset_surface_review_detail.csv"
                ),
                "boundary": "Asset-surface review mapping only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v492 records manual layout review acceptance criteria.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_manual_review_acceptance_criteria.csv"
                ),
                "boundary": "Criteria register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v492 makes Paper 4 ready for Quarto patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_manuscript_readiness_delta.csv"
                ),
                "boundary": "Manual review, approval and final captions are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v492 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_review_decision_options.csv"
                ),
                "boundary": "Patch is not authorized in v492.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v492 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v492_manuscript_readiness_delta.csv"
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
                "executable_item": "v492 packages manual layout review.",
                "status": "manual_layout_review_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v493 audits caption signoff gaps without mutation",
                "last_wave": "v492",
                "execution_result": "manual_layout_review_packet_created_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v492")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Layout Review Packet v492

Generated: {status["generated_at_utc"]}

## Result

v492 packages the four target surfaces and ten draft assets for manual layout
review. It creates acceptance criteria and decision options, but keeps every
review item pending and does not authorize a Quarto patch.

## Counts

- Review surface rows: `{status["review_surface_rows_v492"]}`.
- Asset review detail rows: `{status["asset_review_detail_rows_v492"]}`.
- Acceptance criteria rows: `{status["acceptance_criteria_rows_v492"]}`.
- Criteria ready rows: `{status["criteria_ready_rows_v492"]}`.
- Review decision option rows: `{status["review_decision_option_rows_v492"]}`.
- Review pending rows: `{status["review_pending_rows_v492"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v492"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v492"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v492 is a manual review packet only. It does not edit Quarto, apply a patch,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V492_MANUAL_LAYOUT_REVIEW_PACKET_START -->"
    end = "<!-- V492_MANUAL_LAYOUT_REVIEW_PACKET_END -->"
    block = f"""
{start}

## Wave v492: Manual Layout Review Packet

Generated: {status["generated_at_utc"]}

### Objective

v492 packages the four target surfaces and ten draft assets for manual layout
review without editing book sources.

### Results

- Review surface rows:
  `{status["review_surface_rows_v492"]}`.
- Asset review detail rows:
  `{status["asset_review_detail_rows_v492"]}`.
- Acceptance criteria rows:
  `{status["acceptance_criteria_rows_v492"]}`.
- Criteria ready rows:
  `{status["criteria_ready_rows_v492"]}`.
- Review decision option rows:
  `{status["review_decision_option_rows_v492"]}`.
- Review pending rows:
  `{status["review_pending_rows_v492"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v492"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v492"]}`.
- Book sources modified:
  `{status["book_sources_modified_v492"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v492"]}`.

### Interpretation

The review packet is now concrete, but patching remains blocked by pending
manual review, missing final caption signoff and missing explicit patch
approval.

### Claim Impact

- Allowed: manual layout review packet, asset-surface mapping and acceptance
  criteria register.
- Still prohibited: Quarto patch readiness/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v492 in the living notebook. v493 should audit caption signoff gaps without
modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v491 = _read_status(PRIOR_PATCH_PREFLIGHT_VERSION)
    if v491["next_artifact_v491"] != "paper4_v492_manual_layout_review_packet.md":
        raise RuntimeError("v492 expects v491 to route to manual layout review packet.")

    review_packet = _manual_layout_review_packet()
    asset_detail = _asset_surface_review_detail()
    criteria = _manual_review_acceptance_criteria()
    decision_options = _review_decision_options()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v492_manual_layout_review_packet.csv", review_packet)
    write_csv(TABLE_DIR / "paper4_v492_asset_surface_review_detail.csv", asset_detail)
    write_csv(TABLE_DIR / "paper4_v492_manual_review_acceptance_criteria.csv", criteria)
    write_csv(TABLE_DIR / "paper4_v492_review_decision_options.csv", decision_options)
    write_csv(TABLE_DIR / "paper4_v492_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v492_claim_matrix_delta.csv", claim_matrix)

    patch_allowed_rows = int(review_packet["patch_allowed_v492"].astype(bool).sum()) + int(
        asset_detail["patch_allowed_v492"].astype(bool).sum()
    )
    status = {
        "phase": "v492_manual_layout_review_packet",
        "schema_version": "2026-05-17.492",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_patch_preflight_version_v492": PRIOR_PATCH_PREFLIGHT_VERSION,
        "manual_layout_review_packet_created_v492": True,
        "review_surface_rows_v492": len(review_packet),
        "asset_review_detail_rows_v492": len(asset_detail),
        "acceptance_criteria_rows_v492": len(criteria),
        "criteria_ready_rows_v492": int(criteria["criteria_ready_v492"].astype(bool).sum()),
        "review_decision_option_rows_v492": len(decision_options),
        "review_pending_rows_v492": int(
            review_packet["manual_review_required_v492"].astype(bool).sum()
        ),
        "patch_allowed_rows_v492": patch_allowed_rows,
        "readiness_delta_rows_v492": len(readiness),
        "ready_for_quarto_patch_v492": False,
        "quarto_patch_applied_v492": False,
        "book_sources_modified_v492": False,
        "book_references_modified_v492": False,
        "submission_ready_claim_allowed_v492": False,
        "working_champion_claim_allowed_v492": False,
        "paper1_promotion_allowed_v492": False,
        "paper4_working_champion_changed_v492": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v492": NEXT_ARTIFACT,
        "claim_boundary": (
            "v492 packages manual layout review only; patching, final captions, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v492 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v492": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
