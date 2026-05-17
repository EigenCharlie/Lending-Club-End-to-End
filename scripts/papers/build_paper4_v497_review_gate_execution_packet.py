#!/usr/bin/env python3
"""Build Paper 4 v497 review gate execution packet artifacts."""

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

VERSION = 497
PRIOR_REVIEW_GATE_VERSION = 496
NEXT_ARTIFACT = "paper4_v498_review_gate_completion_gap_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v497_review_gate_execution_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _review_gate_execution_packet() -> pd.DataFrame:
    queue = pd.read_csv(TABLE_DIR / "paper4_v496_execution_priority_queue.csv")
    ready = queue.loc[queue["execution_ready_v496"].astype(bool)].copy()
    item_count_by_gate = {
        "manual_layout_surface_review": 4,
        "caption_claim_safety_review": 10,
    }
    source_by_gate = {
        "manual_layout_surface_review": "paper4_v492_manual_layout_review_packet.csv",
        "caption_claim_safety_review": "paper4_v493_caption_claim_safety_matrix.csv",
    }
    rows = []
    for _, row in ready.sort_values("priority_v496").iterrows():
        gate_id = row["review_gate_id_v496"]
        rows.append(
            {
                "execution_gate_id_v497": gate_id,
                "priority_v497": int(row["priority_v496"]),
                "source_artifact_v497": source_by_gate[gate_id],
                "review_item_count_v497": item_count_by_gate[gate_id],
                "execution_status_v497": "packet_ready_review_pending",
                "execution_started_v497": False,
                "execution_completed_v497": False,
                "patch_allowed_v497": False,
            }
        )
    return pd.DataFrame(rows)


def _layout_surface_review_inputs() -> pd.DataFrame:
    surfaces = pd.read_csv(TABLE_DIR / "paper4_v492_manual_layout_review_packet.csv")
    rows = []
    for _, row in surfaces.iterrows():
        rows.append(
            {
                "layout_review_input_id_v497": row["review_item_id_v492"],
                "target_file_v497": row["target_file_v492"],
                "target_block_v497": row["target_block_v492"],
                "asset_sequence_v497": row["asset_sequence_v492"],
                "layout_item_count_v497": int(row["layout_item_count_v492"]),
                "review_focus_v497": row["review_focus_v492"],
                "review_status_v497": "pending_review",
                "accepted_for_patch_v497": False,
                "patch_allowed_v497": False,
            }
        )
    return pd.DataFrame(rows)


def _caption_claim_safety_review_inputs() -> pd.DataFrame:
    captions = pd.read_csv(TABLE_DIR / "paper4_v493_caption_claim_safety_matrix.csv")
    rows = []
    for idx, row in captions.iterrows():
        rows.append(
            {
                "caption_review_input_id_v497": f"caption_review_{idx + 1:02d}",
                "asset_id_v497": row["asset_id_v493"],
                "target_block_v497": row["target_block_v493"],
                "draft_caption_exists_v497": bool(row["draft_caption_exists_v493"]),
                "caption_final_v497": bool(row["caption_final_v493"]),
                "claim_boundary_v497": row["claim_boundary_v493"],
                "overclaim_review_required_v497": bool(row["overclaim_review_required_v493"]),
                "review_status_v497": "pending_review",
                "accepted_for_final_caption_v497": False,
                "patch_allowed_v497": False,
            }
        )
    return pd.DataFrame(rows)


def _execution_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "control_id_v497": "no_book_source_mutation",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "review packet only",
            },
            {
                "control_id_v497": "no_quarto_patch",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "patch remains blocked",
            },
            {
                "control_id_v497": "no_caption_finalization",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "caption finalization deferred",
            },
            {
                "control_id_v497": "no_patch_approval_claim",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "approval remains missing",
            },
            {
                "control_id_v497": "no_render_or_submission_claim",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "render/submission gates remain open",
            },
            {
                "control_id_v497": "no_final_promotion",
                "control_active_v497": True,
                "mutation_allowed_v497": False,
                "control_boundary_v497": "Paper Estrella boundary remains active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v497": "review_gate_execution_packet_created",
                "ready_v497": True,
                "evidence_artifact_v497": "paper4_v497_review_gate_execution_packet.csv",
                "claim_boundary_v497": "execution packet only",
            },
            {
                "readiness_gate_v497": "layout_surface_review_inputs_created",
                "ready_v497": True,
                "evidence_artifact_v497": "paper4_v497_layout_surface_review_inputs.csv",
                "claim_boundary_v497": "layout review inputs only",
            },
            {
                "readiness_gate_v497": "caption_claim_safety_review_inputs_created",
                "ready_v497": True,
                "evidence_artifact_v497": "paper4_v497_caption_claim_safety_review_inputs.csv",
                "claim_boundary_v497": "caption safety inputs only",
            },
            {
                "readiness_gate_v497": "execution_control_register_created",
                "ready_v497": True,
                "evidence_artifact_v497": "paper4_v497_execution_control_register.csv",
                "claim_boundary_v497": "control register only",
            },
            {
                "readiness_gate_v497": "ready_for_quarto_patch",
                "ready_v497": False,
                "evidence_artifact_v497": "reviews remain pending",
                "claim_boundary_v497": "patch remains blocked",
            },
            {
                "readiness_gate_v497": "book_sources_or_references_modified",
                "ready_v497": False,
                "evidence_artifact_v497": "book sources unchanged",
                "claim_boundary_v497": "no Quarto/book mutation in v497",
            },
            {
                "readiness_gate_v497": "submission_ready",
                "ready_v497": False,
                "evidence_artifact_v497": "future approval, patch, render and venue gates",
                "claim_boundary_v497": "not a submission package",
            },
            {
                "readiness_gate_v497": "paper4_final_promotion_created",
                "ready_v497": False,
                "evidence_artifact_v497": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v497": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v497_review_gate_execution_packet_created",
                "allowed": True,
                "artifact": "paper4_v497_review_gate_execution_packet.csv",
                "boundary": "execution packet only",
            },
            {
                "claim_id": "v497_review_inputs_created",
                "allowed": True,
                "artifact": "paper4_v497_layout_and_caption_review_inputs",
                "boundary": "review inputs only",
            },
            {
                "claim_id": "v497_execution_controls_preserved",
                "allowed": True,
                "artifact": "paper4_v497_execution_control_register.csv",
                "boundary": "controls preserved only",
            },
            {
                "claim_id": "v497_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v497_manuscript_readiness_delta.csv",
                "boundary": "reviews remain pending",
            },
            {
                "claim_id": "v497_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v497_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v497_final_promotion",
                "allowed": False,
                "artifact": "paper4_v497_manuscript_readiness_delta.csv",
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
                "claim": "v497 packages executable Paper 4 review gates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_review_gate_execution_packet.csv"
                ),
                "boundary": "Review-gate execution packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v497 creates layout and caption review input packets.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_layout_surface_review_inputs.csv"
                ),
                "boundary": "Review inputs only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v497 preserves no-mutation execution controls.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_execution_control_register.csv"
                ),
                "boundary": "Execution control register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v497 completes manual reviews or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_manuscript_readiness_delta.csv"
                ),
                "boundary": "Reviews remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v497 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v497 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v497_manuscript_readiness_delta.csv"
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
                "executable_item": "v497 packages executable review gates.",
                "status": "review_gate_execution_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v498 audits review gate completion gaps",
                "last_wave": "v497",
                "execution_result": "review_gate_execution_packet_created_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v497")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Gate Execution Packet v497

Generated: {status["generated_at_utc"]}

## Result

v497 packages the two immediately executable review gates from v496: manual
layout surface review and caption claim-safety review. It creates review inputs
and execution controls, but does not start or complete review, finalize
captions, approve patching, mutate Quarto, or promote Paper 4.

## Counts

- Execution gate rows: `{status["execution_gate_rows_v497"]}`.
- Layout surface input rows: `{status["layout_surface_input_rows_v497"]}`.
- Caption claim-safety input rows: `{status["caption_claim_safety_input_rows_v497"]}`.
- Execution control rows: `{status["execution_control_rows_v497"]}`.
- Active control rows: `{status["active_control_rows_v497"]}`.
- Execution completed rows: `{status["execution_completed_rows_v497"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v497"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v497"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v497 is an execution packet only. It does not complete reviews, finalize
captions, obtain approval, edit Quarto, apply a patch, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V497_REVIEW_GATE_EXECUTION_PACKET_START -->"
    end = "<!-- V497_REVIEW_GATE_EXECUTION_PACKET_END -->"
    block = f"""
{start}

## Wave v497: Review Gate Execution Packet

Generated: {status["generated_at_utc"]}

### Objective

v497 packages the two immediately executable review gates from v496 without
starting review completion, finalizing captions, approving patching, or editing
book sources.

### Results

- Execution gate rows:
  `{status["execution_gate_rows_v497"]}`.
- Layout surface input rows:
  `{status["layout_surface_input_rows_v497"]}`.
- Caption claim-safety input rows:
  `{status["caption_claim_safety_input_rows_v497"]}`.
- Execution control rows:
  `{status["execution_control_rows_v497"]}`.
- Active control rows:
  `{status["active_control_rows_v497"]}`.
- Execution completed rows:
  `{status["execution_completed_rows_v497"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v497"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v497"]}`.
- Book sources modified:
  `{status["book_sources_modified_v497"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v497"]}`.

### Interpretation

The packet is now ready for review execution, but no review outcome has been
recorded. The next useful step is a completion-gap audit.

### Claim Impact

- Allowed: executable review-gate packet, review input packets and no-mutation
  execution controls.
- Still prohibited: completed review/signoff claims, Quarto patch
  readiness/application, Quarto/book-reference mutation, submission readiness,
  Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v497 in the living notebook. v498 should audit review gate completion gaps
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v496 = _read_status(PRIOR_REVIEW_GATE_VERSION)
    if v496["next_artifact_v496"] != "paper4_v497_review_gate_execution_packet.md":
        raise RuntimeError("v497 expects v496 to route to review gate execution packet.")

    execution_packet = _review_gate_execution_packet()
    layout_inputs = _layout_surface_review_inputs()
    caption_inputs = _caption_claim_safety_review_inputs()
    controls = _execution_control_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v497_review_gate_execution_packet.csv", execution_packet)
    write_csv(TABLE_DIR / "paper4_v497_layout_surface_review_inputs.csv", layout_inputs)
    write_csv(TABLE_DIR / "paper4_v497_caption_claim_safety_review_inputs.csv", caption_inputs)
    write_csv(TABLE_DIR / "paper4_v497_execution_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v497_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v497_claim_matrix_delta.csv", claim_matrix)

    patch_allowed_rows = (
        int(execution_packet["patch_allowed_v497"].astype(bool).sum())
        + int(layout_inputs["patch_allowed_v497"].astype(bool).sum())
        + int(caption_inputs["patch_allowed_v497"].astype(bool).sum())
    )
    status = {
        "phase": "v497_review_gate_execution_packet",
        "schema_version": "2026-05-17.497",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_review_gate_version_v497": PRIOR_REVIEW_GATE_VERSION,
        "review_gate_execution_packet_created_v497": True,
        "execution_gate_rows_v497": len(execution_packet),
        "layout_surface_input_rows_v497": len(layout_inputs),
        "caption_claim_safety_input_rows_v497": len(caption_inputs),
        "execution_control_rows_v497": len(controls),
        "active_control_rows_v497": int(controls["control_active_v497"].astype(bool).sum()),
        "execution_started_rows_v497": int(
            execution_packet["execution_started_v497"].astype(bool).sum()
        ),
        "execution_completed_rows_v497": int(
            execution_packet["execution_completed_v497"].astype(bool).sum()
        ),
        "layout_inputs_accepted_rows_v497": int(
            layout_inputs["accepted_for_patch_v497"].astype(bool).sum()
        ),
        "caption_inputs_accepted_rows_v497": int(
            caption_inputs["accepted_for_final_caption_v497"].astype(bool).sum()
        ),
        "patch_allowed_rows_v497": patch_allowed_rows,
        "readiness_delta_rows_v497": len(readiness),
        "ready_for_quarto_patch_v497": False,
        "quarto_patch_applied_v497": False,
        "book_sources_modified_v497": False,
        "book_references_modified_v497": False,
        "submission_ready_claim_allowed_v497": False,
        "working_champion_claim_allowed_v497": False,
        "paper1_promotion_allowed_v497": False,
        "paper4_working_champion_changed_v497": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v497": NEXT_ARTIFACT,
        "claim_boundary": (
            "v497 packages executable review gates only; review completion, captions, "
            "approval, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v497 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v497": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
