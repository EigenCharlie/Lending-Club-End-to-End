#!/usr/bin/env python3
"""Build Paper 4 v488 layout dry-run packet artifacts."""

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

VERSION = 488
PRIOR_PAIRING_VERSION = 487
NEXT_ARTIFACT = "paper4_v489_layout_consistency_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v488_layout_dry_run_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _target_for_asset(asset_id: str) -> tuple[str, str]:
    if asset_id in {"T5", "F4"}:
        return (
            "book/chapters/19-paper-mega-extension/19f-sequential-decision-framework.qmd",
            "methods_protocol",
        )
    if asset_id in {"T1", "F1"}:
        return (
            "book/chapters/19-paper-mega-extension/19bv-v33-cvar-certificate.qmd",
            "results_evidence_cvar",
        )
    if asset_id in {"T2", "F2", "T4", "F3"}:
        return (
            "book/chapters/19-paper-mega-extension/19bx-v35-online-macro-validation.qmd",
            "results_evidence_governance_online",
        )
    return (
        "book/chapters/19-paper-mega-extension/19ca-v38-final-synthesis.qmd",
        "discussion_limitations",
    )


def _layout_dry_run_packet() -> pd.DataFrame:
    pairings = pd.read_csv(TABLE_DIR / "paper4_v487_caption_asset_pairing_matrix.csv")
    rows = []
    for _, row in pairings.sort_values("insertion_order_v487").iterrows():
        asset_id = str(row["asset_id_v487"])
        target_file, target_block = _target_for_asset(asset_id)
        rows.append(
            {
                "layout_item_id_v488": f"dryrun_{asset_id}",
                "asset_id_v488": asset_id,
                "asset_type_v488": row["asset_type_v487"],
                "source_asset_v488": row["source_asset_v487"],
                "target_file_v488": target_file,
                "target_block_v488": target_block,
                "layout_order_v488": int(row["insertion_order_v487"]),
                "caption_text_v488": row["draft_caption_v487"],
                "layout_mode_v488": "dry_run_only",
                "ready_for_layout_dry_run_v488": True,
                "ready_for_quarto_patch_v488": False,
                "caption_final_v488": False,
                "inserted_into_quarto_v488": False,
                "claim_boundary_v488": row["claim_boundary_v487"],
            }
        )
    return pd.DataFrame(rows)


def _target_surface_summary(layout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target_file, target_block), group in layout.groupby(
        ["target_file_v488", "target_block_v488"],
        sort=True,
    ):
        rows.append(
            {
                "target_file_v488": target_file,
                "target_block_v488": target_block,
                "layout_item_count_v488": len(group),
                "table_count_v488": int(group["asset_type_v488"].eq("table").sum()),
                "figure_count_v488": int(group["asset_type_v488"].eq("figure").sum()),
                "asset_sequence_v488": ";".join(group["asset_id_v488"]),
                "target_ready_for_patch_v488": False,
                "book_mutation_allowed_v488": False,
            }
        )
    return pd.DataFrame(rows)


def _render_gate_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "render_gate_id_v488": "layout_packet_exists",
                "gate_ready_v488": True,
                "required_before_patch_v488": True,
                "claim_boundary_v488": "dry-run packet exists only",
            },
            {
                "render_gate_id_v488": "target_surfaces_identified",
                "gate_ready_v488": True,
                "required_before_patch_v488": True,
                "claim_boundary_v488": "target identification only",
            },
            {
                "render_gate_id_v488": "manual_patch_approval_present",
                "gate_ready_v488": False,
                "required_before_patch_v488": True,
                "claim_boundary_v488": "approval missing",
            },
            {
                "render_gate_id_v488": "captions_final",
                "gate_ready_v488": False,
                "required_before_patch_v488": True,
                "claim_boundary_v488": "captions non-final",
            },
            {
                "render_gate_id_v488": "quarto_patch_applied",
                "gate_ready_v488": False,
                "required_before_patch_v488": False,
                "claim_boundary_v488": "no patch in v488",
            },
            {
                "render_gate_id_v488": "post_patch_render_passed",
                "gate_ready_v488": False,
                "required_before_patch_v488": False,
                "claim_boundary_v488": "render deferred until patch exists",
            },
        ]
    )


def _no_patch_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "no_patch_item_v488": "book_sources",
                "mutation_allowed_v488": False,
                "mutation_performed_v488": False,
                "blocking_reason_v488": "layout dry-run only",
            },
            {
                "no_patch_item_v488": "book_references",
                "mutation_allowed_v488": False,
                "mutation_performed_v488": False,
                "blocking_reason_v488": "no bibliography update in v488",
            },
            {
                "no_patch_item_v488": "caption_finalization",
                "mutation_allowed_v488": False,
                "mutation_performed_v488": False,
                "blocking_reason_v488": "editorial signoff missing",
            },
            {
                "no_patch_item_v488": "submission_package",
                "mutation_allowed_v488": False,
                "mutation_performed_v488": False,
                "blocking_reason_v488": "venue and render gates missing",
            },
            {
                "no_patch_item_v488": "final_promotion",
                "mutation_allowed_v488": False,
                "mutation_performed_v488": FORBIDDEN_FINAL_PROMOTION.exists(),
                "blocking_reason_v488": "Paper Estrella boundary remains active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v488": "layout_dry_run_packet_created",
                "ready_v488": True,
                "evidence_artifact_v488": "paper4_v488_layout_dry_run_packet.csv",
                "claim_boundary_v488": "layout dry-run only",
            },
            {
                "readiness_gate_v488": "target_surface_summary_created",
                "ready_v488": True,
                "evidence_artifact_v488": "paper4_v488_target_surface_summary.csv",
                "claim_boundary_v488": "target summary only",
            },
            {
                "readiness_gate_v488": "render_gate_plan_created",
                "ready_v488": True,
                "evidence_artifact_v488": "paper4_v488_render_gate_plan.csv",
                "claim_boundary_v488": "render gate plan only",
            },
            {
                "readiness_gate_v488": "no_patch_register_created",
                "ready_v488": True,
                "evidence_artifact_v488": "paper4_v488_no_patch_register.csv",
                "claim_boundary_v488": "no-patch register only",
            },
            {
                "readiness_gate_v488": "ready_for_quarto_patch",
                "ready_v488": False,
                "evidence_artifact_v488": "manual approval and final captions missing",
                "claim_boundary_v488": "patch remains blocked",
            },
            {
                "readiness_gate_v488": "book_sources_or_references_modified",
                "ready_v488": False,
                "evidence_artifact_v488": "book sources unchanged",
                "claim_boundary_v488": "no Quarto/book mutation in v488",
            },
            {
                "readiness_gate_v488": "submission_ready",
                "ready_v488": False,
                "evidence_artifact_v488": "future approval, patch, render and venue gates",
                "claim_boundary_v488": "not a submission package",
            },
            {
                "readiness_gate_v488": "paper4_final_promotion_created",
                "ready_v488": False,
                "evidence_artifact_v488": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v488": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v488_layout_dry_run_packet_created",
                "allowed": True,
                "artifact": "paper4_v488_layout_dry_run_packet.csv",
                "boundary": "layout dry-run only",
            },
            {
                "claim_id": "v488_target_surface_summary_created",
                "allowed": True,
                "artifact": "paper4_v488_target_surface_summary.csv",
                "boundary": "target summary only",
            },
            {
                "claim_id": "v488_render_gate_plan_created",
                "allowed": True,
                "artifact": "paper4_v488_render_gate_plan.csv",
                "boundary": "render gate plan only",
            },
            {
                "claim_id": "v488_quarto_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v488_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v488_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v488_manuscript_readiness_delta.csv",
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
                "claim": "v488 creates a Paper 4 layout dry-run packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v488_layout_dry_run_packet.csv"
                ),
                "boundary": "Layout dry-run only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v488 identifies target surfaces and render gates for future review.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v488_target_surface_summary.csv"
                ),
                "boundary": "Future review plan only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v488 makes Paper 4 ready for Quarto patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v488_manuscript_readiness_delta.csv"
                ),
                "boundary": "Manual approval and final captions are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v488 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v488_no_patch_register.csv"
                ),
                "boundary": "No book mutation in v488.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v488 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v488_manuscript_readiness_delta.csv"
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
                "executable_item": "v488 creates layout dry-run packet.",
                "status": "layout_dry_run_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v489 audits layout consistency",
                "last_wave": "v488",
                "execution_result": "layout_dry_run_created_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v488")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Layout Dry-Run Packet v488

Generated: {status["generated_at_utc"]}

## Result

v488 creates a layout dry-run packet from the v487 caption-asset pairings. It
identifies target surfaces and render gates, but does not edit book sources,
apply a patch, render the book, make Paper 4 submission-ready, or promote Paper
4.

## Counts

- Layout rows: `{status["layout_rows_v488"]}`.
- Target surface rows: `{status["target_surface_rows_v488"]}`.
- Render gate rows: `{status["render_gate_rows_v488"]}`.
- No-patch rows: `{status["no_patch_rows_v488"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v488"]}`.
- Book sources modified: `{status["book_sources_modified_v488"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v488 is a layout dry-run only. Manual approval, final captions, patching, render
validation, submission readiness and final promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V488_LAYOUT_DRY_RUN_PACKET_START -->"
    end = "<!-- V488_LAYOUT_DRY_RUN_PACKET_END -->"
    block = f"""
{start}

## Wave v488: Layout Dry-Run Packet

Generated: {status["generated_at_utc"]}

### Objective

v488 converts the v487 caption-asset pairs into a dry-run layout packet without
editing Quarto sources.

### Results

- Layout rows:
  `{status["layout_rows_v488"]}`.
- Target surface rows:
  `{status["target_surface_rows_v488"]}`.
- Render gate rows:
  `{status["render_gate_rows_v488"]}`.
- No-patch rows:
  `{status["no_patch_rows_v488"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v488"]}`.
- Book sources modified:
  `{status["book_sources_modified_v488"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v488"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v488"]}`.

### Interpretation

The paper now has a concrete layout dry-run surface, but the patch is still
blocked by manual approval and final-caption gates.

### Claim Impact

- Allowed: layout dry-run packet, target-surface summary, render gate plan and
  no-patch register.
- Still prohibited: patch readiness/application, Quarto/book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v488 in the living notebook. v489 should audit layout consistency without
modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v487 = _read_status(PRIOR_PAIRING_VERSION)
    if v487["next_artifact_v487"] != "paper4_v488_layout_dry_run_packet.md":
        raise RuntimeError("v488 expects v487 to route to layout dry-run packet.")

    layout = _layout_dry_run_packet()
    targets = _target_surface_summary(layout)
    render_gates = _render_gate_plan()
    no_patch = _no_patch_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v488_layout_dry_run_packet.csv", layout)
    write_csv(TABLE_DIR / "paper4_v488_target_surface_summary.csv", targets)
    write_csv(TABLE_DIR / "paper4_v488_render_gate_plan.csv", render_gates)
    write_csv(TABLE_DIR / "paper4_v488_no_patch_register.csv", no_patch)
    write_csv(TABLE_DIR / "paper4_v488_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v488_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v488_layout_dry_run_packet",
        "schema_version": "2026-05-17.488",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pairing_version_v488": PRIOR_PAIRING_VERSION,
        "layout_dry_run_packet_created_v488": True,
        "layout_rows_v488": len(layout),
        "target_surface_rows_v488": len(targets),
        "render_gate_rows_v488": len(render_gates),
        "no_patch_rows_v488": len(no_patch),
        "readiness_delta_rows_v488": len(readiness),
        "layout_rows_ready_v488": int(layout["ready_for_layout_dry_run_v488"].sum()),
        "ready_for_quarto_patch_v488": False,
        "quarto_patch_applied_v488": False,
        "book_sources_modified_v488": False,
        "book_references_modified_v488": False,
        "submission_ready_claim_allowed_v488": False,
        "working_champion_claim_allowed_v488": False,
        "paper1_promotion_allowed_v488": False,
        "paper4_working_champion_changed_v488": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v488": NEXT_ARTIFACT,
        "claim_boundary": (
            "v488 creates a layout dry-run only; patching, final captions, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v488 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v488": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
