#!/usr/bin/env python3
"""Build Paper 4 v489 layout consistency audit artifacts."""

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

VERSION = 489
PRIOR_LAYOUT_DRY_RUN_VERSION = 488
NEXT_ARTIFACT = "paper4_v490_layout_review_decision.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v489_layout_consistency_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _layout_consistency_checks() -> pd.DataFrame:
    layout = pd.read_csv(TABLE_DIR / "paper4_v488_layout_dry_run_packet.csv")
    targets = pd.read_csv(TABLE_DIR / "paper4_v488_target_surface_summary.csv")
    gates = pd.read_csv(TABLE_DIR / "paper4_v488_render_gate_plan.csv")
    no_patch = pd.read_csv(TABLE_DIR / "paper4_v488_no_patch_register.csv")
    checks = [
        ("layout_row_count", len(layout) == 10, f"{len(layout)} layout rows"),
        (
            "layout_order_contiguous",
            list(layout["layout_order_v488"]) == list(range(1, 11)),
            "layout order 1-10",
        ),
        (
            "all_items_dry_run_ready",
            layout["ready_for_layout_dry_run_v488"].astype(bool).all(),
            "all layout items ready for dry-run",
        ),
        (
            "no_items_ready_for_patch",
            not layout["ready_for_quarto_patch_v488"].astype(bool).any(),
            "no layout item ready for patch",
        ),
        (
            "target_surface_count",
            len(targets) == 4,
            f"{len(targets)} target surfaces",
        ),
        (
            "render_gates_preserve_blockers",
            not gates.loc[gates["render_gate_id_v488"].str.contains("approval|final|patch|render")]
            ["gate_ready_v488"]
            .astype(bool)
            .any(),
            "approval/final/patch/render gates remain blocked",
        ),
        (
            "no_patch_register_clean",
            not no_patch["mutation_performed_v488"].astype(bool).any(),
            "no mutations performed",
        ),
        (
            "final_promotion_absent",
            not FORBIDDEN_FINAL_PROMOTION.exists(),
            "paper4_final_promotion.json absent",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check_id_v489": check_id,
                "passed_v489": passed,
                "evidence_v489": evidence,
                "claim_boundary_v489": "layout consistency audit only",
            }
            for check_id, passed, evidence in checks
        ]
    )


def _target_consistency_matrix() -> pd.DataFrame:
    targets = pd.read_csv(TABLE_DIR / "paper4_v488_target_surface_summary.csv")
    rows = []
    for _, row in targets.iterrows():
        item_count = int(row["layout_item_count_v488"])
        table_count = int(row["table_count_v488"])
        figure_count = int(row["figure_count_v488"])
        rows.append(
            {
                "target_file_v489": row["target_file_v488"],
                "target_block_v489": row["target_block_v488"],
                "layout_item_count_v489": item_count,
                "asset_sequence_v489": row["asset_sequence_v488"],
                "has_asset_sequence_v489": item_count > 0,
                "table_or_figure_present_v489": (table_count + figure_count) == item_count,
                "target_ready_for_patch_v489": False,
                "book_mutation_allowed_v489": False,
                "target_consistent_v489": (
                    item_count > 0 and (table_count + figure_count) == item_count
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_blocker_matrix() -> pd.DataFrame:
    gates = pd.read_csv(TABLE_DIR / "paper4_v488_render_gate_plan.csv")
    blocked = gates.loc[~gates["gate_ready_v488"].astype(bool)].copy()
    return pd.DataFrame(
        [
            {
                "render_blocker_id_v489": row["render_gate_id_v488"],
                "required_before_patch_v489": bool(row["required_before_patch_v488"]),
                "blocker_open_v489": True,
                "resolved_by_v489": False,
                "claim_boundary_v489": row["claim_boundary_v488"],
            }
            for _, row in blocked.iterrows()
        ]
    )


def _patch_safety_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v489": "keep_layout_as_dry_run",
                "recommended_v489": True,
                "patch_allowed_v489": False,
                "decision_boundary_v489": "layout consistency passed but patch gates open",
            },
            {
                "decision_id_v489": "require_manual_approval_before_patch",
                "recommended_v489": True,
                "patch_allowed_v489": False,
                "decision_boundary_v489": "manual approval missing",
            },
            {
                "decision_id_v489": "require_final_caption_signoff_before_patch",
                "recommended_v489": True,
                "patch_allowed_v489": False,
                "decision_boundary_v489": "captions remain non-final",
            },
            {
                "decision_id_v489": "apply_quarto_patch_now",
                "recommended_v489": False,
                "patch_allowed_v489": False,
                "decision_boundary_v489": "not authorized in v489",
            },
            {
                "decision_id_v489": "declare_submission_ready",
                "recommended_v489": False,
                "patch_allowed_v489": False,
                "decision_boundary_v489": "venue/render/external gates missing",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v489": "layout_consistency_audit_created",
                "ready_v489": True,
                "evidence_artifact_v489": "paper4_v489_layout_consistency_checks.csv",
                "claim_boundary_v489": "audit only",
            },
            {
                "readiness_gate_v489": "target_consistency_matrix_created",
                "ready_v489": True,
                "evidence_artifact_v489": "paper4_v489_target_consistency_matrix.csv",
                "claim_boundary_v489": "target consistency only",
            },
            {
                "readiness_gate_v489": "render_blocker_matrix_created",
                "ready_v489": True,
                "evidence_artifact_v489": "paper4_v489_render_blocker_matrix.csv",
                "claim_boundary_v489": "blocker matrix only",
            },
            {
                "readiness_gate_v489": "patch_safety_decision_created",
                "ready_v489": True,
                "evidence_artifact_v489": "paper4_v489_patch_safety_decision.csv",
                "claim_boundary_v489": "decision matrix only",
            },
            {
                "readiness_gate_v489": "ready_for_quarto_patch",
                "ready_v489": False,
                "evidence_artifact_v489": "manual approval and final captions missing",
                "claim_boundary_v489": "patch remains blocked",
            },
            {
                "readiness_gate_v489": "book_sources_or_references_modified",
                "ready_v489": False,
                "evidence_artifact_v489": "book sources unchanged",
                "claim_boundary_v489": "no Quarto/book mutation in v489",
            },
            {
                "readiness_gate_v489": "submission_ready",
                "ready_v489": False,
                "evidence_artifact_v489": "future approval, patch, render and venue gates",
                "claim_boundary_v489": "not a submission package",
            },
            {
                "readiness_gate_v489": "paper4_final_promotion_created",
                "ready_v489": False,
                "evidence_artifact_v489": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v489": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v489_layout_consistency_audit_created",
                "allowed": True,
                "artifact": "paper4_v489_layout_consistency_checks.csv",
                "boundary": "layout audit only",
            },
            {
                "claim_id": "v489_target_and_render_blockers_mapped",
                "allowed": True,
                "artifact": "paper4_v489_render_blocker_matrix.csv",
                "boundary": "blocker mapping only",
            },
            {
                "claim_id": "v489_patch_safety_decision_created",
                "allowed": True,
                "artifact": "paper4_v489_patch_safety_decision.csv",
                "boundary": "safety decision only",
            },
            {
                "claim_id": "v489_quarto_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v489_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v489_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v489_manuscript_readiness_delta.csv",
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
                "claim": "v489 audits Paper 4 layout dry-run consistency.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_layout_consistency_checks.csv"
                ),
                "boundary": "Layout audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v489 preserves render blockers before any Quarto patch.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_render_blocker_matrix.csv"
                ),
                "boundary": "Render blocker preservation only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v489 records a patch safety decision that keeps patching blocked.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_patch_safety_decision.csv"
                ),
                "boundary": "Patch safety decision only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v489 makes Paper 4 ready for Quarto patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_manuscript_readiness_delta.csv"
                ),
                "boundary": "Manual approval and final captions are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v489 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_patch_safety_decision.csv"
                ),
                "boundary": "Patch is not authorized in v489.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v489 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v489_manuscript_readiness_delta.csv"
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
    legacy_claims = {
        "v489 maps render blockers after the layout dry-run.",
    }
    replaced_claims = set(additions["claim"]) | legacy_claims
    out = current.loc[~current["claim"].isin(replaced_claims)].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Manuscript",
                "executable_item": "v489 audits layout dry-run consistency.",
                "status": "layout_consistency_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v490 records layout review decision",
                "last_wave": "v489",
                "execution_result": "layout_consistency_audit_passed_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v489")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Layout Consistency Audit v489

Generated: {status["generated_at_utc"]}

## Result

v489 audits the v488 layout dry-run for row coverage, target consistency, render
blockers and patch safety. The layout is internally consistent, but patching,
final captions, render validation, submission readiness and final promotion
remain blocked.

## Counts

- Consistency check rows: `{status["consistency_check_rows_v489"]}`.
- Passed consistency checks: `{status["passed_consistency_checks_v489"]}`.
- Target consistency rows: `{status["target_consistency_rows_v489"]}`.
- Render blocker rows: `{status["render_blocker_rows_v489"]}`.
- Patch safety rows: `{status["patch_safety_rows_v489"]}`.
- Layout audit passed: `{status["layout_consistency_audit_passed_v489"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v489"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v489 is an audit only. It does not edit Quarto, apply a patch, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V489_LAYOUT_CONSISTENCY_AUDIT_START -->"
    end = "<!-- V489_LAYOUT_CONSISTENCY_AUDIT_END -->"
    block = f"""
{start}

## Wave v489: Layout Consistency Audit

Generated: {status["generated_at_utc"]}

### Objective

v489 audits the v488 layout dry-run for coverage, target consistency, render
blockers and patch safety without editing book sources.

### Results

- Consistency check rows:
  `{status["consistency_check_rows_v489"]}`.
- Passed consistency checks:
  `{status["passed_consistency_checks_v489"]}`.
- Target consistency rows:
  `{status["target_consistency_rows_v489"]}`.
- Render blocker rows:
  `{status["render_blocker_rows_v489"]}`.
- Patch safety rows:
  `{status["patch_safety_rows_v489"]}`.
- Layout audit passed:
  `{status["layout_consistency_audit_passed_v489"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v489"]}`.
- Book sources modified:
  `{status["book_sources_modified_v489"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v489"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v489"]}`.

### Interpretation

The layout dry-run is internally consistent, but the safest next move is a
layout review decision rather than any book-source mutation.

### Claim Impact

- Allowed: layout consistency audit, target consistency matrix, render blocker
  matrix and patch safety decision.
- Still prohibited: Quarto patch readiness/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v489 in the living notebook. v490 should record a layout review decision
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v488 = _read_status(PRIOR_LAYOUT_DRY_RUN_VERSION)
    if v488["next_artifact_v488"] != "paper4_v489_layout_consistency_audit.md":
        raise RuntimeError("v489 expects v488 to route to layout consistency audit.")

    checks = _layout_consistency_checks()
    targets = _target_consistency_matrix()
    blockers = _render_blocker_matrix()
    safety = _patch_safety_decision()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v489_layout_consistency_checks.csv", checks)
    write_csv(TABLE_DIR / "paper4_v489_target_consistency_matrix.csv", targets)
    write_csv(TABLE_DIR / "paper4_v489_render_blocker_matrix.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v489_patch_safety_decision.csv", safety)
    write_csv(TABLE_DIR / "paper4_v489_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v489_claim_matrix_delta.csv", claim_matrix)

    passed_checks = int(checks["passed_v489"].astype(bool).sum())
    target_passes = int(targets["target_consistent_v489"].astype(bool).sum())
    audit_passed = passed_checks == len(checks) and target_passes == len(targets)
    status = {
        "phase": "v489_layout_consistency_audit",
        "schema_version": "2026-05-17.489",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_layout_dry_run_version_v489": PRIOR_LAYOUT_DRY_RUN_VERSION,
        "layout_consistency_audit_created_v489": True,
        "layout_consistency_audit_passed_v489": audit_passed,
        "consistency_check_rows_v489": len(checks),
        "passed_consistency_checks_v489": passed_checks,
        "target_consistency_rows_v489": len(targets),
        "target_consistency_pass_rows_v489": target_passes,
        "target_surface_consistent_rows_v489": target_passes,
        "render_blocker_rows_v489": len(blockers),
        "render_blockers_preserved_rows_v489": int(
            blockers["blocker_open_v489"].astype(bool).sum()
        ),
        "patch_safety_rows_v489": len(safety),
        "safe_to_patch_rows_v489": int(safety["patch_allowed_v489"].astype(bool).sum()),
        "readiness_delta_rows_v489": len(readiness),
        "ready_for_quarto_patch_v489": False,
        "quarto_patch_applied_v489": False,
        "book_sources_modified_v489": False,
        "book_references_modified_v489": False,
        "submission_ready_claim_allowed_v489": False,
        "working_champion_claim_allowed_v489": False,
        "paper1_promotion_allowed_v489": False,
        "paper4_working_champion_changed_v489": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v489": NEXT_ARTIFACT,
        "claim_boundary": (
            "v489 audits layout consistency only; patching, final captions, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v489 must not create final Paper 4 promotion.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v489": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
