#!/usr/bin/env python3
"""Build Paper 4 v491 patch readiness preflight artifacts."""

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

VERSION = 491
PRIOR_LAYOUT_REVIEW_VERSION = 490
NEXT_ARTIFACT = "paper4_v492_manual_layout_review_packet.md"
PREFLIGHT_MD = NOTEBOOK.parent / "paper4_v491_patch_readiness_preflight.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _preflight_gap_matrix() -> pd.DataFrame:
    seed = pd.read_csv(TABLE_DIR / "paper4_v490_patch_readiness_preflight_seed.csv")
    rows = []
    for _, row in seed.iterrows():
        passed = bool(row["ready_for_v491_v490"])
        rows.append(
            {
                "preflight_item_v491": row["preflight_item_v490"],
                "source_status_v491": row["seed_status_v490"],
                "preflight_pass_v491": passed,
                "blocks_patch_v491": not passed,
                "required_next_action_v491": row["required_next_action_v490"],
            }
        )
    return pd.DataFrame(rows)


def _manual_review_surface_checklist() -> pd.DataFrame:
    queue = pd.read_csv(TABLE_DIR / "paper4_v490_manual_review_queue.csv")
    rows = []
    for _, row in queue.iterrows():
        rows.append(
            {
                "surface_check_id_v491": row["review_item_id_v490"],
                "target_file_v491": row["target_file_v490"],
                "target_block_v491": row["target_block_v490"],
                "asset_sequence_v491": row["asset_sequence_v490"],
                "layout_item_count_v491": int(row["layout_item_count_v490"]),
                "manual_review_pending_v491": row["review_status_v490"]
                == "pending_manual_review",
                "blocks_patch_v491": True,
                "review_required_before_patch_v491": True,
            }
        )
    return pd.DataFrame(rows)


def _blocker_resolution_plan() -> pd.DataFrame:
    gates = pd.read_csv(TABLE_DIR / "paper4_v490_patch_gate_register.csv")
    open_gates = gates.loc[gates["blocks_patch_v490"].astype(bool)].copy()
    action_by_gate = {
        "manual_patch_approval_present": "capture explicit manual patch approval",
        "captions_final": "complete final caption signoff",
        "quarto_patch_applied": "apply a future controlled patch only after approvals",
        "post_patch_render_passed": "run post-patch render after a patch exists",
    }
    return pd.DataFrame(
        [
            {
                "blocker_id_v491": row["patch_gate_id_v490"],
                "required_before_patch_v491": bool(row["required_before_patch_v490"]),
                "resolution_action_v491": action_by_gate[row["patch_gate_id_v490"]],
                "resolution_owner_v491": "manual_editorial_review",
                "resolved_v491": False,
                "blocks_patch_v491": True,
                "claim_boundary_v491": row["claim_boundary_v490"],
            }
            for _, row in open_gates.iterrows()
        ]
    )


def _readiness_scorecard(gaps: pd.DataFrame, checklist: pd.DataFrame) -> pd.DataFrame:
    gap_map = dict(zip(gaps["preflight_item_v491"], gaps["preflight_pass_v491"], strict=False))
    return pd.DataFrame(
        [
            {
                "scorecard_gate_v491": "layout_audit_passed",
                "pass_v491": bool(gap_map["layout_audit_status"]),
                "evidence_artifact_v491": "paper4_v489_layout_consistency_checks.csv",
                "claim_boundary_v491": "layout audit evidence only",
            },
            {
                "scorecard_gate_v491": "manual_review_queue_exists",
                "pass_v491": bool(gap_map["manual_review_queue"]),
                "evidence_artifact_v491": "paper4_v490_manual_review_queue.csv",
                "claim_boundary_v491": "queue exists but review remains pending",
            },
            {
                "scorecard_gate_v491": "manual_review_completed",
                "pass_v491": not checklist["manual_review_pending_v491"].astype(bool).any(),
                "evidence_artifact_v491": "paper4_v491_manual_review_surface_checklist.csv",
                "claim_boundary_v491": "manual review still pending",
            },
            {
                "scorecard_gate_v491": "manual_patch_approval_present",
                "pass_v491": bool(gap_map["manual_patch_approval"]),
                "evidence_artifact_v491": "paper4_v491_preflight_gap_matrix.csv",
                "claim_boundary_v491": "approval missing",
            },
            {
                "scorecard_gate_v491": "final_caption_signoff_present",
                "pass_v491": bool(gap_map["final_caption_signoff"]),
                "evidence_artifact_v491": "paper4_v491_preflight_gap_matrix.csv",
                "claim_boundary_v491": "caption signoff missing",
            },
            {
                "scorecard_gate_v491": "post_patch_render_gate_passed",
                "pass_v491": bool(gap_map["post_patch_render_gate"]),
                "evidence_artifact_v491": "paper4_v491_preflight_gap_matrix.csv",
                "claim_boundary_v491": "render deferred until patch exists",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v491": "patch_readiness_preflight_created",
                "ready_v491": True,
                "evidence_artifact_v491": "paper4_v491_preflight_gap_matrix.csv",
                "claim_boundary_v491": "preflight only",
            },
            {
                "readiness_gate_v491": "manual_review_surface_checklist_created",
                "ready_v491": True,
                "evidence_artifact_v491": "paper4_v491_manual_review_surface_checklist.csv",
                "claim_boundary_v491": "manual review checklist only",
            },
            {
                "readiness_gate_v491": "blocker_resolution_plan_created",
                "ready_v491": True,
                "evidence_artifact_v491": "paper4_v491_blocker_resolution_plan.csv",
                "claim_boundary_v491": "blocker plan only",
            },
            {
                "readiness_gate_v491": "readiness_scorecard_created",
                "ready_v491": True,
                "evidence_artifact_v491": "paper4_v491_readiness_scorecard.csv",
                "claim_boundary_v491": "scorecard only",
            },
            {
                "readiness_gate_v491": "ready_for_quarto_patch",
                "ready_v491": False,
                "evidence_artifact_v491": "open blockers remain",
                "claim_boundary_v491": "patch remains blocked",
            },
            {
                "readiness_gate_v491": "book_sources_or_references_modified",
                "ready_v491": False,
                "evidence_artifact_v491": "book sources unchanged",
                "claim_boundary_v491": "no Quarto/book mutation in v491",
            },
            {
                "readiness_gate_v491": "submission_ready",
                "ready_v491": False,
                "evidence_artifact_v491": "future approval, patch, render and venue gates",
                "claim_boundary_v491": "not a submission package",
            },
            {
                "readiness_gate_v491": "paper4_final_promotion_created",
                "ready_v491": False,
                "evidence_artifact_v491": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v491": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v491_patch_readiness_preflight_created",
                "allowed": True,
                "artifact": "paper4_v491_preflight_gap_matrix.csv",
                "boundary": "preflight only",
            },
            {
                "claim_id": "v491_manual_review_surface_checklist_created",
                "allowed": True,
                "artifact": "paper4_v491_manual_review_surface_checklist.csv",
                "boundary": "checklist only",
            },
            {
                "claim_id": "v491_open_blockers_identified",
                "allowed": True,
                "artifact": "paper4_v491_blocker_resolution_plan.csv",
                "boundary": "blocker identification only",
            },
            {
                "claim_id": "v491_quarto_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v491_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v491_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v491_manuscript_readiness_delta.csv",
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
                "claim": "v491 preflights Paper 4 patch readiness.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_preflight_gap_matrix.csv"
                ),
                "boundary": "Patch-readiness preflight only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v491 identifies manual layout review surfaces.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_manual_review_surface_checklist.csv"
                ),
                "boundary": "Manual review checklist only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v491 records unresolved patch blockers.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_blocker_resolution_plan.csv"
                ),
                "boundary": "Open blocker register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v491 makes Paper 4 ready for Quarto patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_readiness_scorecard.csv"
                ),
                "boundary": "Manual review, approval, captions and render gates are open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v491 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch is not authorized in v491.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v491 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v491_manuscript_readiness_delta.csv"
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
                "executable_item": "v491 preflights patch readiness.",
                "status": "patch_readiness_preflight_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v492 packages manual layout review without mutation",
                "last_wave": "v491",
                "execution_result": "patch_readiness_preflight_failed_with_open_blockers",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v491")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _preflight_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Patch Readiness Preflight v491

Generated: {status["generated_at_utc"]}

## Result

v491 preflights the v490 layout review decision. The preflight preserves the
passed layout audit and review queue as useful inputs, but patch readiness fails
because manual review, patch approval, final caption signoff and post-patch
render gates remain open.

## Counts

- Preflight gap rows: `{status["preflight_gap_rows_v491"]}`.
- Preflight pass rows: `{status["preflight_pass_rows_v491"]}`.
- Manual review surface rows: `{status["manual_review_surface_rows_v491"]}`.
- Unresolved blocker rows: `{status["unresolved_blocker_rows_v491"]}`.
- Scorecard rows: `{status["scorecard_rows_v491"]}`.
- Scorecard pass rows: `{status["scorecard_pass_rows_v491"]}`.
- Patch readiness passed: `{status["patch_readiness_passed_v491"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v491"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v491 is a preflight only. It does not edit Quarto, apply a patch, render the
book, make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4
as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V491_PATCH_READINESS_PREFLIGHT_START -->"
    end = "<!-- V491_PATCH_READINESS_PREFLIGHT_END -->"
    block = f"""
{start}

## Wave v491: Patch Readiness Preflight

Generated: {status["generated_at_utc"]}

### Objective

v491 preflights whether the v490 layout review decision is enough to authorize
any Quarto patch. It is not enough; this wave documents the open blockers.

### Results

- Preflight gap rows:
  `{status["preflight_gap_rows_v491"]}`.
- Preflight pass rows:
  `{status["preflight_pass_rows_v491"]}`.
- Manual review surface rows:
  `{status["manual_review_surface_rows_v491"]}`.
- Manual review pending rows:
  `{status["manual_review_pending_rows_v491"]}`.
- Unresolved blocker rows:
  `{status["unresolved_blocker_rows_v491"]}`.
- Scorecard rows:
  `{status["scorecard_rows_v491"]}`.
- Scorecard pass rows:
  `{status["scorecard_pass_rows_v491"]}`.
- Patch readiness passed:
  `{status["patch_readiness_passed_v491"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v491"]}`.
- Book sources modified:
  `{status["book_sources_modified_v491"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v491"]}`.

### Interpretation

Patch readiness still fails. The next useful executable step is to package the
manual layout review so the four target surfaces can be reviewed without source
mutation.

### Claim Impact

- Allowed: patch-readiness preflight, manual layout review surface checklist and
  unresolved blocker register.
- Still prohibited: Quarto patch readiness/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v491 in the living notebook. v492 should package the manual layout review
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v490 = _read_status(PRIOR_LAYOUT_REVIEW_VERSION)
    if v490["next_artifact_v490"] != "paper4_v491_patch_readiness_preflight.md":
        raise RuntimeError("v491 expects v490 to route to patch readiness preflight.")

    gaps = _preflight_gap_matrix()
    checklist = _manual_review_surface_checklist()
    blockers = _blocker_resolution_plan()
    scorecard = _readiness_scorecard(gaps, checklist)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v491_preflight_gap_matrix.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v491_manual_review_surface_checklist.csv", checklist)
    write_csv(TABLE_DIR / "paper4_v491_blocker_resolution_plan.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v491_readiness_scorecard.csv", scorecard)
    write_csv(TABLE_DIR / "paper4_v491_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v491_claim_matrix_delta.csv", claim_matrix)

    preflight_pass_rows = int(gaps["preflight_pass_v491"].astype(bool).sum())
    scorecard_pass_rows = int(scorecard["pass_v491"].astype(bool).sum())
    patch_readiness_passed = bool(scorecard["pass_v491"].astype(bool).all())
    status = {
        "phase": "v491_patch_readiness_preflight",
        "schema_version": "2026-05-17.491",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_layout_review_version_v491": PRIOR_LAYOUT_REVIEW_VERSION,
        "patch_readiness_preflight_created_v491": True,
        "preflight_gap_rows_v491": len(gaps),
        "preflight_pass_rows_v491": preflight_pass_rows,
        "manual_review_surface_rows_v491": len(checklist),
        "manual_review_pending_rows_v491": int(
            checklist["manual_review_pending_v491"].astype(bool).sum()
        ),
        "blocker_resolution_rows_v491": len(blockers),
        "unresolved_blocker_rows_v491": int(blockers["blocks_patch_v491"].astype(bool).sum()),
        "scorecard_rows_v491": len(scorecard),
        "scorecard_pass_rows_v491": scorecard_pass_rows,
        "patch_readiness_passed_v491": patch_readiness_passed,
        "readiness_delta_rows_v491": len(readiness),
        "ready_for_quarto_patch_v491": False,
        "quarto_patch_applied_v491": False,
        "book_sources_modified_v491": False,
        "book_references_modified_v491": False,
        "submission_ready_claim_allowed_v491": False,
        "working_champion_claim_allowed_v491": False,
        "paper1_promotion_allowed_v491": False,
        "paper4_working_champion_changed_v491": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v491": NEXT_ARTIFACT,
        "claim_boundary": (
            "v491 preflights patch readiness only; review, approval, captions, render, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v491 must not create final Paper 4 promotion.")

    PREFLIGHT_MD.write_text(_preflight_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v491": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
