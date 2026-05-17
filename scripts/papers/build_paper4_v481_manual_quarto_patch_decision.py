#!/usr/bin/env python3
"""Build Paper 4 v481 manual Quarto patch decision artifacts."""

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

VERSION = 481
PRIOR_INSERTION_PLAN_VERSION = 480
NEXT_ARTIFACT = "paper4_v482_post_plan_synthesis_packet.md"
DECISION_MD = NOTEBOOK.parent / "paper4_v481_manual_quarto_patch_decision.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _patch_decision_register() -> pd.DataFrame:
    plan = pd.read_csv(TABLE_DIR / "paper4_v480_quarto_insertion_plan.csv")
    rows = []
    for _, row in plan.iterrows():
        rows.append(
            {
                "stub_id_v481": row["stub_id_v480"],
                "target_file_v481": row["target_file_v480"],
                "patch_allowed_v481": False,
                "patch_applied_v481": False,
                "decision_reason_v481": (
                    "explicit human approval and manual review are required before "
                    "editing book sources"
                ),
                "required_next_review_v481": "manual patch approval",
                "claim_boundary_v481": row["claim_boundary_v480"],
            }
        )
    return pd.DataFrame(rows)


def _manual_review_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id_v481": "explicit_user_patch_approval",
                "satisfied_v481": False,
                "blocks_patch_v481": True,
                "evidence_v481": "no explicit patch approval in v481",
            },
            {
                "requirement_id_v481": "caption_finalization_review",
                "satisfied_v481": False,
                "blocks_patch_v481": True,
                "evidence_v481": "captions remain draft",
            },
            {
                "requirement_id_v481": "asset_path_review",
                "satisfied_v481": False,
                "blocks_patch_v481": True,
                "evidence_v481": "visual assets need manual placement review",
            },
            {
                "requirement_id_v481": "post_patch_quarto_render_plan",
                "satisfied_v481": True,
                "blocks_patch_v481": False,
                "evidence_v481": "v480 gate requires render after future patch",
            },
            {
                "requirement_id_v481": "rollback_plan_available",
                "satisfied_v481": True,
                "blocks_patch_v481": False,
                "evidence_v481": "paper4_v480_rollback_plan.csv",
            },
            {
                "requirement_id_v481": "final_promotion_absent",
                "satisfied_v481": not FORBIDDEN_FINAL_PROMOTION.exists(),
                "blocks_patch_v481": False,
                "evidence_v481": "paper4_final_promotion.json absent",
            },
        ]
    )


def _next_action_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id_v481": "request_explicit_patch_approval",
                "priority_v481": 1,
                "action_ready_v481": True,
                "result_expected_v481": "human decision on book-source edit",
            },
            {
                "action_id_v481": "finalize_caption_language_under_caveats",
                "priority_v481": 2,
                "action_ready_v481": True,
                "result_expected_v481": "caption text approved for insertion",
            },
            {
                "action_id_v481": "prepare_manual_patch_diff",
                "priority_v481": 3,
                "action_ready_v481": False,
                "result_expected_v481": "requires explicit patch approval first",
            },
            {
                "action_id_v481": "run_post_patch_quarto_render",
                "priority_v481": 4,
                "action_ready_v481": False,
                "result_expected_v481": "requires patch to exist",
            },
            {
                "action_id_v481": "rerun_paper4_guardrails_after_patch",
                "priority_v481": 5,
                "action_ready_v481": False,
                "result_expected_v481": "requires patch and render results",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v481": "manual_patch_decision_created",
                "ready_v481": True,
                "evidence_artifact_v481": "paper4_v481_patch_decision_register.csv",
                "claim_boundary_v481": "decision only",
            },
            {
                "readiness_gate_v481": "manual_review_requirements_created",
                "ready_v481": True,
                "evidence_artifact_v481": "paper4_v481_manual_review_requirements.csv",
                "claim_boundary_v481": "requirements only",
            },
            {
                "readiness_gate_v481": "next_action_queue_created",
                "ready_v481": True,
                "evidence_artifact_v481": "paper4_v481_next_action_queue.csv",
                "claim_boundary_v481": "next-action queue only",
            },
            {
                "readiness_gate_v481": "patch_allowed",
                "ready_v481": False,
                "evidence_artifact_v481": "explicit approval missing",
                "claim_boundary_v481": "patch remains blocked",
            },
            {
                "readiness_gate_v481": "book_sources_or_references_modified",
                "ready_v481": False,
                "evidence_artifact_v481": "book sources unchanged",
                "claim_boundary_v481": "no Quarto/book mutation in v481",
            },
            {
                "readiness_gate_v481": "submission_ready",
                "ready_v481": False,
                "evidence_artifact_v481": "future approval, patch and render validation",
                "claim_boundary_v481": "not a submission package",
            },
            {
                "readiness_gate_v481": "paper4_final_promotion_created",
                "ready_v481": False,
                "evidence_artifact_v481": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v481": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v481_manual_patch_decision_created",
                "allowed": True,
                "artifact": "paper4_v481_patch_decision_register.csv",
                "boundary": "decision only",
            },
            {
                "claim_id": "v481_manual_review_requirements_created",
                "allowed": True,
                "artifact": "paper4_v481_manual_review_requirements.csv",
                "boundary": "requirements only",
            },
            {
                "claim_id": "v481_next_action_queue_created",
                "allowed": True,
                "artifact": "paper4_v481_next_action_queue.csv",
                "boundary": "next-action queue only",
            },
            {
                "claim_id": "v481_quarto_patch_allowed_or_applied",
                "allowed": False,
                "artifact": "paper4_v481_manuscript_readiness_delta.csv",
                "boundary": "explicit approval missing",
            },
            {
                "claim_id": "v481_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v481_manuscript_readiness_delta.csv",
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
                "claim": "v481 records a manual Quarto patch decision.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v481_patch_decision_register.csv"
                ),
                "boundary": "Decision only; patch not authorized.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v481 records review requirements and next actions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v481_manual_review_requirements.csv"
                ),
                "boundary": "Planning only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v481 authorizes or applies a Quarto patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v481_patch_decision_register.csv"
                ),
                "boundary": "Explicit user approval is missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v481 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v481_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v481 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v481_manuscript_readiness_delta.csv"
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
                "executable_item": "v481 records manual Quarto patch decision.",
                "status": "manual_quarto_patch_decision_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v482 synthesizes post-plan manuscript state",
                "last_wave": "v481",
                "execution_result": "manual_patch_blocked_without_explicit_approval",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v481")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Quarto Patch Decision v481

Generated: {status["generated_at_utc"]}

## Result

v481 records that the controlled insertion plan exists, but an actual Quarto
patch is not authorized because explicit human approval and manual caption/asset
review are still missing. No book source is edited.

## Counts

- Patch decision rows: `{status["patch_decision_rows_v481"]}`.
- Manual review requirement rows: `{status["manual_review_requirement_rows_v481"]}`.
- Next-action rows: `{status["next_action_rows_v481"]}`.
- Patch allowed: `{status["patch_allowed_v481"]}`.
- Patch applied: `{status["patch_applied_v481"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v481 is a decision record only. It does not authorize a patch, edit Quarto,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V481_MANUAL_QUARTO_PATCH_DECISION_START -->"
    end = "<!-- V481_MANUAL_QUARTO_PATCH_DECISION_END -->"
    block = f"""
{start}

## Wave v481: Manual Quarto Patch Decision

Generated: {status["generated_at_utc"]}

### Objective

v481 records whether the v480 controlled insertion plan is allowed to become a
book-source patch.

### Results

- Patch decision rows:
  `{status["patch_decision_rows_v481"]}`.
- Manual review requirement rows:
  `{status["manual_review_requirement_rows_v481"]}`.
- Next-action rows:
  `{status["next_action_rows_v481"]}`.
- Patch allowed:
  `{status["patch_allowed_v481"]}`.
- Patch applied:
  `{status["patch_applied_v481"]}`.
- Book sources modified:
  `{status["book_sources_modified_v481"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v481"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v481"]}`.

### Interpretation

The safe decision is to keep the patch blocked until explicit approval and
manual review exist. The living-lab artifacts are ready for review, not for
automatic insertion.

### Claim Impact

- Allowed: decision register, manual review requirements and next-action queue.
- Still prohibited: Quarto patch authorization/application, book-source mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v481 in the living notebook. v482 should synthesize the post-plan manuscript
state and preserve the patch-blocked boundary.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v480 = _read_status(PRIOR_INSERTION_PLAN_VERSION)
    if v480["next_artifact_v480"] != "paper4_v481_manual_quarto_patch_decision.md":
        raise RuntimeError("v481 expects v480 to route to manual patch decision.")

    decision = _patch_decision_register()
    requirements = _manual_review_requirements()
    actions = _next_action_queue()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v481_patch_decision_register.csv", decision)
    write_csv(TABLE_DIR / "paper4_v481_manual_review_requirements.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v481_next_action_queue.csv", actions)
    write_csv(TABLE_DIR / "paper4_v481_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v481_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v481_manual_quarto_patch_decision",
        "schema_version": "2026-05-17.481",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_insertion_plan_version_v481": PRIOR_INSERTION_PLAN_VERSION,
        "manual_quarto_patch_decision_created_v481": True,
        "patch_decision_rows_v481": len(decision),
        "manual_review_requirement_rows_v481": len(requirements),
        "blocking_requirement_rows_v481": int(requirements["blocks_patch_v481"].astype(bool).sum()),
        "next_action_rows_v481": len(actions),
        "patch_allowed_v481": bool(decision["patch_allowed_v481"].astype(bool).any()),
        "patch_applied_v481": bool(decision["patch_applied_v481"].astype(bool).any()),
        "book_sources_modified_v481": False,
        "book_references_modified_v481": False,
        "submission_ready_claim_allowed_v481": False,
        "working_champion_claim_allowed_v481": False,
        "paper1_promotion_allowed_v481": False,
        "paper4_working_champion_changed_v481": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v481": NEXT_ARTIFACT,
        "claim_boundary": (
            "v481 records patch denial only; actual patching, final prose, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v481 must not create final Paper 4 promotion.")

    DECISION_MD.write_text(_decision_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v481": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
