#!/usr/bin/env python3
"""Build Paper 4 v482 post-plan synthesis packet artifacts."""

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

VERSION = 482
PRIOR_PATCH_DECISION_VERSION = 481
NEXT_ARTIFACT = "paper4_v483_manual_review_packet.md"
SYNTHESIS_MD = NOTEBOOK.parent / "paper4_v482_post_plan_synthesis_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _post_plan_wave_summary() -> pd.DataFrame:
    rows = []
    summaries = {
        476: (
            "caption_and_insertion_plan",
            "draft captions and insertion order created",
            "captions remain draft; no Quarto insertion",
        ),
        477: (
            "post_visual_package_manuscript_delta",
            "visual package mapped into manuscript sections",
            "assets remain uninserted and blocker caveats stay active",
        ),
        478: (
            "section_text_stub_bundle",
            "section text stubs and callout sentences drafted",
            "stubs are not final prose and are not in book sources",
        ),
        479: (
            "stub_claim_consistency_audit",
            "stub-claim audit passed with no prohibited assertions",
            "audit is not insertion authorization",
        ),
        480: (
            "controlled_quarto_insertion_plan",
            "candidate Quarto targets, gates and rollback plan identified",
            "manual review is mandatory before any patch",
        ),
        481: (
            "manual_quarto_patch_decision",
            "manual patch decision recorded as blocked",
            "explicit approval, caption review and asset review are missing",
        ),
    }
    for version, (phase, result, blocker) in summaries.items():
        rows.append(
            {
                "wave_v482": f"v{version}",
                "phase_v482": phase,
                "primary_result_v482": result,
                "preserved_blocker_v482": blocker,
                "book_sources_modified_v482": False,
                "final_promotion_created_v482": False,
                "claim_boundary_v482": "living notebook evidence only",
            }
        )
    return pd.DataFrame(rows)


def _manuscript_state_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_gate_v482": "visual_package_selected_and_captioned",
                "state_ready_v482": True,
                "evidence_v482": "paper4_v476_caption_plan.csv",
                "claim_boundary_v482": "draft visual package only",
            },
            {
                "state_gate_v482": "visual_package_mapped_to_manuscript",
                "state_ready_v482": True,
                "evidence_v482": "paper4_v477_visual_section_delta.csv",
                "claim_boundary_v482": "manuscript delta only",
            },
            {
                "state_gate_v482": "section_text_stubs_available",
                "state_ready_v482": True,
                "evidence_v482": "paper4_v478_section_text_stubs.csv",
                "claim_boundary_v482": "draft stubs only",
            },
            {
                "state_gate_v482": "stub_claim_consistency_audited",
                "state_ready_v482": True,
                "evidence_v482": "paper4_v479_stub_consistency_checks.csv",
                "claim_boundary_v482": "audit only",
            },
            {
                "state_gate_v482": "controlled_insertion_plan_available",
                "state_ready_v482": True,
                "evidence_v482": "paper4_v480_quarto_insertion_plan.csv",
                "claim_boundary_v482": "plan only",
            },
            {
                "state_gate_v482": "manual_patch_authorized",
                "state_ready_v482": False,
                "evidence_v482": "paper4_v481_patch_decision_register.csv",
                "claim_boundary_v482": "explicit approval missing",
            },
            {
                "state_gate_v482": "book_sources_or_references_modified",
                "state_ready_v482": False,
                "evidence_v482": "book sources unchanged",
                "claim_boundary_v482": "no Quarto/book mutation",
            },
            {
                "state_gate_v482": "submission_ready_or_final_promotion",
                "state_ready_v482": False,
                "evidence_v482": "future venue, patch, render and external validation",
                "claim_boundary_v482": "Paper Estrella remains protected",
            },
        ]
    )


def _open_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v482": "explicit_patch_approval_missing",
                "blocks_v482": "manual_patch_authorization",
                "evidence_v482": "paper4_v481_manual_review_requirements.csv",
                "resolved_v482": False,
            },
            {
                "blocker_id_v482": "caption_finalization_review_missing",
                "blocks_v482": "caption_finalization",
                "evidence_v482": "paper4_v481_manual_review_requirements.csv",
                "resolved_v482": False,
            },
            {
                "blocker_id_v482": "asset_path_review_missing",
                "blocks_v482": "safe_visual_insertion",
                "evidence_v482": "paper4_v481_manual_review_requirements.csv",
                "resolved_v482": False,
            },
            {
                "blocker_id_v482": "post_patch_render_missing",
                "blocks_v482": "render_validated_patch",
                "evidence_v482": "paper4_v480_pre_patch_gate_checklist.csv",
                "resolved_v482": False,
            },
            {
                "blocker_id_v482": "venue_and_external_validation_missing",
                "blocks_v482": "submission_ready_claim",
                "evidence_v482": "paper4_v480_manuscript_readiness_delta.csv",
                "resolved_v482": False,
            },
            {
                "blocker_id_v482": "paper4_final_promotion_forbidden",
                "blocks_v482": "final_paper4_or_paper_estrella_replacement_claim",
                "evidence_v482": "paper4_final_promotion.json absent",
                "resolved_v482": False,
            },
        ]
    )


def _manual_review_packet_seed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_item_id_v482": "approve_or_reject_book_source_patch",
                "priority_v482": 1,
                "ready_for_review_v482": True,
                "source_artifact_v482": "paper4_v481_patch_decision_register.csv",
                "review_question_v482": "Is a manual book-source patch approved?",
            },
            {
                "review_item_id_v482": "review_caption_language",
                "priority_v482": 2,
                "ready_for_review_v482": True,
                "source_artifact_v482": "paper4_v476_caption_plan.csv",
                "review_question_v482": "Which captions can become final under caveats?",
            },
            {
                "review_item_id_v482": "review_target_file_anchors",
                "priority_v482": 3,
                "ready_for_review_v482": True,
                "source_artifact_v482": "paper4_v480_quarto_insertion_plan.csv",
                "review_question_v482": "Are target files and anchors appropriate?",
            },
            {
                "review_item_id_v482": "confirm_post_patch_render_gate",
                "priority_v482": 4,
                "ready_for_review_v482": False,
                "source_artifact_v482": "paper4_v480_pre_patch_gate_checklist.csv",
                "review_question_v482": "Which render gate must pass after a patch?",
            },
            {
                "review_item_id_v482": "confirm_post_patch_guardrail_scope",
                "priority_v482": 5,
                "ready_for_review_v482": False,
                "source_artifact_v482": "tests/test_docs/test_paper4_living_lab_guardrails.py",
                "review_question_v482": "Which guardrails must rerun after patching?",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v482": "post_plan_synthesis_packet_created",
                "ready_v482": True,
                "evidence_artifact_v482": "paper4_v482_post_plan_wave_summary.csv",
                "claim_boundary_v482": "synthesis packet only",
            },
            {
                "readiness_gate_v482": "manuscript_state_matrix_created",
                "ready_v482": True,
                "evidence_artifact_v482": "paper4_v482_manuscript_state_matrix.csv",
                "claim_boundary_v482": "state matrix only",
            },
            {
                "readiness_gate_v482": "open_blockers_preserved",
                "ready_v482": True,
                "evidence_artifact_v482": "paper4_v482_open_blocker_register.csv",
                "claim_boundary_v482": "blockers remain open",
            },
            {
                "readiness_gate_v482": "manual_review_packet_seed_created",
                "ready_v482": True,
                "evidence_artifact_v482": "paper4_v482_manual_review_packet_seed.csv",
                "claim_boundary_v482": "review seed only",
            },
            {
                "readiness_gate_v482": "manual_patch_authorized",
                "ready_v482": False,
                "evidence_artifact_v482": "paper4_v481_patch_decision_register.csv",
                "claim_boundary_v482": "patch remains blocked",
            },
            {
                "readiness_gate_v482": "book_sources_or_references_modified",
                "ready_v482": False,
                "evidence_artifact_v482": "book sources unchanged",
                "claim_boundary_v482": "no Quarto/book mutation in v482",
            },
            {
                "readiness_gate_v482": "submission_ready",
                "ready_v482": False,
                "evidence_artifact_v482": "future approval, patch, render and venue gates",
                "claim_boundary_v482": "not a submission package",
            },
            {
                "readiness_gate_v482": "paper4_final_promotion_created",
                "ready_v482": False,
                "evidence_artifact_v482": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v482": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v482_post_plan_synthesis_packet_created",
                "allowed": True,
                "artifact": "paper4_v482_post_plan_wave_summary.csv",
                "boundary": "synthesis only",
            },
            {
                "claim_id": "v482_manuscript_state_matrix_created",
                "allowed": True,
                "artifact": "paper4_v482_manuscript_state_matrix.csv",
                "boundary": "state matrix only",
            },
            {
                "claim_id": "v482_open_blockers_preserved",
                "allowed": True,
                "artifact": "paper4_v482_open_blocker_register.csv",
                "boundary": "blockers remain open",
            },
            {
                "claim_id": "v482_manual_review_packet_seed_created",
                "allowed": True,
                "artifact": "paper4_v482_manual_review_packet_seed.csv",
                "boundary": "review seed only",
            },
            {
                "claim_id": "v482_patch_authorized_or_applied",
                "allowed": False,
                "artifact": "paper4_v482_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v482_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v482_manuscript_readiness_delta.csv",
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
                "claim": "v482 synthesizes the post-plan Paper 4 manuscript state.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_manuscript_state_matrix.csv"
                ),
                "boundary": "Synthesis only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v482 preserves open blockers after the controlled insertion plan.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_open_blocker_register.csv"
                ),
                "boundary": "Blockers remain open.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v482 seeds a manual review packet for future decisions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_manual_review_packet_seed.csv"
                ),
                "boundary": "Review seed only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v482 authorizes or applies a Quarto patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked by v481 decision.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v482 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v482 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v482_manuscript_readiness_delta.csv"
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
                "executable_item": "v482 synthesizes post-plan manuscript state.",
                "status": "post_plan_synthesis_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v483 builds manual review packet",
                "last_wave": "v482",
                "execution_result": "post_plan_state_synthesized_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v482")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _synthesis_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Plan Synthesis Packet v482

Generated: {status["generated_at_utc"]}

## Result

v482 synthesizes the v476-v481 manuscript-planning arc. The package records
which manuscript assets are useful now, which blockers remain open, and what a
future manual review packet should inspect. It does not authorize a Quarto
patch, edit book sources, make Paper 4 submission-ready, replace Paper Estrella,
or promote Paper 4.

## Counts

- Wave summary rows: `{status["post_plan_wave_summary_rows_v482"]}`.
- Manuscript state rows: `{status["manuscript_state_rows_v482"]}`.
- Open blocker rows: `{status["open_blocker_rows_v482"]}`.
- Manual review seed rows: `{status["manual_review_seed_rows_v482"]}`.
- Patch authorized: `{status["patch_authorized_v482"]}`.
- Patch applied: `{status["patch_applied_v482"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v482 is synthesis and review preparation only. Manual patch authorization,
Quarto edits, render validation, venue formatting, submission readiness, Paper
Estrella replacement and final Paper 4 promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V482_POST_PLAN_SYNTHESIS_PACKET_START -->"
    end = "<!-- V482_POST_PLAN_SYNTHESIS_PACKET_END -->"
    block = f"""
{start}

## Wave v482: Post-Plan Synthesis Packet

Generated: {status["generated_at_utc"]}

### Objective

v482 synthesizes the v476-v481 manuscript-planning sequence into a reviewable
post-plan state while preserving the no-patch boundary.

### Results

- Wave summary rows:
  `{status["post_plan_wave_summary_rows_v482"]}`.
- Manuscript state rows:
  `{status["manuscript_state_rows_v482"]}`.
- Open blocker rows:
  `{status["open_blocker_rows_v482"]}`.
- Manual review seed rows:
  `{status["manual_review_seed_rows_v482"]}`.
- Patch authorized:
  `{status["patch_authorized_v482"]}`.
- Patch applied:
  `{status["patch_applied_v482"]}`.
- Book sources modified:
  `{status["book_sources_modified_v482"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v482"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v482"]}`.

### Interpretation

The manuscript-planning arc has produced usable drafting and review evidence,
but the patch remains blocked. The next useful move is a manual review packet,
not automatic source mutation.

### Claim Impact

- Allowed: post-plan synthesis, manuscript state matrix, open-blocker register
  and manual review packet seed.
- Still prohibited: patch authorization/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v482 in the living notebook. v483 should build the manual review packet
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v481 = _read_status(PRIOR_PATCH_DECISION_VERSION)
    if v481["next_artifact_v481"] != "paper4_v482_post_plan_synthesis_packet.md":
        raise RuntimeError("v482 expects v481 to route to post-plan synthesis packet.")

    wave_summary = _post_plan_wave_summary()
    state_matrix = _manuscript_state_matrix()
    blockers = _open_blocker_register()
    review_seed = _manual_review_packet_seed()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v482_post_plan_wave_summary.csv", wave_summary)
    write_csv(TABLE_DIR / "paper4_v482_manuscript_state_matrix.csv", state_matrix)
    write_csv(TABLE_DIR / "paper4_v482_open_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v482_manual_review_packet_seed.csv", review_seed)
    write_csv(TABLE_DIR / "paper4_v482_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v482_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v482_post_plan_synthesis_packet",
        "schema_version": "2026-05-17.482",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_patch_decision_version_v482": PRIOR_PATCH_DECISION_VERSION,
        "post_plan_synthesis_packet_created_v482": True,
        "post_plan_wave_summary_rows_v482": len(wave_summary),
        "manuscript_state_rows_v482": len(state_matrix),
        "open_blocker_rows_v482": len(blockers),
        "manual_review_seed_rows_v482": len(review_seed),
        "readiness_delta_rows_v482": len(readiness),
        "patch_authorized_v482": False,
        "patch_applied_v482": False,
        "book_sources_modified_v482": False,
        "book_references_modified_v482": False,
        "submission_ready_claim_allowed_v482": False,
        "working_champion_claim_allowed_v482": False,
        "paper1_promotion_allowed_v482": False,
        "paper4_working_champion_changed_v482": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v482": NEXT_ARTIFACT,
        "claim_boundary": (
            "v482 synthesizes post-plan state only; patching, final prose, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v482 must not create final Paper 4 promotion.")

    SYNTHESIS_MD.write_text(_synthesis_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v482": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
