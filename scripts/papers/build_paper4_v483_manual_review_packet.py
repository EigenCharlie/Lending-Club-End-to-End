#!/usr/bin/env python3
"""Build Paper 4 v483 manual review packet artifacts."""

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

VERSION = 483
PRIOR_SYNTHESIS_VERSION = 482
NEXT_ARTIFACT = "paper4_v484_caption_hardening_dry_run.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v483_manual_review_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _review_packet_items() -> pd.DataFrame:
    seed = pd.read_csv(TABLE_DIR / "paper4_v482_manual_review_packet_seed.csv")
    owners = {
        "approve_or_reject_book_source_patch": "paper_owner",
        "review_caption_language": "paper_editor",
        "review_target_file_anchors": "quarto_editor",
        "confirm_post_patch_render_gate": "validation_owner",
        "confirm_post_patch_guardrail_scope": "validation_owner",
    }
    rows = []
    for _, row in seed.sort_values("priority_v482").iterrows():
        item_id = str(row["review_item_id_v482"])
        rows.append(
            {
                "review_item_id_v483": item_id,
                "priority_v483": int(row["priority_v482"]),
                "review_owner_v483": owners[item_id],
                "source_artifact_v483": row["source_artifact_v482"],
                "review_question_v483": row["review_question_v482"],
                "ready_for_review_v483": bool(row["ready_for_review_v482"]),
                "decision_recorded_v483": False,
                "patch_authorized_by_item_v483": False,
                "claim_boundary_v483": "manual review packet only",
            }
        )
    return pd.DataFrame(rows)


def _evidence_bundle() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evidence_id_v483": "captions",
                "artifact_v483": "paper4_v476_caption_plan.csv",
                "review_use_v483": "caption language and caveat review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "visual_section_delta",
                "artifact_v483": "paper4_v477_visual_section_delta.csv",
                "review_use_v483": "asset-to-section review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "section_stubs",
                "artifact_v483": "paper4_v478_section_text_stubs.csv",
                "review_use_v483": "draft prose review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "stub_claim_audit",
                "artifact_v483": "paper4_v479_stub_consistency_checks.csv",
                "review_use_v483": "claim consistency review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "insertion_plan",
                "artifact_v483": "paper4_v480_quarto_insertion_plan.csv",
                "review_use_v483": "target file and anchor review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "rollback_plan",
                "artifact_v483": "paper4_v480_rollback_plan.csv",
                "review_use_v483": "rollback feasibility review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "patch_decision",
                "artifact_v483": "paper4_v481_patch_decision_register.csv",
                "review_use_v483": "approval state review",
                "included_v483": True,
            },
            {
                "evidence_id_v483": "post_plan_state",
                "artifact_v483": "paper4_v482_manuscript_state_matrix.csv",
                "review_use_v483": "overall manuscript state review",
                "included_v483": True,
            },
        ]
    )


def _decision_option_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_option_id_v483": "keep_patch_blocked",
                "recommended_v483": True,
                "patch_allowed_v483": False,
                "requires_explicit_approval_v483": False,
                "next_artifact_if_selected_v483": NEXT_ARTIFACT,
                "claim_boundary_v483": "continue dry-run editing only",
            },
            {
                "decision_option_id_v483": "caption_hardening_dry_run",
                "recommended_v483": True,
                "patch_allowed_v483": False,
                "requires_explicit_approval_v483": False,
                "next_artifact_if_selected_v483": NEXT_ARTIFACT,
                "claim_boundary_v483": "harden captions without Quarto mutation",
            },
            {
                "decision_option_id_v483": "manual_patch_after_approval",
                "recommended_v483": False,
                "patch_allowed_v483": False,
                "requires_explicit_approval_v483": True,
                "next_artifact_if_selected_v483": "future_manual_patch_plan",
                "claim_boundary_v483": "not authorized in v483",
            },
            {
                "decision_option_id_v483": "declare_submission_ready",
                "recommended_v483": False,
                "patch_allowed_v483": False,
                "requires_explicit_approval_v483": True,
                "next_artifact_if_selected_v483": "blocked_submission_gate",
                "claim_boundary_v483": "submission and final promotion remain blocked",
            },
        ]
    )


def _risk_control_matrix() -> pd.DataFrame:
    blockers = pd.read_csv(TABLE_DIR / "paper4_v482_open_blocker_register.csv")
    controls = {
        "explicit_patch_approval_missing": "keep patch_authorized false",
        "caption_finalization_review_missing": "run caption hardening dry run",
        "asset_path_review_missing": "keep target-file review item open",
        "post_patch_render_missing": "defer render until patch exists",
        "venue_and_external_validation_missing": "keep submission-ready false",
        "paper4_final_promotion_forbidden": "keep final promotion artifact absent",
    }
    rows = []
    for _, row in blockers.iterrows():
        blocker_id = str(row["blocker_id_v482"])
        rows.append(
            {
                "blocker_id_v483": blocker_id,
                "control_v483": controls[blocker_id],
                "control_active_v483": True,
                "blocker_resolved_v483": False,
                "patch_allowed_after_control_v483": False,
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v483": "manual_review_packet_created",
                "ready_v483": True,
                "evidence_artifact_v483": "paper4_v483_review_packet_items.csv",
                "claim_boundary_v483": "packet only",
            },
            {
                "readiness_gate_v483": "evidence_bundle_created",
                "ready_v483": True,
                "evidence_artifact_v483": "paper4_v483_evidence_bundle.csv",
                "claim_boundary_v483": "review bundle only",
            },
            {
                "readiness_gate_v483": "decision_options_created",
                "ready_v483": True,
                "evidence_artifact_v483": "paper4_v483_decision_option_matrix.csv",
                "claim_boundary_v483": "decision options only",
            },
            {
                "readiness_gate_v483": "risk_controls_created",
                "ready_v483": True,
                "evidence_artifact_v483": "paper4_v483_risk_control_matrix.csv",
                "claim_boundary_v483": "risk controls only",
            },
            {
                "readiness_gate_v483": "patch_authorized",
                "ready_v483": False,
                "evidence_artifact_v483": "paper4_v483_decision_option_matrix.csv",
                "claim_boundary_v483": "patch remains blocked",
            },
            {
                "readiness_gate_v483": "book_sources_or_references_modified",
                "ready_v483": False,
                "evidence_artifact_v483": "book sources unchanged",
                "claim_boundary_v483": "no Quarto/book mutation in v483",
            },
            {
                "readiness_gate_v483": "submission_ready",
                "ready_v483": False,
                "evidence_artifact_v483": "future approval, patch, render and venue gates",
                "claim_boundary_v483": "not a submission package",
            },
            {
                "readiness_gate_v483": "paper4_final_promotion_created",
                "ready_v483": False,
                "evidence_artifact_v483": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v483": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v483_manual_review_packet_created",
                "allowed": True,
                "artifact": "paper4_v483_review_packet_items.csv",
                "boundary": "review packet only",
            },
            {
                "claim_id": "v483_evidence_bundle_created",
                "allowed": True,
                "artifact": "paper4_v483_evidence_bundle.csv",
                "boundary": "evidence bundle only",
            },
            {
                "claim_id": "v483_decision_options_created",
                "allowed": True,
                "artifact": "paper4_v483_decision_option_matrix.csv",
                "boundary": "decision options only",
            },
            {
                "claim_id": "v483_risk_controls_created",
                "allowed": True,
                "artifact": "paper4_v483_risk_control_matrix.csv",
                "boundary": "controls only; blockers remain open",
            },
            {
                "claim_id": "v483_patch_authorized_or_applied",
                "allowed": False,
                "artifact": "paper4_v483_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v483_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v483_manuscript_readiness_delta.csv",
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
                "claim": "v483 creates a manual review packet for Paper 4 manuscript work.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_review_packet_items.csv"
                ),
                "boundary": "Review packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v483 bundles evidence for manual review.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_evidence_bundle.csv"
                ),
                "boundary": "Evidence bundle only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v483 recommends caption hardening as the next dry-run.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_decision_option_matrix.csv"
                ),
                "boundary": "Dry-run recommendation only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v483 authorizes or applies a Quarto patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v483 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v483 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v483_manuscript_readiness_delta.csv"
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
                "executable_item": "v483 builds manual review packet.",
                "status": "manual_review_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v484 hardens captions without book edit",
                "last_wave": "v483",
                "execution_result": "manual_review_packet_created_without_patch",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v483")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manual Review Packet v483

Generated: {status["generated_at_utc"]}

## Result

v483 turns the v482 review seed into a manual review packet. It bundles review
items, evidence, decision options and risk controls. It recommends continuing
with caption hardening as a dry-run and does not authorize any Quarto patch.

## Counts

- Review item rows: `{status["review_item_rows_v483"]}`.
- Evidence bundle rows: `{status["evidence_bundle_rows_v483"]}`.
- Decision option rows: `{status["decision_option_rows_v483"]}`.
- Risk control rows: `{status["risk_control_rows_v483"]}`.
- Patch authorized: `{status["patch_authorized_v483"]}`.
- Patch applied: `{status["patch_applied_v483"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v483 is a review packet only. It does not authorize a patch, edit Quarto, render
the book, make Paper 4 submission-ready, replace Paper Estrella, or promote
Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V483_MANUAL_REVIEW_PACKET_START -->"
    end = "<!-- V483_MANUAL_REVIEW_PACKET_END -->"
    block = f"""
{start}

## Wave v483: Manual Review Packet

Generated: {status["generated_at_utc"]}

### Objective

v483 converts the v482 review seed into a manual review packet with evidence,
decision options and risk controls, without authorizing a patch.

### Results

- Review item rows:
  `{status["review_item_rows_v483"]}`.
- Evidence bundle rows:
  `{status["evidence_bundle_rows_v483"]}`.
- Decision option rows:
  `{status["decision_option_rows_v483"]}`.
- Risk control rows:
  `{status["risk_control_rows_v483"]}`.
- Patch authorized:
  `{status["patch_authorized_v483"]}`.
- Patch applied:
  `{status["patch_applied_v483"]}`.
- Book sources modified:
  `{status["book_sources_modified_v483"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v483"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v483"]}`.

### Interpretation

The review surface is now concrete: reviewers can inspect captions, stubs, target
files, rollback and guardrails. The safe executable next step is caption
hardening, still as a dry-run.

### Claim Impact

- Allowed: manual review packet, evidence bundle, decision options and risk
  controls.
- Still prohibited: patch authorization/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v483 in the living notebook. v484 should harden captions without modifying
book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v482 = _read_status(PRIOR_SYNTHESIS_VERSION)
    if v482["next_artifact_v482"] != "paper4_v483_manual_review_packet.md":
        raise RuntimeError("v483 expects v482 to route to manual review packet.")

    review_items = _review_packet_items()
    evidence = _evidence_bundle()
    options = _decision_option_matrix()
    controls = _risk_control_matrix()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v483_review_packet_items.csv", review_items)
    write_csv(TABLE_DIR / "paper4_v483_evidence_bundle.csv", evidence)
    write_csv(TABLE_DIR / "paper4_v483_decision_option_matrix.csv", options)
    write_csv(TABLE_DIR / "paper4_v483_risk_control_matrix.csv", controls)
    write_csv(TABLE_DIR / "paper4_v483_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v483_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v483_manual_review_packet",
        "schema_version": "2026-05-17.483",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_synthesis_version_v483": PRIOR_SYNTHESIS_VERSION,
        "manual_review_packet_created_v483": True,
        "review_item_rows_v483": len(review_items),
        "evidence_bundle_rows_v483": len(evidence),
        "decision_option_rows_v483": len(options),
        "risk_control_rows_v483": len(controls),
        "readiness_delta_rows_v483": len(readiness),
        "patch_authorized_v483": False,
        "patch_applied_v483": False,
        "book_sources_modified_v483": False,
        "book_references_modified_v483": False,
        "submission_ready_claim_allowed_v483": False,
        "working_champion_claim_allowed_v483": False,
        "paper1_promotion_allowed_v483": False,
        "paper4_working_champion_changed_v483": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v483": NEXT_ARTIFACT,
        "claim_boundary": (
            "v483 creates a manual review packet only; patching, final prose, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v483 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v483": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
