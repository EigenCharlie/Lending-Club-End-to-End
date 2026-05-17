#!/usr/bin/env python3
"""Build Paper 4 v494 patch approval gap packet artifacts."""

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

VERSION = 494
PRIOR_CAPTION_SIGNOFF_VERSION = 493
NEXT_ARTIFACT = "paper4_v495_no_patch_release_synthesis.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v494_patch_approval_gap_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _patch_approval_gap_packet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "approval_gap_id_v494": "explicit_patch_approval_documented",
                "approval_ready_v494": False,
                "required_before_patch_v494": True,
                "blocks_patch_v494": True,
                "gap_boundary_v494": "no explicit patch approval exists",
            },
            {
                "approval_gap_id_v494": "approver_identity_recorded",
                "approval_ready_v494": False,
                "required_before_patch_v494": True,
                "blocks_patch_v494": True,
                "gap_boundary_v494": "no accountable approver recorded",
            },
            {
                "approval_gap_id_v494": "mutation_scope_approved",
                "approval_ready_v494": False,
                "required_before_patch_v494": True,
                "blocks_patch_v494": True,
                "gap_boundary_v494": "book mutation scope not approved",
            },
            {
                "approval_gap_id_v494": "rollback_plan_accepted",
                "approval_ready_v494": False,
                "required_before_patch_v494": True,
                "blocks_patch_v494": True,
                "gap_boundary_v494": "rollback acceptance missing",
            },
            {
                "approval_gap_id_v494": "caption_signoff_synced",
                "approval_ready_v494": False,
                "required_before_patch_v494": True,
                "blocks_patch_v494": True,
                "gap_boundary_v494": "v493 caption signoff remains open",
            },
            {
                "approval_gap_id_v494": "paper_estrella_boundary_reconfirmed",
                "approval_ready_v494": True,
                "required_before_patch_v494": True,
                "blocks_patch_v494": False,
                "gap_boundary_v494": "final promotion remains absent",
            },
        ]
    )


def _approval_request_packet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_item_id_v494": "approve_or_reject_patch_attempt",
                "request_status_v494": "pending",
                "required_response_v494": "approve, reject, or keep blocked",
                "patch_allowed_v494": False,
            },
            {
                "request_item_id_v494": "confirm_target_files",
                "request_status_v494": "pending",
                "required_response_v494": "confirm four target source files",
                "patch_allowed_v494": False,
            },
            {
                "request_item_id_v494": "confirm_caption_signoff_dependency",
                "request_status_v494": "pending",
                "required_response_v494": "confirm captions must be final before patch",
                "patch_allowed_v494": False,
            },
            {
                "request_item_id_v494": "confirm_rollback_expectation",
                "request_status_v494": "pending",
                "required_response_v494": "confirm rollback and render checks",
                "patch_allowed_v494": False,
            },
            {
                "request_item_id_v494": "confirm_no_final_promotion",
                "request_status_v494": "reconfirmed",
                "required_response_v494": "keep Paper Estrella boundary active",
                "patch_allowed_v494": False,
            },
        ]
    )


def _approval_scope_matrix() -> pd.DataFrame:
    review_packet = pd.read_csv(TABLE_DIR / "paper4_v492_manual_layout_review_packet.csv")
    return pd.DataFrame(
        [
            {
                "scope_item_id_v494": row["review_item_id_v492"],
                "target_file_v494": row["target_file_v492"],
                "target_block_v494": row["target_block_v492"],
                "asset_sequence_v494": row["asset_sequence_v492"],
                "scope_reviewed_v494": True,
                "scope_approved_for_patch_v494": False,
                "blocks_patch_v494": True,
            }
            for _, row in review_packet.iterrows()
        ]
    )


def _approval_decision_options() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_option_id_v494": "keep_patch_blocked",
                "recommended_v494": True,
                "patch_allowed_v494": False,
                "decision_boundary_v494": "approval and caption signoff missing",
            },
            {
                "decision_option_id_v494": "prepare_no_patch_release_synthesis",
                "recommended_v494": True,
                "patch_allowed_v494": False,
                "decision_boundary_v494": "summarize useful no-patch evidence",
            },
            {
                "decision_option_id_v494": "request_explicit_patch_approval",
                "recommended_v494": True,
                "patch_allowed_v494": False,
                "decision_boundary_v494": "request only, no mutation",
            },
            {
                "decision_option_id_v494": "apply_quarto_patch_now",
                "recommended_v494": False,
                "patch_allowed_v494": False,
                "decision_boundary_v494": "not authorized in v494",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v494": "patch_approval_gap_packet_created",
                "ready_v494": True,
                "evidence_artifact_v494": "paper4_v494_patch_approval_gap_packet.csv",
                "claim_boundary_v494": "approval gap packet only",
            },
            {
                "readiness_gate_v494": "approval_request_packet_created",
                "ready_v494": True,
                "evidence_artifact_v494": "paper4_v494_approval_request_packet.csv",
                "claim_boundary_v494": "request packet only",
            },
            {
                "readiness_gate_v494": "approval_scope_matrix_created",
                "ready_v494": True,
                "evidence_artifact_v494": "paper4_v494_approval_scope_matrix.csv",
                "claim_boundary_v494": "scope matrix only",
            },
            {
                "readiness_gate_v494": "approval_decision_options_created",
                "ready_v494": True,
                "evidence_artifact_v494": "paper4_v494_approval_decision_options.csv",
                "claim_boundary_v494": "decision options only",
            },
            {
                "readiness_gate_v494": "ready_for_quarto_patch",
                "ready_v494": False,
                "evidence_artifact_v494": "explicit patch approval missing",
                "claim_boundary_v494": "patch remains blocked",
            },
            {
                "readiness_gate_v494": "book_sources_or_references_modified",
                "ready_v494": False,
                "evidence_artifact_v494": "book sources unchanged",
                "claim_boundary_v494": "no Quarto/book mutation in v494",
            },
            {
                "readiness_gate_v494": "submission_ready",
                "ready_v494": False,
                "evidence_artifact_v494": "future approval, patch, render and venue gates",
                "claim_boundary_v494": "not a submission package",
            },
            {
                "readiness_gate_v494": "paper4_final_promotion_created",
                "ready_v494": False,
                "evidence_artifact_v494": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v494": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v494_patch_approval_gap_packet_created",
                "allowed": True,
                "artifact": "paper4_v494_patch_approval_gap_packet.csv",
                "boundary": "approval gap packet only",
            },
            {
                "claim_id": "v494_approval_request_packet_created",
                "allowed": True,
                "artifact": "paper4_v494_approval_request_packet.csv",
                "boundary": "request packet only",
            },
            {
                "claim_id": "v494_approval_scope_matrix_created",
                "allowed": True,
                "artifact": "paper4_v494_approval_scope_matrix.csv",
                "boundary": "scope matrix only",
            },
            {
                "claim_id": "v494_explicit_patch_approval_or_patch_ready",
                "allowed": False,
                "artifact": "paper4_v494_manuscript_readiness_delta.csv",
                "boundary": "approval remains missing and patch remains blocked",
            },
            {
                "claim_id": "v494_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v494_manuscript_readiness_delta.csv",
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
                "claim": "v494 audits explicit patch approval gaps for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_patch_approval_gap_packet.csv"
                ),
                "boundary": "Patch approval gap packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v494 creates a bounded patch approval request packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_approval_request_packet.csv"
                ),
                "boundary": "Approval request packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v494 maps patch scope without approving mutation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_approval_scope_matrix.csv"
                ),
                "boundary": "Patch scope mapping only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v494 obtains explicit patch approval or makes Paper 4 patch-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval remains missing and patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v494 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_approval_decision_options.csv"
                ),
                "boundary": "Patch is not authorized in v494.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v494 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v494_manuscript_readiness_delta.csv"
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
                "executable_item": "v494 audits explicit patch approval gaps.",
                "status": "patch_approval_gap_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v495 synthesizes no-patch release evidence",
                "last_wave": "v494",
                "execution_result": "patch_approval_gap_packet_created_without_approval",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v494")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Patch Approval Gap Packet v494

Generated: {status["generated_at_utc"]}

## Result

v494 audits explicit patch approval gaps. It maps approval requirements, request
items and patch scope, but does not obtain approval or authorize any source
mutation.

## Counts

- Approval gap rows: `{status["approval_gap_rows_v494"]}`.
- Approval ready rows: `{status["approval_ready_rows_v494"]}`.
- Approval blocking rows: `{status["approval_blocking_rows_v494"]}`.
- Approval request rows: `{status["approval_request_rows_v494"]}`.
- Approval scope rows: `{status["approval_scope_rows_v494"]}`.
- Scope approved rows: `{status["scope_approved_rows_v494"]}`.
- Decision option rows: `{status["approval_decision_option_rows_v494"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v494"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v494 is an approval gap audit only. It does not obtain approval, edit Quarto,
apply a patch, render the book, make Paper 4 submission-ready, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V494_PATCH_APPROVAL_GAP_PACKET_START -->"
    end = "<!-- V494_PATCH_APPROVAL_GAP_PACKET_END -->"
    block = f"""
{start}

## Wave v494: Patch Approval Gap Packet

Generated: {status["generated_at_utc"]}

### Objective

v494 audits explicit patch approval gaps without obtaining approval or editing
book sources.

### Results

- Approval gap rows:
  `{status["approval_gap_rows_v494"]}`.
- Approval ready rows:
  `{status["approval_ready_rows_v494"]}`.
- Approval blocking rows:
  `{status["approval_blocking_rows_v494"]}`.
- Approval request rows:
  `{status["approval_request_rows_v494"]}`.
- Approval scope rows:
  `{status["approval_scope_rows_v494"]}`.
- Scope approved rows:
  `{status["scope_approved_rows_v494"]}`.
- Decision option rows:
  `{status["approval_decision_option_rows_v494"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v494"]}`.
- Book sources modified:
  `{status["book_sources_modified_v494"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v494"]}`.

### Interpretation

Explicit patch approval is still absent. The useful next move is a no-patch
release synthesis that captures what the living lab can claim without mutating
the manuscript.

### Claim Impact

- Allowed: approval gap packet, bounded approval request packet and patch scope
  mapping.
- Still prohibited: explicit patch approval claim, Quarto patch
  readiness/application, Quarto/book-reference mutation, submission readiness,
  Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v494 in the living notebook. v495 should synthesize no-patch release
evidence without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v493 = _read_status(PRIOR_CAPTION_SIGNOFF_VERSION)
    if v493["next_artifact_v493"] != "paper4_v494_patch_approval_gap_packet.md":
        raise RuntimeError("v494 expects v493 to route to patch approval gap packet.")

    gaps = _patch_approval_gap_packet()
    requests = _approval_request_packet()
    scope = _approval_scope_matrix()
    options = _approval_decision_options()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v494_patch_approval_gap_packet.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v494_approval_request_packet.csv", requests)
    write_csv(TABLE_DIR / "paper4_v494_approval_scope_matrix.csv", scope)
    write_csv(TABLE_DIR / "paper4_v494_approval_decision_options.csv", options)
    write_csv(TABLE_DIR / "paper4_v494_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v494_claim_matrix_delta.csv", claim_matrix)

    approval_ready_rows = int(gaps["approval_ready_v494"].astype(bool).sum())
    status = {
        "phase": "v494_patch_approval_gap_packet",
        "schema_version": "2026-05-17.494",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_caption_signoff_version_v494": PRIOR_CAPTION_SIGNOFF_VERSION,
        "patch_approval_gap_packet_created_v494": True,
        "approval_gap_rows_v494": len(gaps),
        "approval_ready_rows_v494": approval_ready_rows,
        "approval_blocking_rows_v494": int(gaps["blocks_patch_v494"].astype(bool).sum()),
        "approval_request_rows_v494": len(requests),
        "approval_request_pending_rows_v494": int(
            requests["request_status_v494"].eq("pending").sum()
        ),
        "approval_scope_rows_v494": len(scope),
        "scope_approved_rows_v494": int(
            scope["scope_approved_for_patch_v494"].astype(bool).sum()
        ),
        "approval_decision_option_rows_v494": len(options),
        "patch_allowed_option_rows_v494": int(options["patch_allowed_v494"].astype(bool).sum()),
        "readiness_delta_rows_v494": len(readiness),
        "ready_for_quarto_patch_v494": False,
        "quarto_patch_applied_v494": False,
        "book_sources_modified_v494": False,
        "book_references_modified_v494": False,
        "submission_ready_claim_allowed_v494": False,
        "working_champion_claim_allowed_v494": False,
        "paper1_promotion_allowed_v494": False,
        "paper4_working_champion_changed_v494": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v494": NEXT_ARTIFACT,
        "claim_boundary": (
            "v494 audits patch approval gaps only; approval, patching, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v494 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v494": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
