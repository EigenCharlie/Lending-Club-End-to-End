#!/usr/bin/env python3
"""Build Paper 4 v490 layout review decision artifacts."""

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

VERSION = 490
PRIOR_LAYOUT_AUDIT_VERSION = 489
NEXT_ARTIFACT = "paper4_v491_patch_readiness_preflight.md"
DECISION_MD = NOTEBOOK.parent / "paper4_v490_layout_review_decision.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _layout_review_decision_matrix(v489_status: dict[str, Any]) -> pd.DataFrame:
    audit_passed = bool(v489_status["layout_consistency_audit_passed_v489"])
    return pd.DataFrame(
        [
            {
                "decision_id_v490": "accept_layout_audit_for_manual_review",
                "decision_v490": "accept_for_manual_review_only",
                "recommended_v490": audit_passed,
                "patch_allowed_v490": False,
                "decision_boundary_v490": "layout audit passed but review gates remain open",
            },
            {
                "decision_id_v490": "preserve_no_patch_boundary",
                "decision_v490": "keep_book_sources_unchanged",
                "recommended_v490": True,
                "patch_allowed_v490": False,
                "decision_boundary_v490": "no explicit patch approval exists",
            },
            {
                "decision_id_v490": "require_manual_patch_approval",
                "decision_v490": "block_until_approval",
                "recommended_v490": True,
                "patch_allowed_v490": False,
                "decision_boundary_v490": "manual approval missing",
            },
            {
                "decision_id_v490": "require_final_caption_signoff",
                "decision_v490": "block_until_caption_signoff",
                "recommended_v490": True,
                "patch_allowed_v490": False,
                "decision_boundary_v490": "captions remain non-final",
            },
            {
                "decision_id_v490": "apply_quarto_patch_now",
                "decision_v490": "reject_for_v490",
                "recommended_v490": False,
                "patch_allowed_v490": False,
                "decision_boundary_v490": "v490 records decision only",
            },
        ]
    )


def _manual_review_queue() -> pd.DataFrame:
    targets = pd.read_csv(TABLE_DIR / "paper4_v489_target_consistency_matrix.csv")
    rows = []
    for _, row in targets.sort_values(["target_file_v489", "target_block_v489"]).iterrows():
        rows.append(
            {
                "review_item_id_v490": f"review_{len(rows) + 1:02d}",
                "target_file_v490": row["target_file_v489"],
                "target_block_v490": row["target_block_v489"],
                "asset_sequence_v490": row["asset_sequence_v489"],
                "layout_item_count_v490": int(row["layout_item_count_v489"]),
                "manual_review_required_v490": True,
                "review_status_v490": "pending_manual_review",
                "patch_allowed_v490": False,
            }
        )
    return pd.DataFrame(rows)


def _patch_gate_register() -> pd.DataFrame:
    gates = pd.read_csv(TABLE_DIR / "paper4_v488_render_gate_plan.csv")
    return pd.DataFrame(
        [
            {
                "patch_gate_id_v490": row["render_gate_id_v488"],
                "gate_ready_v490": bool(row["gate_ready_v488"]),
                "required_before_patch_v490": bool(row["required_before_patch_v488"]),
                "blocks_patch_v490": not bool(row["gate_ready_v488"]),
                "resolved_by_v490": False,
                "claim_boundary_v490": row["claim_boundary_v488"],
            }
            for _, row in gates.iterrows()
        ]
    )


def _patch_readiness_preflight_seed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "preflight_item_v490": "layout_audit_status",
                "seed_status_v490": "passed",
                "ready_for_v491_v490": True,
                "required_next_action_v490": "carry v489 status into patch preflight",
            },
            {
                "preflight_item_v490": "manual_review_queue",
                "seed_status_v490": "created_pending_review",
                "ready_for_v491_v490": True,
                "required_next_action_v490": "review target surfaces before any patch",
            },
            {
                "preflight_item_v490": "manual_patch_approval",
                "seed_status_v490": "missing",
                "ready_for_v491_v490": False,
                "required_next_action_v490": "obtain explicit approval before mutation",
            },
            {
                "preflight_item_v490": "final_caption_signoff",
                "seed_status_v490": "missing",
                "ready_for_v491_v490": False,
                "required_next_action_v490": "finalize captions before mutation",
            },
            {
                "preflight_item_v490": "post_patch_render_gate",
                "seed_status_v490": "deferred",
                "ready_for_v491_v490": False,
                "required_next_action_v490": "run render only after patch exists",
            },
            {
                "preflight_item_v490": "final_promotion_gate",
                "seed_status_v490": "forbidden",
                "ready_for_v491_v490": False,
                "required_next_action_v490": "keep Paper Estrella boundary active",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v490": "layout_review_decision_created",
                "ready_v490": True,
                "evidence_artifact_v490": "paper4_v490_layout_review_decision_matrix.csv",
                "claim_boundary_v490": "decision matrix only",
            },
            {
                "readiness_gate_v490": "manual_review_queue_created",
                "ready_v490": True,
                "evidence_artifact_v490": "paper4_v490_manual_review_queue.csv",
                "claim_boundary_v490": "pending manual review only",
            },
            {
                "readiness_gate_v490": "patch_gate_register_created",
                "ready_v490": True,
                "evidence_artifact_v490": "paper4_v490_patch_gate_register.csv",
                "claim_boundary_v490": "patch gate register only",
            },
            {
                "readiness_gate_v490": "patch_readiness_preflight_seed_created",
                "ready_v490": True,
                "evidence_artifact_v490": "paper4_v490_patch_readiness_preflight_seed.csv",
                "claim_boundary_v490": "preflight seed only",
            },
            {
                "readiness_gate_v490": "ready_for_quarto_patch",
                "ready_v490": False,
                "evidence_artifact_v490": "manual approval and final captions missing",
                "claim_boundary_v490": "patch remains blocked",
            },
            {
                "readiness_gate_v490": "book_sources_or_references_modified",
                "ready_v490": False,
                "evidence_artifact_v490": "book sources unchanged",
                "claim_boundary_v490": "no Quarto/book mutation in v490",
            },
            {
                "readiness_gate_v490": "submission_ready",
                "ready_v490": False,
                "evidence_artifact_v490": "future approval, patch, render and venue gates",
                "claim_boundary_v490": "not a submission package",
            },
            {
                "readiness_gate_v490": "paper4_final_promotion_created",
                "ready_v490": False,
                "evidence_artifact_v490": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v490": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v490_layout_review_decision_created",
                "allowed": True,
                "artifact": "paper4_v490_layout_review_decision_matrix.csv",
                "boundary": "decision matrix only",
            },
            {
                "claim_id": "v490_manual_review_queue_created",
                "allowed": True,
                "artifact": "paper4_v490_manual_review_queue.csv",
                "boundary": "pending review only",
            },
            {
                "claim_id": "v490_patch_readiness_preflight_seed_created",
                "allowed": True,
                "artifact": "paper4_v490_patch_readiness_preflight_seed.csv",
                "boundary": "preflight seed only",
            },
            {
                "claim_id": "v490_quarto_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v490_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v490_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v490_manuscript_readiness_delta.csv",
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
                "claim": "v490 records a layout review decision for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_layout_review_decision_matrix.csv"
                ),
                "boundary": "Layout review decision only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v490 creates a manual review queue for layout surfaces.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_manual_review_queue.csv"
                ),
                "boundary": "Pending manual review only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v490 seeds a future patch-readiness preflight.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_patch_readiness_preflight_seed.csv"
                ),
                "boundary": "Preflight seed only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v490 makes Paper 4 ready for Quarto patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_manuscript_readiness_delta.csv"
                ),
                "boundary": "Manual approval and final captions are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v490 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_patch_gate_register.csv"
                ),
                "boundary": "Patch is not authorized in v490.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v490 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v490_manuscript_readiness_delta.csv"
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
                "executable_item": "v490 records layout review decision.",
                "status": "layout_review_decision_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v491 preflights patch readiness without mutation",
                "last_wave": "v490",
                "execution_result": "layout_review_decision_recorded_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v490")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Layout Review Decision v490

Generated: {status["generated_at_utc"]}

## Result

v490 accepts the v489 layout audit for manual review only. It creates a review
queue and a patch-readiness preflight seed, but keeps patching blocked because
manual approval, final caption signoff and post-patch render gates remain open.

## Counts

- Decision rows: `{status["layout_review_decision_rows_v490"]}`.
- Manual review queue rows: `{status["manual_review_queue_rows_v490"]}`.
- Patch gate rows: `{status["patch_gate_rows_v490"]}`.
- Open patch gate rows: `{status["open_patch_gate_rows_v490"]}`.
- Preflight seed rows: `{status["patch_readiness_preflight_seed_rows_v490"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v490"]}`.
- Book sources modified: `{status["book_sources_modified_v490"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v490 is a review decision only. It does not edit Quarto, apply a patch, render
the book, make Paper 4 submission-ready, replace Paper Estrella, or promote
Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V490_LAYOUT_REVIEW_DECISION_START -->"
    end = "<!-- V490_LAYOUT_REVIEW_DECISION_END -->"
    block = f"""
{start}

## Wave v490: Layout Review Decision

Generated: {status["generated_at_utc"]}

### Objective

v490 converts the passed v489 layout consistency audit into an explicit manual
review decision without editing book sources.

### Results

- Decision rows:
  `{status["layout_review_decision_rows_v490"]}`.
- Manual review queue rows:
  `{status["manual_review_queue_rows_v490"]}`.
- Patch gate rows:
  `{status["patch_gate_rows_v490"]}`.
- Open patch gate rows:
  `{status["open_patch_gate_rows_v490"]}`.
- Preflight seed rows:
  `{status["patch_readiness_preflight_seed_rows_v490"]}`.
- Layout accepted for manual review:
  `{status["layout_accepted_for_manual_review_v490"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v490"]}`.
- Book sources modified:
  `{status["book_sources_modified_v490"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v490"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v490"]}`.

### Interpretation

The layout can move into manual review, but the patch itself remains blocked.
This preserves the dry-run evidence while creating a concrete v491 preflight.

### Claim Impact

- Allowed: layout review decision, manual review queue and patch-readiness
  preflight seed.
- Still prohibited: Quarto patch readiness/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v490 in the living notebook. v491 should preflight patch readiness without
modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v489 = _read_status(PRIOR_LAYOUT_AUDIT_VERSION)
    if v489["next_artifact_v489"] != "paper4_v490_layout_review_decision.md":
        raise RuntimeError("v490 expects v489 to route to layout review decision.")

    decisions = _layout_review_decision_matrix(v489)
    review_queue = _manual_review_queue()
    patch_gates = _patch_gate_register()
    preflight_seed = _patch_readiness_preflight_seed()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v490_layout_review_decision_matrix.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v490_manual_review_queue.csv", review_queue)
    write_csv(TABLE_DIR / "paper4_v490_patch_gate_register.csv", patch_gates)
    write_csv(TABLE_DIR / "paper4_v490_patch_readiness_preflight_seed.csv", preflight_seed)
    write_csv(TABLE_DIR / "paper4_v490_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v490_claim_matrix_delta.csv", claim_matrix)

    layout_accepted = bool(
        decisions.loc[
            decisions["decision_id_v490"].eq("accept_layout_audit_for_manual_review"),
            "recommended_v490",
        ].iloc[0]
    )
    status = {
        "phase": "v490_layout_review_decision",
        "schema_version": "2026-05-17.490",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_layout_audit_version_v490": PRIOR_LAYOUT_AUDIT_VERSION,
        "layout_review_decision_created_v490": True,
        "layout_audit_passed_v490": bool(v489["layout_consistency_audit_passed_v489"]),
        "layout_accepted_for_manual_review_v490": layout_accepted,
        "layout_review_decision_rows_v490": len(decisions),
        "manual_review_queue_rows_v490": len(review_queue),
        "manual_review_pending_rows_v490": int(
            review_queue["manual_review_required_v490"].astype(bool).sum()
        ),
        "patch_gate_rows_v490": len(patch_gates),
        "open_patch_gate_rows_v490": int(patch_gates["blocks_patch_v490"].astype(bool).sum()),
        "patch_readiness_preflight_seed_rows_v490": len(preflight_seed),
        "preflight_ready_seed_rows_v490": int(
            preflight_seed["ready_for_v491_v490"].astype(bool).sum()
        ),
        "readiness_delta_rows_v490": len(readiness),
        "ready_for_quarto_patch_v490": False,
        "quarto_patch_applied_v490": False,
        "book_sources_modified_v490": False,
        "book_references_modified_v490": False,
        "submission_ready_claim_allowed_v490": False,
        "working_champion_claim_allowed_v490": False,
        "paper1_promotion_allowed_v490": False,
        "paper4_working_champion_changed_v490": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v490": NEXT_ARTIFACT,
        "claim_boundary": (
            "v490 records a layout review decision only; patching, final captions, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v490 must not create final Paper 4 promotion.")

    DECISION_MD.write_text(_decision_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v490": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
