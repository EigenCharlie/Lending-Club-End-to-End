#!/usr/bin/env python3
"""Build Paper 4 v493 caption signoff gap packet artifacts."""

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

VERSION = 493
PRIOR_MANUAL_LAYOUT_REVIEW_VERSION = 492
NEXT_ARTIFACT = "paper4_v494_patch_approval_gap_packet.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v493_caption_signoff_gap_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _caption_signoff_gap_packet() -> pd.DataFrame:
    layout = pd.read_csv(TABLE_DIR / "paper4_v488_layout_dry_run_packet.csv")
    rows = []
    for _, row in layout.sort_values("layout_order_v488").iterrows():
        caption_text = str(row["caption_text_v488"])
        caption_final = bool(row["caption_final_v488"])
        rows.append(
            {
                "caption_gap_id_v493": f"caption_{row['asset_id_v488']}",
                "asset_id_v493": row["asset_id_v488"],
                "asset_type_v493": row["asset_type_v488"],
                "target_block_v493": row["target_block_v488"],
                "layout_order_v493": int(row["layout_order_v488"]),
                "draft_caption_exists_v493": bool(caption_text.strip()),
                "caption_word_count_v493": len(caption_text.split()),
                "caption_final_v493": caption_final,
                "signoff_status_v493": "missing_final_caption_signoff",
                "required_signoff_action_v493": "manual editorial caption approval",
                "blocks_patch_v493": not caption_final,
                "claim_boundary_v493": row["claim_boundary_v488"],
            }
        )
    return pd.DataFrame(rows)


def _caption_gate_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_block, group in gaps.groupby("target_block_v493", sort=True):
        final_count = int(group["caption_final_v493"].astype(bool).sum())
        asset_count = len(group)
        rows.append(
            {
                "target_block_v493": target_block,
                "caption_asset_count_v493": asset_count,
                "draft_caption_count_v493": int(group["draft_caption_exists_v493"].sum()),
                "final_caption_count_v493": final_count,
                "pending_caption_count_v493": asset_count - final_count,
                "all_captions_final_v493": final_count == asset_count,
                "blocks_patch_v493": final_count != asset_count,
            }
        )
    return pd.DataFrame(rows)


def _caption_claim_safety_matrix(gaps: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id_v493": row["asset_id_v493"],
                "target_block_v493": row["target_block_v493"],
                "draft_caption_exists_v493": bool(row["draft_caption_exists_v493"]),
                "caption_final_v493": bool(row["caption_final_v493"]),
                "claim_boundary_v493": row["claim_boundary_v493"],
                "overclaim_review_required_v493": True,
                "draft_use_allowed_v493": True,
                "final_caption_claim_allowed_v493": False,
            }
            for _, row in gaps.iterrows()
        ]
    )


def _signoff_action_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signoff_action_id_v493": "review_all_draft_captions",
                "action_required_v493": True,
                "action_complete_v493": False,
                "blocks_patch_v493": True,
                "action_boundary_v493": "manual caption review required",
            },
            {
                "signoff_action_id_v493": "verify_claim_boundaries_per_caption",
                "action_required_v493": True,
                "action_complete_v493": False,
                "blocks_patch_v493": True,
                "action_boundary_v493": "avoid overclaiming in final captions",
            },
            {
                "signoff_action_id_v493": "capture_final_caption_signoff",
                "action_required_v493": True,
                "action_complete_v493": False,
                "blocks_patch_v493": True,
                "action_boundary_v493": "final signoff missing",
            },
            {
                "signoff_action_id_v493": "sync_signoff_with_patch_approval",
                "action_required_v493": True,
                "action_complete_v493": False,
                "blocks_patch_v493": True,
                "action_boundary_v493": "patch approval still missing",
            },
            {
                "signoff_action_id_v493": "apply_quarto_patch_now",
                "action_required_v493": False,
                "action_complete_v493": False,
                "blocks_patch_v493": True,
                "action_boundary_v493": "not authorized in v493",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v493": "caption_signoff_gap_packet_created",
                "ready_v493": True,
                "evidence_artifact_v493": "paper4_v493_caption_signoff_gap_packet.csv",
                "claim_boundary_v493": "caption gap packet only",
            },
            {
                "readiness_gate_v493": "caption_gate_summary_created",
                "ready_v493": True,
                "evidence_artifact_v493": "paper4_v493_caption_gate_summary.csv",
                "claim_boundary_v493": "caption gate summary only",
            },
            {
                "readiness_gate_v493": "caption_claim_safety_matrix_created",
                "ready_v493": True,
                "evidence_artifact_v493": "paper4_v493_caption_claim_safety_matrix.csv",
                "claim_boundary_v493": "claim safety matrix only",
            },
            {
                "readiness_gate_v493": "signoff_action_register_created",
                "ready_v493": True,
                "evidence_artifact_v493": "paper4_v493_signoff_action_register.csv",
                "claim_boundary_v493": "action register only",
            },
            {
                "readiness_gate_v493": "ready_for_quarto_patch",
                "ready_v493": False,
                "evidence_artifact_v493": "caption signoff and patch approval missing",
                "claim_boundary_v493": "patch remains blocked",
            },
            {
                "readiness_gate_v493": "book_sources_or_references_modified",
                "ready_v493": False,
                "evidence_artifact_v493": "book sources unchanged",
                "claim_boundary_v493": "no Quarto/book mutation in v493",
            },
            {
                "readiness_gate_v493": "submission_ready",
                "ready_v493": False,
                "evidence_artifact_v493": "future approval, patch, render and venue gates",
                "claim_boundary_v493": "not a submission package",
            },
            {
                "readiness_gate_v493": "paper4_final_promotion_created",
                "ready_v493": False,
                "evidence_artifact_v493": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v493": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v493_caption_signoff_gap_packet_created",
                "allowed": True,
                "artifact": "paper4_v493_caption_signoff_gap_packet.csv",
                "boundary": "caption signoff gap packet only",
            },
            {
                "claim_id": "v493_caption_claim_safety_matrix_created",
                "allowed": True,
                "artifact": "paper4_v493_caption_claim_safety_matrix.csv",
                "boundary": "claim safety matrix only",
            },
            {
                "claim_id": "v493_open_caption_signoff_actions_identified",
                "allowed": True,
                "artifact": "paper4_v493_signoff_action_register.csv",
                "boundary": "open action register only",
            },
            {
                "claim_id": "v493_final_captions_or_patch_ready",
                "allowed": False,
                "artifact": "paper4_v493_manuscript_readiness_delta.csv",
                "boundary": "captions remain non-final and patch remains blocked",
            },
            {
                "claim_id": "v493_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v493_manuscript_readiness_delta.csv",
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
                "claim": "v493 audits caption signoff gaps for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_caption_signoff_gap_packet.csv"
                ),
                "boundary": "Caption signoff gap packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v493 maps caption claim-safety review needs.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_caption_claim_safety_matrix.csv"
                ),
                "boundary": "Caption claim-safety matrix only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v493 records open caption signoff actions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_signoff_action_register.csv"
                ),
                "boundary": "Open signoff action register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v493 finalizes captions or makes Paper 4 ready for patching.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_manuscript_readiness_delta.csv"
                ),
                "boundary": "Captions remain non-final and patch approval is missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v493 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_signoff_action_register.csv"
                ),
                "boundary": "Patch is not authorized in v493.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v493 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v493_manuscript_readiness_delta.csv"
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
                "executable_item": "v493 audits caption signoff gaps.",
                "status": "caption_signoff_gap_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v494 audits explicit patch approval gap without mutation",
                "last_wave": "v493",
                "execution_result": "caption_signoff_gap_packet_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v493")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption Signoff Gap Packet v493

Generated: {status["generated_at_utc"]}

## Result

v493 audits caption signoff gaps for the ten draft assets. Draft captions exist,
but none are final; therefore caption signoff remains a live blocker before any
Quarto patch.

## Counts

- Caption gap rows: `{status["caption_gap_rows_v493"]}`.
- Draft caption rows: `{status["draft_caption_rows_v493"]}`.
- Caption final rows: `{status["caption_final_rows_v493"]}`.
- Caption pending rows: `{status["caption_pending_rows_v493"]}`.
- Target block summary rows: `{status["target_block_summary_rows_v493"]}`.
- Claim safety rows: `{status["claim_safety_rows_v493"]}`.
- Open signoff action rows: `{status["open_signoff_action_rows_v493"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v493"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v493 is a caption signoff gap audit only. It does not finalize captions, edit
Quarto, apply a patch, render the book, make Paper 4 submission-ready, replace
Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V493_CAPTION_SIGNOFF_GAP_PACKET_START -->"
    end = "<!-- V493_CAPTION_SIGNOFF_GAP_PACKET_END -->"
    block = f"""
{start}

## Wave v493: Caption Signoff Gap Packet

Generated: {status["generated_at_utc"]}

### Objective

v493 audits caption signoff gaps for the ten draft assets without finalizing
captions or editing book sources.

### Results

- Caption gap rows:
  `{status["caption_gap_rows_v493"]}`.
- Draft caption rows:
  `{status["draft_caption_rows_v493"]}`.
- Caption final rows:
  `{status["caption_final_rows_v493"]}`.
- Caption pending rows:
  `{status["caption_pending_rows_v493"]}`.
- Target block summary rows:
  `{status["target_block_summary_rows_v493"]}`.
- Claim safety rows:
  `{status["claim_safety_rows_v493"]}`.
- Open signoff action rows:
  `{status["open_signoff_action_rows_v493"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v493"]}`.
- Book sources modified:
  `{status["book_sources_modified_v493"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v493"]}`.

### Interpretation

All ten draft captions exist, but none are final. Caption signoff is therefore a
real blocker, not a bookkeeping detail.

### Claim Impact

- Allowed: caption signoff gap audit, caption claim-safety matrix and open
  signoff action register.
- Still prohibited: final caption claim, Quarto patch readiness/application,
  Quarto/book-reference mutation, submission readiness, Paper Estrella
  replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v493 in the living notebook. v494 should audit the explicit patch approval
gap without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v492 = _read_status(PRIOR_MANUAL_LAYOUT_REVIEW_VERSION)
    if v492["next_artifact_v492"] != "paper4_v493_caption_signoff_gap_packet.md":
        raise RuntimeError("v493 expects v492 to route to caption signoff gap packet.")

    gaps = _caption_signoff_gap_packet()
    summary = _caption_gate_summary(gaps)
    claim_safety = _caption_claim_safety_matrix(gaps)
    actions = _signoff_action_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v493_caption_signoff_gap_packet.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v493_caption_gate_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v493_caption_claim_safety_matrix.csv", claim_safety)
    write_csv(TABLE_DIR / "paper4_v493_signoff_action_register.csv", actions)
    write_csv(TABLE_DIR / "paper4_v493_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v493_claim_matrix_delta.csv", claim_matrix)

    caption_final_rows = int(gaps["caption_final_v493"].astype(bool).sum())
    open_action_rows = int(actions["blocks_patch_v493"].astype(bool).sum())
    status = {
        "phase": "v493_caption_signoff_gap_packet",
        "schema_version": "2026-05-17.493",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_layout_review_version_v493": PRIOR_MANUAL_LAYOUT_REVIEW_VERSION,
        "caption_signoff_gap_packet_created_v493": True,
        "caption_gap_rows_v493": len(gaps),
        "draft_caption_rows_v493": int(gaps["draft_caption_exists_v493"].astype(bool).sum()),
        "caption_final_rows_v493": caption_final_rows,
        "caption_pending_rows_v493": len(gaps) - caption_final_rows,
        "target_block_summary_rows_v493": len(summary),
        "target_blocks_all_caption_final_v493": int(
            summary["all_captions_final_v493"].astype(bool).sum()
        ),
        "claim_safety_rows_v493": len(claim_safety),
        "overclaim_review_required_rows_v493": int(
            claim_safety["overclaim_review_required_v493"].astype(bool).sum()
        ),
        "signoff_action_rows_v493": len(actions),
        "open_signoff_action_rows_v493": open_action_rows,
        "readiness_delta_rows_v493": len(readiness),
        "ready_for_quarto_patch_v493": False,
        "quarto_patch_applied_v493": False,
        "book_sources_modified_v493": False,
        "book_references_modified_v493": False,
        "submission_ready_claim_allowed_v493": False,
        "working_champion_claim_allowed_v493": False,
        "paper1_promotion_allowed_v493": False,
        "paper4_working_champion_changed_v493": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v493": NEXT_ARTIFACT,
        "claim_boundary": (
            "v493 audits caption signoff gaps only; captions remain non-final, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v493 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v493": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
