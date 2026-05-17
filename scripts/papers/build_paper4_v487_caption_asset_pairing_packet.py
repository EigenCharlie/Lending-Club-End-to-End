#!/usr/bin/env python3
"""Build Paper 4 v487 caption-asset pairing packet artifacts."""

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

VERSION = 487
PRIOR_CAPTION_DECISION_VERSION = 486
NEXT_ARTIFACT = "paper4_v488_layout_dry_run_packet.md"
PAIRING_MD = NOTEBOOK.parent / "paper4_v487_caption_asset_pairing_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _caption_asset_pairings() -> pd.DataFrame:
    decisions = pd.read_csv(TABLE_DIR / "paper4_v486_caption_review_decision_matrix.csv")
    asset_map = pd.read_csv(TABLE_DIR / "paper4_v477_visual_asset_manuscript_map.csv")
    merged = decisions.merge(
        asset_map,
        left_on="asset_id_v486",
        right_on="asset_id_v477",
        how="left",
        suffixes=("_caption", "_asset"),
    )
    rows = []
    for _, row in merged.sort_values("insertion_order_v477").iterrows():
        rows.append(
            {
                "asset_id_v487": row["asset_id_v486"],
                "asset_type_v487": row["asset_type_v486"],
                "source_asset_v487": row["source_asset_v477"],
                "manuscript_section_v487": row["manuscript_section_v477"],
                "insertion_order_v487": int(row["insertion_order_v477"]),
                "draft_caption_v487": row["hardened_caption_v486"],
                "caption_accepted_for_draft_v487": bool(row["accepted_for_manuscript_draft_v486"]),
                "asset_caption_pair_ready_v487": True,
                "layout_ready_dry_run_v487": True,
                "caption_final_v487": False,
                "inserted_into_quarto_v487": False,
                "claim_boundary_v487": row["claim_boundary_v486"],
            }
        )
    return pd.DataFrame(rows)


def _section_pairing_summary(pairings: pd.DataFrame) -> pd.DataFrame:
    grouped = pairings.groupby("manuscript_section_v487", sort=True)
    rows = []
    for section, group in grouped:
        rows.append(
            {
                "manuscript_section_v487": section,
                "asset_count_v487": len(group),
                "table_count_v487": int(group["asset_type_v487"].eq("table").sum()),
                "figure_count_v487": int(group["asset_type_v487"].eq("figure").sum()),
                "all_pairs_ready_v487": bool(
                    group["asset_caption_pair_ready_v487"].astype(bool).all()
                ),
                "all_inserted_into_quarto_v487": False,
                "asset_sequence_v487": ";".join(group["asset_id_v487"]),
            }
        )
    return pd.DataFrame(rows)


def _layout_dry_run_seed(pairings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pairings.iterrows():
        rows.append(
            {
                "layout_item_id_v487": f"layout_{row['asset_id_v487']}",
                "asset_id_v487": row["asset_id_v487"],
                "manuscript_section_v487": row["manuscript_section_v487"],
                "layout_order_v487": row["insertion_order_v487"],
                "layout_action_v487": "place asset and accepted draft caption in dry-run layout",
                "ready_for_layout_dry_run_v487": True,
                "ready_for_quarto_patch_v487": False,
                "claim_boundary_v487": row["claim_boundary_v487"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v487": "caption_asset_pairing_packet_created",
                "ready_v487": True,
                "evidence_artifact_v487": "paper4_v487_caption_asset_pairing_matrix.csv",
                "claim_boundary_v487": "pairing packet only",
            },
            {
                "readiness_gate_v487": "section_pairing_summary_created",
                "ready_v487": True,
                "evidence_artifact_v487": "paper4_v487_section_pairing_summary.csv",
                "claim_boundary_v487": "section summary only",
            },
            {
                "readiness_gate_v487": "layout_dry_run_seed_created",
                "ready_v487": True,
                "evidence_artifact_v487": "paper4_v487_layout_dry_run_seed.csv",
                "claim_boundary_v487": "dry-run seed only",
            },
            {
                "readiness_gate_v487": "caption_asset_pairs_ready_for_dry_run",
                "ready_v487": True,
                "evidence_artifact_v487": "paper4_v487_caption_asset_pairing_matrix.csv",
                "claim_boundary_v487": "dry-run readiness only",
            },
            {
                "readiness_gate_v487": "captions_final",
                "ready_v487": False,
                "evidence_artifact_v487": "editorial signoff missing",
                "claim_boundary_v487": "captions remain non-final",
            },
            {
                "readiness_gate_v487": "assets_inserted_into_quarto",
                "ready_v487": False,
                "evidence_artifact_v487": "book sources unchanged",
                "claim_boundary_v487": "no Quarto/book mutation in v487",
            },
            {
                "readiness_gate_v487": "submission_ready",
                "ready_v487": False,
                "evidence_artifact_v487": "future approval, patch, render and venue gates",
                "claim_boundary_v487": "not a submission package",
            },
            {
                "readiness_gate_v487": "paper4_final_promotion_created",
                "ready_v487": False,
                "evidence_artifact_v487": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v487": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v487_caption_asset_pairing_packet_created",
                "allowed": True,
                "artifact": "paper4_v487_caption_asset_pairing_matrix.csv",
                "boundary": "pairing packet only",
            },
            {
                "claim_id": "v487_layout_dry_run_seed_created",
                "allowed": True,
                "artifact": "paper4_v487_layout_dry_run_seed.csv",
                "boundary": "dry-run seed only",
            },
            {
                "claim_id": "v487_pairs_ready_for_layout_dry_run",
                "allowed": True,
                "artifact": "paper4_v487_caption_asset_pairing_matrix.csv",
                "boundary": "dry-run readiness only",
            },
            {
                "claim_id": "v487_assets_inserted_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v487_manuscript_readiness_delta.csv",
                "boundary": "no insertion or final captions",
            },
            {
                "claim_id": "v487_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v487_manuscript_readiness_delta.csv",
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
                "claim": "v487 pairs accepted captions with selected Paper 4 assets.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v487_caption_asset_pairing_matrix.csv"
                ),
                "boundary": "Pairing packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v487 creates a layout dry-run seed.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v487_layout_dry_run_seed.csv"
                ),
                "boundary": "Dry-run seed only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v487 inserts assets or finalizes captions in Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v487_manuscript_readiness_delta.csv"
                ),
                "boundary": "Assets remain uninserted and captions non-final.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v487 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v487_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v487 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v487_manuscript_readiness_delta.csv"
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
                "executable_item": "v487 pairs captions with selected assets.",
                "status": "caption_asset_pairing_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v488 creates layout dry-run packet",
                "last_wave": "v487",
                "execution_result": "caption_asset_pairs_ready_for_layout_dry_run",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v487")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _pairing_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption-Asset Pairing Packet v487

Generated: {status["generated_at_utc"]}

## Result

v487 pairs draft-accepted captions with selected Paper 4 tables and figures. It
creates section summaries and a dry-run layout seed, but does not finalize
captions or insert assets into Quarto.

## Counts

- Caption-asset pair rows: `{status["caption_asset_pair_rows_v487"]}`.
- Section summary rows: `{status["section_summary_rows_v487"]}`.
- Layout seed rows: `{status["layout_seed_rows_v487"]}`.
- Pairs ready for dry-run: `{status["pairs_ready_for_dry_run_v487"]}`.
- Captions final: `{status["captions_final_v487"]}`.
- Assets inserted into Quarto: `{status["assets_inserted_into_quarto_v487"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v487 is a pairing packet only. Layout, Quarto insertion, final captions,
submission readiness and final promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V487_CAPTION_ASSET_PAIRING_PACKET_START -->"
    end = "<!-- V487_CAPTION_ASSET_PAIRING_PACKET_END -->"
    block = f"""
{start}

## Wave v487: Caption-Asset Pairing Packet

Generated: {status["generated_at_utc"]}

### Objective

v487 pairs draft-accepted captions with selected assets and prepares a layout
dry-run seed without editing Quarto.

### Results

- Caption-asset pair rows:
  `{status["caption_asset_pair_rows_v487"]}`.
- Section summary rows:
  `{status["section_summary_rows_v487"]}`.
- Layout seed rows:
  `{status["layout_seed_rows_v487"]}`.
- Pairs ready for dry-run:
  `{status["pairs_ready_for_dry_run_v487"]}`.
- Captions final:
  `{status["captions_final_v487"]}`.
- Assets inserted into Quarto:
  `{status["assets_inserted_into_quarto_v487"]}`.
- Book sources modified:
  `{status["book_sources_modified_v487"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v487"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v487"]}`.

### Interpretation

Captions and assets are now paired for a layout dry-run, not for Quarto mutation.

### Claim Impact

- Allowed: caption-asset pairing, section pairing summary and layout dry-run seed.
- Still prohibited: final captions, asset insertion, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v487 in the living notebook. v488 should create a layout dry-run packet
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v486 = _read_status(PRIOR_CAPTION_DECISION_VERSION)
    if v486["next_artifact_v486"] != "paper4_v487_caption_asset_pairing_packet.md":
        raise RuntimeError("v487 expects v486 to route to caption-asset pairing.")

    pairings = _caption_asset_pairings()
    section_summary = _section_pairing_summary(pairings)
    layout_seed = _layout_dry_run_seed(pairings)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v487_caption_asset_pairing_matrix.csv", pairings)
    write_csv(TABLE_DIR / "paper4_v487_section_pairing_summary.csv", section_summary)
    write_csv(TABLE_DIR / "paper4_v487_layout_dry_run_seed.csv", layout_seed)
    write_csv(TABLE_DIR / "paper4_v487_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v487_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v487_caption_asset_pairing_packet",
        "schema_version": "2026-05-17.487",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_caption_decision_version_v487": PRIOR_CAPTION_DECISION_VERSION,
        "caption_asset_pairing_packet_created_v487": True,
        "caption_asset_pair_rows_v487": len(pairings),
        "section_summary_rows_v487": len(section_summary),
        "layout_seed_rows_v487": len(layout_seed),
        "readiness_delta_rows_v487": len(readiness),
        "pairs_ready_for_dry_run_v487": int(pairings["asset_caption_pair_ready_v487"].sum()),
        "captions_final_v487": False,
        "assets_inserted_into_quarto_v487": False,
        "book_sources_modified_v487": False,
        "book_references_modified_v487": False,
        "submission_ready_claim_allowed_v487": False,
        "working_champion_claim_allowed_v487": False,
        "paper1_promotion_allowed_v487": False,
        "paper4_working_champion_changed_v487": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v487": NEXT_ARTIFACT,
        "claim_boundary": (
            "v487 pairs captions with assets for dry-run only; final captions, insertion, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v487 must not create final Paper 4 promotion.")

    PAIRING_MD.write_text(_pairing_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v487": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
