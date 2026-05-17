#!/usr/bin/env python3
"""Build Paper 4 v476 caption and insertion plan artifacts."""

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

VERSION = 476
PRIOR_SELECTION_VERSION = 475
NEXT_ARTIFACT = "paper4_v477_post_visual_package_manuscript_delta.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v476_caption_and_insertion_plan.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _caption_plan() -> pd.DataFrame:
    tables = pd.read_csv(TABLE_DIR / "paper4_v475_primary_table_selection.csv")
    figures = pd.read_csv(TABLE_DIR / "paper4_v475_primary_figure_selection.csv")
    table_captions = {
        "T1": (
            "Local return/CVaR frontier chain for v338, v347 and v353. "
            "The table supports only a local frontier claim; full-v55 proof and "
            "working-champion language remain blocked."
        ),
        "T2": (
            "Tight source-governance ranking after the domain refresh. "
            "Grade A is documented as the primary source blocker without "
            "authorizing cap relaxation."
        ),
        "T3": (
            "Dynamic replay gap for the current local frontier. v338 remains the "
            "dynamic proxy anchor, while v353 lacks a dynamic replay trace."
        ),
        "T4": (
            "Internal online conformal monitoring proxy summary. The evidence is "
            "internal and replay-based; external holdout and production monitoring "
            "claims remain blocked."
        ),
        "T5": (
            "Formal SPO-DLA claim boundary matrix. Bounded historical audit "
            "language is permitted, while SPO+/DLA theorem and CRC guarantee claims "
            "remain blocked."
        ),
        "T6": (
            "Contractual IFRS9 requirement-gap audit. The table supports proxy and "
            "gap language only; contractual/accounting compliance remains blocked."
        ),
    }
    figure_captions = {
        "F1": (
            "Return, ECL and tail-risk frontier context. The figure is retained as "
            "visual context and must be read with the current v467 local frontier "
            "table."
        ),
        "F2": (
            "Worst-source governance coverage context. The figure motivates the "
            "source-governance lane without implying legal fairness certification or "
            "cap approval."
        ),
        "F3": (
            "Online conformal method-search context. The figure supports internal "
            "monitoring proxy discussion, not production monitoring."
        ),
        "F4": (
            "Regret-auditability frontier context for historical SPO-DLA positioning. "
            "The figure does not establish a formal theorem."
        ),
    }
    rows = []
    for _, row in tables.iterrows():
        slot = str(row["table_slot_v475"])
        rows.append(
            {
                "asset_id_v476": slot,
                "asset_type_v476": "table",
                "source_asset_v476": row["source_artifact_v475"],
                "manuscript_section_v476": row["manuscript_section_v475"],
                "draft_caption_v476": table_captions[slot],
                "caption_final_v476": False,
                "source_claim_or_lane_v476": row["supports_claim_id_v475"],
                "required_caveat_v476": row["required_caveat_v475"],
                "claim_boundary_v476": row["claim_boundary_v475"],
            }
        )
    for _, row in figures.iterrows():
        slot = str(row["figure_slot_v475"])
        rows.append(
            {
                "asset_id_v476": slot,
                "asset_type_v476": "figure",
                "source_asset_v476": row["source_figure_v475"],
                "manuscript_section_v476": row["manuscript_section_v475"],
                "draft_caption_v476": figure_captions[slot],
                "caption_final_v476": False,
                "source_claim_or_lane_v476": row["supports_domain_lane_v475"],
                "required_caveat_v476": row["required_caveat_v475"],
                "claim_boundary_v476": row["claim_boundary_v475"],
            }
        )
    return pd.DataFrame(rows)


def _insertion_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "insertion_order_v476": 1,
                "asset_id_v476": "T5",
                "manuscript_section_v476": "methods_protocol",
                "insertion_reason_v476": "establish formal claim boundaries before results",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "insertion plan only",
            },
            {
                "insertion_order_v476": 2,
                "asset_id_v476": "F4",
                "manuscript_section_v476": "methods_protocol",
                "insertion_reason_v476": "visualize regret-auditability positioning",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "historical context only",
            },
            {
                "insertion_order_v476": 3,
                "asset_id_v476": "T1",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "lead results with local return/CVaR frontier",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "local frontier only",
            },
            {
                "insertion_order_v476": 4,
                "asset_id_v476": "F1",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "provide tail-risk visual context",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "visual context only",
            },
            {
                "insertion_order_v476": 5,
                "asset_id_v476": "T2",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "show grade-A source blocker evidence",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "source diagnostic only",
            },
            {
                "insertion_order_v476": 6,
                "asset_id_v476": "F2",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "provide source-governance visual context",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "no cap approval claim",
            },
            {
                "insertion_order_v476": 7,
                "asset_id_v476": "T4",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "summarize internal online proxy gates",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "internal proxy only",
            },
            {
                "insertion_order_v476": 8,
                "asset_id_v476": "F3",
                "manuscript_section_v476": "results_evidence",
                "insertion_reason_v476": "visualize online method search context",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "no production monitoring claim",
            },
            {
                "insertion_order_v476": 9,
                "asset_id_v476": "T3",
                "manuscript_section_v476": "discussion_limitations",
                "insertion_reason_v476": "make v353 dynamic replay gap explicit",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "no live dynamic claim",
            },
            {
                "insertion_order_v476": 10,
                "asset_id_v476": "T6",
                "manuscript_section_v476": "discussion_limitations",
                "insertion_reason_v476": "make IFRS9 contractual gaps explicit",
                "inserted_into_quarto_v476": False,
                "claim_boundary_v476": "no contractual IFRS9 claim",
            },
        ]
    )


def _caveat_matrix(captions: pd.DataFrame) -> pd.DataFrame:
    return captions[
        [
            "asset_id_v476",
            "asset_type_v476",
            "source_asset_v476",
            "required_caveat_v476",
            "claim_boundary_v476",
        ]
    ].assign(
        caveat_required_v476=True,
        final_caption_or_insertion_allowed_v476=False,
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v476": "draft_captions_created",
                "ready_v476": True,
                "evidence_artifact_v476": "paper4_v476_caption_plan.csv",
                "claim_boundary_v476": "draft captions only",
            },
            {
                "readiness_gate_v476": "insertion_plan_created",
                "ready_v476": True,
                "evidence_artifact_v476": "paper4_v476_insertion_plan.csv",
                "claim_boundary_v476": "plan only; no Quarto edit",
            },
            {
                "readiness_gate_v476": "asset_caveats_mapped",
                "ready_v476": True,
                "evidence_artifact_v476": "paper4_v476_asset_caveat_matrix.csv",
                "claim_boundary_v476": "caveat mapping only",
            },
            {
                "readiness_gate_v476": "captions_final",
                "ready_v476": False,
                "evidence_artifact_v476": "future_caption_editing",
                "claim_boundary_v476": "draft captions not final",
            },
            {
                "readiness_gate_v476": "assets_inserted_into_quarto",
                "ready_v476": False,
                "evidence_artifact_v476": "book sources unchanged",
                "claim_boundary_v476": "no Quarto/book promotion in v476",
            },
            {
                "readiness_gate_v476": "submission_ready",
                "ready_v476": False,
                "evidence_artifact_v476": "future venue and manuscript edit",
                "claim_boundary_v476": "selection and captions are not submission package",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v476_draft_captions_created",
                "allowed": True,
                "artifact": "paper4_v476_caption_plan.csv",
                "boundary": "draft captions only",
            },
            {
                "claim_id": "v476_insertion_plan_created",
                "allowed": True,
                "artifact": "paper4_v476_insertion_plan.csv",
                "boundary": "plan only; no Quarto edit",
            },
            {
                "claim_id": "v476_asset_caveats_mapped",
                "allowed": True,
                "artifact": "paper4_v476_asset_caveat_matrix.csv",
                "boundary": "caveat map only",
            },
            {
                "claim_id": "v476_assets_inserted_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v476_manuscript_readiness_delta.csv",
                "boundary": "no insertion or final captions",
            },
            {
                "claim_id": "v476_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v476_manuscript_readiness_delta.csv",
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
                "claim": "v476 creates draft captions for selected Paper 4 assets.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v476_caption_plan.csv"
                ),
                "boundary": "Draft captions only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v476 creates an insertion plan for selected Paper 4 assets.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v476_insertion_plan.csv"
                ),
                "boundary": "Plan only; no Quarto edit.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v476 inserts assets or finalizes captions in Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v476_manuscript_readiness_delta.csv"
                ),
                "boundary": "No book source mutation in v476.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v476 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v476_manuscript_readiness_delta.csv"
                ),
                "boundary": "Venue and manuscript edit gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v476 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v476_manuscript_readiness_delta.csv"
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
                "executable_item": "v476 creates captions and insertion plan.",
                "status": "caption_and_insertion_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v477 maps visual package into manuscript delta",
                "last_wave": "v476",
                "execution_result": "draft_captions_insertion_plan_without_quarto_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v476")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption and Insertion Plan v476

Generated: {status["generated_at_utc"]}

## Result

v476 drafts captions and an insertion order for the primary table/figure package
selected in v475. It keeps every caption provisional and records that no asset
has been inserted into Quarto.

## Counts

- Caption rows: `{status["caption_rows_v476"]}`.
- Insertion rows: `{status["insertion_rows_v476"]}`.
- Caveat rows: `{status["asset_caveat_rows_v476"]}`.
- Captions final: `{status["captions_final_v476"]}`.
- Assets inserted into Quarto: `{status["assets_inserted_into_quarto_v476"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v476 is a draft caption and insertion plan only. It does not edit Quarto,
finalize captions, insert tables or figures, make Paper 4 submission-ready,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V476_CAPTION_AND_INSERTION_PLAN_START -->"
    end = "<!-- V476_CAPTION_AND_INSERTION_PLAN_END -->"
    block = f"""
{start}

## Wave v476: Caption and Insertion Plan

Generated: {status["generated_at_utc"]}

### Objective

v476 drafts captions and an insertion plan for the v475 table/figure package
without editing Quarto sources.

### Results

- Caption rows:
  `{status["caption_rows_v476"]}`.
- Insertion rows:
  `{status["insertion_rows_v476"]}`.
- Caveat rows:
  `{status["asset_caveat_rows_v476"]}`.
- Captions final:
  `{status["captions_final_v476"]}`.
- Assets inserted into Quarto:
  `{status["assets_inserted_into_quarto_v476"]}`.
- Book sources modified:
  `{status["book_sources_modified_v476"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v476"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v476"]}`.

### Interpretation

The visual package is now editorially usable: every selected asset has a draft
caption, insertion order and caveat. It is still not inserted or final.

### Claim Impact

- Allowed: draft captions, insertion order and caveat map.
- Still prohibited: asset insertion, final captions, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v476 in the living notebook. v477 should map the visual package back into
the manuscript delta without editing book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v475 = _read_status(PRIOR_SELECTION_VERSION)
    if v475["next_artifact_v475"] != "paper4_v476_caption_and_insertion_plan.md":
        raise RuntimeError("v476 expects v475 to route to caption and insertion plan.")

    captions = _caption_plan()
    insertion = _insertion_plan()
    caveats = _caveat_matrix(captions)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v476_caption_plan.csv", captions)
    write_csv(TABLE_DIR / "paper4_v476_insertion_plan.csv", insertion)
    write_csv(TABLE_DIR / "paper4_v476_asset_caveat_matrix.csv", caveats)
    write_csv(TABLE_DIR / "paper4_v476_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v476_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v476_caption_and_insertion_plan",
        "schema_version": "2026-05-17.476",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_table_figure_selection_version_v476": PRIOR_SELECTION_VERSION,
        "caption_and_insertion_plan_created_v476": True,
        "caption_rows_v476": len(captions),
        "insertion_rows_v476": len(insertion),
        "asset_caveat_rows_v476": len(caveats),
        "captions_final_v476": False,
        "assets_inserted_into_quarto_v476": False,
        "book_sources_modified_v476": False,
        "book_references_modified_v476": False,
        "submission_ready_claim_allowed_v476": False,
        "working_champion_claim_allowed_v476": False,
        "paper1_promotion_allowed_v476": False,
        "paper4_working_champion_changed_v476": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v476": NEXT_ARTIFACT,
        "claim_boundary": (
            "v476 creates draft captions and insertion plan only; asset insertion, "
            "submission and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v476 must not create final Paper 4 promotion.")

    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v476": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
