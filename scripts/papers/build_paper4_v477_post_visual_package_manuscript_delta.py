#!/usr/bin/env python3
"""Build Paper 4 v477 post-visual-package manuscript delta artifacts."""

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

VERSION = 477
PRIOR_CAPTION_PLAN_VERSION = 476
NEXT_ARTIFACT = "paper4_v478_section_text_stub_bundle.md"
DELTA_MD = NOTEBOOK.parent / "paper4_v477_post_visual_package_manuscript_delta.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _visual_asset_map() -> pd.DataFrame:
    captions = pd.read_csv(TABLE_DIR / "paper4_v476_caption_plan.csv")
    insertion = pd.read_csv(TABLE_DIR / "paper4_v476_insertion_plan.csv")
    roles = {
        "T5": "methods_boundary_table",
        "F4": "methods_context_figure",
        "T1": "frontier_results_table",
        "F1": "frontier_context_figure",
        "T2": "source_governance_table",
        "F2": "source_context_figure",
        "T4": "online_proxy_table",
        "F3": "online_context_figure",
        "T3": "dynamic_gap_limitations_table",
        "T6": "ifrs9_gap_limitations_table",
    }
    merged = insertion.merge(
        captions,
        on=["asset_id_v476", "manuscript_section_v476"],
        how="left",
        suffixes=("_insertion", "_caption"),
    )
    rows = []
    for _, row in merged.sort_values("insertion_order_v476").iterrows():
        asset_id = str(row["asset_id_v476"])
        rows.append(
            {
                "asset_id_v477": asset_id,
                "asset_type_v477": row["asset_type_v476"],
                "source_asset_v477": row["source_asset_v476"],
                "insertion_order_v477": int(row["insertion_order_v476"]),
                "manuscript_section_v477": row["manuscript_section_v476"],
                "callout_role_v477": roles[asset_id],
                "draft_caption_v477": row["draft_caption_v476"],
                "must_preserve_caveat_v477": True,
                "caption_final_v477": False,
                "inserted_into_quarto_v477": False,
                "claim_boundary_v477": row["claim_boundary_v476_caption"],
            }
        )
    return pd.DataFrame(rows)


def _visual_section_delta(asset_map: pd.DataFrame) -> pd.DataFrame:
    section_assets = (
        asset_map.groupby("manuscript_section_v477")["asset_id_v477"]
        .apply(lambda values: ";".join(values))
        .to_dict()
    )
    return pd.DataFrame(
        [
            {
                "manuscript_section_v477": "methods_protocol",
                "asset_bundle_v477": section_assets["methods_protocol"],
                "visual_delta_v477": (
                    "Use T5 and F4 to introduce the formal claim-boundary protocol "
                    "before results."
                ),
                "main_text_claim_allowed_v477": True,
                "appendix_only_v477": False,
                "required_caveat_v477": (
                    "historical/oracle-surrogate audit only; no SPO+/DLA theorem"
                ),
                "claim_boundary_v477": "method visual callouts only",
            },
            {
                "manuscript_section_v477": "results_evidence_cvar",
                "asset_bundle_v477": "T1;F1",
                "visual_delta_v477": (
                    "Use the T1/F1 pair to state the local return/CVaR frontier "
                    "claim with legacy visual context."
                ),
                "main_text_claim_allowed_v477": True,
                "appendix_only_v477": False,
                "required_caveat_v477": "local v338-v347-v353 chain only",
                "claim_boundary_v477": "no full-v55 optimality or champion claim",
            },
            {
                "manuscript_section_v477": "results_evidence_governance_online",
                "asset_bundle_v477": "T2;F2;T4;F3",
                "visual_delta_v477": (
                    "Use source-governance and online-proxy assets as bounded "
                    "diagnostics, not deployment evidence."
                ),
                "main_text_claim_allowed_v477": True,
                "appendix_only_v477": False,
                "required_caveat_v477": (
                    "source blocker and internal monitoring proxy language only"
                ),
                "claim_boundary_v477": "no cap approval, external holdout or production claim",
            },
            {
                "manuscript_section_v477": "discussion_limitations",
                "asset_bundle_v477": "T3;T6",
                "visual_delta_v477": (
                    "Use T3 and T6 to make the dynamic replay and IFRS9 contractual "
                    "gaps visible inside limitations."
                ),
                "main_text_claim_allowed_v477": True,
                "appendix_only_v477": False,
                "required_caveat_v477": (
                    "dynamic replay and contractual/accounting blockers remain open"
                ),
                "claim_boundary_v477": "gap evidence only",
            },
            {
                "manuscript_section_v477": "appendix_reproducibility",
                "asset_bundle_v477": ";".join(asset_map["asset_id_v477"]),
                "visual_delta_v477": (
                    "Index the v475-v477 visual-selection, caption and caveat "
                    "provenance for reproducibility."
                ),
                "main_text_claim_allowed_v477": False,
                "appendix_only_v477": True,
                "required_caveat_v477": "appendix index is not a venue-ready supplement",
                "claim_boundary_v477": "provenance index only",
            },
        ]
    )


def _blocker_visual_caveats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v477": "full_v55_global_proof_missing",
                "affected_assets_v477": "T1;F1",
                "manuscript_section_v477": "results_evidence_cvar",
                "required_caption_caveat_v477": (
                    "local frontier evidence only; no full-v55 proof"
                ),
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
            {
                "blocker_id_v477": "grade_a_primary_source_blocker",
                "affected_assets_v477": "T2;F2",
                "manuscript_section_v477": "results_evidence_governance_online",
                "required_caption_caveat_v477": (
                    "diagnostic source blocker only; no cap relaxation"
                ),
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
            {
                "blocker_id_v477": "v353_online_temporal_gate_missing",
                "affected_assets_v477": "T4;F3",
                "manuscript_section_v477": "results_evidence_governance_online",
                "required_caption_caveat_v477": (
                    "internal online proxy only; no external holdout"
                ),
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
            {
                "blocker_id_v477": "v353_dynamic_proxy_trace_missing",
                "affected_assets_v477": "T3",
                "manuscript_section_v477": "discussion_limitations",
                "required_caption_caveat_v477": "v353 dynamic replay trace remains missing",
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
            {
                "blocker_id_v477": "formal_theorem_or_proof_missing",
                "affected_assets_v477": "T5;F4",
                "manuscript_section_v477": "methods_protocol",
                "required_caption_caveat_v477": (
                    "bounded historical audit only; no formal theorem"
                ),
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
            {
                "blocker_id_v477": "contractual_ifrs9_requirements_missing",
                "affected_assets_v477": "T6",
                "manuscript_section_v477": "discussion_limitations",
                "required_caption_caveat_v477": (
                    "requirement audit only; no accounting compliance claim"
                ),
                "blocker_preserved_v477": True,
                "resolved_by_visual_package_v477": False,
            },
        ]
    )


def _caption_revision_queue(asset_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in asset_map.iterrows():
        rows.append(
            {
                "asset_id_v477": row["asset_id_v477"],
                "caption_revision_action_v477": (
                    "tighten caption into venue style while preserving caveat"
                ),
                "caption_caveat_reviewed_v477": True,
                "caption_ready_for_quarto_v477": False,
                "caption_final_v477": False,
                "required_evidence_v477": row["source_asset_v477"],
                "claim_boundary_v477": row["claim_boundary_v477"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v477": "visual_package_mapped_to_sections",
                "ready_v477": True,
                "evidence_artifact_v477": "paper4_v477_visual_section_delta.csv",
                "claim_boundary_v477": "manuscript delta only",
            },
            {
                "readiness_gate_v477": "asset_callout_roles_assigned",
                "ready_v477": True,
                "evidence_artifact_v477": "paper4_v477_visual_asset_manuscript_map.csv",
                "claim_boundary_v477": "callout map only",
            },
            {
                "readiness_gate_v477": "blocker_caveats_preserved",
                "ready_v477": True,
                "evidence_artifact_v477": "paper4_v477_blocker_visual_caveat_map.csv",
                "claim_boundary_v477": "blockers remain open",
            },
            {
                "readiness_gate_v477": "caption_revision_queue_created",
                "ready_v477": True,
                "evidence_artifact_v477": "paper4_v477_caption_revision_queue.csv",
                "claim_boundary_v477": "revision queue only",
            },
            {
                "readiness_gate_v477": "captions_final",
                "ready_v477": False,
                "evidence_artifact_v477": "future_caption_editing",
                "claim_boundary_v477": "no final captions in v477",
            },
            {
                "readiness_gate_v477": "assets_inserted_into_quarto",
                "ready_v477": False,
                "evidence_artifact_v477": "book sources unchanged",
                "claim_boundary_v477": "no Quarto/book promotion in v477",
            },
            {
                "readiness_gate_v477": "submission_ready",
                "ready_v477": False,
                "evidence_artifact_v477": "future venue and manuscript edit",
                "claim_boundary_v477": "not a submission package",
            },
            {
                "readiness_gate_v477": "paper4_final_promotion_created",
                "ready_v477": False,
                "evidence_artifact_v477": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v477": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v477_visual_package_mapped_to_sections",
                "allowed": True,
                "artifact": "paper4_v477_visual_section_delta.csv",
                "boundary": "manuscript delta only",
            },
            {
                "claim_id": "v477_asset_callout_roles_assigned",
                "allowed": True,
                "artifact": "paper4_v477_visual_asset_manuscript_map.csv",
                "boundary": "callout map only",
            },
            {
                "claim_id": "v477_blocker_caveats_preserved",
                "allowed": True,
                "artifact": "paper4_v477_blocker_visual_caveat_map.csv",
                "boundary": "blocker-preserving caveat map only",
            },
            {
                "claim_id": "v477_caption_revision_queue_created",
                "allowed": True,
                "artifact": "paper4_v477_caption_revision_queue.csv",
                "boundary": "revision queue only",
            },
            {
                "claim_id": "v477_assets_inserted_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v477_manuscript_readiness_delta.csv",
                "boundary": "no insertion or final captions",
            },
            {
                "claim_id": "v477_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v477_manuscript_readiness_delta.csv",
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
                "claim": "v477 maps the selected visual package into manuscript sections.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_visual_section_delta.csv"
                ),
                "boundary": "Manuscript delta only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v477 assigns asset callout roles for future manuscript editing.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_visual_asset_manuscript_map.csv"
                ),
                "boundary": "Callout map only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v477 preserves blocker caveats for visual assets.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_blocker_visual_caveat_map.csv"
                ),
                "boundary": "Blockers remain open.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v477 inserts the visual package or finalizes captions in Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_manuscript_readiness_delta.csv"
                ),
                "boundary": "No book source mutation in v477.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v477 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_manuscript_readiness_delta.csv"
                ),
                "boundary": "Venue, caption-finalization and insertion gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v477 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v477_manuscript_readiness_delta.csv"
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
                "executable_item": "v477 maps visual package into manuscript delta.",
                "status": "post_visual_package_manuscript_delta_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v478 drafts section text stubs from visual manuscript delta"
                ),
                "last_wave": "v477",
                "execution_result": (
                    "visual_package_mapped_to_manuscript_without_quarto_edit"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v477")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _delta_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Visual Package Manuscript Delta v477

Generated: {status["generated_at_utc"]}

## Result

v477 maps the selected table/figure package, draft captions and insertion order
into a manuscript delta. It assigns each asset a callout role, preserves blocker
caveats and creates a caption revision queue. It does not insert assets into
Quarto, finalize captions, make Paper 4 submission-ready, or promote Paper 4.

## Counts

- Visual section deltas: `{status["visual_section_delta_rows_v477"]}`.
- Visual asset map rows: `{status["visual_asset_map_rows_v477"]}`.
- Blocker visual caveat rows: `{status["blocker_visual_caveat_rows_v477"]}`.
- Caption revision rows: `{status["caption_revision_rows_v477"]}`.
- Captions final: `{status["captions_final_v477"]}`.
- Assets inserted into Quarto: `{status["assets_inserted_into_quarto_v477"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v477 is a post-visual-package manuscript delta only. Caption finalization,
Quarto insertion, book-reference updates, venue formatting, external validation,
submission readiness, Paper Estrella replacement and final Paper 4 promotion
remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V477_POST_VISUAL_PACKAGE_MANUSCRIPT_DELTA_START -->"
    end = "<!-- V477_POST_VISUAL_PACKAGE_MANUSCRIPT_DELTA_END -->"
    block = f"""
{start}

## Wave v477: Post-Visual Package Manuscript Delta

Generated: {status["generated_at_utc"]}

### Objective

v477 maps the selected visual package back into manuscript sections, callout
roles, blocker caveats and a caption revision queue without editing Quarto.

### Results

- Visual section deltas:
  `{status["visual_section_delta_rows_v477"]}`.
- Visual asset map rows:
  `{status["visual_asset_map_rows_v477"]}`.
- Blocker visual caveat rows:
  `{status["blocker_visual_caveat_rows_v477"]}`.
- Caption revision rows:
  `{status["caption_revision_rows_v477"]}`.
- Captions final:
  `{status["captions_final_v477"]}`.
- Assets inserted into Quarto:
  `{status["assets_inserted_into_quarto_v477"]}`.
- Book sources modified:
  `{status["book_sources_modified_v477"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v477"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v477"]}`.

### Interpretation

The visual package has moved from selection into manuscript editing control: each
asset has a section, role and caveat, and each caption has a future revision
action. The package is still not inserted or final.

### Claim Impact

- Allowed: visual section mapping, asset callout roles, blocker caveat
  preservation and caption revision queue.
- Still prohibited: asset insertion, final captions, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v477 in the living notebook. v478 should draft section text stubs from this
visual manuscript delta without editing book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v476 = _read_status(PRIOR_CAPTION_PLAN_VERSION)
    if v476["next_artifact_v476"] != "paper4_v477_post_visual_package_manuscript_delta.md":
        raise RuntimeError("v477 expects v476 to route to post-visual-package delta.")

    asset_map = _visual_asset_map()
    section_delta = _visual_section_delta(asset_map)
    blocker_caveats = _blocker_visual_caveats()
    caption_queue = _caption_revision_queue(asset_map)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v477_visual_asset_manuscript_map.csv", asset_map)
    write_csv(TABLE_DIR / "paper4_v477_visual_section_delta.csv", section_delta)
    write_csv(TABLE_DIR / "paper4_v477_blocker_visual_caveat_map.csv", blocker_caveats)
    write_csv(TABLE_DIR / "paper4_v477_caption_revision_queue.csv", caption_queue)
    write_csv(TABLE_DIR / "paper4_v477_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v477_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v477_post_visual_package_manuscript_delta",
        "schema_version": "2026-05-17.477",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_caption_plan_version_v477": PRIOR_CAPTION_PLAN_VERSION,
        "post_visual_package_manuscript_delta_created_v477": True,
        "visual_package_mapped_to_manuscript_v477": True,
        "visual_section_delta_rows_v477": len(section_delta),
        "visual_asset_map_rows_v477": len(asset_map),
        "blocker_visual_caveat_rows_v477": len(blocker_caveats),
        "caption_revision_rows_v477": len(caption_queue),
        "readiness_delta_rows_v477": len(readiness),
        "captions_final_v477": False,
        "assets_inserted_into_quarto_v477": False,
        "book_sources_modified_v477": False,
        "book_references_modified_v477": False,
        "submission_ready_claim_allowed_v477": False,
        "working_champion_claim_allowed_v477": False,
        "paper1_promotion_allowed_v477": False,
        "paper4_working_champion_changed_v477": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v477": NEXT_ARTIFACT,
        "claim_boundary": (
            "v477 maps the visual package into manuscript editing control only; "
            "caption finalization, insertion, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v477 must not create final Paper 4 promotion.")

    DELTA_MD.write_text(_delta_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v477": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
