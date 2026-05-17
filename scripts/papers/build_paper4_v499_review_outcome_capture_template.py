#!/usr/bin/env python3
"""Build Paper 4 v499 review outcome capture template artifacts."""

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

VERSION = 499
PRIOR_COMPLETION_GAP_VERSION = 498
NEXT_ARTIFACT = "paper4_v500_review_outcome_template_consistency_audit.md"
TEMPLATE_MD = NOTEBOOK.parent / "paper4_v499_review_outcome_capture_template.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _review_outcome_capture_template() -> pd.DataFrame:
    layout = pd.read_csv(TABLE_DIR / "paper4_v498_layout_completion_gap_matrix.csv")
    captions = pd.read_csv(TABLE_DIR / "paper4_v498_caption_completion_gap_matrix.csv")
    rows: list[dict[str, Any]] = []
    for _, row in layout.iterrows():
        rows.append(
            {
                "outcome_template_id_v499": (
                    f"layout_outcome_{row['layout_completion_gap_id_v498']}"
                ),
                "review_domain_v499": "layout_surface",
                "source_review_id_v499": row["layout_completion_gap_id_v498"],
                "asset_id_v499": row["asset_sequence_v498"],
                "target_file_v499": row["target_file_v498"],
                "target_block_v499": row["target_block_v498"],
                "review_item_count_v499": int(row["layout_item_count_v498"]),
                "required_outcome_decision_v499": "accept_revise_reject_or_defer",
                "outcome_status_v499": "not_captured",
                "review_completed_v499": False,
                "accepted_for_patch_v499": False,
                "accepted_for_final_caption_v499": False,
                "patch_allowed_v499": False,
                "claim_boundary_v499": "layout outcome capture template only",
            }
        )
    for _, row in captions.iterrows():
        rows.append(
            {
                "outcome_template_id_v499": (
                    f"caption_outcome_{row['caption_completion_gap_id_v498']}"
                ),
                "review_domain_v499": "caption_claim_safety",
                "source_review_id_v499": row["caption_completion_gap_id_v498"],
                "asset_id_v499": row["asset_id_v498"],
                "target_file_v499": "pending_manual_layout_mapping",
                "target_block_v499": row["target_block_v498"],
                "review_item_count_v499": 1,
                "required_outcome_decision_v499": "accept_revise_reject_or_defer",
                "outcome_status_v499": "not_captured",
                "review_completed_v499": False,
                "accepted_for_patch_v499": False,
                "accepted_for_final_caption_v499": False,
                "patch_allowed_v499": False,
                "claim_boundary_v499": "caption safety outcome capture template only",
            }
        )
    return pd.DataFrame(rows)


def _outcome_field_dictionary() -> pd.DataFrame:
    rows = [
        {
            "capture_field_v499": "reviewer_id",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "stable reviewer identifier",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "review_timestamp_utc",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "ISO-8601 UTC timestamp",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "outcome_decision",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "accept|revise|reject|defer",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "revision_required",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "true|false",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "claim_boundary_ok",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "true|false",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "caption_final_allowed",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "true|false",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "patch_scope_ok",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "true|false",
            "claim_boundary_v499": "template field only",
        },
        {
            "capture_field_v499": "reviewer_notes",
            "required_capture_field_v499": True,
            "field_captured_v499": False,
            "allowed_values_v499": "free text with evidence references",
            "claim_boundary_v499": "template field only",
        },
    ]
    return pd.DataFrame(rows)


def _capture_control_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "control_id_v499": "no_outcomes_captured",
                "control_active_v499": True,
                "control_result_v499": "outcome rows remain not_captured",
                "blocks_patch_v499": True,
            },
            {
                "control_id_v499": "no_caption_finalized",
                "control_active_v499": True,
                "control_result_v499": "caption finalization remains false",
                "blocks_patch_v499": True,
            },
            {
                "control_id_v499": "no_patch_approval",
                "control_active_v499": True,
                "control_result_v499": "patch approval remains absent",
                "blocks_patch_v499": True,
            },
            {
                "control_id_v499": "no_book_mutation",
                "control_active_v499": True,
                "control_result_v499": "book sources and references remain unchanged",
                "blocks_patch_v499": True,
            },
            {
                "control_id_v499": "no_render_submission",
                "control_active_v499": True,
                "control_result_v499": "no submission render is authorized",
                "blocks_patch_v499": True,
            },
            {
                "control_id_v499": "no_final_promotion",
                "control_active_v499": True,
                "control_result_v499": "final promotion artifact remains absent",
                "blocks_patch_v499": True,
            },
        ]
    )


def _template_readiness_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "template_readiness_item_v499": "layout_outcome_rows_ready",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_review_outcome_capture_template.csv",
                "claim_boundary_v499": "layout capture rows only",
            },
            {
                "template_readiness_item_v499": "caption_outcome_rows_ready",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_review_outcome_capture_template.csv",
                "claim_boundary_v499": "caption capture rows only",
            },
            {
                "template_readiness_item_v499": "required_fields_declared",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_outcome_field_dictionary.csv",
                "claim_boundary_v499": "field declaration only",
            },
            {
                "template_readiness_item_v499": "capture_controls_active",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_capture_control_register.csv",
                "claim_boundary_v499": "control register only",
            },
            {
                "template_readiness_item_v499": "review_outcomes_captured",
                "ready_v499": False,
                "evidence_artifact_v499": "paper4_v499_review_outcome_capture_template.csv",
                "claim_boundary_v499": "no outcomes captured in v499",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v499": "review_outcome_capture_template_created",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_review_outcome_capture_template.csv",
                "claim_boundary_v499": "capture template only",
            },
            {
                "readiness_gate_v499": "outcome_field_dictionary_created",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_outcome_field_dictionary.csv",
                "claim_boundary_v499": "required field dictionary only",
            },
            {
                "readiness_gate_v499": "capture_control_register_created",
                "ready_v499": True,
                "evidence_artifact_v499": "paper4_v499_capture_control_register.csv",
                "claim_boundary_v499": "capture control register only",
            },
            {
                "readiness_gate_v499": "review_outcomes_captured",
                "ready_v499": False,
                "evidence_artifact_v499": "all outcome statuses are not_captured",
                "claim_boundary_v499": "no review outcomes captured",
            },
            {
                "readiness_gate_v499": "ready_for_quarto_patch",
                "ready_v499": False,
                "evidence_artifact_v499": "review outcomes and approval remain absent",
                "claim_boundary_v499": "patch remains blocked",
            },
            {
                "readiness_gate_v499": "book_sources_or_references_modified",
                "ready_v499": False,
                "evidence_artifact_v499": "book sources unchanged",
                "claim_boundary_v499": "no Quarto/book mutation in v499",
            },
            {
                "readiness_gate_v499": "submission_ready",
                "ready_v499": False,
                "evidence_artifact_v499": "future outcome, signoff, patch and render gates",
                "claim_boundary_v499": "not a submission package",
            },
            {
                "readiness_gate_v499": "paper4_final_promotion_created",
                "ready_v499": False,
                "evidence_artifact_v499": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v499": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v499_review_outcome_capture_template_created",
                "allowed": True,
                "artifact": "paper4_v499_review_outcome_capture_template.csv",
                "boundary": "review outcome capture template only",
            },
            {
                "claim_id": "v499_required_review_outcome_fields_declared",
                "allowed": True,
                "artifact": "paper4_v499_outcome_field_dictionary.csv",
                "boundary": "field dictionary only",
            },
            {
                "claim_id": "v499_capture_controls_preserved",
                "allowed": True,
                "artifact": "paper4_v499_capture_control_register.csv",
                "boundary": "controls preserve no-outcome and no-patch state",
            },
            {
                "claim_id": "v499_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v499_review_outcome_capture_template.csv",
                "boundary": "all outcomes remain not_captured",
            },
            {
                "claim_id": "v499_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v499_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v499_final_promotion",
                "allowed": False,
                "artifact": "paper4_v499_manuscript_readiness_delta.csv",
                "boundary": "no final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v499 creates a Paper 4 review outcome capture template.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_review_outcome_capture_template.csv"
                ),
                "boundary": "Review outcome capture template only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v499 defines required review outcome fields.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_outcome_field_dictionary.csv"
                ),
                "boundary": "Required field dictionary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v499 preserves capture controls without recording outcomes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_capture_control_register.csv"
                ),
                "boundary": "No-outcome and no-patch controls only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v499 captures completed review outcomes or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_review_outcome_capture_template.csv"
                ),
                "boundary": "All outcome rows remain not_captured.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v499 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v499 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v499_manuscript_readiness_delta.csv"
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
                "executable_item": "v499 creates a review outcome capture template.",
                "status": "review_outcome_capture_template_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v500 audits template consistency before outcomes",
                "last_wave": "v499",
                "execution_result": "review_outcome_template_created_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v499")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _template_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Outcome Capture Template v499

Generated: {status["generated_at_utc"]}

## Result

v499 converts the open v498 completion gaps into a structured capture template
for future layout and caption claim-safety review outcomes. No review outcome is
recorded, no caption is finalized, and no patch is authorized.

## Counts

- Outcome template rows: `{status["outcome_template_rows_v499"]}`.
- Layout outcome template rows: `{status["layout_outcome_template_rows_v499"]}`.
- Caption outcome template rows: `{status["caption_outcome_template_rows_v499"]}`.
- Outcome captured rows: `{status["outcome_captured_rows_v499"]}`.
- Review completed rows: `{status["review_completed_rows_v499"]}`.
- Accepted for patch rows: `{status["accepted_for_patch_rows_v499"]}`.
- Accepted for final caption rows: `{status["accepted_for_final_caption_rows_v499"]}`.
- Required field rows: `{status["required_field_rows_v499"]}`.
- Captured required field rows: `{status["captured_required_field_rows_v499"]}`.
- Active capture control rows: `{status["active_capture_control_rows_v499"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v499"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v499 is a capture-template artifact only. It does not capture completed review
outcomes, finalize captions, approve patch scope, edit Quarto, render the book,
make Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as
final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V499_REVIEW_OUTCOME_CAPTURE_TEMPLATE_START -->"
    end = "<!-- V499_REVIEW_OUTCOME_CAPTURE_TEMPLATE_END -->"
    block = f"""
{start}

## Wave v499: Review Outcome Capture Template

Generated: {status["generated_at_utc"]}

### Objective

v499 turns the v498 open completion gaps into a structured template for future
review outcomes while preserving the no-outcome, no-caption-final and no-patch
state.

### Results

- Outcome template rows:
  `{status["outcome_template_rows_v499"]}`.
- Layout outcome template rows:
  `{status["layout_outcome_template_rows_v499"]}`.
- Caption outcome template rows:
  `{status["caption_outcome_template_rows_v499"]}`.
- Outcome captured rows:
  `{status["outcome_captured_rows_v499"]}`.
- Review completed rows:
  `{status["review_completed_rows_v499"]}`.
- Accepted for patch rows:
  `{status["accepted_for_patch_rows_v499"]}`.
- Accepted for final caption rows:
  `{status["accepted_for_final_caption_rows_v499"]}`.
- Required field rows:
  `{status["required_field_rows_v499"]}`.
- Captured required field rows:
  `{status["captured_required_field_rows_v499"]}`.
- Active capture control rows:
  `{status["active_capture_control_rows_v499"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v499"]}`.
- Book sources modified:
  `{status["book_sources_modified_v499"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v499"]}`.

### Interpretation

Paper 4 now has review packets, a completion-gap audit and a review-outcome
capture template. The next useful step is to audit template consistency before
any human outcome capture or Quarto patching.

### Claim Impact

- Allowed: capture-template creation, required field declaration and active
  no-outcome/no-patch controls.
- Still prohibited: completed review/signoff claims, final captions, Quarto
  patch readiness/application, Quarto/book-reference mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v499 in the living notebook. v500 should audit template consistency before
capturing outcomes or modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v498 = _read_status(PRIOR_COMPLETION_GAP_VERSION)
    if v498["next_artifact_v498"] != "paper4_v499_review_outcome_capture_template.md":
        raise RuntimeError("v499 expects v498 to route to review outcome template.")

    template = _review_outcome_capture_template()
    field_dictionary = _outcome_field_dictionary()
    controls = _capture_control_register()
    readiness_summary = _template_readiness_summary()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v499_review_outcome_capture_template.csv", template)
    write_csv(TABLE_DIR / "paper4_v499_outcome_field_dictionary.csv", field_dictionary)
    write_csv(TABLE_DIR / "paper4_v499_capture_control_register.csv", controls)
    write_csv(TABLE_DIR / "paper4_v499_template_readiness_summary.csv", readiness_summary)
    write_csv(TABLE_DIR / "paper4_v499_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v499_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v499_review_outcome_capture_template",
        "schema_version": "2026-05-17.499",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_completion_gap_version_v499": PRIOR_COMPLETION_GAP_VERSION,
        "review_outcome_capture_template_created_v499": True,
        "outcome_template_rows_v499": len(template),
        "layout_outcome_template_rows_v499": int(
            template["review_domain_v499"].eq("layout_surface").sum()
        ),
        "caption_outcome_template_rows_v499": int(
            template["review_domain_v499"].eq("caption_claim_safety").sum()
        ),
        "outcome_captured_rows_v499": int(template["outcome_status_v499"].ne("not_captured").sum()),
        "review_completed_rows_v499": int(template["review_completed_v499"].astype(bool).sum()),
        "accepted_for_patch_rows_v499": int(
            template["accepted_for_patch_v499"].astype(bool).sum()
        ),
        "accepted_for_final_caption_rows_v499": int(
            template["accepted_for_final_caption_v499"].astype(bool).sum()
        ),
        "field_dictionary_rows_v499": len(field_dictionary),
        "required_field_rows_v499": int(
            field_dictionary["required_capture_field_v499"].astype(bool).sum()
        ),
        "captured_required_field_rows_v499": int(
            field_dictionary["field_captured_v499"].astype(bool).sum()
        ),
        "capture_control_rows_v499": len(controls),
        "active_capture_control_rows_v499": int(
            controls["control_active_v499"].astype(bool).sum()
        ),
        "template_readiness_rows_v499": len(readiness_summary),
        "readiness_delta_rows_v499": len(readiness),
        "ready_for_quarto_patch_v499": False,
        "quarto_patch_applied_v499": False,
        "book_sources_modified_v499": False,
        "book_references_modified_v499": False,
        "submission_ready_claim_allowed_v499": False,
        "working_champion_claim_allowed_v499": False,
        "paper1_promotion_allowed_v499": False,
        "paper4_working_champion_changed_v499": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v499": NEXT_ARTIFACT,
        "claim_boundary": (
            "v499 creates a review outcome capture template only; no outcomes, "
            "captions, approval, patching, submission or final promotion are allowed"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v499 must not create final Paper 4 promotion.")
    if status["outcome_captured_rows_v499"] != 0:
        raise RuntimeError("v499 must not capture review outcomes.")
    if status["accepted_for_patch_rows_v499"] != 0:
        raise RuntimeError("v499 must not approve a Quarto patch.")

    TEMPLATE_MD.write_text(_template_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v499": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
