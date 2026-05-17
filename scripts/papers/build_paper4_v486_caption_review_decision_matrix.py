#!/usr/bin/env python3
"""Build Paper 4 v486 caption review decision matrix artifacts."""

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

VERSION = 486
PRIOR_CAPTION_AUDIT_VERSION = 485
NEXT_ARTIFACT = "paper4_v487_caption_asset_pairing_packet.md"
DECISION_MD = NOTEBOOK.parent / "paper4_v486_caption_review_decision_matrix.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _caption_review_decisions() -> pd.DataFrame:
    captions = pd.read_csv(TABLE_DIR / "paper4_v484_hardened_caption_dry_run.csv")
    quality = pd.read_csv(TABLE_DIR / "paper4_v485_caption_quality_matrix.csv")
    decisions = captions.merge(
        quality[["asset_id_v485", "caption_quality_pass_v485"]],
        left_on="asset_id_v484",
        right_on="asset_id_v485",
        how="left",
    )
    rows = []
    for _, row in decisions.iterrows():
        rows.append(
            {
                "asset_id_v486": row["asset_id_v484"],
                "asset_type_v486": row["asset_type_v484"],
                "hardened_caption_v486": row["hardened_caption_v484"],
                "audit_passed_v486": bool(row["caption_quality_pass_v485"]),
                "review_decision_v486": "accept_for_draft_under_caveat",
                "accepted_for_manuscript_draft_v486": True,
                "requires_manual_editor_signoff_v486": True,
                "caption_final_v486": False,
                "caption_inserted_into_quarto_v486": False,
                "claim_boundary_v486": row["claim_boundary_v484"],
            }
        )
    return pd.DataFrame(rows)


def _decision_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "summary_metric_v486": "caption_review_decisions",
                "metric_value_v486": 10,
                "claim_boundary_v486": "decision matrix only",
            },
            {
                "summary_metric_v486": "accepted_for_manuscript_draft",
                "metric_value_v486": 10,
                "claim_boundary_v486": "draft acceptance only",
            },
            {
                "summary_metric_v486": "requires_manual_editor_signoff",
                "metric_value_v486": 10,
                "claim_boundary_v486": "manual signoff still required",
            },
            {
                "summary_metric_v486": "captions_final",
                "metric_value_v486": 0,
                "claim_boundary_v486": "no final captions",
            },
            {
                "summary_metric_v486": "captions_inserted_into_quarto",
                "metric_value_v486": 0,
                "claim_boundary_v486": "no Quarto insertion",
            },
            {
                "summary_metric_v486": "final_promotion_created",
                "metric_value_v486": int(FORBIDDEN_FINAL_PROMOTION.exists()),
                "claim_boundary_v486": "final promotion remains forbidden",
            },
        ]
    )


def _revision_action_register() -> pd.DataFrame:
    decisions = _caption_review_decisions()
    rows = []
    for _, row in decisions.iterrows():
        rows.append(
            {
                "asset_id_v486": row["asset_id_v486"],
                "revision_action_v486": "editorial signoff before final caption",
                "action_required_v486": True,
                "blocks_final_caption_v486": True,
                "blocks_quarto_insertion_v486": True,
                "claim_boundary_v486": row["claim_boundary_v486"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v486": "caption_review_decision_matrix_created",
                "ready_v486": True,
                "evidence_artifact_v486": "paper4_v486_caption_review_decision_matrix.csv",
                "claim_boundary_v486": "decision matrix only",
            },
            {
                "readiness_gate_v486": "caption_decision_summary_created",
                "ready_v486": True,
                "evidence_artifact_v486": "paper4_v486_caption_decision_summary.csv",
                "claim_boundary_v486": "summary only",
            },
            {
                "readiness_gate_v486": "caption_revision_actions_created",
                "ready_v486": True,
                "evidence_artifact_v486": "paper4_v486_caption_revision_action_register.csv",
                "claim_boundary_v486": "revision actions only",
            },
            {
                "readiness_gate_v486": "captions_accepted_for_draft",
                "ready_v486": True,
                "evidence_artifact_v486": "paper4_v486_caption_review_decision_matrix.csv",
                "claim_boundary_v486": "draft acceptance only",
            },
            {
                "readiness_gate_v486": "captions_final",
                "ready_v486": False,
                "evidence_artifact_v486": "editorial signoff missing",
                "claim_boundary_v486": "captions remain non-final",
            },
            {
                "readiness_gate_v486": "captions_inserted_into_quarto",
                "ready_v486": False,
                "evidence_artifact_v486": "book sources unchanged",
                "claim_boundary_v486": "no Quarto/book mutation in v486",
            },
            {
                "readiness_gate_v486": "submission_ready",
                "ready_v486": False,
                "evidence_artifact_v486": "future approval, patch, render and venue gates",
                "claim_boundary_v486": "not a submission package",
            },
            {
                "readiness_gate_v486": "paper4_final_promotion_created",
                "ready_v486": False,
                "evidence_artifact_v486": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v486": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v486_caption_review_decision_matrix_created",
                "allowed": True,
                "artifact": "paper4_v486_caption_review_decision_matrix.csv",
                "boundary": "decision matrix only",
            },
            {
                "claim_id": "v486_captions_accepted_for_draft_under_caveat",
                "allowed": True,
                "artifact": "paper4_v486_caption_review_decision_matrix.csv",
                "boundary": "draft acceptance only",
            },
            {
                "claim_id": "v486_caption_revision_actions_created",
                "allowed": True,
                "artifact": "paper4_v486_caption_revision_action_register.csv",
                "boundary": "revision actions only",
            },
            {
                "claim_id": "v486_captions_final_or_inserted",
                "allowed": False,
                "artifact": "paper4_v486_manuscript_readiness_delta.csv",
                "boundary": "editorial signoff and patch authorization missing",
            },
            {
                "claim_id": "v486_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v486_manuscript_readiness_delta.csv",
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
                "claim": "v486 accepts hardened captions for draft use under caveats.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v486_caption_review_decision_matrix.csv"
                ),
                "boundary": "Draft acceptance only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v486 creates revision actions before final caption signoff.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v486_caption_revision_action_register.csv"
                ),
                "boundary": "Revision action register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v486 finalizes captions or inserts them into Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v486_manuscript_readiness_delta.csv"
                ),
                "boundary": "Captions remain non-final and uninserted.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v486 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v486_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v486 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v486_manuscript_readiness_delta.csv"
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
                "executable_item": "v486 records caption review decisions.",
                "status": "caption_review_decision_matrix_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v487 pairs accepted captions with assets",
                "last_wave": "v486",
                "execution_result": "captions_accepted_for_draft_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v486")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption Review Decision Matrix v486

Generated: {status["generated_at_utc"]}

## Result

v486 records caption-review decisions after the v485 audit. All hardened
captions are accepted for draft use under caveats, while manual editor signoff,
final caption status and Quarto insertion remain blocked.

## Counts

- Caption decision rows: `{status["caption_decision_rows_v486"]}`.
- Draft accepted rows: `{status["draft_accepted_rows_v486"]}`.
- Revision action rows: `{status["revision_action_rows_v486"]}`.
- Captions final: `{status["captions_final_v486"]}`.
- Captions inserted into Quarto: `{status["captions_inserted_into_quarto_v486"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v486 records draft-level caption decisions only. It does not finalize captions,
insert captions into Quarto, make Paper 4 submission-ready, replace Paper
Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V486_CAPTION_REVIEW_DECISION_MATRIX_START -->"
    end = "<!-- V486_CAPTION_REVIEW_DECISION_MATRIX_END -->"
    block = f"""
{start}

## Wave v486: Caption Review Decision Matrix

Generated: {status["generated_at_utc"]}

### Objective

v486 records review decisions for v485-audited captions, accepting them only for
draft use under caveats.

### Results

- Caption decision rows:
  `{status["caption_decision_rows_v486"]}`.
- Draft accepted rows:
  `{status["draft_accepted_rows_v486"]}`.
- Revision action rows:
  `{status["revision_action_rows_v486"]}`.
- Captions final:
  `{status["captions_final_v486"]}`.
- Captions inserted into Quarto:
  `{status["captions_inserted_into_quarto_v486"]}`.
- Book sources modified:
  `{status["book_sources_modified_v486"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v486"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v486"]}`.

### Interpretation

The captions have crossed from audited dry-run text into draft-accepted material,
but still require editor signoff before final use.

### Claim Impact

- Allowed: draft caption acceptance and revision action register.
- Still prohibited: final captions, Quarto insertion, book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v486 in the living notebook. v487 should pair draft-accepted captions with
their assets without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v485 = _read_status(PRIOR_CAPTION_AUDIT_VERSION)
    if v485["next_artifact_v485"] != "paper4_v486_caption_review_decision_matrix.md":
        raise RuntimeError("v486 expects v485 to route to caption review decisions.")
    if not v485["caption_consistency_audit_passed_v485"]:
        raise RuntimeError("v486 requires a passing v485 caption consistency audit.")

    decisions = _caption_review_decisions()
    summary = _decision_summary()
    revision_actions = _revision_action_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v486_caption_review_decision_matrix.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v486_caption_decision_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v486_caption_revision_action_register.csv", revision_actions)
    write_csv(TABLE_DIR / "paper4_v486_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v486_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v486_caption_review_decision_matrix",
        "schema_version": "2026-05-17.486",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_caption_audit_version_v486": PRIOR_CAPTION_AUDIT_VERSION,
        "caption_review_decision_matrix_created_v486": True,
        "caption_decision_rows_v486": len(decisions),
        "draft_accepted_rows_v486": int(
            decisions["accepted_for_manuscript_draft_v486"].astype(bool).sum()
        ),
        "revision_action_rows_v486": len(revision_actions),
        "readiness_delta_rows_v486": len(readiness),
        "captions_final_v486": False,
        "captions_inserted_into_quarto_v486": False,
        "book_sources_modified_v486": False,
        "book_references_modified_v486": False,
        "submission_ready_claim_allowed_v486": False,
        "working_champion_claim_allowed_v486": False,
        "paper1_promotion_allowed_v486": False,
        "paper4_working_champion_changed_v486": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v486": NEXT_ARTIFACT,
        "claim_boundary": (
            "v486 accepts captions for draft only; final captions, insertion, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v486 must not create final Paper 4 promotion.")

    DECISION_MD.write_text(_decision_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v486": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
