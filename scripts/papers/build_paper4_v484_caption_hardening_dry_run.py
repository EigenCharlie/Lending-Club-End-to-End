#!/usr/bin/env python3
"""Build Paper 4 v484 caption hardening dry-run artifacts."""

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

VERSION = 484
PRIOR_REVIEW_PACKET_VERSION = 483
NEXT_ARTIFACT = "paper4_v485_caption_consistency_audit.md"
DRY_RUN_MD = NOTEBOOK.parent / "paper4_v484_caption_hardening_dry_run.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _hardened_caption_rows() -> pd.DataFrame:
    captions = pd.read_csv(TABLE_DIR / "paper4_v476_caption_plan.csv")
    hardened = {
        "T1": (
            "Local return and tail-risk frontier for the v338-v347-v353 chain. "
            "Evidence is local to the comparable universe; full-v55 optimality and "
            "working-champion claims remain outside scope."
        ),
        "T2": (
            "Tight source-governance ranking after the domain refresh. Grade A is "
            "retained as the primary blocker; the table does not authorize cap "
            "relaxation or fairness certification."
        ),
        "T3": (
            "Dynamic replay gap for the current local frontier. v338 remains the "
            "historical dynamic proxy anchor, while v353 lacks a dynamic replay trace."
        ),
        "T4": (
            "Internal online conformal monitoring proxy gates. The evidence is "
            "replay-based and internal; external holdout and production monitoring "
            "claims remain blocked."
        ),
        "T5": (
            "Formal SPO-DLA claim-boundary matrix. The table supports bounded "
            "historical audit language only; formal theorem and CRC guarantee claims "
            "remain blocked."
        ),
        "T6": (
            "Contractual IFRS9 requirement-gap audit. The table supports proxy and "
            "gap language only; contractual cashflow and accounting compliance claims "
            "remain blocked."
        ),
        "F1": (
            "Return, ECL and tail-risk frontier context. The figure provides visual "
            "context for the local frontier and must be read with the current T1 "
            "evidence table."
        ),
        "F2": (
            "Worst-source governance coverage context. The figure motivates the "
            "source-governance blocker without implying cap approval or legal "
            "fairness certification."
        ),
        "F3": (
            "Online conformal method-search context. The figure supports internal "
            "monitoring proxy discussion only, not production monitoring."
        ),
        "F4": (
            "Regret-auditability frontier context for historical SPO-DLA positioning. "
            "The figure does not establish a formal theorem or deployment guarantee."
        ),
    }
    rows = []
    for _, row in captions.iterrows():
        asset_id = str(row["asset_id_v476"])
        rows.append(
            {
                "asset_id_v484": asset_id,
                "asset_type_v484": row["asset_type_v476"],
                "source_asset_v484": row["source_asset_v476"],
                "source_caption_v484": row["draft_caption_v476"],
                "hardened_caption_v484": hardened[asset_id],
                "caption_hardened_v484": True,
                "caption_final_v484": False,
                "inserted_into_quarto_v484": False,
                "manual_review_required_v484": True,
                "required_caveat_v484": row["required_caveat_v476"],
                "claim_boundary_v484": row["claim_boundary_v476"],
            }
        )
    return pd.DataFrame(rows)


def _caveat_preservation_audit(hardened: pd.DataFrame) -> pd.DataFrame:
    required_terms = {
        "T1": ["local", "full-v55", "working-champion"],
        "T2": ["primary blocker", "cap"],
        "T3": ["v353", "dynamic replay"],
        "T4": ["internal", "external holdout", "production monitoring"],
        "T5": ["bounded", "formal theorem", "blocked"],
        "T6": ["contractual", "accounting compliance", "blocked"],
        "F1": ["visual context", "T1"],
        "F2": ["source-governance", "fairness certification"],
        "F3": ["internal", "not production monitoring"],
        "F4": ["does not establish", "formal theorem"],
    }
    rows = []
    for _, row in hardened.iterrows():
        asset_id = str(row["asset_id_v484"])
        text = str(row["hardened_caption_v484"]).lower()
        terms = required_terms[asset_id]
        present = all(term.lower() in text for term in terms)
        rows.append(
            {
                "asset_id_v484": asset_id,
                "required_terms_v484": ";".join(terms),
                "required_terms_present_v484": present,
                "caveat_preserved_v484": present,
                "caption_final_v484": False,
                "claim_boundary_v484": row["claim_boundary_v484"],
            }
        )
    return pd.DataFrame(rows)


def _caption_review_delta(hardened: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in hardened.iterrows():
        rows.append(
            {
                "asset_id_v484": row["asset_id_v484"],
                "review_action_v484": "manual editor review of hardened caption",
                "ready_for_manual_review_v484": True,
                "ready_for_quarto_insertion_v484": False,
                "caption_final_v484": False,
                "inserted_into_quarto_v484": False,
                "claim_boundary_v484": row["claim_boundary_v484"],
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v484": "caption_hardening_dry_run_created",
                "ready_v484": True,
                "evidence_artifact_v484": "paper4_v484_hardened_caption_dry_run.csv",
                "claim_boundary_v484": "dry-run hardened captions only",
            },
            {
                "readiness_gate_v484": "caveat_preservation_audit_created",
                "ready_v484": True,
                "evidence_artifact_v484": "paper4_v484_caption_caveat_preservation_audit.csv",
                "claim_boundary_v484": "caveat audit only",
            },
            {
                "readiness_gate_v484": "caption_review_delta_created",
                "ready_v484": True,
                "evidence_artifact_v484": "paper4_v484_caption_review_delta.csv",
                "claim_boundary_v484": "manual review queue only",
            },
            {
                "readiness_gate_v484": "captions_final",
                "ready_v484": False,
                "evidence_artifact_v484": "manual review required",
                "claim_boundary_v484": "captions remain non-final",
            },
            {
                "readiness_gate_v484": "captions_inserted_into_quarto",
                "ready_v484": False,
                "evidence_artifact_v484": "book sources unchanged",
                "claim_boundary_v484": "no Quarto/book mutation in v484",
            },
            {
                "readiness_gate_v484": "submission_ready",
                "ready_v484": False,
                "evidence_artifact_v484": "future approval, patch, render and venue gates",
                "claim_boundary_v484": "not a submission package",
            },
            {
                "readiness_gate_v484": "paper4_final_promotion_created",
                "ready_v484": False,
                "evidence_artifact_v484": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v484": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v484_caption_hardening_dry_run_created",
                "allowed": True,
                "artifact": "paper4_v484_hardened_caption_dry_run.csv",
                "boundary": "dry-run captions only",
            },
            {
                "claim_id": "v484_caveat_preservation_audit_created",
                "allowed": True,
                "artifact": "paper4_v484_caption_caveat_preservation_audit.csv",
                "boundary": "caveat audit only",
            },
            {
                "claim_id": "v484_caption_review_delta_created",
                "allowed": True,
                "artifact": "paper4_v484_caption_review_delta.csv",
                "boundary": "review delta only",
            },
            {
                "claim_id": "v484_captions_final_or_inserted",
                "allowed": False,
                "artifact": "paper4_v484_manuscript_readiness_delta.csv",
                "boundary": "manual review and patch authorization missing",
            },
            {
                "claim_id": "v484_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v484_manuscript_readiness_delta.csv",
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
                "claim": "v484 hardens Paper 4 captions as a dry-run.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v484_hardened_caption_dry_run.csv"
                ),
                "boundary": "Dry-run captions only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v484 verifies caveats remain present in hardened captions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v484_caption_caveat_preservation_audit.csv"
                ),
                "boundary": "Caveat audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v484 finalizes captions or inserts them into Quarto.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v484_manuscript_readiness_delta.csv"
                ),
                "boundary": "Captions remain non-final and uninserted.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v484 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v484_manuscript_readiness_delta.csv"
                ),
                "boundary": "Approval, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v484 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v484_manuscript_readiness_delta.csv"
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
                "executable_item": "v484 hardens captions in dry-run form.",
                "status": "caption_hardening_dry_run_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v485 audits hardened caption consistency",
                "last_wave": "v484",
                "execution_result": "captions_hardened_without_finalization_or_insertion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v484")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _dry_run_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Caption Hardening Dry Run v484

Generated: {status["generated_at_utc"]}

## Result

v484 creates hardened draft captions for the selected Paper 4 assets and audits
that required caveats remain present. The captions are ready for manual review,
but they are not final and are not inserted into Quarto.

## Counts

- Hardened caption rows: `{status["hardened_caption_rows_v484"]}`.
- Caveat audit rows: `{status["caveat_audit_rows_v484"]}`.
- Caveats preserved rows: `{status["caveats_preserved_rows_v484"]}`.
- Caption review rows: `{status["caption_review_rows_v484"]}`.
- Captions final: `{status["captions_final_v484"]}`.
- Captions inserted into Quarto: `{status["captions_inserted_into_quarto_v484"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v484 is a caption hardening dry-run only. Captions remain non-final, manual
review remains required, Quarto insertion is not authorized, and submission or
final-promotion claims remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V484_CAPTION_HARDENING_DRY_RUN_START -->"
    end = "<!-- V484_CAPTION_HARDENING_DRY_RUN_END -->"
    block = f"""
{start}

## Wave v484: Caption Hardening Dry Run

Generated: {status["generated_at_utc"]}

### Objective

v484 hardens the v476 draft captions while preserving caveats, without marking
captions final or inserting them into Quarto.

### Results

- Hardened caption rows:
  `{status["hardened_caption_rows_v484"]}`.
- Caveat audit rows:
  `{status["caveat_audit_rows_v484"]}`.
- Caveats preserved rows:
  `{status["caveats_preserved_rows_v484"]}`.
- Caption review rows:
  `{status["caption_review_rows_v484"]}`.
- Captions final:
  `{status["captions_final_v484"]}`.
- Captions inserted into Quarto:
  `{status["captions_inserted_into_quarto_v484"]}`.
- Book sources modified:
  `{status["book_sources_modified_v484"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v484"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v484"]}`.

### Interpretation

The captions are now more manuscript-like, but still deliberately dry-run
material. The next useful move is a consistency audit, not insertion.

### Claim Impact

- Allowed: hardened draft captions, caveat preservation audit and caption review
  delta.
- Still prohibited: final captions, Quarto insertion, book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v484 in the living notebook. v485 should audit hardened caption consistency
without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v483 = _read_status(PRIOR_REVIEW_PACKET_VERSION)
    if v483["next_artifact_v483"] != "paper4_v484_caption_hardening_dry_run.md":
        raise RuntimeError("v484 expects v483 to route to caption hardening dry-run.")

    hardened = _hardened_caption_rows()
    caveats = _caveat_preservation_audit(hardened)
    review_delta = _caption_review_delta(hardened)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v484_hardened_caption_dry_run.csv", hardened)
    write_csv(TABLE_DIR / "paper4_v484_caption_caveat_preservation_audit.csv", caveats)
    write_csv(TABLE_DIR / "paper4_v484_caption_review_delta.csv", review_delta)
    write_csv(TABLE_DIR / "paper4_v484_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v484_claim_matrix_delta.csv", claim_matrix)

    caveat_passes = int(caveats["caveat_preserved_v484"].astype(bool).sum())
    status = {
        "phase": "v484_caption_hardening_dry_run",
        "schema_version": "2026-05-17.484",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_review_packet_version_v484": PRIOR_REVIEW_PACKET_VERSION,
        "caption_hardening_dry_run_created_v484": True,
        "hardened_caption_rows_v484": len(hardened),
        "caveat_audit_rows_v484": len(caveats),
        "caveats_preserved_rows_v484": caveat_passes,
        "caption_review_rows_v484": len(review_delta),
        "readiness_delta_rows_v484": len(readiness),
        "captions_final_v484": False,
        "captions_inserted_into_quarto_v484": False,
        "book_sources_modified_v484": False,
        "book_references_modified_v484": False,
        "submission_ready_claim_allowed_v484": False,
        "working_champion_claim_allowed_v484": False,
        "paper1_promotion_allowed_v484": False,
        "paper4_working_champion_changed_v484": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v484": NEXT_ARTIFACT,
        "claim_boundary": (
            "v484 hardens draft captions only; final captions, insertion, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v484 must not create final Paper 4 promotion.")

    DRY_RUN_MD.write_text(_dry_run_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v484": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
