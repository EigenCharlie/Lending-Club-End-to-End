#!/usr/bin/env python3
"""Build Paper 4 v495 no-patch release synthesis artifacts."""

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

VERSION = 495
PRIOR_PATCH_APPROVAL_VERSION = 494
NEXT_ARTIFACT = "paper4_v496_review_gate_prioritization.md"
SYNTHESIS_MD = NOTEBOOK.parent / "paper4_v495_no_patch_release_synthesis.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _no_patch_release_synthesis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "synthesis_item_id_v495": "layout_consistency_audit",
                "source_wave_v495": "v489",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": False,
                "release_language_v495": "layout dry-run is internally consistent",
            },
            {
                "synthesis_item_id_v495": "manual_review_decision",
                "source_wave_v495": "v490",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": True,
                "release_language_v495": "layout accepted only for manual review",
            },
            {
                "synthesis_item_id_v495": "patch_readiness_preflight",
                "source_wave_v495": "v491",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": True,
                "release_language_v495": "patch readiness preflight failed with open blockers",
            },
            {
                "synthesis_item_id_v495": "manual_layout_review_packet",
                "source_wave_v495": "v492",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": True,
                "release_language_v495": "four surfaces and ten assets are queued for review",
            },
            {
                "synthesis_item_id_v495": "caption_signoff_gap",
                "source_wave_v495": "v493",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": True,
                "release_language_v495": "ten draft captions exist but zero are final",
            },
            {
                "synthesis_item_id_v495": "patch_approval_gap",
                "source_wave_v495": "v494",
                "useful_no_patch_evidence_v495": True,
                "blocks_patch_v495": True,
                "release_language_v495": "explicit patch approval is absent",
            },
        ]
    )


def _bounded_release_claim_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_claim_id_v495": "layout_evidence_ready_for_review",
                "allowed_v495": True,
                "evidence_artifact_v495": "paper4_v489_layout_consistency_checks.csv",
                "claim_boundary_v495": "review-ready evidence only",
            },
            {
                "release_claim_id_v495": "manual_review_queue_defined",
                "allowed_v495": True,
                "evidence_artifact_v495": "paper4_v492_manual_layout_review_packet.csv",
                "claim_boundary_v495": "manual review queue only",
            },
            {
                "release_claim_id_v495": "caption_and_approval_blockers_documented",
                "allowed_v495": True,
                "evidence_artifact_v495": "paper4_v493_and_v494_gap_packets",
                "claim_boundary_v495": "blockers documented only",
            },
            {
                "release_claim_id_v495": "paper4_ready_for_quarto_patch",
                "allowed_v495": False,
                "evidence_artifact_v495": "paper4_v491_readiness_scorecard.csv",
                "claim_boundary_v495": "patch readiness failed",
            },
            {
                "release_claim_id_v495": "paper4_submission_ready",
                "allowed_v495": False,
                "evidence_artifact_v495": "paper4_v495_manuscript_readiness_delta.csv",
                "claim_boundary_v495": "submission gates remain open",
            },
            {
                "release_claim_id_v495": "paper4_final_or_champion_replacement",
                "allowed_v495": False,
                "evidence_artifact_v495": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v495": "Paper Estrella boundary remains active",
            },
        ]
    )


def _next_work_queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_work_id_v495": "prioritize_review_gates",
                "recommended_next_v495": True,
                "blocks_patch_v495": True,
                "next_artifact_v495": NEXT_ARTIFACT,
            },
            {
                "next_work_id_v495": "complete_caption_signoff",
                "recommended_next_v495": True,
                "blocks_patch_v495": True,
                "next_artifact_v495": "future_caption_signoff_decision",
            },
            {
                "next_work_id_v495": "obtain_explicit_patch_approval",
                "recommended_next_v495": True,
                "blocks_patch_v495": True,
                "next_artifact_v495": "future_patch_approval_decision",
            },
            {
                "next_work_id_v495": "apply_quarto_patch",
                "recommended_next_v495": False,
                "blocks_patch_v495": True,
                "next_artifact_v495": "blocked_until_signoff_and_approval",
            },
            {
                "next_work_id_v495": "declare_paper4_final",
                "recommended_next_v495": False,
                "blocks_patch_v495": True,
                "next_artifact_v495": "forbidden_final_promotion",
            },
        ]
    )


def _release_decision_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v495": "release_no_patch_synthesis",
                "recommended_v495": True,
                "patch_allowed_v495": False,
                "decision_boundary_v495": "living notebook synthesis only",
            },
            {
                "decision_id_v495": "prioritize_review_gates_next",
                "recommended_v495": True,
                "patch_allowed_v495": False,
                "decision_boundary_v495": "review gates must be prioritized first",
            },
            {
                "decision_id_v495": "request_patch_after_signoff",
                "recommended_v495": False,
                "patch_allowed_v495": False,
                "decision_boundary_v495": "signoff and approval not complete",
            },
            {
                "decision_id_v495": "publish_or_finalize_paper4",
                "recommended_v495": False,
                "patch_allowed_v495": False,
                "decision_boundary_v495": "submission/final promotion blocked",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v495": "no_patch_release_synthesis_created",
                "ready_v495": True,
                "evidence_artifact_v495": "paper4_v495_no_patch_release_synthesis.csv",
                "claim_boundary_v495": "no-patch synthesis only",
            },
            {
                "readiness_gate_v495": "bounded_release_claim_register_created",
                "ready_v495": True,
                "evidence_artifact_v495": "paper4_v495_bounded_release_claim_register.csv",
                "claim_boundary_v495": "bounded claim register only",
            },
            {
                "readiness_gate_v495": "next_work_queue_created",
                "ready_v495": True,
                "evidence_artifact_v495": "paper4_v495_next_work_queue.csv",
                "claim_boundary_v495": "future work queue only",
            },
            {
                "readiness_gate_v495": "release_decision_matrix_created",
                "ready_v495": True,
                "evidence_artifact_v495": "paper4_v495_release_decision_matrix.csv",
                "claim_boundary_v495": "decision matrix only",
            },
            {
                "readiness_gate_v495": "ready_for_quarto_patch",
                "ready_v495": False,
                "evidence_artifact_v495": "caption signoff and approval missing",
                "claim_boundary_v495": "patch remains blocked",
            },
            {
                "readiness_gate_v495": "book_sources_or_references_modified",
                "ready_v495": False,
                "evidence_artifact_v495": "book sources unchanged",
                "claim_boundary_v495": "no Quarto/book mutation in v495",
            },
            {
                "readiness_gate_v495": "submission_ready",
                "ready_v495": False,
                "evidence_artifact_v495": "future approval, patch, render and venue gates",
                "claim_boundary_v495": "not a submission package",
            },
            {
                "readiness_gate_v495": "paper4_final_promotion_created",
                "ready_v495": False,
                "evidence_artifact_v495": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v495": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v495_no_patch_release_synthesis_created",
                "allowed": True,
                "artifact": "paper4_v495_no_patch_release_synthesis.csv",
                "boundary": "no-patch synthesis only",
            },
            {
                "claim_id": "v495_bounded_release_claim_register_created",
                "allowed": True,
                "artifact": "paper4_v495_bounded_release_claim_register.csv",
                "boundary": "bounded claim register only",
            },
            {
                "claim_id": "v495_next_work_queue_created",
                "allowed": True,
                "artifact": "paper4_v495_next_work_queue.csv",
                "boundary": "future work queue only",
            },
            {
                "claim_id": "v495_patch_or_submission_ready",
                "allowed": False,
                "artifact": "paper4_v495_manuscript_readiness_delta.csv",
                "boundary": "patch and submission remain blocked",
            },
            {
                "claim_id": "v495_final_promotion",
                "allowed": False,
                "artifact": "paper4_v495_manuscript_readiness_delta.csv",
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
                "claim": "v495 synthesizes no-patch Paper 4 release evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_no_patch_release_synthesis.csv"
                ),
                "boundary": "No-patch synthesis only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v495 registers bounded release claims for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_bounded_release_claim_register.csv"
                ),
                "boundary": "Bounded claim register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v495 creates the next executable review-gate queue.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_next_work_queue.csv"
                ),
                "boundary": "Next work queue only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v495 makes Paper 4 ready for Quarto patching or submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch and submission gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v495 edits book sources or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_release_decision_matrix.csv"
                ),
                "boundary": "Patch is not authorized in v495.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v495 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v495_manuscript_readiness_delta.csv"
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
                "executable_item": "v495 synthesizes no-patch release evidence.",
                "status": "no_patch_release_synthesis_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v496 prioritizes review gates for execution",
                "last_wave": "v495",
                "execution_result": "no_patch_release_synthesis_created_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v495")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _synthesis_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 No-Patch Release Synthesis v495

Generated: {status["generated_at_utc"]}

## Result

v495 synthesizes the v489-v494 manuscript-review chain into a no-patch release
packet. It identifies useful bounded evidence and the next work queue, but does
not make Paper 4 patch-ready, submitted, final, or promoted.

## Counts

- Synthesis rows: `{status["synthesis_rows_v495"]}`.
- Useful no-patch evidence rows: `{status["useful_no_patch_evidence_rows_v495"]}`.
- Blocking synthesis rows: `{status["blocking_synthesis_rows_v495"]}`.
- Release claim rows: `{status["release_claim_rows_v495"]}`.
- Allowed release claim rows: `{status["allowed_release_claim_rows_v495"]}`.
- Next work queue rows: `{status["next_work_queue_rows_v495"]}`.
- Release decision rows: `{status["release_decision_rows_v495"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v495"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v495 is a no-patch synthesis only. It does not edit Quarto, apply a patch,
render the book, make Paper 4 submission-ready, replace Paper Estrella, or
promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V495_NO_PATCH_RELEASE_SYNTHESIS_START -->"
    end = "<!-- V495_NO_PATCH_RELEASE_SYNTHESIS_END -->"
    block = f"""
{start}

## Wave v495: No-Patch Release Synthesis

Generated: {status["generated_at_utc"]}

### Objective

v495 synthesizes the v489-v494 no-patch manuscript-review chain into bounded
release evidence and a next-work queue.

### Results

- Synthesis rows:
  `{status["synthesis_rows_v495"]}`.
- Useful no-patch evidence rows:
  `{status["useful_no_patch_evidence_rows_v495"]}`.
- Blocking synthesis rows:
  `{status["blocking_synthesis_rows_v495"]}`.
- Release claim rows:
  `{status["release_claim_rows_v495"]}`.
- Allowed release claim rows:
  `{status["allowed_release_claim_rows_v495"]}`.
- Next work queue rows:
  `{status["next_work_queue_rows_v495"]}`.
- Release decision rows:
  `{status["release_decision_rows_v495"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v495"]}`.
- Book sources modified:
  `{status["book_sources_modified_v495"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v495"]}`.

### Interpretation

The living lab now has a clean no-patch release packet: useful evidence is
available for review, but patching, submission and final promotion remain
blocked.

### Claim Impact

- Allowed: no-patch release synthesis, bounded release claim register and next
  executable review-gate queue.
- Still prohibited: Quarto patch readiness/application, Quarto/book-reference
  mutation, submission readiness, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v495 in the living notebook. v496 should prioritize review gates for
execution without modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v494 = _read_status(PRIOR_PATCH_APPROVAL_VERSION)
    if v494["next_artifact_v494"] != "paper4_v495_no_patch_release_synthesis.md":
        raise RuntimeError("v495 expects v494 to route to no-patch release synthesis.")

    synthesis = _no_patch_release_synthesis()
    release_claims = _bounded_release_claim_register()
    next_work = _next_work_queue()
    decisions = _release_decision_matrix()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v495_no_patch_release_synthesis.csv", synthesis)
    write_csv(TABLE_DIR / "paper4_v495_bounded_release_claim_register.csv", release_claims)
    write_csv(TABLE_DIR / "paper4_v495_next_work_queue.csv", next_work)
    write_csv(TABLE_DIR / "paper4_v495_release_decision_matrix.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v495_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v495_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v495_no_patch_release_synthesis",
        "schema_version": "2026-05-17.495",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_patch_approval_version_v495": PRIOR_PATCH_APPROVAL_VERSION,
        "no_patch_release_synthesis_created_v495": True,
        "synthesis_rows_v495": len(synthesis),
        "useful_no_patch_evidence_rows_v495": int(
            synthesis["useful_no_patch_evidence_v495"].astype(bool).sum()
        ),
        "blocking_synthesis_rows_v495": int(synthesis["blocks_patch_v495"].astype(bool).sum()),
        "release_claim_rows_v495": len(release_claims),
        "allowed_release_claim_rows_v495": int(release_claims["allowed_v495"].astype(bool).sum()),
        "next_work_queue_rows_v495": len(next_work),
        "recommended_next_work_rows_v495": int(
            next_work["recommended_next_v495"].astype(bool).sum()
        ),
        "release_decision_rows_v495": len(decisions),
        "recommended_release_decision_rows_v495": int(
            decisions["recommended_v495"].astype(bool).sum()
        ),
        "readiness_delta_rows_v495": len(readiness),
        "ready_for_quarto_patch_v495": False,
        "quarto_patch_applied_v495": False,
        "book_sources_modified_v495": False,
        "book_references_modified_v495": False,
        "submission_ready_claim_allowed_v495": False,
        "working_champion_claim_allowed_v495": False,
        "paper1_promotion_allowed_v495": False,
        "paper4_working_champion_changed_v495": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v495": NEXT_ARTIFACT,
        "claim_boundary": (
            "v495 synthesizes no-patch release evidence only; patching, submission "
            "and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v495 must not create final Paper 4 promotion.")

    SYNTHESIS_MD.write_text(_synthesis_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v495": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
