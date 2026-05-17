#!/usr/bin/env python3
"""Build Paper 4 v462 manuscript readiness delta artifacts."""

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

VERSION = 462
PRIOR_BOUNDED_RELATED_WORK_VERSION = 461
NEXT_ARTIFACT = "paper4_v463_paper_specific_bibliography_plan.md"
DELTA_MD = NOTEBOOK.parent / "paper4_v462_manuscript_readiness_delta.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _readiness_delta_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v462": "abstract_conclusion_draft",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v455_abstract_conclusion_draft.md",
                "delta_since_v452_v462": "created after scaffold",
                "claim_boundary_v462": "draft prose only",
            },
            {
                "readiness_gate_v462": "methods_results_draft",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v453_methods_results_draft.md",
                "delta_since_v452_v462": "created after scaffold",
                "claim_boundary_v462": "draft prose only",
            },
            {
                "readiness_gate_v462": "discussion_limitations_draft",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v454_discussion_limitations_draft.md",
                "delta_since_v452_v462": "created after methods/results",
                "claim_boundary_v462": "draft prose only",
            },
            {
                "readiness_gate_v462": "bounded_related_work_draft",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v461_bounded_related_work_draft.md",
                "delta_since_v452_v462": "newly created in v461",
                "claim_boundary_v462": "bounded draft only",
            },
            {
                "readiness_gate_v462": "manuscript_assembly_packet",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v456_manuscript_assembly_packet.md",
                "delta_since_v452_v462": "major prose assembled",
                "claim_boundary_v462": "assembly packet only",
            },
            {
                "readiness_gate_v462": "post_assembly_pytest_clean",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v457_pytest_probe_summary.csv",
                "delta_since_v452_v462": "1195-test regression refresh passed",
                "claim_boundary_v462": "internal regression only",
            },
            {
                "readiness_gate_v462": "post_assembly_render_decision",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v458_post_assembly_render_decision.md",
                "delta_since_v452_v462": "render not required while book sources unchanged",
                "claim_boundary_v462": "decision only",
            },
            {
                "readiness_gate_v462": "venue_agnostic_structure_packet",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v459_target_venue_structure_packet.md",
                "delta_since_v452_v462": "target structure mapped without venue selection",
                "claim_boundary_v462": "venue-agnostic structure only",
            },
            {
                "readiness_gate_v462": "verified_anchor_gap_audit",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_v460_related_work_citation_gap_audit.md",
                "delta_since_v452_v462": "citation gaps audited without new sources",
                "claim_boundary_v462": "audit only",
            },
            {
                "readiness_gate_v462": "paper_specific_bibliography",
                "ready_v462": False,
                "evidence_artifact_v462": NEXT_ARTIFACT,
                "delta_since_v452_v462": "still missing",
                "claim_boundary_v462": "do not claim final bibliography",
            },
            {
                "readiness_gate_v462": "target_venue_selected",
                "ready_v462": False,
                "evidence_artifact_v462": "future_target_venue_decision",
                "delta_since_v452_v462": "still missing",
                "claim_boundary_v462": "do not claim venue compliance",
            },
            {
                "readiness_gate_v462": "external_validation_complete",
                "ready_v462": False,
                "evidence_artifact_v462": "future_external_validation_protocol",
                "delta_since_v452_v462": "still missing",
                "claim_boundary_v462": "do not claim external generalization",
            },
            {
                "readiness_gate_v462": "final_promotion_absent",
                "ready_v462": True,
                "evidence_artifact_v462": "paper4_final_promotion_gate_not_created",
                "delta_since_v452_v462": "invariant preserved",
                "claim_boundary_v462": "Paper Estrella remains protected",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v462": "paper_specific_bibliography_not_planned",
                "blocking_v462": True,
                "evidence_count_v462": 1,
                "required_next_artifact_v462": NEXT_ARTIFACT,
                "claim_boundary_v462": "next executable local manuscript task",
            },
            {
                "blocker_id_v462": "target_venue_not_selected",
                "blocking_v462": True,
                "evidence_count_v462": 0,
                "required_next_artifact_v462": "future_target_venue_decision",
                "claim_boundary_v462": "do not claim venue compliance",
            },
            {
                "blocker_id_v462": "external_dataset_validation_not_run",
                "blocking_v462": True,
                "evidence_count_v462": 0,
                "required_next_artifact_v462": "future_external_validation_protocol",
                "claim_boundary_v462": "do not claim external generalization",
            },
            {
                "blocker_id_v462": "systematic_literature_search_not_run",
                "blocking_v462": True,
                "evidence_count_v462": 1,
                "required_next_artifact_v462": "future_recent_literature_search_log",
                "claim_boundary_v462": "do not claim systematic review",
            },
            {
                "blocker_id_v462": "paper4_final_promotion_forbidden",
                "blocking_v462": True,
                "evidence_count_v462": 1,
                "required_next_artifact_v462": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v462": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _next_wave_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_next_wave_v462": "paper_specific_bibliography_plan",
                "selected_v462": True,
                "reason_v462": "local, executable, unblocks references without new source claims",
                "next_artifact_v462": NEXT_ARTIFACT,
                "claim_boundary_v462": "planning only",
            },
            {
                "candidate_next_wave_v462": "target_venue_selection",
                "selected_v462": False,
                "reason_v462": "requires user/venue preference",
                "next_artifact_v462": "future_target_venue_decision",
                "claim_boundary_v462": "not selected automatically",
            },
            {
                "candidate_next_wave_v462": "external_validation_protocol",
                "selected_v462": False,
                "reason_v462": "requires dataset/scope decision",
                "next_artifact_v462": "future_external_validation_protocol",
                "claim_boundary_v462": "not executed in v462",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v462_manuscript_readiness_delta_created",
                "allowed": True,
                "artifact": "paper4_v462_manuscript_readiness_delta.md",
                "boundary": "readiness synthesis only",
            },
            {
                "claim_id": "v462_major_manuscript_components_exist",
                "allowed": True,
                "artifact": "paper4_v462_readiness_delta_matrix.csv",
                "boundary": "major components exist but final package is incomplete",
            },
            {
                "claim_id": "v462_next_wave_selected_for_bibliography_plan",
                "allowed": True,
                "artifact": "paper4_v462_next_wave_decision.csv",
                "boundary": "next local task only",
            },
            {
                "claim_id": "v462_submission_ready_or_venue_compliant",
                "allowed": False,
                "artifact": "paper4_v462_remaining_blockers.csv",
                "boundary": "venue and bibliography blockers remain",
            },
            {
                "claim_id": "v462_external_validation_or_systematic_review_complete",
                "allowed": False,
                "artifact": "paper4_v462_remaining_blockers.csv",
                "boundary": "external validation and systematic review remain open",
            },
            {
                "claim_id": "v462_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v462 synthesizes Paper 4 manuscript readiness delta.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v462_manuscript_readiness_delta.md"
                ),
                "boundary": "Readiness synthesis only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v462 shows major manuscript components now exist.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v462_readiness_delta_matrix.csv"
                ),
                "boundary": "Components exist; final package remains incomplete.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v462 makes Paper 4 submission-ready or venue-compliant.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v462_remaining_blockers.csv"
                ),
                "boundary": "Venue and bibliography blockers remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v462 completes external validation or systematic literature review.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v462_remaining_blockers.csv"
                ),
                "boundary": "Both remain future work.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v462 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v462_remaining_blockers.csv"
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
                "executable_item": "v462 synthesizes manuscript readiness delta.",
                "status": "manuscript_readiness_delta_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v463 plans Paper 4 specific bibliography work",
                "last_wave": "v462",
                "execution_result": "readiness_delta_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v462")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _delta_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manuscript Readiness Delta v462

Generated: {status["generated_at_utc"]}

## Delta

The Paper 4 manuscript package has moved from scaffold to a guarded
venue-agnostic draft packet. Abstract/conclusion, methods/results,
discussion/limitations, bounded related work, an assembly packet, post-assembly
pytest, render decision, target structure, and citation gap audit now exist.

## Current Readiness Counts

- Readiness gates recorded: `{status["readiness_gate_count_v462"]}`.
- Ready gates: `{status["ready_gate_count_v462"]}`.
- Not-ready gates: `{status["not_ready_gate_count_v462"]}`.
- Blocking rows: `{status["blocking_row_count_v462"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Main Remaining Work

The next local executable task is a Paper 4 specific bibliography plan. Target
venue selection, systematic literature search, external validation, and any
Quarto promotion remain separate future decisions.

## Required Caveat

v462 is a readiness synthesis only. It does not select a target venue, complete
a bibliography, run external validation, claim a systematic literature review,
submit the paper, replace Paper Estrella, or promote Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v462"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V462_MANUSCRIPT_READINESS_DELTA_START -->"
    end = "<!-- V462_MANUSCRIPT_READINESS_DELTA_END -->"
    block = f"""
{start}

## Wave v462: Manuscript Readiness Delta

Generated: {status["generated_at_utc"]}

### Objective

v462 synthesizes what became ready after v456-v461 and selects the next local
executable manuscript task.

### Results

- Readiness gates recorded:
  `{status["readiness_gate_count_v462"]}`.
- Ready gates:
  `{status["ready_gate_count_v462"]}`.
- Not-ready gates:
  `{status["not_ready_gate_count_v462"]}`.
- Blocking rows:
  `{status["blocking_row_count_v462"]}`.
- Selected next wave:
  `{status["selected_next_wave_v462"]}`.
- Submission ready:
  `{status["submission_ready_v462"]}`.
- External validation complete:
  `{status["external_validation_complete_v462"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v462"]}`.

### Interpretation

The manuscript has become substantially more usable for Paper 4 drafting, but
the remaining blockers are still manuscript-critical: bibliography planning,
venue selection, systematic literature search and external validation.

### Claim Impact

- Allowed: manuscript readiness delta and next local task selection.
- Still prohibited: submission readiness, venue compliance, external validation,
  systematic review, champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v462 in the living notebook. v463 should plan the Paper 4 specific
bibliography without editing the global bibliography or claiming finality.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v461 = _read_status(461)
    if v461["next_artifact_v461"] != "paper4_v462_manuscript_readiness_delta.md":
        raise RuntimeError("v462 expects v461 to route to manuscript readiness delta.")
    if v461["bounded_related_work_draft_created_v461"] is not True:
        raise RuntimeError("v462 expects v461 bounded related-work draft.")

    readiness = _readiness_delta_matrix()
    blockers = _remaining_blockers()
    next_wave = _next_wave_decision()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v462_readiness_delta_matrix.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v462_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v462_next_wave_decision.csv", next_wave)
    write_csv(TABLE_DIR / "paper4_v462_claim_matrix_delta.csv", claim_matrix)

    ready_count = int(readiness["ready_v462"].astype(bool).sum())
    selected_next = next_wave.loc[next_wave["selected_v462"].astype(bool)].iloc[0]
    status = {
        "phase": "v462_manuscript_readiness_delta",
        "schema_version": "2026-05-17.462",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_bounded_related_work_version_v462": PRIOR_BOUNDED_RELATED_WORK_VERSION,
        "readiness_gate_count_v462": len(readiness),
        "ready_gate_count_v462": ready_count,
        "not_ready_gate_count_v462": int(len(readiness) - ready_count),
        "blocking_row_count_v462": int(blockers["blocking_v462"].astype(bool).sum()),
        "selected_next_wave_v462": str(selected_next["candidate_next_wave_v462"]),
        "manuscript_readiness_delta_created_v462": True,
        "major_manuscript_components_exist_v462": True,
        "paper_specific_bibliography_complete_v462": False,
        "target_venue_selected_v462": False,
        "submission_ready_v462": False,
        "external_validation_complete_v462": False,
        "systematic_literature_review_complete_v462": False,
        "working_champion_claim_allowed_v462": False,
        "paper1_promotion_allowed_v462": False,
        "paper4_working_champion_changed_v462": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v462": NEXT_ARTIFACT,
        "claim_boundary": (
            "v462 synthesizes readiness only; bibliography, venue, systematic "
            "review, external validation, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v462 must not create final Paper 4 promotion.")

    DELTA_MD.write_text(_delta_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v462": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
