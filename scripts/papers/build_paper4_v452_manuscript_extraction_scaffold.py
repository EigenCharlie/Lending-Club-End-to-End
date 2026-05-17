#!/usr/bin/env python3
"""Build Paper 4 v452 manuscript extraction scaffold artifacts."""

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

VERSION = 452
PRIOR_READINESS_SYNTHESIS_VERSION = 451
NEXT_ARTIFACT = "paper4_v453_methods_results_draft.md"
SCAFFOLD_MD = NOTEBOOK.parent / "paper4_v452_manuscript_extraction_scaffold.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _section_scaffold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id_v452": "abstract",
                "target_role_v452": "compress contribution and validation envelope",
                "primary_artifact_v452": "paper4_v451_release_readiness_gate_summary.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "bounded readiness, not final acceptance",
            },
            {
                "section_id_v452": "introduction",
                "target_role_v452": "motivate dynamic risk lab and decision governance gap",
                "primary_artifact_v452": "paper4_living_lab_notebook.md",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "motivation from living-lab history",
            },
            {
                "section_id_v452": "methods",
                "target_role_v452": "describe SDAM, CVaR/source governance, replay and gates",
                "primary_artifact_v452": "paper4_current_claim_boundaries.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "method description from generated artifacts",
            },
            {
                "section_id_v452": "results",
                "target_role_v452": "summarize validation gates and executable wave outcomes",
                "primary_artifact_v452": "paper4_v451_release_readiness_gate_summary.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "validation results only",
            },
            {
                "section_id_v452": "quarto_reproducibility",
                "target_role_v452": "report Paper 4 and full-book render readiness",
                "primary_artifact_v452": "paper4_v449_full_book_render_probe_summary.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "render/reproducibility only",
            },
            {
                "section_id_v452": "limitations",
                "target_role_v452": (
                    "state no external dataset, no legal fairness, no final promotion"
                ),
                "primary_artifact_v452": "paper4_v451_remaining_blockers.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "must include prohibited claims",
            },
            {
                "section_id_v452": "conclusion",
                "target_role_v452": "close with bounded readiness and next validation agenda",
                "primary_artifact_v452": "paper4_v451_publishable_language_bank.csv",
                "draft_allowed_v452": True,
                "claim_boundary_v452": "no submission-ready or final claim",
            },
        ]
    )


def _figure_table_shortlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id_v452": "tbl_validation_gates",
                "item_type_v452": "table",
                "source_artifact_v452": "paper4_v451_release_readiness_gate_summary.csv",
                "manuscript_section_v452": "results",
                "include_priority_v452": 1,
                "claim_boundary_v452": "validation gates only",
            },
            {
                "item_id_v452": "tbl_claim_boundaries",
                "item_type_v452": "table",
                "source_artifact_v452": "paper4_current_claim_boundaries.csv",
                "manuscript_section_v452": "methods",
                "include_priority_v452": 2,
                "claim_boundary_v452": "allowed/prohibited claims",
            },
            {
                "item_id_v452": "tbl_quarto_render",
                "item_type_v452": "table",
                "source_artifact_v452": "paper4_v449_full_book_render_probe_summary.csv",
                "manuscript_section_v452": "quarto_reproducibility",
                "include_priority_v452": 3,
                "claim_boundary_v452": "full-book render readiness",
            },
            {
                "item_id_v452": "tbl_language_bank",
                "item_type_v452": "table",
                "source_artifact_v452": "paper4_v451_publishable_language_bank.csv",
                "manuscript_section_v452": "limitations",
                "include_priority_v452": 4,
                "claim_boundary_v452": "publication language controls",
            },
            {
                "item_id_v452": "fig_regret_auditability_frontier",
                "item_type_v452": "figure",
                "source_artifact_v452": "paper4_fig9_regret_auditability_v2.png",
                "manuscript_section_v452": "results",
                "include_priority_v452": 5,
                "claim_boundary_v452": "historical Paper 4 figure; verify caption before use",
            },
        ]
    )


def _claim_language_scaffold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "language_id_v452": "abstract_gate_sentence",
                "allowed_v452": True,
                "draft_text_v452": (
                    "Across 451 living-lab waves, the current Paper 4 package "
                    "reaches bounded release-readiness under full pytest, Ruff, "
                    "chapter render, and full-book render gates."
                ),
                "section_id_v452": "abstract",
                "claim_boundary_v452": "bounded readiness only",
            },
            {
                "language_id_v452": "results_gate_sentence",
                "allowed_v452": True,
                "draft_text_v452": (
                    "The post-render validation gate passed 1188 tests with zero "
                    "repository Ruff diagnostics."
                ),
                "section_id_v452": "results",
                "claim_boundary_v452": "v450 evidence only",
            },
            {
                "language_id_v452": "quarto_sentence",
                "allowed_v452": True,
                "draft_text_v452": (
                    "The compact registered Paper 4 chapter renders both in "
                    "isolation and within the 122-page official Quarto book."
                ),
                "section_id_v452": "quarto_reproducibility",
                "claim_boundary_v452": "registered pages only",
            },
            {
                "language_id_v452": "final_paper_sentence",
                "allowed_v452": False,
                "draft_text_v452": "Paper 4 is final, submitted, or externally validated.",
                "section_id_v452": "prohibited",
                "claim_boundary_v452": "manuscript extraction and external validation are pending",
            },
            {
                "language_id_v452": "champion_replacement_sentence",
                "allowed_v452": False,
                "draft_text_v452": "Paper 4 replaces the official Paper Estrella champion.",
                "section_id_v452": "prohibited",
                "claim_boundary_v452": "no final promotion or champion update is created",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v452": "section_drafts_not_written",
                "blocking_v452": True,
                "evidence_count_v452": 7,
                "required_next_artifact_v452": NEXT_ARTIFACT,
                "claim_boundary_v452": "scaffold is not prose draft",
            },
            {
                "blocker_id_v452": "external_dataset_validation_not_run",
                "blocking_v452": True,
                "evidence_count_v452": 0,
                "required_next_artifact_v452": "future_external_validation_protocol",
                "claim_boundary_v452": "do not claim external generalization",
            },
            {
                "blocker_id_v452": "paper4_final_promotion_forbidden",
                "blocking_v452": True,
                "evidence_count_v452": 1,
                "required_next_artifact_v452": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v452": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v452_manuscript_extraction_scaffold_created",
                "allowed": True,
                "artifact": "paper4_v452_manuscript_extraction_scaffold.md",
                "boundary": "scaffold only",
            },
            {
                "claim_id": "v452_section_map_created",
                "allowed": True,
                "artifact": "paper4_v452_manuscript_section_scaffold.csv",
                "boundary": "section map, not prose draft",
            },
            {
                "claim_id": "v452_claim_language_scaffold_created",
                "allowed": True,
                "artifact": "paper4_v452_claim_language_scaffold.csv",
                "boundary": "language controls only",
            },
            {
                "claim_id": "v452_methods_results_draft_complete",
                "allowed": False,
                "artifact": "paper4_v452_remaining_blockers.csv",
                "boundary": "draft deferred to v453",
            },
            {
                "claim_id": "v452_release_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v452_remaining_blockers.csv",
                "boundary": "scaffold is not final promotion",
            },
            {
                "claim_id": "v452_working_champion_or_final_promotion",
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
                "claim": "v452 creates a manuscript extraction scaffold for Paper 4.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v452_manuscript_extraction_scaffold.md"
                ),
                "boundary": "Scaffold only; not a manuscript draft or submission.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v452 maps manuscript sections to validated artifacts.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v452_manuscript_section_scaffold.csv"
                ),
                "boundary": "Map only; section prose remains pending.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v452 completes Methods/Results prose or final manuscript extraction.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v452_remaining_blockers.csv"
                ),
                "boundary": "Draft prose is deferred to v453.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v452 makes Paper 4 final, submitted, or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v452_remaining_blockers.csv"
                ),
                "boundary": "No submission, external validation or final promotion is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v452 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v452_remaining_blockers.csv"
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
                "executable_item": "v452 creates a manuscript extraction scaffold.",
                "status": "manuscript_extraction_scaffold_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v453 drafts Methods/Results prose without final promotion",
                "last_wave": "v452",
                "execution_result": "manuscript_scaffold_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v452")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _scaffold_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manuscript Extraction Scaffold v452

Generated: {status["generated_at_utc"]}

v452 turns the bounded readiness synthesis into a manuscript extraction scaffold.
It maps sections, source artifacts, figure/table candidates, and allowed claim
language without writing a final manuscript.

## Scaffold

- Sections mapped: `{status["section_count_v452"]}`.
- Figure/table candidates shortlisted: `{status["figure_table_shortlist_count_v452"]}`.
- Claim-language rows: `{status["claim_language_row_count_v452"]}`.
- Methods/Results draft complete: `{status["methods_results_draft_complete_v452"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Immediate Draft Targets

1. Methods: SDAM/CVaR/source-governance/replay validation stack.
2. Results: v448-v450 clean validation gates and bounded readiness.
3. Limitations: no external dataset, no legal fairness certification, no final
   promotion.

## Required Caveat

This is a scaffold, not a submitted manuscript, external validation package, or
final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v452"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V452_MANUSCRIPT_EXTRACTION_SCAFFOLD_START -->"
    end = "<!-- V452_MANUSCRIPT_EXTRACTION_SCAFFOLD_END -->"
    block = f"""
{start}

## Wave v452: Manuscript Extraction Scaffold

Generated: {status["generated_at_utc"]}

### Objective

v452 maps the bounded readiness package into a manuscript extraction scaffold:
sections, source artifacts, figure/table candidates and controlled claim
language.

### Results

- Sections mapped:
  `{status["section_count_v452"]}`.
- Figure/table candidates:
  `{status["figure_table_shortlist_count_v452"]}`.
- Claim-language rows:
  `{status["claim_language_row_count_v452"]}`.
- Manuscript scaffold created:
  `{status["manuscript_extraction_scaffold_created_v452"]}`.
- Methods/Results draft complete:
  `{status["methods_results_draft_complete_v452"]}`.
- External validation complete:
  `{status["external_validation_complete_v452"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v452"]}`.

### Interpretation

The project now has a concrete extraction map for transforming validated
living-lab evidence into manuscript prose. The next step is drafting Methods and
Results while preserving the boundaries from v451.

### Claim Impact

- Allowed: manuscript extraction scaffold and section-to-artifact map.
- Still prohibited: completed manuscript, external validation, champion
  replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v452 in the living notebook. v453 should draft Methods/Results prose
without final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v451 = _read_status(451)
    if v451["next_artifact_v451"] != "paper4_v452_manuscript_extraction_scaffold.md":
        raise RuntimeError("v452 expects v451 to route to manuscript extraction scaffold.")
    if v451["all_validation_gates_clean_v451"] is not True:
        raise RuntimeError("v452 expects v451 validation gates to be clean.")

    sections = _section_scaffold()
    shortlist = _figure_table_shortlist()
    language = _claim_language_scaffold()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v452_manuscript_section_scaffold.csv", sections)
    write_csv(TABLE_DIR / "paper4_v452_figure_table_shortlist.csv", shortlist)
    write_csv(TABLE_DIR / "paper4_v452_claim_language_scaffold.csv", language)
    write_csv(TABLE_DIR / "paper4_v452_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v452_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v452_manuscript_extraction_scaffold",
        "schema_version": "2026-05-17.452",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_readiness_synthesis_version_v452": PRIOR_READINESS_SYNTHESIS_VERSION,
        "section_count_v452": len(sections),
        "figure_table_shortlist_count_v452": len(shortlist),
        "claim_language_row_count_v452": len(language),
        "allowed_language_row_count_v452": int(language["allowed_v452"].astype(bool).sum()),
        "prohibited_language_row_count_v452": int((~language["allowed_v452"].astype(bool)).sum()),
        "manuscript_extraction_scaffold_created_v452": True,
        "section_map_created_v452": True,
        "figure_table_shortlist_created_v452": True,
        "claim_language_scaffold_created_v452": True,
        "methods_results_draft_complete_v452": False,
        "manuscript_extraction_complete_v452": False,
        "external_validation_complete_v452": False,
        "working_champion_claim_allowed_v452": False,
        "paper1_promotion_allowed_v452": False,
        "paper4_working_champion_changed_v452": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v452": NEXT_ARTIFACT,
        "claim_boundary": (
            "v452 is manuscript extraction scaffold only; Methods/Results prose, "
            "external validation and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v452 must not create final Paper 4 promotion.")

    SCAFFOLD_MD.write_text(_scaffold_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v452": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
