#!/usr/bin/env python3
"""Build Paper 4 v453 Methods/Results draft artifacts."""

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

VERSION = 453
PRIOR_MANUSCRIPT_SCAFFOLD_VERSION = 452
NEXT_ARTIFACT = "paper4_v454_discussion_limitations_draft.md"
DRAFT_MD = NOTEBOOK.parent / "paper4_v453_methods_results_draft.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _section_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id_v453": "methods_living_lab_protocol",
                "draft_role_v453": "describe executable wave protocol and guardrail ledger",
                "source_artifact_v453": "paper4_living_lab_notebook.md",
                "draft_created_v453": True,
                "claim_boundary_v453": "protocol description only",
            },
            {
                "section_id_v453": "methods_validation_gates",
                "draft_role_v453": "define pytest, Ruff, Quarto, archive and no-promotion gates",
                "source_artifact_v453": "paper4_v451_release_readiness_gate_summary.csv",
                "draft_created_v453": True,
                "claim_boundary_v453": "validation gates from v448-v451",
            },
            {
                "section_id_v453": "results_post_render_validation",
                "draft_role_v453": "report v450 post-render pytest and Ruff results",
                "source_artifact_v453": "paper4_v450_validation_gate_summary.csv",
                "draft_created_v453": True,
                "claim_boundary_v453": "post-render validation results only",
            },
            {
                "section_id_v453": "results_quarto_readiness",
                "draft_role_v453": "report Paper 4 chapter and full-book render results",
                "source_artifact_v453": "paper4_v448_quarto_render_probe_summary.csv",
                "draft_created_v453": True,
                "claim_boundary_v453": "render readiness only",
            },
            {
                "section_id_v453": "results_claim_boundaries",
                "draft_role_v453": "state allowed and prohibited publication language",
                "source_artifact_v453": "paper4_v452_claim_language_scaffold.csv",
                "draft_created_v453": True,
                "claim_boundary_v453": "language controls only",
            },
        ]
    )


def _claim_sentence_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id_v453": "methods_wave_protocol",
                "allowed_v453": True,
                "sentence_v453": (
                    "Each living-lab wave materializes a versioned status file, "
                    "tables, a notebook entry, and a guardrail test before the "
                    "next executable item is selected."
                ),
                "source_artifact_v453": "paper4_living_lab_notebook.md",
                "claim_boundary_v453": "process description",
            },
            {
                "sentence_id_v453": "results_pytest_ruff",
                "allowed_v453": True,
                "sentence_v453": (
                    "After the full-book render probe, the repository passed "
                    "1188 tests with 2 skipped tests, 13 warnings, and zero Ruff "
                    "diagnostics."
                ),
                "source_artifact_v453": "paper4_v450_validation_gate_summary.csv",
                "claim_boundary_v453": "v450 validation evidence",
            },
            {
                "sentence_id_v453": "results_quarto",
                "allowed_v453": True,
                "sentence_v453": (
                    "The registered Paper 4 chapter rendered as 10 official pages, "
                    "and the full Quarto book rendered 122 registered pages."
                ),
                "source_artifact_v453": "paper4_v448_quarto_render_probe_summary.csv",
                "claim_boundary_v453": "registered Quarto pages only",
            },
            {
                "sentence_id_v453": "results_archive",
                "allowed_v453": True,
                "sentence_v453": (
                    "The historical Paper 4 archive remains preserved on disk while "
                    "excluded from the official rendered chapter."
                ),
                "source_artifact_v453": "paper4_v449_paper4_surface_in_full_book.csv",
                "claim_boundary_v453": "archive policy only",
            },
            {
                "sentence_id_v453": "prohibited_final",
                "allowed_v453": False,
                "sentence_v453": "These results make Paper 4 final or externally validated.",
                "source_artifact_v453": "paper4_v453_remaining_blockers.csv",
                "claim_boundary_v453": "final/external claims remain blocked",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v453": "discussion_limitations_not_drafted",
                "blocking_v453": True,
                "evidence_count_v453": 1,
                "required_next_artifact_v453": NEXT_ARTIFACT,
                "claim_boundary_v453": "Methods/Results draft needs discussion and limitations",
            },
            {
                "blocker_id_v453": "abstract_conclusion_not_drafted",
                "blocking_v453": True,
                "evidence_count_v453": 1,
                "required_next_artifact_v453": "paper4_v455_abstract_conclusion_draft.md",
                "claim_boundary_v453": "front/back matter remains pending",
            },
            {
                "blocker_id_v453": "external_dataset_validation_not_run",
                "blocking_v453": True,
                "evidence_count_v453": 0,
                "required_next_artifact_v453": "future_external_validation_protocol",
                "claim_boundary_v453": "do not claim external generalization",
            },
            {
                "blocker_id_v453": "paper4_final_promotion_forbidden",
                "blocking_v453": True,
                "evidence_count_v453": 1,
                "required_next_artifact_v453": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v453": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v453_methods_results_draft_created",
                "allowed": True,
                "artifact": "paper4_v453_methods_results_draft.md",
                "boundary": "Methods/Results prose only",
            },
            {
                "claim_id": "v453_claim_sentence_trace_created",
                "allowed": True,
                "artifact": "paper4_v453_claim_sentence_trace.csv",
                "boundary": "sentence-to-artifact trace",
            },
            {
                "claim_id": "v453_discussion_limitations_complete",
                "allowed": False,
                "artifact": "paper4_v453_remaining_blockers.csv",
                "boundary": "deferred to v454",
            },
            {
                "claim_id": "v453_complete_manuscript_or_submission",
                "allowed": False,
                "artifact": "paper4_v453_remaining_blockers.csv",
                "boundary": "draft is partial",
            },
            {
                "claim_id": "v453_external_validation_complete",
                "allowed": False,
                "artifact": "paper4_v453_remaining_blockers.csv",
                "boundary": "no external validation run",
            },
            {
                "claim_id": "v453_working_champion_or_final_promotion",
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
                "claim": "v453 drafts Methods/Results prose from validated Paper 4 gates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v453_methods_results_draft.md"
                ),
                "boundary": "Partial manuscript draft only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v453 traces draft sentences to Paper 4 artifacts.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v453_claim_sentence_trace.csv"
                ),
                "boundary": "Traceability for drafted Methods/Results sentences.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": (
                    "v453 completes discussion, limitations, abstract, conclusion or submission."
                ),
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v453_remaining_blockers.csv"
                ),
                "boundary": "Those sections remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v453 makes Paper 4 final or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v453_remaining_blockers.csv"
                ),
                "boundary": "Partial draft does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v453 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v453_remaining_blockers.csv"
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
                "executable_item": "v453 drafts Methods/Results prose from validated gates.",
                "status": "methods_results_draft_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v454 drafts discussion/limitations without final promotion",
                "last_wave": "v453",
                "execution_result": "methods_results_draft_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v453")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _draft_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Methods/Results Draft v453

Generated: {status["generated_at_utc"]}

## Methods Draft

Paper 4 is maintained as a living-lab protocol in which each executable wave
produces a versioned status file, tabular evidence, a notebook entry, and a
guardrail test. The current validation surface is defined by four operational
gates: full repository pytest, repository Ruff, Quarto rendering of the compact
registered Paper 4 chapter, and rendering of the full official Quarto book. A
negative promotion gate is also enforced: the final Paper 4 promotion artifact
must remain absent.

The manuscript extraction is therefore evidence-first. Claims are admitted only
when they can be traced to generated artifacts, and every claim boundary is kept
in `paper4_current_claim_boundaries.csv`. Historical Paper 4 pages remain on
disk as an archive, while the official book renders only the compact registered
Paper 4 surface.

## Results Draft

After the full-book render probe, the repository passed 1188 tests with 2
skipped tests, 13 warnings, and zero Ruff diagnostics. The official Paper 4
chapter rendered as 10 registered pages, and the full Quarto book rendered 122
registered pages. The archive policy remained clean: historical Paper 4 files
were preserved on disk but excluded from the official rendered chapter.

Together, these gates support a bounded readiness statement: the living-lab
evidence package is internally reproducible and ready for manuscript extraction.
They do not support final-paper, submission, deployment, legal fairness,
external-validation, or champion-replacement claims.

## Required Caveat

v453 is a partial Methods/Results draft. Discussion, limitations, abstract,
conclusion, external validation, and final promotion remain blocked.

## Next Executable Wave

Build `{status["next_artifact_v453"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V453_METHODS_RESULTS_DRAFT_START -->"
    end = "<!-- V453_METHODS_RESULTS_DRAFT_END -->"
    block = f"""
{start}

## Wave v453: Methods/Results Draft

Generated: {status["generated_at_utc"]}

### Objective

v453 drafts Methods/Results prose from validated Paper 4 gates and records a
sentence-to-artifact trace.

### Results

- Draft sections created:
  `{status["draft_section_count_v453"]}`.
- Traceable claim sentences:
  `{status["claim_sentence_count_v453"]}`.
- Allowed claim sentences:
  `{status["allowed_sentence_count_v453"]}`.
- Methods/Results draft created:
  `{status["methods_results_draft_created_v453"]}`.
- Discussion/limitations complete:
  `{status["discussion_limitations_complete_v453"]}`.
- Complete manuscript:
  `{status["complete_manuscript_v453"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v453"]}`.

### Interpretation

Methods/Results prose now exists as a traceable draft, but the manuscript is not
complete. The next needed section is discussion/limitations, especially the
negative claims around external validation, legal fairness and final promotion.

### Claim Impact

- Allowed: partial Methods/Results draft and sentence-to-artifact trace.
- Still prohibited: complete manuscript, submission, external validation,
  champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v453 in the living notebook. v454 should draft discussion/limitations
without final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v452 = _read_status(452)
    if v452["next_artifact_v452"] != "paper4_v453_methods_results_draft.md":
        raise RuntimeError("v453 expects v452 to route to Methods/Results draft.")
    if v452["manuscript_extraction_scaffold_created_v452"] is not True:
        raise RuntimeError("v453 expects v452 scaffold to exist.")

    sections = _section_inventory()
    trace = _claim_sentence_trace()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v453_draft_section_inventory.csv", sections)
    write_csv(TABLE_DIR / "paper4_v453_claim_sentence_trace.csv", trace)
    write_csv(TABLE_DIR / "paper4_v453_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v453_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v453_methods_results_draft",
        "schema_version": "2026-05-17.453",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manuscript_scaffold_version_v453": PRIOR_MANUSCRIPT_SCAFFOLD_VERSION,
        "draft_section_count_v453": len(sections),
        "claim_sentence_count_v453": len(trace),
        "allowed_sentence_count_v453": int(trace["allowed_v453"].astype(bool).sum()),
        "prohibited_sentence_count_v453": int((~trace["allowed_v453"].astype(bool)).sum()),
        "methods_results_draft_created_v453": True,
        "claim_sentence_trace_created_v453": True,
        "discussion_limitations_complete_v453": False,
        "abstract_conclusion_complete_v453": False,
        "complete_manuscript_v453": False,
        "external_validation_complete_v453": False,
        "working_champion_claim_allowed_v453": False,
        "paper1_promotion_allowed_v453": False,
        "paper4_working_champion_changed_v453": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v453": NEXT_ARTIFACT,
        "claim_boundary": (
            "v453 is partial Methods/Results draft only; discussion, limitations, "
            "external validation and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v453 must not create final Paper 4 promotion.")

    DRAFT_MD.write_text(_draft_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v453": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
