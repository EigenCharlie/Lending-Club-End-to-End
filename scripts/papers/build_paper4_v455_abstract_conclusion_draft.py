#!/usr/bin/env python3
"""Build Paper 4 v455 abstract/conclusion draft artifacts."""

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

VERSION = 455
PRIOR_DISCUSSION_LIMITATIONS_VERSION = 454
NEXT_ARTIFACT = "paper4_v456_manuscript_assembly_packet.md"
DRAFT_MD = NOTEBOOK.parent / "paper4_v455_abstract_conclusion_draft.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _sentence_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id_v455": "abstract_problem",
                "section_v455": "abstract",
                "allowed_v455": True,
                "sentence_v455": (
                    "We study Paper 4 as a living-lab protocol for auditable "
                    "credit-risk decision research rather than as a final deployed policy."
                ),
                "source_artifact_v455": "paper4_v453_methods_results_draft.md",
                "claim_boundary_v455": "problem framing only",
            },
            {
                "sentence_id_v455": "abstract_evidence",
                "section_v455": "abstract",
                "allowed_v455": True,
                "sentence_v455": (
                    "The current package passes full pytest, repository Ruff, "
                    "Paper 4 Quarto render, full-book render, archive-governance, "
                    "and no-promotion gates."
                ),
                "source_artifact_v455": "paper4_v451_release_readiness_gate_summary.csv",
                "claim_boundary_v455": "bounded readiness gates",
            },
            {
                "sentence_id_v455": "abstract_caveat",
                "section_v455": "abstract",
                "allowed_v455": True,
                "sentence_v455": (
                    "These gates support manuscript extraction but not external "
                    "validation, legal fairness certification, submission readiness, "
                    "or final promotion."
                ),
                "source_artifact_v455": "paper4_v454_limitation_register.csv",
                "claim_boundary_v455": "required caveat",
            },
            {
                "sentence_id_v455": "conclusion_value",
                "section_v455": "conclusion",
                "allowed_v455": True,
                "sentence_v455": (
                    "The main value of the current Paper 4 laboratory is the "
                    "conversion of many exploratory waves into a traceable, "
                    "guarded evidence package."
                ),
                "source_artifact_v455": "paper4_living_lab_notebook.md",
                "claim_boundary_v455": "laboratory contribution",
            },
            {
                "sentence_id_v455": "conclusion_next",
                "section_v455": "conclusion",
                "allowed_v455": True,
                "sentence_v455": (
                    "The next executable step is manuscript assembly, followed by "
                    "external validation protocol design if a broader claim is needed."
                ),
                "source_artifact_v455": "paper4_v454_remaining_blockers.csv",
                "claim_boundary_v455": "future work",
            },
            {
                "sentence_id_v455": "prohibited_final",
                "section_v455": "prohibited",
                "allowed_v455": False,
                "sentence_v455": (
                    "The abstract may state that Paper 4 is final and submission-ready."
                ),
                "source_artifact_v455": "paper4_v455_remaining_blockers.csv",
                "claim_boundary_v455": "final/submission claims remain blocked",
            },
        ]
    )


def _draft_completion_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draft_component_v455": "methods_results",
                "source_version_v455": 453,
                "created_v455": True,
                "source_artifact_v455": "paper4_v453_methods_results_draft.md",
                "claim_boundary_v455": "partial prose exists",
            },
            {
                "draft_component_v455": "discussion_limitations",
                "source_version_v455": 454,
                "created_v455": True,
                "source_artifact_v455": "paper4_v454_discussion_limitations_draft.md",
                "claim_boundary_v455": "partial prose exists",
            },
            {
                "draft_component_v455": "abstract_conclusion",
                "source_version_v455": 455,
                "created_v455": True,
                "source_artifact_v455": "paper4_v455_abstract_conclusion_draft.md",
                "claim_boundary_v455": "partial prose exists",
            },
            {
                "draft_component_v455": "assembled_manuscript",
                "source_version_v455": 456,
                "created_v455": False,
                "source_artifact_v455": "paper4_v456_manuscript_assembly_packet.md",
                "claim_boundary_v455": "assembly deferred",
            },
            {
                "draft_component_v455": "external_validation_protocol",
                "source_version_v455": 0,
                "created_v455": False,
                "source_artifact_v455": "future_external_validation_protocol",
                "claim_boundary_v455": "future validation work",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v455": "manuscript_not_assembled",
                "blocking_v455": True,
                "evidence_count_v455": 1,
                "required_next_artifact_v455": NEXT_ARTIFACT,
                "claim_boundary_v455": "major prose pieces exist but are not assembled",
            },
            {
                "blocker_id_v455": "post_assembly_full_pytest_not_run",
                "blocking_v455": True,
                "evidence_count_v455": 1,
                "required_next_artifact_v455": "paper4_v457_post_assembly_pytest_probe.md",
                "claim_boundary_v455": "assembly changes would need regression refresh",
            },
            {
                "blocker_id_v455": "external_dataset_validation_not_run",
                "blocking_v455": True,
                "evidence_count_v455": 0,
                "required_next_artifact_v455": "future_external_validation_protocol",
                "claim_boundary_v455": "do not claim external generalization",
            },
            {
                "blocker_id_v455": "paper4_final_promotion_forbidden",
                "blocking_v455": True,
                "evidence_count_v455": 1,
                "required_next_artifact_v455": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v455": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v455_abstract_conclusion_draft_created",
                "allowed": True,
                "artifact": "paper4_v455_abstract_conclusion_draft.md",
                "boundary": "abstract/conclusion prose only",
            },
            {
                "claim_id": "v455_major_prose_components_created",
                "allowed": True,
                "artifact": "paper4_v455_draft_completion_inventory.csv",
                "boundary": "major partial prose components exist",
            },
            {
                "claim_id": "v455_assembled_manuscript_complete",
                "allowed": False,
                "artifact": "paper4_v455_remaining_blockers.csv",
                "boundary": "assembly deferred to v456",
            },
            {
                "claim_id": "v455_submission_ready_or_external_validation",
                "allowed": False,
                "artifact": "paper4_v455_remaining_blockers.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v455_working_champion_or_final_promotion",
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
                "claim": "v455 drafts Paper 4 abstract and conclusion prose.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v455_abstract_conclusion_draft.md"
                ),
                "boundary": "Partial prose only; not assembled manuscript.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v455 creates the major prose components for manuscript assembly.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v455_draft_completion_inventory.csv"
                ),
                "boundary": "Components exist; assembly is deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v455 completes assembled manuscript, submission package, or validation.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v455_remaining_blockers.csv"
                ),
                "boundary": "Assembly, submission and validation remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v455 makes Paper 4 final, submitted, or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v455_remaining_blockers.csv"
                ),
                "boundary": "Partial draft does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v455 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v455_remaining_blockers.csv"
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
                "executable_item": "v455 drafts abstract and conclusion prose.",
                "status": "abstract_conclusion_draft_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v456 assembles a manuscript packet without final promotion",
                "last_wave": "v455",
                "execution_result": "abstract_conclusion_draft_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v455")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _draft_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Abstract/Conclusion Draft v455

Generated: {status["generated_at_utc"]}

## Abstract Draft

We study Paper 4 as a living-lab protocol for auditable credit-risk decision
research rather than as a final deployed policy. The current package converts
hundreds of exploratory waves into a bounded evidence stack with full pytest,
repository Ruff, Paper 4 Quarto render, full-book render, archive-governance,
and no-promotion gates all clean. These gates support manuscript extraction, but
not external validation, legal fairness certification, submission readiness, or
final promotion.

## Conclusion Draft

The current Paper 4 laboratory is valuable because it turns exploratory research
into a traceable, guarded evidence package. Methods/Results, Discussion/
Limitations, and Abstract/Conclusion draft components now exist, each tied to
claim boundaries and generated artifacts. The next executable step is manuscript
assembly, followed by a regression refresh if assembly changes the repository.

## Required Caveat

v455 creates abstract/conclusion prose but does not assemble the manuscript,
submit the paper, externally validate it, replace Paper Estrella, or promote
Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v455"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V455_ABSTRACT_CONCLUSION_DRAFT_START -->"
    end = "<!-- V455_ABSTRACT_CONCLUSION_DRAFT_END -->"
    block = f"""
{start}

## Wave v455: Abstract/Conclusion Draft

Generated: {status["generated_at_utc"]}

### Objective

v455 drafts abstract/conclusion prose after v453 Methods/Results and v454
Discussion/Limitations.

### Results

- Abstract/conclusion sentences:
  `{status["abstract_conclusion_sentence_count_v455"]}`.
- Allowed abstract/conclusion sentences:
  `{status["allowed_abstract_conclusion_sentence_count_v455"]}`.
- Major prose components created:
  `{status["major_prose_component_count_v455"]}`.
- Abstract/conclusion draft created:
  `{status["abstract_conclusion_draft_created_v455"]}`.
- Assembled manuscript complete:
  `{status["assembled_manuscript_complete_v455"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v455"]}`.

### Interpretation

The major prose components now exist, but they are not yet assembled into a
manuscript packet and have not been rerun through a post-assembly validation
cycle.

### Claim Impact

- Allowed: abstract/conclusion draft and major prose component inventory.
- Still prohibited: assembled manuscript, submission, external validation,
  champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v455 in the living notebook. v456 should assemble a manuscript packet
without final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v454 = _read_status(454)
    if v454["next_artifact_v454"] != "paper4_v455_abstract_conclusion_draft.md":
        raise RuntimeError("v455 expects v454 to route to abstract/conclusion draft.")
    if v454["discussion_limitations_draft_created_v454"] is not True:
        raise RuntimeError("v455 expects v454 discussion/limitations draft to exist.")

    sentences = _sentence_inventory()
    completion = _draft_completion_inventory()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v455_abstract_conclusion_sentences.csv", sentences)
    write_csv(TABLE_DIR / "paper4_v455_draft_completion_inventory.csv", completion)
    write_csv(TABLE_DIR / "paper4_v455_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v455_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v455_abstract_conclusion_draft",
        "schema_version": "2026-05-17.455",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_discussion_limitations_version_v455": PRIOR_DISCUSSION_LIMITATIONS_VERSION,
        "abstract_conclusion_sentence_count_v455": len(sentences),
        "allowed_abstract_conclusion_sentence_count_v455": int(
            sentences["allowed_v455"].astype(bool).sum()
        ),
        "prohibited_abstract_conclusion_sentence_count_v455": int(
            (~sentences["allowed_v455"].astype(bool)).sum()
        ),
        "major_prose_component_count_v455": int(completion["created_v455"].astype(bool).sum()),
        "abstract_conclusion_draft_created_v455": True,
        "major_prose_components_created_v455": True,
        "assembled_manuscript_complete_v455": False,
        "post_assembly_full_pytest_run_v455": False,
        "external_validation_complete_v455": False,
        "working_champion_claim_allowed_v455": False,
        "paper1_promotion_allowed_v455": False,
        "paper4_working_champion_changed_v455": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v455": NEXT_ARTIFACT,
        "claim_boundary": (
            "v455 is abstract/conclusion draft only; manuscript assembly, "
            "post-assembly pytest, external validation and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v455 must not create final Paper 4 promotion.")

    DRAFT_MD.write_text(_draft_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v455": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
