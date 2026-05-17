#!/usr/bin/env python3
"""Build Paper 4 v456 manuscript assembly packet artifacts."""

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

VERSION = 456
PRIOR_ABSTRACT_CONCLUSION_VERSION = 455
NEXT_ARTIFACT = "paper4_v457_post_assembly_pytest_probe.md"
NOTES_DIR = NOTEBOOK.parent
ASSEMBLY_MD = NOTES_DIR / "paper4_v456_manuscript_assembly_packet.md"
GOAL_PROMPT_MD = NOTES_DIR / "paper4_v456_goal_prompt.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _source_component_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component_id_v456": "bounded_readiness_synthesis",
                "source_version_v456": 451,
                "source_artifact_v456": "paper4_v451_release_readiness_synthesis.md",
                "used_in_packet_v456": True,
                "assembly_role_v456": "validation status and bounded readiness language",
                "claim_boundary_v456": "internal readiness only",
            },
            {
                "component_id_v456": "manuscript_scaffold",
                "source_version_v456": 452,
                "source_artifact_v456": "paper4_v452_manuscript_extraction_scaffold.md",
                "used_in_packet_v456": True,
                "assembly_role_v456": "section order and manuscript extraction map",
                "claim_boundary_v456": "scaffold only",
            },
            {
                "component_id_v456": "methods_results_draft",
                "source_version_v456": 453,
                "source_artifact_v456": "paper4_v453_methods_results_draft.md",
                "used_in_packet_v456": True,
                "assembly_role_v456": "methods and results prose",
                "claim_boundary_v456": "draft prose only",
            },
            {
                "component_id_v456": "discussion_limitations_draft",
                "source_version_v456": 454,
                "source_artifact_v456": "paper4_v454_discussion_limitations_draft.md",
                "used_in_packet_v456": True,
                "assembly_role_v456": "discussion, limitations and caveats",
                "claim_boundary_v456": "draft prose only",
            },
            {
                "component_id_v456": "abstract_conclusion_draft",
                "source_version_v456": 455,
                "source_artifact_v456": "paper4_v455_abstract_conclusion_draft.md",
                "used_in_packet_v456": True,
                "assembly_role_v456": "abstract and conclusion prose",
                "claim_boundary_v456": "draft prose only",
            },
        ]
    )


def _assembly_section_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id_v456": "abstract",
                "assembled_order_v456": 1,
                "section_title_v456": "Abstract",
                "source_artifact_v456": "paper4_v455_abstract_conclusion_draft.md",
                "source_version_v456": 455,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "front matter draft only",
            },
            {
                "section_id_v456": "introduction_scope",
                "assembled_order_v456": 2,
                "section_title_v456": "Introduction and Scope",
                "source_artifact_v456": "paper4_v452_manuscript_extraction_scaffold.md",
                "source_version_v456": 452,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "problem framing without deployment claim",
            },
            {
                "section_id_v456": "methods",
                "assembled_order_v456": 3,
                "section_title_v456": "Methods",
                "source_artifact_v456": "paper4_v453_methods_results_draft.md",
                "source_version_v456": 453,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "execution protocol only",
            },
            {
                "section_id_v456": "results",
                "assembled_order_v456": 4,
                "section_title_v456": "Results",
                "source_artifact_v456": "paper4_v453_methods_results_draft.md",
                "source_version_v456": 453,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "internal validation gates only",
            },
            {
                "section_id_v456": "discussion_limitations",
                "assembled_order_v456": 5,
                "section_title_v456": "Discussion and Limitations",
                "source_artifact_v456": "paper4_v454_discussion_limitations_draft.md",
                "source_version_v456": 454,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "bounded interpretation and caveats",
            },
            {
                "section_id_v456": "conclusion",
                "assembled_order_v456": 6,
                "section_title_v456": "Conclusion",
                "source_artifact_v456": "paper4_v455_abstract_conclusion_draft.md",
                "source_version_v456": 455,
                "included_in_packet_v456": True,
                "claim_boundary_v456": "next-step conclusion only",
            },
        ]
    )


def _readiness_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id_v456": "major_prose_components_assembled",
                "ready_v456": True,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "paper4_v456_manuscript_assembly_packet.md",
                "next_action_v456": "run post-assembly regression checks",
                "claim_boundary_v456": "assembly packet exists",
            },
            {
                "gate_id_v456": "claim_boundaries_attached",
                "ready_v456": True,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "paper4_current_claim_boundaries.csv",
                "next_action_v456": "keep claim map synchronized",
                "claim_boundary_v456": "bounded language only",
            },
            {
                "gate_id_v456": "post_assembly_full_pytest",
                "ready_v456": False,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "paper4_v457_post_assembly_pytest_probe.md",
                "next_action_v456": "execute full pytest after manuscript assembly",
                "claim_boundary_v456": "regression refresh pending",
            },
            {
                "gate_id_v456": "paper4_quarto_render_after_assembly",
                "ready_v456": False,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "future_post_assembly_quarto_probe",
                "next_action_v456": "rerender Paper 4 surface if Quarto content changes",
                "claim_boundary_v456": "render refresh pending if promoted to Quarto",
            },
            {
                "gate_id_v456": "external_dataset_validation",
                "ready_v456": False,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "future_external_validation_protocol",
                "next_action_v456": (
                    "design external validation protocol if broader claim is needed"
                ),
                "claim_boundary_v456": "no external generalization claim",
            },
            {
                "gate_id_v456": "submission_package_ready",
                "ready_v456": False,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "future_target_venue_packet",
                "next_action_v456": "choose target venue and format requirements",
                "claim_boundary_v456": "not submission ready",
            },
            {
                "gate_id_v456": "final_promotion_absent",
                "ready_v456": True,
                "blocking_if_false_v456": True,
                "evidence_artifact_v456": "paper4_final_promotion_gate_not_created",
                "next_action_v456": "keep forbidden promotion artifact absent",
                "claim_boundary_v456": "Paper Estrella remains protected",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v456": "post_assembly_full_pytest_not_run",
                "blocking_v456": True,
                "evidence_count_v456": 1,
                "required_next_artifact_v456": NEXT_ARTIFACT,
                "claim_boundary_v456": "run regression after assembly before stronger claim",
            },
            {
                "blocker_id_v456": "post_assembly_render_not_refreshed",
                "blocking_v456": True,
                "evidence_count_v456": 0,
                "required_next_artifact_v456": "future_post_assembly_quarto_probe",
                "claim_boundary_v456": "Quarto refresh remains conditional future work",
            },
            {
                "blocker_id_v456": "external_dataset_validation_not_run",
                "blocking_v456": True,
                "evidence_count_v456": 0,
                "required_next_artifact_v456": "future_external_validation_protocol",
                "claim_boundary_v456": "do not claim external generalization",
            },
            {
                "blocker_id_v456": "target_venue_not_selected",
                "blocking_v456": True,
                "evidence_count_v456": 0,
                "required_next_artifact_v456": "future_target_venue_packet",
                "claim_boundary_v456": "do not claim submission readiness",
            },
            {
                "blocker_id_v456": "paper4_final_promotion_forbidden",
                "blocking_v456": True,
                "evidence_count_v456": 1,
                "required_next_artifact_v456": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v456": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v456_manuscript_assembly_packet_created",
                "allowed": True,
                "artifact": "paper4_v456_manuscript_assembly_packet.md",
                "boundary": "assembled draft packet only",
            },
            {
                "claim_id": "v456_source_sections_mapped",
                "allowed": True,
                "artifact": "paper4_v456_assembly_section_map.csv",
                "boundary": "source-to-section map only",
            },
            {
                "claim_id": "v456_reusable_goal_prompt_created",
                "allowed": True,
                "artifact": "paper4_v456_goal_prompt.md",
                "boundary": "continued-work prompt only",
            },
            {
                "claim_id": "v456_post_assembly_regression_complete",
                "allowed": False,
                "artifact": "paper4_v456_manuscript_readiness_matrix.csv",
                "boundary": "full pytest rerun deferred to v457",
            },
            {
                "claim_id": "v456_submission_ready_or_external_validation",
                "allowed": False,
                "artifact": "paper4_v456_remaining_blockers.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v456_working_champion_or_final_promotion",
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
                "claim": "v456 assembles a Paper 4 manuscript packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v456_manuscript_assembly_packet.md"
                ),
                "boundary": "Assembly packet only; not submission-ready manuscript.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v456 maps assembled manuscript sections to source artifacts.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v456_assembly_section_map.csv"
                ),
                "boundary": "Traceability map only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v456 provides a reusable goal prompt for continued Paper 4 work.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/paper4_v456_goal_prompt.md"
                ),
                "boundary": "Prompt preserves no-promotion and bounded-claim rules.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v456 completes post-assembly full pytest and render validation.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v456_manuscript_readiness_matrix.csv"
                ),
                "boundary": "Regression and render refresh remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v456 makes Paper 4 final, submitted, or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v456_remaining_blockers.csv"
                ),
                "boundary": "Assembly packet does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v456 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v456_remaining_blockers.csv"
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
                "lane": "Validation",
                "executable_item": "v456 assembles manuscript packet and goal prompt.",
                "status": "manuscript_assembly_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v457 runs full pytest after manuscript assembly",
                "last_wave": "v456",
                "execution_result": "manuscript_packet_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v456")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _assembly_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Manuscript Assembly Packet v456

Generated: {status["generated_at_utc"]}

## Working Title

Auditable Living-Lab Protocols for Credit-Risk Decision Research

## Assembly Claim

This packet assembles the current Paper 4 prose components into a traceable
manuscript draft package. It is not a final manuscript, submission package,
external validation result, deployment policy, or Paper Estrella replacement.

## Abstract

We study Paper 4 as a living-lab protocol for auditable credit-risk decision
research rather than as a final deployed policy. The current package converts
hundreds of exploratory waves into a bounded evidence stack with full pytest,
repository Ruff, Paper 4 Quarto render, full-book render, archive-governance,
and no-promotion gates all clean. These gates support manuscript extraction, but
not external validation, legal fairness certification, submission readiness, or
final promotion.

## Introduction and Scope

Paper 4 is framed as a reproducible laboratory for stress-testing constrained
credit-risk decision research. The contribution is not a new deployment
decision. The contribution is a guarded execution protocol: each wave leaves a
status artifact, tabular evidence, claim boundaries, and a regression or render
gate when the wave touches executable behavior.

## Methods

Paper 4 is maintained as a living-lab protocol in which each executable wave
produces a versioned status file, tabular evidence, a notebook entry, and a
guardrail test. The current validation surface is defined by full repository
pytest, repository Ruff, Quarto rendering of the compact registered Paper 4
chapter, rendering of the full official Quarto book, and a negative promotion
gate requiring the final Paper 4 promotion artifact to remain absent.

## Results

The latest bounded readiness stack includes a clean post-render full pytest
probe, zero repository Ruff diagnostics, a clean official Paper 4 Quarto chapter
render, a clean full-book render, preserved archive governance, and an absent
final promotion artifact. These results support internal reproducibility and
manuscript extraction, not external generalization or submission readiness.

## Discussion and Limitations

The strongest current result is the conversion of exploratory research into a
traceable evidence package. The narrow readiness claim is useful because every
admitted claim points to a generated artifact and every prohibited claim remains
represented in the claim-boundary tables. The package does not establish
external generalization, legal fair-lending certification, deployment readiness,
or champion replacement.

## Conclusion

The current Paper 4 laboratory turns many execution waves into a guarded
manuscript source. The assembled packet can now drive post-assembly regression
checks and later target-venue editing, while preserving the boundary that Paper
4 is not final, submitted, externally validated, or promoted.

## Immediate Goal Prompt

Use `{status["goal_prompt_artifact_v456"]}` to continue with `{status["next_artifact_v456"]}`.

## Required Caveat

v456 assembles a manuscript packet and reusable goal prompt only. It does not
complete post-assembly full pytest, external validation, target-venue formatting,
submission, Paper Estrella replacement, or final Paper 4 promotion.
"""


def _goal_prompt_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Executable Goal Prompt v456

Generated: {status["generated_at_utc"]}

Goal:
Continue the Paper 4 living-lab from v456 by executing the next highest-value
pending waves while preserving all claim boundaries.

Non-negotiable rules:
- Do not create `reports/paper_material/paper4/status/paper4_final_promotion.json`.
- Do not claim Paper 4 is final, submitted, externally validated, deployed, or
  a replacement for Paper Estrella.
- Keep every new wave traceable through a status JSON, tables, notebook entry,
  guardrail test, and small commit.
- Restore model side effects if full pytest rewrites model status artifacts.

Immediate executable queue:
1. Build `{status["next_artifact_v456"]}` by running a post-assembly full pytest
   probe and recording pass/fail evidence.
2. If clean, add a v457 guardrail and commit/push the result.
3. Decide whether v458 should rerender Paper 4/full book or refine the assembled
   manuscript for target-venue structure.
4. Keep external validation as a separate future protocol unless the data and
   scope are explicitly available.

Useful current evidence:
- v451 bounded readiness synthesis.
- v452 manuscript extraction scaffold.
- v453 Methods/Results draft.
- v454 Discussion/Limitations draft.
- v455 Abstract/Conclusion draft.
- v456 manuscript assembly packet.

Success condition:
Produce the next executable artifact without weakening the no-promotion,
no-champion-replacement, no-external-validation, and no-submission boundaries.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V456_MANUSCRIPT_ASSEMBLY_PACKET_START -->"
    end = "<!-- V456_MANUSCRIPT_ASSEMBLY_PACKET_END -->"
    block = f"""
{start}

## Wave v456: Manuscript Assembly Packet

Generated: {status["generated_at_utc"]}

### Objective

v456 assembles the current Paper 4 prose components into a manuscript packet and
records a reusable continuation goal prompt.

### Results

- Source components used:
  `{status["source_component_count_v456"]}`.
- Assembled sections:
  `{status["assembled_section_count_v456"]}`.
- Readiness gates recorded:
  `{status["readiness_gate_count_v456"]}`.
- Ready gates:
  `{status["ready_gate_count_v456"]}`.
- Blocking gaps:
  `{status["blocking_gap_count_v456"]}`.
- Assembly packet created:
  `{status["manuscript_assembly_packet_created_v456"]}`.
- Goal prompt created:
  `{status["goal_prompt_created_v456"]}`.
- Post-assembly full pytest run:
  `{status["post_assembly_full_pytest_run_v456"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v456"]}`.

### Interpretation

The major manuscript components are now assembled into one packet, but this is
not yet a submission-ready manuscript. The next evidence-producing wave is a
post-assembly full pytest probe.

### Claim Impact

- Allowed: manuscript assembly packet, source-section traceability, reusable
  continuation goal prompt.
- Still prohibited: post-assembly regression completion, external validation,
  submission readiness, champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v456 in the living notebook. v457 should run post-assembly regression
checks before any stronger manuscript-readiness language.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v455 = _read_status(455)
    if v455["next_artifact_v455"] != "paper4_v456_manuscript_assembly_packet.md":
        raise RuntimeError("v456 expects v455 to route to manuscript assembly.")
    if v455["major_prose_components_created_v455"] is not True:
        raise RuntimeError("v456 expects major prose components from v455.")

    components = _source_component_inventory()
    sections = _assembly_section_map()
    readiness = _readiness_matrix()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v456_source_component_inventory.csv", components)
    write_csv(TABLE_DIR / "paper4_v456_assembly_section_map.csv", sections)
    write_csv(TABLE_DIR / "paper4_v456_manuscript_readiness_matrix.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v456_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v456_claim_matrix_delta.csv", claim_matrix)

    ready_gates = int(readiness["ready_v456"].astype(bool).sum())
    blocking_gaps = int(
        ((~readiness["ready_v456"].astype(bool)) & readiness["blocking_if_false_v456"]).sum()
    )
    status = {
        "phase": "v456_manuscript_assembly_packet",
        "schema_version": "2026-05-17.456",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_abstract_conclusion_version_v456": PRIOR_ABSTRACT_CONCLUSION_VERSION,
        "source_component_count_v456": len(components),
        "assembled_section_count_v456": len(sections),
        "readiness_gate_count_v456": len(readiness),
        "ready_gate_count_v456": ready_gates,
        "blocking_gap_count_v456": blocking_gaps,
        "manuscript_assembly_packet_created_v456": True,
        "source_sections_mapped_v456": True,
        "goal_prompt_created_v456": True,
        "complete_manuscript_ready_v456": False,
        "post_assembly_full_pytest_run_v456": False,
        "submission_package_ready_v456": False,
        "external_validation_complete_v456": False,
        "working_champion_claim_allowed_v456": False,
        "paper1_promotion_allowed_v456": False,
        "paper4_working_champion_changed_v456": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "goal_prompt_artifact_v456": "paper4_v456_goal_prompt.md",
        "next_artifact_v456": NEXT_ARTIFACT,
        "claim_boundary": (
            "v456 is a manuscript assembly packet and goal prompt only; "
            "post-assembly pytest, external validation, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v456 must not create final Paper 4 promotion.")

    ASSEMBLY_MD.write_text(_assembly_markdown(status), encoding="utf-8")
    GOAL_PROMPT_MD.write_text(_goal_prompt_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v456": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
