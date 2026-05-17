#!/usr/bin/env python3
"""Build Paper 4 v459 target-venue structure packet artifacts."""

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

VERSION = 459
PRIOR_RENDER_DECISION_VERSION = 458
NEXT_ARTIFACT = "paper4_v460_related_work_citation_gap_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v459_target_venue_structure_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _candidate_structure_families() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "structure_family_v459": "reproducibility_artifact_paper",
                "fit_level_v459": "high",
                "selected_v459": False,
                "why_useful_v459": "matches living-lab gates, artifacts and auditability",
                "missing_before_selection_v459": "venue-specific requirements and citations",
                "claim_boundary_v459": "candidate family only",
            },
            {
                "structure_family_v459": "credit_risk_decision_science_paper",
                "fit_level_v459": "medium",
                "selected_v459": False,
                "why_useful_v459": "matches domain framing and risk decision context",
                "missing_before_selection_v459": "external validation and domain literature bridge",
                "claim_boundary_v459": "candidate family only",
            },
            {
                "structure_family_v459": "ml_systems_methods_paper",
                "fit_level_v459": "medium",
                "selected_v459": False,
                "why_useful_v459": "matches pipeline, validation and governance emphasis",
                "missing_before_selection_v459": "systems baseline framing and venue format",
                "claim_boundary_v459": "candidate family only",
            },
        ]
    )


def _target_venue_structure_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id_v459": "abstract",
                "recommended_order_v459": 1,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v455_abstract_conclusion_draft.md",
                "required_next_action_v459": "tighten to target word limit after venue selection",
                "claim_boundary_v459": "draft abstract exists",
            },
            {
                "section_id_v459": "introduction_scope",
                "recommended_order_v459": 2,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v456_manuscript_assembly_packet.md",
                "required_next_action_v459": "adapt motivation to selected audience",
                "claim_boundary_v459": "scope prose exists",
            },
            {
                "section_id_v459": "related_work_positioning",
                "recommended_order_v459": 3,
                "section_ready_v459": False,
                "source_artifact_v459": "paper4_v460_related_work_citation_gap_audit.md",
                "required_next_action_v459": "audit missing citations and related-work anchors",
                "claim_boundary_v459": "related work not yet drafted",
            },
            {
                "section_id_v459": "methods_protocol",
                "recommended_order_v459": 4,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v453_methods_results_draft.md",
                "required_next_action_v459": (
                    "compress implementation details after venue selection"
                ),
                "claim_boundary_v459": "methods draft exists",
            },
            {
                "section_id_v459": "results_evidence",
                "recommended_order_v459": 5,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v451_release_readiness_synthesis.md",
                "required_next_action_v459": "choose primary tables and figures",
                "claim_boundary_v459": "internal validation evidence only",
            },
            {
                "section_id_v459": "discussion_limitations",
                "recommended_order_v459": 6,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v454_discussion_limitations_draft.md",
                "required_next_action_v459": "preserve caveats during venue editing",
                "claim_boundary_v459": "discussion and limitations draft exists",
            },
            {
                "section_id_v459": "reproducibility_artifacts",
                "recommended_order_v459": 7,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v457_post_assembly_pytest_probe.md",
                "required_next_action_v459": (
                    "convert gate list to artifact appendix after venue selection"
                ),
                "claim_boundary_v459": "internal reproducibility only",
            },
            {
                "section_id_v459": "conclusion",
                "recommended_order_v459": 8,
                "section_ready_v459": True,
                "source_artifact_v459": "paper4_v455_abstract_conclusion_draft.md",
                "required_next_action_v459": "align final paragraph with venue contribution style",
                "claim_boundary_v459": "draft conclusion exists",
            },
            {
                "section_id_v459": "references",
                "recommended_order_v459": 9,
                "section_ready_v459": False,
                "source_artifact_v459": "paper4_v460_related_work_citation_gap_audit.md",
                "required_next_action_v459": (
                    "build verified citation log and references.bib update"
                ),
                "claim_boundary_v459": "citation set not yet audited",
            },
        ]
    )


def _gap_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap_id_v459": "related_work_not_audited",
                "blocking_v459": True,
                "next_artifact_v459": NEXT_ARTIFACT,
                "claim_boundary_v459": "do not claim scholarly positioning is complete",
            },
            {
                "gap_id_v459": "verified_citation_log_missing",
                "blocking_v459": True,
                "next_artifact_v459": NEXT_ARTIFACT,
                "claim_boundary_v459": "do not cite unverified literature",
            },
            {
                "gap_id_v459": "target_venue_not_selected",
                "blocking_v459": True,
                "next_artifact_v459": "future_target_venue_decision",
                "claim_boundary_v459": "do not claim target-venue compliance",
            },
            {
                "gap_id_v459": "external_dataset_validation_not_run",
                "blocking_v459": True,
                "next_artifact_v459": "future_external_validation_protocol",
                "claim_boundary_v459": "do not claim external generalization",
            },
            {
                "gap_id_v459": "conditional_quarto_render_if_promoted",
                "blocking_v459": True,
                "next_artifact_v459": "future_post_promotion_quarto_render_probe",
                "claim_boundary_v459": "rerender if content enters book/",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v459_target_venue_structure_packet_created",
                "allowed": True,
                "artifact": "paper4_v459_target_venue_structure_packet.md",
                "boundary": "venue-agnostic structure packet only",
            },
            {
                "claim_id": "v459_section_readiness_map_created",
                "allowed": True,
                "artifact": "paper4_v459_target_venue_structure_map.csv",
                "boundary": "section readiness map only",
            },
            {
                "claim_id": "v459_related_work_gap_identified",
                "allowed": True,
                "artifact": "paper4_v459_section_gap_register.csv",
                "boundary": "gap identification only",
            },
            {
                "claim_id": "v459_target_venue_selected_or_compliant",
                "allowed": False,
                "artifact": "paper4_v459_section_gap_register.csv",
                "boundary": "target venue not selected",
            },
            {
                "claim_id": "v459_submission_ready_or_externally_validated",
                "allowed": False,
                "artifact": "paper4_v459_remaining_blockers.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v459_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _remaining_blockers(gaps: pd.DataFrame) -> pd.DataFrame:
    return gaps.rename(
        columns={
            "gap_id_v459": "blocker_id_v459",
            "next_artifact_v459": "required_next_artifact_v459",
        }
    ).assign(evidence_count_v459=[1, 0, 0, 0, 1])[
        [
            "blocker_id_v459",
            "blocking_v459",
            "evidence_count_v459",
            "required_next_artifact_v459",
            "claim_boundary_v459",
        ]
    ]


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v459 maps Paper 4 into a venue-agnostic target structure.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v459_target_venue_structure_packet.md"
                ),
                "boundary": "Structure packet only; no venue selected.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v459 identifies related-work and citation gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v459_section_gap_register.csv"
                ),
                "boundary": "Gap register only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v459 selects a target venue or proves venue compliance.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v459_section_gap_register.csv"
                ),
                "boundary": "Target venue remains undecided.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v459 makes Paper 4 submitted, externally validated, or final.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v459_remaining_blockers.csv"
                ),
                "boundary": "Submission and external validation remain blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v459 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v459_remaining_blockers.csv"
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
                "executable_item": "v459 maps assembled packet to target-venue structure.",
                "status": "target_venue_structure_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v460 audits related work and citation gaps",
                "last_wave": "v459",
                "execution_result": "target_structure_created_without_venue_selection",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v459")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Target-Venue Structure Packet v459

Generated: {status["generated_at_utc"]}

## Purpose

v459 converts the v456 assembled manuscript packet into a venue-agnostic
submission structure. It does not select a venue and does not claim submission
readiness.

## Recommended Structure

1. Abstract
2. Introduction and Scope
3. Related Work and Positioning
4. Methods and Living-Lab Protocol
5. Results and Evidence Gates
6. Discussion and Limitations
7. Reproducibility Artifacts
8. Conclusion
9. References

## Current Readiness

- Candidate structure families: `{status["candidate_structure_count_v459"]}`.
- Mapped sections: `{status["mapped_section_count_v459"]}`.
- Ready sections: `{status["ready_section_count_v459"]}`.
- Missing sections: `{status["missing_section_count_v459"]}`.
- Blocking gaps: `{status["blocking_gap_count_v459"]}`.
- Target venue selected: `{status["target_venue_selected_v459"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Main Finding

The manuscript now has enough internal evidence for a venue-agnostic structure,
but not enough scholarly positioning for target-venue language. The next useful
wave is a related-work and citation gap audit.

## Required Caveat

v459 is a structure packet only. It does not choose a target venue, verify
citations, complete related work, create external validation, submit the paper,
replace Paper Estrella, or promote Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v459"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V459_TARGET_VENUE_STRUCTURE_PACKET_START -->"
    end = "<!-- V459_TARGET_VENUE_STRUCTURE_PACKET_END -->"
    block = f"""
{start}

## Wave v459: Target-Venue Structure Packet

Generated: {status["generated_at_utc"]}

### Objective

v459 maps the assembled Paper 4 packet into a venue-agnostic manuscript
structure and identifies section gaps.

### Results

- Candidate structure families:
  `{status["candidate_structure_count_v459"]}`.
- Mapped sections:
  `{status["mapped_section_count_v459"]}`.
- Ready sections:
  `{status["ready_section_count_v459"]}`.
- Missing sections:
  `{status["missing_section_count_v459"]}`.
- Blocking gaps:
  `{status["blocking_gap_count_v459"]}`.
- Target venue selected:
  `{status["target_venue_selected_v459"]}`.
- Related-work audit complete:
  `{status["related_work_audit_complete_v459"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v459"]}`.

### Interpretation

Paper 4 now has a target-venue structure packet, but the scholarly positioning
and verified citation set remain the next executable gap.

### Claim Impact

- Allowed: venue-agnostic structure map and related-work/citation gap register.
- Still prohibited: venue selection, venue compliance, submission readiness,
  external validation, champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v459 in the living notebook. v460 should audit related work and citations
before target-venue claims.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v458 = _read_status(458)
    if v458["next_artifact_v458"] != "paper4_v459_target_venue_structure_packet.md":
        raise RuntimeError("v459 expects v458 to route to target-venue structure.")
    if v458["render_decision_recorded_v458"] is not True:
        raise RuntimeError("v459 expects v458 render decision to be recorded.")

    families = _candidate_structure_families()
    section_map = _target_venue_structure_map()
    gaps = _gap_register()
    blockers = _remaining_blockers(gaps)
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v459_candidate_structure_families.csv", families)
    write_csv(TABLE_DIR / "paper4_v459_target_venue_structure_map.csv", section_map)
    write_csv(TABLE_DIR / "paper4_v459_section_gap_register.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v459_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v459_claim_matrix_delta.csv", claim_matrix)

    ready_sections = int(section_map["section_ready_v459"].astype(bool).sum())
    status = {
        "phase": "v459_target_venue_structure_packet",
        "schema_version": "2026-05-17.459",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_render_decision_version_v459": PRIOR_RENDER_DECISION_VERSION,
        "candidate_structure_count_v459": len(families),
        "mapped_section_count_v459": len(section_map),
        "ready_section_count_v459": ready_sections,
        "missing_section_count_v459": int(len(section_map) - ready_sections),
        "blocking_gap_count_v459": int(gaps["blocking_v459"].astype(bool).sum()),
        "target_venue_structure_packet_created_v459": True,
        "target_venue_selected_v459": False,
        "target_venue_compliance_complete_v459": False,
        "related_work_audit_complete_v459": False,
        "external_validation_complete_v459": False,
        "working_champion_claim_allowed_v459": False,
        "paper1_promotion_allowed_v459": False,
        "paper4_working_champion_changed_v459": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v459": NEXT_ARTIFACT,
        "claim_boundary": (
            "v459 is a venue-agnostic manuscript structure packet; target venue, "
            "related work, external validation, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v459 must not create final Paper 4 promotion.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v459": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
