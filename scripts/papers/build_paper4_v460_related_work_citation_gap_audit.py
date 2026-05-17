#!/usr/bin/env python3
"""Build Paper 4 v460 related-work and citation gap audit artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    ROOT,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 460
PRIOR_TARGET_STRUCTURE_VERSION = 459
PRIOR_VERIFIED_SOURCE_LOG_VERSION = 381
NEXT_ARTIFACT = "paper4_v461_bounded_related_work_draft.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v460_related_work_citation_gap_audit.md"
BIB_PATH = ROOT / "book" / "references.bib"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _local_bib_entry_count(path: Path = BIB_PATH) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("@"))


def _anchor_inventory(source_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source_log.iterrows():
        rows.append(
            {
                "source_id_v460": row["source_id_v381"],
                "citation_key_v460": row["citation_key_v381"],
                "title_v460": row["title_v381"],
                "year_v460": int(row["year_v381"]),
                "source_type_v460": row["source_type_v381"],
                "verified_v460": bool(row["verified_v381"]),
                "paper4_use_v460": row["paper4_use_v381"],
                "claim_boundary_v460": row["claim_boundary_v381"],
            }
        )
    return pd.DataFrame(rows)


def _section_citation_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section_id_v460": "abstract",
                "verified_anchor_count_v460": 0,
                "coverage_status_v460": "not_needed_in_abstract",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "keep citations out unless target venue requires them",
                "claim_boundary_v460": "abstract can summarize evidence without citation expansion",
            },
            {
                "section_id_v460": "introduction_scope",
                "verified_anchor_count_v460": 2,
                "coverage_status_v460": "partial",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "use conformal and governance anchors cautiously",
                "claim_boundary_v460": "motivation only; no full literature review",
            },
            {
                "section_id_v460": "related_work_positioning",
                "verified_anchor_count_v460": 5,
                "coverage_status_v460": "partial_gap",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "draft bounded positioning and mark recent search gaps",
                "claim_boundary_v460": "bounded related-work draft only",
            },
            {
                "section_id_v460": "methods_protocol",
                "verified_anchor_count_v460": 4,
                "coverage_status_v460": "partial",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "cite CVaR, conformal and predict-then-optimize anchors",
                "claim_boundary_v460": "method context only",
            },
            {
                "section_id_v460": "results_evidence",
                "verified_anchor_count_v460": 0,
                "coverage_status_v460": "internal_evidence",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "cite generated tables rather than external papers",
                "claim_boundary_v460": "internal validation evidence only",
            },
            {
                "section_id_v460": "discussion_limitations",
                "verified_anchor_count_v460": 4,
                "coverage_status_v460": "partial",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "use IFRS, Regulation B and online conformal caveats",
                "claim_boundary_v460": "limitations context only",
            },
            {
                "section_id_v460": "reproducibility_artifacts",
                "verified_anchor_count_v460": 1,
                "coverage_status_v460": "partial_gap",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "connect artifact governance to local validation gates",
                "claim_boundary_v460": "artifact auditability only",
            },
            {
                "section_id_v460": "conclusion",
                "verified_anchor_count_v460": 0,
                "coverage_status_v460": "not_needed_in_conclusion",
                "ready_for_bounded_draft_v460": True,
                "next_action_v460": "avoid adding unsupported finality language",
                "claim_boundary_v460": "conclusion can route future work",
            },
            {
                "section_id_v460": "references",
                "verified_anchor_count_v460": 9,
                "coverage_status_v460": "not_bibliography_complete",
                "ready_for_bounded_draft_v460": False,
                "next_action_v460": "format references only after target venue selection",
                "claim_boundary_v460": "verified anchor set is not final bibliography",
            },
        ]
    )


def _citation_gap_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap_id_v460": "recent_credit_portfolio_literature_not_systematic",
                "blocking_v460": True,
                "needed_for_v460": "target-venue related work",
                "next_artifact_v460": "future_recent_literature_search_log",
                "claim_boundary_v460": "do not claim systematic literature coverage",
            },
            {
                "gap_id_v460": "venue_specific_reference_style_not_selected",
                "blocking_v460": True,
                "needed_for_v460": "submission formatting",
                "next_artifact_v460": "future_target_venue_decision",
                "claim_boundary_v460": "do not claim venue compliance",
            },
            {
                "gap_id_v460": "references_bib_not_curated_for_paper4",
                "blocking_v460": True,
                "needed_for_v460": "paper-specific bibliography",
                "next_artifact_v460": "future_paper4_references_bib",
                "claim_boundary_v460": "book bibliography is broader than Paper 4",
            },
            {
                "gap_id_v460": "external_validation_literature_and_data_missing",
                "blocking_v460": True,
                "needed_for_v460": "external generalization claim",
                "next_artifact_v460": "future_external_validation_protocol",
                "claim_boundary_v460": "do not claim external validation",
            },
            {
                "gap_id_v460": "legal_fairness_sources_not_approval",
                "blocking_v460": True,
                "needed_for_v460": "legal fair-lending claim",
                "next_artifact_v460": "future_legal_fairness_review",
                "claim_boundary_v460": "Regulation B context is not legal certification",
            },
            {
                "gap_id_v460": "bounded_related_work_draft_not_written",
                "blocking_v460": True,
                "needed_for_v460": "next manuscript prose wave",
                "next_artifact_v460": NEXT_ARTIFACT,
                "claim_boundary_v460": "audit precedes related-work prose",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v460_related_work_citation_gap_audit_created",
                "allowed": True,
                "artifact": "paper4_v460_related_work_citation_gap_audit.md",
                "boundary": "audit only",
            },
            {
                "claim_id": "v460_verified_v381_anchor_log_reused",
                "allowed": True,
                "artifact": "paper4_v381_verified_literature_source_log.csv",
                "boundary": "verified anchors only; no new source claims",
            },
            {
                "claim_id": "v460_bounded_related_work_draft_can_start",
                "allowed": True,
                "artifact": "paper4_v460_section_citation_coverage.csv",
                "boundary": "bounded draft can cite verified anchors and mark gaps",
            },
            {
                "claim_id": "v460_bibliography_complete_or_systematic_review",
                "allowed": False,
                "artifact": "paper4_v460_citation_gap_register.csv",
                "boundary": "systematic search and paper-specific bibliography remain open",
            },
            {
                "claim_id": "v460_submission_ready_or_externally_validated",
                "allowed": False,
                "artifact": "paper4_v460_citation_gap_register.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v460_working_champion_or_final_promotion",
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
                "claim": "v460 audits Paper 4 related-work and citation gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v460_related_work_citation_gap_audit.md"
                ),
                "boundary": "Gap audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v460 reuses verified v381 anchors for bounded related-work drafting.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v460_related_work_anchor_inventory.csv"
                ),
                "boundary": "No new external sources are added in v460.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v460 completes a systematic literature review or final bibliography.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v460_citation_gap_register.csv"
                ),
                "boundary": "Systematic search and final bibliography remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v460 makes Paper 4 submitted, externally validated, or final.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v460_citation_gap_register.csv"
                ),
                "boundary": "Citation audit does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v460 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v460_citation_gap_register.csv"
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
                "executable_item": "v460 audits related-work and citation gaps.",
                "status": "related_work_citation_gap_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v461 drafts bounded related work from verified anchors",
                "last_wave": "v460",
                "execution_result": "citation_gap_audit_created_without_new_sources",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v460")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Related-Work/Citation Gap Audit v460

Generated: {status["generated_at_utc"]}

## Scope

v460 audits local citation readiness for Paper 4. It reuses the verified v381
source log and inspects the local book bibliography count, but it does not add
new external sources, perform a systematic review, or edit `book/references.bib`.

## Results

- Local `book/references.bib` entries: `{status["local_bib_entry_count_v460"]}`.
- Verified v381 anchor rows reused: `{status["verified_anchor_count_v460"]}`.
- Section citation coverage rows: `{status["section_citation_coverage_rows_v460"]}`.
- Sections ready for bounded related-work draft:
  `{status["bounded_related_work_ready_section_count_v460"]}`.
- Citation gaps recorded: `{status["citation_gap_count_v460"]}`.
- Open citation gaps: `{status["open_citation_gap_count_v460"]}`.
- New external sources added: `{status["new_external_sources_added_v460"]}`.
- References bibliography modified: `{status["references_bib_modified_v460"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Interpretation

The verified v381 anchors are enough to start a bounded related-work draft that
clearly labels open gaps. They are not enough to claim a systematic literature
review, target-venue compliance, final bibliography, external validation, legal
fairness certification, or submission readiness.

## Required Caveat

v460 is an audit only. It does not create new citations, verify new papers,
complete related work, select a venue, edit the bibliography, replace Paper
Estrella, or promote Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v460"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V460_RELATED_WORK_CITATION_GAP_AUDIT_START -->"
    end = "<!-- V460_RELATED_WORK_CITATION_GAP_AUDIT_END -->"
    block = f"""
{start}

## Wave v460: Related-Work/Citation Gap Audit

Generated: {status["generated_at_utc"]}

### Objective

v460 audits which verified anchors can support bounded related-work drafting and
which citation gaps remain open.

### Results

- Local bibliography entries:
  `{status["local_bib_entry_count_v460"]}`.
- Verified v381 anchors reused:
  `{status["verified_anchor_count_v460"]}`.
- Section citation coverage rows:
  `{status["section_citation_coverage_rows_v460"]}`.
- Sections ready for bounded related-work draft:
  `{status["bounded_related_work_ready_section_count_v460"]}`.
- Citation gaps recorded:
  `{status["citation_gap_count_v460"]}`.
- New external sources added:
  `{status["new_external_sources_added_v460"]}`.
- References bibliography modified:
  `{status["references_bib_modified_v460"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v460"]}`.

### Interpretation

The verified v381 anchors can support a bounded related-work draft, but they do
not close the systematic-search, venue-style, external-validation or final
bibliography gaps.

### Claim Impact

- Allowed: citation gap audit and bounded related-work draft readiness.
- Still prohibited: systematic review, final bibliography, venue compliance,
  submission readiness, external validation, champion replacement and final
  promotion.

### Quarto Promotion Decision

Keep v460 in the living notebook. v461 should draft bounded related work from
verified anchors while naming the open gaps.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v459 = _read_status(459)
    if v459["next_artifact_v459"] != "paper4_v460_related_work_citation_gap_audit.md":
        raise RuntimeError("v460 expects v459 to route to citation gap audit.")
    if v459["target_venue_structure_packet_created_v459"] is not True:
        raise RuntimeError("v460 expects v459 target-venue structure packet.")

    source_log = pd.read_csv(TABLE_DIR / "paper4_v381_verified_literature_source_log.csv")
    if not source_log["verified_v381"].astype(bool).all():
        raise RuntimeError("v460 requires all v381 source anchors to be verified.")

    anchors = _anchor_inventory(source_log)
    coverage = _section_citation_coverage()
    gaps = _citation_gap_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v460_related_work_anchor_inventory.csv", anchors)
    write_csv(TABLE_DIR / "paper4_v460_section_citation_coverage.csv", coverage)
    write_csv(TABLE_DIR / "paper4_v460_citation_gap_register.csv", gaps)
    write_csv(TABLE_DIR / "paper4_v460_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v460_related_work_citation_gap_audit",
        "schema_version": "2026-05-17.460",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_target_structure_version_v460": PRIOR_TARGET_STRUCTURE_VERSION,
        "prior_verified_source_log_version_v460": PRIOR_VERIFIED_SOURCE_LOG_VERSION,
        "local_bib_entry_count_v460": _local_bib_entry_count(),
        "verified_anchor_count_v460": int(anchors["verified_v460"].astype(bool).sum()),
        "section_citation_coverage_rows_v460": len(coverage),
        "bounded_related_work_ready_section_count_v460": int(
            coverage["ready_for_bounded_draft_v460"].astype(bool).sum()
        ),
        "citation_gap_count_v460": len(gaps),
        "open_citation_gap_count_v460": int(gaps["blocking_v460"].astype(bool).sum()),
        "related_work_citation_gap_audit_created_v460": True,
        "verified_source_log_reused_from_v381_v460": True,
        "new_external_sources_added_v460": False,
        "references_bib_modified_v460": False,
        "systematic_literature_review_complete_v460": False,
        "bibliography_complete_v460": False,
        "target_venue_selected_v460": False,
        "external_validation_complete_v460": False,
        "working_champion_claim_allowed_v460": False,
        "paper1_promotion_allowed_v460": False,
        "paper4_working_champion_changed_v460": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v460": NEXT_ARTIFACT,
        "claim_boundary": (
            "v460 audits local verified anchors and citation gaps; no new sources, "
            "systematic review, bibliography completion, submission or final "
            "promotion are created"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v460 must not create final Paper 4 promotion.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v460": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
