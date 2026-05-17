#!/usr/bin/env python3
"""Build Paper 4 v461 bounded related-work draft artifacts."""

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

VERSION = 461
PRIOR_CITATION_GAP_AUDIT_VERSION = 460
NEXT_ARTIFACT = "paper4_v462_manuscript_readiness_delta.md"
DRAFT_MD = NOTEBOOK.parent / "paper4_v461_bounded_related_work_draft.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _paragraph_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paragraph_id_v461": "conformal_foundations",
                "draft_order_v461": 1,
                "citation_keys_v461": (
                    "vovk2005algorithmic;romano2019conformalized;"
                    "gibbs2021adaptive;angelopoulos2024conformal"
                ),
                "planned_role_v461": "position uncertainty and conformal context",
                "allowed_v461": True,
                "claim_boundary_v461": "bounded conformal related-work paragraph",
            },
            {
                "paragraph_id_v461": "risk_optimization_foundations",
                "draft_order_v461": 2,
                "citation_keys_v461": "rockafellar2000optimization;rockafellar2002conditional",
                "planned_role_v461": "position CVaR and tail-risk optimization context",
                "allowed_v461": True,
                "claim_boundary_v461": "risk measure context only",
            },
            {
                "paragraph_id_v461": "predict_then_optimize_boundary",
                "draft_order_v461": 3,
                "citation_keys_v461": "elmachtoub2021smart",
                "planned_role_v461": "distinguish Paper 4 from end-to-end SPO training",
                "allowed_v461": True,
                "claim_boundary_v461": "method boundary only",
            },
            {
                "paragraph_id_v461": "regulatory_context_boundaries",
                "draft_order_v461": 4,
                "citation_keys_v461": "ifrs2026ifrs9;cfpb2026regulationb",
                "planned_role_v461": "state accounting and fair-lending caveats",
                "allowed_v461": True,
                "claim_boundary_v461": "context only, not legal or accounting certification",
            },
            {
                "paragraph_id_v461": "open_gap_caveat",
                "draft_order_v461": 5,
                "citation_keys_v461": "paper4_v460_citation_gap_register.csv",
                "planned_role_v461": "name systematic-search and bibliography gaps",
                "allowed_v461": True,
                "claim_boundary_v461": "gap caveat only",
            },
        ]
    )


def _sentence_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id_v461": "conformal_context",
                "allowed_v461": True,
                "citation_keys_v461": (
                    "vovk2005algorithmic;romano2019conformalized;"
                    "gibbs2021adaptive;angelopoulos2024conformal"
                ),
                "sentence_v461": (
                    "Paper 4 should position conformal prediction as the uncertainty "
                    "language around the living-lab protocol, while separating static "
                    "interval use from online or risk-control extensions."
                ),
                "claim_boundary_v461": "bounded related-work context",
            },
            {
                "sentence_id_v461": "risk_optimization_context",
                "allowed_v461": True,
                "citation_keys_v461": "rockafellar2000optimization;rockafellar2002conditional",
                "sentence_v461": (
                    "CVaR and general loss-distribution references can support the "
                    "tail-risk and scenario-loss vocabulary without proving global "
                    "optimality for Paper 4."
                ),
                "claim_boundary_v461": "risk-measure context only",
            },
            {
                "sentence_id_v461": "pto_boundary",
                "allowed_v461": True,
                "citation_keys_v461": "elmachtoub2021smart",
                "sentence_v461": (
                    "Predict-then-optimize work should be cited as a boundary: Paper 4 "
                    "is auditable pipeline evidence, not differentiable SPO training."
                ),
                "claim_boundary_v461": "method boundary only",
            },
            {
                "sentence_id_v461": "regulatory_caveat",
                "allowed_v461": True,
                "citation_keys_v461": "ifrs2026ifrs9;cfpb2026regulationb",
                "sentence_v461": (
                    "IFRS 9 and Regulation B can frame accounting and fairness limits, "
                    "but cannot be used to claim contractual compliance or legal "
                    "certification."
                ),
                "claim_boundary_v461": "regulatory context only",
            },
            {
                "sentence_id_v461": "gap_disclosure",
                "allowed_v461": True,
                "citation_keys_v461": "paper4_v460_citation_gap_register.csv",
                "sentence_v461": (
                    "The related-work section must state that recent credit-portfolio "
                    "literature, venue style, and paper-specific bibliography work remain open."
                ),
                "claim_boundary_v461": "gap disclosure only",
            },
            {
                "sentence_id_v461": "prohibited_complete_review",
                "allowed_v461": False,
                "citation_keys_v461": "NONE",
                "sentence_v461": (
                    "Paper 4 has completed a systematic literature review and final bibliography."
                ),
                "claim_boundary_v461": "systematic-review and final-bibliography claims blocked",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v461": "systematic_literature_search_not_run",
                "blocking_v461": True,
                "evidence_count_v461": 1,
                "required_next_artifact_v461": "future_recent_literature_search_log",
                "claim_boundary_v461": "do not claim systematic literature coverage",
            },
            {
                "blocker_id_v461": "paper_specific_bibliography_not_curated",
                "blocking_v461": True,
                "evidence_count_v461": 1,
                "required_next_artifact_v461": "future_paper4_references_bib",
                "claim_boundary_v461": "do not claim final bibliography",
            },
            {
                "blocker_id_v461": "target_venue_not_selected",
                "blocking_v461": True,
                "evidence_count_v461": 0,
                "required_next_artifact_v461": "future_target_venue_decision",
                "claim_boundary_v461": "do not claim venue compliance",
            },
            {
                "blocker_id_v461": "manuscript_readiness_delta_not_synthesized",
                "blocking_v461": True,
                "evidence_count_v461": 1,
                "required_next_artifact_v461": NEXT_ARTIFACT,
                "claim_boundary_v461": "synthesize readiness after related-work draft",
            },
            {
                "blocker_id_v461": "paper4_final_promotion_forbidden",
                "blocking_v461": True,
                "evidence_count_v461": 1,
                "required_next_artifact_v461": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v461": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v461_bounded_related_work_draft_created",
                "allowed": True,
                "artifact": "paper4_v461_bounded_related_work_draft.md",
                "boundary": "bounded related-work draft only",
            },
            {
                "claim_id": "v461_citation_sentence_trace_created",
                "allowed": True,
                "artifact": "paper4_v461_citation_sentence_trace.csv",
                "boundary": "sentence-to-anchor traceability",
            },
            {
                "claim_id": "v461_verified_anchor_only_policy_preserved",
                "allowed": True,
                "artifact": "paper4_v460_related_work_anchor_inventory.csv",
                "boundary": "no new citations added",
            },
            {
                "claim_id": "v461_systematic_review_or_final_bibliography",
                "allowed": False,
                "artifact": "paper4_v461_remaining_blockers.csv",
                "boundary": "systematic search and final bibliography remain open",
            },
            {
                "claim_id": "v461_submission_ready_or_externally_validated",
                "allowed": False,
                "artifact": "paper4_v461_remaining_blockers.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v461_working_champion_or_final_promotion",
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
                "claim": "v461 drafts bounded related-work prose from verified anchors.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v461_bounded_related_work_draft.md"
                ),
                "boundary": "Bounded draft only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v461 traces related-work sentences to citation anchors.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v461_citation_sentence_trace.csv"
                ),
                "boundary": "Traceability map only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v461 completes systematic literature review or final bibliography.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v461_remaining_blockers.csv"
                ),
                "boundary": "Systematic search and final bibliography remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v461 makes Paper 4 submitted, externally validated, or final.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v461_remaining_blockers.csv"
                ),
                "boundary": "Related-work draft does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v461 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v461_remaining_blockers.csv"
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
                "executable_item": "v461 drafts bounded related work from verified anchors.",
                "status": "bounded_related_work_draft_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v462 synthesizes manuscript readiness delta",
                "last_wave": "v461",
                "execution_result": "bounded_related_work_draft_created_without_finalization",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v461")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _draft_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Bounded Related-Work Draft v461

Generated: {status["generated_at_utc"]}

## Draft

Paper 4 should be positioned as an auditable living-lab protocol at the
intersection of conformal uncertainty, risk-aware optimization, and governed
credit-risk decision research. Foundational conformal work supports the use of
uncertainty statements, while CQR, adaptive conformal inference, and conformal
risk control provide adjacent methods that should be described as context or
future extensions rather than as claims implemented by Paper 4.

The risk and optimization framing can cite CVaR foundations to support the
tail-risk vocabulary and general loss-distribution caveats. These references
support the language of scenario losses and risk measures; they do not prove
global optimality, integer optimality, or Paper 4 promotion.

Predict-then-optimize work should be used as a method boundary. Paper 4 is not a
differentiable SPO training result; its present contribution is a traceable
pipeline that turns execution waves into guarded manuscript evidence.

IFRS 9 and Regulation B can provide accounting and fair-lending context, but
only as boundaries. Paper 4 does not claim contractual IFRS 9 compliance, legal
fair-lending certification, or protected-attribute review.

The section must explicitly disclose open gaps: recent credit-portfolio
literature has not been systematically reviewed, a target venue has not been
selected, and the Paper 4 bibliography has not been curated as a final
paper-specific reference set.

## Required Caveat

v461 is a bounded related-work draft based on verified v381 anchors and v460
gap analysis. It does not create new citations, complete a systematic review,
curate a final bibliography, submit Paper 4, replace Paper Estrella, or promote
Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v461"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V461_BOUNDED_RELATED_WORK_DRAFT_START -->"
    end = "<!-- V461_BOUNDED_RELATED_WORK_DRAFT_END -->"
    block = f"""
{start}

## Wave v461: Bounded Related-Work Draft

Generated: {status["generated_at_utc"]}

### Objective

v461 drafts bounded related-work prose from verified v381 anchors and the v460
citation gap audit.

### Results

- Paragraph plan rows:
  `{status["paragraph_plan_rows_v461"]}`.
- Citation sentence trace rows:
  `{status["citation_sentence_trace_rows_v461"]}`.
- Allowed citation sentences:
  `{status["allowed_citation_sentence_count_v461"]}`.
- Prohibited citation sentences:
  `{status["prohibited_citation_sentence_count_v461"]}`.
- Verified anchors reused:
  `{status["verified_anchor_count_from_v460"]}`.
- New external sources added:
  `{status["new_external_sources_added_v461"]}`.
- Systematic literature review complete:
  `{status["systematic_literature_review_complete_v461"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v461"]}`.

### Interpretation

The related-work prose can now be drafted in a bounded way, but it remains
explicitly short of a systematic review or final bibliography.

### Claim Impact

- Allowed: bounded related-work draft and citation sentence trace.
- Still prohibited: systematic review, final bibliography, venue compliance,
  submission readiness, external validation, champion replacement and final
  promotion.

### Quarto Promotion Decision

Keep v461 in the living notebook. v462 should synthesize the manuscript
readiness delta after adding related-work prose.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v460 = _read_status(460)
    if v460["next_artifact_v460"] != "paper4_v461_bounded_related_work_draft.md":
        raise RuntimeError("v461 expects v460 to route to bounded related-work draft.")
    if v460["verified_source_log_reused_from_v381_v460"] is not True:
        raise RuntimeError("v461 expects v460 to reuse verified v381 anchors.")

    paragraph_plan = _paragraph_plan()
    sentence_trace = _sentence_trace()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v461_related_work_paragraph_plan.csv", paragraph_plan)
    write_csv(TABLE_DIR / "paper4_v461_citation_sentence_trace.csv", sentence_trace)
    write_csv(TABLE_DIR / "paper4_v461_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v461_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v461_bounded_related_work_draft",
        "schema_version": "2026-05-17.461",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_citation_gap_audit_version_v461": PRIOR_CITATION_GAP_AUDIT_VERSION,
        "paragraph_plan_rows_v461": len(paragraph_plan),
        "citation_sentence_trace_rows_v461": len(sentence_trace),
        "allowed_citation_sentence_count_v461": int(
            sentence_trace["allowed_v461"].astype(bool).sum()
        ),
        "prohibited_citation_sentence_count_v461": int(
            (~sentence_trace["allowed_v461"].astype(bool)).sum()
        ),
        "verified_anchor_count_from_v460": int(v460["verified_anchor_count_v460"]),
        "bounded_related_work_draft_created_v461": True,
        "citation_sentence_trace_created_v461": True,
        "new_external_sources_added_v461": False,
        "references_bib_modified_v461": False,
        "systematic_literature_review_complete_v461": False,
        "bibliography_complete_v461": False,
        "target_venue_selected_v461": False,
        "external_validation_complete_v461": False,
        "working_champion_claim_allowed_v461": False,
        "paper1_promotion_allowed_v461": False,
        "paper4_working_champion_changed_v461": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v461": NEXT_ARTIFACT,
        "claim_boundary": (
            "v461 drafts bounded related work from verified anchors; systematic "
            "review, final bibliography, external validation, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v461 must not create final Paper 4 promotion.")

    DRAFT_MD.write_text(_draft_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v461": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
