#!/usr/bin/env python3
"""Build Paper 4 v474 post-domain manuscript delta artifacts."""

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

VERSION = 474
PRIOR_DOMAIN_SYNTHESIS_VERSION = 473
NEXT_ARTIFACT = "paper4_v475_primary_table_figure_selection.md"
DELTA_MD = NOTEBOOK.parent / "paper4_v474_post_domain_manuscript_delta.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _section_delta() -> pd.DataFrame:
    synthesis = pd.read_csv(TABLE_DIR / "paper4_v473_domain_execution_synthesis.csv")
    lane_result = dict(
        zip(
            synthesis["domain_lane_v473"],
            synthesis["primary_allowed_result_v473"],
            strict=False,
        )
    )
    lane_blocker = dict(
        zip(
            synthesis["domain_lane_v473"],
            synthesis["primary_open_blocker_v473"],
            strict=False,
        )
    )
    return pd.DataFrame(
        [
            {
                "manuscript_section_v474": "abstract",
                "section_delta_v474": (
                    "Add one sentence that Paper 4 now has a six-lane bounded domain "
                    "evidence bundle."
                ),
                "primary_artifacts_v474": (
                    "paper4_v473_domain_execution_synthesis.csv;"
                    "paper4_v473_allowed_domain_claims.csv"
                ),
                "main_text_claim_allowed_v474": True,
                "appendix_only_v474": False,
                "required_caveat_v474": (
                    "not a working champion, submission package or final promotion"
                ),
                "claim_boundary_v474": "high-level bounded contribution only",
            },
            {
                "manuscript_section_v474": "methods_protocol",
                "section_delta_v474": (
                    "Describe the living-lab mechanism that converts domain waves into "
                    "status, table, notebook and guardrail evidence."
                ),
                "primary_artifacts_v474": (
                    "paper4_v466_domain_lane_backlog.csv;"
                    "paper4_v473_open_domain_blockers.csv"
                ),
                "main_text_claim_allowed_v474": True,
                "appendix_only_v474": False,
                "required_caveat_v474": "domain synthesis does not resolve blockers",
                "claim_boundary_v474": "protocol description only",
            },
            {
                "manuscript_section_v474": "results_evidence",
                "section_delta_v474": (
                    f"Report the six-lane results: {lane_result['cvar_tail_risk']}; "
                    f"{lane_result['source_governance']}; "
                    f"{lane_result['dynamic_replay']}; online/internal proxy evidence; "
                    f"{lane_result['spo_dla']}; {lane_result['ifrs9_proxy']}."
                ),
                "primary_artifacts_v474": (
                    "paper4_v467_cvar_frontier_probe.csv;"
                    "paper4_v468_source_governance_refresh.csv;"
                    "paper4_v469_current_frontier_dynamic_gap.csv;"
                    "paper4_v470_online_monitoring_proxy_summary.csv;"
                    "paper4_v471_formal_claim_boundary_matrix.csv;"
                    "paper4_v472_ifrs9_proxy_boundary_summary.csv"
                ),
                "main_text_claim_allowed_v474": True,
                "appendix_only_v474": False,
                "required_caveat_v474": "local/proxy/internal/bounded evidence only",
                "claim_boundary_v474": "results summary without champion language",
            },
            {
                "manuscript_section_v474": "discussion_limitations",
                "section_delta_v474": (
                    f"Use blockers directly: {lane_blocker['cvar_tail_risk']}; "
                    f"{lane_blocker['dynamic_replay']}; "
                    f"{lane_blocker['online_monitoring']}; "
                    f"{lane_blocker['ifrs9_proxy']}."
                ),
                "primary_artifacts_v474": "paper4_v473_open_domain_blockers.csv",
                "main_text_claim_allowed_v474": True,
                "appendix_only_v474": False,
                "required_caveat_v474": (
                    "global, dynamic, online, legal, accounting and venue gates remain open"
                ),
                "claim_boundary_v474": "limitation language is mandatory",
            },
            {
                "manuscript_section_v474": "appendix_reproducibility",
                "section_delta_v474": (
                    "Index all v467-v473 generated artifacts and guardrail tests as the "
                    "post-domain reproducibility appendix."
                ),
                "primary_artifacts_v474": (
                    "paper4_v473_allowed_domain_claims.csv;"
                    "tests/test_docs/test_paper4_living_lab_guardrails.py"
                ),
                "main_text_claim_allowed_v474": False,
                "appendix_only_v474": True,
                "required_caveat_v474": "artifact appendix is not a venue-ready supplement",
                "claim_boundary_v474": "appendix index only",
            },
            {
                "manuscript_section_v474": "references",
                "section_delta_v474": (
                    "Keep citation integration as a dry-run until book references or venue "
                    "style are explicitly updated."
                ),
                "primary_artifacts_v474": (
                    "paper4_v464_references_subset.bib;"
                    "paper4_v465_citation_integration_map.csv"
                ),
                "main_text_claim_allowed_v474": False,
                "appendix_only_v474": True,
                "required_caveat_v474": "bibliography remains dry-run and venue agnostic",
                "claim_boundary_v474": "no book references mutation",
            },
        ]
    )


def _claim_placement() -> pd.DataFrame:
    allowed = pd.read_csv(TABLE_DIR / "paper4_v473_allowed_domain_claims.csv")
    main_text_claims = {
        "v467_v353_local_return_cvar_frontier",
        "v468_grade_a_primary_blocker_documented",
        "v469_v353_dynamic_gap_documented",
        "v470_online_monitoring_proxy_created",
        "v471_bounded_historical_spo_dla_language",
        "v472_contractual_requirement_gap_documented",
    }
    rows = []
    for _, row in allowed.iterrows():
        claim_id = str(row["claim_id_v473"])
        rows.append(
            {
                "claim_id_v474": claim_id,
                "source_wave_v474": row["wave_v473"],
                "recommended_placement_v474": (
                    "main_text" if claim_id in main_text_claims else "appendix"
                ),
                "source_artifact_v474": row["artifact_v473"],
                "boundary_v474": row["boundary_v473"],
                "claim_text_action_v474": (
                    "state with caveat" if claim_id in main_text_claims else "index as support"
                ),
            }
        )
    return pd.DataFrame(rows)


def _blocker_to_limitations() -> pd.DataFrame:
    blockers = pd.read_csv(TABLE_DIR / "paper4_v473_open_domain_blockers.csv")
    selected = {
        "proxy_gap_persists_on_local_frontier": "results and limitations",
        "full_v55_global_proof_missing": "limitations",
        "grade_a_primary_source_blocker": "results and limitations",
        "v353_dynamic_proxy_trace_missing": "limitations",
        "v353_online_temporal_gate_missing": "limitations",
        "formal_theorem_or_proof_missing": "methods limitations",
        "contractual_ifrs9_requirements_missing": "limitations",
        "v353_ifrs9_proxy_gate_missing": "limitations",
        "paper4_final_promotion_forbidden": "governance caveat",
    }
    out = blockers.loc[blockers["blocker_id_v473"].isin(selected)].copy()
    out = out.drop_duplicates(subset=["blocker_id_v473"], keep="first")
    out["recommended_section_v474"] = out["blocker_id_v473"].map(selected)
    out["limitation_language_required_v474"] = True
    out["claim_boundary_v474"] = "blocker must remain explicit in manuscript delta"
    return out[
        [
            "wave_v473",
            "blocker_id_v473",
            "evidence_count_v473",
            "recommended_section_v474",
            "required_next_artifact_v473",
            "limitation_language_required_v474",
            "claim_boundary_v474",
        ]
    ]


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v474": "post_domain_manuscript_delta_created",
                "ready_v474": True,
                "evidence_artifact_v474": "paper4_v474_post_domain_manuscript_delta.md",
                "claim_boundary_v474": "delta only; not final manuscript",
            },
            {
                "readiness_gate_v474": "main_text_claims_selected",
                "ready_v474": True,
                "evidence_artifact_v474": "paper4_v474_claim_placement_plan.csv",
                "claim_boundary_v474": "bounded claim placement only",
            },
            {
                "readiness_gate_v474": "limitations_backed_by_blockers",
                "ready_v474": True,
                "evidence_artifact_v474": "paper4_v474_blocker_to_limitations_map.csv",
                "claim_boundary_v474": "limitations must preserve blockers",
            },
            {
                "readiness_gate_v474": "primary_tables_figures_selected",
                "ready_v474": False,
                "evidence_artifact_v474": NEXT_ARTIFACT,
                "claim_boundary_v474": "selection deferred to v475",
            },
            {
                "readiness_gate_v474": "book_sources_or_references_modified",
                "ready_v474": False,
                "evidence_artifact_v474": "book sources unchanged",
                "claim_boundary_v474": "no Quarto/book promotion in v474",
            },
            {
                "readiness_gate_v474": "submission_ready",
                "ready_v474": False,
                "evidence_artifact_v474": "future venue decision and external validation",
                "claim_boundary_v474": "venue/external gates remain missing",
            },
            {
                "readiness_gate_v474": "paper4_final_promotion_created",
                "ready_v474": False,
                "evidence_artifact_v474": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v474": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v474_post_domain_manuscript_delta_created",
                "allowed": True,
                "artifact": "paper4_v474_post_domain_manuscript_delta.md",
                "boundary": "manuscript delta only",
            },
            {
                "claim_id": "v474_main_text_and_appendix_claims_mapped",
                "allowed": True,
                "artifact": "paper4_v474_claim_placement_plan.csv",
                "boundary": "bounded placement plan only",
            },
            {
                "claim_id": "v474_limitations_backed_by_open_blockers",
                "allowed": True,
                "artifact": "paper4_v474_blocker_to_limitations_map.csv",
                "boundary": "limitation mapping only",
            },
            {
                "claim_id": "v474_primary_tables_figures_finalized",
                "allowed": False,
                "artifact": "paper4_v474_manuscript_readiness_delta.csv",
                "boundary": "selection deferred to v475",
            },
            {
                "claim_id": "v474_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v474_manuscript_readiness_delta.csv",
                "boundary": "venue, external validation and final promotion remain blocked",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v474 maps post-domain evidence into manuscript sections.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v474_manuscript_section_delta.csv"
                ),
                "boundary": "Manuscript delta only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v474 maps bounded claims to main text and appendix placement.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v474_claim_placement_plan.csv"
                ),
                "boundary": "Placement plan only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v474 finalizes Paper 4 primary tables and figures.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v474_manuscript_readiness_delta.csv"
                ),
                "boundary": "Table/figure selection is deferred to v475.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v474 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v474_manuscript_readiness_delta.csv"
                ),
                "boundary": "Venue, external validation and bibliography gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v474 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v474_manuscript_readiness_delta.csv"
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
                "executable_item": "v474 maps domain evidence into manuscript delta.",
                "status": "post_domain_manuscript_delta_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v475 selects primary Paper 4 tables and figures",
                "last_wave": "v474",
                "execution_result": "domain_evidence_mapped_to_manuscript_without_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v474")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _delta_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Domain Manuscript Delta v474

Generated: {status["generated_at_utc"]}

## Result

v474 maps the six-lane domain execution synthesis into manuscript sections. It
selects bounded claims for main-text or appendix placement and maps the most
important blockers into limitations. It does not select final tables/figures,
edit Quarto sources, edit book references, make Paper 4 submission-ready, or
promote Paper 4.

## Counts

- Manuscript section deltas: `{status["manuscript_section_delta_rows_v474"]}`.
- Claim placement rows: `{status["claim_placement_rows_v474"]}`.
- Main-text claim rows: `{status["main_text_claim_rows_v474"]}`.
- Appendix claim rows: `{status["appendix_claim_rows_v474"]}`.
- Limitation blocker rows: `{status["limitation_blocker_rows_v474"]}`.
- Submission-ready claim allowed: `{status["submission_ready_claim_allowed_v474"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v474 is a manuscript delta only. Table/figure selection, target-venue formatting,
book-reference updates, external validation, submission readiness, Paper Estrella
replacement and final Paper 4 promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V474_POST_DOMAIN_MANUSCRIPT_DELTA_START -->"
    end = "<!-- V474_POST_DOMAIN_MANUSCRIPT_DELTA_END -->"
    block = f"""
{start}

## Wave v474: Post-Domain Manuscript Delta

Generated: {status["generated_at_utc"]}

### Objective

v474 maps the v467-v473 domain evidence into manuscript sections while
preserving all open blockers.

### Results

- Manuscript section deltas:
  `{status["manuscript_section_delta_rows_v474"]}`.
- Claim placement rows:
  `{status["claim_placement_rows_v474"]}`.
- Main-text claim rows:
  `{status["main_text_claim_rows_v474"]}`.
- Appendix claim rows:
  `{status["appendix_claim_rows_v474"]}`.
- Limitation blocker rows:
  `{status["limitation_blocker_rows_v474"]}`.
- Primary tables/figures selected:
  `{status["primary_tables_figures_selected_v474"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v474"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v474"]}`.

### Interpretation

The domain evidence is now usable for manuscript editing: a small set of bounded
claims belongs in the main text, the rest belongs in an appendix, and the main
limitations are backed by explicit blocker rows.

### Claim Impact

- Allowed: manuscript-section delta, bounded claim placement, blocker-backed
  limitations.
- Still prohibited: primary table/figure finalization, submission readiness,
  Quarto/book-reference promotion, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v474 in the living notebook. v475 should select primary tables and figures
without editing book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v473 = _read_status(PRIOR_DOMAIN_SYNTHESIS_VERSION)
    if v473["next_artifact_v473"] != "paper4_v474_post_domain_manuscript_delta.md":
        raise RuntimeError("v474 expects v473 to route to post-domain manuscript delta.")

    section_delta = _section_delta()
    placement = _claim_placement()
    limitations = _blocker_to_limitations()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v474_manuscript_section_delta.csv", section_delta)
    write_csv(TABLE_DIR / "paper4_v474_claim_placement_plan.csv", placement)
    write_csv(TABLE_DIR / "paper4_v474_blocker_to_limitations_map.csv", limitations)
    write_csv(TABLE_DIR / "paper4_v474_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v474_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v474_post_domain_manuscript_delta",
        "schema_version": "2026-05-17.474",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_domain_synthesis_version_v474": PRIOR_DOMAIN_SYNTHESIS_VERSION,
        "post_domain_manuscript_delta_created_v474": True,
        "manuscript_section_delta_rows_v474": len(section_delta),
        "claim_placement_rows_v474": len(placement),
        "main_text_claim_rows_v474": int(
            placement["recommended_placement_v474"].eq("main_text").sum()
        ),
        "appendix_claim_rows_v474": int(
            placement["recommended_placement_v474"].eq("appendix").sum()
        ),
        "limitation_blocker_rows_v474": len(limitations),
        "readiness_delta_rows_v474": len(readiness),
        "primary_tables_figures_selected_v474": False,
        "book_sources_modified_v474": False,
        "book_references_modified_v474": False,
        "submission_ready_claim_allowed_v474": False,
        "working_champion_claim_allowed_v474": False,
        "paper1_promotion_allowed_v474": False,
        "paper4_working_champion_changed_v474": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v474": NEXT_ARTIFACT,
        "claim_boundary": (
            "v474 maps post-domain evidence into manuscript sections only; "
            "table/figure selection, submission and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v474 must not create final Paper 4 promotion.")

    DELTA_MD.write_text(_delta_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v474": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
