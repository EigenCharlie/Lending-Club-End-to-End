#!/usr/bin/env python3
"""Build Paper 4 v475 primary table and figure selection artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    PAPER4_ROOT,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 475
PRIOR_MANUSCRIPT_DELTA_VERSION = 474
NEXT_ARTIFACT = "paper4_v476_caption_and_insertion_plan.md"
SELECTION_MD = NOTEBOOK.parent / "paper4_v475_primary_table_figure_selection.md"
FIGURE_DIR = PAPER4_ROOT / "figures"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _path_exists(path_text: str) -> bool:
    if path_text.endswith(".csv") or path_text.endswith(".parquet"):
        return (TABLE_DIR / path_text).exists()
    if path_text.endswith(".png"):
        return (FIGURE_DIR / path_text).exists()
    return Path(path_text).exists()


def _primary_tables() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_slot_v475": "T1",
                "manuscript_section_v475": "results_evidence",
                "source_artifact_v475": "paper4_v467_cvar_frontier_probe.csv",
                "supports_claim_id_v475": "v467_v353_local_return_cvar_frontier",
                "recommended_table_title_v475": "Local Return and CVaR Frontier Chain",
                "main_text_priority_v475": 1,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "local v338-v347-v353 chain only",
                "claim_boundary_v475": "no full-v55 optimality or champion claim",
            },
            {
                "table_slot_v475": "T2",
                "manuscript_section_v475": "results_evidence",
                "source_artifact_v475": "paper4_v468_tight_source_rankings.csv",
                "supports_claim_id_v475": "v468_grade_a_primary_blocker_documented",
                "recommended_table_title_v475": "Tight Source Governance Ranking",
                "main_text_priority_v475": 2,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "diagnostic source blocker statement only",
                "claim_boundary_v475": "no source cap approval or mutation",
            },
            {
                "table_slot_v475": "T3",
                "manuscript_section_v475": "discussion_limitations",
                "source_artifact_v475": "paper4_v469_current_frontier_dynamic_gap.csv",
                "supports_claim_id_v475": "v469_v353_dynamic_gap_documented",
                "recommended_table_title_v475": "Current Frontier Dynamic Replay Gap",
                "main_text_priority_v475": 3,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "v353 lacks dynamic replay trace",
                "claim_boundary_v475": "no live dynamic deployment language",
            },
            {
                "table_slot_v475": "T4",
                "manuscript_section_v475": "results_evidence",
                "source_artifact_v475": "paper4_v470_online_monitoring_proxy_summary.csv",
                "supports_claim_id_v475": "v470_online_monitoring_proxy_created",
                "recommended_table_title_v475": "Internal Online Monitoring Proxy Gates",
                "main_text_priority_v475": 4,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "internal proxy only; no external holdout",
                "claim_boundary_v475": "no production monitoring claim",
            },
            {
                "table_slot_v475": "T5",
                "manuscript_section_v475": "methods_protocol",
                "source_artifact_v475": "paper4_v471_formal_claim_boundary_matrix.csv",
                "supports_claim_id_v475": "v471_bounded_historical_spo_dla_language",
                "recommended_table_title_v475": "SPO-DLA Formal Claim Boundary Matrix",
                "main_text_priority_v475": 5,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "historical/oracle-surrogate audit only",
                "claim_boundary_v475": "no formal theorem or CRC guarantee",
            },
            {
                "table_slot_v475": "T6",
                "manuscript_section_v475": "discussion_limitations",
                "source_artifact_v475": "paper4_v472_ifrs9_requirement_audit.csv",
                "supports_claim_id_v475": "v472_contractual_requirement_gap_documented",
                "recommended_table_title_v475": "Contractual IFRS9 Requirement Gap Audit",
                "main_text_priority_v475": 6,
                "selected_for_manuscript_draft_v475": True,
                "final_table_v475": False,
                "required_caveat_v475": "requirement audit only",
                "claim_boundary_v475": "no contractual accounting compliance claim",
            },
        ]
    )


def _primary_figures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_slot_v475": "F1",
                "manuscript_section_v475": "results_evidence",
                "source_figure_v475": "paper4_fig5_return_ecl_tail_frontier.png",
                "recommended_figure_title_v475": "Return, ECL and Tail-Risk Frontier",
                "supports_domain_lane_v475": "cvar_tail_risk",
                "selected_for_manuscript_draft_v475": True,
                "final_figure_v475": False,
                "required_caveat_v475": "legacy frontier figure; align caption to v467 local claim",
                "claim_boundary_v475": "visual context only; v467 table carries current claim",
            },
            {
                "figure_slot_v475": "F2",
                "manuscript_section_v475": "results_evidence",
                "source_figure_v475": "paper4_fig10_mdcp_worst_source_v2.png",
                "recommended_figure_title_v475": "Worst-Source Governance Coverage",
                "supports_domain_lane_v475": "source_governance",
                "selected_for_manuscript_draft_v475": True,
                "final_figure_v475": False,
                "required_caveat_v475": "source-governance visual context only",
                "claim_boundary_v475": "no cap relaxation or legal fairness certification",
            },
            {
                "figure_slot_v475": "F3",
                "manuscript_section_v475": "results_evidence",
                "source_figure_v475": "paper4_fig12_online_method_search.png",
                "recommended_figure_title_v475": "Online Conformal Method Search",
                "supports_domain_lane_v475": "online_monitoring",
                "selected_for_manuscript_draft_v475": True,
                "final_figure_v475": False,
                "required_caveat_v475": "internal replay-selected method search",
                "claim_boundary_v475": "no external holdout or production monitoring claim",
            },
            {
                "figure_slot_v475": "F4",
                "manuscript_section_v475": "methods_protocol",
                "source_figure_v475": "paper4_fig4_regret_auditability_frontier.png",
                "recommended_figure_title_v475": "Regret-Auditability Frontier",
                "supports_domain_lane_v475": "spo_dla",
                "selected_for_manuscript_draft_v475": True,
                "final_figure_v475": False,
                "required_caveat_v475": "historical audit positioning only",
                "claim_boundary_v475": "no formal SPO-DLA theorem claim",
            },
        ]
    )


def _appendix_tables() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "appendix_table_id_v475": "A1",
                "source_artifact_v475": "paper4_v473_allowed_domain_claims.csv",
                "appendix_role_v475": "bounded claim inventory",
                "selected_for_appendix_v475": True,
                "claim_boundary_v475": "index only",
            },
            {
                "appendix_table_id_v475": "A2",
                "source_artifact_v475": "paper4_v473_open_domain_blockers.csv",
                "appendix_role_v475": "open blocker inventory",
                "selected_for_appendix_v475": True,
                "claim_boundary_v475": "blockers remain open",
            },
            {
                "appendix_table_id_v475": "A3",
                "source_artifact_v475": "paper4_v474_claim_placement_plan.csv",
                "appendix_role_v475": "claim placement provenance",
                "selected_for_appendix_v475": True,
                "claim_boundary_v475": "placement plan only",
            },
            {
                "appendix_table_id_v475": "A4",
                "source_artifact_v475": "paper4_v474_blocker_to_limitations_map.csv",
                "appendix_role_v475": "limitations provenance",
                "selected_for_appendix_v475": True,
                "claim_boundary_v475": "limitations mapping only",
            },
        ]
    )


def _readiness_delta(tables: pd.DataFrame, figures: pd.DataFrame) -> pd.DataFrame:
    table_artifacts_exist = all(_path_exists(path) for path in tables["source_artifact_v475"])
    figure_artifacts_exist = all(_path_exists(path) for path in figures["source_figure_v475"])
    return pd.DataFrame(
        [
            {
                "readiness_gate_v475": "primary_tables_selected",
                "ready_v475": True,
                "evidence_artifact_v475": "paper4_v475_primary_table_selection.csv",
                "claim_boundary_v475": "draft selection only",
            },
            {
                "readiness_gate_v475": "primary_figures_selected",
                "ready_v475": True,
                "evidence_artifact_v475": "paper4_v475_primary_figure_selection.csv",
                "claim_boundary_v475": "draft selection only",
            },
            {
                "readiness_gate_v475": "selected_table_artifacts_exist",
                "ready_v475": table_artifacts_exist,
                "evidence_artifact_v475": "paper4_v475_primary_table_selection.csv",
                "claim_boundary_v475": "path existence only",
            },
            {
                "readiness_gate_v475": "selected_figure_artifacts_exist",
                "ready_v475": figure_artifacts_exist,
                "evidence_artifact_v475": "paper4_v475_primary_figure_selection.csv",
                "claim_boundary_v475": "path existence only",
            },
            {
                "readiness_gate_v475": "captions_and_insertion_plan_created",
                "ready_v475": False,
                "evidence_artifact_v475": NEXT_ARTIFACT,
                "claim_boundary_v475": "deferred to v476",
            },
            {
                "readiness_gate_v475": "book_sources_or_references_modified",
                "ready_v475": False,
                "evidence_artifact_v475": "book sources unchanged",
                "claim_boundary_v475": "no Quarto/book promotion in v475",
            },
            {
                "readiness_gate_v475": "final_tables_figures",
                "ready_v475": False,
                "evidence_artifact_v475": "future manuscript edit",
                "claim_boundary_v475": "selection is not final insertion",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v475_primary_tables_selected_for_draft",
                "allowed": True,
                "artifact": "paper4_v475_primary_table_selection.csv",
                "boundary": "draft table selection only",
            },
            {
                "claim_id": "v475_primary_figures_selected_for_draft",
                "allowed": True,
                "artifact": "paper4_v475_primary_figure_selection.csv",
                "boundary": "draft figure selection only",
            },
            {
                "claim_id": "v475_appendix_tables_selected_for_draft",
                "allowed": True,
                "artifact": "paper4_v475_appendix_table_selection.csv",
                "boundary": "appendix index only",
            },
            {
                "claim_id": "v475_tables_figures_final_or_inserted",
                "allowed": False,
                "artifact": "paper4_v475_manuscript_readiness_delta.csv",
                "boundary": "captions/insertion deferred to v476",
            },
            {
                "claim_id": "v475_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v475_manuscript_readiness_delta.csv",
                "boundary": "no submission or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v475 selects primary Paper 4 tables for a manuscript draft.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v475_primary_table_selection.csv"
                ),
                "boundary": "Draft selection only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v475 selects primary Paper 4 figures for a manuscript draft.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v475_primary_figure_selection.csv"
                ),
                "boundary": "Draft selection only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v475 finalizes or inserts Paper 4 tables and figures.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v475_manuscript_readiness_delta.csv"
                ),
                "boundary": "Captions and insertion plan are deferred to v476.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v475 makes Paper 4 submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v475_manuscript_readiness_delta.csv"
                ),
                "boundary": "Venue, external validation and manuscript insertion remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v475 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v475_manuscript_readiness_delta.csv"
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
                "executable_item": "v475 selects primary tables and figures.",
                "status": "primary_table_figure_selection_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v476 writes captions and insertion plan",
                "last_wave": "v475",
                "execution_result": "draft_tables_figures_selected_without_insertion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v475")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _selection_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Primary Table/Figure Selection v475

Generated: {status["generated_at_utc"]}

## Result

v475 selects a compact draft set of primary tables and figures for the Paper 4
manuscript. The selection is evidence-facing and bounded: it does not insert
assets into Quarto, write captions, finalize the visual package, or make Paper 4
submission-ready.

## Counts

- Primary tables selected: `{status["primary_table_count_v475"]}`.
- Primary figures selected: `{status["primary_figure_count_v475"]}`.
- Appendix tables selected: `{status["appendix_table_count_v475"]}`.
- Selected table artifacts exist: `{status["selected_table_artifacts_exist_v475"]}`.
- Selected figure artifacts exist: `{status["selected_figure_artifacts_exist_v475"]}`.
- Final tables/figures: `{status["final_tables_figures_v475"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v475 is a draft selection only. Captions, insertion order, Quarto edits, final
tables/figures, submission readiness, Paper Estrella replacement and final Paper
4 promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V475_PRIMARY_TABLE_FIGURE_SELECTION_START -->"
    end = "<!-- V475_PRIMARY_TABLE_FIGURE_SELECTION_END -->"
    block = f"""
{start}

## Wave v475: Primary Table/Figure Selection

Generated: {status["generated_at_utc"]}

### Objective

v475 selects draft primary tables and figures from the post-domain manuscript
delta without inserting them into Quarto.

### Results

- Primary tables selected:
  `{status["primary_table_count_v475"]}`.
- Primary figures selected:
  `{status["primary_figure_count_v475"]}`.
- Appendix tables selected:
  `{status["appendix_table_count_v475"]}`.
- Selected table artifacts exist:
  `{status["selected_table_artifacts_exist_v475"]}`.
- Selected figure artifacts exist:
  `{status["selected_figure_artifacts_exist_v475"]}`.
- Captions/insertion plan created:
  `{status["captions_insertion_plan_created_v475"]}`.
- Final tables/figures:
  `{status["final_tables_figures_v475"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v475"]}`.

### Interpretation

Paper 4 now has a compact draft visual/table package that follows the bounded
main-text claims from v474. The package still needs captions and an insertion
plan before any Quarto manuscript edit.

### Claim Impact

- Allowed: draft selection of primary tables, figures and appendix tables.
- Still prohibited: final insertion, final tables/figures, submission readiness,
  Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v475 in the living notebook. v476 should write captions and an insertion
plan without editing book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v474 = _read_status(PRIOR_MANUSCRIPT_DELTA_VERSION)
    if v474["next_artifact_v474"] != "paper4_v475_primary_table_figure_selection.md":
        raise RuntimeError("v475 expects v474 to route to primary table/figure selection.")

    tables = _primary_tables()
    figures = _primary_figures()
    appendix = _appendix_tables()
    readiness = _readiness_delta(tables, figures)
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v475_primary_table_selection.csv", tables)
    write_csv(TABLE_DIR / "paper4_v475_primary_figure_selection.csv", figures)
    write_csv(TABLE_DIR / "paper4_v475_appendix_table_selection.csv", appendix)
    write_csv(TABLE_DIR / "paper4_v475_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v475_claim_matrix_delta.csv", claim_matrix)

    selected_table_artifacts_exist = all(
        _path_exists(path) for path in tables["source_artifact_v475"]
    )
    selected_figure_artifacts_exist = all(
        _path_exists(path) for path in figures["source_figure_v475"]
    )
    status = {
        "phase": "v475_primary_table_figure_selection",
        "schema_version": "2026-05-17.475",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manuscript_delta_version_v475": PRIOR_MANUSCRIPT_DELTA_VERSION,
        "primary_table_figure_selection_created_v475": True,
        "primary_table_count_v475": len(tables),
        "primary_figure_count_v475": len(figures),
        "appendix_table_count_v475": len(appendix),
        "selected_table_artifacts_exist_v475": selected_table_artifacts_exist,
        "selected_figure_artifacts_exist_v475": selected_figure_artifacts_exist,
        "captions_insertion_plan_created_v475": False,
        "book_sources_modified_v475": False,
        "book_references_modified_v475": False,
        "final_tables_figures_v475": False,
        "submission_ready_claim_allowed_v475": False,
        "working_champion_claim_allowed_v475": False,
        "paper1_promotion_allowed_v475": False,
        "paper4_working_champion_changed_v475": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v475": NEXT_ARTIFACT,
        "claim_boundary": (
            "v475 selects draft table/figure candidates only; captions, insertion, "
            "submission and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v475 must not create final Paper 4 promotion.")

    SELECTION_MD.write_text(_selection_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v475": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
