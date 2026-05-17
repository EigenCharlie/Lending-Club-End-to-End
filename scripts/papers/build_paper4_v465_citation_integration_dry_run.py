#!/usr/bin/env python3
"""Build Paper 4 v465 citation integration dry-run artifacts."""

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

VERSION = 465
PRIOR_BIBLIOGRAPHY_SUBSET_VERSION = 464
NEXT_ARTIFACT = "paper4_v466_domain_execution_backlog_refocus.md"
DRY_RUN_MD = NOTEBOOK.parent / "paper4_v465_citation_integration_dry_run.md"
BIB_SUBSET = NOTEBOOK.parent.parent / "references" / "paper4_v464_references_subset.bib"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _split_keys(raw: str) -> list[str]:
    keys = [key.strip() for key in str(raw).split(";") if key.strip()]
    return [
        key
        for key in keys
        if key not in {"NONE", "paper4_v460_citation_gap_register.csv"}
    ]


def _keys_used_in_trace(trace: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    allowed = trace.loc[trace["allowed_v461"].astype(bool)]
    for raw in allowed["citation_keys_v461"]:
        keys.update(_split_keys(str(raw)))
    return keys


def _integration_map(subset_keys: set[str]) -> pd.DataFrame:
    rows = [
        {
            "integration_id_v465": "related_work_conformal_foundations",
            "target_section_v465": "Related Work and Positioning",
            "citation_keys_v465": (
                "vovk2005algorithmic;romano2019conformalized;"
                "gibbs2021adaptive;angelopoulos2024conformal"
            ),
            "integration_allowed_v465": True,
            "integration_text_v465": (
                "Cite conformal foundations and adjacent conformal extensions as "
                "context, not as implemented Paper 4 guarantees."
            ),
            "claim_boundary_v465": "bounded related-work placement only",
        },
        {
            "integration_id_v465": "methods_risk_optimization",
            "target_section_v465": "Methods and Living-Lab Protocol",
            "citation_keys_v465": "rockafellar2000optimization;rockafellar2002conditional",
            "integration_allowed_v465": True,
            "integration_text_v465": (
                "Cite CVaR foundations beside tail-risk vocabulary and scenario-loss caveats."
            ),
            "claim_boundary_v465": "risk-measure context only",
        },
        {
            "integration_id_v465": "related_work_predict_then_optimize_boundary",
            "target_section_v465": "Related Work and Positioning",
            "citation_keys_v465": "elmachtoub2021smart",
            "integration_allowed_v465": True,
            "integration_text_v465": (
                "Cite predict-then-optimize as a boundary, while stating Paper 4 is not "
                "differentiable SPO training."
            ),
            "claim_boundary_v465": "method-boundary placement only",
        },
        {
            "integration_id_v465": "limitations_regulatory_context",
            "target_section_v465": "Discussion and Limitations",
            "citation_keys_v465": "ifrs2026ifrs9;cfpb2026regulationb",
            "integration_allowed_v465": True,
            "integration_text_v465": (
                "Cite IFRS 9 and Regulation B as context for accounting and legal "
                "caveats, not compliance certification."
            ),
            "claim_boundary_v465": "regulatory context only",
        },
        {
            "integration_id_v465": "references_subset_dry_run",
            "target_section_v465": "References",
            "citation_keys_v465": ";".join(sorted(subset_keys)),
            "integration_allowed_v465": True,
            "integration_text_v465": (
                "Use the reports-side v464 subset as a dry-run reference set only."
            ),
            "claim_boundary_v465": "not final bibliography and not book references edit",
        },
        {
            "integration_id_v465": "prohibited_final_bibliography",
            "target_section_v465": "References",
            "citation_keys_v465": "NONE",
            "integration_allowed_v465": False,
            "integration_text_v465": (
                "State that the Paper 4 bibliography is final and venue compliant."
            ),
            "claim_boundary_v465": "final-bibliography and venue-compliance claims blocked",
        },
    ]
    out = pd.DataFrame(rows)
    out["all_keys_in_subset_v465"] = [
        set(_split_keys(raw)).issubset(subset_keys)
        for raw in out["citation_keys_v465"]
    ]
    return out


def _key_consistency(subset_inventory: pd.DataFrame, used_keys: set[str]) -> pd.DataFrame:
    rows = []
    for _, row in subset_inventory.iterrows():
        key = str(row["citation_key_v464"])
        rows.append(
            {
                "citation_key_v465": key,
                "in_subset_bib_v465": True,
                "used_in_related_work_trace_v465": key in used_keys,
                "integration_ready_v465": key in used_keys,
                "claim_boundary_v465": "subset key consistency only",
            }
        )
    return pd.DataFrame(rows)


def _integration_readiness() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v465": "citation_integration_map_created",
                "ready_v465": True,
                "evidence_artifact_v465": "paper4_v465_citation_integration_map.csv",
                "claim_boundary_v465": "dry-run citation placement only",
            },
            {
                "readiness_gate_v465": "subset_bib_available",
                "ready_v465": True,
                "evidence_artifact_v465": "paper4_v464_references_subset.bib",
                "claim_boundary_v465": "reports-side subset only",
            },
            {
                "readiness_gate_v465": "book_references_modified",
                "ready_v465": False,
                "evidence_artifact_v465": "book/references.bib",
                "claim_boundary_v465": "global bibliography intentionally unchanged",
            },
            {
                "readiness_gate_v465": "quarto_sources_modified",
                "ready_v465": False,
                "evidence_artifact_v465": "book/chapters/19-paper-mega-extension",
                "claim_boundary_v465": "no Quarto promotion in v465",
            },
            {
                "readiness_gate_v465": "domain_execution_refocus",
                "ready_v465": False,
                "evidence_artifact_v465": NEXT_ARTIFACT,
                "claim_boundary_v465": "domain backlog refocus deferred to v466",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v465": "domain_execution_backlog_not_refocused",
                "blocking_v465": True,
                "evidence_count_v465": 1,
                "required_next_artifact_v465": NEXT_ARTIFACT,
                "claim_boundary_v465": "return next to executable experimental lanes",
            },
            {
                "blocker_id_v465": "book_references_not_modified",
                "blocking_v465": True,
                "evidence_count_v465": 1,
                "required_next_artifact_v465": "future_book_references_patch",
                "claim_boundary_v465": "global bibliography intentionally unchanged",
            },
            {
                "blocker_id_v465": "quarto_not_promoted",
                "blocking_v465": True,
                "evidence_count_v465": 1,
                "required_next_artifact_v465": "future_post_promotion_quarto_render_probe",
                "claim_boundary_v465": "rerender only after book source changes",
            },
            {
                "blocker_id_v465": "target_venue_not_selected",
                "blocking_v465": True,
                "evidence_count_v465": 0,
                "required_next_artifact_v465": "future_target_venue_decision",
                "claim_boundary_v465": "do not claim venue compliance",
            },
            {
                "blocker_id_v465": "paper4_final_promotion_forbidden",
                "blocking_v465": True,
                "evidence_count_v465": 1,
                "required_next_artifact_v465": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v465": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v465_citation_integration_dry_run_created",
                "allowed": True,
                "artifact": "paper4_v465_citation_integration_dry_run.md",
                "boundary": "dry-run placement only",
            },
            {
                "claim_id": "v465_all_trace_keys_present_in_subset",
                "allowed": True,
                "artifact": "paper4_v465_subset_key_consistency.csv",
                "boundary": "key consistency with reports-side subset",
            },
            {
                "claim_id": "v465_book_references_or_quarto_modified",
                "allowed": False,
                "artifact": "paper4_v465_remaining_blockers.csv",
                "boundary": "no global references or Quarto mutation",
            },
            {
                "claim_id": "v465_final_bibliography_or_submission_ready",
                "allowed": False,
                "artifact": "paper4_v465_remaining_blockers.csv",
                "boundary": "not final, venue compliant, or submitted",
            },
            {
                "claim_id": "v465_working_champion_or_final_promotion",
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
                "claim": "v465 maps Paper 4 citation integration as a dry-run.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v465_citation_integration_dry_run.md"
                ),
                "boundary": "Dry-run placement only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v465 verifies related-work citation keys exist in the subset bib.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v465_subset_key_consistency.csv"
                ),
                "boundary": "Subset consistency only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v465 modifies book references or promotes Quarto content.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v465_remaining_blockers.csv"
                ),
                "boundary": "No global references or book source mutation.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v465 makes Paper 4 bibliography final or submission-ready.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v465_remaining_blockers.csv"
                ),
                "boundary": "Venue and final bibliography remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v465 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v465_remaining_blockers.csv"
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
                "lane": "Execution Planning",
                "executable_item": "v465 maps citation integration without promotion.",
                "status": "citation_integration_dry_run_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v466 refocuses executable domain backlog for CVaR/source governance/"
                    "dynamic replay/online/SPO-DLA/IFRS9 proxy"
                ),
                "last_wave": "v465",
                "execution_result": "citation_integration_dry_run_without_quarto_promotion",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v465")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _dry_run_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Citation Integration Dry-Run v465

Generated: {status["generated_at_utc"]}

## Result

v465 maps where the verified v381/v464 citation subset can be integrated into
the Paper 4 manuscript draft. It does not edit `book/references.bib`, modify
Quarto sources, or promote Paper 4.

## Counts

- Citation integration rows: `{status["citation_integration_row_count_v465"]}`.
- Subset bibliography keys: `{status["subset_bib_key_count_v465"]}`.
- Related-work trace keys used: `{status["trace_key_count_v465"]}`.
- All trace keys in subset: `{status["all_trace_keys_in_subset_v465"]}`.
- Book references modified: `{status["book_references_modified_v465"]}`.
- Quarto sources modified: `{status["quarto_sources_modified_v465"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v465 is a citation integration dry-run only. It does not create final
bibliography, edit the global bibliography, promote Quarto content, select a
target venue, submit Paper 4, replace Paper Estrella, or promote Paper 4 as
final.

## Next Executable Wave

Build `{status["next_artifact_v465"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V465_CITATION_INTEGRATION_DRY_RUN_START -->"
    end = "<!-- V465_CITATION_INTEGRATION_DRY_RUN_END -->"
    block = f"""
{start}

## Wave v465: Citation Integration Dry-Run

Generated: {status["generated_at_utc"]}

### Objective

v465 maps citation integration from the reports-side bibliography subset without
editing book references or promoting Quarto content.

### Results

- Citation integration rows:
  `{status["citation_integration_row_count_v465"]}`.
- Subset bibliography keys:
  `{status["subset_bib_key_count_v465"]}`.
- Related-work trace keys:
  `{status["trace_key_count_v465"]}`.
- All trace keys in subset:
  `{status["all_trace_keys_in_subset_v465"]}`.
- Book references modified:
  `{status["book_references_modified_v465"]}`.
- Quarto sources modified:
  `{status["quarto_sources_modified_v465"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v465"]}`.

### Interpretation

The citation placement path is now mapped, so the next highest-value local task
is to refocus the executable experimental backlog toward the domain lanes named
by the Paper 4 goal.

### Claim Impact

- Allowed: citation placement dry-run and subset key consistency.
- Still prohibited: global bibliography mutation, Quarto promotion, final
  bibliography, submission readiness, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v465 in the living notebook. v466 should refocus the executable domain
backlog toward CVaR, source governance, dynamic replay, online, SPO-DLA and
IFRS9 proxy experiments.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v464 = _read_status(464)
    if v464["next_artifact_v464"] != "paper4_v465_citation_integration_dry_run.md":
        raise RuntimeError("v465 expects v464 to route to citation integration dry-run.")
    if v464["bibliography_subset_dry_run_created_v464"] is not True:
        raise RuntimeError("v465 expects v464 bibliography subset dry-run.")
    if not BIB_SUBSET.exists():
        raise RuntimeError("v465 expects the v464 bibliography subset artifact.")

    subset_inventory = pd.read_csv(TABLE_DIR / "paper4_v464_bib_entry_inventory.csv")
    sentence_trace = pd.read_csv(TABLE_DIR / "paper4_v461_citation_sentence_trace.csv")
    subset_keys = set(subset_inventory["citation_key_v464"].astype(str))
    used_keys = _keys_used_in_trace(sentence_trace)
    missing_keys = sorted(used_keys - subset_keys)
    if missing_keys:
        raise RuntimeError(f"v465 trace keys missing from subset bibliography: {missing_keys}")

    integration_map = _integration_map(subset_keys)
    consistency = _key_consistency(subset_inventory, used_keys)
    readiness = _integration_readiness()
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v465_citation_integration_map.csv", integration_map)
    write_csv(TABLE_DIR / "paper4_v465_subset_key_consistency.csv", consistency)
    write_csv(TABLE_DIR / "paper4_v465_integration_readiness.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v465_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v465_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v465_citation_integration_dry_run",
        "schema_version": "2026-05-17.465",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_bibliography_subset_version_v465": PRIOR_BIBLIOGRAPHY_SUBSET_VERSION,
        "citation_integration_row_count_v465": len(integration_map),
        "subset_bib_key_count_v465": len(subset_keys),
        "trace_key_count_v465": len(used_keys),
        "all_trace_keys_in_subset_v465": used_keys.issubset(subset_keys),
        "citation_integration_dry_run_created_v465": True,
        "subset_key_consistency_created_v465": True,
        "book_references_modified_v465": False,
        "quarto_sources_modified_v465": False,
        "final_bibliography_complete_v465": False,
        "target_venue_selected_v465": False,
        "domain_backlog_refocus_created_v465": False,
        "working_champion_claim_allowed_v465": False,
        "paper1_promotion_allowed_v465": False,
        "paper4_working_champion_changed_v465": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v465": NEXT_ARTIFACT,
        "claim_boundary": (
            "v465 maps citation integration only; global bibliography edits, Quarto "
            "promotion, final bibliography, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v465 must not create final Paper 4 promotion.")

    DRY_RUN_MD.write_text(_dry_run_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v465": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
