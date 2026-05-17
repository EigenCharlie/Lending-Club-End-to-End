#!/usr/bin/env python3
"""Build Paper 4 v463 paper-specific bibliography plan artifacts."""

from __future__ import annotations

import json
import re
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

VERSION = 463
PRIOR_READINESS_DELTA_VERSION = 462
NEXT_ARTIFACT = "paper4_v464_bibliography_subset_dry_run.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v463_paper_specific_bibliography_plan.md"
BIB_PATH = ROOT / "book" / "references.bib"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _bib_text(path: Path = BIB_PATH) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _bib_has_key(bib_text: str, key: str) -> bool:
    return re.search(rf"@\w+\s*\{{\s*{re.escape(key)}\s*,", bib_text) is not None


def _reference_subset_plan(anchors: pd.DataFrame, bib_text: str) -> pd.DataFrame:
    rows = []
    for _, row in anchors.iterrows():
        key = str(row["citation_key_v460"])
        exact_match = _bib_has_key(bib_text, key)
        rows.append(
            {
                "citation_key_v463": key,
                "source_id_v463": row["source_id_v460"],
                "verified_v463": bool(row["verified_v460"]),
                "exact_key_in_book_bib_v463": exact_match,
                "planned_action_v463": (
                    "reuse_existing_book_key"
                    if exact_match
                    else "add_to_paper4_subset_bib_dry_run"
                ),
                "claim_boundary_v463": "plan only; do not edit bibliography in v463",
            }
        )
    return pd.DataFrame(rows)


def _bibliography_action_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id_v463": "create_paper4_subset_bib_dry_run",
                "selected_v463": True,
                "required_next_artifact_v463": NEXT_ARTIFACT,
                "success_condition_v463": (
                    "draft a Paper 4 subset bibliography without editing book refs"
                ),
                "claim_boundary_v463": "dry run only",
            },
            {
                "action_id_v463": "preserve_verified_v381_metadata",
                "selected_v463": True,
                "required_next_artifact_v463": NEXT_ARTIFACT,
                "success_condition_v463": "carry DOI/URL/title/author metadata from v381",
                "claim_boundary_v463": "reuse verified metadata only",
            },
            {
                "action_id_v463": "do_not_edit_book_references_yet",
                "selected_v463": True,
                "required_next_artifact_v463": "future_book_references_patch",
                "success_condition_v463": "wait until venue/style decision before global bib edits",
                "claim_boundary_v463": "no global bibliography mutation",
            },
            {
                "action_id_v463": "target_venue_style_decision",
                "selected_v463": False,
                "required_next_artifact_v463": "future_target_venue_decision",
                "success_condition_v463": "choose IEEE/ACM/journal style later",
                "claim_boundary_v463": "venue not selected",
            },
            {
                "action_id_v463": "systematic_literature_search",
                "selected_v463": False,
                "required_next_artifact_v463": "future_recent_literature_search_log",
                "success_condition_v463": "search and verify new sources separately",
                "claim_boundary_v463": "no new source discovery in v463",
            },
        ]
    )


def _remaining_blockers(missing_key_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v463": "paper4_subset_bib_dry_run_not_written",
                "blocking_v463": True,
                "evidence_count_v463": missing_key_count,
                "required_next_artifact_v463": NEXT_ARTIFACT,
                "claim_boundary_v463": "next local bibliography task",
            },
            {
                "blocker_id_v463": "book_references_not_modified",
                "blocking_v463": True,
                "evidence_count_v463": 1,
                "required_next_artifact_v463": "future_book_references_patch",
                "claim_boundary_v463": "intentional no-mutation boundary",
            },
            {
                "blocker_id_v463": "target_venue_not_selected",
                "blocking_v463": True,
                "evidence_count_v463": 0,
                "required_next_artifact_v463": "future_target_venue_decision",
                "claim_boundary_v463": "do not claim venue style compliance",
            },
            {
                "blocker_id_v463": "systematic_literature_search_not_run",
                "blocking_v463": True,
                "evidence_count_v463": 1,
                "required_next_artifact_v463": "future_recent_literature_search_log",
                "claim_boundary_v463": "do not claim systematic review",
            },
            {
                "blocker_id_v463": "paper4_final_promotion_forbidden",
                "blocking_v463": True,
                "evidence_count_v463": 1,
                "required_next_artifact_v463": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v463": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v463_paper_specific_bibliography_plan_created",
                "allowed": True,
                "artifact": "paper4_v463_paper_specific_bibliography_plan.md",
                "boundary": "plan only",
            },
            {
                "claim_id": "v463_v381_keys_checked_against_book_bib",
                "allowed": True,
                "artifact": "paper4_v463_reference_subset_plan.csv",
                "boundary": "exact-key check only",
            },
            {
                "claim_id": "v463_bibliography_or_book_references_modified",
                "allowed": False,
                "artifact": "paper4_v463_remaining_blockers.csv",
                "boundary": "v463 does not edit bibliography files",
            },
            {
                "claim_id": "v463_final_bibliography_or_venue_style_complete",
                "allowed": False,
                "artifact": "paper4_v463_remaining_blockers.csv",
                "boundary": "venue and final references remain open",
            },
            {
                "claim_id": "v463_working_champion_or_final_promotion",
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
                "claim": "v463 plans a Paper 4 specific bibliography subset.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v463_paper_specific_bibliography_plan.md"
                ),
                "boundary": "Plan only; no bibliography edit.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v463 checks verified v381 keys against book references.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v463_reference_subset_plan.csv"
                ),
                "boundary": "Exact-key inventory only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v463 modifies book references or completes final bibliography.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v463_remaining_blockers.csv"
                ),
                "boundary": "No bibliography file is modified in v463.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v463 makes Paper 4 venue-style compliant or submitted.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v463_remaining_blockers.csv"
                ),
                "boundary": "Venue and style remain future decisions.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v463 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v463_remaining_blockers.csv"
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
                "executable_item": "v463 plans Paper 4 specific bibliography subset.",
                "status": "paper_specific_bibliography_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v464 writes bibliography subset dry-run without global edit",
                "last_wave": "v463",
                "execution_result": "bibliography_plan_created_without_reference_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v463")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Specific Bibliography Plan v463

Generated: {status["generated_at_utc"]}

## Finding

The verified v381 citation keys are not present as exact keys in
`book/references.bib`. v463 therefore records a plan for a Paper 4 specific
bibliography subset dry-run instead of editing the global book bibliography.

## Counts

- Verified anchors checked: `{status["verified_anchor_count_v463"]}`.
- Exact key matches in `book/references.bib`:
  `{status["exact_key_matches_in_book_bib_v463"]}`.
- Missing exact keys:
  `{status["missing_exact_key_count_v463"]}`.
- Bibliography actions planned: `{status["bibliography_action_count_v463"]}`.
- `book/references.bib` modified: `{status["book_references_modified_v463"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v463 is a plan only. It does not edit `book/references.bib`, create a final
Paper 4 bibliography, select a target venue, run a systematic search, submit the
paper, replace Paper Estrella, or promote Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v463"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V463_PAPER_SPECIFIC_BIBLIOGRAPHY_PLAN_START -->"
    end = "<!-- V463_PAPER_SPECIFIC_BIBLIOGRAPHY_PLAN_END -->"
    block = f"""
{start}

## Wave v463: Paper-Specific Bibliography Plan

Generated: {status["generated_at_utc"]}

### Objective

v463 plans a Paper 4 specific bibliography subset without editing the global
book bibliography.

### Results

- Verified anchors checked:
  `{status["verified_anchor_count_v463"]}`.
- Exact key matches in book references:
  `{status["exact_key_matches_in_book_bib_v463"]}`.
- Missing exact keys:
  `{status["missing_exact_key_count_v463"]}`.
- Bibliography actions planned:
  `{status["bibliography_action_count_v463"]}`.
- Book references modified:
  `{status["book_references_modified_v463"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v463"]}`.

### Interpretation

The verified source log is usable, but Paper 4 needs a bibliography subset
dry-run before any global references edit.

### Claim Impact

- Allowed: bibliography plan and exact-key inventory.
- Still prohibited: bibliography mutation, final bibliography, venue compliance,
  systematic review, champion replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v463 in the living notebook. v464 should write a bibliography subset
dry-run without changing `book/references.bib`.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v462 = _read_status(462)
    if v462["next_artifact_v462"] != "paper4_v463_paper_specific_bibliography_plan.md":
        raise RuntimeError("v463 expects v462 to route to bibliography plan.")
    if v462["selected_next_wave_v462"] != "paper_specific_bibliography_plan":
        raise RuntimeError("v463 expects v462 to select bibliography plan.")

    anchors = pd.read_csv(TABLE_DIR / "paper4_v460_related_work_anchor_inventory.csv")
    bib_text = _bib_text()
    subset = _reference_subset_plan(anchors, bib_text)
    actions = _bibliography_action_plan()
    missing_count = int((~subset["exact_key_in_book_bib_v463"].astype(bool)).sum())
    blockers = _remaining_blockers(missing_count)
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v463_reference_subset_plan.csv", subset)
    write_csv(TABLE_DIR / "paper4_v463_bibliography_action_plan.csv", actions)
    write_csv(TABLE_DIR / "paper4_v463_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v463_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v463_paper_specific_bibliography_plan",
        "schema_version": "2026-05-17.463",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_readiness_delta_version_v463": PRIOR_READINESS_DELTA_VERSION,
        "verified_anchor_count_v463": len(subset),
        "exact_key_matches_in_book_bib_v463": int(
            subset["exact_key_in_book_bib_v463"].astype(bool).sum()
        ),
        "missing_exact_key_count_v463": missing_count,
        "bibliography_action_count_v463": len(actions),
        "selected_bibliography_action_count_v463": int(actions["selected_v463"].astype(bool).sum()),
        "paper_specific_bibliography_plan_created_v463": True,
        "book_references_modified_v463": False,
        "paper4_subset_bib_created_v463": False,
        "target_venue_selected_v463": False,
        "systematic_literature_review_complete_v463": False,
        "working_champion_claim_allowed_v463": False,
        "paper1_promotion_allowed_v463": False,
        "paper4_working_champion_changed_v463": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v463": NEXT_ARTIFACT,
        "claim_boundary": (
            "v463 plans a Paper 4 bibliography subset; global bibliography edits, "
            "venue style, systematic review, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v463 must not create final Paper 4 promotion.")

    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v463": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
