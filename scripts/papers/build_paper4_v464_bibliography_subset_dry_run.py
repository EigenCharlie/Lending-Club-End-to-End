#!/usr/bin/env python3
"""Build Paper 4 v464 bibliography subset dry-run artifacts."""

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

VERSION = 464
PRIOR_BIBLIOGRAPHY_PLAN_VERSION = 463
NEXT_ARTIFACT = "paper4_v465_citation_integration_dry_run.md"
NOTES_DIR = NOTEBOOK.parent
REFERENCES_DIR = NOTEBOOK.parent.parent / "references"
DRY_RUN_MD = NOTES_DIR / "paper4_v464_bibliography_subset_dry_run.md"
SUBSET_BIB = REFERENCES_DIR / "paper4_v464_references_subset.bib"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _entry_type(source_type: str) -> str:
    return {
        "journal_article": "article",
        "conference_paper": "inproceedings",
        "book": "book",
        "official_accounting_standard": "misc",
        "official_regulatory_standard": "misc",
    }.get(source_type, "misc")


def _authors(raw: str) -> str:
    return " and ".join(part.strip() for part in str(raw).split(";") if part.strip())


def _optional_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _bib_entry(row: pd.Series) -> str:
    entry_type = _entry_type(str(row["source_type_v381"]))
    key = str(row["citation_key_v381"])
    fields = {
        "author": _authors(str(row["authors_v381"])),
        "title": str(row["title_v381"]),
        "year": str(int(row["year_v381"])),
        "url": str(row["canonical_url_v381"]),
    }
    doi = _optional_text(row.get("doi_v381", ""))
    if doi:
        fields["doi"] = doi
    venue = str(row["venue_or_publisher_v381"])
    if entry_type == "article":
        fields["journal"] = venue
    elif entry_type == "inproceedings":
        fields["booktitle"] = venue
    elif entry_type == "book":
        fields["publisher"] = venue
    else:
        fields["howpublished"] = venue
    field_lines = [f"  {name} = {{{value}}}" for name, value in fields.items()]
    return f"@{entry_type}{{{key},\n" + ",\n".join(field_lines) + "\n}\n"


def _subset_bib_text(source_log: pd.DataFrame) -> str:
    header = [
        "% Paper 4 bibliography subset dry-run v464",
        "% Generated from verified v381 source log.",
        "% This file is not book/references.bib and is not a final bibliography.",
        "",
    ]
    entries = [_bib_entry(row) for _, row in source_log.iterrows()]
    return "\n".join(header + entries)


def _entry_inventory(source_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source_log.iterrows():
        rows.append(
            {
                "citation_key_v464": row["citation_key_v381"],
                "entry_type_v464": _entry_type(str(row["source_type_v381"])),
                "verified_v464": bool(row["verified_v381"]),
                "has_doi_v464": bool(_optional_text(row.get("doi_v381", ""))),
                "has_url_v464": bool(str(row["canonical_url_v381"]).strip()),
                "source_type_v464": row["source_type_v381"],
                "claim_boundary_v464": "dry-run bib entry from verified v381 metadata",
            }
        )
    return pd.DataFrame(rows)


def _dry_run_summary(entry_inventory: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "summary_metric_v464": "subset_bib_entries",
                "metric_value_v464": len(entry_inventory),
                "claim_boundary_v464": "dry-run subset only",
            },
            {
                "summary_metric_v464": "entries_with_doi",
                "metric_value_v464": int(entry_inventory["has_doi_v464"].astype(bool).sum()),
                "claim_boundary_v464": "metadata completeness only",
            },
            {
                "summary_metric_v464": "entries_with_url",
                "metric_value_v464": int(entry_inventory["has_url_v464"].astype(bool).sum()),
                "claim_boundary_v464": "metadata completeness only",
            },
            {
                "summary_metric_v464": "book_references_modified",
                "metric_value_v464": 0,
                "claim_boundary_v464": "global bibliography unchanged",
            },
            {
                "summary_metric_v464": "final_bibliography_complete",
                "metric_value_v464": 0,
                "claim_boundary_v464": "venue/style and systematic search remain open",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v464": "citation_integration_not_dry_run",
                "blocking_v464": True,
                "evidence_count_v464": 1,
                "required_next_artifact_v464": NEXT_ARTIFACT,
                "claim_boundary_v464": "next local citation-placement task",
            },
            {
                "blocker_id_v464": "book_references_not_modified",
                "blocking_v464": True,
                "evidence_count_v464": 1,
                "required_next_artifact_v464": "future_book_references_patch",
                "claim_boundary_v464": "global bibliography intentionally unchanged",
            },
            {
                "blocker_id_v464": "target_venue_not_selected",
                "blocking_v464": True,
                "evidence_count_v464": 0,
                "required_next_artifact_v464": "future_target_venue_decision",
                "claim_boundary_v464": "do not claim venue style compliance",
            },
            {
                "blocker_id_v464": "systematic_literature_search_not_run",
                "blocking_v464": True,
                "evidence_count_v464": 1,
                "required_next_artifact_v464": "future_recent_literature_search_log",
                "claim_boundary_v464": "do not claim systematic review",
            },
            {
                "blocker_id_v464": "paper4_final_promotion_forbidden",
                "blocking_v464": True,
                "evidence_count_v464": 1,
                "required_next_artifact_v464": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v464": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v464_bibliography_subset_dry_run_created",
                "allowed": True,
                "artifact": "paper4_v464_references_subset.bib",
                "boundary": "reports-side dry-run only",
            },
            {
                "claim_id": "v464_verified_metadata_preserved",
                "allowed": True,
                "artifact": "paper4_v464_bib_entry_inventory.csv",
                "boundary": "entries derive from v381 verified source log",
            },
            {
                "claim_id": "v464_book_references_modified_or_final_bib",
                "allowed": False,
                "artifact": "paper4_v464_remaining_blockers.csv",
                "boundary": "global bibliography unchanged; not final",
            },
            {
                "claim_id": "v464_submission_ready_or_venue_style_complete",
                "allowed": False,
                "artifact": "paper4_v464_remaining_blockers.csv",
                "boundary": "venue/style remains open",
            },
            {
                "claim_id": "v464_working_champion_or_final_promotion",
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
                "claim": "v464 creates a reports-side Paper 4 bibliography subset dry-run.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/references/"
                    "paper4_v464_references_subset.bib"
                ),
                "boundary": "Dry-run subset only; not global bibliography.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v464 preserves verified v381 bibliography metadata.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v464_bib_entry_inventory.csv"
                ),
                "boundary": "Metadata copy from verified source log.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v464 modifies book references or completes final bibliography.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v464_remaining_blockers.csv"
                ),
                "boundary": "book/references.bib remains unchanged.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v464 makes Paper 4 venue-style compliant or submitted.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v464_remaining_blockers.csv"
                ),
                "boundary": "Venue/style remains future work.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v464 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v464_remaining_blockers.csv"
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
                "executable_item": "v464 writes bibliography subset dry-run.",
                "status": "bibliography_subset_dry_run_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v465 maps citation integration without Quarto promotion",
                "last_wave": "v464",
                "execution_result": "bibliography_subset_dry_run_created_without_global_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v464")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _dry_run_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Bibliography Subset Dry-Run v464

Generated: {status["generated_at_utc"]}

## Output

v464 creates a reports-side bibliography subset at
`reports/paper_material/paper4/references/paper4_v464_references_subset.bib`.
It does not edit `book/references.bib`.

## Counts

- Subset entries: `{status["subset_bib_entry_count_v464"]}`.
- Entries with DOI: `{status["entries_with_doi_v464"]}`.
- Entries with URL: `{status["entries_with_url_v464"]}`.
- Book references modified: `{status["book_references_modified_v464"]}`.
- Final bibliography complete: `{status["final_bibliography_complete_v464"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v464 is a bibliography dry-run only. It does not modify the global book
bibliography, select a target venue, complete venue style, run a systematic
literature search, submit the paper, replace Paper Estrella, or promote Paper 4
as final.

## Next Executable Wave

Build `{status["next_artifact_v464"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V464_BIBLIOGRAPHY_SUBSET_DRY_RUN_START -->"
    end = "<!-- V464_BIBLIOGRAPHY_SUBSET_DRY_RUN_END -->"
    block = f"""
{start}

## Wave v464: Bibliography Subset Dry-Run

Generated: {status["generated_at_utc"]}

### Objective

v464 writes a reports-side Paper 4 bibliography subset from verified v381
metadata without editing `book/references.bib`.

### Results

- Subset entries:
  `{status["subset_bib_entry_count_v464"]}`.
- Entries with DOI:
  `{status["entries_with_doi_v464"]}`.
- Entries with URL:
  `{status["entries_with_url_v464"]}`.
- Book references modified:
  `{status["book_references_modified_v464"]}`.
- Final bibliography complete:
  `{status["final_bibliography_complete_v464"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v464"]}`.

### Interpretation

Paper 4 now has a local bibliography subset dry-run, but no global bibliography
or Quarto source was changed.

### Claim Impact

- Allowed: reports-side subset bibliography dry-run and metadata inventory.
- Still prohibited: global bibliography mutation, final bibliography, venue
  style compliance, submission readiness, champion replacement and final
  promotion.

### Quarto Promotion Decision

Keep v464 in the living notebook. v465 should map citation integration without
promoting Quarto or editing global book references.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v463 = _read_status(463)
    if v463["next_artifact_v463"] != "paper4_v464_bibliography_subset_dry_run.md":
        raise RuntimeError("v464 expects v463 to route to bibliography subset dry-run.")
    if v463["paper_specific_bibliography_plan_created_v463"] is not True:
        raise RuntimeError("v464 expects v463 bibliography plan.")

    source_log = pd.read_csv(TABLE_DIR / "paper4_v381_verified_literature_source_log.csv")
    if not source_log["verified_v381"].astype(bool).all():
        raise RuntimeError("v464 requires verified v381 source metadata.")

    subset_text = _subset_bib_text(source_log)
    entry_inventory = _entry_inventory(source_log)
    summary = _dry_run_summary(entry_inventory)
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_BIB.write_text(subset_text, encoding="utf-8")
    write_csv(TABLE_DIR / "paper4_v464_bib_entry_inventory.csv", entry_inventory)
    write_csv(TABLE_DIR / "paper4_v464_bibliography_dry_run_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v464_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v464_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v464_bibliography_subset_dry_run",
        "schema_version": "2026-05-17.464",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_bibliography_plan_version_v464": PRIOR_BIBLIOGRAPHY_PLAN_VERSION,
        "subset_bib_entry_count_v464": len(entry_inventory),
        "entries_with_doi_v464": int(entry_inventory["has_doi_v464"].astype(bool).sum()),
        "entries_with_url_v464": int(entry_inventory["has_url_v464"].astype(bool).sum()),
        "bibliography_subset_dry_run_created_v464": True,
        "subset_bib_artifact_v464": "paper4_v464_references_subset.bib",
        "book_references_modified_v464": False,
        "final_bibliography_complete_v464": False,
        "target_venue_selected_v464": False,
        "systematic_literature_review_complete_v464": False,
        "working_champion_claim_allowed_v464": False,
        "paper1_promotion_allowed_v464": False,
        "paper4_working_champion_changed_v464": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v464": NEXT_ARTIFACT,
        "claim_boundary": (
            "v464 creates a reports-side bibliography subset dry-run; global "
            "book references, venue style, systematic review, submission and final "
            "promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v464 must not create final Paper 4 promotion.")

    DRY_RUN_MD.write_text(_dry_run_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v464": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
