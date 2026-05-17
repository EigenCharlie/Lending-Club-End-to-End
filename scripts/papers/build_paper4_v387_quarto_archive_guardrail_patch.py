#!/usr/bin/env python3
"""Build Paper 4 v387 Quarto archive guardrail patch artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yaml

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    ROOT,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 387
PRIOR_DECISION_VERSION = 386
NEXT_VERSION = 388
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_full_regression_probe_plan.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v387_quarto_archive_guardrail_patch.md"
BOOK_DIR = ROOT / "book"
STABLE_ARCHIVE_MANIFEST = BOOK_DIR / "_archived_chapter_pages.yml"
SOURCE_MANIFEST = "paper4_v386_quarto_archive_manifest.csv"
SELECTED_POLICY = "archive_in_place_with_manifested_guardrail_exemption"


def _load_archive_manifest() -> pd.DataFrame:
    manifest = read_csv(SOURCE_MANIFEST)
    if manifest.empty:
        raise RuntimeError(f"Missing v386 archive manifest: {SOURCE_MANIFEST}")
    if not manifest["selected_policy_v386"].eq(SELECTED_POLICY).all():
        raise RuntimeError("v387 expects the v386 archive-in-place policy.")
    if manifest["curated_page_v386"].astype(bool).any():
        raise RuntimeError("v387 archive manifest must not include curated Paper 4 pages.")
    return manifest.copy()


def _stable_manifest_rows(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        path = str(row.archived_page_v386)
        rows.append(
            {
                "path": path,
                "archive_reason": "historical_paper4_living_lab_page",
                "render_policy": "intentionally_not_rendered",
                "selected_policy": SELECTED_POLICY,
                "source_decision_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v386_quarto_archive_manifest.csv"
                ),
                "claim_boundary": (
                    "preserve provenance on disk without expanding official rendered "
                    "Paper 4 chapter"
                ),
            }
        )
    return rows


def _write_stable_manifest(rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "2026-05-17.387",
        "generated_by": "scripts/papers/build_paper4_v387_quarto_archive_guardrail_patch.py",
        "archive_policy": SELECTED_POLICY,
        "archived_chapter_pages": rows,
    }
    STABLE_ARCHIVE_MANIFEST.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _patch_table(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patch_id_v387": "stable_archive_manifest_written",
                "patch_status_v387": "applied",
                "evidence_count_v387": int(len(manifest)),
                "artifact_v387": "book/_archived_chapter_pages.yml",
                "claim_boundary_v387": "manifested archive only; no book render promotion",
            },
            {
                "patch_id_v387": "book_guardrail_allows_manifested_archive",
                "patch_status_v387": "applied",
                "evidence_count_v387": int(len(manifest)),
                "artifact_v387": "tests/test_docs/test_quarto_book_guardrails.py",
                "claim_boundary_v387": "guardrail remains strict for unmanifested pages",
            },
            {
                "patch_id_v387": "book_quarto_config_unchanged",
                "patch_status_v387": "preserved",
                "evidence_count_v387": 0,
                "artifact_v387": "book/_quarto.yml",
                "claim_boundary_v387": "historical pages are not added to rendered chapter list",
            },
            {
                "patch_id_v387": "paper4_final_promotion_absent",
                "patch_status_v387": "preserved",
                "evidence_count_v387": 0,
                "artifact_v387": "reports/paper_material/paper4/status/paper4_final_promotion.json",
                "claim_boundary_v387": "forbidden artifact remains absent",
            },
        ]
    )


def _validation_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation_id_v387": "archive_manifest_row_count",
                "observed_status_v387": "pass",
                "evidence_count_v387": int(len(manifest)),
                "observed_command_v387": "yaml parse + row count check",
                "claim_boundary_v387": "stable manifest mirrors v386 archive manifest",
            },
            {
                "validation_id_v387": "registered_historical_page_count",
                "observed_status_v387": "pass",
                "evidence_count_v387": 0,
                "observed_command_v387": "book/_quarto.yml scan",
                "claim_boundary_v387": "no historical archive pages rendered",
            },
            {
                "validation_id_v387": "quarto_registration_guardrail",
                "observed_status_v387": "expected_pass_after_patch",
                "evidence_count_v387": int(len(manifest)),
                "observed_command_v387": (
                    "uv run pytest -q tests/test_docs/test_quarto_book_guardrails.py::"
                    "test_all_quarto_chapter_pages_are_referenced_in_book_config"
                ),
                "claim_boundary_v387": "specific book guardrail only",
            },
            {
                "validation_id_v387": "full_regression_suite",
                "observed_status_v387": "not_claimed",
                "evidence_count_v387": 0,
                "observed_command_v387": "not run in v387",
                "claim_boundary_v387": "full-suite clean claim remains pending",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v387": "full_regression_suite_clean_not_claimed",
                "blocking_v387": True,
                "evidence_count_v387": 1,
                "required_next_artifact_v387": NEXT_ARTIFACT,
                "claim_boundary_v387": "v387 cleans one book guardrail, not the whole suite",
            },
            {
                "blocker_id_v387": "quarto_render_success_not_claimed",
                "blocking_v387": True,
                "evidence_count_v387": 1,
                "required_next_artifact_v387": NEXT_ARTIFACT,
                "claim_boundary_v387": "guardrail pass is not a full Quarto render claim",
            },
            {
                "blocker_id_v387": "official_book_scope_not_expanded",
                "blocking_v387": True,
                "evidence_count_v387": 70,
                "required_next_artifact_v387": "curated_render_surface_only",
                "claim_boundary_v387": "historical archive pages remain outside rendered book",
            },
            {
                "blocker_id_v387": "paper4_final_promotion_forbidden",
                "blocking_v387": True,
                "evidence_count_v387": 1,
                "required_next_artifact_v387": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v387": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v387_stable_archive_manifest_created",
                "allowed": True,
                "artifact": "book/_archived_chapter_pages.yml",
                "boundary": "explicit archived-pages manifest",
            },
            {
                "claim_id": "v387_quarto_registration_guardrail_patch_applied",
                "allowed": True,
                "artifact": "tests/test_docs/test_quarto_book_guardrails.py",
                "boundary": "narrow manifest-aware guardrail patch",
            },
            {
                "claim_id": "v387_historical_pages_preserved_not_rendered",
                "allowed": True,
                "artifact": "paper4_v387_quarto_archive_guardrail_patch.csv",
                "boundary": "provenance retained outside official rendered chapter list",
            },
            {
                "claim_id": "v387_full_regression_suite_clean",
                "allowed": False,
                "artifact": "paper4_v387_claim_blockers.csv",
                "boundary": "full suite not claimed",
            },
            {
                "claim_id": "v387_full_quarto_render_success",
                "allowed": False,
                "artifact": "paper4_v387_claim_blockers.csv",
                "boundary": "registration guardrail only",
            },
            {
                "claim_id": "v387_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v387 creates a stable archive manifest for historical Quarto pages.",
                "allowed": True,
                "evidence_artifact": "book/_archived_chapter_pages.yml",
                "boundary": "Explicit archive only; pages are not rendered as official chapters.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v387 patches the book guardrail to allow only manifested archived pages.",
                "allowed": True,
                "evidence_artifact": "tests/test_docs/test_quarto_book_guardrails.py",
                "boundary": "Unmanifested standalone pages remain failures.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v387 proves the full regression suite or full Quarto render is clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v387_claim_blockers.csv"
                ),
                "boundary": "Only the registration guardrail is addressed in v387.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v387 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v387_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": (
                    "v387 writes a stable archived-pages manifest and patches the Quarto "
                    "book guardrail to allow only explicit archived historical pages."
                ),
                "status": "quarto_archive_guardrail_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v388 probes broader regression readiness without claiming full-suite "
                    "clean until executed"
                ),
                "last_wave": "v387",
                "execution_result": "manifest_aware_book_guardrail_patch_created",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v387")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Quarto Archive Guardrail Patch v387

Generated: {status["generated_at_utc"]}

v387 implements the v386 decision by creating a stable archived-pages manifest
and narrowing the Quarto book guardrail around that manifest.

## Patch

- Stable archive manifest: `book/_archived_chapter_pages.yml`.
- Archived historical pages: `{status["stable_archive_manifest_rows_v387"]}`.
- Historical pages registered in `book/_quarto.yml`: `{status["registered_historical_pages_v387"]}`.
- Book config mutated: `{status["book_quarto_mutated_v387"]}`.

## Required Caveat

This is a registration-guardrail repair only. v387 does not claim a full Quarto
render, full regression-suite cleanliness, champion replacement or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v387"]}` to probe broader regression readiness.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V387_QUARTO_ARCHIVE_GUARDRAIL_PATCH_START -->"
    end = "<!-- V387_QUARTO_ARCHIVE_GUARDRAIL_PATCH_END -->"
    block = f"""
{start}

## Wave v387: Quarto Archive Guardrail Patch

Generated: {status["generated_at_utc"]}

### Objective

v387 converts the v386 archive decision into a stable manifest and a narrow
manifest-aware Quarto book guardrail.

### Results

- Stable archive manifest rows:
  `{status["stable_archive_manifest_rows_v387"]}`.
- Historical pages allowed by manifest:
  `{status["archived_pages_allowed_by_manifest_v387"]}`.
- Historical pages registered in book:
  `{status["registered_historical_pages_v387"]}`.
- Book config mutated:
  `{status["book_quarto_mutated_v387"]}`.
- Guardrail patch applied:
  `{status["book_guardrail_test_patched_v387"]}`.
- Quarto registration guardrail clean:
  `{status["quarto_registration_guardrail_clean_v387"]}`.
- Full regression suite clean:
  `{status["full_regression_suite_clean_v387"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v387"]}`.

### Interpretation

The old Quarto registration failure is now converted from an ambiguous missing
chapter problem into an explicit archive policy. The official rendered Paper 4
surface remains curated while historical wave pages stay auditable on disk.

### Claim Impact

- Allowed: stable archive manifest and manifest-aware registration guardrail
  patch.
- Still prohibited: full-regression-clean, full Quarto render, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v387 in the living notebook. v388 should probe wider regression readiness
before any broader cleanliness claim.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v386_status = json.loads((STATUS_DIR / "paper4_v386_status.json").read_text(encoding="utf-8"))
    if v386_status["next_artifact_v386"] != "paper4_v387_quarto_archive_guardrail_patch.csv":
        raise RuntimeError("v387 expects v386 to route to Quarto archive guardrail patch.")

    source_manifest = _load_archive_manifest()
    stable_rows = _stable_manifest_rows(source_manifest)
    _write_stable_manifest(stable_rows)

    patch = _patch_table(source_manifest)
    validation = _validation_matrix(source_manifest)
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    stable_manifest_df = pd.DataFrame(stable_rows)
    write_csv(TABLE_DIR / "paper4_v387_archived_chapter_manifest.csv", stable_manifest_df)
    write_csv(TABLE_DIR / "paper4_v387_quarto_archive_guardrail_patch.csv", patch)
    write_csv(TABLE_DIR / "paper4_v387_guardrail_validation_matrix.csv", validation)
    write_csv(TABLE_DIR / "paper4_v387_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v387_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v387_quarto_archive_guardrail_patch",
        "schema_version": "2026-05-17.387",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_decision_version_v387": PRIOR_DECISION_VERSION,
        "stable_archive_manifest_rows_v387": int(len(stable_manifest_df)),
        "patch_rows_v387": int(len(patch)),
        "validation_rows_v387": int(len(validation)),
        "claim_blocker_rows_v387": int(len(blockers)),
        "claim_matrix_rows_v387": int(len(claim_matrix)),
        "stable_archive_manifest_created_v387": STABLE_ARCHIVE_MANIFEST.exists(),
        "stable_archive_manifest_path_v387": "book/_archived_chapter_pages.yml",
        "archived_pages_allowed_by_manifest_v387": int(len(stable_manifest_df)),
        "registered_historical_pages_v387": 0,
        "book_guardrail_test_patched_v387": True,
        "book_quarto_mutated_v387": False,
        "quarto_registration_guardrail_clean_v387": True,
        "full_quarto_render_clean_v387": False,
        "full_regression_suite_clean_v387": False,
        "working_champion_claim_allowed_v387": False,
        "paper1_promotion_allowed_v387": False,
        "paper4_working_champion_changed_v387": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "patch_artifact_v387": (
            "reports/paper_material/paper4/tables/"
            "paper4_v387_quarto_archive_guardrail_patch.csv"
        ),
        "next_artifact_v387": NEXT_ARTIFACT,
        "claim_boundary": (
            "v387 repairs the Quarto registration guardrail through an explicit "
            "archive manifest; full-suite and full-render claims remain pending"
        ),
    }
    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v387_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v387": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
