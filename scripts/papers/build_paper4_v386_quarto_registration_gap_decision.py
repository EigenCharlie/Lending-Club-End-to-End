#!/usr/bin/env python3
"""Build Paper 4 v386 Quarto registration-gap decision artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 386
PRIOR_VALIDATION_VERSION = 385
NEXT_VERSION = 387
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_quarto_archive_guardrail_patch.csv"
DECISION_MD = NOTEBOOK.parent / "paper4_v386_quarto_registration_gap_decision.md"
SELECTED_POLICY = "archive_in_place_with_manifested_guardrail_exemption"
TRIAGE_REGISTER = "paper4_v385_quarto_missing_pages_register.csv"


def _load_missing_pages() -> pd.DataFrame:
    missing = read_csv(TRIAGE_REGISTER)
    if missing.empty:
        raise RuntimeError(f"Missing v385 triage register: {TRIAGE_REGISTER}")
    if missing["is_curated_paper4_page_v385"].astype(bool).any():
        raise RuntimeError("v386 expects no curated Paper 4 pages in the missing set.")
    return missing.copy()


def _archive_manifest(missing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(missing.itertuples(index=False), start=1):
        page = str(row.missing_page_v385)
        page_name = Path(page).name
        rows.append(
            {
                "archive_id_v386": f"paper4_archive_{ordinal:03d}",
                "archived_page_v386": page,
                "page_name_v386": page_name,
                "chapter_group_v386": str(Path(page).parent),
                "source_gap_class_v386": str(row.gap_class_v385),
                "selected_policy_v386": SELECTED_POLICY,
                "register_in_book_v386": False,
                "move_file_v386": False,
                "delete_file_v386": False,
                "guardrail_exception_needed_v386": True,
                "archive_manifest_needed_v386": True,
                "curated_page_v386": bool(row.is_curated_paper4_page_v385),
                "decision_reason_v386": (
                    "historical living-lab page preserved on disk; official book "
                    "renders only curated Paper 4 synthesis pages"
                ),
                "source_triage_artifact_v386": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v385_quarto_missing_pages_register.csv"
                ),
            }
        )
    return pd.DataFrame(rows)


def _policy_matrix(missing_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_option_v386": "register_all_historical_pages_in_book",
                "decision_v386": "rejected",
                "selected_v386": False,
                "evidence_count_v386": missing_count,
                "expected_guardrail_effect_v386": "would satisfy registration guardrail",
                "risk_v386": "bloats official chapter and reverses curated-paper boundary",
                "next_action_v386": "do_not_apply",
                "claim_boundary_v386": "not selected; historical pages stay out of render list",
            },
            {
                "policy_option_v386": "move_pages_outside_book_chapters",
                "decision_v386": "deferred",
                "selected_v386": False,
                "evidence_count_v386": missing_count,
                "expected_guardrail_effect_v386": "would remove files from chapter scan",
                "risk_v386": "path churn may break provenance links and historical references",
                "next_action_v386": "keep_as_later_archive_migration_option",
                "claim_boundary_v386": "not selected for immediate patch",
            },
            {
                "policy_option_v386": "delete_historical_pages",
                "decision_v386": "rejected",
                "selected_v386": False,
                "evidence_count_v386": missing_count,
                "expected_guardrail_effect_v386": "would remove missing pages",
                "risk_v386": "destroys living-lab traceability",
                "next_action_v386": "do_not_apply",
                "claim_boundary_v386": "prohibited by provenance requirement",
            },
            {
                "policy_option_v386": SELECTED_POLICY,
                "decision_v386": "selected",
                "selected_v386": True,
                "evidence_count_v386": missing_count,
                "expected_guardrail_effect_v386": (
                    "will allow explicit archived pages while keeping curated book pages strict"
                ),
                "risk_v386": "requires a narrow guardrail patch and archive manifest",
                "next_action_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "decision only; patch not applied in v386",
            },
            {
                "policy_option_v386": "ignore_without_manifest",
                "decision_v386": "rejected",
                "selected_v386": False,
                "evidence_count_v386": missing_count,
                "expected_guardrail_effect_v386": "would leave current failure ambiguous",
                "risk_v386": "hides a book-quality failure without audit trail",
                "next_action_v386": "do_not_apply",
                "claim_boundary_v386": "explicitly prohibited",
            },
        ]
    )


def _decision_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    missing_count = int(len(manifest))
    return pd.DataFrame(
        [
            {
                "decision_id_v386": "selected_archive_policy",
                "decision_v386": SELECTED_POLICY,
                "allowed_v386": True,
                "evidence_count_v386": missing_count,
                "next_action_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "decision artifact only; no guardrail patch yet",
            },
            {
                "decision_id_v386": "register_historical_pages_in_book",
                "decision_v386": "do_not_register",
                "allowed_v386": False,
                "evidence_count_v386": missing_count,
                "next_action_v386": "keep_curated_render_set",
                "claim_boundary_v386": "curated Paper 4 chapter remains the rendered surface",
            },
            {
                "decision_id_v386": "preserve_historical_pages_on_disk",
                "decision_v386": "preserve",
                "allowed_v386": True,
                "evidence_count_v386": missing_count,
                "next_action_v386": "manifest_as_archived_pages",
                "claim_boundary_v386": "provenance retained; not promoted to official text",
            },
            {
                "decision_id_v386": "apply_guardrail_patch",
                "decision_v386": "defer_to_v387",
                "allowed_v386": False,
                "evidence_count_v386": missing_count,
                "next_action_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "v386 decides only",
            },
            {
                "decision_id_v386": "paper4_final_promotion",
                "decision_v386": "forbidden",
                "allowed_v386": False,
                "evidence_count_v386": 0,
                "next_action_v386": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v386": "final promotion remains absent",
            },
        ]
    )


def _claim_blockers(missing_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v386": "guardrail_patch_not_applied",
                "blocking_v386": True,
                "evidence_count_v386": missing_count,
                "required_next_artifact_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "v386 selects policy but does not patch tests or manifests",
            },
            {
                "blocker_id_v386": "quarto_registration_guardrail_still_not_clean",
                "blocking_v386": True,
                "evidence_count_v386": missing_count,
                "required_next_artifact_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "book guardrail remains pending until v387",
            },
            {
                "blocker_id_v386": "full_regression_suite_clean_not_claimed",
                "blocking_v386": True,
                "evidence_count_v386": missing_count,
                "required_next_artifact_v386": NEXT_ARTIFACT,
                "claim_boundary_v386": "full-suite clean claim remains blocked",
            },
            {
                "blocker_id_v386": "paper4_final_promotion_forbidden",
                "blocking_v386": True,
                "evidence_count_v386": 1,
                "required_next_artifact_v386": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v386": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v386_quarto_registration_gap_decision_created",
                "allowed": True,
                "artifact": "paper4_v386_quarto_registration_gap_decision.md",
                "boundary": "decision artifact only",
            },
            {
                "claim_id": "v386_archive_manifest_draft_created",
                "allowed": True,
                "artifact": "paper4_v386_quarto_archive_manifest.csv",
                "boundary": "manifest draft, not yet book guardrail input",
            },
            {
                "claim_id": "v386_archive_in_place_policy_selected",
                "allowed": True,
                "artifact": "paper4_v386_quarto_registration_policy_matrix.csv",
                "boundary": "selected for v387 patch",
            },
            {
                "claim_id": "v386_book_config_or_guardrail_patched",
                "allowed": False,
                "artifact": "paper4_v386_claim_blockers.csv",
                "boundary": "v386 applies no patch",
            },
            {
                "claim_id": "v386_full_regression_suite_clean",
                "allowed": False,
                "artifact": "paper4_v386_claim_blockers.csv",
                "boundary": "suite clean claim remains blocked",
            },
            {
                "claim_id": "v386_working_champion_or_final_promotion",
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
                "claim": "v386 selects an explicit archive policy for historical Quarto pages.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v386_quarto_registration_policy_matrix.csv"
                ),
                "boundary": "Decision only; v386 does not patch the book guardrail.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v386 preserves historical Paper 4 pages without rendering them.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v386_quarto_archive_manifest.csv"
                ),
                "boundary": "Archive-manifest draft; official rendered chapter remains curated.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v386 fixes the Quarto registration guardrail or full regression suite.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v386_claim_blockers.csv"
                ),
                "boundary": "Guardrail patch deferred to v387.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v386 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v386_claim_blockers.csv"
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
                    "v386 selects an archive-in-place policy for the 70 historical "
                    "standalone Quarto pages."
                ),
                "status": "quarto_registration_gap_decision_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v387 writes a stable archive manifest and narrows the book guardrail "
                    "to allow only explicit archived pages"
                ),
                "last_wave": "v386",
                "execution_result": "archive_in_place_policy_selected_guardrail_patch_deferred",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v386")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any], policy: pd.DataFrame) -> str:
    policy_lines = "\n".join(
        "- `{}`: `{}`{}".format(
            row.policy_option_v386,
            row.decision_v386,
            " (selected)" if bool(row.selected_v386) else "",
        )
        for row in policy.itertuples(index=False)
    )
    return f"""# Paper 4 Quarto Registration Gap Decision v386

Generated: {status["generated_at_utc"]}

v386 turns the v385 validation gap into an explicit governance decision.

## Selected Policy

`{status["selected_policy_v386"]}`

The 70 historical standalone pages remain on disk for provenance, but they should
not be rendered as official Paper 4 book chapters. The official rendered surface
stays limited to the curated Paper 4 pages.

## Options Reviewed

{policy_lines}

## Required Caveat

v386 is a decision packet only. It does not mutate `book/_quarto.yml`, does not
patch the Quarto book guardrail, does not make the full regression suite clean,
and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v386"]}` to write the stable archive manifest and
patch the book guardrail narrowly around explicitly archived historical pages.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V386_QUARTO_REGISTRATION_GAP_DECISION_START -->"
    end = "<!-- V386_QUARTO_REGISTRATION_GAP_DECISION_END -->"
    block = f"""
{start}

## Wave v386: Quarto Registration Gap Decision

Generated: {status["generated_at_utc"]}

### Objective

v386 decides what to do with the 70 historical standalone Quarto pages isolated
by v385.

### Results

- Missing page rows reviewed:
  `{status["missing_page_rows_v386"]}`.
- Archive manifest draft rows:
  `{status["archive_manifest_rows_v386"]}`.
- Selected policy:
  `{status["selected_policy_v386"]}`.
- Register all historical pages in book:
  `{status["register_all_historical_pages_v386"]}`.
- Guardrail patch applied:
  `{status["guardrail_patch_applied_v386"]}`.
- Book config mutated:
  `{status["book_quarto_mutated_v386"]}`.
- Full regression suite clean:
  `{status["full_regression_suite_clean_v386"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v386"]}`.

### Interpretation

The right immediate path is not to render every historical wave page. The archive
policy preserves the living-lab trace while keeping the official Paper 4 chapter
curated and bounded.

### Claim Impact

- Allowed: v386 selected an archive-in-place policy and produced a draft archive
  manifest.
- Still prohibited: Quarto guardrail fixed, full-regression-clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v386 in the living notebook. v387 should convert the decision into a narrow
guardrail patch and stable archive manifest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v385_status = json.loads((STATUS_DIR / "paper4_v385_status.json").read_text(encoding="utf-8"))
    if v385_status["next_artifact_v385"] != "paper4_v386_quarto_registration_gap_decision.md":
        raise RuntimeError("v386 expects v385 to route to Quarto registration-gap decision.")

    missing = _load_missing_pages()
    manifest = _archive_manifest(missing)
    policy = _policy_matrix(len(manifest))
    decisions = _decision_summary(manifest)
    blockers = _claim_blockers(len(manifest))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v386_quarto_archive_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v386_quarto_registration_policy_matrix.csv", policy)
    write_csv(TABLE_DIR / "paper4_v386_quarto_registration_gap_decision.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v386_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v386_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    selected_count = int(policy["selected_v386"].astype(bool).sum())
    status = {
        "phase": "v386_quarto_registration_gap_decision",
        "schema_version": "2026-05-17.386",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_validation_version_v386": PRIOR_VALIDATION_VERSION,
        "missing_page_rows_v386": int(len(missing)),
        "archive_manifest_rows_v386": int(len(manifest)),
        "policy_option_rows_v386": int(len(policy)),
        "decision_rows_v386": int(len(decisions)),
        "claim_blocker_rows_v386": int(len(blockers)),
        "claim_matrix_rows_v386": int(len(claim_matrix)),
        "selected_policy_rows_v386": selected_count,
        "selected_policy_v386": SELECTED_POLICY,
        "archive_in_place_policy_selected_v386": selected_count == 1,
        "register_all_historical_pages_v386": False,
        "delete_historical_pages_v386": False,
        "move_historical_pages_v386": False,
        "preserve_historical_pages_v386": True,
        "guardrail_patch_applied_v386": False,
        "book_quarto_mutated_v386": False,
        "quarto_registration_guardrail_clean_v386": False,
        "full_regression_suite_clean_v386": False,
        "working_champion_claim_allowed_v386": False,
        "paper1_promotion_allowed_v386": False,
        "paper4_working_champion_changed_v386": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "decision_artifact_v386": (
            "reports/paper_material/paper4/notes/"
            "paper4_v386_quarto_registration_gap_decision.md"
        ),
        "next_artifact_v386": NEXT_ARTIFACT,
        "claim_boundary": (
            "v386 selects an archive-in-place policy for historical Quarto pages; "
            "guardrail patch and full-suite clean claims remain deferred"
        ),
    }
    DECISION_MD.write_text(_decision_markdown(status, policy), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v386_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v386": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
