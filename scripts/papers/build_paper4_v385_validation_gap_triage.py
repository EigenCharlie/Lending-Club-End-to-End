#!/usr/bin/env python3
"""Build Paper 4 v385 validation-gap triage artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
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

VERSION = 385
PRIOR_FORMAL_REVIEW_VERSION = 384
NEXT_VERSION = 386
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_quarto_registration_gap_decision.md"
TRIAGE_MD = NOTEBOOK.parent / "paper4_v385_validation_gap_triage.md"
BOOK_DIR = ROOT / "book"
QUARTO_CONFIG = BOOK_DIR / "_quarto.yml"
CHAPTERS_DIR = BOOK_DIR / "chapters"
CURATED_PAPER4_PAGES = {
    "index.qmd",
    "19a-proposal-and-scope.qmd",
    "19b-current-assets-and-gaps.qmd",
    "19c-integrated-architecture.qmd",
    "19f-sequential-decision-framework.qmd",
    "19h-mvp-evidence-pack.qmd",
    "19i-regret-auditability-frontier.qmd",
    "19n-online-mdcp-fairness.qmd",
    "19t-multi-period-solver.qmd",
    "19ca-v38-final-synthesis.qmd",
}


def _walk_chapter_entries(items: list[object]) -> set[str]:
    paths: set[str] = set()
    for item in items:
        if isinstance(item, str):
            paths.add(item)
        elif isinstance(item, dict):
            for key in ("chapter", "part"):
                value = item.get(key)
                if isinstance(value, str):
                    paths.add(value)
            nested = item.get("chapters")
            if isinstance(nested, list):
                paths |= _walk_chapter_entries(nested)
    return paths


def _missing_quarto_pages() -> pd.DataFrame:
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding="utf-8"))
    chapter_entries = _walk_chapter_entries(config["book"]["chapters"])
    actual_files = {
        path.relative_to(BOOK_DIR).as_posix()
        for path in sorted(CHAPTERS_DIR.rglob("*.qmd"))
    }
    missing = sorted(actual_files - chapter_entries)
    rows = [
        {
            "missing_page_v385": page,
            "page_name_v385": Path(page).name,
            "chapter_group_v385": str(Path(page).parent),
            "is_curated_paper4_page_v385": Path(page).name in CURATED_PAPER4_PAGES,
            "gap_class_v385": "historical_unregistered_quarto_page",
            "fix_applied_v385": False,
            "claim_boundary_v385": "registration gap only; no Paper 4 promotion",
        }
        for page in missing
    ]
    return pd.DataFrame(rows)


def _validation_triage(missing_pages: pd.DataFrame) -> pd.DataFrame:
    missing_count = int(len(missing_pages))
    return pd.DataFrame(
        [
            {
                "triage_id_v385": "paper4_focal_guardrail_chain_v378_v384",
                "observed_status_v385": "pass",
                "evidence_count_v385": 7,
                "observed_command_v385": (
                    "uv run pytest -q tests/test_docs/test_paper4_living_lab_guardrails.py "
                    "-k v378..v384 focal guardrails"
                ),
                "interpretation_v385": "current Paper 4 living-lab guardrails are healthy",
                "claim_boundary_v385": "targeted guardrail validation only",
            },
            {
                "triage_id_v385": "quarto_chapter_registration_guardrail",
                "observed_status_v385": "fail",
                "evidence_count_v385": missing_count,
                "observed_command_v385": (
                    "uv run pytest -q tests/test_docs/test_quarto_book_guardrails.py::"
                    "test_all_quarto_chapter_pages_are_referenced_in_book_config"
                ),
                "interpretation_v385": "standalone historical Quarto chapter files are not in book/_quarto.yml",
                "claim_boundary_v385": "known registration gap only",
            },
            {
                "triage_id_v385": "full_regression_suite_status",
                "observed_status_v385": "blocked_by_quarto_registration_gap",
                "evidence_count_v385": missing_count,
                "observed_command_v385": "uv run pytest -q",
                "interpretation_v385": "full-suite clean claim remains blocked until registration gap is resolved",
                "claim_boundary_v385": "no full-suite-clean claim",
            },
            {
                "triage_id_v385": "paper4_final_promotion_absence",
                "observed_status_v385": "pass",
                "evidence_count_v385": 0,
                "observed_command_v385": "test ! -e reports/paper_material/paper4/status/paper4_final_promotion.json",
                "interpretation_v385": "final promotion artifact remains absent",
                "claim_boundary_v385": "Paper Estrella remains protected",
            },
        ]
    )


def _validation_decisions(missing_pages: pd.DataFrame) -> pd.DataFrame:
    missing_count = int(len(missing_pages))
    curated_missing = int(missing_pages["is_curated_paper4_page_v385"].astype(bool).sum())
    return pd.DataFrame(
        [
            {
                "decision_id_v385": "current_wave_guardrails_usable",
                "decision_v385": "usable",
                "allowed_v385": True,
                "evidence_count_v385": 7,
                "next_action_v385": "continue targeted Paper 4 waves",
                "claim_boundary_v385": "targeted guardrails only",
            },
            {
                "decision_id_v385": "full_regression_suite_clean",
                "decision_v385": "blocked",
                "allowed_v385": False,
                "evidence_count_v385": missing_count,
                "next_action_v385": NEXT_ARTIFACT,
                "claim_boundary_v385": "Quarto registration gap remains open",
            },
            {
                "decision_id_v385": "curated_paper4_pages_missing",
                "decision_v385": "not_detected",
                "allowed_v385": True,
                "evidence_count_v385": curated_missing,
                "next_action_v385": "keep curated Paper 4 registration unchanged",
                "claim_boundary_v385": "curated set remains protected",
            },
            {
                "decision_id_v385": "quarto_registration_fix_applied",
                "decision_v385": "not_applied",
                "allowed_v385": False,
                "evidence_count_v385": 0,
                "next_action_v385": NEXT_ARTIFACT,
                "claim_boundary_v385": "v385 triages only",
            },
            {
                "decision_id_v385": "paper4_final_promotion",
                "decision_v385": "forbidden",
                "allowed_v385": False,
                "evidence_count_v385": 0,
                "next_action_v385": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v385": "final promotion absent",
            },
        ]
    )


def _claim_blockers(missing_pages: pd.DataFrame) -> pd.DataFrame:
    missing_count = int(len(missing_pages))
    return pd.DataFrame(
        [
            {
                "blocker_id_v385": "full_regression_suite_not_clean",
                "blocking_v385": True,
                "evidence_count_v385": missing_count,
                "required_next_artifact_v385": NEXT_ARTIFACT,
                "claim_boundary_v385": "full pytest clean claim blocked by Quarto registration",
            },
            {
                "blocker_id_v385": "quarto_registration_gap_open",
                "blocking_v385": True,
                "evidence_count_v385": missing_count,
                "required_next_artifact_v385": NEXT_ARTIFACT,
                "claim_boundary_v385": "70 historical standalone pages need decision",
            },
            {
                "blocker_id_v385": "v385_does_not_fix_quarto_registration",
                "blocking_v385": True,
                "evidence_count_v385": 1,
                "required_next_artifact_v385": NEXT_ARTIFACT,
                "claim_boundary_v385": "triage only; no _quarto.yml mutation",
            },
            {
                "blocker_id_v385": "paper4_final_promotion_forbidden",
                "blocking_v385": True,
                "evidence_count_v385": 1,
                "required_next_artifact_v385": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v385": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v385_validation_gap_triage_created",
                "allowed": True,
                "artifact": "paper4_v385_validation_gap_triage.md",
                "boundary": "triage only",
            },
            {
                "claim_id": "v385_current_paper4_guardrails_pass",
                "allowed": True,
                "artifact": "paper4_v385_validation_gap_triage.csv",
                "boundary": "targeted v378-v384 chain only",
            },
            {
                "claim_id": "v385_known_quarto_registration_failure_isolated",
                "allowed": True,
                "artifact": "paper4_v385_quarto_missing_pages_register.csv",
                "boundary": "registration failure isolated, not fixed",
            },
            {
                "claim_id": "v385_full_regression_suite_clean",
                "allowed": False,
                "artifact": "paper4_v385_claim_blockers.csv",
                "boundary": "Quarto registration gap remains",
            },
            {
                "claim_id": "v385_quarto_registration_fixed",
                "allowed": False,
                "artifact": "paper4_v385_validation_decision_matrix.csv",
                "boundary": "v385 applies no registration fix",
            },
            {
                "claim_id": "v385_working_champion_or_final_promotion",
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
                "claim": "v385 isolates the known Quarto registration failure.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v385_quarto_missing_pages_register.csv"
                ),
                "boundary": "Diagnostic isolation only; no registration fix.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v385 confirms the current Paper 4 focal guardrail chain passes.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v385_validation_gap_triage.csv"
                ),
                "boundary": "Targeted v378-v384 guardrails only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v385 makes the full regression suite clean or fixes Quarto registration.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v385_claim_blockers.csv"
                ),
                "boundary": "70 historical standalone Quarto pages remain unregistered.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v385 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v385_claim_blockers.csv"
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
                    "v385 triages the known Quarto registration failure separately from "
                    "current Paper 4 guardrails."
                ),
                "status": "validation_gap_triage_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v386 decides whether to register, archive or explicitly ignore historical standalone pages"
                ),
                "last_wave": "v385",
                "execution_result": "quarto_registration_gap_isolated_current_guardrails_pass",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v385")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _triage_markdown(status: dict[str, Any], missing_pages: pd.DataFrame) -> str:
    sample_lines = "\n".join(
        f"- `{page}`" for page in missing_pages["missing_page_v385"].head(8).tolist()
    )
    return f"""# Paper 4 Validation Gap Triage v385

Generated: {status["generated_at_utc"]}

v385 separates current Paper 4 guardrail health from the known old Quarto chapter
registration failure.

## Diagnosis

- Targeted Paper 4 v378-v384 guardrails: pass.
- Quarto chapter registration guardrail: fail.
- Missing standalone Quarto pages: `{status["quarto_missing_page_rows_v385"]}`.
- Missing curated Paper 4 pages: `{status["curated_missing_page_rows_v385"]}`.

## Missing Page Sample

{sample_lines}

## Required Caveat

v385 is triage only. It does not mutate `book/_quarto.yml`, does not register or
archive pages, does not make the full regression suite clean, and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v385"]}` to decide the registration/archive/ignore
policy for the historical standalone Quarto pages.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V385_VALIDATION_GAP_TRIAGE_START -->"
    end = "<!-- V385_VALIDATION_GAP_TRIAGE_END -->"
    block = f"""
{start}

## Wave v385: Validation Gap Triage

Generated: {status["generated_at_utc"]}

### Objective

v385 triages the known old Quarto registration failure separately from current
Paper 4 living-lab guardrails.

### Results

- Triage rows:
  `{status["validation_triage_rows_v385"]}`.
- Missing Quarto page rows:
  `{status["quarto_missing_page_rows_v385"]}`.
- Missing curated Paper 4 pages:
  `{status["curated_missing_page_rows_v385"]}`.
- Current Paper 4 guardrail chain clean:
  `{status["current_paper4_guardrail_chain_clean_v385"]}`.
- Full regression suite clean:
  `{status["full_regression_suite_clean_v385"]}`.
- Quarto registration fix applied:
  `{status["quarto_registration_fix_applied_v385"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v385"]}`.

### Interpretation

The current Paper 4 waves are healthy under targeted guardrails. The full-suite
claim remains blocked by a separate historical Quarto registration gap: 70
standalone chapter files are absent from `book/_quarto.yml`.

### Claim Impact

- Allowed: validation-gap triage and targeted guardrail pass statement.
- Still prohibited: full-regression-clean, Quarto-registration-fixed, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v385 in the living notebook. v386 should decide what to do with the
historical standalone Quarto pages without promoting Paper 4.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v384_status = json.loads((STATUS_DIR / "paper4_v384_status.json").read_text(encoding="utf-8"))
    if v384_status["next_artifact_v384"] != "paper4_v385_validation_gap_triage.md":
        raise RuntimeError("v385 expects v384 to route to validation gap triage.")
    missing_pages = _missing_quarto_pages()
    triage = _validation_triage(missing_pages)
    decisions = _validation_decisions(missing_pages)
    blockers = _claim_blockers(missing_pages)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v385_quarto_missing_pages_register.csv", missing_pages)
    write_csv(TABLE_DIR / "paper4_v385_validation_gap_triage.csv", triage)
    write_csv(TABLE_DIR / "paper4_v385_validation_decision_matrix.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v385_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v385_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    curated_missing = int(missing_pages["is_curated_paper4_page_v385"].astype(bool).sum())
    status = {
        "phase": "v385_validation_gap_triage",
        "schema_version": "2026-05-17.385",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_formal_review_version_v385": PRIOR_FORMAL_REVIEW_VERSION,
        "validation_triage_rows_v385": int(len(triage)),
        "validation_decision_rows_v385": int(len(decisions)),
        "quarto_missing_page_rows_v385": int(len(missing_pages)),
        "curated_missing_page_rows_v385": curated_missing,
        "claim_blocker_rows_v385": int(len(blockers)),
        "claim_matrix_rows_v385": int(len(claim_matrix)),
        "known_quarto_registration_failure_isolated_v385": True,
        "current_paper4_guardrail_chain_clean_v385": True,
        "targeted_guardrail_selected_tests_v385": 7,
        "full_regression_suite_clean_v385": False,
        "quarto_registration_guardrail_clean_v385": False,
        "quarto_registration_fix_applied_v385": False,
        "curated_paper4_pages_missing_v385": False,
        "working_champion_claim_allowed_v385": False,
        "paper1_promotion_allowed_v385": False,
        "paper4_working_champion_changed_v385": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "triage_artifact_v385": (
            "reports/paper_material/paper4/notes/paper4_v385_validation_gap_triage.md"
        ),
        "next_artifact_v385": NEXT_ARTIFACT,
        "claim_boundary": (
            "v385 isolates a known Quarto registration gap; current Paper 4 focal "
            "guardrails pass, but full regression clean claim remains blocked"
        ),
    }
    TRIAGE_MD.write_text(_triage_markdown(status, missing_pages), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v385_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v385": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
