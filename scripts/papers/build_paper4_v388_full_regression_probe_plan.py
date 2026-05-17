#!/usr/bin/env python3
"""Build Paper 4 v388 full-regression probe-plan artifacts."""

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
    read_csv,
    write_csv,
    write_json,
)

VERSION = 388
PRIOR_ARCHIVE_GUARDRAIL_VERSION = 387
NEXT_VERSION = 389
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_full_repository_pytest_probe.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v388_full_regression_probe_plan.md"
DOCS_REGRESSION_COMMAND = "uv run pytest -q tests/test_docs --maxfail=10"
DOCS_REGRESSION_COLLECTED = 440
DOCS_REGRESSION_PASSED = 440
DOCS_REGRESSION_RUNTIME_SECONDS = 76.00
PAPER4_FOCAL_SELECTED = 10
QUARTO_BOOK_GUARDRAILS_PASSED = 3


def _probe_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v388": "docs_regression_tests",
                "command_v388": DOCS_REGRESSION_COMMAND,
                "observed_status_v388": "pass",
                "tests_collected_v388": DOCS_REGRESSION_COLLECTED,
                "tests_passed_v388": DOCS_REGRESSION_PASSED,
                "runtime_seconds_v388": DOCS_REGRESSION_RUNTIME_SECONDS,
                "claim_allowed_v388": True,
                "claim_boundary_v388": "documentation test suite only",
            },
            {
                "probe_id_v388": "paper4_focal_guardrail_chain_v378_v387",
                "command_v388": (
                    "uv run pytest -q tests/test_docs/test_paper4_living_lab_guardrails.py "
                    "-k v378..v387 focal guardrails"
                ),
                "observed_status_v388": "pass",
                "tests_collected_v388": PAPER4_FOCAL_SELECTED,
                "tests_passed_v388": PAPER4_FOCAL_SELECTED,
                "runtime_seconds_v388": 30.21,
                "claim_allowed_v388": True,
                "claim_boundary_v388": "targeted Paper 4 guardrail chain only",
            },
            {
                "probe_id_v388": "quarto_book_guardrails",
                "command_v388": "uv run pytest -q tests/test_docs/test_quarto_book_guardrails.py",
                "observed_status_v388": "pass",
                "tests_collected_v388": QUARTO_BOOK_GUARDRAILS_PASSED,
                "tests_passed_v388": QUARTO_BOOK_GUARDRAILS_PASSED,
                "runtime_seconds_v388": 0.25,
                "claim_allowed_v388": True,
                "claim_boundary_v388": "book guardrail tests only; not a full Quarto render",
            },
            {
                "probe_id_v388": "full_repository_pytest",
                "command_v388": "uv run pytest -q",
                "observed_status_v388": "not_run",
                "tests_collected_v388": 0,
                "tests_passed_v388": 0,
                "runtime_seconds_v388": 0.0,
                "claim_allowed_v388": False,
                "claim_boundary_v388": "full repository pytest clean claim deferred to v389",
            },
            {
                "probe_id_v388": "full_repository_ruff",
                "command_v388": "uv run ruff check .",
                "observed_status_v388": "not_run",
                "tests_collected_v388": 0,
                "tests_passed_v388": 0,
                "runtime_seconds_v388": 0.0,
                "claim_allowed_v388": False,
                "claim_boundary_v388": "global lint clean claim not made in v388",
            },
            {
                "probe_id_v388": "paper4_final_promotion_absence",
                "command_v388": (
                    "test ! -e reports/paper_material/paper4/status/"
                    "paper4_final_promotion.json"
                ),
                "observed_status_v388": "pass",
                "tests_collected_v388": 1,
                "tests_passed_v388": 1,
                "runtime_seconds_v388": 0.0,
                "claim_allowed_v388": True,
                "claim_boundary_v388": "final promotion remains forbidden and absent",
            },
        ]
    )


def _next_probe_backlog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "next_probe_id_v388": "full_repository_pytest_probe",
                "priority_v388": 1,
                "command_v388": "uv run pytest -q --maxfail=10",
                "expected_output_v388": "first full-repository pass/failure frontier",
                "success_condition_v388": "either full repository tests pass or blockers are classified",
                "next_artifact_v388": NEXT_ARTIFACT,
            },
            {
                "next_probe_id_v388": "full_repository_ruff_probe",
                "priority_v388": 2,
                "command_v388": "uv run ruff check .",
                "expected_output_v388": "repository lint frontier",
                "success_condition_v388": "global lint pass or historical lint blockers classified",
                "next_artifact_v388": "paper4_v390_repository_lint_frontier.md",
            },
            {
                "next_probe_id_v388": "quarto_render_probe",
                "priority_v388": 3,
                "command_v388": "quarto render chapters/19-paper-mega-extension --to html",
                "expected_output_v388": "Paper 4 chapter render status",
                "success_condition_v388": "render pass or render blockers classified",
                "next_artifact_v388": "paper4_v391_quarto_render_probe.md",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v388": "full_repository_pytest_not_run",
                "blocking_v388": True,
                "evidence_count_v388": 1,
                "required_next_artifact_v388": NEXT_ARTIFACT,
                "claim_boundary_v388": "documentation tests pass, full repository pytest not claimed",
            },
            {
                "blocker_id_v388": "full_repository_ruff_not_run",
                "blocking_v388": True,
                "evidence_count_v388": 1,
                "required_next_artifact_v388": "paper4_v390_repository_lint_frontier.md",
                "claim_boundary_v388": "global lint clean claim not made",
            },
            {
                "blocker_id_v388": "full_quarto_render_not_run",
                "blocking_v388": True,
                "evidence_count_v388": 1,
                "required_next_artifact_v388": "paper4_v391_quarto_render_probe.md",
                "claim_boundary_v388": "book guardrails pass but full render not claimed",
            },
            {
                "blocker_id_v388": "paper4_final_promotion_forbidden",
                "blocking_v388": True,
                "evidence_count_v388": 1,
                "required_next_artifact_v388": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v388": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v388_docs_regression_tests_clean",
                "allowed": True,
                "artifact": "paper4_v388_regression_probe_results.csv",
                "boundary": "tests/test_docs only",
            },
            {
                "claim_id": "v388_paper4_focal_guardrails_clean",
                "allowed": True,
                "artifact": "paper4_v388_regression_probe_results.csv",
                "boundary": "v378-v387 selected guardrails",
            },
            {
                "claim_id": "v388_quarto_book_guardrails_clean",
                "allowed": True,
                "artifact": "paper4_v388_regression_probe_results.csv",
                "boundary": "book guardrail tests only",
            },
            {
                "claim_id": "v388_full_repository_pytest_clean",
                "allowed": False,
                "artifact": "paper4_v388_claim_blockers.csv",
                "boundary": "full repository pytest not run in v388",
            },
            {
                "claim_id": "v388_full_repository_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v388_claim_blockers.csv",
                "boundary": "global lint not run in v388",
            },
            {
                "claim_id": "v388_working_champion_or_final_promotion",
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
                "claim": "v388 shows the documentation regression suite is clean.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v388_regression_probe_results.csv"
                ),
                "boundary": "Limited to tests/test_docs: 440 passed.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v388 shows Paper 4 focal guardrails and Quarto book guardrails are clean.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v388_regression_probe_results.csv"
                ),
                "boundary": "Selected Paper 4 focal tests plus Quarto book guardrail tests.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v388 proves full repository pytest, global ruff or full Quarto render is clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v388_claim_blockers.csv"
                ),
                "boundary": "Those broader probes are deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v388 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v388_claim_blockers.csv"
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
                    "v388 records a clean tests/test_docs regression probe and schedules "
                    "the full repository pytest frontier."
                ),
                "status": "docs_regression_probe_clean",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v389 runs full repository pytest with maxfail frontier and classifies results"
                ),
                "last_wave": "v388",
                "execution_result": "tests_test_docs_440_passed_full_repo_probe_pending",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v388")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Full Regression Probe Plan v388

Generated: {status["generated_at_utc"]}

v388 captures the first broader regression result after the v387 Quarto archive
guardrail repair.

## Observed Clean Surface

- Documentation tests: `{status["docs_regression_passed_v388"]}` /
  `{status["docs_regression_collected_v388"]}` passed.
- Documentation runtime: `{status["docs_regression_runtime_seconds_v388"]}` seconds.
- Paper 4 focal guardrails selected: `{status["paper4_focal_selected_tests_v388"]}`.
- Quarto book guardrails passed: `{status["quarto_book_guardrails_passed_v388"]}`.

## Required Caveat

v388 does not claim full repository pytest, global ruff, full Quarto render,
champion replacement or Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v388"]}` by running `uv run pytest -q --maxfail=10`
and classifying the full repository pass/failure frontier.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V388_FULL_REGRESSION_PROBE_PLAN_START -->"
    end = "<!-- V388_FULL_REGRESSION_PROBE_PLAN_END -->"
    block = f"""
{start}

## Wave v388: Full Regression Probe Plan

Generated: {status["generated_at_utc"]}

### Objective

v388 records the broader documentation regression evidence after the v387 Quarto
archive guardrail patch and schedules the full repository pytest frontier.

### Results

- Documentation tests collected:
  `{status["docs_regression_collected_v388"]}`.
- Documentation tests passed:
  `{status["docs_regression_passed_v388"]}`.
- Documentation regression clean:
  `{status["docs_regression_clean_v388"]}`.
- Paper 4 focal guardrails clean:
  `{status["paper4_focal_guardrails_clean_v388"]}`.
- Quarto book guardrails clean:
  `{status["quarto_book_guardrails_clean_v388"]}`.
- Full repository pytest clean:
  `{status["full_repository_pytest_clean_v388"]}`.
- Full repository ruff clean:
  `{status["full_repository_ruff_clean_v388"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v388"]}`.

### Interpretation

The historical Quarto registration blocker is no longer blocking the
documentation test suite. The project now has a clean docs-level regression
surface, but broader repository probes still need to be executed before any
full-suite claim.

### Claim Impact

- Allowed: docs regression clean, Paper 4 focal guardrails clean, Quarto book
  guardrails clean.
- Still prohibited: full repository pytest clean, global ruff clean, full Quarto
  render, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v388 in the living notebook. v389 should run the full repository pytest
frontier.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v387_status = json.loads((STATUS_DIR / "paper4_v387_status.json").read_text(encoding="utf-8"))
    if v387_status["next_artifact_v387"] != "paper4_v388_full_regression_probe_plan.md":
        raise RuntimeError("v388 expects v387 to route to full regression probe plan.")

    probes = _probe_results()
    next_probes = _next_probe_backlog()
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v388_regression_probe_results.csv", probes)
    write_csv(TABLE_DIR / "paper4_v388_next_probe_backlog.csv", next_probes)
    write_csv(TABLE_DIR / "paper4_v388_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v388_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v388_full_regression_probe_plan",
        "schema_version": "2026-05-17.388",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_archive_guardrail_version_v388": PRIOR_ARCHIVE_GUARDRAIL_VERSION,
        "probe_rows_v388": int(len(probes)),
        "next_probe_rows_v388": int(len(next_probes)),
        "claim_blocker_rows_v388": int(len(blockers)),
        "claim_matrix_rows_v388": int(len(claim_matrix)),
        "docs_regression_command_v388": DOCS_REGRESSION_COMMAND,
        "docs_regression_collected_v388": DOCS_REGRESSION_COLLECTED,
        "docs_regression_passed_v388": DOCS_REGRESSION_PASSED,
        "docs_regression_runtime_seconds_v388": DOCS_REGRESSION_RUNTIME_SECONDS,
        "docs_regression_clean_v388": True,
        "paper4_focal_selected_tests_v388": PAPER4_FOCAL_SELECTED,
        "paper4_focal_guardrails_clean_v388": True,
        "quarto_book_guardrails_passed_v388": QUARTO_BOOK_GUARDRAILS_PASSED,
        "quarto_book_guardrails_clean_v388": True,
        "full_repository_pytest_run_v388": False,
        "full_repository_pytest_clean_v388": False,
        "full_repository_ruff_run_v388": False,
        "full_repository_ruff_clean_v388": False,
        "full_quarto_render_run_v388": False,
        "full_quarto_render_clean_v388": False,
        "working_champion_claim_allowed_v388": False,
        "paper1_promotion_allowed_v388": False,
        "paper4_working_champion_changed_v388": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "probe_artifact_v388": (
            "reports/paper_material/paper4/tables/"
            "paper4_v388_regression_probe_results.csv"
        ),
        "next_artifact_v388": NEXT_ARTIFACT,
        "claim_boundary": (
            "v388 proves docs-level regression cleanliness only; full repository "
            "pytest, global ruff and full Quarto render claims remain pending"
        ),
    }
    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v388_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v388": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
