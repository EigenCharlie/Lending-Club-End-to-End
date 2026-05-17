#!/usr/bin/env python3
"""Build Paper 4 v391 targeted lint repair batch artifacts."""

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

VERSION = 391
PRIOR_LINT_FRONTIER_VERSION = 390
NEXT_VERSION = 392
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_lint_policy.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v391_targeted_lint_repair_batch.md"
TARGET_FILE = "tests/test_docs/test_paper4_living_lab_guardrails.py"
PAPER4_GUARDRAIL_TEST_COMMAND = "uv run pytest -q tests/test_docs/test_paper4_living_lab_guardrails.py"
PAPER4_GUARDRAIL_TESTS_PASSED = 406
PAPER4_GUARDRAIL_TEST_RUNTIME_SECONDS = 101.22


def _repair_actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repair_id_v391": "paper4_guardrail_f541_fix",
                "target_file_v391": TARGET_FILE,
                "rule_code_v391": "F541",
                "pre_error_count_v391": 1,
                "post_error_count_v391": 0,
                "errors_reduced_v391": 1,
                "repair_method_v391": "ruff_safe_fix",
                "claim_boundary_v391": "test string cleanup only",
            },
            {
                "repair_id_v391": "paper4_guardrail_f841_config_reads",
                "target_file_v391": TARGET_FILE,
                "rule_code_v391": "F841",
                "pre_error_count_v391": 19,
                "post_error_count_v391": 0,
                "errors_reduced_v391": 19,
                "repair_method_v391": "ruff_unsafe_fix_preserving_read_side_effect",
                "claim_boundary_v391": "unused local assignments removed; read_text side effect retained",
            },
            {
                "repair_id_v391": "paper4_guardrail_targeted_lint_check",
                "target_file_v391": TARGET_FILE,
                "rule_code_v391": "F841,F541",
                "pre_error_count_v391": 20,
                "post_error_count_v391": 0,
                "errors_reduced_v391": 20,
                "repair_method_v391": "uv run ruff check target --select F841,F541",
                "claim_boundary_v391": "targeted lint surface only",
            },
        ]
    )


def _lint_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_v391": "global_ruff_total_errors",
                "before_v391": 282,
                "after_v391": 262,
                "delta_v391": -20,
                "claim_boundary_v391": "global ruff still fails",
            },
            {
                "metric_v391": "global_ruff_fixable_errors",
                "before_v391": 88,
                "after_v391": 68,
                "delta_v391": -20,
                "claim_boundary_v391": "fixable count reduced; unresolved lint remains",
            },
            {
                "metric_v391": "paper4_guardrail_file_errors",
                "before_v391": 20,
                "after_v391": 0,
                "delta_v391": -20,
                "claim_boundary_v391": "target file clean for current ruff scan",
            },
            {
                "metric_v391": "f841_unused_local_errors",
                "before_v391": 26,
                "after_v391": 7,
                "delta_v391": -19,
                "claim_boundary_v391": "remaining F841 errors are outside target file",
            },
            {
                "metric_v391": "f541_fstring_without_placeholder_errors",
                "before_v391": 5,
                "after_v391": 4,
                "delta_v391": -1,
                "claim_boundary_v391": "remaining F541 errors are outside target file",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v391": "global_ruff_not_clean",
                "blocking_v391": True,
                "evidence_count_v391": 262,
                "required_next_artifact_v391": NEXT_ARTIFACT,
                "claim_boundary_v391": "targeted repair reduced errors but global ruff still fails",
            },
            {
                "blocker_id_v391": "notebook_lint_surface_unrepaired",
                "blocking_v391": True,
                "evidence_count_v391": 180,
                "required_next_artifact_v391": NEXT_ARTIFACT,
                "claim_boundary_v391": "notebook import/cell hygiene remains dominant",
            },
            {
                "blocker_id_v391": "full_repository_pytest_not_rerun_after_lint_repair",
                "blocking_v391": True,
                "evidence_count_v391": 1,
                "required_next_artifact_v391": "paper4_v393_post_lint_pytest_refresh.md",
                "claim_boundary_v391": "v391 validates target tests, not full pytest",
            },
            {
                "blocker_id_v391": "paper4_final_promotion_forbidden",
                "blocking_v391": True,
                "evidence_count_v391": 1,
                "required_next_artifact_v391": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v391": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v391_targeted_lint_repair_created",
                "allowed": True,
                "artifact": "paper4_v391_targeted_lint_repair_batch.csv",
                "boundary": "Paper 4 guardrail lint subset only",
            },
            {
                "claim_id": "v391_paper4_guardrail_f841_f541_clean",
                "allowed": True,
                "artifact": TARGET_FILE,
                "boundary": "F841/F541 selected rules only",
            },
            {
                "claim_id": "v391_global_ruff_errors_reduced",
                "allowed": True,
                "artifact": "paper4_v391_lint_delta.csv",
                "boundary": "282 to 262 diagnostics; still failing",
            },
            {
                "claim_id": "v391_global_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v391_claim_blockers.csv",
                "boundary": "262 diagnostics remain",
            },
            {
                "claim_id": "v391_full_repository_pytest_clean_after_repair",
                "allowed": False,
                "artifact": "paper4_v391_claim_blockers.csv",
                "boundary": "full pytest not rerun after lint repair",
            },
            {
                "claim_id": "v391_working_champion_or_final_promotion",
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
                "claim": "v391 cleans the Paper 4 guardrail file for F841/F541 lint.",
                "allowed": True,
                "evidence_artifact": TARGET_FILE,
                "boundary": "Selected lint rules only; global ruff still fails.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v391 reduces global ruff diagnostics from 282 to 262.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v391_lint_delta.csv"
                ),
                "boundary": "Reduction only; not a global lint clean claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v391 proves global ruff or post-repair full pytest is clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v391_claim_blockers.csv"
                ),
                "boundary": "Global ruff still fails and full pytest was not rerun in v391.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v391 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v391_claim_blockers.csv"
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
                    "v391 repairs the safest Paper 4 guardrail lint subset and records "
                    "the remaining repository lint frontier."
                ),
                "status": "targeted_paper4_guardrail_lint_repair_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v392 decides how to handle notebook lint without damaging executed notebooks"
                ),
                "last_wave": "v391",
                "execution_result": "ruff_global_reduced_282_to_262_target_file_clean",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v391")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Targeted Lint Repair Batch v391

Generated: {status["generated_at_utc"]}

v391 applies the safest repair from the v390 lint frontier: F841/F541 cleanup in
the Paper 4 living-lab guardrail file.

## Result

- Target file: `{status["target_file_v391"]}`.
- Target-file F841/F541 diagnostics after repair:
  `{status["target_file_selected_lint_errors_after_v391"]}`.
- Global ruff diagnostics before/after:
  `{status["global_ruff_errors_before_v391"]}` ->
  `{status["global_ruff_errors_after_v391"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v391"]}`.
- Paper 4 guardrail file tests:
  `{status["paper4_guardrail_file_tests_passed_v391"]}` passed.

## Required Caveat

v391 is a targeted lint repair only. It does not claim global ruff cleanliness,
post-repair full pytest cleanliness, full Quarto render success, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v391"]}` to decide the notebook lint policy.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V391_TARGETED_LINT_REPAIR_BATCH_START -->"
    end = "<!-- V391_TARGETED_LINT_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v391: Targeted Lint Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v391 repairs the safest lint subset from v390: F841/F541 diagnostics in the
Paper 4 living-lab guardrail file.

### Results

- Target file:
  `{status["target_file_v391"]}`.
- Target selected lint errors before:
  `{status["target_file_selected_lint_errors_before_v391"]}`.
- Target selected lint errors after:
  `{status["target_file_selected_lint_errors_after_v391"]}`.
- Global ruff errors before:
  `{status["global_ruff_errors_before_v391"]}`.
- Global ruff errors after:
  `{status["global_ruff_errors_after_v391"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v391"]}`.
- Full pytest rerun after repair:
  `{status["full_repository_pytest_rerun_after_repair_v391"]}`.
- Paper 4 guardrail file tests passed:
  `{status["paper4_guardrail_file_tests_passed_v391"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v391"]}`.

### Interpretation

The long Paper 4 guardrail file is no longer part of the F841/F541 lint blocker.
The remaining lint frontier is now more clearly a notebook and Streamlit-page
cleanup problem.

### Claim Impact

- Allowed: targeted lint repair and global diagnostic reduction.
- Still prohibited: global ruff clean, post-repair full pytest clean, full Quarto
  render, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v391 in the living notebook. v392 should decide notebook lint policy before
bulk notebook mutation.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v390_status = json.loads((STATUS_DIR / "paper4_v390_status.json").read_text(encoding="utf-8"))
    if v390_status["next_artifact_v390"] != "paper4_v391_targeted_lint_repair_batch.md":
        raise RuntimeError("v391 expects v390 to route to targeted lint repair batch.")

    actions = _repair_actions()
    delta = _lint_delta()
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v391_targeted_lint_repair_batch.csv", actions)
    write_csv(TABLE_DIR / "paper4_v391_lint_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v391_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v391_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v391_targeted_lint_repair_batch",
        "schema_version": "2026-05-17.391",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_lint_frontier_version_v391": PRIOR_LINT_FRONTIER_VERSION,
        "repair_action_rows_v391": int(len(actions)),
        "lint_delta_rows_v391": int(len(delta)),
        "claim_blocker_rows_v391": int(len(blockers)),
        "claim_matrix_rows_v391": int(len(claim_matrix)),
        "target_file_v391": TARGET_FILE,
        "target_file_selected_lint_errors_before_v391": 20,
        "target_file_selected_lint_errors_after_v391": 0,
        "target_file_errors_reduced_v391": 20,
        "global_ruff_errors_before_v391": 282,
        "global_ruff_errors_after_v391": 262,
        "global_ruff_errors_reduced_v391": 20,
        "global_ruff_fixable_before_v391": 88,
        "global_ruff_fixable_after_v391": 68,
        "global_ruff_clean_v391": False,
        "paper4_guardrail_file_lint_clean_for_selected_rules_v391": True,
        "paper4_guardrail_file_test_command_v391": PAPER4_GUARDRAIL_TEST_COMMAND,
        "paper4_guardrail_file_tests_passed_v391": PAPER4_GUARDRAIL_TESTS_PASSED,
        "paper4_guardrail_file_test_runtime_seconds_v391": (
            PAPER4_GUARDRAIL_TEST_RUNTIME_SECONDS
        ),
        "full_repository_pytest_rerun_after_repair_v391": False,
        "full_repository_pytest_clean_after_repair_v391": False,
        "full_quarto_render_run_v391": False,
        "full_quarto_render_clean_v391": False,
        "working_champion_claim_allowed_v391": False,
        "paper1_promotion_allowed_v391": False,
        "paper4_working_champion_changed_v391": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "repair_artifact_v391": (
            "reports/paper_material/paper4/tables/"
            "paper4_v391_targeted_lint_repair_batch.csv"
        ),
        "next_artifact_v391": NEXT_ARTIFACT,
        "claim_boundary": (
            "v391 reduces global ruff diagnostics from 282 to 262 by cleaning "
            "the Paper 4 guardrail F841/F541 subset; global lint remains blocked"
        ),
    }
    REPAIR_MD.write_text(_repair_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v391_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v391": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
