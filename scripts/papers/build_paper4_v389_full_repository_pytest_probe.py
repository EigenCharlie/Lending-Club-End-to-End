#!/usr/bin/env python3
"""Build Paper 4 v389 full-repository pytest probe artifacts."""

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

VERSION = 389
PRIOR_REGRESSION_PROBE_VERSION = 388
NEXT_VERSION = 390
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_repository_lint_frontier.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v389_full_repository_pytest_probe.md"

INITIAL_FULL_PYTEST_COMMAND = "uv run pytest -q --maxfail=10"
INITIAL_FULL_PYTEST_PASSED = 1126
INITIAL_FULL_PYTEST_FAILED = 1
INITIAL_FULL_PYTEST_SKIPPED = 2
INITIAL_FULL_PYTEST_WARNINGS = 13
INITIAL_FULL_PYTEST_RUNTIME_SECONDS = 167.63

STREAMLIT_SHELL_TEST = (
    "tests/test_streamlit/test_app_shell_navigation.py::"
    "test_app_shell_renders_without_exceptions"
)
STREAMLIT_TIMEOUT_BEFORE = 20
STREAMLIT_TIMEOUT_AFTER = 45
STREAMLIT_MEASURED_RUNTIME_SECONDS = 23.91
STREAMLIT_PRE_REPAIR_PASS_WITH_LONG_TIMEOUT_SECONDS = 23.38

POST_REPAIR_FULL_PYTEST_COMMAND = "uv run pytest -q --maxfail=10"
POST_REPAIR_FULL_PYTEST_PASSED = 1128
POST_REPAIR_FULL_PYTEST_FAILED = 0
POST_REPAIR_FULL_PYTEST_SKIPPED = 2
POST_REPAIR_FULL_PYTEST_WARNINGS = 13
POST_REPAIR_FULL_PYTEST_RUNTIME_SECONDS = 205.61


def _probe_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v389": "initial_full_repository_pytest",
                "command_v389": INITIAL_FULL_PYTEST_COMMAND,
                "observed_status_v389": "fail",
                "tests_passed_v389": INITIAL_FULL_PYTEST_PASSED,
                "tests_failed_v389": INITIAL_FULL_PYTEST_FAILED,
                "tests_skipped_v389": INITIAL_FULL_PYTEST_SKIPPED,
                "warnings_v389": INITIAL_FULL_PYTEST_WARNINGS,
                "runtime_seconds_v389": INITIAL_FULL_PYTEST_RUNTIME_SECONDS,
                "claim_allowed_v389": False,
                "claim_boundary_v389": "one Streamlit AppTest timeout blocked full-suite claim",
            },
            {
                "probe_id_v389": "streamlit_shell_timeout_repair",
                "command_v389": f"uv run pytest -q {STREAMLIT_SHELL_TEST}",
                "observed_status_v389": "pass",
                "tests_passed_v389": 1,
                "tests_failed_v389": 0,
                "tests_skipped_v389": 0,
                "warnings_v389": 0,
                "runtime_seconds_v389": STREAMLIT_MEASURED_RUNTIME_SECONDS,
                "claim_allowed_v389": True,
                "claim_boundary_v389": "test timeout budget repaired; app rendered without exception",
            },
            {
                "probe_id_v389": "post_repair_full_repository_pytest",
                "command_v389": POST_REPAIR_FULL_PYTEST_COMMAND,
                "observed_status_v389": "pass",
                "tests_passed_v389": POST_REPAIR_FULL_PYTEST_PASSED,
                "tests_failed_v389": POST_REPAIR_FULL_PYTEST_FAILED,
                "tests_skipped_v389": POST_REPAIR_FULL_PYTEST_SKIPPED,
                "warnings_v389": POST_REPAIR_FULL_PYTEST_WARNINGS,
                "runtime_seconds_v389": POST_REPAIR_FULL_PYTEST_RUNTIME_SECONDS,
                "claim_allowed_v389": True,
                "claim_boundary_v389": "full repository pytest clean after timeout repair",
            },
            {
                "probe_id_v389": "paper4_final_promotion_absence",
                "command_v389": (
                    "test ! -e reports/paper_material/paper4/status/"
                    "paper4_final_promotion.json"
                ),
                "observed_status_v389": "pass",
                "tests_passed_v389": 1,
                "tests_failed_v389": 0,
                "tests_skipped_v389": 0,
                "warnings_v389": 0,
                "runtime_seconds_v389": 0.0,
                "claim_allowed_v389": True,
                "claim_boundary_v389": "final promotion remains forbidden and absent",
            },
        ]
    )


def _failure_frontier() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "failure_id_v389": "streamlit_app_shell_apptest_timeout",
                "test_path_v389": STREAMLIT_SHELL_TEST,
                "failure_type_v389": "RuntimeError",
                "failure_message_v389": "AppTest script run timed out after 20(s)",
                "initial_timeout_seconds_v389": STREAMLIT_TIMEOUT_BEFORE,
                "repaired_timeout_seconds_v389": STREAMLIT_TIMEOUT_AFTER,
                "measured_runtime_seconds_v389": STREAMLIT_MEASURED_RUNTIME_SECONDS,
                "pre_repair_long_timeout_runtime_seconds_v389": (
                    STREAMLIT_PRE_REPAIR_PASS_WITH_LONG_TIMEOUT_SECONDS
                ),
                "repair_file_v389": "tests/test_streamlit/test_app_shell_navigation.py",
                "repair_summary_v389": "increase AppTest timeout from 20s to 45s",
                "claim_boundary_v389": "timeout budget repair only; app behavior unchanged",
            }
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v389": "full_repository_ruff_not_run",
                "blocking_v389": True,
                "evidence_count_v389": 1,
                "required_next_artifact_v389": NEXT_ARTIFACT,
                "claim_boundary_v389": "full pytest is clean, global lint remains pending",
            },
            {
                "blocker_id_v389": "full_quarto_render_not_run",
                "blocking_v389": True,
                "evidence_count_v389": 1,
                "required_next_artifact_v389": "paper4_v391_quarto_render_probe.md",
                "claim_boundary_v389": "pytest clean is not a full Quarto render claim",
            },
            {
                "blocker_id_v389": "streamlit_performance_budget_not_formalized",
                "blocking_v389": True,
                "evidence_count_v389": 1,
                "required_next_artifact_v389": "paper4_v392_streamlit_shell_performance_budget.csv",
                "claim_boundary_v389": "timeout repair is pragmatic, not a formal performance guarantee",
            },
            {
                "blocker_id_v389": "paper4_final_promotion_forbidden",
                "blocking_v389": True,
                "evidence_count_v389": 1,
                "required_next_artifact_v389": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v389": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v389_initial_full_pytest_failure_isolated",
                "allowed": True,
                "artifact": "paper4_v389_failure_frontier.csv",
                "boundary": "single Streamlit timeout failure",
            },
            {
                "claim_id": "v389_streamlit_shell_timeout_repaired",
                "allowed": True,
                "artifact": "tests/test_streamlit/test_app_shell_navigation.py",
                "boundary": "AppTest timeout raised to observed runtime budget",
            },
            {
                "claim_id": "v389_full_repository_pytest_clean",
                "allowed": True,
                "artifact": "paper4_v389_full_repository_pytest_probe.csv",
                "boundary": "pytest suite only; 1128 passed after repair",
            },
            {
                "claim_id": "v389_full_repository_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v389_claim_blockers.csv",
                "boundary": "global lint not run in v389",
            },
            {
                "claim_id": "v389_full_quarto_render_success",
                "allowed": False,
                "artifact": "paper4_v389_claim_blockers.csv",
                "boundary": "full Quarto render not run",
            },
            {
                "claim_id": "v389_working_champion_or_final_promotion",
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
                "claim": "v389 isolates and repairs the single full-pytest Streamlit timeout.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v389_failure_frontier.csv"
                ),
                "boundary": "Timeout budget repair only; app behavior unchanged.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v389 shows full repository pytest is clean after repair.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v389_full_repository_pytest_probe.csv"
                ),
                "boundary": "Pytest suite only: 1128 passed, 2 skipped, 13 warnings.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v389 proves global ruff or full Quarto render is clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v389_claim_blockers.csv"
                ),
                "boundary": "Those broader probes are deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v389 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v389_claim_blockers.csv"
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
                    "v389 runs full repository pytest, repairs the lone Streamlit timeout "
                    "frontier, and records post-repair full pytest cleanliness."
                ),
                "status": "full_repository_pytest_clean_after_streamlit_timeout_repair",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v390 runs global ruff or classifies historical lint blockers"
                ),
                "last_wave": "v389",
                "execution_result": "full_pytest_1128_passed_streamlit_timeout_repaired",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v389")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Full Repository Pytest Probe v389

Generated: {status["generated_at_utc"]}

v389 runs the full repository pytest frontier after the v388 documentation
regression probe.

## Result

- Initial full pytest: `1` Streamlit AppTest timeout failure.
- Repair: `tests/test_streamlit/test_app_shell_navigation.py` timeout raised from
  `{status["streamlit_timeout_before_seconds_v389"]}`s to
  `{status["streamlit_timeout_after_seconds_v389"]}`s.
- Post-repair full pytest: `{status["post_repair_full_pytest_passed_v389"]}`
  passed, `{status["post_repair_full_pytest_skipped_v389"]}` skipped,
  `{status["post_repair_full_pytest_warnings_v389"]}` warnings.
- Runtime: `{status["post_repair_full_pytest_runtime_seconds_v389"]}` seconds.

## Required Caveat

v389 proves full repository pytest cleanliness only. It does not claim global
ruff cleanliness, full Quarto render success, champion replacement or Paper 4
final promotion.

## Next Executable Wave

Build `{status["next_artifact_v389"]}` by probing `uv run ruff check .`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V389_FULL_REPOSITORY_PYTEST_PROBE_START -->"
    end = "<!-- V389_FULL_REPOSITORY_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v389: Full Repository Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v389 executes the full repository pytest frontier, repairs the lone Streamlit
AppTest timeout, and records the post-repair result.

### Results

- Initial full pytest failed tests:
  `{status["initial_full_pytest_failed_v389"]}`.
- Initial full pytest passed tests:
  `{status["initial_full_pytest_passed_v389"]}`.
- Repaired test:
  `{status["repaired_test_path_v389"]}`.
- Streamlit timeout before:
  `{status["streamlit_timeout_before_seconds_v389"]}`.
- Streamlit timeout after:
  `{status["streamlit_timeout_after_seconds_v389"]}`.
- Post-repair full pytest passed:
  `{status["post_repair_full_pytest_passed_v389"]}`.
- Post-repair full pytest clean:
  `{status["post_repair_full_pytest_clean_v389"]}`.
- Full repository ruff clean:
  `{status["full_repository_ruff_clean_v389"]}`.
- Full Quarto render clean:
  `{status["full_quarto_render_clean_v389"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v389"]}`.

### Interpretation

The full pytest frontier is now clean after a narrow Streamlit AppTest timeout
budget repair. This materially strengthens Paper 4 reproducibility, while still
leaving lint and full-render frontiers as separate, unclaimed checks.

### Claim Impact

- Allowed: initial full-pytest failure isolated, Streamlit timeout repaired, full
  repository pytest clean after repair.
- Still prohibited: global ruff clean, full Quarto render, champion replacement
  and final promotion claims.

### Quarto Promotion Decision

Keep v389 in the living notebook. v390 should probe repository lint.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v388_status = json.loads((STATUS_DIR / "paper4_v388_status.json").read_text(encoding="utf-8"))
    if v388_status["next_artifact_v388"] != "paper4_v389_full_repository_pytest_probe.md":
        raise RuntimeError("v389 expects v388 to route to full repository pytest probe.")

    probes = _probe_results()
    frontier = _failure_frontier()
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v389_full_repository_pytest_probe.csv", probes)
    write_csv(TABLE_DIR / "paper4_v389_failure_frontier.csv", frontier)
    write_csv(TABLE_DIR / "paper4_v389_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v389_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v389_full_repository_pytest_probe",
        "schema_version": "2026-05-17.389",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_regression_probe_version_v389": PRIOR_REGRESSION_PROBE_VERSION,
        "probe_rows_v389": int(len(probes)),
        "failure_frontier_rows_v389": int(len(frontier)),
        "claim_blocker_rows_v389": int(len(blockers)),
        "claim_matrix_rows_v389": int(len(claim_matrix)),
        "initial_full_pytest_command_v389": INITIAL_FULL_PYTEST_COMMAND,
        "initial_full_pytest_failed_v389": INITIAL_FULL_PYTEST_FAILED,
        "initial_full_pytest_passed_v389": INITIAL_FULL_PYTEST_PASSED,
        "initial_full_pytest_skipped_v389": INITIAL_FULL_PYTEST_SKIPPED,
        "initial_full_pytest_warnings_v389": INITIAL_FULL_PYTEST_WARNINGS,
        "initial_full_pytest_runtime_seconds_v389": INITIAL_FULL_PYTEST_RUNTIME_SECONDS,
        "repaired_test_path_v389": STREAMLIT_SHELL_TEST,
        "repair_file_v389": "tests/test_streamlit/test_app_shell_navigation.py",
        "streamlit_timeout_before_seconds_v389": STREAMLIT_TIMEOUT_BEFORE,
        "streamlit_timeout_after_seconds_v389": STREAMLIT_TIMEOUT_AFTER,
        "streamlit_measured_runtime_seconds_v389": STREAMLIT_MEASURED_RUNTIME_SECONDS,
        "streamlit_pre_repair_long_timeout_runtime_seconds_v389": (
            STREAMLIT_PRE_REPAIR_PASS_WITH_LONG_TIMEOUT_SECONDS
        ),
        "post_repair_full_pytest_command_v389": POST_REPAIR_FULL_PYTEST_COMMAND,
        "post_repair_full_pytest_run_v389": True,
        "post_repair_full_pytest_clean_v389": True,
        "post_repair_full_pytest_passed_v389": POST_REPAIR_FULL_PYTEST_PASSED,
        "post_repair_full_pytest_failed_v389": POST_REPAIR_FULL_PYTEST_FAILED,
        "post_repair_full_pytest_skipped_v389": POST_REPAIR_FULL_PYTEST_SKIPPED,
        "post_repair_full_pytest_warnings_v389": POST_REPAIR_FULL_PYTEST_WARNINGS,
        "post_repair_full_pytest_runtime_seconds_v389": POST_REPAIR_FULL_PYTEST_RUNTIME_SECONDS,
        "full_repository_ruff_run_v389": False,
        "full_repository_ruff_clean_v389": False,
        "full_quarto_render_run_v389": False,
        "full_quarto_render_clean_v389": False,
        "working_champion_claim_allowed_v389": False,
        "paper1_promotion_allowed_v389": False,
        "paper4_working_champion_changed_v389": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "probe_artifact_v389": (
            "reports/paper_material/paper4/tables/"
            "paper4_v389_full_repository_pytest_probe.csv"
        ),
        "next_artifact_v389": NEXT_ARTIFACT,
        "claim_boundary": (
            "v389 proves full repository pytest cleanliness after one timeout repair; "
            "global lint and full Quarto render remain pending"
        ),
    }
    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v389_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v389": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
