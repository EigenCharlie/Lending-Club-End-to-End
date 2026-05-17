#!/usr/bin/env python3
"""Build Paper 4 v406 post-sys.path-refactor pytest probe artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_v403_post_notebook_mutation_pytest_probe import (
    _notebook_diff_clean,
    _run_pytest,
    _run_ruff_json,
)
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

VERSION = 406
PRIOR_SYS_PATH_REFACTOR_VERSION = 405
NEXT_PASS_ARTIFACT = "paper4_v407_notebook_non_e402_lint_triage.md"
NEXT_FAIL_ARTIFACT = "paper4_v407_post_sys_path_refactor_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v406_post_sys_path_refactor_pytest_probe.md"


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v406": "full_repository_pytest",
                "command_v406": pytest_result["command"],
                "exit_code_v406": int(pytest_result["exit_code"]),
                "passed_v406": bool(pytest_result["passed"]),
                "runtime_seconds_v406": float(pytest_result["runtime_seconds"]),
                "collected_items_v406": int(pytest_result["collected_items"]),
                "summary_line_v406": str(pytest_result["summary_line"]),
                "claim_boundary_v406": "post-sys.path-refactor pytest probe",
            }
        ]
    )


def _lint_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in items)
    rows = []
    for code, count in sorted(counts.items()):
        rows.append(
            {
                "lint_code_v406": code,
                "diagnostic_count_v406": int(count),
                "claim_boundary_v406": "notebook lint remains visible after sys.path refactor",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, pytest_passed: bool, global_after: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v406": "non_e402_notebook_lint_remaining",
            "blocking_v406": True,
            "evidence_count_v406": global_after,
            "required_next_artifact_v406": NEXT_PASS_ARTIFACT,
            "claim_boundary_v406": "non-E402 notebook lint remains",
        },
        {
            "blocker_id_v406": "paper4_final_promotion_forbidden",
            "blocking_v406": True,
            "evidence_count_v406": 1,
            "required_next_artifact_v406": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v406": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v406": "full_repository_pytest_failed",
                "blocking_v406": True,
                "evidence_count_v406": 1,
                "required_next_artifact_v406": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v406": "pytest failure must be triaged before lint cleanup",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(pytest_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v406_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v406_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v405 sys.path refactor",
            },
            {
                "claim_id": "v406_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v406_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v406_notebook_e402_remains_clear",
                "allowed": True,
                "artifact": "paper4_v406_notebook_lint_snapshot.csv",
                "boundary": "E402 count remains 0",
            },
            {
                "claim_id": "v406_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v406_claim_blockers.csv",
                "boundary": "20 non-E402 notebook diagnostics remain",
            },
            {
                "claim_id": "v406_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(pytest_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v406 runs full repository pytest after notebook E402 is cleared.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v406_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v406 full repository pytest passes after notebook E402 is cleared.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v406_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v406 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v406_claim_blockers.csv",
                "boundary": "20 non-E402 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v406 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v406_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(pytest_passed: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v406 runs the post-v405 full repository pytest probe.",
                "status": (
                    "post_sys_path_refactor_pytest_probe_passed"
                    if pytest_passed
                    else "post_sys_path_refactor_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v407 triages remaining non-E402 notebook lint"
                    if pytest_passed
                    else "v407 triages pytest failures before lint cleanup"
                ),
                "last_wave": "v406",
                "execution_result": (
                    "full_repository_pytest_passed_after_notebook_e402_clearance"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_notebook_e402_clearance"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v406")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Sys.path-Refactor Pytest Probe v406

Generated: {status["generated_at_utc"]}

v406 runs full repository pytest after v405 clears notebook E402.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v406"]}`.
- Pytest passed: `{status["pytest_passed_v406"]}`.
- Collected items: `{status["pytest_collected_items_v406"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v406"]}`.
- Summary: `{status["pytest_summary_line_v406"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v406"]}`.
- Notebook E402 diagnostics: `{status["global_notebook_e402_v406"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v406 does not clear the remaining non-E402 notebook lint and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v406"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V406_POST_SYS_PATH_REFACTOR_PYTEST_PROBE_START -->"
    end = "<!-- V406_POST_SYS_PATH_REFACTOR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v406: Post-Sys.path-Refactor Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v406 runs full repository pytest after v405 clears notebook E402.

### Results

- Pytest command:
  `{status["pytest_command_v406"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v406"]}`.
- Pytest passed:
  `{status["pytest_passed_v406"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v406"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v406"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v406"]}`.
- Notebook E402 diagnostics:
  `{status["global_notebook_e402_v406"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v406"]}`.

### Interpretation

Notebook E402 is closed and post-refactor pytest decides whether the next lane
can focus on the remaining non-E402 notebook diagnostics.

### Claim Impact

- Allowed: full repository pytest was executed after notebook E402 clearance.
- Conditional: pytest pass claim follows the captured exit code.
- Still prohibited: all notebook lint clean, repository ruff clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v406 in the living notebook. Route v407 according to the pytest result.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v406 expects clean notebook diff before pytest probe.")

    v405_status = json.loads((STATUS_DIR / "paper4_v405_status.json").read_text(encoding="utf-8"))
    if v405_status["next_artifact_v405"] != "paper4_v406_post_sys_path_refactor_pytest_probe.md":
        raise RuntimeError("v406 expects v405 to route to post-sys.path-refactor pytest probe.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    current_global = _run_ruff_json()
    counts = Counter(item["code"] for item in current_global)
    pytest_summary = _pytest_summary_table(pytest_result)
    lint_snapshot = _lint_snapshot(current_global)
    blockers = _claim_blockers(pytest_passed=pytest_passed, global_after=len(current_global))
    claim_matrix = _claim_matrix(pytest_passed)
    _update_claim_boundaries(pytest_passed)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v406_pytest_probe_summary.csv", pytest_summary)
    write_csv(TABLE_DIR / "paper4_v406_notebook_lint_snapshot.csv", lint_snapshot)
    write_csv(TABLE_DIR / "paper4_v406_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v406_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v406_post_sys_path_refactor_pytest_probe",
        "schema_version": "2026-05-17.406",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_sys_path_refactor_version_v406": PRIOR_SYS_PATH_REFACTOR_VERSION,
        "pytest_command_v406": pytest_result["command"],
        "pytest_exit_code_v406": int(pytest_result["exit_code"]),
        "pytest_passed_v406": pytest_passed,
        "pytest_runtime_seconds_v406": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v406": int(pytest_result["collected_items"]),
        "pytest_summary_line_v406": str(pytest_result["summary_line"]),
        "global_notebook_diagnostics_v406": int(len(current_global)),
        "global_notebook_e402_v406": int(counts.get("E402", 0)),
        "global_notebook_i001_v406": int(counts.get("I001", 0)),
        "global_notebook_f401_v406": int(counts.get("F401", 0)),
        "global_notebook_f821_v406": int(counts.get("F821", 0)),
        "global_ruff_clean_v406": False,
        "full_repository_pytest_run_v406": True,
        "full_repository_pytest_passed_v406": pytest_passed,
        "full_quarto_render_run_v406": False,
        "working_champion_claim_allowed_v406": False,
        "paper1_promotion_allowed_v406": False,
        "paper4_working_champion_changed_v406": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v406": next_artifact,
        "claim_boundary": (
            "v406 records post-E402-clearance pytest evidence; non-E402 lint and "
            "final promotion claims remain blocked"
        ),
    }
    if status["global_notebook_e402_v406"] != 0:
        raise RuntimeError("v406 expected notebook E402 to remain clear.")
    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v406": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
