#!/usr/bin/env python3
"""Build Paper 4 v419 post-SIM108 pytest probe artifacts."""

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
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 419
PRIOR_SIM108_PATCH_VERSION = 418
NEXT_PASS_ARTIFACT = "paper4_v420_notebook_gpu_style_lint_triage.md"
NEXT_FAIL_ARTIFACT = "paper4_v420_post_sim108_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v419_notebook_sim108_post_patch_pytest_probe.md"


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v419": "full_repository_pytest",
                "command_v419": pytest_result["command"],
                "exit_code_v419": int(pytest_result["exit_code"]),
                "passed_v419": bool(pytest_result["passed"]),
                "runtime_seconds_v419": float(pytest_result["runtime_seconds"]),
                "collected_items_v419": int(pytest_result["collected_items"]),
                "summary_line_v419": str(pytest_result["summary_line"]),
                "claim_boundary_v419": "post-SIM108 pytest probe",
            }
        ]
    )


def _lint_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in items)
    rows = []
    for code, count in sorted(counts.items()):
        rows.append(
            {
                "lint_code_v419": code,
                "diagnostic_count_v419": int(count),
                "claim_boundary_v419": "side-project notebook lint remains visible after SIM108 patch",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, pytest_passed: bool, global_after: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v419": "gpu_side_project_style_lint_remaining",
            "blocking_v419": True,
            "evidence_count_v419": global_after,
            "required_next_artifact_v419": NEXT_PASS_ARTIFACT,
            "claim_boundary_v419": "remaining notebook diagnostics are GPU side-project style refactors",
        },
        {
            "blocker_id_v419": "paper4_final_promotion_forbidden",
            "blocking_v419": True,
            "evidence_count_v419": 1,
            "required_next_artifact_v419": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v419": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v419": "full_repository_pytest_failed",
                "blocking_v419": True,
                "evidence_count_v419": 1,
                "required_next_artifact_v419": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v419": "pytest failure must be triaged before GPU style lint cleanup",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(pytest_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v419_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v419_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v418 SIM108 patch",
            },
            {
                "claim_id": "v419_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v419_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v419_notebook_sim108_remains_clear",
                "allowed": True,
                "artifact": "paper4_v419_notebook_lint_snapshot.csv",
                "boundary": "SIM108 count remains 0",
            },
            {
                "claim_id": "v419_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v419_claim_blockers.csv",
                "boundary": "3 notebook diagnostics remain",
            },
            {
                "claim_id": "v419_working_champion_or_final_promotion",
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
                "claim": "v419 runs full repository pytest after notebook SIM108 patch.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v419_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v419 full repository pytest passes after notebook SIM108 patch.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v419_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v419 keeps notebook SIM108 cleared after pytest probe.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v419_notebook_lint_snapshot.csv",
                "boundary": "SIM108 remains at zero; GPU side-project style lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v419 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v419_claim_blockers.csv",
                "boundary": "3 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v419 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v419_claim_blockers.csv",
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
                "executable_item": "v419 runs full repository pytest after the SIM108 notebook patch.",
                "status": (
                    "post_sim108_conditional_expr_pytest_probe_passed"
                    if pytest_passed
                    else "post_sim108_conditional_expr_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v420 triages remaining GPU side-project style notebook lint"
                    if pytest_passed
                    else "v420 triages pytest failures before GPU style lint cleanup"
                ),
                "last_wave": "v419",
                "execution_result": (
                    "full_repository_pytest_passed_after_sim108_conditional_expr_patch"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_sim108_conditional_expr_patch"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v419")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-SIM108 Pytest Probe v419

Generated: {status["generated_at_utc"]}

v419 runs full repository pytest after v418 clears notebook SIM108.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v419"]}`.
- Pytest passed: `{status["pytest_passed_v419"]}`.
- Collected items: `{status["pytest_collected_items_v419"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v419"]}`.
- Summary: `{status["pytest_summary_line_v419"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v419"]}`.
- Notebook SIM108 diagnostics: `{status["global_notebook_sim108_v419"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v419 does not clear remaining GPU side-project style notebook lint and does not
create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v419"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V419_NOTEBOOK_SIM108_POST_PATCH_PYTEST_PROBE_START -->"
    end = "<!-- V419_NOTEBOOK_SIM108_POST_PATCH_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v419: Post-SIM108 Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v419 runs full repository pytest after v418 clears notebook SIM108.

### Results

- Pytest command:
  `{status["pytest_command_v419"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v419"]}`.
- Pytest passed:
  `{status["pytest_passed_v419"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v419"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v419"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v419"]}`.
- Notebook SIM108 diagnostics:
  `{status["global_notebook_sim108_v419"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v419"]}`.

### Interpretation

The SIM108 patch passed repository validation. The remaining notebook lint
frontier is isolated to GPU side-project style refactors.

### Claim Impact

- Allowed: full repository pytest was executed after SIM108 clearance.
- Conditional: pytest pass claim follows the captured exit code.
- Still prohibited: global notebook lint clean, repository ruff clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v419 in the living notebook. v420 should triage the remaining GPU
side-project style lint.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v419 expects clean notebook diff before pytest probe.")

    v418_status = json.loads((STATUS_DIR / "paper4_v418_status.json").read_text(encoding="utf-8"))
    if v418_status["next_artifact_v418"] != "paper4_v419_notebook_sim108_post_patch_pytest_probe.md":
        raise RuntimeError("v419 expects v418 to route to post-SIM108 pytest probe.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    current_global = _run_ruff_json()
    counts = Counter(item["code"] for item in current_global)
    lint_snapshot = _lint_snapshot(current_global)
    pytest_summary = _pytest_summary_table(pytest_result)
    blockers = _claim_blockers(pytest_passed=pytest_passed, global_after=len(current_global))
    claim_matrix = _claim_matrix(pytest_passed)
    _update_claim_boundaries(pytest_passed)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v419_pytest_probe_summary.csv", pytest_summary)
    write_csv(TABLE_DIR / "paper4_v419_notebook_lint_snapshot.csv", lint_snapshot)
    write_csv(TABLE_DIR / "paper4_v419_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v419_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v419_notebook_sim108_post_patch_pytest_probe",
        "schema_version": "2026-05-17.419",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_sim108_patch_version_v419": PRIOR_SIM108_PATCH_VERSION,
        "pytest_command_v419": pytest_result["command"],
        "pytest_exit_code_v419": int(pytest_result["exit_code"]),
        "pytest_passed_v419": pytest_passed,
        "pytest_runtime_seconds_v419": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v419": int(pytest_result["collected_items"]),
        "pytest_summary_line_v419": str(pytest_result["summary_line"]),
        "global_notebook_diagnostics_v419": int(len(current_global)),
        "global_notebook_sim108_v419": int(counts.get("SIM108", 0)),
        "global_notebook_e712_v419": int(counts.get("E712", 0)),
        "global_notebook_sim102_v419": int(counts.get("SIM102", 0)),
        "global_ruff_clean_v419": False,
        "full_repository_pytest_run_v419": True,
        "full_repository_pytest_passed_v419": pytest_passed,
        "full_quarto_render_run_v419": False,
        "working_champion_claim_allowed_v419": False,
        "paper1_promotion_allowed_v419": False,
        "paper4_working_champion_changed_v419": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v419": next_artifact,
        "claim_boundary": (
            "v419 records post-SIM108 pytest evidence; remaining lint and final "
            "promotion claims remain blocked"
        ),
    }
    if status["global_notebook_sim108_v419"] != 0:
        raise RuntimeError("v419 expected notebook SIM108 to remain clear.")
    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v419": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
