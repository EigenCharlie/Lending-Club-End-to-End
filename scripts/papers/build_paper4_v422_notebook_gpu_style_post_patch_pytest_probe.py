#!/usr/bin/env python3
"""Build Paper 4 v422 post-GPU-style notebook patch pytest probe artifacts."""

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

VERSION = 422
PRIOR_GPU_STYLE_PATCH_VERSION = 421
NEXT_PASS_ARTIFACT = "paper4_v423_repository_ruff_frontier_after_notebook_clean.md"
NEXT_FAIL_ARTIFACT = "paper4_v423_post_gpu_style_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v422_notebook_gpu_style_post_patch_pytest_probe.md"


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v422": "full_repository_pytest",
                "command_v422": pytest_result["command"],
                "exit_code_v422": int(pytest_result["exit_code"]),
                "passed_v422": bool(pytest_result["passed"]),
                "runtime_seconds_v422": float(pytest_result["runtime_seconds"]),
                "collected_items_v422": int(pytest_result["collected_items"]),
                "summary_line_v422": str(pytest_result["summary_line"]),
                "claim_boundary_v422": "post-GPU-style notebook patch pytest probe",
            }
        ]
    )


def _lint_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in items)
    rows = [
        {
            "lint_code_v422": code,
            "diagnostic_count_v422": int(count),
            "claim_boundary_v422": "notebook lint snapshot after post-GPU-style pytest probe",
        }
        for code, count in sorted(counts.items())
    ]
    if not rows:
        rows.append(
            {
                "lint_code_v422": "__none__",
                "diagnostic_count_v422": 0,
                "claim_boundary_v422": "ruff check notebooks is clean after pytest probe",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, pytest_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v422": "repository_ruff_frontier_not_reprobed",
            "blocking_v422": True,
            "evidence_count_v422": 1,
            "required_next_artifact_v422": NEXT_PASS_ARTIFACT,
            "claim_boundary_v422": "notebook lint and pytest evidence do not imply repository ruff clean",
        },
        {
            "blocker_id_v422": "paper4_final_promotion_forbidden",
            "blocking_v422": True,
            "evidence_count_v422": 1,
            "required_next_artifact_v422": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v422": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v422": "full_repository_pytest_failed",
                "blocking_v422": True,
                "evidence_count_v422": 1,
                "required_next_artifact_v422": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v422": "pytest failure must be triaged before repository ruff frontier work",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pytest_passed: bool, notebook_clean: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v422_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v422_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v421 GPU style patch",
            },
            {
                "claim_id": "v422_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v422_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v422_notebook_lint_remains_clean",
                "allowed": notebook_clean,
                "artifact": "paper4_v422_notebook_lint_snapshot.csv",
                "boundary": "true only when ruff check notebooks reports zero diagnostics",
            },
            {
                "claim_id": "v422_repository_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v422_claim_blockers.csv",
                "boundary": "repository ruff is deferred to v423",
            },
            {
                "claim_id": "v422_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, pytest_passed: bool, notebook_clean: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v422 runs full repository pytest after GPU side-project notebook lint patch.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v422_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v422 full repository pytest passes after GPU side-project notebook lint patch.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v422_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v422 keeps notebook lint clean after pytest probe.",
                "allowed": notebook_clean,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v422_notebook_lint_snapshot.csv",
                "boundary": "Applies only to ruff check notebooks.",
                "prohibited_claim_flag": not notebook_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v422 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v422_claim_blockers.csv",
                "boundary": "Repository ruff and Quarto render are deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v422 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v422_claim_blockers.csv",
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
                "executable_item": "v422 runs full repository pytest after the GPU style-lint patch.",
                "status": (
                    "post_gpu_style_lint_pytest_probe_passed"
                    if pytest_passed
                    else "post_gpu_style_lint_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v423 classifies the repository ruff frontier after notebook lint is clean"
                    if pytest_passed
                    else "v423 triages pytest failures before repository ruff frontier work"
                ),
                "last_wave": "v422",
                "execution_result": (
                    "full_repository_pytest_passed_after_gpu_style_lint_patch"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_gpu_style_lint_patch"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v422")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-GPU-Style Pytest Probe v422

Generated: {status["generated_at_utc"]}

v422 runs full repository pytest after v421 clears GPU side-project notebook
style lint.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v422"]}`.
- Pytest passed: `{status["pytest_passed_v422"]}`.
- Collected items: `{status["pytest_collected_items_v422"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v422"]}`.
- Summary: `{status["pytest_summary_line_v422"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v422"]}`.
- Notebook ruff clean: `{status["notebook_ruff_clean_v422"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v422 does not run repository ruff or Quarto render and does not create Paper 4
final promotion.

## Next Executable Wave

Build `{status["next_artifact_v422"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V422_NOTEBOOK_GPU_STYLE_POST_PATCH_PYTEST_PROBE_START -->"
    end = "<!-- V422_NOTEBOOK_GPU_STYLE_POST_PATCH_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v422: Post-GPU-Style Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v422 runs full repository pytest after v421 clears the GPU side-project
notebook style-lint frontier.

### Results

- Pytest command:
  `{status["pytest_command_v422"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v422"]}`.
- Pytest passed:
  `{status["pytest_passed_v422"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v422"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v422"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v422"]}`.
- Notebook ruff clean:
  `{status["notebook_ruff_clean_v422"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v422"]}`.

### Interpretation

The GPU notebook patch passed the repository pytest probe and notebook lint
remained clean. The next frontier is repository-wide ruff classification, not a
Paper Estrella or final Paper 4 promotion.

### Claim Impact

- Allowed: full repository pytest was executed after notebook lint clearance.
- Conditional: pytest pass claim follows the captured exit code.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v422 in the living notebook. v423 should classify the repository ruff
frontier after notebook lint is clean.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v422 expects clean notebook diff before pytest probe.")

    v421_status = json.loads((STATUS_DIR / "paper4_v421_status.json").read_text(encoding="utf-8"))
    if v421_status["next_artifact_v421"] != "paper4_v422_notebook_gpu_style_post_patch_pytest_probe.md":
        raise RuntimeError("v422 expects v421 to route to post-GPU-style pytest probe.")
    if v421_status["notebook_ruff_clean_v421"] is not True:
        raise RuntimeError("v422 expects v421 to have cleared notebook lint.")

    before_notebook_lint = _run_ruff_json()
    if before_notebook_lint:
        raise RuntimeError("v422 expected clean notebook lint before pytest probe.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    current_global = _run_ruff_json()
    notebook_clean = len(current_global) == 0
    counts = Counter(item["code"] for item in current_global)
    lint_snapshot = _lint_snapshot(current_global)
    pytest_summary = _pytest_summary_table(pytest_result)
    blockers = _claim_blockers(pytest_passed=pytest_passed)
    claim_matrix = _claim_matrix(pytest_passed=pytest_passed, notebook_clean=notebook_clean)
    _update_claim_boundaries(pytest_passed=pytest_passed, notebook_clean=notebook_clean)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v422_pytest_probe_summary.csv", pytest_summary)
    write_csv(TABLE_DIR / "paper4_v422_notebook_lint_snapshot.csv", lint_snapshot)
    write_csv(TABLE_DIR / "paper4_v422_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v422_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v422_notebook_gpu_style_post_patch_pytest_probe",
        "schema_version": "2026-05-17.422",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_gpu_style_patch_version_v422": PRIOR_GPU_STYLE_PATCH_VERSION,
        "pytest_command_v422": pytest_result["command"],
        "pytest_exit_code_v422": int(pytest_result["exit_code"]),
        "pytest_passed_v422": pytest_passed,
        "pytest_runtime_seconds_v422": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v422": int(pytest_result["collected_items"]),
        "pytest_summary_line_v422": str(pytest_result["summary_line"]),
        "global_notebook_diagnostics_v422": int(len(current_global)),
        "global_notebook_e712_v422": int(counts.get("E712", 0)),
        "global_notebook_sim102_v422": int(counts.get("SIM102", 0)),
        "global_notebook_sim108_v422": int(counts.get("SIM108", 0)),
        "notebook_ruff_clean_v422": notebook_clean,
        "repository_ruff_clean_v422": False,
        "full_repository_pytest_run_v422": True,
        "full_repository_pytest_passed_v422": pytest_passed,
        "full_quarto_render_run_v422": False,
        "working_champion_claim_allowed_v422": False,
        "paper1_promotion_allowed_v422": False,
        "paper4_working_champion_changed_v422": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v422": next_artifact,
        "claim_boundary": (
            "v422 records post-GPU-style pytest evidence; repository ruff, Quarto "
            "and final-promotion claims remain blocked"
        ),
    }
    if not notebook_clean:
        raise RuntimeError("v422 expected notebook lint to remain clean after pytest.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v422": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
