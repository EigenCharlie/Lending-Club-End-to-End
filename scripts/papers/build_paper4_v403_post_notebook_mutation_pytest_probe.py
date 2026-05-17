#!/usr/bin/env python3
"""Build Paper 4 v403 post-notebook-mutation pytest probe artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

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

VERSION = 403
PRIOR_NOTEBOOK_MUTATION_VERSION = 402
NEXT_PASS_ARTIFACT = "paper4_v404_notebook_sys_path_project_import_refactor_plan.md"
NEXT_FAIL_ARTIFACT = "paper4_v404_post_mutation_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v403_post_notebook_mutation_pytest_probe.md"
PYTEST_COMMAND = ["uv", "run", "pytest", "-q", "--tb=short"]


def _run_ruff_json(codes: list[str] | None = None) -> list[dict[str, Any]]:
    command = ["uv", "run", "ruff", "check", "notebooks", "--output-format", "json"]
    if codes is not None:
        command[5:5] = ["--select", ",".join(codes)]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff notebook probe failed")
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("ruff notebook JSON output is not a list")
    return payload


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _run_pytest() -> dict[str, Any]:
    started = datetime.now(UTC)
    result = subprocess.run(
        PYTEST_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    runtime = round((datetime.now(UTC) - started).total_seconds(), 3)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)
    summary_line = ""
    for line in reversed(combined.splitlines()):
        if " in " in line and "==" in line:
            summary_line = line.strip()
            break
    collected_match = re.search(r"collected\s+(\d+)\s+items", combined)
    return {
        "command": " ".join(PYTEST_COMMAND),
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "runtime_seconds": runtime,
        "collected_items": int(collected_match.group(1)) if collected_match else 0,
        "summary_line": summary_line,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-80:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-80:]),
    }


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v403": "full_repository_pytest",
                "command_v403": pytest_result["command"],
                "exit_code_v403": int(pytest_result["exit_code"]),
                "passed_v403": bool(pytest_result["passed"]),
                "runtime_seconds_v403": float(pytest_result["runtime_seconds"]),
                "collected_items_v403": int(pytest_result["collected_items"]),
                "summary_line_v403": str(pytest_result["summary_line"]),
                "claim_boundary_v403": "post-notebook-mutation pytest probe",
            }
        ]
    )


def _lint_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in items)
    rows = []
    for code, count in sorted(counts.items()):
        rows.append(
            {
                "lint_code_v403": code,
                "diagnostic_count_v403": int(count),
                "claim_boundary_v403": "notebook lint remains visible after pytest probe",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, pytest_passed: bool, e402_after: int, global_after: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v403": "sys_path_project_import_e402_remaining",
            "blocking_v403": True,
            "evidence_count_v403": e402_after,
            "required_next_artifact_v403": NEXT_PASS_ARTIFACT,
            "claim_boundary_v403": "sys.path/project-import cells remain",
        },
        {
            "blocker_id_v403": "global_notebook_lint_not_clean",
            "blocking_v403": True,
            "evidence_count_v403": global_after,
            "required_next_artifact_v403": NEXT_PASS_ARTIFACT,
            "claim_boundary_v403": "E402 and semantic/manual notebook lint remain",
        },
        {
            "blocker_id_v403": "paper4_final_promotion_forbidden",
            "blocking_v403": True,
            "evidence_count_v403": 1,
            "required_next_artifact_v403": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v403": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v403": "full_repository_pytest_failed",
                "blocking_v403": True,
                "evidence_count_v403": 1,
                "required_next_artifact_v403": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v403": "pytest failure must be triaged before deeper notebook mutation",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(pytest_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v403_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v403_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v402 notebook mutation",
            },
            {
                "claim_id": "v403_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v403_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v403_notebook_lint_snapshot_created",
                "allowed": True,
                "artifact": "paper4_v403_notebook_lint_snapshot.csv",
                "boundary": "lint visibility only",
            },
            {
                "claim_id": "v403_sys_path_project_import_e402_repaired",
                "allowed": False,
                "artifact": "paper4_v403_claim_blockers.csv",
                "boundary": "sys.path cells deferred",
            },
            {
                "claim_id": "v403_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v403_claim_blockers.csv",
                "boundary": "62 notebook diagnostics remain",
            },
            {
                "claim_id": "v403_working_champion_or_final_promotion",
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
                "claim": "v403 runs full repository pytest after v402 notebook mutation.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v403_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v403 full repository pytest passes after v402 notebook mutation.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v403_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v403 clears E402 or global notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v403_claim_blockers.csv",
                "boundary": "42 E402 and 62 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v403 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v403_claim_blockers.csv",
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
                "executable_item": "v403 runs the post-v402 full repository pytest probe.",
                "status": (
                    "post_notebook_mutation_pytest_probe_passed"
                    if pytest_passed
                    else "post_notebook_mutation_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v404 plans sys.path/project-import E402 cells"
                    if pytest_passed
                    else "v404 triages pytest failures before deeper mutation"
                ),
                "last_wave": "v403",
                "execution_result": (
                    "full_repository_pytest_passed_after_v402"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_v402"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v403")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Notebook-Mutation Pytest Probe v403

Generated: {status["generated_at_utc"]}

v403 runs the full repository pytest command after the v400/v402 notebook
mutations.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v403"]}`.
- Pytest passed: `{status["pytest_passed_v403"]}`.
- Collected items: `{status["pytest_collected_items_v403"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v403"]}`.
- Summary: `{status["pytest_summary_line_v403"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v403"]}`.
- Notebook E402 diagnostics: `{status["global_notebook_e402_v403"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v403 does not repair sys.path/project-import E402 cells, does not clear notebook
or repository ruff, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v403"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V403_POST_NOTEBOOK_MUTATION_PYTEST_PROBE_START -->"
    end = "<!-- V403_POST_NOTEBOOK_MUTATION_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v403: Post-Notebook-Mutation Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v403 runs full repository pytest after the v400/v402 notebook mutations and
records whether validation can proceed toward sys.path E402 planning.

### Results

- Pytest command:
  `{status["pytest_command_v403"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v403"]}`.
- Pytest passed:
  `{status["pytest_passed_v403"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v403"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v403"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v403"]}`.
- Notebook E402 diagnostics:
  `{status["global_notebook_e402_v403"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v403"]}`.

### Interpretation

Pytest status now gates whether the next executable wave can continue directly
to sys.path/project-import E402 planning or must first triage test failures.

### Claim Impact

- Allowed: full repository pytest was executed after notebook mutation.
- Conditional: pytest pass claim follows the captured exit code.
- Still prohibited: notebook lint clean, sys.path E402 repaired, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v403 in the living notebook. Route the next wave according to the pytest
result.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v403 expects clean notebook diff before pytest probe.")

    v402_status = json.loads((STATUS_DIR / "paper4_v402_status.json").read_text(encoding="utf-8"))
    if v402_status["next_artifact_v402"] != "paper4_v403_post_notebook_mutation_pytest_probe.md":
        raise RuntimeError("v403 expects v402 to route to post-notebook-mutation pytest probe.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    current_global = _run_ruff_json()
    counts = Counter(item["code"] for item in current_global)
    lint_snapshot = _lint_snapshot(current_global)
    pytest_summary = _pytest_summary_table(pytest_result)
    blockers = _claim_blockers(
        pytest_passed=pytest_passed,
        e402_after=counts.get("E402", 0),
        global_after=len(current_global),
    )
    claim_matrix = _claim_matrix(pytest_passed)
    _update_claim_boundaries(pytest_passed)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v403_pytest_probe_summary.csv", pytest_summary)
    write_csv(TABLE_DIR / "paper4_v403_notebook_lint_snapshot.csv", lint_snapshot)
    write_csv(TABLE_DIR / "paper4_v403_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v403_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v403_post_notebook_mutation_pytest_probe",
        "schema_version": "2026-05-17.403",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_notebook_mutation_version_v403": PRIOR_NOTEBOOK_MUTATION_VERSION,
        "pytest_command_v403": pytest_result["command"],
        "pytest_exit_code_v403": int(pytest_result["exit_code"]),
        "pytest_passed_v403": pytest_passed,
        "pytest_runtime_seconds_v403": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v403": int(pytest_result["collected_items"]),
        "pytest_summary_line_v403": str(pytest_result["summary_line"]),
        "global_notebook_diagnostics_v403": int(len(current_global)),
        "global_notebook_e402_v403": int(counts.get("E402", 0)),
        "global_notebook_i001_v403": int(counts.get("I001", 0)),
        "global_notebook_f821_v403": int(counts.get("F821", 0)),
        "global_ruff_clean_v403": False,
        "full_repository_pytest_run_v403": True,
        "full_repository_pytest_passed_v403": pytest_passed,
        "full_quarto_render_run_v403": False,
        "working_champion_claim_allowed_v403": False,
        "paper1_promotion_allowed_v403": False,
        "paper4_working_champion_changed_v403": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v403": next_artifact,
        "claim_boundary": (
            "v403 records post-notebook-mutation pytest evidence; lint and final "
            "promotion claims remain blocked"
        ),
    }
    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v403": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
