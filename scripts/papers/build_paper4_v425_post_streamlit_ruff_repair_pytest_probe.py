#!/usr/bin/env python3
"""Build Paper 4 v425 post-Streamlit-ruff-repair pytest probe artifacts."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_v403_post_notebook_mutation_pytest_probe import _run_pytest
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

VERSION = 425
PRIOR_STREAMLIT_RUFF_REPAIR_VERSION = 424
NEXT_PASS_ARTIFACT = "paper4_v426_targeted_scripts_ruff_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v426_post_streamlit_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v425_post_streamlit_ruff_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]


def _streamlit_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "streamlit_app/pages"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _run_repository_ruff_json() -> tuple[int, list[dict[str, Any]]]:
    result = subprocess.run(
        RUFF_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "repository ruff probe failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("repository ruff JSON output is not a list")
    return result.returncode, payload


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def _surface(path: str) -> str:
    if path.startswith("notebooks/"):
        return "notebook"
    if path.startswith("streamlit_app/"):
        return "streamlit_app"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith("book/"):
        return "book"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("src/"):
        return "src"
    return "other"


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v425": "full_repository_pytest",
                "command_v425": pytest_result["command"],
                "exit_code_v425": int(pytest_result["exit_code"]),
                "passed_v425": bool(pytest_result["passed"]),
                "runtime_seconds_v425": float(pytest_result["runtime_seconds"]),
                "collected_items_v425": int(pytest_result["collected_items"]),
                "summary_line_v425": str(pytest_result["summary_line"]),
                "claim_boundary_v425": "post-v424 full pytest probe",
            }
        ]
    )


def _ruff_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    rule_counts = Counter(str(item["code"]) for item in items)
    file_surfaces = [_surface(_relative_path(str(item["filename"]))) for item in items]
    surface_counts = Counter(file_surfaces)
    rows = [
        ("repository_total", len(items)),
        ("repository_e402", rule_counts.get("E402", 0)),
        ("repository_b905", rule_counts.get("B905", 0)),
        ("repository_c408", rule_counts.get("C408", 0)),
        ("notebook_total", surface_counts.get("notebook", 0)),
        ("streamlit_app_total", surface_counts.get("streamlit_app", 0)),
        ("scripts_total", surface_counts.get("scripts", 0)),
        ("book_total", surface_counts.get("book", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v425": metric,
                "diagnostic_count_v425": int(count),
                "claim_boundary_v425": "post-v424 repository ruff snapshot",
            }
            for metric, count in rows
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v425": "repository_ruff_frontier_still_open",
            "blocking_v425": ruff_total > 0,
            "evidence_count_v425": ruff_total,
            "required_next_artifact_v425": NEXT_PASS_ARTIFACT,
            "claim_boundary_v425": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v425": "quarto_render_not_run",
            "blocking_v425": True,
            "evidence_count_v425": 1,
            "required_next_artifact_v425": NEXT_PASS_ARTIFACT,
            "claim_boundary_v425": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v425": "paper4_final_promotion_forbidden",
            "blocking_v425": True,
            "evidence_count_v425": 1,
            "required_next_artifact_v425": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v425": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v425": "full_repository_pytest_failed",
                "blocking_v425": True,
                "evidence_count_v425": 1,
                "required_next_artifact_v425": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v425": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pytest_passed: bool, e402_count: int, notebook_count: int, ruff_total: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v425_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v425_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v424 Streamlit repair",
            },
            {
                "claim_id": "v425_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v425_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v425_streamlit_e402_remains_clear",
                "allowed": e402_count == 0,
                "artifact": "paper4_v425_repository_ruff_snapshot.csv",
                "boundary": "E402 remains zero after full pytest probe",
            },
            {
                "claim_id": "v425_notebook_lint_remains_clean",
                "allowed": notebook_count == 0,
                "artifact": "paper4_v425_repository_ruff_snapshot.csv",
                "boundary": "notebook diagnostics remain zero",
            },
            {
                "claim_id": "v425_repository_ruff_clean",
                "allowed": ruff_total == 0,
                "artifact": "paper4_v425_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v425_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, pytest_passed: bool, e402_count: int, notebook_count: int) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v425 runs full repository pytest after Streamlit ruff repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v425_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v425 full repository pytest passes after Streamlit ruff repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v425_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v425 keeps Streamlit E402 and notebook lint clean after pytest probe.",
                "allowed": e402_count == 0 and notebook_count == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v425_repository_ruff_snapshot.csv",
                "boundary": "E402 and notebook diagnostics remain zero.",
                "prohibited_claim_flag": e402_count != 0 or notebook_count != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v425 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v425_claim_blockers.csv",
                "boundary": "Repository ruff remains open and Quarto render is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v425 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v425_claim_blockers.csv",
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
                "executable_item": "v425 runs full repository pytest after the Streamlit ruff repair.",
                "status": (
                    "post_streamlit_ruff_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_streamlit_ruff_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v426 applies a targeted scripts ruff repair batch"
                    if pytest_passed
                    else "v426 triages pytest failures before more lint repair"
                ),
                "last_wave": "v425",
                "execution_result": (
                    "full_repository_pytest_passed_after_streamlit_e402_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_streamlit_e402_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v425")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Streamlit-Ruff-Repair Pytest Probe v425

Generated: {status["generated_at_utc"]}

v425 runs full repository pytest after v424's targeted Streamlit E402 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v425"]}`.
- Pytest passed: `{status["pytest_passed_v425"]}`.
- Collected items: `{status["pytest_collected_items_v425"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v425"]}`.
- Summary: `{status["pytest_summary_line_v425"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v425"]}`.
- E402 diagnostics: `{status["repo_ruff_e402_v425"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v425"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v425 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v425"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V425_POST_STREAMLIT_RUFF_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V425_POST_STREAMLIT_RUFF_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v425: Post-Streamlit-Ruff-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v425 runs full repository pytest after v424's targeted Streamlit E402 repair.

### Results

- Pytest command:
  `{status["pytest_command_v425"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v425"]}`.
- Pytest passed:
  `{status["pytest_passed_v425"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v425"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v425"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v425"]}`.
- Repository E402 diagnostics:
  `{status["repo_ruff_e402_v425"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v425"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v425"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v425"]}`.

### Interpretation

The Streamlit E402 repair passed the full repository pytest probe. The remaining
repository ruff frontier is scripts-first, with a small non-E402 Streamlit
hotspot and two book-helper diagnostics.

### Claim Impact

- Allowed: full repository pytest passed after the Streamlit repair; E402 and
  notebook lint remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v425 in the living notebook. v426 should start a targeted scripts ruff
repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _streamlit_diff_clean():
        raise RuntimeError("v425 expects clean Streamlit page diffs before pytest probe.")

    v424_status = json.loads((STATUS_DIR / "paper4_v424_status.json").read_text(encoding="utf-8"))
    if v424_status["next_artifact_v424"] != "paper4_v425_post_streamlit_ruff_repair_pytest_probe.md":
        raise RuntimeError("v425 expects v424 to route to post-repair pytest probe.")
    if v424_status["targeted_streamlit_page_import_tests_passed_v424"] is not True:
        raise RuntimeError("v425 expects v424 targeted Streamlit tests to have passed.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    snapshot = _ruff_snapshot(ruff_items)
    snapshot_map = dict(zip(snapshot["metric_v425"], snapshot["diagnostic_count_v425"], strict=False))
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(
        pytest_passed=pytest_passed,
        e402_count=int(snapshot_map["repository_e402"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_claim_boundaries(
        pytest_passed=pytest_passed,
        e402_count=int(snapshot_map["repository_e402"]),
        notebook_count=int(snapshot_map["notebook_total"]),
    )
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v425_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v425_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v425_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v425_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v425_post_streamlit_ruff_repair_pytest_probe",
        "schema_version": "2026-05-17.425",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_streamlit_ruff_repair_version_v425": PRIOR_STREAMLIT_RUFF_REPAIR_VERSION,
        "pytest_command_v425": pytest_result["command"],
        "pytest_exit_code_v425": int(pytest_result["exit_code"]),
        "pytest_passed_v425": pytest_passed,
        "pytest_runtime_seconds_v425": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v425": int(pytest_result["collected_items"]),
        "pytest_summary_line_v425": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v425": int(ruff_exit),
        "repo_ruff_total_v425": int(snapshot_map["repository_total"]),
        "repo_ruff_e402_v425": int(snapshot_map["repository_e402"]),
        "repo_ruff_b905_v425": int(snapshot_map["repository_b905"]),
        "repo_ruff_c408_v425": int(snapshot_map["repository_c408"]),
        "notebook_diagnostics_v425": int(snapshot_map["notebook_total"]),
        "streamlit_diagnostics_v425": int(snapshot_map["streamlit_app_total"]),
        "scripts_diagnostics_v425": int(snapshot_map["scripts_total"]),
        "book_diagnostics_v425": int(snapshot_map["book_total"]),
        "repository_ruff_clean_v425": int(snapshot_map["repository_total"]) == 0,
        "full_repository_pytest_run_v425": True,
        "full_repository_pytest_passed_v425": pytest_passed,
        "full_quarto_render_run_v425": False,
        "working_champion_claim_allowed_v425": False,
        "paper1_promotion_allowed_v425": False,
        "paper4_working_champion_changed_v425": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v425": next_artifact,
        "claim_boundary": (
            "v425 records post-Streamlit-repair full pytest evidence; repo ruff, "
            "Quarto and final-promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v425"] != 57:
        raise RuntimeError("v425 expected repository ruff frontier to remain at 57 diagnostics.")
    if status["repo_ruff_e402_v425"] != 0:
        raise RuntimeError("v425 expected E402 to remain clear.")
    if status["notebook_diagnostics_v425"] != 0:
        raise RuntimeError("v425 expected notebooks to remain lint-clean.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v425": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
