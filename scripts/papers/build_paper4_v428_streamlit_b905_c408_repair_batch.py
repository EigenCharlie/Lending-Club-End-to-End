#!/usr/bin/env python3
"""Build Paper 4 v428 targeted Streamlit B905/C408 repair batch artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
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

VERSION = 428
PRIOR_POST_SCRIPTS_REPAIR_PYTEST_VERSION = 427
TARGET_FILE = "streamlit_app/pages/model_interpretability.py"
TARGET_RULES = ["B905", "C408"]
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_TARGET_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    TARGET_FILE,
    "--select",
    ",".join(TARGET_RULES),
    "--output-format",
    "json",
]
RUFF_FIX_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    TARGET_FILE,
    "--select",
    ",".join(TARGET_RULES),
    "--fix",
    "--unsafe-fixes",
]
TARGETED_TEST_COMMAND = ["uv", "run", "pytest", "-q", "tests/test_streamlit/test_page_imports.py"]
NEXT_ARTIFACT = "paper4_v429_post_streamlit_b905_c408_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v428_streamlit_b905_c408_repair_batch.md"


def _run_json_command(command: list[str]) -> tuple[int, list[dict[str, Any]]]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or f"{' '.join(command)} failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("ruff JSON output is not a list")
    return result.returncode, payload


def _run_ruff_fix() -> dict[str, Any]:
    result = subprocess.run(
        RUFF_FIX_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "Streamlit B905/C408 ruff fix failed")
    return {
        "command": " ".join(RUFF_FIX_COMMAND),
        "exit_code": int(result.returncode),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


def _run_targeted_tests() -> dict[str, Any]:
    started = datetime.now(UTC)
    result = subprocess.run(
        TARGETED_TEST_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    runtime = round((datetime.now(UTC) - started).total_seconds(), 3)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)
    summary_line = ""
    for line in reversed(combined.splitlines()):
        if " in " in line and ("==" in line or " passed" in line):
            summary_line = line.strip()
            break
    collected_match = re.search(r"collected\s+(\d+)\s+items", combined)
    return {
        "command": " ".join(TARGETED_TEST_COMMAND),
        "exit_code": int(result.returncode),
        "passed": result.returncode == 0,
        "runtime_seconds": runtime,
        "collected_items": int(collected_match.group(1)) if collected_match else 0,
        "summary_line": summary_line,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-40:]),
    }


def _target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", TARGET_FILE],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _changed_target_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", TARGET_FILE],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


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
    return "other"


def _snapshot_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    rule_counts = Counter(str(item["code"]) for item in items)
    surface_counts = Counter(_surface(_relative_path(str(item["filename"]))) for item in items)
    streamlit_rules = Counter(
        str(item["code"])
        for item in items
        if _surface(_relative_path(str(item["filename"]))) == "streamlit_app"
    )
    return {
        "repository_total": int(len(items)),
        "repository_b905": int(rule_counts.get("B905", 0)),
        "repository_c408": int(rule_counts.get("C408", 0)),
        "repository_e402": int(rule_counts.get("E402", 0)),
        "streamlit_app_total": int(surface_counts.get("streamlit_app", 0)),
        "streamlit_app_b905": int(streamlit_rules.get("B905", 0)),
        "streamlit_app_c408": int(streamlit_rules.get("C408", 0)),
        "scripts_total": int(surface_counts.get("scripts", 0)),
        "book_total": int(surface_counts.get("book", 0)),
        "notebook_total": int(surface_counts.get("notebook", 0)),
    }


def _actions(before_target_items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in before_target_items:
        fix = item.get("fix") or {}
        edits = fix.get("edits") or []
        rows.append(
            {
                "action_id_v428": f"streamlit_model_interpretability_{len(rows) + 1:02d}",
                "file_path_v428": _relative_path(str(item["filename"])),
                "row_v428": int((item.get("location") or {}).get("row") or 0),
                "rule_code_v428": str(item["code"]),
                "message_v428": str(item["message"]),
                "fix_message_v428": str(fix.get("message") or ""),
                "fix_edit_count_v428": int(len(edits)),
                "mutation_applied_v428": True,
                "claim_boundary_v428": "targeted Streamlit model_interpretability B905/C408 repair",
            }
        )
    return pd.DataFrame(rows)


def _delta_table(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before = _snapshot_counts(before_items)
    after = _snapshot_counts(after_items)
    metrics = [
        "repository_total",
        "repository_b905",
        "repository_c408",
        "repository_e402",
        "streamlit_app_total",
        "streamlit_app_b905",
        "streamlit_app_c408",
        "scripts_total",
        "book_total",
        "notebook_total",
    ]
    return pd.DataFrame(
        [
            {
                "metric_v428": metric,
                "before_v428": int(before[metric]),
                "after_v428": int(after[metric]),
                "delta_v428": int(after[metric] - before[metric]),
                "claim_boundary_v428": "ruff-count delta only; repository ruff remains open unless after=0",
            }
            for metric in metrics
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v428": metric,
                "diagnostic_count_v428": int(count),
                "claim_boundary_v428": "post-v428 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


def _test_summary_table(test_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_id_v428": "streamlit_page_imports",
                "command_v428": test_result["command"],
                "exit_code_v428": int(test_result["exit_code"]),
                "passed_v428": bool(test_result["passed"]),
                "runtime_seconds_v428": float(test_result["runtime_seconds"]),
                "collected_items_v428": int(test_result["collected_items"]),
                "summary_line_v428": str(test_result["summary_line"]),
                "claim_boundary_v428": "targeted import smoke test only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, targeted_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v428": "repository_ruff_frontier_still_open",
            "blocking_v428": after_total > 0,
            "evidence_count_v428": after_total,
            "required_next_artifact_v428": NEXT_ARTIFACT,
            "claim_boundary_v428": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v428": "full_repository_pytest_deferred_after_streamlit_repair",
            "blocking_v428": True,
            "evidence_count_v428": 1,
            "required_next_artifact_v428": NEXT_ARTIFACT,
            "claim_boundary_v428": "targeted import tests do not replace full pytest",
        },
        {
            "blocker_id_v428": "paper4_final_promotion_forbidden",
            "blocking_v428": True,
            "evidence_count_v428": 1,
            "required_next_artifact_v428": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v428": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not targeted_passed:
        rows.insert(
            0,
            {
                "blocker_id_v428": "targeted_streamlit_page_import_tests_failed",
                "blocking_v428": True,
                "evidence_count_v428": 1,
                "required_next_artifact_v428": "paper4_v429_streamlit_repair_failure_triage.md",
                "claim_boundary_v428": "targeted test failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(
    *,
    after_total: int,
    streamlit_after: int,
    notebook_after: int,
    targeted_passed: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v428_streamlit_b905_c408_repair_applied",
                "allowed": True,
                "artifact": "paper4_v428_streamlit_b905_c408_actions.csv",
                "boundary": "single Streamlit hotspot repair batch",
            },
            {
                "claim_id": "v428_streamlit_b905_c408_cleared",
                "allowed": streamlit_after == 0,
                "artifact": "paper4_v428_repository_ruff_delta.csv",
                "boundary": "true only for Streamlit B905/C408 target diagnostics",
            },
            {
                "claim_id": "v428_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v428_repository_ruff_delta.csv",
                "boundary": "repository ruff diagnostic count decreases",
            },
            {
                "claim_id": "v428_notebook_lint_remains_clean",
                "allowed": notebook_after == 0,
                "artifact": "paper4_v428_repository_ruff_after_snapshot.csv",
                "boundary": "notebook diagnostics remain zero",
            },
            {
                "claim_id": "v428_targeted_streamlit_page_import_tests_passed",
                "allowed": targeted_passed,
                "artifact": "paper4_v428_streamlit_page_import_test_summary.csv",
                "boundary": "targeted smoke only",
            },
            {
                "claim_id": "v428_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v428_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v428_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v428_claim_blockers.csv",
                "boundary": "full pytest deferred to v429",
            },
            {
                "claim_id": "v428_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    after_total: int,
    streamlit_after: int,
    notebook_after: int,
    targeted_passed: bool,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v428 clears the Streamlit B905/C408 model-interpretability hotspot.",
                "allowed": streamlit_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_repository_ruff_delta.csv",
                "boundary": "Streamlit target only; scripts/book diagnostics remain.",
                "prohibited_claim_flag": streamlit_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v428 reduces repository ruff diagnostics after Streamlit B905/C408 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff remains open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v428 keeps notebook lint clean after Streamlit B905/C408 repair.",
                "allowed": notebook_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_repository_ruff_after_snapshot.csv",
                "boundary": "Notebook diagnostics remain zero.",
                "prohibited_claim_flag": notebook_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v428 passes targeted Streamlit page-import tests after B905/C408 repair.",
                "allowed": targeted_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_streamlit_page_import_test_summary.csv",
                "boundary": "Targeted smoke only; full pytest deferred.",
                "prohibited_claim_flag": not targeted_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v428 proves repository ruff clean or full pytest clean after Streamlit repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_claim_blockers.csv",
                "boundary": f"{after_total} repository ruff diagnostics remain and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v428 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v428_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(after_total: int) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v428 applies a targeted Streamlit B905/C408 repair batch.",
                "status": "targeted_streamlit_b905_c408_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v429 full repository pytest passes after Streamlit B905/C408 repair",
                "last_wave": "v428",
                "execution_result": f"repo_ruff_reduced_46_to_{after_total}_streamlit_hotspot_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v428")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], fix_result: dict[str, Any], test_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Streamlit B905/C408 Repair Batch v428

Generated: {status["generated_at_utc"]}

v428 applies ruff's targeted B905/C408 fixes to
`streamlit_app/pages/model_interpretability.py`.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v428"]}` ->
  `{status["repo_ruff_total_after_v428"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_before_v428"]}` ->
  `{status["streamlit_diagnostics_after_v428"]}`.
- Repository B905/C408 after: `{status["repo_ruff_b905_after_v428"]}` /
  `{status["repo_ruff_c408_after_v428"]}`.
- Changed Streamlit files: `{status["changed_streamlit_files_v428"]}`.
- Targeted page-import tests passed: `{status["targeted_streamlit_page_import_tests_passed_v428"]}`.
- Targeted test summary: `{test_result["summary_line"]}`.
- Ruff fix command: `{fix_result["command"]}`.

## Required Caveat

v428 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v428"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V428_STREAMLIT_B905_C408_REPAIR_BATCH_START -->"
    end = "<!-- V428_STREAMLIT_B905_C408_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v428: Streamlit B905/C408 Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v428 applies ruff's targeted B905/C408 fixes to
`streamlit_app/pages/model_interpretability.py`.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v428"]}` ->
  `{status["repo_ruff_total_after_v428"]}`.
- Streamlit diagnostics before/after:
  `{status["streamlit_diagnostics_before_v428"]}` ->
  `{status["streamlit_diagnostics_after_v428"]}`.
- Repository B905/C408 after:
  `{status["repo_ruff_b905_after_v428"]}` /
  `{status["repo_ruff_c408_after_v428"]}`.
- Notebook diagnostics after:
  `{status["notebook_diagnostics_after_v428"]}`.
- Changed Streamlit files:
  `{status["changed_streamlit_files_v428"]}`.
- Targeted Streamlit page-import tests passed:
  `{status["targeted_streamlit_page_import_tests_passed_v428"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v428"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v428"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v428"]}`.

### Interpretation

The Streamlit non-E402 hotspot is cleared. The remaining repository ruff
frontier is now scripts/book only, so v429 should run full pytest before
touching the next scripts frontier.

### Claim Impact

- Allowed: targeted Streamlit B905/C408 repair applied, repository ruff count
  reduced, notebook lint remained clean, and targeted page-import tests passed.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v428 in the living notebook. v429 should run the post-Streamlit-repair full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _target_diff_clean():
        raise RuntimeError("v428 expects clean target Streamlit file diffs before mutation.")

    v427_status = json.loads((STATUS_DIR / "paper4_v427_status.json").read_text(encoding="utf-8"))
    if v427_status["next_artifact_v427"] != "paper4_v428_streamlit_b905_c408_repair_batch.md":
        raise RuntimeError("v428 expects v427 to route to Streamlit B905/C408 repair.")
    if v427_status["full_repository_pytest_passed_v427"] is not True:
        raise RuntimeError("v428 expects v427 full pytest to pass.")
    if int(v427_status["streamlit_diagnostics_v427"]) != 8:
        raise RuntimeError("v428 expects eight Streamlit diagnostics before repair.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_target_items = _run_json_command(RUFF_TARGET_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 46:
        raise RuntimeError("v428 expected repository ruff to fail with 46 diagnostics before repair.")
    if len(before_target_items) != 8:
        raise RuntimeError("v428 expected eight target Streamlit B905/C408 diagnostics.")

    before_counts = _snapshot_counts(before_repo_items)
    actions = _actions(before_target_items)
    fix_result = _run_ruff_fix()
    changed_files = _changed_target_files()
    _, after_target_items = _run_json_command(RUFF_TARGET_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    test_result = _run_targeted_tests()
    targeted_passed = bool(test_result["passed"])
    delta = _delta_table(before_repo_items, after_repo_items)
    after_snapshot = _after_snapshot(after_repo_items)
    after_total = int(len(after_repo_items))
    streamlit_after = int(after_counts["streamlit_app_total"])
    notebook_after = int(after_counts["notebook_total"])
    blockers = _claim_blockers(after_total=after_total, targeted_passed=targeted_passed)
    claim_matrix = _claim_matrix(
        after_total=after_total,
        streamlit_after=streamlit_after,
        notebook_after=notebook_after,
        targeted_passed=targeted_passed,
    )
    _update_claim_boundaries(
        after_total=after_total,
        streamlit_after=streamlit_after,
        notebook_after=notebook_after,
        targeted_passed=targeted_passed,
    )
    _update_backlog(after_total)

    write_csv(TABLE_DIR / "paper4_v428_streamlit_b905_c408_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v428_repository_ruff_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v428_repository_ruff_after_snapshot.csv", after_snapshot)
    write_csv(TABLE_DIR / "paper4_v428_streamlit_page_import_test_summary.csv", _test_summary_table(test_result))
    write_csv(TABLE_DIR / "paper4_v428_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v428_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v428_streamlit_b905_c408_repair_batch",
        "schema_version": "2026-05-17.428",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_scripts_repair_pytest_version_v428": PRIOR_POST_SCRIPTS_REPAIR_PYTEST_VERSION,
        "actions_v428": int(len(actions)),
        "target_file_v428": TARGET_FILE,
        "target_rules_v428": ",".join(TARGET_RULES),
        "repo_ruff_exit_code_before_v428": int(before_repo_exit),
        "repo_ruff_exit_code_after_v428": int(after_repo_exit),
        "repo_ruff_total_before_v428": before_counts["repository_total"],
        "repo_ruff_total_after_v428": after_counts["repository_total"],
        "repo_ruff_total_reduced_v428": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_b905_before_v428": before_counts["repository_b905"],
        "repo_ruff_b905_after_v428": after_counts["repository_b905"],
        "repo_ruff_c408_before_v428": before_counts["repository_c408"],
        "repo_ruff_c408_after_v428": after_counts["repository_c408"],
        "repo_ruff_e402_after_v428": after_counts["repository_e402"],
        "streamlit_diagnostics_before_v428": before_counts["streamlit_app_total"],
        "streamlit_diagnostics_after_v428": streamlit_after,
        "scripts_diagnostics_after_v428": after_counts["scripts_total"],
        "book_diagnostics_after_v428": after_counts["book_total"],
        "notebook_diagnostics_after_v428": notebook_after,
        "changed_streamlit_files_v428": int(len(changed_files)),
        "changed_streamlit_file_list_v428": changed_files,
        "targeted_streamlit_page_import_test_command_v428": test_result["command"],
        "targeted_streamlit_page_import_tests_exit_code_v428": int(test_result["exit_code"]),
        "targeted_streamlit_page_import_tests_passed_v428": targeted_passed,
        "targeted_streamlit_page_import_tests_summary_v428": str(test_result["summary_line"]),
        "repository_ruff_clean_v428": after_counts["repository_total"] == 0,
        "notebook_ruff_clean_v428": notebook_after == 0,
        "full_repository_pytest_run_v428": False,
        "full_quarto_render_run_v428": False,
        "working_champion_claim_allowed_v428": False,
        "paper1_promotion_allowed_v428": False,
        "paper4_working_champion_changed_v428": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v428": NEXT_ARTIFACT,
        "claim_boundary": (
            "v428 applies a targeted Streamlit B905/C408 repair; repository ruff "
            "and full pytest/final promotion claims remain blocked"
        ),
    }
    if len(after_target_items) != 0:
        raise RuntimeError("v428 expected target Streamlit B905/C408 diagnostics to be cleared.")
    if status["repo_ruff_total_after_v428"] != 38:
        raise RuntimeError("v428 expected repository ruff to contract to 38 diagnostics.")
    if status["streamlit_diagnostics_after_v428"] != 0:
        raise RuntimeError("v428 expected Streamlit diagnostics to be cleared.")
    if status["repo_ruff_b905_after_v428"] != 0 or status["repo_ruff_c408_after_v428"] != 0:
        raise RuntimeError("v428 expected repository B905/C408 to be cleared.")
    if sorted(changed_files) != [TARGET_FILE]:
        raise RuntimeError("v428 changed an unexpected Streamlit file set.")
    if not targeted_passed:
        raise RuntimeError("v428 targeted Streamlit page import tests failed.")

    REPAIR_MD.write_text(_repair_markdown(status, fix_result, test_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v428": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
