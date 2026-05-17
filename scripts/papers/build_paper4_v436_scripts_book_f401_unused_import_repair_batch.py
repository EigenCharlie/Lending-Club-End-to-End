#!/usr/bin/env python3
"""Build Paper 4 v436 targeted F401 unused-import repair batch artifacts."""

from __future__ import annotations

import json
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

VERSION = 436
PRIOR_POST_I001_REPAIR_PYTEST_VERSION = 435
TARGET_FILES = [
    "book/_helpers/plot_helpers.py",
    "scripts/papers/build_paper4_v398_notebook_historical_e402_policy.py",
    "scripts/papers/build_paper4_v399_notebook_e402_cell_refactor_plan.py",
    "scripts/papers/build_paper4_v406_post_sys_path_refactor_pytest_probe.py",
]
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_TARGET_F401_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    *TARGET_FILES,
    "--select",
    "F401",
    "--output-format",
    "json",
]
RUFF_FIX_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    *TARGET_FILES,
    "--select",
    "F401",
    "--fix",
]
NEXT_ARTIFACT = "paper4_v437_post_scripts_f401_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v436_scripts_book_f401_unused_import_repair_batch.md"


def _run_json_command(command: list[str]) -> tuple[int, list[dict[str, Any]]]:
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or f"{' '.join(command)} failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("ruff JSON output is not a list")
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
    return "other"


def _target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *TARGET_FILES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _changed_target_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *TARGET_FILES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _run_ruff_fix() -> dict[str, Any]:
    result = subprocess.run(RUFF_FIX_COMMAND, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "targeted F401 ruff fix failed")
    return {
        "command": " ".join(RUFF_FIX_COMMAND),
        "exit_code": int(result.returncode),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


def _run_pycompile(paths: list[str]) -> dict[str, Any]:
    command = ["uv", "run", "python", "-m", "py_compile", *paths]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    return {
        "command": " ".join(command),
        "exit_code": int(result.returncode),
        "passed": result.returncode == 0,
        "compiled_files": len(paths),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


def _snapshot_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    rule_counts = Counter(str(item["code"]) for item in items)
    surface_counts = Counter(_surface(_relative_path(str(item["filename"]))) for item in items)
    return {
        "repository_total": int(len(items)),
        "repository_b023": int(rule_counts.get("B023", 0)),
        "repository_f401": int(rule_counts.get("F401", 0)),
        "repository_up022": int(rule_counts.get("UP022", 0)),
        "repository_sim108": int(rule_counts.get("SIM108", 0)),
        "repository_c405": int(rule_counts.get("C405", 0)),
        "repository_sim223": int(rule_counts.get("SIM223", 0)),
        "repository_i001": int(rule_counts.get("I001", 0)),
        "repository_f841": int(rule_counts.get("F841", 0)),
        "repository_b007": int(rule_counts.get("B007", 0)),
        "repository_b905": int(rule_counts.get("B905", 0)),
        "repository_c408": int(rule_counts.get("C408", 0)),
        "scripts_total": int(surface_counts.get("scripts", 0)),
        "book_total": int(surface_counts.get("book", 0)),
        "streamlit_app_total": int(surface_counts.get("streamlit_app", 0)),
        "notebook_total": int(surface_counts.get("notebook", 0)),
    }


def _actions(before_target_f401: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in before_target_f401:
        fix = item.get("fix") or {}
        edits = fix.get("edits") or []
        message = str(item["message"])
        unused_import = message.split("`")[1] if "`" in message else ""
        rows.append(
            {
                "action_id_v436": f"scripts_book_f401_unused_import_{len(rows) + 1:02d}",
                "file_path_v436": _relative_path(str(item["filename"])),
                "surface_v436": _surface(_relative_path(str(item["filename"]))),
                "row_v436": int((item.get("location") or {}).get("row") or 0),
                "rule_code_v436": str(item["code"]),
                "unused_import_v436": unused_import,
                "message_v436": message,
                "fix_message_v436": str(fix.get("message") or ""),
                "fix_edit_count_v436": int(len(edits)),
                "mutation_applied_v436": True,
                "claim_boundary_v436": "targeted scripts/book F401 unused-import repair",
            }
        )
    return pd.DataFrame(rows)


def _delta_table(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before = _snapshot_counts(before_items)
    after = _snapshot_counts(after_items)
    metrics = [
        "repository_total",
        "repository_b023",
        "repository_f401",
        "repository_up022",
        "repository_sim108",
        "repository_c405",
        "repository_sim223",
        "repository_i001",
        "repository_f841",
        "repository_b007",
        "repository_b905",
        "repository_c408",
        "scripts_total",
        "book_total",
        "streamlit_app_total",
        "notebook_total",
    ]
    return pd.DataFrame(
        [
            {
                "metric_v436": metric,
                "before_v436": int(before[metric]),
                "after_v436": int(after[metric]),
                "delta_v436": int(after[metric] - before[metric]),
                "claim_boundary_v436": "ruff-count delta only; repository ruff remains open unless after=0",
            }
            for metric in metrics
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v436": metric,
                "diagnostic_count_v436": int(count),
                "claim_boundary_v436": "post-v436 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


def _pycompile_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compile_id_v436": "changed_scripts_and_book_py_compile",
                "command_v436": result["command"],
                "exit_code_v436": int(result["exit_code"]),
                "passed_v436": bool(result["passed"]),
                "compiled_files_v436": int(result["compiled_files"]),
                "stdout_tail_v436": str(result["stdout_tail"]),
                "stderr_tail_v436": str(result["stderr_tail"]),
                "claim_boundary_v436": "syntax/bytecode check only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, pycompile_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v436": "repository_ruff_frontier_still_open",
            "blocking_v436": after_total > 0,
            "evidence_count_v436": after_total,
            "required_next_artifact_v436": NEXT_ARTIFACT,
            "claim_boundary_v436": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v436": "full_repository_pytest_deferred_after_f401_repair",
            "blocking_v436": True,
            "evidence_count_v436": 1,
            "required_next_artifact_v436": NEXT_ARTIFACT,
            "claim_boundary_v436": "py_compile does not replace full pytest",
        },
        {
            "blocker_id_v436": "paper4_final_promotion_forbidden",
            "blocking_v436": True,
            "evidence_count_v436": 1,
            "required_next_artifact_v436": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v436": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pycompile_passed:
        rows.insert(
            0,
            {
                "blocker_id_v436": "changed_files_pycompile_failed",
                "blocking_v436": True,
                "evidence_count_v436": 1,
                "required_next_artifact_v436": "paper4_v437_f401_pycompile_failure_triage.md",
                "claim_boundary_v436": "compile failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, after_total: int, f401_after: int, pycompile_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v436_f401_unused_import_repair_applied",
                "allowed": True,
                "artifact": "paper4_v436_scripts_book_f401_actions.csv",
                "boundary": "targeted scripts/book F401 unused-import repair batch",
            },
            {
                "claim_id": "v436_f401_cleared",
                "allowed": f401_after == 0,
                "artifact": "paper4_v436_repository_ruff_delta.csv",
                "boundary": "true only for F401 diagnostics",
            },
            {
                "claim_id": "v436_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v436_repository_ruff_delta.csv",
                "boundary": "repository ruff diagnostic count decreases",
            },
            {
                "claim_id": "v436_changed_files_pycompile_passed",
                "allowed": pycompile_passed,
                "artifact": "paper4_v436_pycompile_summary.csv",
                "boundary": "syntax/bytecode check for changed scripts/book files",
            },
            {
                "claim_id": "v436_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v436_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v436_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v436_claim_blockers.csv",
                "boundary": "full pytest deferred to v437",
            },
            {
                "claim_id": "v436_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, after_total: int, f401_after: int, pycompile_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v436 clears targeted scripts/book F401 unused-import diagnostics.",
                "allowed": f401_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v436_repository_ruff_delta.csv",
                "boundary": "F401 only; B023/UP022/SIM108/C405/SIM223 remain separate frontiers.",
                "prohibited_claim_flag": f401_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v436 reduces repository ruff diagnostics after F401 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v436_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff remains open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v436 changed scripts/book files compile after F401 repair.",
                "allowed": pycompile_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v436_pycompile_summary.csv",
                "boundary": "py_compile only; full pytest deferred.",
                "prohibited_claim_flag": not pycompile_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v436 proves repository ruff clean or full pytest clean after F401 repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v436_claim_blockers.csv",
                "boundary": f"{after_total} repository ruff diagnostics remain and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v436 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v436_claim_blockers.csv",
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
                "executable_item": "v436 applies targeted scripts/book F401 unused-import repair.",
                "status": "targeted_scripts_book_f401_unused_import_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v437 full repository pytest passes after F401 unused-import repair",
                "last_wave": "v436",
                "execution_result": f"repo_ruff_reduced_18_to_{after_total}_f401_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v436")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], fix_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Scripts/Book F401 Unused-Import Repair Batch v436

Generated: {status["generated_at_utc"]}

v436 applies targeted unused-import repairs across scripts/book F401 diagnostics.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v436"]}` ->
  `{status["repo_ruff_total_after_v436"]}`.
- Repository F401 diagnostics: `{status["repo_ruff_f401_before_v436"]}` ->
  `{status["repo_ruff_f401_after_v436"]}`.
- Scripts diagnostics: `{status["scripts_diagnostics_before_v436"]}` ->
  `{status["scripts_diagnostics_after_v436"]}`.
- Book diagnostics: `{status["book_diagnostics_before_v436"]}` ->
  `{status["book_diagnostics_after_v436"]}`.
- Changed files: `{status["changed_files_v436"]}`.
- py_compile passed: `{status["changed_files_pycompile_passed_v436"]}`.
- Ruff fix command: `{fix_result["command"]}`.

## Required Caveat

v436 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v436"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V436_SCRIPTS_BOOK_F401_UNUSED_IMPORT_REPAIR_BATCH_START -->"
    end = "<!-- V436_SCRIPTS_BOOK_F401_UNUSED_IMPORT_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v436: Scripts/Book F401 Unused-Import Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v436 applies targeted unused-import repairs across scripts/book F401 diagnostics.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v436"]}` ->
  `{status["repo_ruff_total_after_v436"]}`.
- Repository F401 before/after:
  `{status["repo_ruff_f401_before_v436"]}` ->
  `{status["repo_ruff_f401_after_v436"]}`.
- Scripts diagnostics before/after:
  `{status["scripts_diagnostics_before_v436"]}` ->
  `{status["scripts_diagnostics_after_v436"]}`.
- Book diagnostics before/after:
  `{status["book_diagnostics_before_v436"]}` ->
  `{status["book_diagnostics_after_v436"]}`.
- Changed files:
  `{status["changed_files_v436"]}`.
- py_compile passed:
  `{status["changed_files_pycompile_passed_v436"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v436"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v436"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v436"]}`.

### Interpretation

The F401 frontier is cleared by removing unused imports from the book helper
and three Paper 4 script builders. Remaining repository ruff diagnostics are
now B023, UP022, SIM108, C405 and SIM223.

### Claim Impact

- Allowed: targeted scripts/book F401 repair applied, repository ruff count
  reduced, and changed files compile.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v436 in the living notebook. v437 should run the post-F401-repair full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _target_diff_clean():
        raise RuntimeError("v436 expects clean target diffs before mutation.")

    v435_status = json.loads((STATUS_DIR / "paper4_v435_status.json").read_text(encoding="utf-8"))
    if v435_status["next_artifact_v435"] != "paper4_v436_scripts_book_f401_unused_import_repair_batch.md":
        raise RuntimeError("v436 expects v435 to route to scripts/book F401 repair.")
    if v435_status["full_repository_pytest_passed_v435"] is not True:
        raise RuntimeError("v436 expects v435 full pytest to pass.")
    if int(v435_status["repo_ruff_f401_v435"]) != 4:
        raise RuntimeError("v436 expects four F401 diagnostics before repair.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_target_f401 = _run_json_command(RUFF_TARGET_F401_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 18:
        raise RuntimeError("v436 expected repository ruff to fail with 18 diagnostics before repair.")
    if len(before_target_f401) != 4:
        raise RuntimeError("v436 expected four target F401 diagnostics before repair.")

    before_counts = _snapshot_counts(before_repo_items)
    actions = _actions(before_target_f401)
    fix_result = _run_ruff_fix()
    changed_files = _changed_target_files()
    _, after_target_f401 = _run_json_command(RUFF_TARGET_F401_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    pycompile_result = _run_pycompile(changed_files)
    pycompile_passed = bool(pycompile_result["passed"])
    after_total = int(len(after_repo_items))
    f401_after = int(len(after_target_f401))

    delta = _delta_table(before_repo_items, after_repo_items)
    after_snapshot = _after_snapshot(after_repo_items)
    blockers = _claim_blockers(after_total=after_total, pycompile_passed=pycompile_passed)
    claim_matrix = _claim_matrix(
        after_total=after_total,
        f401_after=f401_after,
        pycompile_passed=pycompile_passed,
    )
    _update_claim_boundaries(
        after_total=after_total,
        f401_after=f401_after,
        pycompile_passed=pycompile_passed,
    )
    _update_backlog(after_total)

    write_csv(TABLE_DIR / "paper4_v436_scripts_book_f401_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v436_repository_ruff_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v436_repository_ruff_after_snapshot.csv", after_snapshot)
    write_csv(TABLE_DIR / "paper4_v436_pycompile_summary.csv", _pycompile_summary(pycompile_result))
    write_csv(TABLE_DIR / "paper4_v436_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v436_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v436_scripts_book_f401_unused_import_repair_batch",
        "schema_version": "2026-05-17.436",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_i001_repair_pytest_version_v436": PRIOR_POST_I001_REPAIR_PYTEST_VERSION,
        "actions_v436": int(len(actions)),
        "repo_ruff_exit_code_before_v436": int(before_repo_exit),
        "repo_ruff_exit_code_after_v436": int(after_repo_exit),
        "repo_ruff_total_before_v436": before_counts["repository_total"],
        "repo_ruff_total_after_v436": after_counts["repository_total"],
        "repo_ruff_total_reduced_v436": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_f401_before_v436": before_counts["repository_f401"],
        "repo_ruff_f401_after_v436": after_counts["repository_f401"],
        "repo_ruff_f401_reduced_v436": before_counts["repository_f401"] - after_counts["repository_f401"],
        "repo_ruff_b023_after_v436": after_counts["repository_b023"],
        "repo_ruff_up022_after_v436": after_counts["repository_up022"],
        "repo_ruff_sim108_after_v436": after_counts["repository_sim108"],
        "repo_ruff_c405_after_v436": after_counts["repository_c405"],
        "repo_ruff_sim223_after_v436": after_counts["repository_sim223"],
        "repo_ruff_i001_after_v436": after_counts["repository_i001"],
        "repo_ruff_f841_after_v436": after_counts["repository_f841"],
        "repo_ruff_b007_after_v436": after_counts["repository_b007"],
        "repo_ruff_b905_after_v436": after_counts["repository_b905"],
        "repo_ruff_c408_after_v436": after_counts["repository_c408"],
        "scripts_diagnostics_before_v436": before_counts["scripts_total"],
        "scripts_diagnostics_after_v436": after_counts["scripts_total"],
        "book_diagnostics_before_v436": before_counts["book_total"],
        "book_diagnostics_after_v436": after_counts["book_total"],
        "streamlit_diagnostics_after_v436": after_counts["streamlit_app_total"],
        "notebook_diagnostics_after_v436": after_counts["notebook_total"],
        "changed_files_v436": int(len(changed_files)),
        "changed_file_list_v436": changed_files,
        "changed_files_pycompile_run_v436": True,
        "changed_files_pycompile_passed_v436": pycompile_passed,
        "changed_files_pycompile_files_v436": int(pycompile_result["compiled_files"]),
        "repository_ruff_clean_v436": after_counts["repository_total"] == 0,
        "full_repository_pytest_run_v436": False,
        "full_quarto_render_run_v436": False,
        "working_champion_claim_allowed_v436": False,
        "paper1_promotion_allowed_v436": False,
        "paper4_working_champion_changed_v436": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v436": NEXT_ARTIFACT,
        "claim_boundary": (
            "v436 applies a targeted scripts/book F401 repair; repository ruff "
            "and full pytest/final promotion claims remain blocked"
        ),
    }
    if f401_after != 0:
        raise RuntimeError("v436 expected F401 to be cleared.")
    if status["repo_ruff_total_after_v436"] != 14:
        raise RuntimeError("v436 expected repository ruff to contract to 14 diagnostics.")
    if status["repo_ruff_f401_after_v436"] != 0:
        raise RuntimeError("v436 expected repository F401 to be cleared.")
    if status["streamlit_diagnostics_after_v436"] != 0 or status["notebook_diagnostics_after_v436"] != 0:
        raise RuntimeError("v436 expected Streamlit and notebooks to remain clean.")
    if not pycompile_passed:
        raise RuntimeError("v436 changed scripts/book files did not compile.")

    REPAIR_MD.write_text(_repair_markdown(status, fix_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v436": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
