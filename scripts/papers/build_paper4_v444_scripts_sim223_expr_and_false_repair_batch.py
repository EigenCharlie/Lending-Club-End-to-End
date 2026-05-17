#!/usr/bin/env python3
"""Build Paper 4 v444 targeted SIM223 expr-and-false repair artifacts."""

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

VERSION = 444
PRIOR_POST_C405_REPAIR_PYTEST_VERSION = 443
TARGET_FILES = ["scripts/papers/build_paper4_v41_v44_living_lab_wave.py"]
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_TARGET_SIM223_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    *TARGET_FILES,
    "--select",
    "SIM223",
    "--output-format",
    "json",
]
RUFF_TARGET_UP018_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    *TARGET_FILES,
    "--select",
    "UP018",
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
    "SIM223",
    "--fix",
    "--unsafe-fixes",
]
RUFF_UP018_FIX_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    *TARGET_FILES,
    "--select",
    "UP018",
    "--fix",
]
NEXT_ARTIFACT = "paper4_v445_post_scripts_sim223_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v444_scripts_sim223_expr_and_false_repair_batch.md"


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


def _run_ruff_fix(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "targeted ruff fix failed")
    return {
        "command": " ".join(command),
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
        "repository_sim223": int(rule_counts.get("SIM223", 0)),
        "repository_c405": int(rule_counts.get("C405", 0)),
        "repository_up018": int(rule_counts.get("UP018", 0)),
        "repository_sim108": int(rule_counts.get("SIM108", 0)),
        "repository_up022": int(rule_counts.get("UP022", 0)),
        "repository_f401": int(rule_counts.get("F401", 0)),
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


def _actions(
    diagnostics: list[dict[str, Any]],
    *,
    action_label: str,
    claim_boundary: str,
    unsafe_fix_applied: bool,
) -> pd.DataFrame:
    rows = []
    for item in diagnostics:
        fix = item.get("fix") or {}
        rule = str(item["code"]).lower()
        rows.append(
            {
                "action_id_v444": f"scripts_{rule}_{action_label}_{len(rows) + 1:02d}",
                "file_path_v444": _relative_path(str(item["filename"])),
                "surface_v444": _surface(_relative_path(str(item["filename"]))),
                "row_v444": int((item.get("location") or {}).get("row") or 0),
                "rule_code_v444": str(item["code"]),
                "message_v444": str(item["message"]),
                "fix_message_v444": str(fix.get("message") or ""),
                "fix_edit_count_v444": int(len(fix.get("edits") or [])),
                "unsafe_fix_applied_v444": unsafe_fix_applied,
                "mutation_applied_v444": True,
                "claim_boundary_v444": claim_boundary,
            }
        )
    return pd.DataFrame(rows)


def _frame_from_counts(
    before_items: list[dict[str, Any]], after_items: list[dict[str, Any]], suffix: str
) -> pd.DataFrame:
    before = _snapshot_counts(before_items)
    after = _snapshot_counts(after_items)
    metrics = [
        "repository_total",
        "repository_b023",
        "repository_sim223",
        "repository_c405",
        "repository_up018",
        "repository_sim108",
        "repository_up022",
        "repository_f401",
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
                f"metric_{suffix}": metric,
                f"before_{suffix}": int(before[metric]),
                f"after_{suffix}": int(after[metric]),
                f"delta_{suffix}": int(after[metric] - before[metric]),
                f"claim_boundary_{suffix}": "ruff-count delta only; repository ruff remains open unless after=0",
            }
            for metric in metrics
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v444": metric,
                "diagnostic_count_v444": int(count),
                "claim_boundary_v444": "post-v444 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


def _pycompile_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compile_id_v444": "changed_scripts_py_compile",
                "command_v444": result["command"],
                "exit_code_v444": int(result["exit_code"]),
                "passed_v444": bool(result["passed"]),
                "compiled_files_v444": int(result["compiled_files"]),
                "stdout_tail_v444": str(result["stdout_tail"]),
                "stderr_tail_v444": str(result["stderr_tail"]),
                "claim_boundary_v444": "syntax/bytecode check only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, pycompile_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v444": "repository_ruff_frontier_still_open",
            "blocking_v444": after_total > 0,
            "evidence_count_v444": after_total,
            "required_next_artifact_v444": NEXT_ARTIFACT,
            "claim_boundary_v444": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v444": "full_repository_pytest_deferred_after_sim223_repair",
            "blocking_v444": True,
            "evidence_count_v444": 1,
            "required_next_artifact_v444": NEXT_ARTIFACT,
            "claim_boundary_v444": "py_compile does not replace full pytest",
        },
        {
            "blocker_id_v444": "paper4_final_promotion_forbidden",
            "blocking_v444": True,
            "evidence_count_v444": 1,
            "required_next_artifact_v444": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v444": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pycompile_passed:
        rows.insert(
            0,
            {
                "blocker_id_v444": "changed_scripts_pycompile_failed",
                "blocking_v444": True,
                "evidence_count_v444": 1,
                "required_next_artifact_v444": "paper4_v445_sim223_pycompile_failure_triage.md",
                "claim_boundary_v444": "compile failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, after_total: int, sim223_after: int, pycompile_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v444_sim223_expr_and_false_repair_applied",
                "allowed": True,
                "artifact": "paper4_v444_scripts_sim223_actions.csv",
                "boundary": "targeted scripts SIM223 expr-and-false repair batch",
            },
            {
                "claim_id": "v444_sim223_cleared",
                "allowed": sim223_after == 0,
                "artifact": "paper4_v444_repository_ruff_delta.csv",
                "boundary": "true only for SIM223 diagnostics",
            },
            {
                "claim_id": "v444_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v444_repository_ruff_delta.csv",
                "boundary": "repository ruff diagnostic count decreases",
            },
            {
                "claim_id": "v444_changed_scripts_pycompile_passed",
                "allowed": pycompile_passed,
                "artifact": "paper4_v444_pycompile_summary.csv",
                "boundary": "syntax/bytecode check for changed scripts",
            },
            {
                "claim_id": "v444_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v444_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v444_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v444_claim_blockers.csv",
                "boundary": "full pytest deferred to v445",
            },
            {
                "claim_id": "v444_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, after_total: int, sim223_after: int, pycompile_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v444 clears targeted scripts SIM223 expr-and-false diagnostics.",
                "allowed": sim223_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v444_repository_ruff_delta.csv",
                "boundary": "SIM223 only; B023 remains a separate manual frontier.",
                "prohibited_claim_flag": sim223_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v444 reduces repository ruff diagnostics after SIM223 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v444_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff remains open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v444 changed scripts compile after SIM223 repair.",
                "allowed": pycompile_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v444_pycompile_summary.csv",
                "boundary": "py_compile only; full pytest deferred.",
                "prohibited_claim_flag": not pycompile_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v444 proves repository ruff clean or full pytest clean after SIM223 repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v444_claim_blockers.csv",
                "boundary": f"{after_total} repository ruff diagnostics remain and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v444 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v444_claim_blockers.csv",
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
                "executable_item": "v444 applies targeted scripts SIM223 expr-and-false repair.",
                "status": "targeted_scripts_sim223_expr_and_false_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v445 full repository pytest passes after SIM223 repair",
                "last_wave": "v444",
                "execution_result": f"repo_ruff_reduced_8_to_{after_total}_sim223_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v444")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], fix_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Scripts SIM223 Expr-And-False Repair Batch v444

Generated: {status["generated_at_utc"]}

v444 applies the targeted SIM223 expr-and-false repair.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v444"]}` ->
  `{status["repo_ruff_total_after_v444"]}`.
- Repository SIM223 diagnostics: `{status["repo_ruff_sim223_before_v444"]}` ->
  `{status["repo_ruff_sim223_after_v444"]}`.
- Changed files: `{status["changed_files_v444"]}`.
- py_compile passed: `{status["changed_scripts_pycompile_passed_v444"]}`.
- Ruff fix command: `{fix_result["command"]}`.

## Required Caveat

v444 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v444"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V444_SCRIPTS_SIM223_EXPR_AND_FALSE_REPAIR_BATCH_START -->"
    end = "<!-- V444_SCRIPTS_SIM223_EXPR_AND_FALSE_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v444: Scripts SIM223 Expr-And-False Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v444 applies the targeted SIM223 expr-and-false repair in `build_paper4_v41_v44_living_lab_wave.py`.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v444"]}` ->
  `{status["repo_ruff_total_after_v444"]}`.
- Repository SIM223 before/after:
  `{status["repo_ruff_sim223_before_v444"]}` ->
  `{status["repo_ruff_sim223_after_v444"]}`.
- Changed files:
  `{status["changed_files_v444"]}`.
- py_compile passed:
  `{status["changed_scripts_pycompile_passed_v444"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v444"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v444"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v444"]}`.

### Interpretation

The SIM223 frontier is cleared by replacing the unreachable `... and False`
expression with `False`. Remaining repository ruff diagnostics are now B023.

### Claim Impact

- Allowed: targeted scripts SIM223 repair applied, repository ruff count reduced,
  and changed scripts compile.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v444 in the living notebook. v445 should run the post-SIM223-repair full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _target_diff_clean():
        raise RuntimeError("v444 expects clean target diffs before mutation.")

    v443_status = json.loads((STATUS_DIR / "paper4_v443_status.json").read_text(encoding="utf-8"))
    if v443_status["next_artifact_v443"] != "paper4_v444_scripts_sim223_expr_and_false_repair_batch.md":
        raise RuntimeError("v444 expects v443 to route to scripts SIM223 repair.")
    if v443_status["full_repository_pytest_passed_v443"] is not True:
        raise RuntimeError("v444 expects v443 full pytest to pass.")
    if int(v443_status["repo_ruff_sim223_v443"]) != 1:
        raise RuntimeError("v444 expects one SIM223 diagnostic before repair.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_target_sim223 = _run_json_command(RUFF_TARGET_SIM223_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 8:
        raise RuntimeError("v444 expected repository ruff to fail with 8 diagnostics before repair.")
    if len(before_target_sim223) != 1:
        raise RuntimeError("v444 expected one target SIM223 diagnostic before repair.")

    before_counts = _snapshot_counts(before_repo_items)
    sim223_actions = _actions(
        before_target_sim223,
        action_label="expr_and_false",
        claim_boundary="targeted scripts SIM223 expr-and-false repair",
        unsafe_fix_applied=True,
    )
    sim223_fix_result = _run_ruff_fix(RUFF_FIX_COMMAND)
    _, induced_up018 = _run_json_command(RUFF_TARGET_UP018_COMMAND)
    up018_actions = _actions(
        induced_up018,
        action_label="induced_literal_cleanup",
        claim_boundary="induced UP018 cleanup after SIM223 repair",
        unsafe_fix_applied=False,
    )
    up018_fix_result = None
    if induced_up018:
        up018_fix_result = _run_ruff_fix(RUFF_UP018_FIX_COMMAND)
    actions = pd.concat([sim223_actions, up018_actions], ignore_index=True)
    fix_commands = [sim223_fix_result["command"]]
    if up018_fix_result is not None:
        fix_commands.append(up018_fix_result["command"])
    fix_result = {"command": " | ".join(fix_commands)}
    changed_files = _changed_target_files()
    _, after_target_sim223 = _run_json_command(RUFF_TARGET_SIM223_COMMAND)
    _, after_target_up018 = _run_json_command(RUFF_TARGET_UP018_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    pycompile_result = _run_pycompile(changed_files)
    pycompile_passed = bool(pycompile_result["passed"])
    after_total = int(len(after_repo_items))
    sim223_after = int(len(after_target_sim223))
    up018_after = int(len(after_target_up018))

    write_csv(TABLE_DIR / "paper4_v444_scripts_sim223_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v444_repository_ruff_delta.csv", _frame_from_counts(before_repo_items, after_repo_items, "v444"))
    write_csv(TABLE_DIR / "paper4_v444_repository_ruff_after_snapshot.csv", _after_snapshot(after_repo_items))
    write_csv(TABLE_DIR / "paper4_v444_pycompile_summary.csv", _pycompile_summary(pycompile_result))
    write_csv(TABLE_DIR / "paper4_v444_claim_blockers.csv", _claim_blockers(after_total=after_total, pycompile_passed=pycompile_passed))
    write_csv(
        TABLE_DIR / "paper4_v444_claim_matrix_delta.csv",
        _claim_matrix(after_total=after_total, sim223_after=sim223_after, pycompile_passed=pycompile_passed),
    )
    _update_claim_boundaries(
        after_total=after_total,
        sim223_after=sim223_after,
        pycompile_passed=pycompile_passed,
    )
    _update_backlog(after_total)

    status = {
        "phase": "v444_scripts_sim223_expr_and_false_repair_batch",
        "schema_version": "2026-05-17.444",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_c405_repair_pytest_version_v444": PRIOR_POST_C405_REPAIR_PYTEST_VERSION,
        "actions_v444": int(len(actions)),
        "repo_ruff_exit_code_before_v444": int(before_repo_exit),
        "repo_ruff_exit_code_after_v444": int(after_repo_exit),
        "repo_ruff_total_before_v444": before_counts["repository_total"],
        "repo_ruff_total_after_v444": after_counts["repository_total"],
        "repo_ruff_total_reduced_v444": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_sim223_before_v444": before_counts["repository_sim223"],
        "repo_ruff_sim223_after_v444": after_counts["repository_sim223"],
        "repo_ruff_sim223_reduced_v444": before_counts["repository_sim223"] - after_counts["repository_sim223"],
        "repo_ruff_b023_after_v444": after_counts["repository_b023"],
        "repo_ruff_c405_after_v444": after_counts["repository_c405"],
        "repo_ruff_up018_induced_v444": int(len(induced_up018)),
        "repo_ruff_up018_after_v444": after_counts["repository_up018"],
        "repo_ruff_sim108_after_v444": after_counts["repository_sim108"],
        "repo_ruff_up022_after_v444": after_counts["repository_up022"],
        "repo_ruff_f401_after_v444": after_counts["repository_f401"],
        "repo_ruff_i001_after_v444": after_counts["repository_i001"],
        "repo_ruff_f841_after_v444": after_counts["repository_f841"],
        "repo_ruff_b007_after_v444": after_counts["repository_b007"],
        "repo_ruff_b905_after_v444": after_counts["repository_b905"],
        "repo_ruff_c408_after_v444": after_counts["repository_c408"],
        "scripts_diagnostics_before_v444": before_counts["scripts_total"],
        "scripts_diagnostics_after_v444": after_counts["scripts_total"],
        "book_diagnostics_after_v444": after_counts["book_total"],
        "streamlit_diagnostics_after_v444": after_counts["streamlit_app_total"],
        "notebook_diagnostics_after_v444": after_counts["notebook_total"],
        "changed_files_v444": int(len(changed_files)),
        "changed_file_list_v444": changed_files,
        "changed_scripts_pycompile_run_v444": True,
        "changed_scripts_pycompile_passed_v444": pycompile_passed,
        "changed_scripts_pycompile_files_v444": int(pycompile_result["compiled_files"]),
        "repository_ruff_clean_v444": after_counts["repository_total"] == 0,
        "full_repository_pytest_run_v444": False,
        "full_quarto_render_run_v444": False,
        "working_champion_claim_allowed_v444": False,
        "paper1_promotion_allowed_v444": False,
        "paper4_working_champion_changed_v444": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v444": NEXT_ARTIFACT,
        "claim_boundary": (
            "v444 applies a targeted scripts SIM223 repair; repository ruff "
            "and full pytest/final promotion claims remain blocked"
        ),
    }
    if sim223_after != 0:
        raise RuntimeError("v444 expected SIM223 to be cleared.")
    if up018_after != 0:
        raise RuntimeError("v444 expected induced UP018 to be cleared.")
    if status["repo_ruff_total_after_v444"] != 7:
        raise RuntimeError("v444 expected repository ruff to contract to 7 diagnostics.")
    if status["streamlit_diagnostics_after_v444"] != 0 or status["notebook_diagnostics_after_v444"] != 0:
        raise RuntimeError("v444 expected Streamlit and notebooks to remain clean.")
    if not pycompile_passed:
        raise RuntimeError("v444 changed scripts did not compile.")

    REPAIR_MD.write_text(_repair_markdown(status, fix_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v444": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
