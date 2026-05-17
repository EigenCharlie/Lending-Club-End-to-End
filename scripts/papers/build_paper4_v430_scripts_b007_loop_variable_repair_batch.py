#!/usr/bin/env python3
"""Build Paper 4 v430 targeted scripts B007 loop-variable repair batch artifacts."""

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

VERSION = 430
PRIOR_POST_STREAMLIT_REPAIR_PYTEST_VERSION = 429
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_SCRIPTS_B007_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    "scripts/papers",
    "--select",
    "B007",
    "--output-format",
    "json",
]
RUFF_FIX_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    "scripts/papers",
    "--select",
    "B007",
    "--fix",
    "--unsafe-fixes",
]
MANUAL_PATCH_FILE = "scripts/papers/build_paper4_v15_dynamic_stress_engine.py"
NEXT_ARTIFACT = "paper4_v431_post_scripts_b007_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v430_scripts_b007_loop_variable_repair_batch.md"


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


def _changed_expected_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            "scripts/papers/build_paper4_extended_experiments.py",
            "scripts/papers/build_paper4_v15_dynamic_stress_engine.py",
            "scripts/papers/build_paper4_v20_dla_cvar_spo_resolution.py",
            "scripts/papers/build_paper4_v24_dla_cvar_spo_upgrade.py",
            "scripts/papers/build_paper4_v45_v48_living_lab_wave.py",
            "scripts/papers/build_paper4_v5_blocker_resolution.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _expected_diff_clean() -> bool:
    return not _changed_expected_files()


def _run_ruff_fix() -> dict[str, Any]:
    result = subprocess.run(
        RUFF_FIX_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "scripts B007 ruff fix failed")
    return {
        "command": " ".join(RUFF_FIX_COMMAND),
        "exit_code": int(result.returncode),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


def _apply_manual_policy_id_patch() -> bool:
    path = ROOT / MANUAL_PATCH_FILE
    text = path.read_text(encoding="utf-8")
    old = '    for policy_id, policy_book in books.groupby("policy_id", sort=False):\n'
    new = '    for _policy_id, policy_book in books.groupby("policy_id", sort=False):\n'
    if new in text:
        return False
    if old not in text:
        raise RuntimeError("v430 manual B007 patch target was not found.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _run_pycompile(paths: list[str]) -> dict[str, Any]:
    command = ["uv", "run", "python", "-m", "py_compile", *paths]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
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
        "repository_b007": int(rule_counts.get("B007", 0)),
        "repository_f841": int(rule_counts.get("F841", 0)),
        "repository_b023": int(rule_counts.get("B023", 0)),
        "repository_i001": int(rule_counts.get("I001", 0)),
        "repository_b905": int(rule_counts.get("B905", 0)),
        "repository_c408": int(rule_counts.get("C408", 0)),
        "scripts_total": int(surface_counts.get("scripts", 0)),
        "book_total": int(surface_counts.get("book", 0)),
        "streamlit_app_total": int(surface_counts.get("streamlit_app", 0)),
        "notebook_total": int(surface_counts.get("notebook", 0)),
    }


def _actions(before_scripts_b007: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in before_scripts_b007:
        fix = item.get("fix") or {}
        edits = fix.get("edits") or []
        file_path = _relative_path(str(item["filename"]))
        variable = str(item["message"]).split("`")[1] if "`" in str(item["message"]) else ""
        rows.append(
            {
                "action_id_v430": f"scripts_b007_loop_variable_{len(rows) + 1:02d}",
                "file_path_v430": file_path,
                "row_v430": int((item.get("location") or {}).get("row") or 0),
                "rule_code_v430": str(item["code"]),
                "unused_variable_v430": variable,
                "message_v430": str(item["message"]),
                "fix_message_v430": str(fix.get("message") or ""),
                "fix_edit_count_v430": int(len(edits)),
                "mutation_strategy_v430": "ruff_fix" if edits else "manual_underscore_prefix",
                "mutation_applied_v430": True,
                "claim_boundary_v430": "targeted scripts B007 unused loop variable repair",
            }
        )
    return pd.DataFrame(rows)


def _delta_table(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before = _snapshot_counts(before_items)
    after = _snapshot_counts(after_items)
    metrics = [
        "repository_total",
        "repository_b007",
        "repository_f841",
        "repository_b023",
        "repository_i001",
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
                "metric_v430": metric,
                "before_v430": int(before[metric]),
                "after_v430": int(after[metric]),
                "delta_v430": int(after[metric] - before[metric]),
                "claim_boundary_v430": "ruff-count delta only; repository ruff remains open unless after=0",
            }
            for metric in metrics
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v430": metric,
                "diagnostic_count_v430": int(count),
                "claim_boundary_v430": "post-v430 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


def _pycompile_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compile_id_v430": "changed_scripts_py_compile",
                "command_v430": result["command"],
                "exit_code_v430": int(result["exit_code"]),
                "passed_v430": bool(result["passed"]),
                "compiled_files_v430": int(result["compiled_files"]),
                "stdout_tail_v430": str(result["stdout_tail"]),
                "stderr_tail_v430": str(result["stderr_tail"]),
                "claim_boundary_v430": "syntax/bytecode check only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, pycompile_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v430": "repository_ruff_frontier_still_open",
            "blocking_v430": after_total > 0,
            "evidence_count_v430": after_total,
            "required_next_artifact_v430": NEXT_ARTIFACT,
            "claim_boundary_v430": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v430": "full_repository_pytest_deferred_after_scripts_repair",
            "blocking_v430": True,
            "evidence_count_v430": 1,
            "required_next_artifact_v430": NEXT_ARTIFACT,
            "claim_boundary_v430": "py_compile does not replace full pytest",
        },
        {
            "blocker_id_v430": "paper4_final_promotion_forbidden",
            "blocking_v430": True,
            "evidence_count_v430": 1,
            "required_next_artifact_v430": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v430": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pycompile_passed:
        rows.insert(
            0,
            {
                "blocker_id_v430": "changed_scripts_pycompile_failed",
                "blocking_v430": True,
                "evidence_count_v430": 1,
                "required_next_artifact_v430": "paper4_v431_scripts_pycompile_failure_triage.md",
                "claim_boundary_v430": "compile failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, after_total: int, b007_after: int, pycompile_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v430_scripts_b007_repair_applied",
                "allowed": True,
                "artifact": "paper4_v430_scripts_b007_actions.csv",
                "boundary": "targeted B007 loop-variable repair batch",
            },
            {
                "claim_id": "v430_scripts_b007_cleared",
                "allowed": b007_after == 0,
                "artifact": "paper4_v430_repository_ruff_delta.csv",
                "boundary": "true only for scripts/papers B007 diagnostics",
            },
            {
                "claim_id": "v430_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v430_repository_ruff_delta.csv",
                "boundary": "repository ruff diagnostic count decreases",
            },
            {
                "claim_id": "v430_changed_scripts_pycompile_passed",
                "allowed": pycompile_passed,
                "artifact": "paper4_v430_pycompile_summary.csv",
                "boundary": "syntax/bytecode check for changed scripts",
            },
            {
                "claim_id": "v430_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v430_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v430_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v430_claim_blockers.csv",
                "boundary": "full pytest deferred to v431",
            },
            {
                "claim_id": "v430_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, after_total: int, b007_after: int, pycompile_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v430 clears scripts/papers B007 unused loop-variable diagnostics.",
                "allowed": b007_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v430_repository_ruff_delta.csv",
                "boundary": "Scripts B007 only; other scripts/book diagnostics remain.",
                "prohibited_claim_flag": b007_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v430 reduces repository ruff diagnostics after scripts B007 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v430_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff remains open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v430 changed scripts compile after B007 repair.",
                "allowed": pycompile_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v430_pycompile_summary.csv",
                "boundary": "py_compile only; full pytest deferred.",
                "prohibited_claim_flag": not pycompile_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v430 proves repository ruff clean or full pytest clean after scripts repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v430_claim_blockers.csv",
                "boundary": f"{after_total} repository ruff diagnostics remain and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v430 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v430_claim_blockers.csv",
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
                "executable_item": "v430 applies targeted scripts B007 loop-variable repair.",
                "status": "targeted_scripts_b007_loop_variable_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v431 full repository pytest passes after scripts B007 repair",
                "last_wave": "v430",
                "execution_result": f"repo_ruff_reduced_38_to_{after_total}_scripts_b007_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v430")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], fix_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Scripts B007 Loop-Variable Repair Batch v430

Generated: {status["generated_at_utc"]}

v430 applies targeted unused loop-variable repairs across scripts/papers B007 diagnostics.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v430"]}` ->
  `{status["repo_ruff_total_after_v430"]}`.
- Repository B007 diagnostics: `{status["repo_ruff_b007_before_v430"]}` ->
  `{status["repo_ruff_b007_after_v430"]}`.
- Scripts diagnostics: `{status["scripts_diagnostics_before_v430"]}` ->
  `{status["scripts_diagnostics_after_v430"]}`.
- Changed script files: `{status["changed_script_files_v430"]}`.
- py_compile passed: `{status["changed_scripts_pycompile_passed_v430"]}`.
- Ruff fix command: `{fix_result["command"]}`.
- Manual patch file: `{MANUAL_PATCH_FILE}`.

## Required Caveat

v430 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v430"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V430_SCRIPTS_B007_LOOP_VARIABLE_REPAIR_BATCH_START -->"
    end = "<!-- V430_SCRIPTS_B007_LOOP_VARIABLE_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v430: Scripts B007 Loop-Variable Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v430 applies targeted unused loop-variable repairs across scripts/papers B007
diagnostics.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v430"]}` ->
  `{status["repo_ruff_total_after_v430"]}`.
- Repository B007 before/after:
  `{status["repo_ruff_b007_before_v430"]}` ->
  `{status["repo_ruff_b007_after_v430"]}`.
- Scripts diagnostics before/after:
  `{status["scripts_diagnostics_before_v430"]}` ->
  `{status["scripts_diagnostics_after_v430"]}`.
- Changed script files:
  `{status["changed_script_files_v430"]}`.
- py_compile passed:
  `{status["changed_scripts_pycompile_passed_v430"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v430"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v430"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v430"]}`.

### Interpretation

The scripts B007 frontier is cleared, including the one non-autofixable grouped
loop key in v15. Remaining repository ruff diagnostics are non-B007 scripts/book
items.

### Claim Impact

- Allowed: targeted scripts B007 repair applied, repository ruff count reduced,
  and changed scripts compile.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v430 in the living notebook. v431 should run the post-scripts-B007-repair
full pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _expected_diff_clean():
        raise RuntimeError("v430 expects clean target script diffs before mutation.")

    v429_status = json.loads((STATUS_DIR / "paper4_v429_status.json").read_text(encoding="utf-8"))
    if v429_status["next_artifact_v429"] != "paper4_v430_scripts_b007_loop_variable_repair_batch.md":
        raise RuntimeError("v430 expects v429 to route to scripts B007 repair.")
    if v429_status["full_repository_pytest_passed_v429"] is not True:
        raise RuntimeError("v430 expects v429 full pytest to pass.")
    if int(v429_status["repo_ruff_b007_v429"]) != 8:
        raise RuntimeError("v430 expects eight B007 diagnostics before repair.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_scripts_b007 = _run_json_command(RUFF_SCRIPTS_B007_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 38:
        raise RuntimeError("v430 expected repository ruff to fail with 38 diagnostics before repair.")
    if len(before_scripts_b007) != 8:
        raise RuntimeError("v430 expected eight scripts/papers B007 diagnostics before repair.")

    before_counts = _snapshot_counts(before_repo_items)
    actions = _actions(before_scripts_b007)
    fix_result = _run_ruff_fix()
    manual_patch_applied = _apply_manual_policy_id_patch()
    changed_files = _changed_expected_files()
    _, after_scripts_b007 = _run_json_command(RUFF_SCRIPTS_B007_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    pycompile_result = _run_pycompile(changed_files)
    pycompile_passed = bool(pycompile_result["passed"])
    delta = _delta_table(before_repo_items, after_repo_items)
    after_snapshot = _after_snapshot(after_repo_items)
    after_total = int(len(after_repo_items))
    b007_after = int(len(after_scripts_b007))
    blockers = _claim_blockers(after_total=after_total, pycompile_passed=pycompile_passed)
    claim_matrix = _claim_matrix(
        after_total=after_total,
        b007_after=b007_after,
        pycompile_passed=pycompile_passed,
    )
    _update_claim_boundaries(
        after_total=after_total,
        b007_after=b007_after,
        pycompile_passed=pycompile_passed,
    )
    _update_backlog(after_total)

    write_csv(TABLE_DIR / "paper4_v430_scripts_b007_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v430_repository_ruff_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v430_repository_ruff_after_snapshot.csv", after_snapshot)
    write_csv(TABLE_DIR / "paper4_v430_pycompile_summary.csv", _pycompile_summary(pycompile_result))
    write_csv(TABLE_DIR / "paper4_v430_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v430_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v430_scripts_b007_loop_variable_repair_batch",
        "schema_version": "2026-05-17.430",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_streamlit_repair_pytest_version_v430": PRIOR_POST_STREAMLIT_REPAIR_PYTEST_VERSION,
        "actions_v430": int(len(actions)),
        "manual_patch_applied_v430": manual_patch_applied,
        "repo_ruff_exit_code_before_v430": int(before_repo_exit),
        "repo_ruff_exit_code_after_v430": int(after_repo_exit),
        "repo_ruff_total_before_v430": before_counts["repository_total"],
        "repo_ruff_total_after_v430": after_counts["repository_total"],
        "repo_ruff_total_reduced_v430": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_b007_before_v430": before_counts["repository_b007"],
        "repo_ruff_b007_after_v430": after_counts["repository_b007"],
        "repo_ruff_b007_reduced_v430": before_counts["repository_b007"] - after_counts["repository_b007"],
        "repo_ruff_f841_after_v430": after_counts["repository_f841"],
        "repo_ruff_b023_after_v430": after_counts["repository_b023"],
        "repo_ruff_i001_after_v430": after_counts["repository_i001"],
        "repo_ruff_b905_after_v430": after_counts["repository_b905"],
        "repo_ruff_c408_after_v430": after_counts["repository_c408"],
        "scripts_diagnostics_before_v430": before_counts["scripts_total"],
        "scripts_diagnostics_after_v430": after_counts["scripts_total"],
        "book_diagnostics_after_v430": after_counts["book_total"],
        "streamlit_diagnostics_after_v430": after_counts["streamlit_app_total"],
        "notebook_diagnostics_after_v430": after_counts["notebook_total"],
        "changed_script_files_v430": int(len(changed_files)),
        "changed_script_file_list_v430": changed_files,
        "changed_scripts_pycompile_run_v430": True,
        "changed_scripts_pycompile_passed_v430": pycompile_passed,
        "changed_scripts_pycompile_files_v430": int(pycompile_result["compiled_files"]),
        "repository_ruff_clean_v430": after_counts["repository_total"] == 0,
        "full_repository_pytest_run_v430": False,
        "full_quarto_render_run_v430": False,
        "working_champion_claim_allowed_v430": False,
        "paper1_promotion_allowed_v430": False,
        "paper4_working_champion_changed_v430": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v430": NEXT_ARTIFACT,
        "claim_boundary": (
            "v430 applies a targeted scripts B007 repair; repository ruff "
            "and full pytest/final promotion claims remain blocked"
        ),
    }
    if b007_after != 0:
        raise RuntimeError("v430 expected scripts/papers B007 to be cleared.")
    if status["repo_ruff_total_after_v430"] != 30:
        raise RuntimeError("v430 expected repository ruff to contract to 30 diagnostics.")
    if status["repo_ruff_b007_after_v430"] != 0:
        raise RuntimeError("v430 expected repository B007 to be cleared.")
    if status["streamlit_diagnostics_after_v430"] != 0 or status["notebook_diagnostics_after_v430"] != 0:
        raise RuntimeError("v430 expected Streamlit and notebooks to remain clean.")
    if not pycompile_passed:
        raise RuntimeError("v430 changed scripts did not compile.")

    REPAIR_MD.write_text(_repair_markdown(status, fix_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v430": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
