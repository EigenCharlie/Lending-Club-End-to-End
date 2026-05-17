#!/usr/bin/env python3
"""Build Paper 4 v426 targeted scripts B905 ruff repair batch artifacts."""

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

VERSION = 426
PRIOR_POST_STREAMLIT_REPAIR_PYTEST_VERSION = 425
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_SCRIPTS_B905_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    "scripts/papers",
    "--select",
    "B905",
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
    "B905",
    "--fix",
    "--unsafe-fixes",
]
NEXT_ARTIFACT = "paper4_v427_post_scripts_ruff_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v426_targeted_scripts_ruff_repair_batch.md"


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


def _changed_script_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "scripts/papers"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _script_diff_clean() -> bool:
    return not _changed_script_files()


def _run_ruff_fix() -> dict[str, Any]:
    result = subprocess.run(
        RUFF_FIX_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "scripts B905 ruff fix failed")
    return {
        "command": " ".join(RUFF_FIX_COMMAND),
        "exit_code": result.returncode,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


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
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "compiled_files": len(paths),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


def _actions(before_scripts_b905: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in before_scripts_b905:
        fix = item.get("fix") or {}
        edits = fix.get("edits") or []
        rows.append(
            {
                "action_id_v426": f"scripts_b905_strict_false_{len(rows) + 1:02d}",
                "file_path_v426": _relative_path(str(item["filename"])),
                "row_v426": int((item.get("location") or {}).get("row") or 0),
                "rule_code_v426": str(item["code"]),
                "message_v426": str(item["message"]),
                "fix_message_v426": str(fix.get("message") or ""),
                "fix_edit_count_v426": int(len(edits)),
                "mutation_applied_v426": True,
                "claim_boundary_v426": "adds explicit strict=False to preserve default zip behavior",
            }
        )
    return pd.DataFrame(rows)


def _snapshot_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    rule_counts = Counter(str(item["code"]) for item in items)
    surface_counts = Counter(_surface(_relative_path(str(item["filename"]))) for item in items)
    return {
        "repository_total": int(len(items)),
        "repository_b905": int(rule_counts.get("B905", 0)),
        "repository_e402": int(rule_counts.get("E402", 0)),
        "repository_c408": int(rule_counts.get("C408", 0)),
        "scripts_total": int(surface_counts.get("scripts", 0)),
        "streamlit_app_total": int(surface_counts.get("streamlit_app", 0)),
        "book_total": int(surface_counts.get("book", 0)),
        "notebook_total": int(surface_counts.get("notebook", 0)),
    }


def _delta_table(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before = _snapshot_counts(before_items)
    after = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v426": metric,
                "before_v426": int(before[metric]),
                "after_v426": int(after[metric]),
                "delta_v426": int(after[metric] - before[metric]),
                "claim_boundary_v426": "ruff-count delta only; repository ruff remains open unless after=0",
            }
            for metric in [
                "repository_total",
                "repository_b905",
                "repository_e402",
                "repository_c408",
                "scripts_total",
                "streamlit_app_total",
                "book_total",
                "notebook_total",
            ]
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v426": metric,
                "diagnostic_count_v426": int(count),
                "claim_boundary_v426": "post-v426 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


def _pycompile_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compile_id_v426": "changed_scripts_py_compile",
                "command_v426": result["command"],
                "exit_code_v426": int(result["exit_code"]),
                "passed_v426": bool(result["passed"]),
                "compiled_files_v426": int(result["compiled_files"]),
                "stdout_tail_v426": str(result["stdout_tail"]),
                "stderr_tail_v426": str(result["stderr_tail"]),
                "claim_boundary_v426": "syntax/bytecode check only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, pycompile_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v426": "repository_ruff_frontier_still_open",
            "blocking_v426": after_total > 0,
            "evidence_count_v426": after_total,
            "required_next_artifact_v426": NEXT_ARTIFACT,
            "claim_boundary_v426": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v426": "full_repository_pytest_deferred_after_scripts_repair",
            "blocking_v426": True,
            "evidence_count_v426": 1,
            "required_next_artifact_v426": NEXT_ARTIFACT,
            "claim_boundary_v426": "py_compile does not replace full pytest",
        },
        {
            "blocker_id_v426": "paper4_final_promotion_forbidden",
            "blocking_v426": True,
            "evidence_count_v426": 1,
            "required_next_artifact_v426": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v426": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pycompile_passed:
        rows.insert(
            0,
            {
                "blocker_id_v426": "changed_scripts_pycompile_failed",
                "blocking_v426": True,
                "evidence_count_v426": 1,
                "required_next_artifact_v426": "paper4_v427_scripts_pycompile_failure_triage.md",
                "claim_boundary_v426": "compile failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, after_total: int, repo_b905_after: int, scripts_b905_after: int, pycompile_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v426_scripts_b905_repair_applied",
                "allowed": True,
                "artifact": "paper4_v426_scripts_b905_actions.csv",
                "boundary": "adds explicit strict=False to scripts/papers zip calls",
            },
            {
                "claim_id": "v426_scripts_b905_cleared",
                "allowed": scripts_b905_after == 0,
                "artifact": "paper4_v426_repository_ruff_delta.csv",
                "boundary": "true only for scripts/papers B905",
            },
            {
                "claim_id": "v426_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v426_repository_ruff_delta.csv",
                "boundary": "repository ruff count decreases",
            },
            {
                "claim_id": "v426_changed_scripts_pycompile_passed",
                "allowed": pycompile_passed,
                "artifact": "paper4_v426_pycompile_summary.csv",
                "boundary": "syntax/bytecode check for changed scripts",
            },
            {
                "claim_id": "v426_repository_b905_cleared",
                "allowed": repo_b905_after == 0,
                "artifact": "paper4_v426_repository_ruff_after_snapshot.csv",
                "boundary": "false while Streamlit B905 remains",
            },
            {
                "claim_id": "v426_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v426_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v426_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v426_claim_blockers.csv",
                "boundary": "full pytest deferred to v427",
            },
            {
                "claim_id": "v426_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, after_total: int, scripts_b905_after: int, pycompile_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v426 clears scripts/papers B905 diagnostics with explicit strict=False.",
                "allowed": scripts_b905_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v426_repository_ruff_delta.csv",
                "boundary": "Scripts B905 only; Streamlit B905 remains.",
                "prohibited_claim_flag": scripts_b905_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v426 reduces repository ruff diagnostics after scripts repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v426_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff remains open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v426 changed scripts compile after B905 repair.",
                "allowed": pycompile_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v426_pycompile_summary.csv",
                "boundary": "py_compile only; full pytest deferred.",
                "prohibited_claim_flag": not pycompile_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v426 proves repository ruff clean or full pytest clean after scripts repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v426_claim_blockers.csv",
                "boundary": f"{after_total} repository ruff diagnostics remain and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v426 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v426_claim_blockers.csv",
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
                "executable_item": "v426 applies explicit strict=False to scripts/papers B905 zip calls.",
                "status": "targeted_scripts_b905_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v427 full repository pytest passes after scripts B905 repair",
                "last_wave": "v426",
                "execution_result": f"repo_ruff_reduced_57_to_{after_total}_scripts_b905_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v426")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], fix_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Targeted Scripts Ruff Repair Batch v426

Generated: {status["generated_at_utc"]}

v426 applies explicit `strict=False` to scripts/papers B905 `zip()` calls.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v426"]}` ->
  `{status["repo_ruff_total_after_v426"]}`.
- Repository B905 diagnostics: `{status["repo_ruff_b905_before_v426"]}` ->
  `{status["repo_ruff_b905_after_v426"]}`.
- Scripts diagnostics: `{status["scripts_diagnostics_before_v426"]}` ->
  `{status["scripts_diagnostics_after_v426"]}`.
- Scripts B905 after: `{status["scripts_b905_after_v426"]}`.
- Changed script files: `{status["changed_script_files_v426"]}`.
- py_compile passed: `{status["changed_scripts_pycompile_passed_v426"]}`.
- Ruff fix command: `{fix_result["command"]}`.

## Required Caveat

v426 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v426"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V426_TARGETED_SCRIPTS_RUFF_REPAIR_BATCH_START -->"
    end = "<!-- V426_TARGETED_SCRIPTS_RUFF_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v426: Targeted Scripts Ruff Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v426 applies explicit `strict=False` to scripts/papers B905 `zip()` calls.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v426"]}` ->
  `{status["repo_ruff_total_after_v426"]}`.
- Repository B905 before/after:
  `{status["repo_ruff_b905_before_v426"]}` ->
  `{status["repo_ruff_b905_after_v426"]}`.
- Scripts diagnostics before/after:
  `{status["scripts_diagnostics_before_v426"]}` ->
  `{status["scripts_diagnostics_after_v426"]}`.
- Scripts B905 after:
  `{status["scripts_b905_after_v426"]}`.
- Changed script files:
  `{status["changed_script_files_v426"]}`.
- py_compile passed:
  `{status["changed_scripts_pycompile_passed_v426"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v426"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v426"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v426"]}`.

### Interpretation

The scripts/papers B905 frontier is closed by making default `zip()` behavior
explicit. Remaining ruff diagnostics are Streamlit B905/C408, scripts B007/F841
B023/I001/F401/UP022/SIM108/C405/SIM223, and two book-helper diagnostics.

### Claim Impact

- Allowed: scripts B905 repair applied, repository ruff count reduced, changed
  scripts compile.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v426 in the living notebook. v427 should run the post-scripts-repair full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _script_diff_clean():
        raise RuntimeError("v426 expects clean scripts/papers diffs before mutation.")

    v425_status = json.loads((STATUS_DIR / "paper4_v425_status.json").read_text(encoding="utf-8"))
    if v425_status["next_artifact_v425"] != "paper4_v426_targeted_scripts_ruff_repair_batch.md":
        raise RuntimeError("v426 expects v425 to route to scripts ruff repair.")
    if int(v425_status["repo_ruff_total_v425"]) != 57:
        raise RuntimeError("v426 expects the v425 57-diagnostic frontier.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_scripts_b905 = _run_json_command(RUFF_SCRIPTS_B905_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 57:
        raise RuntimeError("v426 expected repository ruff to fail with 57 diagnostics before repair.")
    if len(before_scripts_b905) != 11:
        raise RuntimeError("v426 expected 11 scripts/papers B905 diagnostics before repair.")

    before_counts = _snapshot_counts(before_repo_items)
    actions = _actions(before_scripts_b905)
    fix_result = _run_ruff_fix()
    changed_files = _changed_script_files()
    _, after_scripts_b905 = _run_json_command(RUFF_SCRIPTS_B905_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    pycompile_result = _run_pycompile(changed_files)
    delta = _delta_table(before_repo_items, after_repo_items)
    after_snapshot = _after_snapshot(after_repo_items)

    scripts_b905_after = int(len(after_scripts_b905))
    pycompile_passed = bool(pycompile_result["passed"])
    blockers = _claim_blockers(after_total=len(after_repo_items), pycompile_passed=pycompile_passed)
    claim_matrix = _claim_matrix(
        after_total=len(after_repo_items),
        repo_b905_after=after_counts["repository_b905"],
        scripts_b905_after=scripts_b905_after,
        pycompile_passed=pycompile_passed,
    )
    _update_claim_boundaries(
        after_total=len(after_repo_items),
        scripts_b905_after=scripts_b905_after,
        pycompile_passed=pycompile_passed,
    )
    _update_backlog(len(after_repo_items))

    write_csv(TABLE_DIR / "paper4_v426_scripts_b905_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v426_repository_ruff_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v426_repository_ruff_after_snapshot.csv", after_snapshot)
    write_csv(TABLE_DIR / "paper4_v426_pycompile_summary.csv", _pycompile_summary(pycompile_result))
    write_csv(TABLE_DIR / "paper4_v426_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v426_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v426_targeted_scripts_ruff_repair_batch",
        "schema_version": "2026-05-17.426",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_streamlit_repair_pytest_version_v426": PRIOR_POST_STREAMLIT_REPAIR_PYTEST_VERSION,
        "actions_v426": int(len(actions)),
        "changed_script_files_v426": int(len(changed_files)),
        "changed_script_file_list_v426": changed_files,
        "repo_ruff_exit_code_before_v426": int(before_repo_exit),
        "repo_ruff_exit_code_after_v426": int(after_repo_exit),
        "repo_ruff_total_before_v426": before_counts["repository_total"],
        "repo_ruff_total_after_v426": after_counts["repository_total"],
        "repo_ruff_total_reduced_v426": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_b905_before_v426": before_counts["repository_b905"],
        "repo_ruff_b905_after_v426": after_counts["repository_b905"],
        "repo_ruff_b905_reduced_v426": before_counts["repository_b905"] - after_counts["repository_b905"],
        "scripts_diagnostics_before_v426": before_counts["scripts_total"],
        "scripts_diagnostics_after_v426": after_counts["scripts_total"],
        "scripts_b905_before_v426": int(len(before_scripts_b905)),
        "scripts_b905_after_v426": scripts_b905_after,
        "streamlit_diagnostics_after_v426": after_counts["streamlit_app_total"],
        "book_diagnostics_after_v426": after_counts["book_total"],
        "notebook_diagnostics_after_v426": after_counts["notebook_total"],
        "changed_scripts_pycompile_run_v426": True,
        "changed_scripts_pycompile_passed_v426": pycompile_passed,
        "changed_scripts_pycompile_files_v426": int(pycompile_result["compiled_files"]),
        "repository_ruff_clean_v426": after_counts["repository_total"] == 0,
        "full_repository_pytest_run_v426": False,
        "full_quarto_render_run_v426": False,
        "working_champion_claim_allowed_v426": False,
        "paper1_promotion_allowed_v426": False,
        "paper4_working_champion_changed_v426": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v426": NEXT_ARTIFACT,
        "claim_boundary": (
            "v426 applies a targeted scripts/papers B905 repair; repository ruff "
            "and full pytest/final promotion claims remain blocked"
        ),
    }
    if scripts_b905_after != 0:
        raise RuntimeError("v426 expected scripts/papers B905 to be cleared.")
    if status["repo_ruff_total_after_v426"] != 46:
        raise RuntimeError("v426 expected repository ruff to contract to 46 diagnostics.")
    if status["repo_ruff_b905_after_v426"] != 3:
        raise RuntimeError("v426 expected three repository B905 diagnostics to remain in Streamlit.")
    if not pycompile_passed:
        raise RuntimeError("v426 changed scripts did not compile.")

    REPAIR_MD.write_text(_repair_markdown(status, fix_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v426": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
