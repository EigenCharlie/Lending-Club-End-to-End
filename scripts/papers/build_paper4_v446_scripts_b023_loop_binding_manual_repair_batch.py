#!/usr/bin/env python3
"""Build Paper 4 v446 manual B023 loop-binding repair artifacts."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
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

VERSION = 446
PRIOR_POST_SIM223_REPAIR_PYTEST_VERSION = 445
NEXT_ARTIFACT = "paper4_v447_post_scripts_b023_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v446_scripts_b023_loop_binding_manual_repair_batch.md"
RUFF_REPO_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
RUFF_B023_COMMAND = [
    "uv",
    "run",
    "ruff",
    "check",
    "scripts/papers",
    "--select",
    "B023",
    "--output-format",
    "json",
]
TARGET_FILES = [
    "scripts/papers/build_paper4_v10_resolution_wave.py",
    "scripts/papers/build_paper4_v11_promising_lanes.py",
    "scripts/papers/build_paper4_v12_resolution_wave.py",
    "scripts/papers/build_paper4_v41_v44_living_lab_wave.py",
]
COUNT_CODES = [
    "B023",
    "SIM223",
    "C405",
    "UP018",
    "SIM108",
    "UP022",
    "F401",
    "I001",
    "F841",
    "B007",
    "B905",
    "C408",
]


@dataclass(frozen=True)
class ManualReplacement:
    action_id: str
    file_path: str
    row: int
    captured_variables: str
    diagnostics_cleared: int
    old: str
    new: str


REPLACEMENTS = [
    ManualReplacement(
        action_id="scripts_b023_v10_path_id_lambda_bind_01",
        file_path="scripts/papers/build_paper4_v10_resolution_wave.py",
        row=1138,
        captured_variables="path_id",
        diagnostics_cleared=1,
        old='.map(lambda m: 1 + 0.12 * math.sin(2 * math.pi * _stable_uniform(path_id, m, "cohort")))',
        new=(
            ".map(\n"
            "                lambda m, path_id=path_id: 1\n"
            "                + 0.12 * math.sin(2 * math.pi * _stable_uniform(path_id, m, \"cohort\"))\n"
            "            )"
        ),
    ),
    ManualReplacement(
        action_id="scripts_b023_v11_path_policy_lambda_bind_01",
        file_path="scripts/papers/build_paper4_v11_promising_lanes.py",
        row=936,
        captured_variables="path_id,policy_id",
        diagnostics_cleared=2,
        old="lambda m: (\n                        1\n                        + 0.10",
        new=(
            "lambda m, path_id=path_id, policy_id=policy_id: (\n"
            "                        1\n"
            "                        + 0.10"
        ),
    ),
    ManualReplacement(
        action_id="scripts_b023_v12_path_policy_lambda_bind_01",
        file_path="scripts/papers/build_paper4_v12_resolution_wave.py",
        row=1514,
        captured_variables="path_id,policy_id",
        diagnostics_cleared=2,
        old="lambda m: (\n                        1\n                        + 0.16",
        new=(
            "lambda m, path_id=path_id, policy_id=policy_id: (\n"
            "                        1\n"
            "                        + 0.16"
        ),
    ),
    ManualReplacement(
        action_id="scripts_b023_v41_total_exposure_lambda_bind_01",
        file_path="scripts/papers/build_paper4_v41_v44_living_lab_wave.py",
        row=319,
        captured_variables="total_exposure",
        diagnostics_cleared=2,
        old=(
            "        group = group.assign(\n"
            "            exposure_share=lambda d: d[\"exposure\"] / total_exposure if total_exposure else 0.0\n"
            "        )"
        ),
        new=(
            "        group = group.assign(\n"
            "            exposure_share=lambda d, total_exposure=total_exposure: (\n"
            "                d[\"exposure\"] / total_exposure if total_exposure else 0.0\n"
            "            )\n"
            "        )"
        ),
    ),
]


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


def _snapshot_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    rule_counts = Counter(str(item["code"]) for item in items)
    surface_counts = Counter(_surface(_relative_path(str(item["filename"]))) for item in items)
    counts = {"repository_total": int(len(items))}
    for code in COUNT_CODES:
        counts[f"repository_{code.lower()}"] = int(rule_counts.get(code, 0))
    counts["scripts_total"] = int(surface_counts.get("scripts", 0))
    counts["book_total"] = int(surface_counts.get("book", 0))
    counts["streamlit_app_total"] = int(surface_counts.get("streamlit_app", 0))
    counts["notebook_total"] = int(surface_counts.get("notebook", 0))
    return counts


def _apply_replacements() -> pd.DataFrame:
    rows = []
    for replacement in REPLACEMENTS:
        path = ROOT / replacement.file_path
        text = path.read_text(encoding="utf-8")
        if replacement.old not in text:
            raise RuntimeError(f"v446 replacement anchor missing: {replacement.action_id}")
        path.write_text(text.replace(replacement.old, replacement.new, 1), encoding="utf-8")
        rows.append(
            {
                "action_id_v446": replacement.action_id,
                "file_path_v446": replacement.file_path,
                "surface_v446": _surface(replacement.file_path),
                "row_v446": replacement.row,
                "rule_code_v446": "B023",
                "captured_variables_v446": replacement.captured_variables,
                "diagnostics_cleared_v446": replacement.diagnostics_cleared,
                "mutation_applied_v446": True,
                "claim_boundary_v446": "manual lambda default binding for loop variables",
            }
        )
    return pd.DataFrame(rows)


def _ruff_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
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
                "metric_v446": metric,
                "before_v446": int(before[metric]),
                "after_v446": int(after[metric]),
                "delta_v446": int(after[metric] - before[metric]),
                "claim_boundary_v446": "ruff-count delta only; pytest deferred to v447",
            }
            for metric in metrics
        ]
    )


def _after_snapshot(after_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = _snapshot_counts(after_items)
    return pd.DataFrame(
        [
            {
                "metric_v446": metric,
                "diagnostic_count_v446": int(count),
                "claim_boundary_v446": "post-v446 repository ruff snapshot",
            }
            for metric, count in counts.items()
        ]
    )


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


def _pycompile_summary(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compile_id_v446": "changed_scripts_py_compile",
                "command_v446": result["command"],
                "exit_code_v446": int(result["exit_code"]),
                "passed_v446": bool(result["passed"]),
                "compiled_files_v446": int(result["compiled_files"]),
                "stdout_tail_v446": str(result["stdout_tail"]),
                "stderr_tail_v446": str(result["stderr_tail"]),
                "claim_boundary_v446": "syntax/bytecode check only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, pycompile_passed: bool, repo_total_after: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v446": "full_repository_pytest_deferred_after_b023_repair",
            "blocking_v446": True,
            "evidence_count_v446": 1,
            "required_next_artifact_v446": NEXT_ARTIFACT,
            "claim_boundary_v446": "py_compile and ruff clean do not replace full pytest",
        },
        {
            "blocker_id_v446": "quarto_render_not_run",
            "blocking_v446": True,
            "evidence_count_v446": 1,
            "required_next_artifact_v446": NEXT_ARTIFACT,
            "claim_boundary_v446": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v446": "paper4_final_promotion_forbidden",
            "blocking_v446": True,
            "evidence_count_v446": 1,
            "required_next_artifact_v446": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v446": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if repo_total_after > 0:
        rows.insert(
            0,
            {
                "blocker_id_v446": "repository_ruff_frontier_still_open",
                "blocking_v446": True,
                "evidence_count_v446": repo_total_after,
                "required_next_artifact_v446": NEXT_ARTIFACT,
                "claim_boundary_v446": "repository ruff clean claim blocked while diagnostics remain",
            },
        )
    if not pycompile_passed:
        rows.insert(
            0,
            {
                "blocker_id_v446": "changed_scripts_pycompile_failed",
                "blocking_v446": True,
                "evidence_count_v446": 1,
                "required_next_artifact_v446": "paper4_v447_b023_pycompile_failure_triage.md",
                "claim_boundary_v446": "compile failure must be triaged before pytest",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pycompile_passed: bool, repo_total_after: int, b023_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v446_b023_loop_binding_repair_applied",
                "allowed": True,
                "artifact": "paper4_v446_scripts_b023_actions.csv",
                "boundary": "manual lambda default binding only",
            },
            {
                "claim_id": "v446_b023_cleared",
                "allowed": b023_after == 0,
                "artifact": "paper4_v446_repository_ruff_delta.csv",
                "boundary": "true only for B023 diagnostics",
            },
            {
                "claim_id": "v446_repository_ruff_clean",
                "allowed": repo_total_after == 0,
                "artifact": "paper4_v446_repository_ruff_after_snapshot.csv",
                "boundary": "ruff clean only; pytest still deferred",
            },
            {
                "claim_id": "v446_changed_scripts_pycompile_passed",
                "allowed": pycompile_passed,
                "artifact": "paper4_v446_pycompile_summary.csv",
                "boundary": "syntax/bytecode check for changed scripts",
            },
            {
                "claim_id": "v446_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v446_claim_blockers.csv",
                "boundary": "full pytest deferred to v447",
            },
            {
                "claim_id": "v446_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *, repo_total_after: int, b023_after: int, pycompile_passed: bool
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v446 clears manual scripts B023 loop-binding diagnostics.",
                "allowed": b023_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v446_repository_ruff_delta.csv",
                "boundary": "B023 only; semantics still require full pytest.",
                "prohibited_claim_flag": b023_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v446 makes repository ruff clean after B023 repair.",
                "allowed": repo_total_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v446_repository_ruff_after_snapshot.csv",
                "boundary": "Ruff clean only; Quarto and full pytest remain separate gates.",
                "prohibited_claim_flag": repo_total_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v446 changed scripts compile after B023 repair.",
                "allowed": pycompile_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v446_pycompile_summary.csv",
                "boundary": "py_compile only; full pytest deferred.",
                "prohibited_claim_flag": not pycompile_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v446 proves full pytest or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v446_claim_blockers.csv",
                "boundary": "Full pytest and Quarto render are deferred to later waves.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v446 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v446_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(repo_total_after: int) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v446 applies manual scripts B023 loop-binding repair.",
                "status": "manual_scripts_b023_loop_binding_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v447 full repository pytest passes after manual B023 repair",
                "last_wave": "v446",
                "execution_result": f"repo_ruff_reduced_7_to_{repo_total_after}_b023_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v446")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Scripts B023 Loop-Binding Manual Repair Batch v446

Generated: {status["generated_at_utc"]}

v446 applies four manual lambda default-binding repairs that clear seven B023
diagnostics.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v446"]}` ->
  `{status["repo_ruff_total_after_v446"]}`.
- Repository B023 diagnostics: `{status["repo_ruff_b023_before_v446"]}` ->
  `{status["repo_ruff_b023_after_v446"]}`.
- Changed files: `{status["changed_files_v446"]}`.
- py_compile passed: `{status["changed_scripts_pycompile_passed_v446"]}`.
- Repository ruff clean: `{status["repository_ruff_clean_v446"]}`.

## Required Caveat

v446 does not claim full pytest clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v446"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V446_SCRIPTS_B023_LOOP_BINDING_MANUAL_REPAIR_BATCH_START -->"
    end = "<!-- V446_SCRIPTS_B023_LOOP_BINDING_MANUAL_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v446: Scripts B023 Loop-Binding Manual Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v446 applies manual lambda default-binding repairs across four scripts.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v446"]}` ->
  `{status["repo_ruff_total_after_v446"]}`.
- Repository B023 diagnostics before/after:
  `{status["repo_ruff_b023_before_v446"]}` ->
  `{status["repo_ruff_b023_after_v446"]}`.
- Replacement actions:
  `{status["actions_v446"]}`.
- Diagnostics cleared by action table:
  `{status["diagnostics_cleared_v446"]}`.
- py_compile passed:
  `{status["changed_scripts_pycompile_passed_v446"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v446"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v446"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v446"]}`.

### Interpretation

The manual B023 frontier is cleared without introducing new lint diagnostics.
This is the first living-lab state where repository ruff is clean, but full
pytest and Quarto render are still explicit future gates.

### Claim Impact

- Allowed: manual B023 repair applied, repository ruff clean, changed scripts
  compile.
- Still prohibited: full pytest clean after B023 repair, Quarto render clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v446 in the living notebook. v447 should run the post-B023-repair full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _target_diff_clean():
        raise RuntimeError("v446 expects clean target diffs before manual repair.")

    v445_status = json.loads((STATUS_DIR / "paper4_v445_status.json").read_text(encoding="utf-8"))
    if v445_status["next_artifact_v445"] != "paper4_v446_scripts_b023_loop_binding_manual_repair_batch.md":
        raise RuntimeError("v446 expects v445 to route to manual B023 repair.")
    if v445_status["full_repository_pytest_passed_v445"] is not True:
        raise RuntimeError("v446 expects v445 full pytest to pass.")
    if int(v445_status["repo_ruff_b023_v445"]) != 7:
        raise RuntimeError("v446 expects seven B023 diagnostics before repair.")

    before_repo_exit, before_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    _, before_b023_items = _run_json_command(RUFF_B023_COMMAND)
    if before_repo_exit != 1 or len(before_repo_items) != 7 or len(before_b023_items) != 7:
        raise RuntimeError("v446 expected exactly seven repository B023 diagnostics before repair.")

    before_counts = _snapshot_counts(before_repo_items)
    actions = _apply_replacements()
    changed_files = _changed_target_files()
    _, after_b023_items = _run_json_command(RUFF_B023_COMMAND)
    after_repo_exit, after_repo_items = _run_json_command(RUFF_REPO_COMMAND)
    after_counts = _snapshot_counts(after_repo_items)
    pycompile_result = _run_pycompile(changed_files)
    pycompile_passed = bool(pycompile_result["passed"])
    b023_after = int(len(after_b023_items))

    write_csv(TABLE_DIR / "paper4_v446_scripts_b023_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v446_repository_ruff_delta.csv", _ruff_delta(before_repo_items, after_repo_items))
    write_csv(TABLE_DIR / "paper4_v446_repository_ruff_after_snapshot.csv", _after_snapshot(after_repo_items))
    write_csv(TABLE_DIR / "paper4_v446_pycompile_summary.csv", _pycompile_summary(pycompile_result))
    write_csv(
        TABLE_DIR / "paper4_v446_claim_blockers.csv",
        _claim_blockers(pycompile_passed=pycompile_passed, repo_total_after=len(after_repo_items)),
    )
    write_csv(
        TABLE_DIR / "paper4_v446_claim_matrix_delta.csv",
        _claim_matrix(
            pycompile_passed=pycompile_passed,
            repo_total_after=len(after_repo_items),
            b023_after=b023_after,
        ),
    )
    _update_claim_boundaries(
        repo_total_after=len(after_repo_items),
        b023_after=b023_after,
        pycompile_passed=pycompile_passed,
    )
    _update_backlog(len(after_repo_items))

    diagnostics_cleared = int(actions["diagnostics_cleared_v446"].sum())
    status = {
        "phase": "v446_scripts_b023_loop_binding_manual_repair_batch",
        "schema_version": "2026-05-17.446",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_sim223_repair_pytest_version_v446": PRIOR_POST_SIM223_REPAIR_PYTEST_VERSION,
        "actions_v446": int(len(actions)),
        "diagnostics_cleared_v446": diagnostics_cleared,
        "repo_ruff_exit_code_before_v446": int(before_repo_exit),
        "repo_ruff_exit_code_after_v446": int(after_repo_exit),
        "repo_ruff_total_before_v446": before_counts["repository_total"],
        "repo_ruff_total_after_v446": after_counts["repository_total"],
        "repo_ruff_total_reduced_v446": before_counts["repository_total"] - after_counts["repository_total"],
        "repo_ruff_b023_before_v446": before_counts["repository_b023"],
        "repo_ruff_b023_after_v446": after_counts["repository_b023"],
        "repo_ruff_b023_reduced_v446": before_counts["repository_b023"] - after_counts["repository_b023"],
        "repo_ruff_sim223_after_v446": after_counts["repository_sim223"],
        "repo_ruff_c405_after_v446": after_counts["repository_c405"],
        "repo_ruff_up018_after_v446": after_counts["repository_up018"],
        "repo_ruff_sim108_after_v446": after_counts["repository_sim108"],
        "repo_ruff_up022_after_v446": after_counts["repository_up022"],
        "repo_ruff_f401_after_v446": after_counts["repository_f401"],
        "repo_ruff_i001_after_v446": after_counts["repository_i001"],
        "repo_ruff_f841_after_v446": after_counts["repository_f841"],
        "repo_ruff_b007_after_v446": after_counts["repository_b007"],
        "repo_ruff_b905_after_v446": after_counts["repository_b905"],
        "repo_ruff_c408_after_v446": after_counts["repository_c408"],
        "scripts_diagnostics_before_v446": before_counts["scripts_total"],
        "scripts_diagnostics_after_v446": after_counts["scripts_total"],
        "book_diagnostics_after_v446": after_counts["book_total"],
        "streamlit_diagnostics_after_v446": after_counts["streamlit_app_total"],
        "notebook_diagnostics_after_v446": after_counts["notebook_total"],
        "changed_files_v446": int(len(changed_files)),
        "changed_file_list_v446": changed_files,
        "changed_scripts_pycompile_run_v446": True,
        "changed_scripts_pycompile_passed_v446": pycompile_passed,
        "changed_scripts_pycompile_files_v446": int(pycompile_result["compiled_files"]),
        "repository_ruff_clean_v446": after_counts["repository_total"] == 0,
        "full_repository_pytest_run_v446": False,
        "full_quarto_render_run_v446": False,
        "working_champion_claim_allowed_v446": False,
        "paper1_promotion_allowed_v446": False,
        "paper4_working_champion_changed_v446": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v446": NEXT_ARTIFACT,
        "claim_boundary": (
            "v446 clears manual B023 loop-binding diagnostics; full pytest "
            "and final promotion claims remain blocked"
        ),
    }
    if b023_after != 0 or status["repo_ruff_total_after_v446"] != 0:
        raise RuntimeError("v446 expected repository ruff to be clean after B023 repair.")
    if diagnostics_cleared != 7:
        raise RuntimeError("v446 expected the action table to clear seven B023 diagnostics.")
    if not pycompile_passed:
        raise RuntimeError("v446 changed scripts did not compile.")

    REPAIR_MD.write_text(_repair_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v446": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
