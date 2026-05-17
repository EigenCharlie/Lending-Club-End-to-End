#!/usr/bin/env python3
"""Build Paper 4 v401 notebook E402 setup warning-filter plan artifacts."""

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

VERSION = 401
PRIOR_E402_PLAN_VERSION = 399
PRIOR_LOCAL_HOIST_VERSION = 400
NEXT_VERSION = 402
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_warning_filter_only_reorder_batch.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v401_notebook_e402_setup_warning_refactor_plan.md"
SOURCE_PLAN = TABLE_DIR / "paper4_v399_notebook_e402_cell_refactor_plan.csv"


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


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _source_lines(notebook_path: str, cell_number: int) -> list[str]:
    payload = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
    return list(payload["cells"][cell_number - 1].get("source", []))


def _is_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def _is_warning_filter_line(line: str) -> bool:
    return "warnings.filterwarnings" in line


def _is_sys_path_insert_line(line: str) -> bool:
    return "sys.path.insert" in line


def _is_project_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("from src.") or stripped.startswith("import src")


def _current_e402_cell_counts(items: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, int]]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for item in items:
        key = (_relative_path(str(item["filename"])), int(item.get("cell") or 0))
        location = item.get("location") or {}
        grouped.setdefault(key, []).append(int(location.get("row") or 0))
    return {
        key: {
            "current_e402_diagnostic_count": len(rows),
            "first_e402_row": min(rows),
            "last_e402_row": max(rows),
        }
        for key, rows in grouped.items()
    }


def _setup_refactor_class(lines: list[str]) -> tuple[str, str, str, str]:
    sys_path_count = sum(1 for line in lines if _is_sys_path_insert_line(line))
    if sys_path_count:
        return (
            "sys_path_project_import_cell",
            "batch_2_sys_path_project_import_refactor",
            "medium_path_semantics",
            "validate project import path before moving sys.path or project imports",
        )
    return (
        "warning_filter_only_cell",
        "batch_1_warning_filter_only_reorder",
        "low_to_medium_warning_filter_timing",
        "move warning filters below the import block while preserving filter order",
    )


def _plan_rows(source_plan: pd.DataFrame, current_e402: list[dict[str, Any]]) -> pd.DataFrame:
    setup = source_plan.loc[
        source_plan["planned_batch_v399"].eq("batch_2_setup_warning_filter_refactor")
    ].copy()
    current_counts = _current_e402_cell_counts(current_e402)
    rows = []
    for idx, row in enumerate(setup.itertuples(index=False), start=1):
        notebook_path = str(row.notebook_path_v399)
        cell_number = int(row.cell_v399)
        lines = _source_lines(notebook_path, cell_number)
        cell_key = (notebook_path, cell_number)
        counts = current_counts[cell_key]
        refactor_class, batch, risk, action = _setup_refactor_class(lines)
        rows.append(
            {
                "plan_id_v401": f"setup_warning_plan_{idx:02d}",
                "notebook_path_v401": notebook_path,
                "cell_v401": cell_number,
                "current_e402_diagnostic_count_v401": counts["current_e402_diagnostic_count"],
                "first_e402_row_v401": counts["first_e402_row"],
                "last_e402_row_v401": counts["last_e402_row"],
                "warning_filter_line_count_v401": sum(1 for line in lines if _is_warning_filter_line(line)),
                "sys_path_insert_line_count_v401": sum(1 for line in lines if _is_sys_path_insert_line(line)),
                "project_import_line_count_v401": sum(1 for line in lines if _is_project_import_line(line)),
                "import_line_count_v401": sum(1 for line in lines if _is_import_line(line)),
                "setup_refactor_class_v401": refactor_class,
                "planned_batch_v401": batch,
                "risk_class_v401": risk,
                "recommended_action_v401": action,
                "mutation_allowed_v401": False,
                "claim_boundary_v401": "plan only; no notebook mutation in v401",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(plan: pd.DataFrame) -> pd.DataFrame:
    order = {
        "batch_1_warning_filter_only_reorder": 1,
        "batch_2_sys_path_project_import_refactor": 2,
    }
    rows = []
    for batch, group in plan.groupby("planned_batch_v401", sort=False):
        rows.append(
            {
                "batch_id_v401": batch,
                "execution_order_v401": order.get(batch, 99),
                "cell_count_v401": int(len(group)),
                "e402_diagnostic_count_v401": int(
                    group["current_e402_diagnostic_count_v401"].sum()
                ),
                "risk_class_v401": (
                    "lowest_remaining"
                    if batch == "batch_1_warning_filter_only_reorder"
                    else "requires_project_import_path_review"
                ),
                "mutation_allowed_v401": False,
                "next_action_v401": (
                    NEXT_ARTIFACT
                    if batch == "batch_1_warning_filter_only_reorder"
                    else "paper4_v403_notebook_sys_path_project_import_refactor_plan.md"
                ),
                "claim_boundary_v401": "batch plan only; no notebook mutation in v401",
            }
        )
    return pd.DataFrame(rows).sort_values("execution_order_v401", ignore_index=True)


def _claim_blockers(plan: pd.DataFrame, *, global_after_v400: int) -> pd.DataFrame:
    warning_only = plan.loc[plan["planned_batch_v401"].eq("batch_1_warning_filter_only_reorder")]
    sys_path = plan.loc[plan["planned_batch_v401"].eq("batch_2_sys_path_project_import_refactor")]
    return pd.DataFrame(
        [
            {
                "blocker_id_v401": "warning_filter_only_batch_not_applied_yet",
                "blocking_v401": True,
                "evidence_count_v401": int(warning_only["current_e402_diagnostic_count_v401"].sum()),
                "required_next_artifact_v401": NEXT_ARTIFACT,
                "claim_boundary_v401": "v401 selects warning-filter-only reorder but does not mutate notebooks",
            },
            {
                "blocker_id_v401": "sys_path_project_import_cells_deferred",
                "blocking_v401": True,
                "evidence_count_v401": int(sys_path["current_e402_diagnostic_count_v401"].sum()),
                "required_next_artifact_v401": "paper4_v403_notebook_sys_path_project_import_refactor_plan.md",
                "claim_boundary_v401": "project import path semantics need a separate review",
            },
            {
                "blocker_id_v401": "global_notebook_lint_not_clean",
                "blocking_v401": True,
                "evidence_count_v401": global_after_v400,
                "required_next_artifact_v401": NEXT_ARTIFACT,
                "claim_boundary_v401": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v401": "full_repository_pytest_not_rerun",
                "blocking_v401": True,
                "evidence_count_v401": 1,
                "required_next_artifact_v401": "paper4_v403_post_notebook_mutation_pytest_probe.md",
                "claim_boundary_v401": "focused validation only; full pytest deferred until after mutation",
            },
            {
                "blocker_id_v401": "paper4_final_promotion_forbidden",
                "blocking_v401": True,
                "evidence_count_v401": 1,
                "required_next_artifact_v401": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v401": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v401_setup_warning_refactor_plan_created",
                "allowed": True,
                "artifact": "paper4_v401_notebook_warning_filter_refactor_plan.csv",
                "boundary": "9 remaining setup warning-filter cells planned",
            },
            {
                "claim_id": "v401_warning_filter_only_batch_selected",
                "allowed": True,
                "artifact": "paper4_v401_notebook_warning_filter_batch_plan.csv",
                "boundary": "6 cells selected for v402",
            },
            {
                "claim_id": "v401_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v401",
            },
            {
                "claim_id": "v401_e402_or_global_lint_clean",
                "allowed": False,
                "artifact": "paper4_v401_claim_blockers.csv",
                "boundary": "112 E402 and 132 notebook diagnostics remain",
            },
            {
                "claim_id": "v401_sys_path_refactor_applied",
                "allowed": False,
                "artifact": "paper4_v401_claim_blockers.csv",
                "boundary": "sys.path cells deferred",
            },
            {
                "claim_id": "v401_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v401 plans all 9 remaining setup warning-filter E402 cells.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v401_notebook_warning_filter_refactor_plan.csv"
                ),
                "boundary": "Plan only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v401 selects a 6-cell warning-filter-only reorder batch for v402.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v401_notebook_warning_filter_batch_plan.csv"
                ),
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v401 repairs E402 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v401_claim_blockers.csv",
                "boundary": "No notebook mutation; 112 E402 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v401 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v401_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": (
                    "v401 plans the remaining setup warning-filter E402 frontier and "
                    "selects the warning-filter-only reorder batch."
                ),
                "status": "notebook_setup_warning_refactor_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v402 applies only warning-filter-only reorder cells with roundtrip checks",
                "last_wave": "v401",
                "execution_result": "setup_warning_e402_9_cells_planned_first_batch_6_cells_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v401")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook E402 Setup Warning-Filter Refactor Plan v401

Generated: {status["generated_at_utc"]}

v401 plans the remaining setup-cell E402 frontier after v400's local import
hoist batch. It does not mutate notebooks.

## Result

- Setup warning-filter cells planned: `{status["setup_warning_filter_cells_v401"]}`.
- Setup warning-filter E402 diagnostics: `{status["setup_warning_filter_e402_diagnostics_v401"]}`.
- Warning-filter-only first batch cells: `{status["warning_filter_only_cells_v401"]}`.
- Warning-filter-only first batch diagnostics: `{status["warning_filter_only_e402_diagnostics_v401"]}`.
- Sys.path/project-import cells deferred: `{status["sys_path_project_import_cells_v401"]}`.
- Sys.path/project-import diagnostics deferred: `{status["sys_path_project_import_e402_diagnostics_v401"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v401"]}`.

## Required Caveat

v401 does not repair E402, does not make notebook or repository ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v401"]}` for the warning-filter-only setup cells.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V401_NOTEBOOK_E402_SETUP_WARNING_REFACTOR_PLAN_START -->"
    end = "<!-- V401_NOTEBOOK_E402_SETUP_WARNING_REFACTOR_PLAN_END -->"
    block = f"""
{start}

## Wave v401: Notebook E402 Setup Warning-Filter Refactor Plan

Generated: {status["generated_at_utc"]}

### Objective

v401 plans the 9 remaining setup warning-filter E402 cells and separates the
lowest remaining mutation batch from cells that depend on project import path
semantics.

### Results

- Setup warning-filter cells:
  `{status["setup_warning_filter_cells_v401"]}`.
- Setup warning-filter E402 diagnostics:
  `{status["setup_warning_filter_e402_diagnostics_v401"]}`.
- Warning-filter-only first batch cells:
  `{status["warning_filter_only_cells_v401"]}`.
- Warning-filter-only first batch diagnostics:
  `{status["warning_filter_only_e402_diagnostics_v401"]}`.
- Sys.path/project-import cells deferred:
  `{status["sys_path_project_import_cells_v401"]}`.
- Sys.path/project-import diagnostics deferred:
  `{status["sys_path_project_import_e402_diagnostics_v401"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v401"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v401"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v401"]}`.

### Interpretation

The next safe mutation should target only the 6 warning-filter-only setup cells.
The 3 sys.path/project-import cells remain separate because moving imports across
path injection can change notebook execution semantics.

### Claim Impact

- Allowed: 9-cell setup warning-filter refactor plan and first-batch selection.
- Still prohibited: E402 repaired, notebook lint clean, repository ruff clean,
  sys.path refactor applied, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v401 in the living notebook. v402 should apply only the warning-filter-only
reorder batch with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v401 expects clean notebook diff because it is plan-only.")

    v400_status = json.loads((STATUS_DIR / "paper4_v400_status.json").read_text(encoding="utf-8"))
    if v400_status["next_artifact_v400"] != "paper4_v401_notebook_e402_setup_warning_refactor_plan.md":
        raise RuntimeError("v401 expects v400 to route to setup warning-filter refactor planning.")

    source_plan = pd.read_csv(SOURCE_PLAN)
    current_global = _run_ruff_json()
    current_e402 = _run_ruff_json(["E402"])
    before_notebook_diff = _notebook_diff_clean()
    plan = _plan_rows(source_plan, current_e402)
    batch_plan = _batch_plan(plan)
    blockers = _claim_blockers(plan, global_after_v400=len(current_global))
    claim_matrix = _claim_matrix()
    counts = Counter(current["code"] for current in current_global)

    warning_only = plan.loc[plan["planned_batch_v401"].eq("batch_1_warning_filter_only_reorder")]
    sys_path = plan.loc[plan["planned_batch_v401"].eq("batch_2_sys_path_project_import_refactor")]
    planned_e402 = int(plan["current_e402_diagnostic_count_v401"].sum())
    if planned_e402 != len(current_e402):
        raise RuntimeError("v401 setup plan does not cover all current E402 diagnostics.")

    write_csv(TABLE_DIR / "paper4_v401_notebook_warning_filter_refactor_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v401_notebook_warning_filter_batch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v401_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v401_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v401_notebook_e402_setup_warning_refactor_plan",
        "schema_version": "2026-05-17.401",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_e402_plan_version_v401": PRIOR_E402_PLAN_VERSION,
        "prior_local_hoist_version_v401": PRIOR_LOCAL_HOIST_VERSION,
        "setup_warning_filter_cells_v401": int(len(plan)),
        "setup_warning_filter_e402_diagnostics_v401": planned_e402,
        "batch_plan_rows_v401": int(len(batch_plan)),
        "claim_blocker_rows_v401": int(len(blockers)),
        "claim_matrix_rows_v401": int(len(claim_matrix)),
        "warning_filter_only_cells_v401": int(len(warning_only)),
        "warning_filter_only_e402_diagnostics_v401": int(
            warning_only["current_e402_diagnostic_count_v401"].sum()
        ),
        "sys_path_project_import_cells_v401": int(len(sys_path)),
        "sys_path_project_import_e402_diagnostics_v401": int(
            sys_path["current_e402_diagnostic_count_v401"].sum()
        ),
        "selected_first_batch_v401": "batch_1_warning_filter_only_reorder",
        "sys_path_project_import_batch_deferred_v401": True,
        "notebooks_mutated_v401": False,
        "notebook_diff_clean_before_v401": before_notebook_diff,
        "notebook_diff_clean_after_v401": _notebook_diff_clean(),
        "global_notebook_diagnostics_v401": int(len(current_global)),
        "global_notebook_e402_v401": int(counts.get("E402", 0)),
        "global_notebook_f821_v401": int(counts.get("F821", 0)),
        "global_ruff_clean_v401": False,
        "full_repository_pytest_run_v401": False,
        "full_quarto_render_run_v401": False,
        "working_champion_claim_allowed_v401": False,
        "paper1_promotion_allowed_v401": False,
        "paper4_working_champion_changed_v401": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v401": NEXT_ARTIFACT,
        "claim_boundary": (
            "v401 plans setup warning-filter E402 refactors and selects a first "
            "warning-filter-only batch; no notebooks are mutated and final "
            "promotion remains blocked"
        ),
    }
    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v401": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
