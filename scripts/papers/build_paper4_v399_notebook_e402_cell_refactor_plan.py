#!/usr/bin/env python3
"""Build Paper 4 v399 notebook E402 cell refactor plan artifacts."""

from __future__ import annotations

import json
import subprocess
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

VERSION = 399
PRIOR_E402_POLICY_VERSION = 398
NEXT_VERSION = 400
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_e402_local_import_hoist_batch.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v399_notebook_e402_cell_refactor_plan.md"
SOURCE_CELL_SUMMARY = TABLE_DIR / "paper4_v398_notebook_historical_e402_cell_summary.csv"


def _run_e402_json() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "notebooks", "--select", "E402", "--output-format", "json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff E402 probe failed")
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("ruff E402 JSON output is not a list")
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


def _source_lines(notebook_path: str, cell_number: int) -> list[str]:
    payload = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
    return list(payload["cells"][cell_number - 1].get("source", []))


def _is_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def _is_executable_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _pre_import_executable_count(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if _is_import_line(line):
            return sum(1 for prior in lines[:idx] if _is_executable_line(prior))
    return 0


def _has_import_after_executable(lines: list[str]) -> bool:
    seen_executable = False
    for line in lines:
        if _is_import_line(line) and seen_executable:
            return True
        if _is_executable_line(line) and not _is_import_line(line):
            seen_executable = True
    return False


def _classify_cell(lines: list[str], cell_number: int) -> tuple[str, str, str, int]:
    has_warning_filter = any("warnings.filterwarnings" in line for line in lines)
    pre_import_exec = _pre_import_executable_count(lines)
    import_count = sum(1 for line in lines if _is_import_line(line))
    if cell_number == 2 and has_warning_filter:
        return (
            "setup_warning_filter_cell",
            "batch_2_setup_warning_filter_refactor",
            "medium_warning_filter_semantics",
            import_count,
        )
    if pre_import_exec > 0 or _has_import_after_executable(lines):
        return (
            "local_delayed_import_cell",
            "batch_1_local_import_hoist",
            "low_to_medium_local_dependency",
            import_count,
        )
    return (
        "historical_setup_import_cell",
        "batch_2_setup_warning_filter_refactor",
        "medium_setup_cell",
        import_count,
    )


def _recommended_action(cell_class: str) -> str:
    if cell_class == "local_delayed_import_cell":
        return "hoist local import to notebook setup cell after dependency check"
    if cell_class == "setup_warning_filter_cell":
        return "split or reorder warning filters only after preserving import-time warning intent"
    return "inspect setup cell before import movement"


def _plan_rows(cell_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(cell_summary.itertuples(index=False), start=1):
        notebook_path = str(row.notebook_path_v398)
        cell_number = int(row.cell_v398)
        lines = _source_lines(notebook_path, cell_number)
        cell_class, batch, risk, import_count = _classify_cell(lines, cell_number)
        rows.append(
            {
                "plan_id_v399": f"e402_cell_plan_{idx:02d}",
                "notebook_path_v399": notebook_path,
                "cell_v399": cell_number,
                "e402_diagnostic_count_v399": int(row.e402_diagnostic_count_v398),
                "first_e402_row_v399": int(row.first_e402_row_v398),
                "last_e402_row_v399": int(row.last_e402_row_v398),
                "cell_class_v399": cell_class,
                "planned_batch_v399": batch,
                "risk_class_v399": risk,
                "import_line_count_v399": int(import_count),
                "pre_import_executable_lines_v399": int(_pre_import_executable_count(lines)),
                "recommended_action_v399": _recommended_action(cell_class),
                "mutation_allowed_v399": False,
                "claim_boundary_v399": "plan only; no notebook mutation in v399",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(plan: pd.DataFrame) -> pd.DataFrame:
    order = {
        "batch_1_local_import_hoist": 1,
        "batch_2_setup_warning_filter_refactor": 2,
    }
    rows = []
    for batch, group in plan.groupby("planned_batch_v399", sort=False):
        rows.append(
            {
                "batch_id_v399": batch,
                "execution_order_v399": order.get(batch, 99),
                "cell_count_v399": int(len(group)),
                "e402_diagnostic_count_v399": int(group["e402_diagnostic_count_v399"].sum()),
                "risk_class_v399": (
                    "lowest_available"
                    if batch == "batch_1_local_import_hoist"
                    else "requires_warning_semantics_review"
                ),
                "mutation_allowed_v399": False,
                "next_action_v399": (
                    NEXT_ARTIFACT
                    if batch == "batch_1_local_import_hoist"
                    else "paper4_v401_notebook_e402_setup_warning_refactor_plan.md"
                ),
                "claim_boundary_v399": "batch plan only; no notebook mutation in v399",
            }
        )
    return pd.DataFrame(rows).sort_values("execution_order_v399", ignore_index=True)


def _claim_blockers(plan: pd.DataFrame) -> pd.DataFrame:
    local = plan.loc[plan["planned_batch_v399"].eq("batch_1_local_import_hoist")]
    setup = plan.loc[plan["planned_batch_v399"].eq("batch_2_setup_warning_filter_refactor")]
    return pd.DataFrame(
        [
            {
                "blocker_id_v399": "local_import_hoist_not_applied_yet",
                "blocking_v399": True,
                "evidence_count_v399": int(local["e402_diagnostic_count_v399"].sum()),
                "required_next_artifact_v399": NEXT_ARTIFACT,
                "claim_boundary_v399": "v399 plans local import hoists but does not mutate notebooks",
            },
            {
                "blocker_id_v399": "setup_warning_filter_cells_require_review",
                "blocking_v399": True,
                "evidence_count_v399": int(setup["e402_diagnostic_count_v399"].sum()),
                "required_next_artifact_v399": "paper4_v401_notebook_e402_setup_warning_refactor_plan.md",
                "claim_boundary_v399": "warning filter semantics must be preserved",
            },
            {
                "blocker_id_v399": "global_notebook_lint_not_clean",
                "blocking_v399": True,
                "evidence_count_v399": 139,
                "required_next_artifact_v399": NEXT_ARTIFACT,
                "claim_boundary_v399": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v399": "paper4_final_promotion_forbidden",
                "blocking_v399": True,
                "evidence_count_v399": 1,
                "required_next_artifact_v399": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v399": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v399_e402_cell_refactor_plan_created",
                "allowed": True,
                "artifact": "paper4_v399_notebook_e402_cell_refactor_plan.csv",
                "boundary": "15 cells planned",
            },
            {
                "claim_id": "v399_first_e402_batch_selected",
                "allowed": True,
                "artifact": "paper4_v399_notebook_e402_batch_plan.csv",
                "boundary": "6 local delayed-import cells selected for v400",
            },
            {
                "claim_id": "v399_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v399",
            },
            {
                "claim_id": "v399_e402_or_global_lint_clean",
                "allowed": False,
                "artifact": "paper4_v399_claim_blockers.csv",
                "boundary": "119 E402 and 139 notebook diagnostics remain",
            },
            {
                "claim_id": "v399_setup_warning_refactor_applied",
                "allowed": False,
                "artifact": "paper4_v399_claim_blockers.csv",
                "boundary": "setup warning-filter cells only planned",
            },
            {
                "claim_id": "v399_working_champion_or_final_promotion",
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
                "claim": "v399 plans all 15 historical E402 notebook cells.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v399_notebook_e402_cell_refactor_plan.csv"
                ),
                "boundary": "Plan only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v399 selects a 6-cell local import hoist batch for v400.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v399_notebook_e402_batch_plan.csv"
                ),
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v399 repairs E402 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v399_claim_blockers.csv",
                "boundary": "No notebook mutation; 119 E402 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v399 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v399_claim_blockers.csv",
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
                    "v399 converts the 15-cell historical E402 frontier into an executable "
                    "cell-local refactor plan."
                ),
                "status": "notebook_e402_cell_refactor_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v400 applies only the local delayed-import hoist batch with roundtrip checks",
                "last_wave": "v399",
                "execution_result": "e402_15_cells_planned_first_batch_6_cells_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v399")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook E402 Cell Refactor Plan v399

Generated: {status["generated_at_utc"]}

v399 turns the v398 E402 policy into an executable cell-level plan without
mutating notebooks.

## Result

- Planned E402 cells: `{status["cell_plan_rows_v399"]}`.
- E402 diagnostics covered: `{status["e402_diagnostics_planned_v399"]}`.
- First batch cells: `{status["first_batch_cells_v399"]}`.
- First batch diagnostics: `{status["first_batch_e402_diagnostics_v399"]}`.
- Setup warning-filter cells: `{status["setup_warning_filter_cells_v399"]}`.
- Setup warning-filter diagnostics: `{status["setup_warning_filter_e402_diagnostics_v399"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v399"]}`.

## Required Caveat

v399 does not repair E402, does not make notebook or repository ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v399"]}` for the local delayed-import hoist batch.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V399_NOTEBOOK_E402_CELL_REFACTOR_PLAN_START -->"
    end = "<!-- V399_NOTEBOOK_E402_CELL_REFACTOR_PLAN_END -->"
    block = f"""
{start}

## Wave v399: Notebook E402 Cell Refactor Plan

Generated: {status["generated_at_utc"]}

### Objective

v399 converts the historical E402 frontier into an executable plan by separating
lower-risk local delayed imports from setup cells with warning-filter semantics.

### Results

- Cell plan rows:
  `{status["cell_plan_rows_v399"]}`.
- E402 diagnostics planned:
  `{status["e402_diagnostics_planned_v399"]}`.
- First batch cells:
  `{status["first_batch_cells_v399"]}`.
- First batch diagnostics:
  `{status["first_batch_e402_diagnostics_v399"]}`.
- Setup warning-filter cells:
  `{status["setup_warning_filter_cells_v399"]}`.
- Setup warning-filter diagnostics:
  `{status["setup_warning_filter_e402_diagnostics_v399"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v399"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v399"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v399"]}`.

### Interpretation

The next safe mutation should target only the 6 local delayed-import cells. The
9 setup cells need a separate warning-filter semantics review before any import
movement.

### Claim Impact

- Allowed: 15-cell E402 refactor plan and first-batch selection.
- Still prohibited: E402 repaired, notebook lint clean, repository ruff clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v399 in the living notebook. v400 should apply the local delayed-import
hoist batch with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v399 expects clean notebook diff because it is plan-only.")

    v398_status = json.loads((STATUS_DIR / "paper4_v398_status.json").read_text(encoding="utf-8"))
    if v398_status["next_artifact_v398"] != "paper4_v399_notebook_e402_cell_refactor_plan.csv":
        raise RuntimeError("v399 expects v398 to route to E402 cell refactor plan.")

    cell_summary = pd.read_csv(SOURCE_CELL_SUMMARY)
    current_e402 = _run_e402_json()
    diff_clean_before = _notebook_diff_clean()
    plan = _plan_rows(cell_summary)
    batch_plan = _batch_plan(plan)
    blockers = _claim_blockers(plan)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v399_notebook_e402_cell_refactor_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v399_notebook_e402_batch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v399_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v399_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    first_batch = plan.loc[plan["planned_batch_v399"].eq("batch_1_local_import_hoist")]
    setup = plan.loc[plan["planned_batch_v399"].eq("batch_2_setup_warning_filter_refactor")]
    status = {
        "phase": "v399_notebook_e402_cell_refactor_plan",
        "schema_version": "2026-05-17.399",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_e402_policy_version_v399": PRIOR_E402_POLICY_VERSION,
        "cell_plan_rows_v399": int(len(plan)),
        "batch_plan_rows_v399": int(len(batch_plan)),
        "claim_blocker_rows_v399": int(len(blockers)),
        "claim_matrix_rows_v399": int(len(claim_matrix)),
        "e402_diagnostics_planned_v399": int(plan["e402_diagnostic_count_v399"].sum()),
        "current_e402_diagnostics_v399": int(len(current_e402)),
        "first_batch_cells_v399": int(len(first_batch)),
        "first_batch_e402_diagnostics_v399": int(first_batch["e402_diagnostic_count_v399"].sum()),
        "setup_warning_filter_cells_v399": int(len(setup)),
        "setup_warning_filter_e402_diagnostics_v399": int(setup["e402_diagnostic_count_v399"].sum()),
        "selected_first_batch_v399": "batch_1_local_import_hoist",
        "setup_warning_filter_batch_deferred_v399": True,
        "notebooks_mutated_v399": False,
        "notebook_diff_clean_before_v399": diff_clean_before,
        "notebook_diff_clean_after_v399": _notebook_diff_clean(),
        "global_notebook_diagnostics_v399": 139,
        "global_ruff_clean_v399": False,
        "full_repository_pytest_run_v399": False,
        "full_quarto_render_run_v399": False,
        "working_champion_claim_allowed_v399": False,
        "paper1_promotion_allowed_v399": False,
        "paper4_working_champion_changed_v399": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v399": NEXT_ARTIFACT,
        "claim_boundary": (
            "v399 plans E402 cell refactors and selects a first batch; no notebooks "
            "are mutated and final promotion remains blocked"
        ),
    }
    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v399_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v399": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
