#!/usr/bin/env python3
"""Build Paper 4 v404 notebook sys.path/project-import refactor plan artifacts."""

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

VERSION = 404
PRIOR_PYTEST_PROBE_VERSION = 403
NEXT_VERSION = 405
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_sys_path_project_import_refactor_batch.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v404_notebook_sys_path_project_import_refactor_plan.md"
SOURCE_PLAN = TABLE_DIR / "paper4_v401_notebook_warning_filter_refactor_plan.csv"

IMPORT_PROBE = """
import src
from src.models.survival import make_survival_target
from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio
from src.optimization.robust_opt import scenario_analysis
from src.evaluation.ifrs9 import assign_stage, compute_ecl, ecl_with_conformal_range
print("imports_ok")
""".strip()


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


def _run_import_probe() -> pd.DataFrame:
    rows = []
    for probe_id, cwd in [
        ("repo_root_cwd", ROOT),
        ("notebooks_cwd", ROOT / "notebooks"),
    ]:
        result = subprocess.run(
            ["uv", "run", "python", "-c", IMPORT_PROBE],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        rows.append(
            {
                "probe_id_v404": probe_id,
                "cwd_v404": str(cwd.relative_to(ROOT)) if cwd != ROOT else ".",
                "exit_code_v404": int(result.returncode),
                "passed_v404": result.returncode == 0,
                "stdout_v404": result.stdout.strip(),
                "stderr_tail_v404": "\n".join(result.stderr.splitlines()[-20:]),
                "claim_boundary_v404": "import viability probe only; no notebook mutation",
            }
        )
    return pd.DataFrame(rows)


def _plan_rows(source_plan: pd.DataFrame, current_e402: list[dict[str, Any]]) -> pd.DataFrame:
    sys_path_plan = source_plan.loc[
        source_plan["planned_batch_v401"].eq("batch_2_sys_path_project_import_refactor")
    ].copy()
    current_counts = _current_e402_cell_counts(current_e402)
    rows = []
    for idx, row in enumerate(sys_path_plan.itertuples(index=False), start=1):
        notebook_path = str(row.notebook_path_v401)
        cell_number = int(row.cell_v401)
        lines = _source_lines(notebook_path, cell_number)
        counts = current_counts[(notebook_path, cell_number)]
        rows.append(
            {
                "plan_id_v404": f"sys_path_plan_{idx:02d}",
                "notebook_path_v404": notebook_path,
                "cell_v404": cell_number,
                "current_e402_diagnostic_count_v404": counts["current_e402_diagnostic_count"],
                "first_e402_row_v404": counts["first_e402_row"],
                "last_e402_row_v404": counts["last_e402_row"],
                "warning_filter_line_count_v404": sum(
                    1 for line in lines if "warnings.filterwarnings" in line
                ),
                "sys_path_insert_line_count_v404": sum(1 for line in lines if "sys.path.insert" in line),
                "import_sys_line_count_v404": sum(1 for line in lines if line.strip() == "import sys"),
                "project_import_line_count_v404": sum(1 for line in lines if _is_project_import_line(line)),
                "import_line_count_v404": sum(1 for line in lines if _is_import_line(line)),
                "recommended_action_v404": (
                    "remove sys.path.insert and import sys, move warning filters below imports, "
                    "then apply I001 normalization"
                ),
                "mutation_allowed_v404": False,
                "claim_boundary_v404": "plan only; no notebook mutation in v404",
            }
        )
    return pd.DataFrame(rows)


def _expected_lint_delta(current_global: list[dict[str, Any]], plan: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(item["code"] for item in current_global)
    e402_reduction = int(plan["current_e402_diagnostic_count_v404"].sum())
    expected_total_after = len(current_global) - e402_reduction
    rows = [
        ("global_notebook_total", len(current_global), expected_total_after),
        ("global_notebook_e402", counts.get("E402", 0), 0),
        ("global_notebook_i001", counts.get("I001", 0), 0),
        ("global_notebook_f401", counts.get("F401", 0), 0),
        ("global_notebook_f821", counts.get("F821", 0), counts.get("F821", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v404": metric,
                "current_v404": current,
                "expected_after_v405": expected,
                "expected_delta_v405": expected - current,
                "claim_boundary_v404": "expected lint delta only; no mutation in v404",
            }
            for metric, current, expected in rows
        ]
    )


def _claim_blockers(*, e402_after: int, global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v404": "sys_path_project_import_refactor_not_applied_yet",
                "blocking_v404": True,
                "evidence_count_v404": e402_after,
                "required_next_artifact_v404": NEXT_ARTIFACT,
                "claim_boundary_v404": "v404 plans sys.path/project-import refactor but does not mutate notebooks",
            },
            {
                "blocker_id_v404": "global_notebook_lint_not_clean",
                "blocking_v404": True,
                "evidence_count_v404": global_after,
                "required_next_artifact_v404": NEXT_ARTIFACT,
                "claim_boundary_v404": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v404": "post_refactor_pytest_not_run",
                "blocking_v404": True,
                "evidence_count_v404": 1,
                "required_next_artifact_v404": "paper4_v406_post_sys_path_refactor_pytest_probe.md",
                "claim_boundary_v404": "post-refactor validation deferred until after v405 mutation",
            },
            {
                "blocker_id_v404": "paper4_final_promotion_forbidden",
                "blocking_v404": True,
                "evidence_count_v404": 1,
                "required_next_artifact_v404": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v404": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix(import_probe_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v404_sys_path_refactor_plan_created",
                "allowed": True,
                "artifact": "paper4_v404_notebook_sys_path_refactor_plan.csv",
                "boundary": "3 remaining sys.path/project-import cells planned",
            },
            {
                "claim_id": "v404_import_viability_probe_passed",
                "allowed": import_probe_passed,
                "artifact": "paper4_v404_import_viability_probe.csv",
                "boundary": "true only when root and notebooks cwd import probes pass",
            },
            {
                "claim_id": "v404_expected_e402_clearance_modeled",
                "allowed": True,
                "artifact": "paper4_v404_expected_lint_delta.csv",
                "boundary": "expected delta only; no notebook mutation",
            },
            {
                "claim_id": "v404_sys_path_project_import_refactor_applied",
                "allowed": False,
                "artifact": "paper4_v404_claim_blockers.csv",
                "boundary": "application deferred to v405",
            },
            {
                "claim_id": "v404_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v404_claim_blockers.csv",
                "boundary": "62 notebook diagnostics remain",
            },
            {
                "claim_id": "v404_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(import_probe_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v404 plans the 3 remaining sys.path/project-import E402 cells.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v404_notebook_sys_path_refactor_plan.csv"
                ),
                "boundary": "Plan only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v404 import viability probes pass without notebook sys.path injection.",
                "allowed": import_probe_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v404_import_viability_probe.csv",
                "boundary": "Root and notebooks cwd probe only.",
                "prohibited_claim_flag": not import_probe_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v404 repairs E402 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v404_claim_blockers.csv",
                "boundary": "No notebook mutation; 42 E402 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v404 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v404_claim_blockers.csv",
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
                "executable_item": "v404 plans sys.path/project-import E402 refactor after pytest pass.",
                "status": "notebook_sys_path_project_import_refactor_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v405 clears remaining E402 without F401/I001 side effects",
                "last_wave": "v404",
                "execution_result": "sys_path_e402_3_cells_42_diagnostics_planned_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v404")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Sys.path/Project-Import Refactor Plan v404

Generated: {status["generated_at_utc"]}

v404 plans the final E402 group: 3 setup cells where `sys.path.insert(...)`
precedes project imports.

## Result

- Sys.path/project-import cells planned: `{status["sys_path_project_import_cells_v404"]}`.
- Current E402 diagnostics planned: `{status["sys_path_project_import_e402_diagnostics_v404"]}`.
- Import viability probes passed: `{status["import_viability_all_passed_v404"]}`.
- Current notebook diagnostics: `{status["global_notebook_diagnostics_v404"]}`.
- Expected diagnostics after v405: `{status["expected_global_notebook_diagnostics_after_v405"]}`.
- Expected E402 after v405: `{status["expected_global_notebook_e402_after_v405"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v404"]}`.

## Required Caveat

v404 does not repair E402, does not make notebook or repository ruff clean, does
not run post-refactor pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v404"]}` for the sys.path/project-import refactor
batch.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V404_NOTEBOOK_SYS_PATH_PROJECT_IMPORT_REFACTOR_PLAN_START -->"
    end = "<!-- V404_NOTEBOOK_SYS_PATH_PROJECT_IMPORT_REFACTOR_PLAN_END -->"
    block = f"""
{start}

## Wave v404: Notebook Sys.path/Project-Import Refactor Plan

Generated: {status["generated_at_utc"]}

### Objective

v404 plans the 3 remaining E402 setup cells where `sys.path.insert(...)` precedes
project imports and validates whether `src` imports work without notebook-local
path injection.

### Results

- Sys.path/project-import cells:
  `{status["sys_path_project_import_cells_v404"]}`.
- Sys.path/project-import E402 diagnostics:
  `{status["sys_path_project_import_e402_diagnostics_v404"]}`.
- Import viability probes passed:
  `{status["import_viability_all_passed_v404"]}`.
- Current notebook diagnostics:
  `{status["global_notebook_diagnostics_v404"]}`.
- Expected notebook diagnostics after v405:
  `{status["expected_global_notebook_diagnostics_after_v405"]}`.
- Expected E402 after v405:
  `{status["expected_global_notebook_e402_after_v405"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v404"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v404"]}`.

### Interpretation

The import probes indicate that project imports resolve from both repository root
and `notebooks/` cwd without notebook-local `sys.path.insert(...)`. v405 can
therefore test the narrow mutation: remove path injection and unused `import
sys`, move warning filters below imports, and normalize import ordering.

### Claim Impact

- Allowed: sys.path/project-import refactor plan and import viability probe.
- Still prohibited: E402 repaired, notebook lint clean, repository ruff clean,
  post-refactor pytest passed, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v404 in the living notebook. v405 should apply the planned sys.path
refactor batch with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v404 expects clean notebook diff because it is plan-only.")

    v403_status = json.loads((STATUS_DIR / "paper4_v403_status.json").read_text(encoding="utf-8"))
    if v403_status["next_artifact_v403"] != "paper4_v404_notebook_sys_path_project_import_refactor_plan.md":
        raise RuntimeError("v404 expects v403 to route to sys.path/project-import refactor planning.")
    if not v403_status["pytest_passed_v403"]:
        raise RuntimeError("v404 requires the v403 post-mutation pytest probe to pass.")

    source_plan = pd.read_csv(SOURCE_PLAN)
    current_global = _run_ruff_json()
    current_e402 = _run_ruff_json(["E402"])
    import_probe = _run_import_probe()
    import_probe_passed = bool(import_probe["passed_v404"].astype(bool).all())
    plan = _plan_rows(source_plan, current_e402)
    expected_lint_delta = _expected_lint_delta(current_global, plan)
    blockers = _claim_blockers(e402_after=len(current_e402), global_after=len(current_global))
    claim_matrix = _claim_matrix(import_probe_passed)

    write_csv(TABLE_DIR / "paper4_v404_notebook_sys_path_refactor_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v404_import_viability_probe.csv", import_probe)
    write_csv(TABLE_DIR / "paper4_v404_expected_lint_delta.csv", expected_lint_delta)
    write_csv(TABLE_DIR / "paper4_v404_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v404_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries(import_probe_passed)
    _update_backlog()

    status = {
        "phase": "v404_notebook_sys_path_project_import_refactor_plan",
        "schema_version": "2026-05-17.404",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pytest_probe_version_v404": PRIOR_PYTEST_PROBE_VERSION,
        "sys_path_project_import_cells_v404": int(len(plan)),
        "sys_path_project_import_e402_diagnostics_v404": int(
            plan["current_e402_diagnostic_count_v404"].sum()
        ),
        "import_viability_probe_rows_v404": int(len(import_probe)),
        "import_viability_all_passed_v404": import_probe_passed,
        "notebooks_mutated_v404": False,
        "notebook_diff_clean_before_v404": True,
        "notebook_diff_clean_after_v404": _notebook_diff_clean(),
        "global_notebook_diagnostics_v404": int(len(current_global)),
        "global_notebook_e402_v404": int(Counter(item["code"] for item in current_global).get("E402", 0)),
        "global_notebook_i001_v404": int(Counter(item["code"] for item in current_global).get("I001", 0)),
        "expected_global_notebook_diagnostics_after_v405": 20,
        "expected_global_notebook_e402_after_v405": 0,
        "global_ruff_clean_v404": False,
        "full_repository_pytest_run_v404": False,
        "full_quarto_render_run_v404": False,
        "working_champion_claim_allowed_v404": False,
        "paper1_promotion_allowed_v404": False,
        "paper4_working_champion_changed_v404": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v404": NEXT_ARTIFACT,
        "claim_boundary": (
            "v404 plans sys.path/project-import E402 refactor after pytest pass; no "
            "notebooks are mutated and final promotion remains blocked"
        ),
    }
    if status["sys_path_project_import_e402_diagnostics_v404"] != 42:
        raise RuntimeError("v404 expected 42 sys.path/project-import E402 diagnostics.")
    if not import_probe_passed:
        raise RuntimeError("v404 import viability probes did not pass.")

    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v404": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
