#!/usr/bin/env python3
"""Build Paper 4 v415 remaining style-only notebook lint triage artifacts."""

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

VERSION = 415
PRIOR_PYTEST_PROBE_VERSION = 414
NEXT_ARTIFACT = "paper4_v416_notebook_e741_comprehension_var_patch.md"
TRIAGE_MD = NOTEBOOK.parent / "paper4_v415_notebook_remaining_style_lint_triage.md"
STYLE_CODES = {"E741", "SIM108", "E712", "SIM102"}


def _run_ruff_json() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "notebooks", "--output-format", "json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff notebook probe failed")
    return json.loads(result.stdout or "[]")


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def _cell_source(notebook_path: str, cell_number: int) -> list[str]:
    payload = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
    return list(payload["cells"][cell_number - 1].get("source", []))


def _classification(code: str, notebook_path: str) -> tuple[str, str, bool]:
    if code == "E741":
        return (
            "safe_single_symbol_rename",
            "Rename list-comprehension variable l to line.",
            True,
        )
    if code == "SIM108":
        return (
            "manual_semantics_preserving_ifelse_refactor",
            "Convert simple assignment if/else to explicit conditional expression after E741.",
            False,
        )
    if code in {"E712", "SIM102"} and "side_projects" in notebook_path:
        return (
            "side_project_gpu_boolean_style_refactor",
            "Defer GPU benchmark boolean-style cleanup until core notebooks are clean.",
            False,
        )
    return ("manual_review", "Manual review required.", False)


def _triage_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        notebook_path = _relative_path(str(item["filename"]))
        cell_number = int(item.get("cell") or 0)
        row_number = int((item.get("location") or {}).get("row") or 0)
        lines = _cell_source(notebook_path, cell_number)
        source_line = lines[row_number - 1].strip() if 0 < row_number <= len(lines) else ""
        lint_code = str(item["code"])
        category, recommendation, selected = _classification(lint_code, notebook_path)
        rows.append(
            {
                "diagnostic_id_v415": f"style_lint_{len(rows) + 1:02d}",
                "notebook_path_v415": notebook_path,
                "cell_v415": cell_number,
                "row_v415": row_number,
                "lint_code_v415": lint_code,
                "message_v415": str(item["message"]),
                "source_line_v415": source_line,
                "triage_category_v415": category,
                "recommended_action_v415": recommendation,
                "selected_for_v416_v415": selected,
                "mutation_allowed_v415": False,
                "claim_boundary_v415": "style lint triage only; no notebook mutation",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(triage: pd.DataFrame) -> pd.DataFrame:
    selected = triage.loc[triage["selected_for_v416_v415"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "patch_batch_id_v415": "batch_1_e741_comprehension_variable",
                "diagnostic_count_v415": int(len(selected)),
                "notebook_count_v415": int(selected["notebook_path_v415"].nunique()),
                "recommended_next_artifact_v415": NEXT_ARTIFACT,
                "mutation_allowed_v415": False,
                "claim_boundary_v415": "selection only; E741 patch deferred to v416",
            }
        ]
    )


def _claim_blockers(global_after: int, e741_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v415": "e741_comprehension_patch_not_applied_yet",
                "blocking_v415": True,
                "evidence_count_v415": e741_after,
                "required_next_artifact_v415": NEXT_ARTIFACT,
                "claim_boundary_v415": "v415 triages but does not mutate notebooks",
            },
            {
                "blocker_id_v415": "style_notebook_lint_remaining",
                "blocking_v415": True,
                "evidence_count_v415": global_after,
                "required_next_artifact_v415": NEXT_ARTIFACT,
                "claim_boundary_v415": "style notebook lint remains",
            },
            {
                "blocker_id_v415": "paper4_final_promotion_forbidden",
                "blocking_v415": True,
                "evidence_count_v415": 1,
                "required_next_artifact_v415": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v415": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v415_remaining_style_lint_triage_created",
                "allowed": True,
                "artifact": "paper4_v415_notebook_remaining_style_lint_manifest.csv",
                "boundary": "6 style-only diagnostics inventoried",
            },
            {
                "claim_id": "v415_e741_batch_selected",
                "allowed": True,
                "artifact": "paper4_v415_notebook_style_lint_batch_plan.csv",
                "boundary": "single E741 patch selected for v416",
            },
            {
                "claim_id": "v415_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v415",
            },
            {
                "claim_id": "v415_style_lint_repaired",
                "allowed": False,
                "artifact": "paper4_v415_claim_blockers.csv",
                "boundary": "triage only",
            },
            {
                "claim_id": "v415_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v415_claim_blockers.csv",
                "boundary": "6 notebook diagnostics remain",
            },
            {
                "claim_id": "v415_working_champion_or_final_promotion",
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
                "claim": "v415 inventories the remaining 6 style-only notebook diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v415_notebook_remaining_style_lint_manifest.csv",
                "boundary": "Triage only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v415 selects the E741 comprehension-variable patch for v416.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v415_notebook_style_lint_batch_plan.csv",
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v415 repairs style lint or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v415_claim_blockers.csv",
                "boundary": "No notebook mutation; 6 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v415 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v415_claim_blockers.csv",
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
                "executable_item": "v415 triages the remaining style-only notebook lint frontier.",
                "status": "notebook_remaining_style_lint_triage_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v416 applies the E741 comprehension-variable patch with roundtrip checks",
                "last_wave": "v415",
                "execution_result": "six_style_diagnostics_triaged_e741_first_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v415")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _triage_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Remaining Style-Only Notebook Lint Triage v415

Generated: {status["generated_at_utc"]}

v415 inventories the remaining style-only notebook lint frontier.

## Result

- Notebook diagnostics: `{status["global_notebook_diagnostics_v415"]}`.
- E741 diagnostics: `{status["e741_diagnostics_v415"]}`.
- SIM108 diagnostics: `{status["sim108_diagnostics_v415"]}`.
- E712 diagnostics: `{status["e712_diagnostics_v415"]}`.
- SIM102 diagnostics: `{status["sim102_diagnostics_v415"]}`.
- Selected next artifact: `{status["next_artifact_v415"]}`.

## Required Caveat

v415 is non-mutating. It does not repair style lint, clear notebook lint, or
create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V415_NOTEBOOK_REMAINING_STYLE_LINT_TRIAGE_START -->"
    end = "<!-- V415_NOTEBOOK_REMAINING_STYLE_LINT_TRIAGE_END -->"
    block = f"""
{start}

## Wave v415: Remaining Style-Only Notebook Lint Triage

Generated: {status["generated_at_utc"]}

### Objective

v415 inventories the 6 remaining style-only notebook diagnostics and selects the
lowest-risk next mutation batch.

### Results

- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v415"]}`.
- E741/SIM108/E712/SIM102:
  `{status["e741_diagnostics_v415"]}` /
  `{status["sim108_diagnostics_v415"]}` /
  `{status["e712_diagnostics_v415"]}` /
  `{status["sim102_diagnostics_v415"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v415"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v415"]}`.

### Interpretation

The semantic lint blockers are closed; the next safe batch is the single E741
list-comprehension variable rename in notebook 03.

### Claim Impact

- Allowed: remaining style-lint inventory and E741 batch selection.
- Still prohibited: style lint repaired, notebook lint clean, repository ruff
  clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v415 in the living notebook. v416 should apply the E741 patch with
roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    clean_before = _notebook_diff_clean()
    if not clean_before:
        raise RuntimeError("v415 expects clean notebook diff before triage.")

    v414_status = json.loads((STATUS_DIR / "paper4_v414_status.json").read_text(encoding="utf-8"))
    if v414_status["next_artifact_v414"] != "paper4_v415_notebook_remaining_style_lint_triage.md":
        raise RuntimeError("v415 expects v414 to route to remaining style lint triage.")

    global_items = _run_ruff_json()
    counts = Counter(item["code"] for item in global_items)
    if set(counts) - STYLE_CODES:
        raise RuntimeError(f"v415 expected style-only diagnostics; got {sorted(counts)}")
    triage = _triage_rows(global_items)
    batch_plan = _batch_plan(triage)
    blockers = _claim_blockers(
        global_after=len(global_items),
        e741_after=counts.get("E741", 0),
    )
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v415_notebook_remaining_style_lint_manifest.csv", triage)
    write_csv(TABLE_DIR / "paper4_v415_notebook_style_lint_batch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v415_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v415_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v415_notebook_remaining_style_lint_triage",
        "schema_version": "2026-05-17.415",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pytest_probe_version_v415": PRIOR_PYTEST_PROBE_VERSION,
        "global_notebook_diagnostics_v415": int(len(global_items)),
        "e741_diagnostics_v415": int(counts.get("E741", 0)),
        "sim108_diagnostics_v415": int(counts.get("SIM108", 0)),
        "e712_diagnostics_v415": int(counts.get("E712", 0)),
        "sim102_diagnostics_v415": int(counts.get("SIM102", 0)),
        "selected_for_v416_rows_v415": int(triage["selected_for_v416_v415"].astype(bool).sum()),
        "patch_plan_rows_v415": int(len(batch_plan)),
        "notebooks_mutated_v415": False,
        "notebook_diff_clean_before_v415": clean_before,
        "notebook_diff_clean_after_v415": _notebook_diff_clean(),
        "global_ruff_clean_v415": False,
        "full_repository_pytest_run_v415": False,
        "full_quarto_render_run_v415": False,
        "working_champion_claim_allowed_v415": False,
        "paper1_promotion_allowed_v415": False,
        "paper4_working_champion_changed_v415": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v415": NEXT_ARTIFACT,
        "claim_boundary": (
            "v415 triages remaining style-only notebook lint and selects E741; no "
            "notebook mutation or final promotion is allowed"
        ),
    }
    if status["global_notebook_diagnostics_v415"] != 6:
        raise RuntimeError("v415 expected 6 remaining notebook diagnostics.")
    if status["selected_for_v416_rows_v415"] != 1:
        raise RuntimeError("v415 expected exactly one E741 diagnostic selected.")
    if status["notebook_diff_clean_after_v415"] is not True:
        raise RuntimeError("v415 unexpectedly mutated notebooks.")
    TRIAGE_MD.write_text(_triage_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v415": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
