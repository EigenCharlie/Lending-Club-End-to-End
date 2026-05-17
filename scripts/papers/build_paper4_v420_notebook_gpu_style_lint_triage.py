#!/usr/bin/env python3
"""Build Paper 4 v420 GPU side-project style-lint triage artifacts."""

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

VERSION = 420
PRIOR_PYTEST_PROBE_VERSION = 419
NEXT_ARTIFACT = "paper4_v421_notebook_gpu_style_lint_patch.md"
TRIAGE_MD = NOTEBOOK.parent / "paper4_v420_notebook_gpu_style_lint_triage.md"
TARGET_NOTEBOOK = "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb"


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


def _recommended_patch(code: str, source_line: str) -> tuple[str, str]:
    if code == "E712" and "rows_match_cpu" in source_line:
        return (
            "rows_match_cpu_fillna_boolean_mask",
            'replace `(cudf_df["rows_match_cpu"] == True)` with '
            '`cudf_df["rows_match_cpu"].fillna(False).astype(bool)`',
        )
    if code == "SIM102":
        return (
            "quality_pass_guard_hoist",
            "hoist quality_pass availability into has_quality_pass before the failed-quality block",
        )
    if code == "E712" and "quality_pass" in source_line:
        return (
            "quality_pass_eq_false_mask",
            'replace `quality["quality_pass"] == False` with `quality["quality_pass"].eq(False)`',
        )
    return ("manual_review", "manual review required")


def _triage_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    target_items = [item for item in items if _relative_path(str(item["filename"])) == TARGET_NOTEBOOK]
    for item in target_items:
        notebook_path = _relative_path(str(item["filename"]))
        cell_number = int(item.get("cell") or 0)
        row_number = int((item.get("location") or {}).get("row") or 0)
        lines = _cell_source(notebook_path, cell_number)
        source_line = lines[row_number - 1].strip() if 0 < row_number <= len(lines) else ""
        category, recommendation = _recommended_patch(str(item["code"]), source_line)
        rows.append(
            {
                "diagnostic_id_v420": f"gpu_style_lint_{len(rows) + 1:02d}",
                "notebook_path_v420": notebook_path,
                "cell_v420": cell_number,
                "row_v420": row_number,
                "lint_code_v420": str(item["code"]),
                "message_v420": str(item["message"]),
                "source_line_v420": source_line,
                "triage_category_v420": category,
                "recommended_action_v420": recommendation,
                "selected_for_v421_v420": category != "manual_review",
                "mutation_allowed_v420": False,
                "claim_boundary_v420": "GPU side-project style lint triage only; no notebook mutation",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(triage: pd.DataFrame) -> pd.DataFrame:
    selected = triage.loc[triage["selected_for_v421_v420"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "patch_batch_id_v420": "batch_1_gpu_side_project_style_lint",
                "diagnostic_count_v420": int(len(selected)),
                "notebook_count_v420": int(selected["notebook_path_v420"].nunique()),
                "recommended_next_artifact_v420": NEXT_ARTIFACT,
                "mutation_allowed_v420": False,
                "claim_boundary_v420": "selection only; GPU style-lint patch deferred to v421",
            }
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v420": "gpu_style_lint_patch_not_applied_yet",
                "blocking_v420": True,
                "evidence_count_v420": global_after,
                "required_next_artifact_v420": NEXT_ARTIFACT,
                "claim_boundary_v420": "v420 triages but does not mutate notebooks",
            },
            {
                "blocker_id_v420": "paper4_final_promotion_forbidden",
                "blocking_v420": True,
                "evidence_count_v420": 1,
                "required_next_artifact_v420": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v420": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v420_gpu_style_lint_triage_created",
                "allowed": True,
                "artifact": "paper4_v420_notebook_gpu_style_lint_manifest.csv",
                "boundary": "3 GPU side-project style diagnostics inventoried",
            },
            {
                "claim_id": "v420_gpu_style_lint_patch_selected",
                "allowed": True,
                "artifact": "paper4_v420_notebook_gpu_style_lint_patch_plan.csv",
                "boundary": "3 diagnostics selected for v421 patch",
            },
            {
                "claim_id": "v420_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v420",
            },
            {
                "claim_id": "v420_gpu_style_lint_repaired",
                "allowed": False,
                "artifact": "paper4_v420_claim_blockers.csv",
                "boundary": "triage only",
            },
            {
                "claim_id": "v420_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v420_claim_blockers.csv",
                "boundary": "3 notebook diagnostics remain",
            },
            {
                "claim_id": "v420_working_champion_or_final_promotion",
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
                "claim": "v420 inventories the remaining 3 GPU side-project notebook style diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v420_notebook_gpu_style_lint_manifest.csv",
                "boundary": "Triage only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v420 selects GPU side-project style-lint patches for v421.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v420_notebook_gpu_style_lint_patch_plan.csv",
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v420 repairs GPU style lint or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v420_claim_blockers.csv",
                "boundary": "No notebook mutation; 3 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v420 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v420_claim_blockers.csv",
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
                "executable_item": "v420 triages remaining GPU side-project notebook style lint.",
                "status": "notebook_gpu_style_lint_triage_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v421 applies GPU side-project style-lint patches with roundtrip checks",
                "last_wave": "v420",
                "execution_result": "three_gpu_style_diagnostics_triaged_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v420")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _triage_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 GPU Side-Project Style-Lint Triage v420

Generated: {status["generated_at_utc"]}

v420 reviews the final notebook-lint frontier before mutation.

## Result

- Notebook diagnostics: `{status["global_notebook_diagnostics_v420"]}`.
- E712 diagnostics: `{status["e712_diagnostics_v420"]}`.
- SIM102 diagnostics: `{status["sim102_diagnostics_v420"]}`.
- Selected for v421: `{status["selected_for_v421_rows_v420"]}`.
- Next artifact: `{status["next_artifact_v420"]}`.

## Required Caveat

v420 is non-mutating. It does not repair GPU style lint, clear notebook lint, or
create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V420_NOTEBOOK_GPU_STYLE_LINT_TRIAGE_START -->"
    end = "<!-- V420_NOTEBOOK_GPU_STYLE_LINT_TRIAGE_END -->"
    block = f"""
{start}

## Wave v420: GPU Side-Project Style-Lint Triage

Generated: {status["generated_at_utc"]}

### Objective

v420 reviews the final 3 notebook diagnostics in the GPU side-project notebook.

### Results

- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v420"]}`.
- E712/SIM102:
  `{status["e712_diagnostics_v420"]}` /
  `{status["sim102_diagnostics_v420"]}`.
- Selected for v421:
  `{status["selected_for_v421_rows_v420"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v420"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v420"]}`.

### Interpretation

The remaining notebook-lint frontier is isolated to GPU side-project boolean
style and a small quality-pass guard refactor.

### Claim Impact

- Allowed: GPU style-lint inventory and v421 patch selection.
- Still prohibited: GPU style lint repaired, notebook lint clean, repository
  ruff clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v420 in the living notebook. v421 should apply the GPU style-lint patch
with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    clean_before = _notebook_diff_clean()
    if not clean_before:
        raise RuntimeError("v420 expects clean notebook diff before triage.")

    v419_status = json.loads((STATUS_DIR / "paper4_v419_status.json").read_text(encoding="utf-8"))
    if v419_status["next_artifact_v419"] != "paper4_v420_notebook_gpu_style_lint_triage.md":
        raise RuntimeError("v420 expects v419 to route to GPU style lint triage.")

    global_items = _run_ruff_json()
    counts = Counter(item["code"] for item in global_items)
    triage = _triage_rows(global_items)
    batch_plan = _batch_plan(triage)
    blockers = _claim_blockers(global_after=len(global_items))
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v420_notebook_gpu_style_lint_manifest.csv", triage)
    write_csv(TABLE_DIR / "paper4_v420_notebook_gpu_style_lint_patch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v420_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v420_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v420_notebook_gpu_style_lint_triage",
        "schema_version": "2026-05-17.420",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pytest_probe_version_v420": PRIOR_PYTEST_PROBE_VERSION,
        "global_notebook_diagnostics_v420": int(len(global_items)),
        "e712_diagnostics_v420": int(counts.get("E712", 0)),
        "sim102_diagnostics_v420": int(counts.get("SIM102", 0)),
        "selected_for_v421_rows_v420": int(triage["selected_for_v421_v420"].astype(bool).sum()),
        "patch_plan_rows_v420": int(len(batch_plan)),
        "notebooks_mutated_v420": False,
        "notebook_diff_clean_before_v420": clean_before,
        "notebook_diff_clean_after_v420": _notebook_diff_clean(),
        "global_ruff_clean_v420": False,
        "full_repository_pytest_run_v420": False,
        "full_quarto_render_run_v420": False,
        "working_champion_claim_allowed_v420": False,
        "paper1_promotion_allowed_v420": False,
        "paper4_working_champion_changed_v420": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v420": NEXT_ARTIFACT,
        "claim_boundary": (
            "v420 triages final GPU side-project style lint and selects v421; no "
            "notebook mutation or final promotion is allowed"
        ),
    }
    if status["global_notebook_diagnostics_v420"] != 3:
        raise RuntimeError("v420 expected exactly three remaining diagnostics.")
    if status["selected_for_v421_rows_v420"] != 3:
        raise RuntimeError("v420 expected all three diagnostics selected.")
    if status["notebook_diff_clean_after_v420"] is not True:
        raise RuntimeError("v420 unexpectedly mutated notebooks.")
    TRIAGE_MD.write_text(_triage_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v420": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
