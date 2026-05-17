#!/usr/bin/env python3
"""Build Paper 4 v405 notebook sys.path/project-import refactor artifacts."""

from __future__ import annotations

import hashlib
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

VERSION = 405
PRIOR_SYS_PATH_PLAN_VERSION = 404
NEXT_VERSION = 406
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_post_sys_path_refactor_pytest_probe.md"
REFACTOR_MD = NOTEBOOK.parent / "paper4_v405_notebook_sys_path_project_import_refactor_batch.md"
PLAN_PATH = TABLE_DIR / "paper4_v404_notebook_sys_path_refactor_plan.csv"


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


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _changed_notebook_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _notebook_signature(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cells = payload.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    non_code_cells = [cell for cell in cells if cell.get("cell_type") != "code"]
    outputs_payload = [
        {"execution_count": cell.get("execution_count"), "outputs": cell.get("outputs", [])}
        for cell in code_cells
    ]
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "cell_type_sequence_hash": _sha256_text(_canonical([cell.get("cell_type") for cell in cells])),
        "non_code_source_hash": _sha256_text(_canonical([cell.get("source", "") for cell in non_code_cells])),
        "outputs_hash": _sha256_text(_canonical(outputs_payload)),
        "metadata_hash": _sha256_text(
            _canonical(
                {
                    "notebook_metadata": payload.get("metadata", {}),
                    "cell_metadata": [cell.get("metadata", {}) for cell in cells],
                    "nbformat": payload.get("nbformat"),
                    "nbformat_minor": payload.get("nbformat_minor"),
                }
            )
        ),
    }


def _is_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def _last_import_block_end(source: list[str]) -> int:
    last_end = 0
    idx = 0
    while idx < len(source):
        if _is_import_line(source[idx]):
            paren_balance = source[idx].count("(") - source[idx].count(")")
            idx += 1
            while paren_balance > 0 and idx < len(source):
                paren_balance += source[idx].count("(") - source[idx].count(")")
                idx += 1
            last_end = idx
            continue
        idx += 1
    return last_end


def _patch_sys_path_cell(source: list[str]) -> dict[str, int]:
    warning_filters = [line for line in source if "warnings.filterwarnings" in line]
    if not warning_filters:
        raise RuntimeError("target cell has no warnings.filterwarnings lines")
    sys_path_removed = sum(1 for line in source if "sys.path.insert" in line)
    import_sys_removed = sum(1 for line in source if line.strip() == "import sys")
    source[:] = [
        line
        for line in source
        if "warnings.filterwarnings" not in line
        and "sys.path.insert" not in line
        and line.strip() != "import sys"
    ]
    insert_at = _last_import_block_end(source)
    insert_lines: list[str] = []
    if insert_at > 0 and source[insert_at - 1] != "\n":
        insert_lines.append("\n")
    insert_lines.extend(warning_filters)
    if insert_at < len(source) and source[insert_at] != "\n":
        insert_lines.append("\n")
    source[insert_at:insert_at] = insert_lines
    return {
        "warning_filters_moved": len(warning_filters),
        "sys_path_insert_lines_removed": sys_path_removed,
        "import_sys_lines_removed": import_sys_removed,
    }


def _patch_notebooks(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    payloads: dict[str, dict[str, Any]] = {}
    for row in plan.itertuples(index=False):
        notebook_path = str(row.notebook_path_v404)
        cell_number = int(row.cell_v404)
        payload = payloads.get(notebook_path)
        if payload is None:
            payload = _read_notebook(ROOT / notebook_path)
            payloads[notebook_path] = payload
        source = payload["cells"][cell_number - 1]["source"]
        patch_counts = _patch_sys_path_cell(source)
        rows.append(
            {
                "action_id_v405": f"sys_path_refactor_{len(rows) + 1:02d}",
                "notebook_path_v405": notebook_path,
                "cell_v405": cell_number,
                "warning_filter_lines_moved_v405": patch_counts["warning_filters_moved"],
                "sys_path_insert_lines_removed_v405": patch_counts["sys_path_insert_lines_removed"],
                "import_sys_lines_removed_v405": patch_counts["import_sys_lines_removed"],
                "import_sort_normalization_applied_v405": True,
                "mutation_applied_v405": True,
                "claim_boundary_v405": "sys.path/project-import setup cell refactor",
            }
        )
    for notebook_path, payload in payloads.items():
        _write_notebook(ROOT / notebook_path, payload)
    return pd.DataFrame(rows)


def _run_import_sort_fix(target_notebooks: list[str]) -> str:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", *target_notebooks, "--select", "I001", "--fix"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff I001 fix failed")
    return result.stdout.strip()


def _roundtrip_integrity(
    before_signatures: dict[str, dict[str, Any]],
    after_signatures: dict[str, dict[str, Any]],
    changed_files: list[str],
) -> pd.DataFrame:
    rows = []
    for notebook_path in changed_files:
        before = before_signatures[notebook_path]
        after = after_signatures[notebook_path]
        rows.append(
            {
                "notebook_path_v405": notebook_path,
                "file_sha256_before_v405": before["file_sha256"],
                "file_sha256_after_v405": after["file_sha256"],
                "file_changed_v405": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v405": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v405": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v405": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v405": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v405": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v405": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v405": "setup cell code-source patch only",
            }
        )
    return pd.DataFrame(rows)


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_e402", before_counts.get("E402", 0), after_counts.get("E402", 0)),
        ("global_notebook_i001", before_counts.get("I001", 0), after_counts.get("I001", 0)),
        ("global_notebook_f401", before_counts.get("F401", 0), after_counts.get("F401", 0)),
        ("global_notebook_f821", before_counts.get("F821", 0), after_counts.get("F821", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v405": metric,
                "before_v405": before,
                "after_v405": after,
                "delta_v405": after - before,
                "claim_boundary_v405": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(*, global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v405": "non_e402_notebook_lint_remaining",
                "blocking_v405": True,
                "evidence_count_v405": global_after,
                "required_next_artifact_v405": "paper4_v407_notebook_non_e402_lint_triage.md",
                "claim_boundary_v405": "non-E402 notebook lint remains",
            },
            {
                "blocker_id_v405": "post_refactor_pytest_not_run",
                "blocking_v405": True,
                "evidence_count_v405": 1,
                "required_next_artifact_v405": NEXT_ARTIFACT,
                "claim_boundary_v405": "post-refactor validation deferred until v406",
            },
            {
                "blocker_id_v405": "paper4_final_promotion_forbidden",
                "blocking_v405": True,
                "evidence_count_v405": 1,
                "required_next_artifact_v405": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v405": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v405_sys_path_project_import_refactor_applied",
                "allowed": True,
                "artifact": "paper4_v405_notebook_sys_path_refactor_actions.csv",
                "boundary": "3 sys.path/project-import setup cells",
            },
            {
                "claim_id": "v405_e402_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v405_notebook_lint_delta.csv",
                "boundary": "notebook E402 only",
            },
            {
                "claim_id": "v405_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v405_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v405_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v405_claim_blockers.csv",
                "boundary": "20 non-E402 notebook diagnostics remain",
            },
            {
                "claim_id": "v405_post_refactor_pytest_passed",
                "allowed": False,
                "artifact": "paper4_v405_claim_blockers.csv",
                "boundary": "post-refactor pytest deferred to v406",
            },
            {
                "claim_id": "v405_working_champion_or_final_promotion",
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
                "claim": "v405 clears notebook E402 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v405_notebook_lint_delta.csv",
                "boundary": "Notebook E402 only; non-E402 lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v405 reduces notebook lint diagnostics from 62 to 20.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v405_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v405 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v405_claim_blockers.csv",
                "boundary": "20 non-E402 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v405 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v405_claim_blockers.csv",
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
                "executable_item": "v405 applies the sys.path/project-import E402 refactor batch.",
                "status": "notebook_sys_path_project_import_refactor_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v406 full repository pytest passes after final E402 notebook mutation",
                "last_wave": "v405",
                "execution_result": "notebook_e402_cleared_lint_reduced_62_to_20",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v405")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _refactor_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Sys.path/Project-Import Refactor Batch v405

Generated: {status["generated_at_utc"]}

v405 applies the final E402 notebook batch: 3 sys.path/project-import setup
cells.

## Result

- E402 diagnostics: `{status["global_notebook_e402_before_v405"]}` ->
  `{status["global_notebook_e402_after_v405"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v405"]}` ->
  `{status["global_notebook_diagnostics_after_v405"]}`.
- F401 diagnostics after refactor: `{status["global_notebook_f401_after_v405"]}`.
- I001 diagnostics after normalization: `{status["global_notebook_i001_after_v405"]}`.
- Changed notebook files: `{status["changed_notebook_files_v405"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v405"]}`.

## Required Caveat

v405 clears notebook E402, but does not clear all notebook lint, does not run
post-refactor pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v405"]}` as a post-sys.path-refactor pytest probe.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V405_NOTEBOOK_SYS_PATH_PROJECT_IMPORT_REFACTOR_BATCH_START -->"
    end = "<!-- V405_NOTEBOOK_SYS_PATH_PROJECT_IMPORT_REFACTOR_BATCH_END -->"
    block = f"""
{start}

## Wave v405: Notebook Sys.path/Project-Import Refactor Batch

Generated: {status["generated_at_utc"]}

### Objective

v405 applies the v404 sys.path/project-import refactor batch and tests whether
the remaining notebook E402 frontier can be cleared without F401 or I001 side
effects.

### Results

- E402 before/after:
  `{status["global_notebook_e402_before_v405"]}` ->
  `{status["global_notebook_e402_after_v405"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v405"]}` ->
  `{status["global_notebook_diagnostics_after_v405"]}`.
- F401 after refactor:
  `{status["global_notebook_f401_after_v405"]}`.
- I001 after normalization:
  `{status["global_notebook_i001_after_v405"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v405"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v405"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v405"]}`.

### Interpretation

The notebook E402 frontier is now closed. The remaining notebook lint is
non-E402 semantic/style debt and needs a separate triage lane after post-refactor
pytest.

### Claim Impact

- Allowed: notebook E402 cleared and lint reduced to 20 diagnostics.
- Still prohibited: all notebook lint clean, repository ruff clean, post-refactor
  pytest passed, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v405 in the living notebook. v406 should run a post-sys.path-refactor pytest
probe before further lint cleanup.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v405 expects clean notebook diff before mutation.")

    v404_status = json.loads((STATUS_DIR / "paper4_v404_status.json").read_text(encoding="utf-8"))
    if v404_status["next_artifact_v404"] != "paper4_v405_notebook_sys_path_project_import_refactor_batch.md":
        raise RuntimeError("v405 expects v404 to route to sys.path/project-import refactor batch.")

    plan = pd.read_csv(PLAN_PATH)
    target_notebooks = sorted(plan["notebook_path_v404"].unique())
    before_global = _run_ruff_json()
    before_counts = Counter(item["code"] for item in before_global)
    before_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in target_notebooks
    }

    actions = _patch_notebooks(plan)
    i001_fix_output = _run_import_sort_fix(target_notebooks)

    changed_files = _changed_notebook_files()
    after_global = _run_ruff_json()
    after_counts = Counter(item["code"] for item in after_global)
    after_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in changed_files
    }
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    integrity_columns = [
        "cell_count_preserved_v405",
        "code_cell_count_preserved_v405",
        "cell_type_sequence_preserved_v405",
        "non_code_source_preserved_v405",
        "outputs_preserved_v405",
        "metadata_preserved_v405",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(global_after=len(after_global))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v405_notebook_sys_path_refactor_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v405_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v405_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v405_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v405_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v405_notebook_sys_path_project_import_refactor_batch",
        "schema_version": "2026-05-17.405",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_sys_path_plan_version_v405": PRIOR_SYS_PATH_PLAN_VERSION,
        "sys_path_project_import_cells_v405": int(len(plan)),
        "sys_path_project_import_e402_diagnostics_v405": int(
            plan["current_e402_diagnostic_count_v404"].sum()
        ),
        "action_rows_v405": int(len(actions)),
        "i001_fix_command_output_v405": i001_fix_output,
        "global_notebook_diagnostics_before_v405": int(len(before_global)),
        "global_notebook_diagnostics_after_v405": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v405": int(len(before_global) - len(after_global)),
        "global_notebook_e402_before_v405": int(before_counts.get("E402", 0)),
        "global_notebook_e402_after_v405": int(after_counts.get("E402", 0)),
        "global_notebook_e402_reduced_v405": int(
            before_counts.get("E402", 0) - after_counts.get("E402", 0)
        ),
        "global_notebook_i001_after_v405": int(after_counts.get("I001", 0)),
        "global_notebook_f401_after_v405": int(after_counts.get("F401", 0)),
        "global_notebook_f821_after_v405": int(after_counts.get("F821", 0)),
        "changed_notebook_files_v405": int(len(changed_files)),
        "changed_notebook_file_list_v405": changed_files,
        "roundtrip_integrity_rows_v405": int(len(integrity)),
        "roundtrip_integrity_all_passed_v405": integrity_passed,
        "global_ruff_clean_v405": False,
        "full_repository_pytest_run_v405": False,
        "full_quarto_render_run_v405": False,
        "working_champion_claim_allowed_v405": False,
        "paper1_promotion_allowed_v405": False,
        "paper4_working_champion_changed_v405": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v405": NEXT_ARTIFACT,
        "claim_boundary": (
            "v405 clears notebook E402 with roundtrip checks; non-E402 lint and final "
            "promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v405 notebook roundtrip integrity failed.")
    if status["global_notebook_e402_after_v405"] != 0:
        raise RuntimeError("v405 did not clear notebook E402 diagnostics.")
    if status["global_notebook_i001_after_v405"] != 0:
        raise RuntimeError("v405 left an import-sort side effect.")
    if status["global_notebook_f401_after_v405"] != 0:
        raise RuntimeError("v405 left unused sys import side effects.")
    if status["global_notebook_diagnostics_after_v405"] != 20:
        raise RuntimeError("v405 did not reach the expected 20 non-E402 notebook diagnostics.")

    REFACTOR_MD.write_text(_refactor_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v405": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
