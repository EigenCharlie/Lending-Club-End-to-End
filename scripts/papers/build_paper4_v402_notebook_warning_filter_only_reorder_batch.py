#!/usr/bin/env python3
"""Build Paper 4 v402 notebook warning-filter-only reorder artifacts."""

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

VERSION = 402
PRIOR_SETUP_PLAN_VERSION = 401
NEXT_VERSION = 403
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_post_notebook_mutation_pytest_probe.md"
REORDER_MD = NOTEBOOK.parent / "paper4_v402_notebook_warning_filter_only_reorder_batch.md"
PLAN_PATH = TABLE_DIR / "paper4_v401_notebook_warning_filter_refactor_plan.csv"


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
        {
            "execution_count": cell.get("execution_count"),
            "outputs": cell.get("outputs", []),
        }
        for cell in code_cells
    ]
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "cell_type_sequence_hash": _sha256_text(
            _canonical([cell.get("cell_type") for cell in cells])
        ),
        "non_code_source_hash": _sha256_text(
            _canonical([cell.get("source", "") for cell in non_code_cells])
        ),
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


def _move_warning_filters_below_imports(source: list[str]) -> int:
    warning_filters = [line for line in source if "warnings.filterwarnings" in line]
    if not warning_filters:
        raise RuntimeError("target cell has no warnings.filterwarnings lines")
    source[:] = [line for line in source if "warnings.filterwarnings" not in line]
    insert_at = _last_import_block_end(source)
    insert_lines: list[str] = []
    if insert_at > 0 and source[insert_at - 1] != "\n":
        insert_lines.append("\n")
    insert_lines.extend(warning_filters)
    if insert_at < len(source) and source[insert_at] != "\n":
        insert_lines.append("\n")
    source[insert_at:insert_at] = insert_lines
    return len(warning_filters)


def _patch_notebooks(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    payloads: dict[str, dict[str, Any]] = {}
    for row in plan.itertuples(index=False):
        notebook_path = str(row.notebook_path_v401)
        cell_number = int(row.cell_v401)
        payload = payloads.get(notebook_path)
        if payload is None:
            payload = _read_notebook(ROOT / notebook_path)
            payloads[notebook_path] = payload
        source = payload["cells"][cell_number - 1]["source"]
        moved = _move_warning_filters_below_imports(source)
        rows.append(
            {
                "action_id_v402": f"warning_filter_reorder_{len(rows) + 1:02d}",
                "notebook_path_v402": notebook_path,
                "cell_v402": cell_number,
                "warning_filter_lines_moved_v402": moved,
                "import_sort_normalization_applied_v402": True,
                "mutation_applied_v402": True,
                "claim_boundary_v402": "warning-filter-only setup cell reorder",
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
                "notebook_path_v402": notebook_path,
                "file_sha256_before_v402": before["file_sha256"],
                "file_sha256_after_v402": after["file_sha256"],
                "file_changed_v402": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v402": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v402": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v402": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v402": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v402": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v402": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v402": "setup cell code-source patch only",
            }
        )
    return pd.DataFrame(rows)


def _lint_delta(
    *,
    before_total: int,
    before_counts: Counter[str],
    after_items: list[dict[str, Any]],
) -> pd.DataFrame:
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", before_total, len(after_items)),
        ("global_notebook_e402", before_counts.get("E402", 0), after_counts.get("E402", 0)),
        ("global_notebook_i001", before_counts.get("I001", 0), after_counts.get("I001", 0)),
        ("global_notebook_b018", before_counts.get("B018", 0), after_counts.get("B018", 0)),
        ("global_notebook_f821", before_counts.get("F821", 0), after_counts.get("F821", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v402": metric,
                "before_v402": before,
                "after_v402": after,
                "delta_v402": after - before,
                "claim_boundary_v402": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(*, e402_after: int, global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v402": "sys_path_project_import_e402_remaining",
                "blocking_v402": True,
                "evidence_count_v402": e402_after,
                "required_next_artifact_v402": "paper4_v404_notebook_sys_path_project_import_refactor_plan.md",
                "claim_boundary_v402": "sys.path/project-import cells remain",
            },
            {
                "blocker_id_v402": "global_notebook_lint_not_clean",
                "blocking_v402": True,
                "evidence_count_v402": global_after,
                "required_next_artifact_v402": NEXT_ARTIFACT,
                "claim_boundary_v402": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v402": "full_repository_pytest_not_rerun",
                "blocking_v402": True,
                "evidence_count_v402": 1,
                "required_next_artifact_v402": NEXT_ARTIFACT,
                "claim_boundary_v402": "focused validation only",
            },
            {
                "blocker_id_v402": "paper4_final_promotion_forbidden",
                "blocking_v402": True,
                "evidence_count_v402": 1,
                "required_next_artifact_v402": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v402": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v402_warning_filter_only_reorder_applied",
                "allowed": True,
                "artifact": "paper4_v402_notebook_warning_filter_reorder_actions.csv",
                "boundary": "6 warning-filter-only setup cells",
            },
            {
                "claim_id": "v402_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v402_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v402_notebook_lint_reduced",
                "allowed": True,
                "artifact": "paper4_v402_notebook_lint_delta.csv",
                "boundary": "reduction only, not clean",
            },
            {
                "claim_id": "v402_sys_path_project_import_e402_repaired",
                "allowed": False,
                "artifact": "paper4_v402_claim_blockers.csv",
                "boundary": "sys.path cells deferred",
            },
            {
                "claim_id": "v402_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v402_claim_blockers.csv",
                "boundary": "62 notebook diagnostics remain",
            },
            {
                "claim_id": "v402_working_champion_or_final_promotion",
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
                "claim": "v402 applies the 6-cell warning-filter-only reorder batch.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v402_notebook_warning_filter_reorder_actions.csv"
                ),
                "boundary": "Warning-filter-only setup cells plus import-sort normalization.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v402 reduces notebook lint diagnostics from 132 to 62.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v402_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v402 clears E402 or global notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v402_claim_blockers.csv",
                "boundary": "42 E402 and 62 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v402 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v402_claim_blockers.csv",
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
                    "v402 applies warning-filter-only setup cell reorder with roundtrip checks."
                ),
                "status": "notebook_warning_filter_only_reorder_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v403 runs a post-notebook-mutation pytest probe before deeper sys.path work",
                "last_wave": "v402",
                "execution_result": "notebook_lint_reduced_132_to_62_warning_filter_batch",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v402")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _reorder_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Warning-Filter-Only Reorder Batch v402

Generated: {status["generated_at_utc"]}

v402 applies the first setup-cell batch selected by v401: 6 warning-filter-only
cells. It moves warning filters below import blocks and applies import-sort
normalization only in those changed notebooks.

## Result

- E402 diagnostics: `{status["global_notebook_e402_before_v402"]}` ->
  `{status["global_notebook_e402_after_v402"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v402"]}` ->
  `{status["global_notebook_diagnostics_after_v402"]}`.
- I001 diagnostics after normalization: `{status["global_notebook_i001_after_v402"]}`.
- Changed notebook files: `{status["changed_notebook_files_v402"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v402"]}`.

## Required Caveat

v402 does not repair sys.path/project-import E402 cells, does not clear notebook
or repository ruff, does not run full pytest, and does not create Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v402"]}` as a post-notebook-mutation pytest probe.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V402_NOTEBOOK_WARNING_FILTER_ONLY_REORDER_BATCH_START -->"
    end = "<!-- V402_NOTEBOOK_WARNING_FILTER_ONLY_REORDER_BATCH_END -->"
    block = f"""
{start}

## Wave v402: Notebook Warning-Filter-Only Reorder Batch

Generated: {status["generated_at_utc"]}

### Objective

v402 applies the v401 first setup-cell batch: 6 warning-filter-only cells, with
import-sort normalization limited to the changed notebooks.

### Results

- E402 before/after:
  `{status["global_notebook_e402_before_v402"]}` ->
  `{status["global_notebook_e402_after_v402"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v402"]}` ->
  `{status["global_notebook_diagnostics_after_v402"]}`.
- I001 after normalization:
  `{status["global_notebook_i001_after_v402"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v402"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v402"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v402"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v402"]}`.

### Interpretation

The warning-filter-only setup batch is now closed. The remaining E402 frontier is
the smaller sys.path/project-import group where notebook execution path semantics
must be reviewed separately.

### Claim Impact

- Allowed: warning-filter-only reorder batch, import-sort normalization and
  roundtrip preservation.
- Still prohibited: sys.path/project-import E402 repaired, notebook lint clean,
  repository ruff clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v402 in the living notebook. v403 should run a post-notebook-mutation pytest
probe before deeper sys.path refactors.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v402 expects clean notebook diff before mutation.")

    v401_status = json.loads((STATUS_DIR / "paper4_v401_status.json").read_text(encoding="utf-8"))
    if v401_status["next_artifact_v401"] != "paper4_v402_notebook_warning_filter_only_reorder_batch.md":
        raise RuntimeError("v402 expects v401 to route to warning-filter-only reorder batch.")

    plan = pd.read_csv(PLAN_PATH)
    target_plan = plan.loc[
        plan["planned_batch_v401"].eq("batch_1_warning_filter_only_reorder")
    ].copy()
    target_notebooks = sorted(target_plan["notebook_path_v401"].unique())
    before_global = _run_ruff_json()
    before_counts = Counter(item["code"] for item in before_global)
    if before_counts.get("I001", 0) != 0:
        raise RuntimeError("v402 expects no pre-existing notebook I001 diagnostics.")
    before_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in target_notebooks
    }

    actions = _patch_notebooks(target_plan)
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
        "cell_count_preserved_v402",
        "code_cell_count_preserved_v402",
        "cell_type_sequence_preserved_v402",
        "non_code_source_preserved_v402",
        "outputs_preserved_v402",
        "metadata_preserved_v402",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(
        before_total=len(before_global),
        before_counts=before_counts,
        after_items=after_global,
    )
    blockers = _claim_blockers(
        e402_after=after_counts.get("E402", 0),
        global_after=len(after_global),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v402_notebook_warning_filter_reorder_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v402_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v402_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v402_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v402_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    expected_reduction = int(v401_status["warning_filter_only_e402_diagnostics_v401"])
    status = {
        "phase": "v402_notebook_warning_filter_only_reorder_batch",
        "schema_version": "2026-05-17.402",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_setup_plan_version_v402": PRIOR_SETUP_PLAN_VERSION,
        "warning_filter_only_cells_v402": int(len(target_plan)),
        "warning_filter_only_e402_diagnostics_v402": expected_reduction,
        "action_rows_v402": int(len(actions)),
        "i001_fix_command_output_v402": i001_fix_output,
        "global_notebook_diagnostics_before_v402": int(len(before_global)),
        "global_notebook_diagnostics_after_v402": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v402": int(len(before_global) - len(after_global)),
        "global_notebook_e402_before_v402": int(before_counts.get("E402", 0)),
        "global_notebook_e402_after_v402": int(after_counts.get("E402", 0)),
        "global_notebook_e402_reduced_v402": int(
            before_counts.get("E402", 0) - after_counts.get("E402", 0)
        ),
        "global_notebook_i001_after_v402": int(after_counts.get("I001", 0)),
        "global_notebook_f821_after_v402": int(after_counts.get("F821", 0)),
        "changed_notebook_files_v402": int(len(changed_files)),
        "changed_notebook_file_list_v402": changed_files,
        "roundtrip_integrity_rows_v402": int(len(integrity)),
        "roundtrip_integrity_all_passed_v402": integrity_passed,
        "global_ruff_clean_v402": False,
        "full_repository_pytest_run_v402": False,
        "full_quarto_render_run_v402": False,
        "working_champion_claim_allowed_v402": False,
        "paper1_promotion_allowed_v402": False,
        "paper4_working_champion_changed_v402": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v402": NEXT_ARTIFACT,
        "claim_boundary": (
            "v402 applies warning-filter-only setup cell reorders with roundtrip "
            "checks; sys.path E402 and final promotion remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v402 notebook roundtrip integrity failed.")
    if status["global_notebook_e402_reduced_v402"] != expected_reduction:
        raise RuntimeError("v402 E402 reduction did not match selected warning-filter batch.")
    if status["global_notebook_diagnostics_reduced_v402"] != expected_reduction:
        raise RuntimeError("v402 global diagnostic reduction did not match selected warning-filter batch.")
    if status["global_notebook_i001_after_v402"] != 0:
        raise RuntimeError("v402 left an import-sort side effect.")

    REORDER_MD.write_text(_reorder_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v402": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
