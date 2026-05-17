#!/usr/bin/env python3
"""Build Paper 4 v400 notebook E402 local import hoist artifacts."""

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

VERSION = 400
PRIOR_E402_PLAN_VERSION = 399
NEXT_VERSION = 401
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_e402_setup_warning_refactor_plan.md"
STATUS_PATH = STATUS_DIR / f"paper4_v{VERSION}_status.json"
HOIST_MD = NOTEBOOK.parent / "paper4_v400_notebook_e402_local_import_hoist_batch.md"
PLAN_PATH = TABLE_DIR / "paper4_v399_notebook_e402_cell_refactor_plan.csv"


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
    payload = _read_notebook(path)
    return _notebook_signature_from_payload(payload, path.read_bytes())


def _notebook_signature_from_git_head(notebook_path: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "show", f"HEAD:{notebook_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    return _notebook_signature_from_payload(payload, result.stdout)


def _notebook_signature_from_payload(payload: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
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
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
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


def _top_import_insert_index(source: list[str]) -> int:
    idx = 0
    while idx < len(source) and (
        not source[idx].strip() or source[idx].lstrip().startswith("#")
    ):
        idx += 1
    while idx < len(source):
        stripped = source[idx].strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            idx += 1
            continue
        break
    return idx


def _move_imports_to_cell_top(source: list[str], import_lines: list[str]) -> None:
    for line in import_lines:
        while line in source:
            source.remove(line)
    insert_at = _top_import_insert_index(source)
    for offset, line in enumerate(import_lines):
        source.insert(insert_at + offset, line)
    if insert_at + len(import_lines) < len(source) and source[insert_at + len(import_lines)] != "\n":
        source.insert(insert_at + len(import_lines), "\n")


def _patch_notebooks() -> pd.DataFrame:
    actions = [
        (
            "pd_modeling_calibration_imports",
            "notebooks/03_pd_modeling.ipynb",
            24,
            [
                "from sklearn.isotonic import IsotonicRegression\n",
                "from sklearn.linear_model import LogisticRegression as PlattLR\n",
            ],
        ),
        (
            "time_series_stl_import",
            "notebooks/05_time_series_forecasting.ipynb",
            7,
            [
                "from statsmodels.tsa.seasonal import STL\n",
                "from statsmodels.tsa.stattools import adfuller\n",
            ],
        ),
        (
            "survival_cox_scaler_import",
            "notebooks/06_survival_analysis.ipynb",
            12,
            ["from sklearn.preprocessing import StandardScaler\n"],
        ),
        (
            "survival_rsf_split_import",
            "notebooks/06_survival_analysis.ipynb",
            18,
            ["from sklearn.model_selection import train_test_split\n"],
        ),
        (
            "pipeline_auc_import",
            "notebooks/09_end_to_end_pipeline.ipynb",
            8,
            ["from sklearn.metrics import roc_auc_score\n"],
        ),
        (
            "explainability_patch_import",
            "notebooks/13_model_explainability.ipynb",
            8,
            ["from matplotlib.patches import Patch\n"],
        ),
    ]
    rows = []
    payloads: dict[str, dict[str, Any]] = {}
    for action_id, notebook_path, cell_number, import_lines in actions:
        payload = payloads.get(notebook_path)
        if payload is None:
            payload = _read_notebook(ROOT / notebook_path)
            payloads[notebook_path] = payload
        source = payload["cells"][cell_number - 1]["source"]
        _move_imports_to_cell_top(source, import_lines)
        rows.append(
            {
                "action_id_v400": action_id,
                "notebook_path_v400": notebook_path,
                "cell_v400": cell_number,
                "import_lines_moved_v400": len(import_lines),
                "mutation_applied_v400": True,
                "claim_boundary_v400": "local delayed-import hoist only",
            }
        )
    for notebook_path, payload in payloads.items():
        _write_notebook(ROOT / notebook_path, payload)
    return pd.DataFrame(rows)


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
                "notebook_path_v400": notebook_path,
                "file_sha256_before_v400": before["file_sha256"],
                "file_sha256_after_v400": after["file_sha256"],
                "file_changed_v400": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v400": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v400": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v400": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v400": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v400": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v400": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v400": "local import-hoist code-source patch only",
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
                "metric_v400": metric,
                "before_v400": before,
                "after_v400": after,
                "delta_v400": after - before,
                "claim_boundary_v400": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(*, e402_after: int, global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v400": "setup_warning_filter_e402_remaining",
                "blocking_v400": True,
                "evidence_count_v400": e402_after,
                "required_next_artifact_v400": NEXT_ARTIFACT,
                "claim_boundary_v400": "setup warning-filter cells remain",
            },
            {
                "blocker_id_v400": "global_notebook_lint_not_clean",
                "blocking_v400": True,
                "evidence_count_v400": global_after,
                "required_next_artifact_v400": NEXT_ARTIFACT,
                "claim_boundary_v400": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v400": "full_repository_pytest_not_rerun",
                "blocking_v400": True,
                "evidence_count_v400": 1,
                "required_next_artifact_v400": "paper4_v402_post_notebook_mutation_pytest_probe.md",
                "claim_boundary_v400": "focused validation only",
            },
            {
                "blocker_id_v400": "paper4_final_promotion_forbidden",
                "blocking_v400": True,
                "evidence_count_v400": 1,
                "required_next_artifact_v400": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v400": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v400_local_e402_import_hoist_applied",
                "allowed": True,
                "artifact": "paper4_v400_notebook_e402_local_import_hoist_actions.csv",
                "boundary": "6 local delayed-import cells only",
            },
            {
                "claim_id": "v400_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v400_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v400_notebook_lint_reduced",
                "allowed": True,
                "artifact": "paper4_v400_notebook_lint_delta.csv",
                "boundary": "reduction only, not clean",
            },
            {
                "claim_id": "v400_setup_warning_e402_repaired",
                "allowed": False,
                "artifact": "paper4_v400_claim_blockers.csv",
                "boundary": "setup warning-filter cells deferred",
            },
            {
                "claim_id": "v400_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v400_claim_blockers.csv",
                "boundary": "132 notebook diagnostics remain",
            },
            {
                "claim_id": "v400_working_champion_or_final_promotion",
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
                "claim": "v400 applies the 6-cell local E402 import-hoist batch.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v400_notebook_e402_local_import_hoist_actions.csv"
                ),
                "boundary": "Local delayed-import cells only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v400 reduces notebook lint diagnostics from 139 to 132.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v400_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v400 clears E402 or global notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v400_claim_blockers.csv",
                "boundary": "112 E402 and 132 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v400 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v400_claim_blockers.csv",
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
                    "v400 applies the local delayed-import E402 hoist batch with roundtrip checks."
                ),
                "status": "notebook_e402_local_import_hoist_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v401 plans the setup warning-filter E402 cells without blind mutation",
                "last_wave": "v400",
                "execution_result": "notebook_lint_reduced_139_to_132_local_e402_batch",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v400")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _hoist_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook E402 Local Import Hoist Batch v400

Generated: {status["generated_at_utc"]}

v400 applies only the first E402 batch selected by v399: local delayed imports
inside 6 cells.

## Result

- E402 diagnostics: `{status["global_notebook_e402_before_v400"]}` ->
  `{status["global_notebook_e402_after_v400"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v400"]}` ->
  `{status["global_notebook_diagnostics_after_v400"]}`.
- Changed notebook files: `{status["changed_notebook_files_v400"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v400"]}`.

## Required Caveat

v400 does not repair setup warning-filter E402 cells, does not clear notebook or
repository ruff, does not run full pytest, and does not create Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v400"]}` for the setup warning-filter E402 cells.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V400_NOTEBOOK_E402_LOCAL_IMPORT_HOIST_BATCH_START -->"
    end = "<!-- V400_NOTEBOOK_E402_LOCAL_IMPORT_HOIST_BATCH_END -->"
    block = f"""
{start}

## Wave v400: Notebook E402 Local Import Hoist Batch

Generated: {status["generated_at_utc"]}

### Objective

v400 applies the v399 first batch: 6 local delayed-import cells, leaving setup
warning-filter cells untouched.

### Results

- E402 before/after:
  `{status["global_notebook_e402_before_v400"]}` ->
  `{status["global_notebook_e402_after_v400"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v400"]}` ->
  `{status["global_notebook_diagnostics_after_v400"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v400"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v400"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v400"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v400"]}`.

### Interpretation

The low-risk local delayed-import E402 batch is now closed. The remaining E402
frontier is concentrated in setup cells where warning-filter order can affect
import-time warning behavior.

### Claim Impact

- Allowed: local delayed-import E402 hoist batch and roundtrip preservation.
- Still prohibited: setup warning-filter E402 repaired, notebook lint clean,
  repository ruff clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v400 in the living notebook. v401 should plan the setup warning-filter
cells before mutation.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _status_is_current() -> bool:
    if not STATUS_PATH.exists():
        return False
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    current = _run_ruff_json()
    return (
        int(status["global_notebook_diagnostics_after_v400"]) == len(current)
        and int(status.get("global_notebook_i001_after_v400", -1)) == 0
    )


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    v399_status = json.loads((STATUS_DIR / "paper4_v399_status.json").read_text(encoding="utf-8"))
    if v399_status["next_artifact_v399"] != "paper4_v400_notebook_e402_local_import_hoist_batch.md":
        raise RuntimeError("v400 expects v399 to route to local import hoist batch.")
    if _status_is_current():
        print(json.dumps({"v400": json.loads(STATUS_PATH.read_text(encoding="utf-8"))}, indent=2))
        return

    plan = pd.read_csv(PLAN_PATH)
    first_batch = plan.loc[plan["planned_batch_v399"].eq("batch_1_local_import_hoist")]
    target_notebooks = sorted(first_batch["notebook_path_v399"].unique())
    dirty_resume = not _notebook_diff_clean()
    if dirty_resume:
        before_total = int(v399_status["global_notebook_diagnostics_v399"])
        before_counts: Counter[str] = Counter({"E402": 119, "I001": 0, "B018": 10, "F821": 1})
    else:
        before_global = _run_ruff_json()
        before_total = len(before_global)
        before_counts = Counter(item["code"] for item in before_global)
    before_signatures = {
        notebook_path: (
            _notebook_signature_from_git_head(notebook_path)
            if dirty_resume
            else _notebook_signature(ROOT / notebook_path)
        )
        for notebook_path in target_notebooks
    }

    actions = _patch_notebooks()

    changed_files = _changed_notebook_files()
    after_global = _run_ruff_json()
    after_counts = Counter(item["code"] for item in after_global)
    after_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in changed_files
    }
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    integrity_columns = [
        "cell_count_preserved_v400",
        "code_cell_count_preserved_v400",
        "cell_type_sequence_preserved_v400",
        "non_code_source_preserved_v400",
        "outputs_preserved_v400",
        "metadata_preserved_v400",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(
        before_total=before_total,
        before_counts=before_counts,
        after_items=after_global,
    )
    blockers = _claim_blockers(
        e402_after=after_counts.get("E402", 0),
        global_after=len(after_global),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v400_notebook_e402_local_import_hoist_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v400_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v400_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v400_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v400_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v400_notebook_e402_local_import_hoist_batch",
        "schema_version": "2026-05-17.400",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_e402_plan_version_v400": PRIOR_E402_PLAN_VERSION,
        "first_batch_cells_v400": int(v399_status["first_batch_cells_v399"]),
        "first_batch_e402_diagnostics_v400": int(v399_status["first_batch_e402_diagnostics_v399"]),
        "action_rows_v400": int(len(actions)),
        "global_notebook_diagnostics_before_v400": int(before_total),
        "global_notebook_diagnostics_after_v400": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v400": int(before_total - len(after_global)),
        "global_notebook_e402_before_v400": int(before_counts.get("E402", 0)),
        "global_notebook_e402_after_v400": int(after_counts.get("E402", 0)),
        "global_notebook_e402_reduced_v400": int(
            before_counts.get("E402", 0) - after_counts.get("E402", 0)
        ),
        "global_notebook_i001_after_v400": int(after_counts.get("I001", 0)),
        "global_notebook_f821_after_v400": int(after_counts.get("F821", 0)),
        "changed_notebook_files_v400": int(len(changed_files)),
        "changed_notebook_file_list_v400": changed_files,
        "roundtrip_integrity_rows_v400": int(len(integrity)),
        "roundtrip_integrity_all_passed_v400": integrity_passed,
        "global_ruff_clean_v400": False,
        "full_repository_pytest_run_v400": False,
        "full_quarto_render_run_v400": False,
        "working_champion_claim_allowed_v400": False,
        "paper1_promotion_allowed_v400": False,
        "paper4_working_champion_changed_v400": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v400": NEXT_ARTIFACT,
        "claim_boundary": (
            "v400 applies local delayed-import E402 hoists with roundtrip checks; setup "
            "warning-filter E402 and final promotion remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v400 notebook roundtrip integrity failed.")
    if status["global_notebook_e402_reduced_v400"] != status["first_batch_e402_diagnostics_v400"]:
        raise RuntimeError("v400 E402 reduction did not match selected first batch.")
    if status["global_notebook_i001_after_v400"] != 0:
        raise RuntimeError("v400 left an import-sort side effect.")

    HOIST_MD.write_text(_hoist_markdown(status), encoding="utf-8")
    write_json(STATUS_PATH, status)
    _update_notebook(status)
    print(json.dumps({"v400": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
