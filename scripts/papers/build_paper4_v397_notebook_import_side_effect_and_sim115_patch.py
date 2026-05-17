#!/usr/bin/env python3
"""Build Paper 4 v397 notebook import side-effect and SIM115 patch artifacts."""

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

VERSION = 397
PRIOR_UNSAFE_APPLICATION_VERSION = 396
NEXT_VERSION = 398
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_historical_e402_policy.md"
STATUS_PATH = STATUS_DIR / f"paper4_v{VERSION}_status.json"
PATCH_MD = NOTEBOOK.parent / "paper4_v397_notebook_import_side_effect_and_sim115_patch.md"
TARGET_CODES = ["E402", "I001", "SIM115", "B905", "SIM105"]
TARGET_NOTEBOOKS = [
    "notebooks/01_eda_lending_club.ipynb",
    "notebooks/08_portfolio_optimization.ipynb",
    "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb",
]


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
        raise RuntimeError(result.stderr or "ruff notebook JSON probe failed")
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _notebook_signature_from_payload(
    payload: dict[str, Any],
    raw_bytes: bytes,
) -> dict[str, Any]:
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


def _move_line_before(source: list[str], *, line: str, before_line: str) -> None:
    while line in source:
        source.remove(line)
    insert_at = source.index(before_line)
    source.insert(insert_at, line)


def _ensure_blank_after(source: list[str], *, line: str) -> None:
    idx = source.index(line)
    if idx + 1 < len(source) and source[idx + 1] != "\n":
        source.insert(idx + 1, "\n")


def _patch_notebooks() -> pd.DataFrame:
    actions = [
        {
            "action_id_v397": "eda_contextlib_before_warning_filter",
            "notebook_path_v397": "notebooks/01_eda_lending_club.ipynb",
            "cell_index_v397": 2,
            "rule_target_v397": "E402/I001",
            "action_v397": "move contextlib import before warning filter side effect",
            "claim_boundary_v397": "contextlib side effect only",
        },
        {
            "action_id_v397": "gpu_cugraph_contextlib_sort",
            "notebook_path_v397": "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb",
            "cell_index_v397": 11,
            "rule_target_v397": "I001",
            "action_v397": "sort contextlib before networkx import",
            "claim_boundary_v397": "contextlib side effect only",
        },
        {
            "action_id_v397": "gpu_cuopt_contextlib_sort",
            "notebook_path_v397": "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb",
            "cell_index_v397": 13,
            "rule_target_v397": "I001",
            "action_v397": "sort contextlib before scipy imports",
            "claim_boundary_v397": "contextlib side effect only",
        },
        {
            "action_id_v397": "portfolio_feature_config_context_manager",
            "notebook_path_v397": "notebooks/08_portfolio_optimization.ipynb",
            "cell_index_v397": 4,
            "rule_target_v397": "SIM115",
            "action_v397": "replace pickle.load(open(...)) with context manager",
            "claim_boundary_v397": "manual SIM115 refactor only",
        },
    ]

    eda_path = ROOT / "notebooks/01_eda_lending_club.ipynb"
    eda = _read_notebook(eda_path)
    eda_source = eda["cells"][1]["source"]
    _move_line_before(
        eda_source,
        line="import contextlib\n",
        before_line="import warnings\n",
    )
    _write_notebook(eda_path, eda)

    gpu_path = ROOT / "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb"
    gpu = _read_notebook(gpu_path)
    cell_11 = gpu["cells"][10]["source"]
    _move_line_before(cell_11, line="import contextlib\n", before_line="import networkx as nx\n")
    _ensure_blank_after(cell_11, line="import contextlib\n")
    cell_13 = gpu["cells"][12]["source"]
    _move_line_before(
        cell_13,
        line="import contextlib\n",
        before_line="from scipy.optimize import Bounds, LinearConstraint, linprog, milp\n",
    )
    _ensure_blank_after(cell_13, line="import contextlib\n")
    _write_notebook(gpu_path, gpu)

    portfolio_path = ROOT / "notebooks/08_portfolio_optimization.ipynb"
    portfolio = _read_notebook(portfolio_path)
    source = portfolio["cells"][3]["source"]
    old_line = '    feature_config = pickle.load(open(DATA_DIR / "feature_config.pkl", "rb"))\n'
    if old_line in source:
        idx = source.index(old_line)
        source[idx : idx + 1] = [
            '    with open(DATA_DIR / "feature_config.pkl", "rb") as feature_config_file:\n',
            "        feature_config = pickle.load(feature_config_file)\n",
        ]
    _write_notebook(portfolio_path, portfolio)
    return pd.DataFrame(actions)


def _diagnostic_rows(items: list[dict[str, Any]], *, stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        location = item.get("location") or {}
        rows.append(
            {
                "stage_v397": stage,
                "diagnostic_id_v397": f"{stage}_{idx:03d}",
                "notebook_path_v397": _relative_path(str(item["filename"])),
                "cell_v397": int(item.get("cell") or 0),
                "row_v397": int(location.get("row") or 0),
                "rule_code_v397": str(item["code"]),
                "message_v397": str(item["message"]),
                "claim_boundary_v397": "v397 target lint subset only",
            }
        )
    return pd.DataFrame(rows)


def _lint_delta(
    *,
    before_global: list[dict[str, Any]],
    after_global: list[dict[str, Any]],
    before_target: list[dict[str, Any]],
    after_target: list[dict[str, Any]],
) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_global)
    after_counts = Counter(item["code"] for item in after_global)
    before_target_counts = Counter(item["code"] for item in before_target)
    after_target_counts = Counter(item["code"] for item in after_target)
    rows = [
        ("global_notebook_total", len(before_global), len(after_global)),
        ("global_notebook_e402", before_counts.get("E402", 0), after_counts.get("E402", 0)),
        ("global_notebook_i001", before_counts.get("I001", 0), after_counts.get("I001", 0)),
        ("global_notebook_sim115", before_counts.get("SIM115", 0), after_counts.get("SIM115", 0)),
        ("global_notebook_b905", before_counts.get("B905", 0), after_counts.get("B905", 0)),
        ("global_notebook_sim105", before_counts.get("SIM105", 0), after_counts.get("SIM105", 0)),
        ("target_subset_total", len(before_target), len(after_target)),
        ("target_subset_e402", before_target_counts.get("E402", 0), after_target_counts.get("E402", 0)),
        ("target_subset_i001", before_target_counts.get("I001", 0), after_target_counts.get("I001", 0)),
        (
            "target_subset_sim115",
            before_target_counts.get("SIM115", 0),
            after_target_counts.get("SIM115", 0),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v397": metric,
                "before_v397": before,
                "after_v397": after,
                "delta_v397": after - before,
                "claim_boundary_v397": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _lint_delta_from_counts(
    *,
    before_global_total: int,
    after_global: list[dict[str, Any]],
    before_counts: Counter[str],
    before_target: pd.DataFrame,
    after_target: list[dict[str, Any]],
) -> pd.DataFrame:
    after_counts = Counter(item["code"] for item in after_global)
    before_target_counts = Counter(before_target["rule_code_v397"])
    after_target_counts = Counter(item["code"] for item in after_target)
    rows = [
        ("global_notebook_total", before_global_total, len(after_global)),
        ("global_notebook_e402", before_counts.get("E402", 0), after_counts.get("E402", 0)),
        ("global_notebook_i001", before_counts.get("I001", 0), after_counts.get("I001", 0)),
        ("global_notebook_sim115", before_counts.get("SIM115", 0), after_counts.get("SIM115", 0)),
        ("global_notebook_b905", before_counts.get("B905", 0), after_counts.get("B905", 0)),
        ("global_notebook_sim105", before_counts.get("SIM105", 0), after_counts.get("SIM105", 0)),
        ("target_subset_total", len(before_target), len(after_target)),
        ("target_subset_e402", before_target_counts.get("E402", 0), after_target_counts.get("E402", 0)),
        ("target_subset_i001", before_target_counts.get("I001", 0), after_target_counts.get("I001", 0)),
        (
            "target_subset_sim115",
            before_target_counts.get("SIM115", 0),
            after_target_counts.get("SIM115", 0),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v397": metric,
                "before_v397": before,
                "after_v397": after,
                "delta_v397": after - before,
                "claim_boundary_v397": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


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
                "notebook_path_v397": notebook_path,
                "file_sha256_before_v397": before["file_sha256"],
                "file_sha256_after_v397": after["file_sha256"],
                "file_changed_v397": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v397": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v397": (
                    before["code_cell_count"] == after["code_cell_count"]
                ),
                "cell_type_sequence_preserved_v397": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v397": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v397": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v397": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v397": "targeted notebook code-source patch only",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, global_after: int, e402_after: int, f821_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v397": "historical_e402_notebook_frontier_remaining",
                "blocking_v397": True,
                "evidence_count_v397": e402_after,
                "required_next_artifact_v397": NEXT_ARTIFACT,
                "claim_boundary_v397": "historical import-order policy remains",
            },
            {
                "blocker_id_v397": "notebook_f821_execution_context_remaining",
                "blocking_v397": True,
                "evidence_count_v397": f821_after,
                "required_next_artifact_v397": "paper4_v399_notebook_execution_context_audit.md",
                "claim_boundary_v397": "undefined-name requires execution-context audit",
            },
            {
                "blocker_id_v397": "global_notebook_lint_not_clean",
                "blocking_v397": True,
                "evidence_count_v397": global_after,
                "required_next_artifact_v397": NEXT_ARTIFACT,
                "claim_boundary_v397": "global notebook lint remains blocked",
            },
            {
                "blocker_id_v397": "paper4_final_promotion_forbidden",
                "blocking_v397": True,
                "evidence_count_v397": 1,
                "required_next_artifact_v397": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v397": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v397_import_side_effects_cleaned",
                "allowed": True,
                "artifact": "paper4_v397_notebook_lint_delta.csv",
                "boundary": "v396 contextlib side effects only",
            },
            {
                "claim_id": "v397_sim115_manual_refactor_applied",
                "allowed": True,
                "artifact": "paper4_v397_notebook_patch_actions.csv",
                "boundary": "single feature_config open call only",
            },
            {
                "claim_id": "v397_notebook_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v397_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v397_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v397_claim_blockers.csv",
                "boundary": "139 notebook diagnostics remain",
            },
            {
                "claim_id": "v397_full_repository_pytest_clean_after_notebook_mutation",
                "allowed": False,
                "artifact": "paper4_v397_claim_blockers.csv",
                "boundary": "full pytest not rerun in v397",
            },
            {
                "claim_id": "v397_working_champion_or_final_promotion",
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
                "claim": "v397 cleans v396 import-lint side effects and the SIM115 finding.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v397_notebook_lint_delta.csv"
                ),
                "boundary": "Targeted cleanup only; historical notebook lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v397 reduces notebook lint diagnostics from 144 to 139.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v397_notebook_lint_delta.csv"
                ),
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v397 clears notebook lint or proves repository ruff clean.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v397_claim_blockers.csv",
                "boundary": "139 notebook diagnostics and 119 E402 findings remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v397 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v397_claim_blockers.csv",
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
                    "v397 cleans the v396 contextlib import-lint side effects and the "
                    "remaining SIM115 manual refactor."
                ),
                "status": "notebook_import_side_effect_and_sim115_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v398 decides how to handle the historical E402 notebook frontier",
                "last_wave": "v397",
                "execution_result": "notebook_lint_reduced_144_to_139_targeted_cleanup",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v397")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Import Side-Effect and SIM115 Patch v397

Generated: {status["generated_at_utc"]}

v397 cleans the small lint frontier introduced by v396 and the remaining SIM115
manual-refactor diagnostic.

## Result

- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v397"]}` ->
  `{status["global_notebook_diagnostics_after_v397"]}`.
- E402 diagnostics: `{status["global_notebook_e402_before_v397"]}` ->
  `{status["global_notebook_e402_after_v397"]}`.
- I001 diagnostics: `{status["global_notebook_i001_before_v397"]}` ->
  `{status["global_notebook_i001_after_v397"]}`.
- SIM115 diagnostics: `{status["global_notebook_sim115_before_v397"]}` ->
  `{status["global_notebook_sim115_after_v397"]}`.
- Changed notebook files: `{status["changed_notebook_files_v397"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v397"]}`.

## Required Caveat

v397 does not clear notebook lint, does not make repository-wide ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v397"]}` to govern the historical E402 notebook
frontier.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V397_NOTEBOOK_IMPORT_SIDE_EFFECT_AND_SIM115_PATCH_START -->"
    end = "<!-- V397_NOTEBOOK_IMPORT_SIDE_EFFECT_AND_SIM115_PATCH_END -->"
    block = f"""
{start}

## Wave v397: Notebook Import Side-Effect and SIM115 Patch

Generated: {status["generated_at_utc"]}

### Objective

v397 cleans the contextlib import-order side effects introduced by v396 and
patches the remaining SIM115 context-manager finding.

### Results

- Global notebook diagnostics before:
  `{status["global_notebook_diagnostics_before_v397"]}`.
- Global notebook diagnostics after:
  `{status["global_notebook_diagnostics_after_v397"]}`.
- E402 before/after:
  `{status["global_notebook_e402_before_v397"]}` ->
  `{status["global_notebook_e402_after_v397"]}`.
- I001 before/after:
  `{status["global_notebook_i001_before_v397"]}` ->
  `{status["global_notebook_i001_after_v397"]}`.
- SIM115 before/after:
  `{status["global_notebook_sim115_before_v397"]}` ->
  `{status["global_notebook_sim115_after_v397"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v397"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v397"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v397"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v397"]}`.

### Interpretation

The v396 side effects are closed and the selected B905/SIM105/SIM115 subset is
now clean. The next lint frontier is the historical E402 notebook policy rather
than a side effect from the recent repair waves.

### Claim Impact

- Allowed: targeted cleanup and roundtrip preservation.
- Still prohibited: notebook lint clean, repository ruff clean, full pytest clean
  after notebook mutation, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v397 in the living notebook. v398 should decide how to handle the
historical E402 notebook frontier without blind bulk mutation.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _status_is_current() -> bool:
    if not STATUS_PATH.exists():
        return False
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    current_global = _run_ruff_json()
    return (
        int(status["global_notebook_diagnostics_after_v397"]) == len(current_global)
        and int(status["global_notebook_i001_after_v397"]) == 0
        and int(status["global_notebook_sim115_after_v397"]) == 0
    )


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if _status_is_current():
        print(json.dumps({"v397": json.loads(STATUS_PATH.read_text(encoding="utf-8"))}, indent=2))
        return
    v396_status = json.loads((STATUS_DIR / "paper4_v396_status.json").read_text(encoding="utf-8"))
    if v396_status["next_artifact_v396"] != (
        "paper4_v397_notebook_import_side_effect_and_sim115_patch.md"
    ):
        raise RuntimeError("v397 expects v396 to route to import side-effect patch.")

    dirty_resume = not _notebook_diff_clean()
    diagnostics_path = TABLE_DIR / "paper4_v397_notebook_lint_diagnostics.csv"
    if dirty_resume and not diagnostics_path.exists():
        raise RuntimeError("v397 dirty notebook diff exists without resumable artifacts.")

    if dirty_resume:
        existing_diagnostics = pd.read_csv(diagnostics_path)
        before_target_rows = existing_diagnostics.loc[
            existing_diagnostics["stage_v397"].eq("before")
        ].copy()
        before_global_total = int(v396_status["global_notebook_diagnostics_after_v396"])
        before_counts: Counter[str] = Counter(
            {
                "E402": int(v396_status["global_notebook_e402_after_v396"]),
                "I001": int(v396_status["global_notebook_i001_after_v396"]),
                "SIM115": 1,
                "B905": int(v396_status["global_notebook_b905_after_v396"]),
                "SIM105": int(v396_status["global_notebook_sim105_after_v396"]),
            }
        )
    else:
        before_global = _run_ruff_json()
        before_target = _run_ruff_json(TARGET_CODES)
        before_target_rows = _diagnostic_rows(before_target, stage="before")
        before_global_total = len(before_global)
        before_counts = Counter(item["code"] for item in before_global)
    before_signatures = {
        notebook_path: _notebook_signature_from_git_head(notebook_path)
        for notebook_path in TARGET_NOTEBOOKS
    }

    actions = _patch_notebooks()

    changed_files = _changed_notebook_files()
    after_global = _run_ruff_json()
    after_target = _run_ruff_json(TARGET_CODES)
    after_counts = Counter(item["code"] for item in after_global)
    after_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in changed_files
    }
    diagnostics = pd.concat(
        [before_target_rows, _diagnostic_rows(after_target, stage="after")],
        ignore_index=True,
    )
    lint_delta = _lint_delta_from_counts(
        before_global_total=before_global_total,
        after_global=after_global,
        before_counts=before_counts,
        before_target=before_target_rows,
        after_target=after_target,
    )
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    integrity_columns = [
        "cell_count_preserved_v397",
        "code_cell_count_preserved_v397",
        "cell_type_sequence_preserved_v397",
        "non_code_source_preserved_v397",
        "outputs_preserved_v397",
        "metadata_preserved_v397",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    blockers = _claim_blockers(
        global_after=len(after_global),
        e402_after=after_counts.get("E402", 0),
        f821_after=after_counts.get("F821", 0),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v397_notebook_patch_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v397_notebook_lint_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v397_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v397_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v397_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v397_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v397_notebook_import_side_effect_and_sim115_patch",
        "schema_version": "2026-05-17.397",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_unsafe_application_version_v397": PRIOR_UNSAFE_APPLICATION_VERSION,
        "patch_action_rows_v397": int(len(actions)),
        "target_diagnostic_rows_before_v397": int(len(before_target_rows)),
        "target_diagnostic_rows_after_v397": int(len(after_target)),
        "target_diagnostics_reduced_v397": int(len(before_target_rows) - len(after_target)),
        "global_notebook_diagnostics_before_v397": int(before_global_total),
        "global_notebook_diagnostics_after_v397": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v397": int(before_global_total - len(after_global)),
        "global_notebook_e402_before_v397": int(before_counts.get("E402", 0)),
        "global_notebook_e402_after_v397": int(after_counts.get("E402", 0)),
        "global_notebook_i001_before_v397": int(before_counts.get("I001", 0)),
        "global_notebook_i001_after_v397": int(after_counts.get("I001", 0)),
        "global_notebook_sim115_before_v397": int(before_counts.get("SIM115", 0)),
        "global_notebook_sim115_after_v397": int(after_counts.get("SIM115", 0)),
        "global_notebook_b905_after_v397": int(after_counts.get("B905", 0)),
        "global_notebook_sim105_after_v397": int(after_counts.get("SIM105", 0)),
        "global_notebook_f821_after_v397": int(after_counts.get("F821", 0)),
        "changed_notebook_files_v397": int(len(changed_files)),
        "changed_notebook_file_list_v397": changed_files,
        "roundtrip_integrity_rows_v397": int(len(integrity)),
        "roundtrip_integrity_all_passed_v397": integrity_passed,
        "global_ruff_clean_v397": False,
        "full_repository_pytest_run_v397": False,
        "full_quarto_render_run_v397": False,
        "working_champion_claim_allowed_v397": False,
        "paper1_promotion_allowed_v397": False,
        "paper4_working_champion_changed_v397": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v397": NEXT_ARTIFACT,
        "claim_boundary": (
            "v397 cleans targeted import side effects and SIM115; historical notebook "
            "E402 and final promotion remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v397 notebook roundtrip integrity failed.")
    if status["global_notebook_b905_after_v397"] != 0 or status["global_notebook_sim105_after_v397"] != 0:
        raise RuntimeError("v397 unexpectedly reopened B905/SIM105.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_PATH, status)
    _update_notebook(status)
    print(json.dumps({"v397": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
