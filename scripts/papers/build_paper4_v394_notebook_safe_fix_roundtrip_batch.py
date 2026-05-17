#!/usr/bin/env python3
"""Build Paper 4 v394 notebook safe-fix roundtrip artifacts."""

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

VERSION = 394
PRIOR_DRY_RUN_VERSION = 393
NEXT_VERSION = 395
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_unsafe_fix_review.md"
STATUS_PATH = STATUS_DIR / f"paper4_v{VERSION}_status.json"
SAFE_FIX_MD = NOTEBOOK.parent / "paper4_v394_notebook_safe_fix_roundtrip_batch.md"
SELECTED_CODES = ["F541", "I001", "W293", "F401", "B905", "SIM105", "SIM115", "UP017"]
RUFF_SAFE_FIX_COMMAND = (
    "uv run ruff check notebooks --select "
    f"{','.join(SELECTED_CODES)} --fix"
)


def _run_ruff_json(*, selected_only: bool) -> list[dict[str, Any]]:
    command = ["uv", "run", "ruff", "check", "notebooks", "--output-format", "json"]
    if selected_only:
        command[5:5] = ["--select", ",".join(SELECTED_CODES)]
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


def _run_safe_fix() -> str:
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "notebooks",
            "--select",
            ",".join(SELECTED_CODES),
            "--fix",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff notebook safe fix failed")
    return result.stdout.strip()


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _notebook_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_signature(path: Path) -> dict[str, Any]:
    payload = _notebook_payload(path)
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
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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


def _diagnostic_rows(items: list[dict[str, Any]], *, stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        fix = item.get("fix") or {}
        location = item.get("location") or {}
        rows.append(
            {
                "stage_v394": stage,
                "diagnostic_id_v394": f"{stage}_{idx:03d}",
                "notebook_path_v394": _relative_path(str(item["filename"])),
                "cell_v394": int(item.get("cell") or 0),
                "row_v394": int(location.get("row") or 0),
                "rule_code_v394": str(item["code"]),
                "message_v394": str(item["message"]),
                "has_ruff_fix_v394": bool(item.get("fix")),
                "fix_applicability_v394": str(fix.get("applicability") or "none"),
                "claim_boundary_v394": "selected notebook lint subset only",
            }
        )
    return pd.DataFrame(rows)


def _summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    keys = sorted(
        set(zip(before["rule_code_v394"], before["fix_applicability_v394"], strict=False))
        | set(zip(after["rule_code_v394"], after["fix_applicability_v394"], strict=False))
    )
    rows = []
    for rule_code, applicability in keys:
        before_count = int(
            before.loc[
                before["rule_code_v394"].eq(rule_code)
                & before["fix_applicability_v394"].eq(applicability)
            ].shape[0]
        )
        after_count = int(
            after.loc[
                after["rule_code_v394"].eq(rule_code)
                & after["fix_applicability_v394"].eq(applicability)
            ].shape[0]
        )
        rows.append(
            {
                "rule_code_v394": rule_code,
                "fix_applicability_v394": applicability,
                "diagnostic_count_before_v394": before_count,
                "diagnostic_count_after_v394": after_count,
                "diagnostics_reduced_v394": before_count - after_count,
                "action_v394": (
                    "applied_by_safe_fix_batch"
                    if applicability == "safe" and before_count > after_count
                    else "deferred_or_unfixed"
                ),
                "claim_boundary_v394": "safe applicability only; unsafe fixes deferred",
            }
        )
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
                "notebook_path_v394": notebook_path,
                "file_sha256_before_v394": before["file_sha256"],
                "file_sha256_after_v394": after["file_sha256"],
                "file_changed_v394": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v394": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v394": (
                    before["code_cell_count"] == after["code_cell_count"]
                ),
                "cell_type_sequence_preserved_v394": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v394": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v394": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v394": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v394": "code-source lint repair only",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(
    *,
    unsafe_deferred: int,
    selected_after: int,
    global_after: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v394": "unsafe_notebook_fixes_deferred",
                "blocking_v394": True,
                "evidence_count_v394": unsafe_deferred,
                "required_next_artifact_v394": NEXT_ARTIFACT,
                "claim_boundary_v394": "Ruff-unsafe notebook fixes require a separate review",
            },
            {
                "blocker_id_v394": "selected_notebook_lint_subset_not_clean",
                "blocking_v394": True,
                "evidence_count_v394": selected_after,
                "required_next_artifact_v394": NEXT_ARTIFACT,
                "claim_boundary_v394": "SIM105/B905/SIM115 remain after safe-only fix",
            },
            {
                "blocker_id_v394": "global_notebook_lint_not_clean",
                "blocking_v394": True,
                "evidence_count_v394": global_after,
                "required_next_artifact_v394": "paper4_v396_notebook_import_reorder_policy.md",
                "claim_boundary_v394": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v394": "paper4_final_promotion_forbidden",
                "blocking_v394": True,
                "evidence_count_v394": 1,
                "required_next_artifact_v394": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v394": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v394_safe_applicability_notebook_fixes_applied",
                "allowed": True,
                "artifact": "paper4_v394_notebook_safe_fix_summary.csv",
                "boundary": "13 Ruff-safe notebook fixes only",
            },
            {
                "claim_id": "v394_notebook_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v394_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v394_notebook_lint_reduced",
                "allowed": True,
                "artifact": "paper4_v394_notebook_safe_fix_summary.csv",
                "boundary": "selected lint reduced, not clean",
            },
            {
                "claim_id": "v394_unsafe_notebook_fixes_applied",
                "allowed": False,
                "artifact": "paper4_v394_claim_blockers.csv",
                "boundary": "unsafe fixes deferred",
            },
            {
                "claim_id": "v394_global_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v394_claim_blockers.csv",
                "boundary": "notebook and repository ruff still fail",
            },
            {
                "claim_id": "v394_working_champion_or_final_promotion",
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
                "claim": "v394 applies Ruff-safe notebook fixes with roundtrip preservation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v394_notebook_roundtrip_integrity.csv"
                ),
                "boundary": "13 safe-applicability fixes only; no unsafe Ruff fixes.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v394 reduces notebook lint diagnostics from 158 to 145.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v394_notebook_safe_fix_summary.csv"
                ),
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v394 applies Ruff unsafe fixes or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v394_claim_blockers.csv"
                ),
                "boundary": "Unsafe fixes are deferred and 145 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v394 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v394_claim_blockers.csv"
                ),
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
                    "v394 applies only Ruff-safe notebook fixes and records roundtrip "
                    "integrity before reviewing Ruff-unsafe candidates."
                ),
                "status": "notebook_safe_fix_roundtrip_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v395 reviews the 5 Ruff-unsafe notebook fixes without blind mutation"
                ),
                "last_wave": "v394",
                "execution_result": "notebook_lint_reduced_158_to_145_safe_only",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v394")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _safe_fix_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Safe-Fix Roundtrip Batch v394

Generated: {status["generated_at_utc"]}

v394 executes the conservative subset of the v393 dry-run manifest: only Ruff
fixes with `safe` applicability are applied to notebooks.

## Result

- Selected notebook diagnostics before: `{status["selected_diagnostics_before_v394"]}`.
- Selected notebook diagnostics after: `{status["selected_diagnostics_after_v394"]}`.
- Safe-applicability fixes applied: `{status["safe_applicability_fixes_applied_v394"]}`.
- Ruff-unsafe fixes deferred: `{status["unsafe_applicability_deferred_v394"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v394"]}` ->
  `{status["global_notebook_diagnostics_after_v394"]}`.
- Changed notebook files: `{status["changed_notebook_files_v394"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v394"]}`.

## Required Caveat

v394 does not apply Ruff-unsafe fixes, does not clear notebook lint, does not
make repository-wide ruff clean, does not run full pytest, and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v394"]}` for the 5 Ruff-unsafe notebook fixes.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V394_NOTEBOOK_SAFE_FIX_ROUNDTRIP_BATCH_START -->"
    end = "<!-- V394_NOTEBOOK_SAFE_FIX_ROUNDTRIP_BATCH_END -->"
    block = f"""
{start}

## Wave v394: Notebook Safe-Fix Roundtrip Batch

Generated: {status["generated_at_utc"]}

### Objective

v394 applies only Ruff-safe notebook fixes from the v393 dry-run manifest and
tests whether notebook structure, markdown, metadata and outputs remain stable.

### Results

- Selected notebook diagnostics before:
  `{status["selected_diagnostics_before_v394"]}`.
- Selected notebook diagnostics after:
  `{status["selected_diagnostics_after_v394"]}`.
- Safe-applicability fixes applied:
  `{status["safe_applicability_fixes_applied_v394"]}`.
- Ruff-unsafe fixes deferred:
  `{status["unsafe_applicability_deferred_v394"]}`.
- Global notebook diagnostics before:
  `{status["global_notebook_diagnostics_before_v394"]}`.
- Global notebook diagnostics after:
  `{status["global_notebook_diagnostics_after_v394"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v394"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v394"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v394"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v394"]}`.

### Interpretation

This is the first controlled notebook mutation in the lint frontier. It reduces
notebook diagnostics without touching outputs or markdown, while separating the
5 Ruff-unsafe fixes into a dedicated review wave.

### Claim Impact

- Allowed: 13 Ruff-safe notebook fixes and roundtrip preservation evidence.
- Still prohibited: unsafe fixes applied, notebook lint clean, repository ruff
  clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v394 in the living notebook. v395 should review Ruff-unsafe notebook fixes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _status_is_current() -> bool:
    if not STATUS_PATH.exists():
        return False
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    selected_now = _run_ruff_json(selected_only=True)
    return int(status["selected_diagnostics_after_v394"]) == len(selected_now)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if _status_is_current():
        print(json.dumps({"v394": json.loads(STATUS_PATH.read_text(encoding="utf-8"))}, indent=2))
        return
    if not _notebook_diff_clean():
        raise RuntimeError("v394 expects clean notebook diff before applying safe fixes.")

    v393_status = json.loads((STATUS_DIR / "paper4_v393_status.json").read_text(encoding="utf-8"))
    if v393_status["next_artifact_v393"] != "paper4_v394_notebook_safe_fix_roundtrip_batch.md":
        raise RuntimeError("v394 expects v393 to route to notebook safe fix roundtrip batch.")

    before_selected_items = _run_ruff_json(selected_only=True)
    before_global_items = _run_ruff_json(selected_only=False)
    before_selected = _diagnostic_rows(before_selected_items, stage="before")
    before_paths = sorted(set(before_selected["notebook_path_v394"]))
    before_signatures = {
        path: _notebook_signature(ROOT / path)
        for path in before_paths
    }

    _run_safe_fix()

    changed_files = _changed_notebook_files()
    after_selected_items = _run_ruff_json(selected_only=True)
    after_global_items = _run_ruff_json(selected_only=False)
    after_selected = _diagnostic_rows(after_selected_items, stage="after")
    after_signatures = {
        path: _notebook_signature(ROOT / path)
        for path in changed_files
    }

    summary = _summary(before_selected, after_selected)
    diagnostics = pd.concat([before_selected, after_selected], ignore_index=True)
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    safe_applied = int(
        summary.loc[
            summary["fix_applicability_v394"].eq("safe"),
            "diagnostics_reduced_v394",
        ].sum()
    )
    unsafe_deferred = int(before_selected["fix_applicability_v394"].eq("unsafe").sum())
    nonfixable_selected = int(before_selected["fix_applicability_v394"].eq("none").sum())
    global_before_counts = Counter(item["code"] for item in before_global_items)
    global_after_counts = Counter(item["code"] for item in after_global_items)
    blockers = _claim_blockers(
        unsafe_deferred=unsafe_deferred,
        selected_after=len(after_selected),
        global_after=len(after_global_items),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v394_notebook_safe_fix_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v394_notebook_safe_fix_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v394_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v394_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v394_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    integrity_columns = [
        "cell_count_preserved_v394",
        "code_cell_count_preserved_v394",
        "cell_type_sequence_preserved_v394",
        "non_code_source_preserved_v394",
        "outputs_preserved_v394",
        "metadata_preserved_v394",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    status = {
        "phase": "v394_notebook_safe_fix_roundtrip_batch",
        "schema_version": "2026-05-17.394",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dry_run_version_v394": PRIOR_DRY_RUN_VERSION,
        "ruff_safe_fix_command_v394": RUFF_SAFE_FIX_COMMAND,
        "selected_diagnostics_before_v394": int(len(before_selected)),
        "selected_diagnostics_after_v394": int(len(after_selected)),
        "selected_diagnostics_reduced_v394": int(len(before_selected) - len(after_selected)),
        "v393_safe_after_roundtrip_rows_v394": int(v393_status["safe_after_roundtrip_rows_v393"]),
        "safe_applicability_fixes_applied_v394": safe_applied,
        "unsafe_applicability_deferred_v394": unsafe_deferred,
        "nonfixable_selected_rows_v394": nonfixable_selected,
        "global_notebook_diagnostics_before_v394": int(len(before_global_items)),
        "global_notebook_diagnostics_after_v394": int(len(after_global_items)),
        "global_notebook_diagnostics_reduced_v394": int(
            len(before_global_items) - len(after_global_items)
        ),
        "global_notebook_e402_after_v394": int(global_after_counts.get("E402", 0)),
        "global_notebook_f821_after_v394": int(global_after_counts.get("F821", 0)),
        "global_notebook_b905_after_v394": int(global_after_counts.get("B905", 0)),
        "global_notebook_sim105_after_v394": int(global_after_counts.get("SIM105", 0)),
        "changed_notebook_files_v394": int(len(changed_files)),
        "changed_notebook_file_list_v394": changed_files,
        "roundtrip_integrity_rows_v394": int(len(integrity)),
        "roundtrip_integrity_all_passed_v394": integrity_passed,
        "unsafe_fixes_applied_v394": False,
        "global_ruff_clean_v394": False,
        "full_repository_pytest_run_v394": False,
        "full_quarto_render_run_v394": False,
        "working_champion_claim_allowed_v394": False,
        "paper1_promotion_allowed_v394": False,
        "paper4_working_champion_changed_v394": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "summary_artifact_v394": (
            "reports/paper_material/paper4/tables/"
            "paper4_v394_notebook_safe_fix_summary.csv"
        ),
        "next_artifact_v394": NEXT_ARTIFACT,
        "claim_boundary": (
            "v394 applies only Ruff-safe notebook fixes; unsafe fixes, notebook lint "
            "cleanliness and final promotion remain blocked"
        ),
    }
    if global_before_counts.get("E402", 0) != global_after_counts.get("E402", 0):
        raise RuntimeError("v394 safe-fix batch unexpectedly changed E402 count.")
    if not integrity_passed:
        raise RuntimeError("v394 notebook roundtrip integrity failed.")

    SAFE_FIX_MD.write_text(_safe_fix_markdown(status), encoding="utf-8")
    write_json(STATUS_PATH, status)
    _update_notebook(status)
    print(json.dumps({"v394": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
