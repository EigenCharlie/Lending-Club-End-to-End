#!/usr/bin/env python3
"""Build Paper 4 v396 notebook Ruff-unsafe fix application artifacts."""

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

VERSION = 396
PRIOR_UNSAFE_REVIEW_VERSION = 395
NEXT_VERSION = 397
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_import_side_effect_and_sim115_patch.md"
STATUS_PATH = STATUS_DIR / f"paper4_v{VERSION}_status.json"
APPLICATION_MD = NOTEBOOK.parent / "paper4_v396_notebook_unsafe_fix_roundtrip_application.md"
UNSAFE_CODES = ["B905", "SIM105"]
RESIDUAL_SELECTED_CODES = ["B905", "SIM105", "SIM115"]
UNSAFE_FIX_COMMAND = (
    "uv run ruff check notebooks --select B905,SIM105 --fix --unsafe-fixes"
)


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


def _run_unsafe_fix() -> str:
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "notebooks",
            "--select",
            ",".join(UNSAFE_CODES),
            "--fix",
            "--unsafe-fixes",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff notebook unsafe fix failed")
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
                "stage_v396": stage,
                "diagnostic_id_v396": f"{stage}_{idx:03d}",
                "notebook_path_v396": _relative_path(str(item["filename"])),
                "cell_v396": int(item.get("cell") or 0),
                "row_v396": int(location.get("row") or 0),
                "rule_code_v396": str(item["code"]),
                "message_v396": str(item["message"]),
                "has_ruff_fix_v396": bool(item.get("fix")),
                "fix_applicability_v396": str(fix.get("applicability") or "none"),
                "claim_boundary_v396": "residual selected notebook lint subset only",
            }
        )
    return pd.DataFrame(rows)


def _summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    keys = sorted(
        set(zip(before["rule_code_v396"], before["fix_applicability_v396"], strict=False))
        | set(zip(after["rule_code_v396"], after["fix_applicability_v396"], strict=False))
    )
    rows = []
    for rule_code, applicability in keys:
        before_count = int(
            before.loc[
                before["rule_code_v396"].eq(rule_code)
                & before["fix_applicability_v396"].eq(applicability)
            ].shape[0]
        )
        after_count = int(
            after.loc[
                after["rule_code_v396"].eq(rule_code)
                & after["fix_applicability_v396"].eq(applicability)
            ].shape[0]
        )
        rows.append(
            {
                "rule_code_v396": rule_code,
                "fix_applicability_v396": applicability,
                "diagnostic_count_before_v396": before_count,
                "diagnostic_count_after_v396": after_count,
                "diagnostics_reduced_v396": before_count - after_count,
                "action_v396": (
                    "applied_by_guarded_unsafe_batch"
                    if rule_code in UNSAFE_CODES and before_count > after_count
                    else "deferred_or_unfixed"
                ),
                "claim_boundary_v396": "guarded unsafe application only",
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
                "notebook_path_v396": notebook_path,
                "file_sha256_before_v396": before["file_sha256"],
                "file_sha256_after_v396": after["file_sha256"],
                "file_changed_v396": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v396": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v396": (
                    before["code_cell_count"] == after["code_cell_count"]
                ),
                "cell_type_sequence_preserved_v396": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v396": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v396": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v396": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v396": "code-source unsafe lint repair only",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, selected_after: int, global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v396": "sim115_manual_refactor_remaining",
                "blocking_v396": True,
                "evidence_count_v396": selected_after,
                "required_next_artifact_v396": NEXT_ARTIFACT,
                "claim_boundary_v396": "nonfixable selected lint remains",
            },
            {
                "blocker_id_v396": "unsafe_fix_import_side_effects_remaining",
                "blocking_v396": True,
                "evidence_count_v396": 4,
                "required_next_artifact_v396": NEXT_ARTIFACT,
                "claim_boundary_v396": "contextlib imports introduced E402/I001 cleanup work",
            },
            {
                "blocker_id_v396": "global_notebook_lint_not_clean",
                "blocking_v396": True,
                "evidence_count_v396": global_after,
                "required_next_artifact_v396": "paper4_v398_notebook_import_reorder_policy.md",
                "claim_boundary_v396": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v396": "full_repository_ruff_not_clean",
                "blocking_v396": True,
                "evidence_count_v396": 1,
                "required_next_artifact_v396": "paper4_v399_repository_lint_refresh.md",
                "claim_boundary_v396": "v396 only reduces notebook lint subset",
            },
            {
                "blocker_id_v396": "paper4_final_promotion_forbidden",
                "blocking_v396": True,
                "evidence_count_v396": 1,
                "required_next_artifact_v396": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v396": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v396_approved_unsafe_notebook_fixes_applied",
                "allowed": True,
                "artifact": "paper4_v396_notebook_unsafe_fix_summary.csv",
                "boundary": "5 B905/SIM105 fixes only",
            },
            {
                "claim_id": "v396_notebook_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v396_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v396_selected_notebook_lint_reduced_to_sim115",
                "allowed": True,
                "artifact": "paper4_v396_notebook_unsafe_fix_summary.csv",
                "boundary": "selected subset still has SIM115",
            },
            {
                "claim_id": "v396_import_lint_side_effects_detected",
                "allowed": True,
                "artifact": "paper4_v396_claim_blockers.csv",
                "boundary": "4 import-order side effects routed to v397",
            },
            {
                "claim_id": "v396_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v396_claim_blockers.csv",
                "boundary": "144 notebook diagnostics remain",
            },
            {
                "claim_id": "v396_full_repository_pytest_clean_after_notebook_mutation",
                "allowed": False,
                "artifact": "paper4_v396_claim_blockers.csv",
                "boundary": "full pytest not rerun in v396",
            },
            {
                "claim_id": "v396_working_champion_or_final_promotion",
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
                "claim": "v396 applies 5 approved Ruff-unsafe notebook fixes with roundtrip preservation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v396_notebook_roundtrip_integrity.csv"
                ),
                "boundary": "Only B905/SIM105 fixes approved by v395.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v396 reduces notebook lint diagnostics from 145 to 144.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v396_notebook_unsafe_fix_summary.csv"
                ),
                "boundary": "Net reduction only; 4 import-lint side effects are documented.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v396 clears notebook lint or proves full repository ruff clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v396_claim_blockers.csv"
                ),
                "boundary": "SIM115, import side effects and 144 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v396 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v396_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    stale_claims = {
        "v396 reduces notebook lint diagnostics from 145 to 140.",
        "v396 clears notebook lint or proves full repository ruff clean.",
    }
    out = current.loc[
        ~current["claim"].isin(set(additions["claim"]) | stale_claims)
    ].copy()
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
                    "v396 applies the v395-approved B905/SIM105 notebook fixes and "
                    "records roundtrip integrity."
                ),
                "status": "notebook_unsafe_fix_roundtrip_application_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v397 cleans contextlib import side effects and handles the SIM115 manual refactor"
                ),
                "last_wave": "v396",
                "execution_result": "selected_lint_reduced_6_to_1_global_lint_145_to_144",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v396")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _application_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Ruff-Unsafe Fix Roundtrip Application v396

Generated: {status["generated_at_utc"]}

v396 applies the 5 B905/SIM105 fixes reviewed in v395 and validates notebook
roundtrip integrity.

## Result

- Residual selected diagnostics before: `{status["selected_diagnostics_before_v396"]}`.
- Residual selected diagnostics after: `{status["selected_diagnostics_after_v396"]}`.
- Approved unsafe fixes applied: `{status["approved_unsafe_fixes_applied_v396"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v396"]}` ->
  `{status["global_notebook_diagnostics_after_v396"]}`.
- Import-lint side effects detected: `{status["global_notebook_import_lint_side_effect_rows_v396"]}`.
- Changed notebook files: `{status["changed_notebook_files_v396"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v396"]}`.

## Required Caveat

v396 does not clear notebook lint, does not make repository-wide ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v396"]}` for the remaining SIM115 manual refactor
and contextlib import-lint side effects.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V396_NOTEBOOK_UNSAFE_FIX_ROUNDTRIP_APPLICATION_START -->"
    end = "<!-- V396_NOTEBOOK_UNSAFE_FIX_ROUNDTRIP_APPLICATION_END -->"
    block = f"""
{start}

## Wave v396: Notebook Ruff-Unsafe Fix Roundtrip Application

Generated: {status["generated_at_utc"]}

### Objective

v396 applies the 5 B905/SIM105 fixes approved by v395 and repeats v394's
roundtrip integrity checks.

### Results

- Selected diagnostics before:
  `{status["selected_diagnostics_before_v396"]}`.
- Selected diagnostics after:
  `{status["selected_diagnostics_after_v396"]}`.
- Approved unsafe fixes applied:
  `{status["approved_unsafe_fixes_applied_v396"]}`.
- Global notebook diagnostics before:
  `{status["global_notebook_diagnostics_before_v396"]}`.
- Global notebook diagnostics after:
  `{status["global_notebook_diagnostics_after_v396"]}`.
- Import-lint side effects detected:
  `{status["global_notebook_import_lint_side_effect_rows_v396"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v396"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v396"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v396"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v396"]}`.

### Interpretation

The B905/SIM105 residual lint is closed without changing outputs, markdown,
metadata or notebook structure. The selected lint frontier is now reduced to the
single SIM115 manual-refactor diagnostic, but the inserted `contextlib` imports
create 4 import-lint side effects that must be cleaned separately.

### Claim Impact

- Allowed: 5 reviewed unsafe fixes applied with roundtrip preservation.
- Still prohibited: notebook lint clean, repository ruff clean, full pytest clean
  after notebook mutation, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v396 in the living notebook. v397 should clean the contextlib import side
effects and handle the remaining SIM115 manual context-manager refactor.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def _status_is_current() -> bool:
    if not STATUS_PATH.exists():
        return False
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    selected_now = _run_ruff_json(RESIDUAL_SELECTED_CODES)
    return int(status["selected_diagnostics_after_v396"]) == len(selected_now)


def _build_status(
    *,
    started: datetime,
    v395_status: dict[str, Any],
    before_selected: pd.DataFrame,
    after_selected: pd.DataFrame,
    before_global_count: int,
    before_global_e402: int,
    after_global_items: list[dict[str, Any]],
    summary: pd.DataFrame,
    integrity: pd.DataFrame,
    changed_files: list[str],
) -> dict[str, Any]:
    integrity_columns = [
        "cell_count_preserved_v396",
        "code_cell_count_preserved_v396",
        "cell_type_sequence_preserved_v396",
        "non_code_source_preserved_v396",
        "outputs_preserved_v396",
        "metadata_preserved_v396",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    global_after_counts = Counter(item["code"] for item in after_global_items)
    approved_applied = int(
        summary.loc[
            summary["rule_code_v396"].isin(UNSAFE_CODES),
            "diagnostics_reduced_v396",
        ].sum()
    )
    expected_after_without_side_effects = before_global_count - approved_applied
    import_side_effect_rows = max(0, len(after_global_items) - expected_after_without_side_effects)
    return {
        "phase": "v396_notebook_unsafe_fix_roundtrip_application",
        "schema_version": "2026-05-17.396",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_unsafe_review_version_v396": PRIOR_UNSAFE_REVIEW_VERSION,
        "unsafe_fix_command_v396": UNSAFE_FIX_COMMAND,
        "selected_diagnostics_before_v396": int(len(before_selected)),
        "selected_diagnostics_after_v396": int(len(after_selected)),
        "selected_diagnostics_reduced_v396": int(len(before_selected) - len(after_selected)),
        "approved_unsafe_fixes_approved_by_v395": int(
            v395_status["approved_for_roundtrip_application_v395"]
        ),
        "approved_unsafe_fixes_applied_v396": approved_applied,
        "sim115_remaining_v396": int(global_after_counts.get("SIM115", 0)),
        "global_notebook_diagnostics_before_v396": before_global_count,
        "global_notebook_diagnostics_after_v396": int(len(after_global_items)),
        "global_notebook_diagnostics_reduced_v396": int(
            before_global_count - len(after_global_items)
        ),
        "global_notebook_e402_before_v396": before_global_e402,
        "global_notebook_e402_after_v396": int(global_after_counts.get("E402", 0)),
        "global_notebook_e402_delta_v396": int(
            global_after_counts.get("E402", 0) - before_global_e402
        ),
        "global_notebook_i001_after_v396": int(global_after_counts.get("I001", 0)),
        "global_notebook_import_lint_side_effect_rows_v396": int(import_side_effect_rows),
        "global_notebook_f821_after_v396": int(global_after_counts.get("F821", 0)),
        "global_notebook_b905_after_v396": int(global_after_counts.get("B905", 0)),
        "global_notebook_sim105_after_v396": int(global_after_counts.get("SIM105", 0)),
        "changed_notebook_files_v396": int(len(changed_files)),
        "changed_notebook_file_list_v396": changed_files,
        "roundtrip_integrity_rows_v396": int(len(integrity)),
        "roundtrip_integrity_all_passed_v396": integrity_passed,
        "global_ruff_clean_v396": False,
        "full_repository_pytest_run_v396": False,
        "full_quarto_render_run_v396": False,
        "working_champion_claim_allowed_v396": False,
        "paper1_promotion_allowed_v396": False,
        "paper4_working_champion_changed_v396": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v396": NEXT_ARTIFACT,
        "claim_boundary": (
            "v396 applies v395-approved unsafe fixes with roundtrip checks; import "
            "side effects, notebook lint cleanliness and final promotion remain blocked"
        ),
    }


def _finalize_from_existing_application(
    *,
    started: datetime,
    v395_status: dict[str, Any],
) -> None:
    diagnostics_path = TABLE_DIR / "paper4_v396_notebook_unsafe_fix_diagnostics.csv"
    summary_path = TABLE_DIR / "paper4_v396_notebook_unsafe_fix_summary.csv"
    integrity_path = TABLE_DIR / "paper4_v396_notebook_roundtrip_integrity.csv"
    if not diagnostics_path.exists() or not summary_path.exists() or not integrity_path.exists():
        raise RuntimeError("v396 dirty notebook diff exists without resumable artifacts.")

    diagnostics = pd.read_csv(diagnostics_path)
    before_selected = diagnostics.loc[diagnostics["stage_v396"].eq("before")].copy()
    after_selected = _diagnostic_rows(_run_ruff_json(RESIDUAL_SELECTED_CODES), stage="after")
    diagnostics = pd.concat([before_selected, after_selected], ignore_index=True)
    summary = pd.read_csv(summary_path)
    integrity = pd.read_csv(integrity_path)
    changed_files = _changed_notebook_files()
    after_global_items = _run_ruff_json()
    blockers = _claim_blockers(
        selected_after=len(after_selected),
        global_after=len(after_global_items),
    )
    claim_matrix = _claim_matrix()

    write_csv(diagnostics_path, diagnostics)
    write_csv(summary_path, summary)
    write_csv(integrity_path, integrity)
    write_csv(TABLE_DIR / "paper4_v396_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v396_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = _build_status(
        started=started,
        v395_status=v395_status,
        before_selected=before_selected,
        after_selected=after_selected,
        before_global_count=int(v395_status["global_notebook_diagnostics_v395"]),
        before_global_e402=int(v395_status["global_notebook_e402_v395"]),
        after_global_items=after_global_items,
        summary=summary,
        integrity=integrity,
        changed_files=changed_files,
    )
    if not status["roundtrip_integrity_all_passed_v396"]:
        raise RuntimeError("v396 notebook roundtrip integrity failed.")

    APPLICATION_MD.write_text(_application_markdown(status), encoding="utf-8")
    write_json(STATUS_PATH, status)
    _update_notebook(status)
    print(json.dumps({"v396": status}, indent=2, sort_keys=True))


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if _status_is_current():
        print(json.dumps({"v396": json.loads(STATUS_PATH.read_text(encoding="utf-8"))}, indent=2))
        return

    v395_status = json.loads((STATUS_DIR / "paper4_v395_status.json").read_text(encoding="utf-8"))
    if v395_status["next_artifact_v395"] != (
        "paper4_v396_notebook_unsafe_fix_roundtrip_application.md"
    ):
        raise RuntimeError("v396 expects v395 to route to unsafe fix application.")
    if not _notebook_diff_clean():
        _finalize_from_existing_application(started=started, v395_status=v395_status)
        return

    before_selected_items = _run_ruff_json(RESIDUAL_SELECTED_CODES)
    before_global_items = _run_ruff_json()
    before_selected = _diagnostic_rows(before_selected_items, stage="before")
    before_paths = sorted(
        set(
            before_selected.loc[
                before_selected["rule_code_v396"].isin(UNSAFE_CODES),
                "notebook_path_v396",
            ]
        )
    )
    before_signatures = {path: _notebook_signature(ROOT / path) for path in before_paths}

    _run_unsafe_fix()

    changed_files = _changed_notebook_files()
    after_selected_items = _run_ruff_json(RESIDUAL_SELECTED_CODES)
    after_global_items = _run_ruff_json()
    after_selected = _diagnostic_rows(after_selected_items, stage="after")
    after_signatures = {path: _notebook_signature(ROOT / path) for path in changed_files}

    summary = _summary(before_selected, after_selected)
    diagnostics = pd.concat([before_selected, after_selected], ignore_index=True)
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    blockers = _claim_blockers(
        selected_after=len(after_selected),
        global_after=len(after_global_items),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v396_notebook_unsafe_fix_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v396_notebook_unsafe_fix_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v396_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v396_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v396_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    global_before_counts = Counter(item["code"] for item in before_global_items)
    status = _build_status(
        started=started,
        v395_status=v395_status,
        before_selected=before_selected,
        after_selected=after_selected,
        before_global_count=int(len(before_global_items)),
        before_global_e402=int(global_before_counts.get("E402", 0)),
        after_global_items=after_global_items,
        summary=summary,
        integrity=integrity,
        changed_files=changed_files,
    )
    if not status["roundtrip_integrity_all_passed_v396"]:
        raise RuntimeError("v396 notebook roundtrip integrity failed.")

    APPLICATION_MD.write_text(_application_markdown(status), encoding="utf-8")
    write_json(STATUS_PATH, status)
    _update_notebook(status)
    print(json.dumps({"v396": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
