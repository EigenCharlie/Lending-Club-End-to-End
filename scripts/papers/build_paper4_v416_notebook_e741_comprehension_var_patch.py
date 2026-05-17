#!/usr/bin/env python3
"""Build Paper 4 v416 E741 comprehension-variable notebook patch artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
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

VERSION = 416
PRIOR_STYLE_TRIAGE_VERSION = 415
TARGET_NOTEBOOK = "notebooks/03_pd_modeling.ipynb"
TARGET_CELL = 37
OLD_LINE = "labels = [l.get_label() for l in lines]\n"
NEW_LINE = "labels = [line.get_label() for line in lines]\n"
NEXT_ARTIFACT = "paper4_v417_notebook_sim108_conditional_expr_review.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v416_notebook_e741_comprehension_var_patch.md"


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


def _changed_notebook_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _read_notebook(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _write_notebook(path: str, payload: dict[str, Any]) -> None:
    (ROOT / path).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _notebook_signature(path: str) -> dict[str, Any]:
    raw = (ROOT / path).read_bytes()
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


def _patch_notebook() -> pd.DataFrame:
    payload = _read_notebook(TARGET_NOTEBOOK)
    source = payload["cells"][TARGET_CELL - 1]["source"]
    if OLD_LINE not in source:
        raise RuntimeError("expected E741 line missing")
    source[source.index(OLD_LINE)] = NEW_LINE
    _write_notebook(TARGET_NOTEBOOK, payload)
    return pd.DataFrame(
        [
            {
                "action_id_v416": "e741_comprehension_var_patch_01",
                "notebook_path_v416": TARGET_NOTEBOOK,
                "cell_v416": TARGET_CELL,
                "old_expression_v416": OLD_LINE.strip(),
                "new_expression_v416": NEW_LINE.strip(),
                "mutation_applied_v416": True,
                "claim_boundary_v416": "E741 comprehension-variable rename only",
            }
        ]
    )


def _roundtrip_integrity(before: dict[str, Any], after: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notebook_path_v416": TARGET_NOTEBOOK,
                "file_sha256_before_v416": before["file_sha256"],
                "file_sha256_after_v416": after["file_sha256"],
                "file_changed_v416": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v416": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v416": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v416": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v416": before["non_code_source_hash"] == after["non_code_source_hash"],
                "outputs_preserved_v416": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v416": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v416": "single-line code-source patch only",
            }
        ]
    )


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_e741", before_counts.get("E741", 0), after_counts.get("E741", 0)),
        ("global_notebook_sim108", before_counts.get("SIM108", 0), after_counts.get("SIM108", 0)),
        ("global_notebook_e712", before_counts.get("E712", 0), after_counts.get("E712", 0)),
        ("global_notebook_sim102", before_counts.get("SIM102", 0), after_counts.get("SIM102", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v416": metric,
                "before_v416": before,
                "after_v416": after,
                "delta_v416": after - before,
                "claim_boundary_v416": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(global_after: int, sim108_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v416": "sim108_conditional_expr_review_deferred",
                "blocking_v416": True,
                "evidence_count_v416": sim108_after,
                "required_next_artifact_v416": NEXT_ARTIFACT,
                "claim_boundary_v416": "SIM108 refactors need review before mutation",
            },
            {
                "blocker_id_v416": "style_notebook_lint_remaining",
                "blocking_v416": True,
                "evidence_count_v416": global_after,
                "required_next_artifact_v416": NEXT_ARTIFACT,
                "claim_boundary_v416": "style notebook lint remains",
            },
            {
                "blocker_id_v416": "paper4_final_promotion_forbidden",
                "blocking_v416": True,
                "evidence_count_v416": 1,
                "required_next_artifact_v416": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v416": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v416_e741_comprehension_var_patch_applied",
                "allowed": True,
                "artifact": "paper4_v416_notebook_e741_comprehension_var_actions.csv",
                "boundary": "one list-comprehension variable rename",
            },
            {
                "claim_id": "v416_e741_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v416_notebook_lint_delta.csv",
                "boundary": "E741 only",
            },
            {
                "claim_id": "v416_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v416_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v416_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v416_claim_blockers.csv",
                "boundary": "5 notebook diagnostics remain",
            },
            {
                "claim_id": "v416_working_champion_or_final_promotion",
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
                "claim": "v416 clears notebook E741 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v416_notebook_lint_delta.csv",
                "boundary": "E741 only; other style notebook lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v416 reduces notebook lint diagnostics from 6 to 5.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v416_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v416 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v416_claim_blockers.csv",
                "boundary": "5 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v416 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v416_claim_blockers.csv",
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
                "executable_item": "v416 applies the notebook 03 E741 comprehension-variable rename.",
                "status": "notebook_e741_comprehension_var_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v417 reviews SIM108 conditional-expression refactors before mutation",
                "last_wave": "v416",
                "execution_result": "notebook_e741_cleared_lint_reduced_6_to_5",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v416")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook E741 Comprehension-Variable Patch v416

Generated: {status["generated_at_utc"]}

v416 applies the E741 patch selected by v415.

## Result

- E741 diagnostics: `{status["global_notebook_e741_before_v416"]}` ->
  `{status["global_notebook_e741_after_v416"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v416"]}` ->
  `{status["global_notebook_diagnostics_after_v416"]}`.
- Changed notebook files: `{status["changed_notebook_files_v416"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v416"]}`.

## Required Caveat

v416 does not clear remaining SIM/E712 style lint and does not create Paper 4
final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V416_NOTEBOOK_E741_COMPREHENSION_VAR_PATCH_START -->"
    end = "<!-- V416_NOTEBOOK_E741_COMPREHENSION_VAR_PATCH_END -->"
    block = f"""
{start}

## Wave v416: Notebook E741 Comprehension-Variable Patch

Generated: {status["generated_at_utc"]}

### Objective

v416 renames the ambiguous list-comprehension variable `l` to `line`.

### Results

- E741 before/after:
  `{status["global_notebook_e741_before_v416"]}` ->
  `{status["global_notebook_e741_after_v416"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v416"]}` ->
  `{status["global_notebook_diagnostics_after_v416"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v416"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v416"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v416"]}`.

### Interpretation

E741 is now closed. The remaining style frontier is SIM108, E712 and SIM102.

### Claim Impact

- Allowed: E741 cleared and notebook lint reduced to 5 diagnostics.
- Still prohibited: notebook lint clean, repository ruff clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v416 in the living notebook. v417 should review SIM108 refactors before
mutation.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v416 expects clean notebook diff before mutation.")

    v415_status = json.loads((STATUS_DIR / "paper4_v415_status.json").read_text(encoding="utf-8"))
    if v415_status["next_artifact_v415"] != "paper4_v416_notebook_e741_comprehension_var_patch.md":
        raise RuntimeError("v416 expects v415 to route to E741 patch.")

    before_global = _run_ruff_json()
    before_counts = Counter(item["code"] for item in before_global)
    before_signature = _notebook_signature(TARGET_NOTEBOOK)
    actions = _patch_notebook()
    changed_files = _changed_notebook_files()
    after_global = _run_ruff_json()
    after_counts = Counter(item["code"] for item in after_global)
    after_signature = _notebook_signature(TARGET_NOTEBOOK)
    integrity = _roundtrip_integrity(before_signature, after_signature)
    integrity_columns = [
        "cell_count_preserved_v416",
        "code_cell_count_preserved_v416",
        "cell_type_sequence_preserved_v416",
        "non_code_source_preserved_v416",
        "outputs_preserved_v416",
        "metadata_preserved_v416",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(
        global_after=len(after_global),
        sim108_after=after_counts.get("SIM108", 0),
    )
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v416_notebook_e741_comprehension_var_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v416_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v416_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v416_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v416_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v416_notebook_e741_comprehension_var_patch",
        "schema_version": "2026-05-17.416",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_style_triage_version_v416": PRIOR_STYLE_TRIAGE_VERSION,
        "action_rows_v416": int(len(actions)),
        "global_notebook_diagnostics_before_v416": int(len(before_global)),
        "global_notebook_diagnostics_after_v416": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v416": int(len(before_global) - len(after_global)),
        "global_notebook_e741_before_v416": int(before_counts.get("E741", 0)),
        "global_notebook_e741_after_v416": int(after_counts.get("E741", 0)),
        "global_notebook_e741_reduced_v416": int(
            before_counts.get("E741", 0) - after_counts.get("E741", 0)
        ),
        "changed_notebook_files_v416": int(len(changed_files)),
        "changed_notebook_file_list_v416": changed_files,
        "roundtrip_integrity_rows_v416": int(len(integrity)),
        "roundtrip_integrity_all_passed_v416": integrity_passed,
        "global_ruff_clean_v416": False,
        "full_repository_pytest_run_v416": False,
        "full_quarto_render_run_v416": False,
        "working_champion_claim_allowed_v416": False,
        "paper1_promotion_allowed_v416": False,
        "paper4_working_champion_changed_v416": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v416": NEXT_ARTIFACT,
        "claim_boundary": (
            "v416 clears E741 with a local variable rename; remaining lint and final "
            "promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v416 notebook roundtrip integrity failed.")
    if status["global_notebook_e741_after_v416"] != 0:
        raise RuntimeError("v416 did not clear E741.")
    if status["global_notebook_diagnostics_after_v416"] != 5:
        raise RuntimeError("v416 did not reach expected 5 diagnostics.")
    if changed_files != [TARGET_NOTEBOOK]:
        raise RuntimeError("v416 changed an unexpected notebook set.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v416": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
