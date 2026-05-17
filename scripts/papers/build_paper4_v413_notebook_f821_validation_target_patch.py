#!/usr/bin/env python3
"""Build Paper 4 v413 F821 validation-target notebook patch artifacts."""

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

VERSION = 413
PRIOR_F821_AUDIT_VERSION = 412
TARGET_NOTEBOOK = "notebooks/02_feature_engineering.ipynb"
TARGET_CELL = 32
NEXT_ARTIFACT = "paper4_v414_notebook_f821_post_patch_pytest_probe.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v413_notebook_f821_validation_target_patch.md"


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


def _find_block(source: list[str], block: list[str]) -> int:
    for idx in range(0, len(source) - len(block) + 1):
        if source[idx : idx + len(block)] == block:
            return idx
    raise RuntimeError("expected F821 validation block was not found")


def _patch_notebook() -> pd.DataFrame:
    payload = _read_notebook(TARGET_NOTEBOOK)
    source = payload["cells"][TARGET_CELL - 1]["source"]
    old_block = [
        "# Validate with Pandera schema\n",
        "\n",
        "try:\n",
        "    loan_master_schema.validate(train_fe, lazy=True)\n",
        "    print(\"Pandera validation PASSED for train_fe\")\n",
    ]
    new_block = [
        "# Validate with Pandera schema\n",
        "\n",
        "validation_target = script_train if \"script_train\" in globals() else train\n",
        "validation_target_name = \"script_train\" if \"script_train\" in globals() else \"train\"\n",
        "\n",
        "try:\n",
        "    loan_master_schema.validate(validation_target, lazy=True)\n",
        "    print(f\"Pandera validation PASSED for {validation_target_name}\")\n",
    ]
    start_idx = _find_block(source, old_block)
    source[start_idx : start_idx + len(old_block)] = new_block
    _write_notebook(TARGET_NOTEBOOK, payload)
    return pd.DataFrame(
        [
            {
                "action_id_v413": "f821_validation_target_patch_01",
                "notebook_path_v413": TARGET_NOTEBOOK,
                "cell_v413": TARGET_CELL,
                "old_reference_v413": "train_fe",
                "new_reference_v413": "validation_target",
                "validation_target_rule_v413": "script_train if available else train",
                "mutation_applied_v413": True,
                "claim_boundary_v413": "F821 validation-target patch only",
            }
        ]
    )


def _roundtrip_integrity(
    before: dict[str, Any],
    after: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notebook_path_v413": TARGET_NOTEBOOK,
                "file_sha256_before_v413": before["file_sha256"],
                "file_sha256_after_v413": after["file_sha256"],
                "file_changed_v413": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v413": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v413": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v413": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v413": before["non_code_source_hash"] == after["non_code_source_hash"],
                "outputs_preserved_v413": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v413": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v413": "single-cell code-source patch only",
            }
        ]
    )


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_f821", before_counts.get("F821", 0), after_counts.get("F821", 0)),
        ("global_notebook_e741", before_counts.get("E741", 0), after_counts.get("E741", 0)),
        ("global_notebook_sim108", before_counts.get("SIM108", 0), after_counts.get("SIM108", 0)),
        ("global_notebook_e712", before_counts.get("E712", 0), after_counts.get("E712", 0)),
        ("global_notebook_sim102", before_counts.get("SIM102", 0), after_counts.get("SIM102", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v413": metric,
                "before_v413": before,
                "after_v413": after,
                "delta_v413": after - before,
                "claim_boundary_v413": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v413": "post_f821_pytest_not_run",
                "blocking_v413": True,
                "evidence_count_v413": 1,
                "required_next_artifact_v413": NEXT_ARTIFACT,
                "claim_boundary_v413": "post-validation-target pytest deferred to v414",
            },
            {
                "blocker_id_v413": "global_notebook_lint_not_clean",
                "blocking_v413": True,
                "evidence_count_v413": global_after,
                "required_next_artifact_v413": NEXT_ARTIFACT,
                "claim_boundary_v413": "style notebook lint remains",
            },
            {
                "blocker_id_v413": "paper4_final_promotion_forbidden",
                "blocking_v413": True,
                "evidence_count_v413": 1,
                "required_next_artifact_v413": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v413": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v413_f821_validation_target_patch_applied",
                "allowed": True,
                "artifact": "paper4_v413_notebook_f821_validation_target_actions.csv",
                "boundary": "one explicit validation-target patch",
            },
            {
                "claim_id": "v413_f821_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v413_notebook_lint_delta.csv",
                "boundary": "F821 only",
            },
            {
                "claim_id": "v413_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v413_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v413_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v413_claim_blockers.csv",
                "boundary": "6 notebook diagnostics remain",
            },
            {
                "claim_id": "v413_post_f821_pytest_passed",
                "allowed": False,
                "artifact": "paper4_v413_claim_blockers.csv",
                "boundary": "post-patch pytest deferred to v414",
            },
            {
                "claim_id": "v413_working_champion_or_final_promotion",
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
                "claim": "v413 clears notebook F821 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v413_notebook_lint_delta.csv",
                "boundary": "F821 only; style notebook lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v413 reduces notebook lint diagnostics from 7 to 6.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v413_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v413 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v413_claim_blockers.csv",
                "boundary": "6 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v413 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v413_claim_blockers.csv",
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
                "executable_item": "v413 applies the F821 validation-target patch in notebook 02.",
                "status": "notebook_f821_validation_target_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v414 full repository pytest passes after F821 patch",
                "last_wave": "v413",
                "execution_result": "notebook_f821_cleared_lint_reduced_7_to_6",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v413")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook F821 Validation-Target Patch v413

Generated: {status["generated_at_utc"]}

v413 applies the validation-target patch selected by v412.

## Result

- F821 diagnostics: `{status["global_notebook_f821_before_v413"]}` ->
  `{status["global_notebook_f821_after_v413"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v413"]}` ->
  `{status["global_notebook_diagnostics_after_v413"]}`.
- Changed notebook files: `{status["changed_notebook_files_v413"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v413"]}`.

## Required Caveat

v413 does not clear remaining style notebook lint, does not run post-patch
pytest, and does not create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V413_NOTEBOOK_F821_VALIDATION_TARGET_PATCH_START -->"
    end = "<!-- V413_NOTEBOOK_F821_VALIDATION_TARGET_PATCH_END -->"
    block = f"""
{start}

## Wave v413: Notebook F821 Validation-Target Patch

Generated: {status["generated_at_utc"]}

### Objective

v413 replaces the undefined `train_fe` validation reference with an explicit
`validation_target`.

### Results

- F821 before/after:
  `{status["global_notebook_f821_before_v413"]}` ->
  `{status["global_notebook_f821_after_v413"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v413"]}` ->
  `{status["global_notebook_diagnostics_after_v413"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v413"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v413"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v413"]}`.

### Interpretation

The semantic F821 blocker is now closed. The remaining notebook lint frontier is
style-only and still requires post-patch pytest before further cleanup.

### Claim Impact

- Allowed: F821 cleared and notebook lint reduced to 6 diagnostics.
- Still prohibited: notebook lint clean, repository ruff clean, post-patch pytest
  passed, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v413 in the living notebook. v414 should run post-F821 full pytest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v413 expects clean notebook diff before mutation.")

    v412_status = json.loads((STATUS_DIR / "paper4_v412_status.json").read_text(encoding="utf-8"))
    if v412_status["next_artifact_v412"] != "paper4_v413_notebook_f821_validation_target_patch.md":
        raise RuntimeError("v413 expects v412 to route to validation-target patch.")

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
        "cell_count_preserved_v413",
        "code_cell_count_preserved_v413",
        "cell_type_sequence_preserved_v413",
        "non_code_source_preserved_v413",
        "outputs_preserved_v413",
        "metadata_preserved_v413",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(len(after_global))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v413_notebook_f821_validation_target_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v413_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v413_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v413_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v413_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v413_notebook_f821_validation_target_patch",
        "schema_version": "2026-05-17.413",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_f821_audit_version_v413": PRIOR_F821_AUDIT_VERSION,
        "action_rows_v413": int(len(actions)),
        "global_notebook_diagnostics_before_v413": int(len(before_global)),
        "global_notebook_diagnostics_after_v413": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v413": int(len(before_global) - len(after_global)),
        "global_notebook_f821_before_v413": int(before_counts.get("F821", 0)),
        "global_notebook_f821_after_v413": int(after_counts.get("F821", 0)),
        "global_notebook_f821_reduced_v413": int(
            before_counts.get("F821", 0) - after_counts.get("F821", 0)
        ),
        "changed_notebook_files_v413": int(len(changed_files)),
        "changed_notebook_file_list_v413": changed_files,
        "roundtrip_integrity_rows_v413": int(len(integrity)),
        "roundtrip_integrity_all_passed_v413": integrity_passed,
        "global_ruff_clean_v413": False,
        "full_repository_pytest_run_v413": False,
        "full_quarto_render_run_v413": False,
        "working_champion_claim_allowed_v413": False,
        "paper1_promotion_allowed_v413": False,
        "paper4_working_champion_changed_v413": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v413": NEXT_ARTIFACT,
        "claim_boundary": (
            "v413 clears F821 with an explicit validation target; remaining lint and "
            "final promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v413 notebook roundtrip integrity failed.")
    if status["global_notebook_f821_after_v413"] != 0:
        raise RuntimeError("v413 did not clear F821.")
    if status["global_notebook_diagnostics_after_v413"] != 6:
        raise RuntimeError("v413 did not reach expected 6 diagnostics.")
    if changed_files != [TARGET_NOTEBOOK]:
        raise RuntimeError("v413 changed an unexpected notebook set.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v413": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
