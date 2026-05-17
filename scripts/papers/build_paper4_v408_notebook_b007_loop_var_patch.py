#!/usr/bin/env python3
"""Build Paper 4 v408 notebook B007 loop-variable patch artifacts."""

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

VERSION = 408
PRIOR_TRIAGE_VERSION = 407
NEXT_ARTIFACT = "paper4_v409_notebook_b018_display_review.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v408_notebook_b007_loop_var_patch.md"


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


def _read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _patch_notebooks() -> pd.DataFrame:
    actions = [
        (
            "eda_status_breakdown_unused_name",
            "notebooks/01_eda_lending_club.ipynb",
            12,
            "for i, (val, name) in enumerate(zip(status_counts.values, status_counts.index, strict=False)):\n",
            "for i, (val, _name) in enumerate(zip(status_counts.values, status_counts.index, strict=False)):\n",
        ),
        (
            "explainability_perm_unused_i",
            "notebooks/13_model_explainability.ipynb",
            10,
            "for i, row in perm_df.head(15).iterrows():\n",
            "for _i, row in perm_df.head(15).iterrows():\n",
        ),
        (
            "explainability_family_unused_idx",
            "notebooks/13_model_explainability.ipynb",
            14,
            "for i, (idx, row) in enumerate(fa.iterrows()):\n",
            "for i, (_idx, row) in enumerate(fa.iterrows()):\n",
        ),
    ]
    payloads: dict[str, dict[str, Any]] = {}
    rows = []
    for action_id, notebook_path, cell_number, old_line, new_line in actions:
        payload = payloads.get(notebook_path)
        if payload is None:
            payload = _read_notebook(ROOT / notebook_path)
            payloads[notebook_path] = payload
        source = payload["cells"][cell_number - 1]["source"]
        if old_line not in source:
            raise RuntimeError(f"expected B007 target line missing for {action_id}")
        source[source.index(old_line)] = new_line
        rows.append(
            {
                "action_id_v408": action_id,
                "notebook_path_v408": notebook_path,
                "cell_v408": cell_number,
                "old_loop_var_v408": old_line.strip(),
                "new_loop_var_v408": new_line.strip(),
                "mutation_applied_v408": True,
                "claim_boundary_v408": "B007 unused loop variable rename only",
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
                "notebook_path_v408": notebook_path,
                "file_sha256_before_v408": before["file_sha256"],
                "file_sha256_after_v408": after["file_sha256"],
                "file_changed_v408": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v408": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v408": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v408": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v408": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v408": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v408": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v408": "single-line code-source rename only",
            }
        )
    return pd.DataFrame(rows)


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_b007", before_counts.get("B007", 0), after_counts.get("B007", 0)),
        ("global_notebook_b018", before_counts.get("B018", 0), after_counts.get("B018", 0)),
        ("global_notebook_f821", before_counts.get("F821", 0), after_counts.get("F821", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v408": metric,
                "before_v408": before,
                "after_v408": after,
                "delta_v408": after - before,
                "claim_boundary_v408": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v408": "b018_display_review_deferred",
                "blocking_v408": True,
                "evidence_count_v408": 10,
                "required_next_artifact_v408": NEXT_ARTIFACT,
                "claim_boundary_v408": "display semantics require review",
            },
            {
                "blocker_id_v408": "global_notebook_lint_not_clean",
                "blocking_v408": True,
                "evidence_count_v408": global_after,
                "required_next_artifact_v408": NEXT_ARTIFACT,
                "claim_boundary_v408": "non-B007 notebook lint remains",
            },
            {
                "blocker_id_v408": "paper4_final_promotion_forbidden",
                "blocking_v408": True,
                "evidence_count_v408": 1,
                "required_next_artifact_v408": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v408": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v408_b007_loop_var_patch_applied",
                "allowed": True,
                "artifact": "paper4_v408_notebook_b007_loop_var_actions.csv",
                "boundary": "3 B007 unused loop variable renames",
            },
            {
                "claim_id": "v408_b007_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v408_notebook_lint_delta.csv",
                "boundary": "B007 only",
            },
            {
                "claim_id": "v408_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v408_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v408_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v408_claim_blockers.csv",
                "boundary": "17 notebook diagnostics remain",
            },
            {
                "claim_id": "v408_working_champion_or_final_promotion",
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
                "claim": "v408 clears notebook B007 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v408_notebook_lint_delta.csv",
                "boundary": "B007 only; other lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v408 reduces notebook lint diagnostics from 20 to 17.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v408_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v408 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v408_claim_blockers.csv",
                "boundary": "17 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v408 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v408_claim_blockers.csv",
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
                "executable_item": "v408 applies the B007 loop-variable rename batch.",
                "status": "notebook_b007_loop_var_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v409 reviews B018 display-expression semantics before mutation",
                "last_wave": "v408",
                "execution_result": "notebook_b007_cleared_lint_reduced_20_to_17",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v408")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook B007 Loop-Variable Patch v408

Generated: {status["generated_at_utc"]}

v408 applies the B007 first batch selected by v407.

## Result

- B007 diagnostics: `{status["global_notebook_b007_before_v408"]}` ->
  `{status["global_notebook_b007_after_v408"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v408"]}` ->
  `{status["global_notebook_diagnostics_after_v408"]}`.
- Changed notebook files: `{status["changed_notebook_files_v408"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v408"]}`.

## Required Caveat

v408 does not clear remaining B018/F821/style notebook lint and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v408"]}` for the B018 display-expression review.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V408_NOTEBOOK_B007_LOOP_VAR_PATCH_START -->"
    end = "<!-- V408_NOTEBOOK_B007_LOOP_VAR_PATCH_END -->"
    block = f"""
{start}

## Wave v408: Notebook B007 Loop-Variable Patch

Generated: {status["generated_at_utc"]}

### Objective

v408 applies the 3-diagnostic B007 loop-variable rename batch selected by v407.

### Results

- B007 before/after:
  `{status["global_notebook_b007_before_v408"]}` ->
  `{status["global_notebook_b007_after_v408"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v408"]}` ->
  `{status["global_notebook_diagnostics_after_v408"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v408"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v408"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v408"]}`.

### Interpretation

B007 is now closed. The remaining largest block is B018 display-expression
semantics, which needs review before mutation because notebook expressions can be
intentional outputs.

### Claim Impact

- Allowed: B007 cleared and notebook lint reduced to 17 diagnostics.
- Still prohibited: notebook lint clean, repository ruff clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v408 in the living notebook. v409 should review B018 display expressions
before mutation.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v408 expects clean notebook diff before mutation.")

    v407_status = json.loads((STATUS_DIR / "paper4_v407_status.json").read_text(encoding="utf-8"))
    if v407_status["next_artifact_v407"] != "paper4_v408_notebook_b007_loop_var_patch.md":
        raise RuntimeError("v408 expects v407 to route to B007 loop-var patch.")

    target_notebooks = [
        "notebooks/01_eda_lending_club.ipynb",
        "notebooks/13_model_explainability.ipynb",
    ]
    before_global = _run_ruff_json()
    before_counts = Counter(item["code"] for item in before_global)
    before_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
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
        "cell_count_preserved_v408",
        "code_cell_count_preserved_v408",
        "cell_type_sequence_preserved_v408",
        "non_code_source_preserved_v408",
        "outputs_preserved_v408",
        "metadata_preserved_v408",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(len(after_global))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v408_notebook_b007_loop_var_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v408_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v408_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v408_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v408_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v408_notebook_b007_loop_var_patch",
        "schema_version": "2026-05-17.408",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_non_e402_triage_version_v408": PRIOR_TRIAGE_VERSION,
        "action_rows_v408": int(len(actions)),
        "global_notebook_diagnostics_before_v408": int(len(before_global)),
        "global_notebook_diagnostics_after_v408": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v408": int(len(before_global) - len(after_global)),
        "global_notebook_b007_before_v408": int(before_counts.get("B007", 0)),
        "global_notebook_b007_after_v408": int(after_counts.get("B007", 0)),
        "global_notebook_b007_reduced_v408": int(
            before_counts.get("B007", 0) - after_counts.get("B007", 0)
        ),
        "changed_notebook_files_v408": int(len(changed_files)),
        "changed_notebook_file_list_v408": changed_files,
        "roundtrip_integrity_rows_v408": int(len(integrity)),
        "roundtrip_integrity_all_passed_v408": integrity_passed,
        "global_ruff_clean_v408": False,
        "full_repository_pytest_run_v408": False,
        "full_quarto_render_run_v408": False,
        "working_champion_claim_allowed_v408": False,
        "paper1_promotion_allowed_v408": False,
        "paper4_working_champion_changed_v408": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v408": NEXT_ARTIFACT,
        "claim_boundary": (
            "v408 clears B007 notebook lint with roundtrip checks; remaining lint and "
            "final promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v408 notebook roundtrip integrity failed.")
    if status["global_notebook_b007_after_v408"] != 0:
        raise RuntimeError("v408 did not clear B007.")
    if status["global_notebook_diagnostics_after_v408"] != 17:
        raise RuntimeError("v408 did not reach expected 17 diagnostics.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v408": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
