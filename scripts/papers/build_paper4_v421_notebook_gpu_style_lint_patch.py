#!/usr/bin/env python3
"""Build Paper 4 v421 GPU side-project style-lint notebook patch artifacts."""

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

VERSION = 421
PRIOR_GPU_STYLE_TRIAGE_VERSION = 420
TARGET_NOTEBOOK = "notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb"
NEXT_ARTIFACT = "paper4_v422_notebook_gpu_style_post_patch_pytest_probe.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v421_notebook_gpu_style_lint_patch.md"

CUDF_OLD_LINE = (
    '        (cudf_df["rows_match_cpu"] == True) & '
    '(cudf_df["checksum_rel_err_cpu"] <= CONFIG["consistency_rel_tol"]),\n'
)
CUDF_NEW_LINES = [
    '        cudf_df["rows_match_cpu"].fillna(False).astype(bool)\n',
    '        & (cudf_df["checksum_rel_err_cpu"] <= CONFIG["consistency_rel_tol"]),\n',
]

QUALITY_GUARD_OLD_BLOCK = [
    'for name in ["cuml_quality_df", "cupy_df"]:\n',
    "    if name in globals() and isinstance(globals()[name], pd.DataFrame) and len(globals()[name]):\n",
    '        if "quality_pass" in globals()[name].columns:\n',
    "            tmp = globals()[name].copy()\n",
    '            tmp["section"] = name.replace("_quality_df", "").replace("_df", "")\n',
    "            checks.append(tmp)\n",
]
QUALITY_GUARD_NEW_BLOCK = [
    'for name in ["cuml_quality_df", "cupy_df"]:\n',
    "    candidate = globals().get(name)\n",
    "    if (\n",
    "        isinstance(candidate, pd.DataFrame)\n",
    "        and len(candidate)\n",
    '        and "quality_pass" in candidate.columns\n',
    "    ):\n",
    "        tmp = candidate.copy()\n",
    '        tmp["section"] = name.replace("_quality_df", "").replace("_df", "")\n',
    "        checks.append(tmp)\n",
]

QUALITY_FAIL_OLD_LINE = '        failed = quality[quality["quality_pass"] == False]\n'
QUALITY_FAIL_NEW_LINE = '        failed = quality[quality["quality_pass"].eq(False)]\n'


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
    raise RuntimeError("expected GPU style-lint block was not found")


def _patch_notebook() -> pd.DataFrame:
    payload = _read_notebook(TARGET_NOTEBOOK)
    cell_7 = payload["cells"][7 - 1]["source"]
    cell_17 = payload["cells"][17 - 1]["source"]
    rows = []

    if CUDF_OLD_LINE not in cell_7:
        raise RuntimeError("expected cuDF rows_match_cpu E712 line missing")
    cudf_idx = cell_7.index(CUDF_OLD_LINE)
    cell_7[cudf_idx : cudf_idx + 1] = CUDF_NEW_LINES
    rows.append(
        {
            "action_id_v421": "gpu_style_lint_01_rows_match_cpu_boolean_mask",
            "notebook_path_v421": TARGET_NOTEBOOK,
            "cell_v421": 7,
            "old_source_v421": CUDF_OLD_LINE.strip(),
            "new_source_v421": "".join(CUDF_NEW_LINES).strip(),
            "lint_code_v421": "E712",
            "mutation_applied_v421": True,
            "claim_boundary_v421": "cuDF consistency mask style patch only",
        }
    )

    guard_idx = _find_block(cell_17, QUALITY_GUARD_OLD_BLOCK)
    cell_17[guard_idx : guard_idx + len(QUALITY_GUARD_OLD_BLOCK)] = QUALITY_GUARD_NEW_BLOCK
    rows.append(
        {
            "action_id_v421": "gpu_style_lint_02_quality_pass_guard",
            "notebook_path_v421": TARGET_NOTEBOOK,
            "cell_v421": 17,
            "old_source_v421": "".join(QUALITY_GUARD_OLD_BLOCK).strip(),
            "new_source_v421": "".join(QUALITY_GUARD_NEW_BLOCK).strip(),
            "lint_code_v421": "SIM102",
            "mutation_applied_v421": True,
            "claim_boundary_v421": "quality_pass guard consolidation only",
        }
    )

    if QUALITY_FAIL_OLD_LINE not in cell_17:
        raise RuntimeError("expected quality_pass E712 line missing")
    fail_idx = cell_17.index(QUALITY_FAIL_OLD_LINE)
    cell_17[fail_idx] = QUALITY_FAIL_NEW_LINE
    rows.append(
        {
            "action_id_v421": "gpu_style_lint_03_quality_pass_eq_false",
            "notebook_path_v421": TARGET_NOTEBOOK,
            "cell_v421": 17,
            "old_source_v421": QUALITY_FAIL_OLD_LINE.strip(),
            "new_source_v421": QUALITY_FAIL_NEW_LINE.strip(),
            "lint_code_v421": "E712",
            "mutation_applied_v421": True,
            "claim_boundary_v421": "quality_pass comparison style patch only",
        }
    )

    _write_notebook(TARGET_NOTEBOOK, payload)
    return pd.DataFrame(rows)


def _roundtrip_integrity(before: dict[str, Any], after: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notebook_path_v421": TARGET_NOTEBOOK,
                "file_sha256_before_v421": before["file_sha256"],
                "file_sha256_after_v421": after["file_sha256"],
                "file_changed_v421": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v421": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v421": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v421": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v421": before["non_code_source_hash"] == after["non_code_source_hash"],
                "outputs_preserved_v421": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v421": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v421": "three code-source patches only",
            }
        ]
    )


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_e712", before_counts.get("E712", 0), after_counts.get("E712", 0)),
        ("global_notebook_sim102", before_counts.get("SIM102", 0), after_counts.get("SIM102", 0)),
        ("global_notebook_sim108", before_counts.get("SIM108", 0), after_counts.get("SIM108", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v421": metric,
                "before_v421": before,
                "after_v421": after,
                "delta_v421": after - before,
                "claim_boundary_v421": "notebook-lint delta only; repository ruff not claimed",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v421": "post_gpu_style_patch_pytest_not_run",
                "blocking_v421": True,
                "evidence_count_v421": 1,
                "required_next_artifact_v421": NEXT_ARTIFACT,
                "claim_boundary_v421": "post-GPU-style patch full pytest deferred to v422",
            },
            {
                "blocker_id_v421": "repository_ruff_clean_not_run",
                "blocking_v421": True,
                "evidence_count_v421": 1,
                "required_next_artifact_v421": NEXT_ARTIFACT,
                "claim_boundary_v421": "notebook lint is clean, but repository ruff is not claimed",
            },
            {
                "blocker_id_v421": "paper4_final_promotion_forbidden",
                "blocking_v421": True,
                "evidence_count_v421": 1,
                "required_next_artifact_v421": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v421": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v421_gpu_style_lint_patch_applied",
                "allowed": True,
                "artifact": "paper4_v421_notebook_gpu_style_lint_actions.csv",
                "boundary": "three GPU side-project style patches",
            },
            {
                "claim_id": "v421_notebook_lint_cleared",
                "allowed": True,
                "artifact": "paper4_v421_notebook_lint_delta.csv",
                "boundary": "ruff check notebooks reports zero diagnostics after patch",
            },
            {
                "claim_id": "v421_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v421_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v421_post_gpu_style_pytest_passed",
                "allowed": False,
                "artifact": "paper4_v421_claim_blockers.csv",
                "boundary": "post-patch pytest deferred to v422",
            },
            {
                "claim_id": "v421_repository_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v421_claim_blockers.csv",
                "boundary": "repository ruff was not run and is not claimed",
            },
            {
                "claim_id": "v421_working_champion_or_final_promotion",
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
                "claim": "v421 clears the remaining GPU side-project notebook style diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v421_notebook_lint_delta.csv",
                "boundary": "Notebook lint only; post-patch pytest deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v421 clears notebook lint diagnostics from 3 to 0.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v421_notebook_lint_delta.csv",
                "boundary": "Applies only to ruff check notebooks.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v421 preserves notebook roundtrip integrity for the GPU side-project patch.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v421_notebook_roundtrip_integrity.csv",
                "boundary": "Cell structure, non-code source, outputs and metadata preserved.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v421 proves post-patch pytest, repository ruff, or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v421_claim_blockers.csv",
                "boundary": "Those probes are deferred and cannot be claimed from v421.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v421 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v421_claim_blockers.csv",
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
                "executable_item": "v421 applies the GPU side-project notebook style-lint patch.",
                "status": "notebook_gpu_style_lint_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v422 full repository pytest passes after GPU style-lint patch",
                "last_wave": "v421",
                "execution_result": "notebook_gpu_style_lint_cleared_3_to_0",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v421")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 GPU Side-Project Style-Lint Patch v421

Generated: {status["generated_at_utc"]}

v421 applies the v420-selected GPU side-project style-lint patch.

## Result

- Notebook diagnostics: `{status["global_notebook_diagnostics_before_v421"]}` ->
  `{status["global_notebook_diagnostics_after_v421"]}`.
- E712 diagnostics: `{status["global_notebook_e712_before_v421"]}` ->
  `{status["global_notebook_e712_after_v421"]}`.
- SIM102 diagnostics: `{status["global_notebook_sim102_before_v421"]}` ->
  `{status["global_notebook_sim102_after_v421"]}`.
- Changed notebook files: `{status["changed_notebook_files_v421"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v421"]}`.
- Notebook ruff clean: `{status["notebook_ruff_clean_v421"]}`.

## Required Caveat

v421 clears notebook lint only. It does not run post-patch pytest, repository
ruff, Quarto render, or create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V421_NOTEBOOK_GPU_STYLE_LINT_PATCH_START -->"
    end = "<!-- V421_NOTEBOOK_GPU_STYLE_LINT_PATCH_END -->"
    block = f"""
{start}

## Wave v421: GPU Side-Project Style-Lint Patch

Generated: {status["generated_at_utc"]}

### Objective

v421 applies the three GPU side-project style-lint patches selected by v420.

### Results

- Notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v421"]}` ->
  `{status["global_notebook_diagnostics_after_v421"]}`.
- E712 before/after:
  `{status["global_notebook_e712_before_v421"]}` ->
  `{status["global_notebook_e712_after_v421"]}`.
- SIM102 before/after:
  `{status["global_notebook_sim102_before_v421"]}` ->
  `{status["global_notebook_sim102_after_v421"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v421"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v421"]}`.
- Notebook ruff clean:
  `{status["notebook_ruff_clean_v421"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v421"]}`.

### Interpretation

The notebook-lint frontier is cleared for `ruff check notebooks`. The next
step is a post-mutation repository pytest probe before using the notebook
cleanup as broader regression evidence.

### Claim Impact

- Allowed: GPU side-project style lint patched, notebook lint reduced from 3 to
  0, and roundtrip integrity preserved.
- Still prohibited: post-patch pytest passed, repository ruff clean, Quarto
  render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v421 in the living notebook. v422 should run the post-GPU-style full
pytest probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v421 expects clean notebook diff before mutation.")

    v420_status = json.loads((STATUS_DIR / "paper4_v420_status.json").read_text(encoding="utf-8"))
    if v420_status["next_artifact_v420"] != "paper4_v421_notebook_gpu_style_lint_patch.md":
        raise RuntimeError("v421 expects v420 to route to GPU style-lint patch.")
    if int(v420_status["selected_for_v421_rows_v420"]) != 3:
        raise RuntimeError("v421 expects three v420-selected diagnostics.")

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
        "cell_count_preserved_v421",
        "code_cell_count_preserved_v421",
        "cell_type_sequence_preserved_v421",
        "non_code_source_preserved_v421",
        "outputs_preserved_v421",
        "metadata_preserved_v421",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v421_notebook_gpu_style_lint_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v421_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v421_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v421_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v421_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v421_notebook_gpu_style_lint_patch",
        "schema_version": "2026-05-17.421",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_gpu_style_triage_version_v421": PRIOR_GPU_STYLE_TRIAGE_VERSION,
        "action_rows_v421": int(len(actions)),
        "global_notebook_diagnostics_before_v421": int(len(before_global)),
        "global_notebook_diagnostics_after_v421": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v421": int(len(before_global) - len(after_global)),
        "global_notebook_e712_before_v421": int(before_counts.get("E712", 0)),
        "global_notebook_e712_after_v421": int(after_counts.get("E712", 0)),
        "global_notebook_e712_reduced_v421": int(
            before_counts.get("E712", 0) - after_counts.get("E712", 0)
        ),
        "global_notebook_sim102_before_v421": int(before_counts.get("SIM102", 0)),
        "global_notebook_sim102_after_v421": int(after_counts.get("SIM102", 0)),
        "global_notebook_sim102_reduced_v421": int(
            before_counts.get("SIM102", 0) - after_counts.get("SIM102", 0)
        ),
        "changed_notebook_files_v421": int(len(changed_files)),
        "changed_notebook_file_list_v421": changed_files,
        "roundtrip_integrity_rows_v421": int(len(integrity)),
        "roundtrip_integrity_all_passed_v421": integrity_passed,
        "notebook_ruff_clean_v421": len(after_global) == 0,
        "repository_ruff_clean_v421": False,
        "full_repository_pytest_run_v421": False,
        "full_quarto_render_run_v421": False,
        "working_champion_claim_allowed_v421": False,
        "paper1_promotion_allowed_v421": False,
        "paper4_working_champion_changed_v421": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v421": NEXT_ARTIFACT,
        "claim_boundary": (
            "v421 clears notebook lint with a GPU side-project patch; post-patch "
            "pytest, repository ruff, Quarto and final-promotion claims remain blocked"
        ),
    }
    if len(before_global) != 3:
        raise RuntimeError("v421 expected exactly three notebook diagnostics before mutation.")
    if before_counts.get("E712", 0) != 2 or before_counts.get("SIM102", 0) != 1:
        raise RuntimeError("v421 expected two E712 and one SIM102 diagnostic before mutation.")
    if not integrity_passed:
        raise RuntimeError("v421 notebook roundtrip integrity failed.")
    if not status["notebook_ruff_clean_v421"]:
        raise RuntimeError("v421 did not clear notebook lint.")
    if changed_files != [TARGET_NOTEBOOK]:
        raise RuntimeError("v421 changed an unexpected notebook set.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v421": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
