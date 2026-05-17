#!/usr/bin/env python3
"""Build Paper 4 v418 SIM108 conditional-expression notebook patch artifacts."""

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

VERSION = 418
PRIOR_SIM108_REVIEW_VERSION = 417
TARGET_NOTEBOOK = "notebooks/04_conformal_prediction.ipynb"
NEXT_ARTIFACT = "paper4_v419_notebook_sim108_post_patch_pytest_probe.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v418_notebook_sim108_conditional_expr_patch.md"

PATCHES = [
    {
        "action_id_v418": "sim108_conditional_expr_01",
        "cell_v418": 4,
        "old_block": [
            "if calibrator is not None:\n",
            "    y_prob_test_cal = calibrator.predict(y_prob_test)\n",
            "else:\n",
            "    y_prob_test_cal = y_prob_test\n",
        ],
        "new_block": [
            "y_prob_test_cal = calibrator.predict(y_prob_test) if calibrator is not None else y_prob_test\n",
        ],
        "claim_boundary_v418": "SIM108 calibrator conditional-expression patch only",
    },
    {
        "action_id_v418": "sim108_conditional_expr_02",
        "cell_v418": 25,
        "old_block": [
            "if ead_col in test.columns:\n",
            "    ead = test[ead_col].values\n",
            "else:\n",
            "    ead = np.ones(len(test)) * 15000  # default\n",
        ],
        "new_block": [
            "ead = test[ead_col].values if ead_col in test.columns else np.ones(len(test)) * 15000\n",
        ],
        "claim_boundary_v418": "SIM108 EAD conditional-expression patch only",
    },
]


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
    raise RuntimeError("expected SIM108 block was not found")


def _patch_notebook() -> pd.DataFrame:
    payload = _read_notebook(TARGET_NOTEBOOK)
    rows = []
    for patch in PATCHES:
        source = payload["cells"][int(patch["cell_v418"]) - 1]["source"]
        old_block = list(patch["old_block"])
        new_block = list(patch["new_block"])
        start_idx = _find_block(source, old_block)
        source[start_idx : start_idx + len(old_block)] = new_block
        rows.append(
            {
                "action_id_v418": patch["action_id_v418"],
                "notebook_path_v418": TARGET_NOTEBOOK,
                "cell_v418": int(patch["cell_v418"]),
                "old_block_v418": "".join(old_block).strip(),
                "new_statement_v418": "".join(new_block).strip(),
                "mutation_applied_v418": True,
                "claim_boundary_v418": patch["claim_boundary_v418"],
            }
        )
    _write_notebook(TARGET_NOTEBOOK, payload)
    return pd.DataFrame(rows)


def _roundtrip_integrity(before: dict[str, Any], after: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notebook_path_v418": TARGET_NOTEBOOK,
                "file_sha256_before_v418": before["file_sha256"],
                "file_sha256_after_v418": after["file_sha256"],
                "file_changed_v418": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v418": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v418": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v418": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v418": before["non_code_source_hash"] == after["non_code_source_hash"],
                "outputs_preserved_v418": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v418": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v418": "two-cell code-source patch only",
            }
        ]
    )


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_sim108", before_counts.get("SIM108", 0), after_counts.get("SIM108", 0)),
        ("global_notebook_e712", before_counts.get("E712", 0), after_counts.get("E712", 0)),
        ("global_notebook_sim102", before_counts.get("SIM102", 0), after_counts.get("SIM102", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v418": metric,
                "before_v418": before,
                "after_v418": after,
                "delta_v418": after - before,
                "claim_boundary_v418": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v418": "post_sim108_pytest_not_run",
                "blocking_v418": True,
                "evidence_count_v418": 1,
                "required_next_artifact_v418": NEXT_ARTIFACT,
                "claim_boundary_v418": "post-SIM108 pytest deferred to v419",
            },
            {
                "blocker_id_v418": "style_notebook_lint_remaining",
                "blocking_v418": True,
                "evidence_count_v418": global_after,
                "required_next_artifact_v418": NEXT_ARTIFACT,
                "claim_boundary_v418": "E712/SIM102 side-project style notebook lint remains",
            },
            {
                "blocker_id_v418": "paper4_final_promotion_forbidden",
                "blocking_v418": True,
                "evidence_count_v418": 1,
                "required_next_artifact_v418": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v418": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v418_sim108_conditional_expr_patch_applied",
                "allowed": True,
                "artifact": "paper4_v418_notebook_sim108_conditional_expr_actions.csv",
                "boundary": "two SIM108 conditional-expression patches",
            },
            {
                "claim_id": "v418_sim108_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v418_notebook_lint_delta.csv",
                "boundary": "SIM108 only",
            },
            {
                "claim_id": "v418_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v418_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v418_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v418_claim_blockers.csv",
                "boundary": "3 notebook diagnostics remain",
            },
            {
                "claim_id": "v418_post_sim108_pytest_passed",
                "allowed": False,
                "artifact": "paper4_v418_claim_blockers.csv",
                "boundary": "post-patch pytest deferred to v419",
            },
            {
                "claim_id": "v418_working_champion_or_final_promotion",
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
                "claim": "v418 clears notebook SIM108 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v418_notebook_lint_delta.csv",
                "boundary": "SIM108 only; side-project style notebook lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v418 reduces notebook lint diagnostics from 5 to 3.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v418_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v418 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v418_claim_blockers.csv",
                "boundary": "3 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v418 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v418_claim_blockers.csv",
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
                "executable_item": "v418 applies both SIM108 conditional-expression notebook patches.",
                "status": "notebook_sim108_conditional_expr_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v419 full repository pytest passes after SIM108 patch",
                "last_wave": "v418",
                "execution_result": "notebook_sim108_cleared_lint_reduced_5_to_3",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v418")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook SIM108 Conditional-Expression Patch v418

Generated: {status["generated_at_utc"]}

v418 applies the SIM108 patch selected by v417.

## Result

- SIM108 diagnostics: `{status["global_notebook_sim108_before_v418"]}` ->
  `{status["global_notebook_sim108_after_v418"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v418"]}` ->
  `{status["global_notebook_diagnostics_after_v418"]}`.
- Changed notebook files: `{status["changed_notebook_files_v418"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v418"]}`.

## Required Caveat

v418 does not clear remaining side-project style notebook lint, does not run
post-patch pytest, and does not create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V418_NOTEBOOK_SIM108_CONDITIONAL_EXPR_PATCH_START -->"
    end = "<!-- V418_NOTEBOOK_SIM108_CONDITIONAL_EXPR_PATCH_END -->"
    block = f"""
{start}

## Wave v418: Notebook SIM108 Conditional-Expression Patch

Generated: {status["generated_at_utc"]}

### Objective

v418 applies both SIM108 conditional-expression patches selected by v417.

### Results

- SIM108 before/after:
  `{status["global_notebook_sim108_before_v418"]}` ->
  `{status["global_notebook_sim108_after_v418"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v418"]}` ->
  `{status["global_notebook_diagnostics_after_v418"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v418"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v418"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v418"]}`.

### Interpretation

SIM108 is now closed. The remaining notebook lint frontier is isolated to the
GPU side-project notebook.

### Claim Impact

- Allowed: SIM108 cleared and notebook lint reduced to 3 diagnostics.
- Still prohibited: notebook lint clean, repository ruff clean, post-patch pytest
  passed, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v418 in the living notebook. v419 should run post-SIM108 full pytest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v418 expects clean notebook diff before mutation.")

    v417_status = json.loads((STATUS_DIR / "paper4_v417_status.json").read_text(encoding="utf-8"))
    if v417_status["next_artifact_v417"] != "paper4_v418_notebook_sim108_conditional_expr_patch.md":
        raise RuntimeError("v418 expects v417 to route to SIM108 patch.")

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
        "cell_count_preserved_v418",
        "code_cell_count_preserved_v418",
        "cell_type_sequence_preserved_v418",
        "non_code_source_preserved_v418",
        "outputs_preserved_v418",
        "metadata_preserved_v418",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(global_after=len(after_global))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v418_notebook_sim108_conditional_expr_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v418_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v418_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v418_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v418_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v418_notebook_sim108_conditional_expr_patch",
        "schema_version": "2026-05-17.418",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_sim108_review_version_v418": PRIOR_SIM108_REVIEW_VERSION,
        "action_rows_v418": int(len(actions)),
        "global_notebook_diagnostics_before_v418": int(len(before_global)),
        "global_notebook_diagnostics_after_v418": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v418": int(len(before_global) - len(after_global)),
        "global_notebook_sim108_before_v418": int(before_counts.get("SIM108", 0)),
        "global_notebook_sim108_after_v418": int(after_counts.get("SIM108", 0)),
        "global_notebook_sim108_reduced_v418": int(
            before_counts.get("SIM108", 0) - after_counts.get("SIM108", 0)
        ),
        "changed_notebook_files_v418": int(len(changed_files)),
        "changed_notebook_file_list_v418": changed_files,
        "roundtrip_integrity_rows_v418": int(len(integrity)),
        "roundtrip_integrity_all_passed_v418": integrity_passed,
        "global_ruff_clean_v418": False,
        "full_repository_pytest_run_v418": False,
        "full_quarto_render_run_v418": False,
        "working_champion_claim_allowed_v418": False,
        "paper1_promotion_allowed_v418": False,
        "paper4_working_champion_changed_v418": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v418": NEXT_ARTIFACT,
        "claim_boundary": (
            "v418 clears SIM108 with conditional expressions; remaining lint and final "
            "promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v418 notebook roundtrip integrity failed.")
    if status["global_notebook_sim108_after_v418"] != 0:
        raise RuntimeError("v418 did not clear SIM108.")
    if status["global_notebook_diagnostics_after_v418"] != 3:
        raise RuntimeError("v418 did not reach expected 3 diagnostics.")
    if changed_files != [TARGET_NOTEBOOK]:
        raise RuntimeError("v418 changed an unexpected notebook set.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v418": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
