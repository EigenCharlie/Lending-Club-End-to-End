#!/usr/bin/env python3
"""Build Paper 4 v410 B018 fig.show notebook patch artifacts."""

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

VERSION = 410
PRIOR_B018_REVIEW_VERSION = 409
NEXT_ARTIFACT = "paper4_v411_notebook_b018_post_patch_pytest_probe.md"
PATCH_MD = NOTEBOOK.parent / "paper4_v410_notebook_b018_fig_show_patch.md"
REVIEW_PATH = TABLE_DIR / "paper4_v409_notebook_b018_display_review.csv"


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


def _patch_notebooks(review: pd.DataFrame) -> pd.DataFrame:
    payloads: dict[str, dict[str, Any]] = {}
    rows = []
    for row in review.itertuples(index=False):
        notebook_path = str(row.notebook_path_v409)
        cell_number = int(row.cell_v409)
        old_line = f"{str(row.display_expression_v409)}\n"
        new_line = f"{str(row.recommended_patch_v409)}\n"
        payload = payloads.get(notebook_path)
        if payload is None:
            payload = _read_notebook(ROOT / notebook_path)
            payloads[notebook_path] = payload
        source = payload["cells"][cell_number - 1]["source"]
        if old_line not in source:
            raise RuntimeError(f"expected B018 display line missing: {notebook_path} cell {cell_number}")
        source[source.index(old_line)] = new_line
        rows.append(
            {
                "action_id_v410": f"b018_fig_show_{len(rows) + 1:02d}",
                "notebook_path_v410": notebook_path,
                "cell_v410": cell_number,
                "old_display_expression_v410": old_line.strip(),
                "new_display_statement_v410": new_line.strip(),
                "mutation_applied_v410": True,
                "claim_boundary_v410": "B018 explicit fig.show display patch only",
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
                "notebook_path_v410": notebook_path,
                "file_sha256_before_v410": before["file_sha256"],
                "file_sha256_after_v410": after["file_sha256"],
                "file_changed_v410": before["file_sha256"] != after["file_sha256"],
                "cell_count_preserved_v410": before["cell_count"] == after["cell_count"],
                "code_cell_count_preserved_v410": before["code_cell_count"] == after["code_cell_count"],
                "cell_type_sequence_preserved_v410": (
                    before["cell_type_sequence_hash"] == after["cell_type_sequence_hash"]
                ),
                "non_code_source_preserved_v410": (
                    before["non_code_source_hash"] == after["non_code_source_hash"]
                ),
                "outputs_preserved_v410": before["outputs_hash"] == after["outputs_hash"],
                "metadata_preserved_v410": before["metadata_hash"] == after["metadata_hash"],
                "claim_boundary_v410": "single-line code-source display patch only",
            }
        )
    return pd.DataFrame(rows)


def _lint_delta(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_counts = Counter(item["code"] for item in before_items)
    after_counts = Counter(item["code"] for item in after_items)
    rows = [
        ("global_notebook_total", len(before_items), len(after_items)),
        ("global_notebook_b018", before_counts.get("B018", 0), after_counts.get("B018", 0)),
        ("global_notebook_f821", before_counts.get("F821", 0), after_counts.get("F821", 0)),
        ("global_notebook_e741", before_counts.get("E741", 0), after_counts.get("E741", 0)),
        ("global_notebook_sim108", before_counts.get("SIM108", 0), after_counts.get("SIM108", 0)),
        ("global_notebook_e712", before_counts.get("E712", 0), after_counts.get("E712", 0)),
        ("global_notebook_sim102", before_counts.get("SIM102", 0), after_counts.get("SIM102", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v410": metric,
                "before_v410": before,
                "after_v410": after,
                "delta_v410": after - before,
                "claim_boundary_v410": "lint reduction only; global lint not clean",
            }
            for metric, before, after in rows
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v410": "f821_execution_context_deferred",
                "blocking_v410": True,
                "evidence_count_v410": 1,
                "required_next_artifact_v410": "paper4_v412_notebook_f821_execution_context_audit.md",
                "claim_boundary_v410": "undefined execution context requires audit",
            },
            {
                "blocker_id_v410": "global_notebook_lint_not_clean",
                "blocking_v410": True,
                "evidence_count_v410": global_after,
                "required_next_artifact_v410": NEXT_ARTIFACT,
                "claim_boundary_v410": "non-B018 notebook lint remains",
            },
            {
                "blocker_id_v410": "post_b018_pytest_not_run",
                "blocking_v410": True,
                "evidence_count_v410": 1,
                "required_next_artifact_v410": NEXT_ARTIFACT,
                "claim_boundary_v410": "post-display-patch pytest deferred to v411",
            },
            {
                "blocker_id_v410": "paper4_final_promotion_forbidden",
                "blocking_v410": True,
                "evidence_count_v410": 1,
                "required_next_artifact_v410": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v410": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v410_b018_fig_show_patch_applied",
                "allowed": True,
                "artifact": "paper4_v410_notebook_b018_fig_show_actions.csv",
                "boundary": "10 explicit fig.show replacements",
            },
            {
                "claim_id": "v410_b018_cleared_from_notebooks",
                "allowed": True,
                "artifact": "paper4_v410_notebook_lint_delta.csv",
                "boundary": "B018 only",
            },
            {
                "claim_id": "v410_roundtrip_integrity_preserved",
                "allowed": True,
                "artifact": "paper4_v410_notebook_roundtrip_integrity.csv",
                "boundary": "outputs, markdown, metadata and cell structure preserved",
            },
            {
                "claim_id": "v410_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v410_claim_blockers.csv",
                "boundary": "7 notebook diagnostics remain",
            },
            {
                "claim_id": "v410_post_b018_pytest_passed",
                "allowed": False,
                "artifact": "paper4_v410_claim_blockers.csv",
                "boundary": "post-patch pytest deferred to v411",
            },
            {
                "claim_id": "v410_working_champion_or_final_promotion",
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
                "claim": "v410 clears notebook B018 diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v410_notebook_lint_delta.csv",
                "boundary": "B018 only; other lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v410 reduces notebook lint diagnostics from 17 to 7.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v410_notebook_lint_delta.csv",
                "boundary": "Reduction only; notebook lint is not clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v410 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v410_claim_blockers.csv",
                "boundary": "7 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v410 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v410_claim_blockers.csv",
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
                "executable_item": "v410 applies explicit fig.show replacements for B018 notebook displays.",
                "status": "notebook_b018_fig_show_patch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v411 full repository pytest passes after B018 display patch",
                "last_wave": "v410",
                "execution_result": "notebook_b018_cleared_lint_reduced_17_to_7",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v410")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _patch_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook B018 Fig.show Patch v410

Generated: {status["generated_at_utc"]}

v410 applies the explicit `fig.show()` display patch selected by v409.

## Result

- B018 diagnostics: `{status["global_notebook_b018_before_v410"]}` ->
  `{status["global_notebook_b018_after_v410"]}`.
- Global notebook diagnostics: `{status["global_notebook_diagnostics_before_v410"]}` ->
  `{status["global_notebook_diagnostics_after_v410"]}`.
- Changed notebook files: `{status["changed_notebook_files_v410"]}`.
- Roundtrip integrity passed: `{status["roundtrip_integrity_all_passed_v410"]}`.

## Required Caveat

v410 does not clear remaining F821/SIM/style notebook lint, does not run
post-patch pytest, and does not create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V410_NOTEBOOK_B018_FIG_SHOW_PATCH_START -->"
    end = "<!-- V410_NOTEBOOK_B018_FIG_SHOW_PATCH_END -->"
    block = f"""
{start}

## Wave v410: Notebook B018 Fig.show Patch

Generated: {status["generated_at_utc"]}

### Objective

v410 replaces the 10 reviewed bare plotly figure display expressions with
explicit `fig.show()` calls.

### Results

- B018 before/after:
  `{status["global_notebook_b018_before_v410"]}` ->
  `{status["global_notebook_b018_after_v410"]}`.
- Global notebook diagnostics before/after:
  `{status["global_notebook_diagnostics_before_v410"]}` ->
  `{status["global_notebook_diagnostics_after_v410"]}`.
- Changed notebook files:
  `{status["changed_notebook_files_v410"]}`.
- Roundtrip integrity passed:
  `{status["roundtrip_integrity_all_passed_v410"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v410"]}`.

### Interpretation

B018 is now closed while preserving explicit notebook display intent. The
remaining lint frontier is small and still not clean.

### Claim Impact

- Allowed: B018 cleared and notebook lint reduced to 7 diagnostics.
- Still prohibited: notebook lint clean, repository ruff clean, post-patch pytest
  passed, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v410 in the living notebook. v411 should run post-B018 full pytest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v410 expects clean notebook diff before mutation.")

    v409_status = json.loads((STATUS_DIR / "paper4_v409_status.json").read_text(encoding="utf-8"))
    if v409_status["next_artifact_v409"] != "paper4_v410_notebook_b018_fig_show_patch.md":
        raise RuntimeError("v410 expects v409 to route to fig.show patch.")

    review = pd.read_csv(REVIEW_PATH)
    target_notebooks = sorted(review["notebook_path_v409"].unique())
    before_global = _run_ruff_json()
    before_counts = Counter(item["code"] for item in before_global)
    before_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in target_notebooks
    }

    actions = _patch_notebooks(review)
    changed_files = _changed_notebook_files()
    after_global = _run_ruff_json()
    after_counts = Counter(item["code"] for item in after_global)
    after_signatures = {
        notebook_path: _notebook_signature(ROOT / notebook_path)
        for notebook_path in changed_files
    }
    integrity = _roundtrip_integrity(before_signatures, after_signatures, changed_files)
    integrity_columns = [
        "cell_count_preserved_v410",
        "code_cell_count_preserved_v410",
        "cell_type_sequence_preserved_v410",
        "non_code_source_preserved_v410",
        "outputs_preserved_v410",
        "metadata_preserved_v410",
    ]
    integrity_passed = bool(integrity[integrity_columns].astype(bool).all().all())
    lint_delta = _lint_delta(before_global, after_global)
    blockers = _claim_blockers(len(after_global))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v410_notebook_b018_fig_show_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v410_notebook_lint_delta.csv", lint_delta)
    write_csv(TABLE_DIR / "paper4_v410_notebook_roundtrip_integrity.csv", integrity)
    write_csv(TABLE_DIR / "paper4_v410_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v410_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v410_notebook_b018_fig_show_patch",
        "schema_version": "2026-05-17.410",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_b018_review_version_v410": PRIOR_B018_REVIEW_VERSION,
        "action_rows_v410": int(len(actions)),
        "global_notebook_diagnostics_before_v410": int(len(before_global)),
        "global_notebook_diagnostics_after_v410": int(len(after_global)),
        "global_notebook_diagnostics_reduced_v410": int(len(before_global) - len(after_global)),
        "global_notebook_b018_before_v410": int(before_counts.get("B018", 0)),
        "global_notebook_b018_after_v410": int(after_counts.get("B018", 0)),
        "global_notebook_b018_reduced_v410": int(
            before_counts.get("B018", 0) - after_counts.get("B018", 0)
        ),
        "changed_notebook_files_v410": int(len(changed_files)),
        "changed_notebook_file_list_v410": changed_files,
        "roundtrip_integrity_rows_v410": int(len(integrity)),
        "roundtrip_integrity_all_passed_v410": integrity_passed,
        "global_ruff_clean_v410": False,
        "full_repository_pytest_run_v410": False,
        "full_quarto_render_run_v410": False,
        "working_champion_claim_allowed_v410": False,
        "paper1_promotion_allowed_v410": False,
        "paper4_working_champion_changed_v410": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v410": NEXT_ARTIFACT,
        "claim_boundary": (
            "v410 clears B018 notebook lint with explicit fig.show calls; remaining "
            "lint and final promotion claims remain blocked"
        ),
    }
    if not integrity_passed:
        raise RuntimeError("v410 notebook roundtrip integrity failed.")
    if status["global_notebook_b018_after_v410"] != 0:
        raise RuntimeError("v410 did not clear B018.")
    if status["global_notebook_diagnostics_after_v410"] != 7:
        raise RuntimeError("v410 did not reach expected 7 diagnostics.")
    if sorted(changed_files) != target_notebooks:
        raise RuntimeError("v410 changed an unexpected notebook set.")

    PATCH_MD.write_text(_patch_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v410": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
