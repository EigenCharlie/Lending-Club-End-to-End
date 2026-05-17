#!/usr/bin/env python3
"""Build Paper 4 v398 historical notebook E402 policy artifacts."""

from __future__ import annotations

import json
import subprocess
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

VERSION = 398
PRIOR_SIDE_EFFECT_PATCH_VERSION = 397
NEXT_VERSION = 399
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_e402_cell_refactor_plan.csv"
POLICY_MD = NOTEBOOK.parent / "paper4_v398_notebook_historical_e402_policy.md"
RUFF_E402_COMMAND = "uv run ruff check notebooks --select E402 --output-format json"


def _run_e402_json() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "notebooks", "--select", "E402", "--output-format", "json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff E402 probe failed")
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("ruff E402 JSON output is not a list")
    return payload


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _manifest(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        location = item.get("location") or {}
        rows.append(
            {
                "diagnostic_id_v398": f"e402_{idx:03d}",
                "notebook_path_v398": _relative_path(str(item["filename"])),
                "cell_v398": int(item.get("cell") or 0),
                "row_v398": int(location.get("row") or 0),
                "column_v398": int(location.get("column") or 0),
                "rule_code_v398": str(item["code"]),
                "message_v398": str(item["message"]),
                "has_ruff_fix_v398": bool(item.get("fix")),
                "mutation_allowed_v398": False,
                "policy_class_v398": "historical_cell_setup_import_order",
                "claim_boundary_v398": "E402 manifest only; no notebook mutation in v398",
            }
        )
    return pd.DataFrame(rows)


def _file_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for notebook_path, group in manifest.groupby("notebook_path_v398", sort=True):
        rows.append(
            {
                "notebook_path_v398": notebook_path,
                "e402_diagnostic_count_v398": int(len(group)),
                "affected_cell_count_v398": int(group["cell_v398"].nunique()),
                "first_e402_row_v398": int(group["row_v398"].min()),
                "last_e402_row_v398": int(group["row_v398"].max()),
                "requires_cell_refactor_plan_v398": True,
                "claim_boundary_v398": "file-level E402 policy summary only",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["e402_diagnostic_count_v398", "notebook_path_v398"],
        ascending=[False, True],
        ignore_index=True,
    )


def _cell_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (notebook_path, cell), group in manifest.groupby(
        ["notebook_path_v398", "cell_v398"],
        sort=True,
    ):
        rows.append(
            {
                "notebook_path_v398": notebook_path,
                "cell_v398": int(cell),
                "e402_diagnostic_count_v398": int(len(group)),
                "first_e402_row_v398": int(group["row_v398"].min()),
                "last_e402_row_v398": int(group["row_v398"].max()),
                "planned_v399_action_v398": "inspect_cell_setup_before_reorder",
                "claim_boundary_v398": "cell needs local execution-order review",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["e402_diagnostic_count_v398", "notebook_path_v398", "cell_v398"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _decision_table(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v398": "bulk_notebook_import_reorder",
                "decision_v398": "rejected",
                "selected_v398": False,
                "evidence_count_v398": int(len(manifest)),
                "rationale_v398": "Could move imports across setup statements in executed notebooks.",
                "next_artifact_v398": "do_not_apply",
                "claim_boundary_v398": "no blind notebook mutation",
            },
            {
                "decision_id_v398": "ignore_or_exclude_notebook_e402",
                "decision_v398": "rejected",
                "selected_v398": False,
                "evidence_count_v398": int(len(manifest)),
                "rationale_v398": "Would hide a real reproducibility and style frontier.",
                "next_artifact_v398": "do_not_apply",
                "claim_boundary_v398": "lint debt remains visible",
            },
            {
                "decision_id_v398": "cell_local_refactor_plan",
                "decision_v398": "selected",
                "selected_v398": True,
                "evidence_count_v398": int(manifest[["notebook_path_v398", "cell_v398"]].drop_duplicates().shape[0]),
                "rationale_v398": "Only 15 cells need local review before any import movement.",
                "next_artifact_v398": NEXT_ARTIFACT,
                "claim_boundary_v398": "planning only; no mutation in v398",
            },
            {
                "decision_id_v398": "f821_execution_context_first",
                "decision_v398": "deferred",
                "selected_v398": False,
                "evidence_count_v398": 1,
                "rationale_v398": "Important, but v398 is scoped to E402 governance.",
                "next_artifact_v398": "paper4_v400_notebook_execution_context_audit.md",
                "claim_boundary_v398": "separate blocker",
            },
        ]
    )


def _claim_blockers(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v398": "historical_e402_notebook_frontier_remaining",
                "blocking_v398": True,
                "evidence_count_v398": int(len(manifest)),
                "required_next_artifact_v398": NEXT_ARTIFACT,
                "claim_boundary_v398": "E402 requires cell-local refactor planning",
            },
            {
                "blocker_id_v398": "bulk_notebook_import_reorder_rejected",
                "blocking_v398": True,
                "evidence_count_v398": int(manifest[["notebook_path_v398", "cell_v398"]].drop_duplicates().shape[0]),
                "required_next_artifact_v398": NEXT_ARTIFACT,
                "claim_boundary_v398": "no blind notebook import movement",
            },
            {
                "blocker_id_v398": "global_notebook_lint_not_clean",
                "blocking_v398": True,
                "evidence_count_v398": 139,
                "required_next_artifact_v398": NEXT_ARTIFACT,
                "claim_boundary_v398": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v398": "paper4_final_promotion_forbidden",
                "blocking_v398": True,
                "evidence_count_v398": 1,
                "required_next_artifact_v398": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v398": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v398_historical_e402_manifest_created",
                "allowed": True,
                "artifact": "paper4_v398_notebook_historical_e402_manifest.csv",
                "boundary": "119 E402 diagnostics inventoried",
            },
            {
                "claim_id": "v398_cell_local_refactor_policy_selected",
                "allowed": True,
                "artifact": "paper4_v398_notebook_historical_e402_decision.csv",
                "boundary": "planning only",
            },
            {
                "claim_id": "v398_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v398",
            },
            {
                "claim_id": "v398_e402_or_global_lint_clean",
                "allowed": False,
                "artifact": "paper4_v398_claim_blockers.csv",
                "boundary": "119 E402 and 139 notebook diagnostics remain",
            },
            {
                "claim_id": "v398_bulk_notebook_reorder_approved",
                "allowed": False,
                "artifact": "paper4_v398_notebook_historical_e402_decision.csv",
                "boundary": "bulk reorder rejected",
            },
            {
                "claim_id": "v398_working_champion_or_final_promotion",
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
                "claim": "v398 inventories 119 historical E402 notebook diagnostics.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v398_notebook_historical_e402_manifest.csv"
                ),
                "boundary": "Manifest and policy only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v398 selects cell-local E402 refactor planning instead of bulk reorder.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v398_notebook_historical_e402_decision.csv"
                ),
                "boundary": "Planning only; future cells require local review.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v398 clears E402 or global notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v398_claim_blockers.csv",
                "boundary": "119 E402 and 139 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v398 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v398_claim_blockers.csv",
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
                    "v398 inventories the historical E402 notebook frontier and selects "
                    "cell-local refactor planning instead of bulk reorder."
                ),
                "status": "notebook_historical_e402_policy_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v399 builds the 15-cell E402 refactor plan without mutating notebooks",
                "last_wave": "v398",
                "execution_result": "e402_119_diagnostics_15_cells_policy_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v398")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _policy_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Historical Notebook E402 Policy v398

Generated: {status["generated_at_utc"]}

v398 inventories the post-v397 historical E402 notebook frontier without
mutating notebooks.

## Result

- E402 diagnostics: `{status["e402_diagnostics_v398"]}`.
- Notebook files: `{status["e402_file_rows_v398"]}`.
- Affected cells: `{status["e402_cell_rows_v398"]}`.
- Top notebook: `{status["top_e402_notebook_v398"]}`
  (`{status["top_e402_count_v398"]}` diagnostics).
- Selected policy: `{status["selected_policy_v398"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v398"]}`.

## Required Caveat

v398 does not repair E402, does not make notebook or repository ruff clean, does
not run full pytest, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v398"]}` for the 15 affected E402 cells.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V398_NOTEBOOK_HISTORICAL_E402_POLICY_START -->"
    end = "<!-- V398_NOTEBOOK_HISTORICAL_E402_POLICY_END -->"
    block = f"""
{start}

## Wave v398: Historical Notebook E402 Policy

Generated: {status["generated_at_utc"]}

### Objective

v398 inventories the post-v397 historical notebook E402 frontier and decides how
to proceed without blind bulk mutation.

### Results

- E402 diagnostics:
  `{status["e402_diagnostics_v398"]}`.
- Notebook files:
  `{status["e402_file_rows_v398"]}`.
- Affected cells:
  `{status["e402_cell_rows_v398"]}`.
- Selected policy:
  `{status["selected_policy_v398"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v398"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v398"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v398"]}`.

### Interpretation

The remaining E402 debt is historical and concentrated in 15 notebook cells.
Bulk import movement is rejected; the next wave should produce a cell-local
refactor plan before any notebook mutation.

### Claim Impact

- Allowed: E402 manifest, file/cell summaries and cell-local policy decision.
- Still prohibited: E402 repaired, notebook lint clean, repository ruff clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v398 in the living notebook. v399 should build the 15-cell E402 refactor
plan.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v398 expects clean notebook diff because it is policy-only.")

    v397_status = json.loads((STATUS_DIR / "paper4_v397_status.json").read_text(encoding="utf-8"))
    if v397_status["next_artifact_v397"] != "paper4_v398_notebook_historical_e402_policy.md":
        raise RuntimeError("v398 expects v397 to route to historical E402 policy.")

    diff_clean_before = _notebook_diff_clean()
    items = _run_e402_json()
    manifest = _manifest(items)
    file_summary = _file_summary(manifest)
    cell_summary = _cell_summary(manifest)
    decisions = _decision_table(manifest)
    blockers = _claim_blockers(manifest)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v398_notebook_historical_e402_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v398_notebook_historical_e402_file_summary.csv", file_summary)
    write_csv(TABLE_DIR / "paper4_v398_notebook_historical_e402_cell_summary.csv", cell_summary)
    write_csv(TABLE_DIR / "paper4_v398_notebook_historical_e402_decision.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v398_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v398_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    top_file = file_summary.iloc[0]
    status = {
        "phase": "v398_notebook_historical_e402_policy",
        "schema_version": "2026-05-17.398",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_side_effect_patch_version_v398": PRIOR_SIDE_EFFECT_PATCH_VERSION,
        "ruff_e402_command_v398": RUFF_E402_COMMAND,
        "e402_diagnostics_v398": int(len(manifest)),
        "e402_file_rows_v398": int(len(file_summary)),
        "e402_cell_rows_v398": int(len(cell_summary)),
        "decision_rows_v398": int(len(decisions)),
        "claim_blocker_rows_v398": int(len(blockers)),
        "claim_matrix_rows_v398": int(len(claim_matrix)),
        "top_e402_notebook_v398": str(top_file["notebook_path_v398"]),
        "top_e402_count_v398": int(top_file["e402_diagnostic_count_v398"]),
        "selected_policy_v398": "cell_local_refactor_plan",
        "bulk_import_reorder_allowed_v398": False,
        "notebook_exclusion_allowed_v398": False,
        "notebooks_mutated_v398": False,
        "notebook_diff_clean_before_v398": diff_clean_before,
        "notebook_diff_clean_after_v398": _notebook_diff_clean(),
        "global_notebook_diagnostics_v398": 139,
        "global_ruff_clean_v398": False,
        "full_repository_pytest_run_v398": False,
        "full_quarto_render_run_v398": False,
        "working_champion_claim_allowed_v398": False,
        "paper1_promotion_allowed_v398": False,
        "paper4_working_champion_changed_v398": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v398": NEXT_ARTIFACT,
        "claim_boundary": (
            "v398 inventories historical E402 and selects cell-local planning; no "
            "notebooks are mutated and final promotion remains blocked"
        ),
    }
    POLICY_MD.write_text(_policy_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v398_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v398": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
