#!/usr/bin/env python3
"""Build Paper 4 v412 F821 execution-context audit artifacts."""

from __future__ import annotations

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

VERSION = 412
PRIOR_PYTEST_PROBE_VERSION = 411
NEXT_ARTIFACT = "paper4_v413_notebook_f821_validation_target_patch.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v412_notebook_f821_execution_context_audit.md"
TARGET_NOTEBOOK = "notebooks/02_feature_engineering.ipynb"
TARGET_CELL = 32
TARGET_UNDEFINED_NAME = "train_fe"


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


def _read_notebook(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _cell_source(path: str, cell_number: int) -> list[str]:
    payload = _read_notebook(path)
    return list(payload["cells"][cell_number - 1].get("source", []))


def _contains_assignment(source: str, name: str) -> bool:
    assignment_prefixes = (f"{name} =", f"{name}:")
    return any(line.lstrip().startswith(assignment_prefixes) for line in source.splitlines())


def _f821_audit_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        notebook_path = Path(item["filename"]).relative_to(ROOT).as_posix()
        cell_number = int(item.get("cell") or 0)
        row_number = int((item.get("location") or {}).get("row") or 0)
        lines = _cell_source(notebook_path, cell_number)
        code_line = lines[row_number - 1].strip()
        rows.append(
            {
                "audit_id_v412": f"f821_execution_context_{len(rows) + 1:02d}",
                "notebook_path_v412": notebook_path,
                "cell_v412": cell_number,
                "row_v412": row_number,
                "undefined_name_v412": TARGET_UNDEFINED_NAME,
                "code_line_v412": code_line,
                "diagnosis_v412": (
                    "train_fe is referenced in the final Pandera validation cell but is not "
                    "assigned anywhere in the notebook execution context"
                ),
                "recommended_patch_v412": (
                    "introduce validation_target = script_train if script_train exists else train, "
                    "then validate validation_target"
                ),
                "mutation_allowed_v412": False,
                "claim_boundary_v412": "execution-context audit only; no notebook mutation",
            }
        )
    return pd.DataFrame(rows)


def _context_rows() -> pd.DataFrame:
    payload = _read_notebook(TARGET_NOTEBOOK)
    joined_sources = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    notebook_source = "\n".join(joined_sources)
    target_source = "".join(_cell_source(TARGET_NOTEBOOK, TARGET_CELL))
    return pd.DataFrame(
        [
            {
                "context_id_v412": "train_fe_assignment_scan",
                "notebook_path_v412": TARGET_NOTEBOOK,
                "evidence_v412": str(_contains_assignment(notebook_source, "train_fe")),
                "interpretation_v412": "no train_fe assignment found in code cells",
                "claim_boundary_v412": "static notebook source scan only",
            },
            {
                "context_id_v412": "script_train_assignment_scan",
                "notebook_path_v412": TARGET_NOTEBOOK,
                "evidence_v412": str(_contains_assignment(notebook_source, "script_train")),
                "interpretation_v412": "script_train is assigned when train_fe.parquet exists",
                "claim_boundary_v412": "static notebook source scan only",
            },
            {
                "context_id_v412": "train_assignment_scan",
                "notebook_path_v412": TARGET_NOTEBOOK,
                "evidence_v412": str(_contains_assignment(notebook_source, "train")),
                "interpretation_v412": "train is the in-memory engineered dataframe used earlier",
                "claim_boundary_v412": "static notebook source scan only",
            },
            {
                "context_id_v412": "target_cell_source",
                "notebook_path_v412": TARGET_NOTEBOOK,
                "evidence_v412": target_source.strip(),
                "interpretation_v412": "target cell validates train_fe directly",
                "claim_boundary_v412": "cell source captured for patch planning",
            },
        ]
    )


def _patch_plan(audit: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patch_batch_id_v412": "batch_1_f821_validation_target",
                "diagnostic_count_v412": int(len(audit)),
                "notebook_count_v412": int(audit["notebook_path_v412"].nunique()),
                "recommended_next_artifact_v412": NEXT_ARTIFACT,
                "mutation_allowed_v412": False,
                "claim_boundary_v412": "selection only; validation-target patch deferred to v413",
            }
        ]
    )


def _claim_blockers(global_after: int, f821_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v412": "f821_validation_target_patch_not_applied_yet",
                "blocking_v412": True,
                "evidence_count_v412": f821_after,
                "required_next_artifact_v412": NEXT_ARTIFACT,
                "claim_boundary_v412": "v412 audits but does not mutate the notebook",
            },
            {
                "blocker_id_v412": "global_notebook_lint_not_clean",
                "blocking_v412": True,
                "evidence_count_v412": global_after,
                "required_next_artifact_v412": NEXT_ARTIFACT,
                "claim_boundary_v412": "F821 and style notebook lint remain",
            },
            {
                "blocker_id_v412": "paper4_final_promotion_forbidden",
                "blocking_v412": True,
                "evidence_count_v412": 1,
                "required_next_artifact_v412": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v412": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v412_f821_execution_context_audit_created",
                "allowed": True,
                "artifact": "paper4_v412_notebook_f821_execution_context_audit.csv",
                "boundary": "one F821 execution-context diagnostic audited",
            },
            {
                "claim_id": "v412_f821_validation_target_patch_selected",
                "allowed": True,
                "artifact": "paper4_v412_notebook_f821_patch_plan.csv",
                "boundary": "patch selected for v413",
            },
            {
                "claim_id": "v412_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v412",
            },
            {
                "claim_id": "v412_f821_repaired",
                "allowed": False,
                "artifact": "paper4_v412_claim_blockers.csv",
                "boundary": "audit only",
            },
            {
                "claim_id": "v412_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v412_claim_blockers.csv",
                "boundary": "7 notebook diagnostics remain",
            },
            {
                "claim_id": "v412_working_champion_or_final_promotion",
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
                "claim": "v412 audits the remaining F821 notebook execution-context diagnostic.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v412_notebook_f821_execution_context_audit.csv",
                "boundary": "Audit only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v412 selects a validation-target patch for the train_fe F821.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v412_notebook_f821_patch_plan.csv",
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v412 repairs F821 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v412_claim_blockers.csv",
                "boundary": "No notebook mutation; 7 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v412 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v412_claim_blockers.csv",
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
                "executable_item": "v412 audits the remaining F821 train_fe execution-context diagnostic.",
                "status": "notebook_f821_execution_context_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v413 applies the validation-target patch with roundtrip checks",
                "last_wave": "v412",
                "execution_result": "f821_train_fe_context_audited_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v412")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook F821 Execution-Context Audit v412

Generated: {status["generated_at_utc"]}

v412 audits the one remaining F821 diagnostic before mutation.

## Result

- F821 diagnostics reviewed: `{status["f821_diagnostics_v412"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v412"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v412"]}`.
- Recommended next artifact: `{status["next_artifact_v412"]}`.

## Interpretation

The final validation cell references `train_fe`, but the notebook never assigns
that name. Earlier cells already use the in-memory `train` dataframe and may
load `script_train` from the canonical parquet artifact.

## Required Caveat

v412 is non-mutating. It does not repair F821, clear notebook lint, or create
Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V412_NOTEBOOK_F821_EXECUTION_CONTEXT_AUDIT_START -->"
    end = "<!-- V412_NOTEBOOK_F821_EXECUTION_CONTEXT_AUDIT_END -->"
    block = f"""
{start}

## Wave v412: Notebook F821 Execution-Context Audit

Generated: {status["generated_at_utc"]}

### Objective

v412 audits the one remaining F821 notebook diagnostic before mutation.

### Results

- F821 diagnostics reviewed:
  `{status["f821_diagnostics_v412"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v412"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v412"]}`.
- Notebook diff clean after audit:
  `{status["notebook_diff_clean_after_v412"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v412"]}`.

### Interpretation

`train_fe` is referenced in the Pandera validation cell but is not assigned in
the notebook. v413 should introduce an explicit validation target using
`script_train` when available, otherwise the in-memory `train` dataframe.

### Claim Impact

- Allowed: F821 execution-context audit and v413 patch selection.
- Still prohibited: F821 repaired, notebook lint clean, repository ruff clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v412 in the living notebook. v413 should apply the validation-target patch
with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    clean_before = _notebook_diff_clean()
    if not clean_before:
        raise RuntimeError("v412 expects clean notebook diff before audit.")

    v411_status = json.loads((STATUS_DIR / "paper4_v411_status.json").read_text(encoding="utf-8"))
    if v411_status["next_artifact_v411"] != "paper4_v412_notebook_f821_execution_context_audit.md":
        raise RuntimeError("v412 expects v411 to route to F821 execution-context audit.")

    global_items = _run_ruff_json()
    counts = Counter(item["code"] for item in global_items)
    f821_items = _run_ruff_json(["F821"])
    audit = _f821_audit_rows(f821_items)
    context = _context_rows()
    patch_plan = _patch_plan(audit)
    blockers = _claim_blockers(
        global_after=len(global_items),
        f821_after=counts.get("F821", 0),
    )
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v412_notebook_f821_execution_context_audit.csv", audit)
    write_csv(TABLE_DIR / "paper4_v412_notebook_f821_context_evidence.csv", context)
    write_csv(TABLE_DIR / "paper4_v412_notebook_f821_patch_plan.csv", patch_plan)
    write_csv(TABLE_DIR / "paper4_v412_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v412_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v412_notebook_f821_execution_context_audit",
        "schema_version": "2026-05-17.412",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_pytest_probe_version_v412": PRIOR_PYTEST_PROBE_VERSION,
        "global_notebook_diagnostics_v412": int(len(global_items)),
        "f821_diagnostics_v412": int(len(f821_items)),
        "f821_target_notebook_v412": TARGET_NOTEBOOK,
        "f821_target_cell_v412": TARGET_CELL,
        "notebooks_mutated_v412": False,
        "notebook_diff_clean_before_v412": clean_before,
        "notebook_diff_clean_after_v412": _notebook_diff_clean(),
        "patch_plan_rows_v412": int(len(patch_plan)),
        "global_ruff_clean_v412": False,
        "full_repository_pytest_run_v412": False,
        "full_quarto_render_run_v412": False,
        "working_champion_claim_allowed_v412": False,
        "paper1_promotion_allowed_v412": False,
        "paper4_working_champion_changed_v412": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v412": NEXT_ARTIFACT,
        "claim_boundary": (
            "v412 audits F821 execution context and selects a v413 patch; no notebook "
            "mutation or final promotion is allowed"
        ),
    }
    if status["f821_diagnostics_v412"] != 1:
        raise RuntimeError("v412 expected exactly one F821 diagnostic.")
    if status["notebook_diff_clean_after_v412"] is not True:
        raise RuntimeError("v412 unexpectedly mutated notebooks.")
    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v412": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
