#!/usr/bin/env python3
"""Build Paper 4 v407 non-E402 notebook lint triage artifacts."""

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

VERSION = 407
PRIOR_POST_E402_PYTEST_VERSION = 406
NEXT_VERSION = 408
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_b007_loop_var_patch.md"
TRIAGE_MD = NOTEBOOK.parent / "paper4_v407_notebook_non_e402_lint_triage.md"


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
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("ruff notebook JSON output is not a list")
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


def _fix_applicability(item: dict[str, Any]) -> str:
    fix = item.get("fix")
    if not fix:
        return "none"
    return str(fix.get("applicability") or "unknown")


def _planned_batch(code: str, notebook_path: str) -> tuple[str, str, str]:
    if code == "B007":
        return (
            "batch_1_b007_loop_var_rename",
            "low_safe_loop_variable_cleanup",
            "rename unused loop variables to underscore-prefixed names",
        )
    if code == "B018":
        return (
            "batch_2_b018_notebook_display_review",
            "medium_display_semantics",
            "review useless expressions as possible intentional notebook display outputs",
        )
    if code == "F821":
        return (
            "batch_3_f821_execution_context_audit",
            "high_execution_context",
            "audit undefined notebook execution dependency before mutation",
        )
    if "side_projects" in notebook_path:
        return (
            "batch_4_side_project_style_cleanup",
            "medium_side_project_semantics",
            "defer side-project style cleanup until main Paper 4 notebooks stay stable",
        )
    return (
        "batch_5_manual_style_cleanup",
        "medium_manual_style",
        "manual readability cleanup after safer batches",
    )


def _manifest(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(items, start=1):
        notebook_path = _relative_path(str(item["filename"]))
        code = str(item["code"])
        batch, risk, action = _planned_batch(code, notebook_path)
        location = item.get("location") or {}
        rows.append(
            {
                "diagnostic_id_v407": f"non_e402_{idx:03d}",
                "notebook_path_v407": notebook_path,
                "cell_v407": int(item.get("cell") or 0),
                "row_v407": int(location.get("row") or 0),
                "rule_code_v407": code,
                "message_v407": str(item["message"]),
                "fix_applicability_v407": _fix_applicability(item),
                "planned_batch_v407": batch,
                "risk_class_v407": risk,
                "recommended_action_v407": action,
                "mutation_allowed_v407": False,
                "claim_boundary_v407": "triage only; no notebook mutation in v407",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(manifest: pd.DataFrame) -> pd.DataFrame:
    order = {
        "batch_1_b007_loop_var_rename": 1,
        "batch_2_b018_notebook_display_review": 2,
        "batch_3_f821_execution_context_audit": 3,
        "batch_4_side_project_style_cleanup": 4,
        "batch_5_manual_style_cleanup": 5,
    }
    rows = []
    for batch, group in manifest.groupby("planned_batch_v407", sort=False):
        rows.append(
            {
                "batch_id_v407": batch,
                "execution_order_v407": order.get(batch, 99),
                "diagnostic_count_v407": int(len(group)),
                "rule_codes_v407": ",".join(sorted(group["rule_code_v407"].unique())),
                "risk_class_v407": str(group["risk_class_v407"].iloc[0]),
                "mutation_allowed_v407": False,
                "next_action_v407": (
                    NEXT_ARTIFACT
                    if batch == "batch_1_b007_loop_var_rename"
                    else "deferred_after_b007_patch"
                ),
                "claim_boundary_v407": "batch plan only; no notebook mutation in v407",
            }
        )
    return pd.DataFrame(rows).sort_values("execution_order_v407", ignore_index=True)


def _claim_blockers(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = manifest["planned_batch_v407"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "blocker_id_v407": "b007_loop_var_patch_not_applied_yet",
                "blocking_v407": True,
                "evidence_count_v407": int(counts.get("batch_1_b007_loop_var_rename", 0)),
                "required_next_artifact_v407": NEXT_ARTIFACT,
                "claim_boundary_v407": "v407 selects but does not mutate B007 loop variables",
            },
            {
                "blocker_id_v407": "b018_display_review_deferred",
                "blocking_v407": True,
                "evidence_count_v407": int(counts.get("batch_2_b018_notebook_display_review", 0)),
                "required_next_artifact_v407": "paper4_v409_notebook_b018_display_review.md",
                "claim_boundary_v407": "display semantics require review",
            },
            {
                "blocker_id_v407": "f821_execution_context_deferred",
                "blocking_v407": True,
                "evidence_count_v407": int(counts.get("batch_3_f821_execution_context_audit", 0)),
                "required_next_artifact_v407": "paper4_v410_notebook_f821_execution_context_audit.md",
                "claim_boundary_v407": "undefined execution context requires audit",
            },
            {
                "blocker_id_v407": "paper4_final_promotion_forbidden",
                "blocking_v407": True,
                "evidence_count_v407": 1,
                "required_next_artifact_v407": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v407": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v407_non_e402_lint_triage_created",
                "allowed": True,
                "artifact": "paper4_v407_notebook_non_e402_lint_manifest.csv",
                "boundary": "20 non-E402 notebook diagnostics inventoried",
            },
            {
                "claim_id": "v407_b007_first_batch_selected",
                "allowed": True,
                "artifact": "paper4_v407_notebook_non_e402_lint_batch_plan.csv",
                "boundary": "3 B007 diagnostics selected for v408",
            },
            {
                "claim_id": "v407_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v407",
            },
            {
                "claim_id": "v407_non_e402_lint_repaired",
                "allowed": False,
                "artifact": "paper4_v407_claim_blockers.csv",
                "boundary": "triage only",
            },
            {
                "claim_id": "v407_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v407_claim_blockers.csv",
                "boundary": "20 notebook diagnostics remain",
            },
            {
                "claim_id": "v407_working_champion_or_final_promotion",
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
                "claim": "v407 inventories the 20 remaining non-E402 notebook diagnostics.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v407_notebook_non_e402_lint_manifest.csv"
                ),
                "boundary": "Triage only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v407 selects a 3-diagnostic B007 loop-variable batch for v408.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v407_notebook_non_e402_lint_batch_plan.csv"
                ),
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v407 repairs non-E402 lint or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v407_claim_blockers.csv",
                "boundary": "No notebook mutation; 20 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v407 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v407_claim_blockers.csv",
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
                "executable_item": "v407 triages remaining non-E402 notebook lint after E402 clearance.",
                "status": "notebook_non_e402_lint_triage_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v408 applies only B007 loop-variable renames with roundtrip checks",
                "last_wave": "v407",
                "execution_result": "non_e402_lint_20_diagnostics_triaged_b007_first_batch",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v407")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _triage_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Non-E402 Notebook Lint Triage v407

Generated: {status["generated_at_utc"]}

v407 inventories the remaining non-E402 notebook lint after v405 cleared E402
and v406 passed full pytest.

## Result

- Remaining notebook diagnostics: `{status["global_notebook_diagnostics_v407"]}`.
- E402 diagnostics: `{status["global_notebook_e402_v407"]}`.
- B007 diagnostics selected for v408: `{status["b007_diagnostics_v407"]}`.
- B018 display-review diagnostics deferred: `{status["b018_diagnostics_v407"]}`.
- F821 execution-context diagnostics deferred: `{status["f821_diagnostics_v407"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v407"]}`.

## Required Caveat

v407 does not repair non-E402 lint, does not make notebook or repository ruff
clean, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v407"]}` for the B007 loop-variable batch.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V407_NOTEBOOK_NON_E402_LINT_TRIAGE_START -->"
    end = "<!-- V407_NOTEBOOK_NON_E402_LINT_TRIAGE_END -->"
    block = f"""
{start}

## Wave v407: Notebook Non-E402 Lint Triage

Generated: {status["generated_at_utc"]}

### Objective

v407 inventories the 20 remaining non-E402 notebook diagnostics after E402
clearance and selects the first low-risk cleanup batch.

### Results

- Remaining notebook diagnostics:
  `{status["global_notebook_diagnostics_v407"]}`.
- E402 diagnostics:
  `{status["global_notebook_e402_v407"]}`.
- B007 diagnostics selected for v408:
  `{status["b007_diagnostics_v407"]}`.
- B018 display-review diagnostics deferred:
  `{status["b018_diagnostics_v407"]}`.
- F821 execution-context diagnostics deferred:
  `{status["f821_diagnostics_v407"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v407"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v407"]}`.

### Interpretation

The safest next mutation is the small B007 loop-variable rename batch. B018
diagnostics may be intentional display expressions, and F821 needs execution
context review before mutation.

### Claim Impact

- Allowed: non-E402 lint triage and B007 first-batch selection.
- Still prohibited: non-E402 lint repaired, notebook lint clean, repository ruff
  clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v407 in the living notebook. v408 should apply only the B007 loop-variable
patch with roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v407 expects clean notebook diff because it is plan-only.")

    v406_status = json.loads((STATUS_DIR / "paper4_v406_status.json").read_text(encoding="utf-8"))
    if v406_status["next_artifact_v406"] != "paper4_v407_notebook_non_e402_lint_triage.md":
        raise RuntimeError("v407 expects v406 to route to non-E402 lint triage.")
    if not v406_status["pytest_passed_v406"]:
        raise RuntimeError("v407 requires v406 pytest to pass.")

    items = _run_ruff_json()
    counts = Counter(item["code"] for item in items)
    if counts.get("E402", 0) != 0:
        raise RuntimeError("v407 expects notebook E402 to be clear.")
    manifest = _manifest(items)
    batch_plan = _batch_plan(manifest)
    blockers = _claim_blockers(manifest)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v407_notebook_non_e402_lint_manifest.csv", manifest)
    write_csv(TABLE_DIR / "paper4_v407_notebook_non_e402_lint_batch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v407_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v407_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v407_notebook_non_e402_lint_triage",
        "schema_version": "2026-05-17.407",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_e402_pytest_version_v407": PRIOR_POST_E402_PYTEST_VERSION,
        "global_notebook_diagnostics_v407": int(len(items)),
        "global_notebook_e402_v407": int(counts.get("E402", 0)),
        "b007_diagnostics_v407": int(counts.get("B007", 0)),
        "b018_diagnostics_v407": int(counts.get("B018", 0)),
        "f821_diagnostics_v407": int(counts.get("F821", 0)),
        "batch_plan_rows_v407": int(len(batch_plan)),
        "claim_blocker_rows_v407": int(len(blockers)),
        "claim_matrix_rows_v407": int(len(claim_matrix)),
        "selected_first_batch_v407": "batch_1_b007_loop_var_rename",
        "notebooks_mutated_v407": False,
        "notebook_diff_clean_before_v407": True,
        "notebook_diff_clean_after_v407": _notebook_diff_clean(),
        "global_ruff_clean_v407": False,
        "full_repository_pytest_run_v407": False,
        "full_quarto_render_run_v407": False,
        "working_champion_claim_allowed_v407": False,
        "paper1_promotion_allowed_v407": False,
        "paper4_working_champion_changed_v407": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v407": NEXT_ARTIFACT,
        "claim_boundary": (
            "v407 triages non-E402 notebook lint and selects B007; no notebooks "
            "are mutated and final promotion remains blocked"
        ),
    }
    TRIAGE_MD.write_text(_triage_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v407": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
