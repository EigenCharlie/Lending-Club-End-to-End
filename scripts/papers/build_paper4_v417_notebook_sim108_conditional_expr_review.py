#!/usr/bin/env python3
"""Build Paper 4 v417 SIM108 conditional-expression review artifacts."""

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

VERSION = 417
PRIOR_E741_PATCH_VERSION = 416
NEXT_ARTIFACT = "paper4_v418_notebook_sim108_conditional_expr_patch.md"
REVIEW_MD = NOTEBOOK.parent / "paper4_v417_notebook_sim108_conditional_expr_review.md"


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


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def _cell_source(notebook_path: str, cell_number: int) -> list[str]:
    payload = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
    return list(payload["cells"][cell_number - 1].get("source", []))


def _recommended_patch(source_line: str) -> str:
    if source_line == "if calibrator is not None:":
        return "y_prob_test_cal = calibrator.predict(y_prob_test) if calibrator is not None else y_prob_test"
    if source_line == "if ead_col in test.columns:":
        return "ead = test[ead_col].values if ead_col in test.columns else np.ones(len(test)) * 15000"
    return "manual_review"


def _review_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in [entry for entry in items if entry["code"] == "SIM108"]:
        notebook_path = _relative_path(str(item["filename"]))
        cell_number = int(item.get("cell") or 0)
        row_number = int((item.get("location") or {}).get("row") or 0)
        lines = _cell_source(notebook_path, cell_number)
        source_line = lines[row_number - 1].strip()
        assigned_line = lines[row_number].strip() if row_number < len(lines) else ""
        else_line = lines[row_number + 1].strip() if row_number + 1 < len(lines) else ""
        fallback_line = lines[row_number + 2].strip() if row_number + 2 < len(lines) else ""
        patch = _recommended_patch(source_line)
        rows.append(
            {
                "review_id_v417": f"sim108_conditional_expr_{len(rows) + 1:02d}",
                "notebook_path_v417": notebook_path,
                "cell_v417": cell_number,
                "row_v417": row_number,
                "source_if_line_v417": source_line,
                "then_assignment_v417": assigned_line,
                "else_line_v417": else_line,
                "else_assignment_v417": fallback_line,
                "recommended_patch_v417": patch,
                "semantics_review_v417": (
                    "single_assignment_ifelse_no_branch_side_effects"
                    if patch != "manual_review"
                    else "requires_manual_review"
                ),
                "selected_for_v418_v417": patch != "manual_review",
                "mutation_allowed_v417": False,
                "claim_boundary_v417": "SIM108 review only; no notebook mutation",
            }
        )
    return pd.DataFrame(rows)


def _batch_plan(review: pd.DataFrame) -> pd.DataFrame:
    selected = review.loc[review["selected_for_v418_v417"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "patch_batch_id_v417": "batch_1_sim108_conditional_expr",
                "diagnostic_count_v417": int(len(selected)),
                "notebook_count_v417": int(selected["notebook_path_v417"].nunique()),
                "recommended_next_artifact_v417": NEXT_ARTIFACT,
                "mutation_allowed_v417": False,
                "claim_boundary_v417": "selection only; SIM108 patch deferred to v418",
            }
        ]
    )


def _claim_blockers(global_after: int, sim108_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v417": "sim108_conditional_expr_patch_not_applied_yet",
                "blocking_v417": True,
                "evidence_count_v417": sim108_after,
                "required_next_artifact_v417": NEXT_ARTIFACT,
                "claim_boundary_v417": "v417 reviews but does not mutate notebooks",
            },
            {
                "blocker_id_v417": "style_notebook_lint_remaining",
                "blocking_v417": True,
                "evidence_count_v417": global_after,
                "required_next_artifact_v417": NEXT_ARTIFACT,
                "claim_boundary_v417": "SIM108/E712/SIM102 style notebook lint remains",
            },
            {
                "blocker_id_v417": "paper4_final_promotion_forbidden",
                "blocking_v417": True,
                "evidence_count_v417": 1,
                "required_next_artifact_v417": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v417": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v417_sim108_conditional_expr_review_created",
                "allowed": True,
                "artifact": "paper4_v417_notebook_sim108_conditional_expr_review.csv",
                "boundary": "2 SIM108 diagnostics reviewed",
            },
            {
                "claim_id": "v417_sim108_patch_selected",
                "allowed": True,
                "artifact": "paper4_v417_notebook_sim108_patch_plan.csv",
                "boundary": "2 conditional-expression patches selected for v418",
            },
            {
                "claim_id": "v417_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v417",
            },
            {
                "claim_id": "v417_sim108_repaired",
                "allowed": False,
                "artifact": "paper4_v417_claim_blockers.csv",
                "boundary": "review only",
            },
            {
                "claim_id": "v417_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v417_claim_blockers.csv",
                "boundary": "5 notebook diagnostics remain",
            },
            {
                "claim_id": "v417_working_champion_or_final_promotion",
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
                "claim": "v417 reviews the remaining 2 SIM108 notebook diagnostics.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v417_notebook_sim108_conditional_expr_review.csv",
                "boundary": "Review only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v417 selects SIM108 conditional-expression patches for v418.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v417_notebook_sim108_patch_plan.csv",
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v417 repairs SIM108 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v417_claim_blockers.csv",
                "boundary": "No notebook mutation; 5 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v417 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v417_claim_blockers.csv",
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
                "executable_item": "v417 reviews SIM108 conditional-expression notebook refactors before mutation.",
                "status": "notebook_sim108_conditional_expr_review_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v418 applies both SIM108 conditional-expression patches with roundtrip checks",
                "last_wave": "v417",
                "execution_result": "two_sim108_refactors_reviewed_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v417")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _review_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook SIM108 Conditional-Expression Review v417

Generated: {status["generated_at_utc"]}

v417 reviews the remaining SIM108 notebook diagnostics before mutation.

## Result

- SIM108 diagnostics reviewed: `{status["sim108_diagnostics_v417"]}`.
- Selected for v418: `{status["selected_for_v418_rows_v417"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v417"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v417"]}`.
- Next artifact: `{status["next_artifact_v417"]}`.

## Required Caveat

v417 is non-mutating. It does not repair SIM108, clear notebook lint, or create
Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V417_NOTEBOOK_SIM108_CONDITIONAL_EXPR_REVIEW_START -->"
    end = "<!-- V417_NOTEBOOK_SIM108_CONDITIONAL_EXPR_REVIEW_END -->"
    block = f"""
{start}

## Wave v417: Notebook SIM108 Conditional-Expression Review

Generated: {status["generated_at_utc"]}

### Objective

v417 reviews the two remaining SIM108 notebook diagnostics before mutation.

### Results

- SIM108 diagnostics reviewed:
  `{status["sim108_diagnostics_v417"]}`.
- Selected for v418:
  `{status["selected_for_v418_rows_v417"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v417"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v417"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v417"]}`.

### Interpretation

Both SIM108 rows are single-assignment if/else blocks in notebook 04 and are
eligible for an explicit conditional-expression patch.

### Claim Impact

- Allowed: SIM108 review and v418 patch selection.
- Still prohibited: SIM108 repaired, notebook lint clean, repository ruff clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v417 in the living notebook. v418 should apply the SIM108 patch with
roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    clean_before = _notebook_diff_clean()
    if not clean_before:
        raise RuntimeError("v417 expects clean notebook diff before review.")

    v416_status = json.loads((STATUS_DIR / "paper4_v416_status.json").read_text(encoding="utf-8"))
    if v416_status["next_artifact_v416"] != "paper4_v417_notebook_sim108_conditional_expr_review.md":
        raise RuntimeError("v417 expects v416 to route to SIM108 review.")

    global_items = _run_ruff_json()
    counts = Counter(item["code"] for item in global_items)
    review = _review_rows(global_items)
    batch_plan = _batch_plan(review)
    blockers = _claim_blockers(
        global_after=len(global_items),
        sim108_after=counts.get("SIM108", 0),
    )
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v417_notebook_sim108_conditional_expr_review.csv", review)
    write_csv(TABLE_DIR / "paper4_v417_notebook_sim108_patch_plan.csv", batch_plan)
    write_csv(TABLE_DIR / "paper4_v417_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v417_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v417_notebook_sim108_conditional_expr_review",
        "schema_version": "2026-05-17.417",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_e741_patch_version_v417": PRIOR_E741_PATCH_VERSION,
        "global_notebook_diagnostics_v417": int(len(global_items)),
        "sim108_diagnostics_v417": int(counts.get("SIM108", 0)),
        "e712_diagnostics_v417": int(counts.get("E712", 0)),
        "sim102_diagnostics_v417": int(counts.get("SIM102", 0)),
        "selected_for_v418_rows_v417": int(review["selected_for_v418_v417"].astype(bool).sum()),
        "patch_plan_rows_v417": int(len(batch_plan)),
        "notebooks_mutated_v417": False,
        "notebook_diff_clean_before_v417": clean_before,
        "notebook_diff_clean_after_v417": _notebook_diff_clean(),
        "global_ruff_clean_v417": False,
        "full_repository_pytest_run_v417": False,
        "full_quarto_render_run_v417": False,
        "working_champion_claim_allowed_v417": False,
        "paper1_promotion_allowed_v417": False,
        "paper4_working_champion_changed_v417": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v417": NEXT_ARTIFACT,
        "claim_boundary": (
            "v417 reviews SIM108 conditional-expression refactors and selects v418; no "
            "notebook mutation or final promotion is allowed"
        ),
    }
    if status["sim108_diagnostics_v417"] != 2:
        raise RuntimeError("v417 expected exactly two SIM108 diagnostics.")
    if status["selected_for_v418_rows_v417"] != 2:
        raise RuntimeError("v417 expected both SIM108 diagnostics selected.")
    if status["notebook_diff_clean_after_v417"] is not True:
        raise RuntimeError("v417 unexpectedly mutated notebooks.")
    REVIEW_MD.write_text(_review_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v417": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
