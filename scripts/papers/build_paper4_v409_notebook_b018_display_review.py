#!/usr/bin/env python3
"""Build Paper 4 v409 B018 notebook display review artifacts."""

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

VERSION = 409
PRIOR_B007_PATCH_VERSION = 408
NEXT_VERSION = 410
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_b018_fig_show_patch.md"
REVIEW_MD = NOTEBOOK.parent / "paper4_v409_notebook_b018_display_review.md"


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
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def _read_cell_lines(notebook_path: str, cell_number: int) -> list[str]:
    payload = json.loads((ROOT / notebook_path).read_text(encoding="utf-8"))
    return list(payload["cells"][cell_number - 1].get("source", []))


def _display_review_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    b018_items = [item for item in items if item["code"] == "B018"]
    for idx, item in enumerate(b018_items, start=1):
        notebook_path = _relative_path(str(item["filename"]))
        cell_number = int(item.get("cell") or 0)
        row_number = int((item.get("location") or {}).get("row") or 0)
        lines = _read_cell_lines(notebook_path, cell_number)
        expression = lines[row_number - 1].strip()
        following = lines[row_number].strip() if row_number < len(lines) else ""
        is_exported = following.startswith("export_figure(")
        rows.append(
            {
                "display_id_v409": f"b018_display_{idx:02d}",
                "notebook_path_v409": notebook_path,
                "cell_v409": cell_number,
                "row_v409": row_number,
                "display_expression_v409": expression,
                "following_statement_v409": following,
                "display_semantics_v409": (
                    "intentional_plotly_display_then_export"
                    if is_exported
                    else "requires_manual_review"
                ),
                "recommended_patch_v409": (
                    f"{expression}.show()"
                    if is_exported and expression.startswith("fig")
                    else "manual_review"
                ),
                "mutation_allowed_v409": False,
                "claim_boundary_v409": "display review only; no notebook mutation in v409",
            }
        )
    return pd.DataFrame(rows)


def _patch_plan(review: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patch_batch_id_v409": "batch_1_b018_fig_show_patch",
                "diagnostic_count_v409": int(len(review)),
                "notebook_count_v409": int(review["notebook_path_v409"].nunique()),
                "recommended_next_artifact_v409": NEXT_ARTIFACT,
                "mutation_allowed_v409": False,
                "claim_boundary_v409": "selection only; fig.show patch deferred to v410",
            }
        ]
    )


def _claim_blockers(global_after: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v409": "b018_fig_show_patch_not_applied_yet",
                "blocking_v409": True,
                "evidence_count_v409": 10,
                "required_next_artifact_v409": NEXT_ARTIFACT,
                "claim_boundary_v409": "v409 reviews but does not mutate display expressions",
            },
            {
                "blocker_id_v409": "global_notebook_lint_not_clean",
                "blocking_v409": True,
                "evidence_count_v409": global_after,
                "required_next_artifact_v409": NEXT_ARTIFACT,
                "claim_boundary_v409": "B018 and other notebook lint remain",
            },
            {
                "blocker_id_v409": "paper4_final_promotion_forbidden",
                "blocking_v409": True,
                "evidence_count_v409": 1,
                "required_next_artifact_v409": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v409": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v409_b018_display_review_created",
                "allowed": True,
                "artifact": "paper4_v409_notebook_b018_display_review.csv",
                "boundary": "10 B018 display expressions reviewed",
            },
            {
                "claim_id": "v409_fig_show_patch_selected",
                "allowed": True,
                "artifact": "paper4_v409_notebook_b018_patch_plan.csv",
                "boundary": "fig.show patch selected for v410",
            },
            {
                "claim_id": "v409_notebooks_preserved_unmodified",
                "allowed": True,
                "artifact": "git diff --name-only -- notebooks",
                "boundary": "no notebook mutation in v409",
            },
            {
                "claim_id": "v409_b018_repaired",
                "allowed": False,
                "artifact": "paper4_v409_claim_blockers.csv",
                "boundary": "review only",
            },
            {
                "claim_id": "v409_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v409_claim_blockers.csv",
                "boundary": "17 notebook diagnostics remain",
            },
            {
                "claim_id": "v409_working_champion_or_final_promotion",
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
                "claim": "v409 reviews the 10 B018 notebook display expressions.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v409_notebook_b018_display_review.csv",
                "boundary": "Review only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v409 selects a fig.show patch for B018 display expressions.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v409_notebook_b018_patch_plan.csv",
                "boundary": "Selection only; application deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v409 repairs B018 or clears notebook lint.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v409_claim_blockers.csv",
                "boundary": "No notebook mutation; 17 diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v409 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v409_claim_blockers.csv",
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
                "executable_item": "v409 reviews B018 display-expression semantics before mutation.",
                "status": "notebook_b018_display_review_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v410 applies fig.show display patch with roundtrip checks",
                "last_wave": "v409",
                "execution_result": "b018_10_display_expressions_reviewed_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v409")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _review_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook B018 Display Review v409

Generated: {status["generated_at_utc"]}

v409 reviews the 10 B018 expressions remaining after v408.

## Result

- B018 diagnostics reviewed: `{status["b018_diagnostics_v409"]}`.
- Intentional display-then-export rows: `{status["intentional_display_rows_v409"]}`.
- Proposed next patch: `{status["next_artifact_v409"]}`.
- Notebooks mutated: `{status["notebooks_mutated_v409"]}`.

## Required Caveat

v409 does not repair B018, does not clear notebook or repository ruff, and does
not create Paper 4 final promotion.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V409_NOTEBOOK_B018_DISPLAY_REVIEW_START -->"
    end = "<!-- V409_NOTEBOOK_B018_DISPLAY_REVIEW_END -->"
    block = f"""
{start}

## Wave v409: Notebook B018 Display Review

Generated: {status["generated_at_utc"]}

### Objective

v409 reviews the 10 remaining B018 notebook expressions before mutation.

### Results

- B018 diagnostics reviewed:
  `{status["b018_diagnostics_v409"]}`.
- Intentional display-then-export rows:
  `{status["intentional_display_rows_v409"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v409"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v409"]}`.

### Interpretation

All B018 rows are plotly figure display expressions immediately followed by
`export_figure(...)`. v410 can replace the bare figure expressions with explicit
`fig.show()` calls while preserving display intent.

### Claim Impact

- Allowed: B018 display review and fig.show patch selection.
- Still prohibited: B018 repaired, notebook lint clean, repository ruff clean,
  champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v409 in the living notebook. v410 should apply the fig.show patch with
roundtrip checks.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v409 expects clean notebook diff because it is review-only.")

    v408_status = json.loads((STATUS_DIR / "paper4_v408_status.json").read_text(encoding="utf-8"))
    if v408_status["next_artifact_v408"] != "paper4_v409_notebook_b018_display_review.md":
        raise RuntimeError("v409 expects v408 to route to B018 display review.")

    items = _run_ruff_json()
    counts = Counter(item["code"] for item in items)
    review = _display_review_rows(items)
    if len(review) != 10:
        raise RuntimeError("v409 expected 10 B018 display rows.")
    patch_plan = _patch_plan(review)
    blockers = _claim_blockers(len(items))
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v409_notebook_b018_display_review.csv", review)
    write_csv(TABLE_DIR / "paper4_v409_notebook_b018_patch_plan.csv", patch_plan)
    write_csv(TABLE_DIR / "paper4_v409_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v409_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v409_notebook_b018_display_review",
        "schema_version": "2026-05-17.409",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_b007_patch_version_v409": PRIOR_B007_PATCH_VERSION,
        "global_notebook_diagnostics_v409": int(len(items)),
        "b018_diagnostics_v409": int(counts.get("B018", 0)),
        "intentional_display_rows_v409": int(
            review["display_semantics_v409"].eq("intentional_plotly_display_then_export").sum()
        ),
        "patch_plan_rows_v409": int(len(patch_plan)),
        "notebooks_mutated_v409": False,
        "notebook_diff_clean_before_v409": True,
        "notebook_diff_clean_after_v409": _notebook_diff_clean(),
        "global_ruff_clean_v409": False,
        "full_repository_pytest_run_v409": False,
        "full_quarto_render_run_v409": False,
        "working_champion_claim_allowed_v409": False,
        "paper1_promotion_allowed_v409": False,
        "paper4_working_champion_changed_v409": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v409": NEXT_ARTIFACT,
        "claim_boundary": (
            "v409 reviews B018 display semantics and selects fig.show patch; no "
            "notebooks are mutated and final promotion remains blocked"
        ),
    }
    REVIEW_MD.write_text(_review_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v409": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
