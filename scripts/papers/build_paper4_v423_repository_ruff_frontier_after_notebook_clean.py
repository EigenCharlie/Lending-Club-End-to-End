#!/usr/bin/env python3
"""Build Paper 4 v423 repository ruff frontier after notebook lint is clean."""

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

VERSION = 423
PRIOR_POST_PATCH_PYTEST_PROBE_VERSION = 422
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
NEXT_FAIL_ARTIFACT = "paper4_v424_targeted_repo_ruff_repair_batch.md"
NEXT_PASS_ARTIFACT = "paper4_v424_quarto_render_probe_after_ruff_clean.md"
FRONTIER_MD = NOTEBOOK.parent / "paper4_v423_repository_ruff_frontier_after_notebook_clean.md"


def _run_repository_ruff_json() -> tuple[int, str, str, list[dict[str, Any]]]:
    result = subprocess.run(
        RUFF_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "repository ruff probe failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("repository ruff JSON output is not a list")
    return result.returncode, result.stdout, result.stderr, payload


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def _surface(path: str) -> str:
    if path.startswith("notebooks/"):
        return "notebook"
    if path == "tests/test_docs/test_paper4_living_lab_guardrails.py":
        return "paper4_guardrail_test"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("streamlit_app/"):
        return "streamlit_app"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith("src/"):
        return "src"
    if path.startswith("book/"):
        return "book"
    return "other"


def _fixable(item: dict[str, Any]) -> bool:
    fix = item.get("fix")
    return bool(fix and fix.get("edits"))


def _normalized_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        path = _relative_path(str(item["filename"]))
        rows.append(
            {
                "file_path_v423": path,
                "surface_v423": _surface(path),
                "rule_code_v423": str(item["code"]),
                "message_v423": str(item["message"]),
                "row_v423": int((item.get("location") or {}).get("row") or 0),
                "column_v423": int((item.get("location") or {}).get("column") or 0),
                "fixable_v423": _fixable(item),
                "claim_boundary_v423": "repository ruff diagnostic inventory only",
            }
        )
    return pd.DataFrame(rows)


def _rule_frontier(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "rule_code_v423",
                "diagnostic_count_v423",
                "fixable_count_v423",
                "file_count_v423",
                "top_surface_v423",
                "sample_message_v423",
                "repair_priority_v423",
                "claim_boundary_v423",
            ]
        )
    rows = []
    grouped = diagnostics.groupby("rule_code_v423", sort=False)
    for code, group in grouped:
        surface_counts = Counter(group["surface_v423"])
        rows.append(
            {
                "rule_code_v423": code,
                "diagnostic_count_v423": int(len(group)),
                "fixable_count_v423": int(group["fixable_v423"].astype(bool).sum()),
                "file_count_v423": int(group["file_path_v423"].nunique()),
                "top_surface_v423": surface_counts.most_common(1)[0][0],
                "sample_message_v423": str(group.iloc[0]["message_v423"]),
                "repair_priority_v423": 0,
                "claim_boundary_v423": "classification only; no repair applied in v423",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v423", "fixable_count_v423", "rule_code_v423"],
        ascending=[False, False, True],
    )
    out["repair_priority_v423"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _hotspot_files(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "file_path_v423",
                "surface_v423",
                "diagnostic_count_v423",
                "fixable_count_v423",
                "rule_codes_v423",
                "repair_priority_v423",
                "claim_boundary_v423",
            ]
        )
    rows = []
    for path, group in diagnostics.groupby("file_path_v423", sort=False):
        rows.append(
            {
                "file_path_v423": path,
                "surface_v423": str(group.iloc[0]["surface_v423"]),
                "diagnostic_count_v423": int(len(group)),
                "fixable_count_v423": int(group["fixable_v423"].astype(bool).sum()),
                "rule_codes_v423": ",".join(sorted(set(group["rule_code_v423"]))),
                "repair_priority_v423": 0,
                "claim_boundary_v423": "hotspot classification only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v423", "fixable_count_v423", "file_path_v423"],
        ascending=[False, False, True],
    )
    out["repair_priority_v423"] = range(1, len(out) + 1)
    return out.head(25).reset_index(drop=True)


def _surface_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(
            [
                {
                    "surface_v423": "__none__",
                    "diagnostic_count_v423": 0,
                    "fixable_count_v423": 0,
                    "file_count_v423": 0,
                    "rule_count_v423": 0,
                    "recommended_next_action_v423": "route to Quarto render probe",
                    "mutation_allowed_v423": False,
                    "claim_boundary_v423": "repository ruff clean; no mutation in v423",
                }
            ]
        )
    rows = []
    for surface, group in diagnostics.groupby("surface_v423", sort=False):
        rows.append(
            {
                "surface_v423": surface,
                "diagnostic_count_v423": int(len(group)),
                "fixable_count_v423": int(group["fixable_v423"].astype(bool).sum()),
                "file_count_v423": int(group["file_path_v423"].nunique()),
                "rule_count_v423": int(group["rule_code_v423"].nunique()),
                "recommended_next_action_v423": "targeted repair batch if selected for v424",
                "mutation_allowed_v423": False,
                "claim_boundary_v423": "surface summary only; no repair applied",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v423", "fixable_count_v423", "surface_v423"],
        ascending=[False, False, True],
    )


def _repair_plan(rule_frontier: pd.DataFrame, hotspots: pd.DataFrame, total: int) -> pd.DataFrame:
    if total == 0:
        return pd.DataFrame(
            [
                {
                    "repair_lane_v423": "quarto_render_probe",
                    "priority_v423": 1,
                    "target_surface_v423": "__none__",
                    "target_rules_v423": "__none__",
                    "target_files_v423": "__none__",
                    "estimated_diagnostic_count_v423": 0,
                    "recommended_next_artifact_v423": NEXT_PASS_ARTIFACT,
                    "mutation_allowed_v423": False,
                    "claim_boundary_v423": "repo ruff already clean; route to render probe",
                }
            ]
        )
    top_rule = rule_frontier.iloc[0]
    top_hotspot = hotspots.iloc[0]
    return pd.DataFrame(
        [
            {
                "repair_lane_v423": "top_rule_targeted_repair",
                "priority_v423": 1,
                "target_surface_v423": str(top_rule["top_surface_v423"]),
                "target_rules_v423": str(top_rule["rule_code_v423"]),
                "target_files_v423": "top hotspots only",
                "estimated_diagnostic_count_v423": int(top_rule["diagnostic_count_v423"]),
                "recommended_next_artifact_v423": NEXT_FAIL_ARTIFACT,
                "mutation_allowed_v423": False,
                "claim_boundary_v423": "selected for v424, but no mutation in v423",
            },
            {
                "repair_lane_v423": "top_hotspot_file_repair",
                "priority_v423": 2,
                "target_surface_v423": str(top_hotspot["surface_v423"]),
                "target_rules_v423": str(top_hotspot["rule_codes_v423"]),
                "target_files_v423": str(top_hotspot["file_path_v423"]),
                "estimated_diagnostic_count_v423": int(top_hotspot["diagnostic_count_v423"]),
                "recommended_next_artifact_v423": NEXT_FAIL_ARTIFACT,
                "mutation_allowed_v423": False,
                "claim_boundary_v423": "candidate v424 batch; no mutation in v423",
            },
        ]
    )


def _claim_blockers(*, total: int, notebook_count: int, next_artifact: str) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v423": "repository_ruff_frontier_open",
            "blocking_v423": total > 0,
            "evidence_count_v423": total,
            "required_next_artifact_v423": next_artifact,
            "claim_boundary_v423": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v423": "quarto_render_not_run",
            "blocking_v423": True,
            "evidence_count_v423": 1,
            "required_next_artifact_v423": next_artifact,
            "claim_boundary_v423": "Quarto render is not implied by ruff or pytest probes",
        },
        {
            "blocker_id_v423": "paper4_final_promotion_forbidden",
            "blocking_v423": True,
            "evidence_count_v423": 1,
            "required_next_artifact_v423": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v423": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if notebook_count:
        rows.insert(
            1,
            {
                "blocker_id_v423": "notebook_ruff_regression_detected",
                "blocking_v423": True,
                "evidence_count_v423": notebook_count,
                "required_next_artifact_v423": "paper4_v424_notebook_ruff_regression_triage.md",
                "claim_boundary_v423": "notebook lint clean claim would need repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, total: int, notebook_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v423_repository_ruff_probe_run",
                "allowed": True,
                "artifact": "paper4_v423_repository_ruff_rule_frontier.csv",
                "boundary": "repository ruff command executed and inventoried",
            },
            {
                "claim_id": "v423_notebook_lint_remains_clean",
                "allowed": notebook_count == 0,
                "artifact": "paper4_v423_repository_ruff_surface_summary.csv",
                "boundary": "true only if notebook diagnostics remain at zero",
            },
            {
                "claim_id": "v423_full_repository_pytest_clean_inherited",
                "allowed": True,
                "artifact": "paper4_v422_pytest_probe_summary.csv",
                "boundary": "inherits v422 full pytest success",
            },
            {
                "claim_id": "v423_repository_ruff_clean",
                "allowed": total == 0,
                "artifact": "paper4_v423_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v423_lint_repair_applied",
                "allowed": False,
                "artifact": "paper4_v423_repository_ruff_repair_plan.csv",
                "boundary": "v423 is non-mutating",
            },
            {
                "claim_id": "v423_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, total: int, notebook_count: int) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v423 runs repository-wide ruff after notebook lint is clean.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v423_repository_ruff_rule_frontier.csv",
                "boundary": "Execution and classification only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v423 keeps notebook lint clean in the repository ruff frontier.",
                "allowed": notebook_count == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v423_repository_ruff_surface_summary.csv",
                "boundary": "Notebook diagnostics remain zero.",
                "prohibited_claim_flag": notebook_count != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v423 proves repository ruff is clean.",
                "allowed": total == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v423_claim_blockers.csv",
                "boundary": "Allowed only if total diagnostics are zero.",
                "prohibited_claim_flag": total != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v423 repairs repository ruff diagnostics.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v423_repository_ruff_repair_plan.csv",
                "boundary": "No files are mutated by v423.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v423 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v423_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(*, total: int, next_artifact: str) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v423 classifies repository-wide ruff after notebook lint is clean.",
                "status": (
                    "repository_ruff_frontier_after_notebook_clean_classified"
                    if total
                    else "repository_ruff_clean_after_notebook_cleanup"
                ),
                "next_artifact": next_artifact,
                "success_condition": (
                    "v424 applies a targeted repository ruff repair batch"
                    if total
                    else "v424 probes Quarto render after ruff clean"
                ),
                "last_wave": "v423",
                "execution_result": (
                    f"repo_ruff_failed_{total}_diagnostics_notebooks_clean_no_mutation"
                    if total
                    else "repo_ruff_clean_after_notebook_lint_cleanup_no_mutation"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v423")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _frontier_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Repository Ruff Frontier After Notebook Clean v423

Generated: {status["generated_at_utc"]}

v423 runs repository-wide ruff after v421/v422 cleared and validated notebook
lint.

## Result

- Ruff command: `{status["ruff_command_v423"]}`.
- Ruff exit code: `{status["ruff_exit_code_v423"]}`.
- Total diagnostics: `{status["ruff_total_diagnostics_v423"]}`.
- Fixable diagnostics: `{status["ruff_fixable_diagnostics_v423"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v423"]}`.
- Top rule: `{status["top_rule_v423"]}` with `{status["top_rule_count_v423"]}` diagnostics.
- Top file: `{status["top_file_v423"]}` with `{status["top_file_diagnostics_v423"]}` diagnostics.
- Top surface: `{status["top_surface_v423"]}`.

## Required Caveat

v423 is non-mutating. It does not repair repository ruff diagnostics, run Quarto
render, or create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v423"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V423_REPOSITORY_RUFF_FRONTIER_AFTER_NOTEBOOK_CLEAN_START -->"
    end = "<!-- V423_REPOSITORY_RUFF_FRONTIER_AFTER_NOTEBOOK_CLEAN_END -->"
    block = f"""
{start}

## Wave v423: Repository Ruff Frontier After Notebook Clean

Generated: {status["generated_at_utc"]}

### Objective

v423 runs `uv run ruff check . --output-format json` after notebook lint is
clean and full repository pytest has passed.

### Results

- Ruff exit code:
  `{status["ruff_exit_code_v423"]}`.
- Total diagnostics:
  `{status["ruff_total_diagnostics_v423"]}`.
- Fixable diagnostics:
  `{status["ruff_fixable_diagnostics_v423"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v423"]}`.
- Top rule:
  `{status["top_rule_v423"]}`.
- Top file:
  `{status["top_file_v423"]}`.
- Top surface:
  `{status["top_surface_v423"]}`.
- Full pytest clean inherited from v422:
  `{status["full_repository_pytest_clean_v423"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v423"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v423"]}`.

### Interpretation

The notebook-lint frontier remains closed, but repository-wide ruff is still an
open validation frontier. v424 should repair a small targeted batch from the
current rule/file frontier rather than re-opening notebook bulk rewrites.

### Claim Impact

- Allowed: repository ruff frontier classified and notebook lint remains clean.
- Still prohibited: repository ruff clean if diagnostics remain, Quarto render
  clean, lint repair, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v423 in the living notebook. v424 should execute the first targeted
repository ruff repair batch if diagnostics remain.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v422_status = json.loads((STATUS_DIR / "paper4_v422_status.json").read_text(encoding="utf-8"))
    if v422_status["next_artifact_v422"] != "paper4_v423_repository_ruff_frontier_after_notebook_clean.md":
        raise RuntimeError("v423 expects v422 to route to repository ruff frontier.")
    if v422_status["notebook_ruff_clean_v422"] is not True:
        raise RuntimeError("v423 expects v422 notebook ruff to be clean.")
    if v422_status["full_repository_pytest_passed_v422"] is not True:
        raise RuntimeError("v423 expects v422 full pytest to have passed.")

    exit_code, stdout, stderr, items = _run_repository_ruff_json()
    diagnostics = _normalized_items(items)
    total = int(len(diagnostics))
    notebook_count = int((diagnostics["surface_v423"].eq("notebook")).sum()) if total else 0
    fixable_count = int(diagnostics["fixable_v423"].astype(bool).sum()) if total else 0
    rule_frontier = _rule_frontier(diagnostics)
    hotspots = _hotspot_files(diagnostics)
    surface_summary = _surface_summary(diagnostics)
    next_artifact = NEXT_FAIL_ARTIFACT if total else NEXT_PASS_ARTIFACT
    repair_plan = _repair_plan(rule_frontier, hotspots, total)
    blockers = _claim_blockers(total=total, notebook_count=notebook_count, next_artifact=next_artifact)
    claim_matrix = _claim_matrix(total=total, notebook_count=notebook_count)
    _update_claim_boundaries(total=total, notebook_count=notebook_count)
    _update_backlog(total=total, next_artifact=next_artifact)

    write_csv(TABLE_DIR / "paper4_v423_repository_ruff_diagnostics.csv", diagnostics)
    write_csv(TABLE_DIR / "paper4_v423_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v423_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v423_repository_ruff_surface_summary.csv", surface_summary)
    write_csv(TABLE_DIR / "paper4_v423_repository_ruff_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v423_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v423_claim_matrix_delta.csv", claim_matrix)

    top_rule = rule_frontier.iloc[0] if total else {}
    top_file = hotspots.iloc[0] if total else {}
    top_surface = surface_summary.iloc[0] if total else {}
    status = {
        "phase": "v423_repository_ruff_frontier_after_notebook_clean",
        "schema_version": "2026-05-17.423",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_patch_pytest_probe_version_v423": PRIOR_POST_PATCH_PYTEST_PROBE_VERSION,
        "ruff_command_v423": " ".join(RUFF_COMMAND),
        "ruff_exit_code_v423": int(exit_code),
        "ruff_stderr_tail_v423": "\n".join(stderr.splitlines()[-20:]),
        "ruff_total_diagnostics_v423": total,
        "ruff_fixable_diagnostics_v423": fixable_count,
        "notebook_diagnostics_v423": notebook_count,
        "notebook_ruff_clean_v423": notebook_count == 0,
        "rule_frontier_rows_v423": int(len(rule_frontier)),
        "hotspot_file_rows_v423": int(len(hotspots)),
        "surface_summary_rows_v423": int(len(surface_summary)),
        "repair_plan_rows_v423": int(len(repair_plan)),
        "top_rule_v423": str(top_rule.get("rule_code_v423", "__none__")),
        "top_rule_count_v423": int(top_rule.get("diagnostic_count_v423", 0)),
        "top_file_v423": str(top_file.get("file_path_v423", "__none__")),
        "top_file_diagnostics_v423": int(top_file.get("diagnostic_count_v423", 0)),
        "top_surface_v423": str(top_surface.get("surface_v423", "__none__")),
        "repository_ruff_clean_v423": total == 0,
        "full_repository_pytest_clean_v423": bool(v422_status["full_repository_pytest_passed_v422"]),
        "full_quarto_render_run_v423": False,
        "working_champion_claim_allowed_v423": False,
        "paper1_promotion_allowed_v423": False,
        "paper4_working_champion_changed_v423": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v423": next_artifact,
        "claim_boundary": (
            "v423 dynamically classifies repository ruff after notebook lint is clean; "
            "no repair, Quarto render or final promotion is performed"
        ),
    }
    if notebook_count != 0:
        raise RuntimeError("v423 expected notebook diagnostics to remain zero.")
    if total and exit_code == 0:
        raise RuntimeError("v423 saw diagnostics but ruff exit code was zero.")
    if not total and exit_code != 0:
        raise RuntimeError("v423 saw no diagnostics but ruff exit code was nonzero.")
    if stdout.strip() and total == 0 and json.loads(stdout) != []:
        raise RuntimeError("v423 expected empty JSON output for clean ruff.")

    FRONTIER_MD.write_text(_frontier_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v423": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
