#!/usr/bin/env python3
"""Build Paper 4 v424 targeted repository ruff repair batch artifacts."""

from __future__ import annotations

import json
import re
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

VERSION = 424
PRIOR_REPOSITORY_RUFF_FRONTIER_VERSION = 423
TARGET_FILES = [
    "streamlit_app/pages/model_interpretability.py",
    "streamlit_app/pages/data_story.py",
    "streamlit_app/pages/portfolio_optimizer.py",
    "streamlit_app/pages/causal_intelligence.py",
]
NOQA_LINE = "# ruff: noqa: E402 - Streamlit pages bootstrap repo root before app-local imports.\n"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
TARGETED_TEST_COMMAND = ["uv", "run", "pytest", "-q", "tests/test_streamlit/test_page_imports.py"]
NEXT_ARTIFACT = "paper4_v425_post_streamlit_ruff_repair_pytest_probe.md"
REPAIR_MD = NOTEBOOK.parent / "paper4_v424_targeted_repo_ruff_repair_batch.md"


def _run_repository_ruff_json() -> tuple[int, list[dict[str, Any]]]:
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
    return result.returncode, payload


def _run_targeted_tests() -> dict[str, Any]:
    started = datetime.now(UTC)
    result = subprocess.run(
        TARGETED_TEST_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    runtime = round((datetime.now(UTC) - started).total_seconds(), 3)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part)
    summary_line = ""
    for line in reversed(combined.splitlines()):
        if " in " in line and ("==" in line or " passed" in line):
            summary_line = line.strip()
            break
    collected_match = re.search(r"collected\s+(\d+)\s+items", combined)
    return {
        "command": " ".join(TARGETED_TEST_COMMAND),
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "runtime_seconds": runtime,
        "collected_items": int(collected_match.group(1)) if collected_match else 0,
        "summary_line": summary_line,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-40:]),
    }


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


def _items_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        path = _relative_path(str(item["filename"]))
        rows.append(
            {
                "file_path_v424": path,
                "surface_v424": _surface(path),
                "rule_code_v424": str(item["code"]),
                "message_v424": str(item["message"]),
                "row_v424": int((item.get("location") or {}).get("row") or 0),
                "fixable_v424": bool(item.get("fix") and item["fix"].get("edits")),
                "claim_boundary_v424": "post-repair repository ruff diagnostic inventory",
            }
        )
    return pd.DataFrame(rows)


def _target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *TARGET_FILES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _changed_target_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *TARGET_FILES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _insert_streamlit_e402_noqa() -> pd.DataFrame:
    rows = []
    for relative_path in TARGET_FILES:
        path = ROOT / relative_path
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        already_present = any(line.startswith("# ruff: noqa: E402") for line in lines[:8])
        if not already_present:
            future_idx = next(
                idx for idx, line in enumerate(lines) if line.startswith("from __future__ import annotations")
            )
            insert_idx = future_idx + 1
            if insert_idx < len(lines) and lines[insert_idx].strip() == "":
                insert_idx += 1
            lines.insert(insert_idx, NOQA_LINE)
            path.write_text("".join(lines), encoding="utf-8")
        rows.append(
            {
                "action_id_v424": f"streamlit_e402_bootstrap_noqa_{len(rows) + 1:02d}",
                "file_path_v424": relative_path,
                "rule_code_v424": "E402",
                "justification_v424": "Streamlit page bootstraps repository root before app-local imports",
                "mutation_applied_v424": not already_present,
                "claim_boundary_v424": "file-level E402 exception only; no UI behavior changed",
            }
        )
    return pd.DataFrame(rows)


def _rule_counts(items: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item["code"]) for item in items)


def _surface_counts(frame: pd.DataFrame) -> Counter[str]:
    if frame.empty:
        return Counter()
    return Counter(frame["surface_v424"])


def _delta_table(before_items: list[dict[str, Any]], after_items: list[dict[str, Any]]) -> pd.DataFrame:
    before_rules = _rule_counts(before_items)
    after_rules = _rule_counts(after_items)
    after_frame = _items_frame(after_items)
    before_frame = _items_frame(before_items)
    before_surfaces = _surface_counts(before_frame)
    after_surfaces = _surface_counts(after_frame)
    rows = [
        ("repository_total", len(before_items), len(after_items)),
        ("repository_e402", before_rules.get("E402", 0), after_rules.get("E402", 0)),
        (
            "streamlit_app_total",
            before_surfaces.get("streamlit_app", 0),
            after_surfaces.get("streamlit_app", 0),
        ),
        ("notebook_total", before_surfaces.get("notebook", 0), after_surfaces.get("notebook", 0)),
        ("scripts_total", before_surfaces.get("scripts", 0), after_surfaces.get("scripts", 0)),
        ("book_total", before_surfaces.get("book", 0), after_surfaces.get("book", 0)),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v424": metric,
                "before_v424": int(before),
                "after_v424": int(after),
                "delta_v424": int(after - before),
                "claim_boundary_v424": "ruff-count delta only; repository ruff still open unless after=0",
            }
            for metric, before, after in rows
        ]
    )


def _rule_frontier_after(after_frame: pd.DataFrame) -> pd.DataFrame:
    if after_frame.empty:
        return pd.DataFrame(
            [
                {
                    "rule_code_v424": "__none__",
                    "diagnostic_count_v424": 0,
                    "fixable_count_v424": 0,
                    "file_count_v424": 0,
                    "top_surface_v424": "__none__",
                    "repair_priority_v424": 1,
                    "claim_boundary_v424": "repository ruff clean after v424",
                }
            ]
        )
    rows = []
    for code, group in after_frame.groupby("rule_code_v424", sort=False):
        surface_counts = Counter(group["surface_v424"])
        rows.append(
            {
                "rule_code_v424": code,
                "diagnostic_count_v424": int(len(group)),
                "fixable_count_v424": int(group["fixable_v424"].astype(bool).sum()),
                "file_count_v424": int(group["file_path_v424"].nunique()),
                "top_surface_v424": surface_counts.most_common(1)[0][0],
                "repair_priority_v424": 0,
                "claim_boundary_v424": "post-v424 frontier only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v424", "fixable_count_v424", "rule_code_v424"],
        ascending=[False, False, True],
    )
    out["repair_priority_v424"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _surface_summary_after(after_frame: pd.DataFrame) -> pd.DataFrame:
    if after_frame.empty:
        return pd.DataFrame(
            [
                {
                    "surface_v424": "__none__",
                    "diagnostic_count_v424": 0,
                    "fixable_count_v424": 0,
                    "file_count_v424": 0,
                    "mutation_allowed_v424": False,
                    "claim_boundary_v424": "repository ruff clean after v424",
                }
            ]
        )
    rows = []
    for surface, group in after_frame.groupby("surface_v424", sort=False):
        rows.append(
            {
                "surface_v424": surface,
                "diagnostic_count_v424": int(len(group)),
                "fixable_count_v424": int(group["fixable_v424"].astype(bool).sum()),
                "file_count_v424": int(group["file_path_v424"].nunique()),
                "mutation_allowed_v424": False,
                "claim_boundary_v424": "post-v424 surface summary",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v424", "fixable_count_v424", "surface_v424"],
        ascending=[False, False, True],
    )


def _test_summary_table(test_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test_id_v424": "streamlit_page_imports",
                "command_v424": test_result["command"],
                "exit_code_v424": int(test_result["exit_code"]),
                "passed_v424": bool(test_result["passed"]),
                "runtime_seconds_v424": float(test_result["runtime_seconds"]),
                "collected_items_v424": int(test_result["collected_items"]),
                "summary_line_v424": str(test_result["summary_line"]),
                "claim_boundary_v424": "targeted import smoke test only; full pytest deferred",
            }
        ]
    )


def _claim_blockers(*, after_total: int, targeted_passed: bool) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v424": "repository_ruff_frontier_still_open",
            "blocking_v424": after_total > 0,
            "evidence_count_v424": after_total,
            "required_next_artifact_v424": NEXT_ARTIFACT,
            "claim_boundary_v424": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v424": "full_repository_pytest_deferred_after_repair",
            "blocking_v424": True,
            "evidence_count_v424": 1,
            "required_next_artifact_v424": NEXT_ARTIFACT,
            "claim_boundary_v424": "targeted import tests do not replace full pytest",
        },
        {
            "blocker_id_v424": "paper4_final_promotion_forbidden",
            "blocking_v424": True,
            "evidence_count_v424": 1,
            "required_next_artifact_v424": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v424": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not targeted_passed:
        rows.insert(
            0,
            {
                "blocker_id_v424": "targeted_streamlit_page_import_tests_failed",
                "blocking_v424": True,
                "evidence_count_v424": 1,
                "required_next_artifact_v424": "paper4_v425_streamlit_repair_failure_triage.md",
                "claim_boundary_v424": "targeted test failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, after_total: int, notebook_after: int, targeted_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v424_streamlit_e402_bootstrap_exception_applied",
                "allowed": True,
                "artifact": "paper4_v424_streamlit_e402_actions.csv",
                "boundary": "four Streamlit page bootstrap exceptions",
            },
            {
                "claim_id": "v424_repository_ruff_reduced",
                "allowed": True,
                "artifact": "paper4_v424_repository_ruff_delta.csv",
                "boundary": "repository ruff diagnostic count decreases",
            },
            {
                "claim_id": "v424_notebook_lint_remains_clean",
                "allowed": notebook_after == 0,
                "artifact": "paper4_v424_repository_ruff_delta.csv",
                "boundary": "notebook diagnostics remain zero",
            },
            {
                "claim_id": "v424_targeted_streamlit_page_import_tests_passed",
                "allowed": targeted_passed,
                "artifact": "paper4_v424_streamlit_page_import_test_summary.csv",
                "boundary": "targeted smoke only",
            },
            {
                "claim_id": "v424_repository_ruff_clean",
                "allowed": after_total == 0,
                "artifact": "paper4_v424_claim_blockers.csv",
                "boundary": "true only if repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v424_full_repository_pytest_passed_after_repair",
                "allowed": False,
                "artifact": "paper4_v424_claim_blockers.csv",
                "boundary": "full pytest deferred to v425",
            },
            {
                "claim_id": "v424_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, after_total: int, notebook_after: int, targeted_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v424 reduces repository ruff diagnostics with a targeted Streamlit E402 batch.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v424_repository_ruff_delta.csv",
                "boundary": "Reduction only; repository ruff still open unless after count is zero.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v424 keeps notebook lint clean after Streamlit repair.",
                "allowed": notebook_after == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v424_repository_ruff_delta.csv",
                "boundary": "Notebook diagnostics remain zero.",
                "prohibited_claim_flag": notebook_after != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v424 passes targeted Streamlit page-import tests after repair.",
                "allowed": targeted_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v424_streamlit_page_import_test_summary.csv",
                "boundary": "Targeted smoke only; full pytest deferred.",
                "prohibited_claim_flag": not targeted_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v424 proves repository ruff clean or full pytest clean after repair.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v424_claim_blockers.csv",
                "boundary": "Repository ruff remains open and full pytest is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v424 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v424_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(after_total: int) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v424 applies a targeted Streamlit E402 bootstrap exception batch.",
                "status": "targeted_streamlit_e402_repair_batch_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v425 full repository pytest passes after the Streamlit ruff repair",
                "last_wave": "v424",
                "execution_result": f"repo_ruff_reduced_107_to_{after_total}_streamlit_e402_cleared",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v424")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _repair_markdown(status: dict[str, Any], test_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Targeted Repo Ruff Repair Batch v424

Generated: {status["generated_at_utc"]}

v424 applies a targeted Streamlit E402 bootstrap exception batch.

## Result

- Repository diagnostics: `{status["repo_ruff_total_before_v424"]}` ->
  `{status["repo_ruff_total_after_v424"]}`.
- Repository E402 diagnostics: `{status["repo_ruff_e402_before_v424"]}` ->
  `{status["repo_ruff_e402_after_v424"]}`.
- Notebook diagnostics after: `{status["notebook_diagnostics_after_v424"]}`.
- Changed Streamlit files: `{status["changed_streamlit_files_v424"]}`.
- Targeted page-import tests passed: `{status["targeted_streamlit_page_import_tests_passed_v424"]}`.
- Targeted test summary: `{test_result["summary_line"]}`.

## Required Caveat

v424 does not claim repository ruff clean, full pytest clean, Quarto render, or
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v424"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V424_TARGETED_REPO_RUFF_REPAIR_BATCH_START -->"
    end = "<!-- V424_TARGETED_REPO_RUFF_REPAIR_BATCH_END -->"
    block = f"""
{start}

## Wave v424: Targeted Repo Ruff Repair Batch

Generated: {status["generated_at_utc"]}

### Objective

v424 applies a targeted Streamlit E402 bootstrap exception batch selected by
v423.

### Results

- Repository ruff diagnostics before/after:
  `{status["repo_ruff_total_before_v424"]}` ->
  `{status["repo_ruff_total_after_v424"]}`.
- Repository E402 before/after:
  `{status["repo_ruff_e402_before_v424"]}` ->
  `{status["repo_ruff_e402_after_v424"]}`.
- Streamlit diagnostics before/after:
  `{status["streamlit_diagnostics_before_v424"]}` ->
  `{status["streamlit_diagnostics_after_v424"]}`.
- Notebook diagnostics after:
  `{status["notebook_diagnostics_after_v424"]}`.
- Changed Streamlit files:
  `{status["changed_streamlit_files_v424"]}`.
- Targeted Streamlit page-import tests passed:
  `{status["targeted_streamlit_page_import_tests_passed_v424"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v424"]}`.
- Full repository pytest run:
  `{status["full_repository_pytest_run_v424"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v424"]}`.

### Interpretation

The justified Streamlit bootstrap E402 frontier is closed. Repository ruff now
contracts to scripts/book plus the remaining non-E402 Streamlit hotspot in
`model_interpretability.py`.

### Claim Impact

- Allowed: targeted Streamlit E402 repair applied, repo ruff count reduced, and
  targeted page-import tests passed.
- Still prohibited: repository ruff clean, full pytest clean after repair,
  Quarto render clean, champion replacement and final promotion claims.

### Quarto Promotion Decision

Keep v424 in the living notebook. v425 should run the post-repair full pytest
probe before further lint repair.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _target_diff_clean():
        raise RuntimeError("v424 expects clean target Streamlit file diffs before mutation.")

    v423_status = json.loads((STATUS_DIR / "paper4_v423_status.json").read_text(encoding="utf-8"))
    if v423_status["next_artifact_v423"] != "paper4_v424_targeted_repo_ruff_repair_batch.md":
        raise RuntimeError("v424 expects v423 to route to targeted repo ruff repair.")
    if int(v423_status["ruff_total_diagnostics_v423"]) != 107:
        raise RuntimeError("v424 expects the v423 107-diagnostic frontier.")

    before_exit, before_items = _run_repository_ruff_json()
    if before_exit != 1 or len(before_items) != 107:
        raise RuntimeError("v424 expected repository ruff to fail with 107 diagnostics before repair.")
    before_counts = _rule_counts(before_items)
    before_frame = _items_frame(before_items)
    before_surfaces = _surface_counts(before_frame)

    actions = _insert_streamlit_e402_noqa()
    changed_files = _changed_target_files()
    after_exit, after_items = _run_repository_ruff_json()
    after_counts = _rule_counts(after_items)
    after_frame = _items_frame(after_items)
    after_surfaces = _surface_counts(after_frame)
    delta = _delta_table(before_items, after_items)
    after_rule_frontier = _rule_frontier_after(after_frame)
    after_surface_summary = _surface_summary_after(after_frame)
    test_result = _run_targeted_tests()
    test_summary = _test_summary_table(test_result)

    after_total = int(len(after_items))
    notebook_after = int(after_surfaces.get("notebook", 0))
    targeted_passed = bool(test_result["passed"])
    blockers = _claim_blockers(after_total=after_total, targeted_passed=targeted_passed)
    claim_matrix = _claim_matrix(
        after_total=after_total,
        notebook_after=notebook_after,
        targeted_passed=targeted_passed,
    )
    _update_claim_boundaries(
        after_total=after_total,
        notebook_after=notebook_after,
        targeted_passed=targeted_passed,
    )
    _update_backlog(after_total)

    write_csv(TABLE_DIR / "paper4_v424_streamlit_e402_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v424_repository_ruff_delta.csv", delta)
    write_csv(TABLE_DIR / "paper4_v424_repository_ruff_after_diagnostics.csv", after_frame)
    write_csv(TABLE_DIR / "paper4_v424_repository_ruff_after_rule_frontier.csv", after_rule_frontier)
    write_csv(TABLE_DIR / "paper4_v424_repository_ruff_after_surface_summary.csv", after_surface_summary)
    write_csv(TABLE_DIR / "paper4_v424_streamlit_page_import_test_summary.csv", test_summary)
    write_csv(TABLE_DIR / "paper4_v424_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v424_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v424_targeted_repo_ruff_repair_batch",
        "schema_version": "2026-05-17.424",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_repository_ruff_frontier_version_v424": PRIOR_REPOSITORY_RUFF_FRONTIER_VERSION,
        "actions_v424": int(len(actions)),
        "repo_ruff_exit_code_before_v424": int(before_exit),
        "repo_ruff_exit_code_after_v424": int(after_exit),
        "repo_ruff_total_before_v424": int(len(before_items)),
        "repo_ruff_total_after_v424": after_total,
        "repo_ruff_total_reduced_v424": int(len(before_items) - after_total),
        "repo_ruff_e402_before_v424": int(before_counts.get("E402", 0)),
        "repo_ruff_e402_after_v424": int(after_counts.get("E402", 0)),
        "repo_ruff_e402_reduced_v424": int(before_counts.get("E402", 0) - after_counts.get("E402", 0)),
        "streamlit_diagnostics_before_v424": int(before_surfaces.get("streamlit_app", 0)),
        "streamlit_diagnostics_after_v424": int(after_surfaces.get("streamlit_app", 0)),
        "notebook_diagnostics_after_v424": notebook_after,
        "changed_streamlit_files_v424": int(len(changed_files)),
        "changed_streamlit_file_list_v424": changed_files,
        "targeted_streamlit_page_import_test_command_v424": test_result["command"],
        "targeted_streamlit_page_import_tests_exit_code_v424": int(test_result["exit_code"]),
        "targeted_streamlit_page_import_tests_passed_v424": targeted_passed,
        "targeted_streamlit_page_import_tests_summary_v424": str(test_result["summary_line"]),
        "repository_ruff_clean_v424": after_total == 0,
        "notebook_ruff_clean_v424": notebook_after == 0,
        "full_repository_pytest_run_v424": False,
        "full_quarto_render_run_v424": False,
        "working_champion_claim_allowed_v424": False,
        "paper1_promotion_allowed_v424": False,
        "paper4_working_champion_changed_v424": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v424": NEXT_ARTIFACT,
        "claim_boundary": (
            "v424 applies a targeted Streamlit E402 bootstrap exception repair; "
            "repo ruff remains open and full pytest/final promotion claims stay blocked"
        ),
    }
    if status["repo_ruff_e402_after_v424"] != 0:
        raise RuntimeError("v424 expected E402 to be cleared.")
    if status["repo_ruff_total_after_v424"] != 57:
        raise RuntimeError("v424 expected repository ruff to contract to 57 diagnostics.")
    if notebook_after != 0:
        raise RuntimeError("v424 expected notebook diagnostics to remain zero.")
    if sorted(changed_files) != sorted(TARGET_FILES):
        raise RuntimeError("v424 changed an unexpected Streamlit file set.")
    if not targeted_passed:
        raise RuntimeError("v424 targeted Streamlit page import tests failed.")

    REPAIR_MD.write_text(_repair_markdown(status, test_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v424": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
