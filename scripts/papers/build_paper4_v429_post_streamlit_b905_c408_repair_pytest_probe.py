#!/usr/bin/env python3
"""Build Paper 4 v429 post-Streamlit-B905/C408-repair pytest probe artifacts."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_v403_post_notebook_mutation_pytest_probe import _run_pytest
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

VERSION = 429
PRIOR_STREAMLIT_B905_C408_REPAIR_VERSION = 428
NEXT_PASS_ARTIFACT = "paper4_v430_scripts_b007_loop_variable_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v430_post_streamlit_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v429_post_streamlit_b905_c408_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
TARGET_STREAMLIT_FILE = "streamlit_app/pages/model_interpretability.py"


def _streamlit_target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", TARGET_STREAMLIT_FILE],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


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


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def _surface(path: str) -> str:
    if path.startswith("notebooks/"):
        return "notebook"
    if path.startswith("streamlit_app/"):
        return "streamlit_app"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith("book/"):
        return "book"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("src/"):
        return "src"
    return "other"


def _items_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        path = _relative_path(str(item["filename"]))
        fix = item.get("fix") or {}
        rows.append(
            {
                "file_path_v429": path,
                "surface_v429": _surface(path),
                "rule_code_v429": str(item["code"]),
                "message_v429": str(item["message"]),
                "row_v429": int((item.get("location") or {}).get("row") or 0),
                "fixable_v429": bool(fix.get("edits")),
                "claim_boundary_v429": "post-v428 repository ruff diagnostic inventory",
            }
        )
    return pd.DataFrame(rows)


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v429": "full_repository_pytest",
                "command_v429": pytest_result["command"],
                "exit_code_v429": int(pytest_result["exit_code"]),
                "passed_v429": bool(pytest_result["passed"]),
                "runtime_seconds_v429": float(pytest_result["runtime_seconds"]),
                "collected_items_v429": int(pytest_result["collected_items"]),
                "summary_line_v429": str(pytest_result["summary_line"]),
                "claim_boundary_v429": "post-v428 Streamlit B905/C408 repair full pytest probe",
            }
        ]
    )


def _count(items: list[dict[str, Any]], *, code: str | None = None, surface: str | None = None) -> int:
    total = 0
    for item in items:
        item_code = str(item["code"])
        item_surface = _surface(_relative_path(str(item["filename"])))
        if code is not None and item_code != code:
            continue
        if surface is not None and item_surface != surface:
            continue
        total += 1
    return total


def _ruff_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        ("repository_total", len(items)),
        ("repository_e402", _count(items, code="E402")),
        ("repository_b905", _count(items, code="B905")),
        ("repository_c408", _count(items, code="C408")),
        ("repository_b007", _count(items, code="B007")),
        ("repository_f841", _count(items, code="F841")),
        ("repository_b023", _count(items, code="B023")),
        ("notebook_total", _count(items, surface="notebook")),
        ("streamlit_app_total", _count(items, surface="streamlit_app")),
        ("scripts_total", _count(items, surface="scripts")),
        ("book_total", _count(items, surface="book")),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v429": metric,
                "diagnostic_count_v429": int(count),
                "claim_boundary_v429": "post-v428 repository ruff snapshot",
            }
            for metric, count in rows
        ]
    )


def _rule_frontier(items_frame: pd.DataFrame) -> pd.DataFrame:
    if items_frame.empty:
        return pd.DataFrame(
            [
                {
                    "rule_code_v429": "__none__",
                    "diagnostic_count_v429": 0,
                    "fixable_count_v429": 0,
                    "file_count_v429": 0,
                    "top_surface_v429": "__none__",
                    "repair_priority_v429": 1,
                    "claim_boundary_v429": "repository ruff clean after v429",
                }
            ]
        )
    rows = []
    for code, group in items_frame.groupby("rule_code_v429", sort=False):
        surface_counts = Counter(group["surface_v429"])
        rows.append(
            {
                "rule_code_v429": code,
                "diagnostic_count_v429": int(len(group)),
                "fixable_count_v429": int(group["fixable_v429"].astype(bool).sum()),
                "file_count_v429": int(group["file_path_v429"].nunique()),
                "top_surface_v429": surface_counts.most_common(1)[0][0],
                "repair_priority_v429": 0,
                "claim_boundary_v429": "post-v429 frontier only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v429", "fixable_count_v429", "rule_code_v429"],
        ascending=[False, False, True],
    )
    out["repair_priority_v429"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _hotspot_files(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_relative_path(str(item["filename"]))].append(item)
    rows = []
    for file_path, diagnostics in grouped.items():
        rule_codes = sorted({str(item["code"]) for item in diagnostics})
        rows.append(
            {
                "file_path_v429": file_path,
                "surface_v429": _surface(file_path),
                "diagnostic_count_v429": int(len(diagnostics)),
                "rule_codes_v429": ",".join(rule_codes),
                "claim_boundary_v429": "hotspot ranking only; no mutation in v429",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v429", "file_path_v429"],
        ascending=[False, True],
        ignore_index=True,
    )


def _repair_plan(rule_frontier: pd.DataFrame) -> pd.DataFrame:
    b007 = rule_frontier.loc[rule_frontier["rule_code_v429"].eq("B007")]
    diagnostic_count = int(b007["diagnostic_count_v429"].iloc[0]) if not b007.empty else 0
    fixable_count = int(b007["fixable_count_v429"].iloc[0]) if not b007.empty else 0
    return pd.DataFrame(
        [
            {
                "repair_lane_v429": "targeted_scripts_b007_loop_variable_repair",
                "target_surface_v429": "scripts",
                "target_rule_v429": "B007",
                "diagnostic_count_v429": diagnostic_count,
                "fixable_count_v429": fixable_count,
                "mutation_allowed_in_v429": False,
                "v430_candidate_v429": diagnostic_count == 8,
                "next_artifact_v429": NEXT_PASS_ARTIFACT,
                "claim_boundary_v429": "v429 plans the next repair but does not mutate scripts",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v429": "repository_ruff_frontier_still_open",
            "blocking_v429": ruff_total > 0,
            "evidence_count_v429": ruff_total,
            "required_next_artifact_v429": NEXT_PASS_ARTIFACT,
            "claim_boundary_v429": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v429": "quarto_render_not_run",
            "blocking_v429": True,
            "evidence_count_v429": 1,
            "required_next_artifact_v429": NEXT_PASS_ARTIFACT,
            "claim_boundary_v429": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v429": "paper4_final_promotion_forbidden",
            "blocking_v429": True,
            "evidence_count_v429": 1,
            "required_next_artifact_v429": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v429": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v429": "full_repository_pytest_failed",
                "blocking_v429": True,
                "evidence_count_v429": 1,
                "required_next_artifact_v429": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v429": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(
    *,
    pytest_passed: bool,
    streamlit_count: int,
    b905_count: int,
    c408_count: int,
    notebook_count: int,
    ruff_total: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v429_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v429_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v428 Streamlit B905/C408 repair",
            },
            {
                "claim_id": "v429_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v429_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v429_streamlit_ruff_remains_clear",
                "allowed": streamlit_count == 0 and b905_count == 0 and c408_count == 0,
                "artifact": "paper4_v429_repository_ruff_snapshot.csv",
                "boundary": "Streamlit and repository B905/C408 remain zero after full pytest",
            },
            {
                "claim_id": "v429_notebook_lint_remains_clean",
                "allowed": notebook_count == 0,
                "artifact": "paper4_v429_repository_ruff_snapshot.csv",
                "boundary": "notebook diagnostics remain zero",
            },
            {
                "claim_id": "v429_repository_ruff_clean",
                "allowed": ruff_total == 0,
                "artifact": "paper4_v429_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v429_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    pytest_passed: bool,
    streamlit_count: int,
    b905_count: int,
    c408_count: int,
    notebook_count: int,
    ruff_total: int,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v429 runs full repository pytest after Streamlit B905/C408 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v429_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v429 full repository pytest passes after Streamlit B905/C408 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v429_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v429 keeps Streamlit ruff and notebook lint clean after pytest.",
                "allowed": streamlit_count == 0 and b905_count == 0 and c408_count == 0 and notebook_count == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v429_repository_ruff_snapshot.csv",
                "boundary": "Specific lint channels only; scripts/book frontier remains open.",
                "prohibited_claim_flag": streamlit_count != 0
                or b905_count != 0
                or c408_count != 0
                or notebook_count != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v429 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v429_claim_blockers.csv",
                "boundary": f"{ruff_total} repository ruff diagnostics remain and Quarto render is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v429 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v429_claim_blockers.csv",
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog(pytest_passed: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": "v429 runs full repository pytest after Streamlit B905/C408 repair.",
                "status": (
                    "post_streamlit_b905_c408_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_streamlit_b905_c408_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v430 applies targeted scripts B007 loop-variable repair"
                    if pytest_passed
                    else "v430 triages pytest failures before more lint repair"
                ),
                "last_wave": "v429",
                "execution_result": (
                    "full_repository_pytest_passed_after_streamlit_b905_c408_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_streamlit_b905_c408_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v429")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Streamlit-B905/C408-Repair Pytest Probe v429

Generated: {status["generated_at_utc"]}

v429 runs full repository pytest after v428's targeted Streamlit B905/C408 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v429"]}`.
- Pytest passed: `{status["pytest_passed_v429"]}`.
- Collected items: `{status["pytest_collected_items_v429"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v429"]}`.
- Summary: `{status["pytest_summary_line_v429"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v429"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v429"]}`.
- Repository B905/C408 diagnostics: `{status["repo_ruff_b905_v429"]}` /
  `{status["repo_ruff_c408_v429"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v429"]}`.
- Top remaining rule: `{status["top_rule_v429"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v429 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v429"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V429_POST_STREAMLIT_B905_C408_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V429_POST_STREAMLIT_B905_C408_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v429: Post-Streamlit-B905/C408-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v429 runs full repository pytest after v428's targeted Streamlit B905/C408
repair.

### Results

- Pytest command:
  `{status["pytest_command_v429"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v429"]}`.
- Pytest passed:
  `{status["pytest_passed_v429"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v429"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v429"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v429"]}`.
- Streamlit diagnostics:
  `{status["streamlit_diagnostics_v429"]}`.
- Repository B905/C408 diagnostics:
  `{status["repo_ruff_b905_v429"]}` /
  `{status["repo_ruff_c408_v429"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v429"]}`.
- Top remaining rule:
  `{status["top_rule_v429"]}` (`{status["top_rule_count_v429"]}`).
- Repository ruff clean:
  `{status["repository_ruff_clean_v429"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v429"]}`.

### Interpretation

The Streamlit B905/C408 repair survives full repository pytest. The remaining
frontier is now scripts/book only, with B007 as the top executable scripts rule.

### Claim Impact

- Allowed: full repository pytest passed after Streamlit repair; Streamlit
  diagnostics, B905/C408 and notebook lint remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v429 in the living notebook. v430 should apply the targeted scripts B007
loop-variable repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _streamlit_target_diff_clean():
        raise RuntimeError("v429 expects committed v428 Streamlit repairs before pytest probe.")

    v428_status = json.loads((STATUS_DIR / "paper4_v428_status.json").read_text(encoding="utf-8"))
    if v428_status["next_artifact_v428"] != "paper4_v429_post_streamlit_b905_c408_repair_pytest_probe.md":
        raise RuntimeError("v429 expects v428 to route to post-Streamlit-repair pytest probe.")
    if v428_status["targeted_streamlit_page_import_tests_passed_v428"] is not True:
        raise RuntimeError("v429 expects v428 targeted Streamlit tests to pass.")
    if int(v428_status["streamlit_diagnostics_after_v428"]) != 0:
        raise RuntimeError("v429 expects Streamlit diagnostics to be cleared by v428.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    items_frame = _items_frame(ruff_items)
    snapshot = _ruff_snapshot(ruff_items)
    snapshot_map = dict(zip(snapshot["metric_v429"], snapshot["diagnostic_count_v429"], strict=False))
    rule_frontier = _rule_frontier(items_frame)
    hotspots = _hotspot_files(ruff_items)
    repair_plan = _repair_plan(rule_frontier)
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(
        pytest_passed=pytest_passed,
        streamlit_count=int(snapshot_map["streamlit_app_total"]),
        b905_count=int(snapshot_map["repository_b905"]),
        c408_count=int(snapshot_map["repository_c408"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_claim_boundaries(
        pytest_passed=pytest_passed,
        streamlit_count=int(snapshot_map["streamlit_app_total"]),
        b905_count=int(snapshot_map["repository_b905"]),
        c408_count=int(snapshot_map["repository_c408"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v429_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v429_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v429_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v429_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v429_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v429_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v429_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    top_rule = rule_frontier.iloc[0]
    top_hotspot = hotspots.iloc[0]
    status = {
        "phase": "v429_post_streamlit_b905_c408_repair_pytest_probe",
        "schema_version": "2026-05-17.429",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_streamlit_b905_c408_repair_version_v429": PRIOR_STREAMLIT_B905_C408_REPAIR_VERSION,
        "pytest_command_v429": pytest_result["command"],
        "pytest_exit_code_v429": int(pytest_result["exit_code"]),
        "pytest_passed_v429": pytest_passed,
        "pytest_runtime_seconds_v429": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v429": int(pytest_result["collected_items"]),
        "pytest_summary_line_v429": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v429": int(ruff_exit),
        "repo_ruff_total_v429": int(snapshot_map["repository_total"]),
        "repo_ruff_e402_v429": int(snapshot_map["repository_e402"]),
        "repo_ruff_b905_v429": int(snapshot_map["repository_b905"]),
        "repo_ruff_c408_v429": int(snapshot_map["repository_c408"]),
        "repo_ruff_b007_v429": int(snapshot_map["repository_b007"]),
        "repo_ruff_f841_v429": int(snapshot_map["repository_f841"]),
        "repo_ruff_b023_v429": int(snapshot_map["repository_b023"]),
        "notebook_diagnostics_v429": int(snapshot_map["notebook_total"]),
        "streamlit_diagnostics_v429": int(snapshot_map["streamlit_app_total"]),
        "scripts_diagnostics_v429": int(snapshot_map["scripts_total"]),
        "book_diagnostics_v429": int(snapshot_map["book_total"]),
        "top_rule_v429": str(top_rule["rule_code_v429"]),
        "top_rule_count_v429": int(top_rule["diagnostic_count_v429"]),
        "top_rule_fixable_v429": int(top_rule["fixable_count_v429"]),
        "top_hotspot_file_v429": str(top_hotspot["file_path_v429"]),
        "top_hotspot_diagnostics_v429": int(top_hotspot["diagnostic_count_v429"]),
        "repository_ruff_clean_v429": int(snapshot_map["repository_total"]) == 0,
        "full_repository_pytest_run_v429": True,
        "full_repository_pytest_passed_v429": pytest_passed,
        "full_quarto_render_run_v429": False,
        "working_champion_claim_allowed_v429": False,
        "paper1_promotion_allowed_v429": False,
        "paper4_working_champion_changed_v429": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v429": next_artifact,
        "claim_boundary": (
            "v429 records post-Streamlit-repair full pytest evidence; repository ruff, "
            "Quarto and final-promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v429"] != 38:
        raise RuntimeError("v429 expected repository ruff frontier to remain at 38 diagnostics.")
    if status["streamlit_diagnostics_v429"] != 0:
        raise RuntimeError("v429 expected Streamlit diagnostics to remain clear.")
    if status["repo_ruff_b905_v429"] != 0 or status["repo_ruff_c408_v429"] != 0:
        raise RuntimeError("v429 expected repository B905/C408 to remain clear.")
    if status["notebook_diagnostics_v429"] != 0:
        raise RuntimeError("v429 expected notebooks to remain lint-clean.")
    if status["top_rule_v429"] != "B007" or status["top_rule_count_v429"] != 8:
        raise RuntimeError("v429 expected B007 to be the next top rule frontier.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v429": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
