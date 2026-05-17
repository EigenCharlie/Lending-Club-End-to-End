#!/usr/bin/env python3
"""Build Paper 4 v431 post-scripts-B007-repair pytest probe artifacts."""

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

VERSION = 431
PRIOR_SCRIPTS_B007_REPAIR_VERSION = 430
NEXT_PASS_ARTIFACT = "paper4_v432_scripts_f841_unused_variable_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v432_post_b007_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v431_post_scripts_b007_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
B007_TARGET_FILES = [
    "scripts/papers/build_paper4_extended_experiments.py",
    "scripts/papers/build_paper4_v15_dynamic_stress_engine.py",
    "scripts/papers/build_paper4_v20_dla_cvar_spo_resolution.py",
    "scripts/papers/build_paper4_v24_dla_cvar_spo_upgrade.py",
    "scripts/papers/build_paper4_v45_v48_living_lab_wave.py",
    "scripts/papers/build_paper4_v5_blocker_resolution.py",
]


def _b007_target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *B007_TARGET_FILES],
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
                "file_path_v431": path,
                "surface_v431": _surface(path),
                "rule_code_v431": str(item["code"]),
                "message_v431": str(item["message"]),
                "row_v431": int((item.get("location") or {}).get("row") or 0),
                "fixable_v431": bool(fix.get("edits")),
                "claim_boundary_v431": "post-v430 repository ruff diagnostic inventory",
            }
        )
    return pd.DataFrame(rows)


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v431": "full_repository_pytest",
                "command_v431": pytest_result["command"],
                "exit_code_v431": int(pytest_result["exit_code"]),
                "passed_v431": bool(pytest_result["passed"]),
                "runtime_seconds_v431": float(pytest_result["runtime_seconds"]),
                "collected_items_v431": int(pytest_result["collected_items"]),
                "summary_line_v431": str(pytest_result["summary_line"]),
                "claim_boundary_v431": "post-v430 scripts B007 repair full pytest probe",
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
        ("repository_b007", _count(items, code="B007")),
        ("repository_f841", _count(items, code="F841")),
        ("repository_b023", _count(items, code="B023")),
        ("repository_i001", _count(items, code="I001")),
        ("repository_f401", _count(items, code="F401")),
        ("repository_up022", _count(items, code="UP022")),
        ("repository_sim108", _count(items, code="SIM108")),
        ("repository_b905", _count(items, code="B905")),
        ("repository_c408", _count(items, code="C408")),
        ("notebook_total", _count(items, surface="notebook")),
        ("streamlit_app_total", _count(items, surface="streamlit_app")),
        ("scripts_total", _count(items, surface="scripts")),
        ("book_total", _count(items, surface="book")),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v431": metric,
                "diagnostic_count_v431": int(count),
                "claim_boundary_v431": "post-v430 repository ruff snapshot",
            }
            for metric, count in rows
        ]
    )


def _rule_frontier(items_frame: pd.DataFrame) -> pd.DataFrame:
    if items_frame.empty:
        return pd.DataFrame(
            [
                {
                    "rule_code_v431": "__none__",
                    "diagnostic_count_v431": 0,
                    "fixable_count_v431": 0,
                    "file_count_v431": 0,
                    "top_surface_v431": "__none__",
                    "repair_priority_v431": 1,
                    "claim_boundary_v431": "repository ruff clean after v431",
                }
            ]
        )
    rows = []
    for code, group in items_frame.groupby("rule_code_v431", sort=False):
        surface_counts = Counter(group["surface_v431"])
        rows.append(
            {
                "rule_code_v431": code,
                "diagnostic_count_v431": int(len(group)),
                "fixable_count_v431": int(group["fixable_v431"].astype(bool).sum()),
                "file_count_v431": int(group["file_path_v431"].nunique()),
                "top_surface_v431": surface_counts.most_common(1)[0][0],
                "repair_priority_v431": 0,
                "claim_boundary_v431": "post-v431 frontier only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v431", "fixable_count_v431", "rule_code_v431"],
        ascending=[False, False, True],
    )
    out["repair_priority_v431"] = range(1, len(out) + 1)
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
                "file_path_v431": file_path,
                "surface_v431": _surface(file_path),
                "diagnostic_count_v431": int(len(diagnostics)),
                "rule_codes_v431": ",".join(rule_codes),
                "claim_boundary_v431": "hotspot ranking only; no mutation in v431",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v431", "file_path_v431"],
        ascending=[False, True],
        ignore_index=True,
    )


def _repair_plan(rule_frontier: pd.DataFrame) -> pd.DataFrame:
    f841 = rule_frontier.loc[rule_frontier["rule_code_v431"].eq("F841")]
    diagnostic_count = int(f841["diagnostic_count_v431"].iloc[0]) if not f841.empty else 0
    fixable_count = int(f841["fixable_count_v431"].iloc[0]) if not f841.empty else 0
    return pd.DataFrame(
        [
            {
                "repair_lane_v431": "targeted_scripts_f841_unused_variable_repair",
                "target_surface_v431": "scripts",
                "target_rule_v431": "F841",
                "diagnostic_count_v431": diagnostic_count,
                "fixable_count_v431": fixable_count,
                "mutation_allowed_in_v431": False,
                "v432_candidate_v431": diagnostic_count == 7 and fixable_count == 7,
                "next_artifact_v431": NEXT_PASS_ARTIFACT,
                "claim_boundary_v431": "v431 plans the next repair but does not mutate scripts",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v431": "repository_ruff_frontier_still_open",
            "blocking_v431": ruff_total > 0,
            "evidence_count_v431": ruff_total,
            "required_next_artifact_v431": NEXT_PASS_ARTIFACT,
            "claim_boundary_v431": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v431": "quarto_render_not_run",
            "blocking_v431": True,
            "evidence_count_v431": 1,
            "required_next_artifact_v431": NEXT_PASS_ARTIFACT,
            "claim_boundary_v431": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v431": "paper4_final_promotion_forbidden",
            "blocking_v431": True,
            "evidence_count_v431": 1,
            "required_next_artifact_v431": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v431": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v431": "full_repository_pytest_failed",
                "blocking_v431": True,
                "evidence_count_v431": 1,
                "required_next_artifact_v431": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v431": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(
    *,
    pytest_passed: bool,
    b007_count: int,
    streamlit_count: int,
    notebook_count: int,
    ruff_total: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v431_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v431_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v430 scripts B007 repair",
            },
            {
                "claim_id": "v431_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v431_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v431_scripts_b007_remains_clear",
                "allowed": b007_count == 0,
                "artifact": "paper4_v431_repository_ruff_snapshot.csv",
                "boundary": "scripts/repository B007 remains zero after full pytest",
            },
            {
                "claim_id": "v431_streamlit_and_notebooks_remain_clean",
                "allowed": streamlit_count == 0 and notebook_count == 0,
                "artifact": "paper4_v431_repository_ruff_snapshot.csv",
                "boundary": "Streamlit and notebook diagnostics remain zero",
            },
            {
                "claim_id": "v431_repository_ruff_clean",
                "allowed": ruff_total == 0,
                "artifact": "paper4_v431_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v431_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    pytest_passed: bool,
    b007_count: int,
    streamlit_count: int,
    notebook_count: int,
    ruff_total: int,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v431 runs full repository pytest after scripts B007 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v431_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v431 full repository pytest passes after scripts B007 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v431_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v431 keeps B007, Streamlit and notebook lint clean after pytest.",
                "allowed": b007_count == 0 and streamlit_count == 0 and notebook_count == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v431_repository_ruff_snapshot.csv",
                "boundary": "Specific lint channels only; repository ruff remains open.",
                "prohibited_claim_flag": b007_count != 0
                or streamlit_count != 0
                or notebook_count != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v431 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v431_claim_blockers.csv",
                "boundary": f"{ruff_total} repository ruff diagnostics remain and Quarto render is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v431 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v431_claim_blockers.csv",
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
                "executable_item": "v431 runs full repository pytest after scripts B007 repair.",
                "status": (
                    "post_scripts_b007_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_scripts_b007_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v432 applies targeted scripts F841 unused-variable repair"
                    if pytest_passed
                    else "v432 triages pytest failures before more lint repair"
                ),
                "last_wave": "v431",
                "execution_result": (
                    "full_repository_pytest_passed_after_scripts_b007_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_scripts_b007_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v431")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Scripts-B007-Repair Pytest Probe v431

Generated: {status["generated_at_utc"]}

v431 runs full repository pytest after v430's targeted scripts B007 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v431"]}`.
- Pytest passed: `{status["pytest_passed_v431"]}`.
- Collected items: `{status["pytest_collected_items_v431"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v431"]}`.
- Summary: `{status["pytest_summary_line_v431"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v431"]}`.
- Repository B007 diagnostics: `{status["repo_ruff_b007_v431"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v431"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v431"]}`.
- Top remaining rule: `{status["top_rule_v431"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v431 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v431"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V431_POST_SCRIPTS_B007_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V431_POST_SCRIPTS_B007_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v431: Post-Scripts-B007-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v431 runs full repository pytest after v430's targeted scripts B007 repair.

### Results

- Pytest command:
  `{status["pytest_command_v431"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v431"]}`.
- Pytest passed:
  `{status["pytest_passed_v431"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v431"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v431"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v431"]}`.
- Repository B007 diagnostics:
  `{status["repo_ruff_b007_v431"]}`.
- Streamlit diagnostics:
  `{status["streamlit_diagnostics_v431"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v431"]}`.
- Top remaining rule:
  `{status["top_rule_v431"]}` (`{status["top_rule_count_v431"]}`).
- Repository ruff clean:
  `{status["repository_ruff_clean_v431"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v431"]}`.

### Interpretation

The scripts B007 repair survives full repository pytest. F841 is now tied with
B023 by count, but F841 is fully fixable and therefore the next executable
batch.

### Claim Impact

- Allowed: full repository pytest passed after scripts B007 repair; B007,
  Streamlit and notebook lint remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v431 in the living notebook. v432 should apply the targeted scripts F841
unused-variable repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _b007_target_diff_clean():
        raise RuntimeError("v431 expects committed v430 scripts B007 repairs before pytest probe.")

    v430_status = json.loads((STATUS_DIR / "paper4_v430_status.json").read_text(encoding="utf-8"))
    if v430_status["next_artifact_v430"] != "paper4_v431_post_scripts_b007_repair_pytest_probe.md":
        raise RuntimeError("v431 expects v430 to route to post-B007-repair pytest probe.")
    if v430_status["changed_scripts_pycompile_passed_v430"] is not True:
        raise RuntimeError("v431 expects v430 changed scripts to compile.")
    if int(v430_status["repo_ruff_b007_after_v430"]) != 0:
        raise RuntimeError("v431 expects B007 to be cleared by v430.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    items_frame = _items_frame(ruff_items)
    snapshot = _ruff_snapshot(ruff_items)
    snapshot_map = dict(zip(snapshot["metric_v431"], snapshot["diagnostic_count_v431"], strict=False))
    rule_frontier = _rule_frontier(items_frame)
    hotspots = _hotspot_files(ruff_items)
    repair_plan = _repair_plan(rule_frontier)
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(
        pytest_passed=pytest_passed,
        b007_count=int(snapshot_map["repository_b007"]),
        streamlit_count=int(snapshot_map["streamlit_app_total"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_claim_boundaries(
        pytest_passed=pytest_passed,
        b007_count=int(snapshot_map["repository_b007"]),
        streamlit_count=int(snapshot_map["streamlit_app_total"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v431_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v431_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v431_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v431_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v431_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v431_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v431_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    top_rule = rule_frontier.iloc[0]
    top_hotspot = hotspots.iloc[0]
    status = {
        "phase": "v431_post_scripts_b007_repair_pytest_probe",
        "schema_version": "2026-05-17.431",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scripts_b007_repair_version_v431": PRIOR_SCRIPTS_B007_REPAIR_VERSION,
        "pytest_command_v431": pytest_result["command"],
        "pytest_exit_code_v431": int(pytest_result["exit_code"]),
        "pytest_passed_v431": pytest_passed,
        "pytest_runtime_seconds_v431": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v431": int(pytest_result["collected_items"]),
        "pytest_summary_line_v431": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v431": int(ruff_exit),
        "repo_ruff_total_v431": int(snapshot_map["repository_total"]),
        "repo_ruff_b007_v431": int(snapshot_map["repository_b007"]),
        "repo_ruff_f841_v431": int(snapshot_map["repository_f841"]),
        "repo_ruff_b023_v431": int(snapshot_map["repository_b023"]),
        "repo_ruff_i001_v431": int(snapshot_map["repository_i001"]),
        "repo_ruff_f401_v431": int(snapshot_map["repository_f401"]),
        "repo_ruff_up022_v431": int(snapshot_map["repository_up022"]),
        "repo_ruff_sim108_v431": int(snapshot_map["repository_sim108"]),
        "repo_ruff_b905_v431": int(snapshot_map["repository_b905"]),
        "repo_ruff_c408_v431": int(snapshot_map["repository_c408"]),
        "notebook_diagnostics_v431": int(snapshot_map["notebook_total"]),
        "streamlit_diagnostics_v431": int(snapshot_map["streamlit_app_total"]),
        "scripts_diagnostics_v431": int(snapshot_map["scripts_total"]),
        "book_diagnostics_v431": int(snapshot_map["book_total"]),
        "top_rule_v431": str(top_rule["rule_code_v431"]),
        "top_rule_count_v431": int(top_rule["diagnostic_count_v431"]),
        "top_rule_fixable_v431": int(top_rule["fixable_count_v431"]),
        "top_hotspot_file_v431": str(top_hotspot["file_path_v431"]),
        "top_hotspot_diagnostics_v431": int(top_hotspot["diagnostic_count_v431"]),
        "repository_ruff_clean_v431": int(snapshot_map["repository_total"]) == 0,
        "full_repository_pytest_run_v431": True,
        "full_repository_pytest_passed_v431": pytest_passed,
        "full_quarto_render_run_v431": False,
        "working_champion_claim_allowed_v431": False,
        "paper1_promotion_allowed_v431": False,
        "paper4_working_champion_changed_v431": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v431": next_artifact,
        "claim_boundary": (
            "v431 records post-B007-repair full pytest evidence; repository ruff, "
            "Quarto and final-promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v431"] != 30:
        raise RuntimeError("v431 expected repository ruff frontier to remain at 30 diagnostics.")
    if status["repo_ruff_b007_v431"] != 0:
        raise RuntimeError("v431 expected B007 to remain clear.")
    if status["streamlit_diagnostics_v431"] != 0 or status["notebook_diagnostics_v431"] != 0:
        raise RuntimeError("v431 expected Streamlit and notebooks to remain clean.")
    if status["top_rule_v431"] != "F841" or status["top_rule_count_v431"] != 7:
        raise RuntimeError("v431 expected F841 to be the next top executable rule frontier.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v431": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
