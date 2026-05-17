#!/usr/bin/env python3
"""Build Paper 4 v447 post-scripts-B023-repair pytest probe artifacts."""

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

VERSION = 447
PRIOR_SCRIPTS_B023_REPAIR_VERSION = 446
NEXT_PASS_ARTIFACT = "paper4_v448_quarto_render_probe.md"
NEXT_FAIL_ARTIFACT = "paper4_v448_post_b023_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v447_post_scripts_b023_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
B023_TARGET_FILES = [
    "scripts/papers/build_paper4_v10_resolution_wave.py",
    "scripts/papers/build_paper4_v11_promising_lanes.py",
    "scripts/papers/build_paper4_v12_resolution_wave.py",
    "scripts/papers/build_paper4_v41_v44_living_lab_wave.py",
]
COUNT_CODES = [
    "B023",
    "SIM223",
    "C405",
    "UP018",
    "SIM108",
    "UP022",
    "F401",
    "I001",
    "F841",
    "B007",
    "B905",
    "C408",
]


def _b023_target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *B023_TARGET_FILES],
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
    return "other"


def _run_repository_ruff_json() -> tuple[int, list[dict[str, Any]]]:
    result = subprocess.run(RUFF_COMMAND, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "repository ruff probe failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("repository ruff JSON output is not a list")
    return result.returncode, payload


def _snapshot_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    rule_counts = Counter(str(item["code"]) for item in items)
    surface_counts = Counter(_surface(_relative_path(str(item["filename"]))) for item in items)
    counts = {"repository_total": int(len(items))}
    for code in COUNT_CODES:
        counts[f"repository_{code.lower()}"] = int(rule_counts.get(code, 0))
    counts["notebook_total"] = int(surface_counts.get("notebook", 0))
    counts["streamlit_app_total"] = int(surface_counts.get("streamlit_app", 0))
    counts["scripts_total"] = int(surface_counts.get("scripts", 0))
    counts["book_total"] = int(surface_counts.get("book", 0))
    return counts


def _snapshot_frame(counts: dict[str, int]) -> pd.DataFrame:
    ordered = [
        "repository_total",
        "repository_b023",
        "repository_sim223",
        "repository_c405",
        "repository_up018",
        "repository_sim108",
        "repository_up022",
        "repository_f401",
        "repository_i001",
        "repository_f841",
        "repository_b007",
        "repository_b905",
        "repository_c408",
        "notebook_total",
        "streamlit_app_total",
        "scripts_total",
        "book_total",
    ]
    return pd.DataFrame(
        [
            {
                "metric_v447": metric,
                "diagnostic_count_v447": int(counts[metric]),
                "claim_boundary_v447": "post-v446 repository ruff snapshot",
            }
            for metric in ordered
        ]
    )


def _rule_frontier(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["code"])].append(item)
    rows = []
    for code, diagnostics in grouped.items():
        surfaces = Counter(_surface(_relative_path(str(item["filename"]))) for item in diagnostics)
        fixable = sum(1 for item in diagnostics if (item.get("fix") or {}).get("edits"))
        rows.append(
            {
                "rule_code_v447": code,
                "diagnostic_count_v447": int(len(diagnostics)),
                "fixable_count_v447": int(fixable),
                "file_count_v447": int(len({_relative_path(str(item["filename"])) for item in diagnostics})),
                "top_surface_v447": surfaces.most_common(1)[0][0],
                "repair_priority_v447": 0,
                "claim_boundary_v447": "post-v447 frontier only",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "rule_code_v447",
                "diagnostic_count_v447",
                "fixable_count_v447",
                "file_count_v447",
                "top_surface_v447",
                "repair_priority_v447",
                "claim_boundary_v447",
            ]
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v447", "fixable_count_v447", "rule_code_v447"],
        ascending=[False, False, True],
    )
    out["repair_priority_v447"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _hotspot_files(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_relative_path(str(item["filename"]))].append(item)
    rows = []
    for file_path, diagnostics in grouped.items():
        rows.append(
            {
                "file_path_v447": file_path,
                "surface_v447": _surface(file_path),
                "diagnostic_count_v447": int(len(diagnostics)),
                "rule_codes_v447": ",".join(sorted({str(item["code"]) for item in diagnostics})),
                "claim_boundary_v447": "hotspot ranking only; no mutation in v447",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "file_path_v447",
                "surface_v447",
                "diagnostic_count_v447",
                "rule_codes_v447",
                "claim_boundary_v447",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v447", "file_path_v447"],
        ascending=[False, True],
        ignore_index=True,
    )


def _top_executable_rule(rule_frontier: pd.DataFrame) -> pd.Series:
    candidates = rule_frontier.loc[rule_frontier["fixable_count_v447"].astype(int).gt(0)]
    return candidates.iloc[0] if not candidates.empty else rule_frontier.iloc[0]


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v447": "full_repository_pytest",
                "command_v447": pytest_result["command"],
                "exit_code_v447": int(pytest_result["exit_code"]),
                "passed_v447": bool(pytest_result["passed"]),
                "runtime_seconds_v447": float(pytest_result["runtime_seconds"]),
                "collected_items_v447": int(pytest_result["collected_items"]),
                "summary_line_v447": str(pytest_result["summary_line"]),
                "claim_boundary_v447": "post-v446 scripts B023 repair full pytest probe",
            }
        ]
    )


def _repair_plan(*, pytest_passed: bool, repo_clean: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repair_lane_v447": "quarto_render_probe",
                "target_surface_v447": "book",
                "target_rule_v447": "NONE",
                "diagnostic_count_v447": 0,
                "fixable_count_v447": 0,
                "mutation_allowed_in_v447": False,
                "v448_candidate_v447": bool(pytest_passed and repo_clean),
                "next_artifact_v447": NEXT_PASS_ARTIFACT,
                "claim_boundary_v447": "v447 plans Quarto render probe but does not render",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v447": "quarto_render_not_run",
            "blocking_v447": True,
            "evidence_count_v447": 1,
            "required_next_artifact_v447": NEXT_PASS_ARTIFACT,
            "claim_boundary_v447": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v447": "paper4_final_promotion_forbidden",
            "blocking_v447": True,
            "evidence_count_v447": 1,
            "required_next_artifact_v447": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v447": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v447": "full_repository_pytest_failed",
                "blocking_v447": True,
                "evidence_count_v447": 1,
                "required_next_artifact_v447": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v447": "pytest failure must be triaged before more lint repair",
            },
        )
    if ruff_total > 0:
        rows.insert(
            0,
            {
                "blocker_id_v447": "repository_ruff_frontier_still_open",
                "blocking_v447": True,
                "evidence_count_v447": ruff_total,
                "required_next_artifact_v447": NEXT_PASS_ARTIFACT,
                "claim_boundary_v447": "repository ruff clean claim blocked while diagnostics remain",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pytest_passed: bool, counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v447_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v447_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v446 manual B023 repair",
            },
            {
                "claim_id": "v447_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v447_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v447_b023_remains_clear",
                "allowed": counts["repository_b023"] == 0,
                "artifact": "paper4_v447_repository_ruff_snapshot.csv",
                "boundary": "repository B023 remains zero after full pytest",
            },
            {
                "claim_id": "v447_prior_lint_channels_remain_clean",
                "allowed": counts["repository_b023"] == 0
                and counts["repository_c405"] == 0
                and counts["repository_up018"] == 0
                and counts["repository_sim108"] == 0
                and counts["repository_up022"] == 0
                and counts["repository_f401"] == 0
                and counts["repository_i001"] == 0
                and counts["repository_f841"] == 0
                and counts["streamlit_app_total"] == 0
                and counts["notebook_total"] == 0
                and counts["book_total"] == 0,
                "artifact": "paper4_v447_repository_ruff_snapshot.csv",
                "boundary": "previously cleared lint channels remain zero",
            },
            {
                "claim_id": "v447_repository_ruff_clean",
                "allowed": counts["repository_total"] == 0,
                "artifact": "paper4_v447_repository_ruff_snapshot.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v447_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, pytest_passed: bool, counts: dict[str, int]) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    channels_clean = (
        counts["repository_b023"] == 0
        and counts["repository_c405"] == 0
        and counts["repository_up018"] == 0
        and counts["repository_sim108"] == 0
        and counts["repository_up022"] == 0
        and counts["repository_f401"] == 0
        and counts["repository_i001"] == 0
        and counts["streamlit_app_total"] == 0
        and counts["notebook_total"] == 0
        and counts["book_total"] == 0
    )
    additions = pd.DataFrame(
        [
            {
                "claim": "v447 runs full repository pytest after scripts B023 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v447_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v447 full repository pytest passes after scripts B023 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v447_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v447 keeps B023 and prior automated lint channels clean after pytest.",
                "allowed": channels_clean,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v447_repository_ruff_snapshot.csv",
                "boundary": "Repository ruff is clean; Quarto render remains separate.",
                "prohibited_claim_flag": not channels_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v447 proves Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v447_claim_blockers.csv",
                "boundary": "Quarto render is deferred to v448.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v447 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v447_claim_blockers.csv",
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
                "executable_item": "v447 runs full repository pytest after scripts B023 repair.",
                "status": (
                    "post_scripts_b023_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_scripts_b023_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v448 Quarto render probe runs after clean pytest and ruff"
                    if pytest_passed
                    else "v448 triages pytest failures before Quarto render"
                ),
                "last_wave": "v447",
                "execution_result": (
                    "full_repository_pytest_passed_after_scripts_b023_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_scripts_b023_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v447")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Scripts-B023-Repair Pytest Probe v447

Generated: {status["generated_at_utc"]}

v447 runs full repository pytest after v446's manual scripts B023 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v447"]}`.
- Pytest passed: `{status["pytest_passed_v447"]}`.
- Collected items: `{status["pytest_collected_items_v447"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v447"]}`.
- Summary: `{status["pytest_summary_line_v447"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v447"]}`.
- Repository B023 diagnostics: `{status["repo_ruff_b023_v447"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v447"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v447"]}`.
- Book diagnostics: `{status["book_diagnostics_v447"]}`.
- Top executable rule: `{status["top_executable_rule_v447"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v447 does not claim Quarto render or Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v447"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V447_POST_SCRIPTS_B023_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V447_POST_SCRIPTS_B023_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v447: Post-Scripts-B023-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v447 runs full repository pytest after v446's manual scripts B023 repair.

### Results

- Pytest command:
  `{status["pytest_command_v447"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v447"]}`.
- Pytest passed:
  `{status["pytest_passed_v447"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v447"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v447"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v447"]}`.
- Repository B023 diagnostics:
  `{status["repo_ruff_b023_v447"]}`.
- Streamlit diagnostics:
  `{status["streamlit_diagnostics_v447"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v447"]}`.
- Book diagnostics:
  `{status["book_diagnostics_v447"]}`.
- Remaining Ruff frontier:
  `{status["top_rule_v447"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v447"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v447"]}`.

### Interpretation

The manual B023 repair survives full repository pytest, and repository Ruff
remains clean. The next gate is an explicit Quarto render probe.

### Claim Impact

- Allowed: full repository pytest passed after manual B023 repair; repository
  ruff remains clean.
- Still prohibited: Quarto render clean, champion replacement and final
  promotion claims.

### Quarto Promotion Decision

Keep v447 in the living notebook. v448 should run the Quarto render probe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _b023_target_diff_clean():
        raise RuntimeError("v447 expects committed v446 scripts B023 repairs before pytest probe.")

    v446_status = json.loads((STATUS_DIR / "paper4_v446_status.json").read_text(encoding="utf-8"))
    if v446_status["next_artifact_v446"] != "paper4_v447_post_scripts_b023_repair_pytest_probe.md":
        raise RuntimeError("v447 expects v446 to route to post-B023-repair pytest probe.")
    if v446_status["changed_scripts_pycompile_passed_v446"] is not True:
        raise RuntimeError("v447 expects v446 changed scripts to compile.")
    if int(v446_status["repo_ruff_b023_after_v446"]) != 0:
        raise RuntimeError("v447 expects B023 to be cleared by v446.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    counts = _snapshot_counts(ruff_items)
    snapshot = _snapshot_frame(counts)
    rule_frontier = _rule_frontier(ruff_items)
    hotspots = _hotspot_files(ruff_items)
    repo_clean = len(ruff_items) == 0
    if rule_frontier.empty:
        top_rule_code = "NONE"
        top_rule_count = 0
        top_rule_fixable = 0
        top_executable_code = "NONE"
        top_executable_count = 0
        top_executable_fixable = 0
    else:
        top_rule = rule_frontier.iloc[0]
        top_executable = _top_executable_rule(rule_frontier)
        top_rule_code = str(top_rule["rule_code_v447"])
        top_rule_count = int(top_rule["diagnostic_count_v447"])
        top_rule_fixable = int(top_rule["fixable_count_v447"])
        top_executable_code = str(top_executable["rule_code_v447"])
        top_executable_count = int(top_executable["diagnostic_count_v447"])
        top_executable_fixable = int(top_executable["fixable_count_v447"])
    repair_plan = _repair_plan(pytest_passed=pytest_passed, repo_clean=repo_clean)
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(pytest_passed=pytest_passed, counts=counts)
    _update_claim_boundaries(pytest_passed=pytest_passed, counts=counts)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v447_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v447_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v447_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v447_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v447_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v447_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v447_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    top_hotspot_file = "NONE" if hotspots.empty else str(hotspots.iloc[0]["file_path_v447"])
    top_hotspot_diagnostics = 0 if hotspots.empty else int(hotspots.iloc[0]["diagnostic_count_v447"])
    status = {
        "phase": "v447_post_scripts_b023_repair_pytest_probe",
        "schema_version": "2026-05-17.447",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scripts_b023_repair_version_v447": PRIOR_SCRIPTS_B023_REPAIR_VERSION,
        "pytest_command_v447": pytest_result["command"],
        "pytest_exit_code_v447": int(pytest_result["exit_code"]),
        "pytest_passed_v447": pytest_passed,
        "pytest_runtime_seconds_v447": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v447": int(pytest_result["collected_items"]),
        "pytest_summary_line_v447": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v447": int(ruff_exit),
        "repo_ruff_total_v447": counts["repository_total"],
        "repo_ruff_b023_v447": counts["repository_b023"],
        "repo_ruff_sim223_v447": counts["repository_sim223"],
        "repo_ruff_c405_v447": counts["repository_c405"],
        "repo_ruff_up018_v447": counts["repository_up018"],
        "repo_ruff_sim108_v447": counts["repository_sim108"],
        "repo_ruff_up022_v447": counts["repository_up022"],
        "repo_ruff_f401_v447": counts["repository_f401"],
        "repo_ruff_i001_v447": counts["repository_i001"],
        "repo_ruff_f841_v447": counts["repository_f841"],
        "repo_ruff_b007_v447": counts["repository_b007"],
        "repo_ruff_b905_v447": counts["repository_b905"],
        "repo_ruff_c408_v447": counts["repository_c408"],
        "notebook_diagnostics_v447": counts["notebook_total"],
        "streamlit_diagnostics_v447": counts["streamlit_app_total"],
        "scripts_diagnostics_v447": counts["scripts_total"],
        "book_diagnostics_v447": counts["book_total"],
        "top_rule_v447": top_rule_code,
        "top_rule_count_v447": top_rule_count,
        "top_rule_fixable_v447": top_rule_fixable,
        "top_executable_rule_v447": top_executable_code,
        "top_executable_rule_count_v447": top_executable_count,
        "top_executable_rule_fixable_v447": top_executable_fixable,
        "top_hotspot_file_v447": top_hotspot_file,
        "top_hotspot_diagnostics_v447": top_hotspot_diagnostics,
        "repository_ruff_clean_v447": counts["repository_total"] == 0,
        "full_repository_pytest_run_v447": True,
        "full_repository_pytest_passed_v447": pytest_passed,
        "full_quarto_render_run_v447": False,
        "working_champion_claim_allowed_v447": False,
        "paper1_promotion_allowed_v447": False,
        "paper4_working_champion_changed_v447": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v447": next_artifact,
        "claim_boundary": (
            "v447 is a post-B023-repair full pytest probe; repository ruff "
            "and final promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v447"] != 0:
        raise RuntimeError("v447 expected repository ruff to remain clean.")
    if status["repo_ruff_b023_v447"] != 0:
        raise RuntimeError("v447 expected repository B023 to remain clear.")
    if status["streamlit_diagnostics_v447"] != 0 or status["notebook_diagnostics_v447"] != 0:
        raise RuntimeError("v447 expected Streamlit and notebooks to remain clean.")
    if status["book_diagnostics_v447"] != 0:
        raise RuntimeError("v447 expected book diagnostics to remain clean.")
    if status["top_executable_rule_v447"] != "NONE":
        raise RuntimeError("v447 expected no remaining Ruff repair frontier.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v447": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
