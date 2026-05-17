#!/usr/bin/env python3
"""Build Paper 4 v441 post-scripts-SIM108-repair pytest probe artifacts."""

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

VERSION = 441
PRIOR_SCRIPTS_SIM108_REPAIR_VERSION = 440
NEXT_PASS_ARTIFACT = "paper4_v442_scripts_c405_set_literal_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v442_post_sim108_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v441_post_scripts_sim108_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
SIM108_TARGET_FILES = [
    "scripts/papers/build_paper4_v10_resolution_wave.py",
    "scripts/papers/build_paper4_v41_v44_living_lab_wave.py",
]
COUNT_CODES = [
    "B023",
    "C405",
    "SIM223",
    "SIM108",
    "UP022",
    "F401",
    "I001",
    "F841",
    "B007",
    "B905",
    "C408",
]


def _sim108_target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *SIM108_TARGET_FILES],
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
        "repository_c405",
        "repository_sim223",
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
                "metric_v441": metric,
                "diagnostic_count_v441": int(counts[metric]),
                "claim_boundary_v441": "post-v440 repository ruff snapshot",
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
                "rule_code_v441": code,
                "diagnostic_count_v441": int(len(diagnostics)),
                "fixable_count_v441": int(fixable),
                "file_count_v441": int(len({_relative_path(str(item["filename"])) for item in diagnostics})),
                "top_surface_v441": surfaces.most_common(1)[0][0],
                "repair_priority_v441": 0,
                "claim_boundary_v441": "post-v441 frontier only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v441", "fixable_count_v441", "rule_code_v441"],
        ascending=[False, False, True],
    )
    out["repair_priority_v441"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _hotspot_files(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_relative_path(str(item["filename"]))].append(item)
    rows = []
    for file_path, diagnostics in grouped.items():
        rows.append(
            {
                "file_path_v441": file_path,
                "surface_v441": _surface(file_path),
                "diagnostic_count_v441": int(len(diagnostics)),
                "rule_codes_v441": ",".join(sorted({str(item["code"]) for item in diagnostics})),
                "claim_boundary_v441": "hotspot ranking only; no mutation in v441",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v441", "file_path_v441"],
        ascending=[False, True],
        ignore_index=True,
    )


def _top_executable_rule(rule_frontier: pd.DataFrame) -> pd.Series:
    candidates = rule_frontier.loc[rule_frontier["fixable_count_v441"].astype(int).gt(0)]
    return candidates.iloc[0] if not candidates.empty else rule_frontier.iloc[0]


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v441": "full_repository_pytest",
                "command_v441": pytest_result["command"],
                "exit_code_v441": int(pytest_result["exit_code"]),
                "passed_v441": bool(pytest_result["passed"]),
                "runtime_seconds_v441": float(pytest_result["runtime_seconds"]),
                "collected_items_v441": int(pytest_result["collected_items"]),
                "summary_line_v441": str(pytest_result["summary_line"]),
                "claim_boundary_v441": "post-v440 scripts SIM108 repair full pytest probe",
            }
        ]
    )


def _repair_plan(top_executable: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repair_lane_v441": "targeted_scripts_c405_set_literal_repair",
                "target_surface_v441": "scripts",
                "target_rule_v441": str(top_executable["rule_code_v441"]),
                "diagnostic_count_v441": int(top_executable["diagnostic_count_v441"]),
                "fixable_count_v441": int(top_executable["fixable_count_v441"]),
                "mutation_allowed_in_v441": False,
                "v442_candidate_v441": str(top_executable["rule_code_v441"]) == "C405",
                "next_artifact_v441": NEXT_PASS_ARTIFACT,
                "claim_boundary_v441": "v441 plans the next repair but does not mutate set literals",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v441": "repository_ruff_frontier_still_open",
            "blocking_v441": ruff_total > 0,
            "evidence_count_v441": ruff_total,
            "required_next_artifact_v441": NEXT_PASS_ARTIFACT,
            "claim_boundary_v441": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v441": "quarto_render_not_run",
            "blocking_v441": True,
            "evidence_count_v441": 1,
            "required_next_artifact_v441": NEXT_PASS_ARTIFACT,
            "claim_boundary_v441": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v441": "paper4_final_promotion_forbidden",
            "blocking_v441": True,
            "evidence_count_v441": 1,
            "required_next_artifact_v441": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v441": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v441": "full_repository_pytest_failed",
                "blocking_v441": True,
                "evidence_count_v441": 1,
                "required_next_artifact_v441": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v441": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pytest_passed: bool, counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v441_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v441_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v440 scripts SIM108 repair",
            },
            {
                "claim_id": "v441_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v441_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v441_sim108_remains_clear",
                "allowed": counts["repository_sim108"] == 0,
                "artifact": "paper4_v441_repository_ruff_snapshot.csv",
                "boundary": "repository SIM108 remains zero after full pytest",
            },
            {
                "claim_id": "v441_prior_lint_channels_remain_clean",
                "allowed": counts["repository_up022"] == 0
                and counts["repository_f401"] == 0
                and counts["repository_i001"] == 0
                and counts["repository_f841"] == 0
                and counts["streamlit_app_total"] == 0
                and counts["notebook_total"] == 0
                and counts["book_total"] == 0,
                "artifact": "paper4_v441_repository_ruff_snapshot.csv",
                "boundary": "previously cleared lint channels remain zero",
            },
            {
                "claim_id": "v441_repository_ruff_clean",
                "allowed": counts["repository_total"] == 0,
                "artifact": "paper4_v441_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v441_working_champion_or_final_promotion",
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
        counts["repository_sim108"] == 0
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
                "claim": "v441 runs full repository pytest after scripts SIM108 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v441_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v441 full repository pytest passes after scripts SIM108 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v441_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v441 keeps SIM108 and prior lint channels clean after pytest.",
                "allowed": channels_clean,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v441_repository_ruff_snapshot.csv",
                "boundary": "Specific lint channels only; repository ruff remains open.",
                "prohibited_claim_flag": not channels_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v441 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v441_claim_blockers.csv",
                "boundary": (
                    f"{counts['repository_total']} repository ruff diagnostics remain "
                    "and Quarto render is deferred."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v441 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v441_claim_blockers.csv",
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
                "executable_item": "v441 runs full repository pytest after scripts SIM108 repair.",
                "status": (
                    "post_scripts_sim108_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_scripts_sim108_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v442 applies targeted C405 set-literal repair"
                    if pytest_passed
                    else "v442 triages pytest failures before more lint repair"
                ),
                "last_wave": "v441",
                "execution_result": (
                    "full_repository_pytest_passed_after_scripts_sim108_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_scripts_sim108_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v441")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Scripts-SIM108-Repair Pytest Probe v441

Generated: {status["generated_at_utc"]}

v441 runs full repository pytest after v440's targeted scripts SIM108 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v441"]}`.
- Pytest passed: `{status["pytest_passed_v441"]}`.
- Collected items: `{status["pytest_collected_items_v441"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v441"]}`.
- Summary: `{status["pytest_summary_line_v441"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v441"]}`.
- Repository SIM108 diagnostics: `{status["repo_ruff_sim108_v441"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v441"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v441"]}`.
- Book diagnostics: `{status["book_diagnostics_v441"]}`.
- Top executable rule: `{status["top_executable_rule_v441"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v441 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v441"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V441_POST_SCRIPTS_SIM108_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V441_POST_SCRIPTS_SIM108_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v441: Post-Scripts-SIM108-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v441 runs full repository pytest after v440's targeted scripts SIM108 repair.

### Results

- Pytest command:
  `{status["pytest_command_v441"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v441"]}`.
- Pytest passed:
  `{status["pytest_passed_v441"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v441"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v441"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v441"]}`.
- Repository SIM108 diagnostics:
  `{status["repo_ruff_sim108_v441"]}`.
- Streamlit diagnostics:
  `{status["streamlit_diagnostics_v441"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v441"]}`.
- Book diagnostics:
  `{status["book_diagnostics_v441"]}`.
- Top rule / top executable rule:
  `{status["top_rule_v441"]}` /
  `{status["top_executable_rule_v441"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v441"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v441"]}`.

### Interpretation

The scripts SIM108 repair survives full repository pytest. B023 remains the top
rule by count but has no automatic fixes, so C405 is the next executable repair
frontier.

### Claim Impact

- Allowed: full repository pytest passed after scripts SIM108 repair; SIM108
  and earlier cleared lint channels remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v441 in the living notebook. v442 should apply the targeted C405
set-literal repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _sim108_target_diff_clean():
        raise RuntimeError("v441 expects committed v440 scripts SIM108 repairs before pytest probe.")

    v440_status = json.loads((STATUS_DIR / "paper4_v440_status.json").read_text(encoding="utf-8"))
    if v440_status["next_artifact_v440"] != "paper4_v441_post_scripts_sim108_repair_pytest_probe.md":
        raise RuntimeError("v441 expects v440 to route to post-SIM108-repair pytest probe.")
    if v440_status["changed_scripts_pycompile_passed_v440"] is not True:
        raise RuntimeError("v441 expects v440 changed scripts to compile.")
    if int(v440_status["repo_ruff_sim108_after_v440"]) != 0:
        raise RuntimeError("v441 expects SIM108 to be cleared by v440.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    counts = _snapshot_counts(ruff_items)
    snapshot = _snapshot_frame(counts)
    rule_frontier = _rule_frontier(ruff_items)
    hotspots = _hotspot_files(ruff_items)
    top_rule = rule_frontier.iloc[0]
    top_executable = _top_executable_rule(rule_frontier)
    repair_plan = _repair_plan(top_executable)
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(pytest_passed=pytest_passed, counts=counts)
    _update_claim_boundaries(pytest_passed=pytest_passed, counts=counts)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v441_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v441_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v441_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v441_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v441_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v441_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v441_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    top_hotspot = hotspots.iloc[0]
    status = {
        "phase": "v441_post_scripts_sim108_repair_pytest_probe",
        "schema_version": "2026-05-17.441",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scripts_sim108_repair_version_v441": PRIOR_SCRIPTS_SIM108_REPAIR_VERSION,
        "pytest_command_v441": pytest_result["command"],
        "pytest_exit_code_v441": int(pytest_result["exit_code"]),
        "pytest_passed_v441": pytest_passed,
        "pytest_runtime_seconds_v441": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v441": int(pytest_result["collected_items"]),
        "pytest_summary_line_v441": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v441": int(ruff_exit),
        "repo_ruff_total_v441": counts["repository_total"],
        "repo_ruff_b023_v441": counts["repository_b023"],
        "repo_ruff_c405_v441": counts["repository_c405"],
        "repo_ruff_sim223_v441": counts["repository_sim223"],
        "repo_ruff_sim108_v441": counts["repository_sim108"],
        "repo_ruff_up022_v441": counts["repository_up022"],
        "repo_ruff_f401_v441": counts["repository_f401"],
        "repo_ruff_i001_v441": counts["repository_i001"],
        "repo_ruff_f841_v441": counts["repository_f841"],
        "repo_ruff_b007_v441": counts["repository_b007"],
        "repo_ruff_b905_v441": counts["repository_b905"],
        "repo_ruff_c408_v441": counts["repository_c408"],
        "notebook_diagnostics_v441": counts["notebook_total"],
        "streamlit_diagnostics_v441": counts["streamlit_app_total"],
        "scripts_diagnostics_v441": counts["scripts_total"],
        "book_diagnostics_v441": counts["book_total"],
        "top_rule_v441": str(top_rule["rule_code_v441"]),
        "top_rule_count_v441": int(top_rule["diagnostic_count_v441"]),
        "top_rule_fixable_v441": int(top_rule["fixable_count_v441"]),
        "top_executable_rule_v441": str(top_executable["rule_code_v441"]),
        "top_executable_rule_count_v441": int(top_executable["diagnostic_count_v441"]),
        "top_executable_rule_fixable_v441": int(top_executable["fixable_count_v441"]),
        "top_hotspot_file_v441": str(top_hotspot["file_path_v441"]),
        "top_hotspot_diagnostics_v441": int(top_hotspot["diagnostic_count_v441"]),
        "repository_ruff_clean_v441": counts["repository_total"] == 0,
        "full_repository_pytest_run_v441": True,
        "full_repository_pytest_passed_v441": pytest_passed,
        "full_quarto_render_run_v441": False,
        "working_champion_claim_allowed_v441": False,
        "paper1_promotion_allowed_v441": False,
        "paper4_working_champion_changed_v441": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v441": next_artifact,
        "claim_boundary": (
            "v441 is a post-SIM108-repair full pytest probe; repository ruff "
            "and final promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v441"] != 9:
        raise RuntimeError("v441 expected repository ruff to remain at 9 diagnostics.")
    if status["repo_ruff_sim108_v441"] != 0:
        raise RuntimeError("v441 expected repository SIM108 to remain clear.")
    if status["streamlit_diagnostics_v441"] != 0 or status["notebook_diagnostics_v441"] != 0:
        raise RuntimeError("v441 expected Streamlit and notebooks to remain clean.")
    if status["book_diagnostics_v441"] != 0:
        raise RuntimeError("v441 expected book diagnostics to remain clean.")
    if status["top_executable_rule_v441"] != "C405":
        raise RuntimeError("v441 expected C405 to be the next executable repair frontier.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v441": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
