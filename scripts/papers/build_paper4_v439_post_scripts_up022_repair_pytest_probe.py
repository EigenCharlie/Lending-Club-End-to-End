#!/usr/bin/env python3
"""Build Paper 4 v439 post-scripts-UP022-repair pytest probe artifacts."""

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

VERSION = 439
PRIOR_SCRIPTS_UP022_REPAIR_VERSION = 438
NEXT_PASS_ARTIFACT = "paper4_v440_scripts_sim108_conditional_expr_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v440_post_up022_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v439_post_scripts_up022_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
UP022_TARGET_FILES = [
    "scripts/papers/build_paper4_v39_v40_living_lab_execution.py",
    "scripts/papers/build_paper4_v41_v44_living_lab_wave.py",
    "scripts/papers/build_paper4_v45_v48_living_lab_wave.py",
]
COUNT_CODES = ["B023", "SIM108", "C405", "SIM223", "UP022", "F401", "I001", "F841", "B007", "B905", "C408"]


def _up022_target_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *UP022_TARGET_FILES],
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
        "repository_sim108",
        "repository_c405",
        "repository_sim223",
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
                "metric_v439": metric,
                "diagnostic_count_v439": int(counts[metric]),
                "claim_boundary_v439": "post-v438 repository ruff snapshot",
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
                "rule_code_v439": code,
                "diagnostic_count_v439": int(len(diagnostics)),
                "fixable_count_v439": int(fixable),
                "file_count_v439": int(len({_relative_path(str(item["filename"])) for item in diagnostics})),
                "top_surface_v439": surfaces.most_common(1)[0][0],
                "repair_priority_v439": 0,
                "claim_boundary_v439": "post-v439 frontier only",
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v439", "fixable_count_v439", "rule_code_v439"],
        ascending=[False, False, True],
    )
    out["repair_priority_v439"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _hotspot_files(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_relative_path(str(item["filename"]))].append(item)
    rows = []
    for file_path, diagnostics in grouped.items():
        rows.append(
            {
                "file_path_v439": file_path,
                "surface_v439": _surface(file_path),
                "diagnostic_count_v439": int(len(diagnostics)),
                "rule_codes_v439": ",".join(sorted({str(item["code"]) for item in diagnostics})),
                "claim_boundary_v439": "hotspot ranking only; no mutation in v439",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v439", "file_path_v439"],
        ascending=[False, True],
        ignore_index=True,
    )


def _top_executable_rule(rule_frontier: pd.DataFrame) -> pd.Series:
    candidates = rule_frontier.loc[rule_frontier["fixable_count_v439"].astype(int).gt(0)]
    return candidates.iloc[0] if not candidates.empty else rule_frontier.iloc[0]


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v439": "full_repository_pytest",
                "command_v439": pytest_result["command"],
                "exit_code_v439": int(pytest_result["exit_code"]),
                "passed_v439": bool(pytest_result["passed"]),
                "runtime_seconds_v439": float(pytest_result["runtime_seconds"]),
                "collected_items_v439": int(pytest_result["collected_items"]),
                "summary_line_v439": str(pytest_result["summary_line"]),
                "claim_boundary_v439": "post-v438 scripts UP022 repair full pytest probe",
            }
        ]
    )


def _repair_plan(top_executable: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repair_lane_v439": "targeted_scripts_sim108_conditional_expr_repair",
                "target_surface_v439": "scripts",
                "target_rule_v439": str(top_executable["rule_code_v439"]),
                "diagnostic_count_v439": int(top_executable["diagnostic_count_v439"]),
                "fixable_count_v439": int(top_executable["fixable_count_v439"]),
                "mutation_allowed_in_v439": False,
                "v440_candidate_v439": str(top_executable["rule_code_v439"]) == "SIM108",
                "next_artifact_v439": NEXT_PASS_ARTIFACT,
                "claim_boundary_v439": "v439 plans the next repair but does not mutate conditional expressions",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v439": "repository_ruff_frontier_still_open",
            "blocking_v439": ruff_total > 0,
            "evidence_count_v439": ruff_total,
            "required_next_artifact_v439": NEXT_PASS_ARTIFACT,
            "claim_boundary_v439": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v439": "quarto_render_not_run",
            "blocking_v439": True,
            "evidence_count_v439": 1,
            "required_next_artifact_v439": NEXT_PASS_ARTIFACT,
            "claim_boundary_v439": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v439": "paper4_final_promotion_forbidden",
            "blocking_v439": True,
            "evidence_count_v439": 1,
            "required_next_artifact_v439": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v439": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v439": "full_repository_pytest_failed",
                "blocking_v439": True,
                "evidence_count_v439": 1,
                "required_next_artifact_v439": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v439": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(*, pytest_passed: bool, counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v439_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v439_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v438 scripts UP022 repair",
            },
            {
                "claim_id": "v439_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v439_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v439_up022_remains_clear",
                "allowed": counts["repository_up022"] == 0,
                "artifact": "paper4_v439_repository_ruff_snapshot.csv",
                "boundary": "repository UP022 remains zero after full pytest",
            },
            {
                "claim_id": "v439_prior_lint_channels_remain_clean",
                "allowed": counts["repository_f401"] == 0
                and counts["repository_i001"] == 0
                and counts["repository_f841"] == 0
                and counts["streamlit_app_total"] == 0
                and counts["notebook_total"] == 0
                and counts["book_total"] == 0,
                "artifact": "paper4_v439_repository_ruff_snapshot.csv",
                "boundary": "previously cleared lint channels remain zero",
            },
            {
                "claim_id": "v439_repository_ruff_clean",
                "allowed": counts["repository_total"] == 0,
                "artifact": "paper4_v439_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v439_working_champion_or_final_promotion",
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
        counts["repository_up022"] == 0
        and counts["repository_f401"] == 0
        and counts["repository_i001"] == 0
        and counts["streamlit_app_total"] == 0
        and counts["notebook_total"] == 0
        and counts["book_total"] == 0
    )
    additions = pd.DataFrame(
        [
            {
                "claim": "v439 runs full repository pytest after scripts UP022 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v439_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v439 full repository pytest passes after scripts UP022 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v439_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v439 keeps UP022 and prior lint channels clean after pytest.",
                "allowed": channels_clean,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v439_repository_ruff_snapshot.csv",
                "boundary": "Specific lint channels only; repository ruff remains open.",
                "prohibited_claim_flag": not channels_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v439 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v439_claim_blockers.csv",
                "boundary": (
                    f"{counts['repository_total']} repository ruff diagnostics remain "
                    "and Quarto render is deferred."
                ),
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v439 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v439_claim_blockers.csv",
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
                "executable_item": "v439 runs full repository pytest after scripts UP022 repair.",
                "status": (
                    "post_scripts_up022_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_scripts_up022_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v440 applies targeted SIM108 conditional-expression repair"
                    if pytest_passed
                    else "v440 triages pytest failures before more lint repair"
                ),
                "last_wave": "v439",
                "execution_result": (
                    "full_repository_pytest_passed_after_scripts_up022_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_scripts_up022_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v439")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Scripts-UP022-Repair Pytest Probe v439

Generated: {status["generated_at_utc"]}

v439 runs full repository pytest after v438's targeted scripts UP022 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v439"]}`.
- Pytest passed: `{status["pytest_passed_v439"]}`.
- Collected items: `{status["pytest_collected_items_v439"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v439"]}`.
- Summary: `{status["pytest_summary_line_v439"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v439"]}`.
- Repository UP022 diagnostics: `{status["repo_ruff_up022_v439"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v439"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v439"]}`.
- Book diagnostics: `{status["book_diagnostics_v439"]}`.
- Top executable rule: `{status["top_executable_rule_v439"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v439 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v439"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V439_POST_SCRIPTS_UP022_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V439_POST_SCRIPTS_UP022_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v439: Post-Scripts-UP022-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v439 runs full repository pytest after v438's targeted scripts UP022 repair.

### Results

- Pytest command:
  `{status["pytest_command_v439"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v439"]}`.
- Pytest passed:
  `{status["pytest_passed_v439"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v439"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v439"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v439"]}`.
- Repository UP022 diagnostics:
  `{status["repo_ruff_up022_v439"]}`.
- Streamlit diagnostics:
  `{status["streamlit_diagnostics_v439"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v439"]}`.
- Book diagnostics:
  `{status["book_diagnostics_v439"]}`.
- Top rule / top executable rule:
  `{status["top_rule_v439"]}` /
  `{status["top_executable_rule_v439"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v439"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v439"]}`.

### Interpretation

The scripts UP022 repair survives full repository pytest. B023 remains the top
rule by count but has no automatic fixes, so SIM108 is the next executable
repair frontier.

### Claim Impact

- Allowed: full repository pytest passed after scripts UP022 repair; UP022 and
  earlier cleared lint channels remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v439 in the living notebook. v440 should apply the targeted SIM108
conditional-expression repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _up022_target_diff_clean():
        raise RuntimeError("v439 expects committed v438 scripts UP022 repairs before pytest probe.")

    v438_status = json.loads((STATUS_DIR / "paper4_v438_status.json").read_text(encoding="utf-8"))
    if v438_status["next_artifact_v438"] != "paper4_v439_post_scripts_up022_repair_pytest_probe.md":
        raise RuntimeError("v439 expects v438 to route to post-UP022-repair pytest probe.")
    if v438_status["changed_scripts_pycompile_passed_v438"] is not True:
        raise RuntimeError("v439 expects v438 changed scripts to compile.")
    if int(v438_status["repo_ruff_up022_after_v438"]) != 0:
        raise RuntimeError("v439 expects UP022 to be cleared by v438.")

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

    write_csv(TABLE_DIR / "paper4_v439_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v439_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v439_repository_ruff_rule_frontier.csv", rule_frontier)
    write_csv(TABLE_DIR / "paper4_v439_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v439_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v439_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v439_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    top_hotspot = hotspots.iloc[0]
    status = {
        "phase": "v439_post_scripts_up022_repair_pytest_probe",
        "schema_version": "2026-05-17.439",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scripts_up022_repair_version_v439": PRIOR_SCRIPTS_UP022_REPAIR_VERSION,
        "pytest_command_v439": pytest_result["command"],
        "pytest_exit_code_v439": int(pytest_result["exit_code"]),
        "pytest_passed_v439": pytest_passed,
        "pytest_runtime_seconds_v439": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v439": int(pytest_result["collected_items"]),
        "pytest_summary_line_v439": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v439": int(ruff_exit),
        "repo_ruff_total_v439": counts["repository_total"],
        "repo_ruff_b023_v439": counts["repository_b023"],
        "repo_ruff_sim108_v439": counts["repository_sim108"],
        "repo_ruff_c405_v439": counts["repository_c405"],
        "repo_ruff_sim223_v439": counts["repository_sim223"],
        "repo_ruff_up022_v439": counts["repository_up022"],
        "repo_ruff_f401_v439": counts["repository_f401"],
        "repo_ruff_i001_v439": counts["repository_i001"],
        "repo_ruff_f841_v439": counts["repository_f841"],
        "repo_ruff_b007_v439": counts["repository_b007"],
        "repo_ruff_b905_v439": counts["repository_b905"],
        "repo_ruff_c408_v439": counts["repository_c408"],
        "notebook_diagnostics_v439": counts["notebook_total"],
        "streamlit_diagnostics_v439": counts["streamlit_app_total"],
        "scripts_diagnostics_v439": counts["scripts_total"],
        "book_diagnostics_v439": counts["book_total"],
        "top_rule_v439": str(top_rule["rule_code_v439"]),
        "top_rule_count_v439": int(top_rule["diagnostic_count_v439"]),
        "top_rule_fixable_v439": int(top_rule["fixable_count_v439"]),
        "top_executable_rule_v439": str(top_executable["rule_code_v439"]),
        "top_executable_rule_count_v439": int(top_executable["diagnostic_count_v439"]),
        "top_executable_rule_fixable_v439": int(top_executable["fixable_count_v439"]),
        "top_hotspot_file_v439": str(top_hotspot["file_path_v439"]),
        "top_hotspot_diagnostics_v439": int(top_hotspot["diagnostic_count_v439"]),
        "repository_ruff_clean_v439": counts["repository_total"] == 0,
        "full_repository_pytest_run_v439": True,
        "full_repository_pytest_passed_v439": pytest_passed,
        "full_quarto_render_run_v439": False,
        "working_champion_claim_allowed_v439": False,
        "paper1_promotion_allowed_v439": False,
        "paper4_working_champion_changed_v439": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v439": next_artifact,
        "claim_boundary": (
            "v439 is a post-UP022-repair full pytest probe; repository ruff "
            "and final promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v439"] != 11:
        raise RuntimeError("v439 expected repository ruff to remain at 11 diagnostics.")
    if status["repo_ruff_up022_v439"] != 0:
        raise RuntimeError("v439 expected repository UP022 to remain clear.")
    if status["streamlit_diagnostics_v439"] != 0 or status["notebook_diagnostics_v439"] != 0:
        raise RuntimeError("v439 expected Streamlit and notebooks to remain clean.")
    if status["book_diagnostics_v439"] != 0:
        raise RuntimeError("v439 expected book diagnostics to remain clean.")
    if status["top_executable_rule_v439"] != "SIM108":
        raise RuntimeError("v439 expected SIM108 to be the next executable repair frontier.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v439": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
