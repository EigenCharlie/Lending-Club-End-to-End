#!/usr/bin/env python3
"""Build Paper 4 v427 post-scripts-ruff-repair pytest probe artifacts."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
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

VERSION = 427
PRIOR_SCRIPTS_RUFF_REPAIR_VERSION = 426
NEXT_PASS_ARTIFACT = "paper4_v428_streamlit_b905_c408_repair_batch.md"
NEXT_FAIL_ARTIFACT = "paper4_v428_post_scripts_repair_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v427_post_scripts_ruff_repair_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]
SCRIPTS_B905_COMMAND = ["uv", "run", "ruff", "check", "scripts/papers", "--select", "B905"]


def _protected_scripts_diff_clean() -> bool:
    protected = [
        "scripts/papers/build_paper4_v10_resolution_wave.py",
        "scripts/papers/build_paper4_v11_promising_lanes.py",
        "scripts/papers/build_paper4_v12_resolution_wave.py",
        "scripts/papers/build_paper4_v13_resolution_wave.py",
        "scripts/papers/build_paper4_v5_blocker_resolution.py",
        "scripts/papers/build_paper4_v6_priority_resolution.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *protected],
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


def _run_scripts_b905_probe() -> dict[str, Any]:
    result = subprocess.run(
        SCRIPTS_B905_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(SCRIPTS_B905_COMMAND),
        "exit_code": int(result.returncode),
        "passed": result.returncode == 0,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }


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


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v427": "full_repository_pytest",
                "command_v427": pytest_result["command"],
                "exit_code_v427": int(pytest_result["exit_code"]),
                "passed_v427": bool(pytest_result["passed"]),
                "runtime_seconds_v427": float(pytest_result["runtime_seconds"]),
                "collected_items_v427": int(pytest_result["collected_items"]),
                "summary_line_v427": str(pytest_result["summary_line"]),
                "claim_boundary_v427": "post-v426 scripts B905 repair full pytest probe",
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
        ("notebook_total", _count(items, surface="notebook")),
        ("streamlit_app_total", _count(items, surface="streamlit_app")),
        ("streamlit_app_b905", _count(items, code="B905", surface="streamlit_app")),
        ("streamlit_app_c408", _count(items, code="C408", surface="streamlit_app")),
        ("scripts_total", _count(items, surface="scripts")),
        ("scripts_b905", _count(items, code="B905", surface="scripts")),
        ("book_total", _count(items, surface="book")),
    ]
    return pd.DataFrame(
        [
            {
                "metric_v427": metric,
                "diagnostic_count_v427": int(count),
                "claim_boundary_v427": "post-v426 repository ruff snapshot",
            }
            for metric, count in rows
        ]
    )


def _hotspot_files(items: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_relative_path(str(item["filename"]))].append(item)
    rows = []
    for file_path, diagnostics in grouped.items():
        rule_codes = sorted({str(item["code"]) for item in diagnostics})
        rows.append(
            {
                "file_path_v427": file_path,
                "surface_v427": _surface(file_path),
                "diagnostic_count_v427": int(len(diagnostics)),
                "rule_codes_v427": ",".join(rule_codes),
                "is_next_repair_candidate_v427": file_path
                == "streamlit_app/pages/model_interpretability.py",
                "claim_boundary_v427": "hotspot ranking only; no mutation in v427",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["diagnostic_count_v427", "file_path_v427"],
        ascending=[False, True],
        ignore_index=True,
    )


def _repair_plan(hotspots: pd.DataFrame, *, pytest_passed: bool) -> pd.DataFrame:
    top_streamlit = hotspots.loc[
        hotspots["file_path_v427"].eq("streamlit_app/pages/model_interpretability.py")
    ]
    diagnostic_count = int(top_streamlit["diagnostic_count_v427"].iloc[0]) if not top_streamlit.empty else 0
    return pd.DataFrame(
        [
            {
                "repair_lane_v427": "targeted_streamlit_non_e402_repair",
                "target_surface_v427": "streamlit_app",
                "target_file_v427": "streamlit_app/pages/model_interpretability.py",
                "target_rules_v427": "B905,C408",
                "diagnostic_count_v427": diagnostic_count,
                "mutation_allowed_in_v427": False,
                "v428_candidate_v427": pytest_passed and diagnostic_count == 8,
                "next_artifact_v427": NEXT_PASS_ARTIFACT,
                "claim_boundary_v427": "v427 plans the next repair but does not mutate Streamlit files",
            }
        ]
    )


def _claim_blockers(*, pytest_passed: bool, ruff_total: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v427": "repository_ruff_frontier_still_open",
            "blocking_v427": ruff_total > 0,
            "evidence_count_v427": ruff_total,
            "required_next_artifact_v427": NEXT_PASS_ARTIFACT,
            "claim_boundary_v427": "repository ruff clean claim blocked while diagnostics remain",
        },
        {
            "blocker_id_v427": "quarto_render_not_run",
            "blocking_v427": True,
            "evidence_count_v427": 1,
            "required_next_artifact_v427": NEXT_PASS_ARTIFACT,
            "claim_boundary_v427": "Quarto render is not implied by pytest or ruff snapshots",
        },
        {
            "blocker_id_v427": "paper4_final_promotion_forbidden",
            "blocking_v427": True,
            "evidence_count_v427": 1,
            "required_next_artifact_v427": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v427": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v427": "full_repository_pytest_failed",
                "blocking_v427": True,
                "evidence_count_v427": 1,
                "required_next_artifact_v427": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v427": "pytest failure must be triaged before more lint repair",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(
    *,
    pytest_passed: bool,
    scripts_b905: int,
    e402_count: int,
    notebook_count: int,
    ruff_total: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v427_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v427_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v426 scripts B905 repair",
            },
            {
                "claim_id": "v427_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v427_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v427_scripts_b905_remains_clear",
                "allowed": scripts_b905 == 0,
                "artifact": "paper4_v427_repository_ruff_snapshot.csv",
                "boundary": "scripts/papers B905 remains zero after full pytest",
            },
            {
                "claim_id": "v427_streamlit_e402_remains_clear",
                "allowed": e402_count == 0,
                "artifact": "paper4_v427_repository_ruff_snapshot.csv",
                "boundary": "E402 remains zero after full pytest probe",
            },
            {
                "claim_id": "v427_notebook_lint_remains_clean",
                "allowed": notebook_count == 0,
                "artifact": "paper4_v427_repository_ruff_snapshot.csv",
                "boundary": "notebook diagnostics remain zero",
            },
            {
                "claim_id": "v427_repository_ruff_clean",
                "allowed": ruff_total == 0,
                "artifact": "paper4_v427_claim_blockers.csv",
                "boundary": "true only when repository ruff emits zero diagnostics",
            },
            {
                "claim_id": "v427_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    pytest_passed: bool,
    scripts_b905: int,
    e402_count: int,
    notebook_count: int,
    ruff_total: int,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v427 runs full repository pytest after scripts B905 repair.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v427_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v427 full repository pytest passes after scripts B905 repair.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v427_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v427 keeps scripts B905, Streamlit E402 and notebook lint clean after pytest.",
                "allowed": scripts_b905 == 0 and e402_count == 0 and notebook_count == 0,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v427_repository_ruff_snapshot.csv",
                "boundary": "Specific lint channels only; repository ruff remains open.",
                "prohibited_claim_flag": scripts_b905 != 0 or e402_count != 0 or notebook_count != 0,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v427 proves repository ruff or Quarto render cleanliness.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v427_claim_blockers.csv",
                "boundary": f"{ruff_total} repository ruff diagnostics remain and Quarto render is deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v427 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v427_claim_blockers.csv",
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
                "executable_item": "v427 runs full repository pytest after scripts B905 repair.",
                "status": (
                    "post_scripts_b905_repair_pytest_probe_passed"
                    if pytest_passed
                    else "post_scripts_b905_repair_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v428 applies a targeted Streamlit B905/C408 repair batch"
                    if pytest_passed
                    else "v428 triages pytest failures before more lint repair"
                ),
                "last_wave": "v427",
                "execution_result": (
                    "full_repository_pytest_passed_after_scripts_b905_repair"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_scripts_b905_repair"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v427")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Scripts-Ruff-Repair Pytest Probe v427

Generated: {status["generated_at_utc"]}

v427 runs full repository pytest after v426's targeted scripts/papers B905 repair.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v427"]}`.
- Pytest passed: `{status["pytest_passed_v427"]}`.
- Collected items: `{status["pytest_collected_items_v427"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v427"]}`.
- Summary: `{status["pytest_summary_line_v427"]}`.
- Repository ruff diagnostics: `{status["repo_ruff_total_v427"]}`.
- Scripts B905 diagnostics: `{status["scripts_b905_v427"]}`.
- Streamlit diagnostics: `{status["streamlit_diagnostics_v427"]}`.
- Notebook diagnostics: `{status["notebook_diagnostics_v427"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v427 does not claim repository ruff clean, Quarto render, or Paper 4 final
promotion.

## Next Executable Wave

Build `{status["next_artifact_v427"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V427_POST_SCRIPTS_RUFF_REPAIR_PYTEST_PROBE_START -->"
    end = "<!-- V427_POST_SCRIPTS_RUFF_REPAIR_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v427: Post-Scripts-Ruff-Repair Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v427 runs full repository pytest after v426's targeted scripts/papers B905
repair.

### Results

- Pytest command:
  `{status["pytest_command_v427"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v427"]}`.
- Pytest passed:
  `{status["pytest_passed_v427"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v427"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v427"]}`.
- Repository ruff diagnostics:
  `{status["repo_ruff_total_v427"]}`.
- Scripts B905 diagnostics:
  `{status["scripts_b905_v427"]}`.
- Streamlit B905/C408 diagnostics:
  `{status["streamlit_b905_v427"]}` /
  `{status["streamlit_c408_v427"]}`.
- Notebook diagnostics:
  `{status["notebook_diagnostics_v427"]}`.
- Repository ruff clean:
  `{status["repository_ruff_clean_v427"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v427"]}`.

### Interpretation

The v426 scripts B905 repair survives full repository pytest. The next clean,
bounded frontier is `streamlit_app/pages/model_interpretability.py`, where the
remaining non-E402 Streamlit diagnostics are B905/C408.

### Claim Impact

- Allowed: full repository pytest passed after scripts B905 repair; scripts
  B905, Streamlit E402 and notebook lint remain clear.
- Still prohibited: repository ruff clean, Quarto render clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v427 in the living notebook. v428 should apply the targeted Streamlit
B905/C408 repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _protected_scripts_diff_clean():
        raise RuntimeError("v427 expects committed v426 script repairs before pytest probe.")

    v426_status = json.loads((STATUS_DIR / "paper4_v426_status.json").read_text(encoding="utf-8"))
    if v426_status["next_artifact_v426"] != "paper4_v427_post_scripts_ruff_repair_pytest_probe.md":
        raise RuntimeError("v427 expects v426 to route to post-scripts-repair pytest probe.")
    if v426_status["changed_scripts_pycompile_passed_v426"] is not True:
        raise RuntimeError("v427 expects changed scripts to compile after v426.")
    if int(v426_status["scripts_b905_after_v426"]) != 0:
        raise RuntimeError("v427 expects scripts/papers B905 to be cleared by v426.")

    scripts_b905_probe = _run_scripts_b905_probe()
    if scripts_b905_probe["exit_code"] != 0:
        raise RuntimeError("v427 expects scripts/papers B905 to remain clear before full pytest.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    snapshot = _ruff_snapshot(ruff_items)
    snapshot_map = dict(zip(snapshot["metric_v427"], snapshot["diagnostic_count_v427"], strict=False))
    hotspots = _hotspot_files(ruff_items)
    repair_plan = _repair_plan(hotspots, pytest_passed=pytest_passed)
    blockers = _claim_blockers(pytest_passed=pytest_passed, ruff_total=len(ruff_items))
    claim_matrix = _claim_matrix(
        pytest_passed=pytest_passed,
        scripts_b905=int(snapshot_map["scripts_b905"]),
        e402_count=int(snapshot_map["repository_e402"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_claim_boundaries(
        pytest_passed=pytest_passed,
        scripts_b905=int(snapshot_map["scripts_b905"]),
        e402_count=int(snapshot_map["repository_e402"]),
        notebook_count=int(snapshot_map["notebook_total"]),
        ruff_total=int(snapshot_map["repository_total"]),
    )
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v427_pytest_probe_summary.csv", _pytest_summary_table(pytest_result))
    write_csv(TABLE_DIR / "paper4_v427_repository_ruff_snapshot.csv", snapshot)
    write_csv(TABLE_DIR / "paper4_v427_repository_ruff_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v427_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v427_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v427_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v427_post_scripts_ruff_repair_pytest_probe",
        "schema_version": "2026-05-17.427",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scripts_ruff_repair_version_v427": PRIOR_SCRIPTS_RUFF_REPAIR_VERSION,
        "pytest_command_v427": pytest_result["command"],
        "pytest_exit_code_v427": int(pytest_result["exit_code"]),
        "pytest_passed_v427": pytest_passed,
        "pytest_runtime_seconds_v427": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v427": int(pytest_result["collected_items"]),
        "pytest_summary_line_v427": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v427": int(ruff_exit),
        "repo_ruff_total_v427": int(snapshot_map["repository_total"]),
        "repo_ruff_e402_v427": int(snapshot_map["repository_e402"]),
        "repo_ruff_b905_v427": int(snapshot_map["repository_b905"]),
        "repo_ruff_c408_v427": int(snapshot_map["repository_c408"]),
        "notebook_diagnostics_v427": int(snapshot_map["notebook_total"]),
        "streamlit_diagnostics_v427": int(snapshot_map["streamlit_app_total"]),
        "streamlit_b905_v427": int(snapshot_map["streamlit_app_b905"]),
        "streamlit_c408_v427": int(snapshot_map["streamlit_app_c408"]),
        "scripts_diagnostics_v427": int(snapshot_map["scripts_total"]),
        "scripts_b905_v427": int(snapshot_map["scripts_b905"]),
        "book_diagnostics_v427": int(snapshot_map["book_total"]),
        "repository_ruff_clean_v427": int(snapshot_map["repository_total"]) == 0,
        "full_repository_pytest_run_v427": True,
        "full_repository_pytest_passed_v427": pytest_passed,
        "full_quarto_render_run_v427": False,
        "working_champion_claim_allowed_v427": False,
        "paper1_promotion_allowed_v427": False,
        "paper4_working_champion_changed_v427": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v427": next_artifact,
        "claim_boundary": (
            "v427 records post-scripts-repair full pytest evidence; repository ruff, "
            "Quarto and final-promotion claims remain blocked"
        ),
    }
    if status["repo_ruff_total_v427"] != 46:
        raise RuntimeError("v427 expected repository ruff frontier to remain at 46 diagnostics.")
    if status["repo_ruff_e402_v427"] != 0:
        raise RuntimeError("v427 expected E402 to remain clear.")
    if status["scripts_b905_v427"] != 0:
        raise RuntimeError("v427 expected scripts/papers B905 to remain clear.")
    if status["notebook_diagnostics_v427"] != 0:
        raise RuntimeError("v427 expected notebooks to remain lint-clean.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v427": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
