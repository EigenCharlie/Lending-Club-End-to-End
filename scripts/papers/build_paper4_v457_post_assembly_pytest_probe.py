#!/usr/bin/env python3
"""Build Paper 4 v457 post-assembly pytest probe artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
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

VERSION = 457
PRIOR_MANUSCRIPT_ASSEMBLY_VERSION = 456
NEXT_ARTIFACT = "paper4_v458_post_assembly_render_decision.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v457_post_assembly_pytest_probe.md"
RUFF_COMMAND = ["uv", "run", "ruff", "check", ".", "--output-format", "json"]


def _run_repository_ruff_json() -> tuple[int, list[dict[str, Any]]]:
    result = subprocess.run(RUFF_COMMAND, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "repository ruff probe failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("repository ruff JSON output is not a list")
    return result.returncode, payload


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v457": "post_assembly_full_repository_pytest",
                "command_v457": pytest_result["command"],
                "exit_code_v457": int(pytest_result["exit_code"]),
                "passed_v457": bool(pytest_result["passed"]),
                "runtime_seconds_v457": float(pytest_result["runtime_seconds"]),
                "collected_items_v457": int(pytest_result["collected_items"]),
                "summary_line_v457": str(pytest_result["summary_line"]),
                "claim_boundary_v457": "full pytest after v456 manuscript assembly packet",
            }
        ]
    )


def _validation_summary_table(
    *,
    pytest_result: dict[str, Any],
    ruff_exit: int,
    ruff_items: list[dict[str, Any]],
    v456_status: dict[str, Any],
) -> pd.DataFrame:
    final_promotion_absent = not FORBIDDEN_FINAL_PROMOTION.exists()
    return pd.DataFrame(
        [
            {
                "validation_gate_v457": "post_assembly_full_repository_pytest",
                "observed_status_v457": "pass" if pytest_result["passed"] else "fail",
                "evidence_count_v457": int(pytest_result["collected_items"]),
                "claim_allowed_v457": bool(pytest_result["passed"]),
                "claim_boundary_v457": "post-assembly full test suite",
            },
            {
                "validation_gate_v457": "repository_ruff",
                "observed_status_v457": "pass" if ruff_exit == 0 else "fail",
                "evidence_count_v457": len(ruff_items),
                "claim_allowed_v457": ruff_exit == 0 and len(ruff_items) == 0,
                "claim_boundary_v457": "global repository Ruff snapshot",
            },
            {
                "validation_gate_v457": "manuscript_assembly_packet_exists",
                "observed_status_v457": (
                    "pass"
                    if v456_status["manuscript_assembly_packet_created_v456"]
                    else "fail"
                ),
                "evidence_count_v457": int(v456_status["assembled_section_count_v456"]),
                "claim_allowed_v457": bool(
                    v456_status["manuscript_assembly_packet_created_v456"]
                ),
                "claim_boundary_v457": "inherited from v456 manuscript assembly packet",
            },
            {
                "validation_gate_v457": "paper4_final_promotion_absence",
                "observed_status_v457": "pass" if final_promotion_absent else "fail",
                "evidence_count_v457": 1,
                "claim_allowed_v457": final_promotion_absent,
                "claim_boundary_v457": "final promotion remains forbidden and absent",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v457": "post_assembly_render_decision_not_made",
                "blocking_v457": True,
                "evidence_count_v457": 1,
                "required_next_artifact_v457": NEXT_ARTIFACT,
                "claim_boundary_v457": "decide whether Quarto render refresh is required",
            },
            {
                "blocker_id_v457": "external_dataset_validation_not_run",
                "blocking_v457": True,
                "evidence_count_v457": 0,
                "required_next_artifact_v457": "future_external_validation_protocol",
                "claim_boundary_v457": "do not claim external generalization",
            },
            {
                "blocker_id_v457": "target_venue_not_selected",
                "blocking_v457": True,
                "evidence_count_v457": 0,
                "required_next_artifact_v457": "future_target_venue_packet",
                "claim_boundary_v457": "do not claim submission readiness",
            },
            {
                "blocker_id_v457": "paper4_final_promotion_forbidden",
                "blocking_v457": True,
                "evidence_count_v457": 1,
                "required_next_artifact_v457": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v457": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(*, pytest_passed: bool, repo_ruff_clean: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v457_post_assembly_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v457_pytest_probe_summary.csv",
                "boundary": "full pytest executed after v456 assembly packet",
            },
            {
                "claim_id": "v457_post_assembly_full_repository_pytest_clean",
                "allowed": pytest_passed,
                "artifact": "paper4_v457_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v457_repository_ruff_clean",
                "allowed": repo_ruff_clean,
                "artifact": "paper4_v457_validation_gate_summary.csv",
                "boundary": "repository Ruff emits zero diagnostics",
            },
            {
                "claim_id": "v457_post_assembly_regression_refresh_complete",
                "allowed": pytest_passed and repo_ruff_clean,
                "artifact": "paper4_v457_validation_gate_summary.csv",
                "boundary": "pytest and Ruff clean after assembly packet",
            },
            {
                "claim_id": "v457_submission_ready_or_external_validation",
                "allowed": False,
                "artifact": "paper4_v457_claim_blockers.csv",
                "boundary": "not submitted or externally validated",
            },
            {
                "claim_id": "v457_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(*, pytest_passed: bool, repo_ruff_clean: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v457 full repository pytest passes after manuscript assembly.",
                "allowed": pytest_passed,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v457_pytest_probe_summary.csv"
                ),
                "boundary": "Post-assembly full pytest only.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v457 repository Ruff remains clean after manuscript assembly.",
                "allowed": repo_ruff_clean,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v457_validation_gate_summary.csv"
                ),
                "boundary": "Global Ruff only.",
                "prohibited_claim_flag": not repo_ruff_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v457 completes post-assembly regression refresh for the packet.",
                "allowed": pytest_passed and repo_ruff_clean,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v457_validation_gate_summary.csv"
                ),
                "boundary": "Regression refresh only; not submission or external validation.",
                "prohibited_claim_flag": not (pytest_passed and repo_ruff_clean),
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v457 makes Paper 4 final, submitted, or externally validated.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v457_claim_blockers.csv"
                ),
                "boundary": "Regression refresh does not change validation scope.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v457 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v457_claim_blockers.csv"
                ),
                "boundary": (
                    "No final promotion artifact, champion replacement or deployment gate "
                    "is created."
                ),
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
                "lane": "Publication",
                "executable_item": "v457 runs full pytest after manuscript assembly.",
                "status": (
                    "post_assembly_pytest_passed"
                    if pytest_passed
                    else "post_assembly_pytest_failed"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v458 decides whether post-assembly render refresh is needed",
                "last_wave": "v457",
                "execution_result": (
                    "post_assembly_full_pytest_passed"
                    if pytest_passed
                    else "post_assembly_full_pytest_failed"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v457")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Assembly Pytest Probe v457

Generated: {status["generated_at_utc"]}

v457 reruns full repository pytest after the v456 manuscript assembly packet.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v457"]}`.
- Pytest passed: `{status["pytest_passed_v457"]}`.
- Collected items: `{status["pytest_collected_items_v457"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v457"]}`.
- Summary: `{status["pytest_summary_line_v457"]}`.
- Repository Ruff diagnostics: `{status["repo_ruff_total_v457"]}`.
- Assembly packet sections from v456: `{status["assembled_section_count_from_v456"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v457 proves post-assembly pytest and Ruff cleanliness only. It does not create
external validation, target-venue formatting, submission readiness, champion
replacement, Paper Estrella replacement, or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v457"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V457_POST_ASSEMBLY_PYTEST_PROBE_START -->"
    end = "<!-- V457_POST_ASSEMBLY_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v457: Post-Assembly Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v457 reruns full repository pytest after the v456 manuscript assembly packet.

### Results

- Pytest command:
  `{status["pytest_command_v457"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v457"]}`.
- Pytest passed:
  `{status["pytest_passed_v457"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v457"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v457"]}`.
- Repository Ruff diagnostics:
  `{status["repo_ruff_total_v457"]}`.
- Repository Ruff clean:
  `{status["repository_ruff_clean_v457"]}`.
- Assembly packet from v456:
  `{status["manuscript_assembly_packet_created_from_v456"]}`.
- Post-assembly regression refresh complete:
  `{status["post_assembly_regression_refresh_complete_v457"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v457"]}`.

### Interpretation

The v456 manuscript assembly packet survives a full repository pytest refresh
and repository Ruff remains clean. This supports bounded post-assembly
regression language, but not submission, external validation, champion
replacement or final promotion.

### Claim Impact

- Allowed: post-assembly full pytest run and clean regression refresh when both
  pytest and Ruff are clean.
- Still prohibited: submission readiness, external validation, champion
  replacement and final-promotion claims.

### Quarto Promotion Decision

Keep v457 in the living notebook. v458 should decide whether a Quarto render
refresh is needed after the assembly-only notes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v456_status = json.loads((STATUS_DIR / "paper4_v456_status.json").read_text(encoding="utf-8"))
    if v456_status["next_artifact_v456"] != "paper4_v457_post_assembly_pytest_probe.md":
        raise RuntimeError("v457 expects v456 to route to post-assembly pytest probe.")
    if v456_status["manuscript_assembly_packet_created_v456"] is not True:
        raise RuntimeError("v457 expects v456 manuscript assembly packet to exist.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    repo_ruff_clean = ruff_exit == 0 and len(ruff_items) == 0

    validation_summary = _validation_summary_table(
        pytest_result=pytest_result,
        ruff_exit=ruff_exit,
        ruff_items=ruff_items,
        v456_status=v456_status,
    )
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix(pytest_passed=pytest_passed, repo_ruff_clean=repo_ruff_clean)
    _update_claim_boundaries(pytest_passed=pytest_passed, repo_ruff_clean=repo_ruff_clean)
    _update_backlog(pytest_passed)

    write_csv(
        TABLE_DIR / "paper4_v457_pytest_probe_summary.csv",
        _pytest_summary_table(pytest_result),
    )
    write_csv(TABLE_DIR / "paper4_v457_validation_gate_summary.csv", validation_summary)
    write_csv(TABLE_DIR / "paper4_v457_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v457_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v457_post_assembly_pytest_probe",
        "schema_version": "2026-05-17.457",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manuscript_assembly_version_v457": PRIOR_MANUSCRIPT_ASSEMBLY_VERSION,
        "pytest_command_v457": pytest_result["command"],
        "pytest_exit_code_v457": int(pytest_result["exit_code"]),
        "pytest_passed_v457": pytest_passed,
        "pytest_runtime_seconds_v457": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v457": int(pytest_result["collected_items"]),
        "pytest_summary_line_v457": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v457": int(ruff_exit),
        "repo_ruff_total_v457": len(ruff_items),
        "repository_ruff_clean_v457": repo_ruff_clean,
        "manuscript_assembly_packet_created_from_v456": bool(
            v456_status["manuscript_assembly_packet_created_v456"]
        ),
        "assembled_section_count_from_v456": int(v456_status["assembled_section_count_v456"]),
        "post_assembly_full_repository_pytest_run_v457": True,
        "post_assembly_full_repository_pytest_clean_v457": pytest_passed,
        "post_assembly_regression_refresh_complete_v457": pytest_passed and repo_ruff_clean,
        "post_assembly_quarto_render_run_v457": False,
        "external_validation_complete_v457": False,
        "submission_package_ready_v457": False,
        "working_champion_claim_allowed_v457": False,
        "paper1_promotion_allowed_v457": False,
        "paper4_working_champion_changed_v457": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v457": NEXT_ARTIFACT,
        "claim_boundary": (
            "v457 is a post-assembly pytest and Ruff probe; render decision, "
            "submission, external validation and final promotion remain blocked"
        ),
    }
    if not status["pytest_passed_v457"]:
        raise RuntimeError("v457 expected full pytest to pass.")
    if not status["repository_ruff_clean_v457"]:
        raise RuntimeError("v457 expected repository Ruff to remain clean.")
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v457 must not create final Paper 4 promotion.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v457": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
