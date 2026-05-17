#!/usr/bin/env python3
"""Build Paper 4 v450 post-full-book-render pytest probe artifacts."""

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

VERSION = 450
PRIOR_FULL_BOOK_RENDER_VERSION = 449
NEXT_ARTIFACT = "paper4_v451_release_readiness_synthesis.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v450_post_full_book_render_pytest_probe.md"
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
                "probe_id_v450": "post_full_book_render_full_repository_pytest",
                "command_v450": pytest_result["command"],
                "exit_code_v450": int(pytest_result["exit_code"]),
                "passed_v450": bool(pytest_result["passed"]),
                "runtime_seconds_v450": float(pytest_result["runtime_seconds"]),
                "collected_items_v450": int(pytest_result["collected_items"]),
                "summary_line_v450": str(pytest_result["summary_line"]),
                "claim_boundary_v450": "full pytest after v448-v449 render guardrails",
            }
        ]
    )


def _validation_summary_table(
    *,
    pytest_result: dict[str, Any],
    ruff_exit: int,
    ruff_items: list[dict[str, Any]],
    v449_status: dict[str, Any],
) -> pd.DataFrame:
    final_promotion_absent = not FORBIDDEN_FINAL_PROMOTION.exists()
    return pd.DataFrame(
        [
            {
                "validation_gate_v450": "full_repository_pytest",
                "observed_status_v450": "pass" if pytest_result["passed"] else "fail",
                "evidence_count_v450": int(pytest_result["collected_items"]),
                "claim_allowed_v450": bool(pytest_result["passed"]),
                "claim_boundary_v450": "post-render full test suite",
            },
            {
                "validation_gate_v450": "repository_ruff",
                "observed_status_v450": "pass" if ruff_exit == 0 else "fail",
                "evidence_count_v450": len(ruff_items),
                "claim_allowed_v450": ruff_exit == 0 and len(ruff_items) == 0,
                "claim_boundary_v450": "global repository Ruff snapshot",
            },
            {
                "validation_gate_v450": "full_book_quarto_render",
                "observed_status_v450": (
                    "pass" if v449_status["full_book_render_clean_v449"] else "fail"
                ),
                "evidence_count_v450": int(v449_status["rendered_page_count_v449"]),
                "claim_allowed_v450": bool(v449_status["full_book_render_clean_v449"]),
                "claim_boundary_v450": "inherited from v449 full-book render",
            },
            {
                "validation_gate_v450": "paper4_final_promotion_absence",
                "observed_status_v450": "pass" if final_promotion_absent else "fail",
                "evidence_count_v450": 1,
                "claim_allowed_v450": final_promotion_absent,
                "claim_boundary_v450": "final promotion remains forbidden and absent",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v450": "release_readiness_synthesis_not_written",
                "blocking_v450": True,
                "evidence_count_v450": 1,
                "required_next_artifact_v450": NEXT_ARTIFACT,
                "claim_boundary_v450": "clean gates need a bounded readiness synthesis",
            },
            {
                "blocker_id_v450": "paper4_final_promotion_forbidden",
                "blocking_v450": True,
                "evidence_count_v450": 1,
                "required_next_artifact_v450": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v450": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix(
    *,
    pytest_passed: bool,
    repo_ruff_clean: bool,
    full_book_clean: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v450_post_render_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v450_pytest_probe_summary.csv",
                "boundary": "full pytest executed after v449 full-book render",
            },
            {
                "claim_id": "v450_post_render_full_repository_pytest_clean",
                "allowed": pytest_passed,
                "artifact": "paper4_v450_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v450_repository_ruff_clean",
                "allowed": repo_ruff_clean,
                "artifact": "paper4_v450_validation_gate_summary.csv",
                "boundary": "repository Ruff emits zero diagnostics",
            },
            {
                "claim_id": "v450_full_book_render_clean_inherited",
                "allowed": full_book_clean,
                "artifact": "paper4_v449_full_book_render_probe_summary.csv",
                "boundary": "inherited from v449, not rerun in v450",
            },
            {
                "claim_id": "v450_release_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v450_claim_blockers.csv",
                "boundary": "readiness synthesis and final promotion remain separate gates",
            },
            {
                "claim_id": "v450_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(
    *,
    pytest_passed: bool,
    repo_ruff_clean: bool,
    full_book_clean: bool,
) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v450 full repository pytest passes after full-book render.",
                "allowed": pytest_passed,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v450_pytest_probe_summary.csv"
                ),
                "boundary": "Post-render full pytest only.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v450 repository Ruff remains clean after full-book render.",
                "allowed": repo_ruff_clean,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v450_validation_gate_summary.csv"
                ),
                "boundary": "Global Ruff only.",
                "prohibited_claim_flag": not repo_ruff_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v450 inherits the v449 full-book render clean gate.",
                "allowed": full_book_clean,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v449_full_book_render_probe_summary.csv"
                ),
                "boundary": "Full-book render evidence was generated in v449.",
                "prohibited_claim_flag": not full_book_clean,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v450 makes Paper 4 release-ready or final.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v450_claim_blockers.csv"
                ),
                "boundary": "Readiness synthesis and promotion remain deferred.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v450 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v450_claim_blockers.csv"
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
                "lane": "Validation",
                "executable_item": "v450 reruns full pytest after the full-book render probe.",
                "status": (
                    "post_full_book_render_pytest_passed"
                    if pytest_passed
                    else "post_full_book_render_pytest_failed"
                ),
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v451 synthesizes clean gates into bounded release-readiness language"
                ),
                "last_wave": "v450",
                "execution_result": (
                    "post_full_book_render_full_pytest_passed"
                    if pytest_passed
                    else "post_full_book_render_full_pytest_failed"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v450")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Full-Book-Render Pytest Probe v450

Generated: {status["generated_at_utc"]}

v450 reruns full repository pytest after v449's clean full-book Quarto render.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v450"]}`.
- Pytest passed: `{status["pytest_passed_v450"]}`.
- Collected items: `{status["pytest_collected_items_v450"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v450"]}`.
- Summary: `{status["pytest_summary_line_v450"]}`.
- Repository Ruff diagnostics: `{status["repo_ruff_total_v450"]}`.
- Full-book render clean from v449: `{status["full_book_render_clean_from_v449"]}`.
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

v450 proves post-render pytest and Ruff cleanliness only. It does not create a
release readiness synthesis, champion replacement, Paper Estrella replacement,
or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v450"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V450_POST_FULL_BOOK_RENDER_PYTEST_PROBE_START -->"
    end = "<!-- V450_POST_FULL_BOOK_RENDER_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v450: Post-Full-Book-Render Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v450 reruns full repository pytest after v449's clean full-book Quarto render.

### Results

- Pytest command:
  `{status["pytest_command_v450"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v450"]}`.
- Pytest passed:
  `{status["pytest_passed_v450"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v450"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v450"]}`.
- Repository Ruff diagnostics:
  `{status["repo_ruff_total_v450"]}`.
- Repository Ruff clean:
  `{status["repository_ruff_clean_v450"]}`.
- Full-book render clean from v449:
  `{status["full_book_render_clean_from_v449"]}`.
- Release readiness synthesis written:
  `{status["release_readiness_synthesis_written_v450"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v450"]}`.

### Interpretation

The v448-v449 Quarto guardrails and documentation survive a full pytest refresh,
and repository Ruff remains clean. The next useful artifact is a bounded
release-readiness synthesis that distinguishes validated gates from prohibited
promotion claims.

### Claim Impact

- Allowed: post-render full pytest passed; repository Ruff remains clean; v449
  full-book render remains valid.
- Still prohibited: release-ready/final Paper 4, champion replacement and final
  promotion claims.

### Quarto Promotion Decision

Keep v450 in the living notebook. v451 should write the bounded readiness
synthesis without creating final promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v449_status = json.loads((STATUS_DIR / "paper4_v449_status.json").read_text(encoding="utf-8"))
    if v449_status["next_artifact_v449"] != "paper4_v450_post_full_book_render_pytest_probe.md":
        raise RuntimeError("v450 expects v449 to route to post-render pytest probe.")
    if v449_status["full_book_render_clean_v449"] is not True:
        raise RuntimeError("v450 expects v449 full-book render to be clean.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    ruff_exit, ruff_items = _run_repository_ruff_json()
    repo_ruff_clean = ruff_exit == 0 and len(ruff_items) == 0
    full_book_clean = bool(v449_status["full_book_render_clean_v449"])

    validation_summary = _validation_summary_table(
        pytest_result=pytest_result,
        ruff_exit=ruff_exit,
        ruff_items=ruff_items,
        v449_status=v449_status,
    )
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix(
        pytest_passed=pytest_passed,
        repo_ruff_clean=repo_ruff_clean,
        full_book_clean=full_book_clean,
    )
    _update_claim_boundaries(
        pytest_passed=pytest_passed,
        repo_ruff_clean=repo_ruff_clean,
        full_book_clean=full_book_clean,
    )
    _update_backlog(pytest_passed)

    write_csv(
        TABLE_DIR / "paper4_v450_pytest_probe_summary.csv",
        _pytest_summary_table(pytest_result),
    )
    write_csv(TABLE_DIR / "paper4_v450_validation_gate_summary.csv", validation_summary)
    write_csv(TABLE_DIR / "paper4_v450_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v450_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v450_post_full_book_render_pytest_probe",
        "schema_version": "2026-05-17.450",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_full_book_render_version_v450": PRIOR_FULL_BOOK_RENDER_VERSION,
        "pytest_command_v450": pytest_result["command"],
        "pytest_exit_code_v450": int(pytest_result["exit_code"]),
        "pytest_passed_v450": pytest_passed,
        "pytest_runtime_seconds_v450": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v450": int(pytest_result["collected_items"]),
        "pytest_summary_line_v450": str(pytest_result["summary_line"]),
        "repo_ruff_exit_code_v450": int(ruff_exit),
        "repo_ruff_total_v450": len(ruff_items),
        "repository_ruff_clean_v450": repo_ruff_clean,
        "full_book_render_clean_from_v449": full_book_clean,
        "full_book_render_page_count_from_v449": int(v449_status["rendered_page_count_v449"]),
        "post_render_full_repository_pytest_run_v450": True,
        "post_render_full_repository_pytest_clean_v450": pytest_passed,
        "release_readiness_synthesis_written_v450": False,
        "working_champion_claim_allowed_v450": False,
        "paper1_promotion_allowed_v450": False,
        "paper4_working_champion_changed_v450": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v450": NEXT_ARTIFACT,
        "claim_boundary": (
            "v450 is a post-full-book-render pytest and Ruff probe; release "
            "readiness synthesis and final promotion remain blocked"
        ),
    }
    if not status["pytest_passed_v450"]:
        raise RuntimeError("v450 expected full pytest to pass.")
    if not status["repository_ruff_clean_v450"]:
        raise RuntimeError("v450 expected repository Ruff to remain clean.")
    if not status["full_book_render_clean_from_v449"]:
        raise RuntimeError("v450 expected v449 full-book render to be clean.")

    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v450": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
