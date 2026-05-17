#!/usr/bin/env python3
"""Build Paper 4 v411 post-B018-fig.show pytest probe artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.build_paper4_v403_post_notebook_mutation_pytest_probe import (
    _notebook_diff_clean,
    _run_pytest,
    _run_ruff_json,
)
from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    write_csv,
    write_json,
)

VERSION = 411
PRIOR_B018_PATCH_VERSION = 410
NEXT_PASS_ARTIFACT = "paper4_v412_notebook_f821_execution_context_audit.md"
NEXT_FAIL_ARTIFACT = "paper4_v412_post_b018_fig_show_pytest_failure_triage.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v411_notebook_b018_post_patch_pytest_probe.md"


def _pytest_summary_table(pytest_result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id_v411": "full_repository_pytest",
                "command_v411": pytest_result["command"],
                "exit_code_v411": int(pytest_result["exit_code"]),
                "passed_v411": bool(pytest_result["passed"]),
                "runtime_seconds_v411": float(pytest_result["runtime_seconds"]),
                "collected_items_v411": int(pytest_result["collected_items"]),
                "summary_line_v411": str(pytest_result["summary_line"]),
                "claim_boundary_v411": "post-B018 fig.show pytest probe",
            }
        ]
    )


def _lint_snapshot(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in items)
    rows = []
    for code, count in sorted(counts.items()):
        rows.append(
            {
                "lint_code_v411": code,
                "diagnostic_count_v411": int(count),
                "claim_boundary_v411": "notebook lint remains visible after B018 patch",
            }
        )
    return pd.DataFrame(rows)


def _claim_blockers(*, pytest_passed: bool, counts: Counter[str], global_after: int) -> pd.DataFrame:
    rows = [
        {
            "blocker_id_v411": "f821_execution_context_deferred",
            "blocking_v411": True,
            "evidence_count_v411": int(counts.get("F821", 0)),
            "required_next_artifact_v411": NEXT_PASS_ARTIFACT,
            "claim_boundary_v411": "undefined notebook execution context requires audit",
        },
        {
            "blocker_id_v411": "global_notebook_lint_not_clean",
            "blocking_v411": True,
            "evidence_count_v411": global_after,
            "required_next_artifact_v411": NEXT_PASS_ARTIFACT,
            "claim_boundary_v411": "F821 and style notebook lint remain",
        },
        {
            "blocker_id_v411": "paper4_final_promotion_forbidden",
            "blocking_v411": True,
            "evidence_count_v411": 1,
            "required_next_artifact_v411": "paper4_final_promotion_gate_not_created",
            "claim_boundary_v411": "Paper Estrella replacement and final Paper 4 remain prohibited",
        },
    ]
    if not pytest_passed:
        rows.insert(
            0,
            {
                "blocker_id_v411": "full_repository_pytest_failed",
                "blocking_v411": True,
                "evidence_count_v411": 1,
                "required_next_artifact_v411": NEXT_FAIL_ARTIFACT,
                "claim_boundary_v411": "pytest failure must be triaged before more notebook mutation",
            },
        )
    return pd.DataFrame(rows)


def _claim_matrix(pytest_passed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v411_full_repository_pytest_run",
                "allowed": True,
                "artifact": "paper4_v411_pytest_probe_summary.csv",
                "boundary": "pytest command executed after v410 B018 patch",
            },
            {
                "claim_id": "v411_full_repository_pytest_passed",
                "allowed": pytest_passed,
                "artifact": "paper4_v411_pytest_probe_summary.csv",
                "boundary": "true only when pytest exits 0",
            },
            {
                "claim_id": "v411_notebook_b018_remains_clear",
                "allowed": True,
                "artifact": "paper4_v411_notebook_lint_snapshot.csv",
                "boundary": "B018 count remains 0",
            },
            {
                "claim_id": "v411_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v411_claim_blockers.csv",
                "boundary": "7 notebook diagnostics remain",
            },
            {
                "claim_id": "v411_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries(pytest_passed: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v411 runs full repository pytest after notebook B018 fig.show patch.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v411_pytest_probe_summary.csv",
                "boundary": "Execution evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v411 full repository pytest passes after notebook B018 fig.show patch.",
                "allowed": pytest_passed,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v411_pytest_probe_summary.csv",
                "boundary": "Allowed only if pytest exit code is 0.",
                "prohibited_claim_flag": not pytest_passed,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v411 keeps notebook B018 cleared after pytest probe.",
                "allowed": True,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v411_notebook_lint_snapshot.csv",
                "boundary": "B018 remains at zero; other notebook lint remains.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v411 clears global notebook lint or repository ruff.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v411_claim_blockers.csv",
                "boundary": "7 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v411 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": "reports/paper_material/paper4/tables/paper4_v411_claim_blockers.csv",
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
                "executable_item": "v411 runs full repository pytest after the B018 fig.show notebook patch.",
                "status": (
                    "post_b018_fig_show_pytest_probe_passed"
                    if pytest_passed
                    else "post_b018_fig_show_pytest_probe_failed"
                ),
                "next_artifact": NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT,
                "success_condition": (
                    "v412 audits F821 execution context before remaining style lint"
                    if pytest_passed
                    else "v412 triages pytest failures before more notebook mutation"
                ),
                "last_wave": "v411",
                "execution_result": (
                    "full_repository_pytest_passed_after_b018_fig_show_patch"
                    if pytest_passed
                    else "full_repository_pytest_failed_after_b018_fig_show_patch"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v411")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any], pytest_result: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-B018 Fig.show Pytest Probe v411

Generated: {status["generated_at_utc"]}

v411 runs full repository pytest after v410 clears notebook B018.

## Result

- Command: `{pytest_result["command"]}`.
- Exit code: `{status["pytest_exit_code_v411"]}`.
- Pytest passed: `{status["pytest_passed_v411"]}`.
- Collected items: `{status["pytest_collected_items_v411"]}`.
- Runtime seconds: `{status["pytest_runtime_seconds_v411"]}`.
- Summary: `{status["pytest_summary_line_v411"]}`.
- Notebook diagnostics: `{status["global_notebook_diagnostics_v411"]}`.
- Notebook B018 diagnostics: `{status["global_notebook_b018_v411"]}`.

## Stdout Tail

```text
{pytest_result["stdout_tail"]}
```

## Stderr Tail

```text
{pytest_result["stderr_tail"]}
```

## Required Caveat

v411 does not clear remaining F821 or style notebook lint and does not create
Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v411"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V411_NOTEBOOK_B018_POST_PATCH_PYTEST_PROBE_START -->"
    end = "<!-- V411_NOTEBOOK_B018_POST_PATCH_PYTEST_PROBE_END -->"
    block = f"""
{start}

## Wave v411: Post-B018 Fig.show Pytest Probe

Generated: {status["generated_at_utc"]}

### Objective

v411 runs full repository pytest after v410 clears notebook B018.

### Results

- Pytest command:
  `{status["pytest_command_v411"]}`.
- Pytest exit code:
  `{status["pytest_exit_code_v411"]}`.
- Pytest passed:
  `{status["pytest_passed_v411"]}`.
- Pytest collected items:
  `{status["pytest_collected_items_v411"]}`.
- Pytest summary:
  `{status["pytest_summary_line_v411"]}`.
- Notebook diagnostics:
  `{status["global_notebook_diagnostics_v411"]}`.
- Notebook B018 diagnostics:
  `{status["global_notebook_b018_v411"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v411"]}`.

### Interpretation

The post-B018 validation gate decides whether the next wave can audit the
remaining F821 execution-context diagnostic before style-only notebook lint.

### Claim Impact

- Allowed: full repository pytest was executed after B018 clearance.
- Conditional: pytest pass claim follows the captured exit code.
- Still prohibited: global notebook lint clean, repository ruff clean, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v411 in the living notebook. Route v412 according to the pytest result.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v411 expects clean notebook diff before pytest probe.")

    v410_status = json.loads((STATUS_DIR / "paper4_v410_status.json").read_text(encoding="utf-8"))
    if v410_status["next_artifact_v410"] != "paper4_v411_notebook_b018_post_patch_pytest_probe.md":
        raise RuntimeError("v411 expects v410 to route to post-B018 pytest probe.")

    pytest_result = _run_pytest()
    pytest_passed = bool(pytest_result["passed"])
    current_global = _run_ruff_json()
    counts = Counter(item["code"] for item in current_global)
    lint_snapshot = _lint_snapshot(current_global)
    pytest_summary = _pytest_summary_table(pytest_result)
    blockers = _claim_blockers(
        pytest_passed=pytest_passed,
        counts=counts,
        global_after=len(current_global),
    )
    claim_matrix = _claim_matrix(pytest_passed)
    _update_claim_boundaries(pytest_passed)
    _update_backlog(pytest_passed)

    write_csv(TABLE_DIR / "paper4_v411_pytest_probe_summary.csv", pytest_summary)
    write_csv(TABLE_DIR / "paper4_v411_notebook_lint_snapshot.csv", lint_snapshot)
    write_csv(TABLE_DIR / "paper4_v411_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v411_claim_matrix_delta.csv", claim_matrix)

    next_artifact = NEXT_PASS_ARTIFACT if pytest_passed else NEXT_FAIL_ARTIFACT
    status = {
        "phase": "v411_notebook_b018_post_patch_pytest_probe",
        "schema_version": "2026-05-17.411",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_b018_patch_version_v411": PRIOR_B018_PATCH_VERSION,
        "pytest_command_v411": pytest_result["command"],
        "pytest_exit_code_v411": int(pytest_result["exit_code"]),
        "pytest_passed_v411": pytest_passed,
        "pytest_runtime_seconds_v411": float(pytest_result["runtime_seconds"]),
        "pytest_collected_items_v411": int(pytest_result["collected_items"]),
        "pytest_summary_line_v411": str(pytest_result["summary_line"]),
        "global_notebook_diagnostics_v411": int(len(current_global)),
        "global_notebook_b018_v411": int(counts.get("B018", 0)),
        "global_notebook_f821_v411": int(counts.get("F821", 0)),
        "global_notebook_e741_v411": int(counts.get("E741", 0)),
        "global_notebook_sim108_v411": int(counts.get("SIM108", 0)),
        "global_notebook_e712_v411": int(counts.get("E712", 0)),
        "global_notebook_sim102_v411": int(counts.get("SIM102", 0)),
        "global_ruff_clean_v411": False,
        "full_repository_pytest_run_v411": True,
        "full_repository_pytest_passed_v411": pytest_passed,
        "full_quarto_render_run_v411": False,
        "working_champion_claim_allowed_v411": False,
        "paper1_promotion_allowed_v411": False,
        "paper4_working_champion_changed_v411": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v411": next_artifact,
        "claim_boundary": (
            "v411 records post-B018 pytest evidence; remaining lint and final "
            "promotion claims remain blocked"
        ),
    }
    if status["global_notebook_b018_v411"] != 0:
        raise RuntimeError("v411 expected notebook B018 to remain clear.")
    PROBE_MD.write_text(_probe_markdown(status, pytest_result), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v411": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
