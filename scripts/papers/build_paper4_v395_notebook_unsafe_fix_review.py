#!/usr/bin/env python3
"""Build Paper 4 v395 notebook Ruff-unsafe fix review artifacts."""

from __future__ import annotations

import json
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

VERSION = 395
PRIOR_SAFE_FIX_VERSION = 394
NEXT_VERSION = 396
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_unsafe_fix_roundtrip_application.md"
REVIEW_MD = NOTEBOOK.parent / "paper4_v395_notebook_unsafe_fix_review.md"
DIFF_PREVIEW = NOTEBOOK.parent / "paper4_v395_notebook_unsafe_fix_preview.patch"
UNSAFE_CODES = ["B905", "SIM105"]
RESIDUAL_SELECTED_CODES = ["B905", "SIM105", "SIM115"]
UNSAFE_DIFF_COMMAND = (
    "uv run ruff check notebooks --select B905,SIM105 --fix --unsafe-fixes --diff"
)


def _run_ruff_json(codes: list[str] | None = None) -> list[dict[str, Any]]:
    command = ["uv", "run", "ruff", "check", "notebooks", "--output-format", "json"]
    if codes is not None:
        command[5:5] = ["--select", ",".join(codes)]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff notebook JSON probe failed")
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("ruff notebook JSON output is not a list")
    return payload


def _run_unsafe_diff_preview() -> str:
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "notebooks",
            "--select",
            ",".join(UNSAFE_CODES),
            "--fix",
            "--unsafe-fixes",
            "--diff",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr or "ruff unsafe diff preview failed")
    return result.stdout


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix()


def _notebook_diff_clean() -> bool:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "notebooks"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _candidate_rows(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        fix = item.get("fix") or {}
        location = item.get("location") or {}
        rows.append(
            {
                "candidate_id_v395": f"unsafe_candidate_{idx:03d}",
                "notebook_path_v395": _relative_path(str(item["filename"])),
                "cell_v395": int(item.get("cell") or 0),
                "row_v395": int(location.get("row") or 0),
                "rule_code_v395": str(item["code"]),
                "message_v395": str(item["message"]),
                "fix_applicability_v395": str(fix.get("applicability") or "none"),
                "previewed_by_diff_v395": True,
                "approved_for_roundtrip_application_v395": True,
                "claim_boundary_v395": "review approval only; v395 does not mutate notebooks",
            }
        )
    return pd.DataFrame(rows)


def _decision_rows(residual_items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter(item["code"] for item in residual_items)
    return pd.DataFrame(
        [
            {
                "decision_id_v395": "b905_zip_strict_false",
                "rule_code_v395": "B905",
                "residual_count_v395": int(counts.get("B905", 0)),
                "decision_v395": "approved_for_v396_roundtrip_application",
                "rationale_v395": "Adding strict=False makes the existing zip truncation policy explicit.",
                "mutation_in_v395": False,
                "next_artifact_v395": NEXT_ARTIFACT,
                "claim_boundary_v395": "approval only",
            },
            {
                "decision_id_v395": "sim105_contextlib_suppress",
                "rule_code_v395": "SIM105",
                "residual_count_v395": int(counts.get("SIM105", 0)),
                "decision_v395": "approved_for_v396_roundtrip_application",
                "rationale_v395": (
                    "contextlib.suppress(Exception) preserves the broad Exception pass behavior."
                ),
                "mutation_in_v395": False,
                "next_artifact_v395": NEXT_ARTIFACT,
                "claim_boundary_v395": "approval only",
            },
            {
                "decision_id_v395": "sim115_open_context_manager",
                "rule_code_v395": "SIM115",
                "residual_count_v395": int(counts.get("SIM115", 0)),
                "decision_v395": "manual_refactor_deferred",
                "rationale_v395": "Ruff exposes no automatic fix; keep it out of unsafe application batch.",
                "mutation_in_v395": False,
                "next_artifact_v395": "paper4_v397_notebook_manual_context_manager_patch.md",
                "claim_boundary_v395": "manual rewrite required",
            },
        ]
    )


def _claim_blockers(*, unsafe_candidates: int, global_count: int, residual_selected: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v395": "unsafe_fixes_reviewed_not_applied",
                "blocking_v395": True,
                "evidence_count_v395": unsafe_candidates,
                "required_next_artifact_v395": NEXT_ARTIFACT,
                "claim_boundary_v395": "v395 is review-only",
            },
            {
                "blocker_id_v395": "sim115_manual_refactor_remaining",
                "blocking_v395": True,
                "evidence_count_v395": 1,
                "required_next_artifact_v395": "paper4_v397_notebook_manual_context_manager_patch.md",
                "claim_boundary_v395": "nonfixable selected lint remains",
            },
            {
                "blocker_id_v395": "selected_notebook_lint_still_present",
                "blocking_v395": True,
                "evidence_count_v395": residual_selected,
                "required_next_artifact_v395": NEXT_ARTIFACT,
                "claim_boundary_v395": "review does not reduce lint",
            },
            {
                "blocker_id_v395": "global_notebook_lint_not_clean",
                "blocking_v395": True,
                "evidence_count_v395": global_count,
                "required_next_artifact_v395": "paper4_v398_notebook_import_reorder_policy.md",
                "claim_boundary_v395": "E402 and semantic/manual notebook lint remain",
            },
            {
                "blocker_id_v395": "paper4_final_promotion_forbidden",
                "blocking_v395": True,
                "evidence_count_v395": 1,
                "required_next_artifact_v395": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v395": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v395_unsafe_fix_review_created",
                "allowed": True,
                "artifact": "paper4_v395_notebook_unsafe_fix_review.md",
                "boundary": "review and decision only",
            },
            {
                "claim_id": "v395_unsafe_fix_preview_captured",
                "allowed": True,
                "artifact": "paper4_v395_notebook_unsafe_fix_preview.patch",
                "boundary": "diff preview only",
            },
            {
                "claim_id": "v395_five_unsafe_candidates_approved_for_guarded_application",
                "allowed": True,
                "artifact": "paper4_v395_notebook_unsafe_fix_decision.csv",
                "boundary": "approval for v396, not application in v395",
            },
            {
                "claim_id": "v395_notebooks_mutated",
                "allowed": False,
                "artifact": "paper4_v395_claim_blockers.csv",
                "boundary": "no notebook mutation in review wave",
            },
            {
                "claim_id": "v395_notebook_or_repo_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v395_claim_blockers.csv",
                "boundary": "review does not reduce lint",
            },
            {
                "claim_id": "v395_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v395 reviews 5 Ruff-unsafe notebook fixes without mutation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v395_notebook_unsafe_fix_candidates.csv"
                ),
                "boundary": "Review and approval only; notebooks unchanged.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v395 approves B905/SIM105 fixes for guarded v396 application.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v395_notebook_unsafe_fix_decision.csv"
                ),
                "boundary": "Approval only, not application.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v395 mutates notebooks or reduces notebook lint.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v395_claim_blockers.csv"
                ),
                "boundary": "No mutation; 145 notebook diagnostics remain.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v395 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v395_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": (
                    "v395 reviews Ruff-unsafe notebook fixes with a diff preview before "
                    "any guarded application."
                ),
                "status": "notebook_unsafe_fix_review_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v396 applies the approved B905/SIM105 fixes with roundtrip integrity checks"
                ),
                "last_wave": "v395",
                "execution_result": "five_unsafe_candidates_reviewed_no_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v395")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _review_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Ruff-Unsafe Fix Review v395

Generated: {status["generated_at_utc"]}

v395 reviews the residual Ruff-unsafe notebook fixes after v394's safe-only
application batch.

## Result

- Residual selected diagnostics: `{status["residual_selected_diagnostics_v395"]}`.
- Ruff-unsafe candidates reviewed: `{status["unsafe_fix_candidates_v395"]}`.
- Approved for guarded application: `{status["approved_for_roundtrip_application_v395"]}`.
- Nonfixable SIM115 diagnostics deferred: `{status["nonfixable_selected_rows_v395"]}`.
- Global notebook diagnostics remain: `{status["global_notebook_diagnostics_v395"]}`.
- Notebooks mutated in v395: `{status["notebooks_mutated_v395"]}`.

## Decision

B905 and SIM105 are approved for v396 guarded application because the previewed
changes make existing behavior explicit rather than changing the intended
notebook workflow. SIM115 remains deferred because Ruff provides no automatic
fix.

## Required Caveat

v395 does not mutate notebooks, does not reduce lint, does not run full pytest,
and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v395"]}` and rerun roundtrip integrity checks.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V395_NOTEBOOK_UNSAFE_FIX_REVIEW_START -->"
    end = "<!-- V395_NOTEBOOK_UNSAFE_FIX_REVIEW_END -->"
    block = f"""
{start}

## Wave v395: Notebook Ruff-Unsafe Fix Review

Generated: {status["generated_at_utc"]}

### Objective

v395 reviews the 5 B905/SIM105 fixes that Ruff labels unsafe after v394's
safe-only notebook repair batch.

### Results

- Residual selected diagnostics:
  `{status["residual_selected_diagnostics_v395"]}`.
- Ruff-unsafe candidates reviewed:
  `{status["unsafe_fix_candidates_v395"]}`.
- Approved for guarded application:
  `{status["approved_for_roundtrip_application_v395"]}`.
- Nonfixable SIM115 rows:
  `{status["nonfixable_selected_rows_v395"]}`.
- Global notebook diagnostics:
  `{status["global_notebook_diagnostics_v395"]}`.
- Notebooks mutated:
  `{status["notebooks_mutated_v395"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v395"]}`.

### Interpretation

The remaining selected lint frontier is now split cleanly: 5 reviewed B905/SIM105
fixes can move to a guarded application wave, while the SIM115 finding stays a
manual refactor item.

### Claim Impact

- Allowed: unsafe-fix review, preview and approval for guarded v396 application.
- Still prohibited: notebook mutation in v395, lint cleanliness, champion
  replacement and final promotion claims.

### Quarto Promotion Decision

Keep v395 in the living notebook. v396 should apply the approved unsafe fixes
under the same roundtrip integrity checks used in v394.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")
    if not _notebook_diff_clean():
        raise RuntimeError("v395 expects clean notebook diff because it is review-only.")

    v394_status = json.loads((STATUS_DIR / "paper4_v394_status.json").read_text(encoding="utf-8"))
    if v394_status["next_artifact_v394"] != "paper4_v395_notebook_unsafe_fix_review.md":
        raise RuntimeError("v395 expects v394 to route to notebook unsafe fix review.")

    residual_items = _run_ruff_json(RESIDUAL_SELECTED_CODES)
    global_items = _run_ruff_json()
    unsafe_items = [
        item
        for item in residual_items
        if (item.get("fix") or {}).get("applicability") == "unsafe"
    ]
    candidates = _candidate_rows(unsafe_items)
    decisions = _decision_rows(residual_items)
    blockers = _claim_blockers(
        unsafe_candidates=len(candidates),
        global_count=len(global_items),
        residual_selected=len(residual_items),
    )
    claim_matrix = _claim_matrix()
    diff_preview = _run_unsafe_diff_preview()
    sanitized_diff_preview = "\n".join(line.rstrip() for line in diff_preview.splitlines()) + "\n"
    DIFF_PREVIEW.write_text(sanitized_diff_preview, encoding="utf-8")

    write_csv(TABLE_DIR / "paper4_v395_notebook_unsafe_fix_candidates.csv", candidates)
    write_csv(TABLE_DIR / "paper4_v395_notebook_unsafe_fix_decision.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v395_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v395_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    residual_counts = Counter(item["code"] for item in residual_items)
    global_counts = Counter(item["code"] for item in global_items)
    notebooks_mutated_after_preview = not _notebook_diff_clean()
    status = {
        "phase": "v395_notebook_unsafe_fix_review",
        "schema_version": "2026-05-17.395",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_safe_fix_version_v395": PRIOR_SAFE_FIX_VERSION,
        "unsafe_diff_command_v395": UNSAFE_DIFF_COMMAND,
        "residual_selected_diagnostics_v395": int(len(residual_items)),
        "unsafe_fix_candidates_v395": int(len(candidates)),
        "approved_for_roundtrip_application_v395": int(
            candidates["approved_for_roundtrip_application_v395"].astype(bool).sum()
        ),
        "nonfixable_selected_rows_v395": int(residual_counts.get("SIM115", 0)),
        "b905_candidates_v395": int(residual_counts.get("B905", 0)),
        "sim105_candidates_v395": int(residual_counts.get("SIM105", 0)),
        "global_notebook_diagnostics_v395": int(len(global_items)),
        "global_notebook_e402_v395": int(global_counts.get("E402", 0)),
        "global_notebook_f821_v395": int(global_counts.get("F821", 0)),
        "notebooks_mutated_v395": notebooks_mutated_after_preview,
        "diff_preview_lines_v395": int(len(sanitized_diff_preview.splitlines())),
        "unsafe_fix_preview_artifact_v395": (
            "reports/paper_material/paper4/notes/"
            "paper4_v395_notebook_unsafe_fix_preview.patch"
        ),
        "global_ruff_clean_v395": False,
        "full_repository_pytest_run_v395": False,
        "full_quarto_render_run_v395": False,
        "working_champion_claim_allowed_v395": False,
        "paper1_promotion_allowed_v395": False,
        "paper4_working_champion_changed_v395": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v395": NEXT_ARTIFACT,
        "claim_boundary": (
            "v395 reviews Ruff-unsafe notebook fixes and previews the diff; notebooks "
            "are not mutated and final promotion remains blocked"
        ),
    }
    if notebooks_mutated_after_preview:
        raise RuntimeError("v395 diff preview unexpectedly mutated notebooks.")

    REVIEW_MD.write_text(_review_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v395_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v395": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
