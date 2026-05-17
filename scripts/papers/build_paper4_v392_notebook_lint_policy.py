#!/usr/bin/env python3
"""Build Paper 4 v392 notebook lint policy artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    write_csv,
    write_json,
)

VERSION = 392
PRIOR_TARGETED_LINT_VERSION = 391
NEXT_VERSION = 393
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_notebook_lint_dry_run_manifest.csv"
POLICY_MD = NOTEBOOK.parent / "paper4_v392_notebook_lint_policy.md"
SELECTED_POLICY = "dry_run_first_no_bulk_notebook_mutation"


def _surface_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "surface_v392": "notebooks",
                "diagnostic_count_v392": 158,
                "fixable_count_v392": 22,
                "dominant_rule_v392": "E402",
                "dominant_rule_count_v392": 119,
                "mutation_policy_v392": SELECTED_POLICY,
                "claim_boundary_v392": "notebooks are not mutated in v392",
            },
            {
                "surface_v392": "streamlit_app",
                "diagnostic_count_v392": 58,
                "fixable_count_v392": 0,
                "dominant_rule_v392": "E402",
                "dominant_rule_count_v392": 50,
                "mutation_policy_v392": "separate_page_by_page_repair",
                "claim_boundary_v392": "streamlit lint remains a later batch",
            },
            {
                "surface_v392": "scripts",
                "diagnostic_count_v392": 44,
                "fixable_count_v392": 44,
                "dominant_rule_v392": "B023",
                "dominant_rule_count_v392": 7,
                "mutation_policy_v392": "small_script_batches_with_tests",
                "claim_boundary_v392": "legacy scripts need scoped validation",
            },
            {
                "surface_v392": "book",
                "diagnostic_count_v392": 2,
                "fixable_count_v392": 2,
                "dominant_rule_v392": "I001",
                "dominant_rule_count_v392": 1,
                "mutation_policy_v392": "safe_helper_cleanup_later",
                "claim_boundary_v392": "book helper cleanup not applied in v392",
            },
        ]
    )


def _notebook_rule_frontier() -> pd.DataFrame:
    rows = [
        ("E402", 119, 0, "module-import-not-at-top", "high_risk_bulk_cell_reorder"),
        ("B018", 10, 0, "useless-expression", "manual_review"),
        ("F541", 4, 4, "f-string-without-placeholders", "safe_text_fix_after_dry_run"),
        ("B007", 3, 0, "unused-loop-control-variable", "manual_review"),
        ("SIM105", 3, 3, "try-except-pass", "safe_but_semantic_review_needed"),
        ("I001", 3, 3, "unsorted-imports", "safe_after_notebook_roundtrip_check"),
        ("W293", 3, 3, "blank-line-whitespace", "safe_after_notebook_roundtrip_check"),
        ("SIM108", 2, 2, "if-else-block-can-be-ternary", "style_only_defer"),
        ("F401", 2, 2, "unused-import", "safe_after_dependency_check"),
        ("B905", 2, 2, "zip-without-strict", "safe_after_semantic_check"),
        ("E712", 2, 0, "true-false-comparison", "manual_review"),
        ("F821", 1, 0, "undefined-name", "must_fix_only_with_execution_context"),
        ("E741", 1, 0, "ambiguous-variable-name", "manual_review"),
        ("SIM115", 1, 1, "open-file-context-manager", "safe_after_context_check"),
        ("SIM102", 1, 1, "nested-if", "style_only_defer"),
        ("UP017", 1, 1, "datetime-timezone-utc", "safe_after_import_check"),
    ]
    return pd.DataFrame(
        [
            {
                "rule_code_v392": code,
                "notebook_diagnostic_count_v392": count,
                "notebook_fixable_count_v392": fixable,
                "rule_family_v392": family,
                "policy_decision_v392": decision,
                "claim_boundary_v392": "notebook lint frontier only; no mutation in v392",
            }
            for code, count, fixable, family, decision in rows
        ]
    )


def _policy_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_id_v392": "bulk_ruff_fix_notebooks_now",
                "decision_v392": "rejected",
                "selected_v392": False,
                "evidence_count_v392": 158,
                "risk_v392": "could rewrite executed notebooks and obscure provenance",
                "next_action_v392": "do_not_apply",
                "claim_boundary_v392": "notebooks stay untouched in v392",
            },
            {
                "policy_id_v392": SELECTED_POLICY,
                "decision_v392": "selected",
                "selected_v392": True,
                "evidence_count_v392": 158,
                "risk_v392": "adds one planning wave before mutation",
                "next_action_v392": NEXT_ARTIFACT,
                "claim_boundary_v392": "dry-run manifest before any notebook rewrite",
            },
            {
                "policy_id_v392": "exclude_notebooks_from_global_ruff",
                "decision_v392": "rejected",
                "selected_v392": False,
                "evidence_count_v392": 158,
                "risk_v392": "hides lint debt instead of resolving or documenting it",
                "next_action_v392": "do_not_apply",
                "claim_boundary_v392": "global lint frontier remains visible",
            },
            {
                "policy_id_v392": "repair_py_surfaces_first",
                "decision_v392": "deferred",
                "selected_v392": False,
                "evidence_count_v392": 104,
                "risk_v392": "useful but does not answer notebook mutation policy",
                "next_action_v392": "paper4_v394_streamlit_page_lint_repair.md",
                "claim_boundary_v392": "separate workstream",
            },
        ]
    )


def _dry_run_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_id_v392": "capture_current_notebook_ruff_json",
                "priority_v392": 1,
                "command_v392": "uv run ruff check notebooks --output-format json",
                "expected_output_v392": "stable notebook diagnostic manifest",
                "success_condition_v392": "diagnostics parsed by notebook file/rule/cell",
            },
            {
                "step_id_v392": "classify_safe_vs_semantic_notebook_fixes",
                "priority_v392": 2,
                "command_v392": "python scripts/papers/build_paper4_v393_notebook_lint_dry_run_manifest.py",
                "expected_output_v392": NEXT_ARTIFACT,
                "success_condition_v392": "every fix is labeled safe, semantic-review or blocked",
            },
            {
                "step_id_v392": "no_mutation_roundtrip_guard",
                "priority_v392": 3,
                "command_v392": "git diff -- notebooks",
                "expected_output_v392": "no notebook diff in dry-run wave",
                "success_condition_v392": "v393 produces manifest without notebook mutation",
            },
            {
                "step_id_v392": "post_policy_claim_guardrail",
                "priority_v392": 4,
                "command_v392": "uv run pytest -q tests/test_docs/test_paper4_living_lab_guardrails.py -k v393",
                "expected_output_v392": "v393 guardrail passes",
                "success_condition_v392": "dry-run policy is test-covered before repair",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v392": "notebook_bulk_mutation_not_allowed_yet",
                "blocking_v392": True,
                "evidence_count_v392": 158,
                "required_next_artifact_v392": NEXT_ARTIFACT,
                "claim_boundary_v392": "dry-run manifest required before notebook mutation",
            },
            {
                "blocker_id_v392": "global_ruff_not_clean",
                "blocking_v392": True,
                "evidence_count_v392": 262,
                "required_next_artifact_v392": NEXT_ARTIFACT,
                "claim_boundary_v392": "global ruff remains blocked",
            },
            {
                "blocker_id_v392": "notebook_f821_requires_execution_context",
                "blocking_v392": True,
                "evidence_count_v392": 1,
                "required_next_artifact_v392": NEXT_ARTIFACT,
                "claim_boundary_v392": "undefined-name notebook issue cannot be auto-fixed blindly",
            },
            {
                "blocker_id_v392": "paper4_final_promotion_forbidden",
                "blocking_v392": True,
                "evidence_count_v392": 1,
                "required_next_artifact_v392": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v392": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v392_notebook_lint_policy_created",
                "allowed": True,
                "artifact": "paper4_v392_notebook_lint_policy.md",
                "boundary": "policy only",
            },
            {
                "claim_id": "v392_notebook_surface_classified",
                "allowed": True,
                "artifact": "paper4_v392_notebook_rule_frontier.csv",
                "boundary": "158 notebook diagnostics classified",
            },
            {
                "claim_id": "v392_dry_run_first_policy_selected",
                "allowed": True,
                "artifact": "paper4_v392_notebook_lint_policy_decision.csv",
                "boundary": "no notebook mutation in v392",
            },
            {
                "claim_id": "v392_notebooks_repaired",
                "allowed": False,
                "artifact": "paper4_v392_claim_blockers.csv",
                "boundary": "v392 applies no notebook rewrites",
            },
            {
                "claim_id": "v392_global_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v392_claim_blockers.csv",
                "boundary": "262 ruff diagnostics remain",
            },
            {
                "claim_id": "v392_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v392 selects a dry-run-first policy for notebook lint repair.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v392_notebook_lint_policy_decision.csv"
                ),
                "boundary": "Policy only; no notebook mutation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v392 classifies 158 notebook lint diagnostics.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v392_notebook_rule_frontier.csv"
                ),
                "boundary": "Classification only; repair deferred.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v392 repairs notebooks or makes global ruff clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v392_claim_blockers.csv"
                ),
                "boundary": "No notebook rewrites; global ruff still fails.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v392 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v392_claim_blockers.csv"
                ),
                "boundary": "No final promotion artifact, champion replacement or deployment gate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["claim"].isin(additions["claim"])].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Validation",
                "executable_item": (
                    "v392 selects a dry-run-first notebook lint policy before any "
                    "bulk notebook mutation."
                ),
                "status": "notebook_lint_policy_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v393 creates a no-mutation notebook lint dry-run manifest"
                ),
                "last_wave": "v392",
                "execution_result": "notebook_lint_policy_dry_run_first_selected",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v392")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _policy_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Notebook Lint Policy v392

Generated: {status["generated_at_utc"]}

v392 turns the v391 lint reduction into a notebook-specific mutation policy.

## Decision

Selected policy: `{status["selected_policy_v392"]}`.

- Notebook diagnostics: `{status["notebook_diagnostics_v392"]}`.
- Notebook fixable diagnostics: `{status["notebook_fixable_diagnostics_v392"]}`.
- Dominant notebook rule: `{status["notebook_top_rule_v392"]}`
  (`{status["notebook_top_rule_count_v392"]}` findings).
- Notebook bulk mutation applied: `{status["notebook_bulk_mutation_applied_v392"]}`.

## Required Caveat

v392 does not repair notebooks, does not hide notebooks from global ruff, does
not claim global ruff cleanliness, and does not create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v392"]}` as a no-mutation dry-run manifest.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V392_NOTEBOOK_LINT_POLICY_START -->"
    end = "<!-- V392_NOTEBOOK_LINT_POLICY_END -->"
    block = f"""
{start}

## Wave v392: Notebook Lint Policy

Generated: {status["generated_at_utc"]}

### Objective

v392 defines how to approach notebook lint after v391 reduced the Paper 4
guardrail lint subset.

### Results

- Notebook diagnostics:
  `{status["notebook_diagnostics_v392"]}`.
- Notebook fixable diagnostics:
  `{status["notebook_fixable_diagnostics_v392"]}`.
- Selected policy:
  `{status["selected_policy_v392"]}`.
- Notebook bulk mutation applied:
  `{status["notebook_bulk_mutation_applied_v392"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v392"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v392"]}`.

### Interpretation

The notebook lint surface is too large and provenance-sensitive for blind bulk
rewrite. The next wave should produce a dry-run manifest that classifies each
notebook diagnostic before any notebook files are changed.

### Claim Impact

- Allowed: notebook lint policy and frontier classification.
- Still prohibited: notebook repaired, global ruff clean, champion replacement
  and final promotion claims.

### Quarto Promotion Decision

Keep v392 in the living notebook. v393 should build the no-mutation dry-run
manifest.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v391_status = json.loads((STATUS_DIR / "paper4_v391_status.json").read_text(encoding="utf-8"))
    if v391_status["next_artifact_v391"] != "paper4_v392_notebook_lint_policy.md":
        raise RuntimeError("v392 expects v391 to route to notebook lint policy.")

    surface = _surface_summary()
    notebook_rules = _notebook_rule_frontier()
    decisions = _policy_decision()
    dry_run = _dry_run_plan()
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v392_lint_surface_summary.csv", surface)
    write_csv(TABLE_DIR / "paper4_v392_notebook_rule_frontier.csv", notebook_rules)
    write_csv(TABLE_DIR / "paper4_v392_notebook_lint_policy_decision.csv", decisions)
    write_csv(TABLE_DIR / "paper4_v392_notebook_lint_dry_run_plan.csv", dry_run)
    write_csv(TABLE_DIR / "paper4_v392_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v392_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v392_notebook_lint_policy",
        "schema_version": "2026-05-17.392",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_targeted_lint_version_v392": PRIOR_TARGETED_LINT_VERSION,
        "surface_rows_v392": int(len(surface)),
        "notebook_rule_rows_v392": int(len(notebook_rules)),
        "policy_decision_rows_v392": int(len(decisions)),
        "dry_run_plan_rows_v392": int(len(dry_run)),
        "claim_blocker_rows_v392": int(len(blockers)),
        "claim_matrix_rows_v392": int(len(claim_matrix)),
        "selected_policy_v392": SELECTED_POLICY,
        "notebook_diagnostics_v392": 158,
        "notebook_fixable_diagnostics_v392": 22,
        "notebook_top_rule_v392": "E402",
        "notebook_top_rule_count_v392": 119,
        "global_ruff_errors_v392": 262,
        "global_ruff_fixable_errors_v392": 68,
        "global_ruff_clean_v392": False,
        "notebook_bulk_mutation_applied_v392": False,
        "notebook_exclusion_from_global_ruff_allowed_v392": False,
        "dry_run_manifest_required_v392": True,
        "full_repository_pytest_rerun_v392": False,
        "full_quarto_render_run_v392": False,
        "working_champion_claim_allowed_v392": False,
        "paper1_promotion_allowed_v392": False,
        "paper4_working_champion_changed_v392": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "policy_artifact_v392": (
            "reports/paper_material/paper4/notes/paper4_v392_notebook_lint_policy.md"
        ),
        "next_artifact_v392": NEXT_ARTIFACT,
        "claim_boundary": (
            "v392 selects dry-run-first notebook lint governance; notebooks are not "
            "mutated and global ruff remains blocked"
        ),
    }
    POLICY_MD.write_text(_policy_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v392_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v392": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
