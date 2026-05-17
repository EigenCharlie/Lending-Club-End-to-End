#!/usr/bin/env python3
"""Build Paper 4 v390 repository lint frontier artifacts."""

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

VERSION = 390
PRIOR_FULL_PYTEST_VERSION = 389
NEXT_VERSION = 391
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_targeted_lint_repair_batch.md"
LINT_MD = NOTEBOOK.parent / "paper4_v390_repository_lint_frontier.md"
RUFF_COMMAND = "uv run ruff check ."

RULE_COUNTS = [
    ("E402", 169, "module-import-not-at-top", "mostly notebooks and Streamlit pages"),
    ("F841", 26, "unused-local-variable", "mostly historical Paper 4 guardrail config reads"),
    ("B905", 16, "zip-without-strict", "mostly Streamlit page loops"),
    ("B007", 11, "unused-loop-control-variable", "notebook/script cleanup"),
    ("B018", 10, "useless-expression", "notebook cell cleanup"),
    ("I001", 8, "unsorted-imports", "helper and script imports"),
    ("B023", 7, "function-uses-loop-variable", "late-binding loop closure risks"),
    ("F541", 5, "f-string-without-placeholders", "test/notebook cleanup"),
    ("C408", 5, "unnecessary-dict-call", "Streamlit plotly style cleanup"),
    ("SIM108", 4, "if-else-block-can-be-ternary", "notebook style cleanup"),
]

HOTSPOT_FILES = [
    ("streamlit_app/pages/model_interpretability.py", 22, "streamlit_page"),
    ("notebooks/03_pd_modeling.ipynb", 20, "notebook"),
    ("tests/test_docs/test_paper4_living_lab_guardrails.py", 20, "paper4_guardrail_test"),
    ("notebooks/06_survival_analysis.ipynb", 18, "notebook"),
    ("notebooks/13_model_explainability.ipynb", 18, "notebook"),
    ("notebooks/04_conformal_prediction.ipynb", 16, "notebook"),
    ("notebooks/05_time_series_forecasting.ipynb", 16, "notebook"),
    ("notebooks/08_portfolio_optimization.ipynb", 15, "notebook"),
    ("notebooks/09_end_to_end_pipeline.ipynb", 15, "notebook"),
    ("streamlit_app/pages/data_story.py", 13, "streamlit_page"),
    ("streamlit_app/pages/portfolio_optimizer.py", 13, "streamlit_page"),
    ("notebooks/02_feature_engineering.ipynb", 11, "notebook"),
]


def _rule_frontier() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_code_v390": code,
                "error_count_v390": count,
                "rule_family_v390": family,
                "dominant_surface_v390": surface,
                "repair_priority_v390": idx,
                "claim_boundary_v390": "lint frontier only; no bulk repair applied",
            }
            for idx, (code, count, family, surface) in enumerate(RULE_COUNTS, start=1)
        ]
    )


def _hotspot_files() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file_path_v390": path,
                "error_count_v390": count,
                "surface_v390": surface,
                "repair_priority_v390": idx,
                "claim_boundary_v390": "hotspot classification only",
            }
            for idx, (path, count, surface) in enumerate(HOTSPOT_FILES, start=1)
        ]
    )


def _repair_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "repair_lane_v390": "notebook_import_order_and_cell_hygiene",
                "priority_v390": 1,
                "target_rules_v390": "E402,B018,B007,SIM108,F541,F821,E741",
                "estimated_error_count_v390": 180,
                "recommended_mode_v390": "automated_notebook_rewrite_plus_spot_review",
                "next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "notebooks need explicit rewrite policy before mutation",
            },
            {
                "repair_lane_v390": "paper4_guardrail_unused_config_reads",
                "priority_v390": 2,
                "target_rules_v390": "F841,F541",
                "estimated_error_count_v390": 21,
                "recommended_mode_v390": "targeted_apply_patch_in_test_file",
                "next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "safe local cleanup, but not done in v390",
            },
            {
                "repair_lane_v390": "streamlit_page_import_and_style_cleanup",
                "priority_v390": 3,
                "target_rules_v390": "E402,C408,B905",
                "estimated_error_count_v390": 58,
                "recommended_mode_v390": "page_by_page_import_reorder_and_style_patch",
                "next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "avoid broad UI churn in same wave",
            },
            {
                "repair_lane_v390": "legacy_script_lint_cleanup",
                "priority_v390": 4,
                "target_rules_v390": "I001,F401,B023,UP022,SIM115",
                "estimated_error_count_v390": 23,
                "recommended_mode_v390": "small_batches_with_targeted_tests",
                "next_artifact_v390": "paper4_v392_legacy_script_lint_batch.md",
                "claim_boundary_v390": "historical scripts require scoped validation",
            },
        ]
    )


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v390": "global_ruff_not_clean",
                "blocking_v390": True,
                "evidence_count_v390": 282,
                "required_next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "ruff global clean claim blocked by 282 diagnostics",
            },
            {
                "blocker_id_v390": "notebook_lint_surface_unrepaired",
                "blocking_v390": True,
                "evidence_count_v390": 180,
                "required_next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "bulk notebook lint requires explicit rewrite policy",
            },
            {
                "blocker_id_v390": "streamlit_page_lint_surface_unrepaired",
                "blocking_v390": True,
                "evidence_count_v390": 58,
                "required_next_artifact_v390": NEXT_ARTIFACT,
                "claim_boundary_v390": "page import/style cleanup not applied in v390",
            },
            {
                "blocker_id_v390": "paper4_final_promotion_forbidden",
                "blocking_v390": True,
                "evidence_count_v390": 1,
                "required_next_artifact_v390": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v390": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v390_repository_lint_frontier_created",
                "allowed": True,
                "artifact": "paper4_v390_repository_lint_frontier.csv",
                "boundary": "diagnostic frontier only",
            },
            {
                "claim_id": "v390_full_repository_pytest_remains_clean",
                "allowed": True,
                "artifact": "paper4_v389_full_repository_pytest_probe.csv",
                "boundary": "pytest clean inherited from v389",
            },
            {
                "claim_id": "v390_lint_repair_plan_created",
                "allowed": True,
                "artifact": "paper4_v390_lint_repair_plan.csv",
                "boundary": "plan only; no lint repair applied",
            },
            {
                "claim_id": "v390_global_ruff_clean",
                "allowed": False,
                "artifact": "paper4_v390_claim_blockers.csv",
                "boundary": "282 ruff diagnostics remain",
            },
            {
                "claim_id": "v390_full_quarto_render_success",
                "allowed": False,
                "artifact": "paper4_v390_claim_blockers.csv",
                "boundary": "full Quarto render not run",
            },
            {
                "claim_id": "v390_working_champion_or_final_promotion",
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
                "claim": "v390 classifies the repository-wide ruff frontier.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v390_repository_lint_frontier.csv"
                ),
                "boundary": "Diagnostic only: 282 ruff findings remain.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v390 preserves the v389 full pytest clean result.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v389_full_repository_pytest_probe.csv"
                ),
                "boundary": "Pytest clean does not imply lint clean.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v390 proves global ruff or full Quarto render is clean.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v390_claim_blockers.csv"
                ),
                "boundary": "Global ruff failed and full render was not run.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v390 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v390_claim_blockers.csv"
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
                    "v390 classifies repository-wide ruff diagnostics and selects a "
                    "targeted lint repair batch."
                ),
                "status": "repository_lint_frontier_classified",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v391 reduces lint errors in the safest surfaces without breaking full pytest"
                ),
                "last_wave": "v390",
                "execution_result": "ruff_global_failed_282_diagnostics_classified",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v390")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _lint_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Repository Lint Frontier v390

Generated: {status["generated_at_utc"]}

v390 probes `uv run ruff check .` after the full pytest suite became clean in
v389.

## Result

- Ruff command: `{status["ruff_command_v390"]}`.
- Ruff status: `{status["ruff_status_v390"]}`.
- Total diagnostics: `{status["ruff_total_errors_v390"]}`.
- Fixable diagnostics reported by ruff: `{status["ruff_fixable_errors_v390"]}`.
- Top rule: `{status["top_rule_v390"]}` with `{status["top_rule_count_v390"]}` findings.
- Top file: `{status["top_file_v390"]}` with `{status["top_file_error_count_v390"]}` findings.

## Required Caveat

v390 is a lint frontier only. It does not repair the 282 diagnostics, does not
claim global ruff cleanliness, does not run full Quarto render, and does not
create Paper 4 final promotion.

## Next Executable Wave

Build `{status["next_artifact_v390"]}` to start a targeted lint repair batch.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V390_REPOSITORY_LINT_FRONTIER_START -->"
    end = "<!-- V390_REPOSITORY_LINT_FRONTIER_END -->"
    block = f"""
{start}

## Wave v390: Repository Lint Frontier

Generated: {status["generated_at_utc"]}

### Objective

v390 probes `uv run ruff check .` after the v389 full pytest success and
classifies the remaining lint frontier.

### Results

- Ruff status:
  `{status["ruff_status_v390"]}`.
- Ruff total diagnostics:
  `{status["ruff_total_errors_v390"]}`.
- Ruff fixable diagnostics:
  `{status["ruff_fixable_errors_v390"]}`.
- Top rule:
  `{status["top_rule_v390"]}`.
- Top rule count:
  `{status["top_rule_count_v390"]}`.
- Top file:
  `{status["top_file_v390"]}`.
- Full pytest clean inherited from v389:
  `{status["full_repository_pytest_clean_v390"]}`.
- Global ruff clean:
  `{status["global_ruff_clean_v390"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v390"]}`.

### Interpretation

The repo is pytest-clean but not lint-clean. The lint surface is broad and
historical: notebooks dominate import-order diagnostics, while Streamlit pages
and the long Paper 4 guardrail file contain smaller targeted cleanup batches.

### Claim Impact

- Allowed: lint frontier classified and full pytest cleanliness preserved.
- Still prohibited: global ruff clean, full Quarto render, champion replacement
  and final promotion claims.

### Quarto Promotion Decision

Keep v390 in the living notebook. v391 should begin a targeted lint repair batch.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v389_status = json.loads((STATUS_DIR / "paper4_v389_status.json").read_text(encoding="utf-8"))
    if v389_status["next_artifact_v389"] != "paper4_v390_repository_lint_frontier.md":
        raise RuntimeError("v390 expects v389 to route to repository lint frontier.")

    rules = _rule_frontier()
    hotspots = _hotspot_files()
    repair_plan = _repair_plan()
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v390_repository_lint_frontier.csv", rules)
    write_csv(TABLE_DIR / "paper4_v390_lint_hotspot_files.csv", hotspots)
    write_csv(TABLE_DIR / "paper4_v390_lint_repair_plan.csv", repair_plan)
    write_csv(TABLE_DIR / "paper4_v390_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v390_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    top_rule = rules.sort_values("error_count_v390", ascending=False).iloc[0]
    top_file = hotspots.sort_values("error_count_v390", ascending=False).iloc[0]
    status = {
        "phase": "v390_repository_lint_frontier",
        "schema_version": "2026-05-17.390",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_full_pytest_version_v390": PRIOR_FULL_PYTEST_VERSION,
        "ruff_command_v390": RUFF_COMMAND,
        "ruff_status_v390": "fail",
        "ruff_total_errors_v390": 282,
        "ruff_fixable_errors_v390": 88,
        "rule_frontier_rows_v390": int(len(rules)),
        "hotspot_file_rows_v390": int(len(hotspots)),
        "repair_plan_rows_v390": int(len(repair_plan)),
        "claim_blocker_rows_v390": int(len(blockers)),
        "claim_matrix_rows_v390": int(len(claim_matrix)),
        "top_rule_v390": str(top_rule["rule_code_v390"]),
        "top_rule_count_v390": int(top_rule["error_count_v390"]),
        "top_file_v390": str(top_file["file_path_v390"]),
        "top_file_error_count_v390": int(top_file["error_count_v390"]),
        "notebook_surface_dominates_v390": True,
        "streamlit_surface_open_v390": True,
        "paper4_guardrail_surface_open_v390": True,
        "global_ruff_clean_v390": False,
        "full_repository_pytest_clean_v390": bool(v389_status["post_repair_full_pytest_clean_v389"]),
        "full_quarto_render_run_v390": False,
        "full_quarto_render_clean_v390": False,
        "working_champion_claim_allowed_v390": False,
        "paper1_promotion_allowed_v390": False,
        "paper4_working_champion_changed_v390": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "lint_artifact_v390": (
            "reports/paper_material/paper4/tables/"
            "paper4_v390_repository_lint_frontier.csv"
        ),
        "next_artifact_v390": NEXT_ARTIFACT,
        "claim_boundary": (
            "v390 classifies a 282-diagnostic ruff frontier; pytest is clean, "
            "but global lint and full Quarto render claims remain blocked"
        ),
    }
    LINT_MD.write_text(_lint_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v390_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v390": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
