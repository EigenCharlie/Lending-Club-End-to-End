#!/usr/bin/env python3
"""Build Paper 4 v458 post-assembly render decision artifacts."""

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
    write_csv,
    write_json,
)

VERSION = 458
PRIOR_POST_ASSEMBLY_PYTEST_VERSION = 457
NEXT_ARTIFACT = "paper4_v459_target_venue_structure_packet.md"
DECISION_MD = NOTEBOOK.parent / "paper4_v458_post_assembly_render_decision.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _changed_surface_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "surface_id_v458": "paper4_notes",
                "changed_since_v455_v458": True,
                "quarto_source_v458": False,
                "render_required_now_v458": False,
                "example_path_v458": "reports/paper_material/paper4/notes",
                "claim_boundary_v458": "living-lab notes are not official Quarto chapter source",
            },
            {
                "surface_id_v458": "paper4_tables_status",
                "changed_since_v455_v458": True,
                "quarto_source_v458": False,
                "render_required_now_v458": False,
                "example_path_v458": "reports/paper_material/paper4/tables",
                "claim_boundary_v458": "evidence tables and status files only",
            },
            {
                "surface_id_v458": "paper_builder_scripts",
                "changed_since_v455_v458": True,
                "quarto_source_v458": False,
                "render_required_now_v458": False,
                "example_path_v458": "scripts/papers",
                "claim_boundary_v458": "artifact builders are not rendered pages",
            },
            {
                "surface_id_v458": "guardrail_tests",
                "changed_since_v455_v458": True,
                "quarto_source_v458": False,
                "render_required_now_v458": False,
                "example_path_v458": "tests/test_docs/test_paper4_living_lab_guardrails.py",
                "claim_boundary_v458": "test updates are not rendered pages",
            },
            {
                "surface_id_v458": "paper4_quarto_chapter_source",
                "changed_since_v455_v458": False,
                "quarto_source_v458": True,
                "render_required_now_v458": False,
                "example_path_v458": "book/chapters/19-paper-mega-extension",
                "claim_boundary_v458": "no official Paper 4 Quarto source change in v456-v457",
            },
            {
                "surface_id_v458": "full_book_registry",
                "changed_since_v455_v458": False,
                "quarto_source_v458": True,
                "render_required_now_v458": False,
                "example_path_v458": "book/_quarto.yml",
                "claim_boundary_v458": "book registration unchanged",
            },
        ]
    )


def _render_decision_matrix(v457_status: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_gate_v458": "official_paper4_quarto_source_changed",
                "observed_v458": False,
                "render_required_now_v458": False,
                "evidence_artifact_v458": "paper4_v458_changed_surface_inventory.csv",
                "claim_boundary_v458": "official chapter source unchanged",
            },
            {
                "decision_gate_v458": "full_book_registry_changed",
                "observed_v458": False,
                "render_required_now_v458": False,
                "evidence_artifact_v458": "paper4_v458_changed_surface_inventory.csv",
                "claim_boundary_v458": "book registry unchanged",
            },
            {
                "decision_gate_v458": "assembly_artifacts_created",
                "observed_v458": True,
                "render_required_now_v458": False,
                "evidence_artifact_v458": "paper4_v456_manuscript_assembly_packet.md",
                "claim_boundary_v458": "assembly packet remains reports-side evidence",
            },
            {
                "decision_gate_v458": "post_assembly_pytest_clean",
                "observed_v458": bool(
                    v457_status["post_assembly_regression_refresh_complete_v457"]
                ),
                "render_required_now_v458": False,
                "evidence_artifact_v458": "paper4_v457_pytest_probe_summary.csv",
                "claim_boundary_v458": "pytest clean supports deferring render unless book changes",
            },
            {
                "decision_gate_v458": "final_promotion_absent",
                "observed_v458": not FORBIDDEN_FINAL_PROMOTION.exists(),
                "render_required_now_v458": False,
                "evidence_artifact_v458": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v458": "no final promotion or book promotion was created",
            },
        ]
    )


def _remaining_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v458": "target_venue_not_selected",
                "blocking_v458": True,
                "evidence_count_v458": 0,
                "required_next_artifact_v458": NEXT_ARTIFACT,
                "claim_boundary_v458": "choose venue structure before submission language",
            },
            {
                "blocker_id_v458": "conditional_quarto_render_if_promoted",
                "blocking_v458": True,
                "evidence_count_v458": 1,
                "required_next_artifact_v458": "future_post_promotion_quarto_render_probe",
                "claim_boundary_v458": "render refresh becomes required if content enters book/",
            },
            {
                "blocker_id_v458": "external_dataset_validation_not_run",
                "blocking_v458": True,
                "evidence_count_v458": 0,
                "required_next_artifact_v458": "future_external_validation_protocol",
                "claim_boundary_v458": "do not claim external generalization",
            },
            {
                "blocker_id_v458": "paper4_final_promotion_forbidden",
                "blocking_v458": True,
                "evidence_count_v458": 1,
                "required_next_artifact_v458": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v458": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v458_post_assembly_render_decision_recorded",
                "allowed": True,
                "artifact": "paper4_v458_post_assembly_render_decision.md",
                "boundary": "render decision only",
            },
            {
                "claim_id": "v458_no_current_quarto_source_change_detected",
                "allowed": True,
                "artifact": "paper4_v458_changed_surface_inventory.csv",
                "boundary": "v456-v457 touched reports/scripts/tests, not book sources",
            },
            {
                "claim_id": "v458_render_never_needed",
                "allowed": False,
                "artifact": "paper4_v458_remaining_blockers.csv",
                "boundary": "render is required if future content enters book/",
            },
            {
                "claim_id": "v458_target_venue_or_submission_ready",
                "allowed": False,
                "artifact": "paper4_v458_remaining_blockers.csv",
                "boundary": "target venue packet remains next work",
            },
            {
                "claim_id": "v458_working_champion_or_final_promotion",
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
                "claim": "v458 records a post-assembly Quarto render decision.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v458_post_assembly_render_decision.md"
                ),
                "boundary": "Decision only; no render rerun in v458.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v458 finds no current Paper 4 Quarto source change from v456-v457.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v458_changed_surface_inventory.csv"
                ),
                "boundary": "Reports/scripts/tests changed; official book sources did not.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v458 proves future Quarto renders are never needed.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v458_remaining_blockers.csv"
                ),
                "boundary": "A render is required if future content enters book/.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v458 makes Paper 4 target-venue ready, submitted, or external.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v458_remaining_blockers.csv"
                ),
                "boundary": "Target-venue structure and external validation remain pending.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v458 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v458_remaining_blockers.csv"
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


def _update_backlog() -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "Manuscript",
                "executable_item": "v458 records post-assembly render decision.",
                "status": "post_assembly_render_decision_recorded",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v459 maps assembled packet to target-venue structure",
                "last_wave": "v458",
                "execution_result": "render_decision_recorded_without_quarto_rerun",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v458")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-Assembly Render Decision v458

Generated: {status["generated_at_utc"]}

## Decision

Do not rerun Quarto immediately in v458.

## Rationale

The v456-v457 waves assembled reports-side manuscript evidence, added builder
scripts, and expanded guardrail tests. They did not modify the official Paper 4
Quarto chapter source or the full-book registry. The clean v457 post-assembly
pytest and repository Ruff snapshot are therefore sufficient for the current
reports-side assembly packet.

## Conditional Render Rule

If a future wave promotes any assembled text, table, figure, or registration into
`book/chapters/19-paper-mega-extension` or `book/_quarto.yml`, a Paper 4 chapter
render and likely a full-book render become executable validation work again.

## Result

- Changed surfaces recorded: `{status["changed_surface_count_v458"]}`.
- Quarto source changes detected: `{status["quarto_source_change_count_v458"]}`.
- Render required now: `{status["render_required_now_v458"]}`.
- Full-book render required now: `{status["full_book_render_required_now_v458"]}`.
- v457 post-assembly pytest clean: `{status["post_assembly_pytest_clean_from_v457"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v458 is a render decision only. It does not rerun Quarto, select a target venue,
create external validation, make a submission package, replace Paper Estrella, or
promote Paper 4 as final.

## Next Executable Wave

Build `{status["next_artifact_v458"]}`.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V458_POST_ASSEMBLY_RENDER_DECISION_START -->"
    end = "<!-- V458_POST_ASSEMBLY_RENDER_DECISION_END -->"
    block = f"""
{start}

## Wave v458: Post-Assembly Render Decision

Generated: {status["generated_at_utc"]}

### Objective

v458 decides whether v456-v457 require an immediate Quarto render refresh.

### Results

- Changed surfaces recorded:
  `{status["changed_surface_count_v458"]}`.
- Quarto source changes detected:
  `{status["quarto_source_change_count_v458"]}`.
- Render required now:
  `{status["render_required_now_v458"]}`.
- Full-book render required now:
  `{status["full_book_render_required_now_v458"]}`.
- v457 post-assembly pytest clean:
  `{status["post_assembly_pytest_clean_from_v457"]}`.
- Render decision recorded:
  `{status["render_decision_recorded_v458"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v458"]}`.

### Interpretation

Because v456-v457 touched reports-side evidence, scripts and tests rather than
official Quarto book sources, v458 records that no immediate render rerun is
required. A render becomes required again if future content enters `book/`.

### Claim Impact

- Allowed: post-assembly render decision and current no-book-source-change
  statement.
- Still prohibited: claiming renders are never needed, target-venue readiness,
  external validation, champion replacement or final promotion.

### Quarto Promotion Decision

Keep v458 in the living notebook. v459 should map the assembled packet to a
target-venue structure without submission language.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v457 = _read_status(457)
    if v457["next_artifact_v457"] != "paper4_v458_post_assembly_render_decision.md":
        raise RuntimeError("v458 expects v457 to route to render decision.")
    if v457["post_assembly_regression_refresh_complete_v457"] is not True:
        raise RuntimeError("v458 expects v457 post-assembly regression refresh to pass.")

    surfaces = _changed_surface_inventory()
    decision = _render_decision_matrix(v457)
    blockers = _remaining_blockers()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v458_changed_surface_inventory.csv", surfaces)
    write_csv(TABLE_DIR / "paper4_v458_render_decision_matrix.csv", decision)
    write_csv(TABLE_DIR / "paper4_v458_remaining_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v458_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v458_post_assembly_render_decision",
        "schema_version": "2026-05-17.458",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_post_assembly_pytest_version_v458": PRIOR_POST_ASSEMBLY_PYTEST_VERSION,
        "changed_surface_count_v458": int(surfaces["changed_since_v455_v458"].sum()),
        "quarto_source_change_count_v458": int(
            (
                surfaces["changed_since_v455_v458"].astype(bool)
                & surfaces["quarto_source_v458"].astype(bool)
            ).sum()
        ),
        "render_required_now_v458": bool(decision["render_required_now_v458"].any()),
        "full_book_render_required_now_v458": False,
        "post_assembly_pytest_clean_from_v457": bool(
            v457["post_assembly_regression_refresh_complete_v457"]
        ),
        "render_decision_recorded_v458": True,
        "target_venue_structure_created_v458": False,
        "external_validation_complete_v458": False,
        "working_champion_claim_allowed_v458": False,
        "paper1_promotion_allowed_v458": False,
        "paper4_working_champion_changed_v458": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v458": NEXT_ARTIFACT,
        "claim_boundary": (
            "v458 records no immediate render requirement because book sources "
            "were unchanged; target venue, external validation and final promotion "
            "remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v458 must not create final Paper 4 promotion.")

    DECISION_MD.write_text(_decision_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v458": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
