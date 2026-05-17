#!/usr/bin/env python3
"""Build Paper 4 v500 review outcome template consistency audit artifacts."""

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

VERSION = 500
PRIOR_REVIEW_TEMPLATE_VERSION = 499
NEXT_ARTIFACT = "paper4_v501_review_outcome_capture_dry_run.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v500_review_outcome_template_consistency_audit.md"
EXPECTED_FIELDS = {
    "reviewer_id",
    "review_timestamp_utc",
    "outcome_decision",
    "revision_required",
    "claim_boundary_ok",
    "caption_final_allowed",
    "patch_scope_ok",
    "reviewer_notes",
}
EXPECTED_CONTROLS = {
    "no_outcomes_captured",
    "no_caption_finalized",
    "no_patch_approval",
    "no_book_mutation",
    "no_render_submission",
    "no_final_promotion",
}


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _template_domain_coverage(template: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, expected_rows, expected_item_count in [
        ("layout_surface", 4, 10),
        ("caption_claim_safety", 10, 10),
        ("all_review_items", 14, 20),
    ]:
        if domain == "all_review_items":
            subset = template
        else:
            subset = template.loc[template["review_domain_v499"].eq(domain)]
        rows.append(
            {
                "coverage_domain_v500": domain,
                "expected_rows_v500": expected_rows,
                "observed_rows_v500": len(subset),
                "expected_item_count_v500": expected_item_count,
                "observed_item_count_v500": int(subset["review_item_count_v499"].sum()),
                "coverage_passed_v500": (
                    len(subset) == expected_rows
                    and int(subset["review_item_count_v499"].sum()) == expected_item_count
                ),
                "claim_boundary_v500": "template coverage audit only",
            }
        )
    return pd.DataFrame(rows)


def _required_field_coverage(field_dictionary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = set(field_dictionary["capture_field_v499"])
    required_fields = set(
        field_dictionary.loc[
            field_dictionary["required_capture_field_v499"].astype(bool),
            "capture_field_v499",
        ]
    )
    captured_fields = set(
        field_dictionary.loc[
            field_dictionary["field_captured_v499"].astype(bool),
            "capture_field_v499",
        ]
    )
    for field in sorted(EXPECTED_FIELDS):
        rows.append(
            {
                "capture_field_v500": field,
                "declared_v500": field in fields,
                "required_v500": field in required_fields,
                "captured_v500": field in captured_fields,
                "field_coverage_passed_v500": (
                    field in fields and field in required_fields and field not in captured_fields
                ),
                "claim_boundary_v500": "required field coverage audit only",
            }
        )
    return pd.DataFrame(rows)


def _control_integrity_audit(controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    active = set(
        controls.loc[controls["control_active_v499"].astype(bool), "control_id_v499"]
    )
    blocking = set(
        controls.loc[controls["blocks_patch_v499"].astype(bool), "control_id_v499"]
    )
    declared = set(controls["control_id_v499"])
    for control_id in sorted(EXPECTED_CONTROLS):
        rows.append(
            {
                "control_id_v500": control_id,
                "declared_v500": control_id in declared,
                "active_v500": control_id in active,
                "blocks_patch_v500": control_id in blocking,
                "control_integrity_passed_v500": (
                    control_id in declared and control_id in active and control_id in blocking
                ),
                "claim_boundary_v500": "control integrity audit only",
            }
        )
    return pd.DataFrame(rows)


def _template_consistency_audit(
    template: pd.DataFrame,
    domain_coverage: pd.DataFrame,
    field_coverage: pd.DataFrame,
    control_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "consistency_check_id_v500": "template_row_count",
            "observed_value_v500": len(template),
            "expected_value_v500": 14,
            "passed_v500": len(template) == 14,
            "claim_boundary_v500": "row count consistency only",
        },
        {
            "consistency_check_id_v500": "template_ids_unique",
            "observed_value_v500": template["outcome_template_id_v499"].nunique(),
            "expected_value_v500": len(template),
            "passed_v500": template["outcome_template_id_v499"].is_unique,
            "claim_boundary_v500": "identifier uniqueness consistency only",
        },
        {
            "consistency_check_id_v500": "template_domains_expected",
            "observed_value_v500": ",".join(sorted(template["review_domain_v499"].unique())),
            "expected_value_v500": "caption_claim_safety,layout_surface",
            "passed_v500": set(template["review_domain_v499"]) == {
                "caption_claim_safety",
                "layout_surface",
            },
            "claim_boundary_v500": "domain consistency only",
        },
        {
            "consistency_check_id_v500": "domain_coverage_passed",
            "observed_value_v500": int(domain_coverage["coverage_passed_v500"].sum()),
            "expected_value_v500": len(domain_coverage),
            "passed_v500": domain_coverage["coverage_passed_v500"].astype(bool).all(),
            "claim_boundary_v500": "coverage consistency only",
        },
        {
            "consistency_check_id_v500": "required_fields_passed",
            "observed_value_v500": int(
                field_coverage["field_coverage_passed_v500"].sum()
            ),
            "expected_value_v500": len(field_coverage),
            "passed_v500": field_coverage["field_coverage_passed_v500"].astype(bool).all(),
            "claim_boundary_v500": "field consistency only",
        },
        {
            "consistency_check_id_v500": "controls_passed",
            "observed_value_v500": int(
                control_audit["control_integrity_passed_v500"].sum()
            ),
            "expected_value_v500": len(control_audit),
            "passed_v500": control_audit["control_integrity_passed_v500"].astype(bool).all(),
            "claim_boundary_v500": "control consistency only",
        },
        {
            "consistency_check_id_v500": "no_outcomes_captured",
            "observed_value_v500": int(template["outcome_status_v499"].ne("not_captured").sum()),
            "expected_value_v500": 0,
            "passed_v500": template["outcome_status_v499"].eq("not_captured").all(),
            "claim_boundary_v500": "no outcome capture consistency only",
        },
        {
            "consistency_check_id_v500": "no_patch_permission",
            "observed_value_v500": int(template["patch_allowed_v499"].astype(bool).sum()),
            "expected_value_v500": 0,
            "passed_v500": not template["patch_allowed_v499"].astype(bool).any(),
            "claim_boundary_v500": "no patch permission consistency only",
        },
    ]
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v500": "review_outcome_template_consistency_audit_created",
                "ready_v500": True,
                "evidence_artifact_v500": "paper4_v500_template_consistency_audit.csv",
                "claim_boundary_v500": "template consistency audit only",
            },
            {
                "readiness_gate_v500": "domain_coverage_audit_passed",
                "ready_v500": True,
                "evidence_artifact_v500": "paper4_v500_template_domain_coverage.csv",
                "claim_boundary_v500": "coverage audit only",
            },
            {
                "readiness_gate_v500": "required_field_coverage_audit_passed",
                "ready_v500": True,
                "evidence_artifact_v500": "paper4_v500_required_field_coverage.csv",
                "claim_boundary_v500": "field coverage audit only",
            },
            {
                "readiness_gate_v500": "control_integrity_audit_passed",
                "ready_v500": True,
                "evidence_artifact_v500": "paper4_v500_control_integrity_audit.csv",
                "claim_boundary_v500": "control integrity audit only",
            },
            {
                "readiness_gate_v500": "future_capture_dry_run_ready",
                "ready_v500": True,
                "evidence_artifact_v500": "paper4_v500_template_consistency_audit.csv",
                "claim_boundary_v500": "future dry-run readiness only",
            },
            {
                "readiness_gate_v500": "review_outcomes_captured",
                "ready_v500": False,
                "evidence_artifact_v500": "all outcome rows remain not_captured",
                "claim_boundary_v500": "no review outcomes captured",
            },
            {
                "readiness_gate_v500": "ready_for_quarto_patch",
                "ready_v500": False,
                "evidence_artifact_v500": "review outcomes and approval remain absent",
                "claim_boundary_v500": "patch remains blocked",
            },
            {
                "readiness_gate_v500": "paper4_final_promotion_created",
                "ready_v500": False,
                "evidence_artifact_v500": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v500": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v500_review_outcome_template_consistency_audited",
                "allowed": True,
                "artifact": "paper4_v500_template_consistency_audit.csv",
                "boundary": "template consistency audit only",
            },
            {
                "claim_id": "v500_domain_and_field_coverage_passed",
                "allowed": True,
                "artifact": "paper4_v500_template_domain_and_field_audits",
                "boundary": "coverage audit only",
            },
            {
                "claim_id": "v500_future_capture_dry_run_ready",
                "allowed": True,
                "artifact": "paper4_v500_manuscript_readiness_delta.csv",
                "boundary": "future dry-run readiness only",
            },
            {
                "claim_id": "v500_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v500_template_consistency_audit.csv",
                "boundary": "all outcomes remain not_captured",
            },
            {
                "claim_id": "v500_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v500_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v500_final_promotion",
                "allowed": False,
                "artifact": "paper4_v500_manuscript_readiness_delta.csv",
                "boundary": "no final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v500 audits Paper 4 review outcome template consistency.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_template_consistency_audit.csv"
                ),
                "boundary": "Template consistency audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v500 verifies domain, field and control coverage for future capture.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_template_domain_coverage.csv"
                ),
                "boundary": "Coverage verification for future capture only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v500 makes a future outcome-capture dry run executable.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future dry-run readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v500 captures completed review outcomes or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_template_consistency_audit.csv"
                ),
                "boundary": "All outcome rows remain not_captured.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v500 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v500 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v500_manuscript_readiness_delta.csv"
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
                "executable_item": "v500 audits review outcome template consistency.",
                "status": "review_outcome_template_consistency_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v501 performs capture dry run without outcomes",
                "last_wave": "v500",
                "execution_result": "review_outcome_template_audited_without_mutation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v500")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Outcome Template Consistency Audit v500

Generated: {status["generated_at_utc"]}

## Result

v500 audits the v499 review outcome template for row counts, domain coverage,
required fields, control integrity, no-outcome state and no-patch state. All
consistency checks pass, so a future dry run is executable, but no review
outcome has been captured.

## Counts

- Consistency check rows: `{status["consistency_check_rows_v500"]}`.
- Passed consistency check rows: `{status["passed_consistency_check_rows_v500"]}`.
- Failed consistency check rows: `{status["failed_consistency_check_rows_v500"]}`.
- Outcome template rows: `{status["outcome_template_rows_v500"]}`.
- Layout outcome template rows: `{status["layout_outcome_template_rows_v500"]}`.
- Caption outcome template rows: `{status["caption_outcome_template_rows_v500"]}`.
- Required field rows: `{status["required_field_rows_v500"]}`.
- Active control rows: `{status["active_control_rows_v500"]}`.
- Outcome captured rows: `{status["outcome_captured_rows_v500"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v500"]}`.
- Future capture dry run ready: `{status["future_capture_dry_run_ready_v500"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v500"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v500 is a consistency audit only. It does not capture completed review outcomes,
finalize captions, approve patch scope, edit Quarto, render the book, make Paper
4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V500_REVIEW_OUTCOME_TEMPLATE_CONSISTENCY_AUDIT_START -->"
    end = "<!-- V500_REVIEW_OUTCOME_TEMPLATE_CONSISTENCY_AUDIT_END -->"
    block = f"""
{start}

## Wave v500: Review Outcome Template Consistency Audit

Generated: {status["generated_at_utc"]}

### Objective

v500 audits whether the v499 review outcome template is internally consistent
enough to support a future dry run while preserving the no-outcome and no-patch
state.

### Results

- Consistency check rows:
  `{status["consistency_check_rows_v500"]}`.
- Passed consistency check rows:
  `{status["passed_consistency_check_rows_v500"]}`.
- Failed consistency check rows:
  `{status["failed_consistency_check_rows_v500"]}`.
- Outcome template rows:
  `{status["outcome_template_rows_v500"]}`.
- Layout outcome template rows:
  `{status["layout_outcome_template_rows_v500"]}`.
- Caption outcome template rows:
  `{status["caption_outcome_template_rows_v500"]}`.
- Required field rows:
  `{status["required_field_rows_v500"]}`.
- Active control rows:
  `{status["active_control_rows_v500"]}`.
- Outcome captured rows:
  `{status["outcome_captured_rows_v500"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v500"]}`.
- Future capture dry run ready:
  `{status["future_capture_dry_run_ready_v500"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v500"]}`.
- Book sources modified:
  `{status["book_sources_modified_v500"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v500"]}`.

### Interpretation

The template is internally consistent and ready for a controlled dry run. This
still is not human signoff, caption finalization, patch approval or manuscript
promotion.

### Claim Impact

- Allowed: template consistency audit, coverage verification and future dry-run
  readiness.
- Still prohibited: completed review/signoff claims, final captions, Quarto
  patch readiness/application, Quarto/book-reference mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v500 in the living notebook. v501 should run a controlled capture dry run
without recording real review outcomes or modifying book sources.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v499 = _read_status(PRIOR_REVIEW_TEMPLATE_VERSION)
    expected_next = "paper4_v500_review_outcome_template_consistency_audit.md"
    if v499["next_artifact_v499"] != expected_next:
        raise RuntimeError("v500 expects v499 to route to template consistency audit.")

    template = pd.read_csv(TABLE_DIR / "paper4_v499_review_outcome_capture_template.csv")
    field_dictionary = pd.read_csv(TABLE_DIR / "paper4_v499_outcome_field_dictionary.csv")
    controls = pd.read_csv(TABLE_DIR / "paper4_v499_capture_control_register.csv")
    domain_coverage = _template_domain_coverage(template)
    field_coverage = _required_field_coverage(field_dictionary)
    control_audit = _control_integrity_audit(controls)
    consistency = _template_consistency_audit(
        template,
        domain_coverage,
        field_coverage,
        control_audit,
    )
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v500_template_consistency_audit.csv", consistency)
    write_csv(TABLE_DIR / "paper4_v500_template_domain_coverage.csv", domain_coverage)
    write_csv(TABLE_DIR / "paper4_v500_required_field_coverage.csv", field_coverage)
    write_csv(TABLE_DIR / "paper4_v500_control_integrity_audit.csv", control_audit)
    write_csv(TABLE_DIR / "paper4_v500_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v500_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v500_review_outcome_template_consistency_audit",
        "schema_version": "2026-05-17.500",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_review_outcome_template_version_v500": PRIOR_REVIEW_TEMPLATE_VERSION,
        "review_outcome_template_consistency_audit_created_v500": True,
        "consistency_check_rows_v500": len(consistency),
        "passed_consistency_check_rows_v500": int(
            consistency["passed_v500"].astype(bool).sum()
        ),
        "failed_consistency_check_rows_v500": int(
            (~consistency["passed_v500"].astype(bool)).sum()
        ),
        "domain_coverage_rows_v500": len(domain_coverage),
        "domain_coverage_passed_rows_v500": int(
            domain_coverage["coverage_passed_v500"].astype(bool).sum()
        ),
        "required_field_coverage_rows_v500": len(field_coverage),
        "required_field_coverage_passed_rows_v500": int(
            field_coverage["field_coverage_passed_v500"].astype(bool).sum()
        ),
        "control_integrity_rows_v500": len(control_audit),
        "control_integrity_passed_rows_v500": int(
            control_audit["control_integrity_passed_v500"].astype(bool).sum()
        ),
        "outcome_template_rows_v500": len(template),
        "layout_outcome_template_rows_v500": int(
            template["review_domain_v499"].eq("layout_surface").sum()
        ),
        "caption_outcome_template_rows_v500": int(
            template["review_domain_v499"].eq("caption_claim_safety").sum()
        ),
        "unique_template_ids_v500": int(template["outcome_template_id_v499"].nunique()),
        "required_field_rows_v500": int(
            field_dictionary["required_capture_field_v499"].astype(bool).sum()
        ),
        "captured_required_field_rows_v500": int(
            field_dictionary["field_captured_v499"].astype(bool).sum()
        ),
        "active_control_rows_v500": int(
            controls["control_active_v499"].astype(bool).sum()
        ),
        "outcome_captured_rows_v500": int(
            template["outcome_status_v499"].ne("not_captured").sum()
        ),
        "review_completed_rows_v500": int(
            template["review_completed_v499"].astype(bool).sum()
        ),
        "patch_allowed_rows_v500": int(template["patch_allowed_v499"].astype(bool).sum()),
        "readiness_delta_rows_v500": len(readiness),
        "future_capture_dry_run_ready_v500": True,
        "ready_for_quarto_patch_v500": False,
        "quarto_patch_applied_v500": False,
        "book_sources_modified_v500": False,
        "book_references_modified_v500": False,
        "submission_ready_claim_allowed_v500": False,
        "working_champion_claim_allowed_v500": False,
        "paper1_promotion_allowed_v500": False,
        "paper4_working_champion_changed_v500": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v500": NEXT_ARTIFACT,
        "claim_boundary": (
            "v500 audits review outcome template consistency only; dry-run "
            "readiness is allowed but outcomes, patching and promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v500 must not create final Paper 4 promotion.")
    if status["failed_consistency_check_rows_v500"] != 0:
        raise RuntimeError("v500 consistency audit must pass all checks.")
    if status["outcome_captured_rows_v500"] != 0:
        raise RuntimeError("v500 must not capture review outcomes.")
    if status["patch_allowed_rows_v500"] != 0:
        raise RuntimeError("v500 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v500": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
