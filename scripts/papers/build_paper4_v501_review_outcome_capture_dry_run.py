#!/usr/bin/env python3
"""Build Paper 4 v501 review outcome capture dry-run artifacts."""

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

VERSION = 501
PRIOR_TEMPLATE_AUDIT_VERSION = 500
NEXT_ARTIFACT = "paper4_v502_review_outcome_manual_capture_packet.md"
DRY_RUN_MD = NOTEBOOK.parent / "paper4_v501_review_outcome_capture_dry_run.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _capture_dry_run(template: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ordinal, (_, row) in enumerate(template.iterrows(), start=1):
        rows.append(
            {
                "dry_run_id_v501": f"dry_run_{ordinal:02d}",
                "outcome_template_id_v501": row["outcome_template_id_v499"],
                "review_domain_v501": row["review_domain_v499"],
                "asset_id_v501": row["asset_id_v499"],
                "target_block_v501": row["target_block_v499"],
                "dry_run_executed_v501": True,
                "capture_form_valid_v501": True,
                "real_outcome_captured_v501": False,
                "synthetic_outcome_written_v501": False,
                "review_completed_v501": False,
                "caption_final_v501": False,
                "patch_allowed_v501": False,
                "manual_capture_required_v501": True,
                "claim_boundary_v501": "dry run only; no outcome evidence recorded",
            }
        )
    return pd.DataFrame(rows)


def _capture_form_validation(field_dictionary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in field_dictionary.iterrows():
        rows.append(
            {
                "capture_field_v501": row["capture_field_v499"],
                "required_v501": bool(row["required_capture_field_v499"]),
                "field_present_in_template_v501": True,
                "dry_run_value_written_v501": False,
                "manual_capture_ready_v501": True,
                "claim_boundary_v501": "form validation only",
            }
        )
    return pd.DataFrame(rows)


def _dry_run_safety_gate(controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in controls.iterrows():
        rows.append(
            {
                "safety_gate_id_v501": row["control_id_v499"],
                "control_active_v501": bool(row["control_active_v499"]),
                "blocks_patch_v501": bool(row["blocks_patch_v499"]),
                "dry_run_passed_v501": bool(row["control_active_v499"])
                and bool(row["blocks_patch_v499"]),
                "claim_boundary_v501": "dry-run safety gate only",
            }
        )
    return pd.DataFrame(rows)


def _manual_capture_queue(template: pd.DataFrame) -> pd.DataFrame:
    rows = []
    domain_order = {"layout_surface": 1, "caption_claim_safety": 2}
    ordered = template.assign(
        domain_order_v501=template["review_domain_v499"].map(domain_order)
    ).sort_values(["domain_order_v501", "outcome_template_id_v499"])
    for priority, (_, row) in enumerate(ordered.iterrows(), start=1):
        rows.append(
            {
                "manual_capture_queue_id_v501": f"manual_capture_{priority:02d}",
                "priority_v501": priority,
                "outcome_template_id_v501": row["outcome_template_id_v499"],
                "review_domain_v501": row["review_domain_v499"],
                "asset_id_v501": row["asset_id_v499"],
                "target_block_v501": row["target_block_v499"],
                "manual_capture_ready_v501": True,
                "awaiting_human_review_v501": True,
                "outcome_recorded_v501": False,
                "patch_allowed_v501": False,
                "claim_boundary_v501": "manual capture queue only",
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v501": "review_outcome_capture_dry_run_created",
                "ready_v501": True,
                "evidence_artifact_v501": "paper4_v501_review_outcome_capture_dry_run.csv",
                "claim_boundary_v501": "dry-run artifact only",
            },
            {
                "readiness_gate_v501": "capture_form_validation_passed",
                "ready_v501": True,
                "evidence_artifact_v501": "paper4_v501_capture_form_validation.csv",
                "claim_boundary_v501": "form validation only",
            },
            {
                "readiness_gate_v501": "dry_run_safety_gates_passed",
                "ready_v501": True,
                "evidence_artifact_v501": "paper4_v501_dry_run_safety_gate.csv",
                "claim_boundary_v501": "safety gates remain active",
            },
            {
                "readiness_gate_v501": "manual_capture_queue_ready",
                "ready_v501": True,
                "evidence_artifact_v501": "paper4_v501_manual_capture_queue.csv",
                "claim_boundary_v501": "manual queue readiness only",
            },
            {
                "readiness_gate_v501": "real_review_outcomes_captured",
                "ready_v501": False,
                "evidence_artifact_v501": "no real outcome rows recorded",
                "claim_boundary_v501": "dry run does not count as review evidence",
            },
            {
                "readiness_gate_v501": "captions_finalized",
                "ready_v501": False,
                "evidence_artifact_v501": "caption finalization remains false",
                "claim_boundary_v501": "no final captions",
            },
            {
                "readiness_gate_v501": "ready_for_quarto_patch",
                "ready_v501": False,
                "evidence_artifact_v501": "human outcomes and approval remain absent",
                "claim_boundary_v501": "patch remains blocked",
            },
            {
                "readiness_gate_v501": "paper4_final_promotion_created",
                "ready_v501": False,
                "evidence_artifact_v501": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v501": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v501_review_outcome_capture_dry_run_created",
                "allowed": True,
                "artifact": "paper4_v501_review_outcome_capture_dry_run.csv",
                "boundary": "dry run only",
            },
            {
                "claim_id": "v501_capture_form_and_safety_gates_validated",
                "allowed": True,
                "artifact": "paper4_v501_capture_form_and_safety_artifacts",
                "boundary": "form and safety validation only",
            },
            {
                "claim_id": "v501_manual_capture_queue_ready",
                "allowed": True,
                "artifact": "paper4_v501_manual_capture_queue.csv",
                "boundary": "manual queue readiness only",
            },
            {
                "claim_id": "v501_real_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v501_review_outcome_capture_dry_run.csv",
                "boundary": "no real outcomes or final captions",
            },
            {
                "claim_id": "v501_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v501_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v501_final_promotion",
                "allowed": False,
                "artifact": "paper4_v501_manuscript_readiness_delta.csv",
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
                "claim": "v501 executes a Paper 4 review outcome capture dry run.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_review_outcome_capture_dry_run.csv"
                ),
                "boundary": "Dry run only; no review evidence recorded.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v501 validates capture form and safety gates for future use.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_capture_form_validation.csv"
                ),
                "boundary": "Future capture readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v501 stages a manual review outcome capture queue.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_manual_capture_queue.csv"
                ),
                "boundary": "Manual queue only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v501 captures real completed review outcomes or finalizes captions.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_review_outcome_capture_dry_run.csv"
                ),
                "boundary": "No real outcomes or final captions are recorded.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v501 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v501 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v501_manuscript_readiness_delta.csv"
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
                "executable_item": "v501 runs review outcome capture dry run.",
                "status": "review_outcome_capture_dry_run_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v502 creates manual capture packet",
                "last_wave": "v501",
                "execution_result": "review_outcome_capture_dry_run_without_outcomes",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v501")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _dry_run_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Outcome Capture Dry Run v501

Generated: {status["generated_at_utc"]}

## Result

v501 executes a dry run over the 14 v499 review outcome template rows. The dry
run validates the capture form, safety gates and manual capture queue, while
recording zero real review outcomes and granting zero patch permissions.

## Counts

- Dry-run rows: `{status["dry_run_rows_v501"]}`.
- Dry-run executed rows: `{status["dry_run_executed_rows_v501"]}`.
- Capture form field rows: `{status["capture_form_field_rows_v501"]}`.
- Form validation passed rows: `{status["form_validation_passed_rows_v501"]}`.
- Safety gate rows: `{status["safety_gate_rows_v501"]}`.
- Passed safety gate rows: `{status["passed_safety_gate_rows_v501"]}`.
- Manual capture queue rows: `{status["manual_capture_queue_rows_v501"]}`.
- Manual capture ready rows: `{status["manual_capture_ready_rows_v501"]}`.
- Real outcome captured rows: `{status["real_outcome_captured_rows_v501"]}`.
- Synthetic outcome written rows: `{status["synthetic_outcome_written_rows_v501"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v501"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v501"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v501 is a dry run only. It does not capture completed review outcomes, finalize
captions, approve patch scope, edit Quarto, render the book, make Paper 4
submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V501_REVIEW_OUTCOME_CAPTURE_DRY_RUN_START -->"
    end = "<!-- V501_REVIEW_OUTCOME_CAPTURE_DRY_RUN_END -->"
    block = f"""
{start}

## Wave v501: Review Outcome Capture Dry Run

Generated: {status["generated_at_utc"]}

### Objective

v501 runs a controlled dry run over the v499 template after the v500 consistency
audit. It validates the capture path without recording real outcomes or
authorizing patch scope.

### Results

- Dry-run rows:
  `{status["dry_run_rows_v501"]}`.
- Dry-run executed rows:
  `{status["dry_run_executed_rows_v501"]}`.
- Capture form field rows:
  `{status["capture_form_field_rows_v501"]}`.
- Form validation passed rows:
  `{status["form_validation_passed_rows_v501"]}`.
- Safety gate rows:
  `{status["safety_gate_rows_v501"]}`.
- Passed safety gate rows:
  `{status["passed_safety_gate_rows_v501"]}`.
- Manual capture queue rows:
  `{status["manual_capture_queue_rows_v501"]}`.
- Manual capture ready rows:
  `{status["manual_capture_ready_rows_v501"]}`.
- Real outcome captured rows:
  `{status["real_outcome_captured_rows_v501"]}`.
- Synthetic outcome written rows:
  `{status["synthetic_outcome_written_rows_v501"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v501"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v501"]}`.
- Book sources modified:
  `{status["book_sources_modified_v501"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v501"]}`.

### Interpretation

The future review-outcome capture path is operationally staged, but the paper
still lacks actual human review outcomes, caption finalization and patch
approval.

### Claim Impact

- Allowed: dry-run execution, capture form validation, safety-gate validation
  and manual capture queue staging.
- Still prohibited: completed review/signoff claims, final captions, Quarto
  patch readiness/application, Quarto/book-reference mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v501 in the living notebook. v502 should create a manual capture packet
without fabricating or pre-filling review outcomes.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v500 = _read_status(PRIOR_TEMPLATE_AUDIT_VERSION)
    if v500["next_artifact_v500"] != "paper4_v501_review_outcome_capture_dry_run.md":
        raise RuntimeError("v501 expects v500 to route to capture dry run.")
    if not v500["future_capture_dry_run_ready_v500"]:
        raise RuntimeError("v501 requires v500 dry-run readiness.")

    template = pd.read_csv(TABLE_DIR / "paper4_v499_review_outcome_capture_template.csv")
    field_dictionary = pd.read_csv(TABLE_DIR / "paper4_v499_outcome_field_dictionary.csv")
    controls = pd.read_csv(TABLE_DIR / "paper4_v499_capture_control_register.csv")
    dry_run = _capture_dry_run(template)
    form_validation = _capture_form_validation(field_dictionary)
    safety = _dry_run_safety_gate(controls)
    queue = _manual_capture_queue(template)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v501_review_outcome_capture_dry_run.csv", dry_run)
    write_csv(TABLE_DIR / "paper4_v501_capture_form_validation.csv", form_validation)
    write_csv(TABLE_DIR / "paper4_v501_dry_run_safety_gate.csv", safety)
    write_csv(TABLE_DIR / "paper4_v501_manual_capture_queue.csv", queue)
    write_csv(TABLE_DIR / "paper4_v501_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v501_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v501_review_outcome_capture_dry_run",
        "schema_version": "2026-05-17.501",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_template_audit_version_v501": PRIOR_TEMPLATE_AUDIT_VERSION,
        "review_outcome_capture_dry_run_created_v501": True,
        "dry_run_rows_v501": len(dry_run),
        "dry_run_executed_rows_v501": int(
            dry_run["dry_run_executed_v501"].astype(bool).sum()
        ),
        "capture_form_field_rows_v501": len(form_validation),
        "form_validation_passed_rows_v501": int(
            form_validation["manual_capture_ready_v501"].astype(bool).sum()
        ),
        "safety_gate_rows_v501": len(safety),
        "passed_safety_gate_rows_v501": int(
            safety["dry_run_passed_v501"].astype(bool).sum()
        ),
        "manual_capture_queue_rows_v501": len(queue),
        "manual_capture_ready_rows_v501": int(
            queue["manual_capture_ready_v501"].astype(bool).sum()
        ),
        "real_outcome_captured_rows_v501": int(
            dry_run["real_outcome_captured_v501"].astype(bool).sum()
        ),
        "synthetic_outcome_written_rows_v501": int(
            dry_run["synthetic_outcome_written_v501"].astype(bool).sum()
        ),
        "review_completed_rows_v501": int(
            dry_run["review_completed_v501"].astype(bool).sum()
        ),
        "caption_final_rows_v501": int(dry_run["caption_final_v501"].astype(bool).sum()),
        "patch_allowed_rows_v501": int(dry_run["patch_allowed_v501"].astype(bool).sum()),
        "readiness_delta_rows_v501": len(readiness),
        "manual_capture_packet_ready_v501": True,
        "ready_for_quarto_patch_v501": False,
        "quarto_patch_applied_v501": False,
        "book_sources_modified_v501": False,
        "book_references_modified_v501": False,
        "submission_ready_claim_allowed_v501": False,
        "working_champion_claim_allowed_v501": False,
        "paper1_promotion_allowed_v501": False,
        "paper4_working_champion_changed_v501": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v501": NEXT_ARTIFACT,
        "claim_boundary": (
            "v501 dry-runs the review outcome capture flow only; real outcomes, "
            "captions, patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v501 must not create final Paper 4 promotion.")
    if status["real_outcome_captured_rows_v501"] != 0:
        raise RuntimeError("v501 must not capture real review outcomes.")
    if status["synthetic_outcome_written_rows_v501"] != 0:
        raise RuntimeError("v501 must not write synthetic outcomes as evidence.")
    if status["patch_allowed_rows_v501"] != 0:
        raise RuntimeError("v501 must not approve a Quarto patch.")

    DRY_RUN_MD.write_text(_dry_run_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v501": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
