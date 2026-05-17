#!/usr/bin/env python3
"""Build Paper 4 v502 manual review outcome capture packet artifacts."""

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

VERSION = 502
PRIOR_DRY_RUN_VERSION = 501
NEXT_ARTIFACT = "paper4_v503_manual_capture_assignment_gap_audit.md"
PACKET_MD = NOTEBOOK.parent / "paper4_v502_review_outcome_manual_capture_packet.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _manual_capture_packet(queue: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    required_fields = ";".join(fields["capture_field_v501"].astype(str).tolist())
    rows = []
    for _, row in queue.iterrows():
        rows.append(
            {
                "capture_packet_id_v502": row["manual_capture_queue_id_v501"],
                "priority_v502": int(row["priority_v501"]),
                "outcome_template_id_v502": row["outcome_template_id_v501"],
                "review_domain_v502": row["review_domain_v501"],
                "asset_id_v502": row["asset_id_v501"],
                "target_block_v502": row["target_block_v501"],
                "required_fields_v502": required_fields,
                "packet_ready_v502": True,
                "reviewer_action_required_v502": True,
                "real_outcome_prefilled_v502": False,
                "review_completed_v502": False,
                "caption_final_v502": False,
                "patch_allowed_v502": False,
                "claim_boundary_v502": "manual capture packet only",
            }
        )
    return pd.DataFrame(rows)


def _manual_capture_field_checklist(
    packet: pd.DataFrame,
    fields: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, packet_row in packet.iterrows():
        for _, field_row in fields.iterrows():
            rows.append(
                {
                    "capture_packet_id_v502": packet_row["capture_packet_id_v502"],
                    "capture_field_v502": field_row["capture_field_v501"],
                    "field_required_v502": bool(field_row["required_v501"]),
                    "field_present_v502": bool(field_row["field_present_in_template_v501"]),
                    "field_prefilled_v502": False,
                    "human_entry_required_v502": True,
                    "claim_boundary_v502": "manual field checklist only",
                }
            )
    return pd.DataFrame(rows)


def _review_assignment_stub(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        rows.append(
            {
                "assignment_stub_id_v502": row["capture_packet_id_v502"],
                "priority_v502": int(row["priority_v502"]),
                "review_domain_v502": row["review_domain_v502"],
                "asset_id_v502": row["asset_id_v502"],
                "reviewer_assigned_v502": False,
                "assignment_required_v502": True,
                "ready_for_manual_assignment_v502": True,
                "outcome_recorded_v502": False,
                "patch_allowed_v502": False,
                "claim_boundary_v502": "assignment stub only",
            }
        )
    return pd.DataFrame(rows)


def _packet_safety_register(safety: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in safety.iterrows():
        rows.append(
            {
                "packet_safety_id_v502": row["safety_gate_id_v501"],
                "inherited_control_active_v502": bool(row["control_active_v501"]),
                "inherited_blocks_patch_v502": bool(row["blocks_patch_v501"]),
                "packet_safety_passed_v502": bool(row["dry_run_passed_v501"]),
                "claim_boundary_v502": "manual packet safety inheritance only",
            }
        )
    return pd.DataFrame(rows)


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v502": "manual_capture_packet_created",
                "ready_v502": True,
                "evidence_artifact_v502": "paper4_v502_manual_capture_packet.csv",
                "claim_boundary_v502": "manual capture packet only",
            },
            {
                "readiness_gate_v502": "manual_capture_field_checklist_created",
                "ready_v502": True,
                "evidence_artifact_v502": "paper4_v502_manual_capture_field_checklist.csv",
                "claim_boundary_v502": "field checklist only",
            },
            {
                "readiness_gate_v502": "review_assignment_stubs_created",
                "ready_v502": True,
                "evidence_artifact_v502": "paper4_v502_review_assignment_stub.csv",
                "claim_boundary_v502": "assignment stubs only",
            },
            {
                "readiness_gate_v502": "packet_safety_register_created",
                "ready_v502": True,
                "evidence_artifact_v502": "paper4_v502_capture_packet_safety_register.csv",
                "claim_boundary_v502": "safety register only",
            },
            {
                "readiness_gate_v502": "reviewers_assigned",
                "ready_v502": False,
                "evidence_artifact_v502": "reviewer_assigned_v502 remains false",
                "claim_boundary_v502": "awaiting reviewer assignment",
            },
            {
                "readiness_gate_v502": "real_review_outcomes_captured",
                "ready_v502": False,
                "evidence_artifact_v502": "no real outcome rows recorded",
                "claim_boundary_v502": "no review evidence captured",
            },
            {
                "readiness_gate_v502": "ready_for_quarto_patch",
                "ready_v502": False,
                "evidence_artifact_v502": "outcomes, signoff and approval remain absent",
                "claim_boundary_v502": "patch remains blocked",
            },
            {
                "readiness_gate_v502": "paper4_final_promotion_created",
                "ready_v502": False,
                "evidence_artifact_v502": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v502": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v502_manual_capture_packet_created",
                "allowed": True,
                "artifact": "paper4_v502_manual_capture_packet.csv",
                "boundary": "manual capture packet only",
            },
            {
                "claim_id": "v502_field_checklist_and_assignment_stubs_created",
                "allowed": True,
                "artifact": "paper4_v502_manual_capture_field_and_assignment_artifacts",
                "boundary": "manual preparation only",
            },
            {
                "claim_id": "v502_safety_register_inherited",
                "allowed": True,
                "artifact": "paper4_v502_capture_packet_safety_register.csv",
                "boundary": "safety inheritance only",
            },
            {
                "claim_id": "v502_real_reviews_completed_or_captions_final",
                "allowed": False,
                "artifact": "paper4_v502_manual_capture_packet.csv",
                "boundary": "no real outcomes or final captions",
            },
            {
                "claim_id": "v502_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v502_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v502_final_promotion",
                "allowed": False,
                "artifact": "paper4_v502_manuscript_readiness_delta.csv",
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
                "claim": "v502 creates a manual Paper 4 review outcome capture packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_manual_capture_packet.csv"
                ),
                "boundary": "Manual capture packet only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v502 creates field checklist and assignment stubs.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_manual_capture_field_checklist.csv"
                ),
                "boundary": "Manual preparation only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v502 inherits dry-run safety controls into the manual packet.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_capture_packet_safety_register.csv"
                ),
                "boundary": "Safety inheritance only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v502 assigns reviewers or captures completed review outcomes.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_review_assignment_stub.csv"
                ),
                "boundary": "Reviewer assignments and outcomes remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v502 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v502 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v502_manuscript_readiness_delta.csv"
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
                "executable_item": "v502 creates manual capture packet.",
                "status": "manual_capture_packet_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v503 audits reviewer assignment gaps",
                "last_wave": "v502",
                "execution_result": "manual_capture_packet_created_without_outcomes",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v502")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _packet_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Review Outcome Manual Capture Packet v502

Generated: {status["generated_at_utc"]}

## Result

v502 creates a manual capture packet for the 14 review outcome rows, including a
field checklist, reviewer assignment stubs and inherited safety register. It
does not assign reviewers, prefill outcomes, finalize captions or authorize
patching.

## Counts

- Capture packet rows: `{status["capture_packet_rows_v502"]}`.
- Packet ready rows: `{status["packet_ready_rows_v502"]}`.
- Field checklist rows: `{status["field_checklist_rows_v502"]}`.
- Field prefilled rows: `{status["field_prefilled_rows_v502"]}`.
- Reviewer assignment rows: `{status["reviewer_assignment_rows_v502"]}`.
- Reviewer assigned rows: `{status["reviewer_assigned_rows_v502"]}`.
- Safety register rows: `{status["safety_register_rows_v502"]}`.
- Passed safety register rows: `{status["passed_safety_register_rows_v502"]}`.
- Real outcome captured rows: `{status["real_outcome_captured_rows_v502"]}`.
- Review completed rows: `{status["review_completed_rows_v502"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v502"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v502"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v502 is a manual packet only. It does not capture completed review outcomes,
assign reviewers, finalize captions, approve patch scope, edit Quarto, render
the book, make Paper 4 submission-ready, replace Paper Estrella, or promote
Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V502_REVIEW_OUTCOME_MANUAL_CAPTURE_PACKET_START -->"
    end = "<!-- V502_REVIEW_OUTCOME_MANUAL_CAPTURE_PACKET_END -->"
    block = f"""
{start}

## Wave v502: Review Outcome Manual Capture Packet

Generated: {status["generated_at_utc"]}

### Objective

v502 turns the v501 dry-run queue into a manual capture packet with field
checklists, assignment stubs and inherited safety controls.

### Results

- Capture packet rows:
  `{status["capture_packet_rows_v502"]}`.
- Packet ready rows:
  `{status["packet_ready_rows_v502"]}`.
- Field checklist rows:
  `{status["field_checklist_rows_v502"]}`.
- Field prefilled rows:
  `{status["field_prefilled_rows_v502"]}`.
- Reviewer assignment rows:
  `{status["reviewer_assignment_rows_v502"]}`.
- Reviewer assigned rows:
  `{status["reviewer_assigned_rows_v502"]}`.
- Safety register rows:
  `{status["safety_register_rows_v502"]}`.
- Passed safety register rows:
  `{status["passed_safety_register_rows_v502"]}`.
- Real outcome captured rows:
  `{status["real_outcome_captured_rows_v502"]}`.
- Review completed rows:
  `{status["review_completed_rows_v502"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v502"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v502"]}`.
- Book sources modified:
  `{status["book_sources_modified_v502"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v502"]}`.

### Interpretation

The capture packet is ready for a human assignment step. It still has no actual
review decisions, no final captions and no patch permission.

### Claim Impact

- Allowed: manual packet creation, field checklist preparation, assignment
  stubs and inherited safety controls.
- Still prohibited: reviewer assignment, completed review/signoff claims, final
  captions, Quarto patch readiness/application, Quarto/book-reference mutation,
  submission readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v502 in the living notebook. v503 should audit assignment gaps before any
manual capture claims are allowed.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v501 = _read_status(PRIOR_DRY_RUN_VERSION)
    if v501["next_artifact_v501"] != "paper4_v502_review_outcome_manual_capture_packet.md":
        raise RuntimeError("v502 expects v501 to route to manual capture packet.")
    if not v501["manual_capture_packet_ready_v501"]:
        raise RuntimeError("v502 requires v501 manual packet readiness.")

    queue = pd.read_csv(TABLE_DIR / "paper4_v501_manual_capture_queue.csv")
    fields = pd.read_csv(TABLE_DIR / "paper4_v501_capture_form_validation.csv")
    safety = pd.read_csv(TABLE_DIR / "paper4_v501_dry_run_safety_gate.csv")
    packet = _manual_capture_packet(queue, fields)
    checklist = _manual_capture_field_checklist(packet, fields)
    assignments = _review_assignment_stub(packet)
    packet_safety = _packet_safety_register(safety)
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v502_manual_capture_packet.csv", packet)
    write_csv(TABLE_DIR / "paper4_v502_manual_capture_field_checklist.csv", checklist)
    write_csv(TABLE_DIR / "paper4_v502_review_assignment_stub.csv", assignments)
    write_csv(TABLE_DIR / "paper4_v502_capture_packet_safety_register.csv", packet_safety)
    write_csv(TABLE_DIR / "paper4_v502_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v502_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v502_review_outcome_manual_capture_packet",
        "schema_version": "2026-05-17.502",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dry_run_version_v502": PRIOR_DRY_RUN_VERSION,
        "manual_capture_packet_created_v502": True,
        "capture_packet_rows_v502": len(packet),
        "packet_ready_rows_v502": int(packet["packet_ready_v502"].astype(bool).sum()),
        "field_checklist_rows_v502": len(checklist),
        "field_prefilled_rows_v502": int(
            checklist["field_prefilled_v502"].astype(bool).sum()
        ),
        "reviewer_assignment_rows_v502": len(assignments),
        "reviewer_assigned_rows_v502": int(
            assignments["reviewer_assigned_v502"].astype(bool).sum()
        ),
        "safety_register_rows_v502": len(packet_safety),
        "passed_safety_register_rows_v502": int(
            packet_safety["packet_safety_passed_v502"].astype(bool).sum()
        ),
        "real_outcome_captured_rows_v502": int(
            packet["real_outcome_prefilled_v502"].astype(bool).sum()
        ),
        "review_completed_rows_v502": int(
            packet["review_completed_v502"].astype(bool).sum()
        ),
        "caption_final_rows_v502": int(packet["caption_final_v502"].astype(bool).sum()),
        "patch_allowed_rows_v502": int(packet["patch_allowed_v502"].astype(bool).sum()),
        "readiness_delta_rows_v502": len(readiness),
        "reviewer_assignment_gap_audit_ready_v502": True,
        "ready_for_quarto_patch_v502": False,
        "quarto_patch_applied_v502": False,
        "book_sources_modified_v502": False,
        "book_references_modified_v502": False,
        "submission_ready_claim_allowed_v502": False,
        "working_champion_claim_allowed_v502": False,
        "paper1_promotion_allowed_v502": False,
        "paper4_working_champion_changed_v502": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v502": NEXT_ARTIFACT,
        "claim_boundary": (
            "v502 creates a manual review outcome capture packet only; "
            "assignments, outcomes, captions, patching and promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v502 must not create final Paper 4 promotion.")
    if status["real_outcome_captured_rows_v502"] != 0:
        raise RuntimeError("v502 must not prefill real review outcomes.")
    if status["reviewer_assigned_rows_v502"] != 0:
        raise RuntimeError("v502 must not assign reviewers.")
    if status["patch_allowed_rows_v502"] != 0:
        raise RuntimeError("v502 must not approve a Quarto patch.")

    PACKET_MD.write_text(_packet_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v502": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
