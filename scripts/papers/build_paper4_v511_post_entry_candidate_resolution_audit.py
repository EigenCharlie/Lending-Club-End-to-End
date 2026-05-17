#!/usr/bin/env python3
"""Build Paper 4 v511 post-entry candidate resolution audit artifacts."""

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

VERSION = 511
PRIOR_MANUAL_ENTRY_PACKET_VERSION = 510
NEXT_ARTIFACT = "paper4_v512_candidate_input_request_packet.md"
AUDIT_MD = NOTEBOOK.parent / "paper4_v511_post_entry_candidate_resolution_audit.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text())


def _post_entry_candidate_resolution_audit(packet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in packet.iterrows():
        complete = (
            bool(row["candidate_identifier_entered_v510"])
            and bool(row["nomination_fields_entered_v510"])
            and bool(row["nomination_signoff_recorded_v510"])
            and bool(row["candidate_nomination_recorded_v510"])
        )
        rows.append(
            {
                "post_entry_audit_id_v511": row["manual_entry_packet_id_v510"],
                "priority_v511": int(row["priority_v510"]),
                "review_domain_v511": row["review_domain_v510"],
                "reviewer_role_required_v511": row["reviewer_role_required_v510"],
                "manual_entry_packet_ready_v511": bool(
                    row["manual_entry_packet_ready_v510"]
                ),
                "candidate_identifier_entered_v511": bool(
                    row["candidate_identifier_entered_v510"]
                ),
                "nomination_fields_entered_v511": bool(
                    row["nomination_fields_entered_v510"]
                ),
                "nomination_signoff_recorded_v511": bool(
                    row["nomination_signoff_recorded_v510"]
                ),
                "candidate_nomination_recorded_v511": bool(
                    row["candidate_nomination_recorded_v510"]
                ),
                "candidate_resolution_complete_v511": complete,
                "candidate_resolution_gap_open_v511": not complete,
                "eligibility_review_allowed_v511": False,
                "reviewer_assignment_allowed_v511": False,
                "outcome_capture_allowed_v511": False,
                "patch_allowed_v511": False,
                "required_next_step_v511": "collect_manual_entry_inputs",
                "claim_boundary_v511": "post-entry candidate resolution audit only",
            }
        )
    return pd.DataFrame(rows)


def _post_entry_field_completion_audit(fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fields.iterrows():
        entered = bool(row["field_value_entered_v510"])
        rows.append(
            {
                "post_entry_audit_id_v511": row["manual_entry_packet_id_v510"],
                "nomination_field_v511": row["nomination_field_v510"],
                "field_required_v511": bool(row["field_required_v510"]),
                "field_value_entered_v511": entered,
                "field_completion_gap_open_v511": not entered,
                "human_entry_required_v511": bool(row["human_entry_required_v510"]),
                "claim_boundary_v511": "post-entry field completion audit only",
            }
        )
    return pd.DataFrame(rows)


def _resolution_readiness_summary(audit: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        audit.groupby("review_domain_v511", sort=True)
        .agg(
            post_entry_audit_rows_v511=("post_entry_audit_id_v511", "size"),
            complete_resolution_rows_v511=(
                "candidate_resolution_complete_v511",
                "sum",
            ),
            open_resolution_gap_rows_v511=(
                "candidate_resolution_gap_open_v511",
                "sum",
            ),
            eligibility_review_allowed_rows_v511=(
                "eligibility_review_allowed_v511",
                "sum",
            ),
        )
        .reset_index()
    )
    grouped["domain_resolution_ready_v511"] = (
        grouped["open_resolution_gap_rows_v511"].eq(0)
        & grouped["post_entry_audit_rows_v511"].gt(0)
    )
    grouped["claim_boundary_v511"] = "domain readiness summary only"
    return grouped


def _post_entry_blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "post_entry_blocker_id_v511": "no_manual_candidate_identifier_entries",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": True,
                "required_resolution_v511": "enter candidate identifiers",
            },
            {
                "post_entry_blocker_id_v511": "no_manual_field_value_entries",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": True,
                "required_resolution_v511": "enter required nomination field values",
            },
            {
                "post_entry_blocker_id_v511": "no_nomination_signoff",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": True,
                "required_resolution_v511": "record nomination signoff",
            },
            {
                "post_entry_blocker_id_v511": "no_candidate_nomination_recorded",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": True,
                "required_resolution_v511": "record candidate nominations",
            },
            {
                "post_entry_blocker_id_v511": "eligibility_review_blocked",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": False,
                "required_resolution_v511": "start eligibility only after nomination",
            },
            {
                "post_entry_blocker_id_v511": "reviewer_assignment_blocked",
                "blocker_open_v511": True,
                "blocks_eligibility_review_v511": False,
                "required_resolution_v511": "assign reviewers only after eligibility",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v511": "post_entry_candidate_resolution_audit_created",
                "ready_v511": True,
                "evidence_artifact_v511": (
                    "paper4_v511_post_entry_candidate_resolution_audit.csv"
                ),
                "claim_boundary_v511": "post-entry audit only",
            },
            {
                "readiness_gate_v511": "post_entry_field_completion_audit_created",
                "ready_v511": True,
                "evidence_artifact_v511": (
                    "paper4_v511_post_entry_field_completion_audit.csv"
                ),
                "claim_boundary_v511": "post-entry field audit only",
            },
            {
                "readiness_gate_v511": "resolution_readiness_summary_created",
                "ready_v511": True,
                "evidence_artifact_v511": (
                    "paper4_v511_resolution_readiness_summary.csv"
                ),
                "claim_boundary_v511": "readiness summary only",
            },
            {
                "readiness_gate_v511": "candidate_input_request_packet_ready",
                "ready_v511": True,
                "evidence_artifact_v511": "paper4_v511_post_entry_blocker_register.csv",
                "claim_boundary_v511": "future input request readiness only",
            },
            {
                "readiness_gate_v511": "candidate_identifiers_entered",
                "ready_v511": False,
                "evidence_artifact_v511": "manual entry packet remains blank",
                "claim_boundary_v511": "no candidate identifiers entered",
            },
            {
                "readiness_gate_v511": "candidate_nominations_recorded",
                "ready_v511": False,
                "evidence_artifact_v511": "candidate nominations remain absent",
                "claim_boundary_v511": "no candidates nominated",
            },
            {
                "readiness_gate_v511": "ready_for_quarto_patch",
                "ready_v511": False,
                "evidence_artifact_v511": "candidate resolution remains incomplete",
                "claim_boundary_v511": "patch remains blocked",
            },
            {
                "readiness_gate_v511": "paper4_final_promotion_created",
                "ready_v511": False,
                "evidence_artifact_v511": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v511": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v511_post_entry_resolution_audit_created",
                "allowed": True,
                "artifact": "paper4_v511_post_entry_candidate_resolution_audit.csv",
                "boundary": "post-entry resolution audit only",
            },
            {
                "claim_id": "v511_field_completion_audit_created",
                "allowed": True,
                "artifact": "paper4_v511_post_entry_field_completion_audit.csv",
                "boundary": "post-entry field completion audit only",
            },
            {
                "claim_id": "v511_candidate_input_request_packet_ready",
                "allowed": True,
                "artifact": "paper4_v511_manuscript_readiness_delta.csv",
                "boundary": "future candidate input request readiness only",
            },
            {
                "claim_id": "v511_candidates_resolved_or_nominated",
                "allowed": False,
                "artifact": "paper4_v511_post_entry_candidate_resolution_audit.csv",
                "boundary": "no candidates resolved or nominated",
            },
            {
                "claim_id": "v511_patch_ready_or_applied",
                "allowed": False,
                "artifact": "paper4_v511_manuscript_readiness_delta.csv",
                "boundary": "patch remains blocked",
            },
            {
                "claim_id": "v511_final_promotion",
                "allowed": False,
                "artifact": "paper4_v511_manuscript_readiness_delta.csv",
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
                "claim": "v511 audits post-entry candidate resolution status.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_post_entry_candidate_resolution_audit.csv"
                ),
                "boundary": "Post-entry resolution audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v511 audits manual-entry field completion gaps.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_post_entry_field_completion_audit.csv"
                ),
                "boundary": "Field completion audit only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v511 makes candidate input request packet executable next.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_manuscript_readiness_delta.csv"
                ),
                "boundary": "Future candidate input request readiness only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v511 resolves candidates, nominates candidates, or assigns reviewers.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_post_entry_candidate_resolution_audit.csv"
                ),
                "boundary": "Candidate resolution remains incomplete.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v511 makes Paper 4 ready for Quarto patching or applies a patch.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_manuscript_readiness_delta.csv"
                ),
                "boundary": "Patch remains blocked.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v511 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v511_manuscript_readiness_delta.csv"
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
                "executable_item": "v511 audits post-entry candidate resolution.",
                "status": "post_entry_candidate_resolution_audit_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v512 creates candidate input request packet",
                "last_wave": "v511",
                "execution_result": "post_entry_audit_confirmed_no_candidates",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v511")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Post-entry Candidate Resolution Audit v511

Generated: {status["generated_at_utc"]}

## Result

v511 audits the v510 manual-entry packet. No candidate identifiers, nomination
field values, signoffs or candidate nominations have been entered, so all 14
candidate-resolution gaps and all 84 field-completion gaps remain open.

## Counts

- Post-entry audit rows: `{status["post_entry_audit_rows_v511"]}`.
- Open candidate resolution gap rows: `{status["open_candidate_resolution_gap_rows_v511"]}`.
- Candidate resolution complete rows: `{status["candidate_resolution_complete_rows_v511"]}`.
- Candidate identifier entered rows: `{status["candidate_identifier_entered_rows_v511"]}`.
- Nomination fields entered rows: `{status["nomination_fields_entered_rows_v511"]}`.
- Nomination signoff recorded rows: `{status["nomination_signoff_recorded_rows_v511"]}`.
- Candidate nomination recorded rows: `{status["candidate_nomination_recorded_rows_v511"]}`.
- Field completion audit rows: `{status["field_completion_audit_rows_v511"]}`.
- Open field completion gap rows: `{status["open_field_completion_gap_rows_v511"]}`.
- Field value entered rows: `{status["field_value_entered_rows_v511"]}`.
- Domain summary rows: `{status["domain_summary_rows_v511"]}`.
- Domains with open resolution gaps: `{status["domains_with_open_resolution_gaps_rows_v511"]}`.
- Post-entry blocker rows: `{status["post_entry_blocker_rows_v511"]}`.
- Open post-entry blocker rows: `{status["open_post_entry_blocker_rows_v511"]}`.
- Eligibility review allowed rows: `{status["eligibility_review_allowed_rows_v511"]}`.
- Reviewer assignment allowed rows: `{status["reviewer_assignment_allowed_rows_v511"]}`.
- Outcome capture allowed rows: `{status["outcome_capture_allowed_rows_v511"]}`.
- Patch allowed rows: `{status["patch_allowed_rows_v511"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v511"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v511 is a post-entry audit only. It does not resolve candidate identifiers,
nominate candidates, assign reviewers, capture completed review outcomes,
finalize captions, approve patch scope, edit Quarto, render the book, make
Paper 4 submission-ready, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V511_POST_ENTRY_CANDIDATE_RESOLUTION_AUDIT_START -->"
    end = "<!-- V511_POST_ENTRY_CANDIDATE_RESOLUTION_AUDIT_END -->"
    block = f"""
{start}

## Wave v511: Post-entry Candidate Resolution Audit

Generated: {status["generated_at_utc"]}

### Objective

v511 audits the v510 manual-entry packet after creation. Since no human entries
are present, it verifies that candidate resolution remains incomplete.

### Results

- Post-entry audit rows:
  `{status["post_entry_audit_rows_v511"]}`.
- Open candidate resolution gap rows:
  `{status["open_candidate_resolution_gap_rows_v511"]}`.
- Candidate resolution complete rows:
  `{status["candidate_resolution_complete_rows_v511"]}`.
- Candidate identifier entered rows:
  `{status["candidate_identifier_entered_rows_v511"]}`.
- Nomination fields entered rows:
  `{status["nomination_fields_entered_rows_v511"]}`.
- Nomination signoff recorded rows:
  `{status["nomination_signoff_recorded_rows_v511"]}`.
- Candidate nomination recorded rows:
  `{status["candidate_nomination_recorded_rows_v511"]}`.
- Field completion audit rows:
  `{status["field_completion_audit_rows_v511"]}`.
- Open field completion gap rows:
  `{status["open_field_completion_gap_rows_v511"]}`.
- Field value entered rows:
  `{status["field_value_entered_rows_v511"]}`.
- Domain summary rows:
  `{status["domain_summary_rows_v511"]}`.
- Domains with open resolution gaps:
  `{status["domains_with_open_resolution_gaps_rows_v511"]}`.
- Post-entry blocker rows:
  `{status["post_entry_blocker_rows_v511"]}`.
- Open post-entry blocker rows:
  `{status["open_post_entry_blocker_rows_v511"]}`.
- Eligibility review allowed rows:
  `{status["eligibility_review_allowed_rows_v511"]}`.
- Reviewer assignment allowed rows:
  `{status["reviewer_assignment_allowed_rows_v511"]}`.
- Outcome capture allowed rows:
  `{status["outcome_capture_allowed_rows_v511"]}`.
- Patch allowed rows:
  `{status["patch_allowed_rows_v511"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v511"]}`.
- Book sources modified:
  `{status["book_sources_modified_v511"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v511"]}`.

### Interpretation

The living lab now has a post-entry audit proving the manual-entry surface is
still blank. The next executable artifact should request candidate inputs,
rather than opening eligibility review or reviewer assignment.

### Claim Impact

- Allowed: post-entry audit, field-completion audit, domain readiness summary
  and future candidate input request readiness.
- Still prohibited: candidate resolution/nomination, reviewer assignment,
  completed review claims, final captions, Quarto patch readiness/application,
  Quarto/book mutation, submission readiness, Paper Estrella replacement and
  final Paper 4 promotion.

### Quarto Promotion Decision

Keep v511 in the living notebook. v512 should create a candidate input request
packet with explicit evidence requirements and still no fabricated candidates.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v510 = _read_status(PRIOR_MANUAL_ENTRY_PACKET_VERSION)
    expected_next = "paper4_v511_post_entry_candidate_resolution_audit.md"
    if v510["next_artifact_v510"] != expected_next:
        raise RuntimeError("v511 expects v510 to route to post-entry audit.")
    if not v510["post_entry_candidate_resolution_audit_ready_v510"]:
        raise RuntimeError("v511 requires v510 post-entry audit readiness.")

    packet = pd.read_csv(
        TABLE_DIR / "paper4_v510_candidate_resolution_manual_entry_packet.csv"
    )
    fields = pd.read_csv(TABLE_DIR / "paper4_v510_manual_entry_field_template.csv")
    audit = _post_entry_candidate_resolution_audit(packet)
    field_audit = _post_entry_field_completion_audit(fields)
    domain_summary = _resolution_readiness_summary(audit)
    blockers = _post_entry_blocker_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v511_post_entry_candidate_resolution_audit.csv", audit)
    write_csv(TABLE_DIR / "paper4_v511_post_entry_field_completion_audit.csv", field_audit)
    write_csv(TABLE_DIR / "paper4_v511_resolution_readiness_summary.csv", domain_summary)
    write_csv(TABLE_DIR / "paper4_v511_post_entry_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v511_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v511_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v511_post_entry_candidate_resolution_audit",
        "schema_version": "2026-05-17.511",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_manual_entry_packet_version_v511": (
            PRIOR_MANUAL_ENTRY_PACKET_VERSION
        ),
        "post_entry_candidate_resolution_audit_created_v511": True,
        "post_entry_audit_rows_v511": len(audit),
        "open_candidate_resolution_gap_rows_v511": int(
            audit["candidate_resolution_gap_open_v511"].astype(bool).sum()
        ),
        "candidate_resolution_complete_rows_v511": int(
            audit["candidate_resolution_complete_v511"].astype(bool).sum()
        ),
        "candidate_identifier_entered_rows_v511": int(
            audit["candidate_identifier_entered_v511"].astype(bool).sum()
        ),
        "nomination_fields_entered_rows_v511": int(
            audit["nomination_fields_entered_v511"].astype(bool).sum()
        ),
        "nomination_signoff_recorded_rows_v511": int(
            audit["nomination_signoff_recorded_v511"].astype(bool).sum()
        ),
        "candidate_nomination_recorded_rows_v511": int(
            audit["candidate_nomination_recorded_v511"].astype(bool).sum()
        ),
        "field_completion_audit_rows_v511": len(field_audit),
        "open_field_completion_gap_rows_v511": int(
            field_audit["field_completion_gap_open_v511"].astype(bool).sum()
        ),
        "field_value_entered_rows_v511": int(
            field_audit["field_value_entered_v511"].astype(bool).sum()
        ),
        "domain_summary_rows_v511": len(domain_summary),
        "domains_with_open_resolution_gaps_rows_v511": int(
            domain_summary["open_resolution_gap_rows_v511"].astype(bool).sum()
        ),
        "post_entry_blocker_rows_v511": len(blockers),
        "open_post_entry_blocker_rows_v511": int(
            blockers["blocker_open_v511"].astype(bool).sum()
        ),
        "eligibility_review_allowed_rows_v511": int(
            audit["eligibility_review_allowed_v511"].astype(bool).sum()
        ),
        "reviewer_assignment_allowed_rows_v511": int(
            audit["reviewer_assignment_allowed_v511"].astype(bool).sum()
        ),
        "outcome_capture_allowed_rows_v511": int(
            audit["outcome_capture_allowed_v511"].astype(bool).sum()
        ),
        "patch_allowed_rows_v511": int(audit["patch_allowed_v511"].astype(bool).sum()),
        "readiness_delta_rows_v511": len(readiness),
        "candidate_input_request_packet_ready_v511": True,
        "ready_for_quarto_patch_v511": False,
        "quarto_patch_applied_v511": False,
        "book_sources_modified_v511": False,
        "book_references_modified_v511": False,
        "submission_ready_claim_allowed_v511": False,
        "working_champion_claim_allowed_v511": False,
        "paper1_promotion_allowed_v511": False,
        "paper4_working_champion_changed_v511": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v511": NEXT_ARTIFACT,
        "claim_boundary": (
            "v511 audits post-entry candidate resolution only; candidate "
            "resolution, nominations, assignments, outcomes, captions, "
            "patching, submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v511 must not create final Paper 4 promotion.")
    if status["candidate_resolution_complete_rows_v511"] != 0:
        raise RuntimeError("v511 must not complete candidate resolution.")
    if status["candidate_identifier_entered_rows_v511"] != 0:
        raise RuntimeError("v511 must not enter candidate identifiers.")
    if status["candidate_nomination_recorded_rows_v511"] != 0:
        raise RuntimeError("v511 must not record candidate nominations.")
    if status["eligibility_review_allowed_rows_v511"] != 0:
        raise RuntimeError("v511 must not allow eligibility review.")
    if status["reviewer_assignment_allowed_rows_v511"] != 0:
        raise RuntimeError("v511 must not allow reviewer assignment.")
    if status["outcome_capture_allowed_rows_v511"] != 0:
        raise RuntimeError("v511 must not allow outcome capture.")
    if status["patch_allowed_rows_v511"] != 0:
        raise RuntimeError("v511 must not approve a Quarto patch.")

    AUDIT_MD.write_text(_audit_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v511": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
