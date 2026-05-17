#!/usr/bin/env python3
"""Build Paper 4 v480 controlled Quarto insertion plan artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
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

VERSION = 480
PRIOR_STUB_AUDIT_VERSION = 479
NEXT_ARTIFACT = "paper4_v481_manual_quarto_patch_decision.md"
PLAN_MD = NOTEBOOK.parent / "paper4_v480_controlled_quarto_insertion_plan.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _target_file_exists(path_text: str) -> bool:
    return Path(path_text).exists()


def _quarto_insertion_plan() -> pd.DataFrame:
    stubs = pd.read_csv(TABLE_DIR / "paper4_v478_section_text_stubs.csv")
    targets = {
        "stub_methods_protocol": (
            "book/chapters/19-paper-mega-extension/19f-sequential-decision-framework.qmd",
            "Sequential decision framing",
        ),
        "stub_results_evidence_cvar": (
            "book/chapters/19-paper-mega-extension/19bv-v33-cvar-certificate.qmd",
            "CVaR certificate and frontier evidence",
        ),
        "stub_results_evidence_governance_online": (
            "book/chapters/19-paper-mega-extension/19bx-v35-online-macro-validation.qmd",
            "Online and macro validation caveats",
        ),
        "stub_discussion_limitations": (
            "book/chapters/19-paper-mega-extension/19ca-v38-final-synthesis.qmd",
            "Final synthesis limitations",
        ),
        "stub_appendix_reproducibility": (
            "book/chapters/19-paper-mega-extension/19u-artifact-catalog-and-claims.qmd",
            "Artifact catalog and claims",
        ),
    }
    rows = []
    for idx, row in enumerate(stubs.itertuples(index=False), start=1):
        target_file, anchor = targets[row.stub_id_v478]
        rows.append(
            {
                "plan_step_v480": idx,
                "stub_id_v480": row.stub_id_v478,
                "manuscript_section_v480": row.manuscript_section_v478,
                "target_file_v480": target_file,
                "target_file_exists_v480": _target_file_exists(target_file),
                "candidate_anchor_v480": anchor,
                "insertion_mode_v480": "manual_reviewed_patch_only",
                "requires_manual_review_v480": True,
                "ready_for_patch_v480": False,
                "book_mutation_allowed_v480": False,
                "claim_boundary_v480": row.claim_boundary_v478,
            }
        )
    return pd.DataFrame(rows)


def _pre_patch_gate_checklist(plan: pd.DataFrame) -> pd.DataFrame:
    target_files_exist = plan["target_file_exists_v480"].astype(bool).all()
    return pd.DataFrame(
        [
            {
                "gate_id_v480": "v479_audit_passed",
                "passed_v480": True,
                "evidence_v480": "paper4_v479_status.json",
                "required_before_patch_v480": True,
            },
            {
                "gate_id_v480": "target_files_identified",
                "passed_v480": target_files_exist,
                "evidence_v480": "paper4_v480_quarto_insertion_plan.csv",
                "required_before_patch_v480": True,
            },
            {
                "gate_id_v480": "manual_review_required",
                "passed_v480": True,
                "evidence_v480": "paper4_v480_quarto_insertion_plan.csv",
                "required_before_patch_v480": True,
            },
            {
                "gate_id_v480": "rollback_plan_created",
                "passed_v480": True,
                "evidence_v480": "paper4_v480_rollback_plan.csv",
                "required_before_patch_v480": True,
            },
            {
                "gate_id_v480": "book_mutation_allowed",
                "passed_v480": False,
                "evidence_v480": "living notebook only",
                "required_before_patch_v480": True,
            },
            {
                "gate_id_v480": "paper4_final_promotion_absent",
                "passed_v480": not FORBIDDEN_FINAL_PROMOTION.exists(),
                "evidence_v480": "paper4_final_promotion.json absent",
                "required_before_patch_v480": True,
            },
        ]
    )


def _rollback_plan(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in plan.iterrows():
        rows.append(
            {
                "target_file_v480": row["target_file_v480"],
                "rollback_scope_v480": "single planned insertion block",
                "rollback_check_v480": f"git diff -- {row['target_file_v480']}",
                "rollback_action_v480": "do not apply patch before v481 decision",
                "mutation_performed_v480": False,
                "rollback_required_if_patched_v480": True,
            }
        )
    return pd.DataFrame(rows)


def _risk_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "risk_id_v480": "overclaiming_in_inserted_text",
                "risk_open_v480": True,
                "mitigation_v480": "preserve v478 caveats and v479 scan before patch",
            },
            {
                "risk_id_v480": "quarto_render_breakage",
                "risk_open_v480": True,
                "mitigation_v480": "run targeted Quarto render after any future patch",
            },
            {
                "risk_id_v480": "asset_caption_mismatch",
                "risk_open_v480": True,
                "mitigation_v480": "keep captions draft until manual review",
            },
            {
                "risk_id_v480": "book_reference_mutation",
                "risk_open_v480": True,
                "mitigation_v480": "no bibliography mutation in v480",
            },
            {
                "risk_id_v480": "paper_estrella_boundary_violation",
                "risk_open_v480": True,
                "mitigation_v480": "keep final promotion artifact absent",
            },
            {
                "risk_id_v480": "submission_readiness_overstatement",
                "risk_open_v480": True,
                "mitigation_v480": "label plan as insertion planning only",
            },
        ]
    )


def _readiness_delta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readiness_gate_v480": "controlled_quarto_insertion_plan_created",
                "ready_v480": True,
                "evidence_artifact_v480": "paper4_v480_quarto_insertion_plan.csv",
                "claim_boundary_v480": "plan only",
            },
            {
                "readiness_gate_v480": "target_files_identified",
                "ready_v480": True,
                "evidence_artifact_v480": "paper4_v480_quarto_insertion_plan.csv",
                "claim_boundary_v480": "path identification only",
            },
            {
                "readiness_gate_v480": "pre_patch_gates_created",
                "ready_v480": True,
                "evidence_artifact_v480": "paper4_v480_pre_patch_gate_checklist.csv",
                "claim_boundary_v480": "gate checklist only",
            },
            {
                "readiness_gate_v480": "rollback_plan_created",
                "ready_v480": True,
                "evidence_artifact_v480": "paper4_v480_rollback_plan.csv",
                "claim_boundary_v480": "rollback plan only",
            },
            {
                "readiness_gate_v480": "ready_for_quarto_patch",
                "ready_v480": False,
                "evidence_artifact_v480": "manual review required",
                "claim_boundary_v480": "patch deferred to v481 decision",
            },
            {
                "readiness_gate_v480": "book_sources_or_references_modified",
                "ready_v480": False,
                "evidence_artifact_v480": "book sources unchanged",
                "claim_boundary_v480": "no Quarto/book mutation in v480",
            },
            {
                "readiness_gate_v480": "submission_ready",
                "ready_v480": False,
                "evidence_artifact_v480": "future venue and render validation",
                "claim_boundary_v480": "not a submission package",
            },
            {
                "readiness_gate_v480": "paper4_final_promotion_created",
                "ready_v480": False,
                "evidence_artifact_v480": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v480": "Paper Estrella remains protected",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v480_controlled_quarto_insertion_plan_created",
                "allowed": True,
                "artifact": "paper4_v480_quarto_insertion_plan.csv",
                "boundary": "plan only",
            },
            {
                "claim_id": "v480_pre_patch_gate_checklist_created",
                "allowed": True,
                "artifact": "paper4_v480_pre_patch_gate_checklist.csv",
                "boundary": "gate checklist only",
            },
            {
                "claim_id": "v480_rollback_plan_created",
                "allowed": True,
                "artifact": "paper4_v480_rollback_plan.csv",
                "boundary": "rollback plan only",
            },
            {
                "claim_id": "v480_quarto_patch_applied",
                "allowed": False,
                "artifact": "paper4_v480_manuscript_readiness_delta.csv",
                "boundary": "no book source mutation in v480",
            },
            {
                "claim_id": "v480_submission_ready_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v480_manuscript_readiness_delta.csv",
                "boundary": "no submission or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v480 creates a controlled Quarto insertion plan.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v480_quarto_insertion_plan.csv"
                ),
                "boundary": "Plan only; no patch applied.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v480 creates pre-patch gates and rollback instructions.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v480_pre_patch_gate_checklist.csv"
                ),
                "boundary": "Gate and rollback planning only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v480 applies a Quarto patch or edits book sources.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v480_manuscript_readiness_delta.csv"
                ),
                "boundary": "No book source mutation in v480.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v480 makes Paper 4 ready for submission.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v480_manuscript_readiness_delta.csv"
                ),
                "boundary": "Manual review, patch, render and venue gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v480 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v480_manuscript_readiness_delta.csv"
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
                "executable_item": "v480 creates controlled Quarto insertion plan.",
                "status": "controlled_quarto_insertion_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v481 decides whether to apply a manual patch",
                "last_wave": "v480",
                "execution_result": "controlled_insertion_plan_created_without_book_edit",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v480")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _plan_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Controlled Quarto Insertion Plan v480

Generated: {status["generated_at_utc"]}

## Result

v480 creates a controlled insertion plan for the v478-v479 draft material. It
identifies candidate Quarto targets, pre-patch gates, rollback checks and open
risks. It does not edit book sources, apply a patch, render the book, make Paper
4 submission-ready, or promote Paper 4.

## Counts

- Insertion plan rows: `{status["insertion_plan_rows_v480"]}`.
- Pre-patch gate rows: `{status["pre_patch_gate_rows_v480"]}`.
- Rollback rows: `{status["rollback_rows_v480"]}`.
- Risk rows: `{status["risk_rows_v480"]}`.
- Target files exist: `{status["target_files_exist_v480"]}`.
- Ready for Quarto patch: `{status["ready_for_quarto_patch_v480"]}`.
- Quarto patch applied: `{status["quarto_patch_applied_v480"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v480 is planning only. Manual review, actual patching, render validation, venue
formatting, submission readiness, Paper Estrella replacement and final Paper 4
promotion remain blocked.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V480_CONTROLLED_QUARTO_INSERTION_PLAN_START -->"
    end = "<!-- V480_CONTROLLED_QUARTO_INSERTION_PLAN_END -->"
    block = f"""
{start}

## Wave v480: Controlled Quarto Insertion Plan

Generated: {status["generated_at_utc"]}

### Objective

v480 creates a controlled plan for where a future manual patch could insert the
v478-v479 draft material, without editing any book source.

### Results

- Insertion plan rows:
  `{status["insertion_plan_rows_v480"]}`.
- Pre-patch gate rows:
  `{status["pre_patch_gate_rows_v480"]}`.
- Rollback rows:
  `{status["rollback_rows_v480"]}`.
- Risk rows:
  `{status["risk_rows_v480"]}`.
- Target files exist:
  `{status["target_files_exist_v480"]}`.
- Ready for Quarto patch:
  `{status["ready_for_quarto_patch_v480"]}`.
- Quarto patch applied:
  `{status["quarto_patch_applied_v480"]}`.
- Book sources modified:
  `{status["book_sources_modified_v480"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v480"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v480"]}`.

### Interpretation

The manuscript work is close enough to have a controlled insertion plan, but not
close enough to patch automatically. Manual review remains mandatory.

### Claim Impact

- Allowed: controlled insertion plan, pre-patch gate checklist and rollback plan.
- Still prohibited: actual Quarto patch, book-reference mutation, submission
  readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v480 in the living notebook. v481 should decide whether a manual patch is
allowed, still without treating Paper 4 as final.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v479 = _read_status(PRIOR_STUB_AUDIT_VERSION)
    if v479["next_artifact_v479"] != "paper4_v480_controlled_quarto_insertion_plan.md":
        raise RuntimeError("v480 expects v479 to route to controlled insertion plan.")

    plan = _quarto_insertion_plan()
    gates = _pre_patch_gate_checklist(plan)
    rollback = _rollback_plan(plan)
    risks = _risk_register()
    readiness = _readiness_delta()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v480_quarto_insertion_plan.csv", plan)
    write_csv(TABLE_DIR / "paper4_v480_pre_patch_gate_checklist.csv", gates)
    write_csv(TABLE_DIR / "paper4_v480_rollback_plan.csv", rollback)
    write_csv(TABLE_DIR / "paper4_v480_insertion_risk_register.csv", risks)
    write_csv(TABLE_DIR / "paper4_v480_manuscript_readiness_delta.csv", readiness)
    write_csv(TABLE_DIR / "paper4_v480_claim_matrix_delta.csv", claim_matrix)

    status = {
        "phase": "v480_controlled_quarto_insertion_plan",
        "schema_version": "2026-05-17.480",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_stub_audit_version_v480": PRIOR_STUB_AUDIT_VERSION,
        "controlled_quarto_insertion_plan_created_v480": True,
        "insertion_plan_rows_v480": len(plan),
        "pre_patch_gate_rows_v480": len(gates),
        "rollback_rows_v480": len(rollback),
        "risk_rows_v480": len(risks),
        "readiness_delta_rows_v480": len(readiness),
        "target_files_exist_v480": bool(plan["target_file_exists_v480"].astype(bool).all()),
        "manual_review_required_v480": True,
        "ready_for_quarto_patch_v480": False,
        "quarto_patch_applied_v480": False,
        "book_sources_modified_v480": False,
        "book_references_modified_v480": False,
        "submission_ready_claim_allowed_v480": False,
        "working_champion_claim_allowed_v480": False,
        "paper1_promotion_allowed_v480": False,
        "paper4_working_champion_changed_v480": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v480": NEXT_ARTIFACT,
        "claim_boundary": (
            "v480 creates an insertion plan only; actual patching, final prose, "
            "submission and final promotion remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v480 must not create final Paper 4 promotion.")

    PLAN_MD.write_text(_plan_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v480": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
