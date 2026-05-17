#!/usr/bin/env python3
"""Build Paper 4 v379 evidence-gap closure work-order artifacts."""

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

VERSION = 379
PRIOR_GAP_REGISTER_VERSION = 378
NEXT_VERSION = 380
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_manuscript_section_scaffold.md"


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v379 converts v378 open gaps into an evidence-closure work order.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v379_evidence_gap_closure_work_order.csv"
                ),
                "boundary": "Work-order planning only; no gap is closed by declaration.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v379 identifies which open gaps can be executed from current artifacts.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v379_execution_queue.csv"
                ),
                "boundary": "Execution queue only; downstream waves must generate evidence.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v379 closes submission-readiness gaps or authorizes stronger claims.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v379_claim_blockers.csv"
                ),
                "boundary": "Open gaps remain until future artifacts satisfy them.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v379 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v379_claim_blockers.csv"
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
                "lane": "Publishability/Scope",
                "executable_item": (
                    "v379 partitions v378 open gaps into executable current-artifact tasks "
                    "and externally blocked dependencies."
                ),
                "status": "evidence_gap_closure_work_order_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v380 drafts manuscript section scaffolds from v374-v379 without "
                    "changing claim permissions"
                ),
                "last_wave": "v379",
                "execution_result": "open_gaps_partitioned_into_executable_and_external_tasks",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v379")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V379_EVIDENCE_GAP_CLOSURE_WORK_ORDER_START -->"
    end = "<!-- V379_EVIDENCE_GAP_CLOSURE_WORK_ORDER_END -->"
    block = f"""
{start}

## Wave v379: Evidence-Gap Closure Work Order

Generated: {status["generated_at_utc"]}

### Objective

v379 takes the v378 submission-readiness gap register and converts each open
gap into a concrete work order, separating work executable from current
artifacts from work blocked by external data, legal review or approval.

### Results

- Work-order rows:
  `{status["work_order_rows_v379"]}`.
- Executable-now rows:
  `{status["executable_now_rows_v379"]}`.
- External-blocked rows:
  `{status["external_blocked_rows_v379"]}`.
- Execution queue rows:
  `{status["execution_queue_rows_v379"]}`.
- Blocked dependency rows:
  `{status["blocked_dependency_rows_v379"]}`.
- Highest-priority executable rows:
  `{status["highest_priority_executable_rows_v379"]}`.
- Submission-ready claim allowed:
  `{status["submission_ready_claim_allowed_v379"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v379"]}`.

### Interpretation

The next useful path is no longer vague. The lab should first draft manuscript
sections from the existing bounded evidence, preserve Quarto as not promoted,
create a verified literature/source log, and keep global/live/legal claims
blocked unless future evidence changes the gate state.

### Claim Impact

- Allowed: executable work-order and queue statements.
- Still prohibited: declaring gaps closed, submission-ready, live/legal/global
  claims, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v379 in the living notebook. v380 should scaffold the manuscript sections
from v374-v379 while preserving all claim boundaries.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v378_status = json.loads((STATUS_DIR / "paper4_v378_status.json").read_text(encoding="utf-8"))
    if v378_status["next_artifact_v378"] != "paper4_v379_evidence_gap_closure_work_order.csv":
        raise RuntimeError("v379 expects v378 to route to the evidence-gap closure work order.")
    gaps = read_csv("paper4_v378_submission_readiness_gap_register.csv")
    if gaps.empty:
        raise RuntimeError("Missing v378 submission-readiness gap register.")
    open_gaps = gaps.loc[~gaps["currently_satisfied_v378"].astype(bool)].copy()

    work_order = pd.DataFrame(
        [
            {
                f"work_order_id_v{VERSION}": "draft_manuscript_sections_from_patch",
                f"source_gap_id_v{VERSION}": "full_manuscript_draft_not_created",
                f"lane_v{VERSION}": "manuscript",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": (
                    "paper4_v374_paper4_claim_language_section_draft.md;"
                    "paper4_v376_publication_integration_patch.md"
                ),
                f"output_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"executable_task_v{VERSION}": (
                    "scaffold Abstract, Methods, Results, Limitations and Appendix text "
                    "from bounded evidence"
                ),
                f"claim_boundary_v{VERSION}": "draft scaffold only",
            },
            {
                f"work_order_id_v{VERSION}": "plan_quarto_integration_without_promotion",
                f"source_gap_id_v{VERSION}": "quarto_submission_pages_not_promoted",
                f"lane_v{VERSION}": "publication_infrastructure",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": "paper4_v376_section_integration_map.csv",
                f"output_artifact_v{VERSION}": "paper4_v380_manuscript_section_scaffold.md",
                f"executable_task_v{VERSION}": (
                    "record target Quarto sections as future edits without changing curated pages"
                ),
                f"claim_boundary_v{VERSION}": "Quarto not promoted",
            },
            {
                f"work_order_id_v{VERSION}": "create_verified_literature_source_log",
                f"source_gap_id_v{VERSION}": "external_literature_source_log_missing",
                f"lane_v{VERSION}": "citations",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": True,
                f"input_artifacts_v{VERSION}": "paper4_v376_section_integration_map.csv",
                f"output_artifact_v{VERSION}": "paper4_v381_verified_literature_source_log.csv",
                f"executable_task_v{VERSION}": (
                    "search and verify related-work sources before any bibliography claim"
                ),
                f"claim_boundary_v{VERSION}": "no fabricated citations",
            },
            {
                f"work_order_id_v{VERSION}": "decide_global_solver_scope_language",
                f"source_gap_id_v{VERSION}": "full_v55_global_certificate_missing",
                f"lane_v{VERSION}": "solver_global",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
                f"output_artifact_v{VERSION}": "paper4_v382_global_solver_scope_decision.md",
                f"executable_task_v{VERSION}": (
                    "either keep global optimality prohibited or define a separate certificate route"
                ),
                f"claim_boundary_v{VERSION}": "bounded/gap evidence only",
            },
            {
                f"work_order_id_v{VERSION}": "design_targeted_source_governance_audit",
                f"source_gap_id_v{VERSION}": "source_governance_full_v55_source_exact_missing",
                f"lane_v{VERSION}": "source_governance",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": "paper4_v373_sampled_chunk_source_screen.csv",
                f"output_artifact_v{VERSION}": "paper4_v383_source_governance_audit_plan.csv",
                f"executable_task_v{VERSION}": (
                    "turn sampled zero-source-exact evidence into a targeted audit plan"
                ),
                f"claim_boundary_v{VERSION}": "diagnostic only",
            },
            {
                f"work_order_id_v{VERSION}": "acquire_external_live_holdout_panel",
                f"source_gap_id_v{VERSION}": "external_live_holdout_missing",
                f"lane_v{VERSION}": "live_validation",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": False,
                f"requires_external_dependency_v{VERSION}": True,
                f"input_artifacts_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"output_artifact_v{VERSION}": "external_holdout_panel_not_available",
                f"executable_task_v{VERSION}": "requires external/future holdout data before live claims",
                f"claim_boundary_v{VERSION}": "live deployment blocked",
            },
            {
                f"work_order_id_v{VERSION}": "obtain_online_shadow_monitoring_log",
                f"source_gap_id_v{VERSION}": "online_shadow_monitoring_missing",
                f"lane_v{VERSION}": "live_validation",
                f"priority_v{VERSION}": 2,
                f"can_execute_now_v{VERSION}": False,
                f"requires_external_dependency_v{VERSION}": True,
                f"input_artifacts_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"output_artifact_v{VERSION}": "shadow_monitoring_log_not_created",
                f"executable_task_v{VERSION}": "requires shadow deployment environment and monitoring log",
                f"claim_boundary_v{VERSION}": "production monitoring blocked",
            },
            {
                f"work_order_id_v{VERSION}": "obtain_ifrs9_contractual_coverage",
                f"source_gap_id_v{VERSION}": "ifrs9_contractual_coverage_missing",
                f"lane_v{VERSION}": "regulatory_ifrs9",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": False,
                f"requires_external_dependency_v{VERSION}": True,
                f"input_artifacts_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"output_artifact_v{VERSION}": "ifrs9_contractual_coverage_not_complete",
                f"executable_task_v{VERSION}": "requires complete contractual IFRS9 coverage and approval",
                f"claim_boundary_v{VERSION}": "IFRS9 proxy diagnostics only",
            },
            {
                f"work_order_id_v{VERSION}": "obtain_legal_fairness_review",
                f"source_gap_id_v{VERSION}": "legal_fairness_review_missing",
                f"lane_v{VERSION}": "legal_fairness",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": False,
                f"requires_external_dependency_v{VERSION}": True,
                f"input_artifacts_v{VERSION}": "paper4_v375_claim_permission_register.csv",
                f"output_artifact_v{VERSION}": "legal_fairness_review_not_created",
                f"executable_task_v{VERSION}": "requires approved legal fairness attribute review",
                f"claim_boundary_v{VERSION}": "legal fairness compliance blocked",
            },
            {
                f"work_order_id_v{VERSION}": "assemble_formal_spo_dla_review_packet",
                f"source_gap_id_v{VERSION}": "formal_spo_dla_approval_missing",
                f"lane_v{VERSION}": "formal_methods",
                f"priority_v{VERSION}": 2,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": "paper4_v375_live_gate_data_contract.csv",
                f"output_artifact_v{VERSION}": "paper4_v384_formal_spo_dla_review_packet.md",
                f"executable_task_v{VERSION}": (
                    "assemble a review packet while keeping formal claims blocked"
                ),
                f"claim_boundary_v{VERSION}": "historical audit only",
            },
            {
                f"work_order_id_v{VERSION}": "triage_full_regression_quarto_failure",
                f"source_gap_id_v{VERSION}": "full_regression_suite_not_clean",
                f"lane_v{VERSION}": "validation",
                f"priority_v{VERSION}": 1,
                f"can_execute_now_v{VERSION}": True,
                f"requires_external_dependency_v{VERSION}": False,
                f"input_artifacts_v{VERSION}": "tests/test_docs/test_paper4_living_lab_guardrails.py",
                f"output_artifact_v{VERSION}": "paper4_v385_validation_gap_triage.md",
                f"executable_task_v{VERSION}": (
                    "separate known old Quarto registration failure from current wave guardrails"
                ),
                f"claim_boundary_v{VERSION}": "targeted guardrails only",
            },
        ]
    )
    queue = (
        work_order.loc[work_order[f"can_execute_now_v{VERSION}"].astype(bool)]
        .sort_values([f"priority_v{VERSION}", f"work_order_id_v{VERSION}"])
        .reset_index(drop=True)
    )
    queue[f"queue_rank_v{VERSION}"] = range(1, len(queue) + 1)
    external = work_order.loc[
        ~work_order[f"can_execute_now_v{VERSION}"].astype(bool)
        | work_order[f"requires_external_dependency_v{VERSION}"].astype(bool)
    ].copy()
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "work_order_does_not_close_gaps",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(open_gaps)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "future artifacts must close gaps",
            },
            {
                f"blocker_id_v{VERSION}": "external_dependencies_remain",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    external[f"requires_external_dependency_v{VERSION}"].astype(bool).sum()
                ),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "external data/review still unavailable",
            },
            {
                f"blocker_id_v{VERSION}": "submission_ready_claim_still_blocked",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(v378_status["open_gap_rows_v378"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "submission-ready claim remains blocked",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v379_evidence_gap_work_order_created",
                "allowed": True,
                "artifact": "paper4_v379_evidence_gap_closure_work_order.csv",
                "boundary": "work order only",
            },
            {
                "claim_id": "v379_execution_queue_created",
                "allowed": True,
                "artifact": "paper4_v379_execution_queue.csv",
                "boundary": "queue only; tasks not yet complete",
            },
            {
                "claim_id": "v379_submission_gaps_closed",
                "allowed": False,
                "artifact": "paper4_v379_claim_blockers.csv",
                "boundary": "work order does not close gaps",
            },
            {
                "claim_id": "v379_live_legal_or_global_claim_authorized",
                "allowed": False,
                "artifact": "paper4_v379_claim_blockers.csv",
                "boundary": "v375 gates remain blocked",
            },
            {
                "claim_id": "v379_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v379_claim_blockers.csv",
                "boundary": "Paper Estrella remains protected",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v379_evidence_gap_closure_work_order.csv", work_order)
    write_csv(TABLE_DIR / "paper4_v379_execution_queue.csv", queue)
    write_csv(TABLE_DIR / "paper4_v379_blocked_external_dependencies.csv", external)
    write_csv(TABLE_DIR / "paper4_v379_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v379_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    executable_now = int(work_order[f"can_execute_now_v{VERSION}"].astype(bool).sum())
    external_blocked = int((~work_order[f"can_execute_now_v{VERSION}"].astype(bool)).sum())
    highest_priority_executable = int(
        (
            work_order[f"priority_v{VERSION}"].astype(int).eq(1)
            & work_order[f"can_execute_now_v{VERSION}"].astype(bool)
        ).sum()
    )
    status = {
        "phase": "v379_evidence_gap_closure_work_order",
        "schema_version": "2026-05-17.379",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_gap_register_version_v379": PRIOR_GAP_REGISTER_VERSION,
        "prior_v378_gap_register_rows_v379": int(v378_status["gap_register_rows_v378"]),
        "prior_v378_open_gap_rows_v379": int(v378_status["open_gap_rows_v378"]),
        "work_order_rows_v379": int(len(work_order)),
        "executable_now_rows_v379": executable_now,
        "external_blocked_rows_v379": external_blocked,
        "execution_queue_rows_v379": int(len(queue)),
        "blocked_dependency_rows_v379": int(len(external)),
        "highest_priority_executable_rows_v379": highest_priority_executable,
        "claim_blocker_rows_v379": int(len(blockers)),
        "claim_matrix_rows_v379": int(len(claim_matrix)),
        "submission_gaps_closed_v379": False,
        "submission_ready_claim_allowed_v379": False,
        "bounded_living_lab_language_allowed_v379": True,
        "reproducibility_appendix_language_allowed_v379": True,
        "strict_live_deployment_language_allowed_v379": False,
        "contractual_or_legal_language_allowed_v379": False,
        "global_optimality_language_allowed_v379": False,
        "working_champion_claim_allowed_v379": False,
        "paper1_promotion_allowed_v379": False,
        "paper4_working_champion_changed_v379": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v379": NEXT_ARTIFACT,
        "claim_boundary": (
            "v379 creates an evidence-gap work order; open gaps and all stronger "
            "live/legal/global/final claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v379_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v379": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
