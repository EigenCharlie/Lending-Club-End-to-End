#!/usr/bin/env python3
"""Build Paper 4 v383 targeted source-governance audit plan artifacts."""

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

VERSION = 383
PRIOR_SCOPE_VERSION = 382
PRIOR_SOURCE_DIAGNOSTIC_VERSION = 371
PRIOR_PREFILTER_VERSION = 372
PRIOR_STOP_RULE_VERSION = 373
NEXT_VERSION = 384
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_formal_spo_dla_review_packet.md"
AUDIT_PLAN_MD = NOTEBOOK.parent / "paper4_v383_source_governance_audit_plan.md"


def _family_priority(family_retention: pd.DataFrame) -> pd.DataFrame:
    if family_retention.empty:
        raise RuntimeError("Missing v371 source family retention table.")
    out = family_retention.rename(
        columns={
            "source_family_v371": "source_family_v383",
            "budget_return_feasible_rows_v371": "budget_return_feasible_rows_v383",
            "family_source_feasible_rows_v371": "family_source_feasible_rows_v383",
            "family_retention_share_v371": "family_retention_share_v383",
            "binding_rank_v371": "binding_rank_v383",
            "blocker_class_v371": "blocker_class_v383",
            "claim_boundary_v371": "claim_boundary_v383",
        }
    ).copy()
    out["audit_priority_v383"] = out["binding_rank_v383"].astype(int)
    out["audit_action_v383"] = out["source_family_v383"].map(
        {
            "grade": "audit cap slack, add/drop direction and grade-A pressure first",
            "score_decile": "audit secondary tightness after grade is explained",
            "dti_band": "retain as nonbinding control",
            "income_band": "retain as nonbinding control",
            "period": "retain as nonbinding control",
            "state_top20": "retain as nonbinding control",
        }
    )
    out["claim_boundary_v383"] = "family priority diagnostic only"
    return out[
        [
            "source_family_v383",
            "budget_return_feasible_rows_v383",
            "family_source_feasible_rows_v383",
            "family_retention_share_v383",
            "binding_rank_v383",
            "blocker_class_v383",
            "audit_priority_v383",
            "audit_action_v383",
            "claim_boundary_v383",
        ]
    ]


def _audit_plan(v371_status: dict[str, Any], v372_status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "audit_task_id_v383": "audit_grade_a_cap_slack_boundary",
            "priority_v383": 1,
            "source_family_v383": "grade",
            "source_id_v383": "A",
            "input_artifact_v383": "paper4_v371_tight_source_blockers.csv",
            "diagnostic_basis_v383": "grade=A has the tightest slack and zero pass rows",
            "planned_check_v383": "recompute cap slack, tolerance and binding status for grade=A",
            "success_condition_v383": "document whether cap math or true portfolio pressure explains collapse",
            "claim_boundary_v383": "audit only; no cap relaxation",
        },
        {
            "audit_task_id_v383": "audit_grade_a_flow_direction",
            "priority_v383": 1,
            "source_family_v383": "grade",
            "source_id_v383": "A",
            "input_artifact_v383": "paper4_v371_source_pair_flow_diagnostics.csv",
            "diagnostic_basis_v383": "grade=A source pressure dominates budget+return rows",
            "planned_check_v383": "separate add-tight, drop-tight and neutral flows for grade=A",
            "success_condition_v383": "identify whether entering candidates add too much grade=A exposure",
            "claim_boundary_v383": "flow audit only",
        },
        {
            "audit_task_id_v383": "audit_grade_a_relief_counterfactual",
            "priority_v383": 1,
            "source_family_v383": "grade",
            "source_id_v383": "A",
            "input_artifact_v383": "paper4_v372_grade_a_source_relief_prefilter.csv",
            "diagnostic_basis_v383": "grade-A relief has zero return-improving budget rows",
            "planned_check_v383": "quantify the return cost of any feasible relief-style move",
            "success_condition_v383": "record whether source repair is economically dominated",
            "claim_boundary_v383": "counterfactual only; no candidate apply",
        },
        {
            "audit_task_id_v383": "audit_score_decile_secondary_tightness",
            "priority_v383": 2,
            "source_family_v383": "score_decile",
            "source_id_v383": "0",
            "input_artifact_v383": "paper4_v371_tight_source_blockers.csv",
            "diagnostic_basis_v383": "score_decile=0 is secondary and passes 6023 rows before grade blocks",
            "planned_check_v383": "rerun secondary source pass after grade explanation is isolated",
            "success_condition_v383": "determine whether score_decile remains binding after grade treatment",
            "claim_boundary_v383": "secondary audit only",
        },
        {
            "audit_task_id_v383": "audit_sampled_chunk_representativeness",
            "priority_v383": 2,
            "source_family_v383": "all",
            "source_id_v383": "sampled_chunks",
            "input_artifact_v383": "paper4_v373_sampled_chunk_source_screen.csv",
            "diagnostic_basis_v383": "eight sampled chunks produced zero source-exact rows",
            "planned_check_v383": "compare sampled chunks by grade-A pressure and source-exact collapse",
            "success_condition_v383": "justify targeted audit before any new chunk run",
            "claim_boundary_v383": "sample audit only",
        },
        {
            "audit_task_id_v383": "audit_source_cap_contract",
            "priority_v383": 3,
            "source_family_v383": "all",
            "source_id_v383": "source_cap_contract",
            "input_artifact_v383": "paper4_v80_full_pool_milp_gap_source_summary.csv",
            "diagnostic_basis_v383": "source caps are the active feasibility bottleneck",
            "planned_check_v383": "document cap source, tolerance, rounding and family definitions",
            "success_condition_v383": "make source-cap governance auditable without changing caps",
            "claim_boundary_v383": "governance documentation only",
        },
        {
            "audit_task_id_v383": "audit_global_solver_implication",
            "priority_v383": 3,
            "source_family_v383": "all",
            "source_id_v383": "global_scope",
            "input_artifact_v383": "paper4_v382_global_solver_scope_decision.csv",
            "diagnostic_basis_v383": "v382 blocks global optimality and routes to source audit",
            "planned_check_v383": "map audit findings to future certificate requirements",
            "success_condition_v383": "keep bounded solver scope and identify certificate preconditions",
            "claim_boundary_v383": "future-route planning only",
        },
    ]
    out = pd.DataFrame(rows)
    out["primary_blocker_family_v383"] = v371_status["primary_blocker_family_v371"]
    out["grade_a_relief_return_improving_rows_v383"] = int(
        v372_status["grade_a_relief_return_improving_rows_v372"]
    )
    return out


def _blocker_evidence(
    v371_status: dict[str, Any],
    v372_status: dict[str, Any],
    v373_status: dict[str, Any],
    family_priority: pd.DataFrame,
) -> pd.DataFrame:
    grade_row = family_priority.loc[family_priority["source_family_v383"].eq("grade")].iloc[0]
    score_row = family_priority.loc[
        family_priority["source_family_v383"].eq("score_decile")
    ].iloc[0]
    pair_flow = read_csv("paper4_v371_source_pair_flow_diagnostics.csv")
    grade_pressure = float(
        pair_flow.loc[
            pair_flow["flow_category_v371"].eq("source_pressure_drop_other_add_tight")
            & pair_flow["source_family_v371"].eq("grade"),
            "share_of_budget_return_rows_v371",
        ].iloc[0]
    )
    rows = [
        {
            "evidence_id_v383": "grade_family_retention_zero",
            "value_v383": float(grade_row["family_retention_share_v383"]),
            "evidence_artifact_v383": "paper4_v371_source_family_retention.csv",
            "interpretation_v383": "grade is the primary family collapse",
            "claim_boundary_v383": "diagnostic value only",
        },
        {
            "evidence_id_v383": "score_decile_secondary_retention",
            "value_v383": float(score_row["family_retention_share_v383"]),
            "evidence_artifact_v383": "paper4_v371_source_family_retention.csv",
            "interpretation_v383": "score_decile is secondary and partially passes",
            "claim_boundary_v383": "diagnostic value only",
        },
        {
            "evidence_id_v383": "grade_a_pressure_share",
            "value_v383": grade_pressure,
            "evidence_artifact_v383": "paper4_v371_source_pair_flow_diagnostics.csv",
            "interpretation_v383": "budget+return moves mostly add pressure to grade=A",
            "claim_boundary_v383": "flow diagnostic only",
        },
        {
            "evidence_id_v383": "grade_a_relief_return_improving_rows",
            "value_v383": float(v372_status["grade_a_relief_return_improving_rows_v372"]),
            "evidence_artifact_v383": "paper4_v372_grade_a_source_relief_prefilter.csv",
            "interpretation_v383": "no return-improving grade-A relief rows were found",
            "claim_boundary_v383": "counterfactual diagnostic only",
        },
        {
            "evidence_id_v383": "sampled_total_source_exact_rows",
            "value_v383": float(v373_status["sampled_total_source_exact_rows_v373"]),
            "evidence_artifact_v383": "paper4_v373_sampled_chunk_source_screen.csv",
            "interpretation_v383": "sampled chunks all collapse to zero source-exact rows",
            "claim_boundary_v383": "sample diagnostic only",
        },
        {
            "evidence_id_v383": "sampled_chunk_count",
            "value_v383": float(v373_status["sampled_chunk_count_v373"]),
            "evidence_artifact_v383": "paper4_v373_full_v55_chunk_002_or_stop_rule.csv",
            "interpretation_v383": "sample includes representative chunk IDs before audit planning",
            "claim_boundary_v383": "sample diagnostic only",
        },
    ]
    return pd.DataFrame(rows)


def _claim_blockers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v383": "audit_plan_is_diagnostic",
                "blocking_v383": True,
                "evidence_count_v383": 7,
                "required_next_artifact_v383": NEXT_ARTIFACT,
                "claim_boundary_v383": "audit plan does not repair source governance",
            },
            {
                "blocker_id_v383": "source_cap_relaxation_not_authorized",
                "blocking_v383": True,
                "evidence_count_v383": 1,
                "required_next_artifact_v383": "future_source_cap_approval_or_counterfactual",
                "claim_boundary_v383": "no cap change or policy relaxation is made",
            },
            {
                "blocker_id_v383": "global_solver_claims_still_blocked",
                "blocking_v383": True,
                "evidence_count_v383": 1,
                "required_next_artifact_v383": "future_full_v55_certificate_pack",
                "claim_boundary_v383": "v382 global blockers remain active",
            },
            {
                "blocker_id_v383": "paper4_final_promotion_forbidden",
                "blocking_v383": True,
                "evidence_count_v383": 1,
                "required_next_artifact_v383": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v383": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v383_source_governance_audit_plan_created",
                "allowed": True,
                "artifact": "paper4_v383_source_governance_audit_plan.csv",
                "boundary": "audit plan only",
            },
            {
                "claim_id": "v383_family_priority_matrix_created",
                "allowed": True,
                "artifact": "paper4_v383_family_audit_priority_matrix.csv",
                "boundary": "diagnostic prioritization only",
            },
            {
                "claim_id": "v383_zero_source_exact_evidence_can_be_reported",
                "allowed": True,
                "artifact": "paper4_v383_blocker_evidence_map.csv",
                "boundary": "report source collapse as blocker evidence",
            },
            {
                "claim_id": "v383_source_governance_repaired_or_relaxed",
                "allowed": False,
                "artifact": "paper4_v383_claim_blockers.csv",
                "boundary": "no repair, relaxation or candidate apply",
            },
            {
                "claim_id": "v383_global_or_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v383_claim_blockers.csv",
                "boundary": "v382 global blockers remain",
            },
            {
                "claim_id": "v383_working_champion_or_final_promotion",
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
                "claim": "v383 creates a targeted source-governance audit plan.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v383_source_governance_audit_plan.csv"
                ),
                "boundary": "Audit plan only; no cap relaxation or solver proof.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v383 prioritizes grade=A and score_decile=0 source blockers.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v383_family_audit_priority_matrix.csv"
                ),
                "boundary": "Diagnostic prioritization only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v383 repairs source governance or relaxes source caps.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v383_claim_blockers.csv"
                ),
                "boundary": "No source cap or candidate selection change is made.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v383 proves full-v55 global or integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v383_claim_blockers.csv"
                ),
                "boundary": "Audit planning cannot close v382 solver blockers.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v383 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v383_claim_blockers.csv"
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
                "lane": "Formal Methods/DLA",
                "executable_item": (
                    "v383 turns sampled zero source-exact evidence into a targeted "
                    "source-governance audit plan without source-cap relaxation."
                ),
                "status": "source_governance_audit_plan_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v384 assembles the formal SPO/DLA review packet while formal claims stay blocked"
                ),
                "last_wave": "v383",
                "execution_result": "targeted_source_audit_plan_created_no_relaxation",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v383")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _audit_plan_markdown(status: dict[str, Any], audit_plan: pd.DataFrame) -> str:
    task_lines = "\n".join(
        (
            f"- P{int(row['priority_v383'])} `{row['audit_task_id_v383']}`: "
            f"{row['planned_check_v383']}"
        )
        for _, row in audit_plan.iterrows()
    )
    return f"""# Paper 4 Source Governance Audit Plan v383

Generated: {status["generated_at_utc"]}

v383 converts the v371-v373 source collapse into a targeted audit plan. It does
not relax source caps, apply a repair candidate, restart blind chunking or change
the Paper 4 solver claim scope.

## Audit Tasks

{task_lines}

## Evidence Summary

- Primary blocker family: `{status["primary_blocker_family_v383"]}`.
- Secondary blocker family: `{status["secondary_blocker_family_v383"]}`.
- Grade family retention share: `{status["grade_family_retention_share_v383"]}`.
- Grade-A relief return-improving rows: `{status["grade_a_relief_return_improving_rows_v383"]}`.
- Sampled source-exact rows: `{status["sampled_total_source_exact_rows_v383"]}`.

## Required Caveat

This is an audit plan, not a repair. It must not be used to claim source-cap
relaxation, global optimality, integer optimality, champion replacement or final
Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v383"]}` while keeping formal SPO/DLA claims
blocked unless the review packet explicitly satisfies them.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V383_SOURCE_GOVERNANCE_AUDIT_PLAN_START -->"
    end = "<!-- V383_SOURCE_GOVERNANCE_AUDIT_PLAN_END -->"
    block = f"""
{start}

## Wave v383: Source Governance Audit Plan

Generated: {status["generated_at_utc"]}

### Objective

v383 executes the source-governance work order from v379/v382 by turning sampled
zero source-exact evidence into a targeted audit plan. It does not relax caps,
apply a candidate, restart blind chunking or change global solver claims.

### Results

- Audit task rows:
  `{status["audit_plan_rows_v383"]}`.
- Family priority rows:
  `{status["family_priority_rows_v383"]}`.
- Primary blocker family:
  `{status["primary_blocker_family_v383"]}`.
- Secondary blocker family:
  `{status["secondary_blocker_family_v383"]}`.
- Grade family retention share:
  `{status["grade_family_retention_share_v383"]}`.
- Grade-A pressure share:
  `{status["grade_a_pressure_share_v383"]}`.
- Source governance repaired:
  `{status["source_governance_repaired_v383"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v383"]}`.

### Interpretation

The next useful work is not more blind chunking. It is a targeted source audit:
grade=A first, score_decile=0 second, then cap-contract documentation and
future certificate-route implications.

### Claim Impact

- Allowed: targeted audit plan, family priority matrix and zero source-exact
  blocker evidence.
- Still prohibited: source cap relaxation, source governance repair, global or
  integer optimality, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v383 in the living notebook. v384 should assemble the formal SPO/DLA review
packet while keeping formal claims blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v382_status = json.loads((STATUS_DIR / "paper4_v382_status.json").read_text(encoding="utf-8"))
    if v382_status["next_artifact_v382"] != "paper4_v383_source_governance_audit_plan.csv":
        raise RuntimeError("v383 expects v382 to route to the source governance audit plan.")
    v371_status = json.loads((STATUS_DIR / "paper4_v371_status.json").read_text(encoding="utf-8"))
    v372_status = json.loads((STATUS_DIR / "paper4_v372_status.json").read_text(encoding="utf-8"))
    v373_status = json.loads((STATUS_DIR / "paper4_v373_status.json").read_text(encoding="utf-8"))

    family_priority = _family_priority(read_csv("paper4_v371_source_family_retention.csv"))
    audit_plan = _audit_plan(v371_status, v372_status)
    blocker_evidence = _blocker_evidence(v371_status, v372_status, v373_status, family_priority)
    blockers = _claim_blockers()
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v383_family_audit_priority_matrix.csv", family_priority)
    write_csv(TABLE_DIR / "paper4_v383_source_governance_audit_plan.csv", audit_plan)
    write_csv(TABLE_DIR / "paper4_v383_blocker_evidence_map.csv", blocker_evidence)
    write_csv(TABLE_DIR / "paper4_v383_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v383_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    evidence_map = dict(
        zip(blocker_evidence["evidence_id_v383"], blocker_evidence["value_v383"], strict=False)
    )
    status = {
        "phase": "v383_source_governance_audit_plan",
        "schema_version": "2026-05-17.383",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_scope_version_v383": PRIOR_SCOPE_VERSION,
        "prior_source_diagnostic_version_v383": PRIOR_SOURCE_DIAGNOSTIC_VERSION,
        "prior_prefilter_version_v383": PRIOR_PREFILTER_VERSION,
        "prior_stop_rule_version_v383": PRIOR_STOP_RULE_VERSION,
        "audit_plan_rows_v383": int(len(audit_plan)),
        "family_priority_rows_v383": int(len(family_priority)),
        "blocker_evidence_rows_v383": int(len(blocker_evidence)),
        "claim_blocker_rows_v383": int(len(blockers)),
        "claim_matrix_rows_v383": int(len(claim_matrix)),
        "primary_blocker_family_v383": v371_status["primary_blocker_family_v371"],
        "secondary_blocker_family_v383": v371_status["secondary_blocker_family_v371"],
        "grade_family_retention_share_v383": float(evidence_map["grade_family_retention_zero"]),
        "score_decile_family_retention_share_v383": float(
            evidence_map["score_decile_secondary_retention"]
        ),
        "grade_a_pressure_share_v383": float(evidence_map["grade_a_pressure_share"]),
        "grade_a_relief_return_improving_rows_v383": int(
            v372_status["grade_a_relief_return_improving_rows_v372"]
        ),
        "sampled_chunk_count_v383": int(v373_status["sampled_chunk_count_v373"]),
        "sampled_total_source_exact_rows_v383": int(
            v373_status["sampled_total_source_exact_rows_v373"]
        ),
        "audit_plan_created_v383": True,
        "source_cap_relaxation_authorized_v383": False,
        "source_governance_repaired_v383": False,
        "blind_chunking_restarted_v383": False,
        "global_optimality_language_allowed_v383": False,
        "full_universe_integer_optimality_claim_allowed_v383": False,
        "working_champion_claim_allowed_v383": False,
        "paper1_promotion_allowed_v383": False,
        "paper4_working_champion_changed_v383": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "audit_plan_artifact_v383": (
            "reports/paper_material/paper4/tables/"
            "paper4_v383_source_governance_audit_plan.csv"
        ),
        "next_artifact_v383": NEXT_ARTIFACT,
        "claim_boundary": (
            "v383 creates a targeted source-governance audit plan; source cap relaxation, "
            "repair, global/integer optimality and final promotion remain blocked"
        ),
    }
    AUDIT_PLAN_MD.write_text(_audit_plan_markdown(status, audit_plan), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v383_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v383": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
