#!/usr/bin/env python3
"""Build Paper 4 v382 global-solver scope decision artifacts."""

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

VERSION = 382
PRIOR_SOURCE_VERSION = 381
PRIOR_GAP_VERSION = 363
PRIOR_STOP_RULE_VERSION = 373
NEXT_VERSION = 383
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_source_governance_audit_plan.csv"
DECISION_MD = NOTEBOOK.parent / "paper4_v382_global_solver_scope_decision.md"


def _scope_decision(v363_status: dict[str, Any], v373_status: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v382": "global_solver_scope_decision",
                "selected_scope_v382": "bounded_solver_frontier_with_gap_certificate_only",
                "decision_v382": (
                    "Keep global optimality prohibited and write the solver result as a "
                    "bounded frontier plus negative/gap evidence."
                ),
                "primary_evidence_v382": (
                    "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv;"
                    "paper4_v373_full_v55_chunk_002_or_stop_rule.csv"
                ),
                "v71_improving_omitted_columns_v382": int(
                    v363_status["v71_improving_omitted_columns_v363"]
                ),
                "sampled_chunks_with_source_exact_rows_v382": int(
                    v373_status["sampled_chunks_with_source_exact_rows_v373"]
                ),
                "valid_full_v55_dual_bound_certificate_v382": False,
                "separate_certificate_route_required_v382": True,
                "global_optimality_language_allowed_v382": False,
                "claim_boundary_v382": "bounded/gap evidence only",
            }
        ]
    )


def _solver_scope_register(v363_status: dict[str, Any], v373_status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "scope_id_v382": "bounded_fourth_order_no_entry",
            "allowed_v382": True,
            "evidence_artifact_v382": "paper4_v361_v353_fourth_order_or_full_dual_bound.csv",
            "evidence_count_v382": int(v363_status["v361_ordered_fourth_order_rows_v363"]),
            "allowed_language_v382": "bounded fourth-order no-entry screen",
            "prohibited_language_v382": "all-column or global termination",
            "claim_boundary_v382": "bounded subset only",
        },
        {
            "scope_id_v382": "v363_negative_gap_certificate",
            "allowed_v382": True,
            "evidence_artifact_v382": "paper4_v363_v353_full_dual_bound_or_gap_certificate.csv",
            "evidence_count_v382": int(v363_status["v71_improving_omitted_columns_v363"]),
            "allowed_language_v382": "full-v55 global certificate remains blocked by gap evidence",
            "prohibited_language_v382": "valid full-v55 dual-bound certificate",
            "claim_boundary_v382": "negative/gap certificate only",
        },
        {
            "scope_id_v382": "v373_sampled_source_stop_rule",
            "allowed_v382": True,
            "evidence_artifact_v382": "paper4_v373_sampled_chunk_source_screen.csv",
            "evidence_count_v382": int(v373_status["sampled_chunk_count_v373"]),
            "allowed_language_v382": "blind chunking stopped after sampled zero source-exact evidence",
            "prohibited_language_v382": "sampled chunks prove all chunks",
            "claim_boundary_v382": "sampled diagnostic only",
        },
        {
            "scope_id_v382": "full_v55_global_optimality",
            "allowed_v382": False,
            "evidence_artifact_v382": "paper4_v363_dual_bound_requirement_register.csv",
            "evidence_count_v382": int(v363_status["requirements_met_v363"]),
            "allowed_language_v382": "none",
            "prohibited_language_v382": "full-v55 global optimality or solver termination",
            "claim_boundary_v382": "five v363 requirements remain open",
        },
        {
            "scope_id_v382": "full_universe_integer_optimality",
            "allowed_v382": False,
            "evidence_artifact_v382": "paper4_v363_claim_blockers.csv",
            "evidence_count_v382": int(v363_status["integer_certificate_available_v363"]),
            "allowed_language_v382": "none",
            "prohibited_language_v382": "whole-loan full-universe integer optimality",
            "claim_boundary_v382": "integer certificate missing",
        },
        {
            "scope_id_v382": "champion_replacement_or_final_promotion",
            "allowed_v382": False,
            "evidence_artifact_v382": "paper4_final_promotion_gate_not_created",
            "evidence_count_v382": int(FORBIDDEN_FINAL_PROMOTION.exists()),
            "allowed_language_v382": "none",
            "prohibited_language_v382": "Paper Estrella replacement or final Paper 4 promotion",
            "claim_boundary_v382": "promotion forbidden",
        },
    ]
    return pd.DataFrame(rows)


def _certificate_route_requirements(requirements: pd.DataFrame) -> pd.DataFrame:
    if requirements.empty:
        raise RuntimeError("Missing v363 dual-bound requirement register.")
    route = requirements.rename(
        columns={
            "requirement_id_v363": "requirement_id_v382",
            "met_v363": "met_now_v382",
            "evidence_artifact_v363": "current_evidence_artifact_v382",
            "required_next_artifact_v363": "required_next_artifact_v382",
            "claim_boundary_v363": "claim_boundary_v382",
        }
    ).copy()
    route["route_scope_v382"] = "future_full_v55_certificate_route"
    route["can_enable_global_claim_v382"] = route["met_now_v382"].astype(bool)
    route.loc[~route["met_now_v382"].astype(bool), "can_enable_global_claim_v382"] = False
    route["global_claim_requires_all_rows_met_v382"] = True
    route["claim_boundary_v382"] = route["claim_boundary_v382"].astype(str)
    return route[
        [
            "requirement_id_v382",
            "route_scope_v382",
            "met_now_v382",
            "current_evidence_artifact_v382",
            "required_next_artifact_v382",
            "can_enable_global_claim_v382",
            "global_claim_requires_all_rows_met_v382",
            "claim_boundary_v382",
        ]
    ]


def _claim_blockers(v363_status: dict[str, Any], requirements: pd.DataFrame) -> pd.DataFrame:
    open_requirements = int((~requirements["met_now_v382"].astype(bool)).sum())
    return pd.DataFrame(
        [
            {
                "blocker_id_v382": "full_v55_requirements_open",
                "blocking_v382": True,
                "evidence_count_v382": open_requirements,
                "required_next_artifact_v382": "future_full_v55_certificate_pack",
                "claim_boundary_v382": "all v363 requirements must be met before global language",
            },
            {
                "blocker_id_v382": "v71_negative_reduced_cost_persists",
                "blocking_v382": True,
                "evidence_count_v382": int(v363_status["v71_improving_omitted_columns_v363"]),
                "required_next_artifact_v382": "future_all_column_pricing_termination",
                "claim_boundary_v382": "negative reduced-cost rows block termination",
            },
            {
                "blocker_id_v382": "integer_certificate_missing",
                "blocking_v382": True,
                "evidence_count_v382": int(not v363_status["integer_certificate_available_v363"]),
                "required_next_artifact_v382": "future_integer_gap_certificate",
                "claim_boundary_v382": "continuous/bounded evidence is not integer proof",
            },
            {
                "blocker_id_v382": "source_governance_audit_needed",
                "blocking_v382": True,
                "evidence_count_v382": 1,
                "required_next_artifact_v382": NEXT_ARTIFACT,
                "claim_boundary_v382": "v383 must turn sampled zero source-exact evidence into an audit plan",
            },
            {
                "blocker_id_v382": "paper4_final_promotion_forbidden",
                "blocking_v382": True,
                "evidence_count_v382": 1,
                "required_next_artifact_v382": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v382": "Paper Estrella replacement and final Paper 4 remain prohibited",
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v382_bounded_solver_scope_selected",
                "allowed": True,
                "artifact": "paper4_v382_global_solver_scope_decision.md",
                "boundary": "bounded/gap solver language only",
            },
            {
                "claim_id": "v382_separate_certificate_route_defined",
                "allowed": True,
                "artifact": "paper4_v382_certificate_route_requirements.csv",
                "boundary": "future route, not current proof",
            },
            {
                "claim_id": "v382_negative_gap_evidence_can_be_reported",
                "allowed": True,
                "artifact": "paper4_v382_solver_claim_scope_register.csv",
                "boundary": "report blockers and gaps explicitly",
            },
            {
                "claim_id": "v382_full_v55_global_optimality",
                "allowed": False,
                "artifact": "paper4_v382_claim_blockers.csv",
                "boundary": "open requirements and negative reduced costs remain",
            },
            {
                "claim_id": "v382_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v382_claim_blockers.csv",
                "boundary": "integer certificate missing",
            },
            {
                "claim_id": "v382_working_champion_or_final_promotion",
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
                "claim": "v382 selects bounded solver-frontier and gap-only language.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/notes/"
                    "paper4_v382_global_solver_scope_decision.md"
                ),
                "boundary": "Bounded solver scope only; no global optimality claim.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v382 defines a separate full-v55 certificate route.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v382_certificate_route_requirements.csv"
                ),
                "boundary": "Future route only; requirements remain open.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v382 authorizes full-v55 global or integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v382_claim_blockers.csv"
                ),
                "boundary": "Open requirements, negative reduced costs and missing integer certificate block it.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v382 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v382_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v382 fixes the solver claim scope as bounded frontier plus gap evidence "
                    "and keeps full-v55 global optimality prohibited."
                ),
                "status": "global_solver_scope_decision_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v383 turns sampled zero source-exact evidence into a targeted source audit plan"
                ),
                "last_wave": "v382",
                "execution_result": "bounded_solver_scope_selected_global_optimality_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v382")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _decision_markdown(status: dict[str, Any], requirements: pd.DataFrame) -> str:
    open_rows = requirements.loc[~requirements["met_now_v382"].astype(bool)]
    open_lines = "\n".join(
        (
            f"- `{row['requirement_id_v382']}`: {row['claim_boundary_v382']} "
            f"(next: `{row['required_next_artifact_v382']}`)."
        )
        for _, row in open_rows.iterrows()
    )
    return f"""# Paper 4 Global Solver Scope Decision v382

Generated: {status["generated_at_utc"]}

## Decision

Paper 4 should report the solver lane as a bounded frontier plus negative/gap
evidence. It must keep full-v55 global optimality and full-universe integer
optimality prohibited.

## Why

- v71/v363 still record `{status["v71_improving_omitted_columns_v382"]}`
  improving omitted columns.
- v363 has `{status["certificate_requirements_open_v382"]}` open certificate
  requirements.
- v373 sampled `{status["sampled_chunk_count_v382"]}` chunks and found
  `{status["sampled_total_source_exact_rows_v382"]}` source-exact rows.

## Open Certificate Route

{open_lines}

## Required Caveat

This decision supports bounded/gap manuscript language only. It must not be used
to claim global optimality, integer optimality, live deployment, legal/IFRS9
compliance, Paper Estrella replacement or final Paper 4 promotion.

## Next Executable Wave

Build `{status["next_artifact_v382"]}` from the v373 sampled source-screen
evidence.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V382_GLOBAL_SOLVER_SCOPE_DECISION_START -->"
    end = "<!-- V382_GLOBAL_SOLVER_SCOPE_DECISION_END -->"
    block = f"""
{start}

## Wave v382: Global Solver Scope Decision

Generated: {status["generated_at_utc"]}

### Objective

v382 executes the global-solver work order from v379: either authorize a
separate certificate route or keep global optimality prohibited. The decision is
to publish only bounded/gap solver language now and require a separate full-v55
certificate route for any future global claim.

### Results

- Selected scope:
  `{status["selected_scope_v382"]}`.
- Certificate requirements open:
  `{status["certificate_requirements_open_v382"]}`.
- v71 improving omitted columns:
  `{status["v71_improving_omitted_columns_v382"]}`.
- Sampled chunks with source-exact rows:
  `{status["sampled_chunks_with_source_exact_rows_v382"]}`.
- Bounded solver-frontier language allowed:
  `{status["bounded_solver_frontier_language_allowed_v382"]}`.
- Global optimality language allowed:
  `{status["global_optimality_language_allowed_v382"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v382"]}`.

### Interpretation

The valuable Paper 4 result is now cleaner: bounded no-entry evidence and
solver-gap transparency are publishable as audit evidence, while full-v55 global
or integer optimality remains explicitly false.

### Claim Impact

- Allowed: bounded solver-frontier language, negative/gap evidence and a future
  certificate route.
- Still prohibited: full-v55 global optimality, full-universe integer
  optimality, champion replacement and final promotion.

### Quarto Promotion Decision

Keep v382 in the living notebook. v383 should turn the zero source-exact sampled
chunk evidence into a targeted source-governance audit plan.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v381_status = json.loads((STATUS_DIR / "paper4_v381_status.json").read_text(encoding="utf-8"))
    if v381_status["next_artifact_v381"] != "paper4_v382_global_solver_scope_decision.md":
        raise RuntimeError("v382 expects v381 to route to the global solver scope decision.")
    v363_status = json.loads((STATUS_DIR / "paper4_v363_status.json").read_text(encoding="utf-8"))
    v373_status = json.loads((STATUS_DIR / "paper4_v373_status.json").read_text(encoding="utf-8"))
    requirements_v363 = read_csv("paper4_v363_dual_bound_requirement_register.csv")

    decision = _scope_decision(v363_status, v373_status)
    scope_register = _solver_scope_register(v363_status, v373_status)
    route_requirements = _certificate_route_requirements(requirements_v363)
    blockers = _claim_blockers(v363_status, route_requirements)
    claim_matrix = _claim_matrix()

    write_csv(TABLE_DIR / "paper4_v382_global_solver_scope_decision.csv", decision)
    write_csv(TABLE_DIR / "paper4_v382_solver_claim_scope_register.csv", scope_register)
    write_csv(TABLE_DIR / "paper4_v382_certificate_route_requirements.csv", route_requirements)
    write_csv(TABLE_DIR / "paper4_v382_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v382_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    open_requirements = int((~route_requirements["met_now_v382"].astype(bool)).sum())
    met_requirements = int(route_requirements["met_now_v382"].astype(bool).sum())
    status = {
        "phase": "v382_global_solver_scope_decision",
        "schema_version": "2026-05-17.382",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_source_log_version_v382": PRIOR_SOURCE_VERSION,
        "prior_gap_version_v382": PRIOR_GAP_VERSION,
        "prior_stop_rule_version_v382": PRIOR_STOP_RULE_VERSION,
        "scope_decision_rows_v382": int(len(decision)),
        "solver_scope_register_rows_v382": int(len(scope_register)),
        "certificate_route_requirement_rows_v382": int(len(route_requirements)),
        "certificate_requirements_met_v382": met_requirements,
        "certificate_requirements_open_v382": open_requirements,
        "claim_blocker_rows_v382": int(len(blockers)),
        "claim_matrix_rows_v382": int(len(claim_matrix)),
        "selected_scope_v382": decision.iloc[0]["selected_scope_v382"],
        "separate_certificate_route_required_v382": True,
        "bounded_solver_frontier_language_allowed_v382": True,
        "negative_gap_language_allowed_v382": True,
        "full_v55_global_certificate_missing_v382": True,
        "valid_full_v55_dual_bound_certificate_v382": False,
        "global_optimality_language_allowed_v382": False,
        "full_universe_integer_optimality_claim_allowed_v382": False,
        "working_champion_claim_allowed_v382": False,
        "paper1_promotion_allowed_v382": False,
        "paper4_working_champion_changed_v382": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "v71_improving_omitted_columns_v382": int(
            v363_status["v71_improving_omitted_columns_v363"]
        ),
        "sampled_chunk_count_v382": int(v373_status["sampled_chunk_count_v373"]),
        "sampled_chunks_with_source_exact_rows_v382": int(
            v373_status["sampled_chunks_with_source_exact_rows_v373"]
        ),
        "sampled_total_source_exact_rows_v382": int(
            v373_status["sampled_total_source_exact_rows_v373"]
        ),
        "decision_artifact_v382": (
            "reports/paper_material/paper4/notes/"
            "paper4_v382_global_solver_scope_decision.md"
        ),
        "next_artifact_v382": NEXT_ARTIFACT,
        "claim_boundary": (
            "v382 selects bounded/gap solver language and keeps full-v55 global "
            "and integer optimality prohibited"
        ),
    }
    DECISION_MD.write_text(_decision_markdown(status, route_requirements), encoding="utf-8")
    write_json(STATUS_DIR / "paper4_v382_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v382": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
