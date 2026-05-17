#!/usr/bin/env python3
"""Build Paper 4 v467 CVaR tail-risk frontier probe artifacts."""

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

VERSION = 467
PRIOR_DOMAIN_REFOCUS_VERSION = 466
NEXT_ARTIFACT = "paper4_v468_source_governance_refresh.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v467_cvar_tail_risk_frontier_probe.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _first_row(name: str) -> pd.Series:
    data = pd.read_csv(TABLE_DIR / name)
    if data.empty:
        raise RuntimeError(f"Expected non-empty artifact: {name}")
    return data.iloc[0]


def _candidate_frontier() -> pd.DataFrame:
    v338 = _first_row("paper4_v338_apply_next_post_v336_swap.csv")
    v347 = _first_row("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    v353 = _first_row("paper4_v353_v347_apply_expanded_branch_price_candidate.csv")
    rows = [
        {
            "candidate_version_v467": "v338",
            "predecessor_version_v467": "v336",
            "candidate_label_v467": "post_v336_one_swap_candidate",
            "objective_return_v467": float(v338["objective_return_v338"]),
            "scenario_loss_mean_v467": float(v338["scenario_loss_mean_v338"]),
            "scenario_loss_cvar90_v467": float(v338["scenario_loss_cvar90_v338"]),
            "delta_return_vs_predecessor_v467": float(v338["delta_return_vs_v336_v338"]),
            "delta_cvar90_vs_predecessor_v467": float(v338["delta_cvar90_vs_v336_v338"]),
            "missing_proxy_rows_v467": int(v338["missing_proxy_rows_v338"]),
            "source_cap_violations_v467": int(v338["source_cap_violations_v338"]),
            "post_repricing_required_v467": bool(v338["post_v338_repricing_required_v338"]),
            "one_swap_local_optimality_cleared_v467": False,
            "claim_boundary_v467": str(v338["claim_boundary_v338"]),
        },
        {
            "candidate_version_v467": "v347",
            "predecessor_version_v467": "v338",
            "candidate_label_v467": "multi_source_relief_candidate",
            "objective_return_v467": float(v347["objective_return_v347"]),
            "scenario_loss_mean_v467": float(v347["scenario_loss_mean_v347"]),
            "scenario_loss_cvar90_v467": float(v347["scenario_loss_cvar90_v347"]),
            "delta_return_vs_predecessor_v467": float(v347["delta_return_vs_v338_v347"]),
            "delta_cvar90_vs_predecessor_v467": float(v347["delta_cvar90_vs_v338_v347"]),
            "missing_proxy_rows_v467": int(v347["missing_proxy_rows_v347"]),
            "source_cap_violations_v467": int(v347["source_cap_violations_v347"]),
            "post_repricing_required_v467": bool(v347["post_v347_repricing_required_v347"]),
            "one_swap_local_optimality_cleared_v467": False,
            "claim_boundary_v467": str(v347["claim_boundary_v347"]),
        },
        {
            "candidate_version_v467": "v353",
            "predecessor_version_v467": "v347",
            "candidate_label_v467": "expanded_branch_price_candidate",
            "objective_return_v467": float(v353["objective_return_v353"]),
            "scenario_loss_mean_v467": float(v353["scenario_loss_mean_v353"]),
            "scenario_loss_cvar90_v467": float(v353["scenario_loss_cvar90_v353"]),
            "delta_return_vs_predecessor_v467": float(v353["delta_return_vs_v347_v353"]),
            "delta_cvar90_vs_predecessor_v467": float(v353["delta_cvar90_vs_v347_v353"]),
            "missing_proxy_rows_v467": int(v353["missing_proxy_rows_v353"]),
            "source_cap_violations_v467": int(v353["source_cap_violations_v353"]),
            "post_repricing_required_v467": bool(v353["post_v353_repricing_required_v353"]),
            "one_swap_local_optimality_cleared_v467": True,
            "claim_boundary_v467": str(v353["claim_boundary_v353"]),
        },
    ]
    frontier = pd.DataFrame(rows)
    frontier["return_rank_v467"] = (
        frontier["objective_return_v467"].rank(method="dense", ascending=False).astype(int)
    )
    frontier["cvar_rank_v467"] = (
        frontier["scenario_loss_cvar90_v467"].rank(method="dense", ascending=True).astype(int)
    )
    frontier["strict_return_cvar_improvement_vs_predecessor_v467"] = (
        frontier["delta_return_vs_predecessor_v467"].gt(0)
        & frontier["delta_cvar90_vs_predecessor_v467"].lt(0)
    )
    frontier["local_frontier_candidate_v467"] = (
        frontier["return_rank_v467"].eq(1) & frontier["cvar_rank_v467"].eq(1)
    )
    return frontier


def _evidence_stack(frontier: pd.DataFrame) -> pd.DataFrame:
    v354 = _first_row("paper4_v354_post_v353_one_swap_summary.csv")
    v357 = _read_status(357)
    v365 = _read_status(365)
    v367 = _first_row("paper4_v367_route_decision_after_chunk_probe.csv")
    v368_claims = pd.read_csv(TABLE_DIR / "paper4_v368_publishable_claims.csv")
    bounded_fourth_order_allowed = bool(
        v368_claims.loc[
            v368_claims["claim_id_v368"].eq("bounded_fourth_order_no_entry"),
            "allowed_v368",
        ].iloc[0]
    )
    return pd.DataFrame(
        [
            {
                "evidence_id_v467": "frontier_chain_v338_v347_v353",
                "source_artifact_v467": "paper4_v467_cvar_frontier_probe.csv",
                "metric_v467": "local_frontier_candidate",
                "value_v467": str(frontier.loc[frontier["local_frontier_candidate_v467"]].shape[0]),
                "claim_boundary_v467": "local chain only, not full-v55 optimality",
            },
            {
                "evidence_id_v467": "post_v353_one_swap_cleared",
                "source_artifact_v467": "paper4_v354_post_v353_one_swap_summary.csv",
                "metric_v467": "one_swap_improving_rows_v354",
                "value_v467": str(int(v354["one_swap_improving_rows_v354"])),
                "claim_boundary_v467": "one-swap local screen only",
            },
            {
                "evidence_id_v467": "bounded_second_order_no_entry",
                "source_artifact_v467": "paper4_v357_second_order_branch_price_stage_summary.csv",
                "metric_v467": "cvar_feasible_entering_rows_v357",
                "value_v467": str(int(v357["cvar_feasible_entering_rows_v357"])),
                "claim_boundary_v467": "bounded second-order scope only",
            },
            {
                "evidence_id_v467": "bounded_fourth_order_claim_scope",
                "source_artifact_v467": "paper4_v368_publishable_claims.csv",
                "metric_v467": "bounded_fourth_order_no_entry_allowed",
                "value_v467": str(bounded_fourth_order_allowed),
                "claim_boundary_v467": "bounded fourth-order no-entry language only",
            },
            {
                "evidence_id_v467": "full_v55_chunk_plan_not_certificate",
                "source_artifact_v467": "paper4_v365_v353_full_v55_pricing_chunk_plan.csv",
                "metric_v467": "planned_chunk_count_v365",
                "value_v467": str(int(v365["planned_chunk_count_v365"])),
                "claim_boundary_v467": "pricing plan only, not solver certificate",
            },
            {
                "evidence_id_v467": "chunk_route_stops_blind_chunking",
                "source_artifact_v467": "paper4_v367_route_decision_after_chunk_probe.csv",
                "metric_v467": "recommended_route_v367",
                "value_v467": str(v367["recommended_route_v367"]),
                "claim_boundary_v467": "route decision only",
            },
        ]
    )


def _blocker_register(frontier: pd.DataFrame) -> pd.DataFrame:
    best = frontier.loc[frontier["local_frontier_candidate_v467"]].iloc[0]
    v365 = _read_status(365)
    v367 = _first_row("paper4_v367_route_decision_after_chunk_probe.csv")
    return pd.DataFrame(
        [
            {
                "blocker_id_v467": "proxy_gap_persists_on_local_frontier",
                "blocking_v467": True,
                "evidence_count_v467": int(best["missing_proxy_rows_v467"]),
                "required_next_artifact_v467": "paper4_v472_ifrs9_proxy_boundary_probe.md",
                "claim_boundary_v467": "v353 keeps missing proxy rows",
            },
            {
                "blocker_id_v467": "full_v55_global_proof_missing",
                "blocking_v467": True,
                "evidence_count_v467": int(v367["remaining_unpriced_chunks_v367"]),
                "required_next_artifact_v467": "future_full_v55_dual_bound_or_scope_decision",
                "claim_boundary_v467": "remaining chunks are unpriced",
            },
            {
                "blocker_id_v467": "v71_improving_omitted_columns_persist",
                "blocking_v467": True,
                "evidence_count_v467": int(v365["v71_improving_omitted_columns_v365"]),
                "required_next_artifact_v467": "future_dual_bound_gap_resolution",
                "claim_boundary_v467": "prior restricted-master gap remains a blocker",
            },
            {
                "blocker_id_v467": "external_dynamic_online_validation_missing",
                "blocking_v467": True,
                "evidence_count_v467": 1,
                "required_next_artifact_v467": "paper4_v469_to_v470_domain_replay_sequence",
                "claim_boundary_v467": "CVaR frontier evidence is offline/internal",
            },
            {
                "blocker_id_v467": "paper4_final_promotion_forbidden",
                "blocking_v467": True,
                "evidence_count_v467": 1,
                "required_next_artifact_v467": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v467": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v467_v353_local_return_cvar_frontier",
                "allowed": True,
                "artifact": "paper4_v467_cvar_frontier_probe.csv",
                "boundary": "local v338-v347-v353 chain only",
            },
            {
                "claim_id": "v467_bounded_no_entry_evidence_summarized",
                "allowed": True,
                "artifact": "paper4_v467_cvar_evidence_stack.csv",
                "boundary": "bounded branch-price and chunk evidence only",
            },
            {
                "claim_id": "v467_full_v55_global_optimality",
                "allowed": False,
                "artifact": "paper4_v467_cvar_blocker_register.csv",
                "boundary": "global dual-bound certificate missing",
            },
            {
                "claim_id": "v467_working_champion_or_live_deployment",
                "allowed": False,
                "artifact": "paper4_v467_cvar_blocker_register.csv",
                "boundary": "proxy, dynamic, online and deployment gates remain open",
            },
            {
                "claim_id": "v467_paper_estrella_replacement_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no final promotion artifact is created",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v467 identifies v353 as the current local return/CVaR frontier.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v467_cvar_frontier_probe.csv"
                ),
                "boundary": "Local v338-v347-v353 chain only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v467 summarizes bounded no-entry evidence for CVaR claims.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v467_cvar_evidence_stack.csv"
                ),
                "boundary": "Bounded branch-price and chunk evidence only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v467 proves full-v55 global CVaR optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v467_cvar_blocker_register.csv"
                ),
                "boundary": "Full dual-bound/global certificate remains missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v467 authorizes a Paper 4 working champion or live deployment.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v467_cvar_blocker_register.csv"
                ),
                "boundary": "Proxy, dynamic, online and deployment gates remain open.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v467 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v467_cvar_blocker_register.csv"
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
                "lane": "CVaR Tail Risk",
                "executable_item": "v467 probes the local CVaR tail-risk frontier.",
                "status": "cvar_tail_risk_frontier_probe_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v468 refreshes source governance using the v467 blocker register"
                ),
                "last_wave": "v467",
                "execution_result": (
                    "v353_local_frontier_reconfirmed_proxy_global_blockers_active"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v467")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 CVaR Tail-Risk Frontier Probe v467

Generated: {status["generated_at_utc"]}

## Result

v467 finds that v353 is the current local return/CVaR frontier point across the
v338-v347-v353 candidate chain: it has the highest local objective return and
the lowest CVaR90 among those candidates. The claim remains bounded because the
proxy gap, full-v55 dual-bound gap, dynamic/online replay gaps and final
promotion gate all remain unresolved.

## Counts

- Candidate frontier rows: `{status["candidate_frontier_rows_v467"]}`.
- Local frontier candidate: `{status["local_frontier_candidate_v467"]}`.
- Best local objective return: `{status["best_local_objective_return_v467"]}`.
- Best local CVaR90: `{status["best_local_cvar90_v467"]}`.
- Missing proxy rows on local frontier: `{status["missing_proxy_rows_on_frontier_v467"]}`.
- Remaining unpriced full-v55 chunks: `{status["remaining_unpriced_chunks_v467"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v467 supports only local frontier language. It does not prove full-v55 global
optimality, repair proxy coverage, validate live deployment, authorize a Paper 4
working champion, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V467_CVAR_TAIL_RISK_FRONTIER_PROBE_START -->"
    end = "<!-- V467_CVAR_TAIL_RISK_FRONTIER_PROBE_END -->"
    block = f"""
{start}

## Wave v467: CVaR Tail-Risk Frontier Probe

Generated: {status["generated_at_utc"]}

### Objective

v467 probes the CVaR tail-risk lane selected by v466 and separates the strongest
local frontier claim from the blockers that still prevent global or champion
language.

### Results

- Candidate frontier rows:
  `{status["candidate_frontier_rows_v467"]}`.
- Local frontier candidate:
  `{status["local_frontier_candidate_v467"]}`.
- Best local objective return:
  `{status["best_local_objective_return_v467"]}`.
- Best local CVaR90:
  `{status["best_local_cvar90_v467"]}`.
- Missing proxy rows on frontier:
  `{status["missing_proxy_rows_on_frontier_v467"]}`.
- Remaining unpriced chunks:
  `{status["remaining_unpriced_chunks_v467"]}`.
- Full-v55 global proof created:
  `{status["full_v55_global_proof_created_v467"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v467"]}`.

### Interpretation

v353 is valuable future Paper 4 evidence because it improves the local return
and CVaR frontier simultaneously relative to v347. Its usefulness is strongest
as bounded frontier evidence, not as a champion claim, because proxy coverage
and full-v55/global validation remain open.

### Claim Impact

- Allowed: local v353 return/CVaR frontier language and bounded no-entry
  evidence.
- Still prohibited: full-v55 global optimality, working-champion language, live
  deployment readiness, Paper Estrella replacement and final Paper 4 promotion.

### Quarto Promotion Decision

Keep v467 in the living notebook. v468 should refresh source-governance evidence
around the same blocker surface before any broader claim is attempted.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v466 = _read_status(PRIOR_DOMAIN_REFOCUS_VERSION)
    if v466["next_artifact_v466"] != "paper4_v467_cvar_tail_risk_frontier_probe.md":
        raise RuntimeError("v467 expects v466 to route to the CVaR frontier probe.")
    if v466["selected_next_lane_v466"] != "cvar_tail_risk":
        raise RuntimeError("v467 expects the CVaR tail-risk lane to be selected.")

    frontier = _candidate_frontier()
    local = frontier.loc[frontier["local_frontier_candidate_v467"]]
    if len(local) != 1:
        raise RuntimeError("Expected exactly one v467 local frontier candidate.")
    local_row = local.iloc[0]
    if local_row["candidate_version_v467"] != "v353":
        raise RuntimeError("v467 expects v353 to be the current local frontier candidate.")

    evidence_stack = _evidence_stack(frontier)
    blocker_register = _blocker_register(frontier)
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v467_cvar_frontier_probe.csv", frontier)
    write_csv(TABLE_DIR / "paper4_v467_cvar_evidence_stack.csv", evidence_stack)
    write_csv(TABLE_DIR / "paper4_v467_cvar_blocker_register.csv", blocker_register)
    write_csv(TABLE_DIR / "paper4_v467_claim_matrix_delta.csv", claim_matrix)

    v367 = _first_row("paper4_v367_route_decision_after_chunk_probe.csv")
    status = {
        "phase": "v467_cvar_tail_risk_frontier_probe",
        "schema_version": "2026-05-17.467",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_domain_refocus_version_v467": PRIOR_DOMAIN_REFOCUS_VERSION,
        "candidate_frontier_rows_v467": len(frontier),
        "evidence_stack_rows_v467": len(evidence_stack),
        "blocker_register_rows_v467": len(blocker_register),
        "local_frontier_candidate_v467": str(local_row["candidate_version_v467"]),
        "best_local_objective_return_v467": float(local_row["objective_return_v467"]),
        "best_local_cvar90_v467": float(local_row["scenario_loss_cvar90_v467"]),
        "missing_proxy_rows_on_frontier_v467": int(local_row["missing_proxy_rows_v467"]),
        "v353_return_cvar_improves_vs_v347_v467": bool(
            local_row["strict_return_cvar_improvement_vs_predecessor_v467"]
        ),
        "post_v353_one_swap_local_optimality_cleared_v467": bool(
            local_row["one_swap_local_optimality_cleared_v467"]
        ),
        "bounded_no_entry_evidence_summarized_v467": True,
        "remaining_unpriced_chunks_v467": int(v367["remaining_unpriced_chunks_v367"]),
        "full_v55_global_proof_created_v467": False,
        "working_champion_claim_allowed_v467": False,
        "paper1_promotion_allowed_v467": False,
        "paper4_working_champion_changed_v467": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v467": NEXT_ARTIFACT,
        "claim_boundary": (
            "v467 supports local CVaR frontier language only; proxy, global, "
            "dynamic, online, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v467 must not create final Paper 4 promotion.")

    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v467": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
