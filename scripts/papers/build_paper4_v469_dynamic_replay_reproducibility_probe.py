#!/usr/bin/env python3
"""Build Paper 4 v469 dynamic replay reproducibility probe artifacts."""

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

VERSION = 469
PRIOR_SOURCE_GOVERNANCE_VERSION = 468
NEXT_ARTIFACT = "paper4_v470_online_conformal_monitoring_proxy.md"
PROBE_MD = NOTEBOOK.parent / "paper4_v469_dynamic_replay_reproducibility_probe.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _first_row(name: str) -> pd.Series:
    data = pd.read_csv(TABLE_DIR / name)
    if data.empty:
        raise RuntimeError(f"Expected non-empty artifact: {name}")
    return data.iloc[0]


def _dynamic_inventory() -> pd.DataFrame:
    v297 = _first_row("paper4_v297_global_dynamic_gate_summary.csv")
    v340 = _first_row("paper4_v340_dynamic_proxy_or_global_bound_after_v338.csv")
    return pd.DataFrame(
        [
            {
                "dynamic_probe_id_v469": "v297_v295_vs_v293_dynamic_gate",
                "source_artifact_v469": "paper4_v297_global_dynamic_gate_summary.csv",
                "candidate_version_v469": "v295",
                "comparison_versions_v469": "v293",
                "dynamic_proxy_replay_executed_v469": bool(
                    v297["dynamic_proxy_replay_executed_v297"]
                ),
                "dynamic_proxy_trace_rows_v469": int(v297["dynamic_proxy_trace_rows_v297"]),
                "dynamic_proxy_policy_count_v469": int(
                    v297["dynamic_proxy_policy_count_v297"]
                ),
                "dynamic_proxy_period_count_v469": int(
                    v297["dynamic_proxy_period_count_v297"]
                ),
                "period_distribution_match_v469": bool(
                    v297["period_distribution_match_v297"]
                ),
                "live_deployment_claim_allowed_v469": bool(
                    v297["live_deployment_claim_allowed_v297"]
                ),
                "claim_boundary_v469": str(v297["claim_boundary_v297"]),
            },
            {
                "dynamic_probe_id_v469": "v340_v338_dynamic_proxy_gate",
                "source_artifact_v469": "paper4_v340_dynamic_proxy_or_global_bound_after_v338.csv",
                "candidate_version_v469": "v338",
                "comparison_versions_v469": "v295;v316;v330",
                "dynamic_proxy_replay_executed_v469": bool(
                    v340["dynamic_proxy_replay_executed_v340"]
                ),
                "dynamic_proxy_trace_rows_v469": int(v340["dynamic_proxy_trace_rows_v340"]),
                "dynamic_proxy_policy_count_v469": int(
                    v340["dynamic_proxy_policy_count_v340"]
                ),
                "dynamic_proxy_period_count_v469": int(
                    v340["dynamic_proxy_period_count_v340"]
                ),
                "period_distribution_match_v469": bool(v340["period_set_match_v340"]),
                "live_deployment_claim_allowed_v469": bool(
                    v340["strict_live_deployability_claim_allowed_v340"]
                ),
                "claim_boundary_v469": str(v340["claim_boundary_v340"]),
            },
        ]
    )


def _frontier_gap() -> pd.DataFrame:
    frontier = pd.read_csv(TABLE_DIR / "paper4_v467_cvar_frontier_probe.csv")
    v338 = frontier.loc[frontier["candidate_version_v467"].eq("v338")].iloc[0]
    v353 = frontier.loc[frontier["candidate_version_v467"].eq("v353")].iloc[0]
    dynamic = _first_row("paper4_v340_dynamic_proxy_or_global_bound_after_v338.csv")
    return pd.DataFrame(
        [
            {
                "gap_id_v469": "current_frontier_v353_not_dynamic_replayed",
                "local_frontier_version_v469": "v353",
                "latest_dynamic_proxy_candidate_v469": "v338",
                "current_frontier_dynamic_replayed_v469": False,
                "latest_dynamic_proxy_trace_rows_v469": int(
                    dynamic["dynamic_proxy_trace_rows_v340"]
                ),
                "delta_return_v353_vs_v338_static_v469": float(
                    v353["objective_return_v467"] - v338["objective_return_v467"]
                ),
                "delta_cvar90_v353_vs_v338_static_v469": float(
                    v353["scenario_loss_cvar90_v467"] - v338["scenario_loss_cvar90_v467"]
                ),
                "v338_lower_cvar_frontier_tradeoff_vs_v330_v469": bool(
                    dynamic["v338_lower_cvar_frontier_tradeoff_vs_v330_v340"]
                ),
                "v338_dominates_v330_v469": bool(dynamic["v338_dominates_v330_v340"]),
                "claim_boundary_v469": (
                    "v353 static local frontier cannot inherit v338 dynamic proxy claims"
                ),
            }
        ]
    )


def _decision_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id_v469": "do_not_extend_v338_dynamic_claim_to_v353",
                "recommended_v469": True,
                "evidence_v469": "v353 has no dynamic proxy trace artifact",
                "next_artifact_v469": "future_v353_dynamic_replay_build",
                "claim_boundary_v469": "claim transfer is blocked",
            },
            {
                "decision_id_v469": "keep_v338_dynamic_proxy_as_historical_anchor",
                "recommended_v469": True,
                "evidence_v469": "v340 has 1536 trace rows across four policies and three periods",
                "next_artifact_v469": NEXT_ARTIFACT,
                "claim_boundary_v469": "historical proxy anchor only",
            },
            {
                "decision_id_v469": "route_to_online_monitoring_proxy",
                "recommended_v469": True,
                "evidence_v469": "dynamic live claims remain blocked and online lane is next",
                "next_artifact_v469": NEXT_ARTIFACT,
                "claim_boundary_v469": "routing decision only",
            },
            {
                "decision_id_v469": "do_not_claim_live_dynamic_deployment",
                "recommended_v469": True,
                "evidence_v469": "v297 and v340 both mark live deployment claims as false",
                "next_artifact_v469": "future_live_deployment_replay_gate",
                "claim_boundary_v469": "no live deployment language",
            },
        ]
    )


def _blocker_register() -> pd.DataFrame:
    v340_blockers = pd.read_csv(TABLE_DIR / "paper4_v340_claim_blockers.csv")
    dynamic_block = v340_blockers.loc[
        v340_blockers["blocker_id_v340"].eq("dynamic_proxy_not_live_deployment_replay")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "blocker_id_v469": "v353_dynamic_proxy_trace_missing",
                "blocking_v469": True,
                "evidence_count_v469": 1,
                "required_next_artifact_v469": "future_v353_dynamic_replay_build",
                "claim_boundary_v469": "current local frontier lacks dynamic replay",
            },
            {
                "blocker_id_v469": "dynamic_proxy_not_live_deployment_replay",
                "blocking_v469": True,
                "evidence_count_v469": int(dynamic_block["evidence_count_v340"]),
                "required_next_artifact_v469": "future_live_deployment_replay_gate",
                "claim_boundary_v469": "periodized static-book proxy only",
            },
            {
                "blocker_id_v469": "valid_global_gap_certificate_missing",
                "blocking_v469": True,
                "evidence_count_v469": 1,
                "required_next_artifact_v469": "future_branch_price_dual_bound_loop",
                "claim_boundary_v469": "dynamic proxy does not replace solver proof",
            },
            {
                "blocker_id_v469": "online_monitoring_proxy_not_refreshed",
                "blocking_v469": True,
                "evidence_count_v469": 1,
                "required_next_artifact_v469": NEXT_ARTIFACT,
                "claim_boundary_v469": "online lane still pending",
            },
            {
                "blocker_id_v469": "paper4_final_promotion_forbidden",
                "blocking_v469": True,
                "evidence_count_v469": 1,
                "required_next_artifact_v469": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v469": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v469_dynamic_replay_inventory_created",
                "allowed": True,
                "artifact": "paper4_v469_dynamic_replay_inventory.csv",
                "boundary": "existing proxy replay inventory only",
            },
            {
                "claim_id": "v469_v353_dynamic_gap_documented",
                "allowed": True,
                "artifact": "paper4_v469_current_frontier_dynamic_gap.csv",
                "boundary": "gap documentation only",
            },
            {
                "claim_id": "v469_v353_has_dynamic_replay_validation",
                "allowed": False,
                "artifact": "paper4_v469_dynamic_blocker_register.csv",
                "boundary": "v353 dynamic trace missing",
            },
            {
                "claim_id": "v469_live_dynamic_deployment_ready",
                "allowed": False,
                "artifact": "paper4_v469_dynamic_blocker_register.csv",
                "boundary": "proxy replay is not live deployment",
            },
            {
                "claim_id": "v469_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_final_promotion_gate_not_created",
                "boundary": "no champion or final promotion claim",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = pd.read_csv(path)
    additions = pd.DataFrame(
        [
            {
                "claim": "v469 inventories existing dynamic proxy replay evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v469_dynamic_replay_inventory.csv"
                ),
                "boundary": "Existing proxy replay inventory only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v469 documents that v353 lacks dynamic replay validation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v469_current_frontier_dynamic_gap.csv"
                ),
                "boundary": "Gap documentation only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v469 validates v353 as a dynamic replay champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v469_dynamic_blocker_register.csv"
                ),
                "boundary": "v353 has no dynamic proxy trace artifact.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v469 proves live dynamic deployment readiness.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v469_dynamic_blocker_register.csv"
                ),
                "boundary": "Proxy replay is not live deployment validation.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v469 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v469_dynamic_blocker_register.csv"
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
                "lane": "Dynamic Replay",
                "executable_item": "v469 probes dynamic replay reproducibility and gaps.",
                "status": "dynamic_replay_reproducibility_probe_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "v470 refreshes online conformal monitoring as proxy-only evidence"
                ),
                "last_wave": "v469",
                "execution_result": "v338_dynamic_anchor_retained_v353_dynamic_gap_documented",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v469")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _probe_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Dynamic Replay Reproducibility Probe v469

Generated: {status["generated_at_utc"]}

## Result

v469 inventories existing dynamic proxy replay evidence and documents a crucial
gap: the latest local CVaR frontier candidate is v353, but the strongest dynamic
proxy replay anchor remains v338. Therefore v353 cannot inherit dynamic replay
or live deployment language.

## Counts

- Dynamic inventory rows: `{status["dynamic_inventory_rows_v469"]}`.
- Latest dynamic proxy candidate: `{status["latest_dynamic_proxy_candidate_v469"]}`.
- Latest dynamic proxy trace rows: `{status["latest_dynamic_proxy_trace_rows_v469"]}`.
- Current local frontier: `{status["current_local_frontier_v469"]}`.
- Current frontier dynamic replayed: `{status["current_frontier_dynamic_replayed_v469"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v469 is a reproducibility and gap probe only. It does not build a v353 dynamic
trace, validate live deployment, prove global optimality, authorize a working
champion, replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V469_DYNAMIC_REPLAY_REPRODUCIBILITY_PROBE_START -->"
    end = "<!-- V469_DYNAMIC_REPLAY_REPRODUCIBILITY_PROBE_END -->"
    block = f"""
{start}

## Wave v469: Dynamic Replay Reproducibility Probe

Generated: {status["generated_at_utc"]}

### Objective

v469 audits existing dynamic proxy replay evidence and checks whether the v467
local frontier can support dynamic or live claims.

### Results

- Dynamic inventory rows:
  `{status["dynamic_inventory_rows_v469"]}`.
- Latest dynamic proxy candidate:
  `{status["latest_dynamic_proxy_candidate_v469"]}`.
- Latest dynamic proxy trace rows:
  `{status["latest_dynamic_proxy_trace_rows_v469"]}`.
- Current local frontier:
  `{status["current_local_frontier_v469"]}`.
- Current frontier dynamic replayed:
  `{status["current_frontier_dynamic_replayed_v469"]}`.
- v353 static return delta vs v338:
  `{status["delta_return_v353_vs_v338_static_v469"]}`.
- v353 static CVaR delta vs v338:
  `{status["delta_cvar90_v353_vs_v338_static_v469"]}`.
- Live dynamic deployment allowed:
  `{status["live_dynamic_deployment_claim_allowed_v469"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v469"]}`.

### Interpretation

v338 remains the dynamic proxy anchor. v353 is stronger on the local static
return/CVaR frontier, but it has no dynamic replay trace, so it cannot carry
dynamic validation language yet.

### Claim Impact

- Allowed: dynamic replay inventory and v353 dynamic-gap statement.
- Still prohibited: v353 dynamic validation, live deployment readiness,
  working-champion language, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v469 in the living notebook. v470 should refresh online conformal
monitoring as proxy-only evidence.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v468 = _read_status(PRIOR_SOURCE_GOVERNANCE_VERSION)
    if v468["next_artifact_v468"] != "paper4_v469_dynamic_replay_reproducibility_probe.md":
        raise RuntimeError("v469 expects v468 to route to dynamic replay.")

    inventory = _dynamic_inventory()
    gap = _frontier_gap()
    decision = _decision_table()
    blockers = _blocker_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v469_dynamic_replay_inventory.csv", inventory)
    write_csv(TABLE_DIR / "paper4_v469_current_frontier_dynamic_gap.csv", gap)
    write_csv(TABLE_DIR / "paper4_v469_dynamic_replay_decision.csv", decision)
    write_csv(TABLE_DIR / "paper4_v469_dynamic_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v469_claim_matrix_delta.csv", claim_matrix)

    gap_row = gap.iloc[0]
    status = {
        "phase": "v469_dynamic_replay_reproducibility_probe",
        "schema_version": "2026-05-17.469",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_source_governance_version_v469": PRIOR_SOURCE_GOVERNANCE_VERSION,
        "dynamic_replay_reproducibility_probe_created_v469": True,
        "dynamic_inventory_rows_v469": len(inventory),
        "current_local_frontier_v469": str(gap_row["local_frontier_version_v469"]),
        "latest_dynamic_proxy_candidate_v469": str(
            gap_row["latest_dynamic_proxy_candidate_v469"]
        ),
        "latest_dynamic_proxy_trace_rows_v469": int(
            gap_row["latest_dynamic_proxy_trace_rows_v469"]
        ),
        "current_frontier_dynamic_replayed_v469": bool(
            gap_row["current_frontier_dynamic_replayed_v469"]
        ),
        "delta_return_v353_vs_v338_static_v469": float(
            gap_row["delta_return_v353_vs_v338_static_v469"]
        ),
        "delta_cvar90_v353_vs_v338_static_v469": float(
            gap_row["delta_cvar90_v353_vs_v338_static_v469"]
        ),
        "v338_lower_cvar_frontier_tradeoff_vs_v330_v469": bool(
            gap_row["v338_lower_cvar_frontier_tradeoff_vs_v330_v469"]
        ),
        "v338_dominates_v330_v469": bool(gap_row["v338_dominates_v330_v469"]),
        "v353_dynamic_validation_claim_allowed_v469": False,
        "live_dynamic_deployment_claim_allowed_v469": False,
        "working_champion_claim_allowed_v469": False,
        "paper1_promotion_allowed_v469": False,
        "paper4_working_champion_changed_v469": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v469": NEXT_ARTIFACT,
        "claim_boundary": (
            "v469 inventories dynamic proxy replay only; v353 dynamic validation, "
            "live deployment, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v469 must not create final Paper 4 promotion.")

    PROBE_MD.write_text(_probe_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v469": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
