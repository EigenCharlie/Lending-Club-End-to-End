#!/usr/bin/env python3
"""Build Paper 4 v470 online conformal monitoring proxy artifacts."""

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

VERSION = 470
PRIOR_DYNAMIC_REPLAY_VERSION = 469
NEXT_ARTIFACT = "paper4_v471_spo_dla_boundary_probe.md"
PROXY_MD = NOTEBOOK.parent / "paper4_v470_online_conformal_monitoring_proxy.md"


def _read_status(version: int) -> dict[str, Any]:
    return json.loads((STATUS_DIR / f"paper4_v{version}_status.json").read_text(encoding="utf-8"))


def _first_row(name: str) -> pd.Series:
    data = pd.read_csv(TABLE_DIR / name)
    if data.empty:
        raise RuntimeError(f"Expected non-empty artifact: {name}")
    return data.iloc[0]


def _online_summary() -> pd.DataFrame:
    v9_status = json.loads(
        (STATUS_DIR / "paper4_v9_online_goal_status.json").read_text(encoding="utf-8")
    )
    v9 = pd.read_csv(TABLE_DIR / "paper4_v9_online_efficiency_frontier.csv")
    v10 = pd.read_csv(TABLE_DIR / "paper4_v10_online_robustness_summary.csv")
    v10_support = pd.read_csv(TABLE_DIR / "paper4_v10_online_min_support_sensitivity.csv")
    v341_gate = _first_row("paper4_v341_v338_cashflow_online_ifrs9_gate.csv")
    best = v9.loc[v9["online_method_v9"].eq(v9_status["online_best_method_v9"])].iloc[0]
    return pd.DataFrame(
        [
            {
                "summary_id_v470": "online_conformal_monitoring_proxy",
                "v9_online_goal_achieved_v470": bool(v9_status["online_goal_achieved"]),
                "v9_best_method_v470": str(v9_status["online_best_method_v9"]),
                "v9_best_source_month_defended_min_v470": float(
                    best["coverage_source_month_defended_min"]
                ),
                "v9_best_policy_month_defended_min_v470": float(
                    best["coverage_policy_month_defended_min"]
                ),
                "v9_best_avg_width_loan_v470": float(best["avg_width_loan"]),
                "v9_deployable_without_current_outcomes_v470": bool(
                    best["deployable_without_current_outcomes"]
                ),
                "v10_robustness_items_v470": len(v10),
                "v10_robustness_all_pass_v470": bool(v10["pass_v10"].astype(bool).all()),
                "v10_min_support_pass_rows_v470": int(
                    v10_support["gate_source80_policy90_width95"].astype(bool).sum()
                ),
                "v10_min_support_rows_v470": len(v10_support),
                "v341_online_internal_gate_family_rows_v470": int(
                    v341_gate["online_internal_all_gate_family_rows_v341"]
                ),
                "v341_external_holdout_available_v470": False,
                "v341_strict_live_claim_allowed_v470": bool(
                    v341_gate["strict_live_deployability_claim_allowed_v341"]
                ),
                "claim_boundary_v470": (
                    "online monitoring proxy only; no external holdout or production claim"
                ),
            }
        ]
    )


def _internal_gate_inventory() -> pd.DataFrame:
    v9 = _first_row("paper4_v9_online_efficiency_frontier.csv")
    v10 = pd.read_csv(TABLE_DIR / "paper4_v10_online_robustness_summary.csv")
    v323 = pd.read_csv(TABLE_DIR / "paper4_v323_v320_online_temporal_summary.csv")
    v341 = pd.read_csv(TABLE_DIR / "paper4_v341_v338_online_temporal_summary.csv")
    return pd.DataFrame(
        [
            {
                "gate_id_v470": "v9_online_goal",
                "source_artifact_v470": "paper4_v9_online_efficiency_frontier.csv",
                "candidate_scope_v470": "method_grid_replay",
                "internal_gate_pass_v470": bool(v9["goal_pass"]),
                "source_or_policy_rows_v470": 1,
                "worst_source_coverage_v470": float(v9["coverage_source_month_defended_min"]),
                "worst_policy_coverage_v470": float(v9["coverage_policy_month_defended_min"]),
                "external_holdout_available_v470": False,
                "strict_live_claim_allowed_v470": False,
                "claim_boundary_v470": "selected in replay; future-period validation required",
            },
            {
                "gate_id_v470": "v10_robustness",
                "source_artifact_v470": "paper4_v10_online_robustness_summary.csv",
                "candidate_scope_v470": "stress_replay",
                "internal_gate_pass_v470": bool(v10["pass_v10"].astype(bool).all()),
                "source_or_policy_rows_v470": len(v10),
                "worst_source_coverage_v470": float(v10["source_month_defended_min"].min()),
                "worst_policy_coverage_v470": float(v10["policy_month_defended_min"].min()),
                "external_holdout_available_v470": False,
                "strict_live_claim_allowed_v470": False,
                "claim_boundary_v470": "robustness replay only",
            },
            {
                "gate_id_v470": "v323_v320_temporal_online",
                "source_artifact_v470": "paper4_v323_v320_online_temporal_summary.csv",
                "candidate_scope_v470": "v320_selected_book",
                "internal_gate_pass_v470": bool(v323["all_internal_gates_pass_v323"].all()),
                "source_or_policy_rows_v470": len(v323),
                "worst_source_coverage_v470": float(v323["worst_source_coverage_v323"].min()),
                "worst_policy_coverage_v470": float(
                    v323["worst_policy_period_coverage_v323"].min()
                ),
                "external_holdout_available_v470": False,
                "strict_live_claim_allowed_v470": bool(
                    v323["strict_live_claim_allowed_v323"].any()
                ),
                "claim_boundary_v470": "selected-book temporal replay only",
            },
            {
                "gate_id_v470": "v341_v338_temporal_online",
                "source_artifact_v470": "paper4_v341_v338_online_temporal_summary.csv",
                "candidate_scope_v470": "v338_selected_book",
                "internal_gate_pass_v470": bool(v341["all_internal_gates_pass_v341"].all()),
                "source_or_policy_rows_v470": len(v341),
                "worst_source_coverage_v470": float(v341["worst_source_coverage_v341"].min()),
                "worst_policy_coverage_v470": float(
                    v341["worst_policy_period_coverage_v341"].min()
                ),
                "external_holdout_available_v470": False,
                "strict_live_claim_allowed_v470": bool(
                    v341["strict_live_claim_allowed_v341"].any()
                ),
                "claim_boundary_v470": "selected-book temporal replay only",
            },
        ]
    )


def _frontier_online_gap() -> pd.DataFrame:
    v469 = _read_status(469)
    return pd.DataFrame(
        [
            {
                "gap_id_v470": "current_frontier_v353_online_gate_missing",
                "current_local_frontier_v470": str(v469["current_local_frontier_v469"]),
                "latest_online_temporal_candidate_v470": "v338",
                "current_frontier_online_refreshed_v470": False,
                "v338_online_internal_gate_available_v470": True,
                "external_holdout_available_v470": False,
                "strict_live_monitoring_claim_allowed_v470": False,
                "claim_boundary_v470": (
                    "v353 cannot inherit v338 selected-book online replay claims"
                ),
            }
        ]
    )


def _blocker_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id_v470": "v353_online_temporal_gate_missing",
                "blocking_v470": True,
                "evidence_count_v470": 1,
                "required_next_artifact_v470": "future_v353_online_temporal_gate",
                "claim_boundary_v470": "current local frontier lacks online replay",
            },
            {
                "blocker_id_v470": "external_holdout_missing",
                "blocking_v470": True,
                "evidence_count_v470": 1,
                "required_next_artifact_v470": "future_external_online_holdout",
                "claim_boundary_v470": "internal replay is not external validation",
            },
            {
                "blocker_id_v470": "production_monitoring_not_authorized",
                "blocking_v470": True,
                "evidence_count_v470": 1,
                "required_next_artifact_v470": "future_production_monitoring_design",
                "claim_boundary_v470": "no production monitoring control is approved",
            },
            {
                "blocker_id_v470": "spo_dla_boundary_not_refreshed",
                "blocking_v470": True,
                "evidence_count_v470": 1,
                "required_next_artifact_v470": NEXT_ARTIFACT,
                "claim_boundary_v470": "formal method-boundary lane still pending",
            },
            {
                "blocker_id_v470": "paper4_final_promotion_forbidden",
                "blocking_v470": True,
                "evidence_count_v470": 1,
                "required_next_artifact_v470": "paper4_final_promotion_gate_not_created",
                "claim_boundary_v470": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )


def _claim_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "v470_online_monitoring_proxy_created",
                "allowed": True,
                "artifact": "paper4_v470_online_monitoring_proxy_summary.csv",
                "boundary": "proxy-only online monitoring summary",
            },
            {
                "claim_id": "v470_internal_online_gates_inventory_created",
                "allowed": True,
                "artifact": "paper4_v470_online_internal_gate_inventory.csv",
                "boundary": "internal gate inventory only",
            },
            {
                "claim_id": "v470_v353_online_live_validated",
                "allowed": False,
                "artifact": "paper4_v470_current_frontier_online_gap.csv",
                "boundary": "v353 online temporal gate missing",
            },
            {
                "claim_id": "v470_external_or_production_monitoring_ready",
                "allowed": False,
                "artifact": "paper4_v470_online_blocker_register.csv",
                "boundary": "external holdout and production approval missing",
            },
            {
                "claim_id": "v470_working_champion_or_final_promotion",
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
                "claim": "v470 summarizes internal online conformal monitoring proxy evidence.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v470_online_monitoring_proxy_summary.csv"
                ),
                "boundary": "Internal proxy summary only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v470 documents that v353 lacks an online temporal gate.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v470_current_frontier_online_gap.csv"
                ),
                "boundary": "Gap documentation only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v470 validates v353 for live online monitoring.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v470_online_blocker_register.csv"
                ),
                "boundary": "v353 online temporal gate and external holdout are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v470 authorizes production monitoring controls.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v470_online_blocker_register.csv"
                ),
                "boundary": "No production monitoring approval or control design.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v470 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v470_online_blocker_register.csv"
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
                "lane": "Online Monitoring",
                "executable_item": "v470 refreshes online conformal monitoring proxy evidence.",
                "status": "online_monitoring_proxy_created",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": "v471 refreshes SPO-DLA claim boundaries",
                "last_wave": "v470",
                "execution_result": "internal_online_gates_summarized_v353_online_gap_documented",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    out = current.loc[~current["last_wave"].astype(str).eq("v470")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _proxy_markdown(status: dict[str, Any]) -> str:
    return f"""# Paper 4 Online Conformal Monitoring Proxy v470

Generated: {status["generated_at_utc"]}

## Result

v470 summarizes the internal online conformal evidence. v9 achieved the online
goal and v10 passed robustness checks, while v323/v341 provide selected-book
temporal replay evidence. The claim remains proxy-only because external holdout
validation is absent and v353 lacks its own online temporal gate.

## Counts

- v9 online goal achieved: `{status["v9_online_goal_achieved_v470"]}`.
- v9 best average width: `{status["v9_best_avg_width_loan_v470"]}`.
- v10 robustness all pass: `{status["v10_robustness_all_pass_v470"]}`.
- Current frontier online refreshed: `{status["current_frontier_online_refreshed_v470"]}`.
- External holdout available: `{status["external_holdout_available_v470"]}`.
- Live monitoring claim allowed: `{status["live_monitoring_claim_allowed_v470"]}`.
- Final promotion created: `{status["paper4_final_promotion_created"]}`.

## Required Caveat

v470 is internal monitoring proxy evidence only. It does not validate v353 for
live monitoring, provide an external holdout, authorize production controls,
replace Paper Estrella, or promote Paper 4 as final.
"""


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V470_ONLINE_CONFORMAL_MONITORING_PROXY_START -->"
    end = "<!-- V470_ONLINE_CONFORMAL_MONITORING_PROXY_END -->"
    block = f"""
{start}

## Wave v470: Online Conformal Monitoring Proxy

Generated: {status["generated_at_utc"]}

### Objective

v470 refreshes online conformal monitoring evidence and separates internal proxy
gates from live production claims.

### Results

- v9 online goal achieved:
  `{status["v9_online_goal_achieved_v470"]}`.
- v9 best method:
  `{status["v9_best_method_v470"]}`.
- v9 best average width:
  `{status["v9_best_avg_width_loan_v470"]}`.
- v10 robustness all pass:
  `{status["v10_robustness_all_pass_v470"]}`.
- Current local frontier:
  `{status["current_local_frontier_v470"]}`.
- Current frontier online refreshed:
  `{status["current_frontier_online_refreshed_v470"]}`.
- External holdout available:
  `{status["external_holdout_available_v470"]}`.
- Live monitoring claim allowed:
  `{status["live_monitoring_claim_allowed_v470"]}`.
- Final promotion created:
  `{status["paper4_final_promotion_created"]}`.
- Next artifact:
  `{status["next_artifact_v470"]}`.

### Interpretation

The online lane contributes defended internal monitoring evidence, but not live
monitoring validation. The strongest current local frontier, v353, still needs a
candidate-specific online temporal gate before stronger language is allowed.

### Claim Impact

- Allowed: internal online proxy summary and v353 online-gap statement.
- Still prohibited: v353 live online validation, production monitoring controls,
  working-champion language, Paper Estrella replacement and final Paper 4
  promotion.

### Quarto Promotion Decision

Keep v470 in the living notebook. v471 should refresh SPO-DLA formal boundary
language.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    v469 = _read_status(PRIOR_DYNAMIC_REPLAY_VERSION)
    if v469["next_artifact_v469"] != "paper4_v470_online_conformal_monitoring_proxy.md":
        raise RuntimeError("v470 expects v469 to route to online monitoring.")

    summary = _online_summary()
    inventory = _internal_gate_inventory()
    gap = _frontier_online_gap()
    blockers = _blocker_register()
    claim_matrix = _claim_matrix()
    _update_claim_boundaries()
    _update_backlog()

    write_csv(TABLE_DIR / "paper4_v470_online_monitoring_proxy_summary.csv", summary)
    write_csv(TABLE_DIR / "paper4_v470_online_internal_gate_inventory.csv", inventory)
    write_csv(TABLE_DIR / "paper4_v470_current_frontier_online_gap.csv", gap)
    write_csv(TABLE_DIR / "paper4_v470_online_blocker_register.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v470_claim_matrix_delta.csv", claim_matrix)

    row = summary.iloc[0]
    gap_row = gap.iloc[0]
    status = {
        "phase": "v470_online_conformal_monitoring_proxy",
        "schema_version": "2026-05-17.470",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "prior_dynamic_replay_version_v470": PRIOR_DYNAMIC_REPLAY_VERSION,
        "online_monitoring_proxy_created_v470": True,
        "online_internal_gate_inventory_rows_v470": len(inventory),
        "v9_online_goal_achieved_v470": bool(row["v9_online_goal_achieved_v470"]),
        "v9_best_method_v470": str(row["v9_best_method_v470"]),
        "v9_best_source_month_defended_min_v470": float(
            row["v9_best_source_month_defended_min_v470"]
        ),
        "v9_best_policy_month_defended_min_v470": float(
            row["v9_best_policy_month_defended_min_v470"]
        ),
        "v9_best_avg_width_loan_v470": float(row["v9_best_avg_width_loan_v470"]),
        "v10_robustness_all_pass_v470": bool(row["v10_robustness_all_pass_v470"]),
        "v10_min_support_pass_rows_v470": int(row["v10_min_support_pass_rows_v470"]),
        "v10_min_support_rows_v470": int(row["v10_min_support_rows_v470"]),
        "v341_online_internal_gate_family_rows_v470": int(
            row["v341_online_internal_gate_family_rows_v470"]
        ),
        "current_local_frontier_v470": str(gap_row["current_local_frontier_v470"]),
        "latest_online_temporal_candidate_v470": str(
            gap_row["latest_online_temporal_candidate_v470"]
        ),
        "current_frontier_online_refreshed_v470": bool(
            gap_row["current_frontier_online_refreshed_v470"]
        ),
        "external_holdout_available_v470": False,
        "live_monitoring_claim_allowed_v470": False,
        "production_monitoring_controls_authorized_v470": False,
        "working_champion_claim_allowed_v470": False,
        "paper1_promotion_allowed_v470": False,
        "paper4_working_champion_changed_v470": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v470": NEXT_ARTIFACT,
        "claim_boundary": (
            "v470 summarizes internal online proxy evidence only; external holdout, "
            "production monitoring, champion and final promotion claims remain blocked"
        ),
    }
    if status["paper4_final_promotion_created"]:
        raise RuntimeError("v470 must not create final Paper 4 promotion.")

    PROXY_MD.write_text(_proxy_markdown(status), encoding="utf-8")
    write_json(STATUS_DIR / f"paper4_v{VERSION}_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v470": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
