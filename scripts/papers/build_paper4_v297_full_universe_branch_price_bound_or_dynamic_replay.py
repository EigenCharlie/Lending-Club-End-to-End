#!/usr/bin/env python3
"""Build Paper 4 v297 global-bound/dynamic-replay gate artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FORBIDDEN_FINAL_PROMOTION,
    NOTEBOOK,
    STATUS_DIR,
    TABLE_DIR,
    _append_or_replace_block,
    now,
    read_csv,
    read_parquet,
    write_csv,
    write_json,
)

VERSION = 297
INCUMBENT_VERSION = 293
CHALLENGER_VERSION = 295
LOCAL_REPRICE_VERSION = 296
NEXT_VERSION = 298
DIRECT_MIP_GUARD_VERSION = 283


def _portfolio_metrics(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    losses: np.ndarray,
    mean_returns: np.ndarray,
) -> dict[str, Any]:
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    idx = idx_by_id.loc[portfolio["loan_id"].astype(str)].to_numpy()
    scenario_losses = losses[:, idx].sum(axis=1)
    return {
        "selected_rows": int(len(portfolio)),
        "portfolio_exposure": float(portfolio["loan_amnt"].sum()),
        "objective_return": float(mean_returns[idx].sum()),
        "scenario_loss_mean": float(scenario_losses.mean()),
        "scenario_loss_cvar90": v70._tail_cvar(scenario_losses),
        "scenario_losses": scenario_losses,
    }


def _dynamic_proxy_trace(
    *,
    universe: pd.DataFrame,
    policies: dict[str, pd.DataFrame],
    losses: np.ndarray,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for policy_id, portfolio in policies.items():
        portfolio = portfolio.copy()
        portfolio["loan_id"] = portfolio["loan_id"].astype(str)
        portfolio_idx = idx_by_id.loc[portfolio["loan_id"].astype(str)].to_numpy()
        portfolio[f"mean_return_v{VERSION}"] = mean_returns[portfolio_idx]
        periods = sorted(portfolio["period"].astype(str).unique())
        for period in periods:
            period_mask = portfolio["period"].astype(str).le(period).to_numpy()
            period_portfolio = portfolio.loc[period_mask].copy()
            period_idx = idx_by_id.loc[period_portfolio["loan_id"].astype(str)].to_numpy()
            cumulative_losses = losses[:, period_idx].sum(axis=1)
            cumulative_return = float(mean_returns[period_idx].sum())
            cumulative_exposure = float(period_portfolio["loan_amnt"].sum())
            cumulative_rows = int(len(period_portfolio))
            for scenario_id, scenario_loss in enumerate(cumulative_losses):
                rows.append(
                    {
                        "policy_id": policy_id,
                        f"period_v{VERSION}": period,
                        f"scenario_id_v{VERSION}": int(scenario_id),
                        f"cumulative_selected_rows_v{VERSION}": cumulative_rows,
                        f"cumulative_exposure_v{VERSION}": cumulative_exposure,
                        f"cumulative_objective_return_v{VERSION}": cumulative_return,
                        f"cumulative_scenario_loss_v{VERSION}": float(scenario_loss),
                        f"cumulative_wealth_proxy_v{VERSION}": float(
                            cumulative_return - scenario_loss
                        ),
                        f"claim_boundary_v{VERSION}": (
                            "periodized static-book common-path proxy; not live deployment replay"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _dynamic_proxy_summary(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, policy_trace in trace.groupby("policy_id", dropna=False):
        final_period = sorted(policy_trace[f"period_v{VERSION}"].astype(str).unique())[-1]
        final = policy_trace.loc[policy_trace[f"period_v{VERSION}"].astype(str).eq(final_period)]
        final_losses = final[f"cumulative_scenario_loss_v{VERSION}"].to_numpy(float)
        final_wealth = final[f"cumulative_wealth_proxy_v{VERSION}"].to_numpy(float)
        rows.append(
            {
                "policy_id": policy_id,
                f"period_count_v{VERSION}": int(policy_trace[f"period_v{VERSION}"].nunique()),
                f"trace_rows_v{VERSION}": int(len(policy_trace)),
                f"final_period_v{VERSION}": final_period,
                f"final_selected_rows_v{VERSION}": int(
                    final[f"cumulative_selected_rows_v{VERSION}"].max()
                ),
                f"final_exposure_v{VERSION}": float(final[f"cumulative_exposure_v{VERSION}"].max()),
                f"final_objective_return_v{VERSION}": float(
                    final[f"cumulative_objective_return_v{VERSION}"].max()
                ),
                f"final_loss_mean_v{VERSION}": float(final_losses.mean()),
                f"final_loss_cvar90_v{VERSION}": v70._tail_cvar(final_losses),
                f"final_wealth_proxy_mean_v{VERSION}": float(final_wealth.mean()),
                f"final_wealth_proxy_p10_v{VERSION}": float(np.quantile(final_wealth, 0.10)),
                f"worst_period_scenario_wealth_proxy_v{VERSION}": float(
                    policy_trace[f"cumulative_wealth_proxy_v{VERSION}"].min()
                ),
                f"claim_boundary_v{VERSION}": (
                    "dynamic proxy summary only; not a validated live deployment gate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _gate_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "valid_full_universe_gap_certificate",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v297_global_dynamic_gate_summary.csv",
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": (
                    "v297 records resource blocker and proxy evidence but no global bound"
                ),
            },
            {
                f"requirement_id_v{VERSION}": "post_v295_one_swap_gate_cleared",
                f"met_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v296_post_v295_one_swap_summary.csv",
                f"required_next_artifact_v{VERSION}": "none_for_one_swap_scope",
                f"claim_boundary_v{VERSION}": "local one-swap evidence only",
            },
            {
                f"requirement_id_v{VERSION}": "dynamic_proxy_replay_executed",
                f"met_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v297_dynamic_proxy_trace.parquet",
                f"required_next_artifact_v{VERSION}": "future_cashflow_or_live_replay_gate",
                f"claim_boundary_v{VERSION}": "proxy replay only; not deployment validation",
            },
            {
                f"requirement_id_v{VERSION}": "online_holdout_rerun_for_v295",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v297_claim_blockers.csv",
                f"required_next_artifact_v{VERSION}": f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv",
                f"claim_boundary_v{VERSION}": "online/source holdouts not rerun for v295",
            },
            {
                f"requirement_id_v{VERSION}": "ifrs9_proxy_rerun_for_v295",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v297_claim_blockers.csv",
                f"required_next_artifact_v{VERSION}": f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv",
                f"claim_boundary_v{VERSION}": "IFRS9 remains proxy-only and not rerun for v295",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v297 global-bound/dynamic proxy gate for v295.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v297_global_dynamic_gate_summary.csv"
                ),
                "boundary": "Resource-gated global audit plus periodized proxy replay.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v297 dynamic proxy replay favors v295 over v293 on common paths.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v297_dynamic_proxy_summary.csv"
                ),
                "boundary": "Static-book periodized proxy only; not live deployment validation.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v297 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v297_claim_blockers.csv"
                ),
                "boundary": "No full-universe branch-price or dual-bound certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v297 authorizes live deployment or Paper 4 working champion replacement.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v297_claim_blockers.csv"
                ),
                "boundary": "Online, cashflow/live replay and promotion gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v297 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v297_claim_blockers.csv"
                ),
                "boundary": "No final promotion, dynamic validation or deployment gate is created.",
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
                "lane": "Dynamic/Global",
                "executable_item": (
                    "v297 combines the unresolved full-universe branch-price blocker with "
                    "a periodized common-path proxy replay for v293 versus v295."
                ),
                "status": "dynamic_proxy_replay_executed_global_gap_still_open",
                "next_artifact": f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv",
                "success_condition": (
                    "rerun online/source, IFRS9 proxy and SPO-DLA gates for the v295/v296 "
                    "candidate without promoting"
                ),
                "last_wave": "v297",
                "execution_result": "v295_proxy_replay_beats_v293_but_global_dynamic_gates_remain",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v297")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V297_GLOBAL_DYNAMIC_GATE_START -->"
    end = "<!-- V297_GLOBAL_DYNAMIC_GATE_END -->"
    block = f"""
{start}

## Wave v297: Global-Bound / Dynamic Proxy Gate

Generated: {status["generated_at_utc"]}

### Objective

v296 cleared the post-v295 one-swap screen, so v297 moves to the next blocker:
full-universe branch-price/global evidence and dynamic validation. This wave
does not solve the full-v55 integer bound; it records that resource/global
blocker and runs a periodized common-path proxy replay for v293 versus v295.

### Results

- Full-v55 binary variables: `{status["full_binary_variables_v297"]}`.
- Direct MIP guard: `{status["direct_mip_binary_guard_v297"]}`.
- Valid full-universe gap certificate:
  `{status["valid_full_universe_gap_certificate_v297"]}`.
- Dynamic proxy trace rows: `{status["dynamic_proxy_trace_rows_v297"]}`.
- Dynamic proxy policies: `{status["dynamic_proxy_policy_count_v297"]}`.
- Common replay periods: `{status["dynamic_proxy_period_count_v297"]}`.
- Delta return v295 vs v293: `{status["delta_return_v295_vs_v293_v297"]}`.
- Delta CVaR90 v295 vs v293: `{status["delta_cvar90_v295_vs_v293_v297"]}`.
- v295 proxy replay beats v293:
  `{status["v295_dynamic_proxy_beats_v293_v297"]}`.
- Live deployment claim allowed:
  `{status["live_deployment_claim_allowed_v297"]}`.

### Interpretation

v297 strengthens the candidate story without crossing the claim boundary:
v295 beats v293 on return and CVaR in the same periodized static-book proxy
trace, but this is still not a full branch-price certificate, cashflow replay,
online holdout rerun or deployment validation.

### Claim Impact

- Allowed: v297 global/dynamic gate audit and common-path proxy replay.
- Still prohibited: full-universe global optimality, live deployment, Paper
  Estrella replacement, final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v297 in the living notebook. Promotion remains blocked; the next live-lab
wave should rerun online/source, IFRS9 proxy and SPO-DLA gates for v295.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v293 = read_parquet("paper4_v293_diverse_pool_allocations.parquet").reset_index(drop=True)
    v295 = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(drop=True)
    v296_status = json.loads((STATUS_DIR / "paper4_v296_status.json").read_text(encoding="utf-8"))
    v283_status = json.loads((STATUS_DIR / "paper4_v283_status.json").read_text(encoding="utf-8"))
    if universe.empty or v293.empty or v295.empty:
        raise RuntimeError("Missing v55, v293, or v295 inputs for v297.")
    if not bool(v296_status["post_v295_one_swap_local_optimality_cleared_v296"]):
        raise RuntimeError("v297 requires v296 local repricing to be cleared.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    v293["loan_id"] = v293["loan_id"].astype(str)
    v295["loan_id"] = v295["loan_id"].astype(str)
    policies = {
        "v293_diverse_pool_after_v294_gate": v293,
        "v295_broader_pool_after_v296_gate": v295,
    }
    portfolio_metrics = {
        policy_id: _portfolio_metrics(
            universe=universe,
            portfolio=portfolio,
            losses=losses,
            mean_returns=mean_returns,
        )
        for policy_id, portfolio in policies.items()
    }
    trace = _dynamic_proxy_trace(
        universe=universe,
        policies=policies,
        losses=losses,
        mean_returns=mean_returns,
    )
    proxy_summary = _dynamic_proxy_summary(trace)
    v293_metrics = portfolio_metrics["v293_diverse_pool_after_v294_gate"]
    v295_metrics = portfolio_metrics["v295_broader_pool_after_v296_gate"]
    delta_return = float(v295_metrics["objective_return"] - v293_metrics["objective_return"])
    delta_cvar = float(v295_metrics["scenario_loss_cvar90"] - v293_metrics["scenario_loss_cvar90"])
    period_distribution_match = (
        v293["period"].astype(str).value_counts().sort_index().to_dict()
        == v295["period"].astype(str).value_counts().sort_index().to_dict()
    )
    v295_beats_v293 = delta_return > 1e-9 and delta_cvar <= 1e-7 and period_distribution_match

    gate_summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v297_global_bound_dynamic_proxy_gate",
                f"incumbent_version_v{VERSION}": INCUMBENT_VERSION,
                f"challenger_version_v{VERSION}": CHALLENGER_VERSION,
                f"local_reprice_version_v{VERSION}": LOCAL_REPRICE_VERSION,
                f"direct_mip_guard_version_v{VERSION}": DIRECT_MIP_GUARD_VERSION,
                f"full_binary_variables_v{VERSION}": int(v283_status["full_binary_variables_v283"]),
                f"direct_mip_binary_guard_v{VERSION}": int(
                    v283_status["max_binary_vars_for_direct_mip_v283"]
                ),
                f"direct_full_mip_attempted_v{VERSION}": False,
                f"valid_full_universe_gap_certificate_v{VERSION}": False,
                f"dynamic_proxy_replay_executed_v{VERSION}": True,
                f"dynamic_proxy_trace_rows_v{VERSION}": int(len(trace)),
                f"dynamic_proxy_policy_count_v{VERSION}": int(trace["policy_id"].nunique()),
                f"dynamic_proxy_period_count_v{VERSION}": int(
                    trace[f"period_v{VERSION}"].nunique()
                ),
                f"period_distribution_match_v{VERSION}": period_distribution_match,
                f"v293_objective_return_v{VERSION}": float(v293_metrics["objective_return"]),
                f"v295_objective_return_v{VERSION}": float(v295_metrics["objective_return"]),
                f"delta_return_v295_vs_v293_v{VERSION}": delta_return,
                f"v293_cvar90_v{VERSION}": float(v293_metrics["scenario_loss_cvar90"]),
                f"v295_cvar90_v{VERSION}": float(v295_metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_v295_vs_v293_v{VERSION}": delta_cvar,
                f"v295_dynamic_proxy_beats_v293_v{VERSION}": v295_beats_v293,
                f"live_deployment_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "global/dynamic gate audit only; proxy replay is not deployment validation"
                ),
            }
        ]
    )
    requirements = _gate_requirements()
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "valid_global_gap_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no full-v55 branch-price or dual-bound certificate",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_proxy_not_live_deployment_replay",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(trace)),
                f"required_next_artifact_v{VERSION}": "future_cashflow_or_live_replay_gate",
                f"claim_boundary_v{VERSION}": "periodized static-book proxy only",
            },
            {
                f"blocker_id_v{VERSION}": "online_holdout_not_rerun_for_v295",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv"
                ),
                f"claim_boundary_v{VERSION}": "online/source holdouts not rerun for v295",
            },
            {
                f"blocker_id_v{VERSION}": "ifrs9_proxy_not_rerun_for_v295",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_online_ifrs9_spo_dla_gate_expansion.csv"
                ),
                f"claim_boundary_v{VERSION}": "IFRS9/SICR remains proxy-only and not rerun",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_final_promotion_forbidden",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "paper4_final_promotion_gate_not_created",
                f"claim_boundary_v{VERSION}": (
                    "Paper Estrella replacement and final Paper 4 remain prohibited"
                ),
            },
        ]
    )
    claim_matrix = pd.DataFrame(
        [
            {
                "claim_id": "v297_global_dynamic_gate_executed",
                "allowed": True,
                "artifact": "paper4_v297_global_dynamic_gate_summary.csv",
                "boundary": "gate audit plus dynamic proxy replay",
            },
            {
                "claim_id": "v297_v295_dynamic_proxy_beats_v293",
                "allowed": v295_beats_v293,
                "artifact": "paper4_v297_dynamic_proxy_summary.csv",
                "boundary": "periodized static-book proxy only",
            },
            {
                "claim_id": "v297_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v297_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v297_live_deployment_or_working_champion",
                "allowed": False,
                "artifact": "paper4_v297_claim_blockers.csv",
                "boundary": "live deployment and dynamic validation missing",
            },
            {
                "claim_id": "v297_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v297_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v297_global_dynamic_gate_summary.csv", gate_summary)
    write_csv(TABLE_DIR / "paper4_v297_dynamic_proxy_summary.csv", proxy_summary)
    trace.to_parquet(TABLE_DIR / "paper4_v297_dynamic_proxy_trace.parquet", index=False)
    write_csv(TABLE_DIR / "paper4_v297_gate_requirements.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v297_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v297_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = gate_summary.iloc[0]
    status = {
        "phase": "v297_full_universe_branch_price_bound_or_dynamic_replay",
        "schema_version": "2026-05-15.297",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "incumbent_version_v297": INCUMBENT_VERSION,
        "challenger_version_v297": CHALLENGER_VERSION,
        "local_reprice_version_v297": LOCAL_REPRICE_VERSION,
        "full_binary_variables_v297": int(row[f"full_binary_variables_v{VERSION}"]),
        "direct_mip_binary_guard_v297": int(row[f"direct_mip_binary_guard_v{VERSION}"]),
        "direct_full_mip_attempted_v297": False,
        "valid_full_universe_gap_certificate_v297": False,
        "dynamic_proxy_replay_executed_v297": True,
        "dynamic_proxy_trace_rows_v297": int(row[f"dynamic_proxy_trace_rows_v{VERSION}"]),
        "dynamic_proxy_policy_count_v297": int(row[f"dynamic_proxy_policy_count_v{VERSION}"]),
        "dynamic_proxy_period_count_v297": int(row[f"dynamic_proxy_period_count_v{VERSION}"]),
        "period_distribution_match_v297": bool(row[f"period_distribution_match_v{VERSION}"]),
        "delta_return_v295_vs_v293_v297": float(row[f"delta_return_v295_vs_v293_v{VERSION}"]),
        "delta_cvar90_v295_vs_v293_v297": float(row[f"delta_cvar90_v295_vs_v293_v{VERSION}"]),
        "v295_dynamic_proxy_beats_v293_v297": bool(
            row[f"v295_dynamic_proxy_beats_v293_v{VERSION}"]
        ),
        "live_deployment_claim_allowed_v297": False,
        "working_champion_claim_allowed_v297": False,
        "full_universe_integer_optimality_claim_allowed_v297": False,
        "paper1_promotion_allowed_v297": False,
        "paper4_working_champion_changed_v297": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "dynamic_proxy_summary_rows_v297": int(len(proxy_summary)),
        "gate_requirement_rows_v297": int(len(requirements)),
        "claim_blocker_rows_v297": int(len(blockers)),
        "claim_matrix_rows_v297": int(len(claim_matrix)),
        "next_artifact_v297": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v297 executes a global/dynamic gate audit and proxy replay; no global "
            "optimality, deployment, working champion, or final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v297_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v297": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
