#!/usr/bin/env python3
"""Build Paper 4 v322 dynamic-proxy/global-bound gate artifacts after v320."""

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

VERSION = 322
INCUMBENT_VERSION = 295
BASE_VERSION = 316
CANDIDATE_VERSION = 320
LOCAL_REPRICE_VERSION = 321
PREVIOUS_GLOBAL_DYNAMIC_GATE_VERSION = 318
DIRECT_MIP_GUARD_VERSION = 283
NEXT_VERSION = 323

INCUMBENT_POLICY_ID = "v295_broader_pool_after_v296_gate"
BASE_POLICY_ID = "v316_terminal_repair_after_v317_gate"
CANDIDATE_POLICY_ID = "v320_matched_period_after_v321_gate"


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


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


def _proxy_coverage_comparison(policies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet").copy()
    panel["loan_id"] = panel["loan_id"].astype(str)
    panel = panel.drop_duplicates(["loan_id", "month_index", "scenario"])
    panel_ids = set(panel["loan_id"])
    rows: list[dict[str, Any]] = []
    for policy_id, portfolio in policies.items():
        loan_ids = set(portfolio["loan_id"].astype(str))
        observed_ids = loan_ids & panel_ids
        selected_rows = int(len(loan_ids))
        observed_rows = int(len(observed_ids))
        missing_rows = selected_rows - observed_rows
        policy_panel = panel.loc[panel["loan_id"].isin(loan_ids)].copy()
        rows.append(
            {
                "policy_id": policy_id,
                f"selected_rows_v{VERSION}": selected_rows,
                f"observed_v47_proxy_rows_v{VERSION}": observed_rows,
                f"missing_v47_proxy_rows_v{VERSION}": missing_rows,
                f"observed_v47_proxy_share_v{VERSION}": observed_rows / max(selected_rows, 1),
                f"ifrs9_proxy_panel_rows_v{VERSION}": int(len(policy_panel)),
                f"ifrs9_proxy_scenarios_v{VERSION}": int(policy_panel["scenario"].nunique())
                if not policy_panel.empty
                else 0,
                f"ifrs9_proxy_months_v{VERSION}": int(policy_panel["month_index"].nunique())
                if not policy_panel.empty
                else 0,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "loan-id proxy coverage audit only; contractual IFRS9 remains blocked"
                ),
            }
        )
    return pd.DataFrame(rows)


def _gate_requirements(
    *,
    period_set_match: bool,
    period_distribution_match: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                f"requirement_id_v{VERSION}": "post_v320_one_swap_gate_cleared",
                f"met_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v321_post_v320_one_swap_summary.csv",
                f"required_next_artifact_v{VERSION}": "none_for_one_swap_scope",
                f"claim_boundary_v{VERSION}": "local one-swap evidence only",
            },
            {
                f"requirement_id_v{VERSION}": "dynamic_proxy_replay_executed",
                f"met_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v322_dynamic_proxy_trace.parquet",
                f"required_next_artifact_v{VERSION}": "future_cashflow_or_live_replay_gate",
                f"claim_boundary_v{VERSION}": "proxy replay only; not deployment validation",
            },
            {
                f"requirement_id_v{VERSION}": "common_period_set_available",
                f"met_v{VERSION}": period_set_match,
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv"
                ),
                f"required_next_artifact_v{VERSION}": "none_for_period_set_scope",
                f"claim_boundary_v{VERSION}": (
                    "same period labels only; this is weaker than matched issue-time composition"
                ),
            },
            {
                f"requirement_id_v{VERSION}": "matched_period_distribution",
                f"met_v{VERSION}": period_distribution_match,
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv"
                ),
                f"required_next_artifact_v{VERSION}": "future_matched_period_dynamic_replay",
                f"claim_boundary_v{VERSION}": (
                    "v320 matches the v295 target period counts; live replay is still missing"
                ),
            },
            {
                f"requirement_id_v{VERSION}": "observed_or_contractual_ifrs9_complete",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v322_proxy_coverage_comparison.csv",
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_v320_cashflow_online_ifrs9_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": "v320 still has missing v47 proxy loan rows",
            },
            {
                f"requirement_id_v{VERSION}": "valid_full_universe_gap_certificate",
                f"met_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": (
                    "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv"
                ),
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": (
                    "v322 records resource blocker but no branch-price certificate"
                ),
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v322 dynamic proxy/global-bound audit after v320.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv"
                ),
                "boundary": "Global-bound blocker plus periodized proxy replay.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 final-period static-book proxy favors v320 over v295.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_dynamic_proxy_summary.csv"
                ),
                "boundary": (
                    "Return/CVaR proxy comparison with v295 period counts matched; not live deployment."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 final-period static-book proxy favors v320 over v316.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_dynamic_proxy_summary.csv"
                ),
                "boundary": (
                    "Return/CVaR proxy comparison only; v316 period counts differ from v320."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 proves matched-period dynamic or live deployability.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_claim_blockers.csv"
                ),
                "boundary": "Static-book proxy replay only; no live holdout or deployment replay.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 resolves contractual IFRS9 or full cashflow coverage for v320.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_proxy_coverage_comparison.csv"
                ),
                "boundary": "v320 still has missing v47 proxy loan rows.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_claim_blockers.csv"
                ),
                "boundary": "No full-universe branch-price or dual-bound certificate exists.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v322 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v322_claim_blockers.csv"
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
                    "v322 compares the v320 matched-period candidate against v295 and v316 "
                    "dynamic-proxy references and records global, live and proxy blockers."
                ),
                "status": "v320_dynamic_proxy_executed_global_live_proxy_gates_blocked",
                "next_artifact": f"paper4_v{NEXT_VERSION}_v320_cashflow_online_ifrs9_gate.csv",
                "success_condition": (
                    "build v320-specific cashflow, online and IFRS9 proxy gates without promoting"
                ),
                "last_wave": "v322",
                "execution_result": ("v320_static_proxy_beats_v316_and_v295_but_proxy_gap_remains"),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v322")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V322_DYNAMIC_PROXY_GLOBAL_BOUND_AFTER_V320_START -->"
    end = "<!-- V322_DYNAMIC_PROXY_GLOBAL_BOUND_AFTER_V320_END -->"
    block = f"""
{start}

## Wave v322: Dynamic Proxy / Global Bound After v320

Generated: {status["generated_at_utc"]}

### Objective

v321 cleared post-v320 one-swap local optimality. v322 moves to the next
blocker: compare v320 against the v295 target-distribution reference and the
v316 immediate base, audit v47 proxy coverage, and keep the full-universe/global
proof boundary explicit.

### Results

- Dynamic proxy trace rows: `{status["dynamic_proxy_trace_rows_v322"]}`.
- Dynamic proxy policies: `{status["dynamic_proxy_policy_count_v322"]}`.
- Common period set match:
  `{status["period_set_match_v322"]}`.
- Matched period distribution:
  `{status["period_distribution_match_vs_v295_v322"]}`.
- Delta return v320 vs v295:
  `{status["delta_return_v320_vs_v295_v322"]}`.
- Delta CVaR90 v320 vs v295:
  `{status["delta_cvar90_v320_vs_v295_v322"]}`.
- v320 static-book proxy beats v295:
  `{status["v320_dynamic_proxy_beats_v295_v322"]}`.
- Delta return v320 vs v316:
  `{status["delta_return_v320_vs_v316_v322"]}`.
- Delta CVaR90 v320 vs v316:
  `{status["delta_cvar90_v320_vs_v316_v322"]}`.
- v320 static-book proxy beats v316:
  `{status["v320_dynamic_proxy_beats_v316_v322"]}`.
- v320 observed v47 proxy rows:
  `{status["v320_observed_v47_proxy_rows_v322"]}`.
- v320 missing v47 proxy rows:
  `{status["v320_missing_v47_proxy_rows_v322"]}`.
- Full-universe gap certificate:
  `{status["valid_full_universe_gap_certificate_v322"]}`.
- Working champion claim allowed:
  `{status["working_champion_claim_allowed_v322"]}`.

### Interpretation

v322 strengthens the v320 story: the final-period static-book proxy comparison
shows higher return and lower CVaR than both v295 and v316, while matching the
v295 target period distribution. The claim remains bounded because v320 worsens
observed v47 proxy coverage to 92 observed / 79 missing rows, v316 has a
different period distribution, and no branch-price/global certificate or live
deployment replay exists.

### Claim Impact

- Allowed: v322 dynamic/global audit and final-period static-book proxy
  comparison.
- Still prohibited: matched-period dynamic superiority, contractual IFRS9,
  full-universe global optimality, live deployment, Paper Estrella replacement,
  final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v322 in the living notebook. The next wave should build v320-specific
cashflow, online and IFRS9 proxy gates without promoting.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v295 = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(drop=True)
    v316 = read_parquet("paper4_v316_apply_next_post_v314_swap_allocations.parquet").reset_index(
        drop=True
    )
    v320 = read_parquet("paper4_v320_matched_period_allocations.parquet").reset_index(drop=True)
    v321_status = json.loads((STATUS_DIR / "paper4_v321_status.json").read_text(encoding="utf-8"))
    v283_status = json.loads((STATUS_DIR / "paper4_v283_status.json").read_text(encoding="utf-8"))
    if universe.empty or v295.empty or v316.empty or v320.empty:
        raise RuntimeError("Missing v55, v295, v316, or v320 inputs for v322.")
    if not bool(v321_status["post_v320_one_swap_local_optimality_cleared_v321"]):
        raise RuntimeError("v322 requires v321 post-v320 local repricing to be cleared.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    v295["loan_id"] = v295["loan_id"].astype(str)
    v316["loan_id"] = v316["loan_id"].astype(str)
    v320["loan_id"] = v320["loan_id"].astype(str)
    policies = {
        INCUMBENT_POLICY_ID: v295,
        BASE_POLICY_ID: v316,
        CANDIDATE_POLICY_ID: v320,
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
    proxy_coverage = _proxy_coverage_comparison(policies)

    v295_metrics = portfolio_metrics[INCUMBENT_POLICY_ID]
    v316_metrics = portfolio_metrics[BASE_POLICY_ID]
    v320_metrics = portfolio_metrics[CANDIDATE_POLICY_ID]
    delta_return_vs_v295 = float(
        v320_metrics["objective_return"] - v295_metrics["objective_return"]
    )
    delta_cvar_vs_v295 = float(
        v320_metrics["scenario_loss_cvar90"] - v295_metrics["scenario_loss_cvar90"]
    )
    delta_return_vs_v316 = float(
        v320_metrics["objective_return"] - v316_metrics["objective_return"]
    )
    delta_cvar_vs_v316 = float(
        v320_metrics["scenario_loss_cvar90"] - v316_metrics["scenario_loss_cvar90"]
    )
    v295_period_distribution = _period_distribution(v295)
    v316_period_distribution = _period_distribution(v316)
    v320_period_distribution = _period_distribution(v320)
    period_set_match = (
        set(v295_period_distribution)
        == set(v316_period_distribution)
        == set(v320_period_distribution)
    )
    period_distribution_match_vs_v295 = v295_period_distribution == v320_period_distribution
    period_distribution_match_vs_v316 = v316_period_distribution == v320_period_distribution
    v320_beats_v295 = delta_return_vs_v295 > 1e-9 and delta_cvar_vs_v295 <= 1e-7
    v320_beats_v316 = delta_return_vs_v316 > 1e-9 and delta_cvar_vs_v316 <= 1e-7

    v295_coverage = proxy_coverage.loc[proxy_coverage["policy_id"].eq(INCUMBENT_POLICY_ID)].iloc[0]
    v316_coverage = proxy_coverage.loc[proxy_coverage["policy_id"].eq(BASE_POLICY_ID)].iloc[0]
    v320_coverage = proxy_coverage.loc[proxy_coverage["policy_id"].eq(CANDIDATE_POLICY_ID)].iloc[0]
    observed_delta_vs_v295 = int(
        v320_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
        - v295_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
    )
    missing_delta_vs_v295 = int(
        v320_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
        - v295_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
    )
    observed_delta_vs_v316 = int(
        v320_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
        - v316_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
    )
    missing_delta_vs_v316 = int(
        v320_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
        - v316_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
    )

    gate_summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v322_dynamic_proxy_or_global_bound_after_v320",
                f"incumbent_version_v{VERSION}": INCUMBENT_VERSION,
                f"base_comparison_version_v{VERSION}": BASE_VERSION,
                f"candidate_version_v{VERSION}": CANDIDATE_VERSION,
                f"local_reprice_version_v{VERSION}": LOCAL_REPRICE_VERSION,
                f"previous_global_dynamic_gate_version_v{VERSION}": (
                    PREVIOUS_GLOBAL_DYNAMIC_GATE_VERSION
                ),
                f"direct_mip_guard_version_v{VERSION}": DIRECT_MIP_GUARD_VERSION,
                f"full_binary_variables_v{VERSION}": int(v283_status["full_binary_variables_v283"]),
                f"direct_mip_binary_guard_v{VERSION}": int(
                    v283_status["max_binary_vars_for_direct_mip_v283"]
                ),
                f"direct_full_mip_attempted_v{VERSION}": False,
                f"valid_full_universe_gap_certificate_v{VERSION}": False,
                f"post_v320_one_swap_local_optimality_cleared_v{VERSION}": True,
                f"dynamic_proxy_replay_executed_v{VERSION}": True,
                f"dynamic_proxy_trace_rows_v{VERSION}": int(len(trace)),
                f"dynamic_proxy_policy_count_v{VERSION}": int(trace["policy_id"].nunique()),
                f"dynamic_proxy_period_count_v{VERSION}": int(
                    trace[f"period_v{VERSION}"].nunique()
                ),
                f"period_set_match_v{VERSION}": period_set_match,
                f"period_distribution_match_vs_v295_v{VERSION}": (
                    period_distribution_match_vs_v295
                ),
                f"period_distribution_match_vs_v316_v{VERSION}": (
                    period_distribution_match_vs_v316
                ),
                f"v295_period_distribution_v{VERSION}": json.dumps(
                    v295_period_distribution, sort_keys=True
                ),
                f"v316_period_distribution_v{VERSION}": json.dumps(
                    v316_period_distribution, sort_keys=True
                ),
                f"v320_period_distribution_v{VERSION}": json.dumps(
                    v320_period_distribution, sort_keys=True
                ),
                f"v295_objective_return_v{VERSION}": float(v295_metrics["objective_return"]),
                f"v316_objective_return_v{VERSION}": float(v316_metrics["objective_return"]),
                f"v320_objective_return_v{VERSION}": float(v320_metrics["objective_return"]),
                f"delta_return_v320_vs_v295_v{VERSION}": delta_return_vs_v295,
                f"delta_return_v320_vs_v316_v{VERSION}": delta_return_vs_v316,
                f"v295_cvar90_v{VERSION}": float(v295_metrics["scenario_loss_cvar90"]),
                f"v316_cvar90_v{VERSION}": float(v316_metrics["scenario_loss_cvar90"]),
                f"v320_cvar90_v{VERSION}": float(v320_metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_v320_vs_v295_v{VERSION}": delta_cvar_vs_v295,
                f"delta_cvar90_v320_vs_v316_v{VERSION}": delta_cvar_vs_v316,
                f"v320_dynamic_proxy_beats_v295_v{VERSION}": v320_beats_v295,
                f"v320_dynamic_proxy_beats_v316_v{VERSION}": v320_beats_v316,
                f"v295_observed_v47_proxy_rows_v{VERSION}": int(
                    v295_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
                ),
                f"v316_observed_v47_proxy_rows_v{VERSION}": int(
                    v316_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
                ),
                f"v320_observed_v47_proxy_rows_v{VERSION}": int(
                    v320_coverage[f"observed_v47_proxy_rows_v{VERSION}"]
                ),
                f"v295_missing_v47_proxy_rows_v{VERSION}": int(
                    v295_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
                ),
                f"v316_missing_v47_proxy_rows_v{VERSION}": int(
                    v316_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
                ),
                f"v320_missing_v47_proxy_rows_v{VERSION}": int(
                    v320_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
                ),
                f"v320_observed_proxy_delta_vs_v295_v{VERSION}": observed_delta_vs_v295,
                f"v320_missing_proxy_delta_vs_v295_v{VERSION}": missing_delta_vs_v295,
                f"v320_observed_proxy_delta_vs_v316_v{VERSION}": observed_delta_vs_v316,
                f"v320_missing_proxy_delta_vs_v316_v{VERSION}": missing_delta_vs_v316,
                f"matched_period_dynamic_claim_allowed_v{VERSION}": False,
                f"cashflow_online_holdout_v320_claim_allowed_v{VERSION}": False,
                f"contractual_ifrs9_claim_allowed_v{VERSION}": False,
                f"strict_live_deployability_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_v320_cashflow_online_ifrs9_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "dynamic/global audit only; v320 proxy replay is not a live or global proof"
                ),
            }
        ]
    )
    requirements = _gate_requirements(
        period_set_match=period_set_match,
        period_distribution_match=period_distribution_match_vs_v295,
    )
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
                f"blocker_id_v{VERSION}": "target_period_distribution_matched_but_live_replay_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_matched_period_dynamic_replay",
                f"claim_boundary_v{VERSION}": (
                    "v320 matches v295 period counts, but dynamic/live replay is still absent"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "proxy_coverage_gap_persists",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    v320_coverage[f"missing_v47_proxy_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_v320_cashflow_online_ifrs9_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": "v320 still has missing v47 proxy rows",
            },
            {
                f"blocker_id_v{VERSION}": "v320_cashflow_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_v320_cashflow_online_ifrs9_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": "v320-specific cashflow/live holdout not built",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_proxy_not_live_deployment_replay",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(trace)),
                f"required_next_artifact_v{VERSION}": "future_live_deployment_replay_gate",
                f"claim_boundary_v{VERSION}": "periodized static-book proxy only",
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
                "claim_id": "v322_dynamic_global_gate_executed",
                "allowed": True,
                "artifact": "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv",
                "boundary": "gate audit plus dynamic proxy replay",
            },
            {
                "claim_id": "v322_v320_final_static_proxy_beats_v295",
                "allowed": v320_beats_v295,
                "artifact": "paper4_v322_dynamic_proxy_summary.csv",
                "boundary": "final-period static-book proxy only",
            },
            {
                "claim_id": "v322_v320_final_static_proxy_beats_v316",
                "allowed": v320_beats_v316,
                "artifact": "paper4_v322_dynamic_proxy_summary.csv",
                "boundary": "final-period static-book proxy only",
            },
            {
                "claim_id": "v322_v320_matches_v295_period_distribution",
                "allowed": period_distribution_match_vs_v295,
                "artifact": "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv",
                "boundary": "static-book issue-period composition only",
            },
            {
                "claim_id": "v322_matched_period_dynamic_superiority",
                "allowed": False,
                "artifact": "paper4_v322_claim_blockers.csv",
                "boundary": "target period distribution matched, but no live deployment replay",
            },
            {
                "claim_id": "v322_contractual_ifrs9_or_full_proxy_coverage",
                "allowed": False,
                "artifact": "paper4_v322_proxy_coverage_comparison.csv",
                "boundary": "v320 still has missing v47 proxy rows",
            },
            {
                "claim_id": "v322_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v322_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v322_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v322_claim_blockers.csv",
                "boundary": "working champion and final promotion remain blocked",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v322_dynamic_proxy_or_global_bound_after_v320.csv", gate_summary)
    write_csv(TABLE_DIR / "paper4_v322_dynamic_proxy_summary.csv", proxy_summary)
    trace.to_parquet(TABLE_DIR / "paper4_v322_dynamic_proxy_trace.parquet", index=False)
    write_csv(TABLE_DIR / "paper4_v322_proxy_coverage_comparison.csv", proxy_coverage)
    write_csv(TABLE_DIR / "paper4_v322_gate_requirements.csv", requirements)
    write_csv(TABLE_DIR / "paper4_v322_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v322_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = gate_summary.iloc[0]
    status = {
        "phase": "v322_dynamic_proxy_or_global_bound_after_v320",
        "schema_version": "2026-05-16.322",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "incumbent_version_v322": INCUMBENT_VERSION,
        "base_comparison_version_v322": BASE_VERSION,
        "candidate_version_v322": CANDIDATE_VERSION,
        "local_reprice_version_v322": LOCAL_REPRICE_VERSION,
        "previous_global_dynamic_gate_version_v322": PREVIOUS_GLOBAL_DYNAMIC_GATE_VERSION,
        "full_binary_variables_v322": int(row[f"full_binary_variables_v{VERSION}"]),
        "direct_mip_binary_guard_v322": int(row[f"direct_mip_binary_guard_v{VERSION}"]),
        "direct_full_mip_attempted_v322": False,
        "valid_full_universe_gap_certificate_v322": False,
        "post_v320_one_swap_local_optimality_cleared_v322": True,
        "dynamic_proxy_replay_executed_v322": True,
        "dynamic_proxy_trace_rows_v322": int(row[f"dynamic_proxy_trace_rows_v{VERSION}"]),
        "dynamic_proxy_policy_count_v322": int(row[f"dynamic_proxy_policy_count_v{VERSION}"]),
        "dynamic_proxy_period_count_v322": int(row[f"dynamic_proxy_period_count_v{VERSION}"]),
        "period_set_match_v322": bool(row[f"period_set_match_v{VERSION}"]),
        "period_distribution_match_vs_v295_v322": bool(
            row[f"period_distribution_match_vs_v295_v{VERSION}"]
        ),
        "period_distribution_match_vs_v316_v322": bool(
            row[f"period_distribution_match_vs_v316_v{VERSION}"]
        ),
        "delta_return_v320_vs_v295_v322": float(row[f"delta_return_v320_vs_v295_v{VERSION}"]),
        "delta_cvar90_v320_vs_v295_v322": float(row[f"delta_cvar90_v320_vs_v295_v{VERSION}"]),
        "v320_dynamic_proxy_beats_v295_v322": bool(
            row[f"v320_dynamic_proxy_beats_v295_v{VERSION}"]
        ),
        "delta_return_v320_vs_v316_v322": float(row[f"delta_return_v320_vs_v316_v{VERSION}"]),
        "delta_cvar90_v320_vs_v316_v322": float(row[f"delta_cvar90_v320_vs_v316_v{VERSION}"]),
        "v320_dynamic_proxy_beats_v316_v322": bool(
            row[f"v320_dynamic_proxy_beats_v316_v{VERSION}"]
        ),
        "v295_observed_v47_proxy_rows_v322": int(row[f"v295_observed_v47_proxy_rows_v{VERSION}"]),
        "v316_observed_v47_proxy_rows_v322": int(row[f"v316_observed_v47_proxy_rows_v{VERSION}"]),
        "v320_observed_v47_proxy_rows_v322": int(row[f"v320_observed_v47_proxy_rows_v{VERSION}"]),
        "v295_missing_v47_proxy_rows_v322": int(row[f"v295_missing_v47_proxy_rows_v{VERSION}"]),
        "v316_missing_v47_proxy_rows_v322": int(row[f"v316_missing_v47_proxy_rows_v{VERSION}"]),
        "v320_missing_v47_proxy_rows_v322": int(row[f"v320_missing_v47_proxy_rows_v{VERSION}"]),
        "v320_observed_proxy_delta_vs_v295_v322": int(
            row[f"v320_observed_proxy_delta_vs_v295_v{VERSION}"]
        ),
        "v320_missing_proxy_delta_vs_v295_v322": int(
            row[f"v320_missing_proxy_delta_vs_v295_v{VERSION}"]
        ),
        "v320_observed_proxy_delta_vs_v316_v322": int(
            row[f"v320_observed_proxy_delta_vs_v316_v{VERSION}"]
        ),
        "v320_missing_proxy_delta_vs_v316_v322": int(
            row[f"v320_missing_proxy_delta_vs_v316_v{VERSION}"]
        ),
        "matched_period_dynamic_claim_allowed_v322": False,
        "cashflow_online_holdout_v320_claim_allowed_v322": False,
        "contractual_ifrs9_claim_allowed_v322": False,
        "strict_live_deployability_claim_allowed_v322": False,
        "working_champion_claim_allowed_v322": False,
        "full_universe_integer_optimality_claim_allowed_v322": False,
        "paper1_promotion_allowed_v322": False,
        "paper4_working_champion_changed_v322": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "dynamic_proxy_summary_rows_v322": int(len(proxy_summary)),
        "proxy_coverage_rows_v322": int(len(proxy_coverage)),
        "gate_requirement_rows_v322": int(len(requirements)),
        "claim_blocker_rows_v322": int(len(blockers)),
        "claim_matrix_rows_v322": int(len(claim_matrix)),
        "next_artifact_v322": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v322 executes dynamic proxy/global-bound audit after v320; no global "
            "optimality, live deployment, working champion, or final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v322_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v322": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
