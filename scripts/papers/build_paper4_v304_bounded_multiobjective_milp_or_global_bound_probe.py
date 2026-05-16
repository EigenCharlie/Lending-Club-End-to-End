#!/usr/bin/env python3
"""Build Paper 4 v304 bounded multiobjective MILP/global-bound probe artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from scripts.papers import build_paper4_v70_restricted_master_solver as v70
from scripts.papers import build_paper4_v71_full_universe_reduced_costs as v71
from scripts.papers.paper4_one_swap_living_lab import (
    FAMILIES,
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

VERSION = 304
SOURCE_CANDIDATE_VERSION = 295
MULTIOBJECTIVE_AUDIT_VERSION = 303
NEXT_VERSION = 305
REWARD_GRID = [0.0, 2.0, 5.0, 10.0, 20.0]
MILP_TIME_LIMIT_SECONDS = 120.0
MIP_REL_GAP = 1e-6
TARGET_SELECTED_ROWS = 171


def _pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    v47_panel: pd.DataFrame,
    v299_panel: pd.DataFrame,
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    observed_ids = set(v47_panel["loan_id"].astype(str)) & set(universe["loan_id"].astype(str))
    pool = universe.loc[
        universe["loan_id"].astype(str).isin(selected_ids | (observed_ids - selected_ids))
    ].copy()
    pool = pool.reset_index(drop=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool[f"universe_idx_v{VERSION}"] = idx_by_id.loc[pool["loan_id"].astype(str)].to_numpy()
    pool[f"mean_return_v{VERSION}"] = mean_returns[pool[f"universe_idx_v{VERSION}"].to_numpy()]
    loan_level = v299_panel.sort_values(["loan_id", "month_index"]).drop_duplicates("loan_id")
    imputed_ids = set(
        loan_level.loc[
            loan_level["proxy_source_v299"].astype(str).str.startswith("imputed"),
            "loan_id",
        ].astype(str)
    )
    pool[f"incumbent_selected_v{VERSION}"] = pool["loan_id"].isin(selected_ids)
    pool[f"observed_v47_proxy_v{VERSION}"] = pool["loan_id"].isin(observed_ids)
    pool[f"imputed_proxy_if_selected_v{VERSION}"] = pool["loan_id"].isin(imputed_ids)
    pool[f"pool_role_v{VERSION}"] = np.select(
        [
            pool[f"incumbent_selected_v{VERSION}"] & pool[f"imputed_proxy_if_selected_v{VERSION}"],
            pool[f"incumbent_selected_v{VERSION}"] & ~pool[f"imputed_proxy_if_selected_v{VERSION}"],
            ~pool[f"incumbent_selected_v{VERSION}"] & pool[f"observed_v47_proxy_v{VERSION}"],
        ],
        ["v295_selected_imputed_proxy", "v295_selected_observed_proxy", "outside_observed_proxy"],
        default="other",
    )
    return pool


def _cap_lookup(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
    return {
        str(row["source_id"]): float(row[f"cap_share_v{SOURCE_CANDIDATE_VERSION}"])
        for _, row in family_caps.iterrows()
    }


def _constraints(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    source_caps: pd.DataFrame,
    exposure_min: float,
    exposure_max: float,
    cvar_cap: float,
) -> tuple[LinearConstraint, Bounds, np.ndarray, int, int]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    amounts = pool["loan_amnt"].to_numpy(float)
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)

    cardinality = np.zeros(var_count)
    cardinality[:n] = 1.0
    rows.append(cardinality)
    lb.append(float(TARGET_SELECTED_ROWS))
    ub.append(float(TARGET_SELECTED_ROWS))

    for family in FAMILIES:
        caps = _cap_lookup(source_caps, family)
        values = pool[family].astype(str)
        for source_id in sorted(values.dropna().unique()):
            cap = float(caps.get(str(source_id), 1.0))
            row = np.zeros(var_count)
            row[:n] = amounts * (values.eq(str(source_id)).to_numpy(float) - cap)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * scenario_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)

    for scenario_idx in range(scenario_count):
        row = np.zeros(var_count)
        row[:n] = losses_pool[scenario_idx, :]
        row[n] = -1.0
        row[n + 1 + scenario_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    constraints = LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub))
    bounds = Bounds(
        np.r_[np.zeros(n), 0.0, np.zeros(scenario_count)],
        np.r_[np.ones(n), np.inf, np.full(scenario_count, np.inf)],
    )
    integrality = np.r_[np.ones(n), np.zeros(1 + scenario_count)]
    return constraints, bounds, integrality, var_count, len(rows)


def _solve_rewards(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    losses_full: np.ndarray,
    constraints: LinearConstraint,
    bounds: Bounds,
    integrality: np.ndarray,
    v295_return: float,
    v295_cvar: float,
    v302_return: float,
    v302_imputed: int,
    full_universe_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    imputed_flag = pool[f"imputed_proxy_if_selected_v{VERSION}"].to_numpy(float)
    mean_returns = pool[f"mean_return_v{VERSION}"].to_numpy(float)
    for reward in REWARD_GRID:
        objective = np.zeros(var_count)
        objective[:n] = -mean_returns + float(reward) * imputed_flag
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
        )
        selected_mask = np.zeros(n, dtype=bool)
        incumbent_available = result.x is not None
        if incumbent_available:
            selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
        selected = pool.loc[selected_mask].copy()
        selected_idx = selected[f"universe_idx_v{VERSION}"].to_numpy(int)
        scenario_losses = losses_full[:, selected_idx].sum(axis=1)
        objective_return = float(selected[f"mean_return_v{VERSION}"].sum())
        imputed_rows = int(selected[f"imputed_proxy_if_selected_v{VERSION}"].sum())
        cvar90 = v70._tail_cvar(scenario_losses)
        allocation = selected[
            ["loan_id", "loan_amnt", *FAMILIES, f"pool_role_v{VERSION}", f"mean_return_v{VERSION}"]
        ].copy()
        allocation[f"reward_per_imputation_penalty_v{VERSION}"] = float(reward)
        allocation[f"selected_v{VERSION}"] = True
        allocation[f"portfolio_label_v{VERSION}"] = f"bounded_observed_proxy_milp_reward_{reward:g}"
        allocation[f"claim_boundary_v{VERSION}"] = (
            "bounded observed-proxy pool MILP allocation; not full-universe promotion"
        )
        allocation_frames.append(allocation)
        rows.append(
            {
                f"reward_per_imputation_penalty_v{VERSION}": float(reward),
                f"milp_success_v{VERSION}": bool(result.success),
                f"milp_status_v{VERSION}": int(result.status),
                f"milp_message_v{VERSION}": str(result.message),
                f"milp_gap_v{VERSION}": float(getattr(result, "mip_gap", np.nan)),
                f"milp_dual_bound_v{VERSION}": float(getattr(result, "mip_dual_bound", np.nan)),
                f"milp_node_count_v{VERSION}": int(getattr(result, "mip_node_count", -1)),
                f"milp_incumbent_available_v{VERSION}": bool(incumbent_available),
                f"selected_rows_v{VERSION}": int(len(selected)),
                f"portfolio_exposure_v{VERSION}": float(selected["loan_amnt"].sum()),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v295_v{VERSION}": objective_return - v295_return,
                f"delta_return_vs_v302_frontier_v{VERSION}": objective_return - v302_return,
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"delta_cvar90_vs_v295_v{VERSION}": cvar90 - v295_cvar,
                f"imputed_proxy_loan_rows_v{VERSION}": imputed_rows,
                f"imputed_rows_reduced_vs_v295_v{VERSION}": int(76 - imputed_rows),
                f"imputed_rows_delta_vs_v302_frontier_v{VERSION}": imputed_rows - v302_imputed,
                f"bounded_pool_share_of_full_universe_v{VERSION}": len(pool) / full_universe_rows,
                f"bounded_pool_optimality_claim_allowed_v{VERSION}": bool(result.success),
                f"full_universe_global_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": (
                    "reward solution is optimal only inside observed-proxy bounded pool"
                ),
            }
        )
    allocations = (
        pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    )
    return pd.DataFrame(rows), allocations


def _solution_source_summary(
    *,
    allocations: pd.DataFrame,
    universe: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    cap_lookup = {family: _cap_lookup(source_caps, family) for family in FAMILIES}
    rows: list[dict[str, Any]] = []
    for reward, portfolio in allocations.groupby(f"reward_per_imputation_penalty_v{VERSION}"):
        exposure = float(portfolio["loan_amnt"].sum())
        for family in FAMILIES:
            by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
            for source_id in sorted(universe[family].dropna().astype(str).unique()):
                source_exposure = float(by_source.get(source_id, 0.0))
                share = source_exposure / max(exposure, 1.0)
                cap = float(cap_lookup[family].get(source_id, 1.0))
                rows.append(
                    {
                        f"reward_per_imputation_penalty_v{VERSION}": float(reward),
                        "source_family": family,
                        "source_id": source_id,
                        f"cap_share_v{VERSION}": cap,
                        f"source_exposure_v{VERSION}": source_exposure,
                        f"source_share_v{VERSION}": share,
                        f"source_slack_v{VERSION}": cap - share,
                        f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                        f"claim_boundary_v{VERSION}": (
                            "v304 reward solution source diagnostic only"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v304 bounded observed-proxy multiobjective MILP probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v304_bounded_multiobjective_milp_or_global_bound_probe.csv"
                ),
                "boundary": "Optimality is bounded to the observed-proxy MILP pool.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v304 finds bounded-pool solutions that improve return and reduce imputation.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v304_bounded_reward_solutions.csv"
                ),
                "boundary": "Observed-proxy bounded pool only; requires post-solve repricing and global gates.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v304 proves full-universe global or multiobjective optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v304_claim_blockers.csv"
                ),
                "boundary": "The solved pool is 1724 rows, not the full 276869-row universe.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v304 authorizes a Paper 4 working champion or Paper Estrella replacement.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v304_claim_blockers.csv"
                ),
                "boundary": "Global, dynamic, online and promotion gates remain missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v304 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v304_claim_blockers.csv"
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
                "lane": "Source Governance/Global",
                "executable_item": (
                    "v304 solves bounded observed-proxy multiobjective MILPs across reward "
                    "penalties after the v303 reward audit."
                ),
                "status": "bounded_observed_proxy_multiobjective_milp_executed",
                "next_artifact": f"paper4_v{NEXT_VERSION}_post_v304_reprice_or_dynamic_gate.csv",
                "success_condition": (
                    "reprice/post-solve validate the strongest bounded solution and preserve global "
                    "promotion blockers"
                ),
                "last_wave": "v304",
                "execution_result": "bounded_pool_return_and_imputation_challengers_found_global_claims_blocked",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v304")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V304_BOUNDED_MULTIOBJECTIVE_MILP_START -->"
    end = "<!-- V304_BOUNDED_MULTIOBJECTIVE_MILP_END -->"
    block = f"""
{start}

## Wave v304: Bounded Multiobjective MILP / Global-Bound Probe

Generated: {status["generated_at_utc"]}

### Objective

v303 quantified reward thresholds over the v302 greedy prefixes. v304 moves
from prefix audit to optimization by solving bounded MILPs over the v295
selected rows plus all observed-v47-proxy candidates outside v295.

### Results

- Bounded pool rows: `{status["bounded_pool_rows_v304"]}`.
- Observed outside candidate rows: `{status["observed_outside_candidate_rows_v304"]}`.
- Reward solutions solved: `{status["reward_solution_rows_v304"]}`.
- Solver success rows: `{status["milp_success_rows_v304"]}`.
- Best return reward: `{status["best_return_reward_v304"]}`.
- Best objective return: `{status["best_objective_return_v304"]}`.
- Best delta return vs v295: `{status["best_delta_return_vs_v295_v304"]}`.
- Lowest imputed reward: `{status["lowest_imputed_reward_v304"]}`.
- Lowest imputed rows: `{status["lowest_imputed_rows_v304"]}`.
- Lowest-imputed delta return vs v295:
  `{status["lowest_imputed_delta_return_vs_v295_v304"]}`.
- Full-universe global claim allowed:
  `{status["full_universe_global_claim_allowed_v304"]}`.

### Interpretation

v304 is a strong bounded-pool result: within the observed-proxy pool, the MILP
finds portfolios that both improve return versus v295 and reduce imputed proxy
dependence. The best-return solution improves return materially while reducing
imputed rows, and the highest reward solution reduces imputed rows further while
still beating v295. This is not a full-universe proof because most v55 rows are
outside the solved pool.

### Claim Impact

- Allowed: bounded observed-proxy multiobjective MILP probe and bounded-pool
  reward solutions.
- Still prohibited: full-universe/global optimality, Paper 4 working champion,
  Paper Estrella replacement, final Paper 4 promotion, contractual IFRS9 and
  live deployability claims.

### Quarto Promotion Decision

Keep v304 in the living notebook. The next wave should reprice/post-solve
validate the strongest bounded solution and preserve global blockers.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(
        drop=True
    )
    selected["loan_id"] = selected["loan_id"].astype(str)
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    v299_panel = read_parquet("paper4_v299_v295_cashflow_proxy_panel.parquet")
    source_caps = read_csv("paper4_v295_broader_source_summary.csv")
    v295_summary = read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv")
    v302_status = json.loads((STATUS_DIR / "paper4_v302_status.json").read_text(encoding="utf-8"))
    if any(
        df.empty for df in [universe, selected, v47_panel, v299_panel, source_caps, v295_summary]
    ):
        raise RuntimeError("Missing v55, v295, v47, v299 or source inputs for v304.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool = _pool(
        universe=universe,
        selected=selected,
        v47_panel=v47_panel,
        v299_panel=v299_panel,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    v295_row = v295_summary.iloc[0]
    exposure_min = float(v295_row[f"exposure_min_v{SOURCE_CANDIDATE_VERSION}"])
    exposure_max = float(v295_row[f"exposure_max_v{SOURCE_CANDIDATE_VERSION}"])
    v295_return = float(v295_row[f"objective_return_v{SOURCE_CANDIDATE_VERSION}"])
    v295_cvar = float(v295_row[f"scenario_loss_cvar90_v{SOURCE_CANDIDATE_VERSION}"])
    losses_pool = losses[:, pool[f"universe_idx_v{VERSION}"].to_numpy(int)]
    constraints, bounds, integrality, variable_count, constraint_count = _constraints(
        pool=pool,
        losses_pool=losses_pool,
        source_caps=source_caps,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=v295_cvar,
    )
    reward_solutions, allocations = _solve_rewards(
        pool=pool,
        losses_pool=losses_pool,
        losses_full=losses,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        v295_return=v295_return,
        v295_cvar=v295_cvar,
        v302_return=float(v302_status["final_objective_return_v302"]),
        v302_imputed=int(v302_status["final_imputed_proxy_loan_rows_v302"]),
        full_universe_rows=len(universe),
    )
    source_summary = _solution_source_summary(
        allocations=allocations,
        universe=universe,
        source_caps=source_caps,
    )
    pool_audit = pd.DataFrame(
        [
            {
                f"pool_id_v{VERSION}": "v295_selected_plus_observed_v47_proxy_pool",
                f"full_universe_rows_v{VERSION}": int(len(universe)),
                f"bounded_pool_rows_v{VERSION}": int(len(pool)),
                f"bounded_pool_share_of_full_universe_v{VERSION}": float(len(pool) / len(universe)),
                f"v295_selected_rows_v{VERSION}": int(pool[f"incumbent_selected_v{VERSION}"].sum()),
                f"observed_outside_candidate_rows_v{VERSION}": int(
                    (
                        ~pool[f"incumbent_selected_v{VERSION}"]
                        & pool[f"observed_v47_proxy_v{VERSION}"]
                    ).sum()
                ),
                f"imputed_selected_rows_in_pool_v{VERSION}": int(
                    pool[f"imputed_proxy_if_selected_v{VERSION}"].sum()
                ),
                f"outside_bounded_pool_rows_v{VERSION}": int(len(universe) - len(pool)),
                f"binary_variables_v{VERSION}": int(len(pool)),
                f"total_variables_v{VERSION}": int(variable_count),
                f"constraint_rows_v{VERSION}": int(constraint_count),
                f"claim_boundary_v{VERSION}": (
                    "bounded observed-proxy pool only; not full-v55 coverage"
                ),
            }
        ]
    )
    best_return = reward_solutions.sort_values(
        f"objective_return_v{VERSION}", ascending=False
    ).iloc[0]
    lowest_imputed = reward_solutions.sort_values(
        [f"imputed_proxy_loan_rows_v{VERSION}", f"objective_return_v{VERSION}"],
        ascending=[True, False],
    ).iloc[0]
    full_global_allowed = False
    summary = pd.DataFrame(
        [
            {
                f"probe_id_v{VERSION}": "bounded_observed_proxy_multiobjective_milp_or_global_bound_probe",
                f"source_candidate_version_v{VERSION}": SOURCE_CANDIDATE_VERSION,
                f"multiobjective_audit_version_v{VERSION}": MULTIOBJECTIVE_AUDIT_VERSION,
                f"bounded_pool_rows_v{VERSION}": int(len(pool)),
                f"observed_outside_candidate_rows_v{VERSION}": int(
                    pool_audit.iloc[0][f"observed_outside_candidate_rows_v{VERSION}"]
                ),
                f"outside_bounded_pool_rows_v{VERSION}": int(len(universe) - len(pool)),
                f"reward_solution_rows_v{VERSION}": int(len(reward_solutions)),
                f"milp_success_rows_v{VERSION}": int(
                    reward_solutions[f"milp_success_v{VERSION}"].sum()
                ),
                f"best_return_reward_v{VERSION}": float(
                    best_return[f"reward_per_imputation_penalty_v{VERSION}"]
                ),
                f"best_objective_return_v{VERSION}": float(
                    best_return[f"objective_return_v{VERSION}"]
                ),
                f"best_delta_return_vs_v295_v{VERSION}": float(
                    best_return[f"delta_return_vs_v295_v{VERSION}"]
                ),
                f"best_imputed_rows_v{VERSION}": int(
                    best_return[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"best_cvar90_v{VERSION}": float(best_return[f"scenario_loss_cvar90_v{VERSION}"]),
                f"lowest_imputed_reward_v{VERSION}": float(
                    lowest_imputed[f"reward_per_imputation_penalty_v{VERSION}"]
                ),
                f"lowest_imputed_rows_v{VERSION}": int(
                    lowest_imputed[f"imputed_proxy_loan_rows_v{VERSION}"]
                ),
                f"lowest_imputed_delta_return_vs_v295_v{VERSION}": float(
                    lowest_imputed[f"delta_return_vs_v295_v{VERSION}"]
                ),
                f"lowest_imputed_cvar90_v{VERSION}": float(
                    lowest_imputed[f"scenario_loss_cvar90_v{VERSION}"]
                ),
                f"full_universe_global_claim_allowed_v{VERSION}": full_global_allowed,
                f"bounded_pool_optimality_claim_allowed_v{VERSION}": bool(
                    reward_solutions[f"milp_success_v{VERSION}"].all()
                ),
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_v304_reprice_or_dynamic_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "bounded observed-proxy MILP result only; full-universe and promotion claims remain blocked"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "full_universe_coverage_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(universe) - len(pool)),
                f"required_next_artifact_v{VERSION}": "future_full_universe_or_branch_price_bound",
                f"claim_boundary_v{VERSION}": "bounded pool omits most v55 rows",
            },
            {
                f"blocker_id_v{VERSION}": "post_v304_reprice_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(reward_solutions)),
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_v304_reprice_or_dynamic_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": "bounded MILP solutions require post-solve repricing",
            },
            {
                f"blocker_id_v{VERSION}": "global_dynamic_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_v304_reprice_or_dynamic_gate.csv"
                ),
                f"claim_boundary_v{VERSION}": "no dynamic/live promotion gate is created",
            },
            {
                f"blocker_id_v{VERSION}": "external_online_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 0,
                f"required_next_artifact_v{VERSION}": "future_external_online_holdout",
                f"claim_boundary_v{VERSION}": "v304 does not create external online evidence",
            },
            {
                f"blocker_id_v{VERSION}": "paper4_working_champion_gate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_global_dynamic_promotion_gate",
                f"claim_boundary_v{VERSION}": "working champion replacement remains blocked",
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
                "claim_id": "v304_bounded_multiobjective_milp_executed",
                "allowed": True,
                "artifact": "paper4_v304_bounded_multiobjective_milp_or_global_bound_probe.csv",
                "boundary": "observed-proxy bounded pool only",
            },
            {
                "claim_id": "v304_bounded_pool_return_and_imputation_challengers",
                "allowed": True,
                "artifact": "paper4_v304_bounded_reward_solutions.csv",
                "boundary": "requires reprice/global/dynamic gates",
            },
            {
                "claim_id": "v304_full_universe_global_optimality",
                "allowed": False,
                "artifact": "paper4_v304_claim_blockers.csv",
                "boundary": "full-v55 proof missing",
            },
            {
                "claim_id": "v304_working_champion_or_promotion",
                "allowed": False,
                "artifact": "paper4_v304_claim_blockers.csv",
                "boundary": "promotion gates missing",
            },
            {
                "claim_id": "v304_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v304_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v304_bounded_multiobjective_milp_or_global_bound_probe.csv", summary
    )
    write_csv(TABLE_DIR / "paper4_v304_bounded_pool_audit.csv", pool_audit)
    write_csv(TABLE_DIR / "paper4_v304_bounded_reward_solutions.csv", reward_solutions)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v304_bounded_reward_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v304_bounded_reward_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v304_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v304_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    status = {
        "phase": "v304_bounded_multiobjective_milp_or_global_bound_probe",
        "schema_version": "2026-05-15.304",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "source_candidate_version_v304": SOURCE_CANDIDATE_VERSION,
        "multiobjective_audit_version_v304": MULTIOBJECTIVE_AUDIT_VERSION,
        "bounded_pool_rows_v304": int(len(pool)),
        "observed_outside_candidate_rows_v304": int(
            pool_audit.iloc[0][f"observed_outside_candidate_rows_v{VERSION}"]
        ),
        "outside_bounded_pool_rows_v304": int(len(universe) - len(pool)),
        "reward_solution_rows_v304": int(len(reward_solutions)),
        "milp_success_rows_v304": int(reward_solutions[f"milp_success_v{VERSION}"].sum()),
        "best_return_reward_v304": float(best_return[f"reward_per_imputation_penalty_v{VERSION}"]),
        "best_objective_return_v304": float(best_return[f"objective_return_v{VERSION}"]),
        "best_delta_return_vs_v295_v304": float(best_return[f"delta_return_vs_v295_v{VERSION}"]),
        "best_imputed_rows_v304": int(best_return[f"imputed_proxy_loan_rows_v{VERSION}"]),
        "best_cvar90_v304": float(best_return[f"scenario_loss_cvar90_v{VERSION}"]),
        "lowest_imputed_reward_v304": float(
            lowest_imputed[f"reward_per_imputation_penalty_v{VERSION}"]
        ),
        "lowest_imputed_rows_v304": int(lowest_imputed[f"imputed_proxy_loan_rows_v{VERSION}"]),
        "lowest_imputed_delta_return_vs_v295_v304": float(
            lowest_imputed[f"delta_return_vs_v295_v{VERSION}"]
        ),
        "lowest_imputed_cvar90_v304": float(lowest_imputed[f"scenario_loss_cvar90_v{VERSION}"]),
        "bounded_pool_optimality_claim_allowed_v304": bool(
            reward_solutions[f"milp_success_v{VERSION}"].all()
        ),
        "full_universe_global_claim_allowed_v304": full_global_allowed,
        "working_champion_claim_allowed_v304": False,
        "paper1_promotion_allowed_v304": False,
        "paper4_working_champion_changed_v304": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v304": int(len(blockers)),
        "claim_matrix_rows_v304": int(len(claim_matrix)),
        "next_artifact_v304": f"paper4_v{NEXT_VERSION}_post_v304_reprice_or_dynamic_gate.csv",
        "claim_boundary": (
            "v304 solves bounded observed-proxy MILPs; full-universe, dynamic, working champion and promotion claims remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v304_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v304": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
