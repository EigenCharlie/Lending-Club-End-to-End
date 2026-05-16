#!/usr/bin/env python3
"""Build Paper 4 v320 matched-period bounded MILP/branch-price protocol artifacts."""

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

VERSION = 320
BASE_VERSION = 316
MATCH_TARGET_VERSION = 295
GATE_VERSION = 319
NEXT_VERSION = 321
CANDIDATE_POOL_LIMIT = 1500
MILP_TIME_LIMIT_SECONDS = 90.0
MIP_REL_GAP = 1e-6
TARGET_SELECTED_ROWS = 171


def _cap_lookup(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
    return {
        str(row["source_id"]): float(row[f"cap_share_v{BASE_VERSION}"])
        for _, row in family_caps.iterrows()
    }


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


def _build_pool(
    *,
    universe: pd.DataFrame,
    v316: pd.DataFrame,
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    selected_ids = set(v316["loan_id"].astype(str))
    universe = universe.copy()
    universe["loan_id"] = universe["loan_id"].astype(str)
    universe[f"mean_return_v{VERSION}"] = mean_returns
    all_omitted_2019 = universe.loc[
        ~universe["loan_id"].isin(selected_ids) & universe["period"].astype(str).eq("2019")
    ].copy()
    candidates = all_omitted_2019.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
        CANDIDATE_POOL_LIMIT
    )
    selected_universe = universe.loc[universe["loan_id"].isin(selected_ids)].copy()
    pool = pd.concat([selected_universe, candidates], ignore_index=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool[f"incumbent_selected_v{VERSION}"] = pool["loan_id"].isin(selected_ids)
    pool[f"omitted_2019_candidate_v{VERSION}"] = ~pool[f"incumbent_selected_v{VERSION}"]
    pool[f"universe_idx_v{VERSION}"] = idx_by_id.loc[pool["loan_id"]].to_numpy()
    pool[f"pool_role_v{VERSION}"] = np.where(
        pool[f"incumbent_selected_v{VERSION}"],
        "v316_selected_base",
        "top_omitted_2019_candidate",
    )
    return pool, int(len(all_omitted_2019) - len(candidates))


def _constraints(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    source_caps: pd.DataFrame,
    target_period_counts: dict[str, int],
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

    for period, target_count in target_period_counts.items():
        row = np.zeros(var_count)
        row[:n] = pool["period"].astype(str).eq(str(period)).to_numpy(float)
        rows.append(row)
        lb.append(float(target_count))
        ub.append(float(target_count))

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


def _solve_matched_period(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    losses_full: np.ndarray,
    constraints: LinearConstraint,
    bounds: Bounds,
    integrality: np.ndarray,
) -> tuple[object, pd.DataFrame, dict[str, Any]]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    objective = np.zeros(var_count)
    objective[:n] = -pool[f"mean_return_v{VERSION}"].to_numpy(float)
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    selected = pd.DataFrame()
    metrics: dict[str, Any] = {}
    if result.x is not None:
        selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
        selected = pool.loc[selected_mask].copy()
        selected_idx = selected[f"universe_idx_v{VERSION}"].to_numpy(int)
        scenario_losses = losses_full[:, selected_idx].sum(axis=1)
        metrics = {
            "selected_rows": int(len(selected)),
            "portfolio_exposure": float(selected["loan_amnt"].sum()),
            "objective_return": float(selected[f"mean_return_v{VERSION}"].sum()),
            "scenario_loss_mean": float(scenario_losses.mean()),
            "scenario_loss_cvar90": v70._tail_cvar(scenario_losses),
            "period_distribution": _period_distribution(selected),
        }
    return result, selected, metrics


def _source_summary(
    *,
    selected: pd.DataFrame,
    universe: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    cap_lookup = {family: _cap_lookup(source_caps, family) for family in FAMILIES}
    exposure = float(selected["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        by_source = selected.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            source_exposure = float(by_source.get(source_id, 0.0))
            share = source_exposure / max(exposure, 1.0)
            cap = float(cap_lookup[family].get(source_id, 1.0))
            rows.append(
                {
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": (
                        "v320 matched-period bounded MILP source diagnostic only"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _action_table(
    *, pool: pd.DataFrame, selected: pd.DataFrame, observed_ids: set[str]
) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    pool = pool.copy()
    pool["loan_id"] = pool["loan_id"].astype(str)
    actions = pool.loc[
        (pool[f"incumbent_selected_v{VERSION}"] & ~pool["loan_id"].isin(selected_ids))
        | (~pool[f"incumbent_selected_v{VERSION}"] & pool["loan_id"].isin(selected_ids))
    ].copy()
    actions[f"action_v{VERSION}"] = np.where(
        actions["loan_id"].isin(selected_ids), "add_2019_candidate", "drop_v316_selected"
    )
    actions[f"observed_v47_proxy_v{VERSION}"] = actions["loan_id"].isin(observed_ids)
    actions[f"claim_boundary_v{VERSION}"] = (
        "matched-period bounded MILP action list; not a full-universe certificate"
    )
    return actions[
        [
            f"action_v{VERSION}",
            "loan_id",
            "loan_amnt",
            *FAMILIES,
            f"mean_return_v{VERSION}",
            f"observed_v47_proxy_v{VERSION}",
            f"claim_boundary_v{VERSION}",
        ]
    ].sort_values([f"action_v{VERSION}", f"mean_return_v{VERSION}"], ascending=[True, False])


def _protocol_steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                f"step_id_v{VERSION}": "bounded_matched_period_milp",
                f"executed_v{VERSION}": True,
                f"evidence_artifact_v{VERSION}": "paper4_v320_matched_period_milp_summary.csv",
                f"required_for_claim_v{VERSION}": "matched-period bounded candidate",
                f"claim_boundary_v{VERSION}": "bounded to v316 plus top omitted 2019 pool",
            },
            {
                f"step_id_v{VERSION}": "post_v320_one_swap_reprice",
                f"executed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "paper4_v321_post_v320_matched_period_reprice.csv",
                f"required_for_claim_v{VERSION}": "local optimality after matched-period repair",
                f"claim_boundary_v{VERSION}": "must execute before any local-optimal claim",
            },
            {
                f"step_id_v{VERSION}": "full_universe_branch_price_certificate",
                f"executed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"required_for_claim_v{VERSION}": "full-universe global optimality",
                f"claim_boundary_v{VERSION}": "no full-universe certificate in v320",
            },
            {
                f"step_id_v{VERSION}": "cashflow_online_gate_after_repair",
                f"executed_v{VERSION}": False,
                f"evidence_artifact_v{VERSION}": "future_v320_cashflow_online_ifrs9_gate",
                f"required_for_claim_v{VERSION}": "live/IFRS9 extension after changed book",
                f"claim_boundary_v{VERSION}": "v319 applies to v316, not the v320 repaired book",
            },
        ]
    )


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v320 bounded matched-period MILP after v316.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v320_v316_matched_period_or_branch_price_protocol.csv"
                ),
                "boundary": "Bounded to v316 plus top omitted 2019 candidate pool.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v320 finds a period-matched bounded repair improving return and CVaR vs v316.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v320_matched_period_milp_summary.csv"
                ),
                "boundary": "Requires post-v320 repricing and global gates before promotion.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v320 repaired portfolio is post-repair locally optimal.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v320_claim_blockers.csv"
                ),
                "boundary": "Post-v320 one-swap repricing has not been executed.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v320 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v320_claim_blockers.csv"
                ),
                "boundary": "The solved pool is bounded and lacks a branch-price certificate.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v320 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v320_claim_blockers.csv"
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
                    "v320 solves a bounded matched-period MILP using v316 plus top omitted "
                    "2019 candidates to address the v318 period-distribution blocker."
                ),
                "status": "bounded_matched_period_repair_found_requires_repricing",
                "next_artifact": f"paper4_v{NEXT_VERSION}_post_v320_matched_period_reprice.csv",
                "success_condition": (
                    "rerun one-swap/source/CVaR pricing after the matched-period repair before "
                    "any local-optimal or champion claim"
                ),
                "last_wave": "v320",
                "execution_result": (
                    "period_matched_candidate_improves_return_and_cvar_bounded_pool"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v320")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V320_MATCHED_PERIOD_BOUNDED_MILP_START -->"
    end = "<!-- V320_MATCHED_PERIOD_BOUNDED_MILP_END -->"
    block = f"""
{start}

## Wave v320: Matched-Period Bounded MILP

Generated: {status["generated_at_utc"]}

### Objective

v318 showed that v316 beats v295 in final-period proxy metrics but has a period
distribution mismatch. v320 tests whether a bounded repair can match the v295
94/70/7 period distribution while preserving budget, source caps and CVaR.

### Results

- Candidate pool rows: `{status["pool_rows_v320"]}`.
- Omitted 2019 candidates used: `{status["candidate_pool_limit_v320"]}`.
- MILP success:
  `{status["milp_success_v320"]}`.
- Selected rows: `{status["selected_rows_v320"]}`.
- Added rows: `{status["added_rows_v320"]}`.
- Dropped rows: `{status["dropped_rows_v320"]}`.
- Period distribution matched:
  `{status["period_distribution_match_v320"]}`.
- Objective return: `{status["objective_return_v320"]}`.
- Delta return vs v316:
  `{status["delta_return_vs_v316_v320"]}`.
- Scenario CVaR90: `{status["scenario_loss_cvar90_v320"]}`.
- Delta CVaR90 vs v316:
  `{status["delta_cvar90_vs_v316_v320"]}`.
- Post-v320 repricing required:
  `{status["post_v320_repricing_required_v320"]}`.

### Interpretation

v320 is a high-value bounded repair: it closes the period-distribution mismatch
while improving return and lowering CVaR versus v316. This still cannot promote
Paper 4 because the solution is bounded to a candidate pool, needs post-repair
repricing, and lacks a full-universe branch-price certificate.

### Claim Impact

- Allowed: bounded matched-period MILP repair candidate after v316.
- Still prohibited: post-repair local optimality, full-universe global
  optimality, live deployment, Paper Estrella replacement, final Paper 4
  promotion and working champion claims.

### Quarto Promotion Decision

Keep v320 in the living notebook. The next wave should reprice the v320 repaired
candidate against the comparable universe.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v316 = read_parquet("paper4_v316_apply_next_post_v314_swap_allocations.parquet").reset_index(
        drop=True
    )
    v295 = read_parquet("paper4_v295_broader_multi_swap_allocations.parquet").reset_index(drop=True)
    v316_summary = read_csv("paper4_v316_apply_next_post_v314_swap_summary.csv")
    source_caps = read_csv("paper4_v316_apply_next_post_v314_swap_source_summary.csv")
    v319_status = json.loads((STATUS_DIR / "paper4_v319_status.json").read_text(encoding="utf-8"))
    if any(df.empty for df in [universe, v316, v295, v316_summary, source_caps]):
        raise RuntimeError("Missing v55, v316, v295 or source inputs for v320.")
    if bool(v319_status["working_champion_claim_allowed_v319"]):
        raise RuntimeError("v320 expects v319 to keep promotion blocked.")

    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    v316["loan_id"] = v316["loan_id"].astype(str)
    v295["loan_id"] = v295["loan_id"].astype(str)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool, omitted_2019_rows = _build_pool(
        universe=universe,
        v316=v316,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    losses_pool = losses[:, pool[f"universe_idx_v{VERSION}"].to_numpy(int)]
    v316_row = v316_summary.iloc[0]
    exposure_min = 842292.375
    exposure_max = 850000.0
    v316_return = float(v316_row[f"objective_return_v{BASE_VERSION}"])
    v316_cvar = float(v316_row[f"scenario_loss_cvar90_v{BASE_VERSION}"])
    v295_return = float(
        read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv").iloc[0][
            "objective_return_v295"
        ]
    )
    v295_cvar = float(
        read_csv("paper4_v295_broader_multi_swap_or_global_gap_probe.csv").iloc[0][
            "scenario_loss_cvar90_v295"
        ]
    )
    target_period_counts = _period_distribution(v295)
    constraints, bounds, integrality, variable_count, constraint_count = _constraints(
        pool=pool,
        losses_pool=losses_pool,
        source_caps=source_caps,
        target_period_counts=target_period_counts,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=v316_cvar,
    )
    result, selected, metrics = _solve_matched_period(
        pool=pool,
        losses_pool=losses_pool,
        losses_full=losses,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
    )
    if selected.empty:
        raise RuntimeError("v320 matched-period MILP did not return an incumbent solution.")

    base_ids = set(v316["loan_id"].astype(str))
    selected_ids = set(selected["loan_id"].astype(str))
    observed_ids = set(
        read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")["loan_id"].astype(str)
    )
    added_rows = int(len(selected_ids - base_ids))
    dropped_rows = int(len(base_ids - selected_ids))
    observed_rows = int(len(selected_ids & observed_ids))
    missing_rows = int(len(selected_ids) - observed_rows)
    period_match = metrics["period_distribution"] == target_period_counts
    return_delta_vs_v316 = float(metrics["objective_return"] - v316_return)
    cvar_delta_vs_v316 = float(metrics["scenario_loss_cvar90"] - v316_cvar)
    return_delta_vs_v295 = float(metrics["objective_return"] - v295_return)
    cvar_delta_vs_v295 = float(metrics["scenario_loss_cvar90"] - v295_cvar)
    source_summary = _source_summary(selected=selected, universe=universe, source_caps=source_caps)
    source_cap_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    min_source_slack = float(source_summary[f"source_slack_v{VERSION}"].min())
    actions = _action_table(pool=pool, selected=selected, observed_ids=observed_ids)
    protocol = _protocol_steps()

    allocations = selected[
        [
            "loan_id",
            "loan_amnt",
            *FAMILIES,
            f"pool_role_v{VERSION}",
            f"mean_return_v{VERSION}",
            f"incumbent_selected_v{VERSION}",
        ]
    ].copy()
    allocations[f"selected_v{VERSION}"] = True
    allocations[f"observed_v47_proxy_v{VERSION}"] = allocations["loan_id"].isin(observed_ids)
    allocations[f"portfolio_label_v{VERSION}"] = "bounded_matched_period_repair_candidate"
    allocations[f"claim_boundary_v{VERSION}"] = (
        "bounded matched-period MILP allocation; requires post-repair repricing"
    )

    milp_summary = pd.DataFrame(
        [
            {
                f"solver_success_v{VERSION}": bool(result.success),
                f"milp_status_v{VERSION}": int(result.status),
                f"milp_message_v{VERSION}": str(result.message),
                f"milp_fun_v{VERSION}": float(result.fun),
                f"milp_gap_v{VERSION}": float(getattr(result, "mip_gap", np.nan)),
                f"milp_dual_bound_v{VERSION}": float(getattr(result, "mip_dual_bound", np.nan)),
                f"milp_node_count_v{VERSION}": int(getattr(result, "mip_node_count", -1)),
                f"time_limit_seconds_v{VERSION}": MILP_TIME_LIMIT_SECONDS,
                f"candidate_pool_limit_v{VERSION}": CANDIDATE_POOL_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_rows_v{VERSION}": int(metrics["selected_rows"]),
                f"added_rows_v{VERSION}": added_rows,
                f"dropped_rows_v{VERSION}": dropped_rows,
                f"portfolio_exposure_v{VERSION}": float(metrics["portfolio_exposure"]),
                f"objective_return_v{VERSION}": float(metrics["objective_return"]),
                f"delta_return_vs_v316_v{VERSION}": return_delta_vs_v316,
                f"delta_return_vs_v295_v{VERSION}": return_delta_vs_v295,
                f"scenario_loss_mean_v{VERSION}": float(metrics["scenario_loss_mean"]),
                f"scenario_loss_cvar90_v{VERSION}": float(metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_vs_v316_v{VERSION}": cvar_delta_vs_v316,
                f"delta_cvar90_vs_v295_v{VERSION}": cvar_delta_vs_v295,
                f"target_period_distribution_v{VERSION}": json.dumps(
                    target_period_counts, sort_keys=True
                ),
                f"solution_period_distribution_v{VERSION}": json.dumps(
                    metrics["period_distribution"], sort_keys=True
                ),
                f"period_distribution_match_v{VERSION}": period_match,
                f"observed_v47_proxy_rows_v{VERSION}": observed_rows,
                f"missing_v47_proxy_rows_v{VERSION}": missing_rows,
                f"source_cap_violations_v{VERSION}": source_cap_violations,
                f"min_source_slack_v{VERSION}": min_source_slack,
                f"budget_feasible_v{VERSION}": exposure_min
                <= float(metrics["portfolio_exposure"])
                <= exposure_max,
                f"cvar_feasible_v{VERSION}": float(metrics["scenario_loss_cvar90"])
                <= v316_cvar + 1e-7,
                f"source_feasible_v{VERSION}": source_cap_violations == 0,
                f"post_v320_repricing_required_v{VERSION}": True,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"claim_boundary_v{VERSION}": (
                    "bounded matched-period MILP candidate only; no global or promotion claim"
                ),
            }
        ]
    )
    main_summary = pd.DataFrame(
        [
            {
                f"probe_id_v{VERSION}": "v316_matched_period_or_branch_price_protocol",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"match_target_version_v{VERSION}": MATCH_TARGET_VERSION,
                f"gate_version_v{VERSION}": GATE_VERSION,
                f"candidate_pool_limit_v{VERSION}": CANDIDATE_POOL_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"omitted_2019_rows_outside_pool_v{VERSION}": omitted_2019_rows,
                f"constraint_rows_v{VERSION}": constraint_count,
                f"variable_count_v{VERSION}": variable_count,
                f"milp_success_v{VERSION}": bool(result.success),
                f"selected_rows_v{VERSION}": int(metrics["selected_rows"]),
                f"added_rows_v{VERSION}": added_rows,
                f"dropped_rows_v{VERSION}": dropped_rows,
                f"period_distribution_match_v{VERSION}": period_match,
                f"objective_return_v{VERSION}": float(metrics["objective_return"]),
                f"delta_return_vs_v316_v{VERSION}": return_delta_vs_v316,
                f"scenario_loss_cvar90_v{VERSION}": float(metrics["scenario_loss_cvar90"]),
                f"delta_cvar90_vs_v316_v{VERSION}": cvar_delta_vs_v316,
                f"period_matched_candidate_found_v{VERSION}": bool(result.success and period_match),
                f"post_v320_repricing_required_v{VERSION}": True,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_v320_matched_period_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": (
                    "bounded matched-period repair found; post-repair repricing/global gates still missing"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": omitted_2019_rows,
                f"required_next_artifact_v{VERSION}": "future_full_universe_branch_price_bound",
                f"claim_boundary_v{VERSION}": "v320 prices only a top-2019 bounded pool",
            },
            {
                f"blocker_id_v{VERSION}": "post_v320_reprice_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": (
                    f"paper4_v{NEXT_VERSION}_post_v320_matched_period_reprice.csv"
                ),
                f"claim_boundary_v{VERSION}": "matched-period repair needs one-swap repricing",
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_certificate_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no global omitted-column certificate",
            },
            {
                f"blocker_id_v{VERSION}": "cashflow_online_gate_after_repair_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_v320_cashflow_online_ifrs9_gate",
                f"claim_boundary_v{VERSION}": "v319 applies to v316, not the v320 repaired book",
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
                "claim_id": "v320_bounded_matched_period_milp_executed",
                "allowed": True,
                "artifact": "paper4_v320_matched_period_milp_summary.csv",
                "boundary": "bounded candidate pool MILP",
            },
            {
                "claim_id": "v320_period_matched_candidate_found",
                "allowed": bool(result.success and period_match),
                "artifact": "paper4_v320_matched_period_allocations.parquet",
                "boundary": "matches v295 period counts within bounded pool",
            },
            {
                "claim_id": "v320_improves_return_and_lowers_cvar_vs_v316",
                "allowed": return_delta_vs_v316 > 0 and cvar_delta_vs_v316 < 0,
                "artifact": "paper4_v320_matched_period_milp_summary.csv",
                "boundary": "static scenario proxy only; repricing still required",
            },
            {
                "claim_id": "v320_post_repair_local_optimality",
                "allowed": False,
                "artifact": "paper4_v320_claim_blockers.csv",
                "boundary": "post-v320 one-swap repricing missing",
            },
            {
                "claim_id": "v320_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v320_claim_blockers.csv",
                "boundary": "global branch-price certificate missing",
            },
            {
                "claim_id": "v320_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v320_claim_blockers.csv",
                "boundary": "working champion and final promotion remain blocked",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v320_v316_matched_period_or_branch_price_protocol.csv", main_summary
    )
    write_csv(TABLE_DIR / "paper4_v320_matched_period_milp_summary.csv", milp_summary)
    allocations.to_parquet(
        TABLE_DIR / "paper4_v320_matched_period_allocations.parquet", index=False
    )
    write_csv(TABLE_DIR / "paper4_v320_matched_period_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v320_matched_period_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v320_protocol_steps.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v320_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v320_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    summary_row = main_summary.iloc[0]
    milp_row = milp_summary.iloc[0]
    status = {
        "phase": "v320_v316_matched_period_or_branch_price_protocol",
        "schema_version": "2026-05-16.320",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v320": BASE_VERSION,
        "match_target_version_v320": MATCH_TARGET_VERSION,
        "gate_version_v320": GATE_VERSION,
        "candidate_pool_limit_v320": CANDIDATE_POOL_LIMIT,
        "pool_rows_v320": int(summary_row[f"pool_rows_v{VERSION}"]),
        "omitted_2019_rows_outside_pool_v320": int(
            summary_row[f"omitted_2019_rows_outside_pool_v{VERSION}"]
        ),
        "constraint_rows_v320": int(summary_row[f"constraint_rows_v{VERSION}"]),
        "variable_count_v320": int(summary_row[f"variable_count_v{VERSION}"]),
        "milp_success_v320": bool(summary_row[f"milp_success_v{VERSION}"]),
        "milp_gap_v320": float(milp_row[f"milp_gap_v{VERSION}"]),
        "milp_node_count_v320": int(milp_row[f"milp_node_count_v{VERSION}"]),
        "selected_rows_v320": int(summary_row[f"selected_rows_v{VERSION}"]),
        "added_rows_v320": int(summary_row[f"added_rows_v{VERSION}"]),
        "dropped_rows_v320": int(summary_row[f"dropped_rows_v{VERSION}"]),
        "portfolio_exposure_v320": float(milp_row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v320": float(summary_row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v316_v320": float(summary_row[f"delta_return_vs_v316_v{VERSION}"]),
        "delta_return_vs_v295_v320": float(milp_row[f"delta_return_vs_v295_v{VERSION}"]),
        "scenario_loss_cvar90_v320": float(summary_row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v316_v320": float(summary_row[f"delta_cvar90_vs_v316_v{VERSION}"]),
        "delta_cvar90_vs_v295_v320": float(milp_row[f"delta_cvar90_vs_v295_v{VERSION}"]),
        "period_distribution_match_v320": bool(
            summary_row[f"period_distribution_match_v{VERSION}"]
        ),
        "observed_v47_proxy_rows_v320": int(milp_row[f"observed_v47_proxy_rows_v{VERSION}"]),
        "missing_v47_proxy_rows_v320": int(milp_row[f"missing_v47_proxy_rows_v{VERSION}"]),
        "source_cap_violations_v320": int(milp_row[f"source_cap_violations_v{VERSION}"]),
        "min_source_slack_v320": float(milp_row[f"min_source_slack_v{VERSION}"]),
        "budget_feasible_v320": bool(milp_row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v320": bool(milp_row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v320": bool(milp_row[f"source_feasible_v{VERSION}"]),
        "period_matched_candidate_found_v320": bool(
            summary_row[f"period_matched_candidate_found_v{VERSION}"]
        ),
        "post_v320_repricing_required_v320": True,
        "full_universe_integer_optimality_claim_allowed_v320": False,
        "working_champion_claim_allowed_v320": False,
        "paper1_promotion_allowed_v320": False,
        "paper4_working_champion_changed_v320": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "protocol_step_rows_v320": int(len(protocol)),
        "claim_blocker_rows_v320": int(len(blockers)),
        "claim_matrix_rows_v320": int(len(claim_matrix)),
        "next_artifact_v320": summary_row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v320 finds a bounded matched-period repair after v316; post-repair repricing, "
            "global proof, live deployment, working champion, and final promotion remain blocked"
        ),
    }
    write_json(STATUS_DIR / "paper4_v320_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v320": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
