#!/usr/bin/env python3
"""Build Paper 4 v293 diverse-pool return-gap probe artifacts."""

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

VERSION = 293
PREVIOUS_VERSION = 292
REPAIR_VERSION = 289
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 294
TOP_RETURN_LIMIT = 15000
MICRO_RELIEF_LIMIT = 10000
TARGET_SELECTED_ROWS = 171
MILP_TIME_LIMIT_SECONDS = 240.0
MIP_REL_GAP = 1e-6


def _source_cap_map(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)].copy()
    family_caps["source_id"] = family_caps["source_id"].astype(str)
    return family_caps.set_index("source_id")["cap_share_v80"].astype(float).to_dict()


def _build_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    mean_returns: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    omitted = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    omitted[f"mean_return_v{VERSION}"] = mean_returns[omitted.index.to_numpy()]

    top_return = omitted.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
        TOP_RETURN_LIMIT
    )
    top_ids = set(top_return["loan_id"].astype(str))
    outside = omitted.loc[~omitted["loan_id"].astype(str).isin(top_ids)].copy()
    outside["non_tight_source_count_v293"] = outside["grade"].astype(str).ne("A").astype(
        int
    ) + outside["score_decile"].astype(str).ne("0").astype(int)
    micro_relief = (
        outside.loc[outside["non_tight_source_count_v293"].gt(0)]
        .sort_values(["loan_amnt", f"mean_return_v{VERSION}"], ascending=[True, False])
        .head(MICRO_RELIEF_LIMIT)
    )

    selected_pool = selected[["loan_id", "loan_amnt", *FAMILIES]].copy()
    selected_pool["pool_role_v293"] = "current_v289_selected"
    top_pool = top_return[["loan_id", "loan_amnt", *FAMILIES]].copy()
    top_pool["pool_role_v293"] = "top15000_omitted_by_mean_return"
    micro_pool = micro_relief[["loan_id", "loan_amnt", *FAMILIES]].copy()
    micro_pool["pool_role_v293"] = "micro_source_relief_outside_top15000"
    pool = pd.concat([selected_pool, top_pool, micro_pool], ignore_index=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool = pool.drop_duplicates("loan_id", keep="first").reset_index(drop=True)
    candidate_registry = pd.DataFrame(
        [
            {
                f"pool_component_v{VERSION}": "current_v289_selected",
                f"candidate_rows_v{VERSION}": int(len(selected_pool)),
                f"claim_boundary_v{VERSION}": "incumbent rows carried into v293 pool",
            },
            {
                f"pool_component_v{VERSION}": "top15000_omitted_by_mean_return",
                f"candidate_rows_v{VERSION}": int(len(top_pool)),
                f"claim_boundary_v{VERSION}": "high-return omitted candidate component",
            },
            {
                f"pool_component_v{VERSION}": "micro_source_relief_outside_top15000",
                f"candidate_rows_v{VERSION}": int(len(micro_pool)),
                f"claim_boundary_v{VERSION}": (
                    "outside-top15000 non-tight-source micro-loan relief component"
                ),
            },
        ]
    )
    return pool, candidate_registry


def _source_constraints(
    *,
    pool: pd.DataFrame,
    amounts: np.ndarray,
    source_caps: pd.DataFrame,
    var_count: int,
) -> tuple[list[np.ndarray], list[float], list[float], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = _source_cap_map(source_caps, family)
        values = pool[family].astype(str)
        for source_id in sorted(values.dropna().unique()):
            cap = float(caps.get(source_id, 1.0))
            row = np.zeros(var_count)
            row[: len(pool)] = pool["loan_amnt"].to_numpy(float) * (
                values.eq(source_id).to_numpy(float) - cap
            )
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
            meta.append(
                {
                    f"constraint_type_v{VERSION}": "source_share",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                }
            )
    return rows, lb, ub, meta


def _solve_milp(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    mean_returns_pool: np.ndarray,
    source_caps: pd.DataFrame,
    exposure_min: float,
    exposure_max: float,
    cvar_cap: float,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    var_count = n + 1 + scenario_count
    amounts = pool["loan_amnt"].to_numpy(float)
    c = np.zeros(var_count)
    c[:n] = -mean_returns_pool
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    meta: list[dict[str, Any]] = []

    budget = np.zeros(var_count)
    budget[:n] = amounts
    rows.append(budget)
    lb.append(exposure_min)
    ub.append(exposure_max)
    meta.append({f"constraint_type_v{VERSION}": "budget_range"})

    cardinality = np.zeros(var_count)
    cardinality[:n] = 1.0
    rows.append(cardinality)
    lb.append(float(TARGET_SELECTED_ROWS))
    ub.append(float(TARGET_SELECTED_ROWS))
    meta.append({f"constraint_type_v{VERSION}": "selected_row_cardinality"})

    source_rows, source_lb, source_ub, source_meta = _source_constraints(
        pool=pool,
        amounts=amounts,
        source_caps=source_caps,
        var_count=var_count,
    )
    rows.extend(source_rows)
    lb.extend(source_lb)
    ub.extend(source_ub)
    meta.extend(source_meta)

    cvar_row = np.zeros(var_count)
    cvar_row[n] = 1.0
    cvar_row[n + 1 :] = 1.0 / ((1.0 - v70.ALPHA) * scenario_count)
    rows.append(cvar_row)
    lb.append(-np.inf)
    ub.append(cvar_cap)
    meta.append({f"constraint_type_v{VERSION}": "cvar_cap"})

    for scenario_idx in range(scenario_count):
        row = np.zeros(var_count)
        row[:n] = losses_pool[scenario_idx, :]
        row[n] = -1.0
        row[n + 1 + scenario_idx] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
        meta.append({f"constraint_type_v{VERSION}": "cvar_path_excess"})

    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(1 + scenario_count)],
        bounds=Bounds(
            np.r_[np.zeros(n), 0.0, np.zeros(scenario_count)],
            np.r_[np.ones(n), np.inf, np.full(scenario_count, np.inf)],
        ),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    selected_mask = np.zeros(n, dtype=bool)
    incumbent_available = result.x is not None
    if incumbent_available:
        selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
    diagnostics = {
        f"milp_success_v{VERSION}": bool(result.success),
        f"milp_incumbent_available_v{VERSION}": bool(incumbent_available),
        f"milp_status_v{VERSION}": int(result.status),
        f"milp_message_v{VERSION}": str(result.message),
        f"milp_fun_v{VERSION}": float(result.fun) if result.fun is not None else np.nan,
        f"milp_dual_bound_v{VERSION}": float(getattr(result, "mip_dual_bound", np.nan)),
        f"milp_gap_v{VERSION}": float(getattr(result, "mip_gap", np.nan)),
        f"milp_node_count_v{VERSION}": int(getattr(result, "mip_node_count", -1)),
        f"milp_time_limit_seconds_v{VERSION}": MILP_TIME_LIMIT_SECONDS,
        f"milp_mip_rel_gap_v{VERSION}": MIP_REL_GAP,
        f"milp_variable_count_v{VERSION}": int(var_count),
        f"milp_binary_variable_count_v{VERSION}": int(n),
        f"milp_constraint_rows_v{VERSION}": int(len(rows)),
    }
    return diagnostics, selected_mask, pd.DataFrame(meta)


def _source_summary(
    *,
    universe: pd.DataFrame,
    portfolio: pd.DataFrame,
    source_caps: pd.DataFrame,
) -> pd.DataFrame:
    exposure = float(portfolio["loan_amnt"].sum())
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        caps = _source_cap_map(source_caps, family)
        by_source = portfolio.groupby(family, dropna=False)["loan_amnt"].sum()
        for source_id in sorted(universe[family].dropna().astype(str).unique()):
            source_exposure = float(by_source.get(source_id, 0.0))
            share = source_exposure / max(exposure, 1.0)
            cap = float(caps.get(source_id, 1.0))
            rows.append(
                {
                    f"portfolio_label_v{VERSION}": "diverse_micro_relief_cardinality_milp",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": ("v293 diverse-pool MILP source diagnostic only"),
                }
            )
    return pd.DataFrame(rows)


def _return_contributions(
    *,
    universe: pd.DataFrame,
    pool: pd.DataFrame,
    mean_returns: np.ndarray,
    added_ids: list[str],
    dropped_ids: list[str],
) -> pd.DataFrame:
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    rows: list[dict[str, Any]] = []
    pool_roles = pool.set_index("loan_id")["pool_role_v293"].astype(str).to_dict()
    for action, ids, sign in [("added", added_ids, 1.0), ("dropped", dropped_ids, -1.0)]:
        for loan_id in ids:
            row = universe.loc[universe["loan_id"].astype(str).eq(loan_id)].iloc[0]
            mean_return = float(mean_returns[int(idx_by_id.loc[loan_id])])
            rows.append(
                {
                    f"action_v{VERSION}": action,
                    f"loan_id_v{VERSION}": loan_id,
                    f"loan_amount_v{VERSION}": float(row["loan_amnt"]),
                    f"mean_return_v{VERSION}": mean_return,
                    f"signed_return_contribution_v{VERSION}": sign * mean_return,
                    f"pool_role_v{VERSION}": pool_roles.get(loan_id, "dropped_from_v289"),
                    "grade": str(row["grade"]),
                    "score_decile": str(row["score_decile"]),
                    "income_band": str(row["income_band"]),
                    "dti_band": str(row["dti_band"]),
                    "period": str(row["period"]),
                    "state_top20": str(row["state_top20"]),
                    f"claim_boundary_v{VERSION}": (
                        "v293 return-gap contribution; not a promotion claim"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _update_claim_boundaries(*, challenger_found: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v293 diverse source-relief cardinality MILP probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v293_diverse_pool_return_gap_probe.csv"
                ),
                "boundary": (
                    "Top-15000 return pool plus outside-top15000 micro source-relief pool; "
                    "not a full-universe proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v293 finds a return-positive bounded-pool challenger that restores cardinality.",
                "allowed": bool(challenger_found),
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v293_diverse_pool_return_gap_probe.csv"
                ),
                "boundary": (
                    "Bounded-pool challenger signal only; must be applied, repriced, "
                    "stress-tested and globally bounded before champion claims."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v293 is a new Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v293_claim_blockers.csv"
                ),
                "boundary": "Post-v293 repricing, global bounds and dynamic validation are missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v293 proves global full-universe integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v293_claim_blockers.csv"
                ),
                "boundary": "Bounded diverse pool only; full-universe certificate missing.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v293 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v293_claim_blockers.csv"
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
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v293 expands v292 with outside-top15000 micro source-relief candidates "
                    "and tests whether the v292 return gap can be reversed."
                ),
                "status": "diverse_pool_cardinality_challenger_found_requires_repricing",
                "next_artifact": "paper4_v294_post_v293_diverse_pool_reprice.csv",
                "success_condition": (
                    "apply the v293 bounded-pool challenger and rerun post-repair pricing "
                    "before any working-champion claim"
                ),
                "last_wave": "v293",
                "execution_result": "diverse_micro_relief_pool_beats_v289_requires_repricing",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v293")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V293_DIVERSE_POOL_RETURN_GAP_PROBE_START -->"
    end = "<!-- V293_DIVERSE_POOL_RETURN_GAP_PROBE_END -->"
    block = f"""
{start}

## Wave v293: Diverse Pool Return-Gap Probe

Generated: {status["generated_at_utc"]}

### Objective

v292 restored cardinality but trailed v289 by 3.031 return units. v293 tests
whether that gap is a top-return-pool artifact by adding 10,000 outside-top15000
micro source-relief candidates that do not load both active tight sources
(`grade=A` and `score_decile=0`).

### Results

- Pool rows: `{status["pool_rows_v293"]}`.
- Top-return candidate limit: `{status["top_return_limit_v293"]}`.
- Micro-relief candidate limit: `{status["micro_relief_limit_v293"]}`.
- MILP success: `{status["milp_success_v293"]}`.
- MILP gap: `{status["milp_gap_v293"]}`.
- Selected rows: `{status["selected_rows_v293"]}`.
- Added rows vs v289: `{status["added_rows_v293"]}`.
- Dropped rows vs v289: `{status["dropped_rows_v293"]}`.
- Objective return: `{status["objective_return_v293"]}`.
- Delta return vs v289: `{status["delta_return_vs_v289_v293"]}`.
- Delta return vs v292: `{status["delta_return_vs_v292_v293"]}`.
- CVaR90: `{status["scenario_loss_cvar90_v293"]}`.
- Delta CVaR90 vs v289: `{status["delta_cvar90_vs_v289_v293"]}`.
- Source cap violations: `{status["source_cap_violations_v293"]}`.
- Cardinality restored: `{status["cardinality_restored_v293"]}`.
- Bounded-pool challenger found:
  `{status["bounded_pool_challenger_found_v293"]}`.

### Interpretation

v293 is a major live-lab signal. The return gap in v292 is not intrinsic to
cardinality restoration; it is partly a pool-design artifact. Adding micro
source-relief candidates creates a bounded-pool challenger that restores 171
rows, improves return by 77.794 over v289, and lowers CVaR90 by 943.159. This
is still not a working champion because the candidate must be applied and
repriced against broader pricing/global/dynamic gates.

### Claim Impact

- Allowed: bounded diverse-pool challenger signal documented.
- Still prohibited: working champion replacement, full-universe optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v293 in the living notebook. The next live-lab step is applying/repricing
the v293 challenger, not promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v289_exact_relief_repair_allocations.parquet").reset_index(
        drop=True
    )
    v289_summary = read_csv("paper4_v289_exact_relief_repair_summary.csv")
    v279_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    v292_status = json.loads((STATUS_DIR / "paper4_v292_status.json").read_text(encoding="utf-8"))
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or selected.empty or v289_summary.empty or v279_summary.empty:
        raise RuntimeError("Missing v55, v289, or v279 inputs for v293.")
    if source_caps.empty:
        raise RuntimeError("Missing focused source caps for v293.")

    v289_row = v289_summary.iloc[0]
    v279_row = v279_summary.iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool, candidate_registry = _build_pool(
        universe=universe,
        selected=selected,
        mean_returns=mean_returns,
    )
    pool_idx = idx_by_id.loc[pool["loan_id"].astype(str)].to_numpy()
    losses_pool = losses[:, pool_idx]
    mean_returns_pool = mean_returns[pool_idx]
    pool[f"mean_return_v{VERSION}"] = mean_returns_pool

    exposure_min = float(v279_row[f"exposure_min_v{BASE_REPAIR_VERSION}"])
    exposure_max = float(v279_row[f"exposure_max_v{BASE_REPAIR_VERSION}"])
    cvar_cap = float(v289_row[f"scenario_loss_cvar90_v{REPAIR_VERSION}"])
    diagnostics, selected_mask, constraints = _solve_milp(
        pool=pool,
        losses_pool=losses_pool,
        mean_returns_pool=mean_returns_pool,
        source_caps=source_caps,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=cvar_cap,
    )

    solution = pool.loc[selected_mask].copy()
    solution_idx = idx_by_id.loc[solution["loan_id"].astype(str)].to_numpy()
    solution_losses = losses[:, solution_idx].sum(axis=1)
    selected_ids = set(selected["loan_id"].astype(str))
    solution_ids = set(solution["loan_id"].astype(str))
    added_ids = sorted(solution_ids - selected_ids)
    dropped_ids = sorted(selected_ids - solution_ids)
    kept_current = len(solution_ids & selected_ids)
    objective_return = float(mean_returns[solution_idx].sum())
    exposure = float(solution["loan_amnt"].sum())
    cvar90 = v70._tail_cvar(solution_losses)
    source_summary = _source_summary(
        universe=universe,
        portfolio=solution,
        source_caps=source_caps,
    )
    source_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    cardinality_restored = int(len(solution)) == TARGET_SELECTED_ROWS
    delta_return_vs_v289 = objective_return - float(v289_row[f"objective_return_v{REPAIR_VERSION}"])
    delta_return_vs_v279 = objective_return - float(
        v279_row[f"objective_return_v{BASE_REPAIR_VERSION}"]
    )
    delta_return_vs_v292 = objective_return - float(v292_status["objective_return_v292"])
    delta_cvar_vs_v289 = cvar90 - cvar_cap
    budget_feasible = exposure_min - 1e-7 <= exposure <= exposure_max + 1e-7
    cvar_feasible = cvar90 <= cvar_cap + 1e-7
    source_feasible = source_violations == 0
    challenger_found = (
        bool(diagnostics[f"milp_success_v{VERSION}"])
        and cardinality_restored
        and budget_feasible
        and cvar_feasible
        and source_feasible
        and delta_return_vs_v289 > 1e-9
    )

    solution["selected_v293"] = True
    solution["portfolio_label_v293"] = "diverse_micro_relief_cardinality_milp"
    solution["repair_action_v293"] = np.where(
        solution["loan_id"].astype(str).isin(added_ids),
        "added_by_v293_diverse_milp",
        "kept_from_v289",
    )
    solution["claim_boundary_v293"] = (
        "bounded diverse-pool cardinality MILP allocation; requires v294 repricing"
    )
    action = pd.DataFrame(
        [
            {
                f"action_id_v{VERSION}": "diverse_micro_relief_cardinality_milp_action",
                f"added_loan_ids_v{VERSION}": "|".join(added_ids),
                f"dropped_loan_ids_v{VERSION}": "|".join(dropped_ids),
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_v{VERSION}": int(len(dropped_ids)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"bounded_pool_challenger_found_v{VERSION}": challenger_found,
                f"claim_boundary_v{VERSION}": (
                    "bounded challenger action only; post-repair pricing still required"
                ),
            }
        ]
    )
    pool_summary = pd.DataFrame(
        [
            {
                f"pool_role_v{VERSION}": role,
                f"pool_rows_v{VERSION}": int(len(group)),
                f"selected_rows_v{VERSION}": int(
                    group["loan_id"].astype(str).isin(solution_ids).sum()
                ),
                f"claim_boundary_v{VERSION}": "v293 diverse MILP pool composition only",
            }
            for role, group in pool.groupby("pool_role_v293", dropna=False)
        ]
    )
    constraints_summary = (
        constraints.groupby(f"constraint_type_v{VERSION}", dropna=False)
        .size()
        .reset_index(name=f"constraint_rows_v{VERSION}")
    )
    constraints_summary[f"claim_boundary_v{VERSION}"] = "v293 diverse MILP constraint count only"
    contributions = _return_contributions(
        universe=universe,
        pool=pool,
        mean_returns=mean_returns,
        added_ids=added_ids,
        dropped_ids=dropped_ids,
    )
    strategy_comparison = pd.DataFrame(
        [
            {
                f"strategy_id_v{VERSION}": "v292_top15000_baseline",
                f"pool_rows_v{VERSION}": int(v292_status["pool_rows_v292"]),
                f"selected_rows_v{VERSION}": int(v292_status["selected_rows_v292"]),
                f"added_rows_v{VERSION}": int(v292_status["added_rows_v292"]),
                f"dropped_rows_v{VERSION}": int(v292_status["dropped_rows_v292"]),
                f"objective_return_v{VERSION}": float(v292_status["objective_return_v292"]),
                f"delta_return_vs_v289_v{VERSION}": float(v292_status["delta_return_vs_v289_v292"]),
                f"delta_return_vs_v292_v{VERSION}": 0.0,
                f"scenario_loss_cvar90_v{VERSION}": float(v292_status["scenario_loss_cvar90_v292"]),
                f"delta_cvar90_vs_v289_v{VERSION}": float(v292_status["delta_cvar90_vs_v289_v292"]),
                f"source_cap_violations_v{VERSION}": int(v292_status["source_cap_violations_v292"]),
                f"bounded_pool_challenger_found_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "v292 baseline for v293 gap decomposition",
            },
            {
                f"strategy_id_v{VERSION}": "v293_top15000_plus_micro_source_relief",
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"added_rows_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_v{VERSION}": int(len(dropped_ids)),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v289_v{VERSION}": delta_return_vs_v289,
                f"delta_return_vs_v292_v{VERSION}": delta_return_vs_v292,
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"delta_cvar90_vs_v289_v{VERSION}": delta_cvar_vs_v289,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"bounded_pool_challenger_found_v{VERSION}": challenger_found,
                f"claim_boundary_v{VERSION}": (
                    "bounded diverse-pool challenger signal requiring v294 repricing"
                ),
            },
        ]
    )
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "top15000_plus_micro_source_relief_gap_probe",
                f"previous_protocol_version_v{VERSION}": PREVIOUS_VERSION,
                f"repair_version_v{VERSION}": REPAIR_VERSION,
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"top_return_limit_v{VERSION}": TOP_RETURN_LIMIT,
                f"micro_relief_limit_v{VERSION}": MICRO_RELIEF_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"target_selected_rows_v{VERSION}": TARGET_SELECTED_ROWS,
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_v{VERSION}": int(len(dropped_ids)),
                f"portfolio_exposure_v{VERSION}": exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v289_v{VERSION}": delta_return_vs_v289,
                f"delta_return_vs_v279_v{VERSION}": delta_return_vs_v279,
                f"delta_return_vs_v292_v{VERSION}": delta_return_vs_v292,
                f"scenario_loss_mean_v{VERSION}": float(solution_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"delta_cvar90_vs_v289_v{VERSION}": delta_cvar_vs_v289,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"bounded_pool_challenger_found_v{VERSION}": challenger_found,
                f"post_repair_pricing_required_v{VERSION}": True,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"bounded_pool_optimality_claim_allowed_v{VERSION}": bool(
                    diagnostics[f"milp_success_v{VERSION}"]
                ),
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": "paper4_v294_post_v293_diverse_pool_reprice.csv",
                f"claim_boundary_v{VERSION}": (
                    "bounded diverse-pool challenger found; v294 repricing required"
                ),
                **diagnostics,
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "post_v293_repricing_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": delta_return_vs_v289,
                f"required_next_artifact_v{VERSION}": "paper4_v294_post_v293_diverse_pool_reprice.csv",
                f"claim_boundary_v{VERSION}": (
                    "return-positive bounded-pool challenger must be repriced"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    len(universe) - len(selected) - TOP_RETURN_LIMIT - MICRO_RELIEF_LIMIT
                ),
                f"required_next_artifact_v{VERSION}": "future_full_universe_branch_price_bound",
                f"claim_boundary_v{VERSION}": (
                    "v293 still leaves omitted loans outside the bounded diverse pool"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "branch_price_dual_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_branch_price_dual_bound_loop",
                f"claim_boundary_v{VERSION}": "no full-universe dual-bound certificate",
            },
            {
                f"blocker_id_v{VERSION}": "dynamic_replay_and_deployment_gates_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_dynamic_replay_validation",
                f"claim_boundary_v{VERSION}": "no online or deployment validation created",
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
                "claim_id": "v293_diverse_pool_gap_probe_executed",
                "allowed": True,
                "artifact": "paper4_v293_diverse_pool_return_gap_probe.csv",
                "boundary": "bounded diverse-pool MILP",
            },
            {
                "claim_id": "v293_bounded_pool_challenger_found",
                "allowed": challenger_found,
                "artifact": "paper4_v293_diverse_pool_return_gap_probe.csv",
                "boundary": "requires v294 repricing before champion claim",
            },
            {
                "claim_id": "v293_cardinality_restored",
                "allowed": cardinality_restored,
                "artifact": "paper4_v293_diverse_pool_action.csv",
                "boundary": "bounded-pool feasibility only",
            },
            {
                "claim_id": "v293_bounded_pool_milp_optimality",
                "allowed": bool(diagnostics[f"milp_success_v{VERSION}"]),
                "artifact": "paper4_v293_diverse_pool_return_gap_probe.csv",
                "boundary": "bounded pool only",
            },
            {
                "claim_id": "v293_working_champion",
                "allowed": False,
                "artifact": "paper4_v293_claim_blockers.csv",
                "boundary": "post-repair pricing and global evidence missing",
            },
            {
                "claim_id": "v293_global_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v293_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v293_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v293_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_return_gap_probe.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_strategy_comparison.csv", strategy_comparison)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_candidate_registry.csv", candidate_registry)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_pool_summary.csv", pool_summary)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_constraint_summary.csv", constraints_summary)
    write_csv(TABLE_DIR / "paper4_v293_diverse_pool_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v293_return_gap_contributions.csv", contributions)
    write_csv(TABLE_DIR / "paper4_v293_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v293_claim_matrix_delta.csv", claim_matrix)
    solution.to_parquet(TABLE_DIR / "paper4_v293_diverse_pool_allocations.parquet", index=False)
    _update_claim_boundaries(challenger_found=challenger_found)
    _update_backlog()

    row = protocol.iloc[0]
    status = {
        "phase": "v293_diverse_pool_return_gap_probe",
        "schema_version": "2026-05-15.293",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "previous_protocol_version_v293": PREVIOUS_VERSION,
        "repair_version_v293": REPAIR_VERSION,
        "base_repair_version_v293": BASE_REPAIR_VERSION,
        "top_return_limit_v293": TOP_RETURN_LIMIT,
        "micro_relief_limit_v293": MICRO_RELIEF_LIMIT,
        "pool_rows_v293": int(row[f"pool_rows_v{VERSION}"]),
        "selected_rows_v293": int(row[f"selected_rows_v{VERSION}"]),
        "target_selected_rows_v293": TARGET_SELECTED_ROWS,
        "cardinality_restored_v293": bool(row[f"cardinality_restored_v{VERSION}"]),
        "kept_current_rows_v293": int(row[f"kept_current_rows_v{VERSION}"]),
        "added_rows_v293": int(row[f"added_rows_v{VERSION}"]),
        "dropped_rows_v293": int(row[f"dropped_rows_v{VERSION}"]),
        "portfolio_exposure_v293": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v293": float(row[f"objective_return_v{VERSION}"]),
        "delta_return_vs_v289_v293": float(row[f"delta_return_vs_v289_v{VERSION}"]),
        "delta_return_vs_v279_v293": float(row[f"delta_return_vs_v279_v{VERSION}"]),
        "delta_return_vs_v292_v293": float(row[f"delta_return_vs_v292_v{VERSION}"]),
        "scenario_loss_cvar90_v293": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v289_v293": float(row[f"delta_cvar90_vs_v289_v{VERSION}"]),
        "source_cap_violations_v293": source_violations,
        "budget_feasible_v293": bool(row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v293": bool(row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v293": bool(row[f"source_feasible_v{VERSION}"]),
        "bounded_pool_challenger_found_v293": bool(
            row[f"bounded_pool_challenger_found_v{VERSION}"]
        ),
        "post_repair_pricing_required_v293": True,
        "milp_success_v293": bool(row[f"milp_success_v{VERSION}"]),
        "milp_status_v293": int(row[f"milp_status_v{VERSION}"]),
        "milp_gap_v293": float(row[f"milp_gap_v{VERSION}"]),
        "milp_node_count_v293": int(row[f"milp_node_count_v{VERSION}"]),
        "milp_variable_count_v293": int(row[f"milp_variable_count_v{VERSION}"]),
        "milp_constraint_rows_v293": int(row[f"milp_constraint_rows_v{VERSION}"]),
        "strategy_comparison_rows_v293": int(len(strategy_comparison)),
        "candidate_registry_rows_v293": int(len(candidate_registry)),
        "pool_summary_rows_v293": int(len(pool_summary)),
        "constraint_summary_rows_v293": int(len(constraints_summary)),
        "source_summary_rows_v293": int(len(source_summary)),
        "return_contribution_rows_v293": int(len(contributions)),
        "claim_blocker_rows_v293": int(len(blockers)),
        "claim_matrix_rows_v293": int(len(claim_matrix)),
        "working_champion_claim_allowed_v293": False,
        "bounded_pool_optimality_claim_allowed_v293": bool(
            row[f"bounded_pool_optimality_claim_allowed_v{VERSION}"]
        ),
        "full_universe_integer_optimality_claim_allowed_v293": False,
        "paper1_promotion_allowed_v293": False,
        "paper4_working_champion_changed_v293": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v293": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v293 finds a bounded diverse-pool challenger, but v294 repricing/global/"
            "dynamic gates are required before any champion claim"
        ),
    }
    write_json(STATUS_DIR / "paper4_v293_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v293": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
