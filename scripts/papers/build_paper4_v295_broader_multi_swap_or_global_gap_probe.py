#!/usr/bin/env python3
"""Build Paper 4 v295 broader multi-swap/global-gap probe artifacts."""

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

VERSION = 295
INCUMBENT_VERSION = 293
REPRICE_VERSION = 294
BASE_REPAIR_VERSION = 279
NEXT_VERSION = 296
TOP_RETURN_LIMIT = 20000
MICRO_RELIEF_LIMIT = 15000
TARGET_SELECTED_ROWS = 171
MILP_TIME_LIMIT_SECONDS = 300.0
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    omitted = universe.loc[~universe["loan_id"].astype(str).isin(selected_ids)].copy()
    omitted[f"mean_return_v{VERSION}"] = mean_returns[omitted.index.to_numpy()]

    top_return = omitted.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
        TOP_RETURN_LIMIT
    )
    top_ids = set(top_return["loan_id"].astype(str))
    outside = omitted.loc[~omitted["loan_id"].astype(str).isin(top_ids)].copy()
    outside[f"non_tight_source_count_v{VERSION}"] = outside["grade"].astype(str).ne("A").astype(
        int
    ) + outside["score_decile"].astype(str).ne("0").astype(int)
    micro_relief = (
        outside.loc[outside[f"non_tight_source_count_v{VERSION}"].gt(0)]
        .sort_values(["loan_amnt", f"mean_return_v{VERSION}"], ascending=[True, False])
        .head(MICRO_RELIEF_LIMIT)
    )

    selected_pool = selected[["loan_id", "loan_amnt", *FAMILIES]].copy()
    selected_pool[f"pool_role_v{VERSION}"] = "current_v293_selected"
    top_pool = top_return[["loan_id", "loan_amnt", *FAMILIES]].copy()
    top_pool[f"pool_role_v{VERSION}"] = "top20000_post_v293_omitted_by_mean_return"
    micro_pool = micro_relief[["loan_id", "loan_amnt", *FAMILIES]].copy()
    micro_pool[f"pool_role_v{VERSION}"] = "micro_source_relief_outside_top20000"
    pool = pd.concat([selected_pool, top_pool, micro_pool], ignore_index=True)
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool = pool.drop_duplicates("loan_id", keep="first").reset_index(drop=True)
    pool_ids = set(pool["loan_id"].astype(str))
    outside_pool = omitted.loc[~omitted["loan_id"].astype(str).isin(pool_ids)].copy()

    candidate_registry = pd.DataFrame(
        [
            {
                f"pool_component_v{VERSION}": "current_v293_selected",
                f"candidate_rows_v{VERSION}": int(len(selected_pool)),
                f"claim_boundary_v{VERSION}": "incumbent rows carried into v295 pool",
            },
            {
                f"pool_component_v{VERSION}": "top20000_post_v293_omitted_by_mean_return",
                f"candidate_rows_v{VERSION}": int(len(top_pool)),
                f"claim_boundary_v{VERSION}": "post-v293 high-return omitted candidate component",
            },
            {
                f"pool_component_v{VERSION}": "micro_source_relief_outside_top20000",
                f"candidate_rows_v{VERSION}": int(len(micro_pool)),
                f"claim_boundary_v{VERSION}": (
                    "outside-top20000 non-tight-source micro-loan relief component"
                ),
            },
            {
                f"pool_component_v{VERSION}": "outside_v295_pool_not_solved",
                f"candidate_rows_v{VERSION}": int(len(outside_pool)),
                f"claim_boundary_v{VERSION}": (
                    "full-universe rows still outside the bounded v295 MILP pool"
                ),
            },
        ]
    )
    return pool, candidate_registry, outside_pool


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
            row[: len(pool)] = amounts * (values.eq(source_id).to_numpy(float) - cap)
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
    incumbent_ids: set[str],
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
    selected_mask = pool["loan_id"].astype(str).isin(incumbent_ids).to_numpy()
    incumbent_available = result.x is not None
    fallback_used = not incumbent_available
    if incumbent_available:
        selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
    diagnostics = {
        f"milp_success_v{VERSION}": bool(result.success),
        f"milp_incumbent_available_v{VERSION}": bool(incumbent_available),
        f"milp_fallback_to_v293_used_v{VERSION}": bool(fallback_used),
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
                    f"portfolio_label_v{VERSION}": "broader_post_v293_multi_swap_milp",
                    "source_family": family,
                    "source_id": source_id,
                    f"cap_share_v{VERSION}": cap,
                    f"source_exposure_v{VERSION}": source_exposure,
                    f"source_share_v{VERSION}": share,
                    f"source_slack_v{VERSION}": cap - share,
                    f"source_cap_violated_v{VERSION}": share > cap + 1e-7,
                    f"claim_boundary_v{VERSION}": (
                        "v295 broader bounded-pool source diagnostic only"
                    ),
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
    pool_roles = pool.set_index("loan_id")[f"pool_role_v{VERSION}"].astype(str).to_dict()
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
                    f"pool_role_v{VERSION}": pool_roles.get(loan_id, "dropped_from_v293"),
                    "grade": str(row["grade"]),
                    "score_decile": str(row["score_decile"]),
                    "income_band": str(row["income_band"]),
                    "dti_band": str(row["dti_band"]),
                    "period": str(row["period"]),
                    "state_top20": str(row["state_top20"]),
                    f"claim_boundary_v{VERSION}": (
                        "v295 action contribution; not a promotion claim"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gap_diagnostics(
    *,
    universe: pd.DataFrame,
    outside_pool: pd.DataFrame,
    mean_returns: np.ndarray,
    incumbent_objective: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_returns = pd.DataFrame(
        {
            "loan_id": universe["loan_id"].astype(str),
            f"mean_return_v{VERSION}": mean_returns,
            "loan_amnt": universe["loan_amnt"].astype(float),
            "grade": universe["grade"].astype(str),
            "score_decile": universe["score_decile"].astype(str),
        }
    )
    loose_top = universe_returns.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
        TARGET_SELECTED_ROWS
    )
    top_outside = outside_pool.sort_values(f"mean_return_v{VERSION}", ascending=False).head(200)
    positive_outside = outside_pool.loc[outside_pool[f"mean_return_v{VERSION}"].gt(0)].copy()
    tight_outside = positive_outside.loc[
        positive_outside["grade"].astype(str).eq("A")
        | positive_outside["score_decile"].astype(str).eq("0")
    ].copy()
    diagnostics = pd.DataFrame(
        [
            {
                f"diagnostic_id_v{VERSION}": "loose_top171_return_upper_bound_no_constraints",
                f"candidate_rows_v{VERSION}": int(len(universe_returns)),
                f"value_v{VERSION}": float(loose_top[f"mean_return_v{VERSION}"].sum()),
                f"delta_vs_incumbent_v{VERSION}": float(
                    loose_top[f"mean_return_v{VERSION}"].sum() - incumbent_objective
                ),
                f"claim_boundary_v{VERSION}": (
                    "loose no-budget/no-source/no-CVaR upper bound; not a global gap certificate"
                ),
            },
            {
                f"diagnostic_id_v{VERSION}": "outside_v295_pool_rows",
                f"candidate_rows_v{VERSION}": int(len(outside_pool)),
                f"value_v{VERSION}": int(len(outside_pool)),
                f"delta_vs_incumbent_v{VERSION}": np.nan,
                f"claim_boundary_v{VERSION}": (
                    "rows outside bounded v295 MILP pool keep full-universe gap open"
                ),
            },
            {
                f"diagnostic_id_v{VERSION}": "outside_v295_pool_positive_return_rows",
                f"candidate_rows_v{VERSION}": int(len(outside_pool)),
                f"value_v{VERSION}": int(len(positive_outside)),
                f"delta_vs_incumbent_v{VERSION}": np.nan,
                f"claim_boundary_v{VERSION}": (
                    "positive omitted rows outside solved pool require future pricing/bounds"
                ),
            },
            {
                f"diagnostic_id_v{VERSION}": "outside_v295_pool_source_tight_positive_rows",
                f"candidate_rows_v{VERSION}": int(len(outside_pool)),
                f"value_v{VERSION}": int(len(tight_outside)),
                f"delta_vs_incumbent_v{VERSION}": np.nan,
                f"claim_boundary_v{VERSION}": (
                    "source-tight positives outside the solved pool remain a branch-price target"
                ),
            },
        ]
    )
    top_outside = top_outside[["loan_id", "loan_amnt", f"mean_return_v{VERSION}", *FAMILIES]].copy()
    top_outside[f"claim_boundary_v{VERSION}"] = (
        "top outside-pool diagnostic only; not evidence of feasible improving columns"
    )
    return diagnostics, top_outside


def _update_claim_boundaries(*, challenger_found: bool, bounded_optimal: bool) -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "Paper 4 has a v295 broader bounded multi-swap/global-gap probe.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v295_broader_multi_swap_or_global_gap_probe.csv"
                ),
                "boundary": (
                    "Post-v293 bounded pool with expanded return and micro-relief candidates; "
                    "not a full-universe proof."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v295 proves bounded-pool optimality for the expanded post-v293 pool.",
                "allowed": bool(bounded_optimal),
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v295_broader_multi_swap_or_global_gap_probe.csv"
                ),
                "boundary": "Only within the v295 candidate pool and constraints.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v295 finds a bounded-pool improvement over v293.",
                "allowed": bool(challenger_found),
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/"
                    "paper4_v295_broader_multi_swap_or_global_gap_probe.csv"
                ),
                "boundary": (
                    "If true, requires v296 repricing and still cannot promote without "
                    "global/dynamic gates."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v295 proves full-universe global integer optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v295_claim_blockers.csv"
                ),
                "boundary": "Rows remain outside the solved pool and no global bound is produced.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v295 authorizes a Paper 4 working champion.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v295_claim_blockers.csv"
                ),
                "boundary": "Working champion replacement remains blocked by global/dynamic gates.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v295 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v295_claim_blockers.csv"
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


def _update_backlog(*, challenger_found: bool) -> None:
    path = TABLE_DIR / "paper4_living_lab_backlog.csv"
    current = read_csv("paper4_living_lab_backlog.csv")
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_post_v295_broader_pool_reprice.csv"
        if challenger_found
        else f"paper4_v{NEXT_VERSION}_full_universe_branch_price_bound_or_dynamic_replay.csv"
    )
    additions = pd.DataFrame(
        [
            {
                "horizon": "immediate",
                "lane": "CVaR/OCE",
                "executable_item": (
                    "v295 expands the post-v293 multi-swap pool and records residual "
                    "full-universe gap diagnostics after v294 local repricing cleared."
                ),
                "status": (
                    "broader_pool_improvement_found_requires_repricing"
                    if challenger_found
                    else "broader_pool_no_improvement_global_gap_still_open"
                ),
                "next_artifact": next_artifact,
                "success_condition": (
                    "either reprice a new bounded challenger or move to full-universe "
                    "branch-price/dynamic validation without promoting"
                ),
                "last_wave": "v295",
                "execution_result": (
                    "bounded_broader_pool_challenger_found"
                    if challenger_found
                    else "expanded_pool_did_not_beat_v293"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v295")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V295_BROADER_MULTI_SWAP_GLOBAL_GAP_PROBE_START -->"
    end = "<!-- V295_BROADER_MULTI_SWAP_GLOBAL_GAP_PROBE_END -->"
    block = f"""
{start}

## Wave v295: Broader Multi-Swap / Global-Gap Probe

Generated: {status["generated_at_utc"]}

### Objective

After v294 cleared the post-v293 one-swap gate, v295 tests a broader bounded
multi-swap route. It expands the post-v293 MILP pool to 20,000 high-return
omitted loans plus 15,000 micro source-relief candidates, while keeping
cardinality 171, budget, exact source caps and CVaR no worse than v293.

### Results

- Pool rows: `{status["pool_rows_v295"]}`.
- Top-return candidate limit: `{status["top_return_limit_v295"]}`.
- Micro-relief candidate limit: `{status["micro_relief_limit_v295"]}`.
- Outside-pool rows: `{status["outside_pool_rows_v295"]}`.
- MILP success: `{status["milp_success_v295"]}`.
- MILP gap: `{status["milp_gap_v295"]}`.
- Selected rows: `{status["selected_rows_v295"]}`.
- Added rows vs v293: `{status["added_rows_vs_v293_v295"]}`.
- Dropped rows vs v293: `{status["dropped_rows_vs_v293_v295"]}`.
- Objective return: `{status["objective_return_v295"]}`.
- Delta return vs v293: `{status["delta_return_vs_v293_v295"]}`.
- CVaR90: `{status["scenario_loss_cvar90_v295"]}`.
- Delta CVaR90 vs v293: `{status["delta_cvar90_vs_v293_v295"]}`.
- Source cap violations: `{status["source_cap_violations_v295"]}`.
- Broader bounded-pool improvement found:
  `{status["broader_pool_challenger_found_v295"]}`.
- Valid full-universe gap certificate:
  `{status["valid_full_universe_gap_certificate_v295"]}`.

### Interpretation

v295 is a broader multi-swap experiment, not a promotion gate. A successful
bounded MILP can expand what we know about v293, but any improvement still
needs repricing and any non-improvement still leaves rows outside the solved
pool. Full-universe branch-price bounds and dynamic validation remain separate
future gates.

### Claim Impact

- Allowed: broader bounded post-v293 multi-swap probe and residual gap
  diagnostics.
- Still prohibited: working champion replacement, full-universe optimality,
  Paper Estrella replacement, final Paper 4 promotion and live deployment.

### Quarto Promotion Decision

Keep v295 in the living notebook. Promotion remains blocked.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v293_diverse_pool_allocations.parquet").reset_index(drop=True)
    v293_status = json.loads((STATUS_DIR / "paper4_v293_status.json").read_text(encoding="utf-8"))
    v294_status = json.loads((STATUS_DIR / "paper4_v294_status.json").read_text(encoding="utf-8"))
    v279_summary = read_csv("paper4_v279_restricted_pool_milp_repair_summary.csv")
    source_caps = read_csv("paper4_v80_full_pool_milp_gap_source_summary.csv")
    source_caps = source_caps.loc[
        source_caps["portfolio_label_v80"].eq("focused_full_pool_binary_milp")
    ].copy()
    if universe.empty or selected.empty or v279_summary.empty or source_caps.empty:
        raise RuntimeError("Missing v55, v293, v279, or source-cap inputs for v295.")
    if not bool(v294_status["post_v293_one_swap_local_optimality_cleared_v294"]):
        raise RuntimeError("v295 requires the v294 post-v293 one-swap gate to be cleared.")

    v279_row = v279_summary.iloc[0]
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    incumbent_ids = set(selected["loan_id"].astype(str))
    pool, candidate_registry, outside_pool = _build_pool(
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
    incumbent_objective = float(v293_status["objective_return_v293"])
    incumbent_cvar = float(v293_status["scenario_loss_cvar90_v293"])
    diagnostics, selected_mask, constraints = _solve_milp(
        pool=pool,
        losses_pool=losses_pool,
        mean_returns_pool=mean_returns_pool,
        source_caps=source_caps,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        cvar_cap=incumbent_cvar,
        incumbent_ids=incumbent_ids,
    )

    solution = pool.loc[selected_mask].copy()
    solution_idx = idx_by_id.loc[solution["loan_id"].astype(str)].to_numpy()
    solution_losses = losses[:, solution_idx].sum(axis=1)
    solution_ids = set(solution["loan_id"].astype(str))
    added_ids = sorted(solution_ids - incumbent_ids)
    dropped_ids = sorted(incumbent_ids - solution_ids)
    kept_current = len(solution_ids & incumbent_ids)
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
    delta_return_vs_v293 = objective_return - incumbent_objective
    delta_cvar_vs_v293 = cvar90 - incumbent_cvar
    budget_feasible = exposure_min - 1e-7 <= exposure <= exposure_max + 1e-7
    cvar_feasible = cvar90 <= incumbent_cvar + 1e-7
    source_feasible = source_violations == 0
    bounded_optimal = bool(diagnostics[f"milp_success_v{VERSION}"])
    challenger_found = (
        bounded_optimal
        and cardinality_restored
        and budget_feasible
        and cvar_feasible
        and source_feasible
        and delta_return_vs_v293 > 1e-9
    )

    solution["selected_v295"] = True
    solution["portfolio_label_v295"] = "broader_post_v293_multi_swap_milp"
    solution["repair_action_v295"] = np.where(
        solution["loan_id"].astype(str).isin(added_ids),
        "added_by_v295_broader_milp",
        "kept_from_v293",
    )
    solution["claim_boundary_v295"] = (
        "bounded broader post-v293 MILP allocation; no working-champion promotion"
    )
    action = pd.DataFrame(
        [
            {
                f"action_id_v{VERSION}": "broader_post_v293_multi_swap_milp_action",
                f"added_loan_ids_v{VERSION}": "|".join(added_ids),
                f"dropped_loan_ids_v{VERSION}": "|".join(dropped_ids),
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_vs_v293_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_vs_v293_v{VERSION}": int(len(dropped_ids)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"broader_pool_challenger_found_v{VERSION}": challenger_found,
                f"claim_boundary_v{VERSION}": (
                    "bounded multi-swap action only; repricing/global gates still required"
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
                f"claim_boundary_v{VERSION}": "v295 broader MILP pool composition only",
            }
            for role, group in pool.groupby(f"pool_role_v{VERSION}", dropna=False)
        ]
    )
    constraints_summary = (
        constraints.groupby(f"constraint_type_v{VERSION}", dropna=False)
        .size()
        .reset_index(name=f"constraint_rows_v{VERSION}")
    )
    constraints_summary[f"claim_boundary_v{VERSION}"] = "v295 broader MILP constraint count only"
    contributions = _return_contributions(
        universe=universe,
        pool=pool,
        mean_returns=mean_returns,
        added_ids=added_ids,
        dropped_ids=dropped_ids,
    )
    gap_diagnostics, top_outside = _gap_diagnostics(
        universe=universe,
        outside_pool=outside_pool,
        mean_returns=mean_returns,
        incumbent_objective=incumbent_objective,
    )
    strategy_comparison = pd.DataFrame(
        [
            {
                f"strategy_id_v{VERSION}": "v293_incumbent_after_v294_reprice_gate",
                f"pool_rows_v{VERSION}": int(v293_status["pool_rows_v293"]),
                f"selected_rows_v{VERSION}": int(v293_status["selected_rows_v293"]),
                f"objective_return_v{VERSION}": incumbent_objective,
                f"delta_return_vs_v293_v{VERSION}": 0.0,
                f"scenario_loss_cvar90_v{VERSION}": incumbent_cvar,
                f"delta_cvar90_vs_v293_v{VERSION}": 0.0,
                f"source_cap_violations_v{VERSION}": int(v293_status["source_cap_violations_v293"]),
                f"broader_pool_challenger_found_v{VERSION}": False,
                f"claim_boundary_v{VERSION}": "v293 incumbent baseline after v294 local repricing",
            },
            {
                f"strategy_id_v{VERSION}": "v295_top20000_plus_micro_source_relief",
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v293_v{VERSION}": delta_return_vs_v293,
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"delta_cvar90_vs_v293_v{VERSION}": delta_cvar_vs_v293,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"broader_pool_challenger_found_v{VERSION}": challenger_found,
                f"claim_boundary_v{VERSION}": (
                    "bounded broader pool signal requiring repricing/global gates"
                ),
            },
        ]
    )
    next_artifact = (
        f"paper4_v{NEXT_VERSION}_post_v295_broader_pool_reprice.csv"
        if challenger_found
        else f"paper4_v{NEXT_VERSION}_full_universe_branch_price_bound_or_dynamic_replay.csv"
    )
    protocol = pd.DataFrame(
        [
            {
                f"protocol_id_v{VERSION}": "broader_post_v293_multi_swap_or_global_gap_probe",
                f"incumbent_version_v{VERSION}": INCUMBENT_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"base_repair_version_v{VERSION}": BASE_REPAIR_VERSION,
                f"top_return_limit_v{VERSION}": TOP_RETURN_LIMIT,
                f"micro_relief_limit_v{VERSION}": MICRO_RELIEF_LIMIT,
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"outside_pool_rows_v{VERSION}": int(len(outside_pool)),
                f"selected_rows_v{VERSION}": int(len(solution)),
                f"target_selected_rows_v{VERSION}": TARGET_SELECTED_ROWS,
                f"cardinality_restored_v{VERSION}": cardinality_restored,
                f"kept_current_rows_v{VERSION}": kept_current,
                f"added_rows_vs_v293_v{VERSION}": int(len(added_ids)),
                f"dropped_rows_vs_v293_v{VERSION}": int(len(dropped_ids)),
                f"portfolio_exposure_v{VERSION}": exposure,
                f"exposure_min_v{VERSION}": exposure_min,
                f"exposure_max_v{VERSION}": exposure_max,
                f"objective_return_v{VERSION}": objective_return,
                f"incumbent_objective_return_v{VERSION}": incumbent_objective,
                f"delta_return_vs_v293_v{VERSION}": delta_return_vs_v293,
                f"scenario_loss_mean_v{VERSION}": float(solution_losses.mean()),
                f"scenario_loss_cvar90_v{VERSION}": cvar90,
                f"incumbent_cvar90_v{VERSION}": incumbent_cvar,
                f"delta_cvar90_vs_v293_v{VERSION}": delta_cvar_vs_v293,
                f"source_cap_violations_v{VERSION}": source_violations,
                f"budget_feasible_v{VERSION}": budget_feasible,
                f"cvar_feasible_v{VERSION}": cvar_feasible,
                f"source_feasible_v{VERSION}": source_feasible,
                f"broader_pool_challenger_found_v{VERSION}": challenger_found,
                f"bounded_pool_optimality_claim_allowed_v{VERSION}": bounded_optimal,
                f"valid_full_universe_gap_certificate_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "broader bounded multi-swap probe only; no global or promotion claim"
                ),
                **diagnostics,
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "broader_pool_improvement_found",
                f"blocking_v{VERSION}": challenger_found,
                f"evidence_count_v{VERSION}": delta_return_vs_v293,
                f"required_next_artifact_v{VERSION}": next_artifact,
                f"claim_boundary_v{VERSION}": (
                    "if positive, bounded improvement must be repriced before any claim expansion"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(len(outside_pool)),
                f"required_next_artifact_v{VERSION}": ("future_full_universe_branch_price_bound"),
                f"claim_boundary_v{VERSION}": (
                    "v295 still leaves omitted loans outside the bounded solved pool"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_global_gap_certificate_missing",
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
                "claim_id": "v295_broader_multi_swap_probe_executed",
                "allowed": True,
                "artifact": "paper4_v295_broader_multi_swap_or_global_gap_probe.csv",
                "boundary": "bounded broader post-v293 MILP probe",
            },
            {
                "claim_id": "v295_bounded_pool_optimality",
                "allowed": bounded_optimal,
                "artifact": "paper4_v295_broader_multi_swap_or_global_gap_probe.csv",
                "boundary": "bounded v295 pool only",
            },
            {
                "claim_id": "v295_broader_pool_improvement_over_v293",
                "allowed": challenger_found,
                "artifact": "paper4_v295_broader_multi_swap_or_global_gap_probe.csv",
                "boundary": "requires repricing and global/dynamic gates",
            },
            {
                "claim_id": "v295_full_universe_gap_certificate",
                "allowed": False,
                "artifact": "paper4_v295_claim_blockers.csv",
                "boundary": "global certificate missing",
            },
            {
                "claim_id": "v295_working_champion",
                "allowed": False,
                "artifact": "paper4_v295_claim_blockers.csv",
                "boundary": "global/dynamic evidence missing",
            },
            {
                "claim_id": "v295_paper1_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v295_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )

    write_csv(TABLE_DIR / "paper4_v295_broader_multi_swap_or_global_gap_probe.csv", protocol)
    write_csv(TABLE_DIR / "paper4_v295_broader_multi_swap_action.csv", action)
    write_csv(TABLE_DIR / "paper4_v295_broader_strategy_comparison.csv", strategy_comparison)
    write_csv(TABLE_DIR / "paper4_v295_broader_candidate_registry.csv", candidate_registry)
    write_csv(TABLE_DIR / "paper4_v295_broader_pool_summary.csv", pool_summary)
    write_csv(TABLE_DIR / "paper4_v295_broader_constraint_summary.csv", constraints_summary)
    write_csv(TABLE_DIR / "paper4_v295_broader_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v295_return_delta_contributions.csv", contributions)
    write_csv(TABLE_DIR / "paper4_v295_global_gap_diagnostics.csv", gap_diagnostics)
    write_csv(TABLE_DIR / "paper4_v295_outside_pool_top_candidates.csv", top_outside)
    write_csv(TABLE_DIR / "paper4_v295_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v295_claim_matrix_delta.csv", claim_matrix)
    solution.to_parquet(
        TABLE_DIR / "paper4_v295_broader_multi_swap_allocations.parquet", index=False
    )
    _update_claim_boundaries(challenger_found=challenger_found, bounded_optimal=bounded_optimal)
    _update_backlog(challenger_found=challenger_found)

    row = protocol.iloc[0]
    status = {
        "phase": "v295_broader_multi_swap_or_global_gap_probe",
        "schema_version": "2026-05-15.295",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "incumbent_version_v295": INCUMBENT_VERSION,
        "reprice_version_v295": REPRICE_VERSION,
        "base_repair_version_v295": BASE_REPAIR_VERSION,
        "top_return_limit_v295": TOP_RETURN_LIMIT,
        "micro_relief_limit_v295": MICRO_RELIEF_LIMIT,
        "pool_rows_v295": int(row[f"pool_rows_v{VERSION}"]),
        "outside_pool_rows_v295": int(row[f"outside_pool_rows_v{VERSION}"]),
        "selected_rows_v295": int(row[f"selected_rows_v{VERSION}"]),
        "target_selected_rows_v295": TARGET_SELECTED_ROWS,
        "cardinality_restored_v295": bool(row[f"cardinality_restored_v{VERSION}"]),
        "kept_current_rows_v295": int(row[f"kept_current_rows_v{VERSION}"]),
        "added_rows_vs_v293_v295": int(row[f"added_rows_vs_v293_v{VERSION}"]),
        "dropped_rows_vs_v293_v295": int(row[f"dropped_rows_vs_v293_v{VERSION}"]),
        "portfolio_exposure_v295": float(row[f"portfolio_exposure_v{VERSION}"]),
        "objective_return_v295": float(row[f"objective_return_v{VERSION}"]),
        "incumbent_objective_return_v295": float(row[f"incumbent_objective_return_v{VERSION}"]),
        "delta_return_vs_v293_v295": float(row[f"delta_return_vs_v293_v{VERSION}"]),
        "scenario_loss_cvar90_v295": float(row[f"scenario_loss_cvar90_v{VERSION}"]),
        "incumbent_cvar90_v295": float(row[f"incumbent_cvar90_v{VERSION}"]),
        "delta_cvar90_vs_v293_v295": float(row[f"delta_cvar90_vs_v293_v{VERSION}"]),
        "source_cap_violations_v295": source_violations,
        "budget_feasible_v295": bool(row[f"budget_feasible_v{VERSION}"]),
        "cvar_feasible_v295": bool(row[f"cvar_feasible_v{VERSION}"]),
        "source_feasible_v295": bool(row[f"source_feasible_v{VERSION}"]),
        "broader_pool_challenger_found_v295": bool(
            row[f"broader_pool_challenger_found_v{VERSION}"]
        ),
        "bounded_pool_optimality_claim_allowed_v295": bool(
            row[f"bounded_pool_optimality_claim_allowed_v{VERSION}"]
        ),
        "valid_full_universe_gap_certificate_v295": False,
        "milp_success_v295": bool(row[f"milp_success_v{VERSION}"]),
        "milp_incumbent_available_v295": bool(row[f"milp_incumbent_available_v{VERSION}"]),
        "milp_fallback_to_v293_used_v295": bool(row[f"milp_fallback_to_v293_used_v{VERSION}"]),
        "milp_status_v295": int(row[f"milp_status_v{VERSION}"]),
        "milp_gap_v295": float(row[f"milp_gap_v{VERSION}"]),
        "milp_node_count_v295": int(row[f"milp_node_count_v{VERSION}"]),
        "milp_variable_count_v295": int(row[f"milp_variable_count_v{VERSION}"]),
        "milp_binary_variable_count_v295": int(row[f"milp_binary_variable_count_v{VERSION}"]),
        "milp_constraint_rows_v295": int(row[f"milp_constraint_rows_v{VERSION}"]),
        "strategy_comparison_rows_v295": int(len(strategy_comparison)),
        "candidate_registry_rows_v295": int(len(candidate_registry)),
        "pool_summary_rows_v295": int(len(pool_summary)),
        "constraint_summary_rows_v295": int(len(constraints_summary)),
        "source_summary_rows_v295": int(len(source_summary)),
        "return_contribution_rows_v295": int(len(contributions)),
        "gap_diagnostic_rows_v295": int(len(gap_diagnostics)),
        "outside_pool_top_candidate_rows_v295": int(len(top_outside)),
        "claim_blocker_rows_v295": int(len(blockers)),
        "claim_matrix_rows_v295": int(len(claim_matrix)),
        "working_champion_claim_allowed_v295": False,
        "full_universe_integer_optimality_claim_allowed_v295": False,
        "paper1_promotion_allowed_v295": False,
        "paper4_working_champion_changed_v295": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "next_artifact_v295": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v295 is a broader bounded multi-swap/global-gap probe; no working "
            "champion, global optimality, or final promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v295_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v295": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
