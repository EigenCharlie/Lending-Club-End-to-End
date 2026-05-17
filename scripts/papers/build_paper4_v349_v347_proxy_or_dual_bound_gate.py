#!/usr/bin/env python3
"""Build Paper 4 v349 v347 proxy-repair / dual-bound gate artifacts."""

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

VERSION = 349
BASE_VERSION = 347
REFERENCE_VERSION = 338
REPRICE_VERSION = 348
NEXT_VERSION = 350
TARGET_SELECTED_ROWS = 171
MILP_TIME_LIMIT_SECONDS = 60.0
MIP_REL_GAP = 1e-6
OBSERVED_PROXY_WEIGHT = 100_000.0
EXPOSURE_MIN = 842292.375
EXPOSURE_MAX = 850000.0
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v347_dual_bound_after_proxy_gate.csv"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) else numeric


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


def _cap_lookup(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
    return {
        str(row["source_id"]): float(row[f"cap_share_v{BASE_VERSION}"])
        for _, row in family_caps.iterrows()
    }


def _build_pool(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    observed_ids: set[str],
    idx_by_id: pd.Series,
    mean_returns: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["loan_id"].astype(str))
    universe = universe.copy()
    universe["loan_id"] = universe["loan_id"].astype(str)
    universe[f"mean_return_v{VERSION}"] = mean_returns
    universe[f"observed_v47_proxy_v{VERSION}"] = universe["loan_id"].isin(observed_ids)
    observed_omitted = universe.loc[
        ~universe["loan_id"].isin(selected_ids) & universe[f"observed_v47_proxy_v{VERSION}"]
    ].copy()
    pool = pd.concat(
        [universe.loc[universe["loan_id"].isin(selected_ids)], observed_omitted],
        ignore_index=True,
    ).drop_duplicates("loan_id")
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool[f"incumbent_selected_v{VERSION}"] = pool["loan_id"].isin(selected_ids)
    pool[f"observed_candidate_v{VERSION}"] = (
        ~pool[f"incumbent_selected_v{VERSION}"] & pool[f"observed_v47_proxy_v{VERSION}"]
    )
    pool[f"pool_role_v{VERSION}"] = np.where(
        pool[f"incumbent_selected_v{VERSION}"],
        "v347_selected_base",
        "observed_omitted_candidate",
    )
    pool[f"universe_idx_v{VERSION}"] = idx_by_id.loc[pool["loan_id"]].to_numpy(int)
    selected_observed = int(
        pool.loc[pool[f"incumbent_selected_v{VERSION}"], f"observed_v47_proxy_v{VERSION}"].sum()
    )
    selected_missing = int(pool[f"incumbent_selected_v{VERSION}"].sum() - selected_observed)
    pool_summary = pd.DataFrame(
        [
            {
                f"pool_id_v{VERSION}": "v347_plus_all_observed_omitted_candidates",
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_base_rows_v{VERSION}": int(pool[f"incumbent_selected_v{VERSION}"].sum()),
                f"selected_observed_proxy_rows_v{VERSION}": selected_observed,
                f"selected_missing_proxy_rows_v{VERSION}": selected_missing,
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool[f"observed_candidate_v{VERSION}"].sum()
                ),
                f"total_observed_omitted_rows_v{VERSION}": int(len(observed_omitted)),
                f"expanded_pool_includes_all_observed_omitted_v{VERSION}": True,
                f"claim_boundary_v{VERSION}": (
                    "all observed omitted candidates plus v347 selected loans; not full-v55 pricing"
                ),
            }
        ]
    )
    return pool, pool_summary


def _constraints(
    *,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    source_caps: pd.DataFrame,
    target_period_counts: dict[str, int],
    return_floor: float,
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
    lb.append(EXPOSURE_MIN)
    ub.append(EXPOSURE_MAX)

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

    return_row = np.zeros(var_count)
    return_row[:n] = pool[f"mean_return_v{VERSION}"].to_numpy(float)
    rows.append(return_row)
    lb.append(return_floor)
    ub.append(np.inf)

    amounts = pool["loan_amnt"].to_numpy(float)
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


def _solve_tier(
    *,
    tier_id: str,
    pool: pd.DataFrame,
    losses_pool: np.ndarray,
    losses_full: np.ndarray,
    source_caps: pd.DataFrame,
    target_period_counts: dict[str, int],
    return_floor: float,
    cvar_cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    constraints, bounds, integrality, var_count, constraint_count = _constraints(
        pool=pool,
        losses_pool=losses_pool,
        source_caps=source_caps,
        target_period_counts=target_period_counts,
        return_floor=return_floor,
        cvar_cap=cvar_cap,
    )
    n = len(pool)
    scenario_count = losses_pool.shape[0]
    objective = np.zeros(var_count)
    objective[:n] = -(
        OBSERVED_PROXY_WEIGHT * pool[f"observed_v47_proxy_v{VERSION}"].to_numpy(float)
        + pool[f"mean_return_v{VERSION}"].to_numpy(float)
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    selected = pd.DataFrame()
    metrics: dict[str, Any] = {
        "tier_id": tier_id,
        "solver_success": bool(result.success),
        "milp_status": int(result.status),
        "milp_message": str(result.message),
        "milp_fun": _safe_float(result.fun),
        "milp_gap": _safe_float(getattr(result, "mip_gap", None)),
        "milp_dual_bound": _safe_float(getattr(result, "mip_dual_bound", None)),
        "milp_node_count": _safe_int(getattr(result, "mip_node_count", None)),
        "constraint_rows": constraint_count,
        "variable_count": var_count,
        "scenario_count": scenario_count,
        "return_floor": float(return_floor),
        "cvar_cap": float(cvar_cap),
        "incumbent_found": result.x is not None,
    }
    if result.x is not None:
        selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
        selected = pool.loc[selected_mask].copy()
        selected_idx = selected[f"universe_idx_v{VERSION}"].to_numpy(int)
        scenario_losses = losses_full[:, selected_idx].sum(axis=1)
        observed_rows = int(selected[f"observed_v47_proxy_v{VERSION}"].sum())
        missing_rows = int((~selected[f"observed_v47_proxy_v{VERSION}"]).sum())
        metrics.update(
            {
                "selected_rows": int(len(selected)),
                "portfolio_exposure": float(selected["loan_amnt"].sum()),
                "objective_return": float(selected[f"mean_return_v{VERSION}"].sum()),
                "scenario_loss_mean": float(scenario_losses.mean()),
                "scenario_loss_cvar90": v70._tail_cvar(scenario_losses),
                "observed_proxy_rows": observed_rows,
                "missing_proxy_rows": missing_rows,
                "period_distribution": _period_distribution(selected),
            }
        )
    return selected, metrics


def _tier_summary(
    tier_metrics: list[dict[str, Any]],
    reference: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics in tier_metrics:
        objective_return = metrics.get("objective_return")
        cvar = metrics.get("scenario_loss_cvar90")
        rows.append(
            {
                f"tier_id_v{VERSION}": metrics["tier_id"],
                f"solver_success_v{VERSION}": metrics["solver_success"],
                f"milp_status_v{VERSION}": metrics["milp_status"],
                f"milp_message_v{VERSION}": metrics["milp_message"],
                f"milp_fun_v{VERSION}": metrics["milp_fun"],
                f"milp_gap_v{VERSION}": metrics["milp_gap"],
                f"milp_dual_bound_v{VERSION}": metrics["milp_dual_bound"],
                f"milp_node_count_v{VERSION}": metrics["milp_node_count"],
                f"incumbent_found_v{VERSION}": metrics["incumbent_found"],
                f"constraint_rows_v{VERSION}": metrics["constraint_rows"],
                f"variable_count_v{VERSION}": metrics["variable_count"],
                f"return_floor_v{VERSION}": metrics["return_floor"],
                f"cvar_cap_v{VERSION}": metrics["cvar_cap"],
                f"selected_rows_v{VERSION}": metrics.get("selected_rows"),
                f"portfolio_exposure_v{VERSION}": metrics.get("portfolio_exposure"),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v347_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v347_return"]),
                f"delta_return_vs_v338_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v338_return"]),
                f"scenario_loss_mean_v{VERSION}": metrics.get("scenario_loss_mean"),
                f"scenario_loss_cvar90_v{VERSION}": cvar,
                f"delta_cvar90_vs_v347_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v347_cvar"]),
                f"delta_cvar90_vs_v338_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v338_cvar"]),
                f"observed_proxy_rows_v{VERSION}": metrics.get("observed_proxy_rows"),
                f"missing_proxy_rows_v{VERSION}": metrics.get("missing_proxy_rows"),
                f"observed_proxy_delta_vs_v347_v{VERSION}": None
                if metrics.get("observed_proxy_rows") is None
                else int(metrics["observed_proxy_rows"] - reference["v347_observed"]),
                f"missing_proxy_delta_vs_v347_v{VERSION}": None
                if metrics.get("missing_proxy_rows") is None
                else int(metrics["missing_proxy_rows"] - reference["v347_missing"]),
                f"coverage_restores_or_improves_v338_v{VERSION}": None
                if metrics.get("missing_proxy_rows") is None
                else int(metrics["missing_proxy_rows"]) <= int(reference["v338_missing"]),
                f"period_distribution_v{VERSION}": json.dumps(
                    metrics.get("period_distribution", {}), sort_keys=True
                ),
                f"claim_boundary_v{VERSION}": (
                    "all-observed-omitted proxy repair tier; not full-universe pricing"
                ),
            }
        )
    return pd.DataFrame(rows)


def _actions(*, pool: pd.DataFrame, selected: pd.DataFrame, tier_id: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=[
                f"tier_id_v{VERSION}",
                f"action_v{VERSION}",
                "loan_id",
                "loan_amnt",
                *FAMILIES,
                f"mean_return_v{VERSION}",
                f"observed_v47_proxy_v{VERSION}",
                f"claim_boundary_v{VERSION}",
            ]
        )
    selected_ids = set(selected["loan_id"].astype(str))
    work = pool.copy()
    work["loan_id"] = work["loan_id"].astype(str)
    changed = work.loc[
        (work[f"incumbent_selected_v{VERSION}"] & ~work["loan_id"].isin(selected_ids))
        | (~work[f"incumbent_selected_v{VERSION}"] & work["loan_id"].isin(selected_ids))
    ].copy()
    changed[f"tier_id_v{VERSION}"] = tier_id
    changed[f"action_v{VERSION}"] = np.where(
        changed["loan_id"].isin(selected_ids),
        "add_observed_candidate",
        "drop_v347_selected",
    )
    changed[f"claim_boundary_v{VERSION}"] = (
        "v349 coverage-only incumbent action list; not a repair recommendation"
    )
    return changed[
        [
            f"tier_id_v{VERSION}",
            f"action_v{VERSION}",
            "loan_id",
            "loan_amnt",
            *FAMILIES,
            f"mean_return_v{VERSION}",
            f"observed_v47_proxy_v{VERSION}",
            f"claim_boundary_v{VERSION}",
        ]
    ].sort_values([f"action_v{VERSION}", f"mean_return_v{VERSION}"], ascending=[True, False])


def _update_claim_boundaries() -> None:
    path = TABLE_DIR / "paper4_current_claim_boundaries.csv"
    current = read_csv("paper4_current_claim_boundaries.csv")
    additions = pd.DataFrame(
        [
            {
                "claim": "v349 tests v347 proxy repair over all observed omitted candidates.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v349_proxy_repair_tier_summary.csv"
                ),
                "boundary": "All observed omitted candidates plus v347 selected loans; not full-v55.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v349 documents strict and relaxed v347 proxy-repair infeasibility.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v349_proxy_repair_tier_summary.csv"
                ),
                "boundary": (
                    "No repair preserves v347 CVaR with either v347 or v338 return floor "
                    "inside the all-observed-omitted pool."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v349 repairs proxy coverage while preserving v338 return and v347 CVaR.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v349_claim_blockers.csv"
                ),
                "boundary": "Relaxed v338-return/v347-CVaR proxy repair tier is infeasible.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v349 proves a valid branch-price or global integer bound.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v349_claim_blockers.csv"
                ),
                "boundary": "No full-v55 dual-bound loop or global certificate is created.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v349 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v349_claim_blockers.csv"
                ),
                "boundary": "No final promotion, working champion or deployment gate is created.",
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
                    "v349 tests whether the one-swap-cleared v347 candidate can repair "
                    "proxy coverage using all observed omitted candidates under static gates."
                ),
                "status": "strict_and_relaxed_v347_proxy_repair_infeasible",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "build a full-v55 dual-bound/global gate or an explicit proxy-value "
                    "tradeoff protocol without promotion"
                ),
                "last_wave": "v349",
                "execution_result": (
                    "strict_and_relaxed_v347_proxy_repair_infeasible_coverage_only_return_collapse"
                ),
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v349")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V349_V347_PROXY_OR_DUAL_BOUND_GATE_START -->"
    end = "<!-- V349_V347_PROXY_OR_DUAL_BOUND_GATE_END -->"
    block = f"""
{start}

## Wave v349: v347 Proxy Repair / Dual-Bound Gate

Generated: {status["generated_at_utc"]}

### Objective

v348 cleared the immediate one-swap repricing gate for the v347 candidate, but
v347 still has 75 missing proxy rows. v349 tests whether all observed omitted
proxy candidates can repair that gap while preserving the v347 CVaR cap and
economically relevant return floors.

### Results

- Pool rows: `{status["pool_rows_v349"]}`.
- Observed omitted candidate rows:
  `{status["observed_omitted_candidate_rows_v349"]}`.
- Strict v347-return/v347-CVaR repair feasible:
  `{status["strict_v347_repair_feasible_v349"]}`.
- Relaxed v338-return/v347-CVaR repair feasible:
  `{status["relaxed_v338_return_repair_feasible_v349"]}`.
- Coverage-only incumbent found:
  `{status["coverage_only_incumbent_found_v349"]}`.
- Coverage-only missing proxy rows:
  `{status["coverage_only_missing_proxy_rows_v349"]}`.
- Coverage-only return delta vs v347:
  `{status["coverage_only_delta_return_vs_v347_v349"]}`.
- Coverage-only CVaR delta vs v347:
  `{status["coverage_only_delta_cvar90_vs_v347_v349"]}`.
- Valid branch-price bound:
  `{status["valid_branch_price_bound_v349"]}`.

### Interpretation

v349 sharpens the tradeoff. Proxy coverage can be maximized under the v347 CVaR
cap only by accepting a catastrophic return collapse; the strict and relaxed
economic repair tiers are infeasible. The evidence supports a proxy/global
blocker, not a champion.

### Claim Impact

- Allowed: all-observed-omitted proxy repair gate executed; strict and relaxed
  economic repair tiers are infeasible in that scope.
- Still prohibited: proxy-repaired v347 candidate, full-universe/global
  optimality, branch-price certificate, contractual IFRS9, live deployment,
  Paper Estrella replacement, final Paper 4 promotion and working champion
  claims.

### Quarto Promotion Decision

Keep v349 in the living notebook. The next wave should attempt a dual-bound or
explicit proxy-value tradeoff protocol without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    selected = read_parquet("paper4_v347_v338_multi_source_relief_allocations.parquet").reset_index(
        drop=True
    )
    v347_summary = read_csv("paper4_v347_v338_apply_multi_source_relief_candidate.csv")
    v338_summary = read_csv("paper4_v338_apply_next_post_v336_swap.csv")
    source_caps = read_csv("paper4_v347_v338_multi_source_relief_source_summary.csv")
    v348_status = json.loads((STATUS_DIR / "paper4_v348_status.json").read_text(encoding="utf-8"))
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(
        df.empty for df in [universe, selected, v347_summary, v338_summary, source_caps, v47_panel]
    ):
        raise RuntimeError("Missing v349 proxy/dual-bound gate inputs.")
    if not bool(v348_status["post_v347_one_swap_local_optimality_cleared_v348"]):
        raise RuntimeError("v349 expects v348 to clear the post-v347 one-swap gate.")
    if FORBIDDEN_FINAL_PROMOTION.exists():
        raise RuntimeError("Forbidden Paper 4 final promotion artifact exists.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    selected["loan_id"] = selected["loan_id"].astype(str)
    for frame in [universe, selected]:
        for family in FAMILIES:
            frame[family] = frame[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool, pool_summary = _build_pool(
        universe=universe,
        selected=selected,
        observed_ids=observed_ids,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    losses_pool = losses[:, pool[f"universe_idx_v{VERSION}"].to_numpy(int)]

    v347_row = v347_summary.iloc[0]
    v338_row = v338_summary.iloc[0]
    reference = {
        "v347_return": float(v347_row["objective_return_v347"]),
        "v347_cvar": float(v347_row["scenario_loss_cvar90_v347"]),
        "v347_observed": int(v347_row["observed_proxy_rows_v347"]),
        "v347_missing": int(v347_row["missing_proxy_rows_v347"]),
        "v338_return": float(v338_row["objective_return_v338"]),
        "v338_cvar": float(v338_row["scenario_loss_cvar90_v338"]),
        "v338_observed": int(v338_row["observed_proxy_rows_v338"]),
        "v338_missing": int(v338_row["missing_proxy_rows_v338"]),
    }
    target_period_counts = _period_distribution(selected)
    tiers = [
        {
            "tier_id": "strict_v347_return_v347_cvar",
            "return_floor": reference["v347_return"],
            "cvar_cap": reference["v347_cvar"],
        },
        {
            "tier_id": "relaxed_v338_return_v347_cvar",
            "return_floor": reference["v338_return"],
            "cvar_cap": reference["v347_cvar"],
        },
        {
            "tier_id": "coverage_only_v347_cvar",
            "return_floor": -1e12,
            "cvar_cap": reference["v347_cvar"],
        },
    ]
    tier_metrics: list[dict[str, Any]] = []
    coverage_selected = pd.DataFrame()
    for tier in tiers:
        selected_tier, metrics = _solve_tier(
            tier_id=tier["tier_id"],
            pool=pool,
            losses_pool=losses_pool,
            losses_full=losses,
            source_caps=source_caps,
            target_period_counts=target_period_counts,
            return_floor=float(tier["return_floor"]),
            cvar_cap=float(tier["cvar_cap"]),
        )
        tier_metrics.append(metrics)
        if tier["tier_id"] == "coverage_only_v347_cvar":
            coverage_selected = selected_tier

    tier_summary = _tier_summary(tier_metrics, reference)
    strict_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("strict_v347_return_v347_cvar")
    ].iloc[0]
    relaxed_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("relaxed_v338_return_v347_cvar")
    ].iloc[0]
    coverage_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("coverage_only_v347_cvar")
    ].iloc[0]
    strict_feasible = bool(strict_row[f"solver_success_v{VERSION}"])
    relaxed_feasible = bool(relaxed_row[f"solver_success_v{VERSION}"])
    coverage_incumbent_found = bool(coverage_row[f"incumbent_found_v{VERSION}"])
    coverage_missing = (
        None
        if pd.isna(coverage_row[f"missing_proxy_rows_v{VERSION}"])
        else int(coverage_row[f"missing_proxy_rows_v{VERSION}"])
    )
    coverage_delta_return = (
        None
        if pd.isna(coverage_row[f"delta_return_vs_v347_v{VERSION}"])
        else float(coverage_row[f"delta_return_vs_v347_v{VERSION}"])
    )
    coverage_delta_cvar = (
        None
        if pd.isna(coverage_row[f"delta_cvar90_vs_v347_v{VERSION}"])
        else float(coverage_row[f"delta_cvar90_vs_v347_v{VERSION}"])
    )
    main = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v349_v347_proxy_or_dual_bound_gate",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"reference_version_v{VERSION}": REFERENCE_VERSION,
                f"reprice_version_v{VERSION}": REPRICE_VERSION,
                f"pool_rows_v{VERSION}": int(pool_summary[f"pool_rows_v{VERSION}"].iloc[0]),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool_summary[f"observed_omitted_candidate_rows_v{VERSION}"].iloc[0]
                ),
                f"selected_observed_proxy_rows_v{VERSION}": reference["v347_observed"],
                f"selected_missing_proxy_rows_v{VERSION}": reference["v347_missing"],
                f"v338_missing_proxy_rows_v{VERSION}": reference["v338_missing"],
                f"strict_v347_repair_feasible_v{VERSION}": strict_feasible,
                f"relaxed_v338_return_repair_feasible_v{VERSION}": relaxed_feasible,
                f"coverage_only_solver_success_v{VERSION}": bool(
                    coverage_row[f"solver_success_v{VERSION}"]
                ),
                f"coverage_only_incumbent_found_v{VERSION}": coverage_incumbent_found,
                f"coverage_only_observed_proxy_rows_v{VERSION}": None
                if coverage_missing is None
                else TARGET_SELECTED_ROWS - coverage_missing,
                f"coverage_only_missing_proxy_rows_v{VERSION}": coverage_missing,
                f"coverage_only_delta_return_vs_v347_v{VERSION}": coverage_delta_return,
                f"coverage_only_delta_cvar90_vs_v347_v{VERSION}": coverage_delta_cvar,
                f"coverage_only_return_collapse_flag_v{VERSION}": (
                    coverage_delta_return is not None and coverage_delta_return < -1000.0
                ),
                f"valid_branch_price_bound_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "proxy-repair gate over observed omitted candidates; no global bound or promotion"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "strict_v347_proxy_repair_infeasible",
                f"blocking_v{VERSION}": not strict_feasible,
                f"evidence_count_v{VERSION}": int(strict_row[f"milp_status_v{VERSION}"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "strict v347-return/v347-CVaR repair is infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "relaxed_v338_return_proxy_repair_infeasible",
                f"blocking_v{VERSION}": not relaxed_feasible,
                f"evidence_count_v{VERSION}": int(relaxed_row[f"milp_status_v{VERSION}"]),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "relaxed v338-return/v347-CVaR repair is infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "coverage_only_return_collapse",
                f"blocking_v{VERSION}": coverage_delta_return is not None
                and coverage_delta_return < -1000.0,
                f"evidence_count_v{VERSION}": 0
                if coverage_delta_return is None
                else int(abs(coverage_delta_return)),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "coverage-only incumbent restores proxy rows only by destroying return"
                ),
            },
            {
                f"blocker_id_v{VERSION}": "valid_branch_price_bound_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "no full-v55 dual-bound loop or termination certificate",
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
                "claim_id": "v349_proxy_repair_gate_executed",
                "allowed": True,
                "artifact": "paper4_v349_v347_proxy_or_dual_bound_gate.csv",
                "boundary": "all observed omitted proxy candidates; not full-v55 pricing",
            },
            {
                "claim_id": "v349_strict_and_relaxed_proxy_repair_infeasible",
                "allowed": not strict_feasible and not relaxed_feasible,
                "artifact": "paper4_v349_proxy_repair_tier_summary.csv",
                "boundary": "scope-limited all-observed-omitted MILP tiers",
            },
            {
                "claim_id": "v349_proxy_repair_candidate_found",
                "allowed": strict_feasible or relaxed_feasible,
                "artifact": "paper4_v349_claim_blockers.csv",
                "boundary": "requires economic repair feasibility, not coverage-only incumbent",
            },
            {
                "claim_id": "v349_valid_branch_price_bound",
                "allowed": False,
                "artifact": "paper4_v349_claim_blockers.csv",
                "boundary": "formal dual-bound loop missing",
            },
            {
                "claim_id": "v349_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v349_claim_blockers.csv",
                "boundary": "no Paper Estrella or final Paper 4 promotion",
            },
        ]
    )
    coverage_actions = _actions(
        pool=pool,
        selected=coverage_selected,
        tier_id="coverage_only_v347_cvar",
    )

    write_csv(TABLE_DIR / "paper4_v349_v347_proxy_or_dual_bound_gate.csv", main)
    write_csv(TABLE_DIR / "paper4_v349_proxy_repair_pool_summary.csv", pool_summary)
    write_csv(TABLE_DIR / "paper4_v349_proxy_repair_tier_summary.csv", tier_summary)
    write_csv(TABLE_DIR / "paper4_v349_coverage_only_incumbent_actions.csv", coverage_actions)
    write_csv(TABLE_DIR / "paper4_v349_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v349_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = main.iloc[0]
    status = {
        "phase": "v349_v347_proxy_or_dual_bound_gate",
        "schema_version": "2026-05-17.349",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v349": BASE_VERSION,
        "reference_version_v349": REFERENCE_VERSION,
        "reprice_version_v349": REPRICE_VERSION,
        "pool_rows_v349": int(row[f"pool_rows_v{VERSION}"]),
        "observed_omitted_candidate_rows_v349": int(
            row[f"observed_omitted_candidate_rows_v{VERSION}"]
        ),
        "selected_observed_proxy_rows_v349": int(row[f"selected_observed_proxy_rows_v{VERSION}"]),
        "selected_missing_proxy_rows_v349": int(row[f"selected_missing_proxy_rows_v{VERSION}"]),
        "v338_missing_proxy_rows_v349": int(row[f"v338_missing_proxy_rows_v{VERSION}"]),
        "strict_v347_repair_feasible_v349": bool(row[f"strict_v347_repair_feasible_v{VERSION}"]),
        "relaxed_v338_return_repair_feasible_v349": bool(
            row[f"relaxed_v338_return_repair_feasible_v{VERSION}"]
        ),
        "coverage_only_solver_success_v349": bool(row[f"coverage_only_solver_success_v{VERSION}"]),
        "coverage_only_incumbent_found_v349": bool(
            row[f"coverage_only_incumbent_found_v{VERSION}"]
        ),
        "coverage_only_observed_proxy_rows_v349": int(
            row[f"coverage_only_observed_proxy_rows_v{VERSION}"]
        )
        if not pd.isna(row[f"coverage_only_observed_proxy_rows_v{VERSION}"])
        else None,
        "coverage_only_missing_proxy_rows_v349": int(
            row[f"coverage_only_missing_proxy_rows_v{VERSION}"]
        )
        if not pd.isna(row[f"coverage_only_missing_proxy_rows_v{VERSION}"])
        else None,
        "coverage_only_delta_return_vs_v347_v349": None
        if pd.isna(row[f"coverage_only_delta_return_vs_v347_v{VERSION}"])
        else float(row[f"coverage_only_delta_return_vs_v347_v{VERSION}"]),
        "coverage_only_delta_cvar90_vs_v347_v349": None
        if pd.isna(row[f"coverage_only_delta_cvar90_vs_v347_v{VERSION}"])
        else float(row[f"coverage_only_delta_cvar90_vs_v347_v{VERSION}"]),
        "coverage_only_return_collapse_flag_v349": bool(
            row[f"coverage_only_return_collapse_flag_v{VERSION}"]
        ),
        "coverage_only_action_rows_v349": int(len(coverage_actions)),
        "valid_branch_price_bound_v349": False,
        "full_universe_integer_optimality_claim_allowed_v349": False,
        "working_champion_claim_allowed_v349": False,
        "paper1_promotion_allowed_v349": False,
        "paper4_working_champion_changed_v349": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v349": int(len(blockers)),
        "claim_matrix_rows_v349": int(len(claim_matrix)),
        "next_artifact_v349": NEXT_ARTIFACT,
        "claim_boundary": (
            "v349 tests all observed omitted proxy repair for v347; strict/relaxed "
            "repair is infeasible and no global bound or promotion is authorized"
        ),
    }
    write_json(STATUS_DIR / "paper4_v349_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v349": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
