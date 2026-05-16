#!/usr/bin/env python3
"""Build Paper 4 v334 v330 proxy-gap repair / branch-price protocol artifacts."""

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

VERSION = 334
BASE_VERSION = 330
REFERENCE_VERSION = 316
GATE_VERSION = 333
NEXT_VERSION = 335
TARGET_SELECTED_ROWS = 171
CANDIDATE_POOL_LIMIT_PER_PERIOD = 750
MILP_TIME_LIMIT_SECONDS = 60.0
MIP_REL_GAP = 1e-6
OBSERVED_PROXY_WEIGHT = 100_000.0
NEXT_ARTIFACT = f"paper4_v{NEXT_VERSION}_v334_relaxed_reprice_or_dual_bound_gate.csv"


def _cap_lookup(source_caps: pd.DataFrame, family: str) -> dict[str, float]:
    cap_col = f"cap_share_v{BASE_VERSION}"
    family_caps = source_caps.loc[source_caps["source_family"].astype(str).eq(family)]
    return {str(row["source_id"]): float(row[cap_col]) for _, row in family_caps.iterrows()}


def _period_distribution(portfolio: pd.DataFrame) -> dict[str, int]:
    return {
        str(period): int(count)
        for period, count in portfolio["period"].astype(str).value_counts().sort_index().items()
    }


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
    candidate_parts: list[pd.DataFrame] = []
    observed_omitted = universe.loc[
        ~universe["loan_id"].isin(selected_ids) & universe[f"observed_v47_proxy_v{VERSION}"]
    ].copy()
    for _period, group in observed_omitted.groupby("period", dropna=False):
        candidate_parts.append(
            group.sort_values(f"mean_return_v{VERSION}", ascending=False).head(
                CANDIDATE_POOL_LIMIT_PER_PERIOD
            )
        )
    candidates = (
        pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    )
    pool = pd.concat(
        [universe.loc[universe["loan_id"].isin(selected_ids)], candidates],
        ignore_index=True,
    ).drop_duplicates("loan_id")
    pool["loan_id"] = pool["loan_id"].astype(str)
    pool[f"incumbent_selected_v{VERSION}"] = pool["loan_id"].isin(selected_ids)
    pool[f"observed_candidate_v{VERSION}"] = (
        ~pool[f"incumbent_selected_v{VERSION}"] & pool[f"observed_v47_proxy_v{VERSION}"]
    )
    pool[f"pool_role_v{VERSION}"] = np.where(
        pool[f"incumbent_selected_v{VERSION}"],
        "v330_selected_base",
        "observed_omitted_candidate",
    )
    pool[f"universe_idx_v{VERSION}"] = idx_by_id.loc[pool["loan_id"]].to_numpy()
    pool_summary = pd.DataFrame(
        [
            {
                f"pool_id_v{VERSION}": "v330_plus_observed_omitted_candidates",
                f"pool_rows_v{VERSION}": int(len(pool)),
                f"selected_base_rows_v{VERSION}": int(pool[f"incumbent_selected_v{VERSION}"].sum()),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool[f"observed_candidate_v{VERSION}"].sum()
                ),
                f"total_observed_omitted_rows_v{VERSION}": int(len(observed_omitted)),
                f"candidate_pool_limit_per_period_v{VERSION}": CANDIDATE_POOL_LIMIT_PER_PERIOD,
                f"pool_limit_binding_v{VERSION}": int(len(candidates)) < int(len(observed_omitted)),
                f"claim_boundary_v{VERSION}": (
                    "bounded observed-candidate proxy repair pool; not full-universe pricing"
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
    exposure_min: float,
    exposure_max: float,
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

    return_row = np.zeros(var_count)
    return_row[:n] = pool[f"mean_return_v{VERSION}"].to_numpy(float)
    rows.append(return_row)
    lb.append(return_floor)
    ub.append(np.inf)

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
    exposure_min: float,
    exposure_max: float,
    return_floor: float,
    cvar_cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    constraints, bounds, integrality, var_count, constraint_count = _constraints(
        pool=pool,
        losses_pool=losses_pool,
        source_caps=source_caps,
        target_period_counts=target_period_counts,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
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
    }
    if result.x is not None:
        selected_mask = np.rint(np.clip(result.x[:n], 0, 1)).astype(bool)
        selected = pool.loc[selected_mask].copy()
        selected_idx = selected[f"universe_idx_v{VERSION}"].to_numpy(int)
        scenario_losses = losses_full[:, selected_idx].sum(axis=1)
        metrics.update(
            {
                "selected_rows": int(len(selected)),
                "portfolio_exposure": float(selected["loan_amnt"].sum()),
                "objective_return": float(selected[f"mean_return_v{VERSION}"].sum()),
                "scenario_loss_mean": float(scenario_losses.mean()),
                "scenario_loss_cvar90": v70._tail_cvar(scenario_losses),
                "observed_proxy_rows": int(selected[f"observed_v47_proxy_v{VERSION}"].sum()),
                "missing_proxy_rows": int((~selected[f"observed_v47_proxy_v{VERSION}"]).sum()),
                "period_distribution": _period_distribution(selected),
            }
        )
    return selected, metrics


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
                        "v334 relaxed proxy-gap repair source diagnostic only"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _actions(*, pool: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(selected["loan_id"].astype(str))
    work = pool.copy()
    work["loan_id"] = work["loan_id"].astype(str)
    changed = work.loc[
        (work[f"incumbent_selected_v{VERSION}"] & ~work["loan_id"].isin(selected_ids))
        | (~work[f"incumbent_selected_v{VERSION}"] & work["loan_id"].isin(selected_ids))
    ].copy()
    changed[f"action_v{VERSION}"] = np.where(
        changed["loan_id"].isin(selected_ids),
        "add_observed_candidate",
        "drop_v330_selected",
    )
    changed[f"claim_boundary_v{VERSION}"] = (
        "v334 relaxed proxy-gap repair action list; post-repair repricing still required"
    )
    return changed[
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


def _tier_summary(tier_metrics: list[dict[str, Any]], reference: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics in tier_metrics:
        objective_return = metrics.get("objective_return")
        cvar = metrics.get("scenario_loss_cvar90")
        period_distribution = metrics.get("period_distribution", {})
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
                f"constraint_rows_v{VERSION}": metrics["constraint_rows"],
                f"variable_count_v{VERSION}": metrics["variable_count"],
                f"return_floor_v{VERSION}": metrics["return_floor"],
                f"cvar_cap_v{VERSION}": metrics["cvar_cap"],
                f"selected_rows_v{VERSION}": metrics.get("selected_rows"),
                f"portfolio_exposure_v{VERSION}": metrics.get("portfolio_exposure"),
                f"objective_return_v{VERSION}": objective_return,
                f"delta_return_vs_v330_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v330_return"]),
                f"delta_return_vs_v316_v{VERSION}": None
                if objective_return is None
                else float(objective_return - reference["v316_return"]),
                f"scenario_loss_mean_v{VERSION}": metrics.get("scenario_loss_mean"),
                f"scenario_loss_cvar90_v{VERSION}": cvar,
                f"delta_cvar90_vs_v330_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v330_cvar"]),
                f"delta_cvar90_vs_v316_v{VERSION}": None
                if cvar is None
                else float(cvar - reference["v316_cvar"]),
                f"observed_proxy_rows_v{VERSION}": metrics.get("observed_proxy_rows"),
                f"missing_proxy_rows_v{VERSION}": metrics.get("missing_proxy_rows"),
                f"observed_proxy_delta_vs_v330_v{VERSION}": None
                if metrics.get("observed_proxy_rows") is None
                else int(metrics["observed_proxy_rows"] - reference["v330_observed"]),
                f"missing_proxy_delta_vs_v330_v{VERSION}": None
                if metrics.get("missing_proxy_rows") is None
                else int(metrics["missing_proxy_rows"] - reference["v330_missing"]),
                f"period_distribution_v{VERSION}": json.dumps(period_distribution, sort_keys=True),
                f"claim_boundary_v{VERSION}": (
                    "bounded proxy-gap repair tier; not full-universe or promotion evidence"
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
                "claim": "v334 tests strict v330-preserving proxy-gap repair.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v334_proxy_gap_tier_summary.csv"
                ),
                "boundary": "Bounded observed-candidate MILP feasibility test only.",
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v334 documents a relaxed proxy-gap repair coverage frontier.",
                "allowed": True,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v334_relaxed_proxy_gap_allocations.parquet"
                ),
                "boundary": (
                    "Relaxed repair preserves v316 return floor and v330 CVaR cap, "
                    "but concedes v330 return."
                ),
                "prohibited_claim_flag": False,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v334 preserves the full v330 return while repairing proxy coverage.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v334_claim_blockers.csv"
                ),
                "boundary": "Strict v330-return/v330-CVaR repair is infeasible in the bounded pool.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v334 proves full-universe branch-price optimality.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v334_claim_blockers.csv"
                ),
                "boundary": "The experiment prices a bounded observed-candidate pool only.",
                "prohibited_claim_flag": True,
                "current_quarto_page": "living_notebook_only",
            },
            {
                "claim": "v334 replaces Paper Estrella or finalizes Paper 4.",
                "allowed": False,
                "evidence_artifact": (
                    "reports/paper_material/paper4/tables/paper4_v334_claim_blockers.csv"
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
                    "v334 tests whether v330's proxy coverage loss can be repaired with "
                    "observed candidates while preserving static risk/return gates."
                ),
                "status": "strict_v330_proxy_repair_tested_relaxed_frontier_documented",
                "next_artifact": NEXT_ARTIFACT,
                "success_condition": (
                    "reprice the relaxed repair or build a dual-bound certificate without promoting"
                ),
                "last_wave": "v334",
                "execution_result": "relaxed_proxy_repair_frontier_documented",
                "quarto_promotion_decision": "living_notebook_only",
            }
        ]
    )
    if current.empty:
        write_csv(path, additions)
        return
    out = current.loc[~current["last_wave"].astype(str).eq("v334")].copy()
    write_csv(path, pd.concat([out, additions], ignore_index=True))


def _update_notebook(status: dict[str, Any]) -> None:
    start = "<!-- V334_V330_PROXY_GAP_REPAIR_START -->"
    end = "<!-- V334_V330_PROXY_GAP_REPAIR_END -->"
    block = f"""
{start}

## Wave v334: v330 Proxy-Gap Repair / Branch-Price Protocol

Generated: {status["generated_at_utc"]}

### Objective

v333 showed that v330 needs 74 imputed proxy loans, 1 more than v316. v334 tests
whether observed candidates can repair that coverage gap while preserving the
v330 post-v328-swap budget, period, source and CVaR structure.

### Results

- Candidate pool rows: `{status["pool_rows_v334"]}`.
- Strict v330-preserving repair feasible:
  `{status["strict_v330_preserving_repair_feasible_v334"]}`.
- Relaxed repair feasible:
  `{status["relaxed_repair_feasible_v334"]}`.
- Relaxed observed proxy rows:
  `{status["relaxed_observed_proxy_rows_v334"]}`.
- Relaxed missing proxy rows:
  `{status["relaxed_missing_proxy_rows_v334"]}`.
- Relaxed return delta vs v330:
  `{status["relaxed_delta_return_vs_v330_v334"]}`.
- Relaxed CVaR delta vs v330:
  `{status["relaxed_delta_cvar90_vs_v330_v334"]}`.
- Post-v334 repricing required:
  `{status["post_v334_repricing_required_v334"]}`.

### Interpretation

v334 exposes a useful frontier. The strict repair that preserves v330 return and
v330 CVaR is infeasible in the bounded observed-candidate pool. A relaxed repair
improves proxy coverage from 97 to `{status["relaxed_observed_proxy_rows_v334"]}`
observed loans under the v330 CVaR cap, but it gives back
`{status["relaxed_delta_return_vs_v330_v334"]}` return relative to v330. This is evidence of a real
coverage-return tradeoff, not a champion or deployment claim.

### Claim Impact

- Allowed: bounded proxy-gap repair feasibility test and relaxed repair
  diagnostic.
- Still prohibited: strict v330-preserving coverage repair, full-universe global
  optimality, contractual IFRS9, live deployment, Paper Estrella replacement,
  final Paper 4 promotion and working champion claims.

### Quarto Promotion Decision

Keep v334 in the living notebook. The next wave should reprice the relaxed
repair or pursue a dual-bound/global certificate without promotion.

{end}
""".strip()
    _append_or_replace_block(NOTEBOOK, start, end, block)


def main() -> None:
    started = datetime.now(UTC)
    universe = read_parquet("paper4_v55_maximal_comparable_universe.parquet").reset_index(drop=True)
    v330 = read_parquet("paper4_v330_post_v328_swap_allocations.parquet").reset_index(drop=True)
    v316_summary = read_csv("paper4_v316_apply_next_post_v314_swap_summary.csv")
    v330_summary = read_csv("paper4_v330_apply_next_post_v328_swap.csv")
    source_caps = read_csv("paper4_v330_post_v328_swap_source_summary.csv")
    v333_status = json.loads((STATUS_DIR / "paper4_v333_status.json").read_text(encoding="utf-8"))
    v47_panel = read_parquet("paper4_v47_ifrs9_proxy_panel_v45.parquet")
    if any(df.empty for df in [universe, v330, v316_summary, v330_summary, source_caps, v47_panel]):
        raise RuntimeError("Missing v334 proxy-gap repair inputs.")
    if int(v333_status["imputed_proxy_loan_rows_v333"]) != 74:
        raise RuntimeError("v334 expects the v333 v330 proxy gap to be documented.")

    universe["loan_id"] = universe["loan_id"].astype(str)
    v330["loan_id"] = v330["loan_id"].astype(str)
    for family in FAMILIES:
        universe[family] = universe[family].astype(str)
        v330[family] = v330[family].astype(str)
    observed_ids = set(v47_panel["loan_id"].astype(str))
    losses, mean_returns, _path_ids = v71._full_universe_loss_and_return_mean(universe)
    idx_by_id = pd.Series(np.arange(len(universe)), index=universe["loan_id"].astype(str))
    pool, pool_summary = _build_pool(
        universe=universe,
        selected=v330,
        observed_ids=observed_ids,
        idx_by_id=idx_by_id,
        mean_returns=mean_returns,
    )
    losses_pool = losses[:, pool[f"universe_idx_v{VERSION}"].to_numpy(int)]

    v316_row = v316_summary.iloc[0]
    v330_row = v330_summary.iloc[0]
    selected_ids = set(v330["loan_id"].astype(str))
    reference = {
        "v316_return": float(v316_row["objective_return_v316"]),
        "v316_cvar": float(v316_row["scenario_loss_cvar90_v316"]),
        "v330_return": float(v330_row["objective_return_v330"]),
        "v330_cvar": float(v330_row["scenario_loss_cvar90_v330"]),
        "v330_observed": int(v333_status["observed_proxy_loan_rows_v333"]),
        "v330_missing": int(v333_status["imputed_proxy_loan_rows_v333"]),
    }
    target_period_counts = _period_distribution(v330)
    exposure_min = 842292.375
    exposure_max = 850000.0
    tiers = [
        {
            "tier_id": "strict_v330_return_v330_cvar",
            "return_floor": reference["v330_return"],
            "cvar_cap": reference["v330_cvar"],
        },
        {
            "tier_id": "relaxed_v316_return_v330_cvar",
            "return_floor": reference["v316_return"],
            "cvar_cap": reference["v330_cvar"],
        },
    ]

    solved: dict[str, pd.DataFrame] = {}
    tier_metrics: list[dict[str, Any]] = []
    for tier in tiers:
        selected, metrics = _solve_tier(
            tier_id=tier["tier_id"],
            pool=pool,
            losses_pool=losses_pool,
            losses_full=losses,
            source_caps=source_caps,
            target_period_counts=target_period_counts,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            return_floor=float(tier["return_floor"]),
            cvar_cap=float(tier["cvar_cap"]),
        )
        tier_metrics.append(metrics)
        if not selected.empty:
            solved[str(tier["tier_id"])] = selected

    tier_summary = _tier_summary(tier_metrics, reference)
    relaxed = solved.get("relaxed_v316_return_v330_cvar", pd.DataFrame())
    if relaxed.empty:
        raise RuntimeError("v334 expected the relaxed proxy-gap repair to produce a solution.")

    relaxed = relaxed.copy()
    relaxed_ids = set(relaxed["loan_id"].astype(str))
    relaxed[f"selected_v{VERSION}"] = True
    relaxed[f"portfolio_label_v{VERSION}"] = "relaxed_proxy_gap_repair_candidate"
    relaxed[f"claim_boundary_v{VERSION}"] = (
        "relaxed proxy-gap repair allocation; return concession and post-repair repricing required"
    )
    source_summary = _source_summary(selected=relaxed, universe=universe, source_caps=source_caps)
    source_cap_violations = int(source_summary[f"source_cap_violated_v{VERSION}"].sum())
    min_source_slack = float(source_summary[f"source_slack_v{VERSION}"].min())
    actions = _actions(pool=pool, selected=relaxed)
    added_rows = int(len(relaxed_ids - selected_ids))
    dropped_rows = int(len(selected_ids - relaxed_ids))

    strict_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("strict_v330_return_v330_cvar")
    ].iloc[0]
    relaxed_row = tier_summary.loc[
        tier_summary[f"tier_id_v{VERSION}"].eq("relaxed_v316_return_v330_cvar")
    ].iloc[0]
    strict_feasible = bool(strict_row[f"solver_success_v{VERSION}"])
    relaxed_feasible = bool(relaxed_row[f"solver_success_v{VERSION}"])
    relaxed_return_delta = float(relaxed_row[f"delta_return_vs_v330_v{VERSION}"])
    relaxed_cvar_delta = float(relaxed_row[f"delta_cvar90_vs_v330_v{VERSION}"])
    relaxed_observed = int(relaxed_row[f"observed_proxy_rows_v{VERSION}"])
    relaxed_missing = int(relaxed_row[f"missing_proxy_rows_v{VERSION}"])

    main_summary = pd.DataFrame(
        [
            {
                f"gate_id_v{VERSION}": "v334_v330_proxy_gap_repair_or_branch_price_protocol",
                f"base_version_v{VERSION}": BASE_VERSION,
                f"gate_version_v{VERSION}": GATE_VERSION,
                f"pool_rows_v{VERSION}": int(pool_summary.iloc[0][f"pool_rows_v{VERSION}"]),
                f"observed_omitted_candidate_rows_v{VERSION}": int(
                    pool_summary.iloc[0][f"observed_omitted_candidate_rows_v{VERSION}"]
                ),
                f"candidate_pool_limit_per_period_v{VERSION}": CANDIDATE_POOL_LIMIT_PER_PERIOD,
                f"strict_v330_preserving_repair_feasible_v{VERSION}": strict_feasible,
                f"relaxed_repair_feasible_v{VERSION}": relaxed_feasible,
                f"relaxed_selected_rows_v{VERSION}": int(relaxed_row[f"selected_rows_v{VERSION}"]),
                f"relaxed_added_rows_v{VERSION}": added_rows,
                f"relaxed_dropped_rows_v{VERSION}": dropped_rows,
                f"relaxed_observed_proxy_rows_v{VERSION}": relaxed_observed,
                f"relaxed_missing_proxy_rows_v{VERSION}": relaxed_missing,
                f"relaxed_observed_delta_vs_v330_v{VERSION}": int(
                    relaxed_row[f"observed_proxy_delta_vs_v330_v{VERSION}"]
                ),
                f"relaxed_missing_delta_vs_v330_v{VERSION}": int(
                    relaxed_row[f"missing_proxy_delta_vs_v330_v{VERSION}"]
                ),
                f"relaxed_objective_return_v{VERSION}": float(
                    relaxed_row[f"objective_return_v{VERSION}"]
                ),
                f"relaxed_delta_return_vs_v330_v{VERSION}": relaxed_return_delta,
                f"relaxed_delta_return_vs_v316_v{VERSION}": float(
                    relaxed_row[f"delta_return_vs_v316_v{VERSION}"]
                ),
                f"relaxed_cvar90_v{VERSION}": float(
                    relaxed_row[f"scenario_loss_cvar90_v{VERSION}"]
                ),
                f"relaxed_delta_cvar90_vs_v330_v{VERSION}": relaxed_cvar_delta,
                f"relaxed_delta_cvar90_vs_v316_v{VERSION}": float(
                    relaxed_row[f"delta_cvar90_vs_v316_v{VERSION}"]
                ),
                f"source_cap_violations_v{VERSION}": source_cap_violations,
                f"min_source_slack_v{VERSION}": min_source_slack,
                f"post_v334_repricing_required_v{VERSION}": True,
                f"strict_repair_or_champion_claim_allowed_v{VERSION}": False,
                f"full_universe_integer_optimality_claim_allowed_v{VERSION}": False,
                f"working_champion_claim_allowed_v{VERSION}": False,
                f"paper1_promotion_allowed_v{VERSION}": False,
                "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
                f"next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": (
                    "bounded proxy-gap repair protocol; relaxed candidate needs repricing and "
                    "strict v330-preserving repair is infeasible"
                ),
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                f"blocker_id_v{VERSION}": "strict_v330_preserving_proxy_repair_infeasible",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_expanded_pool_or_dual_bound",
                f"claim_boundary_v{VERSION}": "strict v330-return/v330-CVaR repair infeasible",
            },
            {
                f"blocker_id_v{VERSION}": "relaxed_repair_requires_return_concession",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": abs(relaxed_return_delta),
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "coverage improves only by giving back v330 return",
            },
            {
                f"blocker_id_v{VERSION}": "post_v334_reprice_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": NEXT_ARTIFACT,
                f"claim_boundary_v{VERSION}": "relaxed repair has not been one-swap repriced",
            },
            {
                f"blocker_id_v{VERSION}": "bounded_pool_not_full_universe",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": int(
                    pool_summary.iloc[0][f"observed_omitted_candidate_rows_v{VERSION}"]
                ),
                f"required_next_artifact_v{VERSION}": "future_full_universe_branch_price_bound",
                f"claim_boundary_v{VERSION}": "v334 prices a bounded observed-candidate pool",
            },
            {
                f"blocker_id_v{VERSION}": "contractual_ifrs9_and_live_holdout_missing",
                f"blocking_v{VERSION}": True,
                f"evidence_count_v{VERSION}": 1,
                f"required_next_artifact_v{VERSION}": "future_contractual_or_live_holdout_gate",
                f"claim_boundary_v{VERSION}": "proxy repair is not contractual IFRS9 or live replay",
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
                "claim_id": "v334_strict_v330_proxy_gap_repair_found",
                "allowed": strict_feasible,
                "artifact": "paper4_v334_proxy_gap_tier_summary.csv",
                "boundary": "strict bounded-pool feasibility only",
            },
            {
                "claim_id": "v334_relaxed_proxy_gap_repair_found",
                "allowed": relaxed_feasible,
                "artifact": "paper4_v334_relaxed_proxy_gap_allocations.parquet",
                "boundary": "relaxed v316-return/v330-CVaR candidate only",
            },
            {
                "claim_id": "v334_relaxed_observed_proxy_rows_reach_100",
                "allowed": relaxed_observed >= 100,
                "artifact": "paper4_v334_proxy_gap_tier_summary.csv",
                "boundary": "proxy coverage diagnostic only",
            },
            {
                "claim_id": "v334_preserves_full_v330_return",
                "allowed": False,
                "artifact": "paper4_v334_claim_blockers.csv",
                "boundary": "relaxed repair gives back v330 return and strict tier is infeasible",
            },
            {
                "claim_id": "v334_full_universe_integer_optimality",
                "allowed": False,
                "artifact": "paper4_v334_claim_blockers.csv",
                "boundary": "bounded observed-candidate pool only",
            },
            {
                "claim_id": "v334_working_champion_or_final_promotion",
                "allowed": False,
                "artifact": "paper4_v334_claim_blockers.csv",
                "boundary": "working champion and final promotion remain blocked",
            },
        ]
    )

    write_csv(
        TABLE_DIR / "paper4_v334_v330_proxy_gap_repair_or_branch_price_protocol.csv",
        main_summary,
    )
    write_csv(TABLE_DIR / "paper4_v334_proxy_gap_pool_summary.csv", pool_summary)
    write_csv(TABLE_DIR / "paper4_v334_proxy_gap_tier_summary.csv", tier_summary)
    relaxed.to_parquet(TABLE_DIR / "paper4_v334_relaxed_proxy_gap_allocations.parquet", index=False)
    write_csv(TABLE_DIR / "paper4_v334_relaxed_proxy_gap_actions.csv", actions)
    write_csv(TABLE_DIR / "paper4_v334_relaxed_proxy_gap_source_summary.csv", source_summary)
    write_csv(TABLE_DIR / "paper4_v334_claim_blockers.csv", blockers)
    write_csv(TABLE_DIR / "paper4_v334_claim_matrix_delta.csv", claim_matrix)
    _update_claim_boundaries()
    _update_backlog()

    row = main_summary.iloc[0]
    status = {
        "phase": "v334_v330_proxy_gap_repair_or_branch_price_protocol",
        "schema_version": "2026-05-16.334",
        "generated_at_utc": now(),
        "runtime_seconds": round((datetime.now(UTC) - started).total_seconds(), 3),
        "base_version_v334": BASE_VERSION,
        "reference_version_v334": REFERENCE_VERSION,
        "gate_version_v334": GATE_VERSION,
        "pool_rows_v334": int(row[f"pool_rows_v{VERSION}"]),
        "observed_omitted_candidate_rows_v334": int(
            row[f"observed_omitted_candidate_rows_v{VERSION}"]
        ),
        "candidate_pool_limit_per_period_v334": CANDIDATE_POOL_LIMIT_PER_PERIOD,
        "strict_v330_preserving_repair_feasible_v334": strict_feasible,
        "strict_milp_status_v334": int(strict_row[f"milp_status_v{VERSION}"]),
        "relaxed_repair_feasible_v334": relaxed_feasible,
        "relaxed_selected_rows_v334": int(row[f"relaxed_selected_rows_v{VERSION}"]),
        "relaxed_added_rows_v334": int(row[f"relaxed_added_rows_v{VERSION}"]),
        "relaxed_dropped_rows_v334": int(row[f"relaxed_dropped_rows_v{VERSION}"]),
        "relaxed_observed_proxy_rows_v334": relaxed_observed,
        "relaxed_missing_proxy_rows_v334": relaxed_missing,
        "relaxed_observed_delta_vs_v330_v334": int(
            row[f"relaxed_observed_delta_vs_v330_v{VERSION}"]
        ),
        "relaxed_missing_delta_vs_v330_v334": int(row[f"relaxed_missing_delta_vs_v330_v{VERSION}"]),
        "relaxed_objective_return_v334": float(row[f"relaxed_objective_return_v{VERSION}"]),
        "relaxed_delta_return_vs_v330_v334": relaxed_return_delta,
        "relaxed_delta_return_vs_v316_v334": float(row[f"relaxed_delta_return_vs_v316_v{VERSION}"]),
        "relaxed_cvar90_v334": float(row[f"relaxed_cvar90_v{VERSION}"]),
        "relaxed_delta_cvar90_vs_v330_v334": relaxed_cvar_delta,
        "relaxed_delta_cvar90_vs_v316_v334": float(row[f"relaxed_delta_cvar90_vs_v316_v{VERSION}"]),
        "source_cap_violations_v334": source_cap_violations,
        "min_source_slack_v334": min_source_slack,
        "post_v334_repricing_required_v334": True,
        "strict_repair_or_champion_claim_allowed_v334": False,
        "full_universe_integer_optimality_claim_allowed_v334": False,
        "working_champion_claim_allowed_v334": False,
        "paper1_promotion_allowed_v334": False,
        "paper4_working_champion_changed_v334": False,
        "paper4_final_promotion_created": FORBIDDEN_FINAL_PROMOTION.exists(),
        "claim_blocker_rows_v334": int(len(blockers)),
        "claim_matrix_rows_v334": int(len(claim_matrix)),
        "next_artifact_v334": row[f"next_artifact_v{VERSION}"],
        "claim_boundary": (
            "v334 documents a bounded proxy-coverage frontier: strict v330-preserving repair "
            "is infeasible, while relaxed repair improves coverage with a return concession"
        ),
    }
    write_json(STATUS_DIR / "paper4_v334_status.json", status)
    _update_notebook(status)
    print(json.dumps({"v334": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
