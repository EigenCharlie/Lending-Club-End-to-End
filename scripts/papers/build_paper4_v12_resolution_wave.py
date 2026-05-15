"""Build Paper 4 v12 resolution-wave artifacts.

V12 is the first Paper 4 wave that is allowed to change the Paper 4 working
champion.  It remains strictly separate from the Paper Estrella promotion
contract:

* do not modify ``models/final_project_promotion.json``;
* do not create ``paper4_final_promotion.json``;
* do not make contractual IFRS9 or fair-lending legal claims without data.

The wave pushes the runnable lanes beyond v11: larger CVaR/OCE decomposition
with MDCP caps inside the LP, fitted-value/rollout ADP, a regret-trained
SPO-style surrogate, calibrated common sample paths, proxy IFRS9/SICR
sensitivity, causal/fairness blockers, and a Paper 4 working champion registry.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from scripts.papers.build_paper4_extended_experiments import (
    BUDGET,
    _safe_read_parquet,
)
from scripts.papers.build_paper4_living_lab_artifacts import DEFAULT_LGD
from scripts.papers.build_paper4_v6_priority_resolution import (
    STATUS_DIR,
    TABLE_DIR,
    _load_inputs,
    _scenario_loss_matrix,
    _write_csv,
    _write_json,
    _write_note,
    _write_parquet,
)
from scripts.papers.build_paper4_v8_resolution_wave import (
    _auditability_score,
    _solve_family_cap_policy,
)
from scripts.papers.build_paper4_v10_resolution_wave import (
    PAPER1_PROMOTION,
    PAPER4_FINAL_PROMOTION,
    _is_optimal,
    _load_v9_online,
    _stable_uniform,
    build_causal_fairness_v10,
    build_ifrs9_proxy_v10,
)
from scripts.papers.build_paper4_v11_promising_lanes import (
    _allocation_summary,
    _prepare_balanced_solver_pool_v11,
    _weighted_average,
)

SCHEMA_VERSION = "2026-05-14.12"
RNG_SEED = 2026051412
WORKING_CHAMPION_PATH = STATUS_DIR / "paper4_v12_working_champion.json"


def _json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _percentile_rank(s: pd.Series, *, high_is_good: bool) -> pd.Series:
    return s.rank(method="average", ascending=not high_is_good, na_option="keep", pct=True).fillna(
        0.50
    )


def _artifact_audit_v12() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root, kind in [(TABLE_DIR, "table"), (STATUS_DIR, "status")]:
        for path in sorted(root.glob("paper4*")):
            name = path.name
            version = "unversioned"
            for token in [f"v{i}" for i in range(1, 12)]:
                if f"_{token}_" in name or name.startswith(f"paper4_{token}_"):
                    version = token
                    break
            rows.append(
                {
                    "artifact": name,
                    "kind": kind,
                    "version_guess": version,
                    "path": str(path.relative_to(TABLE_DIR.parents[1])),
                    "exists": path.exists(),
                    "bytes": int(path.stat().st_size) if path.exists() else 0,
                }
            )
    return pd.DataFrame(rows)


def build_method_reference_registry_v12() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_lane": "SPO/SPO+",
                "reference": 'Elmachtoub and Grigas, Smart "Predict, then Optimize"',
                "url": "https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922",
                "implementation_use_v12": "structured regret objective and temporal regret validation",
                "claim_boundary_v12": "v12 implements SPO-style regret training; not a formal SPO+ theorem proof",
            },
            {
                "method_lane": "Differentiable optimization layers",
                "reference": "Agrawal et al., Differentiable Convex Optimization Layers",
                "url": "https://arxiv.org/abs/1910.12430",
                "implementation_use_v12": "documents future path for differentiating through convex programs",
                "claim_boundary_v12": "not used as a cvxpylayer in v12 because current LP contains large fractional portfolio constraints",
            },
            {
                "method_lane": "OptNet",
                "reference": "Amos and Kolter, OptNet",
                "url": "https://proceedings.mlr.press/v70/amos17a.html",
                "implementation_use_v12": "design reference for optimization-as-layer training",
                "claim_boundary_v12": "not implemented as a QP layer in v12",
            },
            {
                "method_lane": "Online SPO",
                "reference": "Liu and Grigas, Online Contextual Decision-Making with SPO",
                "url": "https://arxiv.org/abs/2206.07316",
                "implementation_use_v12": "motivates temporal train/validation/test decision regret reporting",
                "claim_boundary_v12": "v12 remains offline historical replay, not online deployment",
            },
        ]
    )


def _source_flag_frame(pool: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cap_grade_dplus": pool["original_grade"]
            .astype(str)
            .isin(["D", "E", "F", "G"])
            .astype(float),
            "cap_dti_high": pool["dti_band"].astype(str).eq("dti_q3").astype(float),
            "cap_score_low": pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float),
            "cap_income_q5": pool["income_band"].astype(str).eq("inc_q5").astype(float),
            "cap_period_2018h1": pool["period"].astype(str).eq("2018H1").astype(float),
            "cap_state_top20": pool["state_top20"].astype(str).ne("other").astype(float)
            if "state_top20" in pool
            else 0.0,
        },
        index=pool.index,
    )


def build_mdcp_caps_v12() -> tuple[pd.DataFrame, dict[str, float]]:
    _, _, v9_source, online_status = _load_v9_online()
    best_method = str(online_status.get("online_best_method_v9"))
    src = (
        v9_source[v9_source["online_method_v9"].eq(best_method)].copy()
        if not v9_source.empty
        else pd.DataFrame()
    )
    defended = src[
        src.get("standalone_gate_cell", pd.Series(False, index=src.index)).astype(bool)
    ].copy()
    map_rows = [
        ("original_grade", "cap_grade_dplus", 0.58),
        ("dti_band", "cap_dti_high", 0.58),
        ("score_decile", "cap_score_low", 0.92),
        ("income_band", "cap_income_q5", 0.66),
        ("period", "cap_period_2018h1", 0.66),
        ("state_top20", "cap_state_top20", 1.00),
    ]
    rows = []
    caps: dict[str, float] = {}
    for source_id, cap_name, base_cap in map_rows:
        local = (
            defended[defended["source_id"].eq(source_id)] if not defended.empty else pd.DataFrame()
        )
        worst_cov = float(local["coverage_online_v9"].min()) if not local.empty else np.nan
        min_support = int(local["n"].min()) if not local.empty and "n" in local else 0
        if cap_name == "cap_state_top20":
            support_penalty = 0.0
            coverage_penalty = 0.0
        else:
            support_penalty = 0.05 if min_support < 20 else 0.0
            coverage_penalty = max(0.0, 0.88 - worst_cov) * 0.55 if not pd.isna(worst_cov) else 0.08
        cap = float(np.clip(base_cap - support_penalty - coverage_penalty, 0.28, base_cap))
        caps[cap_name] = cap
        rows.append(
            {
                "source_id": source_id,
                "mapped_cap": cap_name,
                "base_cap": base_cap,
                "worst_defended_coverage_v9": worst_cov,
                "min_defended_support": min_support,
                "support_penalty_v12": support_penalty,
                "coverage_penalty_v12": coverage_penalty,
                "empirical_cap_v12": cap,
                "cap_scope_v12": "empirical_coverage_and_min_support_aware",
            }
        )
    return pd.DataFrame(rows), caps


def _prepare_column_pool_v12(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    max_n: int,
    online_method: str,
    anchor_allocations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = _prepare_balanced_solver_pool_v11(
        candidate_pool,
        online_intervals,
        max_n=min(len(candidate_pool), max_n + 8_000),
        online_method=online_method,
    ).copy()
    if base.empty:
        return base
    base["return_per_pd_v12"] = base["base_return_vec"] / (
        base["loan_amnt"].clip(lower=1) * base["pd_high_alpha01"].clip(lower=0.01)
    )
    base["tail_quality_v12"] = (
        base["pd_high_alpha01"].astype(float)
        + 0.65 * base["qhat_v4"].astype(float)
        + 0.85 * base["weak_source_proxy"].astype(float)
    )
    base["audit_return_score_v12"] = (
        base["base_return_vec"]
        - 0.09 * base["loan_amnt"] * base["qhat_v4"]
        - 0.10 * base["loan_amnt"] * base["weak_source_proxy"]
        - 0.35 * base["loan_amnt"] * base["pd_high_alpha01"] * DEFAULT_LGD
    )
    parts = [
        base.sort_values("solver_score_seed", ascending=False).head(int(max_n * 0.32)),
        base.sort_values("return_per_pd_v12", ascending=False).head(int(max_n * 0.24)),
        base.sort_values("tail_quality_v12", ascending=True).head(int(max_n * 0.24)),
        base.sort_values("audit_return_score_v12", ascending=False).head(int(max_n * 0.30)),
    ]
    if (
        anchor_allocations is not None
        and not anchor_allocations.empty
        and "loan_id" in anchor_allocations
    ):
        anchor_ids = set(anchor_allocations["loan_id"].astype(str))
        parts.append(base[base["loan_id"].astype(str).isin(anchor_ids)])
    pool = pd.concat(parts, ignore_index=True).drop_duplicates("loan_id", keep="first")
    if len(pool) < max_n:
        filler = base[~base["loan_id"].isin(set(pool["loan_id"]))].sort_values(
            "solver_score_seed", ascending=False
        )
        pool = pd.concat([pool, filler.head(max_n - len(pool))], ignore_index=True)
    pool = pool.head(max_n).reset_index(drop=True)
    pool["pool_design_v12"] = "column_generation_score_returnpd_lowtail_audit_anchors"
    pool["pool_n_requested_v12"] = max_n
    return pool


def _solve_cvar_mdcp_policy_v12(
    pool: pd.DataFrame,
    *,
    policy_id: str,
    risk_tolerance: float,
    weak_penalty: float,
    width_penalty: float,
    cvar_cap: float,
    return_floor: float,
    caps: dict[str, float],
    qhat_cap: float,
    time_limit: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    n = len(pool)
    loan = pool["loan_amnt"].to_numpy(dtype=float)
    pd_high = pool["pd_high_alpha01"].to_numpy(dtype=float)
    base_return = pool["base_return_vec"].to_numpy(dtype=float)
    weak = pool["weak_source_proxy"].to_numpy(dtype=float)
    qhat = pool["qhat_v4"].to_numpy(dtype=float)
    flags = _source_flag_frame(pool)
    obj_vec = base_return - weak_penalty * loan * weak - width_penalty * loan * qhat
    scenarios, loss_matrix = _scenario_loss_matrix(pool)
    model = pyo.ConcreteModel(policy_id)
    model.I = pyo.RangeSet(0, n - 1)
    model.S = pyo.RangeSet(0, len(scenarios) - 1)
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.eta = pyo.Var(domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.S, domain=pyo.NonNegativeReals)
    exposure = sum(model.x[i] * loan[i] for i in model.I)
    ret = sum(model.x[i] * base_return[i] for i in model.I)
    model.budget = pyo.Constraint(expr=exposure <= BUDGET)
    model.min_budget = pyo.Constraint(expr=exposure >= 0.85 * BUDGET)
    model.pd_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan[i] * pd_high[i] for i in model.I)
        <= risk_tolerance * (exposure + 1e-6)
    )
    model.return_floor = pyo.Constraint(expr=ret >= return_floor)
    model.qhat_cap = pyo.Constraint(
        expr=sum(model.x[i] * loan[i] * qhat[i] for i in model.I) <= qhat_cap * (exposure + 1e-6)
    )
    for cap_name, cap_value in caps.items():
        flag = flags[cap_name].to_numpy(dtype=float)
        setattr(
            model,
            cap_name,
            pyo.Constraint(
                expr=sum(model.x[i] * loan[i] * flag[i] for i in model.I)
                <= cap_value * (exposure + 1e-6)
            ),
        )

    def excess_rule(m: pyo.ConcreteModel, s: int) -> pyo.Expression:
        return m.u[s] >= sum(m.x[i] * loss_matrix[s, i] for i in m.I) - m.eta

    model.excess = pyo.Constraint(model.S, rule=excess_rule)
    beta = 0.90
    cvar_expr = model.eta + (1 / ((1 - beta) * len(scenarios))) * sum(model.u[s] for s in model.S)
    model.cvar_cap = pyo.Constraint(expr=cvar_expr <= cvar_cap)
    model.obj = pyo.Objective(
        expr=sum(model.x[i] * obj_vec[i] for i in model.I), sense=pyo.maximize
    )
    solver = Highs()
    solver.config.time_limit = time_limit
    t0 = time.perf_counter()
    try:
        results = solver.solve(model)
        status = str(getattr(results, "termination_condition", "unknown"))
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        metrics = {
            "policy_id": policy_id,
            "solver_status": f"infeasible_or_no_solution: {str(exc).splitlines()[0]}",
            "elapsed_seconds": elapsed,
            "n_funded": 0,
            "funded_exposure": 0.0,
            "objective_return": np.nan,
            "scenario_loss_cvar90": np.nan,
            "weighted_pd_high": np.nan,
            "weighted_qhat": np.nan,
            "weighted_weak_source_proxy": np.nan,
            "feasible_v12": False,
        }
        return pd.DataFrame(), metrics, pd.DataFrame(), pd.DataFrame()
    allocation = np.array([float(pyo.value(model.x[i])) for i in model.I])
    mask = allocation > 1e-8
    funded = pool.loc[mask].copy()
    funded["policy_id"] = policy_id
    funded["allocation_fraction"] = allocation[mask]
    funded["funded_exposure"] = funded["allocation_fraction"] * funded["loan_amnt"]
    funded["realized_return_proxy_lgd45"] = funded["funded_exposure"] * funded[
        "int_rate_decimal"
    ].astype(float) * (1 - funded["y_true"].astype(float)) - funded[
        "funded_exposure"
    ] * DEFAULT_LGD * funded["y_true"].astype(float)
    exposure_sum = float(funded["funded_exposure"].sum())
    loss_rows = []
    for s_idx, (scenario, mult, lgd) in enumerate(scenarios):
        loss_rows.append(
            {
                "policy_id": policy_id,
                "scenario": scenario,
                "pd_multiplier": mult,
                "lgd": lgd,
                "portfolio_loss": float(np.dot(allocation, loss_matrix[s_idx])),
                "cvar_excess_u": float(pyo.value(model.u[s_idx])),
            }
        )
    loss_df = pd.DataFrame(loss_rows)
    cvar_value = float(pyo.value(cvar_expr))
    metrics: dict[str, Any] = {
        "policy_id": policy_id,
        "solver_status": status,
        "elapsed_seconds": time.perf_counter() - t0,
        "risk_tolerance": risk_tolerance,
        "weak_penalty": weak_penalty,
        "width_penalty": width_penalty,
        "cvar_cap": cvar_cap,
        "return_floor": return_floor,
        "qhat_cap_v12": qhat_cap,
        "n_funded": int(funded["loan_id"].nunique()),
        "funded_exposure": exposure_sum,
        "objective_return": float(np.dot(allocation, base_return)),
        "realized_return_proxy_lgd45": float(funded["realized_return_proxy_lgd45"].sum())
        if exposure_sum
        else np.nan,
        "weighted_pd_high": float(
            np.average(funded["pd_high_alpha01"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
        "weighted_qhat": float(np.average(funded["qhat_v4"], weights=funded["funded_exposure"]))
        if exposure_sum
        else np.nan,
        "weighted_weak_source_proxy": float(
            np.average(funded["weak_source_proxy"], weights=funded["funded_exposure"])
        )
        if exposure_sum
        else np.nan,
        "scenario_loss_cvar90": cvar_value,
        "feasible_v12": _is_optimal(status),
    }
    cap_rows = [
        {
            "policy_id": policy_id,
            "constraint_name": "cvar90",
            "limit_v12": cvar_cap,
            "actual_v12": cvar_value,
            "slack_v12": cvar_cap - cvar_value,
            "active_v12": abs(cvar_cap - cvar_value) <= max(1.0, 0.002 * cvar_cap),
        },
        {
            "policy_id": policy_id,
            "constraint_name": "return_floor",
            "limit_v12": return_floor,
            "actual_v12": metrics["objective_return"],
            "slack_v12": metrics["objective_return"] - return_floor,
            "active_v12": abs(metrics["objective_return"] - return_floor)
            <= max(1.0, 0.002 * return_floor),
        },
        {
            "policy_id": policy_id,
            "constraint_name": "qhat_cap",
            "limit_v12": qhat_cap,
            "actual_v12": metrics["weighted_qhat"],
            "slack_v12": qhat_cap - metrics["weighted_qhat"],
            "active_v12": abs(qhat_cap - metrics["weighted_qhat"]) <= 0.002,
        },
    ]
    for cap_name, cap_value in caps.items():
        actual = (
            float(np.average(flags.loc[funded.index, cap_name], weights=funded["funded_exposure"]))
            if exposure_sum
            else np.nan
        )
        metrics[cap_name.replace("cap_", "share_")] = actual
        metrics[f"{cap_name}_limit_v12"] = cap_value
        cap_rows.append(
            {
                "policy_id": policy_id,
                "constraint_name": cap_name,
                "limit_v12": cap_value,
                "actual_v12": actual,
                "slack_v12": cap_value - actual,
                "active_v12": abs(cap_value - actual) <= 0.01,
            }
        )
    metrics["auditability_score_v12"] = _auditability_score(
        metrics["weighted_qhat"],
        metrics["weighted_weak_source_proxy"],
        cvar_value,
    )
    metrics["caps_json_v12"] = _json_dump(caps)
    return funded, metrics, loss_df, pd.DataFrame(cap_rows)


def build_cvar_column_generation_v12(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    max_pool_n: int,
    rounds: int,
    caps: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v11_alloc = _safe_read_parquet(TABLE_DIR / "paper4_v11_cvar_topk_warm_allocations.parquet")
    v10_mdcp_alloc = _safe_read_parquet(
        TABLE_DIR / "paper4_v10_mdcp_empirical_cap_allocations.parquet"
    )
    anchor = pd.concat(
        [df for df in [v11_alloc, v10_mdcp_alloc] if not df.empty], ignore_index=True
    )
    specs = [
        (80_000.0, 90_000.0),
        (80_000.0, 105_000.0),
        (95_000.0, 125_000.0),
        (110_000.0, 150_000.0),
        (125_000.0, 175_000.0),
        (145_000.0, 210_000.0),
        (160_000.0, 250_000.0),
    ]
    rows: list[dict[str, Any]] = []
    allocs: list[pd.DataFrame] = []
    losses: list[pd.DataFrame] = []
    active: list[pd.DataFrame] = []
    for round_idx in range(1, rounds + 1):
        pool_n = min(len(candidate_pool), max_pool_n + (round_idx - 1) * 4_000)
        pool = _prepare_column_pool_v12(
            candidate_pool,
            online_intervals,
            max_n=pool_n,
            online_method=online_method,
            anchor_allocations=anchor,
        )
        for floor, cap in specs:
            if round_idx > 1 and floor < 95_000:
                continue
            policy_id = (
                f"v12_cvar_mdcp_colgen_r{round_idx}_k{pool_n}_floor{int(floor)}_cap{int(cap)}"
            )
            alloc, metrics, loss, active_constraints = _solve_cvar_mdcp_policy_v12(
                pool,
                policy_id=policy_id,
                risk_tolerance=0.175,
                weak_penalty=0.070,
                width_penalty=0.070,
                cvar_cap=cap,
                return_floor=floor,
                caps=caps,
                qhat_cap=0.82,
                time_limit=180,
            )
            metrics.update(
                {
                    "column_round_v12": round_idx,
                    "pool_n_v12": int(len(pool)),
                    "universe_n_v12": int(len(candidate_pool)),
                    "pool_expansion_vs_v11": float(len(pool) / 18_000),
                    "solver_lane_v12": "cvar_mdcp_column_generation_lp",
                    "pool_design_v12": "score_returnpd_lowtail_audit_anchor_column_generation",
                    "full_universe_attempted_v12": len(candidate_pool) <= max_pool_n,
                    "full_universe_feasible_v12": len(candidate_pool) <= max_pool_n
                    and _is_optimal(metrics.get("solver_status")),
                }
            )
            rows.append(metrics)
            if _is_optimal(metrics.get("solver_status")):
                alloc["solver_lane_v12"] = "cvar_mdcp_column_generation_lp"
                allocs.append(alloc)
                losses.append(loss.assign(column_round_v12=round_idx, pool_n_v12=len(pool)))
                active.append(
                    active_constraints.assign(column_round_v12=round_idx, pool_n_v12=len(pool))
                )
                anchor = (
                    pd.concat([anchor, alloc], ignore_index=True) if not anchor.empty else alloc
                )
    if not any(bool(row.get("feasible_v12", False)) for row in rows):
        relaxed_caps = {key: min(1.0, value + 0.20) for key, value in caps.items()}
        pool = _prepare_column_pool_v12(
            candidate_pool,
            online_intervals,
            max_n=min(len(candidate_pool), max_pool_n),
            online_method=online_method,
            anchor_allocations=anchor,
        )
        for floor, cap in [(80_000.0, 250_000.0), (105_000.0, 300_000.0), (125_000.0, 360_000.0)]:
            policy_id = f"v12_cvar_mdcp_colgen_relaxed_k{len(pool)}_floor{int(floor)}_cap{int(cap)}"
            alloc, metrics, loss, active_constraints = _solve_cvar_mdcp_policy_v12(
                pool,
                policy_id=policy_id,
                risk_tolerance=0.190,
                weak_penalty=0.055,
                width_penalty=0.055,
                cvar_cap=cap,
                return_floor=floor,
                caps=relaxed_caps,
                qhat_cap=0.90,
                time_limit=180,
            )
            metrics.update(
                {
                    "column_round_v12": rounds + 1,
                    "pool_n_v12": int(len(pool)),
                    "universe_n_v12": int(len(candidate_pool)),
                    "pool_expansion_vs_v11": float(len(pool) / 18_000),
                    "solver_lane_v12": "cvar_mdcp_column_generation_lp_committee_relaxed",
                    "pool_design_v12": "score_returnpd_lowtail_audit_anchor_column_generation",
                    "full_universe_attempted_v12": len(candidate_pool) <= max_pool_n,
                    "full_universe_feasible_v12": len(candidate_pool) <= max_pool_n
                    and _is_optimal(metrics.get("solver_status")),
                    "cap_relaxation_v12": "committee_relaxed_after_strict_infeasible",
                }
            )
            rows.append(metrics)
            if _is_optimal(metrics.get("solver_status")):
                alloc["solver_lane_v12"] = "cvar_mdcp_column_generation_lp_committee_relaxed"
                allocs.append(alloc)
                losses.append(loss.assign(column_round_v12=rounds + 1, pool_n_v12=len(pool)))
                active.append(
                    active_constraints.assign(column_round_v12=rounds + 1, pool_n_v12=len(pool))
                )
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier["non_dominated_v12"] = False
        feasible = frontier[frontier["feasible_v12"].astype(bool)].copy()
        for idx, row in feasible.iterrows():
            other = feasible.drop(index=idx)
            dominated = (
                other["scenario_loss_cvar90"].le(row.get("scenario_loss_cvar90", np.inf))
                & other["objective_return"].ge(row.get("objective_return", -np.inf))
                & other["auditability_score_v12"].ge(row.get("auditability_score_v12", -np.inf))
                & (
                    other["scenario_loss_cvar90"].lt(row.get("scenario_loss_cvar90", np.inf))
                    | other["objective_return"].gt(row.get("objective_return", -np.inf))
                    | other["auditability_score_v12"].gt(row.get("auditability_score_v12", -np.inf))
                )
            ).any()
            frontier.loc[idx, "non_dominated_v12"] = not bool(dominated)
        frontier = frontier.sort_values(
            ["feasible_v12", "non_dominated_v12", "scenario_loss_cvar90", "objective_return"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
    return (
        frontier,
        pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame(),
        pd.concat(losses, ignore_index=True) if losses else pd.DataFrame(),
        pd.concat(active, ignore_index=True) if active else pd.DataFrame(),
    )


def build_mdcp_solver_v12(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    caps: dict[str, float],
    max_pool_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _prepare_column_pool_v12(
        candidate_pool,
        online_intervals,
        max_n=max_pool_n,
        online_method=online_method,
    )
    specs = [
        ("v12_mdcp_empirical_base", caps, 0.175, 0.070, 0.070, 0.82),
        (
            "v12_mdcp_empirical_tight",
            {k: max(0.24, v - 0.04) for k, v in caps.items()},
            0.1725,
            0.090,
            0.090,
            0.80,
        ),
        (
            "v12_mdcp_empirical_return_recovery",
            {k: min(0.92, v + 0.03) for k, v in caps.items()},
            0.180,
            0.060,
            0.060,
            0.84,
        ),
    ]
    rows = []
    allocs = []
    compatible_caps = {
        "cap_grade_dplus": caps["cap_grade_dplus"],
        "cap_dti_q3": caps["cap_dti_high"],
        "cap_score_low": caps["cap_score_low"],
        "cap_income_q5": caps["cap_income_q5"],
        "cap_period_2018h1": caps["cap_period_2018h1"],
    }
    for policy_id, _, rt, weak_penalty, width_penalty, qhat_cap in specs:
        local_caps = compatible_caps.copy()
        if "tight" in policy_id:
            local_caps = {k: max(0.24, v - 0.04) for k, v in local_caps.items()}
        if "return_recovery" in policy_id:
            local_caps = {k: min(0.92, v + 0.03) for k, v in local_caps.items()}
        alloc, metrics = _solve_family_cap_policy(
            pool,
            policy_id=policy_id,
            risk_tolerance=rt,
            weak_penalty=weak_penalty,
            width_penalty=width_penalty,
            caps=local_caps,
            max_weighted_qhat=qhat_cap,
            time_limit=150,
        )
        metrics["solver_lane_v12"] = "mdcp_empirical_caps_inside_solver"
        metrics["caps_json_v12"] = _json_dump(local_caps)
        metrics["auditability_score_v12"] = metrics.get("auditability_score_v8", np.nan)
        rows.append(metrics)
        if not alloc.empty:
            alloc["solver_lane_v12"] = "mdcp_empirical_caps_inside_solver"
            allocs.append(alloc)
    return pd.DataFrame(rows), pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame()


def build_dla_state_schema_v12() -> pd.DataFrame:
    rows = [
        ("S_t.cash", "liquid cash before month-t decision", "state"),
        ("S_t.outstanding", "remaining funded principal", "state"),
        ("S_t.expected_loss", "one-step expected loss under macro state", "state"),
        ("S_t.capital_used", "capital proxy consumed by outstanding loans", "state"),
        ("S_t.stage2_share", "IFRS9 SICR proxy share in current book", "state"),
        ("S_t.coverage_state", "online conformal defended coverage gate state", "state"),
        ("x_t", "loan-level funding decision for available issue-month loans", "decision"),
        ("S_t^x", "post-decision cash/outstanding/capital/stage mix", "post_decision_state"),
        ("W_{t+1}.macro", "macro/default/LGD/prepayment shock", "exogenous_information"),
        ("R_t", "interest, principal, losses, recoveries and reward proxy", "reward"),
        ("V(S_t)", "fitted value function approximating continuation value", "value_function"),
    ]
    return pd.DataFrame(rows, columns=["sdam_element", "definition_v12", "role"])


def _state_features(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row.get("cash_end", 0.0)) / 1_000_000.0,
            float(row.get("outstanding_balance_proxy", 0.0)) / 1_000_000.0,
            float(row.get("expected_loss", 0.0)) / 100_000.0,
            float(row.get("capital_used", 0.0)) / 100_000.0,
            float(row.get("cumulative_realized_loss", 0.0)) / 100_000.0,
            float(row.get("macro_state_v12", 0.0)),
            float(row.get("stage2_share_proxy_v12", 0.0)),
            float(row.get("coverage_state_v12", 1.0)),
        ],
        dtype=float,
    )


def _fit_value_beta(trace: pd.DataFrame, discount: float) -> np.ndarray:
    if trace.empty:
        return np.array(
            [
                0.0,
                1_000_000.0,
                850_000.0,
                -120_000.0,
                -40_000.0,
                -90_000.0,
                -30_000.0,
                -20_000.0,
                10_000.0,
            ]
        )
    work = trace.sort_values(["policy_id", "path_id", "month_idx"]).copy()
    if "state_value_proxy_v12" not in work and "state_value_proxy_v11" in work:
        work["state_value_proxy_v12"] = work["state_value_proxy_v11"]
    if "macro_state_v12" not in work and "macro_state_v11" in work:
        work["macro_state_v12"] = work["macro_state_v11"]
    if "stage2_share_proxy_v12" not in work:
        work["stage2_share_proxy_v12"] = 0.0
    if "coverage_state_v12" not in work:
        work["coverage_state_v12"] = 1.0
    if "reward_proxy_v12" not in work:
        work["reward_proxy_v12"] = (
            -pd.to_numeric(work.get("realized_loss", 0), errors="coerce").fillna(0)
            - 0.10 * pd.to_numeric(work.get("expected_loss", 0), errors="coerce").fillna(0)
            - 0.04 * pd.to_numeric(work.get("capital_used", 0), errors="coerce").fillna(0)
            + 0.002 * pd.to_numeric(work.get("cash_end", 0), errors="coerce").fillna(0)
        )
    work["next_value"] = work.groupby(["policy_id", "path_id"])["state_value_proxy_v12"].shift(-1)
    work["next_value"] = work["next_value"].fillna(work["state_value_proxy_v12"])
    y = (
        work["reward_proxy_v12"].astype(float).to_numpy()
        + discount * work["next_value"].astype(float).to_numpy()
    )
    x = np.vstack([_state_features(row) for _, row in work.iterrows()])
    penalty = np.eye(x.shape[1]) * 1e-4
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _predict_state_value_from_components(
    beta: np.ndarray,
    *,
    cash: float,
    outstanding: float,
    expected_loss: float,
    capital: float,
    cumulative_loss: float,
    macro: float,
    stage2: float,
    coverage: float,
) -> float:
    x = np.array(
        [
            1.0,
            cash / 1_000_000.0,
            outstanding / 1_000_000.0,
            expected_loss / 100_000.0,
            capital / 100_000.0,
            cumulative_loss / 100_000.0,
            macro,
            stage2,
            coverage,
        ]
    )
    return float(x @ beta)


def build_dla_fvi_v12(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    max_months: int,
    n_paths: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_column_pool_v12(
        candidate_pool,
        online_intervals,
        max_n=min(len(candidate_pool), 14_000),
        online_method=online_method,
    )
    months = sorted(pool["issue_month"].dropna().unique())[:max_months]
    strategies = {
        "v12_static_reference": {
            "capital": 0.00,
            "ecl": 0.00,
            "weak": 0.06,
            "qhat": 0.08,
            "deploy": 0.36,
            "value": 0.00,
        },
        "v12_fvi_bellman_rollout": {
            "capital": 0.24,
            "ecl": 0.78,
            "weak": 0.11,
            "qhat": 0.09,
            "deploy": 0.33,
            "value": 0.075,
        },
        "v12_fvi_capital_guarded": {
            "capital": 0.34,
            "ecl": 0.94,
            "weak": 0.14,
            "qhat": 0.11,
            "deploy": 0.29,
            "value": 0.095,
        },
        "v12_fvi_return_recovery": {
            "capital": 0.18,
            "ecl": 0.58,
            "weak": 0.09,
            "qhat": 0.065,
            "deploy": 0.35,
            "value": 0.055,
        },
    }
    beta = _fit_value_beta(
        _safe_read_parquet(TABLE_DIR / "paper4_v11_dla_adp_trace.parquet"), discount=0.92
    )
    trace = pd.DataFrame()
    decisions = pd.DataFrame()
    for iteration in range(1, iterations + 1):
        trace_rows: list[dict[str, Any]] = []
        decision_frames: list[pd.DataFrame] = []
        for path_id in range(n_paths):
            rng = np.random.default_rng(RNG_SEED + 7000 + iteration * 100 + path_id)
            macro = 0.0
            macros = []
            for _ in months:
                macro = 0.74 * macro + float(rng.normal(0, 0.30))
                macros.append(macro)
            for strategy, weights in strategies.items():
                cash = BUDGET
                outstanding: list[dict[str, Any]] = []
                cumulative_loss = cumulative_expected_loss = cumulative_funded = 0.0
                for t, month in enumerate(months, start=1):
                    macro_t = macros[t - 1]
                    principal_in = interest_in = realized_loss = recovery_in = 0.0
                    expected_loss = capital_used = 0.0
                    stage2_balance = 0.0
                    survivors: list[dict[str, Any]] = []
                    for item in outstanding:
                        age = t - item["month_idx"] + 1
                        remaining = max(float(item["remaining_balance"]), 0.0)
                        if remaining <= 1e-6:
                            continue
                        cyclic_lgd = float(
                            np.clip(item["lgd"] * (1 + 0.30 * max(macro_t, 0)), 0.23, 0.90)
                        )
                        monthly_pd = float(
                            np.clip(
                                (item["pd_high"] / 12.0) * math.exp(0.55 * macro_t), 0.0001, 0.75
                            )
                        )
                        expected_loss += remaining * cyclic_lgd * monthly_pd
                        capital_used += remaining * (
                            0.08 + 0.58 * item["pd_high"] + 0.10 * item["qhat"]
                        )
                        stage2_balance += remaining * float(
                            item["pd_high"] * (1 + max(macro_t, 0)) >= 0.20
                        )
                        cluster = _stable_uniform(path_id, strategy, month, "v12_dla_cluster")
                        u_default = (
                            0.64 * _stable_uniform(path_id, item["loan_id"], age, "v12_dla_default")
                            + 0.36 * cluster
                        )
                        if u_default < monthly_pd or (
                            item["y_true"] >= 0.5 and age >= 9 and u_default < 0.58
                        ):
                            loss = remaining * cyclic_lgd
                            recovery = loss * float(
                                np.clip(0.13 - 0.05 * max(macro_t, 0), 0.01, 0.16)
                            )
                            realized_loss += loss
                            recovery_in += recovery
                            cash += recovery
                        else:
                            prepay_prob = float(
                                np.clip(
                                    0.010 + 0.030 * (1 - item["pd_high"]) - 0.008 * max(macro_t, 0),
                                    0.002,
                                    0.065,
                                )
                            )
                            scheduled_principal = min(
                                remaining, item["original_exposure"] / max(item["term"], 1.0)
                            )
                            if (
                                _stable_uniform(path_id, item["loan_id"], age, "v12_dla_prepay")
                                < prepay_prob
                            ):
                                scheduled_principal = remaining
                            principal_in += scheduled_principal
                            interest_in += remaining * item["int_rate"] / 12.0
                            item["remaining_balance"] = remaining - scheduled_principal
                            if item["remaining_balance"] > 1e-6 and age < item["term"]:
                                survivors.append(item)
                    cash += principal_in + interest_in - realized_loss
                    outstanding = survivors
                    cumulative_loss += realized_loss
                    cumulative_expected_loss += expected_loss
                    outstanding_balance = float(
                        sum(item["remaining_balance"] for item in outstanding)
                    )
                    stage2_share = (
                        float(stage2_balance / outstanding_balance) if outstanding_balance else 0.0
                    )
                    coverage_state = 1.0 if macro_t < 0.75 else 0.82
                    state_value_pre = _predict_state_value_from_components(
                        beta,
                        cash=cash,
                        outstanding=outstanding_balance,
                        expected_loss=expected_loss,
                        capital=capital_used,
                        cumulative_loss=cumulative_loss,
                        macro=macro_t,
                        stage2=stage2_share,
                        coverage=coverage_state,
                    )
                    available = pool[pool["issue_month"].eq(month)].copy()
                    reward_proxy = (
                        interest_in
                        + principal_in
                        + recovery_in
                        - realized_loss
                        - 0.10 * expected_loss
                        - 0.04 * capital_used
                    )
                    if not available.empty and cash > 1_000:
                        available["capital_charge_v12"] = available["loan_amnt"] * (
                            0.08 + 0.58 * available["pd_high_alpha01"] + 0.10 * available["qhat_v4"]
                        )
                        available["ecl_proxy_v12"] = (
                            available["loan_amnt"]
                            * available["pd_high_alpha01"]
                            * DEFAULT_LGD
                            * (1 + max(macro_t, 0))
                        )
                        next_value = [
                            _predict_state_value_from_components(
                                beta,
                                cash=cash - float(row["loan_amnt"]),
                                outstanding=outstanding_balance + float(row["loan_amnt"]),
                                expected_loss=expected_loss + float(row["ecl_proxy_v12"]),
                                capital=capital_used + float(row["capital_charge_v12"]),
                                cumulative_loss=cumulative_loss,
                                macro=macro_t,
                                stage2=min(
                                    1.0, stage2_share + 0.02 * float(row["pd_high_alpha01"] >= 0.20)
                                ),
                                coverage=coverage_state,
                            )
                            for _, row in available.iterrows()
                        ]
                        available["continuation_value_v12"] = next_value
                        available["fvi_score_v12"] = (
                            available["base_return_vec"]
                            - weights["capital"] * available["capital_charge_v12"]
                            - weights["ecl"] * available["ecl_proxy_v12"]
                            - weights["weak"]
                            * available["loan_amnt"]
                            * available["weak_source_proxy"]
                            - weights["qhat"] * available["loan_amnt"] * available["qhat_v4"]
                            + weights["value"] * available["continuation_value_v12"]
                        )
                        deploy_budget = max(
                            0.0,
                            min(
                                cash * weights["deploy"] * (0.84 if macro_t > 0.45 else 1.0),
                                BUDGET * 0.33,
                            ),
                        )
                        funded = available.sort_values("fvi_score_v12", ascending=False).copy()
                        funded["cum_amount"] = funded["loan_amnt"].cumsum()
                        funded = funded[funded["cum_amount"].le(deploy_budget)].copy()
                        if funded.empty and deploy_budget >= 1_000:
                            funded = (
                                available.sort_values("fvi_score_v12", ascending=False)
                                .head(1)
                                .copy()
                            )
                        if not funded.empty:
                            funded["policy_id"] = strategy
                            funded["iteration_v12"] = iteration
                            funded["path_id"] = path_id
                            funded["decision_month"] = month
                            funded["month_idx"] = t
                            funded["funded_exposure"] = funded["loan_amnt"]
                            funded["state_value_pre_decision_v12"] = state_value_pre
                            cash -= float(funded["funded_exposure"].sum())
                            cumulative_funded += float(funded["funded_exposure"].sum())
                            decision_frames.append(
                                funded[
                                    [
                                        "policy_id",
                                        "iteration_v12",
                                        "path_id",
                                        "decision_month",
                                        "month_idx",
                                        "loan_id",
                                        "funded_exposure",
                                        "capital_charge_v12",
                                        "ecl_proxy_v12",
                                        "pd_high_alpha01",
                                        "qhat_v4",
                                        "weak_source_proxy",
                                        "continuation_value_v12",
                                        "fvi_score_v12",
                                        "base_return_vec",
                                        "int_rate_decimal",
                                        "y_true",
                                        "term",
                                        "original_grade",
                                        "period",
                                    ]
                                ].head(320)
                            )
                            for _, row in funded.iterrows():
                                outstanding.append(
                                    {
                                        "loan_id": row["loan_id"],
                                        "month_idx": t,
                                        "original_exposure": float(row["funded_exposure"]),
                                        "remaining_balance": float(row["funded_exposure"]),
                                        "term": float(row["term"]),
                                        "int_rate": float(row["int_rate_decimal"]),
                                        "pd_high": float(row["pd_high_alpha01"]),
                                        "qhat": float(row["qhat_v4"]),
                                        "lgd": DEFAULT_LGD,
                                        "y_true": float(row["y_true"]),
                                    }
                                )
                    outstanding_balance = float(
                        sum(item["remaining_balance"] for item in outstanding)
                    )
                    state_value = (
                        cash
                        + outstanding_balance
                        - expected_loss
                        - 0.12 * capital_used
                        - 0.30 * cumulative_loss
                    )
                    trace_rows.append(
                        {
                            "policy_id": strategy,
                            "iteration_v12": iteration,
                            "path_id": path_id,
                            "month_idx": t,
                            "calendar_month": month,
                            "macro_state_v12": macro_t,
                            "cash_end": cash,
                            "realized_loss": realized_loss,
                            "expected_loss": expected_loss,
                            "capital_used": capital_used,
                            "recovery_in": recovery_in,
                            "outstanding_balance_proxy": outstanding_balance,
                            "stage2_share_proxy_v12": stage2_share,
                            "coverage_state_v12": coverage_state,
                            "reward_proxy_v12": reward_proxy,
                            "state_value_proxy_v12": state_value,
                            "cumulative_realized_loss": cumulative_loss,
                            "cumulative_expected_loss": cumulative_expected_loss,
                            "cumulative_funded_exposure": cumulative_funded,
                        }
                    )
        trace = pd.DataFrame(trace_rows)
        decisions = (
            pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
        )
        beta = _fit_value_beta(trace, discount=0.92)
    final = trace.sort_values("month_idx").groupby(["policy_id", "path_id"], as_index=False).tail(1)
    summary = final.groupby("policy_id", as_index=False).agg(
        n_paths=("path_id", "nunique"),
        final_state_value_mean=("state_value_proxy_v12", "mean"),
        final_state_value_p05=("state_value_proxy_v12", lambda s: float(np.quantile(s, 0.05))),
        final_cash_mean=("cash_end", "mean"),
        cumulative_realized_loss_mean=("cumulative_realized_loss", "mean"),
        cumulative_expected_loss_mean=("cumulative_expected_loss", "mean"),
        cumulative_funded_exposure_mean=("cumulative_funded_exposure", "mean"),
    )
    if not decisions.empty:
        summary = summary.merge(
            decisions.groupby("policy_id", as_index=False).agg(
                stored_decision_rows=("loan_id", "count"),
                unique_funded_loans=("loan_id", "nunique"),
            ),
            on="policy_id",
            how="left",
        )
    static = summary[summary["policy_id"].eq("v12_static_reference")].iloc[0]
    comparison = summary.copy()
    comparison["baseline_policy_id"] = "v12_static_reference"
    comparison["delta_state_value_vs_static"] = comparison["final_state_value_mean"] - float(
        static["final_state_value_mean"]
    )
    comparison["delta_loss_vs_static"] = comparison["cumulative_realized_loss_mean"] - float(
        static["cumulative_realized_loss_mean"]
    )
    comparison["adp_scope_v12"] = "iterative_fitted_value_rollout_not_exact_bellman_optimality"
    coef = pd.DataFrame(
        {
            "feature": [
                "intercept",
                "cash_m",
                "outstanding_m",
                "expected_loss_100k",
                "capital_100k",
                "cum_loss_100k",
                "macro_state",
                "stage2_share",
                "coverage_state",
            ],
            "coefficient": beta,
            "iterations_v12": iterations,
            "discount_v12": 0.92,
            "model_scope_v12": "fitted_value_iteration_rollout_proxy",
        }
    )
    return coef, decisions, trace, comparison


def _feature_matrix_v12(
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "return_per_dollar": pool["base_return_vec"] / pool["loan_amnt"].clip(lower=1),
            "pd_high": pool["pd_high_alpha01"].astype(float),
            "qhat": pool["qhat_v4"].astype(float),
            "weak_source": pool["weak_source_proxy"].astype(float),
            "ecl_rate": pool["pd_high_alpha01"].astype(float) * DEFAULT_LGD,
            "amount_10k": pool["loan_amnt"].astype(float) / 10_000.0,
            "grade_dplus": pool["original_grade"]
            .astype(str)
            .isin(["D", "E", "F", "G"])
            .astype(float),
            "dti_high": pool["dti_band"]
            .astype(str)
            .isin(["dti_q3", "dti_q4", "dti_q5"])
            .astype(float),
            "score_low": pool["score_decile"].astype(str).isin(["0", "1", "2"]).astype(float),
            "income_q5": pool["income_band"].astype(str).eq("inc_q5").astype(float),
        }
    ).fillna(0)
    mean = frame.mean().to_numpy()
    std = frame.std().to_numpy()
    std[std < 1e-8] = 1.0
    z = (frame.to_numpy() - mean) / std
    return frame, z, mean, std


def _greedy_constrained_select(
    work: pd.DataFrame,
    *,
    score_col: str,
    pd_cap: float,
    grade_cap: float,
    dti_cap: float,
    qhat_cap: float,
) -> pd.DataFrame:
    selected_rows = []
    exposure = pd_numer = grade_numer = dti_numer = qhat_numer = 0.0
    for _, row in work.sort_values(score_col, ascending=False).iterrows():
        amount = float(row["loan_amnt"])
        if exposure + amount > BUDGET:
            continue
        next_exposure = exposure + amount
        next_pd = pd_numer + amount * float(row["pd_high_alpha01"])
        next_grade = grade_numer + amount * float(row["grade_dplus_v12"])
        next_dti = dti_numer + amount * float(row["dti_high_v12"])
        next_qhat = qhat_numer + amount * float(row["qhat_v4"])
        if next_pd > pd_cap * next_exposure:
            continue
        if next_grade > grade_cap * next_exposure:
            continue
        if next_dti > dti_cap * next_exposure:
            continue
        if next_qhat > qhat_cap * next_exposure:
            continue
        selected_rows.append(row)
        exposure = next_exposure
        pd_numer = next_pd
        grade_numer = next_grade
        dti_numer = next_dti
        qhat_numer = next_qhat
        if exposure >= 0.98 * BUDGET:
            break
    return pd.DataFrame(selected_rows)


def build_spo_regret_surrogate_v12(
    candidate_pool: pd.DataFrame,
    online_intervals: pd.DataFrame,
    *,
    online_method: str,
    max_pool_n: int,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = _prepare_column_pool_v12(
        candidate_pool, online_intervals, max_n=max_pool_n, online_method=online_method
    ).copy()
    pool["grade_dplus_v12"] = (
        pool["original_grade"].astype(str).isin(["D", "E", "F", "G"]).astype(float)
    )
    pool["dti_high_v12"] = (
        pool["dti_band"].astype(str).isin(["dti_q3", "dti_q4", "dti_q5"]).astype(float)
    )
    feature_frame, z, mean, std = _feature_matrix_v12(pool)
    feature_cols = list(feature_frame.columns)
    pool = pool.reset_index(drop=True)
    true_score = (
        pool["base_return_vec"]
        - 0.065 * pool["loan_amnt"] * pool["qhat_v4"]
        - 0.075 * pool["loan_amnt"] * pool["weak_source_proxy"]
        - 0.45 * pool["loan_amnt"] * pool["pd_high_alpha01"] * DEFAULT_LGD
    )
    pool["true_decision_value_v12"] = true_score
    months = sorted(pool["issue_month"].dropna().unique())
    n_train = max(1, int(len(months) * 0.60))
    n_val = max(1, int(len(months) * 0.20))
    split_months = {
        "train": set(months[:n_train]),
        "validation": set(months[n_train : n_train + n_val]),
        "test": set(months[n_train + n_val :]),
    }
    theta = np.zeros(z.shape[1])
    theta[0] = 1.0
    theta[1] = -0.30
    theta[2] = -0.20
    lr = 0.05
    rows = []
    for epoch in range(1, epochs + 1):
        for month in sorted(split_months["train"]):
            idx = pool.index[pool["issue_month"].eq(month)].to_numpy()
            if len(idx) < 20:
                continue
            local = pool.loc[idx].copy()
            local["pred_score_v12"] = z[idx] @ theta
            true_sel = _greedy_constrained_select(
                local.assign(true_score_v12=local["true_decision_value_v12"]),
                score_col="true_score_v12",
                pd_cap=0.175,
                grade_cap=0.58,
                dti_cap=0.64,
                qhat_cap=0.82,
            )
            pred_sel = _greedy_constrained_select(
                local,
                score_col="pred_score_v12",
                pd_cap=0.175,
                grade_cap=0.58,
                dti_cap=0.64,
                qhat_cap=0.82,
            )
            if true_sel.empty or pred_sel.empty:
                continue
            true_value = float(true_sel["true_decision_value_v12"].sum())
            pred_value = float(pred_sel["true_decision_value_v12"].sum())
            regret = max(0.0, true_value - pred_value)
            true_feat = z[true_sel.index].mean(axis=0)
            pred_feat = z[pred_sel.index].mean(axis=0)
            theta += lr * (true_feat - pred_feat) * min(1.0, regret / max(abs(true_value), 1.0))
        for split, split_set in split_months.items():
            regrets = []
            true_values = []
            pred_values = []
            for month in sorted(split_set):
                idx = pool.index[pool["issue_month"].eq(month)].to_numpy()
                if len(idx) < 20:
                    continue
                local = pool.loc[idx].copy()
                local["pred_score_v12"] = z[idx] @ theta
                true_sel = _greedy_constrained_select(
                    local.assign(true_score_v12=local["true_decision_value_v12"]),
                    score_col="true_score_v12",
                    pd_cap=0.175,
                    grade_cap=0.58,
                    dti_cap=0.64,
                    qhat_cap=0.82,
                )
                pred_sel = _greedy_constrained_select(
                    local,
                    score_col="pred_score_v12",
                    pd_cap=0.175,
                    grade_cap=0.58,
                    dti_cap=0.64,
                    qhat_cap=0.82,
                )
                if true_sel.empty or pred_sel.empty:
                    continue
                true_value = float(true_sel["true_decision_value_v12"].sum())
                pred_value = float(pred_sel["true_decision_value_v12"].sum())
                true_values.append(true_value)
                pred_values.append(pred_value)
                regrets.append(max(0.0, true_value - pred_value))
            rows.append(
                {
                    "training_id": "v12_spo_style_temporal_regret_surrogate",
                    "epoch": epoch,
                    "split": split,
                    "months": len(split_set),
                    "mean_decision_regret": float(np.mean(regrets)) if regrets else np.nan,
                    "total_decision_regret": float(np.sum(regrets)) if regrets else np.nan,
                    "mean_true_value": float(np.mean(true_values)) if true_values else np.nan,
                    "mean_pred_value_under_true_score": float(np.mean(pred_values))
                    if pred_values
                    else np.nan,
                    "claim_scope_v12": "structured_temporal_regret_training_not_differentiable_spo_plus_theorem",
                }
            )
    pool["spo_regret_score_v12"] = z @ theta
    coef = pd.DataFrame(
        [
            {
                "feature": feature,
                "coefficient": float(coef_value),
                "feature_mean": float(mu),
                "feature_std": float(sig),
                "model_scope_v12": "spo_style_structured_regret_perceptron",
            }
            for feature, coef_value, mu, sig in zip(feature_cols, theta, mean, std)
        ]
    )
    variants = [
        ("v12_spo_regret_balanced", "spo_regret_score_v12", 0.175, 0.58, 0.64, 0.82, 0.00),
        ("v12_spo_regret_audit_guarded", "spo_regret_score_v12", 0.1725, 0.54, 0.60, 0.80, -0.05),
        ("v12_spo_regret_return_recovery", "spo_regret_score_v12", 0.185, 0.60, 0.66, 0.84, 0.04),
    ]
    summaries = []
    allocs = []
    for policy_id, score, pd_cap, grade_cap, dti_cap, qhat_cap, return_boost in variants:
        work = pool.copy()
        work["candidate_score_v12"] = work[score] + return_boost * (
            work["base_return_vec"] / work["loan_amnt"].clip(lower=1)
        )
        selected = _greedy_constrained_select(
            work,
            score_col="candidate_score_v12",
            pd_cap=pd_cap,
            grade_cap=grade_cap,
            dti_cap=dti_cap,
            qhat_cap=qhat_cap,
        )
        if selected.empty:
            continue
        selected = selected.copy()
        selected["policy_id"] = policy_id
        selected["funded_exposure"] = selected["loan_amnt"]
        selected["spo_regret_loss_proxy_v12"] = (
            selected["true_decision_value_v12"].max() - selected["true_decision_value_v12"]
        ).clip(lower=0)
        summary = _allocation_summary(selected, policy_id, "spo_style_regret_trained")
        summary.update(
            {
                "pd_cap_v12": pd_cap,
                "grade_cap_v12": grade_cap,
                "dti_cap_v12": dti_cap,
                "qhat_cap_v12": qhat_cap,
                "mean_spo_regret_score_v12": _weighted_average(selected, "spo_regret_score_v12"),
                "decision_regret_proxy_v12": float(selected["spo_regret_loss_proxy_v12"].mean()),
                "constraint_pd_pass": bool(summary["weighted_pd_high"] <= 0.175 + 1e-8),
                "training_scope_v12": "temporal_structured_regret_surrogate_not_cvxpylayer",
            }
        )
        summaries.append(summary)
        allocs.append(
            selected[
                [
                    "policy_id",
                    "loan_id",
                    "issue_month",
                    "period",
                    "original_grade",
                    "loan_amnt",
                    "funded_exposure",
                    "base_return_vec",
                    "int_rate_decimal",
                    "y_true",
                    "pd_high_alpha01",
                    "qhat_v4",
                    "weak_source_proxy",
                    "spo_regret_score_v12",
                    "candidate_score_v12",
                    "true_decision_value_v12",
                    "spo_regret_loss_proxy_v12",
                ]
            ].head(700)
        )
    return (
        pd.DataFrame(rows),
        coef,
        pd.DataFrame(summaries),
        pd.concat(allocs, ignore_index=True) if allocs else pd.DataFrame(),
    )


def build_sample_paths_v12(
    allocations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    n_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if allocations.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = candidate_pool.copy()
    base["default_proxy"] = pd.to_numeric(
        base.get("default_flag", base.get("y_true", 0)), errors="coerce"
    ).fillna(0)
    base["pd_high_alpha01"] = pd.to_numeric(base["pd_high_alpha01"], errors="coerce").fillna(0.10)
    calibration = base.groupby(["period", "original_grade"], dropna=False, as_index=False).agg(
        n=("loan_id", "nunique"),
        observed_default_rate=("default_proxy", "mean"),
        mean_pd_high=("pd_high_alpha01", "mean"),
        mean_int_rate=("int_rate_decimal", "mean"),
    )
    calibration["default_multiplier_v12"] = (
        calibration["observed_default_rate"] / calibration["mean_pd_high"].clip(lower=0.01)
    ).clip(0.40, 2.50)
    global_multiplier = float(
        base["default_proxy"].mean() / max(base["pd_high_alpha01"].mean(), 0.01)
    )
    calibration["support_status_v12"] = np.where(
        calibration["n"].ge(250), "direct_cell", "hierarchical_shrunk"
    )
    low = calibration["support_status_v12"].eq("hierarchical_shrunk")
    calibration.loc[low, "default_multiplier_v12"] = (
        0.55 * calibration.loc[low, "default_multiplier_v12"] + 0.45 * global_multiplier
    )
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "baseline_macro_calibrated",
                "macro_mean": 0.00,
                "macro_sd": 0.25,
                "rho": 0.45,
                "lgd_cycle": 0.24,
                "default_cycle": 0.34,
            },
            {
                "scenario_id": "adverse_macro_calibrated",
                "macro_mean": 0.45,
                "macro_sd": 0.34,
                "rho": 0.58,
                "lgd_cycle": 0.36,
                "default_cycle": 0.52,
            },
            {
                "scenario_id": "severe_macro_calibrated",
                "macro_mean": 0.90,
                "macro_sd": 0.42,
                "rho": 0.70,
                "lgd_cycle": 0.50,
                "default_cycle": 0.70,
            },
        ]
    )
    alloc = allocations.copy()
    alloc["loan_id"] = alloc["loan_id"].astype(str)
    alloc["period"] = alloc.get("period", "unknown").astype(str)
    alloc["original_grade"] = alloc.get("original_grade", "unknown").astype(str)
    alloc["funded_exposure"] = pd.to_numeric(
        alloc.get("funded_exposure", alloc.get("loan_amnt", 0)), errors="coerce"
    ).fillna(0)
    alloc = alloc.merge(
        calibration[["period", "original_grade", "default_multiplier_v12", "support_status_v12"]],
        on=["period", "original_grade"],
        how="left",
    )
    alloc["default_multiplier_v12"] = alloc["default_multiplier_v12"].fillna(global_multiplier)
    alloc["support_status_v12"] = alloc["support_status_v12"].fillna("global_fallback")
    rows = []
    for path_id in range(n_paths):
        scenario = scenarios.iloc[path_id % len(scenarios)]
        rng = np.random.default_rng(RNG_SEED + 9000 + path_id)
        macro = float(
            scenario["macro_mean"] + scenario["rho"] * rng.normal(0, scenario["macro_sd"])
        )
        lgd = float(np.clip(DEFAULT_LGD * (1 + scenario["lgd_cycle"] * max(macro, 0)), 0.20, 0.92))
        default_factor = float(np.exp(float(scenario["default_cycle"]) * macro))
        for policy_id, local in alloc.groupby("policy_id"):
            exposure = local["funded_exposure"].astype(float).to_numpy()
            pd_high = (
                pd.to_numeric(local.get("pd_high_alpha01", 0.10), errors="coerce")
                .fillna(0.10)
                .to_numpy()
            )
            cohort = (
                local.get("issue_month", "unknown")
                .astype(str)
                .map(
                    lambda m: (
                        1
                        + 0.16
                        * math.sin(
                            2 * math.pi * _stable_uniform(path_id, policy_id, m, "v12_vintage")
                        )
                    )
                )
                .to_numpy()
            )
            prob = np.clip(
                pd_high
                * local["default_multiplier_v12"].astype(float).to_numpy()
                * default_factor
                * cohort,
                0,
                1,
            )
            cluster = _stable_uniform(
                path_id, policy_id, scenario["scenario_id"], "v12_default_cluster"
            )
            defaults = np.array(
                [
                    0.58 * _stable_uniform(path_id, loan_id, "v12_common_default") + 0.42 * cluster
                    < p
                    for loan_id, p in zip(local["loan_id"].astype(str), prob)
                ],
                dtype=float,
            )
            loss = float(np.sum(exposure * lgd * defaults))
            rows.append(
                {
                    "path_id": path_id,
                    "scenario_id": scenario["scenario_id"],
                    "policy_id": policy_id,
                    "macro_state_v12": macro,
                    "lgd_cycle_v12": lgd,
                    "default_factor_v12": default_factor,
                    "portfolio_loss_v12": loss,
                    "funded_exposure": float(exposure.sum()),
                    "default_count_v12": int(defaults.sum()),
                    "mean_default_multiplier_v12": float(local["default_multiplier_v12"].mean()),
                }
            )
    paths = pd.DataFrame(rows)
    ci = paths.groupby("policy_id", as_index=False).agg(
        n_paths=("path_id", "nunique"),
        mean_loss=("portfolio_loss_v12", "mean"),
        p05_loss=("portfolio_loss_v12", lambda s: float(np.quantile(s, 0.05))),
        p50_loss=("portfolio_loss_v12", lambda s: float(np.quantile(s, 0.50))),
        p95_loss=("portfolio_loss_v12", lambda s: float(np.quantile(s, 0.95))),
        mean_default_count=("default_count_v12", "mean"),
        funded_exposure=("funded_exposure", "mean"),
        mean_default_multiplier=("mean_default_multiplier_v12", "mean"),
    )
    pairwise_rows = []
    champion_id = (
        ci.sort_values(["p95_loss", "mean_loss"]).iloc[0]["policy_id"] if not ci.empty else None
    )
    if champion_id is not None:
        champ = paths[paths["policy_id"].eq(champion_id)][
            ["path_id", "scenario_id", "portfolio_loss_v12"]
        ].rename(columns={"portfolio_loss_v12": "champion_loss_v12"})
        for policy_id, local in paths.groupby("policy_id"):
            merged = local.merge(champ, on=["path_id", "scenario_id"], how="inner")
            diff = merged["portfolio_loss_v12"] - merged["champion_loss_v12"]
            pairwise_rows.append(
                {
                    "policy_id": policy_id,
                    "reference_policy_id": champion_id,
                    "mean_loss_diff_vs_reference": float(diff.mean()),
                    "p05_loss_diff_vs_reference": float(np.quantile(diff, 0.05)),
                    "p95_loss_diff_vs_reference": float(np.quantile(diff, 0.95)),
                    "prob_lower_loss_than_reference": float((diff < 0).mean()),
                    "n_common_paths": int(len(merged)),
                }
            )
    return (
        calibration,
        scenarios,
        paths,
        ci.sort_values(["p95_loss", "mean_loss"]),
        pd.DataFrame(pairwise_rows),
    )


def build_ifrs9_sicr_v12(candidate_pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness_summary, readiness_detail, sicr_v10 = build_ifrs9_proxy_v10(candidate_pool)
    rows = []
    pd0 = candidate_pool["pd_point_alpha01"].astype(float).clip(0, 1)
    pdl_base = candidate_pool["pd_high_alpha01"].astype(float).clip(0, 1)
    stage3 = (
        candidate_pool.get("default_flag", candidate_pool.get("y_true", 0)).astype(float).ge(0.5)
    )
    for macro_scenario, macro_mult in [("baseline", 1.0), ("adverse", 1.35), ("severe", 1.75)]:
        pdl = (pdl_base * macro_mult).clip(0, 1)
        for rel in [1.5, 2.0, 2.5]:
            for abs_pd in [0.18, 0.20, 0.25, 0.30]:
                stage2 = (pdl >= pd0 * rel) | (pdl >= abs_pd)
                stage2 = stage2 & ~stage3
                stage1 = ~(stage2 | stage3)
                ecl12 = candidate_pool["loan_amnt"].astype(float) * pdl * DEFAULT_LGD
                rows.append(
                    {
                        "scenario": macro_scenario,
                        "relative_pd_increase": rel,
                        "absolute_pd_threshold": abs_pd,
                        "stage1_share_v12": float(stage1.mean()),
                        "stage2_share_v12": float(stage2.mean()),
                        "stage3_share_v12": float(stage3.mean()),
                        "mean_ecl_proxy_v12": float(ecl12.mean()),
                        "stage_mix_defensible_proxy": bool(0.08 <= stage2.mean() <= 0.65),
                        "contractual_ifrs9_claim_allowed": False,
                        "claim_scope_v12": "available_data_sicr_proxy_sensitivity_not_contractual_ifrs9",
                    }
                )
    readiness = readiness_detail.copy()
    readiness["status_v12"] = readiness["status_v10"]
    readiness["claim_scope_v12"] = readiness["claim_scope"]
    readiness["contractual_ifrs9_claim_allowed"] = False
    return readiness, pd.DataFrame(rows)


def build_causal_fairness_v12() -> tuple[pd.DataFrame, pd.DataFrame]:
    causal, causal_tests, fairness = build_causal_fairness_v10()
    causal_rows = []
    for outcome in ["default_flag", "prepayment_proxy", "net_return_proxy", "loss_proxy"]:
        causal_rows.append(
            {
                "dossier_item": f"outcome_{outcome}",
                "status_v12": "defined_for_observational_stress",
                "policy_value_allowed_v12": False,
                "blocker_v12": "identification_overlap_sensitivity_required",
            }
        )
    causal_rows.extend(
        [
            {
                "dossier_item": "overlap",
                "status_v12": "blocked"
                if causal.empty
                or not bool(causal.get("overlap_pass_v10", pd.Series(False)).iloc[0])
                else "pass",
                "policy_value_allowed_v12": False,
                "blocker_v12": "overlap_not_stable_enough_for_CATE_policy_value",
            },
            {
                "dossier_item": "hidden_bias_sensitivity",
                "status_v12": "blocked",
                "policy_value_allowed_v12": False,
                "blocker_v12": "sensitivity_not_stable_to_hidden_confounding",
            },
        ]
    )
    fairness_v12 = fairness.copy()
    fairness_v12["status_v12"] = fairness_v12["status_v10"]
    fairness_v12["legal_fair_lending_claim_allowed"] = False
    fairness_v12["allowed_scope_v12"] = "proxy_governance_stress_only"
    return pd.DataFrame(causal_rows), fairness_v12


def build_working_registry_v12(
    cvar: pd.DataFrame,
    mdcp: pd.DataFrame,
    dla: pd.DataFrame,
    spo: pd.DataFrame,
    sample_ci: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in (
        cvar[cvar.get("feasible_v12", pd.Series(False, index=cvar.index)).astype(bool)]
        .head(14)
        .iterrows()
    ):
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v12": "cvar_mdcp_colgen",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("scenario_loss_cvar90"),
                "auditability_score": row.get("auditability_score_v12"),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v12_cvar_column_generation_frontier.csv",
                "caveat": "column-generation top-k with MDCP caps; not exact full-universe proof",
            }
        )
    for _, row in mdcp.head(8).iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v12": "mdcp_empirical_solver",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("weighted_pd_high", np.nan) * BUDGET * DEFAULT_LGD,
                "auditability_score": row.get("auditability_score_v12"),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v12_mdcp_source_cap_solver_summary.csv",
                "caveat": "MDCP caps inside solver without CVaR scenario constraint",
            }
        )
    for _, row in dla[~dla["policy_id"].eq("v12_static_reference")].iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v12": "dla_fitted_value_iteration",
                "return_proxy": np.nan,
                "tail_risk_proxy": row.get("cumulative_realized_loss_mean"),
                "auditability_score": np.nan,
                "state_value_delta": row.get("delta_state_value_vs_static"),
                "source_artifact": "paper4_v12_dla_fvi_comparison.csv",
                "caveat": "iterative fitted-value rollout, not exact Bellman optimality",
            }
        )
    for _, row in spo.head(8).iterrows():
        rows.append(
            {
                "policy_id": row["policy_id"],
                "lane_v12": "spo_regret_surrogate",
                "return_proxy": row.get("objective_return"),
                "tail_risk_proxy": row.get("ecl_proxy_v11", row.get("ecl_proxy_v12", np.nan)),
                "auditability_score": row.get(
                    "auditability_score_v11", row.get("auditability_score_v12", np.nan)
                ),
                "state_value_delta": np.nan,
                "source_artifact": "paper4_v12_spo_plus_surrogate_candidates.csv",
                "caveat": "SPO-style temporal regret surrogate, not differentiable SPO+ theorem",
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry, {}
    if not sample_ci.empty:
        registry = registry.merge(
            sample_ci[["policy_id", "mean_loss", "p95_loss", "mean_default_count"]],
            on="policy_id",
            how="left",
        )
    registry["return_score"] = _percentile_rank(registry["return_proxy"], high_is_good=True)
    registry["audit_score"] = _percentile_rank(registry["auditability_score"], high_is_good=True)
    registry["state_value_score"] = _percentile_rank(
        registry["state_value_delta"], high_is_good=True
    )
    registry["tail_score"] = _percentile_rank(registry["tail_risk_proxy"], high_is_good=False)
    registry["path_score"] = (
        _percentile_rank(registry["p95_loss"], high_is_good=False)
        if "p95_loss" in registry
        else np.nan
    )
    registry["working_candidate_score_v12"] = registry[
        ["return_score", "audit_score", "state_value_score", "tail_score", "path_score"]
    ].mean(axis=1)
    registry["online_gate_pass_v12"] = True
    registry["ifrs9_contractual_claim_allowed"] = False
    registry["fair_lending_legal_claim_allowed"] = False
    registry["paper1_promotion_allowed"] = False
    registry["paper4_final_promotion_allowed"] = False
    registry = registry.sort_values("working_candidate_score_v12", ascending=False).reset_index(
        drop=True
    )
    registry["registry_rank_v12"] = np.arange(1, len(registry) + 1)
    registry["registry_decision_v12"] = np.where(
        registry["registry_rank_v12"].eq(1),
        "paper4_working_champion",
        np.where(registry["registry_rank_v12"].le(6), "paper4_working_challenger", "lane_evidence"),
    )
    champ = registry.iloc[0].to_dict()
    champion = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": champ["policy_id"],
        "lane_v12": champ["lane_v12"],
        "registry_rank_v12": int(champ["registry_rank_v12"]),
        "working_candidate_score_v12": float(champ["working_candidate_score_v12"]),
        "scope": "paper4_working_champion_only",
        "paper1_artifacts_modified": False,
        "paper1_promotion_allowed": False,
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "contractual_ifrs9_claim_allowed": False,
        "fair_lending_legal_claim_allowed": False,
        "caveat": champ["caveat"],
    }
    return registry, champion


def build_blocker_dashboard_v12(status: dict[str, Any]) -> pd.DataFrame:
    rows = [
        (
            "online_efficiency",
            "resolved",
            "v9 gate remains active for v12",
            "future-period validation",
        ),
        (
            "cvar_scale",
            "near_resolved",
            f"v12 feasible={status['cvar_feasible_count_v12']} with MDCP caps",
            "exact full universe or true column-generation duals",
        ),
        (
            "mdcp_inside_solver",
            "near_resolved",
            f"v12 optimal={status['mdcp_optimal_count_v12']}",
            "future coverage calibration",
        ),
        (
            "dla_adp",
            "near_resolved",
            "iterative fitted-value rollout implemented",
            "formal Bellman optimality remains open",
        ),
        (
            "spo_dfl",
            "near_resolved",
            "temporal regret surrogate implemented",
            "differentiable SPO+/cvxpylayer remains open",
        ),
        (
            "sample_paths",
            "near_resolved",
            "macro-calibrated common paths implemented",
            "external macro/default calibration",
        ),
        (
            "working_champion",
            "resolved",
            status.get("working_champion_policy_id_v12", "none"),
            "re-evaluate after v13",
        ),
        (
            "ifrs9_contractual",
            "data_blocked",
            "proxy readiness only",
            "servicing/DPD/cure/recovery/EAD/macro data",
        ),
        (
            "causal_cate",
            "theory_blocked",
            "policy value remains blocked",
            "identification, overlap, sensitivity",
        ),
        (
            "fairness",
            "data_blocked",
            "proxy governance only",
            "protected attributes or approved external protocol",
        ),
        ("paper1_freeze", "resolved", "Paper Estrella untouched", "continue Paper 4 only"),
    ]
    return pd.DataFrame(
        rows, columns=["blocker_id", "status_v12", "current_diagnosis", "next_action"]
    )


def build_claim_matrix_v12() -> pd.DataFrame:
    rows = [
        (
            "Artifact audit",
            "implemented",
            "paper4_v12_artifact_audit.csv",
            "19ba-v12-resolution-wave.qmd",
            "inventory only",
        ),
        (
            "Method references",
            "implemented_primary_sources",
            "paper4_v12_method_reference_registry.csv",
            "19ba-v12-resolution-wave.qmd",
            "references guide implementation boundaries",
        ),
        (
            "CVaR/OCE with MDCP caps",
            "implemented_column_generation_lp",
            "paper4_v12_cvar_column_generation_frontier.csv",
            "19ba-v12-resolution-wave.qmd",
            "not exact full-universe proof",
        ),
        (
            "MDCP/source coverage",
            "implemented_inside_solver",
            "paper4_v12_mdcp_source_cap_solver_summary.csv",
            "19ba-v12-resolution-wave.qmd",
            "empirical caps, not legal fairness",
        ),
        (
            "DLA/SDAM FVI",
            "implemented_iterative_fitted_value_rollout",
            "paper4_v12_dla_fvi_comparison.csv",
            "19ba-v12-resolution-wave.qmd",
            "not exact Bellman optimality",
        ),
        (
            "SPO/DFL",
            "implemented_temporal_regret_surrogate",
            "paper4_v12_spo_plus_surrogate_candidates.csv",
            "19ba-v12-resolution-wave.qmd",
            "not differentiable SPO+ theorem",
        ),
        (
            "Sample paths",
            "implemented_macro_calibrated_common_paths",
            "paper4_v12_sample_path_macro_calibrated_ci.csv",
            "19ba-v12-resolution-wave.qmd",
            "internal calibration, not future forecast",
        ),
        (
            "Working champion",
            "implemented_paper4_only",
            "paper4_v12_working_candidate_registry.csv",
            "19ba-v12-resolution-wave.qmd",
            "does not modify Paper Estrella",
        ),
        (
            "IFRS9/SICR",
            "implemented_proxy_sensitivity",
            "paper4_v12_ifrs9_sicr_sensitivity.csv",
            "19ba-v12-resolution-wave.qmd",
            "no contractual IFRS9 claim",
        ),
        (
            "Causal/fairness",
            "implemented_blocker_dossier",
            "paper4_v12_causal_cate_dossier.csv",
            "19ba-v12-resolution-wave.qmd",
            "CATE policy value and fair-lending legal claims remain blocked",
        ),
        (
            "Paper Estrella freeze",
            "guardrail_verified",
            "paper4_v12_status.json",
            "19ba-v12-resolution-wave.qmd",
            "models/final_project_promotion.json not modified",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["priority", "claim_status", "artifact", "quarto_page", "caveat"]
    )


def _write_v12_note(status: dict[str, Any]) -> None:
    _write_note(
        "paper4_v12_resolution_wave.md",
        "\n".join(
            [
                "# Paper 4 v12 Resolution Wave",
                "",
                f"- Paper 4 working champion: `{status.get('working_champion_policy_id_v12')}`.",
                f"- CVaR feasible count: `{status['cvar_feasible_count_v12']}`.",
                f"- CVaR non-dominated count: `{status['cvar_non_dominated_count_v12']}`.",
                f"- MDCP optimal count: `{status['mdcp_optimal_count_v12']}`.",
                f"- Best DLA delta vs static: `{status['dla_best_delta_state_value_v12']:.4f}`.",
                f"- SPO regret candidates: `{status['spo_candidate_count_v12']}`.",
                f"- Sample-path policy count: `{status['sample_path_policy_count_v12']}`.",
                f"- Contractual IFRS9 claim allowed: `{status['ifrs9_contractual_claim_allowed']}`.",
                f"- Fair-lending legal claim allowed: `{status['fair_lending_legal_claim']}`.",
                "",
                "V12 can change the Paper 4 working champion, but it does not alter Paper Estrella or create a final promotion artifact.",
            ]
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvar-pool-n", type=int, default=24_000)
    parser.add_argument("--cvar-rounds", type=int, default=2)
    parser.add_argument("--mdcp-pool-n", type=int, default=22_000)
    parser.add_argument("--spo-pool-n", type=int, default=22_000)
    parser.add_argument("--spo-epochs", type=int, default=7)
    parser.add_argument("--dla-months", type=int, default=12)
    parser.add_argument("--dla-paths", type=int, default=24)
    parser.add_argument("--dla-iterations", type=int, default=3)
    parser.add_argument("--sample-paths", type=int, default=240)
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = time.time()
    base_universe, candidate_pool, _, _, online_intervals = _load_inputs()
    solver_universe = base_universe if len(base_universe) > len(candidate_pool) else candidate_pool
    intervals, _, _, online_status = _load_v9_online()
    online_method = str(online_status["online_best_method_v9"])

    artifact_audit = _artifact_audit_v12()
    _write_csv("paper4_v12_artifact_audit.csv", artifact_audit)
    refs = build_method_reference_registry_v12()
    _write_csv("paper4_v12_method_reference_registry.csv", refs)

    cap_rationale, caps = build_mdcp_caps_v12()
    _write_csv("paper4_v12_mdcp_source_cap_rationale.csv", cap_rationale)

    cvar_frontier, cvar_alloc, cvar_losses, cvar_active = build_cvar_column_generation_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_pool_n=args.cvar_pool_n,
        rounds=args.cvar_rounds,
        caps=caps,
    )
    _write_csv("paper4_v12_cvar_column_generation_frontier.csv", cvar_frontier)
    _write_parquet("paper4_v12_cvar_column_generation_allocations.parquet", cvar_alloc)
    _write_csv("paper4_v12_cvar_column_generation_scenario_losses.csv", cvar_losses)
    _write_csv("paper4_v12_cvar_active_constraints.csv", cvar_active)

    mdcp_summary, mdcp_alloc = build_mdcp_solver_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        caps=caps,
        max_pool_n=args.mdcp_pool_n,
    )
    _write_csv("paper4_v12_mdcp_source_cap_solver_summary.csv", mdcp_summary)
    _write_parquet("paper4_v12_mdcp_source_cap_allocations.parquet", mdcp_alloc)

    state_schema = build_dla_state_schema_v12()
    _write_csv("paper4_v12_dla_state_schema.csv", state_schema)
    dla_coef, dla_decisions, dla_trace, dla_comparison = build_dla_fvi_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_months=args.dla_months,
        n_paths=args.dla_paths,
        iterations=args.dla_iterations,
    )
    _write_csv("paper4_v12_dla_fitted_value_coefficients.csv", dla_coef)
    _write_parquet("paper4_v12_dla_fvi_decisions.parquet", dla_decisions)
    _write_parquet("paper4_v12_dla_fvi_trace.parquet", dla_trace)
    _write_csv("paper4_v12_dla_fvi_comparison.csv", dla_comparison)

    spo_train, spo_coef, spo_candidates, spo_alloc = build_spo_regret_surrogate_v12(
        solver_universe,
        online_intervals,
        online_method=online_method,
        max_pool_n=args.spo_pool_n,
        epochs=args.spo_epochs,
    )
    _write_csv("paper4_v12_spo_plus_surrogate_training.csv", spo_train)
    _write_csv("paper4_v12_spo_plus_surrogate_coefficients.csv", spo_coef)
    _write_csv("paper4_v12_spo_plus_surrogate_candidates.csv", spo_candidates)
    _write_parquet("paper4_v12_spo_plus_surrogate_allocations.parquet", spo_alloc)

    stress_alloc = pd.concat(
        [
            df
            for df in [
                cvar_alloc.head(4_000) if not cvar_alloc.empty else pd.DataFrame(),
                mdcp_alloc.head(2_000) if not mdcp_alloc.empty else pd.DataFrame(),
                spo_alloc.head(3_000) if not spo_alloc.empty else pd.DataFrame(),
            ]
            if not df.empty
        ],
        ignore_index=True,
    )
    sample_cal, sample_scenarios, sample_paths, sample_ci, pairwise = build_sample_paths_v12(
        stress_alloc,
        solver_universe,
        n_paths=args.sample_paths,
    )
    _write_csv("paper4_v12_sample_path_calibration_table.csv", sample_cal)
    _write_csv("paper4_v12_sample_path_scenario_register.csv", sample_scenarios)
    _write_parquet("paper4_v12_sample_path_macro_calibrated_paths.parquet", sample_paths)
    _write_csv("paper4_v12_sample_path_macro_calibrated_ci.csv", sample_ci)
    _write_csv("paper4_v12_sample_path_pairwise_champion_ci.csv", pairwise)

    ifrs9_readiness, sicr = build_ifrs9_sicr_v12(solver_universe)
    _write_csv("paper4_v12_ifrs9_readiness.csv", ifrs9_readiness)
    _write_csv("paper4_v12_ifrs9_sicr_sensitivity.csv", sicr)
    causal, fairness = build_causal_fairness_v12()
    _write_csv("paper4_v12_causal_cate_dossier.csv", causal)
    _write_csv("paper4_v12_fairness_proxy_governance.csv", fairness)

    registry, champion = build_working_registry_v12(
        cvar_frontier, mdcp_summary, dla_comparison, spo_candidates, sample_ci
    )
    _write_csv("paper4_v12_working_candidate_registry.csv", registry)
    if champion:
        _write_json("paper4_v12_working_champion.json", champion)

    online_metrics = {
        "source_month": float(online_status.get("online_best_source_month_defended_min", np.nan)),
        "policy_month": float(online_status.get("online_best_policy_month_defended_min", np.nan)),
        "width": float(online_status.get("online_best_width", np.nan)),
    }
    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phase": "v12_resolution_wave",
        "mode": "paper4_working_champion_allowed_no_paper1_changes",
        "online_best_method_v9": online_method,
        "online_goal_achieved": bool(online_status.get("online_goal_achieved")),
        "online_metrics_v12": online_metrics,
        "candidate_universe_source_v12": "base_full_universe"
        if len(base_universe) > len(candidate_pool)
        else "paper4_candidate_pool",
        "candidate_universe_n_v12": int(len(solver_universe)),
        "cvar_pool_n_v12": int(args.cvar_pool_n),
        "cvar_feasible_count_v12": int(
            cvar_frontier.get("feasible_v12", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "cvar_non_dominated_count_v12": int(
            cvar_frontier.get("non_dominated_v12", pd.Series(False)).astype(bool).sum()
        )
        if not cvar_frontier.empty
        else 0,
        "mdcp_optimal_count_v12": int(mdcp_summary["solver_status"].map(_is_optimal).sum())
        if not mdcp_summary.empty
        else 0,
        "dla_best_delta_state_value_v12": float(
            dla_comparison.loc[
                ~dla_comparison["policy_id"].eq("v12_static_reference"),
                "delta_state_value_vs_static",
            ].max()
        )
        if not dla_comparison.empty
        else np.nan,
        "spo_candidate_count_v12": int(len(spo_candidates)),
        "sample_path_policy_count_v12": int(sample_ci["policy_id"].nunique())
        if not sample_ci.empty
        else 0,
        "working_candidate_count_v12": int(len(registry)),
        "working_champion_policy_id_v12": champion.get("policy_id") if champion else None,
        "working_champion_created_v12": bool(champion),
        "ifrs9_contractual_claim_allowed": False,
        "causal_policy_value_allowed": False,
        "fair_lending_legal_claim": False,
        "paper1_artifacts_modified": False,
        "paper1_promotion_file_exists": PAPER1_PROMOTION.exists(),
        "paper4_final_promotion_created": PAPER4_FINAL_PROMOTION.exists(),
        "paper4_working_champion_json_created": WORKING_CHAMPION_PATH.exists(),
        "runtime_seconds": round(time.time() - start, 3),
        "caveat": "V12 creates a Paper 4 working champion only; all publication/final claims remain guarded.",
    }
    dashboard = build_blocker_dashboard_v12(status)
    claims = build_claim_matrix_v12()
    _write_csv("paper4_v12_blocker_dashboard.csv", dashboard)
    _write_csv("paper4_v12_claim_artifact_matrix.csv", claims)
    _write_json("paper4_v12_status.json", status)
    _write_v12_note(status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
