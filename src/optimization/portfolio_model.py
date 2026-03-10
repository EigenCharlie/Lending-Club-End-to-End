"""Portfolio optimization using Pyomo + HiGHS.

Maximizes expected return net of expected loss under credit constraints.
Supports multiple uncertainty policies for PD constraints:
- point_estimate: uses point PD only
- hard_worst_case: uses conformal upper bound
- blended_uncertainty: interpolates between point and upper bound
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from loguru import logger


def compute_effective_pd(
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    *,
    policy_mode: str = "hard_worst_case",
    gamma: float = 1.0,
) -> np.ndarray:
    """Resolve the PD vector used in the portfolio PD constraint.

    Args:
        pd_point: Point PD estimates.
        pd_high: Upper conformal PD bound.
        policy_mode: `point_estimate`, `hard_worst_case`, or `blended_uncertainty`.
        gamma: Blend weight for `blended_uncertainty`.
    """
    point = np.asarray(pd_point, dtype=float)
    high = np.asarray(pd_high, dtype=float)
    mode = str(policy_mode).strip().lower()
    if mode in {"point", "point_estimate", "nonrobust"}:
        return point
    if mode in {"hard_worst_case", "worst_case", "robust"}:
        return high
    if mode == "blended_uncertainty":
        weight = float(np.clip(gamma, 0.0, 1.0))
        return np.clip(point + weight * np.clip(high - point, 0.0, 1.0), 0.0, 1.0)
    raise ValueError(
        f"Unsupported policy_mode={policy_mode!r}. "
        "Use 'point_estimate', 'hard_worst_case', or 'blended_uncertainty'."
    )


def build_portfolio_model(
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
    int_rates: np.ndarray,
    total_budget: float = 1_000_000,
    max_concentration: float = 0.25,
    max_portfolio_pd: float = 0.10,
    robust: bool = True,
    uncertainty_aversion: float = 0.0,
    min_budget_utilization: float = 0.0,
    pd_cap_slack_penalty: float = 0.0,
    pd_constraint_override: np.ndarray | None = None,
) -> pyo.ConcreteModel:
    """Build Pyomo portfolio optimization model.

    Args:
        loans: DataFrame with loan features.
        pd_point: Point PD estimates.
        pd_low: Lower bound PD from conformal prediction.
        pd_high: Upper bound PD from conformal prediction.
        lgd: Loss Given Default estimates.
        int_rates: Interest rates (expected return).
        total_budget: Total capital to allocate.
        max_concentration: Maximum fraction per purpose segment.
        max_portfolio_pd: Maximum portfolio-level default rate.
        robust: If True, use pd_high for risk constraints.
        uncertainty_aversion: Linear penalty weight on PD uncertainty in the objective.
        min_budget_utilization: Optional minimum budget utilization in [0, 1].
        pd_cap_slack_penalty: Optional penalty for weighted-PD cap slack.

    Returns:
        Pyomo ConcreteModel ready to solve.
    """
    n = len(loans)
    model = pyo.ConcreteModel("CreditPortfolioOptimization")

    model.I = pyo.RangeSet(0, n - 1)

    model.int_rate = pyo.Param(model.I, initialize=dict(enumerate(int_rates)))
    pd_constraint = (
        np.asarray(pd_constraint_override, dtype=float)
        if pd_constraint_override is not None
        else (pd_high if robust else pd_point)
    )
    model.pd_point = pyo.Param(model.I, initialize=dict(enumerate(pd_point)))
    model.pd_worst = pyo.Param(model.I, initialize=dict(enumerate(pd_constraint)))
    pd_uncertainty = np.clip(pd_high - pd_point, 0.0, 1.0)
    model.pd_uncertainty = pyo.Param(model.I, initialize=dict(enumerate(pd_uncertainty)))
    model.lgd = pyo.Param(model.I, initialize=dict(enumerate(lgd)))
    model.loan_amnt = pyo.Param(
        model.I,
        initialize=dict(
            enumerate(loans["loan_amnt"].values if "loan_amnt" in loans.columns else np.ones(n))
        ),
    )

    # x[i] = fraction of loan i to fund
    model.x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
    use_pd_slack = pd_cap_slack_penalty > 0
    if use_pd_slack:
        # Slack in weighted-PD units to avoid degenerate zero-investment solutions.
        model.pd_cap_slack = pyo.Var(domain=pyo.NonNegativeReals)

    def objective_rule(m):
        base = sum(
            m.x[i]
            * m.loan_amnt[i]
            * (
                m.int_rate[i]
                - m.pd_point[i] * m.lgd[i]
                - uncertainty_aversion * m.pd_uncertainty[i] * m.lgd[i]
            )
            for i in m.I
        )
        if use_pd_slack:
            return base - pd_cap_slack_penalty * m.pd_cap_slack
        return base

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def budget_rule(m):
        return sum(m.x[i] * m.loan_amnt[i] for i in m.I) <= total_budget

    model.budget = pyo.Constraint(rule=budget_rule)

    min_budget_utilization = float(np.clip(min_budget_utilization, 0.0, 1.0))
    if min_budget_utilization > 0:

        def min_budget_rule(m):
            return (
                sum(m.x[i] * m.loan_amnt[i] for i in m.I) >= min_budget_utilization * total_budget
            )

        model.min_budget = pyo.Constraint(rule=min_budget_rule)

    def pd_cap_rule(m):
        total_exposure = sum(m.x[i] * m.loan_amnt[i] for i in m.I) + 1e-6
        weighted_pd = sum(m.x[i] * m.loan_amnt[i] * m.pd_worst[i] for i in m.I)
        rhs = max_portfolio_pd * total_exposure
        if use_pd_slack:
            rhs = rhs + m.pd_cap_slack
        return weighted_pd <= rhs

    model.pd_cap = pyo.Constraint(rule=pd_cap_rule)

    if "purpose" in loans.columns:
        purposes = loans["purpose"].fillna("unknown").astype(str).unique()
        loan_purpose = loans["purpose"].fillna("unknown").astype(str).values
        for p_idx, purpose in enumerate(purposes):
            mask = [i for i in range(n) if loan_purpose[i] == purpose]

            def concentration_rule(m, _idx=None, _mask=mask):
                total = sum(m.x[i] * m.loan_amnt[i] for i in m.I) + 1e-6
                sector = sum(m.x[i] * m.loan_amnt[i] for i in _mask)
                return sector <= max_concentration * total

            setattr(model, f"concentration_{p_idx}", pyo.Constraint(rule=concentration_rule))

    logger.info(
        f"Built portfolio model: {n} loans, budget={total_budget:,.0f}, robust={robust}, "
        f"uncertainty_aversion={uncertainty_aversion:.3f}, "
        f"min_budget_utilization={min_budget_utilization:.3f}, "
        f"pd_cap_slack_penalty={pd_cap_slack_penalty:.3f}"
    )
    return model


def solve_portfolio(
    model: pyo.ConcreteModel,
    time_limit: int = 300,
    threads: int = 4,
    solver_backend: str = "highs",
) -> dict[str, Any]:
    """Solve portfolio optimization with HiGHS (default) or optional cuOpt."""
    backend = solver_backend.strip().lower()
    if backend == "highs":
        from pyomo.contrib.appsi.solvers import Highs

        solver = Highs()
        solver.config.time_limit = time_limit
        _ = threads  # reserved for future HiGHS appsi configuration
        results = solver.solve(model)
    elif backend == "cuopt":
        solver = pyo.SolverFactory("cuopt")
        if solver is None or not solver.available(False):
            raise RuntimeError(
                "solver_backend='cuopt' requested but Pyomo cuOpt solver is not available "
                "in this environment."
            )
        _ = (time_limit, threads)  # backend-specific options vary by cuOpt deployment
        results = solver.solve(model)
    else:
        raise ValueError(f"Unsupported solver_backend={solver_backend!r}. Use 'highs' or 'cuopt'.")

    allocation = {i: pyo.value(model.x[i]) for i in model.I}
    obj_value = pyo.value(model.obj)
    n_funded = sum(1 for v in allocation.values() if v > 0.01)
    total_allocated = sum(allocation[i] * pyo.value(model.loan_amnt[i]) for i in model.I)
    pd_cap_slack = float(pyo.value(model.pd_cap_slack)) if hasattr(model, "pd_cap_slack") else 0.0
    termination = getattr(results, "termination_condition", None)
    if termination is None and hasattr(results, "solver"):
        termination = getattr(results.solver, "termination_condition", None)

    solution = {
        "allocation": allocation,
        "objective_value": float(obj_value),
        "n_funded": int(n_funded),
        "total_allocated": float(total_allocated),
        "solver_status": str(termination) if termination is not None else "unknown",
        "solver_backend": backend,
        "pd_cap_slack": pd_cap_slack,
    }

    logger.info(
        f"Portfolio solved ({backend}): obj={obj_value:,.2f}, funded={n_funded}/{len(allocation)}, "
        f"allocated={total_allocated:,.0f}, pd_cap_slack={pd_cap_slack:.4f}"
    )
    return solution


def optimize_portfolio_allocation(
    *,
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
    int_rates: np.ndarray,
    total_budget: float = 1_000_000,
    max_concentration: float = 0.25,
    max_portfolio_pd: float = 0.10,
    robust: bool = True,
    uncertainty_aversion: float = 0.0,
    min_budget_utilization: float = 0.0,
    pd_cap_slack_penalty: float = 0.0,
    pd_constraint_override: np.ndarray | None = None,
    time_limit: int = 300,
    threads: int = 4,
    solver_backend: str = "highs",
) -> dict[str, Any]:
    """Unified portfolio solve entrypoint for CPU and native cuOpt backends."""
    backend = solver_backend.strip().lower()
    if backend == "cuopt":
        from src.optimization.cuopt_adapter import solve_portfolio_cuopt_native

        return solve_portfolio_cuopt_native(
            loans=loans,
            pd_point=pd_point,
            pd_high=pd_high,
            lgd=lgd,
            int_rates=int_rates,
            total_budget=total_budget,
            max_concentration=max_concentration,
            max_portfolio_pd=max_portfolio_pd,
            robust=robust,
            uncertainty_aversion=uncertainty_aversion,
            min_budget_utilization=min_budget_utilization,
            pd_cap_slack_penalty=pd_cap_slack_penalty,
            pd_constraint_override=pd_constraint_override,
            time_limit=time_limit,
        )

    model = build_portfolio_model(
        loans=loans,
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        lgd=lgd,
        int_rates=int_rates,
        total_budget=total_budget,
        max_concentration=max_concentration,
        max_portfolio_pd=max_portfolio_pd,
        robust=robust,
        uncertainty_aversion=uncertainty_aversion,
        min_budget_utilization=min_budget_utilization,
        pd_cap_slack_penalty=pd_cap_slack_penalty,
        pd_constraint_override=pd_constraint_override,
    )
    return solve_portfolio(
        model,
        time_limit=time_limit,
        threads=threads,
        solver_backend=backend,
    )


def build_binary_model(
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
    int_rates: np.ndarray,
    total_budget: float = 1_000_000,
    max_portfolio_pd: float = 0.10,
) -> pyo.ConcreteModel:
    """Build MILP approve/reject model (binary decisions)."""
    n = len(loans)
    model = pyo.ConcreteModel("CreditApprovalMILP")

    model.I = pyo.RangeSet(0, n - 1)
    model.int_rate = pyo.Param(model.I, initialize=dict(enumerate(int_rates)))
    model.pd_point = pyo.Param(model.I, initialize=dict(enumerate(pd_point)))
    model.pd_high = pyo.Param(model.I, initialize=dict(enumerate(pd_high)))
    model.lgd = pyo.Param(model.I, initialize=dict(enumerate(lgd)))
    model.loan_amnt = pyo.Param(model.I, initialize=dict(enumerate(loans["loan_amnt"].values)))
    model.x = pyo.Var(model.I, domain=pyo.Binary)

    def objective_rule(m):
        return sum(
            m.x[i] * m.loan_amnt[i] * (m.int_rate[i] - m.pd_point[i] * m.lgd[i]) for i in m.I
        )

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    def budget_rule(m):
        return sum(m.x[i] * m.loan_amnt[i] for i in m.I) <= total_budget

    model.budget = pyo.Constraint(rule=budget_rule)

    def pd_cap_rule(m):
        total = sum(m.x[i] * m.loan_amnt[i] for i in m.I) + 1e-6
        weighted = sum(m.x[i] * m.loan_amnt[i] * m.pd_high[i] for i in m.I)
        return weighted <= max_portfolio_pd * total

    model.pd_cap = pyo.Constraint(rule=pd_cap_rule)

    logger.info(f"Built binary approval model: {n} loans")
    return model
