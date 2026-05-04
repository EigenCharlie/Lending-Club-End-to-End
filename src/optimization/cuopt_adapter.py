"""Native cuOpt adapter for portfolio LP solves.

This bypasses the fragile Pyomo -> cuOpt integration and builds the LP
directly with cuOpt's Python API. The model matches the continuous LP used by
the canonical portfolio optimization path.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix


def _require_cuopt():
    try:
        from cuopt import linear_programming as lp_api
    except Exception as exc:  # pragma: no cover - exercised in RAPIDS env only
        raise RuntimeError(
            "solver_backend='cuopt' requested but native cuOpt Python API is not available."
        ) from exc
    return lp_api


def _extract_primal_solution(solution: Any, n_vars: int) -> np.ndarray:
    values = np.asarray(solution.get_primal_solution(), dtype=float)
    if values.ndim != 1 or len(values) < n_vars:
        raise RuntimeError(
            f"cuOpt primal solution has unexpected shape {values.shape}; expected >= {n_vars}."
        )
    return values


def solve_portfolio_cuopt_native(
    *,
    loans: pd.DataFrame,
    pd_point: np.ndarray,
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
    random_seed: int | None = None,
    presolve: int | None = 1,
) -> dict[str, Any]:
    """Solve the portfolio LP natively with cuOpt."""
    lp_api = _require_cuopt()

    n = len(loans)
    if n == 0:
        raise ValueError("Cannot solve empty portfolio.")

    loan_amounts = (
        loans["loan_amnt"].to_numpy(dtype=float)
        if "loan_amnt" in loans.columns
        else np.ones(n, dtype=float) * 10_000.0
    )
    point = np.asarray(pd_point, dtype=float)
    high = np.asarray(pd_high, dtype=float)
    lgd_arr = np.asarray(lgd, dtype=float)
    rates = np.asarray(int_rates, dtype=float)
    pd_constraint = (
        np.asarray(pd_constraint_override, dtype=float)
        if pd_constraint_override is not None
        else (high if robust else point)
    )
    pd_uncertainty = np.clip(high - point, 0.0, 1.0)

    use_pd_slack = float(pd_cap_slack_penalty) > 0
    obj = loan_amounts * (rates - point * lgd_arr - uncertainty_aversion * pd_uncertainty * lgd_arr)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    row_types: list[str] = []

    # Budget cap
    rows.append(loan_amounts.astype(float))
    rhs.append(float(total_budget))
    row_types.append("L")

    # Optional minimum budget utilization
    min_budget_utilization = float(np.clip(min_budget_utilization, 0.0, 1.0))
    if min_budget_utilization > 0:
        rows.append((-loan_amounts).astype(float))
        rhs.append(float(-min_budget_utilization * total_budget))
        row_types.append("L")

    # Portfolio PD cap: sum(x_i * loan_i * (pd_i - max_pd)) - slack <= 0
    pd_row = loan_amounts * (pd_constraint - float(max_portfolio_pd))
    rows.append(pd_row.astype(float))
    rhs.append(0.0)
    row_types.append("L")

    if "purpose" in loans.columns:
        purposes = loans["purpose"].fillna("unknown").astype(str)
        top_purposes = purposes.unique()
        for purpose in top_purposes:
            mask = (purposes == purpose).to_numpy(dtype=float)
            row = loan_amounts * (mask - float(max_concentration))
            rows.append(row.astype(float))
            rhs.append(0.0)
            row_types.append("L")

    A = np.vstack(rows).astype(np.float64)

    var_lb = np.zeros(n + int(use_pd_slack), dtype=np.float64)
    var_ub = np.ones(n + int(use_pd_slack), dtype=np.float64)
    if use_pd_slack:
        slack_col = np.zeros((A.shape[0], 1), dtype=np.float64)
        pd_cap_row_idx = 2 if min_budget_utilization > 0 else 1
        slack_col[pd_cap_row_idx, 0] = -1.0
        A = np.hstack([A, slack_col])
        obj = np.concatenate([obj.astype(np.float64), np.array([-float(pd_cap_slack_penalty)])])
        var_ub[-1] = float(total_budget)
    else:
        obj = obj.astype(np.float64)

    A_csr = csr_matrix(A)
    dm = lp_api.DataModel()
    dm.set_csr_constraint_matrix(
        A_csr.data.astype(np.float64),
        A_csr.indices.astype(np.int32),
        A_csr.indptr.astype(np.int32),
    )
    dm.set_constraint_bounds(np.asarray(rhs, dtype=np.float64))
    dm.set_row_types(np.asarray(row_types))
    dm.set_objective_coefficients(obj)
    dm.set_maximize(True)
    dm.set_variable_lower_bounds(var_lb)
    dm.set_variable_upper_bounds(var_ub)

    settings = lp_api.SolverSettings()
    with suppress(Exception):
        settings.set_parameter("log_to_console", False)
    settings.set_parameter("time_limit", int(time_limit))
    if random_seed is not None:
        with suppress(Exception):
            settings.set_parameter("random_seed", int(random_seed))
    if presolve is not None:
        with suppress(Exception):
            settings.set_parameter("presolve", int(presolve))

    solution = lp_api.Solve(dm, settings)
    termination_reason = str(solution.get_termination_reason())
    if "Optimal" not in termination_reason and "Feasible" not in termination_reason:
        raise RuntimeError(
            f"cuOpt solve did not produce an acceptable solution: {termination_reason}"
        )

    primal = _extract_primal_solution(solution, n + int(use_pd_slack))
    x = primal[:n]
    pd_cap_slack = float(primal[-1]) if use_pd_slack else 0.0
    allocation = {i: float(x[i]) for i in range(n)}
    total_allocated = float(np.sum(x * loan_amounts))
    n_funded = int(np.sum(x > 0.01))
    obj_value = float(solution.get_primal_objective())

    logger.info(
        "Portfolio solved (cuopt_native): obj={:,.2f}, funded={}/{}, allocated={:,.0f}, pd_cap_slack={:.4f}",
        obj_value,
        n_funded,
        n,
        total_allocated,
        pd_cap_slack,
    )

    return {
        "allocation": allocation,
        "objective_value": obj_value,
        "n_funded": n_funded,
        "total_allocated": total_allocated,
        "solver_status": termination_reason,
        "solver_backend": "cuopt",
        "pd_cap_slack": pd_cap_slack,
    }
