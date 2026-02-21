"""CATE-adjusted portfolio optimization.

Bridges the gap between causal inference (NB07) and portfolio optimization
(NB08) by adjusting PD and interest rates using heterogeneous treatment
effects before feeding into the Pyomo optimizer.

Key idea: for loans where CATE > 0 (rate increase raises default risk),
a targeted rate reduction can lower expected default while maintaining
acceptable return. This enables the optimizer to fund more loans.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.optimization.portfolio_model import build_portfolio_model, solve_portfolio


def apply_cate_adjustment(
    pd_point: np.ndarray,
    int_rates: np.ndarray,
    cate: np.ndarray,
    delta_rate: float = -1.0,
    clip_pd_min: float = 0.001,
    clip_pd_max: float = 0.999,
    clip_rate_min: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CATE-adjusted PD and interest rates.

    For each loan, if CATE > 0 (higher rate increases default risk),
    apply a rate reduction and estimate the resulting PD decrease.

    Note: this is a linear approximation. CATE is a local treatment effect
    estimate, not a structural causal model. The adjustment is valid under
    the assumption that the causal effect is approximately linear near the
    current rate.

    Args:
        pd_point: Point PD estimates per loan.
        int_rates: Current interest rates per loan (decimal, e.g. 0.12).
        cate: Per-loan CATE (effect of +1 unit treatment on default probability).
        delta_rate: Rate change to apply (negative = reduction, in same units as T).
        clip_pd_min: Minimum allowed PD after adjustment.
        clip_pd_max: Maximum allowed PD after adjustment.
        clip_rate_min: Minimum allowed interest rate after adjustment.

    Returns:
        Tuple of (pd_adjusted, rates_adjusted), both clipped to valid ranges.
    """
    pd_point = np.asarray(pd_point, dtype=float)
    int_rates = np.asarray(int_rates, dtype=float)
    cate = np.asarray(cate, dtype=float)

    # Only adjust loans where CATE > 0 (rate-sensitive borrowers)
    eligible = cate > 0
    pd_shift = np.where(eligible, cate * delta_rate, 0.0)

    pd_adjusted = np.clip(pd_point + pd_shift, clip_pd_min, clip_pd_max)
    rates_adjusted = np.clip(
        np.where(eligible, int_rates + delta_rate / 100.0, int_rates),
        clip_rate_min,
        None,
    )

    n_eligible = int(eligible.sum())
    avg_pd_reduction = (
        float(np.mean(pd_point[eligible] - pd_adjusted[eligible])) if n_eligible > 0 else 0.0
    )
    logger.info(
        f"CATE adjustment: {n_eligible}/{len(cate)} loans eligible, "
        f"avg PD reduction: {avg_pd_reduction:.6f}"
    )

    return pd_adjusted, rates_adjusted


def build_cate_adjusted_portfolio(
    loans: pd.DataFrame,
    pd_point: np.ndarray,
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    cate: np.ndarray,
    lgd: np.ndarray,
    int_rates: np.ndarray,
    delta_rate: float = -1.0,
    total_budget: float = 1_000_000,
    max_concentration: float = 0.25,
    max_portfolio_pd: float = 0.10,
    robust: bool = True,
    uncertainty_aversion: float = 0.0,
) -> dict[str, Any]:
    """Run baseline and CATE-adjusted portfolio optimization and compare.

    Calls build_portfolio_model() + solve_portfolio() twice:
    - Baseline: original PD and interest rates
    - CATE-adjusted: reduced PD and rates for eligible loans

    Args:
        loans: DataFrame with loan features (must include loan_amnt, purpose).
        pd_point: Point PD estimates.
        pd_low: Lower conformal bound.
        pd_high: Upper conformal bound.
        cate: Per-loan CATE estimates.
        lgd: Loss Given Default estimates.
        int_rates: Interest rates (decimal).
        delta_rate: Rate change for CATE adjustment (negative = reduction).
        total_budget: Total capital to allocate.
        max_concentration: Maximum per-purpose concentration.
        max_portfolio_pd: Maximum portfolio default rate.
        robust: Whether to use pd_high for PD constraint.
        uncertainty_aversion: Penalty weight on PD uncertainty.

    Returns:
        Dict with keys: baseline, cate_adjusted, comparison_df.
    """
    common_kwargs = {
        "loans": loans,
        "lgd": lgd,
        "total_budget": total_budget,
        "max_concentration": max_concentration,
        "max_portfolio_pd": max_portfolio_pd,
        "robust": robust,
        "uncertainty_aversion": uncertainty_aversion,
    }

    # Baseline portfolio
    logger.info("Building baseline portfolio (no CATE adjustment)")
    model_base = build_portfolio_model(
        pd_point=pd_point,
        pd_low=pd_low,
        pd_high=pd_high,
        int_rates=int_rates,
        **common_kwargs,
    )
    sol_base = solve_portfolio(model_base)

    # CATE-adjusted portfolio
    pd_adj, rates_adj = apply_cate_adjustment(
        pd_point,
        int_rates,
        cate,
        delta_rate=delta_rate,
    )
    # Shift conformal bounds by the same amount
    pd_shift = pd_adj - pd_point
    pd_low_adj = np.clip(pd_low + pd_shift, 0.001, 0.999)
    pd_high_adj = np.clip(pd_high + pd_shift, 0.001, 0.999)

    logger.info("Building CATE-adjusted portfolio")
    model_adj = build_portfolio_model(
        pd_point=pd_adj,
        pd_low=pd_low_adj,
        pd_high=pd_high_adj,
        int_rates=rates_adj,
        **common_kwargs,
    )
    sol_adj = solve_portfolio(model_adj)

    # Build comparison DataFrame
    comparison = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                "objective_value": sol_base["objective_value"],
                "n_funded": sol_base["n_funded"],
                "total_allocated": sol_base["total_allocated"],
            },
            {
                "scenario": "cate_adjusted",
                "objective_value": sol_adj["objective_value"],
                "n_funded": sol_adj["n_funded"],
                "total_allocated": sol_adj["total_allocated"],
            },
        ]
    )

    logger.info(
        f"Comparison — Baseline obj: {sol_base['objective_value']:,.2f}, "
        f"CATE-adjusted obj: {sol_adj['objective_value']:,.2f}"
    )

    return {
        "baseline": sol_base,
        "cate_adjusted": sol_adj,
        "comparison_df": comparison,
        "delta_rate_used": delta_rate,
    }
