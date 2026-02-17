"""Robust optimization using conformal prediction uncertainty sets.

Converts CP intervals into uncertainty sets for robust portfolio optimization.
Supports box uncertainty (intervals) and ellipsoidal uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_box_uncertainty_set(
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    lgd_low: np.ndarray | None = None,
    lgd_high: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build box uncertainty set from conformal prediction intervals.

    The box set is: {PD : PD_low <= PD <= PD_high} for each loan.
    This is directly from split conformal prediction with coverage guarantee.
    """
    uncertainty_set = {
        "pd_low": pd_low,
        "pd_high": pd_high,
        "pd_center": (pd_low + pd_high) / 2,
        "pd_radius": (pd_high - pd_low) / 2,
    }

    if lgd_low is not None and lgd_high is not None:
        uncertainty_set.update(
            {
                "lgd_low": lgd_low,
                "lgd_high": lgd_high,
                "lgd_center": (lgd_low + lgd_high) / 2,
                "lgd_radius": (lgd_high - lgd_low) / 2,
            }
        )

    logger.info(
        f"Box uncertainty set: {len(pd_low)} loans, "
        f"avg PD width={uncertainty_set['pd_radius'].mean() * 2:.4f}"
    )
    return uncertainty_set


def worst_case_expected_loss(
    allocation: np.ndarray,
    loan_amounts: np.ndarray,
    pd_high: np.ndarray,
    lgd_high: np.ndarray | None = None,
    lgd_point: np.ndarray | None = None,
) -> float:
    """Compute worst-case expected loss under box uncertainty.

    Uses upper bounds from conformal prediction for conservative estimate.
    """
    lgd = (
        lgd_high
        if lgd_high is not None
        else (lgd_point if lgd_point is not None else np.ones_like(pd_high) * 0.45)
    )
    return float(np.sum(allocation * loan_amounts * pd_high * lgd))


def scenario_analysis(
    allocation: np.ndarray,
    loan_amounts: np.ndarray,
    pd_low: np.ndarray,
    pd_point: np.ndarray,
    pd_high: np.ndarray,
    lgd: np.ndarray,
) -> pd.DataFrame:
    """Run scenario analysis: best-case, expected, worst-case."""
    scenarios = {
        "best_case": np.sum(allocation * loan_amounts * pd_low * lgd),
        "expected": np.sum(allocation * loan_amounts * pd_point * lgd),
        "worst_case": np.sum(allocation * loan_amounts * pd_high * lgd),
    }
    result = pd.DataFrame([scenarios])
    result["range"] = result["worst_case"] - result["best_case"]
    logger.info(f"Scenario analysis: {scenarios}")
    return result
