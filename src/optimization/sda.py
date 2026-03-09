"""Sequential Decision Analytics for dynamic credit policy.

Uses forecasted default rates with conformal intervals to make
dynamic lending decisions over time (multi-period optimization).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def _extract_policy_forecast_values(row: pd.Series) -> tuple[float, float, str, str, str]:
    """Resolve canonical forecast columns first, then fall back to legacy model outputs."""
    point_model = str(row.get("point_model", "canonical") or "canonical")
    interval_model = str(row.get("interval_model", point_model) or point_model)
    official_status = str(row.get("official_status", "unknown") or "unknown")

    forecast_pd = row.get("y")
    if pd.isna(forecast_pd):
        forecast_pd = row.get(point_model) if point_model in row.index else 0.0
    forecast_pd = float(forecast_pd) if pd.notna(forecast_pd) else 0.0

    pd_upper = row.get("y_hi_90")
    interval_col = f"{interval_model}-hi-90"
    if pd.isna(pd_upper):
        pd_upper = row.get(interval_col) if interval_col in row.index else forecast_pd * 1.2
    pd_upper = float(pd_upper) if pd.notna(pd_upper) else forecast_pd * 1.2

    return forecast_pd, pd_upper, point_model, interval_model, official_status


def dynamic_credit_policy(
    forecasts: pd.DataFrame,
    current_portfolio_pd: float,
    target_pd: float = 0.08,
    max_monthly_origination: float = 10_000_000,
) -> pd.DataFrame:
    """Generate monthly lending policy based on forecasted default rates.

    If forecasted default rate (with upper CP bound) exceeds target,
    tighten lending criteria; if below, expand.

    Args:
        forecasts: DataFrame with columns [ds, y, y_lo_90, y_hi_90].
        current_portfolio_pd: Current portfolio default rate.
        target_pd: Target maximum portfolio default rate.
        max_monthly_origination: Maximum monthly origination volume.

    Returns:
        Policy DataFrame with recommended actions per month.
    """
    policies = []
    for _, row in forecasts.iterrows():
        forecast_pd, pd_upper, point_model, interval_model, official_status = (
            _extract_policy_forecast_values(row)
        )

        # Decision based on worst-case PD vs target
        if pd_upper > target_pd * 1.2:
            action = "TIGHTEN"
            origination_factor = 0.6
            min_grade = "B"
        elif pd_upper > target_pd:
            action = "CAUTIOUS"
            origination_factor = 0.85
            min_grade = "C"
        elif forecast_pd < target_pd * 0.7:
            action = "EXPAND"
            origination_factor = 1.15
            min_grade = "E"
        else:
            action = "MAINTAIN"
            origination_factor = 1.0
            min_grade = "D"

        policies.append(
            {
                "month": row.get("ds", ""),
                "forecast_pd": forecast_pd,
                "pd_upper_90": pd_upper,
                "action": action,
                "recommended_origination": max_monthly_origination * origination_factor,
                "min_grade_threshold": min_grade,
                "point_model": point_model,
                "interval_model": interval_model,
                "official_status": official_status,
            }
        )

    result = pd.DataFrame(policies)
    logger.info(f"Generated {len(result)} monthly policies")
    return result
