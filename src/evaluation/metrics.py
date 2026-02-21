"""Comprehensive evaluation metrics for credit risk models.

Classification: AUC-ROC, KS statistic, Gini, Brier score, ECE.
Regression: MAE, RMSE, R².
Conformal: Coverage, efficiency, singleton rate.
Survival: C-index.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute all classification metrics for PD model."""
    from src.models.calibration import expected_calibration_error

    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    ks = ks_statistic(y_true, y_prob)

    return {
        "auc_roc": auc,
        "gini": gini,
        "brier_score": brier,
        "ece": ece,
        "ks_statistic": ks,
    }


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for discriminatory power."""
    from scipy.stats import ks_2samp

    defaults = y_prob[y_true == 1]
    non_defaults = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(defaults, non_defaults)
    return ks_stat


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for LGD/EAD models."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def conformal_metrics(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Compute conformal prediction quality metrics.

    Args:
        y_intervals: Shape (n, 2) — columns [lower, upper].
                     Compatible with MAPIE >=1.3 predict_interval output.
    """
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    widths = high - low

    covered = (y_true >= low) & (y_true <= high)

    return {
        "empirical_coverage": covered.mean(),
        "target_coverage": 1 - alpha,
        "coverage_gap": abs(covered.mean() - (1 - alpha)),
        "avg_width": widths.mean(),
        "median_width": np.median(widths),
        "width_std": widths.std(),
        "width_90th_pct": np.percentile(widths, 90),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_intervals: np.ndarray | None = None,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Compute all relevant metrics in one call."""
    metrics = classification_metrics(y_true, y_prob)
    if y_intervals is not None:
        metrics.update(conformal_metrics(y_true, y_intervals, alpha))
    return metrics


def forecast_backtest_metrics(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_lo: np.ndarray | None = None,
    forecast_hi: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate time series forecasts against realized actuals.

    Useful for validating IFRS9 forward-looking scenarios against
    observed default rates in the OOT test period (2018-2020).

    Args:
        forecast_values: Point forecasts.
        actual_values: Realized values.
        forecast_lo: Lower prediction interval (optional).
        forecast_hi: Upper prediction interval (optional).

    Returns:
        Dict with MAE, RMSE, directional accuracy, and optionally
        interval coverage and width.
    """
    forecast_values = np.asarray(forecast_values, dtype=float)
    actual_values = np.asarray(actual_values, dtype=float)

    metrics: dict[str, float] = {
        "forecast_mae": float(mean_absolute_error(actual_values, forecast_values)),
        "forecast_rmse": float(np.sqrt(mean_squared_error(actual_values, forecast_values))),
    }

    # Directional accuracy: did forecast correctly predict up/down movement?
    if len(actual_values) > 1:
        actual_direction = np.sign(np.diff(actual_values))
        forecast_direction = np.sign(np.diff(forecast_values))
        directional_match = actual_direction == forecast_direction
        metrics["directional_accuracy"] = float(directional_match.mean())

    # Mean bias (positive = over-forecast)
    metrics["mean_bias"] = float((forecast_values - actual_values).mean())

    # Interval coverage if bounds provided
    if forecast_lo is not None and forecast_hi is not None:
        forecast_lo = np.asarray(forecast_lo, dtype=float)
        forecast_hi = np.asarray(forecast_hi, dtype=float)
        covered = (actual_values >= forecast_lo) & (actual_values <= forecast_hi)
        metrics["interval_coverage"] = float(covered.mean())
        metrics["avg_interval_width"] = float((forecast_hi - forecast_lo).mean())

    return metrics
