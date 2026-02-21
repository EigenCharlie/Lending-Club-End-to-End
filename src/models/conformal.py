"""Conformal prediction utilities using MAPIE >= 1.3.

Includes:
- Global split-conformal intervals for PD probabilities.
- Group-conditional (Mondrian-style) split conformal by segment (e.g., grade).
- Classification set wrappers (LAC/APS/RAPS via MAPIE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin


class ProbabilityRegressor(BaseEstimator, RegressorMixin):
    """Wrap classifier predict_proba as a regression predictor."""

    def __init__(self, classifier):
        self.classifier = classifier
        self.is_fitted_ = True  # required for MAPIE prefit checks

    def fit(self, X, y):
        """Already fitted - no-op for MAPIE interface."""
        return self

    def predict(self, X):
        """Return P(default)."""
        return self.classifier.predict_proba(X)[:, 1]


def apply_probability_calibrator(calibrator: Any, scores: np.ndarray) -> np.ndarray:
    """Apply calibrator robustly across sklearn calibrator API variants."""
    scores = np.asarray(scores, dtype=float)
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)

    if hasattr(calibrator, "transform"):
        out = calibrator.transform(scores)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    if hasattr(calibrator, "predict_proba"):
        out = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    try:
        out = calibrator.predict(scores)
        out = np.asarray(out, dtype=float)
        if out.shape[0] != scores.shape[0]:
            out = np.asarray(calibrator.predict(scores.reshape(-1, 1)), dtype=float)
    except (ValueError, TypeError, IndexError):
        out = np.asarray(calibrator.predict(scores.reshape(-1, 1)), dtype=float)

    return np.clip(out, 0.0, 1.0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile with 'higher' interpolation."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    n = scores.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def create_pd_intervals(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate global split-conformal PD intervals via MAPIE."""
    from mapie.regression import SplitConformalRegressor

    prob_reg = ProbabilityRegressor(classifier)
    mapie = SplitConformalRegressor(
        estimator=prob_reg,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal.astype(float))

    # MAPIE output shape: (n, 2, n_confidence_levels)
    y_pred, y_intervals_raw = mapie.predict_interval(X_test)
    y_intervals = y_intervals_raw[:, :, 0]  # -> (n, 2)

    y_intervals = np.clip(y_intervals, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)

    avg_width = float((y_intervals[:, 1] - y_intervals[:, 0]).mean())
    logger.info(
        f"Conformal PD intervals (global, alpha={alpha}): "
        f"avg_width={avg_width:.4f}, target_coverage={1 - alpha:.0%}"
    )
    return y_pred, y_intervals


def create_pd_intervals_mondrian(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    group_cal: pd.Series,
    group_test: pd.Series,
    alpha: float = 0.1,
    min_group_size: int = 500,
    calibrator: Any | None = None,
    scaled_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create group-conditional split-conformal PD intervals.

    This follows a Mondrian-style approach:
    - compute nonconformity scores on calibration split
    - estimate quantiles within each group
    - apply group-specific radius to test predictions

    Args:
        classifier: fitted classifier with predict_proba.
        X_cal, y_cal: calibration data.
        X_test: test features.
        group_cal, group_test: segmentation labels (e.g., grade).
        alpha: significance level.
        min_group_size: fallback to global quantile for small groups.
        calibrator: optional probability calibrator.
        scaled_scores: if True, scale residuals by sqrt(p*(1-p)).

    Returns:
        y_pred_test, y_intervals, diagnostics
    """
    y_cal_pred_raw = classifier.predict_proba(X_cal)[:, 1]
    y_test_pred_raw = classifier.predict_proba(X_test)[:, 1]
    y_cal_pred = apply_probability_calibrator(calibrator, y_cal_pred_raw)
    y_test_pred = apply_probability_calibrator(calibrator, y_test_pred_raw)

    y_cal_arr = np.asarray(y_cal, dtype=float)
    g_cal = pd.Series(group_cal).fillna("UNKNOWN").astype(str).to_numpy()
    g_test = pd.Series(group_test).fillna("UNKNOWN").astype(str).to_numpy()

    scores = np.abs(y_cal_arr - y_cal_pred)
    if scaled_scores:
        cal_scale = np.sqrt(np.clip(y_cal_pred * (1.0 - y_cal_pred), 1e-6, None))
        test_scale = np.sqrt(np.clip(y_test_pred * (1.0 - y_test_pred), 1e-6, None))
        scores = scores / cal_scale
    else:
        test_scale = np.ones_like(y_test_pred)

    global_q = _conformal_quantile(scores, alpha)
    group_quantiles: dict[str, float] = {}
    group_cal_counts: dict[str, int] = {}
    fallback_groups: list[str] = []

    # Use union so unseen test groups still get a valid quantile.
    all_groups = sorted(set(g_cal).union(set(g_test)))
    for g in all_groups:
        mask = g_cal == g
        n_g = int(mask.sum())
        group_cal_counts[g] = n_g
        if n_g >= min_group_size:
            group_quantiles[g] = _conformal_quantile(scores[mask], alpha)
        else:
            group_quantiles[g] = global_q
            fallback_groups.append(g)

    radii = np.array([group_quantiles[str(g)] for g in g_test], dtype=float) * test_scale
    low = np.clip(y_test_pred - radii, 0.0, 1.0)
    high = np.clip(y_test_pred + radii, 0.0, 1.0)
    y_intervals = np.column_stack([low, high])

    diagnostics = {
        "alpha": alpha,
        "global_quantile": global_q,
        "group_quantiles": group_quantiles,
        "group_cal_counts": group_cal_counts,
        "fallback_groups": fallback_groups,
        "scaled_scores": scaled_scores,
        "min_group_size": min_group_size,
        "avg_width": float((high - low).mean()),
        "median_width": float(np.median(high - low)),
    }
    logger.info(
        "Conformal PD intervals (mondrian): "
        f"groups={len(all_groups)}, avg_width={diagnostics['avg_width']:.4f}, "
        f"fallback_groups={len(fallback_groups)}"
    )
    return y_test_pred, y_intervals, diagnostics


def conditional_coverage_by_group(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compute conditional coverage and width per segment."""
    g = pd.Series(groups).fillna("UNKNOWN").astype(str)
    y_true_arr = np.asarray(y_true, dtype=float)
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    covered = (y_true_arr >= low) & (y_true_arr <= high)
    widths = high - low

    df = pd.DataFrame(
        {
            "group": g,
            "covered": covered.astype(float),
            "width": widths,
        }
    )
    out = (
        df.groupby("group", observed=True)
        .agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            avg_width=("width", "mean"),
            median_width=("width", "median"),
        )
        .reset_index()
        .sort_values("group")
    )
    return out


def create_regression_intervals(
    regressor,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression intervals (for LGD/EAD) using MAPIE."""
    from mapie.regression import SplitConformalRegressor

    mapie = SplitConformalRegressor(
        estimator=regressor,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)

    y_pred, y_intervals_raw = mapie.predict_interval(X_test)
    y_intervals = y_intervals_raw[:, :, 0]

    avg_width = (y_intervals[:, 1] - y_intervals[:, 0]).mean()
    logger.info(f"Conformal regression intervals (alpha={alpha}): avg_width={avg_width:.4f}")
    return y_pred, y_intervals


def create_classification_sets(
    classifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    alpha: float = 0.1,
    method: str = "lac",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate conformal prediction sets for classification."""
    from mapie.classification import SplitConformalClassifier

    mapie = SplitConformalClassifier(
        estimator=classifier,
        confidence_level=1 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)

    y_pred = mapie.predict(X_test)
    _, y_sets_raw = mapie.predict_set(X_test)
    y_sets = y_sets_raw[:, :, 0]

    singleton_rate = (y_sets.sum(axis=1) == 1).mean()
    logger.info(
        f"Conformal sets (alpha={alpha}, method={method}): singleton_rate={singleton_rate:.2%}"
    )
    return y_pred, y_sets


def validate_coverage(
    y_true: np.ndarray,
    y_intervals: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Validate empirical coverage against target."""
    low = y_intervals[:, 0]
    high = y_intervals[:, 1]
    covered = ((y_true >= low) & (y_true <= high)).mean()
    target = 1 - alpha

    metrics = {
        "empirical_coverage": float(covered),
        "target_coverage": float(target),
        "coverage_gap": float(abs(covered - target)),
        "avg_interval_width": float((high - low).mean()),
        "median_interval_width": float(np.median(high - low)),
    }
    logger.info(f"Coverage validation: empirical={covered:.4f} vs target={target:.4f}")
    return metrics
