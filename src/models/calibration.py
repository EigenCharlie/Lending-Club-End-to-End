"""Probability calibration methods.

Available methods: Isotonic and Platt (Sigmoid).
Canonical calibrator is selected at training time via temporal multi-metric validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def calibrate_isotonic(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression calibrator."""
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(proba_cal, y_cal)
    logger.info("Fitted isotonic calibrator")
    return iso


def calibrate_platt(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> LogisticRegression:
    """Fit Platt scaling as logistic regression over raw model scores.

    Returning a score-based calibrator keeps downstream conformal code agnostic
    to feature-space requirements of the base classifier.
    """
    proba_cal = model.predict_proba(X_cal)[:, 1]
    cal_model = LogisticRegression(max_iter=1000)
    cal_model.fit(proba_cal.reshape(-1, 1), y_cal)
    logger.info("Fitted Platt scaling calibrator")
    return cal_model


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_bins: int = 10,
) -> dict[str, float]:
    """Evaluate calibration quality."""
    from sklearn.metrics import brier_score_loss

    ece = expected_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob)
    metrics = {"ece": ece, "brier_score": brier}
    logger.info(f"Calibration [{name}] — ECE: {ece:.4f}, Brier: {brier:.4f}")
    return metrics
