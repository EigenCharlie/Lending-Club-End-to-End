"""Probability calibration methods.

Available methods: Isotonic and Platt (Sigmoid). NB03 selected Platt (ECE=0.0128 on test set) as canonical calibrator.

Migration note (scikit-learn >=1.6):
  cv="prefit" is deprecated. Use FrozenEstimator wrapper instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression


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
) -> CalibratedClassifierCV:
    """Fit Platt scaling (sigmoid) calibrator."""
    cal_model = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    cal_model.fit(X_cal, y_cal)
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
