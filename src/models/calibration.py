"""Probability calibration methods.

Available methods: Isotonic, Platt (Sigmoid), Beta, and Venn-Abers.
Canonical calibrator is selected at training time via temporal multi-metric validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.io_utils import load_pickle_compat


class LogitShiftCalibrator:
    """Lightweight serializable calibrator that shifts log-odds by a fixed delta."""

    def __init__(self, delta: float):
        self.delta = float(delta)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(scores_arr / (1.0 - scores_arr))
        shifted = 1.0 / (1.0 + np.exp(-(logits + self.delta)))
        return np.clip(np.asarray(shifted, dtype=float), 0.0, 1.0)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.transform(scores)

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {"delta": float(self.delta)}

    def __repr__(self) -> str:
        return f"LogitShiftCalibrator(delta={self.delta:.6f})"


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


def calibrate_beta(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    parameters: str = "abm",
) -> Any:
    """Fit beta calibration (Kull et al. 2017).

    Args:
        y_cal: Binary labels.
        proba_cal: Raw probabilities from base model.
        parameters: Beta calibration parameterisation.
            "abm" = 3 parameters (handles asymmetric distortions).
            "am" = 2 parameters (equivalent to Platt when a=b).

    Returns:
        Fitted BetaCalibration object with .predict() method.
    """
    from betacal import BetaCalibration

    bc = BetaCalibration(parameters=parameters)
    bc.fit(proba_cal, y_cal)
    logger.info(f"Fitted beta calibrator (parameters={parameters})")
    return bc


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
    n_bins: int = 10,
) -> dict[str, float]:
    """Evaluate calibration quality."""
    from sklearn.metrics import brier_score_loss, log_loss

    ece = expected_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    metrics = {"ece": ece, "brier_score": brier, "log_loss": logloss}
    logger.info(
        f"Calibration [{name}] — ECE: {ece:.4f}, Brier: {brier:.4f}, Log-loss: {logloss:.4f}"
    )
    return metrics


def load_probability_calibrator(path: str | None) -> Any | None:
    """Load a canonical or shadow calibrator from disk."""
    if not path:
        return None
    return load_pickle_compat(path)
