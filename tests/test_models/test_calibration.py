"""Tests for src/models/calibration.py.

Covers ECE computation, isotonic/Platt calibrators, and evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    calibrate_isotonic,
    evaluate_calibration,
    expected_calibration_error,
)

# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_returns_zero(self):
        # Probabilities exactly match true rates per bin
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration_is_high(self):
        # All predictions = 0 but all labels = 1
        y_true = np.ones(100)
        y_prob = np.zeros(100)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.5

    def test_ece_in_unit_interval(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_empty_bins_handled(self):
        # Only values in [0, 0.1) bin
        y_true = np.array([0, 0, 1])
        y_prob = np.array([0.01, 0.02, 0.05])
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert np.isfinite(ece)


# ---------------------------------------------------------------------------
# calibrate_isotonic
# ---------------------------------------------------------------------------


class TestCalibrateIsotonic:
    def test_returns_isotonic_model(self):
        rng = np.random.default_rng(42)
        y_cal = rng.integers(0, 2, 200).astype(float)
        proba_cal = rng.uniform(0, 1, 200)
        model = calibrate_isotonic(y_cal, proba_cal)
        assert hasattr(model, "predict")

    def test_predictions_in_zero_one(self):
        rng = np.random.default_rng(42)
        y_cal = rng.integers(0, 2, 200).astype(float)
        proba_cal = rng.uniform(0, 1, 200)
        model = calibrate_isotonic(y_cal, proba_cal)
        preds = model.predict(np.linspace(0, 1, 50))
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_monotonic_output(self):
        rng = np.random.default_rng(42)
        y_cal = rng.integers(0, 2, 300).astype(float)
        proba_cal = rng.uniform(0, 1, 300)
        model = calibrate_isotonic(y_cal, proba_cal)
        inputs = np.linspace(0, 1, 100)
        preds = model.predict(inputs)
        # Isotonic regression should be non-decreasing
        assert np.all(np.diff(preds) >= -1e-10)


# ---------------------------------------------------------------------------
# evaluate_calibration
# ---------------------------------------------------------------------------


class TestEvaluateCalibration:
    def test_returns_ece_and_brier(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        metrics = evaluate_calibration(y_true, y_prob, name="test_model")
        assert "ece" in metrics
        assert "brier_score" in metrics

    def test_perfect_predictions_low_brier(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = evaluate_calibration(y_true, y_prob)
        assert metrics["brier_score"] == pytest.approx(0.0)

    def test_brier_in_valid_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        metrics = evaluate_calibration(y_true, y_prob)
        assert 0.0 <= metrics["brier_score"] <= 1.0
