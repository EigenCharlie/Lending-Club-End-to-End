"""Stable Venn-Abers calibration helpers for canonical PD artifacts."""

from __future__ import annotations

import numpy as np


class VennAbersScoreCalibrator:
    """Score-based Venn-Abers calibrator over 1D raw probabilities."""

    def __init__(self) -> None:
        self._wrapped = None
        self._is_fitted = False

    @staticmethod
    def _as_binary_proba(y_prob_raw: np.ndarray) -> np.ndarray:
        p1 = np.clip(np.asarray(y_prob_raw, dtype=float).reshape(-1), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> VennAbersScoreCalibrator:
        from venn_abers import VennAbers

        X = self._as_binary_proba(y_prob_raw)
        y = np.asarray(y_true, dtype=int)
        wrapped = VennAbers()
        wrapped.fit(X, y)
        self._wrapped = wrapped
        self._is_fitted = True
        return self

    def _predict_bounds(self, y_prob_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted or self._wrapped is None:
            raise RuntimeError("VennAbersScoreCalibrator is not fitted.")
        X = self._as_binary_proba(y_prob_raw)
        _, p_bounds = self._wrapped.predict_proba(X)
        p0 = np.clip(np.asarray(p_bounds[:, 0], dtype=float), 0.0, 1.0)
        p1 = np.clip(np.asarray(p_bounds[:, 1], dtype=float), 0.0, 1.0)
        low = np.minimum(p0, p1)
        high = np.maximum(p0, p1)
        return low, high

    def predict(self, y_prob_raw: np.ndarray) -> np.ndarray:
        low, high = self._predict_bounds(y_prob_raw)
        return np.clip((low + high) / 2.0, 0.0, 1.0)

    def predict_proba(self, y_prob_raw: np.ndarray) -> np.ndarray:
        p1 = self.predict(y_prob_raw)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
