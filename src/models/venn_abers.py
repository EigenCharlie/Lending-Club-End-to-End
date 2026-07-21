"""Stable Venn-Abers calibration helpers for canonical PD artifacts."""

from __future__ import annotations

import numpy as np


class VennAbersScoreCalibrator:
    """Score-based Venn-Abers calibrator over 1D raw probabilities."""

    LOG_LOSS_POINT_RULE = "log_loss_minimax"
    LEGACY_POINT_RULE = "midpoint_legacy"

    def __init__(self, *, point_rule: str = LOG_LOSS_POINT_RULE) -> None:
        if point_rule not in {self.LOG_LOSS_POINT_RULE, self.LEGACY_POINT_RULE}:
            raise ValueError(f"Unsupported Venn-Abers point rule: {point_rule}")
        self._wrapped = None
        self._is_fitted = False
        self.point_rule = point_rule

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

    def predict_with_bounds(
        self, y_prob_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the versioned point prediction and Venn--Abers multiprobabilities.

        Pickles created before the point-rule audit have no ``point_rule``
        attribute. They intentionally retain their historical midpoint behavior
        so that loading an unchanged artifact cannot silently alter its outputs.
        Newly fitted calibrators use the Venn--Abers log-loss minimax rule.
        """
        if not self._is_fitted or self._wrapped is None:
            raise RuntimeError("VennAbersScoreCalibrator is not fitted.")
        X = self._as_binary_proba(y_prob_raw)
        p_prime, p_bounds = self._wrapped.predict_proba(X)
        p0 = np.clip(np.asarray(p_bounds[:, 0], dtype=float), 0.0, 1.0)
        p1 = np.clip(np.asarray(p_bounds[:, 1], dtype=float), 0.0, 1.0)
        low = np.minimum(p0, p1)
        high = np.maximum(p0, p1)
        point_rule = getattr(self, "point_rule", self.LEGACY_POINT_RULE)
        if point_rule == self.LOG_LOSS_POINT_RULE:
            point = np.clip(np.asarray(p_prime[:, 1], dtype=float), 0.0, 1.0)
        elif point_rule == self.LEGACY_POINT_RULE:
            point = np.clip((low + high) / 2.0, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported Venn-Abers point rule: {point_rule}")
        return point, low, high

    def _predict_bounds(self, y_prob_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility helper for diagnostics persisted before the point-rule audit."""
        _, low, high = self.predict_with_bounds(y_prob_raw)
        return low, high

    def predict(self, y_prob_raw: np.ndarray) -> np.ndarray:
        point, _, _ = self.predict_with_bounds(y_prob_raw)
        return point

    def predict_proba(self, y_prob_raw: np.ndarray) -> np.ndarray:
        p1 = self.predict(y_prob_raw)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
