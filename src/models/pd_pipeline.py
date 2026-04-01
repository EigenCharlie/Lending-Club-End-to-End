"""sklearn Pipeline wrapper for the PD modeling workflow.

Provides a reproducible fit/predict interface that chains:
  feature resolution → CatBoost classifier → probability calibrator

Compatible with sklearn cross-validation, MAPIE conformal wrappers,
and model serialization via joblib/pickle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _make_feature_selector(
    feature_names: list[str],
    fill_value: float = 0.0,
) -> FunctionTransformer:
    """Create a transformer that selects and orders features.

    Ensures the input DataFrame always has the expected columns in the
    correct order, filling missing ones with ``fill_value``.
    """

    def _select(X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in feature_names:
            out[col] = X[col] if col in X.columns else fill_value
        return out

    return FunctionTransformer(_select, validate=False)


class CatBoostSklearnAdapter(BaseEstimator, ClassifierMixin):
    """Thin adapter making CatBoostClassifier fully sklearn-compatible.

    Handles categorical feature declaration, sample weights via ``fit``,
    and exposes ``predict_proba`` / ``predict`` as expected by sklearn
    Pipeline and MAPIE wrappers.
    """

    def __init__(
        self,
        catboost_params: dict[str, Any] | None = None,
        cat_features: list[str] | None = None,
    ):
        self.catboost_params = catboost_params or {}
        self.cat_features = cat_features or []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple | None = None,
    ) -> CatBoostSklearnAdapter:
        from catboost import CatBoostClassifier, Pool

        self.model_ = CatBoostClassifier(**self.catboost_params)
        cat_feats = [c for c in self.cat_features if c in X.columns]
        train_pool = Pool(X, y, cat_features=cat_feats, weight=sample_weight)

        fit_kwargs: dict[str, Any] = {"use_best_model": True}
        if eval_set is not None:
            from catboost import Pool as _P

            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = _P(X_val, y_val, cat_features=cat_feats)

        self.model_.fit(train_pool, **fit_kwargs)
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


class CalibratedPDPipeline(BaseEstimator, ClassifierMixin):
    """End-to-end PD pipeline: feature selection → CatBoost → calibrator.

    Exposes ``predict_pd()`` for calibrated probabilities and
    ``predict_proba()`` for sklearn compatibility.

    Args:
        feature_names: Ordered list of model features.
        cat_features: Categorical feature names (subset of feature_names).
        catboost_params: CatBoost hyperparameters.
        calibrator: Fitted probability calibrator (Platt/Isotonic/VennAbers/Beta).
    """

    def __init__(
        self,
        feature_names: list[str],
        cat_features: list[str] | None = None,
        catboost_params: dict[str, Any] | None = None,
        calibrator: Any | None = None,
    ):
        self.feature_names = feature_names
        self.cat_features = cat_features or []
        self.catboost_params = catboost_params or {}
        self.calibrator = calibrator

        self._pipeline = Pipeline(
            [
                ("feature_selector", _make_feature_selector(feature_names)),
                ("classifier", CatBoostSklearnAdapter(catboost_params, self.cat_features)),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        **fit_params: Any,
    ) -> CalibratedPDPipeline:
        """Fit the full pipeline (feature selection + CatBoost)."""
        self._pipeline.fit(X, y, **{f"classifier__{k}": v for k, v in fit_params.items()})
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        logger.info(
            f"CalibratedPDPipeline fitted: {len(self.feature_names)} features, "
            f"calibrator={'yes' if self.calibrator else 'no'}"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predict_proba (uncalibrated)."""
        return self._pipeline.predict_proba(X)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated PD estimates."""
        raw = self._pipeline.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            from src.models.conformal import apply_probability_calibrator

            return apply_probability_calibrator(self.calibrator, raw)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary prediction at default threshold."""
        return self._pipeline.predict(X)

    @classmethod
    def from_artifacts(
        cls,
        model_path: str = "models/pd_canonical.cbm",
        calibrator_path: str = "models/pd_canonical_calibrator.pkl",
        contract_path: str = "models/pd_model_contract.json",
    ) -> CalibratedPDPipeline:
        """Load a pre-trained pipeline from canonical artifacts.

        This is the recommended way to instantiate for inference.
        """
        import json
        import pickle
        from pathlib import Path

        from catboost import CatBoostClassifier

        # Load contract
        with open(contract_path) as f:
            contract = json.load(f)
        feature_names = contract["feature_names"]
        cat_features = contract.get("categorical_features", [])

        # Load model
        model = CatBoostClassifier()
        model.load_model(model_path)

        # Load calibrator
        cal_path = Path(calibrator_path)
        calibrator = None
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrator = pickle.load(f)

        # Build pipeline with pre-fitted model
        instance = cls(
            feature_names=feature_names,
            cat_features=cat_features,
            calibrator=calibrator,
        )
        adapter = instance._pipeline.named_steps["classifier"]
        adapter.model_ = model
        adapter.classes_ = np.array([0, 1])
        adapter.is_fitted_ = True
        instance.is_fitted_ = True

        logger.info(
            f"Loaded CalibratedPDPipeline from artifacts: "
            f"{len(feature_names)} features, calibrator={'yes' if calibrator else 'no'}"
        )
        return instance
