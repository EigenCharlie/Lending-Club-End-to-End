"""Exposure at Default (EAD) regression model.

Predicts the outstanding exposure at the time of default.
Only trained on default_flag == 1 subset (ead_dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score


def train_ead_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> tuple[CatBoostRegressor, dict[str, float]]:
    """Train EAD regression model on defaults-only data."""
    default_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 30,
    }
    if params:
        default_params.update(params)

    model = CatBoostRegressor(**default_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"ead_mae": mae, "ead_r2": r2}
    logger.info(f"EAD Model — MAE: {mae:.2f}, R²: {r2:.4f}")
    return model, metrics
