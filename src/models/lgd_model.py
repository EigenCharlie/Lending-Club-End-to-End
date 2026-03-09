"""Loss Given Default (LGD) model — two-stage approach.

Stage 1: Classify P(LGD > 0) — some defaults have full recovery.
Stage 2: Regress conditional LGD for those with LGD > 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

TARGET = "lgd"


def train_two_stage_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str] | None = None,
    params: dict | None = None,
) -> tuple[CatBoostClassifier, CatBoostRegressor, dict[str, float]]:
    """Train two-stage LGD model.

    Stage 1: Binary — is LGD > 0?
    Stage 2: Regression — what is LGD given LGD > 0?
    """
    # Stage 1: Classification
    y_binary = (y_train > 0).astype(int)
    clf_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 30,
    }
    if params:
        clf_params.update(params)
    clf = CatBoostClassifier(
        **clf_params,
    )
    clf.fit(X_train, y_binary, eval_set=(X_test, (y_test > 0).astype(int)))
    stage1_auc = roc_auc_score((y_test > 0).astype(int), clf.predict_proba(X_test)[:, 1])

    # Stage 2: Regression on LGD > 0 subset
    mask_train = y_train > 0
    reg_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 30,
    }
    if params:
        reg_params.update(params)
    reg = CatBoostRegressor(**reg_params)
    reg.fit(X_train[mask_train], y_train[mask_train])

    # Combined prediction
    y_pred = predict_two_stage(clf, reg, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"stage1_auc": stage1_auc, "lgd_mae": mae, "lgd_rmse": rmse}
    logger.info(f"Two-stage LGD — Stage1 AUC: {stage1_auc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return clf, reg, metrics


def train_direct_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> tuple[CatBoostRegressor, dict[str, float]]:
    """Train direct LGD regressor on defaults-only data."""
    default_params = {
        "iterations": 900,
        "learning_rate": 0.03,
        "depth": 7,
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 40,
    }
    if params:
        default_params.update(params)

    reg = CatBoostRegressor(**default_params)
    reg.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred = np.clip(reg.predict(X_test), 0.0, 1.0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = float(np.mean(y_pred - y_test.to_numpy(dtype=float)))

    metrics = {"lgd_mae": float(mae), "lgd_rmse": float(rmse), "lgd_bias": bias}
    logger.info(
        "Direct LGD — MAE: {:.4f}, RMSE: {:.4f}, bias: {:.4f}",
        metrics["lgd_mae"],
        metrics["lgd_rmse"],
        metrics["lgd_bias"],
    )
    return reg, metrics


def predict_two_stage(
    clf: CatBoostClassifier,
    reg: CatBoostRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict LGD using two-stage model."""
    p_positive = clf.predict_proba(X)[:, 1]
    lgd_conditional = reg.predict(X)
    return p_positive * np.clip(lgd_conditional, 0, 1)
