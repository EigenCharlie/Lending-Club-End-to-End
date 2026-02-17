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
) -> tuple[CatBoostClassifier, CatBoostRegressor, dict[str, float]]:
    """Train two-stage LGD model.

    Stage 1: Binary — is LGD > 0?
    Stage 2: Regression — what is LGD given LGD > 0?
    """
    # Stage 1: Classification
    y_binary = (y_train > 0).astype(int)
    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42,
        early_stopping_rounds=30,
    )
    clf.fit(X_train, y_binary, eval_set=(X_test, (y_test > 0).astype(int)))
    stage1_auc = roc_auc_score((y_test > 0).astype(int), clf.predict_proba(X_test)[:, 1])

    # Stage 2: Regression on LGD > 0 subset
    mask_train = y_train > 0
    reg = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42,
        early_stopping_rounds=30,
    )
    reg.fit(X_train[mask_train], y_train[mask_train])

    # Combined prediction
    y_pred = predict_two_stage(clf, reg, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {"stage1_auc": stage1_auc, "lgd_mae": mae, "lgd_rmse": rmse}
    logger.info(f"Two-stage LGD — Stage1 AUC: {stage1_auc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return clf, reg, metrics


def predict_two_stage(
    clf: CatBoostClassifier,
    reg: CatBoostRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict LGD using two-stage model."""
    p_positive = clf.predict_proba(X)[:, 1]
    lgd_conditional = reg.predict(X)
    return p_positive * np.clip(lgd_conditional, 0, 1)
