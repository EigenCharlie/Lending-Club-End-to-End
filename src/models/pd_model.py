"""Probability of Default (PD) models.

Baseline: Logistic Regression with L1/L2 regularization.
Primary: CatBoost classifier with Optuna hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Feature configuration ──
NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "loan_to_income",
    "dti",
    "rev_utilization",
    "num_delinq_2yrs",
    "days_since_last_delinq",
    "int_rate",
    "installment",
]

WOE_FEATURES = [
    "grade_woe",
    "purpose_woe",
    "home_ownership_woe",
]

CATEGORICAL_FEATURES = [
    "int_rate_bucket",
    "term",
]

ALL_FEATURES = NUMERIC_FEATURES + WOE_FEATURES + CATEGORICAL_FEATURES
TARGET = "default_flag"


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return features that actually exist in the DataFrame."""
    return [f for f in ALL_FEATURES if f in df.columns]


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs: Any,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Train logistic regression baseline."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        **kwargs,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    metrics = {"auc_roc": auc, "model_type": "logistic_regression"}
    logger.info(f"Baseline LR — AUC: {auc:.4f}")
    return model, metrics


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[CatBoostClassifier, dict[str, float]]:
    """Train CatBoost PD model."""
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 50,
    }
    if params:
        default_params.update(params)

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "auc_roc": auc,
        "best_iteration": model.get_best_iteration(),
        "model_type": "catboost",
    }
    logger.info(f"CatBoost — AUC: {auc:.4f}, best_iter: {model.get_best_iteration()}")
    return model, metrics


def tune_catboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    cat_features: list[str] | None = None,
) -> dict[str, Any]:
    """Hyperparameter tuning with Optuna."""
    import optuna

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_features),
            eval_set=Pool(X_val, y_val, cat_features=cat_features),
            use_best_model=True,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params
