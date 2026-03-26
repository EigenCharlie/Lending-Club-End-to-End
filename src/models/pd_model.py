"""Probability of Default (PD) modeling utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.feature_engineering import (
    CATBOOST_FEATURES as CANONICAL_CATBOOST_FEATURES,
)
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES as CANONICAL_CATEGORICAL_FEATURES,
)
from src.features.feature_engineering import (
    LOGREG_FEATURES as CANONICAL_LOGREG_FEATURES,
)
from src.features.feature_engineering import (
    NUMERIC_FEATURES as CANONICAL_NUMERIC_FEATURES,
)
from src.features.feature_engineering import (
    WOE_FEATURES as CANONICAL_WOE_FEATURES,
)

# ── Backward-compatible default feature configuration ──
NUMERIC_FEATURES = list(CANONICAL_NUMERIC_FEATURES)
WOE_FEATURES = list(CANONICAL_WOE_FEATURES)
CATEGORICAL_FEATURES = list(CANONICAL_CATEGORICAL_FEATURES)
ALL_FEATURES = list(CANONICAL_CATBOOST_FEATURES) + [
    f for f in CANONICAL_LOGREG_FEATURES if f not in CANONICAL_CATBOOST_FEATURES
]
TARGET = "default_flag"


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return legacy fallback features that exist in the DataFrame."""
    return [f for f in ALL_FEATURES if f in df.columns]


def load_feature_config(feature_config_path: str | Path) -> dict[str, Any]:
    """Load persisted feature config artifact if available."""
    path = Path(feature_config_path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    return cfg if isinstance(cfg, dict) else {}


def resolve_feature_sets(
    df: pd.DataFrame,
    feature_source: str = "auto",
    feature_config_path: str | Path = "data/processed/feature_config.pkl",
) -> dict[str, list[str]]:
    """Resolve feature sets from feature_config first, with legacy fallback."""
    cfg = load_feature_config(feature_config_path)
    use_cfg = feature_source == "feature_config" or (feature_source == "auto" and bool(cfg))

    if use_cfg and cfg:
        catboost = [c for c in cfg.get("CATBOOST_FEATURES", []) if c in df.columns]
        categorical = [c for c in cfg.get("CATEGORICAL_FEATURES", []) if c in catboost]
        logreg = [c for c in cfg.get("LOGREG_FEATURES", []) if c in df.columns]
        if not logreg:
            logreg = [c for c in catboost if c not in categorical]
        return {
            "catboost_features": catboost,
            "logreg_features": logreg,
            "categorical_features": categorical,
            "feature_source": "feature_config",
        }

    fallback = get_available_features(df)
    categorical = [c for c in CATEGORICAL_FEATURES if c in fallback]
    return {
        "catboost_features": fallback,
        "logreg_features": [c for c in fallback if c not in categorical],
        "categorical_features": categorical,
        "feature_source": "legacy_defaults",
    }


def temporal_train_val_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.15,
    date_col: str = "issue_d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into fit/validation keeping temporal order (tail as validation)."""
    if len(train_df) < 10:
        n_val = max(1, int(round(len(train_df) * val_fraction)))
        return train_df.iloc[:-n_val].copy(), train_df.iloc[-n_val:].copy()

    val_fraction = float(np.clip(val_fraction, 0.05, 0.5))
    if date_col in train_df.columns:
        ordered = train_df.sort_values(date_col).reset_index(drop=True)
    else:
        ordered = train_df.reset_index(drop=True)
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    fit = ordered.iloc[:-n_val].copy()
    val = ordered.iloc[-n_val:].copy()
    logger.info(f"Temporal split ({date_col}): fit={len(fit):,}, val={len(val):,}")
    return fit, val


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[LogisticRegression, dict[str, float]]:
    """Train logistic regression baseline."""
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        **kwargs,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    metrics = {"auc_roc": float(auc), "model_type": "logistic_regression"}
    logger.info(f"Baseline LR — AUC: {auc:.4f}")
    return model, metrics


def _catboost_base_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "iterations": 1000,
        "loss_function": "Logloss",
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": 100,
        "early_stopping_rounds": 50,
    }
    if params:
        base.update(params)
    return base


def resolve_monotonic_constraints(
    feature_names: list[str],
    constraints_config: dict[str, int] | None = None,
    config_path: str = "configs/pd_model.yaml",
) -> str | None:
    """Build CatBoost monotonic_constraints string from config.

    Reads the constraint map from YAML challenger_pipeline.monotonic_constraints
    and maps it to feature order. Returns a comma-separated string like
    "0,1,-1,0,..." for CatBoost's monotone_constraints parameter.

    Args:
        feature_names: Ordered list of features used by the model.
        constraints_config: Direct constraint dict {feature: direction}.
            If None, reads from YAML config.
        config_path: Path to pd_model.yaml.

    Returns:
        Constraint string for CatBoost, or None if no constraints configured.
    """
    if constraints_config is None:
        try:
            from pathlib import Path as _P

            import yaml

            cfg_path = _P(config_path)
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                constraints_config = (
                    cfg.get("challenger_pipeline", {}).get("monotonic_constraints") or {}
                )
            else:
                return None
        except Exception:
            return None

    if not constraints_config:
        return None

    vector = [constraints_config.get(feat, 0) for feat in feature_names]
    if all(v == 0 for v in vector):
        return None

    n_constrained = sum(1 for v in vector if v != 0)
    logger.info(f"Monotonic constraints: {n_constrained}/{len(feature_names)} features constrained")
    return ",".join(str(v) for v in vector)


def train_catboost_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    cat_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
    eval_sample_weight: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Train default CatBoost model using temporal validation set."""
    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model = CatBoostClassifier(**_catboost_base_params(params))
    model.fit(
        Pool(X_train, y_train, cat_features=cat_features, weight=sample_weight),
        eval_set=Pool(X_val, y_val, cat_features=cat_features, weight=eval_sample_weight),
        use_best_model=True,
    )
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    metrics: dict[str, Any] = {
        "validation_auc": float(val_auc),
        "best_iteration": int(model.get_best_iteration()),
        "model_type": "catboost_default",
    }
    if X_test is not None and y_test is not None:
        y_test_prob = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_test_prob))
    logger.info(
        f"CatBoost default — val_AUC: {val_auc:.4f}, best_iter: {model.get_best_iteration()}"
    )
    return model, metrics


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[CatBoostClassifier, dict[str, float]]:
    """Backward-compatible CatBoost train helper (uses same set for val/test)."""
    model, metrics = train_catboost_default(
        X_train,
        y_train,
        X_test,
        y_test,
        X_test=X_test,
        y_test=y_test,
        cat_features=cat_features,
        params=params,
    )
    return model, {
        "auc_roc": float(metrics.get("auc_roc", metrics["validation_auc"])),
        "best_iteration": int(metrics["best_iteration"]),
        "model_type": "catboost",
    }


def tune_catboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    cat_features: list[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper returning only best params."""
    from src.models.optuna_tuning import train_catboost_tuned_optuna

    _, metrics = train_catboost_tuned_optuna(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_features=cat_features,
        n_trials=n_trials,
    )
    best_params = metrics.get("best_params", {})
    logger.info(f"Best params (wrapper): {best_params}")
    return best_params


def __getattr__(name: str):
    """Lazy re-export for train_catboost_tuned_optuna (avoids circular import)."""
    if name == "train_catboost_tuned_optuna":
        from src.models.optuna_tuning import train_catboost_tuned_optuna

        return train_catboost_tuned_optuna
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
