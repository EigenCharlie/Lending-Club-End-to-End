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

# ── Backward-compatible default feature configuration ──
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


def train_catboost_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    cat_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Train default CatBoost model using temporal validation set."""
    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model = CatBoostClassifier(**_catboost_base_params(params))
    model.fit(
        Pool(X_train, y_train, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
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


def train_catboost_tuned_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    *,
    cat_features: list[str] | None = None,
    base_params: dict[str, Any] | None = None,
    n_trials: int = 100,
    sampler: str = "tpe",
    pruner: str = "median",
    timeout_minutes: int = 0,
    n_startup_trials: int = 40,
    multivariate_tpe: bool = True,
    pruner_n_startup_trials: int = 20,
    pruner_n_warmup_steps: int = 50,
    use_pruning_callback: bool = True,
    study_storage: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = True,
    refit_full_train: bool = True,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Tune CatBoost with Optuna and return best fitted model and metadata."""
    import optuna

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    base = _catboost_base_params(base_params)
    base["verbose"] = 0

    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=bool(multivariate_tpe),
        )
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=bool(multivariate_tpe),
        )

    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, int(pruner_n_startup_trials)),
            n_warmup_steps=max(1, int(pruner_n_warmup_steps)),
            interval_steps=25,
        )
    elif pruner == "none":
        pruner_obj = optuna.pruners.NopPruner()
    else:
        pruner_obj = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, int(pruner_n_startup_trials)),
            n_warmup_steps=max(1, int(pruner_n_warmup_steps)),
            interval_steps=25,
        )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    def objective(trial: optuna.Trial) -> float:
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        params = {
            **base,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 50.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 64, 254),
            "bootstrap_type": bootstrap_type,
            "random_seed": int(base.get("random_seed", 42)),
        }
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

        model = CatBoostClassifier(**params)
        pruning_callback = None
        callbacks: list[Any] = []
        if use_pruning_callback:
            try:
                from optuna.integration import CatBoostPruningCallback

                pruning_callback = CatBoostPruningCallback(trial, "AUC")
                callbacks = [pruning_callback]
            except Exception as exc:  # pragma: no cover - optional integration path
                if trial.number == 0:
                    logger.warning(
                        "CatBoost pruning callback unavailable; disabling pruning callback: {}", exc
                    )
                pruning_callback = None
                callbacks = []

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            callbacks=callbacks or None,
        )

        if pruning_callback is not None:
            pruning_callback.check_pruned()

        val_auc = model.get_best_score().get("validation", {}).get("AUC")
        if val_auc is None:
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_prob)

        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        return float(val_auc)

    create_study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "sampler": sampler_obj,
        "pruner": pruner_obj,
    }
    if study_storage:
        create_study_kwargs["storage"] = study_storage
        create_study_kwargs["study_name"] = study_name or "pd_catboost_optuna"
        create_study_kwargs["load_if_exists"] = bool(load_if_exists)

    study = optuna.create_study(**create_study_kwargs)
    timeout = None if timeout_minutes <= 0 else int(timeout_minutes * 60)
    study.optimize(
        objective,
        n_trials=max(1, int(n_trials)),
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = {**base, **study.best_params}
    best_params["verbose"] = 100
    selection_model = CatBoostClassifier(**best_params)
    selection_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    y_val_prob = selection_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    best_iteration = int(selection_model.get_best_iteration())

    if refit_full_train:
        full_X = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        full_y = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        full_pool = Pool(full_X, full_y, cat_features=cat_features)
        refit_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
        if best_iteration > 0:
            refit_params["iterations"] = best_iteration + 1
        best_model = CatBoostClassifier(**refit_params)
        best_model.fit(full_pool)
    else:
        best_model = selection_model

    metrics: dict[str, Any] = {
        "validation_auc": float(val_auc),
        "best_iteration": best_iteration,
        "best_params": study.best_params,
        "hpo_trials_executed": len(study.trials),
        "hpo_best_validation_auc": float(study.best_value),
        "refit_full_train": bool(refit_full_train),
        "model_type": "catboost_tuned",
    }
    if X_test is not None and y_test is not None:
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_test_prob))

    logger.info(
        "CatBoost tuned — val_AUC: "
        f"{val_auc:.4f}, best_trial_val_AUC: {study.best_value:.4f}, "
        f"trials={len(study.trials)}"
    )
    return best_model, metrics


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
