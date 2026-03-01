"""Optuna-based CatBoost hyperparameter tuning for PD models."""

from __future__ import annotations

import gc
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.models.pd_model import CATEGORICAL_FEATURES, _catboost_base_params


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
    group_tpe: bool = True,
    warn_independent_sampling: bool = True,
    pruner_n_startup_trials: int = 20,
    pruner_n_warmup_steps: int = 50,
    use_pruning_callback: bool = True,
    study_storage: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = True,
    refit_full_train: bool = True,
    gc_after_trial: bool = True,
    storage_heartbeat_interval: int = 0,
    storage_grace_period: int = 0,
    sqlite_timeout_seconds: int = 60,
    retry_failed_trials: int = 0,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Tune CatBoost with Optuna and return best fitted model and metadata."""
    import optuna

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    base = _catboost_base_params(base_params)
    base["verbose"] = 0

    use_multivariate = bool(multivariate_tpe)
    use_group_tpe = bool(group_tpe and use_multivariate)
    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=use_multivariate,
            group=use_group_tpe,
            warn_independent_sampling=bool(warn_independent_sampling),
        )
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=use_multivariate,
            group=use_group_tpe,
            warn_independent_sampling=bool(warn_independent_sampling),
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
        # Bayesian bootstrap is incompatible with subsample; Bernoulli needs
        # subsample but not bagging_temperature.  Clean inherited base keys
        # before adding the correct bootstrap-specific parameter.
        if bootstrap_type == "Bayesian":
            params.pop("subsample", None)
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        else:
            params.pop("bagging_temperature", None)
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
    retry_callback = None
    if study_storage:
        storage_obj: Any = study_storage
        hb_interval = max(0, int(storage_heartbeat_interval))
        hb_grace = max(0, int(storage_grace_period))
        # For long-running trials on SQLite, use a longer connection timeout and
        # heartbeat to recover stale RUNNING trials after crashes/restarts.
        if str(study_storage).startswith(("sqlite:///", "sqlite+pysqlite:///")):
            engine_kwargs = {"connect_args": {"timeout": max(1, int(sqlite_timeout_seconds))}}
        else:
            engine_kwargs = None
        if hb_interval > 0 or hb_grace > 0:
            try:
                failed_cb = None
                if int(retry_failed_trials) > 0:
                    failed_cb = optuna.storages.RetryFailedTrialCallback(
                        max_retry=int(retry_failed_trials)
                    )
                    retry_callback = failed_cb
                storage_obj = optuna.storages.RDBStorage(
                    url=str(study_storage),
                    engine_kwargs=engine_kwargs,
                    heartbeat_interval=hb_interval or None,
                    grace_period=hb_grace or None,
                    failed_trial_callback=failed_cb,
                )
            except Exception as exc:
                logger.warning(
                    "Optuna RDBStorage heartbeat/retry setup failed; falling back to storage URL. "
                    f"reason={exc}"
                )
        create_study_kwargs["storage"] = storage_obj
        create_study_kwargs["study_name"] = study_name or "pd_catboost_optuna"
        create_study_kwargs["load_if_exists"] = bool(load_if_exists)

    study = optuna.create_study(**create_study_kwargs)
    if retry_callback is not None and hasattr(optuna.storages, "fail_stale_trials"):
        try:
            optuna.storages.fail_stale_trials(study)
        except Exception as exc:
            logger.warning("Optuna stale-trial recovery skipped: {}", exc)
    timeout = None if timeout_minutes <= 0 else int(timeout_minutes * 60)
    requested_trials = int(n_trials)
    if requested_trials > 0:
        study.optimize(
            objective,
            n_trials=requested_trials,
            timeout=timeout,
            show_progress_bar=False,
            gc_after_trial=bool(gc_after_trial),
        )
    else:
        complete_trials = [
            t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None
        ]
        if not complete_trials:
            raise ValueError(
                "n_trials=0 requested, but the Optuna study has no COMPLETE trials to reuse."
            )
        logger.info(
            "Optuna reuse mode enabled (n_trials=0): skipping optimization and reusing "
            "{} existing trials from study '{}'.",
            len(study.trials),
            study.study_name,
        )
    gc.collect()

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
        f"trials={len(study.trials)}, multivariate_tpe={use_multivariate}, group_tpe={use_group_tpe}"
    )
    return best_model, metrics
