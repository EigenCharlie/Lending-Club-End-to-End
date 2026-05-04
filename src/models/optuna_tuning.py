"""Optuna-based CatBoost hyperparameter tuning for PD models."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.calibration import expected_calibration_error
from src.models.pd_model import CATEGORICAL_FEATURES, _catboost_base_params

SEARCH_SPACE_VERSION = "cb_space_v2"


def resolve_optuna_study_name(
    study_name: str | None,
    *,
    search_space_version: str = SEARCH_SPACE_VERSION,
) -> str:
    """Append a stable search-space version to persistent Optuna study names.

    This prevents historical studies with incompatible distributions from being
    reused after search-space changes, which otherwise can fail mid-run with
    dynamic distribution compatibility errors.
    """
    base_name = str(study_name or "pd_catboost_optuna").strip() or "pd_catboost_optuna"
    suffix = f"__{search_space_version.strip()}"
    return base_name if base_name.endswith(suffix) else f"{base_name}{suffix}"


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
    constant_liar: bool = False,
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
    n_jobs: int = 1,
    sample_weight: np.ndarray | None = None,
    eval_sample_weight: np.ndarray | None = None,
    search_space_mode: str = "global",
    local_refine_space: dict[str, Any] | None = None,
    constraints_policy: dict[str, Any] | None = None,
    search_space_version: str = SEARCH_SPACE_VERSION,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Tune CatBoost with Optuna and return best fitted model and metadata."""
    import optuna

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    base = _catboost_base_params(base_params)
    base["verbose"] = 0
    search_space_mode_resolved = str(search_space_mode or "global").strip().lower() or "global"
    local_refine_space = dict(local_refine_space or {})
    constraints_policy = dict(constraints_policy or {})

    incumbent_metrics: dict[str, float] = {}

    def _local_choice(trial: optuna.Trial, name: str, spec: Any, default: Any) -> Any:
        if spec is None:
            return default
        if isinstance(spec, dict):
            if spec.get("choices") is not None:
                return trial.suggest_categorical(name, list(spec["choices"]))
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step")
            log = bool(spec.get("log", False))
            if low is None or high is None:
                return default
            if isinstance(low, int) and isinstance(high, int) and not log:
                return trial.suggest_int(name, int(low), int(high), step=int(step or 1))
            return trial.suggest_float(
                name,
                float(low),
                float(high),
                step=None if log else (float(step) if step is not None else None),
                log=log,
            )
        if isinstance(spec, list):
            return trial.suggest_categorical(name, list(spec))
        return spec

    def _apply_local_feature_priors(trial: optuna.Trial, params: dict[str, Any]) -> None:
        feature_weights_cfg = dict(local_refine_space.get("feature_weights", {}) or {})
        if feature_weights_cfg:
            weights: dict[str, float] = {}
            for feature, spec in feature_weights_cfg.items():
                value = float(_local_choice(trial, f"feature_weight__{feature}", spec, 1.0))
                weights[str(feature)] = value
            if any(abs(value - 1.0) > 1e-12 for value in weights.values()):
                params["feature_weights"] = weights
        penalties_cfg = dict(local_refine_space.get("first_feature_use_penalties", {}) or {})
        if penalties_cfg:
            penalties: dict[str, float] = {}
            for feature, spec in penalties_cfg.items():
                value = float(_local_choice(trial, f"first_use_penalty__{feature}", spec, 0.0))
                penalties[str(feature)] = value
            if any(abs(value) > 1e-12 for value in penalties.values()):
                params["first_feature_use_penalties"] = penalties
        penalties_coeff_spec = local_refine_space.get("penalties_coefficient")
        if penalties_coeff_spec is not None:
            params["penalties_coefficient"] = float(
                _local_choice(trial, "penalties_coefficient", penalties_coeff_spec, 1.0)
            )

    def _materialize_study_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
        params = {**base}
        feature_weights: dict[str, float] = {}
        penalties: dict[str, float] = {}

        for key, value in dict(sampled_params or {}).items():
            key_str = str(key)
            if key_str.startswith("feature_weight__"):
                feature_name = key_str.split("__", 1)[1]
                feature_weights[feature_name] = float(value)
                continue
            if key_str.startswith("first_use_penalty__"):
                feature_name = key_str.split("__", 1)[1]
                penalties[feature_name] = float(value)
                continue
            params[key_str] = value

        if feature_weights and any(
            abs(weight - 1.0) > 1e-12 for weight in feature_weights.values()
        ):
            params["feature_weights"] = feature_weights
        else:
            params.pop("feature_weights", None)
        if penalties and any(abs(weight) > 1e-12 for weight in penalties.values()):
            params["first_feature_use_penalties"] = penalties
        else:
            params.pop("first_feature_use_penalties", None)

        if str(params.get("bootstrap_type", "")).strip() == "Bayesian":
            params.pop("subsample", None)
        else:
            params.pop("bagging_temperature", None)

        if str(params.get("grow_policy", "")).strip() == "Lossguide":
            params.pop("depth", None)
        else:
            params.pop("max_leaves", None)

        if str(params.get("task_type", "")).strip().upper() == "GPU":
            params.pop("rsm", None)

        return {key: value for key, value in params.items() if value is not None}

    def _local_refine_params(trial: optuna.Trial, *, is_gpu: bool) -> dict[str, Any]:
        params = {**base}
        fixed_params = dict(local_refine_space.get("fixed_params", {}) or {})
        params.update(fixed_params)

        params["iterations"] = int(
            _local_choice(
                trial,
                "iterations",
                local_refine_space.get("iterations"),
                base.get("iterations", 3000),
            )
        )
        params["learning_rate"] = float(
            _local_choice(
                trial,
                "learning_rate",
                local_refine_space.get("learning_rate"),
                base.get("learning_rate", 0.03),
            )
        )
        params["l2_leaf_reg"] = float(
            _local_choice(
                trial,
                "l2_leaf_reg",
                local_refine_space.get("l2_leaf_reg"),
                base.get("l2_leaf_reg", 3.0),
            )
        )
        params["min_data_in_leaf"] = int(
            _local_choice(
                trial,
                "min_data_in_leaf",
                local_refine_space.get("min_data_in_leaf"),
                base.get("min_data_in_leaf", 64),
            )
        )
        params["random_strength"] = float(
            _local_choice(
                trial,
                "random_strength",
                local_refine_space.get("random_strength"),
                base.get("random_strength", 1e-6),
            )
        )
        params["border_count"] = int(
            _local_choice(
                trial,
                "border_count",
                local_refine_space.get("border_count"),
                base.get("border_count", 128),
            )
        )
        params["leaf_estimation_iterations"] = int(
            _local_choice(
                trial,
                "leaf_estimation_iterations",
                local_refine_space.get("leaf_estimation_iterations"),
                base.get("leaf_estimation_iterations", 3),
            )
        )
        params["bootstrap_type"] = _local_choice(
            trial,
            "bootstrap_type",
            local_refine_space.get("bootstrap_type"),
            base.get("bootstrap_type", "MVS"),
        )
        params["grow_policy"] = _local_choice(
            trial,
            "grow_policy",
            local_refine_space.get("grow_policy"),
            base.get("grow_policy", "SymmetricTree"),
        )

        if is_gpu:
            params.pop("rsm", None)
        else:
            params["rsm"] = float(
                _local_choice(trial, "rsm", local_refine_space.get("rsm"), base.get("rsm", 1.0))
            )
        if str(params.get("grow_policy", "SymmetricTree")) == "Lossguide":
            params.pop("depth", None)
            params["max_leaves"] = int(
                _local_choice(trial, "max_leaves", local_refine_space.get("max_leaves"), 32)
            )
        else:
            params["depth"] = int(
                _local_choice(trial, "depth", local_refine_space.get("depth"), base.get("depth", 8))
            )
            params.pop("max_leaves", None)
        if str(params.get("bootstrap_type", "MVS")) == "Bayesian":
            params.pop("subsample", None)
            bagging_spec = local_refine_space.get("bagging_temperature", {"low": 0.0, "high": 10.0})
            params["bagging_temperature"] = float(
                _local_choice(
                    trial,
                    "bagging_temperature",
                    bagging_spec,
                    base.get("bagging_temperature", 1.0),
                )
            )
        else:
            params.pop("bagging_temperature", None)
            params["subsample"] = float(
                _local_choice(
                    trial,
                    "subsample",
                    local_refine_space.get("subsample"),
                    base.get("subsample", 0.8),
                )
            )
        _apply_local_feature_priors(trial, params)
        return params

    use_multivariate = bool(multivariate_tpe)
    use_group_tpe = bool(group_tpe and use_multivariate)
    constraints_func = None
    if constraints_policy:
        max_brier_delta = constraints_policy.get("max_brier_delta")
        max_ece_delta = constraints_policy.get("max_ece_delta")
        min_auc_delta = constraints_policy.get("min_auc_delta")

        def constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> list[float]:
            attrs = frozen_trial.user_attrs
            violations: list[float] = []
            if max_brier_delta is not None:
                ceiling = float(incumbent_metrics.get("validation_brier", 0.0)) + float(
                    max_brier_delta
                )
                violations.append(float(attrs.get("validation_brier", float("inf"))) - ceiling)
            if max_ece_delta is not None:
                ceiling = float(incumbent_metrics.get("validation_ece", 0.0)) + float(max_ece_delta)
                violations.append(float(attrs.get("validation_ece", float("inf"))) - ceiling)
            if min_auc_delta is not None:
                floor = float(incumbent_metrics.get("validation_auc", 0.0)) + float(min_auc_delta)
                violations.append(floor - float(attrs.get("validation_auc", float("-inf"))))
            return violations

    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=use_multivariate,
            group=use_group_tpe,
            constant_liar=bool(constant_liar),
            warn_independent_sampling=bool(warn_independent_sampling),
            constraints_func=constraints_func,
        )
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler_obj = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=max(10, int(n_startup_trials)),
            multivariate=use_multivariate,
            group=use_group_tpe,
            constant_liar=bool(constant_liar),
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

    train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=sample_weight)
    val_pool = Pool(X_val, y_val, cat_features=cat_features, weight=eval_sample_weight)
    if constraints_policy:
        incumbent_model = CatBoostClassifier(**base)
        incumbent_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        incumbent_y_val_prob = incumbent_model.predict_proba(X_val)[:, 1]
        incumbent_metrics = {
            "validation_auc": float(roc_auc_score(y_val, incumbent_y_val_prob)),
            "validation_brier": float(brier_score_loss(y_val, incumbent_y_val_prob)),
            "validation_ece": float(expected_calibration_error(y_val, incumbent_y_val_prob)),
        }

    def objective(trial: optuna.Trial) -> float:
        is_gpu = str(base.get("task_type", "")).strip().upper() == "GPU"
        if search_space_mode_resolved == "local_refine":
            params = _local_refine_params(trial, is_gpu=is_gpu)
        else:
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            )
            grow_policy = trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            )
            params = {
                **base,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 100.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
                "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 64, 254),
                "bootstrap_type": bootstrap_type,
                "grow_policy": grow_policy,
                "leaf_estimation_iterations": trial.suggest_int(
                    "leaf_estimation_iterations", 1, 10
                ),
                "random_seed": int(base.get("random_seed", 42)),
            }
            if is_gpu:
                params.pop("rsm", None)
            else:
                params["rsm"] = trial.suggest_float("rsm", 0.5, 1.0)
            if grow_policy == "Lossguide":
                params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)
                params.pop("depth", None)
            else:
                params["depth"] = trial.suggest_int("depth", 4, 10)
            if bootstrap_type == "Bayesian":
                params.pop("subsample", None)
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                )
            else:
                params.pop("bagging_temperature", None)
                params["subsample"] = trial.suggest_float("subsample", 0.5, 0.95)

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
        else:
            y_val_prob = model.predict_proba(X_val)[:, 1]
        val_brier = float(brier_score_loss(y_val, y_val_prob))
        val_ece = float(expected_calibration_error(y_val, y_val_prob))

        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("validation_auc", float(val_auc))
        trial.set_user_attr("validation_brier", val_brier)
        trial.set_user_attr("validation_ece", val_ece)
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
        create_study_kwargs["study_name"] = resolve_optuna_study_name(
            study_name,
            search_space_version=search_space_version,
        )
        create_study_kwargs["load_if_exists"] = bool(load_if_exists)

    study = optuna.create_study(**create_study_kwargs)
    if search_space_mode_resolved == "local_refine" and bool(
        local_refine_space.get("enqueue_base_trial", True)
    ):
        try:
            study.enqueue_trial(
                {
                    key: value
                    for key, value in {
                        "iterations": int(base.get("iterations", 3000)),
                        "learning_rate": float(base.get("learning_rate", 0.03)),
                        "l2_leaf_reg": float(base.get("l2_leaf_reg", 3.0)),
                        "min_data_in_leaf": int(base.get("min_data_in_leaf", 64)),
                        "random_strength": float(base.get("random_strength", 1e-6)),
                        "border_count": int(base.get("border_count", 128)),
                        "leaf_estimation_iterations": int(
                            base.get("leaf_estimation_iterations", 3)
                        ),
                        "bootstrap_type": str(base.get("bootstrap_type", "MVS")),
                        "grow_policy": str(base.get("grow_policy", "SymmetricTree")),
                        "rsm": None
                        if str(base.get("task_type", "")).strip().upper() == "GPU"
                        else float(base.get("rsm", 1.0)),
                        "depth": None
                        if str(base.get("grow_policy", "SymmetricTree")) == "Lossguide"
                        else int(base.get("depth", 8)),
                        "max_leaves": int(base.get("max_leaves", 32))
                        if str(base.get("grow_policy", "SymmetricTree")) == "Lossguide"
                        else None,
                        "subsample": None
                        if str(base.get("bootstrap_type", "MVS")) == "Bayesian"
                        else float(base.get("subsample", 0.8)),
                        "bagging_temperature": float(base.get("bagging_temperature", 1.0))
                        if str(base.get("bootstrap_type", "MVS")) == "Bayesian"
                        else None,
                    }.items()
                    if value is not None
                }
            )
        except Exception as exc:
            logger.warning("Optuna enqueue_trial for local_refine skipped: {}", exc)
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
            n_jobs=max(1, int(n_jobs)),
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

    best_params = _materialize_study_params(study.best_params)
    best_params["verbose"] = 100
    selection_model = CatBoostClassifier(**best_params)
    selection_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    y_val_prob = selection_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    best_iteration = int(selection_model.get_best_iteration())

    if refit_full_train:
        full_X = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        full_y = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        full_weight = None
        if sample_weight is not None and eval_sample_weight is not None:
            full_weight = np.concatenate(
                [
                    np.asarray(sample_weight, dtype=float),
                    np.asarray(eval_sample_weight, dtype=float),
                ]
            )
        full_pool = Pool(full_X, full_y, cat_features=cat_features, weight=full_weight)
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
        "best_params_resolved": best_params,
        "hpo_trials_executed": len(study.trials),
        "hpo_best_validation_auc": float(study.best_value),
        "study_name_resolved": study.study_name,
        "refit_full_train": bool(refit_full_train),
        "model_type": "catboost_tuned",
        "search_space_mode": search_space_mode_resolved,
        "constraint_baseline_metrics": incumbent_metrics,
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


def export_hpo_visualizations(
    study_storage: str,
    study_name: str | None = None,
    output_dir: str = "reports/figures/hpo",
) -> list[str]:
    """Load an Optuna study and export visualization plots as HTML + PNG.

    Args:
        study_storage: SQLite URL (e.g., ``sqlite:///models/optuna_pd_catboost.db``).
        study_name: Study name (auto-resolved with search-space version if None).
        output_dir: Directory to write plots to.

    Returns:
        List of paths to generated plot files.
    """
    from pathlib import Path as _Path

    import optuna

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_name = resolve_optuna_study_name(study_name)
    study = optuna.load_study(study_name=resolved_name, storage=study_storage)
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(complete) < 2:
        logger.warning("Study has <2 complete trials — skipping visualization export.")
        return []

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )
    except ImportError:
        logger.warning("optuna.visualization requires plotly — install plotly to export HPO plots.")
        return []

    plots = {
        "optimization_history": plot_optimization_history(study),
        "param_importances": plot_param_importances(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
        "slice": plot_slice(study),
    }

    saved: list[str] = []
    for name, fig in plots.items():
        html_path = out / f"{name}.html"
        fig.write_html(str(html_path))
        saved.append(str(html_path))
        try:
            png_path = out / f"{name}.png"
            fig.write_image(str(png_path), width=1200, height=700, scale=2)
            saved.append(str(png_path))
        except Exception:
            logger.debug(f"PNG export skipped for {name} (kaleido not installed)")

    logger.info(f"Exported {len(saved)} HPO visualization files to {out}")
    return saved
