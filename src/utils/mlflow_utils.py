"""MLflow + DagsHub experiment tracking utilities."""

from __future__ import annotations

import os
from typing import Any

import mlflow
from loguru import logger


def init_dagshub(
    repo_owner: str | None = None,
    repo_name: str | None = None,
    enable_dvc: bool = False,
) -> None:
    """Initialize DagsHub MLflow tracking using args or env vars (user or org)."""
    import dagshub

    owner = repo_owner or os.getenv("DAGSHUB_OWNER") or os.getenv("DAGSHUB_USER", "YOUR_USER")
    repo = repo_name or os.getenv("DAGSHUB_REPO", "Lending-Club-End-to-End")
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    if token and not os.getenv("DAGSHUB_USER_TOKEN"):
        os.environ["DAGSHUB_USER_TOKEN"] = token

    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True, dvc=enable_dvc)
    logger.info(f"DagsHub initialized: {owner}/{repo} (tracking_uri={mlflow.get_tracking_uri()})")


def log_experiment(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    experiment_name: str | None = None,
    model: Any = None,
    model_name: str = "model",
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a complete experiment to MLflow.

    Returns:
        Run ID string.
    """
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)

        if model is not None:
            mlflow.sklearn.log_model(model, model_name)

        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)

        logger.info(f"Logged run '{run_name}': {metrics}")
        return run.info.run_id


def log_catboost_model(
    model: Any,
    X_sample: Any,
    run_name: str,
    experiment_name: str | None = None,
    registered_model_name: str | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: dict[str, str] | None = None,
) -> str:
    """Log a CatBoost model with inferred signature to MLflow.

    Uses mlflow.catboost autologging format with model signature for
    schema validation at serving time.

    Args:
        model: Fitted CatBoostClassifier.
        X_sample: Sample input DataFrame for signature inference.
        run_name: MLflow run name.
        experiment_name: Optional experiment name.
        registered_model_name: If provided, register in Model Registry.
        params: Hyperparameters to log.
        metrics: Metrics to log.
        tags: Tags to set.
        artifacts: Additional artifacts {name: path}.

    Returns:
        Run ID string.
    """
    try:
        import mlflow.catboost
        from mlflow.models.signature import infer_signature
    except ImportError:
        logger.warning("mlflow.catboost not available — falling back to sklearn logging")
        return log_experiment(
            run_name=run_name,
            params=params or {},
            metrics=metrics or {},
            experiment_name=experiment_name,
            model=model,
            model_name="pd_catboost",
            artifacts=artifacts,
            tags=tags,
        )

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        # Infer signature from sample prediction
        try:
            y_pred_sample = model.predict_proba(X_sample)[:, 1]
            sig = infer_signature(X_sample, y_pred_sample)
        except Exception as exc:
            logger.warning("Signature inference failed: {}", exc)
            sig = None

        mlflow.catboost.log_model(
            model,
            artifact_path="pd_catboost",
            signature=sig,
            registered_model_name=registered_model_name,
        )

        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)

        logger.info(
            f"Logged CatBoost run '{run_name}' (signature={'yes' if sig else 'no'}, "
            f"registry={registered_model_name or 'none'}): {metrics}"
        )
        return run.info.run_id


def log_conformal_experiment(
    run_name: str,
    base_model_params: dict[str, Any],
    conformal_params: dict[str, Any],
    classification_metrics: dict[str, float],
    conformal_metrics: dict[str, float],
) -> str:
    """Log conformal prediction experiment with both model and CP metrics."""
    all_params = {**base_model_params, **{f"cp_{k}": v for k, v in conformal_params.items()}}
    all_metrics = {**classification_metrics, **{f"cp_{k}": v for k, v in conformal_metrics.items()}}

    return log_experiment(
        run_name=run_name,
        params=all_params,
        metrics=all_metrics,
        tags={"experiment_type": "conformal_prediction"},
    )
