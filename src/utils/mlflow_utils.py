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
    """Initialize DagsHub MLflow tracking using args or environment variables."""
    import dagshub

    owner = repo_owner or os.getenv("DAGSHUB_USER", "YOUR_USER")
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
