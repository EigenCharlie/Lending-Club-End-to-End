"""Estimate official causal artifacts for the pricing-intervention research lane."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.causal import (
    build_overlap_diagnostics,
    build_sensitivity_status,
    default_confounders,
    default_effect_modifiers,
    estimate_ate_dowhy,
    estimate_ate_linear_dml,
    estimate_cate_candidates,
    evaluate_overlap_status,
    inspect_causal_environment,
    load_causal_config,
    required_causal_columns,
    sanitize_causal_dataframe,
    specify_causal_graph,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.pipeline_runtime import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_pickle,
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)

CAUSAL_SCHEMA_VERSION = "2026-03-26.1"


def _coerce_treatment(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype(str).str.strip().str.rstrip("%").pipe(pd.to_numeric, errors="coerce")


def _load_split(primary_name: str, fallback_name: str) -> tuple[pd.DataFrame, Path]:
    primary = Path(f"data/processed/{primary_name}.parquet")
    fallback = Path(f"data/processed/{fallback_name}.parquet")
    if primary.exists():
        return pd.read_parquet(primary), primary
    if fallback.exists():
        return pd.read_parquet(fallback), fallback
    raise FileNotFoundError(f"Missing both {primary} and {fallback}")


def _ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "id" not in out.columns:
        out["id"] = np.arange(len(out)).astype(str)
    else:
        out["id"] = out["id"].astype(str)
    return out


def _numeric_frame(
    df: pd.DataFrame, columns: list[str], *, fill_values: dict[str, float]
) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    out = df[columns].copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        fill_value = float(fill_values.get(col, float(out[col].median())))
        if np.isnan(fill_value):
            fill_value = 0.0
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return out


def _artifact_metadata(run_tag: str, dataset_scope: str) -> dict[str, Any]:
    return {
        "dataset_scope": dataset_scope,
        **build_artifact_metadata(
            schema_version=CAUSAL_SCHEMA_VERSION,
            run_tag=run_tag,
            require_explicit=True,
        ),
    }


def _artifact_context_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "id",
        "grade",
        "loan_amnt",
        "annual_inc",
        "dti",
        "int_rate",
        "purpose",
        "home_ownership",
        "default_flag",
    ]
    return [col for col in columns if col in df.columns]


def _parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "y", "on"}:
        return True
    if candidate in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_feature_contract(
    train_df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[list[str], list[str]]:
    variables_cfg = cfg.get("variables", {}) or {}
    configured_effect_modifiers = (
        variables_cfg.get("effect_modifiers") or default_effect_modifiers()
    )
    configured_confounders = variables_cfg.get("confounders") or default_confounders()
    extra_effect_modifiers = variables_cfg.get("extra_effect_modifiers") or []
    extra_confounders = variables_cfg.get("extra_confounders") or []

    effect_modifiers = [
        c
        for c in list(dict.fromkeys([*configured_effect_modifiers, *extra_effect_modifiers]))
        if c in train_df.columns
    ]
    confounders = [
        c
        for c in list(dict.fromkeys([*configured_confounders, *extra_confounders]))
        if c in train_df.columns and c not in effect_modifiers
    ]
    return effect_modifiers, confounders


def _sanitize_test_for_oot(
    df: pd.DataFrame,
    *,
    treatment: str,
    effect_modifiers: list[str],
    fill_values: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _ensure_id(df)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[treatment] = _coerce_treatment(frame[treatment])
    for col in [treatment, *effect_modifiers]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        fill_value = float(fill_values.get(col, 0.0))
        frame[col] = frame[col].fillna(fill_value)
    return frame, {
        "n_rows_test_input": int(len(df)),
        "n_rows_test_after_sanitization": int(len(frame)),
    }


def _save_oot_cate_artifact(
    estimator: Any,
    test_df: pd.DataFrame,
    *,
    treatment: str,
    effect_modifiers: list[str],
    fill_values: dict[str, float],
    data_dir: Path,
) -> pd.DataFrame:
    test_df, _ = _sanitize_test_for_oot(
        test_df,
        treatment=treatment,
        effect_modifiers=effect_modifiers,
        fill_values=fill_values,
    )
    X_test = _numeric_frame(test_df, effect_modifiers, fill_values=fill_values)
    cate = estimator.const_marginal_effect(X_test)
    lb, ub = estimator.const_marginal_effect_interval(X_test, alpha=0.05)

    oot_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(str).to_numpy(),
            "cate": np.asarray(cate, dtype=float).reshape(-1),
            "cate_lb": np.asarray(lb, dtype=float).reshape(-1),
            "cate_ub": np.asarray(ub, dtype=float).reshape(-1),
        }
    )
    for col in _artifact_context_columns(test_df):
        if col not in oot_df.columns:
            oot_df[col] = test_df[col].to_numpy()
    oot_path = data_dir / "cate_estimates_oot.parquet"
    atomic_write_parquet(oot_df, oot_path, index=False)
    logger.info("Saved OOT CATE artifact: {} ({} rows)", oot_path, len(oot_df))
    return oot_df


def main(
    treatment: str = "int_rate",
    sample_size: int | None = None,
    run_tag: str | None = None,
    config_path: str = "configs/causal_lane.yaml",
    cate_n_estimators: int | None = None,
    cate_cv: int | None = None,
    cate_mc_iters: int | None = None,
    cate_criterion: str | None = None,
    cate_min_balancedness_tol: float | None = None,
    cate_honest: bool | None = None,
) -> None:
    cfg = load_causal_config(config_path)
    estimators_cfg = cfg.setdefault("estimators", {})
    causal_forest_cfg = estimators_cfg.setdefault("causal_forest_dml", {})
    linear_dml_cfg = estimators_cfg.setdefault("linear_dml", {})
    if cate_n_estimators is not None:
        causal_forest_cfg["n_estimators"] = int(cate_n_estimators)
    if cate_cv is not None:
        causal_forest_cfg["cv"] = int(cate_cv)
        linear_dml_cfg["cv"] = int(cate_cv)
    if cate_mc_iters is not None:
        causal_forest_cfg["mc_iters"] = int(cate_mc_iters)
        linear_dml_cfg["mc_iters"] = int(cate_mc_iters)
    if cate_criterion is not None:
        causal_forest_cfg["criterion"] = str(cate_criterion)
    if cate_min_balancedness_tol is not None:
        causal_forest_cfg["min_balancedness_tol"] = float(cate_min_balancedness_tol)
    if cate_honest is not None:
        causal_forest_cfg["honest"] = bool(cate_honest)
    run_tag_resolved = resolve_run_tag(run_tag, require_explicit=True)
    stage_name = "causal_effect"
    write_runtime_status(
        stage_name,
        phase="loading_inputs",
        state="running",
        run_tag=run_tag_resolved,
        extra={
            "config_path": config_path,
            "causal_forest_overrides": {
                "n_estimators": causal_forest_cfg.get("n_estimators"),
                "cv": causal_forest_cfg.get("cv"),
                "mc_iters": causal_forest_cfg.get("mc_iters"),
                "criterion": causal_forest_cfg.get("criterion"),
                "min_balancedness_tol": causal_forest_cfg.get("min_balancedness_tol"),
                "honest": causal_forest_cfg.get("honest"),
            },
        },
    )

    env_status = inspect_causal_environment()
    econml_installed = env_status.get("packages", {}).get("econml", {}).get("installed") is not None
    if econml_installed and not env_status.get("compatible", False):
        incompatible = [
            f"{name}={meta.get('installed')} expected {meta.get('expected')}"
            for name, meta in env_status.get("packages", {}).items()
            if not meta.get("compatible", False)
        ]
        raise RuntimeError(
            "Causal environment is incompatible with the official lane: " + ", ".join(incompatible)
        )
    write_runtime_checkpoint(stage_name, "environment_checked", env_status)

    train_df, train_path = _load_split("train_fe", "train")
    test_df, test_path = _load_split("test_fe", "test")
    train_df = _ensure_id(train_df)
    test_df = _ensure_id(test_df)

    if treatment not in train_df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in {train_path}")
    if treatment not in test_df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in {test_path}")

    missing_required_columns = [
        col for col in required_causal_columns(treatment=treatment) if col not in train_df.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "Missing required causal columns for the official DAG/contract: "
            + ", ".join(sorted(missing_required_columns))
        )

    if sample_size is not None and int(sample_size) <= 0:
        sample_size = None
    if sample_size is not None and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    dataset_scope = "full_data" if sample_size is None else "sampled"

    effect_modifiers, confounders = _resolve_feature_contract(train_df, cfg)
    ate_controls = sorted(set(effect_modifiers + confounders))
    if not effect_modifiers:
        raise ValueError("No official effect modifiers found in the feature dataset.")
    if not ate_controls:
        raise ValueError("No controls available for causal identification.")

    data_cfg = cfg.get("data", {})
    train_df[treatment] = _coerce_treatment(train_df[treatment])
    sanitized_train, sanitization = sanitize_causal_dataframe(
        train_df,
        treatment=treatment,
        outcome="default_flag",
        covariate_columns=ate_controls,
        max_covariate_missing_rate=float(data_cfg.get("max_covariate_missing_rate", 0.05)),
        max_row_drop_rate=float(data_cfg.get("max_row_drop_rate", 0.02)),
        impute_covariates=str(data_cfg.get("impute_covariates", "median")),
    )
    fill_values = {
        col: float(sanitized_train[col].median())
        for col in [treatment, *ate_controls]
        if col in sanitized_train.columns
    }
    write_runtime_checkpoint(
        stage_name,
        "inputs_sanitized",
        {
            "dataset_scope": dataset_scope,
            "treatment": treatment,
            **sanitization,
        },
    )

    overlap_cfg = cfg.get("overlap", {})
    overlap = build_overlap_diagnostics(
        sanitized_train,
        treatment=treatment,
        outcome="default_flag",
        segment_columns=overlap_cfg.get("segment_columns"),
        min_segment_size=int(overlap_cfg.get("min_segment_size", 50)),
    )
    overlap["dataset_scope"] = dataset_scope
    overlap["run_tag"] = run_tag_resolved
    overlap_path = Path("data/processed/causal_overlap_diagnostics.parquet")
    atomic_write_parquet(overlap, overlap_path, index=False)
    overlap_status = {
        **_artifact_metadata(run_tag_resolved, dataset_scope),
        **evaluate_overlap_status(
            overlap,
            min_support_ok_share=float(overlap_cfg.get("min_support_ok_share", 0.80)),
        ),
        "n_segments": int(len(overlap)),
        "overlap_artifact_path": str(overlap_path),
    }
    atomic_write_json(Path("models/causal_overlap_status.json"), overlap_status)

    X = _numeric_frame(sanitized_train, effect_modifiers, fill_values=fill_values)
    W = (
        _numeric_frame(sanitized_train, confounders, fill_values=fill_values)
        if confounders
        else None
    )
    Y = pd.to_numeric(sanitized_train["default_flag"], errors="coerce")
    T = pd.to_numeric(sanitized_train[treatment], errors="coerce")

    ate_info = estimate_ate_linear_dml(
        Y=Y,
        T=T,
        X=X,
        W=W,
        cv=int(estimators_cfg.get("linear_dml", {}).get("cv", 3)),
        mc_iters=int(estimators_cfg.get("linear_dml", {}).get("mc_iters", 1)),
        random_state=int(estimators_cfg.get("random_state", 42)),
    )
    write_runtime_status(
        stage_name,
        phase="ate_estimated",
        state="running",
        run_tag=run_tag_resolved,
    )

    dowhy_audit = estimate_ate_dowhy(
        df=sanitized_train,
        treatment=treatment,
        outcome="default_flag",
        common_causes=ate_controls,
        graph=specify_causal_graph(treatment=treatment, outcome="default_flag"),
    )

    sensitivity_cfg = cfg.get("sensitivity", {})
    sensitivity_status = {
        **_artifact_metadata(run_tag_resolved, dataset_scope),
        "estimator_family": "linear_dml",
        **build_sensitivity_status(
            ate_info["estimator"],
            min_robustness_value=float(sensitivity_cfg.get("min_robustness_value", 0.05)),
            alpha=float(sensitivity_cfg.get("alpha", 0.05)),
            c_y=float(sensitivity_cfg.get("c_y", 0.05)),
            c_t=float(sensitivity_cfg.get("c_t", 0.05)),
            rho=float(sensitivity_cfg.get("rho", 1.0)),
        ),
    }
    sensitivity_path = Path("models/causal_sensitivity_status.json")
    atomic_write_json(sensitivity_path, sensitivity_status)

    selection = estimate_cate_candidates(
        Y=Y,
        T=T,
        X=X,
        W=W,
        candidate_names=list(
            estimators_cfg.get("cate_candidates", ["causal_forest_dml", "linear_dml"])
        ),
        random_state=int(estimators_cfg.get("random_state", 42)),
        causal_forest_cfg=estimators_cfg.get("causal_forest_dml", {}),
        linear_dml_cfg=estimators_cfg.get("linear_dml", {}),
        selector=str(estimators_cfg.get("cate_selector", "rscorer")),
    )
    selected_name = str(selection["selected_name"])
    selected = selection["selected"]
    write_runtime_status(
        stage_name,
        phase="cate_estimated",
        state="running",
        run_tag=run_tag_resolved,
        extra={"selected_estimator_family": selected_name},
    )

    selection_status = {
        **_artifact_metadata(run_tag_resolved, dataset_scope),
        "selected_estimator_family": selected_name,
        "selection_reason": selection.get("selection_reason"),
        "candidate_metrics": {
            name: {
                "estimator_family": payload.get("estimator_family"),
                "cate_mean": payload.get("cate_mean"),
                "cate_std": payload.get("cate_std"),
                "selection_score": payload.get("selection_score"),
            }
            for name, payload in selection.get("candidates", {}).items()
        },
        "failures": selection.get("failures", {}),
    }
    selection_path = Path("models/causal_estimator_selection_status.json")
    atomic_write_json(selection_path, selection_status)

    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cate_df = pd.DataFrame(
        {
            "id": sanitized_train["id"].astype(str).to_numpy(),
            "cate": np.asarray(selected["cate"], dtype=float).reshape(-1),
            "cate_lb": np.asarray(selected["cate_lb"], dtype=float).reshape(-1),
            "cate_ub": np.asarray(selected["cate_ub"], dtype=float).reshape(-1),
            treatment: sanitized_train[treatment].to_numpy(),
            "default_flag": pd.to_numeric(
                sanitized_train["default_flag"], errors="coerce"
            ).to_numpy(),
        }
    )
    for col in _artifact_context_columns(sanitized_train):
        if col not in cate_df.columns:
            cate_df[col] = sanitized_train[col].to_numpy()
    cate_path = data_dir / "cate_estimates.parquet"
    atomic_write_parquet(cate_df, cate_path, index=False)

    _save_oot_cate_artifact(
        selected["estimator"],
        test_df,
        treatment=treatment,
        effect_modifiers=effect_modifiers,
        fill_values=fill_values,
        data_dir=data_dir,
    )

    estimator_bundle = {
        "selected_estimator_family": selected_name,
        "effect_modifiers": effect_modifiers,
        "confounders": confounders,
        "treatment": str(treatment),
        "estimator": selected["estimator"],
    }
    atomic_write_pickle(model_dir / "causal_estimator.pkl", estimator_bundle)
    if selected_name == "causal_forest_dml":
        atomic_write_pickle(model_dir / "causal_forest_dml.pkl", selected["estimator"])

    summary_payload = {
        "treatment": treatment,
        "run_tag": run_tag_resolved,
        "n_obs": int(len(sanitized_train)),
        "dataset_scope": dataset_scope,
        "sample_size_requested": None if sample_size is None else int(sample_size),
        "cate_mean": float(np.mean(selected["cate"])),
        "cate_std": float(np.std(selected["cate"])),
        "ci_mean_lb": float(np.mean(selected["cate_lb"])),
        "ci_mean_ub": float(np.mean(selected["cate_ub"])),
        "effect_modifiers": effect_modifiers,
        "confounders": confounders,
        "ate_controls": ate_controls,
        "policy_semantics": cfg.get("defaults", {}).get(
            "policy_semantics", "research_grade_pricing_intervention"
        ),
        "policy_value_method": cfg.get("defaults", {}).get(
            "policy_value_method", "local_cate_discrete_grid"
        ),
        "continuous_treatment_semantics": "const_marginal_effect",
        "estimator_family": selected_name,
        "ate_estimator_family": ate_info.get("estimator_family"),
    }
    atomic_write_pickle(model_dir / "causal_summary.pkl", summary_payload)

    ate_ci = ate_info.get("ate_ci", [None, None])
    status = {
        **_artifact_metadata(run_tag_resolved, dataset_scope),
        "treatment": treatment,
        "treatment_unit": cfg.get("defaults", {}).get("treatment_unit", "percentage_points"),
        "identified_estimand": "linear_dml_const_marginal_effect",
        "identification_strategy": "orthogonal_dml_with_dowhy_audit",
        "ate": float(ate_info["ate"]),
        "ate_ci": ate_ci,
        "ate_audit_dowhy": {
            "ate": dowhy_audit.get("ate"),
            "ate_ci": dowhy_audit.get("ate_ci"),
            "identified_estimand": str(dowhy_audit.get("identified_estimand")),
            "identification_strategy": str(dowhy_audit.get("identification_strategy", "backdoor")),
        },
        "cate_mean": float(np.mean(selected["cate"])),
        "cate_std": float(np.std(selected["cate"])),
        "effect_modifiers": effect_modifiers,
        "confounders": confounders,
        "ate_controls": ate_controls,
        "refutation_summary": list(dowhy_audit.get("refutation_summary", [])),
        "n_obs": int(len(sanitized_train)),
        "n_rows_input": int(sanitization["n_rows_input"]),
        "n_rows_dropped_nonfinite": int(sanitization["n_rows_dropped_nonfinite"]),
        "drop_rate": float(sanitization["drop_rate"]),
        "n_imputed_cells": int(sanitization.get("n_imputed_cells", 0)),
        "source_train_split": str(train_path),
        "source_test_split": str(test_path),
        "cate_artifact_path": str(cate_path),
        "oot_cate_artifact_path": str(data_dir / "cate_estimates_oot.parquet"),
        "overlap_artifact_path": str(overlap_path),
        "overlap_pass": bool(overlap_status["overlap_pass"]),
        "sensitivity_pass": bool(sensitivity_status["sensitivity_pass"]),
        "identification_valid": True,
        "missing_required_columns": [],
        "continuous_treatment_semantics": {
            "estimand": "const_marginal_effect",
            "interpretation": "default_probability_delta_per_1pp_rate_change",
            "policy_safe": False,
        },
        "policy_value_method": cfg.get("defaults", {}).get(
            "policy_value_method", "local_cate_discrete_grid"
        ),
        "policy_evaluation_consistent": False,
        "role": "insights_only",
        "promotion_eligible": False,
        "promotion_state": "insights_only",
        "estimator_family": selected_name,
        "ate_estimator_family": ate_info.get("estimator_family"),
        "official_method": {
            "identification": "LinearDML with DoWhy audit/refutation",
            "heterogeneity": selected_name,
            "policy_semantics": cfg.get("defaults", {}).get(
                "policy_semantics", "research_grade_pricing_intervention"
            ),
        },
        "environment_status": env_status,
    }
    status_path = model_dir / "causal_effect_status.json"
    atomic_write_json(status_path, status)
    write_last_valid_artifact(
        stage_name,
        artifact_key="causal_effect_status",
        artifact_path=status_path,
        run_tag=run_tag_resolved,
        extra={"cate_mean": float(np.mean(selected["cate"])), "n_obs": int(len(sanitized_train))},
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        run_tag=run_tag_resolved,
        extra={
            "status_path": str(status_path),
            "cate_artifact_path": str(cate_path),
            "overlap_artifact_path": str(overlap_path),
            "selected_estimator_family": selected_name,
        },
    )

    logger.info("Saved causal train CATE: {} ({} rows)", cate_path, len(cate_df))
    logger.info("Saved overlap diagnostics: {} ({} rows)", overlap_path, len(overlap))
    logger.info("Saved causal effect status: {}", status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", default="int_rate")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--config", default="configs/causal_lane.yaml")
    parser.add_argument("--cate_n_estimators", type=int, default=None)
    parser.add_argument("--cate_cv", type=int, default=None)
    parser.add_argument("--cate_mc_iters", type=int, default=None)
    parser.add_argument("--cate_criterion", default=None)
    parser.add_argument("--cate_min_balancedness_tol", type=float, default=None)
    parser.add_argument("--cate_honest", type=_parse_bool, default=None)
    args = parser.parse_args()
    main(
        treatment=args.treatment,
        sample_size=args.sample_size,
        run_tag=args.run_tag,
        config_path=args.config,
        cate_n_estimators=args.cate_n_estimators,
        cate_cv=args.cate_cv,
        cate_mc_iters=args.cate_mc_iters,
        cate_criterion=args.cate_criterion,
        cate_min_balancedness_tol=args.cate_min_balancedness_tol,
        cate_honest=args.cate_honest,
    )
