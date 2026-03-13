"""Estimate official causal artifacts for pricing-policy analysis.

Usage:
    uv run python scripts/estimate_causal_effects.py --treatment int_rate
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.causal import (
    build_overlap_diagnostics,
    default_confounders,
    default_effect_modifiers,
    estimate_ate_dowhy,
    estimate_cate,
    specify_causal_graph,
)

CAUSAL_SCHEMA_VERSION = "2026-03-07.1"


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


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    return df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _resolve_run_tag(run_tag: str | None) -> str:
    candidate = str(run_tag or "").strip()
    if candidate:
        return candidate
    env_run_tag = str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    return env_run_tag or "untracked"


def _artifact_metadata(run_tag: str, dataset_scope: str) -> dict[str, Any]:
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": run_tag,
        "dataset_scope": dataset_scope,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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


def _save_oot_cate_artifact(
    estimator: Any,
    test_df: pd.DataFrame,
    *,
    treatment: str,
    effect_modifiers: list[str],
    data_dir: Path,
) -> pd.DataFrame:
    test_df = _ensure_id(test_df)
    test_df[treatment] = _coerce_treatment(test_df[treatment])
    X_test = _numeric_frame(test_df, effect_modifiers)
    cate = estimator.effect(X_test)
    lb, ub = estimator.effect_interval(X_test, alpha=0.05)

    oot_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(str).to_numpy(),
            "cate": np.asarray(cate, dtype=float),
            "cate_lb": np.asarray(lb, dtype=float),
            "cate_ub": np.asarray(ub, dtype=float),
        }
    )
    for col in _artifact_context_columns(test_df):
        if col not in oot_df.columns:
            oot_df[col] = test_df[col].to_numpy()
    oot_path = data_dir / "cate_estimates_oot.parquet"
    oot_df.to_parquet(oot_path, index=False)
    logger.info(f"Saved OOT CATE artifact: {oot_path} ({len(oot_df):,} rows)")
    return oot_df


def main(
    treatment: str = "int_rate",
    sample_size: int | None = None,
    run_tag: str | None = None,
    cate_n_estimators: int = 200,
    cate_cv: int = 3,
    cate_mc_iters: int = 1,
    cate_criterion: str = "mse",
    cate_min_balancedness_tol: float = 0.45,
    cate_honest: bool = True,
) -> None:
    run_tag_resolved = _resolve_run_tag(run_tag)
    train_df, train_path = _load_split("train_fe", "train")
    test_df, test_path = _load_split("test_fe", "test")

    train_df = _ensure_id(train_df)
    test_df = _ensure_id(test_df)
    if treatment not in train_df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in {train_path}")
    if treatment not in test_df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in {test_path}")

    train_df[treatment] = _coerce_treatment(train_df[treatment])
    train_df = train_df.dropna(subset=[treatment, "default_flag"]).copy()
    if sample_size is not None and int(sample_size) <= 0:
        sample_size = None
    if sample_size is not None and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    dataset_scope = "full_data" if sample_size is None else "sampled"
    logger.info(
        f"Loaded {len(train_df):,} loans for causal analysis from {train_path} "
        f"(run_tag={run_tag_resolved})"
    )

    effect_modifiers = [c for c in default_effect_modifiers() if c in train_df.columns]
    confounders = [c for c in default_confounders() if c in train_df.columns]
    ate_controls = sorted(set(effect_modifiers + confounders))
    if not effect_modifiers:
        raise ValueError("No official effect modifiers found in the feature dataset.")
    if not ate_controls:
        raise ValueError("No controls available for causal identification.")

    X = _numeric_frame(train_df, effect_modifiers)
    W = _numeric_frame(train_df, confounders) if confounders else None

    ate_info = estimate_ate_dowhy(
        df=train_df,
        treatment=treatment,
        outcome="default_flag",
        common_causes=ate_controls,
        graph=specify_causal_graph(treatment=treatment, outcome="default_flag"),
    )
    estimator, cate, (lb, ub) = estimate_cate(
        Y=pd.to_numeric(train_df["default_flag"], errors="coerce"),
        T=train_df[treatment],
        X=X,
        W=W,
        n_estimators=cate_n_estimators,
        cv=cate_cv,
        mc_iters=cate_mc_iters,
        criterion=cate_criterion,
        min_balancedness_tol=cate_min_balancedness_tol,
        honest=cate_honest,
    )
    estimator._effect_modifier_columns = list(effect_modifiers)
    estimator._confounder_columns = list(confounders)
    estimator._treatment_name = str(treatment)

    logger.info(
        f"Official causal forest CATE mean={float(np.mean(cate)):.6f} std={float(np.std(cate)):.6f}"
    )
    ate_value = ate_info.get("ate")
    ate_ci = ate_info.get("ate_ci", [None, None])
    if ate_value is None:
        ate_value = float(np.mean(cate))
    if not isinstance(ate_ci, list) or len(ate_ci) < 2 or ate_ci == [None, None]:
        ate_ci = [float(np.mean(lb)), float(np.mean(ub))]

    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cate_df = pd.DataFrame(
        {
            "id": train_df["id"].astype(str).to_numpy(),
            "cate": np.asarray(cate, dtype=float),
            "cate_lb": np.asarray(lb, dtype=float),
            "cate_ub": np.asarray(ub, dtype=float),
            treatment: train_df[treatment].to_numpy(),
            "default_flag": pd.to_numeric(train_df["default_flag"], errors="coerce").to_numpy(),
        }
    )
    for col in _artifact_context_columns(train_df):
        if col not in cate_df.columns:
            cate_df[col] = train_df[col].to_numpy()
    cate_path = data_dir / "cate_estimates.parquet"
    cate_df.to_parquet(cate_path, index=False)

    overlap = build_overlap_diagnostics(
        train_df,
        treatment=treatment,
        outcome="default_flag",
    )
    overlap["dataset_scope"] = dataset_scope
    overlap["run_tag"] = run_tag_resolved
    overlap_path = data_dir / "causal_overlap_diagnostics.parquet"
    overlap.to_parquet(overlap_path, index=False)

    _save_oot_cate_artifact(
        estimator,
        test_df,
        treatment=treatment,
        effect_modifiers=effect_modifiers,
        data_dir=data_dir,
    )

    with open(model_dir / "causal_forest_dml.pkl", "wb") as f:
        pickle.dump(estimator, f)
    with open(model_dir / "causal_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "treatment": treatment,
                "run_tag": run_tag_resolved,
                "n_obs": len(train_df),
                "dataset_scope": dataset_scope,
                "sample_size_requested": None if sample_size is None else int(sample_size),
                "cate_mean": float(np.mean(cate)),
                "cate_std": float(np.std(cate)),
                "ci_mean_lb": float(np.mean(lb)),
                "ci_mean_ub": float(np.mean(ub)),
                "effect_modifiers": effect_modifiers,
                "confounders": confounders,
                "policy_semantics": "local_cate_policy_simulation",
            },
            f,
        )

    status = {
        **_artifact_metadata(run_tag_resolved, dataset_scope),
        "treatment": treatment,
        "treatment_unit": "percentage_points",
        "identified_estimand": str(ate_info.get("identified_estimand")),
        "identification_strategy": str(ate_info.get("identification_strategy", "backdoor")),
        "ate": float(ate_value),
        "ate_ci": ate_ci,
        "cate_mean": float(np.mean(cate)),
        "cate_std": float(np.std(cate)),
        "effect_modifiers": effect_modifiers,
        "confounders": confounders,
        "refutation_summary": list(ate_info.get("refutation_summary", [])),
        "n_obs": int(len(train_df)),
        "source_train_split": str(train_path),
        "source_test_split": str(test_path),
        "cate_artifact_path": str(cate_path),
        "oot_cate_artifact_path": str(data_dir / "cate_estimates_oot.parquet"),
        "overlap_artifact_path": str(overlap_path),
        "official_method": {
            "identification": "DoWhy backdoor identification/refutation",
            "heterogeneity": "EconML CausalForestDML",
            "policy_semantics": "local_cate_policy_simulation",
        },
    }
    status_path = model_dir / "causal_effect_status.json"
    _write_json(status_path, status)

    logger.info(f"Saved causal train CATE: {cate_path} ({len(cate_df):,} rows)")
    logger.info(f"Saved overlap diagnostics: {overlap_path} ({len(overlap):,} rows)")
    logger.info(f"Saved causal effect status: {status_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", default="int_rate")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--cate_n_estimators", type=int, default=200)
    parser.add_argument("--cate_cv", type=int, default=3)
    parser.add_argument("--cate_mc_iters", type=int, default=1)
    parser.add_argument("--cate_criterion", choices=["mse", "het"], default="mse")
    parser.add_argument("--cate_min_balancedness_tol", type=float, default=0.45)
    parser.add_argument(
        "--cate_honest",
        choices=["true", "false"],
        default="true",
    )
    args = parser.parse_args()
    main(
        args.treatment,
        args.sample_size,
        args.run_tag,
        cate_n_estimators=args.cate_n_estimators,
        cate_cv=args.cate_cv,
        cate_mc_iters=args.cate_mc_iters,
        cate_criterion=args.cate_criterion,
        cate_min_balancedness_tol=args.cate_min_balancedness_tol,
        cate_honest=args.cate_honest.lower() == "true",
    )
