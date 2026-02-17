"""Estimate causal effects using Double ML.

Usage: uv run python scripts/estimate_causal_effects.py --treatment int_rate
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.causal import estimate_cate


def _coerce_treatment(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype(str).str.strip().str.rstrip("%").pipe(pd.to_numeric, errors="coerce")


def main(treatment: str = "int_rate", sample_size: int | None = None):
    fe_path = Path("data/processed/train_fe.parquet")
    base_path = Path("data/processed/train.parquet")
    data_path = fe_path if fe_path.exists() else base_path

    df = pd.read_parquet(data_path)
    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in {data_path}")

    df[treatment] = _coerce_treatment(df[treatment])
    df = df.dropna(subset=[treatment, "default_flag"]).copy()
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} loans for causal analysis from {data_path}")

    effect_modifiers = ["loan_amnt", "annual_inc", "dti", "fico_range_low"]
    confounders = ["grade_woe", "purpose_woe", "home_ownership_woe"]
    available_x = [c for c in effect_modifiers if c in df.columns]
    available_w = [c for c in confounders if c in df.columns]

    X = df[available_x].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    W = df[available_w].apply(pd.to_numeric, errors="coerce").fillna(0.0) if available_w else None

    est, cate, (lb, ub) = estimate_cate(
        Y=df["default_flag"],
        T=df[treatment],
        X=X,
        W=W,
    )
    logger.info(f"CATE of {treatment} on default: mean={cate.mean():.6f}")

    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cate_df = pd.DataFrame(
        {
            "cate": cate,
            "cate_lb": lb,
            "cate_ub": ub,
            treatment: df[treatment].to_numpy(),
            "default_flag": df["default_flag"].to_numpy(),
        }
    )
    # Keep decision-relevant identifiers/context for policy simulation.
    for col in [
        "id",
        "grade",
        "loan_amnt",
        "annual_inc",
        "dti",
        "int_rate",
        "purpose",
        "home_ownership",
    ]:
        if col in df.columns and col not in cate_df.columns:
            cate_df[col] = df[col].to_numpy()
    cate_df.to_parquet(data_dir / "cate_estimates.parquet", index=False)

    with open(model_dir / "causal_forest_dml.pkl", "wb") as f:
        pickle.dump(est, f)
    with open(model_dir / "causal_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "treatment": treatment,
                "n_obs": len(df),
                "cate_mean": float(np.mean(cate)),
                "cate_std": float(np.std(cate)),
                "ci_mean_lb": float(np.mean(lb)),
                "ci_mean_ub": float(np.mean(ub)),
                "effect_modifiers": available_x,
                "confounders": available_w,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", default="int_rate")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    main(args.treatment, args.sample_size)
