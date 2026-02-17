"""Simulate pricing policy decisions using CATE estimates.

Transforms causal effects into decision recommendations:
- targeted rate discounts for highly treatment-sensitive segments
- estimated impact on default risk and expected value

Usage:
    uv run python scripts/simulate_causal_policy.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _load_causal_inputs() -> tuple[pd.DataFrame, dict]:
    cate_path = Path("data/processed/cate_estimates.parquet")
    if not cate_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/cate_estimates.parquet. Run estimate_causal_effects first."
        )
    df = pd.read_parquet(cate_path)

    summary_path = Path("models/causal_summary.pkl")
    if summary_path.exists():
        with open(summary_path, "rb") as f:
            summary = pickle.load(f)
    else:
        summary = {"treatment": "int_rate"}
    return df, summary


def _coerce_numeric(df: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    if pd.api.types.is_numeric_dtype(df[col]):
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    else:
        arr = (
            df[col]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
    return np.nan_to_num(arr, nan=default)


def main(
    lgd: float = 0.45,
    high_discount_pp: float = -1.25,
    medium_discount_pp: float = -0.75,
):
    df, summary = _load_causal_inputs()
    if "cate" not in df.columns:
        raise KeyError("CATE column not present in cate_estimates artifact.")

    cate = pd.to_numeric(df["cate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    treatment = summary.get("treatment", "int_rate")
    base_rate = _coerce_numeric(
        df, treatment if treatment in df.columns else "int_rate", default=12.0
    )
    loan_amnt = _coerce_numeric(df, "loan_amnt", default=10_000.0)
    grade = (
        df["grade"].astype(str).fillna("UNKNOWN")
        if "grade" in df.columns
        else pd.Series(["UNKNOWN"] * len(df))
    )

    if "default_flag" in df.columns:
        y = pd.to_numeric(df["default_flag"], errors="coerce").fillna(0.0)
        pd_base = y.groupby(grade).transform("mean").to_numpy(dtype=float)
    else:
        pd_base = np.full(len(df), 0.15, dtype=float)

    q60 = float(np.quantile(cate, 0.60))
    q80 = float(np.quantile(cate, 0.80))
    segment = np.where(
        cate >= q80,
        "high_sensitivity",
        np.where(cate >= q60, "medium_sensitivity", "low_sensitivity"),
    )

    delta_pp = np.zeros(len(df), dtype=float)
    delta_pp[(cate >= q80) & (cate > 0)] = high_discount_pp
    delta_pp[(cate >= q60) & (cate < q80) & (cate > 0)] = medium_discount_pp

    new_rate = np.clip(base_rate + delta_pp, 5.0, None)
    effective_delta_pp = new_rate - base_rate

    pd_shift = cate * effective_delta_pp
    pd_counterfactual = np.clip(pd_base + pd_shift, 0.0, 1.0)
    avoided_pd = np.clip(pd_base - pd_counterfactual, 0.0, 1.0)

    expected_loss_reduction = avoided_pd * loan_amnt * lgd
    revenue_impact = loan_amnt * (effective_delta_pp / 100.0)
    net_value = expected_loss_reduction + revenue_impact

    action = np.where((effective_delta_pp < 0) & (net_value > 0), "decrease_rate", "hold_rate")

    out = pd.DataFrame(
        {
            "cate": cate,
            "segment": segment,
            "grade": grade.to_numpy(dtype=str),
            "base_rate_pp": base_rate,
            "recommended_delta_rate_pp": effective_delta_pp,
            "counterfactual_rate_pp": new_rate,
            "pd_base_proxy": pd_base,
            "pd_counterfactual": pd_counterfactual,
            "expected_pd_reduction": avoided_pd,
            "loan_amnt": loan_amnt,
            "expected_loss_reduction": expected_loss_reduction,
            "revenue_impact": revenue_impact,
            "net_value": net_value,
            "recommended_action": action,
        }
    )
    if "id" in df.columns:
        out["id"] = df["id"].to_numpy()

    summary_segment = (
        out.groupby("segment", observed=True)
        .agg(
            n=("net_value", "size"),
            avg_cate=("cate", "mean"),
            avg_delta_rate_pp=("recommended_delta_rate_pp", "mean"),
            avg_pd_reduction=("expected_pd_reduction", "mean"),
            total_loss_reduction=("expected_loss_reduction", "sum"),
            total_revenue_impact=("revenue_impact", "sum"),
            total_net_value=("net_value", "sum"),
            action_rate=("recommended_action", lambda s: float((s == "decrease_rate").mean())),
        )
        .reset_index()
        .sort_values("segment")
    )

    summary_grade = (
        out.groupby("grade", observed=True)
        .agg(
            n=("net_value", "size"),
            avg_cate=("cate", "mean"),
            avg_pd_reduction=("expected_pd_reduction", "mean"),
            total_net_value=("net_value", "sum"),
            action_rate=("recommended_action", lambda s: float((s == "decrease_rate").mean())),
        )
        .reset_index()
        .sort_values("grade")
    )

    overall = {
        "n_obs": int(len(out)),
        "treatment": str(treatment),
        "cate_mean": float(np.mean(cate)),
        "discount_share": float(np.mean(action == "decrease_rate")),
        "total_loss_reduction": float(np.sum(expected_loss_reduction)),
        "total_revenue_impact": float(np.sum(revenue_impact)),
        "total_net_value": float(np.sum(net_value)),
        "avg_pd_reduction": float(np.mean(avoided_pd)),
    }

    data_dir = Path("data/processed")
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    details_path = data_dir / "causal_policy_simulation.parquet"
    seg_path = data_dir / "causal_policy_segment_summary.parquet"
    grade_path = data_dir / "causal_policy_grade_summary.parquet"
    out.to_parquet(details_path, index=False)
    summary_segment.to_parquet(seg_path, index=False)
    summary_grade.to_parquet(grade_path, index=False)

    with open(model_dir / "causal_policy_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "overall": overall,
                "segment_summary": summary_segment.to_dict(orient="records"),
                "grade_summary": summary_grade.to_dict(orient="records"),
            },
            f,
        )

    logger.info(f"Saved policy simulation details: {details_path} ({len(out):,} rows)")
    logger.info(f"Saved segment policy summary: {seg_path}")
    logger.info(f"Saved grade policy summary: {grade_path}")
    logger.info(f"Overall policy impact: {overall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgd", type=float, default=0.45)
    parser.add_argument("--high_discount_pp", type=float, default=-1.25)
    parser.add_argument("--medium_discount_pp", type=float, default=-0.75)
    args = parser.parse_args()
    main(
        lgd=args.lgd,
        high_discount_pp=args.high_discount_pp,
        medium_discount_pp=args.medium_discount_pp,
    )
