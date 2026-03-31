"""Monotonicity diagnostics for promoted PD champions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pd_band_summary(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    *,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Summarize observed and expected PD by quantile score bands."""
    y_true_arr = np.asarray(y_true, dtype=float)
    pd_arr = np.asarray(pd_scores, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(pd_arr)
    y_true_arr = y_true_arr[mask]
    pd_arr = pd_arr[mask]
    if y_true_arr.size == 0:
        return pd.DataFrame(
            columns=[
                "band",
                "n_obs",
                "mean_predicted_pd",
                "observed_default_rate",
                "rate_gap",
            ]
        )

    df = pd.DataFrame({"y_true": y_true_arr, "pd_score": pd_arr})
    labels = [f"B{i + 1}" for i in range(int(n_bands))]
    df["band"] = pd.qcut(df["pd_score"], q=n_bands, labels=labels, duplicates="drop")
    summary = (
        df.groupby("band", observed=True)
        .agg(
            n_obs=("y_true", "size"),
            mean_predicted_pd=("pd_score", "mean"),
            observed_default_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return summary
    summary["rate_gap"] = (summary["observed_default_rate"] - summary["mean_predicted_pd"]).astype(
        float
    )
    summary = summary.sort_values("mean_predicted_pd", ascending=True).reset_index(drop=True)
    return summary


def adjacent_monotonicity_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare adjacent score bands and flag monotonicity disruptions."""
    if summary_df.empty or len(summary_df) < 2:
        return pd.DataFrame(
            columns=[
                "band_left",
                "band_right",
                "left_observed_default_rate",
                "right_observed_default_rate",
                "left_mean_predicted_pd",
                "right_mean_predicted_pd",
                "observed_rate_step",
                "expected_pd_step",
                "disrupted",
            ]
        )

    rows: list[dict[str, float | str | bool]] = []
    for idx in range(len(summary_df) - 1):
        left = summary_df.iloc[idx]
        right = summary_df.iloc[idx + 1]
        observed_step = float(
            float(right["observed_default_rate"]) - float(left["observed_default_rate"])
        )
        expected_step = float(float(right["mean_predicted_pd"]) - float(left["mean_predicted_pd"]))
        rows.append(
            {
                "band_left": str(left["band"]),
                "band_right": str(right["band"]),
                "left_observed_default_rate": float(left["observed_default_rate"]),
                "right_observed_default_rate": float(right["observed_default_rate"]),
                "left_mean_predicted_pd": float(left["mean_predicted_pd"]),
                "right_mean_predicted_pd": float(right["mean_predicted_pd"]),
                "observed_rate_step": observed_step,
                "expected_pd_step": expected_step,
                "disrupted": bool(observed_step < -1e-9),
            }
        )
    return pd.DataFrame(rows)


def monotonicity_status(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Aggregate monotonicity summary metrics."""
    if summary_df.empty:
        return {
            "n_bands": 0,
            "n_pairs": 0,
            "n_disruptions": 0,
            "disruption_rate": 0.0,
            "max_negative_step": 0.0,
            "overall_pass": False,
        }
    n_pairs = int(len(pair_df))
    n_disruptions = int(pair_df["disrupted"].astype(bool).sum()) if n_pairs else 0
    max_negative_step = float(pair_df["observed_rate_step"].min()) if n_pairs else 0.0
    return {
        "n_bands": int(len(summary_df)),
        "n_pairs": n_pairs,
        "n_disruptions": n_disruptions,
        "disruption_rate": float(n_disruptions / max(n_pairs, 1)),
        "max_negative_step": float(min(max_negative_step, 0.0)),
        "overall_pass": bool(n_disruptions == 0),
    }
