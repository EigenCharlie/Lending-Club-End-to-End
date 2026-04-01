"""Interpretation layer for PD calibration and backtesting diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_overall_gap_materiality(
    observed_rate: float, predicted_rate: float
) -> dict[str, float | str]:
    gap = float(observed_rate - predicted_rate)
    gap_bp = gap * 10_000.0
    abs_gap_bp = abs(gap_bp)
    if abs_gap_bp < 50.0:
        band = "low"
    elif abs_gap_bp < 100.0:
        band = "moderate"
    else:
        band = "high"
    return {
        "gap": gap,
        "gap_bp": gap_bp,
        "abs_gap_bp": abs_gap_bp,
        "materiality_band": band,
    }


def summarize_slice_materiality(
    grade_df: pd.DataFrame,
    band_df: pd.DataFrame,
    *,
    grade_gap_threshold: float = 0.025,
    band_gap_threshold: float = 0.015,
) -> dict[str, float | int]:
    grade = grade_df.copy()
    band = band_df.copy()
    if "observed_default_rate" in grade.columns and "mean_predicted_pd" in grade.columns:
        grade["abs_gap"] = (
            pd.to_numeric(grade["observed_default_rate"], errors="coerce")
            - pd.to_numeric(grade["mean_predicted_pd"], errors="coerce")
        ).abs()
    else:
        grade["abs_gap"] = np.nan
    if "rate_gap" not in band.columns and {"observed_default_rate", "mean_predicted_pd"} <= set(
        band.columns
    ):
        band["rate_gap"] = pd.to_numeric(
            band["observed_default_rate"], errors="coerce"
        ) - pd.to_numeric(band["mean_predicted_pd"], errors="coerce")
    band["abs_gap"] = pd.to_numeric(band["rate_gap"], errors="coerce").abs()
    return {
        "grade_material_breaches": int((grade["abs_gap"] > grade_gap_threshold).sum())
        if not grade.empty
        else 0,
        "max_grade_gap_bp": float(grade["abs_gap"].max() * 10_000.0) if not grade.empty else 0.0,
        "band_material_breaches": int((band["abs_gap"] > band_gap_threshold).sum())
        if not band.empty
        else 0,
        "max_band_gap_bp": float(band["abs_gap"].max() * 10_000.0) if not band.empty else 0.0,
    }


def quarter_materiality_report(
    predictions: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    min_rows_per_quarter: int = 2000,
    score_col: str = "y_prob_final",
) -> pd.DataFrame:
    n = min(len(predictions), len(meta))
    preds = predictions.iloc[:n].reset_index(drop=True).copy()
    meta_df = meta.iloc[:n].reset_index(drop=True).copy()
    issue_q = pd.to_datetime(meta_df["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
    frame = pd.DataFrame(
        {
            "issue_quarter": issue_q,
            "y_true": pd.to_numeric(preds["y_true"], errors="coerce"),
            "y_prob_final": pd.to_numeric(preds[score_col], errors="coerce"),
        }
    ).dropna()
    report = (
        frame.groupby("issue_quarter", observed=True)
        .agg(
            n_obs=("y_true", "size"),
            observed_default_rate=("y_true", "mean"),
            mean_predicted_pd=("y_prob_final", "mean"),
        )
        .reset_index()
        .sort_values("issue_quarter")
    )
    report = report.loc[report["n_obs"] >= int(min_rows_per_quarter)].reset_index(drop=True)
    report["rate_gap"] = report["observed_default_rate"] - report["mean_predicted_pd"]
    report["abs_gap_bp"] = report["rate_gap"].abs() * 10_000.0
    return report


def rare_event_summary(status: dict, report_df: pd.DataFrame) -> dict[str, float | None]:
    summary = dict(status.get("summary", {}) or {})
    deciles = (
        report_df.loc[report_df["report_type"] == "score_decile"].copy()
        if not report_df.empty
        else pd.DataFrame()
    )
    max_decile_gap_bp = None
    if not deciles.empty and {"prevalence", "mean_score"} <= set(deciles.columns):
        max_decile_gap_bp = float(
            (deciles["prevalence"] - deciles["mean_score"]).abs().max() * 10_000.0
        )
    return {
        "global_brier": summary.get("brier"),
        "global_ece": summary.get("ece"),
        "worst_protected_group_ece": status.get("worst_protected_group_ece"),
        "worst_grade_brier": status.get("worst_grade_brier"),
        "max_decile_gap_bp": max_decile_gap_bp,
    }


def validation_interpretation_status(
    *,
    overall_backtesting: dict,
    slice_materiality: dict,
    quarter_report: pd.DataFrame,
    rare_event: dict,
) -> dict[str, object]:
    gap_meta = classify_overall_gap_materiality(
        observed_rate=float(overall_backtesting.get("observed_default_rate", 0.0)),
        predicted_rate=float(overall_backtesting.get("mean_predicted_pd", 0.0)),
    )
    persistent_quarter_gaps = (
        int((quarter_report["abs_gap_bp"] >= 75.0).sum()) if not quarter_report.empty else 0
    )

    severity = "pass"
    if (
        gap_meta["materiality_band"] in {"moderate", "high"}
        or int(slice_materiality.get("grade_material_breaches", 0)) >= 2
        or int(slice_materiality.get("band_material_breaches", 0)) >= 2
        or persistent_quarter_gaps >= 2
    ):
        severity = "warning"
    if (
        gap_meta["materiality_band"] == "high"
        and int(slice_materiality.get("grade_material_breaches", 0)) >= 3
        and persistent_quarter_gaps >= 3
    ):
        severity = "fail"

    signal_type = (
        "large_sample_significance"
        if severity == "pass"
        and float(gap_meta["abs_gap_bp"]) < 50.0
        and not bool(overall_backtesting.get("predicted_pd_inside_jeffreys", False))
        else "material_slice_deviation"
    )
    return {
        "diagnostic_only": True,
        "overall_pass": bool(severity != "fail"),
        "severity": severity,
        "signal_type": signal_type,
        "summary": {
            **gap_meta,
            **slice_materiality,
            **rare_event,
            "persistent_quarter_gaps": persistent_quarter_gaps,
            "n_quarters_evaluated": int(len(quarter_report)),
            "exact_binomial_p_value": overall_backtesting.get("exact_binomial_p_value"),
            "hl_p_value": overall_backtesting.get("hl_p_value"),
            "predicted_pd_inside_jeffreys": overall_backtesting.get("predicted_pd_inside_jeffreys"),
        },
    }
