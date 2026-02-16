"""Temporal backtesting for conformal interval coverage.

Produces monthly and month-grade coverage diagnostics with alert flags.

Usage:
    uv run python scripts/backtest_conformal_coverage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.conformal_artifacts import load_conformal_intervals


def _load_intervals() -> pd.DataFrame:
    df, path, is_legacy = load_conformal_intervals(allow_legacy_fallback=True)
    logger.info(f"Loaded intervals: {path} ({len(df):,} rows, legacy={is_legacy})")
    return df


def _load_test_metadata() -> pd.DataFrame:
    fe = Path("data/processed/test_fe.parquet")
    base = Path("data/processed/test.parquet")
    path = fe if fe.exists() else base
    if not path.exists():
        raise FileNotFoundError("No test dataset found for temporal backtesting.")
    cols = ["issue_d", "grade", "default_flag"]
    df = pd.read_parquet(path)
    keep = [c for c in cols if c in df.columns]
    logger.info(f"Loaded metadata: {path} ({len(df):,} rows)")
    return df[keep].copy()


def _prepare_backtest_frame(intervals: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    n = min(len(intervals), len(meta))
    if len(intervals) != len(meta):
        logger.warning(
            f"Length mismatch intervals={len(intervals):,}, meta={len(meta):,}. "
            f"Using first {n:,} aligned rows."
        )
    out = intervals.iloc[:n].reset_index(drop=True).copy()
    meta = meta.iloc[:n].reset_index(drop=True).copy()

    out["issue_d"] = pd.to_datetime(meta.get("issue_d"), errors="coerce")
    out["month"] = out["issue_d"].dt.to_period("M").dt.to_timestamp()
    if "grade" not in out.columns:
        out["grade"] = meta.get("grade", "UNKNOWN").astype(str)
    out["grade"] = out["grade"].fillna("UNKNOWN").astype(str)
    if "y_true" not in out.columns:
        out["y_true"] = pd.to_numeric(meta.get("default_flag"), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["month"]).reset_index(drop=True)
    return out


def _monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    covered_90 = (df["y_true"] >= df["pd_low_90"]) & (df["y_true"] <= df["pd_high_90"])
    covered_95 = (df["y_true"] >= df["pd_low_95"]) & (df["y_true"] <= df["pd_high_95"])
    width_90 = df["pd_high_90"] - df["pd_low_90"]
    width_95 = df["pd_high_95"] - df["pd_low_95"]

    aux = pd.DataFrame(
        {
            "month": df["month"],
            "covered_90": covered_90.astype(float),
            "covered_95": covered_95.astype(float),
            "width_90": width_90.astype(float),
            "width_95": width_95.astype(float),
        }
    )
    monthly = (
        aux.groupby("month", observed=True)
        .agg(
            n=("covered_90", "size"),
            coverage_90=("covered_90", "mean"),
            coverage_95=("covered_95", "mean"),
            avg_width_90=("width_90", "mean"),
            avg_width_95=("width_95", "mean"),
            p90_width_90=("width_90", lambda s: float(np.quantile(s, 0.90))),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["target_90"] = 0.90
    monthly["target_95"] = 0.95
    monthly["gap_90"] = monthly["coverage_90"] - monthly["target_90"]
    monthly["gap_95"] = monthly["coverage_95"] - monthly["target_95"]
    monthly["coverage_90_roll3"] = monthly["coverage_90"].rolling(3, min_periods=1).mean()
    monthly["coverage_95_roll3"] = monthly["coverage_95"].rolling(3, min_periods=1).mean()
    monthly["avg_width_90_roll3"] = monthly["avg_width_90"].rolling(3, min_periods=1).mean()
    return monthly


def _monthly_grade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    covered_90 = (df["y_true"] >= df["pd_low_90"]) & (df["y_true"] <= df["pd_high_90"])
    covered_95 = (df["y_true"] >= df["pd_low_95"]) & (df["y_true"] <= df["pd_high_95"])
    width_90 = df["pd_high_90"] - df["pd_low_90"]
    width_95 = df["pd_high_95"] - df["pd_low_95"]

    aux = pd.DataFrame(
        {
            "month": df["month"],
            "grade": df["grade"],
            "covered_90": covered_90.astype(float),
            "covered_95": covered_95.astype(float),
            "width_90": width_90.astype(float),
            "width_95": width_95.astype(float),
        }
    )
    by_grade = (
        aux.groupby(["month", "grade"], observed=True)
        .agg(
            n=("covered_90", "size"),
            coverage_90=("covered_90", "mean"),
            coverage_95=("covered_95", "mean"),
            avg_width_90=("width_90", "mean"),
            avg_width_95=("width_95", "mean"),
        )
        .reset_index()
        .sort_values(["month", "grade"])
    )
    by_grade["gap_90"] = by_grade["coverage_90"] - 0.90
    by_grade["gap_95"] = by_grade["coverage_95"] - 0.95
    return by_grade


def _build_alerts(
    monthly: pd.DataFrame,
    by_grade: pd.DataFrame,
    min_n_month: int = 1000,
    min_n_grade: int = 150,
    width_cap_90: float = 0.90,
) -> pd.DataFrame:
    alerts: list[dict[str, object]] = []

    for _, row in monthly.iterrows():
        if int(row["n"]) < min_n_month:
            continue
        cov90 = float(row["coverage_90"])
        cov95 = float(row["coverage_95"])
        w90 = float(row["avg_width_90"])
        month = row["month"]

        if cov90 < 0.88 or cov95 < 0.93 or w90 > width_cap_90:
            severity = "critical" if (cov90 < 0.87 or cov95 < 0.92) else "warning"
            alerts.append(
                {
                    "level": "portfolio",
                    "month": month,
                    "grade": "ALL",
                    "severity": severity,
                    "n": int(row["n"]),
                    "coverage_90": cov90,
                    "coverage_95": cov95,
                    "avg_width_90": w90,
                    "rule": "monthly_portfolio_threshold",
                    "recommended_action": "re-tune mondrian thresholds and review distribution drift",
                }
            )

    for _, row in by_grade.iterrows():
        if int(row["n"]) < min_n_grade:
            continue
        cov90 = float(row["coverage_90"])
        cov95 = float(row["coverage_95"])
        if cov90 < 0.84 or cov95 < 0.90:
            severity = "critical" if cov90 < 0.82 else "warning"
            alerts.append(
                {
                    "level": "grade",
                    "month": row["month"],
                    "grade": str(row["grade"]),
                    "severity": severity,
                    "n": int(row["n"]),
                    "coverage_90": cov90,
                    "coverage_95": cov95,
                    "avg_width_90": float(row["avg_width_90"]),
                    "rule": "monthly_grade_threshold",
                    "recommended_action": "increase group-specific calibration support and inspect subgroup drift",
                }
            )

    if not alerts:
        return pd.DataFrame(
            columns=[
                "level",
                "month",
                "grade",
                "severity",
                "n",
                "coverage_90",
                "coverage_95",
                "avg_width_90",
                "rule",
                "recommended_action",
            ]
        )
    return pd.DataFrame(alerts).sort_values(["month", "level", "grade"]).reset_index(drop=True)


def main():
    intervals = _load_intervals()
    meta = _load_test_metadata()
    df = _prepare_backtest_frame(intervals, meta)

    monthly = _monthly_metrics(df)
    by_grade = _monthly_grade_metrics(df)
    alerts = _build_alerts(monthly, by_grade)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = out_dir / "conformal_backtest_monthly.parquet"
    grade_path = out_dir / "conformal_backtest_monthly_grade.parquet"
    alerts_path = out_dir / "conformal_backtest_alerts.parquet"

    monthly.to_parquet(monthly_path, index=False)
    by_grade.to_parquet(grade_path, index=False)
    alerts.to_parquet(alerts_path, index=False)

    logger.info(f"Saved monthly backtest: {monthly_path} ({len(monthly):,} rows)")
    logger.info(f"Saved monthly grade backtest: {grade_path} ({len(by_grade):,} rows)")
    logger.info(f"Saved backtest alerts: {alerts_path} ({len(alerts):,} rows)")

    if not monthly.empty:
        logger.info(
            "Latest month summary: "
            f"month={monthly.iloc[-1]['month']:%Y-%m}, "
            f"cov90={monthly.iloc[-1]['coverage_90']:.4f}, "
            f"cov95={monthly.iloc[-1]['coverage_95']:.4f}, "
            f"width90={monthly.iloc[-1]['avg_width_90']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
