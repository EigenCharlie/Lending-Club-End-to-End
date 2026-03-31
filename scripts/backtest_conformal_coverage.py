"""Temporal backtesting for conformal interval coverage.

Produces monthly and month-grade coverage diagnostics with alert flags.

Usage:
    uv run python scripts/backtest_conformal_coverage.py
    uv run python scripts/backtest_conformal_coverage.py \
        --intervals-path data/processed/conformal_gap/my_ns/conformal_intervals_mondrian.parquet \
        --output-dir data/processed/conformal_gap/my_ns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.conformal_artifacts import load_conformal_intervals
from src.utils.artifact_metadata import build_artifact_metadata

try:
    from mapie.metrics.regression import (
        coverage_width_based,
        hsic,
        regression_mwi_score,
        regression_ssc,
    )

    _MAPIE_DIAG_AVAILABLE = True
except ImportError:
    _MAPIE_DIAG_AVAILABLE = False
    logger.warning("mapie.metrics.regression not available — HSIC/SSC diagnostics skipped.")


def _load_intervals(intervals_path: str | None = None) -> pd.DataFrame:
    if intervals_path:
        path = Path(intervals_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured conformal intervals artifact not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded intervals: {path} ({len(df):,} rows, legacy=False)")
        return df

    df, path, is_legacy = load_conformal_intervals(allow_legacy_fallback=False)
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
    from sklearn.metrics import brier_score_loss, log_loss

    covered_90 = (df["y_true"] >= df["pd_low_90"]) & (df["y_true"] <= df["pd_high_90"])
    covered_95 = (df["y_true"] >= df["pd_low_95"]) & (df["y_true"] <= df["pd_high_95"])
    width_90 = df["pd_high_90"] - df["pd_low_90"]
    width_95 = df["pd_high_95"] - df["pd_low_95"]

    aux = pd.DataFrame(
        {
            "month": df["month"],
            "y_true": df["y_true"].astype(float).values,
            "y_pred": df["y_pred"].astype(float).values,
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
            brier_score=(
                "y_true",
                lambda s: float(
                    brier_score_loss(aux.loc[s.index, "y_true"], aux.loc[s.index, "y_pred"])
                ),
            ),
            cal_log_loss=(
                "y_true",
                lambda s: float(
                    log_loss(
                        aux.loc[s.index, "y_true"],
                        aux.loc[s.index, "y_pred"].clip(1e-15, 1 - 1e-15),
                        labels=[0.0, 1.0],
                    )
                ),
            ),
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


def _global_diagnostic_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Compute global MAPIE diagnostic metrics: HSIC, SSC, MWI, CWC.

    These metrics test independence between interval size and coverage likelihood
    (HSIC), equity across width strata (SSC), and joint coverage-efficiency
    (Winkler / CWC). All require mapie.metrics.regression.

    Args:
        df: Prepared backtest frame with y_true, pd_low_90, pd_high_90 columns.

    Returns:
        Dict with scalar diagnostics or empty dict if mapie unavailable / data missing.
    """
    if not _MAPIE_DIAG_AVAILABLE:
        return {"mapie_diagnostics_available": False}

    required = {"y_true", "pd_low_90", "pd_high_90"}
    if not required.issubset(df.columns):
        return {"mapie_diagnostics_available": False, "reason": "missing_columns"}

    y = df["y_true"].to_numpy(dtype=float)
    lo = df["pd_low_90"].to_numpy(dtype=float)
    hi = df["pd_high_90"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[valid], lo[valid], hi[valid]

    if y.size < 100:
        return {"mapie_diagnostics_available": False, "reason": "insufficient_samples"}

    # MAPIE 1.3.0 expects y_intervals as 3D: (n_obs, 2, n_confidence_levels)
    y_intervals = np.stack([lo, hi], axis=1)[:, :, np.newaxis]  # (n, 2, 1)
    # y_pis for MWI: same shape (n, 2, 1)
    y_pis = y_intervals.copy()

    diag: dict[str, object] = {"mapie_diagnostics_available": True, "n": int(y.size)}

    # HSIC requires O(n²) kernel matrix — subsample to ~5000 for tractability
    try:
        hsic_n = min(5000, y.size)
        rng = np.random.default_rng(42)
        idx = rng.choice(y.size, size=hsic_n, replace=False)
        hsic_val = hsic(y[idx], y_intervals[idx])
        diag["hsic_90"] = float(np.mean(hsic_val))
        diag["hsic_n_subsample"] = hsic_n
        logger.info(f"HSIC @ 90%: {diag['hsic_90']:.6f} (n={hsic_n}, target ≈ 0)")
    except Exception as exc:
        diag["hsic_90"] = None
        logger.warning(f"HSIC computation failed: {exc}")

    try:
        ssc_val = regression_ssc(y, y_intervals, num_bins=3)
        diag["ssc_90"] = float(np.mean(ssc_val))
        logger.info(f"SSC @ 90%: {diag['ssc_90']:.4f} (higher = better stratified coverage)")
    except Exception as exc:
        diag["ssc_90"] = None
        logger.warning(f"SSC computation failed: {exc}")

    try:
        diag["mwi_90"] = float(regression_mwi_score(y, y_pis, confidence_level=0.90))
        logger.info(f"Mean Winkler (MAPIE) @ 90%: {diag['mwi_90']:.4f}")
    except Exception as exc:
        diag["mwi_90"] = None
        logger.warning(f"regression_mwi_score computation failed: {exc}")

    try:
        empirical_coverage = float(np.mean((y >= lo) & (y <= hi)))
        diag["cwc_90"] = float(coverage_width_based(y, lo, hi, confidence_level=0.90, eta=10.0))
        diag["empirical_coverage_global"] = empirical_coverage
        logger.info(
            f"CWC @ 90%: {diag['cwc_90']:.4f} | empirical_coverage: {empirical_coverage:.4f}"
        )
    except Exception as exc:
        diag["cwc_90"] = None
        logger.warning(f"CWC computation failed: {exc}")

    return diag


def main(intervals_path: str | None = None, output_dir: str = "data/processed"):
    intervals = _load_intervals(intervals_path)
    meta = _load_test_metadata()
    df = _prepare_backtest_frame(intervals, meta)

    monthly = _monthly_metrics(df)
    by_grade = _monthly_grade_metrics(df)
    alerts = _build_alerts(monthly, by_grade)
    global_diag = _global_diagnostic_metrics(df)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = out_dir / "conformal_backtest_monthly.parquet"
    grade_path = out_dir / "conformal_backtest_monthly_grade.parquet"
    alerts_path = out_dir / "conformal_backtest_alerts.parquet"
    diag_path = out_dir / "conformal_diagnostic_metrics.json"

    monthly.to_parquet(monthly_path, index=False)
    by_grade.to_parquet(grade_path, index=False)
    alerts.to_parquet(alerts_path, index=False)
    meta_payload = build_artifact_metadata(schema_version="2026-03-21.1")
    global_diag.update(meta_payload)
    diag_path.write_text(json.dumps(global_diag, indent=2, default=str), encoding="utf-8")

    logger.info(f"Saved monthly backtest: {monthly_path} ({len(monthly):,} rows)")
    logger.info(f"Saved monthly grade backtest: {grade_path} ({len(by_grade):,} rows)")
    logger.info(f"Saved backtest alerts: {alerts_path} ({len(alerts):,} rows)")
    logger.info(f"Saved MAPIE global diagnostics: {diag_path}")

    if not monthly.empty:
        logger.info(
            "Latest month summary: "
            f"month={monthly.iloc[-1]['month']:%Y-%m}, "
            f"cov90={monthly.iloc[-1]['coverage_90']:.4f}, "
            f"cov95={monthly.iloc[-1]['coverage_95']:.4f}, "
            f"width90={monthly.iloc[-1]['avg_width_90']:.4f}"
        )
    if global_diag.get("mapie_diagnostics_available"):
        hsic_90 = global_diag.get("hsic_90")
        ssc_90 = global_diag.get("ssc_90")
        mwi_90 = global_diag.get("mwi_90")
        cwc_90 = global_diag.get("cwc_90")
        logger.info(
            f"MAPIE diagnostics | HSIC={hsic_90:.6f}" if hsic_90 is not None else "HSIC=None"
        )
        logger.info(f"MAPIE diagnostics | SSC={ssc_90:.4f}" if ssc_90 is not None else "SSC=None")
        logger.info(f"MAPIE diagnostics | MWI={mwi_90:.4f}" if mwi_90 is not None else "MWI=None")
        logger.info(f"MAPIE diagnostics | CWC={cwc_90:.4f}" if cwc_90 is not None else "CWC=None")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals-path", default=None)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    main(intervals_path=args.intervals_path, output_dir=args.output_dir)
