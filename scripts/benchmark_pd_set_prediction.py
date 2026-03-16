"""Benchmark binary conformal prediction sets for PD ambiguity/abstention analysis."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from scripts.generate_conformal_intervals import (
    _build_feature_matrix,
    _load_model,
    _resolve_features,
)
from src.models.conformal import create_classification_sets, summarize_prediction_sets
from src.utils.io_utils import read_with_fallback

TARGET_COL = "default_flag"


def _slice_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    work = frame.loc[frame[column].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work[column] = work[column].astype(str)
    summary = (
        work.groupby(column, observed=True)
        .agg(
            n_obs=("y_true", "size"),
            set_coverage=("covered", "mean"),
            singleton_rate=("singleton", "mean"),
            ambiguity_rate=("ambiguous", "mean"),
            default_rate=("y_true", "mean"),
        )
        .reset_index()
        .rename(columns={column: "slice_value"})
    )
    ambiguous_rates = (
        work.loc[work["ambiguous"] == 1]
        .groupby(column, observed=True)["y_true"]
        .mean()
        .rename("default_rate_ambiguous")
        .reset_index()
        .rename(columns={column: "slice_value"})
    )
    summary = summary.drop(columns=["default_rate_ambiguous"], errors="ignore").merge(
        ambiguous_rates, on="slice_value", how="left"
    )
    summary.insert(0, "slice_name", column)
    return summary


def main(
    alpha: float = 0.10,
    method: str = "lac",
    calibration_size_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
) -> None:
    model, _ = _load_model()
    cal_df = read_with_fallback(
        "data/processed/calibration_fe.parquet", "data/processed/calibration.parquet"
    )
    test_df = read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")

    features, categorical = _resolve_features(model, cal_df, test_df)
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].astype(int)
    X_test = _build_feature_matrix(test_df, features, categorical)
    y_test = test_df[TARGET_COL].astype(int)

    y_pred, y_sets = create_classification_sets(
        classifier=model,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        alpha=alpha,
        method=method,
    )
    summary = summarize_prediction_sets(y_test.to_numpy(), y_pred, y_sets)

    cases = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(dtype=int),
            "y_pred_label": np.asarray(y_pred, dtype=int),
            "set_contains_0": y_sets[:, 0].astype(int),
            "set_contains_1": y_sets[:, 1].astype(int),
            "set_size": y_sets.sum(axis=1).astype(int),
            "ambiguous": (y_sets.sum(axis=1) > 1).astype(int),
            "singleton": (y_sets.sum(axis=1) == 1).astype(int),
        }
    )
    cases["covered"] = y_sets[np.arange(len(y_test)), y_test.to_numpy(dtype=int)].astype(int)
    for col in ("grade", "term", "home_ownership", "issue_d"):
        if col in test_df.columns:
            cases[col] = test_df[col].reset_index(drop=True)
    if "id" in test_df.columns:
        cases["id"] = test_df["id"].astype(str).reset_index(drop=True)
    elif "loan_id" in test_df.columns:
        cases["loan_id"] = test_df["loan_id"].astype(str).reset_index(drop=True)
    if "issue_d" in cases.columns:
        cases["issue_quarter"] = (
            pd.to_datetime(cases["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
        )

    slice_reports = []
    for col in ("grade", "term", "issue_quarter"):
        if col in cases.columns:
            report = _slice_summary(cases, col)
            if not report.empty:
                slice_reports.append(report)
    by_slice = pd.concat(slice_reports, ignore_index=True) if slice_reports else pd.DataFrame()

    sensitivity_rows: list[dict[str, float | str | int]] = []
    rng = np.random.RandomState(123)
    for frac in calibration_size_fractions:
        frac_float = float(frac)
        if frac_float <= 0 or frac_float > 1:
            continue
        n_cal = min(len(X_cal), max(1000, int(round(len(X_cal) * frac_float))))
        idx = (
            rng.choice(len(X_cal), size=n_cal, replace=False)
            if n_cal < len(X_cal)
            else np.arange(len(X_cal))
        )
        idx = np.sort(idx)
        y_pred_sub, y_sets_sub = create_classification_sets(
            classifier=model,
            X_cal=X_cal.iloc[idx].reset_index(drop=True),
            y_cal=y_cal.iloc[idx].reset_index(drop=True),
            X_test=X_test,
            alpha=alpha,
            method=method,
        )
        summary_sub = summarize_prediction_sets(y_test.to_numpy(), y_pred_sub, y_sets_sub)
        sensitivity_rows.append(
            {
                "calibration_fraction": frac_float,
                "n_calibration_rows": int(n_cal),
                "set_coverage": float(summary_sub["set_coverage"]),
                "singleton_rate": float(summary_sub["singleton_rate"]),
                "ambiguity_rate": float(summary_sub["ambiguity_rate"]),
                "default_rate_ambiguous": float(summary_sub["default_rate_ambiguous"]),
            }
        )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "pd_set_prediction_cases.parquet"
    cases.to_parquet(cases_path, index=False)
    by_slice_path = out_dir / "pd_set_prediction_by_slice.parquet"
    if not by_slice.empty:
        by_slice.to_parquet(by_slice_path, index=False)
    sensitivity_path = out_dir / "pd_set_prediction_sensitivity.parquet"
    pd.DataFrame(sensitivity_rows).to_parquet(sensitivity_path, index=False)

    status = {
        "schema_version": "2026-03-13.1",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "run_tag": "untracked",
        "status": "research_sidecar",
        "method": method,
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "summary": {k: float(v) for k, v in summary.items()},
        "artifact_path": str(cases_path),
        "by_slice_path": str(by_slice_path),
        "calibration_size_sensitivity_path": str(sensitivity_path),
        "slice_metrics": {
            "grade": by_slice.loc[by_slice["slice_name"] == "grade"].to_dict(orient="records")
            if not by_slice.empty and "slice_name" in by_slice.columns
            else [],
            "term": by_slice.loc[by_slice["slice_name"] == "term"].to_dict(orient="records")
            if not by_slice.empty and "slice_name" in by_slice.columns
            else [],
            "issue_quarter": by_slice.loc[by_slice["slice_name"] == "issue_quarter"].to_dict(
                orient="records"
            )
            if not by_slice.empty and "slice_name" in by_slice.columns
            else [],
        },
        "decision_use_case": {
            "probability_first": True,
            "set_first": False,
            "recommended_guardrail": "ambiguity_defer",
        },
        "promotion_note": (
            "Binary set prediction is currently a research-sidecar for abstention/triage. "
            "It is not part of the operational champion decision policy."
        ),
    }
    status_path = Path("models/pd_set_prediction_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved PD set prediction cases: {}", cases_path)
    logger.info("Saved PD set prediction status: {}", status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--method", default="lac")
    parser.add_argument(
        "--calibration-size-fractions",
        type=float,
        nargs="*",
        default=[0.25, 0.50, 0.75, 1.0],
    )
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        method=args.method,
        calibration_size_fractions=tuple(float(x) for x in args.calibration_size_fractions),
    )
