"""Train LGD/EAD models and export conformal regression interval artifacts.

Usage:
    uv run python scripts/train_lgd_ead.py --sample_size 0 --run-tag <run_tag>
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
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.conformal import _conformal_quantile, create_regression_intervals, validate_coverage
from src.models.ead_model import train_ead_model
from src.models.lgd_model import predict_two_stage, train_direct_lgd, train_two_stage_lgd
from src.models.pd_model import NUMERIC_FEATURES, WOE_FEATURES

SCHEMA_VERSION = "2026-03-02.1"
SNAPSHOT_DATE = pd.Timestamp("2020-09-30")
MATURE_MONTHS_REF = 24.0
MATURE_UNCERTAINTY_LAMBDA = 0.25
TARGET_COVERAGE_ONLINE_90 = 0.905
TARGET_COVERAGE_ONLINE_95 = 0.955
ONLINE_ETA = 0.12
ONLINE_OFFSET_RHO = 0.03
MIN_GROUP_SUPPORT = 500
MIN_YEAR_SUPPORT = 250
MAX_WIDTH_INFLATION = 1.25
MAX_ABS_BIAS = 0.10


def _catboost_backend_params(backend: str) -> dict[str, Any]:
    backend_norm = str(backend).strip().lower()
    if backend_norm == "gpu":
        return {
            "task_type": "GPU",
            "devices": "0",
            "gpu_ram_part": 0.9,
        }
    return {}


class _LGDTwoStageWrapper(BaseEstimator, RegressorMixin):
    """Expose two-stage LGD model as a prefit regressor for conformal intervals."""

    def __init__(self, clf: Any, reg: Any) -> None:
        self.clf = clf
        self.reg = reg
        self.is_fitted_ = True

    def fit(self, X, y):  # noqa: D401, ANN001
        return self

    def predict(self, X):  # noqa: D401, ANN001
        return predict_two_stage(self.clf, self.reg, X)


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "issue_d" not in out.columns:
        return out
    issue_dt = _safe_datetime(out["issue_d"])
    out["issue_d"] = issue_dt
    out["issue_year"] = issue_dt.dt.year.astype("float64")
    out["issue_month"] = issue_dt.dt.month.astype("float64")
    out["issue_quarter"] = issue_dt.dt.quarter.astype("float64")
    base_year = 2007.0
    out["issue_month_index"] = (
        (out["issue_year"] - base_year) * 12.0 + (out["issue_month"] - 1.0)
    ).astype("float64")
    return out


def _load_split(split: str) -> pd.DataFrame:
    fe_path = Path(f"data/processed/{split}_fe.parquet")
    base_path = Path(f"data/processed/{split}.parquet")
    if not base_path.exists() and not fe_path.exists():
        raise FileNotFoundError(f"Neither {fe_path} nor {base_path} exists")

    if fe_path.exists():
        fe_df = pd.read_parquet(fe_path)
        if base_path.exists():
            base_df = pd.read_parquet(base_path)
            target_cols = [
                c
                for c in ["default_flag", "loan_amnt", "lgd", "issue_d", "grade", "term"]
                if c in base_df.columns
            ]
            if target_cols:
                if "id" in fe_df.columns and "id" in base_df.columns:
                    fe_df["id"] = fe_df["id"].astype(str)
                    base_df["id"] = base_df["id"].astype(str)
                    rhs = base_df[["id", *target_cols]].drop_duplicates(subset=["id"])
                    fe_df = fe_df.drop(
                        columns=[c for c in target_cols if c in fe_df.columns], errors="ignore"
                    )
                    fe_df = fe_df.merge(rhs, on="id", how="left", validate="one_to_one")
                elif len(fe_df) == len(base_df):
                    for col in target_cols:
                        fe_df[col] = base_df[col].to_numpy()
        return _add_temporal_features(fe_df)

    return _add_temporal_features(pd.read_parquet(base_path))


def _normalize_sample_size(sample_size: int | None) -> int | None:
    if sample_size is None:
        return None
    sample_size = int(sample_size)
    return None if sample_size <= 0 else sample_size


def _sample_df(
    df: pd.DataFrame, sample_size: int | None, *, random_state: int = 42
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def _point_split_intervals(
    y_cal: np.ndarray,
    y_cal_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_cal_pred, dtype=float))
    q = _conformal_quantile(scores, alpha)
    low = _clip01(np.asarray(y_test_pred, dtype=float) - q)
    high = _clip01(np.asarray(y_test_pred, dtype=float) + q)
    return _clip01(y_test_pred), low, high, float(q)


def _cqr_intervals(
    y_cal: np.ndarray,
    low_cal: np.ndarray,
    high_cal: np.ndarray,
    low_test: np.ndarray,
    high_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y_cal_arr = np.asarray(y_cal, dtype=float)
    low_cal_arr = np.asarray(low_cal, dtype=float)
    high_cal_arr = np.asarray(high_cal, dtype=float)
    low_test_arr = np.asarray(low_test, dtype=float)
    high_test_arr = np.asarray(high_test, dtype=float)

    scores = np.maximum(low_cal_arr - y_cal_arr, y_cal_arr - high_cal_arr)
    q = _conformal_quantile(scores, alpha)
    low = _clip01(low_test_arr - q)
    high = _clip01(high_test_arr + q)
    y_pred = _clip01(0.5 * (low_test_arr + high_test_arr))
    return y_pred, low, high, float(q)


def _maturity_uncertainty_factor(issue_dates: pd.Series) -> np.ndarray:
    issue_dt = _safe_datetime(issue_dates)
    age_months = ((SNAPSHOT_DATE - issue_dt).dt.days.astype(float) / 30.4375).clip(lower=0.0)
    fragility = np.clip((MATURE_MONTHS_REF - age_months) / MATURE_MONTHS_REF, 0.0, 1.0)
    return 1.0 + MATURE_UNCERTAINTY_LAMBDA * fragility.to_numpy(dtype=float)


def _adaptive_online_intervals_by_grade(
    *,
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    issue_dates: pd.Series,
    grades: pd.Series,
    q0: float,
    target_coverage: float,
    eta: float,
    offset_init_by_grade: dict[str, float],
    offset_rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_base = _clip01(y_pred_base)
    g = grades.fillna("UNKNOWN").astype(str).to_numpy()
    dt = _safe_datetime(issue_dates)
    maturity_factor = _maturity_uncertainty_factor(issue_dates)

    order = np.argsort(dt.fillna(pd.Timestamp("1900-01-01")).to_numpy())
    y_pred_adj = np.zeros_like(y_base)
    low = np.zeros_like(y_base)
    high = np.zeros_like(y_base)

    q_by_grade: dict[str, float] = {}
    off_by_grade: dict[str, float] = {
        str(k): float(v) for k, v in offset_init_by_grade.items() if np.isfinite(v)
    }
    global_offset = float(np.mean(list(off_by_grade.values()))) if off_by_grade else 0.0

    for idx in order:
        grade = g[idx]
        q = float(q_by_grade.get(grade, q0))
        offset = float(off_by_grade.get(grade, global_offset))
        p = float(np.clip(y_base[idx] + offset, 0.0, 1.0))
        q_eff = max(0.0, q * float(maturity_factor[idx]))

        low_i = float(np.clip(p - q_eff, 0.0, 1.0))
        high_i = float(np.clip(p + q_eff, 0.0, 1.0))
        hit = 1.0 if (y_true_arr[idx] >= low_i and y_true_arr[idx] <= high_i) else 0.0

        y_pred_adj[idx] = p
        low[idx] = low_i
        high[idx] = high_i

        q_next = max(1e-4, q + eta * (target_coverage - hit))
        q_by_grade[grade] = float(q_next)

        err = float(y_true_arr[idx] - p)
        off_next = (1.0 - offset_rho) * offset + offset_rho * err
        off_by_grade[grade] = float(np.clip(off_next, -0.6, 0.6))

    return y_pred_adj, low, high, q_by_grade, off_by_grade


def _build_interval_frame(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_90: np.ndarray,
    high_90: np.ndarray,
    low_95: np.ndarray,
    high_95: np.ndarray,
    index_df: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "variant": variant,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": _clip01(y_pred),
        "y_low_90": _clip01(low_90),
        "y_high_90": _clip01(high_90),
        "y_low_95": _clip01(low_95),
        "y_high_95": _clip01(high_95),
    }
    payload["width_90"] = payload["y_high_90"] - payload["y_low_90"]
    payload["width_95"] = payload["y_high_95"] - payload["y_low_95"]

    if "id" in index_df.columns:
        payload["id"] = index_df["id"].astype(str).to_numpy()
    if "issue_d" in index_df.columns:
        payload["issue_d"] = _safe_datetime(index_df["issue_d"]).to_numpy()
    if "grade" in index_df.columns:
        payload["grade"] = index_df["grade"].astype(str).to_numpy()

    out = pd.DataFrame(payload)
    out["hit_90"] = (
        (out["y_true"] >= out["y_low_90"]) & (out["y_true"] <= out["y_high_90"])
    ).astype(float)
    out["hit_95"] = (
        (out["y_true"] >= out["y_low_95"]) & (out["y_true"] <= out["y_high_95"])
    ).astype(float)
    return out


def _coverage_summary(
    df: pd.DataFrame,
    *,
    by_col: str,
    min_support: int,
) -> tuple[float | None, pd.DataFrame]:
    if by_col not in df.columns:
        return None, pd.DataFrame(columns=[by_col, "support_n", "coverage_90", "coverage_95"])

    grouped = (
        df.groupby(by_col)
        .agg(
            support_n=("y_true", "size"),
            coverage_90=("hit_90", "mean"),
            coverage_95=("hit_95", "mean"),
            width_90=("width_90", "mean"),
            width_95=("width_95", "mean"),
        )
        .reset_index()
        .sort_values(by_col)
    )
    eligible = grouped[grouped["support_n"] >= int(min_support)].copy()
    if eligible.empty:
        return None, grouped
    return float(eligible["coverage_90"].min()), grouped


def _metrics_payload(
    y_true: np.ndarray,
    low_90: np.ndarray,
    high_90: np.ndarray,
    low_95: np.ndarray,
    high_95: np.ndarray,
) -> dict[str, Any]:
    m90 = validate_coverage(
        np.asarray(y_true, dtype=float), np.column_stack([low_90, high_90]), alpha=0.10
    )
    m95 = validate_coverage(
        np.asarray(y_true, dtype=float), np.column_stack([low_95, high_95]), alpha=0.05
    )
    return {
        "metrics_90": {k: float(v) for k, v in m90.items()},
        "metrics_95": {k: float(v) for k, v in m95.items()},
    }


def _variant_metrics(df: pd.DataFrame) -> dict[str, float]:
    out = {
        "coverage_90": float(df["hit_90"].mean()),
        "coverage_95": float(df["hit_95"].mean()),
        "avg_width_90": float(df["width_90"].mean()),
        "avg_width_95": float(df["width_95"].mean()),
        "bias": float((df["y_pred"] - df["y_true"]).mean()),
    }

    min_grade, grade_cov = _coverage_summary(df, by_col="grade", min_support=MIN_GROUP_SUPPORT)
    min_year, year_cov = _coverage_summary(
        df.assign(issue_year=pd.to_datetime(df["issue_d"], errors="coerce").dt.year),
        by_col="issue_year",
        min_support=MIN_YEAR_SUPPORT,
    )
    out["min_grade_coverage_90"] = float(min_grade) if min_grade is not None else float("nan")
    out["min_year_coverage_90"] = float(min_year) if min_year is not None else float("nan")
    out["n_test"] = float(len(df))
    out["n_grades_eval"] = float(len(grade_cov[grade_cov["support_n"] >= MIN_GROUP_SUPPORT]))
    out["n_years_eval"] = float(len(year_cov[year_cov["support_n"] >= MIN_YEAR_SUPPORT]))
    return out


def _apply_guardrails(
    benchmark: pd.DataFrame,
    *,
    baseline_variant: str,
) -> pd.DataFrame:
    out = benchmark.copy()

    baseline_row = out[out["variant"] == baseline_variant]
    if baseline_row.empty:
        base_w90 = 1.0
        base_w95 = 1.0
    else:
        base_w90 = float(baseline_row.iloc[0]["avg_width_90"])
        base_w95 = float(baseline_row.iloc[0]["avg_width_95"])

    out["width_inflation_90"] = out["avg_width_90"] / max(base_w90, 1e-9)
    out["width_inflation_95"] = out["avg_width_95"] / max(base_w95, 1e-9)

    out["check_coverage_90"] = out["coverage_90"] >= 0.90
    out["check_coverage_95"] = out["coverage_95"] >= 0.95
    out["check_min_grade_coverage_90"] = out["min_grade_coverage_90"] >= 0.85
    out["check_width_inflation_90"] = out["width_inflation_90"] <= MAX_WIDTH_INFLATION
    out["check_width_inflation_95"] = out["width_inflation_95"] <= MAX_WIDTH_INFLATION
    out["check_abs_bias"] = out["bias"].abs() <= MAX_ABS_BIAS

    out["overall_pass"] = (
        out["check_coverage_90"]
        & out["check_coverage_95"]
        & out["check_min_grade_coverage_90"]
        & out["check_width_inflation_90"]
        & out["check_width_inflation_95"]
        & out["check_abs_bias"]
    )
    return out


def _choose_variant(benchmark: pd.DataFrame) -> tuple[str, str]:
    eligible = benchmark[benchmark["overall_pass"]].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(
            ["avg_width_95", "avg_width_90", "bias"], ascending=[True, True, True]
        )
        chosen = str(eligible.iloc[0]["variant"])
        reason = "selected_best_pass_by_width"
        return chosen, reason

    fallback = benchmark.copy()
    fallback["coverage_shortfall"] = (0.90 - fallback["coverage_90"]).clip(lower=0.0) + (
        0.95 - fallback["coverage_95"]
    ).clip(lower=0.0)
    fallback = fallback.sort_values(
        ["coverage_shortfall", "avg_width_95", "avg_width_90"], ascending=[True, True, True]
    )
    chosen = str(fallback.iloc[0]["variant"])
    return chosen, "fallback_min_shortfall"


def _status_template(run_tag: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": run_tag,
        "lgd": {"available": False, "reason": "not_evaluated"},
        "ead": {"available": False, "reason": "not_evaluated"},
    }


def main(
    sample_size: int | None = None,
    run_tag: str | None = None,
    catboost_backend: str = "cpu",
) -> None:
    sample_size = _normalize_sample_size(sample_size)
    backend_params = _catboost_backend_params(catboost_backend)
    resolved_run_tag = (
        str(run_tag or "").strip() or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    )
    if not resolved_run_tag:
        resolved_run_tag = "untracked"

    train = _sample_df(_load_split("train"), sample_size)
    cal = _sample_df(_load_split("calibration"), sample_size)
    test = _sample_df(_load_split("test"), sample_size)

    features = [f for f in NUMERIC_FEATURES + WOE_FEATURES if f in train.columns]
    temporal_features = [
        c
        for c in ["issue_year", "issue_month", "issue_quarter", "issue_month_index"]
        if c in train.columns
    ]
    features = [*features, *temporal_features]

    if not features:
        raise ValueError("No model features available for LGD/EAD training.")

    for df in (train, cal, test):
        df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    status = _status_template(resolved_run_tag)
    status_path = model_dir / "conformal_lgd_ead_status.json"

    # LGD defaults-only slices.
    has_lgd = "lgd" in train.columns and "lgd" in cal.columns and "lgd" in test.columns
    if has_lgd:
        mask_train = train["default_flag"] == 1
        mask_cal = cal["default_flag"] == 1
        mask_test = test["default_flag"] == 1

        if mask_train.any() and mask_cal.any() and mask_test.any():
            train_d = train.loc[mask_train].copy().reset_index(drop=True)
            cal_d = cal.loc[mask_cal].copy().reset_index(drop=True)
            test_d = test.loc[mask_test].copy().reset_index(drop=True)

            X_train = train_d[features]
            y_train = train_d["lgd"].astype(float)
            X_cal = cal_d[features]
            y_cal = cal_d["lgd"].astype(float)
            X_test = test_d[features]
            y_test = test_d["lgd"].astype(float)

            # Variant A: two-stage + split conformal.
            clf, reg, lgd_two_stage_metrics = train_two_stage_lgd(
                X_train,
                y_train,
                X_test,
                y_test,
                params=backend_params,
            )
            with open(model_dir / "lgd_stage1_clf.pkl", "wb") as f:
                pickle.dump(clf, f)
            with open(model_dir / "lgd_stage2_reg.pkl", "wb") as f:
                pickle.dump(reg, f)

            p_cal_two_stage = _clip01(predict_two_stage(clf, reg, X_cal))
            p_test_two_stage = _clip01(predict_two_stage(clf, reg, X_test))
            y_pred_ts_90, low_ts_90, high_ts_90, q_ts_90 = _point_split_intervals(
                y_cal.to_numpy(), p_cal_two_stage, p_test_two_stage, alpha=0.10
            )
            _y_pred_ts_95, low_ts_95, high_ts_95, q_ts_95 = _point_split_intervals(
                y_cal.to_numpy(), p_cal_two_stage, p_test_two_stage, alpha=0.05
            )

            variant_frames: dict[str, pd.DataFrame] = {}
            variant_frames["two_stage_split"] = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred=y_pred_ts_90,
                low_90=low_ts_90,
                high_90=high_ts_90,
                low_95=low_ts_95,
                high_95=high_ts_95,
                index_df=test_d,
                variant="two_stage_split",
            )

            # Variant B: direct regressor + split conformal.
            direct_model, lgd_direct_metrics = train_direct_lgd(
                X_train,
                y_train,
                X_test,
                y_test,
                params=backend_params,
            )
            with open(model_dir / "lgd_direct_reg.pkl", "wb") as f:
                pickle.dump(direct_model, f)

            p_cal_direct = _clip01(direct_model.predict(X_cal))
            p_test_direct = _clip01(direct_model.predict(X_test))
            y_pred_dir_90, low_dir_90, high_dir_90, q_dir_90 = _point_split_intervals(
                y_cal.to_numpy(), p_cal_direct, p_test_direct, alpha=0.10
            )
            _y_pred_dir_95, low_dir_95, high_dir_95, q_dir_95 = _point_split_intervals(
                y_cal.to_numpy(), p_cal_direct, p_test_direct, alpha=0.05
            )

            variant_frames["direct_split"] = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred=y_pred_dir_90,
                low_90=low_dir_90,
                high_90=high_dir_90,
                low_95=low_dir_95,
                high_95=high_dir_95,
                index_df=test_d,
                variant="direct_split",
            )

            # Variant C: direct CQR + conformalized quantiles.
            q05 = CatBoostRegressor(
                iterations=900,
                learning_rate=0.03,
                depth=7,
                loss_function="Quantile:alpha=0.05",
                verbose=0,
                random_seed=42,
                early_stopping_rounds=40,
                **backend_params,
            )
            q95 = CatBoostRegressor(
                iterations=900,
                learning_rate=0.03,
                depth=7,
                loss_function="Quantile:alpha=0.95",
                verbose=0,
                random_seed=42,
                early_stopping_rounds=40,
                **backend_params,
            )
            q025 = CatBoostRegressor(
                iterations=900,
                learning_rate=0.03,
                depth=7,
                loss_function="Quantile:alpha=0.025",
                verbose=0,
                random_seed=42,
                early_stopping_rounds=40,
                **backend_params,
            )
            q975 = CatBoostRegressor(
                iterations=900,
                learning_rate=0.03,
                depth=7,
                loss_function="Quantile:alpha=0.975",
                verbose=0,
                random_seed=42,
                early_stopping_rounds=40,
                **backend_params,
            )
            q05.fit(X_train, y_train, eval_set=(X_test, y_test))
            q95.fit(X_train, y_train, eval_set=(X_test, y_test))
            q025.fit(X_train, y_train, eval_set=(X_test, y_test))
            q975.fit(X_train, y_train, eval_set=(X_test, y_test))

            with open(model_dir / "lgd_q05_reg.pkl", "wb") as f:
                pickle.dump(q05, f)
            with open(model_dir / "lgd_q95_reg.pkl", "wb") as f:
                pickle.dump(q95, f)
            with open(model_dir / "lgd_q025_reg.pkl", "wb") as f:
                pickle.dump(q025, f)
            with open(model_dir / "lgd_q975_reg.pkl", "wb") as f:
                pickle.dump(q975, f)

            q05_cal = q05.predict(X_cal)
            q95_cal = q95.predict(X_cal)
            q05_test = q05.predict(X_test)
            q95_test = q95.predict(X_test)
            y_pred_cqr_90, low_cqr_90, high_cqr_90, q_cqr_90 = _cqr_intervals(
                y_cal.to_numpy(), q05_cal, q95_cal, q05_test, q95_test, alpha=0.10
            )

            q025_cal = q025.predict(X_cal)
            q975_cal = q975.predict(X_cal)
            q025_test = q025.predict(X_test)
            q975_test = q975.predict(X_test)
            _y_pred_cqr_95, low_cqr_95, high_cqr_95, q_cqr_95 = _cqr_intervals(
                y_cal.to_numpy(), q025_cal, q975_cal, q025_test, q975_test, alpha=0.05
            )

            variant_frames["direct_cqr"] = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred=y_pred_cqr_90,
                low_90=low_cqr_90,
                high_90=high_cqr_90,
                low_95=low_cqr_95,
                high_95=high_cqr_95,
                index_df=test_d,
                variant="direct_cqr",
            )

            # Variant D: direct point + adaptive grade-temporal online conformal.
            cal_residual = y_cal.to_numpy() - p_cal_direct
            residual_df = pd.DataFrame(
                {
                    "grade": cal_d.get("grade", pd.Series(["UNKNOWN"] * len(cal_d)))
                    .fillna("UNKNOWN")
                    .astype(str),
                    "residual": cal_residual,
                }
            )
            offset_init_by_grade = (
                residual_df.groupby("grade", observed=True)["residual"].mean().to_dict()
            )

            y_pred_adapt_90, low_adapt_90, high_adapt_90, q_end_90, off_end_90 = (
                _adaptive_online_intervals_by_grade(
                    y_true=y_test.to_numpy(),
                    y_pred_base=p_test_direct,
                    issue_dates=test_d.get("issue_d", pd.Series([pd.NaT] * len(test_d))),
                    grades=test_d.get("grade", pd.Series(["UNKNOWN"] * len(test_d))),
                    q0=q_dir_90,
                    target_coverage=TARGET_COVERAGE_ONLINE_90,
                    eta=ONLINE_ETA,
                    offset_init_by_grade=offset_init_by_grade,
                    offset_rho=ONLINE_OFFSET_RHO,
                )
            )
            _y_pred_adapt_95, low_adapt_95, high_adapt_95, q_end_95, off_end_95 = (
                _adaptive_online_intervals_by_grade(
                    y_true=y_test.to_numpy(),
                    y_pred_base=p_test_direct,
                    issue_dates=test_d.get("issue_d", pd.Series([pd.NaT] * len(test_d))),
                    grades=test_d.get("grade", pd.Series(["UNKNOWN"] * len(test_d))),
                    q0=q_dir_95,
                    target_coverage=TARGET_COVERAGE_ONLINE_95,
                    eta=ONLINE_ETA,
                    offset_init_by_grade=offset_init_by_grade,
                    offset_rho=ONLINE_OFFSET_RHO,
                )
            )

            variant_frames["direct_adaptive_grade_temporal"] = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred=y_pred_adapt_90,
                low_90=low_adapt_90,
                high_90=high_adapt_90,
                low_95=low_adapt_95,
                high_95=high_adapt_95,
                index_df=test_d,
                variant="direct_adaptive_grade_temporal",
            )

            # Benchmark and guardrails.
            benchmark_rows: list[dict[str, Any]] = []
            for variant_name, frame in variant_frames.items():
                row = {"variant": variant_name, **_variant_metrics(frame)}
                benchmark_rows.append(row)

            benchmark = pd.DataFrame(benchmark_rows)
            benchmark = _apply_guardrails(benchmark, baseline_variant="two_stage_split")
            chosen_variant, selection_reason = _choose_variant(benchmark)

            benchmark_path = data_dir / "lgd_variant_benchmark.parquet"
            benchmark.to_parquet(benchmark_path, index=False)

            diagnostics_path = data_dir / "lgd_coverage_diagnostics.parquet"
            diagnostics_df = pd.concat(list(variant_frames.values()), ignore_index=True)
            diagnostics_df.insert(0, "run_tag", resolved_run_tag)
            diagnostics_df.to_parquet(diagnostics_path, index=False)

            selected_df = variant_frames[chosen_variant].copy()
            lgd_intervals_path = data_dir / "conformal_intervals_lgd.parquet"
            selected_df.drop(columns=["variant"], errors="ignore").to_parquet(
                lgd_intervals_path, index=False
            )

            guardrail_row = benchmark[benchmark["variant"] == chosen_variant].iloc[0].to_dict()
            guardrails_payload = {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "run_tag": resolved_run_tag,
                "selected_variant": chosen_variant,
                "selection_reason": selection_reason,
                "checks": {
                    "coverage90_ok": bool(guardrail_row.get("check_coverage_90", False)),
                    "coverage95_ok": bool(guardrail_row.get("check_coverage_95", False)),
                    "min_group_coverage90_ok": bool(
                        guardrail_row.get("check_min_grade_coverage_90", False)
                    ),
                    "width_inflation90_ok": bool(
                        guardrail_row.get("check_width_inflation_90", False)
                    ),
                    "width_inflation95_ok": bool(
                        guardrail_row.get("check_width_inflation_95", False)
                    ),
                    "bias_ok": bool(guardrail_row.get("check_abs_bias", False)),
                },
                "overall_pass": bool(guardrail_row.get("overall_pass", False)),
                "targets": {
                    "coverage_90": 0.90,
                    "coverage_95": 0.95,
                    "min_grade_coverage_90": 0.85,
                    "max_width_inflation": MAX_WIDTH_INFLATION,
                    "max_abs_bias": MAX_ABS_BIAS,
                },
            }
            guardrails_path = model_dir / "lgd_guardrails_status.json"
            guardrails_path.write_text(
                json.dumps(guardrails_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            metrics_90 = validate_coverage(
                selected_df["y_true"].to_numpy(dtype=float),
                selected_df[["y_low_90", "y_high_90"]].to_numpy(dtype=float),
                alpha=0.10,
            )
            metrics_95 = validate_coverage(
                selected_df["y_true"].to_numpy(dtype=float),
                selected_df[["y_low_95", "y_high_95"]].to_numpy(dtype=float),
                alpha=0.05,
            )

            grade_cov = (
                selected_df.groupby("grade", observed=True)
                .agg(
                    support_n=("y_true", "size"),
                    coverage_90=("hit_90", "mean"),
                    coverage_95=("hit_95", "mean"),
                )
                .reset_index()
            )
            grade_cov_path = data_dir / "lgd_coverage_by_grade.parquet"
            grade_cov.to_parquet(grade_cov_path, index=False)

            year_cov = (
                selected_df.assign(
                    issue_year=pd.to_datetime(selected_df["issue_d"], errors="coerce").dt.year
                )
                .groupby("issue_year", observed=True)
                .agg(
                    support_n=("y_true", "size"),
                    coverage_90=("hit_90", "mean"),
                    coverage_95=("hit_95", "mean"),
                )
                .reset_index()
            )
            year_cov_path = data_dir / "lgd_coverage_by_year.parquet"
            year_cov.to_parquet(year_cov_path, index=False)

            status["lgd"] = {
                "available": True,
                "n_train": int(len(train_d)),
                "n_cal": int(len(cal_d)),
                "n_test": int(len(test_d)),
                "schema_variant": "adaptive_lgd_v2",
                "selected_variant": chosen_variant,
                "selection_reason": selection_reason,
                "model_metrics": {
                    "two_stage": {k: float(v) for k, v in lgd_two_stage_metrics.items()},
                    "direct": {k: float(v) for k, v in lgd_direct_metrics.items()},
                },
                "conformal": {
                    "metrics_90": {k: float(v) for k, v in metrics_90.items()},
                    "metrics_95": {k: float(v) for k, v in metrics_95.items()},
                    "policy": {
                        "target_coverage_online_90": TARGET_COVERAGE_ONLINE_90,
                        "target_coverage_online_95": TARGET_COVERAGE_ONLINE_95,
                        "eta": ONLINE_ETA,
                        "offset_rho": ONLINE_OFFSET_RHO,
                        "maturity_uncertainty_lambda": MATURE_UNCERTAINTY_LAMBDA,
                        "maturity_months_reference": MATURE_MONTHS_REF,
                        "q_start_90": float(q_dir_90),
                        "q_start_95": float(q_dir_95),
                        "q_end_by_grade_90": {str(k): float(v) for k, v in q_end_90.items()},
                        "q_end_by_grade_95": {str(k): float(v) for k, v in q_end_95.items()},
                        "offset_end_by_grade_90": {str(k): float(v) for k, v in off_end_90.items()},
                        "offset_end_by_grade_95": {str(k): float(v) for k, v in off_end_95.items()},
                    },
                },
                "guardrails": {
                    "overall_pass": bool(guardrail_row.get("overall_pass", False)),
                    "coverage_90": float(guardrail_row.get("coverage_90", 0.0)),
                    "coverage_95": float(guardrail_row.get("coverage_95", 0.0)),
                    "min_grade_coverage_90": float(guardrail_row.get("min_grade_coverage_90", 0.0)),
                    "width_inflation_90": float(guardrail_row.get("width_inflation_90", 0.0)),
                    "width_inflation_95": float(guardrail_row.get("width_inflation_95", 0.0)),
                    "abs_bias": float(abs(guardrail_row.get("bias", 0.0))),
                    "status_path": str(guardrails_path),
                },
                "artifacts": {
                    "intervals_path": str(lgd_intervals_path),
                    "benchmark_path": str(benchmark_path),
                    "diagnostics_path": str(diagnostics_path),
                    "coverage_by_grade_path": str(grade_cov_path),
                    "coverage_by_year_path": str(year_cov_path),
                },
                "benchmark_variants": benchmark.to_dict(orient="records"),
            }
            logger.info(f"Saved LGD conformal intervals: {lgd_intervals_path}")
            logger.info(
                "LGD selected variant={}, cov90={:.4f}, cov95={:.4f}, width90={:.4f}, width95={:.4f}",
                chosen_variant,
                float(metrics_90["empirical_coverage"]),
                float(metrics_95["empirical_coverage"]),
                float(metrics_90["avg_interval_width"]),
                float(metrics_95["avg_interval_width"]),
            )
        else:
            status["lgd"] = {"available": False, "reason": "insufficient_default_rows"}
    else:
        status["lgd"] = {"available": False, "reason": "missing_lgd_column"}

    # EAD: defaults-only regression target.
    if "loan_amnt" in train.columns and "loan_amnt" in cal.columns and "loan_amnt" in test.columns:
        mask_train = train["default_flag"] == 1
        mask_cal = cal["default_flag"] == 1
        mask_test = test["default_flag"] == 1
        if mask_train.any() and mask_cal.any() and mask_test.any():
            X_train = train.loc[mask_train, features].fillna(0.0)
            y_train = train.loc[mask_train, "loan_amnt"].astype(float)
            X_test = test.loc[mask_test, features].fillna(0.0)
            y_test = test.loc[mask_test, "loan_amnt"].astype(float)
            X_cal = cal.loc[mask_cal, features].fillna(0.0)
            y_cal = cal.loc[mask_cal, "loan_amnt"].astype(float)

            ead_model, ead_metrics = train_ead_model(
                X_train,
                y_train,
                X_test,
                y_test,
                params=backend_params,
            )
            ead_model.save_model(str(model_dir / "ead_catboost.cbm"))

            y_pred_90, y_int_90 = create_regression_intervals(
                ead_model, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.10
            )
            _y_pred_95, y_int_95 = create_regression_intervals(
                ead_model, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.05
            )
            ead_intervals_path = data_dir / "conformal_intervals_ead.parquet"

            ead_df = pd.DataFrame(
                {
                    "y_true": y_test.to_numpy(dtype=float),
                    "y_pred": np.asarray(y_pred_90, dtype=float),
                    "y_low_90": np.maximum(y_int_90[:, 0], 0.0),
                    "y_high_90": np.maximum(y_int_90[:, 1], 0.0),
                    "y_low_95": np.maximum(y_int_95[:, 0], 0.0),
                    "y_high_95": np.maximum(y_int_95[:, 1], 0.0),
                }
            )
            ead_df["width_90"] = ead_df["y_high_90"] - ead_df["y_low_90"]
            ead_df["width_95"] = ead_df["y_high_95"] - ead_df["y_low_95"]
            if "id" in test.columns:
                ead_df["id"] = test.loc[mask_test, "id"].astype(str).to_numpy()
            ead_df.to_parquet(ead_intervals_path, index=False)

            status["ead"] = {
                "available": True,
                "n_train": int(mask_train.sum()),
                "n_cal": int(mask_cal.sum()),
                "n_test": int(mask_test.sum()),
                "model_metrics": {k: float(v) for k, v in ead_metrics.items()},
                "conformal": _metrics_payload(
                    y_test.to_numpy(),
                    ead_df["y_low_90"].to_numpy(),
                    ead_df["y_high_90"].to_numpy(),
                    ead_df["y_low_95"].to_numpy(),
                    ead_df["y_high_95"].to_numpy(),
                ),
                "intervals_path": str(ead_intervals_path),
            }
            logger.info(f"Saved EAD conformal intervals: {ead_intervals_path}")
        else:
            status["ead"] = {"available": False, "reason": "insufficient_default_rows"}
    else:
        status["ead"] = {"available": False, "reason": "missing_loan_amnt_column"}

    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved LGD/EAD conformal status: {status_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--catboost_backend", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        run_tag=args.run_tag,
        catboost_backend=args.catboost_backend,
    )
