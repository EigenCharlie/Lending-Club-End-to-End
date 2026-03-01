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
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.conformal import create_regression_intervals, validate_coverage
from src.models.ead_model import train_ead_model
from src.models.lgd_model import predict_two_stage, train_two_stage_lgd
from src.models.pd_model import NUMERIC_FEATURES, WOE_FEATURES

SCHEMA_VERSION = "2026-03-01.1"


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


def _load_split(split: str) -> pd.DataFrame:
    fe_path = Path(f"data/processed/{split}_fe.parquet")
    base_path = Path(f"data/processed/{split}.parquet")
    path = fe_path if fe_path.exists() else base_path
    return pd.read_parquet(path)


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


def _build_interval_frame(
    *,
    y_true: np.ndarray,
    y_pred_90: np.ndarray,
    intervals_90: np.ndarray,
    intervals_95: np.ndarray,
    index_df: pd.DataFrame,
    value_floor: float = 0.0,
) -> pd.DataFrame:
    low_90 = np.maximum(intervals_90[:, 0], value_floor)
    high_90 = np.maximum(intervals_90[:, 1], value_floor)
    low_95 = np.maximum(intervals_95[:, 0], value_floor)
    high_95 = np.maximum(intervals_95[:, 1], value_floor)
    payload: dict[str, Any] = {
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred_90, dtype=float),
        "y_low_90": low_90,
        "y_high_90": high_90,
        "y_low_95": low_95,
        "y_high_95": high_95,
        "width_90": high_90 - low_90,
        "width_95": high_95 - low_95,
    }
    if "id" in index_df.columns:
        payload["id"] = index_df["id"].astype(str).to_numpy()
    return pd.DataFrame(payload)


def _metrics_payload(
    y_true: np.ndarray, intervals_90: np.ndarray, intervals_95: np.ndarray
) -> dict[str, Any]:
    m90 = validate_coverage(np.asarray(y_true, dtype=float), intervals_90, alpha=0.10)
    m95 = validate_coverage(np.asarray(y_true, dtype=float), intervals_95, alpha=0.05)
    return {
        "metrics_90": {k: float(v) for k, v in m90.items()},
        "metrics_95": {k: float(v) for k, v in m95.items()},
    }


def _status_template(run_tag: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": run_tag,
        "lgd": {"available": False, "reason": "not_evaluated"},
        "ead": {"available": False, "reason": "not_evaluated"},
    }


def main(sample_size: int | None = None, run_tag: str | None = None) -> None:
    sample_size = _normalize_sample_size(sample_size)
    resolved_run_tag = (
        str(run_tag or "").strip() or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    )
    if not resolved_run_tag:
        resolved_run_tag = "untracked"

    train = _sample_df(_load_split("train"), sample_size)
    cal = _sample_df(_load_split("calibration"), sample_size)
    test = _sample_df(_load_split("test"), sample_size)

    features = [f for f in NUMERIC_FEATURES + WOE_FEATURES if f in train.columns]
    if not features:
        raise ValueError("No model features available for LGD/EAD training.")

    for df in (train, cal, test):
        df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    model_dir = Path("models")
    data_dir = Path("data/processed")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    status = _status_template(resolved_run_tag)
    status_path = model_dir / "conformal_lgd_ead_status.json"

    # LGD: defaults-only target with two-stage prediction wrapper.
    if "lgd" in train.columns and "lgd" in cal.columns and "lgd" in test.columns:
        mask_train = train["default_flag"] == 1
        mask_cal = cal["default_flag"] == 1
        mask_test = test["default_flag"] == 1
        if mask_train.any() and mask_cal.any() and mask_test.any():
            X_train = train.loc[mask_train, features].fillna(0.0)
            y_train = train.loc[mask_train, "lgd"].astype(float)
            X_test = test.loc[mask_test, features].fillna(0.0)
            y_test = test.loc[mask_test, "lgd"].astype(float)
            X_cal = cal.loc[mask_cal, features].fillna(0.0)
            y_cal = cal.loc[mask_cal, "lgd"].astype(float)

            clf, reg, lgd_model_metrics = train_two_stage_lgd(X_train, y_train, X_test, y_test)
            with open(model_dir / "lgd_stage1_clf.pkl", "wb") as f:
                pickle.dump(clf, f)
            with open(model_dir / "lgd_stage2_reg.pkl", "wb") as f:
                pickle.dump(reg, f)

            lgd_wrapper = _LGDTwoStageWrapper(clf=clf, reg=reg)
            y_pred_90, y_int_90 = create_regression_intervals(
                lgd_wrapper, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.10
            )
            _y_pred_95, y_int_95 = create_regression_intervals(
                lgd_wrapper, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.05
            )
            lgd_intervals_path = data_dir / "conformal_intervals_lgd.parquet"
            lgd_df = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred_90=y_pred_90,
                intervals_90=y_int_90,
                intervals_95=y_int_95,
                index_df=test.loc[mask_test].reset_index(drop=True),
                value_floor=0.0,
            )
            lgd_df.to_parquet(lgd_intervals_path, index=False)
            status["lgd"] = {
                "available": True,
                "n_train": int(mask_train.sum()),
                "n_cal": int(mask_cal.sum()),
                "n_test": int(mask_test.sum()),
                "model_metrics": {k: float(v) for k, v in lgd_model_metrics.items()},
                "conformal": _metrics_payload(y_test.to_numpy(), y_int_90, y_int_95),
                "intervals_path": str(lgd_intervals_path),
            }
            logger.info(f"Saved LGD conformal intervals: {lgd_intervals_path}")
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

            ead_model, ead_metrics = train_ead_model(X_train, y_train, X_test, y_test)
            ead_model.save_model(str(model_dir / "ead_catboost.cbm"))

            y_pred_90, y_int_90 = create_regression_intervals(
                ead_model, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.10
            )
            _y_pred_95, y_int_95 = create_regression_intervals(
                ead_model, X_cal=X_cal, y_cal=y_cal, X_test=X_test, alpha=0.05
            )
            ead_intervals_path = data_dir / "conformal_intervals_ead.parquet"
            ead_df = _build_interval_frame(
                y_true=y_test.to_numpy(),
                y_pred_90=y_pred_90,
                intervals_90=y_int_90,
                intervals_95=y_int_95,
                index_df=test.loc[mask_test].reset_index(drop=True),
                value_floor=0.0,
            )
            ead_df.to_parquet(ead_intervals_path, index=False)
            status["ead"] = {
                "available": True,
                "n_train": int(mask_train.sum()),
                "n_cal": int(mask_cal.sum()),
                "n_test": int(mask_test.sum()),
                "model_metrics": {k: float(v) for k, v in ead_metrics.items()},
                "conformal": _metrics_payload(y_test.to_numpy(), y_int_90, y_int_95),
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
    args = parser.parse_args()
    main(sample_size=args.sample_size, run_tag=args.run_tag)
