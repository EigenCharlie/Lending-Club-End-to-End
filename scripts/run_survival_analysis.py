"""Run survival analysis (Cox PH + Random Survival Forest).

Usage: uv run python scripts/run_survival_analysis.py
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.pd_model import load_feature_config
from src.models.survival import make_survival_target, train_cox_ph, train_random_survival_forest
from src.utils.pipeline_runtime import (
    atomic_write_parquet,
    atomic_write_pickle,
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)


def _normalize_sample_size(sample_size: int | None, *, full_data: bool = False) -> int | None:
    if full_data:
        return None
    if sample_size is None:
        return None
    sample_size = int(sample_size)
    return None if sample_size <= 0 else sample_size


def _term_to_months(term_series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(term_series):
        term = pd.to_numeric(term_series, errors="coerce")
    else:
        term = term_series.astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
    return term.fillna(36).clip(lower=1, upper=60).astype(float)


def _time_to_event_from_raw_payments(df: pd.DataFrame) -> np.ndarray | None:
    """Build time-to-event from raw payment timestamp when available."""
    if "id" not in df.columns or "issue_d" not in df.columns:
        return None

    raw_path = Path("data/raw/Loan_status_2007-2020Q3.csv")
    if not raw_path.exists():
        return None

    try:
        raw = pd.read_csv(raw_path, usecols=["id", "last_pymnt_d"], low_memory=False)
    except Exception as exc:
        logger.warning(f"Unable to load raw payment timestamps: {exc}")
        return None

    raw["id"] = raw["id"].astype(str)
    raw["last_pymnt_d"] = pd.to_datetime(raw["last_pymnt_d"], format="%b-%Y", errors="coerce")

    tmp = df[["id", "issue_d"]].copy()
    tmp["id"] = tmp["id"].astype(str)
    tmp["issue_d"] = pd.to_datetime(tmp["issue_d"], errors="coerce")
    tmp = tmp.merge(raw, on="id", how="left")
    months = ((tmp["last_pymnt_d"] - tmp["issue_d"]).dt.days / 30.44).to_numpy(dtype=float)
    return months


def _ensure_survival_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time_to_event and event_observed exist.

    If not available, create a transparent proxy from loan term and default flag.
    """
    df = df.copy()
    if "event_observed" not in df.columns and "default_flag" in df.columns:
        df["event_observed"] = df["default_flag"].astype(bool)

    if "time_to_event" not in df.columns:
        if "term" in df.columns:
            term = _term_to_months(df["term"])
        else:
            term = pd.Series(np.full(len(df), 36), index=df.index, dtype=float)

        # Preferred path: build durations from raw payment timestamps.
        from_raw = _time_to_event_from_raw_payments(df)
        if from_raw is not None:
            if "event_observed" in df.columns:
                fallback = np.where(
                    df["event_observed"], np.maximum((term * 0.55).round(), 1), term
                ).astype(float)
            else:
                fallback = term.to_numpy(dtype=float)
            time_to_event = np.asarray(from_raw, dtype=float)
            valid = np.isfinite(time_to_event) & (time_to_event > 0)
            coverage = float(valid.mean()) if len(valid) else 0.0
            time_to_event = np.where(valid, time_to_event, fallback)
            df["time_to_event"] = np.clip(time_to_event, 1.0, 60.0)
            logger.info(
                "time_to_event built from raw payment timestamps "
                f"(coverage={coverage:.1%}, fallback={1.0 - coverage:.1%})"
            )
            return df

        # Fallback proxy: defaults tend to occur before maturity; non-defaults censored at term.
        if "event_observed" in df.columns:
            df["time_to_event"] = np.where(
                df["event_observed"], np.maximum((term * 0.55).round(), 1), term
            )
        else:
            df["time_to_event"] = term

        logger.warning(
            "time_to_event not found; using proxy based on term/default_flag. "
            "For production-grade survival modeling, build targets from payment timestamps."
        )

    return df


def _resolve_survival_features(df: pd.DataFrame) -> list[str]:
    cfg = load_feature_config("data/processed/feature_config.pkl")
    configured = [str(feature) for feature in cfg.get("SURVIVAL_FEATURES", [])]
    if configured:
        features = [feature for feature in configured if feature in df.columns]
        if features:
            return features
    fallback = [
        feature
        for feature in cfg.get("NUMERIC_FEATURES", []) + cfg.get("FLAG_FEATURES", [])
        if feature in df.columns
    ]
    return fallback


def _persist_pickle_with_size_guard(
    path: Path,
    obj: object,
    *,
    max_size_mb: float,
    placeholder_payload: dict[str, object],
) -> dict[str, object]:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = len(raw) / (1024**2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if size_mb <= float(max_size_mb):
        path.write_bytes(raw)
        return {"saved": True, "size_mb": float(round(size_mb, 2)), "placeholder": False}

    with open(path, "wb") as fh:
        pickle.dump(placeholder_payload, fh)
    logger.warning(
        "Skipped full pickle for {} because estimated size {:.1f} MB exceeded {:.1f} MB",
        path,
        size_mb,
        max_size_mb,
    )
    return {
        "saved": False,
        "size_mb": float(round(size_mb, 2)),
        "placeholder": True,
        "reason": "max_size_exceeded",
    }


def main(
    sample_size: int | None = 100_000,
    rsf_n_estimators: int = 200,
    rsf_max_depth: int | None = None,
    rsf_max_samples: float | None = None,
    rsf_min_samples_leaf: int = 5,
    rsf_sample_size: int | None = None,
    rsf_n_jobs: int = -1,
    full_data: bool = False,
    rsf_max_artifact_size_mb: float = 1024.0,
):
    stage_name = "survival"
    write_runtime_status(stage_name, phase="loading_data", state="running")
    data_path = Path("data/processed/loan_master.parquet")
    if not data_path.exists():
        data_path = Path("data/processed/train_fe.parquet")
    df = pd.read_parquet(data_path)
    df = _ensure_survival_targets(df)
    sample_size = _normalize_sample_size(sample_size, full_data=full_data)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    features = _resolve_survival_features(df)
    if not features:
        raise ValueError("No survival features available.")
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    logger.info(f"Loaded {len(df):,} loans for survival analysis from {data_path}")
    write_runtime_checkpoint(
        stage_name,
        "data_loaded",
        {
            "data_path": str(data_path),
            "n_rows": int(len(df)),
            "feature_count": int(len(features)),
            "dataset_scope": "full_data" if sample_size is None else "sampled",
        },
    )

    # Cox PH
    surv_cols = features + ["time_to_event", "event_observed"]
    available = [c for c in surv_cols if c in df.columns]
    t0 = time.perf_counter()
    cph, cox_metrics = train_cox_ph(df[available].dropna())
    cox_training_time = time.perf_counter() - t0
    write_runtime_status(stage_name, phase="cox_trained", state="running")

    # RSF
    df_clean = df[features + ["event_observed", "time_to_event"]].dropna()
    if rsf_sample_size is not None:
        rsf_sample_size = None if int(rsf_sample_size) <= 0 else int(rsf_sample_size)
    if rsf_sample_size is not None and rsf_sample_size < len(df_clean):
        df_clean = df_clean.sample(n=rsf_sample_size, random_state=42).reset_index(drop=True)
    logger.info(
        "RSF training set: rows={}, estimators={}, max_depth={}, max_samples={}, min_samples_leaf={}, n_jobs={}",
        len(df_clean),
        rsf_n_estimators,
        rsf_max_depth,
        rsf_max_samples,
        rsf_min_samples_leaf,
        rsf_n_jobs,
    )
    y = make_survival_target(df_clean, event_col="event_observed", time_col="time_to_event")
    n_train = int(len(df_clean) * 0.8)
    t0 = time.perf_counter()
    rsf, rsf_metrics = train_random_survival_forest(
        df_clean[features].iloc[:n_train],
        y[:n_train],
        df_clean[features].iloc[n_train:],
        y[n_train:],
        n_estimators=rsf_n_estimators,
        min_samples_leaf=rsf_min_samples_leaf,
        max_depth=rsf_max_depth,
        max_samples=rsf_max_samples,
        n_jobs=rsf_n_jobs,
    )
    rsf_training_time = time.perf_counter() - t0
    write_runtime_checkpoint(
        stage_name,
        "survival_models_trained",
        {
            "cox_concordance_index": float(cox_metrics.get("concordance_index", 0.0)),
            "rsf_c_index": float(rsf_metrics.get("c_index", 0.0)),
            "rsf_training_rows": int(len(df_clean)),
        },
    )
    write_runtime_status(stage_name, phase="persisting_artifacts", state="running")
    logger.info(f"Survival analysis complete: Cox={cox_metrics}, RSF={rsf_metrics}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_pickle(model_dir / "cox_ph_model.pkl", cph)
    rsf_artifact = _persist_pickle_with_size_guard(
        model_dir / "rsf_model.pkl",
        rsf,
        max_size_mb=rsf_max_artifact_size_mb,
        placeholder_payload={
            "artifact_type": "rsf_model_placeholder",
            "reason": "full_model_pickle_exceeded_size_guard",
            "max_size_mb": float(rsf_max_artifact_size_mb),
            "features": features,
            "rsf_params": {
                "n_estimators": int(rsf_n_estimators),
                "min_samples_leaf": int(rsf_min_samples_leaf),
                "max_depth": None if rsf_max_depth is None else int(rsf_max_depth),
                "max_samples": None if rsf_max_samples is None else float(rsf_max_samples),
                "n_jobs": int(rsf_n_jobs),
            },
        },
    )
    event_mask = df_clean["event_observed"].astype(bool)
    time_default = df_clean.loc[event_mask, "time_to_event"]
    time_censored = df_clean.loc[~event_mask, "time_to_event"]
    atomic_write_pickle(
        model_dir / "survival_summary.pkl",
        {
            "cox_concordance_index": float(cox_metrics.get("concordance_index", 0.0)),
            "rsf_c_index_test": float(rsf_metrics.get("c_index", 0.0)),
            "cox_training_time": float(cox_training_time),
            "rsf_training_time": float(rsf_training_time),
            "n_loans": int(len(df_clean)),
            "n_events": int(event_mask.sum()),
            "event_rate": float(event_mask.mean()) if len(df_clean) else 0.0,
            "median_time_default": float(time_default.median()) if not time_default.empty else 0.0,
            "median_time_censored": float(time_censored.median())
            if not time_censored.empty
            else 0.0,
            "cox_features": features,
            "rsf_sample_size": int(len(df_clean)),
            "rsf_artifact": rsf_artifact,
            "dataset_scope": "full_data" if sample_size is None else "sampled",
            "sample_size_requested": None if sample_size is None else int(sample_size),
            "rsf_params": {
                "n_estimators": int(rsf_n_estimators),
                "min_samples_leaf": int(rsf_min_samples_leaf),
                "max_depth": None if rsf_max_depth is None else int(rsf_max_depth),
                "max_samples": None if rsf_max_samples is None else float(rsf_max_samples),
                "n_jobs": int(rsf_n_jobs),
            },
        },
    )

    # IFRS9 helper artifact: lifetime PD table by grade.
    if "grade" in df.columns and "default_flag" in df.columns:
        grade_pd = (
            df.groupby("grade", observed=True)["default_flag"]
            .mean()
            .sort_index()
            .clip(lower=0.0001, upper=0.9999)
        )
        lifetime = pd.DataFrame(
            {
                "Grade": grade_pd.index.astype(str),
                "PD_12m": grade_pd.values,
                "PD_24m": 1.0 - (1.0 - grade_pd.values) ** 2,
                "PD_36m": 1.0 - (1.0 - grade_pd.values) ** 3,
                "PD_48m": 1.0 - (1.0 - grade_pd.values) ** 4,
                "PD_60m": 1.0 - (1.0 - grade_pd.values) ** 5,
            }
        )
        out = Path("data/processed/lifetime_pd_table.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(lifetime, out, index=False)
        logger.info(f"Saved lifetime PD table to {out} ({lifetime.shape})")
        write_last_valid_artifact(
            stage_name,
            artifact_key="lifetime_pd_table",
            artifact_path=out,
            extra={"n_grade_rows": int(len(lifetime))},
        )
    write_last_valid_artifact(
        stage_name,
        artifact_key="survival_summary",
        artifact_path=model_dir / "survival_summary.pkl",
        extra={"rsf_artifact": rsf_artifact},
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        extra={
            "summary_path": str(model_dir / "survival_summary.pkl"),
            "cox_model_path": str(model_dir / "cox_ph_model.pkl"),
            "rsf_model_path": str(model_dir / "rsf_model.pkl"),
            "rsf_artifact_placeholder": bool(rsf_artifact.get("placeholder", False)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--full-data", action="store_true", dest="full_data")
    parser.add_argument("--rsf_n_estimators", type=int, default=200)
    parser.add_argument("--rsf_max_depth", type=int, default=None)
    parser.add_argument("--rsf_max_samples", type=float, default=None)
    parser.add_argument("--rsf_min_samples_leaf", type=int, default=5)
    parser.add_argument("--rsf_sample_size", type=int, default=None)
    parser.add_argument("--rsf_n_jobs", type=int, default=-1)
    parser.add_argument("--rsf_max_artifact_size_mb", type=float, default=1024.0)
    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        rsf_n_estimators=args.rsf_n_estimators,
        rsf_max_depth=args.rsf_max_depth,
        rsf_max_samples=args.rsf_max_samples,
        rsf_min_samples_leaf=args.rsf_min_samples_leaf,
        rsf_sample_size=args.rsf_sample_size,
        rsf_n_jobs=args.rsf_n_jobs,
        full_data=args.full_data,
        rsf_max_artifact_size_mb=args.rsf_max_artifact_size_mb,
    )
