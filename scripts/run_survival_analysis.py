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

from src.models.pd_model import NUMERIC_FEATURES, WOE_FEATURES
from src.models.survival import make_survival_target, train_cox_ph, train_random_survival_forest


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


def main(sample_size: int = 100_000, rsf_n_estimators: int = 200):
    data_path = Path("data/processed/loan_master.parquet")
    if not data_path.exists():
        data_path = Path("data/processed/train_fe.parquet")
    df = pd.read_parquet(data_path)
    df = _ensure_survival_targets(df)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    features = [f for f in NUMERIC_FEATURES + WOE_FEATURES if f in df.columns]
    if not features:
        raise ValueError("No survival features available.")
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    logger.info(f"Loaded {len(df):,} loans for survival analysis from {data_path}")

    # Cox PH
    surv_cols = features + ["time_to_event", "event_observed"]
    available = [c for c in surv_cols if c in df.columns]
    t0 = time.perf_counter()
    cph, cox_metrics = train_cox_ph(df[available].dropna())
    cox_training_time = time.perf_counter() - t0

    # RSF
    df_clean = df[features + ["event_observed", "time_to_event"]].dropna()
    y = make_survival_target(df_clean, event_col="event_observed", time_col="time_to_event")
    n_train = int(len(df_clean) * 0.8)
    t0 = time.perf_counter()
    rsf, rsf_metrics = train_random_survival_forest(
        df_clean[features].iloc[:n_train],
        y[:n_train],
        df_clean[features].iloc[n_train:],
        y[n_train:],
        n_estimators=rsf_n_estimators,
    )
    rsf_training_time = time.perf_counter() - t0
    logger.info(f"Survival analysis complete: Cox={cox_metrics}, RSF={rsf_metrics}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cox_ph_model.pkl", "wb") as f:
        pickle.dump(cph, f)
    with open(model_dir / "rsf_model.pkl", "wb") as f:
        pickle.dump(rsf, f)
    with open(model_dir / "survival_summary.pkl", "wb") as f:
        event_mask = df_clean["event_observed"].astype(bool)
        time_default = df_clean.loc[event_mask, "time_to_event"]
        time_censored = df_clean.loc[~event_mask, "time_to_event"]
        pickle.dump(
            {
                "cox_concordance_index": float(cox_metrics.get("concordance_index", 0.0)),
                "rsf_c_index_test": float(rsf_metrics.get("c_index", 0.0)),
                "cox_training_time": float(cox_training_time),
                "rsf_training_time": float(rsf_training_time),
                "n_loans": int(len(df_clean)),
                "n_events": int(event_mask.sum()),
                "event_rate": float(event_mask.mean()) if len(df_clean) else 0.0,
                "median_time_default": float(time_default.median())
                if not time_default.empty
                else 0.0,
                "median_time_censored": float(time_censored.median())
                if not time_censored.empty
                else 0.0,
                "cox_features": features,
                "rsf_sample_size": int(len(df_clean)),
            },
            f,
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
        lifetime.to_parquet(out, index=False)
        logger.info(f"Saved lifetime PD table to {out} ({lifetime.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--rsf_n_estimators", type=int, default=200)
    args = parser.parse_args()
    main(sample_size=args.sample_size, rsf_n_estimators=args.rsf_n_estimators)
