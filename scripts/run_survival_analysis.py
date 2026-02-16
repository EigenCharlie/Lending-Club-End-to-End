"""Run survival analysis (Cox PH + Random Survival Forest).

Usage: uv run python scripts/run_survival_analysis.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.pd_model import NUMERIC_FEATURES, WOE_FEATURES
from src.models.survival import make_survival_target, train_cox_ph, train_random_survival_forest


def _ensure_survival_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time_to_event and event_observed exist.

    If not available, create a transparent proxy from loan term and default flag.
    """
    df = df.copy()
    if "event_observed" not in df.columns and "default_flag" in df.columns:
        df["event_observed"] = df["default_flag"].astype(bool)

    if "time_to_event" not in df.columns:
        if "term" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["term"]):
                term = (
                    df["term"].astype(str).str.extract(r"(\d+)")[0]
                    .pipe(pd.to_numeric, errors="coerce")
                )
            else:
                term = pd.to_numeric(df["term"], errors="coerce")
            term = term.fillna(36).clip(lower=1, upper=60)
        else:
            term = pd.Series(np.full(len(df), 36), index=df.index, dtype=float)

        # Proxy: defaults tend to occur before maturity; non-defaults censored at term.
        if "event_observed" in df.columns:
            df["time_to_event"] = np.where(df["event_observed"], np.maximum((term * 0.55).round(), 1), term)
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
    cph, cox_metrics = train_cox_ph(df[available].dropna())

    # RSF
    df_clean = df[features + ["event_observed", "time_to_event"]].dropna()
    y = make_survival_target(df_clean, event_col="event_observed", time_col="time_to_event")
    n_train = int(len(df_clean) * 0.8)
    rsf, rsf_metrics = train_random_survival_forest(
        df_clean[features].iloc[:n_train],
        y[:n_train],
        df_clean[features].iloc[n_train:],
        y[n_train:],
        n_estimators=rsf_n_estimators,
    )
    logger.info(f"Survival analysis complete: Cox={cox_metrics}, RSF={rsf_metrics}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cox_ph_model.pkl", "wb") as f:
        pickle.dump(cph, f)
    with open(model_dir / "rsf_model.pkl", "wb") as f:
        pickle.dump(rsf, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--rsf_n_estimators", type=int, default=200)
    args = parser.parse_args()
    main(sample_size=args.sample_size, rsf_n_estimators=args.rsf_n_estimators)
