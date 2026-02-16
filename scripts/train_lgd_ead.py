"""Train LGD and EAD models.

Usage: uv run python scripts/train_lgd_ead.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from src.models.ead_model import train_ead_model
from src.models.lgd_model import train_two_stage_lgd
from src.models.pd_model import NUMERIC_FEATURES, WOE_FEATURES


def _load_split(split: str) -> pd.DataFrame:
    fe_path = Path(f"data/processed/{split}_fe.parquet")
    base_path = Path(f"data/processed/{split}.parquet")
    path = fe_path if fe_path.exists() else base_path
    return pd.read_parquet(path)


def main(sample_size: int | None = 300_000):
    train = _load_split("train")
    test = _load_split("test")
    if sample_size is not None:
        if sample_size < len(train):
            train = train.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(test):
            test = test.sample(n=sample_size, random_state=42).reset_index(drop=True)

    features = [f for f in NUMERIC_FEATURES + WOE_FEATURES if f in train.columns]
    if not features:
        raise ValueError("No model features available for LGD/EAD training.")

    train[features] = train[features].apply(pd.to_numeric, errors="coerce")
    test[features] = test[features].apply(pd.to_numeric, errors="coerce")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # LGD
    if "lgd" in train.columns and "lgd" in test.columns:
        mask_train = train["default_flag"] == 1
        mask_test = test["default_flag"] == 1
        if mask_train.any() and mask_test.any():
            clf, reg, lgd_metrics = train_two_stage_lgd(
                train.loc[mask_train, features].fillna(0.0),
                train.loc[mask_train, "lgd"],
                test.loc[mask_test, features].fillna(0.0),
                test.loc[mask_test, "lgd"],
            )
            logger.info(f"LGD metrics: {lgd_metrics}")
            with open(model_dir / "lgd_stage1_clf.pkl", "wb") as f:
                pickle.dump(clf, f)
            with open(model_dir / "lgd_stage2_reg.pkl", "wb") as f:
                pickle.dump(reg, f)

    # EAD
    if "loan_amnt" in train.columns and "loan_amnt" in test.columns:
        ead_train = train[train["default_flag"] == 1]
        ead_test = test[test["default_flag"] == 1]
        if len(ead_train) > 0 and len(ead_test) > 0:
            ead_model, ead_metrics = train_ead_model(
                ead_train[features].fillna(0.0),
                ead_train["loan_amnt"],
                ead_test[features].fillna(0.0),
                ead_test["loan_amnt"],
            )
            logger.info(f"EAD metrics: {ead_metrics}")
            ead_model.save_model(str(model_dir / "ead_catboost.cbm"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=300_000)
    args = parser.parse_args()
    main(sample_size=args.sample_size)
