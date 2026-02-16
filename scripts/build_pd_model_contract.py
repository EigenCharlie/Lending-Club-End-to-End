"""Build and validate canonical PD model contract.

Usage:
    uv run python scripts/build_pd_model_contract.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import shutil

import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from src.models.pd_contract import (
    CANONICAL_CALIBRATOR_PATH,
    CANONICAL_MODEL_PATH,
    CONTRACT_PATH,
    build_contract_payload,
    infer_model_feature_contract,
    resolve_calibrator_path,
    resolve_model_path,
    save_contract,
    validate_features_in_splits,
)


def _read_with_fallback(fe_path: str, base_path: str) -> pd.DataFrame:
    fe = Path(fe_path)
    base = Path(base_path)
    if fe.exists():
        return pd.read_parquet(fe)
    if base.exists():
        return pd.read_parquet(base)
    raise FileNotFoundError(f"Neither {fe} nor {base} exists")


def _materialize_canonical_artifacts(model_path: Path, calibrator_path: Path | None) -> tuple[Path, Path | None]:
    """Copy model/calibrator into canonical paths if needed."""
    CANONICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() != CANONICAL_MODEL_PATH.resolve():
        shutil.copy2(model_path, CANONICAL_MODEL_PATH)
        logger.info(f"Copied canonical PD model -> {CANONICAL_MODEL_PATH}")
    else:
        logger.info(f"Canonical PD model already in place: {CANONICAL_MODEL_PATH}")

    final_calibrator: Path | None = None
    if calibrator_path is not None:
        CANONICAL_CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        if calibrator_path.resolve() != CANONICAL_CALIBRATOR_PATH.resolve():
            shutil.copy2(calibrator_path, CANONICAL_CALIBRATOR_PATH)
            logger.info(f"Copied canonical calibrator -> {CANONICAL_CALIBRATOR_PATH}")
        else:
            logger.info(f"Canonical calibrator already in place: {CANONICAL_CALIBRATOR_PATH}")
        final_calibrator = CANONICAL_CALIBRATOR_PATH

    return CANONICAL_MODEL_PATH, final_calibrator


def main():
    model_path = resolve_model_path()
    calibrator_path = resolve_calibrator_path()
    canonical_model_path, canonical_cal_path = _materialize_canonical_artifacts(model_path, calibrator_path)

    model = CatBoostClassifier()
    model.load_model(str(canonical_model_path))
    features, categorical = infer_model_feature_contract(model)
    if not features:
        raise ValueError("Unable to infer feature names from canonical PD model.")

    train_df = _read_with_fallback("data/processed/train_fe.parquet", "data/processed/train.parquet")
    cal_df = _read_with_fallback("data/processed/calibration_fe.parquet", "data/processed/calibration.parquet")
    test_df = _read_with_fallback("data/processed/test_fe.parquet", "data/processed/test.parquet")

    split_shapes, split_missing = validate_features_in_splits(
        feature_names=features,
        splits={"train": train_df, "calibration": cal_df, "test": test_df},
    )

    payload = build_contract_payload(
        model_path=canonical_model_path,
        calibrator_path=canonical_cal_path,
        feature_names=features,
        categorical_features=categorical,
        split_shapes=split_shapes,
        split_missing_features=split_missing,
    )
    save_contract(payload, CONTRACT_PATH)
    logger.info(f"PD model contract saved: {CONTRACT_PATH}")

    rows = []
    for split, missing in split_missing.items():
        rows.append(
            {
                "split": split,
                "n_rows": split_shapes[split][0],
                "n_cols": split_shapes[split][1],
                "n_missing_required_features": len(missing),
                "all_required_features_present": len(missing) == 0,
            }
        )
    validation_df = pd.DataFrame(rows)
    out_path = Path("data/processed/pd_model_contract_validation.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_parquet(out_path, index=False)
    logger.info(f"Contract validation table saved: {out_path}")

    with open("models/pd_model_contract.pkl", "wb") as f:
        pickle.dump(payload, f)
    logger.info("Contract mirror saved: models/pd_model_contract.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
