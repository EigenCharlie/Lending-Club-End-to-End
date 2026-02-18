"""Train PD model with calibration and conformal prediction.

Usage:
    uv run python scripts/train_pd_model.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.evaluation.metrics import classification_metrics
from src.models.calibration import calibrate_platt, evaluate_calibration
from src.models.conformal import create_pd_intervals, validate_coverage
from src.models.pd_contract import (
    CANONICAL_CALIBRATOR_PATH,
    CANONICAL_MODEL_PATH,
    CONTRACT_PATH,
    build_contract_payload,
    infer_model_feature_contract,
    save_contract,
    validate_features_in_splits,
)
from src.models.pd_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    WOE_FEATURES,
    get_available_features,
    train_baseline,
    train_catboost,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _read_with_fallback(path: str) -> pd.DataFrame:
    """Read configured path; fallback between *_fe and base split if needed."""
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)

    alt = None
    if p.name.endswith("_fe.parquet"):
        alt = p.with_name(p.name.replace("_fe.parquet", ".parquet"))
    elif p.name.endswith(".parquet"):
        alt = p.with_name(p.name.replace(".parquet", "_fe.parquet"))

    if alt is not None and alt.exists():
        logger.warning(f"Configured path not found: {p}. Falling back to {alt}")
        return pd.read_parquet(alt)

    raise FileNotFoundError(f"Neither configured path nor fallback exists: {p}")


def _normalize_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known percent-like string columns when present."""
    df = df.copy()
    for col in ("int_rate", "revol_util"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col].astype(str).str.strip().str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
            )
    if "term" in df.columns and not pd.api.types.is_numeric_dtype(df["term"]):
        df["term"] = (
            df["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    return df


def main(config_path: str = "configs/pd_model.yaml", sample_size: int | None = None):
    config = load_config(config_path)
    logger.info(f"Config loaded from {config_path}")

    train = _normalize_percent_columns(_read_with_fallback(config["data"]["train_path"]))
    test = _normalize_percent_columns(_read_with_fallback(config["data"]["test_path"]))
    cal = _normalize_percent_columns(_read_with_fallback(config["data"]["calibration_path"]))
    if sample_size is not None:
        if sample_size < len(train):
            train = train.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(test):
            test = test.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(cal):
            cal = cal.sample(n=sample_size, random_state=42).reset_index(drop=True)

    features = get_available_features(train)
    if not features:
        raise ValueError("No available PD features found in training data.")
    logger.info(f"Using {len(features)} features")
    feature_config_path = Path("data/processed/feature_config.pkl")
    if not feature_config_path.exists():
        feature_cfg = {
            "CATBOOST_FEATURES": features,
            "LOGREG_FEATURES": [c for c in features if c not in CATEGORICAL_FEATURES],
            "NUMERIC_FEATURES": [c for c in NUMERIC_FEATURES if c in features],
            "WOE_FEATURES": [c for c in WOE_FEATURES if c in features],
            "CATEGORICAL_FEATURES": [c for c in CATEGORICAL_FEATURES if c in features],
            "INTERACTION_FEATURES": [c for c in features if "__" in c],
            "FLAG_FEATURES": [c for c in features if c.endswith("_flag")],
            "iv_scores": {},
        }
        feature_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_config_path, "wb") as f:
            pickle.dump(feature_cfg, f)
        logger.info(f"Generated minimal feature config at {feature_config_path}")

    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    X_cal, y_cal = cal[features], cal[TARGET]

    # Baseline LR
    X_train_num = X_train.select_dtypes(include="number").fillna(0)
    X_test_num = X_test.select_dtypes(include="number").fillna(0)
    lr_model, lr_metrics = train_baseline(X_train_num, y_train, X_test_num, y_test)

    # CatBoost
    cb_model, cb_metrics = train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        params=config["model"]["params"],
    )

    # Calibration (Platt sigmoid — selected in NB03, ECE=0.0128 on test)
    cal_model = calibrate_platt(cb_model, X_cal, y_cal)
    y_prob_test_raw = cb_model.predict_proba(X_test)[:, 1]
    y_prob_calibrated = cal_model.predict_proba(y_prob_test_raw.reshape(-1, 1))[:, 1]
    cal_metrics = evaluate_calibration(y_test.values, y_prob_calibrated, "platt")

    # Conformal
    alpha = 1.0 - config["conformal"].get("confidence_level", 0.9)
    _, y_intervals = create_pd_intervals(cb_model, X_cal, y_cal, X_test, alpha=alpha)
    cp_metrics = validate_coverage(y_test.values.astype(float), y_intervals, alpha)

    all_metrics = {
        **cb_metrics,
        **cal_metrics,
        **{f"cp_{k}": v for k, v in cp_metrics.items()},
        "baseline_auc": lr_metrics["auc_roc"],
    }
    final_test_metrics = classification_metrics(y_test.values, y_prob_calibrated)
    logger.info(f"Final metrics: {all_metrics}")

    model_path = Path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cb_model.save_model(str(model_path))

    cal_path = Path(config["output"]["conformal_path"])
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "wb") as f:
        pickle.dump(cal_model, f)

    # Canonical artifacts for unified downstream loading.
    CANONICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL_CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() != CANONICAL_MODEL_PATH.resolve():
        shutil.copy2(model_path, CANONICAL_MODEL_PATH)
    else:
        # Ensure canonical path exists even if configured path already canonical.
        CANONICAL_MODEL_PATH.touch(exist_ok=True)
    if cal_path.resolve() != CANONICAL_CALIBRATOR_PATH.resolve():
        shutil.copy2(cal_path, CANONICAL_CALIBRATOR_PATH)
    else:
        CANONICAL_CALIBRATOR_PATH.touch(exist_ok=True)

    # Persist model contract for strict feature alignment across scripts.
    model_for_contract = cb_model
    features_contract, categorical_contract = infer_model_feature_contract(model_for_contract)
    split_shapes, split_missing = validate_features_in_splits(
        feature_names=features_contract,
        splits={"train": train, "calibration": cal, "test": test},
    )
    contract_payload = build_contract_payload(
        model_path=CANONICAL_MODEL_PATH,
        calibrator_path=CANONICAL_CALIBRATOR_PATH,
        feature_names=features_contract,
        categorical_features=categorical_contract,
        split_shapes=split_shapes,
        split_missing_features=split_missing,
    )
    save_contract(contract_payload, CONTRACT_PATH)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Calibrator saved to {cal_path}")
    logger.info(f"Canonical PD model saved to {CANONICAL_MODEL_PATH}")
    logger.info(f"Canonical calibrator saved to {CANONICAL_CALIBRATOR_PATH}")
    logger.info(f"PD contract saved to {CONTRACT_PATH}")

    # Persist test predictions + training record for downstream DVC/Streamlit/MLflow contracts.
    test_predictions_path = Path("data/processed/test_predictions.parquet")
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame(
        {
            "loan_id": test["id"].astype(str) if "id" in test.columns else test.index.astype(str),
            "y_true": y_test.values.astype(float),
            "y_prob_lr": lr_model.predict_proba(X_test_num)[:, 1].astype(float),
            "y_prob_cb_default": y_prob_test_raw.astype(float),
            "y_prob_cb_tuned": y_prob_test_raw.astype(float),
            "y_prob_final": y_prob_calibrated.astype(float),
            "pd_calibrated": y_prob_calibrated.astype(float),
            "pd_logreg": lr_model.predict_proba(X_test_num)[:, 1].astype(float),
        }
    )
    preds_df.to_parquet(test_predictions_path, index=False)
    logger.info(f"Saved test predictions to {test_predictions_path}")

    training_record = {
        "best_calibration": "Platt Sigmoid",
        "optuna_best_auc": float(cb_metrics.get("auc_roc", 0.0)),
        "optuna_best_params": config["model"].get("params", {}),
        "baseline_metrics": lr_metrics,
        "catboost_metrics": cb_metrics,
        "calibration_metrics": cal_metrics,
        "conformal_metrics": cp_metrics,
        "final_test_metrics": final_test_metrics,
    }
    record_path = Path("models/pd_training_record.pkl")
    with open(record_path, "wb") as f:
        pickle.dump(training_record, f)
    logger.info(f"Saved training record to {record_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pd_model.yaml")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.sample_size)
