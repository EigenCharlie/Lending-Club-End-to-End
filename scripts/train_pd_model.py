"""Train PD models (LR baseline + CatBoost default/tuned) with robust calibration.

Usage:
    uv run python scripts/train_pd_model.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluation.metrics import classification_metrics
from src.models.calibration import evaluate_calibration, expected_calibration_error
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
    TARGET,
    resolve_feature_sets,
    temporal_train_val_split,
    train_baseline,
    train_catboost_default,
    train_catboost_tuned_optuna,
)
from src.utils.io_utils import read_split_with_fe_fallback


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known percent-like string columns when present."""
    out = df.copy()
    for col in ("int_rate", "revol_util"):
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.rstrip("%")
                .pipe(pd.to_numeric, errors="coerce")
            )
    if "term" in out.columns and not pd.api.types.is_numeric_dtype(out["term"]):
        out["term"] = (
            out["term"].astype(str).str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        )
    return out


def _prepare_catboost_frame(
    df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Build CatBoost matrix with deterministic order and dtypes."""
    out = df.copy()
    categorical_set = set(categorical)

    for col in features:
        if col not in out.columns:
            out[col] = "UNKNOWN" if col in categorical_set else np.nan

    out = out[features].copy()
    for col in features:
        if col in categorical_set:
            out[col] = out[col].astype("string").fillna("UNKNOWN").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_logreg_frame(
    df: pd.DataFrame,
    features: list[str],
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build numeric matrix for LR baseline, imputing with train medians."""
    out = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan

    if fill_values is None:
        fill_values = out.median(numeric_only=True).fillna(0.0)
    out = out.fillna(fill_values).fillna(0.0)
    return out, fill_values


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Return AUC when both classes are present, else None."""
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _fit_calibrator_from_scores(
    method: str,
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
) -> Any:
    """Fit score-based calibrator from raw probabilities."""
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(y_prob_raw.reshape(-1, 1), y_true)
        return model
    if method == "isotonic":
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(y_prob_raw, y_true)
        return model
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_calibrator(calibrator: Any, y_prob_raw: np.ndarray) -> np.ndarray:
    """Apply score-based calibrator."""
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(y_prob_raw)
    raise TypeError(f"Unsupported calibrator type: {type(calibrator)}")


def _build_calibration_backtest_splits(
    cal_df: pd.DataFrame,
    n_folds: int = 4,
    date_col: str = "issue_d",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create anchored temporal folds on calibration set for calibrator selection."""
    if len(cal_df) < 200:
        return []

    if date_col in cal_df.columns:
        ordered = cal_df.sort_values(date_col).reset_index(drop=True)
    else:
        ordered = cal_df.reset_index(drop=True)

    n = len(ordered)
    fold_size = max(1, n // (n_folds + 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(1, n_folds + 1):
        fit_end = fold_size * i
        eval_start = fit_end
        eval_end = min(n, eval_start + fold_size)
        if fit_end < 500 or (eval_end - eval_start) < 100:
            continue

        idx_fit = np.arange(0, fit_end, dtype=int)
        idx_eval = np.arange(eval_start, eval_end, dtype=int)
        splits.append((idx_fit, idx_eval))

    return splits


def _evaluate_calibration_method(
    method: str,
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Backtest calibrator over temporal folds using multi-metric summary."""
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (idx_fit, idx_eval) in enumerate(splits, start=1):
        y_fit = y_true[idx_fit]
        y_eval = y_true[idx_eval]
        p_fit = y_prob_raw[idx_fit]
        p_eval = y_prob_raw[idx_eval]

        if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
            continue

        calibrator = _fit_calibrator_from_scores(method, y_fit, p_fit)
        p_eval_cal = _apply_calibrator(calibrator, p_eval)

        raw_auc = _safe_auc(y_eval, p_eval)
        cal_auc = _safe_auc(y_eval, p_eval_cal)
        auc_drop = 0.0
        if raw_auc is not None and cal_auc is not None:
            auc_drop = float(raw_auc - cal_auc)

        fold_rows.append(
            {
                "fold": fold_id,
                "n_fit": int(len(idx_fit)),
                "n_eval": int(len(idx_eval)),
                "raw_auc": None if raw_auc is None else float(raw_auc),
                "cal_auc": None if cal_auc is None else float(cal_auc),
                "auc_drop": float(auc_drop),
                "brier": float(brier_score_loss(y_eval, p_eval_cal)),
                "ece": float(expected_calibration_error(y_eval, p_eval_cal)),
            }
        )

    if not fold_rows:
        return {
            "method": method,
            "folds_used": 0,
            "mean_brier": float("inf"),
            "mean_ece": float("inf"),
            "mean_auc_drop": float("inf"),
            "brier_variance": float("inf"),
            "ece_variance": float("inf"),
            "stability": float("inf"),
            "folds": [],
        }

    briers = np.array([r["brier"] for r in fold_rows], dtype=float)
    eces = np.array([r["ece"] for r in fold_rows], dtype=float)
    auc_drops = np.array([r["auc_drop"] for r in fold_rows], dtype=float)

    return {
        "method": method,
        "folds_used": int(len(fold_rows)),
        "mean_brier": float(np.mean(briers)),
        "mean_ece": float(np.mean(eces)),
        "mean_auc_drop": float(np.mean(auc_drops)),
        "brier_variance": float(np.var(briers)),
        "ece_variance": float(np.var(eces)),
        "stability": float(np.var(briers) + np.var(eces)),
        "folds": fold_rows,
    }


def _select_calibration_method(
    reports: list[dict[str, Any]],
    auc_drop_limit: float = 0.0015,
) -> tuple[str, dict[str, Any]]:
    """Select calibrator using ordered priorities with AUC-drop constraint."""
    candidates = [r for r in reports if np.isfinite(r.get("mean_brier", np.inf))]
    if not candidates:
        fallback = {
            "selected_method": "platt",
            "selection_reason": "fallback_no_valid_folds",
            "auc_drop_limit": float(auc_drop_limit),
            "candidates": reports,
        }
        return "platt", fallback

    feasible = [r for r in candidates if r["mean_auc_drop"] <= auc_drop_limit]
    selection_reason = "feasible_multi_metric"
    target = feasible

    if not target:
        selection_reason = "constraint_relaxed_auc_drop"
        target = candidates

    selected = sorted(
        target,
        key=lambda r: (
            float(r.get("mean_brier", np.inf)),
            float(r.get("mean_ece", np.inf)),
            float(r.get("stability", np.inf)),
        ),
    )[0]

    report = {
        "selected_method": selected["method"],
        "selection_reason": selection_reason,
        "auc_drop_limit": float(auc_drop_limit),
        "candidates": candidates,
        "feasible_candidates": feasible,
    }
    return str(selected["method"]), report


def _human_calibration_name(method: str) -> str:
    if method == "platt":
        return "Platt Sigmoid"
    if method == "isotonic":
        return "Isotonic Regression"
    return method


def main(config_path: str = "configs/pd_model.yaml", sample_size: int | None = None) -> None:
    config = load_config(config_path)
    logger.info(f"Config loaded from {config_path}")

    train = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["train_path"]))
    test = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["test_path"]))
    cal = _normalize_percent_columns(
        read_split_with_fe_fallback(config["data"]["calibration_path"])
    )

    if sample_size is not None:
        if sample_size < len(train):
            train = train.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(test):
            test = test.sample(n=sample_size, random_state=42).reset_index(drop=True)
        if sample_size < len(cal):
            cal = cal.sample(n=sample_size, random_state=42).reset_index(drop=True)

    feature_src_cfg = config.get("feature_source", {})
    feature_mode = feature_src_cfg.get("mode", "auto")
    feature_config_path = feature_src_cfg.get(
        "feature_config_path", "data/processed/feature_config.pkl"
    )

    feature_sets = resolve_feature_sets(
        train,
        feature_source=feature_mode,
        feature_config_path=feature_config_path,
    )

    catboost_features = feature_sets["catboost_features"]
    logreg_features = feature_sets["logreg_features"]
    categorical_features = feature_sets["categorical_features"]

    # enforce availability in all splits
    catboost_features = [
        c
        for c in catboost_features
        if c in train.columns and c in cal.columns and c in test.columns
    ]
    logreg_features = [
        c for c in logreg_features if c in train.columns and c in cal.columns and c in test.columns
    ]
    categorical_features = [c for c in categorical_features if c in catboost_features]

    if not catboost_features:
        raise ValueError("No CatBoost features resolved across train/cal/test splits.")
    if not logreg_features:
        raise ValueError("No Logistic Regression features resolved across train/cal/test splits.")

    logger.info(
        "Feature source={} | catboost_features={} | logreg_features={} | categorical={}".format(
            feature_sets.get("feature_source", feature_mode),
            len(catboost_features),
            len(logreg_features),
            len(categorical_features),
        )
    )

    val_cfg = config.get("validation", {})
    val_fraction = float(val_cfg.get("val_from_tail_fraction_of_train", 0.15))
    train_fit, train_val = temporal_train_val_split(
        train, val_fraction=val_fraction, date_col="issue_d"
    )

    y_train_fit = train_fit[TARGET].astype(int)
    y_val = train_val[TARGET].astype(int)
    y_cal = cal[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    X_train_fit_cb = _prepare_catboost_frame(train_fit, catboost_features, categorical_features)
    X_val_cb = _prepare_catboost_frame(train_val, catboost_features, categorical_features)
    X_cal_cb = _prepare_catboost_frame(cal, catboost_features, categorical_features)
    X_test_cb = _prepare_catboost_frame(test, catboost_features, categorical_features)

    X_train_fit_lr, lr_fill = _prepare_logreg_frame(train_fit, logreg_features)
    X_test_lr, _ = _prepare_logreg_frame(test, logreg_features, fill_values=lr_fill)

    # Baseline LR
    lr_model, lr_metrics = train_baseline(X_train_fit_lr, y_train_fit, X_test_lr, y_test)

    # CatBoost default
    cb_default_model, cb_default_metrics = train_catboost_default(
        X_train_fit_cb,
        y_train_fit,
        X_val_cb,
        y_val,
        X_test=X_test_cb,
        y_test=y_test,
        cat_features=categorical_features,
        params=config["model"].get("params", {}),
    )

    # CatBoost tuned (Optuna)
    hpo_cfg = config.get("hpo", {})
    if bool(hpo_cfg.get("enabled", True)):
        cb_tuned_model, cb_tuned_metrics = train_catboost_tuned_optuna(
            X_train_fit_cb,
            y_train_fit,
            X_val_cb,
            y_val,
            X_test=X_test_cb,
            y_test=y_test,
            cat_features=categorical_features,
            base_params=config["model"].get("params", {}),
            n_trials=int(hpo_cfg.get("n_trials", 100)),
            sampler=str(hpo_cfg.get("sampler", "tpe")),
            pruner=str(hpo_cfg.get("pruner", "median")),
            timeout_minutes=int(hpo_cfg.get("timeout_minutes", 0)),
            n_startup_trials=int(hpo_cfg.get("n_startup_trials", 40)),
            multivariate_tpe=bool(hpo_cfg.get("multivariate_tpe", True)),
            pruner_n_startup_trials=int(hpo_cfg.get("pruner_n_startup_trials", 20)),
            pruner_n_warmup_steps=int(hpo_cfg.get("pruner_n_warmup_steps", 50)),
            use_pruning_callback=bool(hpo_cfg.get("use_pruning_callback", True)),
            study_storage=hpo_cfg.get("study_storage", None),
            study_name=hpo_cfg.get("study_name", None),
            load_if_exists=bool(hpo_cfg.get("load_if_exists", True)),
            refit_full_train=bool(hpo_cfg.get("refit_full_train", True)),
        )
    else:
        cb_tuned_model, cb_tuned_metrics = train_catboost_default(
            X_train_fit_cb,
            y_train_fit,
            X_val_cb,
            y_val,
            X_test=X_test_cb,
            y_test=y_test,
            cat_features=categorical_features,
            params=config["model"].get("params", {}),
        )
        cb_tuned_metrics["hpo_trials_executed"] = 0
        cb_tuned_metrics["hpo_best_validation_auc"] = float(cb_tuned_metrics["validation_auc"])
        cb_tuned_metrics["best_params"] = config["model"].get("params", {})

    # Raw probabilities
    y_prob_default_test = cb_default_model.predict_proba(X_test_cb)[:, 1]
    y_prob_tuned_test = cb_tuned_model.predict_proba(X_test_cb)[:, 1]
    y_prob_tuned_cal = cb_tuned_model.predict_proba(X_cal_cb)[:, 1]

    # Robust calibration policy selection via temporal folds on calibration set
    cal_splits = _build_calibration_backtest_splits(cal, n_folds=4, date_col="issue_d")
    cal_reports = [
        _evaluate_calibration_method("platt", y_cal.to_numpy(), y_prob_tuned_cal, cal_splits),
        _evaluate_calibration_method("isotonic", y_cal.to_numpy(), y_prob_tuned_cal, cal_splits),
    ]
    selected_cal_method, cal_selection_report = _select_calibration_method(
        cal_reports,
        auc_drop_limit=0.0015,
    )

    calibrator = _fit_calibrator_from_scores(
        selected_cal_method,
        y_cal.to_numpy(),
        y_prob_tuned_cal,
    )
    y_prob_final = _apply_calibrator(calibrator, y_prob_tuned_test)
    cal_metrics = evaluate_calibration(
        y_test.to_numpy(),
        y_prob_final,
        name=selected_cal_method,
    )

    # Conformal (keeps calibration split isolated from model training)
    alpha = 1.0 - float(config["conformal"].get("confidence_level", 0.9))
    _, y_intervals = create_pd_intervals(cb_tuned_model, X_cal_cb, y_cal, X_test_cb, alpha=alpha)
    cp_metrics = validate_coverage(y_test.values.astype(float), y_intervals, alpha)

    final_test_metrics = classification_metrics(y_test.values, y_prob_final)
    tuned_raw_test_metrics = classification_metrics(y_test.values, y_prob_tuned_test)

    # Persist models/calibrator
    model_path = Path(config["output"].get("model_path", "models/pd_catboost_tuned.cbm"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cb_tuned_model.save_model(str(model_path))

    default_model_path = Path(
        config["output"].get("default_model_path", "models/pd_catboost_default.cbm")
    )
    default_model_path.parent.mkdir(parents=True, exist_ok=True)
    cb_default_model.save_model(str(default_model_path))

    tuned_model_path = Path(
        config["output"].get("tuned_model_path", "models/pd_catboost_tuned.cbm")
    )
    tuned_model_path.parent.mkdir(parents=True, exist_ok=True)
    if tuned_model_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, tuned_model_path)

    # Keep legacy path updated for compatibility.
    legacy_model_path = Path("models/pd_catboost.cbm")
    if legacy_model_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, legacy_model_path)

    cal_path = Path(config["output"].get("conformal_path", "models/pd_calibrator.pkl"))
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    # Canonical artifacts for downstream loading.
    CANONICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL_CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() != CANONICAL_MODEL_PATH.resolve():
        shutil.copy2(model_path, CANONICAL_MODEL_PATH)
    if cal_path.resolve() != CANONICAL_CALIBRATOR_PATH.resolve():
        shutil.copy2(cal_path, CANONICAL_CALIBRATOR_PATH)

    # Persist model contract for strict feature alignment across scripts.
    features_contract, categorical_contract = infer_model_feature_contract(cb_tuned_model)
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

    # Persist test predictions for downstream contracts.
    test_predictions_path = Path("data/processed/test_predictions.parquet")
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    y_prob_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    preds_df = pd.DataFrame(
        {
            "loan_id": test["id"].astype(str) if "id" in test.columns else test.index.astype(str),
            "y_true": y_test.values.astype(float),
            "y_prob_lr": y_prob_lr.astype(float),
            "y_prob_cb_default": y_prob_default_test.astype(float),
            "y_prob_cb_tuned": y_prob_tuned_test.astype(float),
            "y_prob_final": y_prob_final.astype(float),
            "pd_calibrated": y_prob_final.astype(float),
            "pd_logreg": y_prob_lr.astype(float),
        }
    )
    preds_df.to_parquet(test_predictions_path, index=False)

    training_record = {
        "best_model": "CatBoost (tuned + calibrated)",
        "best_calibration": _human_calibration_name(selected_cal_method),
        "calibration_selection_report": cal_selection_report,
        "feature_source": feature_sets.get("feature_source", feature_mode),
        "feature_config_path": str(feature_config_path),
        "validation_scheme": val_cfg.get("scheme", "temporal_train_val_cal_test"),
        "feature_count_default": int(len(catboost_features)),
        "feature_count_tuned": int(len(catboost_features)),
        "optuna_best_auc": float(cb_tuned_metrics.get("auc_roc", 0.0)),
        "optuna_best_params": cb_tuned_metrics.get("best_params", {}),
        "hpo_trials_executed": int(cb_tuned_metrics.get("hpo_trials_executed", 0)),
        "hpo_best_validation_auc": float(cb_tuned_metrics.get("hpo_best_validation_auc", 0.0)),
        "baseline_metrics": lr_metrics,
        "catboost_default_metrics": cb_default_metrics,
        "catboost_tuned_metrics": cb_tuned_metrics,
        "catboost_tuned_raw_test_metrics": tuned_raw_test_metrics,
        "calibration_metrics": cal_metrics,
        "conformal_metrics": cp_metrics,
        "final_test_metrics": final_test_metrics,
    }

    record_path = Path("models/pd_training_record.pkl")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "wb") as f:
        pickle.dump(training_record, f)

    logger.info("Saved default model to {}", default_model_path)
    logger.info("Saved tuned model to {}", model_path)
    logger.info("Saved calibrator to {}", cal_path)
    logger.info("Saved canonical model to {}", CANONICAL_MODEL_PATH)
    logger.info("Saved canonical calibrator to {}", CANONICAL_CALIBRATOR_PATH)
    logger.info("Saved PD contract to {}", CONTRACT_PATH)
    logger.info("Saved test predictions to {}", test_predictions_path)
    logger.info("Saved training record to {}", record_path)
    logger.info(
        "Final metrics | AUC={:.4f} Gini={:.4f} KS={:.4f} Brier={:.4f} ECE={:.4f}",
        final_test_metrics["auc_roc"],
        final_test_metrics["gini"],
        final_test_metrics["ks_statistic"],
        final_test_metrics["brier_score"],
        final_test_metrics["ece"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pd_model.yaml")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.sample_size)
