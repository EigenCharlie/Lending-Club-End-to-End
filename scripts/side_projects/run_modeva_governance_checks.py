"""Run Modeva governance diagnostics on canonical PD model (side-task candidate for core).

This script keeps the canonical PD model as source of truth and adds a governance
layer with drift, fairness, robustness, reliability and slicing checks.

Usage:
    python scripts/side_projects/run_modeva_governance_checks.py --config configs/modeva_governance.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from loguru import logger

from src.models.pd_contract import CONTRACT_PATH, resolve_calibrator_path, resolve_model_path


def _check(
    metric_name: str, value: float, threshold: float, comparator: str, scope: str
) -> dict[str, object]:
    if comparator == ">=":
        passed = value >= threshold
    elif comparator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")
    return {
        "scope": scope,
        "metric": metric_name,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "passed": bool(passed),
    }


def _normalize_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _ensure_required_columns(df: pd.DataFrame, required: list[str], split_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {split_name}: {missing}")


def _load_canonical_feature_contract() -> tuple[list[str], list[str]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    feature_names = contract["feature_names"]
    categorical = contract.get("categorical_features", [])
    return feature_names, categorical


def _prepare_model_data(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_names: list[str],
    categorical_features: list[str],
    target_col: str,
    protected_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.Series, dict[str, str]]:
    required = feature_names + [target_col, protected_feature]
    _ensure_required_columns(train_raw, required, "train")
    _ensure_required_columns(test_raw, required, "test")

    train = _normalize_percent_columns(train_raw[required].copy())
    test = _normalize_percent_columns(test_raw[required].copy())

    cat_set = set(categorical_features)
    numeric_features = [c for c in feature_names if c not in cat_set]
    num_fill = train[numeric_features].median(numeric_only=True)

    train[numeric_features] = (
        train[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(num_fill)
    )
    test[numeric_features] = (
        test[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(num_fill)
    )

    cat_fill: dict[str, str] = {}
    for col in categorical_features:
        mode = train[col].mode(dropna=True)
        fill = str(mode.iloc[0]) if len(mode) > 0 else "missing"
        cat_fill[col] = fill
        train[col] = train[col].fillna(fill).astype(str)
        test[col] = test[col].fillna(fill).astype(str)

    protected = pd.concat(
        [train[[protected_feature]], test[[protected_feature]]],
        axis=0,
        ignore_index=True,
    )

    model_train = train[feature_names + [target_col]].copy()
    model_test = test[feature_names + [target_col]].copy()
    return model_train, model_test, protected, numeric_features, num_fill, cat_fill


def _prepare_numeric_drift_data(
    train_model: pd.DataFrame,
    test_model: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_model, test_model], axis=0, ignore_index=True)
    feature_frame = combined[feature_names].copy()
    non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(feature_frame[c])]
    combined_num = pd.get_dummies(combined, columns=non_numeric, drop_first=True)
    combined_num = combined_num.fillna(combined_num.median(numeric_only=True))

    train_num = combined_num.iloc[: len(train_model)][
        [c for c in combined_num.columns if c != target_col] + [target_col]
    ]
    test_num = combined_num.iloc[len(train_model) :][
        [c for c in combined_num.columns if c != target_col] + [target_col]
    ]
    return train_num.reset_index(drop=True), test_num.reset_index(drop=True)


def _load_calibrator(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _apply_calibration(proba: np.ndarray, calibrator: Any | None) -> np.ndarray:
    if calibrator is None:
        return proba
    if hasattr(calibrator, "transform"):
        return np.asarray(calibrator.transform(proba), dtype=float)
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(proba.reshape(-1, 1))[:, 1], dtype=float)
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(proba.reshape(-1, 1)), dtype=float)
    return proba


def _parse_noise_columns(table: pd.DataFrame) -> list[float]:
    vals: list[float] = []
    for col in table.columns:
        try:
            vals.append(float(col))
        except Exception:
            continue
    return sorted(set(vals))


def _resolve_noise_column(table: pd.DataFrame, noise_level: float) -> Any:
    for col in table.columns:
        try:
            if abs(float(col) - noise_level) < 1e-12:
                return col
        except Exception:
            continue
    raise KeyError(
        f"Noise column {noise_level} not found in robustness table columns={list(table.columns)}"
    )


def main(config_path: str = "configs/modeva_governance.yaml") -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        logger.warning("torch import failed before modeva import; proceeding with modeva import.")

    try:
        import modeva
        import modeva.auth as modeva_auth
        from modeva import models
    except Exception as e:
        raise RuntimeError(
            "Modeva import failed. Install/repair Modeva in the runtime environment "
            "before running this governance side-task."
        ) from e

    modeva_auth.Authenticator.run = lambda self, auth_code=None: None

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    diag_cfg = cfg["diagnostics"]
    policy_cfg = cfg["policy"]
    output_cfg = cfg["output"]
    modeva_cfg = cfg["modeva"]

    train = pd.read_parquet(data_cfg["train_path"])
    test = pd.read_parquet(data_cfg["test_path"])

    rs = int(data_cfg.get("random_state", 42))
    n_train = int(data_cfg.get("train_sample_size", len(train)))
    n_test = int(data_cfg.get("test_sample_size", len(test)))
    if n_train < len(train):
        train = train.sample(n=n_train, random_state=rs).reset_index(drop=True)
    if n_test < len(test):
        test = test.sample(n=n_test, random_state=rs).reset_index(drop=True)

    target_col = data_cfg["target"]
    protected_feature = data_cfg["protected_feature"]
    feature_names, categorical_features = _load_canonical_feature_contract()
    logger.info(
        f"Modeva governance on canonical contract features: n_features={len(feature_names)}, "
        f"categorical={len(categorical_features)}"
    )

    model_train, model_test, protected, numeric_features, num_fill, cat_fill = _prepare_model_data(
        train_raw=train,
        test_raw=test,
        feature_names=feature_names,
        categorical_features=categorical_features,
        target_col=target_col,
        protected_feature=protected_feature,
    )
    drift_train, drift_test = _prepare_numeric_drift_data(
        train_model=model_train,
        test_model=model_test,
        feature_names=feature_names,
        target_col=target_col,
    )

    model_path = resolve_model_path()
    calibrator_path = resolve_calibrator_path()
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(model_path))
    calibrator = _load_calibrator(calibrator_path)
    logger.info(f"Loaded canonical model: {model_path}")
    logger.info(f"Loaded canonical calibrator: {calibrator_path if calibrator_path else 'None'}")

    def _predict_proba(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=feature_names)
        x_df = _normalize_percent_columns(x_df)

        for col in numeric_features:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
        x_df[numeric_features] = x_df[numeric_features].fillna(num_fill)

        for col in categorical_features:
            x_df[col] = x_df[col].fillna(cat_fill[col]).astype(str)

        p_raw = cb_model.predict_proba(x_df)[:, 1]
        p_cal = _apply_calibration(p_raw, calibrator)
        p_cal = np.clip(np.asarray(p_cal, dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p_cal, p_cal])

    ds_model = modeva.DataSet(name="modeva_corepd_dataset")
    ds_model.load_dataframe_train_test(model_train, model_test)
    ds_model.set_target(target_col)
    ds_model.set_protected_data(protected)

    ds_drift = modeva.DataSet(name="modeva_drift_dataset")
    ds_drift.load_dataframe_train_test(drift_train, drift_test)
    ds_drift.set_target(target_col)

    core_model = models.MoClassifier(
        predict_proba_function=_predict_proba,
        name=modeva_cfg.get("model_name", "CorePDCanonical"),
    )
    testsuite = modeva.TestSuite(
        dataset=ds_model,
        model=core_model,
        models=[core_model],
        name=modeva_cfg.get("testsuite_name", "modeva_corepd_governance"),
    )

    # Core diagnostics
    acc = testsuite.diagnose_accuracy_table()
    rel = testsuite.diagnose_reliability(
        train_dataset="train",
        test_dataset="test",
        alpha=float(diag_cfg.get("alpha", 0.1)),
    )
    robust = testsuite.diagnose_robustness(
        dataset="test",
        perturb_features=tuple(diag_cfg["perturb_features"]),
        noise_levels=tuple(diag_cfg["noise_levels"]),
        n_repeats=int(diag_cfg.get("robustness_repeats", 5)),
    )
    slicing_acc = testsuite.diagnose_slicing_accuracy(
        features=tuple(diag_cfg["slicing_features"]),
        dataset="test",
        bins=int(diag_cfg.get("slicing_bins", 5)),
    )
    slicing_rob = testsuite.diagnose_slicing_robustness(
        features=tuple(diag_cfg["slicing_features"]),
        dataset="test",
        bins=int(diag_cfg.get("slicing_bins", 5)),
        perturb_features=tuple(diag_cfg["perturb_features"]),
        noise_levels=float(diag_cfg["noise_levels"][-1]),
        n_repeats=int(diag_cfg.get("robustness_repeats", 5)),
    )

    fairness_group = diag_cfg["fairness_group"]
    group_name = fairness_group["name"]
    fairness_cfg = {
        group_name: {
            "feature": fairness_group["feature"],
            "protected": fairness_group["protected"],
            "reference": fairness_group["reference"],
        }
    }
    fair = testsuite.diagnose_fairness(
        group_config=fairness_cfg,
        dataset="test",
        metric=diag_cfg.get("fairness_metric", "AIR"),
    )

    # Data diagnostics (numeric-only drift for PSI stability)
    drift_psi = ds_drift.data_drift_test(
        dataset1="train",
        dataset2="test",
        distance_metric="PSI",
        psi_bins=int(diag_cfg.get("psi_bins", 10)),
    )
    drift_ks = ds_drift.data_drift_test(
        dataset1="train",
        dataset2="test",
        distance_metric="KS",
    )
    outlier = ds_drift.detect_outlier_isolation_forest(
        dataset="train",
        threshold=float(diag_cfg.get("outlier_threshold", 0.995)),
        n_estimators=int(diag_cfg.get("outlier_estimators", 200)),
    )

    # Extract metrics
    acc_table = acc.table
    rel_table = rel.table
    robust_table = robust.table
    fair_table = fair.table
    slicing_acc_table = slicing_acc.table
    slicing_rob_table = slicing_rob.table
    drift_psi_table = drift_psi.table
    drift_ks_table = drift_ks.table

    noise_values = _parse_noise_columns(robust_table)
    base_noise = 0.0 if 0.0 in noise_values else min(noise_values)
    high_noise = max(noise_values)
    base_col = _resolve_noise_column(robust_table, base_noise)
    high_col = _resolve_noise_column(robust_table, high_noise)
    robust_base = float(robust_table[base_col].mean())
    robust_high = float(robust_table[high_col].mean())
    robust_drop = robust_base - robust_high

    outlier_table = outlier.table
    n_out = len(outlier_table["outliers"])
    n_non = len(outlier_table["non-outliers"])
    outlier_rate = float(n_out / (n_out + n_non)) if (n_out + n_non) > 0 else 0.0

    metrics = {
        "test_auc": float(acc_table.loc["test", "AUC"]),
        "train_auc": float(acc_table.loc["train", "AUC"]),
        "auc_gap_test_minus_train": float(acc_table.loc["GAP", "AUC"]),
        "test_brier": float(acc_table.loc["test", "Brier"]),
        "avg_width": float(rel_table.loc[0, "Avg.Width"]),
        "avg_coverage": float(rel_table.loc[0, "Avg.Coverage"]),
        "fairness_air": float(fair_table.loc[diag_cfg.get("fairness_metric", "AIR"), group_name]),
        "max_drift_psi": float(drift_psi_table["Distance_Scores"].max()),
        "max_drift_ks": float(drift_ks_table["Distance_Scores"].max()),
        "robustness_base_metric": robust_base,
        "robustness_high_noise_metric": robust_high,
        "robustness_drop": float(robust_drop),
        "weak_slice_accuracy_ratio": float(slicing_acc_table["Weak"].mean()),
        "weak_slice_robustness_ratio": float(slicing_rob_table["Weak"].mean()),
        "outlier_rate": outlier_rate,
        "noise_base": float(base_noise),
        "noise_high": float(high_noise),
        "sample_train": int(len(model_train)),
        "sample_test": int(len(model_test)),
    }
    metrics_df = pd.DataFrame([metrics])

    checks = [
        _check("test_auc", metrics["test_auc"], float(policy_cfg["min_test_auc"]), ">=", "model"),
        _check(
            "auc_gap",
            abs(metrics["auc_gap_test_minus_train"]),
            float(policy_cfg["max_auc_gap"]),
            "<=",
            "model",
        ),
        _check(
            "avg_coverage",
            metrics["avg_coverage"],
            float(policy_cfg["min_avg_coverage"]),
            ">=",
            "reliability",
        ),
        _check(
            "avg_width",
            metrics["avg_width"],
            float(policy_cfg["max_avg_width"]),
            "<=",
            "reliability",
        ),
        _check(
            "fairness_air_min",
            metrics["fairness_air"],
            float(policy_cfg["min_fairness_air"]),
            ">=",
            "fairness",
        ),
        _check(
            "fairness_air_max",
            metrics["fairness_air"],
            float(policy_cfg["max_fairness_air"]),
            "<=",
            "fairness",
        ),
        _check(
            "max_drift_psi",
            metrics["max_drift_psi"],
            float(policy_cfg["max_drift_psi"]),
            "<=",
            "data_drift",
        ),
        _check(
            "max_drift_ks",
            metrics["max_drift_ks"],
            float(policy_cfg["max_drift_ks"]),
            "<=",
            "data_drift",
        ),
        _check(
            "robustness_drop",
            metrics["robustness_drop"],
            float(policy_cfg["max_robustness_auc_drop"]),
            "<=",
            "robustness",
        ),
        _check(
            "weak_slice_accuracy_ratio",
            metrics["weak_slice_accuracy_ratio"],
            float(policy_cfg["max_weak_slice_accuracy_ratio"]),
            "<=",
            "slicing",
        ),
        _check(
            "weak_slice_robustness_ratio",
            metrics["weak_slice_robustness_ratio"],
            float(policy_cfg["max_weak_slice_robustness_ratio"]),
            "<=",
            "slicing",
        ),
        _check(
            "outlier_rate",
            metrics["outlier_rate"],
            float(policy_cfg["max_outlier_rate"]),
            "<=",
            "data_quality",
        ),
    ]
    checks_df = pd.DataFrame(checks)
    overall_pass = bool(checks_df["passed"].all())

    status = {
        "overall_pass": overall_pass,
        "checks_passed": int(checks_df["passed"].sum()),
        "checks_total": int(len(checks_df)),
        "metrics": metrics,
        "config_path": config_path,
        "model_path": str(model_path),
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
    }

    # Persist artifacts
    status_path = Path(output_cfg["status_json"])
    checks_path = Path(output_cfg["checks_parquet"])
    metrics_path = Path(output_cfg["metrics_parquet"])
    drift_psi_path = Path(output_cfg["drift_psi_parquet"])
    drift_ks_path = Path(output_cfg["drift_ks_parquet"])
    slicing_acc_path = Path(output_cfg["slicing_accuracy_parquet"])
    slicing_rob_path = Path(output_cfg["slicing_robustness_parquet"])
    fairness_path = Path(output_cfg["fairness_parquet"])
    robust_path = Path(output_cfg["robustness_parquet"])
    report_path = Path(output_cfg["report_html"])

    for p in [
        status_path,
        checks_path,
        metrics_path,
        drift_psi_path,
        drift_ks_path,
        slicing_acc_path,
        slicing_rob_path,
        fairness_path,
        robust_path,
        report_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    checks_df.to_parquet(checks_path, index=False)
    metrics_df.to_parquet(metrics_path, index=False)
    drift_psi_table.to_parquet(drift_psi_path)
    drift_ks_table.to_parquet(drift_ks_path)
    slicing_acc_table.to_parquet(slicing_acc_path, index=False)
    slicing_rob_table.to_parquet(slicing_rob_path, index=False)
    fair_table.to_parquet(fairness_path)
    robust_table.to_parquet(robust_path, index=False)

    testsuite.export_report(path=str(report_path))

    logger.info(f"Modeva governance checks saved: {checks_path}")
    logger.info(f"Modeva governance metrics saved: {metrics_path}")
    logger.info(f"Modeva governance status saved: {status_path}")
    logger.info(
        "Modeva governance overall_pass="
        f"{overall_pass} ({status['checks_passed']}/{status['checks_total']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/modeva_governance.yaml")
    args = parser.parse_args()
    main(args.config)
