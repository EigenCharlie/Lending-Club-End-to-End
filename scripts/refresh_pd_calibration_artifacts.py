"""Refresh PD calibration artifacts from the canonical saved model.

Usage:
    uv run python scripts/refresh_pd_calibration_artifacts.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from loguru import logger

import scripts.train_pd_model as tpm
from src.evaluation.metrics import classification_metrics
from src.models.calibration import evaluate_calibration
from src.models.pd_contract import (
    CANONICAL_CALIBRATOR_PATH,
    CANONICAL_MODEL_PATH,
    CONTRACT_PATH,
    load_contract,
)
from src.models.pd_model import TARGET, temporal_train_val_split
from src.utils.baseline_registry import resolve_official_baseline_run_tag
from src.utils.io_utils import read_split_with_fe_fallback
from src.utils.threshold_semantics import write_threshold_semantics


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(model_path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def _resolve_features(contract: dict[str, Any]) -> tuple[list[str], list[str]]:
    features = list(contract.get("feature_names", []) or [])
    categorical = list(contract.get("categorical_features", []) or [])
    if not features:
        raise ValueError(f"No feature_names found in {CONTRACT_PATH}")
    categorical = [c for c in categorical if c in features]
    return features, categorical


def _load_splits(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = tpm._normalize_percent_columns(read_split_with_fe_fallback(cfg["data"]["train_path"]))
    cal = tpm._normalize_percent_columns(
        read_split_with_fe_fallback(cfg["data"]["calibration_path"])
    )
    test = tpm._normalize_percent_columns(read_split_with_fe_fallback(cfg["data"]["test_path"]))
    train, _ = tpm._apply_training_regime(
        train,
        cfg.get("training_regime", {}) or {},
        date_col="issue_d",
    )
    return train, cal, test


def _build_matrices(
    cfg: dict[str, Any],
    features: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, cal, test = _load_splits(cfg)
    val_fraction = float(cfg.get("validation", {}).get("val_from_tail_fraction_of_train", 0.15))
    train_fit, train_val = temporal_train_val_split(
        train, val_fraction=val_fraction, date_col="issue_d"
    )
    X_val = tpm._prepare_catboost_frame(train_val, features, categorical)
    X_cal = tpm._prepare_catboost_frame(cal, features, categorical)
    X_test = tpm._prepare_catboost_frame(test, features, categorical)
    return train_val, cal, test, X_val, X_cal, X_test


def _update_training_record(
    record_path: Path,
    *,
    selected_method: str,
    selection_report: dict[str, Any],
    calibration_metrics: dict[str, Any],
    final_test_metrics: dict[str, Any],
    decision_threshold_artifact: dict[str, Any],
) -> dict[str, Any]:
    with open(record_path, "rb") as f:
        record = pickle.load(f)

    record["best_calibration"] = tpm._human_calibration_name(selected_method)
    record["calibration_selection_report"] = selection_report
    record["calibration_metrics"] = calibration_metrics
    record["final_test_metrics"] = final_test_metrics
    record["decision_threshold"] = decision_threshold_artifact

    with open(record_path, "wb") as f:
        pickle.dump(record, f)
    return record


def _update_test_predictions(
    preds_path: Path,
    *,
    y_true: np.ndarray,
    y_prob_final: np.ndarray,
) -> None:
    preds = pd.read_parquet(preds_path)
    preds = preds.copy()
    preds["y_true"] = np.asarray(y_true, dtype=float)
    preds["y_prob_final"] = np.asarray(y_prob_final, dtype=float)
    preds["pd_calibrated"] = np.asarray(y_prob_final, dtype=float)
    preds.to_parquet(preds_path, index=False)


def _update_seed_replay_status(
    path: Path,
    *,
    selected_method: str,
    final_test_metrics: dict[str, Any],
) -> None:
    if not path.exists():
        return
    payload = _load_json(path)
    payload["generated_at_utc"] = datetime.now(UTC).isoformat()
    payload["selected_calibration_method"] = tpm._human_calibration_name(selected_method)
    payload["oot_auc"] = float(final_test_metrics.get("auc_roc", 0.0))
    payload["brier"] = float(final_test_metrics.get("brier_score", 0.0))
    payload["ece"] = float(final_test_metrics.get("ece", 0.0))
    _save_json(path, payload)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y_true), 1)
    ece = 0.0
    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * (float(np.sum(mask)) / total)
    return float(ece)


def _selector_reliability_bins(
    y_true: np.ndarray, selector_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    ranks = pd.Series(selector_prob).rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, len(ranks)), duplicates="drop")
    frame = pd.DataFrame(
        {"y_true": y_true, "selector_prob": selector_prob, "bin": bins.astype(str)}
    )
    out = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("y_true", "size"),
            empirical_rate=("y_true", "mean"),
            mean_selector_prob=("selector_prob", "mean"),
        )
        .reset_index()
        .sort_values("bin")
    )
    return out.to_dict(orient="records")


def _temporal_metrics(
    issue_dates: pd.Series, y_true: np.ndarray, y_prob: np.ndarray
) -> list[dict[str, Any]]:
    quarters = pd.to_datetime(issue_dates, errors="coerce").dt.to_period("Q").astype(str)
    frame = pd.DataFrame({"issue_quarter": quarters, "y_true": y_true, "y_prob": y_prob})
    frame = frame.loc[frame["issue_quarter"].ne("NaT")].copy()
    rows: list[dict[str, Any]] = []
    for quarter, group in frame.groupby("issue_quarter", observed=True):
        rows.append(
            {
                "issue_quarter": str(quarter),
                "n_obs": int(len(group)),
                "brier": float(np.mean((group["y_true"] - group["y_prob"]) ** 2)),
                "ece": float(
                    _ece(
                        group["y_true"].to_numpy(dtype=float),
                        group["y_prob"].to_numpy(dtype=float),
                    )
                ),
            }
        )
    return rows


def _build_calibration_diagnostics(
    *,
    y_cal: np.ndarray,
    y_prob_tuned_cal: np.ndarray,
    y_test: np.ndarray,
    y_prob_tuned_test: np.ndarray,
    test_issue_dates: pd.Series,
    selected_method: str,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    venn_payload: dict[str, Any] = {}
    for method in ("platt", "isotonic", "venn_abers"):
        calibrator = tpm._fit_calibrator_from_scores(method, y_cal, y_prob_tuned_cal)
        y_prob_eval = tpm._apply_calibrator(calibrator, y_prob_tuned_test)
        candidate_rows.append(
            {
                "method": method,
                "auc": float(classification_metrics(y_test, y_prob_eval).get("auc_roc", 0.0)),
                "brier": float(np.mean((y_test - y_prob_eval) ** 2)),
                "ece": float(_ece(y_test.astype(float), y_prob_eval.astype(float))),
            }
        )
        if method == "venn_abers" and hasattr(calibrator, "_predict_bounds"):
            p0, p1 = calibrator._predict_bounds(y_prob_tuned_test)
            selector_prob = np.where(y_test.astype(int) == 1, p1, p0)
            prevalence = float(np.mean(y_test))
            venn_payload = {
                "mean_p0": float(np.mean(p0)),
                "mean_p1": float(np.mean(p1)),
                "prevalence_observed": prevalence,
                "unbiasedness_in_the_large": bool(
                    float(np.mean(p0)) <= prevalence <= float(np.mean(p1))
                ),
                "avg_width": float(np.mean(p1 - p0)),
                "median_width": float(np.median(p1 - p0)),
                "selector_ece": float(_ece(y_test.astype(float), selector_prob.astype(float))),
                "selector_reliability_bins": _selector_reliability_bins(
                    y_test.astype(float), selector_prob.astype(float)
                ),
                "temporal_stability": _temporal_metrics(
                    test_issue_dates, y_test.astype(float), y_prob_eval.astype(float)
                ),
            }
    return {
        "schema_version": "2026-03-13.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selected_method": str(selected_method),
        "candidate_comparison": candidate_rows,
        "venn_abers": venn_payload,
    }


def main(config_path: str = "configs/pd_model.yaml") -> None:
    cfg = _load_config(config_path)
    contract = load_contract(CONTRACT_PATH)
    if contract is None:
        raise FileNotFoundError(f"Missing PD contract: {CONTRACT_PATH}")

    model_path = Path(contract.get("model_path", CANONICAL_MODEL_PATH.as_posix()))
    calibrator_path = Path(contract.get("calibrator_path", CANONICAL_CALIBRATOR_PATH.as_posix()))
    features, categorical = _resolve_features(contract)
    model = _load_model(model_path)

    train_val, cal, test, X_val, X_cal, X_test = _build_matrices(cfg, features, categorical)
    y_val = train_val[TARGET].astype(int)
    y_cal = cal[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    y_prob_tuned_val = model.predict_proba(X_val)[:, 1]
    y_prob_tuned_cal = model.predict_proba(X_cal)[:, 1]
    y_prob_tuned_test = model.predict_proba(X_test)[:, 1]

    cal_splits = tpm._build_calibration_backtest_splits(cal, n_folds=4, date_col="issue_d")
    candidates = cfg.get("calibration", {}).get("candidates", ["platt", "isotonic", "venn_abers"])
    cal_reports: list[dict[str, Any]] = []
    for method in candidates:
        cal_reports.append(
            tpm._evaluate_calibration_method(
                str(method),
                y_cal.to_numpy(),
                y_prob_tuned_cal,
                cal_splits,
            )
        )

    selected_method, selection_report = tpm._select_calibration_method(
        cal_reports,
        auc_drop_limit=0.0015,
    )
    calibrator = tpm._fit_calibrator_from_scores(
        selected_method,
        y_cal.to_numpy(),
        y_prob_tuned_cal,
    )
    y_prob_final = tpm._apply_calibrator(calibrator, y_prob_tuned_test)
    y_prob_final_val = tpm._apply_calibrator(calibrator, y_prob_tuned_val)
    final_test_metrics = classification_metrics(y_test.values, y_prob_final)
    calibration_metrics = evaluate_calibration(
        y_test.to_numpy(),
        y_prob_final,
        name=selected_method,
    )

    decision_cfg = cfg.get("decision_threshold", {}) or {}
    resolved_run_tag = str(resolve_official_baseline_run_tag(default="untracked") or "untracked")
    fairness_policy_path = str(
        decision_cfg.get("fairness_policy_path", "configs/fairness_policy.yaml")
    )
    with open(fairness_policy_path, encoding="utf-8") as f:
        fairness_cfg = yaml.safe_load(f) or {}
    fairness_policy = fairness_cfg.get("policy", {})
    fairness_attrs = fairness_cfg.get("attributes", [])
    groups_for_threshold = tpm._build_fairness_groups_for_threshold(train_val, fairness_attrs)
    groups_for_threshold_cal = tpm._build_fairness_groups_for_threshold(cal, fairness_attrs)
    thr_min = float(decision_cfg.get("min_threshold", 0.05))
    thr_max = float(decision_cfg.get("max_threshold", 0.95))
    thr_step = float(decision_cfg.get("step", 0.01))
    thresholds = np.arange(thr_min, thr_max + (thr_step / 2.0), thr_step)
    fallback_threshold = float(fairness_policy.get("prediction_threshold", 0.5))
    threshold_result = tpm._select_decision_threshold(
        y_true=y_val.to_numpy(),
        y_prob=y_prob_final_val,
        policy={
            "dpd_threshold": float(fairness_policy.get("dpd_threshold", 0.10)),
            "eo_gap_threshold": float(fairness_policy.get("eo_gap_threshold", 0.10)),
            "dir_threshold": float(fairness_policy.get("dir_threshold", 0.80)),
        },
        groups_dict=groups_for_threshold,
        thresholds=np.asarray(thresholds, dtype=float),
        fallback_threshold=fallback_threshold,
        y_true_secondary=y_cal.to_numpy(),
        y_prob_secondary=y_prob_tuned_cal,
        groups_dict_secondary=groups_for_threshold_cal,
    )
    decision_threshold_artifact = {
        "enabled": True,
        "selected_threshold": float(threshold_result["selected_threshold"]),
        "fallback_threshold": fallback_threshold,
        "selection_metrics": threshold_result["selection_metrics"],
        "search_summary": threshold_result["search_summary"],
        "source": "refresh_from_canonical_model",
        "fairness_policy_path": fairness_policy_path,
        "validation_rows": int(len(train_val)),
        "secondary_validation_rows": int(len(cal)),
        "calibration_method": selected_method,
        "schema_version": "2026-03-13.2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": resolved_run_tag,
    }

    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    legacy_calibrator_path = Path("models/pd_calibrator.pkl")
    with open(legacy_calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)

    decision_threshold_path = Path(
        decision_cfg.get("output_path", "models/decision_threshold.json")
    )
    decision_threshold_v2_path = Path(
        decision_cfg.get("output_path_v2", "models/decision_threshold_v2.json")
    )
    _save_json(decision_threshold_path, decision_threshold_artifact)
    _save_json(decision_threshold_v2_path, decision_threshold_artifact)
    write_threshold_semantics(
        pd_internal_selected_threshold=float(decision_threshold_artifact["selected_threshold"]),
        pd_internal_fallback_threshold=float(
            decision_threshold_artifact.get("fallback_threshold", 0.5)
        ),
        source_artifacts={
            "decision_threshold": str(decision_threshold_path),
            "decision_threshold_v2": str(decision_threshold_v2_path),
        },
        run_tag=resolved_run_tag,
        extra={
            "pd_internal_threshold_source": str(decision_threshold_artifact.get("source", "")),
            "calibration_method": selected_method,
        },
    )
    calibration_diagnostics = _build_calibration_diagnostics(
        y_cal=y_cal.to_numpy(),
        y_prob_tuned_cal=y_prob_tuned_cal,
        y_test=y_test.to_numpy(),
        y_prob_tuned_test=y_prob_tuned_test,
        test_issue_dates=test.get("issue_d", pd.Series([pd.NaT] * len(test))),
        selected_method=selected_method,
    )
    calibration_diagnostics["selected_calibrator"] = selected_method
    calibration_diagnostics["calibrators"] = {
        item["method"]: {
            "auc": float(item["auc"]),
            "brier": float(item["brier"]),
            "ece": float(item["ece"]),
        }
        for item in calibration_diagnostics.get("candidate_comparison", [])
    }
    _save_json(Path("models/pd_calibration_diagnostics.json"), calibration_diagnostics)

    _update_test_predictions(
        Path("data/processed/test_predictions.parquet"),
        y_true=y_test.to_numpy(),
        y_prob_final=y_prob_final,
    )
    _update_training_record(
        Path("models/pd_training_record.pkl"),
        selected_method=selected_method,
        selection_report=selection_report,
        calibration_metrics=calibration_metrics,
        final_test_metrics=final_test_metrics,
        decision_threshold_artifact=decision_threshold_artifact,
    )
    _update_seed_replay_status(
        Path("models/pd_hpo_seed_replay_status.json"),
        selected_method=selected_method,
        final_test_metrics=final_test_metrics,
    )

    logger.info(
        "Calibration refresh complete | selected={} | AUC={:.4f} | Brier={:.6f} | ECE={:.6f}",
        tpm._human_calibration_name(selected_method),
        final_test_metrics["auc_roc"],
        final_test_metrics["brier_score"],
        final_test_metrics["ece"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pd_model.yaml")
    args = parser.parse_args()
    main(args.config)
