"""Train PD models (LR baseline + CatBoost default/tuned) with robust calibration.

Usage:
    uv run python scripts/train_pd_model.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.evaluation.fairness import fairness_report
from src.evaluation.metrics import brier_score_decomposition, classification_metrics
from src.models.calibration import evaluate_calibration, expected_calibration_error
from src.models.conformal import create_pd_intervals, validate_coverage
from src.models.optuna_tuning import resolve_optuna_study_name
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
from src.models.venn_abers import VennAbersScoreCalibrator
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.io_utils import read_split_with_fe_fallback
from src.utils.pipeline_runtime import atomic_write_json
from src.utils.replay_manifest import load_replay_manifest, manifest_section
from src.utils.threshold_semantics import write_threshold_semantics
from src.utils.visualization import plot_murphy_diagram


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_pd_config(config: dict[str, Any], *, config_path: str) -> dict[str, Any]:
    """Fail fast on missing PD config sections and required keys."""
    if not isinstance(config, dict):
        raise ValueError(f"PD config must be a mapping: {config_path}")

    required_sections = ("output", "feature_source", "data", "hpo", "validation")
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(
            f"PD config {config_path} missing required sections: {', '.join(missing_sections)}"
        )

    required_data_keys = ("train_path", "test_path", "calibration_path")
    missing_data_keys = [
        key
        for key in required_data_keys
        if not str((config.get("data", {}) or {}).get(key, "")).strip()
    ]
    if missing_data_keys:
        raise ValueError(
            f"PD config {config_path} missing required data keys: {', '.join(missing_data_keys)}"
        )

    normalized = dict(config)
    normalized["output"] = dict(config.get("output", {}) or {})
    normalized["feature_source"] = dict(config.get("feature_source", {}) or {})
    normalized["data"] = dict(config.get("data", {}) or {})
    normalized["hpo"] = dict(config.get("hpo", {}) or {})
    normalized["validation"] = dict(config.get("validation", {}) or {})
    normalized["model"] = dict(config.get("model", {}) or {})
    normalized["conformal"] = dict(config.get("conformal", {}) or {})
    normalized["calibration"] = dict(config.get("calibration", {}) or {})

    output_defaults = {
        "model_path": "models/pd_canonical.cbm",
        "default_model_path": "models/pd_catboost_default.cbm",
        "tuned_model_path": "models/pd_canonical.cbm",
        "conformal_path": "models/pd_canonical_calibrator.pkl",
        "status_path": "models/pd_training_status.json",
        "checkpoint_dir": "models/pd_training_checkpoints",
        "brier_decomposition_path": "data/processed/brier_score_decomposition.json",
        "murphy_diagram_path": "reports/figures/calibration/murphy_diagram.png",
    }
    for key, value in output_defaults.items():
        normalized["output"].setdefault(key, value)

    normalized["feature_source"].setdefault("mode", "auto")
    normalized["feature_source"].setdefault(
        "feature_config_path", "data/processed/feature_config.pkl"
    )
    return normalized


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    atomic_write_json(target, payload)


def _write_training_status(
    status_path: str | Path,
    *,
    phase: str,
    state: str,
    config_path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage_name": "pd_training",
        "phase": phase,
        "state": state,
        "config_path": config_path,
        "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
        **(extra or {}),
    }
    _write_json(status_path, payload)


def _write_checkpoint(
    checkpoint_dir: str | Path,
    name: str,
    payload: dict[str, Any],
) -> None:
    target = Path(checkpoint_dir) / f"{name}.json"
    _write_json(target, payload)


def _metric_with_aliases(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _validate_replay_expectations(
    *,
    replay_cfg: dict[str, Any],
    final_test_metrics: dict[str, Any],
    feature_names: list[str],
    config_path: str,
) -> None:
    expected = dict(replay_cfg.get("expectations") or {})
    tolerances = dict(replay_cfg.get("tolerances") or {})
    expected_features = [str(x) for x in replay_cfg.get("feature_names", [])]
    if expected_features and list(feature_names) != expected_features:
        raise ValueError("Replay feature order mismatch against frozen manifest.")
    manifest_config_path = str(replay_cfg.get("config_path", "")).strip()
    if manifest_config_path and manifest_config_path != str(config_path):
        raise ValueError("Replay config_path does not match frozen manifest.")

    aliases = {
        "auc_roc": ("auc_roc",),
        "brier_score": ("brier_score",),
        "ece": ("ece",),
        "d2_brier_score": ("d2_brier_score",),
    }
    violations: list[str] = []
    for name, key_aliases in aliases.items():
        expected_value = _metric_with_aliases(expected, name, *key_aliases)
        tolerance = _metric_with_aliases(tolerances, name, *key_aliases)
        observed = _metric_with_aliases(final_test_metrics, name, *key_aliases)
        if expected_value is None or tolerance is None or observed is None:
            continue
        if abs(observed - expected_value) > tolerance:
            violations.append(
                f"{name}: observed={observed:.6f} expected={expected_value:.6f} tol={tolerance:.6f}"
            )
    if violations:
        raise ValueError("Replay metric validation failed: " + "; ".join(violations))


def _gpu_replay_artifact_root() -> Path | None:
    raw = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    return Path(raw) if raw else None


def _artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    root = _gpu_replay_artifact_root()
    if root is None:
        return path
    return root / path


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


def _issue_quarter(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("Q").astype("string")


def _apply_training_regime(
    train: pd.DataFrame,
    regime_cfg: dict[str, Any],
    *,
    date_col: str = "issue_d",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(regime_cfg.get("mode", "standard")).strip().lower() or "standard"
    out = train.copy()
    meta: dict[str, Any] = {"mode": mode}

    if mode.startswith("recent_"):
        n_quarters = int(regime_cfg.get("recent_window_quarters", 12))
        if date_col in out.columns:
            periods = _issue_quarter(out[date_col])
            valid = periods.dropna().astype(str)
            if not valid.empty:
                keep_periods = set(valid.sort_values().unique().tolist()[-n_quarters:])
                mask = periods.astype(str).isin(keep_periods)
                out = out.loc[mask].copy()
                meta["recent_window_quarters"] = int(n_quarters)
                meta["rows_after_recent_window"] = int(len(out))
    elif mode == "full_weighted":
        if date_col in out.columns:
            periods = _issue_quarter(out[date_col])
            codes = periods.astype("category").cat.codes.to_numpy(dtype=float)
            max_code = float(np.nanmax(codes)) if len(codes) else 0.0
            age = np.clip(max_code - codes, a_min=0.0, a_max=None)
            half_life = float(regime_cfg.get("half_life_quarters", 8.0))
            half_life = max(1.0, half_life)
            out["_recency_weight"] = np.power(0.5, age / half_life)
            meta["half_life_quarters"] = float(half_life)
            meta["weight_min"] = float(out["_recency_weight"].min())
            meta["weight_max"] = float(out["_recency_weight"].max())
            meta["weight_mean"] = float(out["_recency_weight"].mean())

    return out.reset_index(drop=True), meta


def _training_weights(df: pd.DataFrame) -> np.ndarray | None:
    if "_recency_weight" not in df.columns:
        return None
    return pd.to_numeric(df["_recency_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)


def _apply_stable_core(
    features: list[str],
    categorical_features: list[str],
    stable_core_cfg: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    if not bool(stable_core_cfg.get("enabled", False)):
        return features, categorical_features, {"enabled": False, "excluded_features": []}
    excluded = stable_core_cfg.get("exclude_features", ["rev_utilization", "high_util_pct"])
    excluded = [str(x) for x in excluded]
    filtered_features = [f for f in features if f not in excluded]
    filtered_categorical = [f for f in categorical_features if f in filtered_features]
    return (
        filtered_features,
        filtered_categorical,
        {
            "enabled": True,
            "excluded_features": excluded,
            "feature_count_after_filter": int(len(filtered_features)),
        },
    )


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
    if method == "venn_abers":
        model = VennAbersScoreCalibrator()
        model.fit(y_prob_raw, y_true)
        return model
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(y_prob_raw.reshape(-1, 1), y_true)
        return model
    if method == "isotonic":
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(y_prob_raw, y_true)
        return model
    if method == "beta":
        from src.models.calibration import calibrate_beta

        return calibrate_beta(y_true, y_prob_raw)
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_calibrator(calibrator: Any, y_prob_raw: np.ndarray) -> np.ndarray:
    """Apply score-based calibrator."""
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(y_prob_raw)
    raise TypeError(f"Unsupported calibrator type: {type(calibrator)}")


def _build_fairness_groups_for_threshold(
    df: pd.DataFrame,
    attributes: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build groups dictionary from fairness attribute config."""
    groups: dict[str, np.ndarray] = {}
    for attr in attributes:
        name = str(attr.get("name", "")).strip()
        col = str(attr.get("column", "")).strip()
        if not name or not col or col not in df.columns:
            continue

        if str(attr.get("binning", "")).strip() == "quartile":
            series = pd.to_numeric(df[col], errors="coerce")
            groups[name] = (
                pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                .astype(str)
                .to_numpy()
            )
        else:
            groups[name] = df[col].astype(str).to_numpy()
    return groups


def _select_decision_threshold(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: dict[str, Any],
    groups_dict: dict[str, np.ndarray],
    thresholds: np.ndarray,
    fallback_threshold: float,
    y_true_secondary: np.ndarray | None = None,
    y_prob_secondary: np.ndarray | None = None,
    groups_dict_secondary: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Select threshold prioritizing fairness pass with optional temporal robustness check."""

    def _fairness_row(
        *,
        y_true_eval: np.ndarray,
        y_prob_eval: np.ndarray,
        groups_eval: dict[str, np.ndarray],
        thr: float,
    ) -> dict[str, Any]:
        report = fairness_report(
            y_true=y_true_eval,
            y_pred_proba=y_prob_eval,
            groups_dict=groups_eval,
            threshold=float(thr),
            dpd_threshold=float(policy["dpd_threshold"]),
            eo_gap_threshold=float(policy["eo_gap_threshold"]),
            dir_threshold=float(policy["dir_threshold"]),
        )
        if report.empty:
            return {
                "overall_pass": False,
                "pass_ratio": 0.0,
                "max_dpd": 1.0,
                "max_eo_gap": 1.0,
                "min_dir": 0.0,
            }
        return {
            "overall_pass": bool(report["passed_all"].all()),
            "pass_ratio": float(report["passed_all"].mean()),
            "max_dpd": float(report["dpd"].max()),
            "max_eo_gap": float(report["eo_gap"].max()),
            "min_dir": float(report["dir"].min()),
        }

    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        primary = _fairness_row(
            y_true_eval=y_true,
            y_prob_eval=y_prob,
            groups_eval=groups_dict,
            thr=float(thr),
        )
        row = {
            "threshold": float(thr),
            "overall_pass": bool(primary["overall_pass"]),
            "pass_ratio": float(primary["pass_ratio"]),
            "max_dpd": float(primary["max_dpd"]),
            "max_eo_gap": float(primary["max_eo_gap"]),
            "min_dir": float(primary["min_dir"]),
        }

        if (
            y_true_secondary is not None
            and y_prob_secondary is not None
            and groups_dict_secondary is not None
            and len(groups_dict_secondary) > 0
        ):
            secondary = _fairness_row(
                y_true_eval=y_true_secondary,
                y_prob_eval=y_prob_secondary,
                groups_eval=groups_dict_secondary,
                thr=float(thr),
            )
            row.update(
                {
                    "overall_pass_secondary": bool(secondary["overall_pass"]),
                    "pass_ratio_secondary": float(secondary["pass_ratio"]),
                    "max_dpd_secondary": float(secondary["max_dpd"]),
                    "max_eo_gap_secondary": float(secondary["max_eo_gap"]),
                    "min_dir_secondary": float(secondary["min_dir"]),
                    "robust_overall_pass": bool(
                        primary["overall_pass"] and secondary["overall_pass"]
                    ),
                    "robust_pass_ratio": float(min(primary["pass_ratio"], secondary["pass_ratio"])),
                    "robust_max_dpd": float(max(primary["max_dpd"], secondary["max_dpd"])),
                    "robust_max_eo_gap": float(max(primary["max_eo_gap"], secondary["max_eo_gap"])),
                    "robust_min_dir": float(min(primary["min_dir"], secondary["min_dir"])),
                }
            )
        else:
            row.update(
                {
                    "overall_pass_secondary": None,
                    "pass_ratio_secondary": None,
                    "max_dpd_secondary": None,
                    "max_eo_gap_secondary": None,
                    "min_dir_secondary": None,
                    "robust_overall_pass": bool(primary["overall_pass"]),
                    "robust_pass_ratio": float(primary["pass_ratio"]),
                    "robust_max_dpd": float(primary["max_dpd"]),
                    "robust_max_eo_gap": float(primary["max_eo_gap"]),
                    "robust_min_dir": float(primary["min_dir"]),
                }
            )
        row["distance_from_fallback"] = float(abs(row["threshold"] - float(fallback_threshold)))
        rows.append(row)

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "robust_overall_pass",
                "robust_pass_ratio",
                "robust_max_eo_gap",
                "robust_max_dpd",
                "robust_min_dir",
                "distance_from_fallback",
            ],
            ascending=[False, False, True, True, False, True],
        )
        .reset_index(drop=True)
    )
    selected = ranking.iloc[0].to_dict()
    return {
        "selected_threshold": float(selected["threshold"]),
        "search_summary": ranking.to_dict(orient="records"),
        "selection_metrics": selected,
    }


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

        brier_raw = float(brier_score_loss(y_eval, p_eval))
        brier_cal = float(brier_score_loss(y_eval, p_eval_cal))

        fold_rows.append(
            {
                "fold": fold_id,
                "n_fit": int(len(idx_fit)),
                "n_eval": int(len(idx_eval)),
                "raw_auc": None if raw_auc is None else float(raw_auc),
                "cal_auc": None if cal_auc is None else float(cal_auc),
                "auc_drop": float(auc_drop),
                "brier": brier_cal,
                "brier_raw": brier_raw,
                "brier_degraded": brier_cal > brier_raw,
                "log_loss": float(log_loss(y_eval, p_eval_cal)),
                "ece": float(expected_calibration_error(y_eval, p_eval_cal)),
            }
        )

    if not fold_rows:
        return {
            "method": method,
            "folds_used": 0,
            "mean_brier": float("inf"),
            "mean_log_loss": float("inf"),
            "mean_ece": float("inf"),
            "mean_auc_drop": float("inf"),
            "brier_variance": float("inf"),
            "ece_variance": float("inf"),
            "stability": float("inf"),
            "degradation_rate": 1.0,
            "folds": [],
        }

    briers = np.array([r["brier"] for r in fold_rows], dtype=float)
    log_losses = np.array([r["log_loss"] for r in fold_rows], dtype=float)
    eces = np.array([r["ece"] for r in fold_rows], dtype=float)
    auc_drops = np.array([r["auc_drop"] for r in fold_rows], dtype=float)
    n_degraded = sum(1 for r in fold_rows if r.get("brier_degraded", False))

    return {
        "method": method,
        "folds_used": int(len(fold_rows)),
        "mean_brier": float(np.mean(briers)),
        "mean_log_loss": float(np.mean(log_losses)),
        "mean_ece": float(np.mean(eces)),
        "mean_auc_drop": float(np.mean(auc_drops)),
        "brier_variance": float(np.var(briers)),
        "ece_variance": float(np.var(eces)),
        "stability": float(np.var(briers) + np.var(eces)),
        "degradation_rate": float(n_degraded / len(fold_rows)),
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
    if method == "venn_abers":
        return "Venn-Abers"
    if method == "beta":
        return "Beta Calibration"
    return method


def _build_walk_forward_splits(
    df: pd.DataFrame,
    *,
    n_windows: int = 3,
    min_train_rows: int = 200_000,
    window_rows: int = 80_000,
    date_col: str = "issue_d",
    max_rows: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build anchored walk-forward temporal splits from train data."""
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        # Keep the latest rows to preserve recent temporal dynamics.
        if date_col in df.columns:
            work = df.sort_values(date_col).tail(int(max_rows)).reset_index(drop=True)
        else:
            work = df.tail(int(max_rows)).reset_index(drop=True)
    elif date_col in df.columns:
        work = df.sort_values(date_col).reset_index(drop=True)
    else:
        work = df.reset_index(drop=True)

    n = len(work)
    if n < (min_train_rows + window_rows + 1):
        return []

    n_windows = max(1, int(n_windows))
    step = max(1, (n - min_train_rows - window_rows) // n_windows)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_windows):
        fit_end = min_train_rows + i * step
        eval_start = fit_end
        eval_end = min(n, eval_start + window_rows)
        if fit_end < min_train_rows or eval_end - eval_start < max(10_000, window_rows // 4):
            continue
        idx_fit = np.arange(0, fit_end, dtype=int)
        idx_eval = np.arange(eval_start, eval_end, dtype=int)
        splits.append((idx_fit, idx_eval))
    return splits


def _evaluate_walk_forward_auc(
    train_df: pd.DataFrame,
    *,
    features: list[str],
    categorical_features: list[str],
    target: str,
    params: dict[str, Any],
    n_windows: int = 3,
    min_train_rows: int = 200_000,
    window_rows: int = 80_000,
    date_col: str = "issue_d",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Evaluate CatBoost params with anchored temporal walk-forward AUC."""
    splits = _build_walk_forward_splits(
        train_df,
        n_windows=n_windows,
        min_train_rows=min_train_rows,
        window_rows=window_rows,
        date_col=date_col,
        max_rows=max_rows,
    )
    if not splits:
        return {
            "enabled": False,
            "reason": "insufficient_rows",
            "n_windows_requested": int(n_windows),
            "n_windows_used": 0,
            "folds": [],
        }

    if date_col in train_df.columns:
        ordered = train_df.sort_values(date_col).reset_index(drop=True)
    else:
        ordered = train_df.reset_index(drop=True)

    folds: list[dict[str, Any]] = []
    for fold_id, (idx_fit, idx_eval) in enumerate(splits, start=1):
        fit_df = ordered.iloc[idx_fit].reset_index(drop=True)
        eval_df = ordered.iloc[idx_eval].reset_index(drop=True)
        y_fit = fit_df[target].astype(int)
        y_eval = eval_df[target].astype(int)
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
            continue

        X_fit = _prepare_catboost_frame(fit_df, features, categorical_features)
        X_eval = _prepare_catboost_frame(eval_df, features, categorical_features)
        fit_weights = _training_weights(fit_df)
        eval_weights = _training_weights(eval_df)
        _, metrics = train_catboost_default(
            X_fit,
            y_fit,
            X_eval,
            y_eval,
            cat_features=categorical_features,
            params=params,
            sample_weight=fit_weights,
            eval_sample_weight=eval_weights,
        )
        folds.append(
            {
                "fold": fold_id,
                "fit_rows": int(len(idx_fit)),
                "eval_rows": int(len(idx_eval)),
                "validation_auc": float(metrics.get("validation_auc", 0.0)),
                "best_iteration": int(metrics.get("best_iteration", 0)),
            }
        )

    if not folds:
        return {
            "enabled": False,
            "reason": "degenerate_target_class",
            "n_windows_requested": int(n_windows),
            "n_windows_used": 0,
            "folds": [],
        }

    aucs = np.array([r["validation_auc"] for r in folds], dtype=float)
    return {
        "enabled": True,
        "n_windows_requested": int(n_windows),
        "n_windows_used": int(len(folds)),
        "mean_validation_auc": float(np.mean(aucs)),
        "median_validation_auc": float(np.median(aucs)),
        "std_validation_auc": float(np.std(aucs)),
        "min_validation_auc": float(np.min(aucs)),
        "max_validation_auc": float(np.max(aucs)),
        "folds": folds,
    }


def _replay_top_optuna_trials(
    *,
    hpo_cfg: dict[str, Any],
    base_params: dict[str, Any],
    X_train_fit_cb: pd.DataFrame,
    y_train_fit: pd.Series,
    X_val_cb: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
    seeds: list[int],
    top_k_trials: int = 3,
    prioritize_gate_pass: bool = True,
    sample_weight: np.ndarray | None = None,
    eval_sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay top Optuna trials across multiple seeds for robustness."""
    report: dict[str, Any] = {
        "enabled": False,
        "reason": "not_run",
        "rows": [],
        "selected_trial": None,
        "selected_params": None,
    }
    if not bool(hpo_cfg.get("enabled", True)):
        report["reason"] = "hpo_disabled"
        return report

    storage = hpo_cfg.get("study_storage")
    study_name = resolve_optuna_study_name(hpo_cfg.get("study_name"))
    if not storage or not study_name:
        report["reason"] = "missing_study_storage_or_name"
        return report

    try:
        import optuna
    except Exception:
        report["reason"] = "optuna_unavailable"
        return report

    try:
        study = optuna.load_study(study_name=str(study_name), storage=str(storage))
    except Exception as exc:
        report["reason"] = f"study_load_failed: {exc}"
        return report

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not complete:
        report["reason"] = "no_complete_trials"
        return report

    top = sorted(complete, key=lambda t: float(t.value), reverse=True)[: max(1, int(top_k_trials))]
    rows: list[dict[str, Any]] = []
    for trial in top:
        fairness_pass_attr = trial.user_attrs.get("fairness_pass")
        conformal_pass_attr = trial.user_attrs.get("conformal_pass")
        governance_pass_attr = trial.user_attrs.get("governance_pass")
        gate_attrs_present = all(
            key in trial.user_attrs
            for key in ("fairness_pass", "conformal_pass", "governance_pass")
        )
        gate_all_pass = (
            bool(fairness_pass_attr and conformal_pass_attr and governance_pass_attr)
            if gate_attrs_present
            else None
        )
        for seed in seeds:
            params = {**base_params, **trial.params, "random_seed": int(seed)}
            model, metrics = train_catboost_default(
                X_train_fit_cb,
                y_train_fit,
                X_val_cb,
                y_val,
                cat_features=cat_features,
                params=params,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
            )
            y_val_prob = model.predict_proba(X_val_cb)[:, 1]
            rows.append(
                {
                    "trial_number": int(trial.number),
                    "seed": int(seed),
                    "validation_auc": float(metrics.get("validation_auc", 0.0)),
                    "validation_brier": float(brier_score_loss(y_val, y_val_prob)),
                    "validation_ece": float(expected_calibration_error(y_val, y_val_prob)),
                    "best_iteration": int(metrics.get("best_iteration", 0)),
                    "trial_best_value": float(trial.value),
                    "fairness_pass": fairness_pass_attr,
                    "conformal_pass": conformal_pass_attr,
                    "governance_pass": governance_pass_attr,
                    "gate_attrs_present": bool(gate_attrs_present),
                    "gate_all_pass": gate_all_pass,
                    "params": trial.params,
                }
            )

    if not rows:
        report["reason"] = "no_replay_rows"
        return report

    replay_df = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for trial_number, grp in replay_df.groupby("trial_number", observed=True):
        gate_present = bool(grp["gate_attrs_present"].fillna(False).any())
        gate_values = grp["gate_all_pass"].dropna().astype(bool)
        if gate_values.empty:
            gate_all_pass_summary: bool | None = None
        else:
            gate_all_pass_summary = bool(gate_values.all())

        if not prioritize_gate_pass:
            gate_tier = 1
        elif gate_all_pass_summary is True:
            gate_tier = 0
        elif gate_all_pass_summary is None:
            gate_tier = 1
        else:
            gate_tier = 2

        summary_rows.append(
            {
                "trial_number": int(trial_number),
                "median_validation_auc": float(grp["validation_auc"].median()),
                "mean_validation_auc": float(grp["validation_auc"].mean()),
                "std_validation_auc": float(grp["validation_auc"].std(ddof=0)),
                "mean_validation_brier": float(grp["validation_brier"].mean()),
                "mean_validation_ece": float(grp["validation_ece"].mean()),
                "gate_attrs_present": gate_present,
                "gate_all_pass": gate_all_pass_summary,
                "gate_tier": int(gate_tier),
            }
        )

    grouped = pd.DataFrame(summary_rows).sort_values(
        [
            "gate_tier",
            "mean_validation_ece",
            "median_validation_auc",
            "mean_validation_brier",
        ],
        ascending=[True, True, False, True],
    )
    selected_trial = int(grouped.iloc[0]["trial_number"])
    selected_trial_obj = next(t for t in top if int(t.number) == selected_trial)
    selected_params = {**base_params, **selected_trial_obj.params}

    report.update(
        {
            "enabled": True,
            "reason": "ok",
            "top_k_trials": int(len(top)),
            "seeds": [int(s) for s in seeds],
            "selection_policy": {
                "prioritize_gate_pass": bool(prioritize_gate_pass),
                "rank_order": [
                    "gate_tier(pass->unknown->fail)",
                    "mean_validation_ece(asc)",
                    "median_validation_auc(desc)",
                    "mean_validation_brier(asc)",
                ],
            },
            "rows": rows,
            "summary_by_trial": grouped.to_dict(orient="records"),
            "selected_trial": selected_trial,
            "selected_params": selected_params,
        }
    )
    return report


def main(
    config_path: str = "configs/pd_model.yaml",
    sample_size: int | None = None,
    training_regime_mode: str | None = None,
    recent_window_quarters: int | None = None,
    half_life_quarters: int | None = None,
    stable_core_enabled: bool | None = None,
    hpo_n_trials: int | None = None,
    hpo_enabled: bool | None = None,
    challenger_enabled: bool | None = None,
    walk_forward_enabled: bool | None = None,
    seed_replay_enabled: bool | None = None,
    catboost_iterations: int | None = None,
    validate_only: bool = False,
    mode: str = "search",
    replay_manifest_path: str | None = None,
) -> None:
    if sample_size is not None and int(sample_size) <= 0:
        sample_size = None
    config = validate_pd_config(load_config(config_path), config_path=config_path)
    run_mode = str(mode or "search").strip().lower() or "search"
    replay_cfg = manifest_section(load_replay_manifest(replay_manifest_path), "pd")
    regime_cfg = dict(config.get("training_regime", {}) or {})
    if training_regime_mode is not None:
        regime_cfg["mode"] = str(training_regime_mode)
    if recent_window_quarters is not None:
        regime_cfg["recent_window_quarters"] = int(recent_window_quarters)
    if half_life_quarters is not None:
        regime_cfg["half_life_quarters"] = int(half_life_quarters)
    config["training_regime"] = regime_cfg

    stable_core_cfg = dict(config.get("stable_core", {}) or {})
    if stable_core_enabled is not None:
        stable_core_cfg["enabled"] = bool(stable_core_enabled)
    config["stable_core"] = stable_core_cfg

    hpo_cfg = dict(config.get("hpo", {}) or {})
    if hpo_n_trials is not None:
        hpo_cfg["n_trials"] = int(hpo_n_trials)
    if hpo_enabled is not None:
        hpo_cfg["enabled"] = bool(hpo_enabled)
    config["hpo"] = hpo_cfg

    challenger_cfg = dict(config.get("challenger_pipeline", {}) or {})
    if challenger_enabled is not None:
        challenger_cfg["enabled"] = bool(challenger_enabled)
    config["challenger_pipeline"] = challenger_cfg

    validation_cfg = dict(config.get("validation", {}) or {})
    walk_cfg = dict(validation_cfg.get("walk_forward", {}) or {})
    if walk_forward_enabled is not None:
        walk_cfg["enabled"] = bool(walk_forward_enabled)
    validation_cfg["walk_forward"] = walk_cfg
    seed_cfg = dict(validation_cfg.get("seed_replay", {}) or {})
    if seed_replay_enabled is not None:
        seed_cfg["enabled"] = bool(seed_replay_enabled)
    validation_cfg["seed_replay"] = seed_cfg
    config["validation"] = validation_cfg

    model_cfg = dict(config.get("model", {}) or {})
    model_params = dict(model_cfg.get("params", {}) or {})
    if catboost_iterations is not None:
        model_params["iterations"] = int(catboost_iterations)
    model_cfg["params"] = model_params
    config["model"] = model_cfg

    if run_mode == "replay":
        if not replay_cfg:
            raise ValueError("Replay mode requires a PD section in the replay manifest.")
        fixed_params = dict(replay_cfg.get("selected_params") or {})
        if not fixed_params:
            raise ValueError("Replay manifest missing pd.selected_params.")
        config["hpo"]["enabled"] = False
        config["validation"]["seed_replay"]["enabled"] = False
        config["challenger_pipeline"]["enabled"] = False
        merged_params = dict(config["model"].get("params", {}) or {})
        merged_params.update(fixed_params)
        config["model"]["params"] = merged_params

    status_path = _artifact_path(
        config["output"].get("status_path", "models/pd_training_status.json")
    )
    checkpoint_dir = _artifact_path(
        config["output"].get("checkpoint_dir", "models/pd_training_checkpoints")
    )
    _write_training_status(
        status_path,
        phase="config_validated",
        state="running",
        config_path=config_path,
        extra={"validate_only": bool(validate_only)},
    )
    _write_checkpoint(
        checkpoint_dir,
        "config_validation",
        {
            "config_path": config_path,
            "output": config["output"],
            "feature_source": config["feature_source"],
            "data": config["data"],
            "hpo_enabled": bool(config["hpo"].get("enabled", True)),
            "validation": config["validation"],
        },
    )

    if validate_only:
        logger.info("PD config validation passed for {}", config_path)
        _write_training_status(
            status_path,
            phase="config_validated",
            state="completed",
            config_path=config_path,
            extra={"validate_only": True},
        )
        return

    logger.info(f"Config loaded from {config_path}")

    train = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["train_path"]))
    test = _normalize_percent_columns(read_split_with_fe_fallback(config["data"]["test_path"]))
    cal = _normalize_percent_columns(
        read_split_with_fe_fallback(config["data"]["calibration_path"])
    )
    _write_training_status(
        status_path,
        phase="data_loaded",
        state="running",
        config_path=config_path,
        extra={
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
        },
    )

    train, regime_meta = _apply_training_regime(train, regime_cfg, date_col="issue_d")

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

    if run_mode == "replay":
        replay_features = [str(x) for x in replay_cfg.get("feature_names", [])]
        replay_categorical = [str(x) for x in replay_cfg.get("categorical_features", [])]
        if replay_features:
            catboost_features = replay_features
            categorical_features = replay_categorical

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

    catboost_features, categorical_features, stable_core_meta = _apply_stable_core(
        catboost_features,
        categorical_features,
        {} if run_mode == "replay" else (config.get("stable_core", {}) or {}),
    )
    logreg_features = [c for c in logreg_features if c in catboost_features]

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
    _write_checkpoint(
        checkpoint_dir,
        "feature_resolution",
        {
            "feature_source": feature_sets.get("feature_source", feature_mode),
            "catboost_features": catboost_features,
            "logreg_features": logreg_features,
            "categorical_features": categorical_features,
        },
    )
    _write_training_status(
        status_path,
        phase="features_resolved",
        state="running",
        config_path=config_path,
        extra={
            "catboost_feature_count": int(len(catboost_features)),
            "logreg_feature_count": int(len(logreg_features)),
        },
    )

    val_cfg = config.get("validation", {})
    walk_cfg = val_cfg.get("walk_forward", {})
    seed_replay_cfg = val_cfg.get("seed_replay", {})
    val_fraction = float(val_cfg.get("val_from_tail_fraction_of_train", 0.15))
    train_fit, train_val = temporal_train_val_split(
        train, val_fraction=val_fraction, date_col="issue_d"
    )
    train_fit_weights = _training_weights(train_fit)
    train_val_weights = _training_weights(train_val)

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
    lr_model, lr_metrics = train_baseline(
        X_train_fit_lr,
        y_train_fit,
        X_test_lr,
        y_test,
        sample_weight=train_fit_weights,
    )
    _write_training_status(
        status_path,
        phase="baseline_trained",
        state="running",
        config_path=config_path,
        extra={"baseline_auc": float(lr_metrics.get("auc_roc", 0.0))},
    )

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
        sample_weight=train_fit_weights,
        eval_sample_weight=train_val_weights,
    )
    _write_training_status(
        status_path,
        phase="default_catboost_trained",
        state="running",
        config_path=config_path,
        extra={"validation_auc": float(cb_default_metrics.get("validation_auc", 0.0))},
    )

    # CatBoost tuned (Optuna)
    hpo_cfg = config.get("hpo", {})
    seed_replay_report: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled_in_config",
        "rows": [],
        "selected_trial": None,
        "selected_params": None,
    }
    if run_mode == "replay":
        replay_params = dict(replay_cfg.get("selected_params") or {})
        cb_tuned_model, cb_tuned_metrics = train_catboost_default(
            X_train_fit_cb,
            y_train_fit,
            X_val_cb,
            y_val,
            X_test=X_test_cb,
            y_test=y_test,
            cat_features=categorical_features,
            params=replay_params,
            sample_weight=train_fit_weights,
            eval_sample_weight=train_val_weights,
        )
        cb_tuned_metrics["hpo_trials_executed"] = 0
        cb_tuned_metrics["hpo_best_validation_auc"] = float(cb_tuned_metrics["validation_auc"])
        cb_tuned_metrics["best_params"] = replay_params
        cb_tuned_metrics["model_type"] = "catboost_replay_manifest"
        seed_replay_report["reason"] = "replay_manifest"
        seed_replay_report["selected_params"] = replay_params
    elif bool(hpo_cfg.get("enabled", True)):
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
            group_tpe=bool(hpo_cfg.get("group_tpe", True)),
            warn_independent_sampling=bool(hpo_cfg.get("warn_independent_sampling", True)),
            constant_liar=bool(hpo_cfg.get("constant_liar", False)),
            pruner_n_startup_trials=int(hpo_cfg.get("pruner_n_startup_trials", 20)),
            pruner_n_warmup_steps=int(hpo_cfg.get("pruner_n_warmup_steps", 50)),
            use_pruning_callback=bool(hpo_cfg.get("use_pruning_callback", True)),
            study_storage=hpo_cfg.get("study_storage", None),
            study_name=hpo_cfg.get("study_name", None),
            load_if_exists=bool(hpo_cfg.get("load_if_exists", True)),
            refit_full_train=bool(hpo_cfg.get("refit_full_train", True)),
            gc_after_trial=bool(hpo_cfg.get("gc_after_trial", True)),
            storage_heartbeat_interval=int(hpo_cfg.get("storage_heartbeat_interval", 0)),
            storage_grace_period=int(hpo_cfg.get("storage_grace_period", 0)),
            sqlite_timeout_seconds=int(hpo_cfg.get("sqlite_timeout_seconds", 60)),
            retry_failed_trials=int(hpo_cfg.get("retry_failed_trials", 0)),
            n_jobs=int(hpo_cfg.get("n_jobs", 1)),
            sample_weight=train_fit_weights,
            eval_sample_weight=train_val_weights,
        )

        if bool(seed_replay_cfg.get("enabled", True)):
            seeds = seed_replay_cfg.get("seeds", [42, 52, 62])
            seeds = [int(s) for s in seeds]
            prioritize_gate_pass = bool(seed_replay_cfg.get("prioritize_gate_pass", True))
            seed_replay_report = _replay_top_optuna_trials(
                hpo_cfg=hpo_cfg,
                base_params=config["model"].get("params", {}),
                X_train_fit_cb=X_train_fit_cb,
                y_train_fit=y_train_fit,
                X_val_cb=X_val_cb,
                y_val=y_val,
                cat_features=categorical_features,
                seeds=seeds,
                top_k_trials=int(seed_replay_cfg.get("top_k_trials", 3)),
                prioritize_gate_pass=prioritize_gate_pass,
                sample_weight=train_fit_weights,
                eval_sample_weight=train_val_weights,
            )
            if seed_replay_report.get("enabled") and seed_replay_report.get("selected_params"):
                selected_params = dict(seed_replay_report["selected_params"])
                replay_model, replay_metrics = train_catboost_default(
                    X_train_fit_cb,
                    y_train_fit,
                    X_val_cb,
                    y_val,
                    X_test=X_test_cb,
                    y_test=y_test,
                    cat_features=categorical_features,
                    params=selected_params,
                    sample_weight=train_fit_weights,
                    eval_sample_weight=train_val_weights,
                )
                cb_tuned_metrics = {
                    **cb_tuned_metrics,
                    **replay_metrics,
                    "model_type": "catboost_tuned_seed_replay_selected",
                    "best_params": selected_params,
                    "seed_replay_selected_trial": seed_replay_report.get("selected_trial"),
                    "seed_replay_enabled": True,
                }
                cb_tuned_model = replay_model
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
            sample_weight=train_fit_weights,
            eval_sample_weight=train_val_weights,
        )
        cb_tuned_metrics["hpo_trials_executed"] = 0
        cb_tuned_metrics["hpo_best_validation_auc"] = float(cb_tuned_metrics["validation_auc"])
        cb_tuned_metrics["best_params"] = config["model"].get("params", {})
        seed_replay_report["reason"] = "hpo_disabled"
    _write_checkpoint(
        checkpoint_dir,
        "hpo_summary",
        {
            "best_params": cb_tuned_metrics.get("best_params", {}),
            "hpo_trials_executed": int(cb_tuned_metrics.get("hpo_trials_executed", 0)),
            "hpo_best_validation_auc": float(cb_tuned_metrics.get("hpo_best_validation_auc", 0.0)),
            "seed_replay_report": seed_replay_report,
        },
    )
    _write_training_status(
        status_path,
        phase="tuned_catboost_trained",
        state="running",
        config_path=config_path,
        extra={
            "validation_auc": float(cb_tuned_metrics.get("hpo_best_validation_auc", 0.0)),
            "best_iteration": int(cb_tuned_metrics.get("best_iteration", 0)),
        },
    )

    walk_forward_report: dict[str, Any]
    if bool(walk_cfg.get("enabled", True)):
        walk_forward_report = _evaluate_walk_forward_auc(
            train,
            features=catboost_features,
            categorical_features=categorical_features,
            target=TARGET,
            params=dict(cb_tuned_metrics.get("best_params", config["model"].get("params", {}))),
            n_windows=int(walk_cfg.get("n_windows", 3)),
            min_train_rows=int(walk_cfg.get("min_train_rows", 200_000)),
            window_rows=int(walk_cfg.get("window_rows", 80_000)),
            date_col=str(walk_cfg.get("date_col", "issue_d")),
            max_rows=(
                None if int(walk_cfg.get("max_rows", 0)) <= 0 else int(walk_cfg.get("max_rows", 0))
            ),
        )
    else:
        walk_forward_report = {
            "enabled": False,
            "reason": "disabled_in_config",
            "n_windows_requested": int(walk_cfg.get("n_windows", 0)),
            "n_windows_used": 0,
            "folds": [],
        }

    # Raw probabilities
    y_prob_default_test = cb_default_model.predict_proba(X_test_cb)[:, 1]
    y_prob_tuned_test = cb_tuned_model.predict_proba(X_test_cb)[:, 1]
    y_prob_tuned_val = cb_tuned_model.predict_proba(X_val_cb)[:, 1]
    y_prob_tuned_cal = cb_tuned_model.predict_proba(X_cal_cb)[:, 1]

    # Robust calibration policy selection via temporal folds on calibration set
    cal_splits = _build_calibration_backtest_splits(cal, n_folds=4, date_col="issue_d")
    cal_cfg = config.get("calibration", {})
    cal_candidates = cal_cfg.get("candidates", ["platt", "isotonic", "venn_abers"])
    cal_reports: list[dict[str, Any]] = []
    if run_mode == "replay":
        selected_cal_method = str(replay_cfg.get("selected_calibration_method", "venn_abers"))
        try:
            cal_reports.append(
                _evaluate_calibration_method(
                    selected_cal_method,
                    y_cal.to_numpy(),
                    y_prob_tuned_cal,
                    cal_splits,
                )
            )
        except Exception as exc:
            cal_reports.append(
                {
                    "method": selected_cal_method,
                    "folds_used": 0,
                    "mean_brier": float("inf"),
                    "mean_log_loss": float("inf"),
                    "mean_ece": float("inf"),
                    "mean_auc_drop": float("inf"),
                    "brier_variance": float("inf"),
                    "ece_variance": float("inf"),
                    "stability": float("inf"),
                    "degradation_rate": 1.0,
                    "error": str(exc),
                    "folds": [],
                }
            )
        cal_selection_report = {
            "selected_method": selected_cal_method,
            "selection_reason": "frozen_replay_manifest",
            "auc_drop_limit": 0.0015,
            "candidates": cal_reports,
            "feasible_candidates": [
                row for row in cal_reports if row.get("method") == selected_cal_method
            ],
        }
    else:
        for method in cal_candidates:
            try:
                cal_reports.append(
                    _evaluate_calibration_method(
                        str(method),
                        y_cal.to_numpy(),
                        y_prob_tuned_cal,
                        cal_splits,
                    )
                )
            except Exception as exc:
                cal_reports.append(
                    {
                        "method": str(method),
                        "folds_used": 0,
                        "mean_brier": float("inf"),
                        "mean_log_loss": float("inf"),
                        "mean_ece": float("inf"),
                        "mean_auc_drop": float("inf"),
                        "brier_variance": float("inf"),
                        "ece_variance": float("inf"),
                        "stability": float("inf"),
                        "degradation_rate": 1.0,
                        "error": str(exc),
                        "folds": [],
                    }
                )
        selected_cal_method, cal_selection_report = _select_calibration_method(
            cal_reports,
            auc_drop_limit=0.0015,
        )
    _write_checkpoint(
        checkpoint_dir,
        "calibration_selection",
        {
            "selected_method": selected_cal_method,
            "selection_report": cal_selection_report,
            "candidate_reports": cal_reports,
        },
    )
    _write_training_status(
        status_path,
        phase="calibration_selected",
        state="running",
        config_path=config_path,
        extra={"selected_calibration_method": selected_cal_method},
    )

    calibrator = _fit_calibrator_from_scores(
        selected_cal_method,
        y_cal.to_numpy(),
        y_prob_tuned_cal,
    )
    y_prob_final = _apply_calibrator(calibrator, y_prob_tuned_test)
    y_prob_final_val = _apply_calibrator(calibrator, y_prob_tuned_val)
    cal_metrics = evaluate_calibration(
        y_test.to_numpy(),
        y_prob_final,
        name=selected_cal_method,
    )

    decision_cfg = config.get("decision_threshold", {})
    resolved_run_tag = resolve_run_tag(require_explicit=True)
    decision_threshold_artifact = {
        "enabled": False,
        "selected_threshold": float(config.get("calibration", {}).get("default_threshold", 0.5)),
        "source": "fallback_default",
        **build_artifact_metadata(
            schema_version="2026-03-01.1",
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    if run_mode == "replay" and replay_cfg.get("decision_threshold_artifact"):
        decision_threshold_artifact = dict(replay_cfg["decision_threshold_artifact"])
        decision_threshold_artifact.update(
            build_artifact_metadata(
                schema_version="2026-03-01.1",
                run_tag=resolved_run_tag,
                require_explicit=True,
            )
        )
        decision_threshold_artifact["source"] = "frozen_replay_manifest"
    elif bool(decision_cfg.get("enabled", True)):
        fairness_policy_path = str(
            decision_cfg.get("fairness_policy_path", "configs/fairness_policy.yaml")
        )
        with open(fairness_policy_path, encoding="utf-8") as f:
            fairness_cfg = yaml.safe_load(f) or {}
        fairness_policy = fairness_cfg.get("policy", {})
        fairness_attrs = fairness_cfg.get("attributes", [])
        groups_for_threshold = _build_fairness_groups_for_threshold(train_val, fairness_attrs)
        groups_for_threshold_cal = _build_fairness_groups_for_threshold(cal, fairness_attrs)

        thr_min = float(decision_cfg.get("min_threshold", 0.05))
        thr_max = float(decision_cfg.get("max_threshold", 0.95))
        thr_step = float(decision_cfg.get("step", 0.01))
        thresholds = np.arange(thr_min, thr_max + (thr_step / 2.0), thr_step)
        fallback_threshold = float(fairness_policy.get("prediction_threshold", 0.5))
        threshold_result = _select_decision_threshold(
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
            "source": "validation_fairness_search",
            "fairness_policy_path": fairness_policy_path,
            "validation_rows": int(len(train_val)),
            "secondary_validation_rows": int(len(cal)),
            "calibration_method": selected_cal_method,
            **build_artifact_metadata(
                schema_version="2026-03-01.1",
                run_tag=resolved_run_tag,
                require_explicit=True,
            ),
        }

    # Conformal (keeps calibration split isolated from model training)
    alpha = 1.0 - float(config["conformal"].get("confidence_level", 0.9))
    _, y_intervals = create_pd_intervals(cb_tuned_model, X_cal_cb, y_cal, X_test_cb, alpha=alpha)
    cp_metrics = validate_coverage(y_test.values.astype(float), y_intervals, alpha)

    final_test_metrics = classification_metrics(y_test.values, y_prob_final)
    tuned_raw_test_metrics = classification_metrics(y_test.values, y_prob_tuned_test)
    calibrated_decomposition = brier_score_decomposition(y_test.values, y_prob_final)
    raw_decomposition = brier_score_decomposition(y_test.values, y_prob_tuned_test)
    if run_mode == "replay":
        _validate_replay_expectations(
            replay_cfg=replay_cfg,
            final_test_metrics=final_test_metrics,
            feature_names=catboost_features,
            config_path=config_path,
        )

    # Statistical calibration hypothesis tests (MAPIE)
    # Tests H0: scores are well-calibrated. High p-value → well calibrated.
    statistical_cal_tests: dict[str, object] = {}
    try:
        from mapie.metrics.calibration import (
            cumulative_differences,
            kolmogorov_smirnov_p_value,
            kuiper_p_value,
            length_scale,
            spiegelhalter_p_value,
        )

        y_test_arr = y_test.values.astype(float)
        # Calibrated (champion) vs uncalibrated
        for tag, probs in [("calibrated", y_prob_final), ("uncalibrated", y_prob_tuned_test)]:
            try:
                ks_p = float(kolmogorov_smirnov_p_value(y_test_arr, probs))
                ku_p = float(kuiper_p_value(y_test_arr, probs))
                sp_p = float(spiegelhalter_p_value(y_test_arr, probs))
                cum_diff = cumulative_differences(y_test_arr, probs)
                sigma = float(length_scale(probs))
                statistical_cal_tests[tag] = {
                    "ks_pvalue": ks_p,
                    "kuiper_pvalue": ku_p,
                    "spiegelhalter_pvalue": sp_p,
                    "length_scale_sigma": sigma,
                    "n": int(len(y_test_arr)),
                }
                logger.info(
                    f"Calibration tests [{tag}]: KS_p={ks_p:.4f} "
                    f"Kuiper_p={ku_p:.4f} Spiegelhalter_p={sp_p:.4f}"
                )
                # Save cumulative differences for the figure (calibrated only)
                if tag == "calibrated":
                    statistical_cal_tests["_cum_diff_calibrated"] = cum_diff.tolist()
                    statistical_cal_tests["_sigma"] = sigma
                elif tag == "uncalibrated":
                    statistical_cal_tests["_cum_diff_uncalibrated"] = cum_diff.tolist()
            except Exception as exc_inner:
                logger.warning(f"Statistical calibration tests [{tag}] failed: {exc_inner}")
                statistical_cal_tests[tag] = {"error": str(exc_inner)}

        # Save cumulative differences parquet for figure generation
        if "_cum_diff_calibrated" in statistical_cal_tests:
            k_idx = np.arange(len(y_test_arr)) / len(y_test_arr)
            sigma_val = float(statistical_cal_tests.get("_sigma", 0.0))
            cum_diff_df = pd.DataFrame(
                {
                    "k": k_idx,
                    "cum_diff_calibrated": statistical_cal_tests.pop("_cum_diff_calibrated"),
                    "cum_diff_uncalibrated": statistical_cal_tests.pop(
                        "_cum_diff_uncalibrated",
                        [float("nan")] * len(k_idx),
                    ),
                    "sigma_upper": sigma_val * 2,
                    "sigma_lower": -sigma_val * 2,
                }
            )
            cum_diff_path = _artifact_path("data/processed/calibration_cumulative_diffs.parquet")
            cum_diff_path.parent.mkdir(parents=True, exist_ok=True)
            cum_diff_df.to_parquet(cum_diff_path, index=False)
            logger.info(f"Saved calibration cumulative diffs: {cum_diff_path}")
            statistical_cal_tests.pop("_sigma", None)

        stat_cal_path = _artifact_path("data/processed/statistical_calibration_tests.json")
        stat_cal_path.parent.mkdir(parents=True, exist_ok=True)
        stat_cal_path.write_text(
            json.dumps(statistical_cal_tests, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"Saved statistical calibration tests: {stat_cal_path}")
    except ImportError:
        logger.warning(
            "mapie.metrics.calibration not available — statistical calibration tests skipped."
        )
    except Exception as exc:
        logger.warning(f"Statistical calibration tests block failed: {exc}")

    brier_decomposition_path = _artifact_path(
        config["output"].get(
            "brier_decomposition_path", "data/processed/brier_score_decomposition.json"
        )
    )
    _write_json(
        brier_decomposition_path,
        {
            "calibrated": calibrated_decomposition,
            "uncalibrated": raw_decomposition,
            "selected_calibration_method": selected_cal_method,
        },
    )

    murphy_diagram_path = _artifact_path(
        config["output"].get(
            "murphy_diagram_path", "reports/figures/calibration/murphy_diagram.png"
        )
    )
    try:
        import matplotlib.pyplot as plt

        ax = plot_murphy_diagram(
            y_test.values,
            {
                "CatBoost tuned raw": y_prob_tuned_test,
                f"CatBoost tuned calibrated ({selected_cal_method})": y_prob_final,
                "LogReg baseline": lr_model.predict_proba(X_test_lr)[:, 1],
            },
            title="Murphy Diagram: Raw vs Calibrated PD Forecasts",
        )
        ax.figure.tight_layout()
        murphy_diagram_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(murphy_diagram_path, dpi=180, bbox_inches="tight")
        plt.close(ax.figure)
        logger.info("Saved Murphy diagram to {}", murphy_diagram_path)
    except Exception as exc:
        logger.warning("Murphy diagram export skipped: {}", exc)

    # Persist models/calibrator
    model_path = _artifact_path(config["output"].get("model_path", "models/pd_catboost_tuned.cbm"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cb_tuned_model.save_model(str(model_path))

    default_model_path = _artifact_path(
        config["output"].get("default_model_path", "models/pd_catboost_default.cbm")
    )
    default_model_path.parent.mkdir(parents=True, exist_ok=True)
    cb_default_model.save_model(str(default_model_path))

    tuned_model_path = _artifact_path(
        config["output"].get("tuned_model_path", "models/pd_catboost_tuned.cbm")
    )
    tuned_model_path.parent.mkdir(parents=True, exist_ok=True)
    if tuned_model_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, tuned_model_path)

    # Optional legacy compatibility copy (disabled by default).
    write_legacy_model_copy = bool(config["output"].get("write_legacy_model_copy", False))
    legacy_model_path = _artifact_path("models/pd_catboost.cbm")
    if write_legacy_model_copy and legacy_model_path.resolve() != model_path.resolve():
        shutil.copy2(model_path, legacy_model_path)
        logger.info("Saved legacy model compatibility copy to {}", legacy_model_path)

    cal_path = _artifact_path(config["output"].get("conformal_path", "models/pd_calibrator.pkl"))
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    decision_threshold_path = _artifact_path(
        decision_cfg.get("output_path", "models/decision_threshold.json")
    )
    decision_threshold_v2_path = _artifact_path(
        decision_cfg.get("output_path_v2", "models/decision_threshold_v2.json")
    )
    decision_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decision_threshold_path, "w", encoding="utf-8") as f:
        json.dump(decision_threshold_artifact, f, indent=2, default=str)
    decision_threshold_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decision_threshold_v2_path, "w", encoding="utf-8") as f:
        json.dump(decision_threshold_artifact, f, indent=2, default=str)
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
            "calibration_method": selected_cal_method,
        },
    )

    logreg_model_path = _artifact_path("models/pd_logreg_baseline.pkl")
    logreg_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(logreg_model_path, "wb") as f:
        pickle.dump(
            {
                "model": lr_model,
                "feature_names": list(logreg_features),
                "fill_values": lr_fill.to_dict(),
            },
            f,
        )

    # Canonical artifacts for downstream loading.
    canonical_model_path = _artifact_path(CANONICAL_MODEL_PATH)
    canonical_calibrator_path = _artifact_path(CANONICAL_CALIBRATOR_PATH)
    contract_path = _artifact_path(CONTRACT_PATH)
    canonical_model_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() != canonical_model_path.resolve():
        shutil.copy2(model_path, canonical_model_path)
    if cal_path.resolve() != canonical_calibrator_path.resolve():
        shutil.copy2(cal_path, canonical_calibrator_path)

    # Persist model contract for strict feature alignment across scripts.
    features_contract, categorical_contract = infer_model_feature_contract(cb_tuned_model)
    split_shapes, split_missing = validate_features_in_splits(
        feature_names=features_contract,
        splits={"train": train, "calibration": cal, "test": test},
    )
    contract_payload = build_contract_payload(
        model_path=canonical_model_path,
        calibrator_path=canonical_calibrator_path,
        feature_names=features_contract,
        categorical_features=categorical_contract,
        split_shapes=split_shapes,
        split_missing_features=split_missing,
    )
    save_contract(contract_payload, contract_path)

    # ── SHAP feature importance export (CatBoost native) ──
    shap_artifact: dict[str, Any] = {"exported": False}
    try:
        from catboost import Pool as _SHAPPool

        shap_pool = _SHAPPool(X_test_cb, cat_features=categorical_features)
        shap_raw = cb_tuned_model.get_feature_importance(type="ShapValues", data=shap_pool)
        # ShapValues returns (n_samples, n_features + 1); last col = expected value
        shap_values = shap_raw[:, :-1]
        shap_expected = float(shap_raw[0, -1])

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = sorted(
            zip(catboost_features, mean_abs_shap.tolist(), strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        shap_dir = _artifact_path("reports/figures/shap")
        shap_dir.mkdir(parents=True, exist_ok=True)

        # Save raw SHAP values (compressed)
        np.savez_compressed(
            str(shap_dir / "shap_values_test.npz"),
            shap_values=shap_values,
            expected_value=np.array([shap_expected]),
            feature_names=np.array(catboost_features),
        )

        # Save top-N importance as JSON for Streamlit/governance
        top_n = min(20, len(shap_importance))
        shap_summary = {
            "expected_value": shap_expected,
            "n_samples": int(shap_values.shape[0]),
            "n_features": int(shap_values.shape[1]),
            "top_features": [
                {"feature": f, "mean_abs_shap": round(v, 6)} for f, v in shap_importance[:top_n]
            ],
        }
        shap_summary_path = shap_dir / "shap_feature_importance.json"
        shap_summary_path.write_text(json.dumps(shap_summary, indent=2), encoding="utf-8")
        shap_artifact = {"exported": True, "n_features": top_n, "path": str(shap_dir)}
        logger.info(
            "SHAP export: top feature={} (|SHAP|={:.4f}), saved to {}",
            shap_importance[0][0],
            shap_importance[0][1],
            shap_dir,
        )
    except Exception as exc:
        logger.warning("SHAP feature importance export skipped: {}", exc)
        shap_artifact["error"] = str(exc)

    # Persist test predictions for downstream contracts.
    test_predictions_path = _artifact_path("data/processed/test_predictions.parquet")
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
        "run_mode": run_mode,
        "replay_manifest_path": str(replay_manifest_path) if replay_manifest_path else None,
        "best_model": "CatBoost (tuned + calibrated)",
        "best_calibration": _human_calibration_name(selected_cal_method),
        "calibration_selection_report": cal_selection_report,
        "feature_source": feature_sets.get("feature_source", feature_mode),
        "feature_config_path": str(feature_config_path),
        "training_regime": regime_meta,
        "stable_core": stable_core_meta,
        "validation_scheme": val_cfg.get("scheme", "temporal_train_val_cal_test"),
        "dataset_scope": "full_data" if sample_size is None else "sampled",
        "sample_size": None if sample_size is None else int(sample_size),
        "feature_count_default": int(len(catboost_features)),
        "feature_count_tuned": int(len(catboost_features)),
        "logreg_feature_names": list(logreg_features),
        "logreg_coefficients": {
            feature: float(coef)
            for feature, coef in zip(
                logreg_features,
                np.asarray(getattr(lr_model, "coef_", np.zeros((1, len(logreg_features))))).ravel(),
                strict=False,
            )
        },
        "optuna_best_auc": float(cb_tuned_metrics.get("auc_roc", 0.0)),
        "optuna_best_params": cb_tuned_metrics.get("best_params", {}),
        "hpo_trials_executed": int(cb_tuned_metrics.get("hpo_trials_executed", 0)),
        "hpo_best_validation_auc": float(cb_tuned_metrics.get("hpo_best_validation_auc", 0.0)),
        "walk_forward_report": walk_forward_report,
        "seed_replay_report": seed_replay_report,
        "decision_threshold": decision_threshold_artifact,
        "baseline_metrics": lr_metrics,
        "catboost_default_metrics": cb_default_metrics,
        "catboost_tuned_metrics": cb_tuned_metrics,
        "catboost_tuned_raw_test_metrics": tuned_raw_test_metrics,
        "brier_decomposition_calibrated": calibrated_decomposition,
        "brier_decomposition_uncalibrated": raw_decomposition,
        "calibration_metrics": cal_metrics,
        "conformal_metrics": cp_metrics,
        "final_test_metrics": final_test_metrics,
        "shap_export": shap_artifact,
        "murphy_diagram_path": str(murphy_diagram_path),
    }

    record_path = _artifact_path("models/pd_training_record.pkl")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "wb") as f:
        pickle.dump(training_record, f)
    if seed_replay_report:
        seed_replay_status_path = _artifact_path("models/pd_hpo_seed_replay_status.json")
        seed_replay_status = {
            "selected_calibration_method": selected_cal_method,
            "validation_auc": float(cb_tuned_metrics.get("hpo_best_validation_auc", 0.0)),
            "oot_auc": float(final_test_metrics.get("auc_roc", 0.0)),
            "brier": float(final_test_metrics.get("brier_score", 0.0)),
            "ece": float(final_test_metrics.get("ece", 0.0)),
            "replay": seed_replay_report,
            **build_artifact_metadata(
                schema_version="2026-03-13.1",
                run_tag=resolved_run_tag,
                require_explicit=True,
            ),
        }
        seed_replay_status_path.write_text(
            json.dumps(seed_replay_status, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved PD HPO seed replay status to {}", seed_replay_status_path)

    logger.info("Saved default model to {}", default_model_path)
    logger.info("Saved tuned model to {}", model_path)
    logger.info("Saved LR baseline to {}", logreg_model_path)
    logger.info("Saved calibrator to {}", cal_path)
    logger.info("Saved decision threshold artifact to {}", decision_threshold_path)
    logger.info("Saved decision threshold v2 artifact to {}", decision_threshold_v2_path)
    logger.info("Saved canonical model to {}", canonical_model_path)
    logger.info("Saved canonical calibrator to {}", canonical_calibrator_path)
    logger.info("Saved PD contract to {}", contract_path)
    logger.info("Saved test predictions to {}", test_predictions_path)
    logger.info("Saved training record to {}", record_path)
    _write_training_status(
        status_path,
        phase="artifacts_saved",
        state="completed",
        config_path=config_path,
        extra={
            "model_path": str(model_path),
            "calibrator_path": str(cal_path),
            "record_path": str(record_path),
            "auc_roc": float(final_test_metrics.get("auc_roc", 0.0)),
            "brier_score": float(final_test_metrics.get("brier_score", 0.0)),
        },
    )
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
    parser.add_argument("--training_regime_mode", default=None)
    parser.add_argument("--recent_window_quarters", type=int, default=None)
    parser.add_argument("--half_life_quarters", type=int, default=None)
    parser.add_argument(
        "--stable_core_enabled",
        choices=["true", "false"],
        default=None,
    )
    parser.add_argument("--hpo_n_trials", type=int, default=None)
    parser.add_argument(
        "--hpo_enabled",
        choices=["true", "false"],
        default=None,
    )
    parser.add_argument(
        "--challenger_enabled",
        choices=["true", "false"],
        default=None,
    )
    parser.add_argument(
        "--walk_forward_enabled",
        choices=["true", "false"],
        default=None,
    )
    parser.add_argument(
        "--seed_replay_enabled",
        choices=["true", "false"],
        default=None,
    )
    parser.add_argument("--catboost_iterations", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mode", choices=["search", "replay"], default="search")
    parser.add_argument("--replay_manifest", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.sample_size,
        training_regime_mode=args.training_regime_mode,
        recent_window_quarters=args.recent_window_quarters,
        half_life_quarters=args.half_life_quarters,
        stable_core_enabled=(
            None if args.stable_core_enabled is None else args.stable_core_enabled.lower() == "true"
        ),
        hpo_n_trials=args.hpo_n_trials,
        hpo_enabled=None if args.hpo_enabled is None else args.hpo_enabled.lower() == "true",
        challenger_enabled=(
            None if args.challenger_enabled is None else args.challenger_enabled.lower() == "true"
        ),
        walk_forward_enabled=(
            None
            if args.walk_forward_enabled is None
            else args.walk_forward_enabled.lower() == "true"
        ),
        seed_replay_enabled=(
            None if args.seed_replay_enabled is None else args.seed_replay_enabled.lower() == "true"
        ),
        catboost_iterations=args.catboost_iterations,
        validate_only=bool(args.validate_only),
        mode=str(args.mode),
        replay_manifest_path=args.replay_manifest,
    )
