"""Run post-promotion PD backtesting diagnostics inspired by ADSFCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from loguru import logger

from src.evaluation.backtesting import (
    hosmer_lemeshow_test,
    jeffreys_interval,
    normal_approximation_backtest,
    two_sided_exact_binomial_test,
)
from src.models.pd_contract import load_contract, resolve_calibrator_path, resolve_model_path
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag
from src.utils.io_utils import load_pickle_compat, read_split_with_fe_fallback

SCHEMA_VERSION = "2026-03-30.1"


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_calibrator() -> Any | None:
    path = resolve_calibrator_path()
    if path is None or not path.exists():
        return None
    try:
        return load_pickle_compat(path)
    except Exception:
        return None


def _apply_calibrator(calibrator: Any | None, y_prob_raw: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(y_prob_raw, dtype=float)
    if hasattr(calibrator, "predict_proba"):
        calibrated = calibrator.predict_proba(np.asarray(y_prob_raw, dtype=float).reshape(-1, 1))
        if getattr(calibrated, "ndim", 0) == 2 and calibrated.shape[1] >= 2:
            return np.asarray(calibrated[:, 1], dtype=float)
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(np.asarray(y_prob_raw, dtype=float)), dtype=float)
    return np.asarray(y_prob_raw, dtype=float)


def _prepare_model_frame(
    df: pd.DataFrame,
    *,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    X = df.reindex(columns=features).copy()
    categorical = {col for col in X.columns if col in categorical_features}
    for col in categorical:
        X[col] = X[col].astype("string").fillna("UNKNOWN").astype(str)
    for col in X.columns:
        if col not in categorical:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


class _CatBoostPredictorAdapter:
    def __init__(self, raw_model: CatBoostClassifier, cat_features: list[str]) -> None:
        self.raw_model = raw_model
        self.cat_features = list(cat_features)

    def predict_proba(self, X_input: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.raw_model.predict_proba(
                Pool(X_input, cat_features=[c for c in self.cat_features if c in X_input.columns])
            ),
            dtype=float,
        )


def _load_model_and_matrix(test_df: pd.DataFrame) -> tuple[_CatBoostPredictorAdapter, pd.DataFrame]:
    raw_model = CatBoostClassifier()
    raw_model.load_model(str(resolve_model_path()))
    contract = load_contract() or {}
    feature_names = [
        str(f)
        for f in (
            contract.get("feature_names", []) or getattr(raw_model, "feature_names_", []) or []
        )
        if f in test_df.columns
    ]
    if not feature_names:
        raise ValueError("PD model contract missing usable feature_names for PD backtesting.")
    model_feature_names = [str(f) for f in getattr(raw_model, "feature_names_", []) or []]
    cat_indices = set(raw_model.get_cat_feature_indices())
    categorical = [
        model_feature_names[idx]
        for idx in sorted(cat_indices)
        if idx < len(model_feature_names) and model_feature_names[idx] in feature_names
    ]
    X = _prepare_model_frame(test_df, features=feature_names, categorical_features=categorical)
    return _CatBoostPredictorAdapter(raw_model, categorical), X


def _row_backtests(n_obs: int, n_defaults: int, pd_ref: float) -> dict[str, float | bool]:
    exact = two_sided_exact_binomial_test(n_defaults=n_defaults, n_obs=n_obs, pd_ref=pd_ref)
    z_test = normal_approximation_backtest(n_defaults=n_defaults, n_obs=n_obs, pd_ref=pd_ref)
    jeffreys = jeffreys_interval(n_defaults=n_defaults, n_obs=n_obs)
    observed_rate = float(n_defaults / max(n_obs, 1))
    return {
        "n_obs": int(n_obs),
        "n_defaults": int(n_defaults),
        "observed_default_rate": observed_rate,
        "mean_predicted_pd": float(pd_ref),
        "exact_binomial_p_value": float(exact["p_value"]),
        "z_score": float(z_test["z_score"]),
        "z_test_p_value": float(z_test["p_value"]),
        "jeffreys_lower": float(jeffreys["lower"]),
        "jeffreys_upper": float(jeffreys["upper"]),
        "predicted_pd_inside_jeffreys": bool(
            float(jeffreys["lower"]) <= float(pd_ref) <= float(jeffreys["upper"])
        ),
    }


def main(config_path: str = "configs/pd_model.champion.yaml", run_tag: str | None = None) -> None:
    cfg = _load_cfg(config_path)
    data_cfg = dict(cfg.get("data") or {})
    test_path = str(data_cfg.get("test_path", "data/processed/test_fe.parquet"))
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[resolve_official_baseline_run_tag()],
        require_explicit=True,
    )

    test_df = read_split_with_fe_fallback(test_path)
    if "default_flag" not in test_df.columns:
        raise KeyError("Missing 'default_flag' in test split for PD backtesting suite.")

    model, X = _load_model_and_matrix(test_df)
    raw_scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    calibrated_scores = _apply_calibrator(_load_calibrator(), raw_scores)
    y_true = test_df["default_flag"].to_numpy(dtype=int)

    overall = _row_backtests(
        n_obs=int(len(test_df)),
        n_defaults=int(np.sum(y_true)),
        pd_ref=float(np.mean(calibrated_scores)),
    )
    hl = hosmer_lemeshow_test(y_true, calibrated_scores, n_groups=10)
    overall.update(
        {
            "hl_statistic": float(hl["hl_statistic"]),
            "hl_p_value": float(hl["hl_p_value"]),
            "hl_n_groups": int(hl["n_groups"]),
        }
    )

    by_grade = []
    if "grade" in test_df.columns:
        for grade, grp in test_df.assign(_pd=calibrated_scores).groupby("grade", observed=True):
            n_obs = int(len(grp))
            if n_obs < 200:
                continue
            metrics = _row_backtests(
                n_obs=n_obs,
                n_defaults=int(grp["default_flag"].sum()),
                pd_ref=float(grp["_pd"].mean()),
            )
            metrics["grade"] = str(grade)
            by_grade.append(metrics)
    by_grade_df = (
        pd.DataFrame(by_grade).sort_values("grade").reset_index(drop=True)
        if by_grade
        else pd.DataFrame()
    )

    band_df = pd.DataFrame({"default_flag": y_true, "pd_score": calibrated_scores})
    band_df["band"] = pd.qcut(band_df["pd_score"], q=10, labels=False, duplicates="drop")
    band_summary = (
        band_df.groupby("band", observed=True)
        .agg(
            n_obs=("default_flag", "size"),
            n_defaults=("default_flag", "sum"),
            observed_default_rate=("default_flag", "mean"),
            mean_predicted_pd=("pd_score", "mean"),
        )
        .reset_index()
    )
    if not band_summary.empty:
        band_summary["band"] = band_summary["band"].astype(int) + 1
        band_summary["rate_gap"] = (
            band_summary["observed_default_rate"] - band_summary["mean_predicted_pd"]
        ).astype(float)

    data_dir = Path("data/processed")
    model_dir = Path("models")
    by_grade_path = data_dir / "pd_backtesting_by_grade.parquet"
    by_band_path = data_dir / "pd_backtesting_by_band.parquet"
    status_path = model_dir / "pd_backtesting_status.json"
    by_grade_path.parent.mkdir(parents=True, exist_ok=True)
    by_grade_df.to_parquet(by_grade_path, index=False)
    band_summary.to_parquet(by_band_path, index=False)

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(
            overall["exact_binomial_p_value"] >= 0.05
            and overall["predicted_pd_inside_jeffreys"]
            and overall["hl_p_value"] >= 0.01
        ),
        "summary": {
            **overall,
            "n_grade_rows": int(len(by_grade_df)),
            "n_band_rows": int(len(band_summary)),
        },
        "grade_rows": by_grade_df.to_dict(orient="records"),
        "artifacts": {
            "by_grade_path": str(by_grade_path),
            "by_band_path": str(by_band_path),
        },
        "config_path": str(config_path),
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "PD backtesting suite saved: {} (exact_p={:.4f}, hl_p={:.4f})",
        status_path,
        overall["exact_binomial_p_value"],
        overall["hl_p_value"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADSFCR-inspired PD backtesting suite")
    parser.add_argument("--config", default="configs/pd_model.champion.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
