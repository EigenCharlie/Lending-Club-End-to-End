"""Run post-promotion monotonicity diagnostics for the canonical PD champion.

Outputs:
- data/processed/monotonicity_band_summary.parquet
- data/processed/monotonicity_pair_report.parquet
- data/processed/monotonicity_feature_report.parquet
- models/monotonicity_audit_status.json
"""

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

from src.evaluation.explainability import monotonic_violation_rate
from src.evaluation.monotonicity import (
    adjacent_monotonicity_report,
    monotonicity_status,
    pd_band_summary,
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
        raise ValueError("PD model contract missing usable feature_names for monotonicity audit.")
    model_feature_names = [str(f) for f in getattr(raw_model, "feature_names_", []) or []]
    cat_indices = set(raw_model.get_cat_feature_indices())
    categorical = [
        model_feature_names[idx]
        for idx in sorted(cat_indices)
        if idx < len(model_feature_names) and model_feature_names[idx] in feature_names
    ]
    X = _prepare_model_frame(test_df, features=feature_names, categorical_features=categorical)
    return _CatBoostPredictorAdapter(raw_model, categorical), X


def _parse_monotonic_constraints(cfg: dict[str, Any]) -> dict[str, int]:
    params = dict((cfg.get("model") or {}).get("params") or {})
    raw = params.get("monotone_constraints")
    constraints: dict[str, int] = {}
    if isinstance(raw, str):
        for part in raw.split(","):
            item = str(part).strip()
            if not item or ":" not in item:
                continue
            name, direction = item.split(":", 1)
            try:
                constraints[str(name).strip()] = int(direction)
            except ValueError:
                continue
    if constraints:
        return constraints
    challenger_constraints = dict(
        (cfg.get("challenger_pipeline") or {}).get("monotonic_constraints") or {}
    )
    return {str(k): int(v) for k, v in challenger_constraints.items()}


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
        raise KeyError("Missing 'default_flag' in test split for monotonicity audit.")

    model, X = _load_model_and_matrix(test_df)
    raw_scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    calibrated_scores = _apply_calibrator(_load_calibrator(), raw_scores)

    band_summary = pd_band_summary(test_df["default_flag"].to_numpy(dtype=float), calibrated_scores)
    pair_report = adjacent_monotonicity_report(band_summary)
    status_summary = monotonicity_status(band_summary, pair_report)

    constraints = _parse_monotonic_constraints(cfg)
    feature_rows: list[dict[str, float | str | int]] = []
    for feature, direction in constraints.items():
        if feature not in X.columns or int(direction) == 0:
            continue
        violation = monotonic_violation_rate(
            model,
            X,
            feature,
            int(direction),
            grid_size=7,
            sample_size=256,
            random_state=42,
        )
        feature_rows.append(
            {
                "feature": str(feature),
                "direction": int(direction),
                "violation_rate": float(violation),
            }
        )

    feature_report = (
        pd.DataFrame(feature_rows)
        .sort_values("violation_rate", ascending=False)
        .reset_index(drop=True)
        if feature_rows
        else pd.DataFrame(columns=["feature", "direction", "violation_rate"])
    )
    max_feature_violation = (
        float(feature_report["violation_rate"].max()) if not feature_report.empty else 0.0
    )
    mean_feature_violation = (
        float(feature_report["violation_rate"].mean()) if not feature_report.empty else 0.0
    )

    data_dir = Path("data/processed")
    model_dir = Path("models")
    band_path = data_dir / "monotonicity_band_summary.parquet"
    pair_path = data_dir / "monotonicity_pair_report.parquet"
    feature_path = data_dir / "monotonicity_feature_report.parquet"
    status_path = model_dir / "monotonicity_audit_status.json"
    for path in [band_path, pair_path, feature_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    band_summary.to_parquet(band_path, index=False)
    pair_report.to_parquet(pair_path, index=False)
    feature_report.to_parquet(feature_path, index=False)

    payload = {
        "diagnostic_only": True,
        "overall_pass": bool(
            status_summary.get("overall_pass", False) and max_feature_violation <= 0.05
        ),
        "summary": {
            **status_summary,
            "n_constrained_features": int(len(feature_report)),
            "mean_feature_violation_rate": mean_feature_violation,
            "max_feature_violation_rate": max_feature_violation,
            "mean_predicted_pd": float(np.mean(calibrated_scores))
            if calibrated_scores.size
            else 0.0,
            "observed_default_rate": float(np.mean(test_df["default_flag"]))
            if len(test_df)
            else 0.0,
        },
        "top_feature_violations": feature_report.head(10).to_dict(orient="records"),
        "artifacts": {
            "band_summary_path": str(band_path),
            "pair_report_path": str(pair_path),
            "feature_report_path": str(feature_path),
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
        "Monotonicity audit saved: {} (disruptions={}, max_feature_violation={:.4f})",
        status_path,
        payload["summary"]["n_disruptions"],
        max_feature_violation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monotonicity diagnostics for canonical PD")
    parser.add_argument("--config", default="configs/pd_model.champion.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
