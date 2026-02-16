"""Model inference service."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from api.dependencies import (
    get_calibrator,
    get_conformal_results,
    get_feature_config,
    get_pd_model,
)


def _risk_category(pd_val: float) -> str:
    if pd_val < 0.05:
        return "Low Risk"
    if pd_val < 0.15:
        return "Medium Risk"
    if pd_val < 0.30:
        return "High Risk"
    return "Very High Risk"


def _select_conformal_profile(alpha: float) -> tuple[str, float]:
    """Map requested alpha to available artifact profiles."""
    # Available artifact levels are 90% and 95% from the Mondrian pipeline.
    if alpha >= 0.10:
        return "diag_90", 0.90
    return "diag_95", 0.95


def _interval_from_mondrian_artifact(
    pd_point: float,
    grade: str,
    diag: dict,
) -> tuple[float, float, float]:
    """Compute [low, high] interval using stored Mondrian quantiles."""
    quantiles = diag.get("group_quantiles", {}) or {}
    global_q = float(diag.get("global_quantile", 0.0))
    q = float(quantiles.get(grade, global_q))

    if bool(diag.get("scaled_scores", False)):
        scale = float(np.sqrt(np.clip(pd_point * (1.0 - pd_point), 1e-6, None)))
    else:
        scale = 1.0

    radius = max(0.0, q * scale)
    pd_low = max(0.0, pd_point - radius)
    pd_high = min(1.0, pd_point + radius)
    return pd_low, pd_high, pd_high - pd_low


def predict_single(loan_data: dict) -> dict:
    """Generate PD prediction for a single loan."""
    model = get_pd_model()
    calibrator = get_calibrator()
    cfg = get_feature_config()

    features = cfg["CATBOOST_FEATURES"]
    cat_features = cfg["CATEGORICAL_FEATURES"]

    # Build feature DataFrame
    df = pd.DataFrame([loan_data])

    # Auto-compute derived features
    if "loan_to_income" not in df.columns or df["loan_to_income"].isna().all():
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1)

    # Ensure all required features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0 if f not in cat_features else "unknown"

    X = df[features].copy()

    # Convert categoricals to string for CatBoost
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype(str)

    # Raw prediction
    pd_raw = float(model.predict_proba(X)[0, 1])

    # Calibrated prediction
    pd_calibrated = float(calibrator.predict(np.array([[pd_raw]]))[0])
    pd_calibrated = float(np.clip(pd_calibrated, 0.001, 0.999))

    ead = loan_data.get("loan_amnt", 10000) * 0.80
    lgd = 0.45
    el = pd_calibrated * lgd * ead

    return {
        "pd_point": round(pd_calibrated, 6),
        "pd_raw": round(pd_raw, 6),
        "lgd_estimate": lgd,
        "ead_estimate": round(ead, 2),
        "expected_loss": round(el, 2),
        "risk_category": _risk_category(pd_calibrated),
    }


def predict_conformal(loan_data: dict, alpha: float = 0.10) -> dict:
    """Generate PD prediction with conformal intervals."""
    base = predict_single(loan_data)
    grade = str(loan_data.get("grade", "C")).strip().upper()
    pd_point = float(base["pd_point"])
    diag_key, confidence_level = _select_conformal_profile(alpha)

    conformal_payload = get_conformal_results()
    diag = conformal_payload.get(diag_key, {}) if isinstance(conformal_payload, dict) else {}

    if isinstance(diag, dict) and "global_quantile" in diag:
        pd_low, pd_high, width = _interval_from_mondrian_artifact(
            pd_point=pd_point,
            grade=grade,
            diag=diag,
        )
    else:
        # Safe fallback if conformal artifact is unavailable/corrupt.
        logger.warning("Using heuristic conformal widths fallback.")
        grade_widths_90 = {
            "A": 0.30, "B": 0.50, "C": 0.65,
            "D": 0.80, "E": 0.90, "F": 0.95, "G": 1.00,
        }
        grade_widths_95 = {k: min(v * 1.25, 1.0) for k, v in grade_widths_90.items()}
        width_map = grade_widths_90 if alpha >= 0.10 else grade_widths_95
        width = float(width_map.get(grade, 0.70))
        pd_low = max(0.0, pd_point - width / 2)
        pd_high = min(1.0, pd_point + width / 2)

    ead = base["ead_estimate"]
    lgd = base["lgd_estimate"]
    el_conservative = pd_high * lgd * ead

    return {
        **base,
        "pd_low": round(pd_low, 6),
        "pd_high": round(pd_high, 6),
        "interval_width": round(width, 4),
        "confidence_level": confidence_level,
        "expected_loss_conservative": round(el_conservative, 2),
    }
