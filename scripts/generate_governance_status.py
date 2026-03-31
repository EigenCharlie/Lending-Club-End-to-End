"""Generate governance drift status for MRM gating.

Builds per-feature drift diagnostics (PSI, KS, CvM) and multivariate C2ST,
then emits:
- data/processed/drift_monitoring.parquet
- models/governance_status.json

Usage:
    uv run python scripts/generate_governance_status.py --config configs/mrm_policy.yaml
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
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_pd_model import _apply_training_regime
from src.evaluation.backtesting import (
    classifier_two_sample_test,
    drift_monitoring_report,
    population_stability_index,
)
from src.evaluation.explainability import dominant_reason_match_rate, rank_overlap_ratio
from src.evaluation.model_shift import interpret_model_shift
from src.models.pd_contract import load_contract, resolve_calibrator_path, resolve_model_path
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag
from src.utils.baseline_registry import resolve_official_baseline_run_tag
from src.utils.io_utils import load_pickle_compat, read_split_with_fe_fallback
from src.utils.threshold_semantics import load_threshold_semantics, resolve_operational_threshold

SCHEMA_VERSION = "2026-03-06.1"


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_feature_contract() -> list[str]:
    contract_path = Path("models/pd_model_contract.json")
    if not contract_path.exists():
        return []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    features = contract.get("feature_names", [])
    if not isinstance(features, list):
        return []
    return [str(f) for f in features]


def _load_training_record() -> dict[str, Any]:
    path = Path("models/pd_training_record.pkl")
    if not path.exists():
        return {}
    try:
        payload = load_pickle_compat(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sample_frame(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


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
        return y_prob_raw.astype(float)
    if hasattr(calibrator, "predict_proba"):
        calibrated = calibrator.predict_proba(y_prob_raw.reshape(-1, 1))
        if calibrated.ndim == 2 and calibrated.shape[1] >= 2:
            return calibrated[:, 1].astype(float)
    if hasattr(calibrator, "predict"):
        calibrated = calibrator.predict(y_prob_raw)
        return np.asarray(calibrated, dtype=float)
    return y_prob_raw.astype(float)


def _prepare_model_frame(
    df: pd.DataFrame,
    *,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    X = df.reindex(columns=features).copy()
    inferred_cat = [
        col
        for col in X.columns
        if col in categorical_features or not pd.api.types.is_numeric_dtype(X[col])
    ]
    for col in inferred_cat:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("missing")
    return X


def _score_and_performance_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    random_state: int,
    max_rows_per_split: int,
    psi_bins: int,
) -> dict[str, float]:
    contract = load_contract()
    if not contract:
        return {}

    feature_names = [str(f) for f in contract.get("feature_names", [])]
    categorical_features = [str(f) for f in contract.get("categorical_features", [])]
    if not feature_names:
        return {}

    model = CatBoostClassifier()
    model.load_model(resolve_model_path().as_posix())
    calibrator = _load_calibrator()

    train_eval = _sample_frame(train_df, max_rows=max_rows_per_split, random_state=random_state)
    test_eval = _sample_frame(test_df, max_rows=max_rows_per_split, random_state=random_state)

    X_train = _prepare_model_frame(
        train_eval,
        features=feature_names,
        categorical_features=categorical_features,
    )
    X_test = _prepare_model_frame(
        test_eval,
        features=feature_names,
        categorical_features=categorical_features,
    )
    y_train = (
        pd.to_numeric(train_eval["default_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    )
    y_test = (
        pd.to_numeric(test_eval["default_flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    )

    cat_cols = [
        col
        for col in X_train.columns
        if col in categorical_features or not pd.api.types.is_numeric_dtype(X_train[col])
    ]
    train_pool = Pool(X_train, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    train_raw = model.predict_proba(train_pool)[:, 1]
    test_raw = model.predict_proba(test_pool)[:, 1]
    train_score = _apply_calibrator(calibrator, train_raw)
    test_score = _apply_calibrator(calibrator, test_raw)

    train_auc = float(roc_auc_score(y_train, train_score))
    test_auc = float(roc_auc_score(y_test, test_score))
    train_brier = float(brier_score_loss(y_train, train_score))
    test_brier = float(brier_score_loss(y_test, test_score))
    train_cal_gap = float(abs(float(np.mean(train_score)) - float(np.mean(y_train))))
    test_cal_gap = float(abs(float(np.mean(test_score)) - float(np.mean(y_test))))
    score_psi = float(
        population_stability_index(
            np.asarray(train_score, dtype=float),
            np.asarray(test_score, dtype=float),
            n_bins=psi_bins,
        )
    )
    return {
        "score_psi": score_psi,
        "auc_train_reference": train_auc,
        "auc_test_oot": test_auc,
        "auc_delta_train_to_test": float(max(train_auc - test_auc, 0.0)),
        "brier_train_reference": train_brier,
        "brier_test_oot": test_brier,
        "brier_increase_train_to_test": float(max(test_brier - train_brier, 0.0)),
        "calibration_gap_train_reference": train_cal_gap,
        "calibration_gap_test_oot": test_cal_gap,
        "calibration_gap_delta": float(max(test_cal_gap - train_cal_gap, 0.0)),
        "train_eval_rows": int(len(train_eval)),
        "test_eval_rows": int(len(test_eval)),
    }


def _resolve_numeric_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    contract_features = _load_feature_contract()
    common = [f for f in contract_features if f in train_df.columns and f in test_df.columns]

    numeric = []
    for f in common:
        if pd.api.types.is_numeric_dtype(train_df[f]) or pd.api.types.is_numeric_dtype(test_df[f]):
            numeric.append(f)

    if numeric:
        return numeric

    # Fallback: numeric intersection from both frames.
    train_num = set(train_df.select_dtypes(include=["number"]).columns)
    test_num = set(test_df.select_dtypes(include=["number"]).columns)
    fallback = sorted(train_num.intersection(test_num))
    return [c for c in fallback if c != "default_flag"][:80]


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _resolve_primary_threshold() -> float:
    semantics = load_threshold_semantics()
    if semantics:
        return resolve_operational_threshold(semantics, default=0.5)
    fairness_status_path = Path("models/fairness_audit_status.json")
    if fairness_status_path.exists():
        try:
            payload = json.loads(fairness_status_path.read_text(encoding="utf-8"))
            return float(payload.get("primary_threshold", payload.get("prediction_threshold", 0.5)))
        except Exception:
            pass
    decision_threshold_path = Path("models/decision_threshold.json")
    if decision_threshold_path.exists():
        try:
            payload = json.loads(decision_threshold_path.read_text(encoding="utf-8"))
            return float(payload.get("selected_threshold", 0.5))
        except Exception:
            pass
    return 0.5


def _build_explanation_drift_report(
    shap_raw: pd.DataFrame,
    *,
    primary_threshold: float,
    min_rank_overlap_top10: float,
    max_shap_psi: float,
    min_reason_code_stability: float,
    min_rows_per_slice: int,
) -> pd.DataFrame:
    feature_cols = [c.replace("shap_", "") for c in shap_raw.columns if c.startswith("shap_")]
    if shap_raw.empty or not feature_cols or "issue_quarter" not in shap_raw.columns:
        return pd.DataFrame()

    periods = sorted(
        [
            p
            for p in shap_raw["issue_quarter"].astype(str).dropna().unique().tolist()
            if p != "unknown"
        ]
    )
    if len(periods) < 2:
        return pd.DataFrame()

    comparison_periods: list[str] = []
    comparison_df = pd.DataFrame()
    for period in reversed(periods):
        comparison_periods.insert(0, period)
        comparison_df = shap_raw.loc[
            shap_raw["issue_quarter"].astype(str).isin(comparison_periods)
        ].copy()
        if len(comparison_df) >= min_rows_per_slice:
            break
    if len(comparison_df) < min_rows_per_slice:
        return pd.DataFrame()

    reference_df = shap_raw.loc[
        ~shap_raw["issue_quarter"].astype(str).isin(comparison_periods)
    ].copy()
    if len(reference_df) < min_rows_per_slice or len(comparison_df) < min_rows_per_slice:
        return pd.DataFrame()
    comparison_period_label = "|".join(comparison_periods)

    segment_pairs: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = [
        ("overall", "all", reference_df, comparison_df)
    ]
    if "grade" in shap_raw.columns:
        for grade in sorted(shap_raw["grade"].dropna().astype(str).unique().tolist()):
            ref_seg = reference_df.loc[reference_df["grade"].astype(str) == grade].copy()
            cmp_seg = comparison_df.loc[comparison_df["grade"].astype(str) == grade].copy()
            if len(ref_seg) < min_rows_per_slice or len(cmp_seg) < min_rows_per_slice:
                continue
            segment_pairs.append(("grade", grade, ref_seg, cmp_seg))

    rows: list[dict[str, Any]] = []
    for segment_type, segment, ref_seg, cmp_seg in segment_pairs:
        ref_ranking = sorted(
            feature_cols,
            key=lambda feature: ref_seg[f"shap_{feature}"].abs().mean(),
            reverse=True,
        )
        cmp_ranking = sorted(
            feature_cols,
            key=lambda feature: cmp_seg[f"shap_{feature}"].abs().mean(),
            reverse=True,
        )
        overlap = rank_overlap_ratio(ref_ranking, cmp_ranking, top_k=10)
        focus_features = list(dict.fromkeys(ref_ranking[:5] + cmp_ranking[:5]))[:5]
        shap_psis = []
        for feature in focus_features:
            col = f"shap_{feature}"
            if col not in ref_seg.columns or col not in cmp_seg.columns:
                continue
            psi = population_stability_index(
                pd.to_numeric(ref_seg[col], errors="coerce").dropna().to_numpy(dtype=float),
                pd.to_numeric(cmp_seg[col], errors="coerce").dropna().to_numpy(dtype=float),
                n_bins=8,
            )
            shap_psis.append({"feature": feature, "psi": float(psi)})

        max_feature_psi = max((row["psi"] for row in shap_psis), default=0.0)
        avg_feature_psi = float(np.mean([row["psi"] for row in shap_psis])) if shap_psis else 0.0
        reason_match_rate, reason_details = dominant_reason_match_rate(
            ref_seg,
            cmp_seg,
            ref_ranking[:10],
            pd_col="pd_calibrated" if "pd_calibrated" in shap_raw.columns else "score_raw",
            threshold=primary_threshold,
            min_rows_per_band=max(15, int(min_rows_per_slice / 4)),
        )
        pass_rank = bool(overlap >= min_rank_overlap_top10)
        pass_dist = bool(max_feature_psi <= max_shap_psi)
        pass_reason = bool(reason_match_rate >= min_reason_code_stability)
        rows.append(
            {
                "segment_type": segment_type,
                "segment": segment,
                "reference_period": "|".join([p for p in periods if p not in comparison_periods]),
                "comparison_period": comparison_period_label,
                "reference_n": int(len(ref_seg)),
                "comparison_n": int(len(cmp_seg)),
                "rank_overlap_top10": float(overlap),
                "avg_shap_psi_top5": float(avg_feature_psi),
                "max_shap_psi_top5": float(max_feature_psi),
                "reason_code_match_rate": float(reason_match_rate),
                "pass_rank_overlap": pass_rank,
                "pass_distribution_shift": pass_dist,
                "pass_reason_code_stability": pass_reason,
                "passed_all": bool(pass_rank and pass_dist and pass_reason),
                "feature_psi_details": json.dumps(shap_psis, default=str),
                "reason_code_details": json.dumps(reason_details, default=str),
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str = "configs/mrm_policy.yaml", run_tag: str | None = None) -> None:
    cfg = _load_cfg(config_path)
    semantics = load_threshold_semantics()
    resolved_run_tag = resolve_run_tag(
        run_tag,
        fallback_candidates=[semantics.get("run_tag"), resolve_official_baseline_run_tag()],
        require_explicit=True,
    )

    triggers = cfg.get("retraining_triggers", {})
    checks = cfg.get("governance_checks", {})
    outputs = cfg.get("governance_output", {})

    psi_threshold = float(triggers.get("psi_threshold", 0.25))
    ks_pvalue_min = float(checks.get("ks_pvalue_min", 0.01))
    cvm_pvalue_min = float(checks.get("cvm_pvalue_min", 0.01))
    c2st_auc_max = float(checks.get("c2st_auc_max", 0.60))
    max_feature_breach_ratio = float(checks.get("max_feature_breach_ratio", 0.15))
    c2st_max_rows = int(checks.get("c2st_max_rows_per_split", 50_000))
    score_psi_max = float(checks.get("score_psi_max", 0.15))
    auc_delta_max = float(checks.get("auc_delta_max", 0.05))
    brier_increase_max = float(checks.get("brier_increase_max", 0.02))
    calibration_gap_delta_max = float(checks.get("calibration_gap_delta_max", 0.02))
    performance_max_rows = int(checks.get("performance_max_rows_per_split", 100_000))
    min_rank_overlap_top10 = float(checks.get("explanation_rank_overlap_top10_min", 0.60))
    max_explanation_shap_psi = float(checks.get("explanation_shap_psi_max", 0.25))
    min_reason_code_stability = float(checks.get("reason_code_stability_min", 0.55))
    explanation_min_rows_per_slice = int(checks.get("explanation_min_rows_per_slice", 80))

    drift_path = Path(
        outputs.get("drift_monitoring_path", "data/processed/drift_monitoring.parquet")
    )
    status_path = Path(outputs.get("governance_status_path", "models/governance_status.json"))
    explanation_drift_path = Path(
        outputs.get("explanation_drift_path", "data/processed/explanation_drift.parquet")
    )
    fairness_status_path = Path(
        outputs.get("fairness_status_path", "models/fairness_audit_status.json")
    )
    fairness_frontier_path = Path(
        outputs.get(
            "fairness_frontier_path",
            "data/processed/fairness_threshold_frontier.parquet",
        )
    )
    challenger_report_path = Path(
        outputs.get("challenger_promotion_report_path", "models/challenger_promotion_report.json")
    )
    model_shift_status_path = Path(
        outputs.get("model_shift_status_path", "models/model_shift_status.json")
    )

    train_df = read_split_with_fe_fallback("data/processed/train_fe.parquet")
    test_df = read_split_with_fe_fallback("data/processed/test_fe.parquet")
    training_record = _load_training_record()
    regime_cfg = training_record.get("training_regime", {}) if training_record else {}
    if isinstance(regime_cfg, dict) and regime_cfg:
        train_df, regime_meta = _apply_training_regime(train_df, regime_cfg, date_col="issue_d")
        logger.info(
            "Applied PD training regime to governance reference population: mode={} rows={}",
            regime_meta.get("mode", regime_cfg.get("mode", "standard")),
            len(train_df),
        )

    features = _resolve_numeric_features(train_df, test_df)
    if not features:
        raise ValueError("No numeric features available for governance drift checks.")

    logger.info("Governance drift checks on {} numeric features", len(features))

    drift_df = drift_monitoring_report(
        train_df=train_df,
        test_df=test_df,
        features=features,
        psi_threshold=psi_threshold,
        ks_pvalue_threshold=ks_pvalue_min,
        cvm_pvalue_threshold=cvm_pvalue_min,
        n_bins=int(checks.get("psi_bins", 10)),
    )

    c2st = classifier_two_sample_test(
        train_df=train_df,
        test_df=test_df,
        features=features,
        max_rows_per_split=c2st_max_rows,
        random_state=int(checks.get("random_state", 42)),
    )

    n_features = int(len(drift_df))
    psi_breaches = (
        int((~drift_df.get("pass_psi", pd.Series(dtype=bool))).sum()) if n_features else 0
    )
    ks_breaches = int((~drift_df.get("pass_ks", pd.Series(dtype=bool))).sum()) if n_features else 0
    cvm_breaches = (
        int((~drift_df.get("pass_cvm", pd.Series(dtype=bool))).sum()) if n_features else 0
    )
    feature_breach_ratio = float(psi_breaches / max(n_features, 1))
    distribution_warning_ratio = float((ks_breaches + cvm_breaches) / max(n_features * 2, 1))

    max_psi = float(drift_df["psi"].max()) if n_features else 0.0
    mean_psi = _safe_mean(drift_df["psi"]) if n_features else 0.0
    min_ks_pvalue = float(drift_df["ks_pvalue"].min()) if n_features else 1.0
    min_cvm_pvalue = float(drift_df["cvm_pvalue"].min()) if n_features else 1.0
    c2st_auc = float(c2st["c2st_auc"])
    c2st_materiality = str(c2st.get("materiality", "none"))
    c2st_effective_driver_count = int(c2st.get("effective_driver_count", 0))
    c2st_top_drivers = list(c2st.get("top_drivers", []) or [])
    performance_report = _score_and_performance_report(
        train_df,
        test_df,
        random_state=int(checks.get("random_state", 42)),
        max_rows_per_split=performance_max_rows,
        psi_bins=int(checks.get("psi_bins", 10)),
    )
    score_psi = float(performance_report.get("score_psi", 0.0))
    auc_delta = float(performance_report.get("auc_delta_train_to_test", 0.0))
    brier_increase = float(performance_report.get("brier_increase_train_to_test", 0.0))
    calibration_gap_delta = float(performance_report.get("calibration_gap_delta", 0.0))

    pass_psi = bool(max_psi <= psi_threshold)
    pass_breach_ratio = bool(feature_breach_ratio <= max_feature_breach_ratio)
    pass_score_psi = bool(score_psi <= score_psi_max)
    pass_auc_delta = bool(auc_delta <= auc_delta_max)
    pass_brier_increase = bool(brier_increase <= brier_increase_max)
    pass_calibration_gap_delta = bool(calibration_gap_delta <= calibration_gap_delta_max)
    pass_c2st = bool(c2st_auc <= c2st_auc_max)
    model_shift = interpret_model_shift(
        c2st_auc=c2st_auc,
        c2st_materiality=c2st_materiality,
        score_psi=score_psi,
        auc_delta=auc_delta,
        brier_increase=brier_increase,
        calibration_gap_delta=calibration_gap_delta,
        distribution_warning_ratio=distribution_warning_ratio,
        score_psi_max=score_psi_max,
        auc_delta_max=auc_delta_max,
        brier_increase_max=brier_increase_max,
        calibration_gap_delta_max=calibration_gap_delta_max,
    )

    drift_path.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_parquet(drift_path, index=False)

    shap_raw_path = Path("data/processed/shap_raw_top20.parquet")
    explanation_drift = pd.DataFrame()
    if shap_raw_path.exists():
        explanation_drift = _build_explanation_drift_report(
            pd.read_parquet(shap_raw_path),
            primary_threshold=_resolve_primary_threshold(),
            min_rank_overlap_top10=min_rank_overlap_top10,
            max_shap_psi=max_explanation_shap_psi,
            min_reason_code_stability=min_reason_code_stability,
            min_rows_per_slice=explanation_min_rows_per_slice,
        )
    explanation_drift_path.parent.mkdir(parents=True, exist_ok=True)
    explanation_drift.to_parquet(explanation_drift_path, index=False)

    fairness_status = {}
    if fairness_status_path.exists():
        try:
            fairness_status = json.loads(fairness_status_path.read_text(encoding="utf-8"))
        except Exception:
            fairness_status = {}
    fairness_pass = bool(fairness_status.get("overall_pass", False))

    challenger_report = {}
    if challenger_report_path.exists():
        try:
            challenger_report = json.loads(challenger_report_path.read_text(encoding="utf-8"))
        except Exception:
            challenger_report = {}
    challenger_promotable = bool(challenger_report.get("challenger_promotable", False))

    explainability_pass = bool(
        (not explanation_drift.empty) and explanation_drift["passed_all"].astype(bool).all()
    )
    reason_code_stability_pass = bool(
        (not explanation_drift.empty)
        and explanation_drift["pass_reason_code_stability"].astype(bool).all()
    )
    predictive_drift_pass = bool(
        pass_psi
        and pass_breach_ratio
        and pass_score_psi
        and pass_auc_delta
        and pass_brier_increase
        and pass_calibration_gap_delta
    )
    overall_pass = bool(predictive_drift_pass and fairness_pass)
    warning_flags = {
        "warn_c2st": bool(not pass_c2st),
        "warn_distribution_tests": bool(ks_breaches > 0 or cvm_breaches > 0),
        "warn_explainability": bool(not explainability_pass),
        "warn_reason_code_stability": bool(not reason_code_stability_pass),
    }

    top_breaches = drift_df.head(10).to_dict(orient="records") if n_features else []
    top_explanation_breaches = (
        explanation_drift.sort_values(
            ["passed_all", "max_shap_psi_top5", "rank_overlap_top10"],
            ascending=[True, False, True],
        )
        .head(10)
        .to_dict(orient="records")
        if not explanation_drift.empty
        else []
    )
    status = {
        "overall_pass": overall_pass,
        "checks": {
            "pass_psi": pass_psi,
            "pass_breach_ratio": pass_breach_ratio,
            "pass_score_psi": pass_score_psi,
            "pass_auc_delta": pass_auc_delta,
            "pass_brier_increase": pass_brier_increase,
            "pass_calibration_gap_delta": pass_calibration_gap_delta,
            "pass_predictive_drift": predictive_drift_pass,
            "pass_fairness": fairness_pass,
            "pass_c2st": pass_c2st,
            "pass_explainability": explainability_pass,
            "pass_reason_code_stability": reason_code_stability_pass,
            **warning_flags,
        },
        "thresholds": {
            "psi_threshold": psi_threshold,
            "ks_pvalue_min": ks_pvalue_min,
            "cvm_pvalue_min": cvm_pvalue_min,
            "c2st_auc_max": c2st_auc_max,
            "max_feature_breach_ratio": max_feature_breach_ratio,
            "score_psi_max": score_psi_max,
            "auc_delta_max": auc_delta_max,
            "brier_increase_max": brier_increase_max,
            "calibration_gap_delta_max": calibration_gap_delta_max,
            "explanation_rank_overlap_top10_min": min_rank_overlap_top10,
            "explanation_shap_psi_max": max_explanation_shap_psi,
            "reason_code_stability_min": min_reason_code_stability,
        },
        "summary": {
            "n_features": n_features,
            "max_psi": max_psi,
            "mean_psi": mean_psi,
            "min_ks_pvalue": min_ks_pvalue,
            "min_cvm_pvalue": min_cvm_pvalue,
            "c2st_auc": c2st_auc,
            "psi_breaches": psi_breaches,
            "ks_breaches": ks_breaches,
            "cvm_breaches": cvm_breaches,
            "feature_breach_ratio": feature_breach_ratio,
            "distribution_warning_ratio": distribution_warning_ratio,
            "c2st_rows_used": int(c2st.get("n_rows", 0)),
            "c2st_materiality": c2st_materiality,
            "c2st_effective_driver_count": c2st_effective_driver_count,
            **performance_report,
            "n_explanation_segments": int(len(explanation_drift)),
            "min_rank_overlap_top10": float(explanation_drift["rank_overlap_top10"].min())
            if not explanation_drift.empty
            else 0.0,
            "max_explanation_shap_psi": float(explanation_drift["max_shap_psi_top5"].max())
            if not explanation_drift.empty
            else 0.0,
            "min_reason_code_stability": float(explanation_drift["reason_code_match_rate"].min())
            if not explanation_drift.empty
            else 0.0,
            "fairness_overall_pass": fairness_pass,
            "fairness_primary_threshold": float(
                fairness_status.get(
                    "primary_threshold", fairness_status.get("prediction_threshold", 0.5)
                )
            )
            if fairness_status
            else 0.5,
            "challenger_promotable": challenger_promotable,
            "model_shift_type": str(model_shift["shift_type"]),
            "governance_posture": str(model_shift["governance_posture"]),
        },
        "warnings": warning_flags,
        "c2st": {
            "auc": c2st_auc,
            "materiality": c2st_materiality,
            "effective_driver_count": c2st_effective_driver_count,
            "top_drivers": c2st_top_drivers,
        },
        "model_shift": model_shift,
        "artifacts": {
            "drift_monitoring_path": str(drift_path),
            "explanation_drift_path": str(explanation_drift_path),
            "fairness_status_path": str(fairness_status_path),
            "fairness_frontier_path": str(fairness_frontier_path),
            "challenger_promotion_report_path": str(challenger_report_path),
            "model_shift_status_path": str(model_shift_status_path),
        },
        "top_drift_features": top_breaches,
        "top_explanation_breaches": top_explanation_breaches,
        "primary_threshold": float(
            fairness_status.get(
                "primary_threshold", fairness_status.get("prediction_threshold", 0.5)
            )
        )
        if fairness_status
        else 0.5,
        "explainability_pass": explainability_pass,
        "explanation_drift_pass": explainability_pass,
        "reason_code_stability_pass": reason_code_stability_pass,
        "challenger_promotable": challenger_promotable,
        "policy_config": config_path,
        **build_artifact_metadata(
            schema_version=SCHEMA_VERSION,
            run_tag=resolved_run_tag,
            require_explicit=True,
        ),
    }

    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    model_shift_status_path.parent.mkdir(parents=True, exist_ok=True)
    model_shift_status_path.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "overall_pass": bool(model_shift["governance_posture"] != "candidate_gate"),
                "summary": model_shift,
                "artifacts": {"governance_status_path": str(status_path)},
                **build_artifact_metadata(
                    schema_version=SCHEMA_VERSION,
                    run_tag=resolved_run_tag,
                    require_explicit=True,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Saved drift monitoring: {}", drift_path)
    logger.info("Saved governance status: {}", status_path)
    logger.info("Saved model-shift interpretation: {}", model_shift_status_path)
    logger.info(
        "Governance checks pass={} (max_psi={:.4f}, score_psi={:.4f}, auc_delta={:.4f}, brier_increase={:.4f}, c2st_auc={:.4f})",
        overall_pass,
        max_psi,
        score_psi,
        auc_delta,
        brier_increase,
        c2st_auc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate governance drift status")
    parser.add_argument("--config", default="configs/mrm_policy.yaml")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    main(config_path=args.config, run_tag=args.run_tag)
