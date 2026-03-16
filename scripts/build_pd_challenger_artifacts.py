"""Build feature-selection and monotonic challenger artifacts for PD modeling.

Usage:
    uv run python scripts/build_pd_challenger_artifacts.py --config configs/pd_model.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluation.explainability import (
    effective_driver_count,
    monotonic_violation_rate,
    rank_overlap_ratio,
)
from src.evaluation.fairness import fairness_report
from src.models.calibration import expected_calibration_error
from src.models.pd_contract import CONTRACT_PATH, load_contract, resolve_model_path
from src.models.pd_model import TARGET, resolve_feature_sets, temporal_train_val_split
from src.utils.io_utils import read_split_with_fe_fallback
from src.utils.threshold_semantics import load_threshold_semantics, resolve_operational_threshold


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_rows(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    if n_rows <= 0 or len(df) <= n_rows:
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


def _prepare_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in features:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    fill_values = X.median(numeric_only=True).fillna(0.0)
    return X.fillna(fill_values).fillna(0.0)


def _boruta_proxy_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.Series, pd.Series]:
    """Boruta-style proxy: compare real feature importances vs shuffled shadows."""
    rng = np.random.RandomState(random_state)
    shadow = X.apply(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    shadow.columns = [f"shadow__{c}" for c in shadow.columns]

    X_aug = pd.concat([X, shadow], axis=1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_aug, y)

    importances = pd.Series(model.feature_importances_, index=X_aug.columns, dtype=float)
    real_imp = importances[[c for c in X.columns if c in importances.index]]
    shadow_imp = importances[[c for c in shadow.columns if c in importances.index]]
    threshold = float(shadow_imp.max()) if len(shadow_imp) else float(real_imp.quantile(0.75))
    support = (real_imp > threshold).astype(bool)
    return real_imp, support


def _permutation_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    max_rows: int,
) -> pd.Series:
    if len(X) > max_rows > 0:
        idx = np.random.RandomState(random_state).choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=150,
        learning_rate=0.05,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    perm = permutation_importance(
        clf,
        X_val,
        y_val,
        n_repeats=3,
        random_state=random_state,
        scoring="roc_auc",
    )
    return pd.Series(perm.importances_mean, index=X.columns, dtype=float)


def _resolve_primary_threshold() -> float:
    semantics = load_threshold_semantics()
    if semantics:
        return resolve_operational_threshold(semantics, default=0.5)
    decision_threshold = Path("models/decision_threshold.json")
    if decision_threshold.exists():
        try:
            payload = json.loads(decision_threshold.read_text(encoding="utf-8"))
            return float(payload.get("selected_threshold", 0.5))
        except Exception:
            pass
    fairness_policy = Path("configs/fairness_policy.yaml")
    if fairness_policy.exists():
        try:
            cfg = yaml.safe_load(fairness_policy.read_text(encoding="utf-8")) or {}
            return float((cfg.get("policy", {}) or {}).get("prediction_threshold", 0.5))
        except Exception:
            pass
    return 0.5


def _build_groups_for_fairness(df: pd.DataFrame) -> dict[str, np.ndarray]:
    fairness_policy = Path("configs/fairness_policy.yaml")
    if not fairness_policy.exists():
        return {}
    try:
        cfg = yaml.safe_load(fairness_policy.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    groups: dict[str, np.ndarray] = {}
    for attr in cfg.get("attributes", []) or []:
        name = str(attr.get("name", "")).strip()
        column = str(attr.get("column", "")).strip()
        if not name or not column or column not in df.columns:
            continue
        if attr.get("binning") == "quartile":
            groups[name] = (
                pd.qcut(df[column], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                .astype(str)
                .to_numpy()
            )
        else:
            groups[name] = df[column].astype(str).to_numpy()
    return groups


def _load_fairness_policy_cfg() -> dict[str, Any]:
    fairness_policy = Path("configs/fairness_policy.yaml")
    if not fairness_policy.exists():
        return {}
    try:
        return yaml.safe_load(fairness_policy.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _cohort_metric_table(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    if "issue_d" in df.columns:
        cohorts = pd.to_datetime(df["issue_d"], errors="coerce").dt.to_period("Q").astype(str)
    else:
        cohorts = pd.Series(["all"] * len(df))
    rows = []
    for cohort in sorted(pd.Series(cohorts).dropna().unique().tolist()):
        mask = pd.Series(cohorts).astype(str).eq(str(cohort)).to_numpy()
        if int(mask.sum()) < 100:
            continue
        rows.append(
            {
                "cohort": str(cohort),
                "auc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "brier": float(brier_score_loss(y_true[mask], y_prob[mask])),
            }
        )
    return pd.DataFrame(rows)


def _importance_stability(
    model,
    test_df: pd.DataFrame,
    features: list[str],
    *,
    random_state: int,
    max_rows: int,
) -> tuple[float, pd.Series]:
    if test_df.empty or len(features) == 0:
        return 0.0, pd.Series(dtype=float)
    if "issue_d" in test_df.columns:
        ordered = test_df.sort_values("issue_d").reset_index(drop=True)
    else:
        ordered = test_df.reset_index(drop=True)
    split = max(int(len(ordered) * 0.5), 1)
    early = ordered.iloc[:split].copy()
    late = ordered.iloc[split:].copy()
    if early.empty or late.empty:
        return 0.0, pd.Series(dtype=float)
    y_early = early[TARGET].astype(int).to_numpy()
    y_late = late[TARGET].astype(int).to_numpy()
    imp_early = _permutation_scores(
        _prepare_numeric_frame(early, features),
        y_early,
        random_state=random_state,
        max_rows=max_rows,
    ).sort_values(ascending=False)
    imp_late = _permutation_scores(
        _prepare_numeric_frame(late, features),
        y_late,
        random_state=random_state + 1,
        max_rows=max_rows,
    ).sort_values(ascending=False)
    overlap = rank_overlap_ratio(
        imp_early.index.astype(str).tolist(),
        imp_late.index.astype(str).tolist(),
        top_k=min(10, len(features)),
    )
    return overlap, imp_late


def _load_champion_test_matrix(test_df: pd.DataFrame) -> tuple[Any | None, pd.DataFrame]:
    if not Path(CONTRACT_PATH).exists():
        return None, pd.DataFrame()
    contract = load_contract(CONTRACT_PATH) or {}
    feature_names = [str(f) for f in contract.get("feature_names", []) if f in test_df.columns]
    if not feature_names:
        return None, pd.DataFrame()
    categorical = set(contract.get("categorical_features", []) or [])
    X = test_df[feature_names].copy()
    categorical = categorical.union(
        {feature for feature in X.columns if not pd.api.types.is_numeric_dtype(X[feature])}
    )
    for feature in feature_names:
        if feature in categorical:
            X[feature] = X[feature].astype("string").fillna("UNKNOWN").astype(str)
    model = CatBoostClassifier()
    model.load_model(str(resolve_model_path()))

    class _CatBoostPredictorAdapter:
        def __init__(self, raw_model: CatBoostClassifier, cat_features: list[str]) -> None:
            self.raw_model = raw_model
            self.cat_features = list(cat_features)

        def predict_proba(self, X_input: pd.DataFrame) -> np.ndarray:
            from catboost import Pool

            return np.asarray(
                self.raw_model.predict_proba(
                    Pool(
                        X_input, cat_features=[c for c in self.cat_features if c in X_input.columns]
                    )
                ),
                dtype=float,
            )

    return _CatBoostPredictorAdapter(model, sorted(categorical)), X


def main(config_path: str = "configs/pd_model.yaml") -> None:
    cfg = _load_config(config_path)
    challenger_cfg = cfg.get("challenger_pipeline", {})
    data_cfg = cfg.get("data", {})

    random_state = int(challenger_cfg.get("random_state", 42))
    sample_rows = int(challenger_cfg.get("sample_rows", 150_000))
    top_k = int(challenger_cfg.get("top_k", 30))
    perm_max_rows = int(challenger_cfg.get("permutation_max_rows", 80_000))
    boruta_n_estimators = int(challenger_cfg.get("boruta_n_estimators", 120))

    train_df = read_split_with_fe_fallback(
        data_cfg.get("train_path", "data/processed/train_fe.parquet")
    )
    train_df = _sample_rows(train_df, sample_rows, random_state)
    if TARGET not in train_df.columns:
        raise KeyError(f"Missing target column '{TARGET}' in sampled training data")

    feature_src_cfg = cfg.get("feature_source", {})
    feature_sets = resolve_feature_sets(
        train_df,
        feature_source=feature_src_cfg.get("mode", "auto"),
        feature_config_path=feature_src_cfg.get(
            "feature_config_path", "data/processed/feature_config.pkl"
        ),
    )

    catboost_features = feature_sets["catboost_features"]
    categorical_features = set(feature_sets["categorical_features"])
    numeric_features = [
        f for f in catboost_features if f not in categorical_features and f in train_df.columns
    ]
    if not numeric_features:
        raise ValueError("No numeric features available for challenger feature selection")

    X = _prepare_numeric_frame(train_df, numeric_features)
    y = train_df[TARGET].astype(int).to_numpy()

    mi = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns, dtype=float)

    boruta_importance, boruta_support = _boruta_proxy_importance(
        X,
        y,
        random_state=random_state,
        n_estimators=boruta_n_estimators,
    )

    perm_scores = _permutation_scores(
        X,
        y,
        random_state=random_state,
        max_rows=perm_max_rows,
    )

    out = pd.DataFrame(
        {
            "feature": X.columns,
            "mutual_info": [float(mi_series.get(c, 0.0)) for c in X.columns],
            "boruta_importance": [float(boruta_importance.get(c, 0.0)) for c in X.columns],
            "boruta_support": [bool(boruta_support.get(c, False)) for c in X.columns],
            "permutation_importance_auc": [float(perm_scores.get(c, 0.0)) for c in X.columns],
        }
    )
    out["rank_mi"] = out["mutual_info"].rank(ascending=False, method="dense")
    out["rank_boruta"] = out["boruta_importance"].rank(ascending=False, method="dense")
    out["rank_permutation"] = out["permutation_importance_auc"].rank(
        ascending=False, method="dense"
    )
    out["rank_aggregate"] = out[["rank_mi", "rank_boruta", "rank_permutation"]].mean(axis=1)
    out["method_votes"] = (
        (out["mutual_info"] > out["mutual_info"].median()).astype(int)
        + out["boruta_support"].astype(int)
        + (out["permutation_importance_auc"] > out["permutation_importance_auc"].median()).astype(
            int
        )
    )
    out = out.sort_values(["rank_aggregate", "method_votes"], ascending=[True, False]).reset_index(
        drop=True
    )
    out["selected_topk"] = False
    out.loc[: max(top_k, 1) - 1, "selected_topk"] = True

    constraints_cfg = challenger_cfg.get("monotonic_constraints", {})
    monotonic_vector = [int(constraints_cfg.get(feature, 0)) for feature in catboost_features]
    selected_features = out.loc[out["selected_topk"], "feature"].astype(str).tolist()

    no_smote_policy = bool(challenger_cfg.get("no_smote", True))
    spec = {
        "schema_version": "2026-03-06.1",
        "sample_rows_used": int(len(train_df)),
        "top_k": top_k,
        "target": TARGET,
        "feature_source": feature_sets.get("feature_source", "unknown"),
        "catboost_feature_count": int(len(catboost_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "selected_features": selected_features,
        "selection_methods": [
            "mrmr_proxy_mutual_info",
            "boruta_proxy_shadow_importance",
            "permutation_importance_auc",
        ],
        "monotonic_constraints": {
            "by_feature": {k: int(v) for k, v in constraints_cfg.items()},
            "vector_in_catboost_order": monotonic_vector,
        },
        "modeling_policies": {
            "no_smote": no_smote_policy,
            "challenger_only": True,
        },
        "notes": [
            "Feature-selection and monotonic constraints are challenger-only until full gate pass.",
            "No synthetic oversampling (SMOTE) is allowed by policy in challenger profiles.",
        ],
    }

    out_path = Path(
        challenger_cfg.get(
            "feature_selection_output", "data/processed/challenger_feature_selection.parquet"
        )
    )
    spec_path = Path(challenger_cfg.get("spec_output", "models/pd_challenger_spec.json"))
    spec_v2_path = Path(challenger_cfg.get("spec_output_v2", "models/pd_challenger_spec_v2.json"))
    promotion_report_path = Path(
        challenger_cfg.get(
            "promotion_report_output",
            "models/challenger_promotion_report.json",
        )
    )

    benchmark_available = False
    promotion_report: dict[str, Any] = {
        "schema_version": "2026-03-06.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_available": False,
        "challenger_promotable": False,
        "selected_features": selected_features,
        "notes": [
            "Benchmark not available because test artifacts or champion references are missing.",
        ],
    }

    test_path = Path(data_cfg.get("test_path", "data/processed/test_fe.parquet"))
    preds_path = Path("data/processed/test_predictions.parquet")
    if test_path.exists() and preds_path.exists() and selected_features:
        test_df = read_split_with_fe_fallback(str(test_path)).reset_index(drop=True)
        preds_df = pd.read_parquet(preds_path).reset_index(drop=True)
        n_eval = min(len(test_df), len(preds_df))
        test_df = test_df.iloc[:n_eval].copy()
        preds_df = preds_df.iloc[:n_eval].copy()

        fit_df, val_df = temporal_train_val_split(
            train_df,
            val_fraction=float(challenger_cfg.get("validation_fraction", 0.15)),
            date_col=str(challenger_cfg.get("date_col", "issue_d")),
        )
        X_fit = _prepare_numeric_frame(fit_df, selected_features)
        X_val = _prepare_numeric_frame(val_df, selected_features)
        X_test = _prepare_numeric_frame(test_df, selected_features)
        y_fit = fit_df[TARGET].astype(int).to_numpy()
        y_val = val_df[TARGET].astype(int).to_numpy()
        y_test = test_df[TARGET].astype(int).to_numpy()

        monotonic_selected = [int(constraints_cfg.get(feature, 0)) for feature in selected_features]
        monotonic_param = monotonic_selected if any(monotonic_selected) else None
        challenger_model = HistGradientBoostingClassifier(
            max_depth=int(challenger_cfg.get("benchmark_max_depth", 6)),
            max_iter=int(challenger_cfg.get("benchmark_max_iter", 220)),
            learning_rate=float(challenger_cfg.get("benchmark_learning_rate", 0.05)),
            random_state=random_state,
            monotonic_cst=monotonic_param,
        )
        challenger_model.fit(X_fit, y_fit)
        val_raw = challenger_model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(val_raw, y_val)
        test_raw = challenger_model.predict_proba(X_test)[:, 1]
        test_cal = np.clip(calibrator.predict(test_raw), 0.0, 1.0)

        if "pd_calibrated" in preds_df.columns:
            champion_prob = pd.to_numeric(preds_df["pd_calibrated"], errors="coerce")
        elif "y_prob_final" in preds_df.columns:
            champion_prob = pd.to_numeric(preds_df["y_prob_final"], errors="coerce")
        else:
            champion_prob = pd.Series(np.nan, index=preds_df.index)
        if champion_prob.isna().all():
            champion_prob = pd.Series(np.nan, index=preds_df.index)
        champion_prob = champion_prob.fillna(
            pd.to_numeric(preds_df.get("y_prob_cb_tuned"), errors="coerce")
        )
        champion_prob = champion_prob.fillna(0.5).to_numpy(dtype=float)

        challenger_metrics = {
            "auc": float(roc_auc_score(y_test, test_cal)),
            "brier": float(brier_score_loss(y_test, test_cal)),
            "ece": float(expected_calibration_error(y_test, test_cal)),
        }
        champion_metrics = {
            "auc": float(roc_auc_score(y_test, champion_prob)),
            "brier": float(brier_score_loss(y_test, champion_prob)),
            "ece": float(expected_calibration_error(y_test, champion_prob)),
        }

        challenger_importance = _permutation_scores(
            X_test,
            y_test,
            random_state=random_state,
            max_rows=min(perm_max_rows, len(X_test)),
        ).sort_values(ascending=False)
        challenger_driver_count = effective_driver_count(challenger_importance)

        shap_summary_path = Path("data/processed/shap_summary.parquet")
        if shap_summary_path.exists():
            shap_summary = pd.read_parquet(shap_summary_path)
            champion_driver_count = effective_driver_count(
                pd.Series(
                    shap_summary["mean_abs_shap"].to_numpy(dtype=float),
                    index=shap_summary["feature"].astype(str),
                )
            )
        else:
            champion_driver_count = len(selected_features)

        challenger_stability, _ = _importance_stability(
            challenger_model,
            test_df[
                selected_features + [TARGET] + ([c for c in ["issue_d"] if c in test_df.columns])
            ].copy(),
            selected_features,
            random_state=random_state,
            max_rows=min(perm_max_rows, max(len(test_df) // 2, 1)),
        )

        shap_raw_path = Path("data/processed/shap_raw_top20.parquet")
        champion_stability = 0.0
        if shap_raw_path.exists():
            shap_raw = pd.read_parquet(shap_raw_path)
            feature_names = [
                c.replace("shap_", "") for c in shap_raw.columns if c.startswith("shap_")
            ]
            if "issue_quarter" in shap_raw.columns and feature_names:
                periods = sorted(
                    [
                        p
                        for p in shap_raw["issue_quarter"].astype(str).dropna().unique().tolist()
                        if p != "unknown"
                    ]
                )
                if len(periods) >= 2:
                    ref = shap_raw.loc[shap_raw["issue_quarter"].astype(str) != periods[-1]].copy()
                    cmp = shap_raw.loc[shap_raw["issue_quarter"].astype(str) == periods[-1]].copy()
                    if not ref.empty and not cmp.empty:
                        ref_rank = sorted(
                            feature_names,
                            key=lambda feature: ref[f"shap_{feature}"].abs().mean(),
                            reverse=True,
                        )
                        cmp_rank = sorted(
                            feature_names,
                            key=lambda feature: cmp[f"shap_{feature}"].abs().mean(),
                            reverse=True,
                        )
                        champion_stability = rank_overlap_ratio(ref_rank, cmp_rank, top_k=10)

        constrained_features = [f for f in selected_features if int(constraints_cfg.get(f, 0)) != 0]
        challenger_violation_rates = [
            monotonic_violation_rate(
                challenger_model,
                X_test[selected_features],
                feature,
                int(constraints_cfg.get(feature, 0)),
                random_state=random_state,
            )
            for feature in constrained_features
        ]
        challenger_violation = (
            float(np.mean(challenger_violation_rates)) if challenger_violation_rates else 0.0
        )

        champion_violation = 0.0
        champion_model, champion_matrix = _load_champion_test_matrix(test_df)
        if champion_model is not None and not champion_matrix.empty and constrained_features:
            champion_rates = [
                monotonic_violation_rate(
                    champion_model,
                    champion_matrix,
                    feature,
                    int(constraints_cfg.get(feature, 0)),
                    random_state=random_state,
                )
                for feature in constrained_features
                if feature in champion_matrix.columns
            ]
            champion_violation = float(np.mean(champion_rates)) if champion_rates else 0.0

        fairness_cfg = _load_fairness_policy_cfg()
        fairness_groups = _build_groups_for_fairness(test_df)
        threshold = _resolve_primary_threshold()
        challenger_fairness_pass = True
        challenger_fairness_rows: list[dict[str, Any]] = []
        if fairness_groups:
            policy = fairness_cfg.get("policy", {}) or {}
            outcome_mode = str(policy.get("outcome_mode", "default")).strip().lower()
            if outcome_mode in {"approval", "approve", "good", "non_default"}:
                y_true_fair = 1.0 - y_test
                y_proba_fair = 1.0 - test_cal
            else:
                y_true_fair = y_test
                y_proba_fair = test_cal
            challenger_fairness = fairness_report(
                y_true=y_true_fair,
                y_pred_proba=y_proba_fair,
                groups_dict=fairness_groups,
                threshold=threshold,
                dpd_threshold=float(policy.get("dpd_threshold", 0.10)),
                eo_gap_threshold=float(policy.get("eo_gap_threshold", 0.11)),
                dir_threshold=float(policy.get("dir_threshold", 0.80)),
            )
            challenger_fairness_pass = bool(challenger_fairness["passed_all"].all())
            challenger_fairness_rows = challenger_fairness.to_dict(orient="records")

        governance_status_path = Path("models/governance_status.json")
        predictive_drift_pass = True
        if governance_status_path.exists():
            try:
                governance_status = json.loads(governance_status_path.read_text(encoding="utf-8"))
                predictive_drift_pass = bool(
                    (governance_status.get("checks", {}) or {}).get("pass_predictive_drift", True)
                )
            except Exception:
                predictive_drift_pass = True

        interpretability_gains = {
            "fewer_effective_drivers": bool(challenger_driver_count < champion_driver_count),
            "more_stable_explanations": bool(challenger_stability >= champion_stability),
            "better_monotonic_consistency": bool(challenger_violation <= champion_violation),
        }
        interpretability_gain_count = int(
            sum(1 for passed in interpretability_gains.values() if passed)
        )

        max_auc_drop = float(challenger_cfg.get("max_auc_drop", 0.01))
        max_brier_increase_pct = float(challenger_cfg.get("max_brier_increase_pct", 0.05))
        min_interpretability_gains = int(challenger_cfg.get("min_interpretability_gains", 2))
        auc_drop = float(champion_metrics["auc"] - challenger_metrics["auc"])
        brier_ratio = (
            float(challenger_metrics["brier"] / max(champion_metrics["brier"], 1e-9)) - 1.0
        )
        promotion_checks = {
            "auc_drop_ok": bool(auc_drop <= max_auc_drop),
            "brier_guardrail_ok": bool(brier_ratio <= max_brier_increase_pct),
            "fairness_ok": bool(challenger_fairness_pass),
            "predictive_drift_ok": bool(predictive_drift_pass),
            "interpretability_gain_count_ok": bool(
                interpretability_gain_count >= min_interpretability_gains
            ),
        }
        benchmark_available = True
        promotion_report = {
            "schema_version": "2026-03-06.1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "benchmark_available": True,
            "selected_features": selected_features,
            "primary_threshold": float(threshold),
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
            "deltas": {
                "auc_drop": auc_drop,
                "brier_increase_pct": brier_ratio,
                "ece_delta": float(challenger_metrics["ece"] - champion_metrics["ece"]),
            },
            "interpretability": {
                "champion_effective_driver_count": int(champion_driver_count),
                "challenger_effective_driver_count": int(challenger_driver_count),
                "champion_explanation_stability": float(champion_stability),
                "challenger_explanation_stability": float(challenger_stability),
                "champion_monotonic_violation_rate": float(champion_violation),
                "challenger_monotonic_violation_rate": float(challenger_violation),
                "gains": interpretability_gains,
                "gain_count": interpretability_gain_count,
            },
            "fairness": {
                "overall_pass": bool(challenger_fairness_pass),
                "attributes": challenger_fairness_rows,
            },
            "promotion_checks": promotion_checks,
            "challenger_promotable": bool(all(promotion_checks.values())),
            "notes": [
                "Monotonic challenger is benchmarked on selected numeric features only.",
                "Production calibrator remains isotonic on the champion model.",
            ],
        }
        spec["benchmark_summary"] = {
            "benchmark_available": True,
            "challenger_promotable": bool(promotion_report["challenger_promotable"]),
            "auc_drop": auc_drop,
            "brier_increase_pct": brier_ratio,
            "interpretability_gain_count": interpretability_gain_count,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_report_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(out_path, index=False)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)
    spec_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_v2_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)
    with open(promotion_report_path, "w", encoding="utf-8") as f:
        json.dump(promotion_report, f, indent=2, default=str)

    logger.info(
        "Challenger artifacts saved: {} ({} rows), {}",
        out_path,
        len(out),
        spec_path,
    )
    logger.info("Challenger spec v2 saved: {}", spec_v2_path)
    logger.info("Challenger promotion report saved: {}", promotion_report_path)
    logger.info(
        "Selected top-k features: {}",
        ", ".join(spec["selected_features"][:10]),
    )
    if benchmark_available:
        logger.info(
            "Challenger promotable={} | auc_drop={:.4f} | brier_increase_pct={:.4f}",
            bool(promotion_report["challenger_promotable"]),
            float(promotion_report["deltas"]["auc_drop"]),
            float(promotion_report["deltas"]["brier_increase_pct"]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PD challenger feature-selection artifacts")
    parser.add_argument("--config", default="configs/pd_model.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
