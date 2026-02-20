"""Export Streamlit-ready artifacts from existing pipeline outputs.

Reads existing parquets, pickles, and models to produce JSON/parquet files
optimized for the Streamlit dashboard. Run after all notebooks have executed.

Usage:
    uv run python scripts/export_streamlit_artifacts.py
"""

from __future__ import annotations

import json
import pickle
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPORT_COUNT = 0


def _save_json(data: dict, path: Path) -> None:
    global EXPORT_COUNT
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved {path.name}")
    EXPORT_COUNT += 1


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    global EXPORT_COUNT
    df.to_parquet(path, index=False)
    logger.info(f"Saved {path.name} ({df.shape[0]} rows x {df.shape[1]} cols)")
    EXPORT_COUNT += 1


def _collect_test_inventory() -> tuple[int, list[dict[str, int | str]]]:
    """Collect pytest node inventory for runtime status metadata."""
    cmd = ["uv", "run", "pytest", "--collect-only", "-q", "-q"]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    counts: Counter[str] = Counter()
    total = 0
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line.startswith("tests/") or "::" not in line:
            continue
        module = line.split("::", maxsplit=1)[0].removeprefix("tests/").removesuffix(".py")
        counts[module] += 1
        total += 1
    breakdown = [{"module": module, "tests": int(n)} for module, n in sorted(counts.items())]
    return total, breakdown


# ── 1. EDA Summary ──
def export_eda_summary() -> None:
    """Summary stats from loan_master for the Executive Summary page."""
    logger.info("Exporting EDA summary...")
    df = pd.read_parquet(DATA_DIR / "loan_master.parquet")

    null_pcts = df.isnull().mean()
    top_nulls = null_pcts[null_pcts > 0.001].sort_values(ascending=False)

    summary = {
        "n_loans": int(len(df)),
        "n_features": int(df.shape[1]),
        "default_rate": round(float(df["default_flag"].mean()), 4),
        "n_defaults": int(df["default_flag"].sum()),
        "date_range": {
            "min": str(df["issue_d"].min().date()),
            "max": str(df["issue_d"].max().date()),
        },
        "loan_amnt": {
            "mean": round(float(df["loan_amnt"].mean()), 2),
            "median": round(float(df["loan_amnt"].median()), 2),
            "min": round(float(df["loan_amnt"].min()), 2),
            "max": round(float(df["loan_amnt"].max()), 2),
        },
        "annual_inc": {
            "mean": round(float(df["annual_inc"].mean()), 2),
            "median": round(float(df["annual_inc"].median()), 2),
        },
        "int_rate": {
            "mean": round(float(df["int_rate"].mean()), 4),
            "median": round(float(df["int_rate"].median()), 4),
        },
        "dti": {
            "mean": round(float(df["dti"].mean()), 4),
            "median": round(float(df["dti"].median()), 4),
        },
        "default_rate_by_grade": {
            g: round(float(r), 4)
            for g, r in df.groupby("grade")["default_flag"].mean().sort_index().items()
        },
        "loan_count_by_grade": {
            g: int(c) for g, c in df["grade"].value_counts().sort_index().items()
        },
        "null_pcts": {col: round(float(pct), 4) for col, pct in top_nulls.items()},
        "term_distribution": {str(t): int(c) for t, c in df["term"].value_counts().items()},
    }
    _save_json(summary, DATA_DIR / "eda_summary.json")


# ── 2. Feature Importance (IV) ──
def export_feature_importance_iv() -> None:
    """IV scores from feature engineering for the Data Story page."""
    logger.info("Exporting feature importance IV scores...")
    with open(DATA_DIR / "feature_config.pkl", "rb") as f:
        cfg = pickle.load(f)

    iv_scores = cfg.get("iv_scores", {})
    # Sort descending
    sorted_iv = dict(sorted(iv_scores.items(), key=lambda x: -x[1]))

    export = {
        "iv_scores": {k: round(v, 4) for k, v in sorted_iv.items()},
        "n_features_total": len(cfg.get("CATBOOST_FEATURES", [])),
        "n_woe_features": len(cfg.get("WOE_FEATURES", [])),
        "feature_lists": {
            "numeric": cfg.get("NUMERIC_FEATURES", []),
            "flag": cfg.get("FLAG_FEATURES", []),
            "categorical": cfg.get("CATEGORICAL_FEATURES", []),
            "woe": cfg.get("WOE_FEATURES", []),
            "interaction": cfg.get("INTERACTION_FEATURES", []),
            "catboost": cfg.get("CATBOOST_FEATURES", []),
            "logreg": cfg.get("LOGREG_FEATURES", []),
        },
    }
    _save_json(export, DATA_DIR / "feature_importance_iv.json")


# ── 3. Model Comparison ──
def export_model_comparison() -> None:
    """Model performance comparison table for the Model Laboratory page."""
    logger.info("Exporting model comparison...")

    # Load predictions
    preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    y_true = preds["y_true"].values

    # Load training record
    with open(MODEL_DIR / "pd_training_record.pkl", "rb") as f:
        rec = pickle.load(f)

    # Compute metrics for each model variant
    models = {
        "Logistic Regression": preds["y_prob_lr"].values,
        "CatBoost (default)": preds["y_prob_cb_default"].values,
        "CatBoost (tuned)": preds["y_prob_cb_tuned"].values,
        "CatBoost (tuned + calibrated)": preds["y_prob_final"].values,
    }

    comparison = []
    for name, y_prob in models.items():
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        from sklearn.metrics import brier_score_loss

        brier = brier_score_loss(y_true, y_prob)
        comparison.append(
            {
                "model": name,
                "auc": round(float(auc), 4),
                "gini": round(float(gini), 4),
                "brier": round(float(brier), 4),
            }
        )

    record_best_model = str(rec.get("best_model", "")).strip()
    if record_best_model in models:
        # Honor canonical model selection from training pipeline.
        best_model = record_best_model
    elif comparison:
        # Fallback: highest AUC, then lowest Brier.
        best_row = sorted(comparison, key=lambda row: (-row["auc"], row["brier"]))[0]
        best_model = str(best_row["model"])
    else:
        best_model = "CatBoost (tuned + calibrated)"

    feature_count_default = int(rec.get("feature_count_default", 0))
    feature_count_tuned = int(rec.get("feature_count_tuned", 0))
    if feature_count_default <= 0 or feature_count_tuned <= 0:
        try:
            with open(DATA_DIR / "feature_config.pkl", "rb") as f:
                feat_cfg = pickle.load(f)
            n_catboost = int(len(feat_cfg.get("CATBOOST_FEATURES", [])))
            if feature_count_default <= 0:
                feature_count_default = n_catboost
            if feature_count_tuned <= 0:
                feature_count_tuned = n_catboost
        except Exception:
            pass

    hpo_trials_executed = int(rec.get("hpo_trials_executed", rec.get("optuna_n_trials", 0)))
    hpo_best_validation_auc = float(
        rec.get("hpo_best_validation_auc", rec.get("optuna_best_auc", 0.0))
    )

    export = {
        "models": comparison,
        "best_model": best_model,
        "best_calibration": rec.get("best_calibration", "Platt Sigmoid"),
        "optuna_best_auc": round(rec.get("optuna_best_auc", 0), 4),
        "optuna_n_trials": hpo_trials_executed,
        "hpo_trials_executed": hpo_trials_executed,
        "hpo_best_validation_auc": round(hpo_best_validation_auc, 4),
        "validation_scheme": rec.get("validation_scheme", ""),
        "feature_count_default": feature_count_default,
        "feature_count_tuned": feature_count_tuned,
        "calibration_selection_report": rec.get("calibration_selection_report", {}),
        "final_test_metrics": rec.get("final_test_metrics", {}),
    }
    _save_json(export, DATA_DIR / "model_comparison.json")


# ── 4. ROC Curve Data ──
def export_roc_curves() -> None:
    """Pre-computed ROC curve points for the Model Laboratory page."""
    logger.info("Exporting ROC curve data...")
    preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    y_true = preds["y_true"].values

    models = {
        "logreg": preds["y_prob_lr"].values,
        "catboost_default": preds["y_prob_cb_default"].values,
        "catboost_tuned": preds["y_prob_cb_tuned"].values,
        "catboost_calibrated": preds["y_prob_final"].values,
    }

    rows = []
    for name, y_prob in models.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Downsample to ~200 points for faster rendering
        n = len(fpr)
        if n > 200:
            idx = np.linspace(0, n - 1, 200, dtype=int)
            fpr, tpr = fpr[idx], tpr[idx]
        for f, t in zip(fpr, tpr, strict=False):
            rows.append({"model": name, "fpr": round(float(f), 6), "tpr": round(float(t), 6)})

    _save_parquet(pd.DataFrame(rows), DATA_DIR / "roc_curve_data.parquet")


# ── 5. Calibration Curve Data ──
def export_calibration_curves() -> None:
    """Pre-computed calibration curves for the Model Laboratory page."""
    logger.info("Exporting calibration curve data...")
    preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    y_true = preds["y_true"].values

    models = {
        "catboost_uncalibrated": preds["y_prob_cb_tuned"].values,
        "catboost_calibrated": preds["y_prob_final"].values,
    }

    rows = []
    for name, y_prob in models.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=20, strategy="uniform")
        for pt, pp in zip(prob_true, prob_pred, strict=False):
            rows.append(
                {
                    "model": name,
                    "predicted_prob": round(float(pp), 6),
                    "observed_freq": round(float(pt), 6),
                }
            )

    _save_parquet(pd.DataFrame(rows), DATA_DIR / "calibration_curve_data.parquet")


# ── 6. Explainability Artifacts ──
def _load_pd_explainability_context(
    sample_size: int = 20000,
) -> tuple[Any, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Load canonical PD model + OOT matrix for explainability exports."""
    from catboost import CatBoostClassifier

    from src.models.pd_contract import CONTRACT_PATH, load_contract, resolve_model_path

    model_path = resolve_model_path()
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    contract = load_contract(CONTRACT_PATH) or {}
    feature_names = list(contract.get("feature_names", []) or getattr(model, "feature_names_", []) or [])
    categorical = list(contract.get("categorical_features", []) or [])

    test = pd.read_parquet(DATA_DIR / "test_fe.parquet")
    feature_names = [c for c in feature_names if c in test.columns]
    if not feature_names:
        raise ValueError("Unable to resolve feature names for explainability export.")

    X = test[feature_names].copy()
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("UNKNOWN").astype(str)

    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)

    if "default_flag" not in test.columns:
        raise ValueError("default_flag column missing in test_fe.parquet for explainability exports.")
    y = test.loc[X.index, "default_flag"].to_numpy(dtype=int)

    return model, X, y, feature_names, categorical


def export_shap_summary() -> None:
    """Export SHAP global + local artifacts for the final PD model."""
    logger.info("Exporting SHAP artifacts...")

    try:
        import shap

        model, X, _, features, _ = _load_pd_explainability_context(sample_size=5000)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs_shap}).sort_values(
            "mean_abs_shap", ascending=False
        )
        _save_parquet(shap_df, DATA_DIR / "shap_summary.parquet")

        top_features = shap_df["feature"].head(20).tolist()
        top_idx = [features.index(f) for f in top_features]
        raw_shap_df = pd.DataFrame(shap_values[:, top_idx], columns=[f"shap_{f}" for f in top_features])
        for f in top_features:
            raw_shap_df[f"val_{f}"] = X[f].values
        _save_parquet(raw_shap_df, DATA_DIR / "shap_raw_top20.parquet")

        # Local explanations for representative risk quantiles.
        y_prob = model.predict_proba(X)[:, 1]
        q10 = np.quantile(y_prob, 0.10)
        q50 = np.quantile(y_prob, 0.50)
        q90 = np.quantile(y_prob, 0.90)
        representative_idx = (
            np.argsort(np.abs(y_prob - q10))[:2].tolist()
            + np.argsort(np.abs(y_prob - q50))[:2].tolist()
            + np.argsort(np.abs(y_prob - q90))[:2].tolist()
        )
        representative_idx = sorted(set(representative_idx))[:6]

        local_rows: list[dict[str, Any]] = []
        for row_id in representative_idx:
            for f, feature_idx in zip(top_features, top_idx, strict=False):
                local_rows.append(
                    {
                        "case_id": int(row_id),
                        "predicted_pd": float(y_prob[row_id]),
                        "feature": f,
                        "feature_value": str(X.iloc[row_id][f]),
                        "shap_value": float(shap_values[row_id, feature_idx]),
                    }
                )
        _save_parquet(pd.DataFrame(local_rows), DATA_DIR / "shap_local_cases.parquet")

    except Exception as e:
        logger.warning(f"SHAP export failed: {e}. Skipping.")


def export_permutation_importance() -> None:
    """Export permutation importance on OOT sample using AUC degradation."""
    logger.info("Exporting permutation importance...")
    try:
        model, X, y, features, _ = _load_pd_explainability_context(sample_size=12000)
        baseline_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

        rng = np.random.default_rng(42)
        rows: list[dict[str, Any]] = []
        for feature in features:
            X_perm = X.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
            auc_perm = roc_auc_score(y, model.predict_proba(X_perm)[:, 1])
            rows.append(
                {
                    "feature": feature,
                    "baseline_auc": float(baseline_auc),
                    "permuted_auc": float(auc_perm),
                    "auc_drop": float(baseline_auc - auc_perm),
                }
            )
        imp_df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
        _save_parquet(imp_df, DATA_DIR / "permutation_importance.parquet")
    except Exception as e:
        logger.warning(f"Permutation importance export failed: {e}. Skipping.")


def export_pdp_ice_top5() -> None:
    """Export PDP/ICE data for top-5 numeric features."""
    logger.info("Exporting PDP/ICE top-5 features...")
    try:
        model, X, _, _, categorical = _load_pd_explainability_context(sample_size=4000)

        imp_path = DATA_DIR / "permutation_importance.parquet"
        if imp_path.exists():
            imp = pd.read_parquet(imp_path)
            ranking = imp["feature"].tolist()
        else:
            ranking = X.columns.tolist()

        numeric_candidates = [f for f in ranking if f in X.columns and f not in set(categorical)]
        top5 = numeric_candidates[:5]
        if not top5:
            raise ValueError("No numeric features available for PDP/ICE export.")

        ice_sample = X.sample(n=min(500, len(X)), random_state=11).copy()
        rows: list[dict[str, Any]] = []

        for feature in top5:
            valid = pd.to_numeric(X[feature], errors="coerce").dropna()
            if valid.empty:
                continue
            grid = np.quantile(valid, np.linspace(0.05, 0.95, 11))
            for value in grid:
                X_tmp = ice_sample.copy()
                X_tmp[feature] = value
                preds = model.predict_proba(X_tmp)[:, 1]
                pdp_value = float(np.mean(preds))
                for obs_id, pred in zip(X_tmp.index.to_numpy(), preds, strict=False):
                    rows.append(
                        {
                            "feature": feature,
                            "grid_value": float(value),
                            "observation_id": int(obs_id),
                            "ice_pred": float(pred),
                            "pdp_pred": pdp_value,
                        }
                    )

        _save_parquet(pd.DataFrame(rows), DATA_DIR / "pdp_ice_top5.parquet")
    except Exception as e:
        logger.warning(f"PDP/ICE export failed: {e}. Skipping.")


# ── 7. KM Curve Data ──
def export_km_curves() -> None:
    """Kaplan-Meier survival curves by grade."""
    logger.info("Exporting KM curve data...")

    try:
        from lifelines import KaplanMeierFitter

        df = pd.read_parquet(DATA_DIR / "loan_master.parquet")

        # Attempt to load raw data for last_pymnt_d
        raw_path = RAW_DIR / "Loan_status_2007-2020Q3.csv"
        if raw_path.exists():
            logger.info("Loading raw CSV for payment dates...")
            raw = pd.read_csv(raw_path, usecols=["id", "last_pymnt_d"], low_memory=False)
            raw["id"] = raw["id"].astype(str)
            df["id"] = df["id"].astype(str)
            df = df.merge(raw[["id", "last_pymnt_d"]], on="id", how="left")
            df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
            df["time_months"] = ((df["last_pymnt_d"] - df["issue_d"]).dt.days / 30.44).clip(lower=1)
            df["time_months"] = df["time_months"].fillna(
                df["term"].map({" 36 months": 36, " 60 months": 60, 36: 36, 60: 60}).fillna(36)
            )
        else:
            logger.warning("Raw CSV not found, using term as proxy for duration.")
            df["time_months"] = (
                df["term"].map({" 36 months": 36, " 60 months": 60, 36: 36, 60: 60}).fillna(36)
            )

        kmf = KaplanMeierFitter()
        rows = []

        for grade in sorted(df["grade"].dropna().unique()):
            mask = df["grade"] == grade
            T = df.loc[mask, "time_months"].values
            E = df.loc[mask, "default_flag"].values.astype(int)

            # Clip extreme values
            T = np.clip(T, 1, 60)

            kmf.fit(T, event_observed=E, label=grade)
            timeline = kmf.survival_function_.index.values
            survival = kmf.survival_function_.iloc[:, 0].values
            ci_low = kmf.confidence_interval_survival_function_.iloc[:, 0].values
            ci_high = kmf.confidence_interval_survival_function_.iloc[:, 1].values

            for t, s, cl, ch in zip(timeline, survival, ci_low, ci_high, strict=False):
                rows.append(
                    {
                        "grade": grade,
                        "timeline": round(float(t), 2),
                        "survival_prob": round(float(s), 6),
                        "ci_low": round(float(cl), 6),
                        "ci_high": round(float(ch), 6),
                    }
                )

        _save_parquet(pd.DataFrame(rows), DATA_DIR / "km_curve_data.parquet")

    except Exception as e:
        logger.warning(f"KM curve export failed: {e}. Skipping.")


# ── 8. Hazard Ratios ──
def export_hazard_ratios() -> None:
    """Cox PH hazard ratios for the Survival Analysis page."""
    logger.info("Exporting hazard ratios...")

    try:
        with open(MODEL_DIR / "cox_ph_model.pkl", "rb") as f:
            cph = pickle.load(f)

        summary = cph.summary
        hr_df = pd.DataFrame(
            {
                "feature": summary.index,
                "coef": summary["coef"].values,
                "exp_coef": summary["exp(coef)"].values,
                "se_coef": summary["se(coef)"].values,
                "z": summary["z"].values,
                "p_value": summary["p"].values,
                "neg_log2_p": summary["-log2(p)"].values,
                "ci_lower": summary["exp(coef) lower 95%"].values,
                "ci_upper": summary["exp(coef) upper 95%"].values,
            }
        )
        hr_df = hr_df.sort_values("neg_log2_p", ascending=False)
        _save_parquet(hr_df, DATA_DIR / "hazard_ratios.parquet")

    except Exception as e:
        logger.warning(f"Hazard ratio export failed: {e}. Skipping.")


# ── 9. Pipeline Summary (JSON) ──
def export_pipeline_summary() -> None:
    """Convert pipeline_results.pkl to JSON for the Executive Summary page."""
    logger.info("Exporting pipeline summary...")

    with open(MODEL_DIR / "pipeline_results.pkl", "rb") as f:
        results = pickle.load(f)

    # Load additional context
    with open(MODEL_DIR / "pd_training_record.pkl", "rb") as f:
        train_rec = pickle.load(f)

    with open(MODEL_DIR / "survival_summary.pkl", "rb") as f:
        surv = pickle.load(f)

    conformal_status = json.loads((MODEL_DIR / "conformal_policy_status.json").read_text())

    summary = {
        "pipeline": results,
        "pd_model": {
            "final_auc": train_rec.get("final_test_metrics", {}).get("auc_roc", 0),
            "final_gini": train_rec.get("final_test_metrics", {}).get("gini", 0),
            "final_brier": train_rec.get("final_test_metrics", {}).get("brier_score", 0),
            "final_ece": train_rec.get("final_test_metrics", {}).get("ece", 0),
            "calibration_method": train_rec.get("best_calibration", ""),
        },
        "conformal": {
            "coverage_90": conformal_status.get("coverage_90", 0),
            "coverage_95": conformal_status.get("coverage_95", 0),
            "overall_pass": conformal_status.get("overall_pass", False),
            "n_checks_passed": conformal_status.get("checks_passed", 0),
        },
        "survival": {
            "cox_concordance": surv.get("cox_concordance_index", 0),
            "rsf_concordance": surv.get("rsf_c_index_test", 0),
        },
        "dataset": {
            "n_loans": surv.get("n_loans", 0),
            "n_events": surv.get("n_events", 0),
            "event_rate": surv.get("event_rate", 0),
        },
    }
    _save_json(summary, DATA_DIR / "pipeline_summary.json")


# ── 10. Dataset Dictionary ──
def export_dataset_dictionary() -> None:
    """Static dataset variable dictionary for Data Story and Glossary pages."""
    logger.info("Exporting dataset dictionary...")
    path = DATA_DIR / "dataset_dictionary.json"
    if path.exists():
        logger.info(f"dataset_dictionary.json already exists ({path.stat().st_size} bytes)")
        return
    logger.warning("dataset_dictionary.json not found — create manually.")


# ── 11. Macro Context Timeline ──
def export_macro_context() -> None:
    """Static macro-economic timeline for temporal annotations."""
    logger.info("Exporting macro context timeline...")
    path = DATA_DIR / "macro_context.json"
    if path.exists():
        logger.info(f"macro_context.json already exists ({path.stat().st_size} bytes)")
        return
    logger.warning("macro_context.json not found — create manually.")


# ── 12. State-Level Aggregates ──
def export_state_aggregates() -> None:
    """State-level loan volume and default rates for choropleth maps."""
    logger.info("Exporting state-level aggregates...")

    interim_path = PROJECT_ROOT / "data" / "interim" / "lending_club_cleaned.parquet"
    if not interim_path.exists():
        logger.warning("lending_club_cleaned.parquet not found. Skipping state aggregates.")
        return

    df = pd.read_parquet(interim_path, columns=["addr_state", "default_flag", "loan_amnt"])
    state_agg = df.groupby("addr_state", as_index=False).agg(
        n_loans=("default_flag", "count"),
        default_rate=("default_flag", "mean"),
        total_volume=("loan_amnt", "sum"),
        avg_loan=("loan_amnt", "mean"),
    )
    state_agg["default_rate"] = state_agg["default_rate"].round(6)
    state_agg["avg_loan"] = state_agg["avg_loan"].round(2)
    _save_parquet(state_agg, DATA_DIR / "state_aggregates.parquet")


# ── 13. Historical ROI by Grade ──
def export_roi_by_grade() -> None:
    """Realized ROI from raw data for retrospective investor analysis."""
    logger.info("Exporting ROI by grade...")

    raw_path = RAW_DIR / "Loan_status_2007-2020Q3.csv"
    if not raw_path.exists():
        logger.warning("Raw CSV not found. Skipping ROI export.")
        return

    cols = ["id", "grade", "term", "loan_status", "funded_amnt", "total_pymnt", "int_rate"]
    df = pd.read_csv(raw_path, usecols=cols, low_memory=False)

    # Keep only terminal loans
    terminal = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(terminal)].copy()
    df["default_flag"] = (df["loan_status"] != "Fully Paid").astype(int)
    df["roi"] = (df["total_pymnt"] - df["funded_amnt"]) / df["funded_amnt"]

    roi_grade = df.groupby("grade", as_index=False).agg(
        n_loans=("roi", "count"),
        default_rate=("default_flag", "mean"),
        roi_mean=("roi", "mean"),
        roi_median=("roi", "median"),
        roi_p10=("roi", lambda x: x.quantile(0.10)),
        roi_p90=("roi", lambda x: x.quantile(0.90)),
        avg_funded=("funded_amnt", "mean"),
        avg_received=("total_pymnt", "mean"),
    )
    for c in ["default_rate", "roi_mean", "roi_median", "roi_p10", "roi_p90"]:
        roi_grade[c] = roi_grade[c].round(6)
    roi_grade["avg_funded"] = roi_grade["avg_funded"].round(2)
    roi_grade["avg_received"] = roi_grade["avg_received"].round(2)

    roi_term = df.groupby(["grade", "term"], as_index=False).agg(
        n_loans=("roi", "count"),
        default_rate=("default_flag", "mean"),
        roi_mean=("roi", "mean"),
    )
    for c in ["default_rate", "roi_mean"]:
        roi_term[c] = roi_term[c].round(6)

    _save_parquet(roi_grade, DATA_DIR / "roi_by_grade.parquet")
    _save_parquet(roi_term, DATA_DIR / "roi_by_grade_term.parquet")


# ── 14. Runtime Status Snapshot ──
def export_runtime_status() -> None:
    """Export runtime status used by narrative/UI pages to avoid stale hardcodes."""
    logger.info("Exporting runtime status snapshot...")
    test_total, test_breakdown = _collect_test_inventory()
    pages_total = len(list((PROJECT_ROOT / "streamlit_app" / "pages").glob("*.py")))
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "test_suite_total": int(test_total),
        "test_breakdown": test_breakdown,
        "streamlit_pages_total": int(pages_total),
    }
    _save_json(payload, DATA_DIR / "runtime_status.json")


# ── Main ──
def main() -> None:
    logger.info("Starting Streamlit artifact export...")

    export_eda_summary()
    export_feature_importance_iv()
    export_model_comparison()
    export_roc_curves()
    export_calibration_curves()
    export_shap_summary()
    export_permutation_importance()
    export_pdp_ice_top5()
    export_km_curves()
    export_hazard_ratios()
    export_pipeline_summary()
    export_dataset_dictionary()
    export_macro_context()
    export_state_aggregates()
    export_roi_by_grade()
    export_runtime_status()

    logger.success(f"Export complete! {EXPORT_COUNT} artifacts generated.")


if __name__ == "__main__":
    main()
