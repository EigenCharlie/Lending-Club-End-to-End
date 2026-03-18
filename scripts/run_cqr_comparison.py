"""CQR (Conformalized Quantile Regression) comparison for Paper Estrella.

Compares three conformal uncertainty methods for PD intervals:
1. Symmetric split-conformal (global, existing)
2. Mondrian split-conformal (grade-conditional, existing)
3. CQR — asymmetric intervals from quantile regression residuals

CQR uses MAPIE ConformalizedQuantileRegressor with CatBoost quantile models,
producing asymmetric [pd_low, pd_high] intervals that adapt to the local
uncertainty distribution, unlike symmetric split-conformal.

Usage:
    uv run python scripts/run_cqr_comparison.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from src.models.conformal import (
    create_pd_intervals,
    create_pd_intervals_mondrian,
)
from src.utils.artifact_metadata import build_artifact_metadata, resolve_run_tag

SCHEMA_VERSION = "2026-03-16.1"

ALPHA = 0.10  # Target: 90% coverage
LOW_RISK_THRESHOLD = 0.10
N_CAL_SPLIT = 0.5  # CQR needs its own cal split (from existing cal set)


def _load_model_and_data() -> tuple:
    """Load CatBoost model, contract, and processed datasets."""
    model_path = Path("models/pd_canonical.cbm")
    contract_path = Path("models/pd_model_contract.json")
    cal_path = Path("data/processed/calibration_fe.parquet")
    test_path = Path("data/processed/test_fe.parquet")

    for p in [model_path, contract_path, cal_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info("Loaded CatBoost classifier from {}", model_path)

    with open(contract_path) as f:
        contract = json.load(f)
    feature_cols: list[str] = contract["feature_names"]

    # Load calibrator (optional)
    cal_pkl = Path("models/pd_canonical_calibrator.pkl")
    calibrator = None
    if cal_pkl.exists():
        with open(cal_pkl, "rb") as f:
            calibrator = pickle.load(f)
        logger.info("Loaded calibrator from {}", cal_pkl)

    # Load data
    cal = pd.read_parquet(cal_path)
    test = pd.read_parquet(test_path)
    logger.info("Loaded cal ({:,}) and test ({:,}) rows", len(cal), len(test))

    # Type conversion using model's cat feature indices
    cat_idx = set(model.get_cat_feature_indices())
    cat_names = {feature_cols[i] for i in cat_idx if i < len(feature_cols)}

    def prep_X(df: pd.DataFrame) -> pd.DataFrame:
        X = df[feature_cols].copy()
        for col in feature_cols:
            if col in cat_names:
                X[col] = X[col].astype("string").fillna("UNKNOWN").astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    X_cal = prep_X(cal)
    X_test = prep_X(test)
    y_cal = cal["default_flag"].astype(float)
    y_test = test["default_flag"].astype(float)
    grade_cal = (
        cal["grade"].fillna("UNKNOWN").astype(str)
        if "grade" in cal.columns
        else pd.Series(["UNKNOWN"] * len(cal))
    )
    grade_test = (
        test["grade"].fillna("UNKNOWN").astype(str)
        if "grade" in test.columns
        else pd.Series(["UNKNOWN"] * len(test))
    )

    return (
        model,
        calibrator,
        feature_cols,
        cat_names,
        X_cal,
        y_cal,
        X_test,
        y_test,
        grade_cal,
        grade_test,
    )


def _run_cqr(
    model_cls,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_names: set[str],
    alpha: float = ALPHA,
) -> tuple[np.ndarray, dict]:
    """Run CQR using CatBoost quantile regressors + MAPIE.

    Splits the calibration set in two halves:
    - First half: fit quantile models (lower/upper quantile at alpha/2)
    - Second half: conformalize residuals

    Returns:
        y_intervals shape (n_test, 2), diagnostics dict
    """
    from mapie.regression import ConformalizedQuantileRegressor

    n_cal = len(X_cal)
    split = int(n_cal * N_CAL_SPLIT)
    idx = np.arange(n_cal)
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, cal_idx = idx[:split], idx[split:]

    X_cqr_train = X_cal.iloc[train_idx]
    y_cqr_train = y_cal.iloc[train_idx]
    X_cqr_cal = X_cal.iloc[cal_idx]
    y_cqr_cal = y_cal.iloc[cal_idx]

    logger.info(
        "CQR split: train_quantile={:,}, conformalize={:,}", len(X_cqr_train), len(X_cqr_cal)
    )

    # MAPIE converts X to numpy before calling predict — CatBoost cat_features must be absent.
    # Use numeric-only columns for CQR quantile regressors.
    num_cols = [c for c in X_cal.columns if c not in cat_names]
    logger.info(
        "CQR using {:,} numeric features (dropping {:,} cat)", len(num_cols), len(cat_names)
    )
    X_cqr_train_num = X_cqr_train[num_cols].fillna(0.0)
    X_cqr_cal_num = X_cqr_cal[num_cols].fillna(0.0)
    X_test_num = X_test[num_cols].fillna(0.0)

    # MAPIE CQR requires 3 quantile regressors: [alpha/2, 1-alpha/2, 0.5]
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    q_mid = 0.5

    def make_qreg(quantile: float) -> CatBoostRegressor:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )

    qreg_lo = make_qreg(q_lo)
    qreg_hi = make_qreg(q_hi)
    qreg_mid = make_qreg(q_mid)

    logger.info("Fitting lower quantile regressor (q={:.3f})...", q_lo)
    qreg_lo.fit(X_cqr_train_num, y_cqr_train)
    logger.info("Fitting upper quantile regressor (q={:.3f})...", q_hi)
    qreg_hi.fit(X_cqr_train_num, y_cqr_train)
    logger.info("Fitting median quantile regressor (q={:.3f})...", q_mid)
    qreg_mid.fit(X_cqr_train_num, y_cqr_train)

    # MAPIE CQR: pass list [alpha/2, 1-alpha/2, 0.5] in that order
    cqr = ConformalizedQuantileRegressor(
        estimator=[qreg_lo, qreg_hi, qreg_mid],
        confidence_level=1.0 - alpha,
        prefit=True,
    )
    cqr.conformalize(X_cqr_cal_num, y_cqr_cal)

    # Predict intervals on test set: returns (y_pred, y_intervals) where
    # y_intervals has shape (n_test, 2, n_alphas) — take [:, :, 0] for single alpha
    _, intervals_raw = cqr.predict_interval(X_test_num)
    lo = np.clip(intervals_raw[:, 0, 0], 0.0, 1.0)
    hi = np.clip(intervals_raw[:, 1, 0], 0.0, 1.0)
    y_intervals = np.column_stack([lo, hi])

    # Diagnostics
    covered = (y_test.to_numpy() >= y_intervals[:, 0]) & (y_test.to_numpy() <= y_intervals[:, 1])
    widths = y_intervals[:, 1] - y_intervals[:, 0]

    diag = {
        "empirical_coverage": float(covered.mean()),
        "avg_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "min_width": float(widths.min()),
        "max_width": float(widths.max()),
        "std_width": float(widths.std()),
        "n_eligible": int((y_intervals[:, 1] < LOW_RISK_THRESHOLD).sum()),
        "n_test": len(y_test),
    }
    logger.info(
        "CQR: coverage={:.4f}, avg_width={:.4f}, eligible={:,}",
        diag["empirical_coverage"],
        diag["avg_width"],
        diag["n_eligible"],
    )
    return y_intervals, diag


def _coverage_diagnostics(
    y_test: np.ndarray,
    y_intervals: np.ndarray,
    grade_test: np.ndarray,
    method: str,
) -> dict:
    """Compute coverage and per-group metrics."""
    covered = (y_test >= y_intervals[:, 0]) & (y_test <= y_intervals[:, 1])
    widths = y_intervals[:, 1] - y_intervals[:, 0]

    group_coverages: dict[str, float] = {}
    for g in sorted(set(grade_test)):
        mask = grade_test == g
        if mask.sum() > 0:
            group_coverages[g] = float(covered[mask].mean())

    return {
        "method": method,
        "empirical_coverage": float(covered.mean()),
        "min_group_coverage": float(min(group_coverages.values()))
        if group_coverages
        else float("nan"),
        "avg_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "std_width": float(widths.std()),
        "n_eligible": int((y_intervals[:, 1] < LOW_RISK_THRESHOLD).sum()),
        "n_test": len(y_test),
        "eligible_pct": float((y_intervals[:, 1] < LOW_RISK_THRESHOLD).mean()),
        "group_coverages": group_coverages,
    }


def main() -> int:
    """Run CQR vs symmetric conformal comparison."""
    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info("CQR comparison starting | run_tag={}", run_tag)

    (
        model,
        calibrator,
        feature_cols,
        cat_names,
        X_cal,
        y_cal,
        X_test,
        y_test,
        grade_cal,
        grade_test,
    ) = _load_model_and_data()

    y_test_arr = np.asarray(y_test, dtype=float)
    grade_test_arr = np.asarray(grade_test, dtype=str)

    results: list[dict] = []

    # ── Method 1: Global split-conformal ──────────────────────────────────────
    logger.info("=== Method 1: Global split-conformal ===")
    y_pred_global, y_int_global = create_pd_intervals(model, X_cal, y_cal, X_test, alpha=ALPHA)
    results.append(
        _coverage_diagnostics(y_test_arr, y_int_global, grade_test_arr, "global_splitconf")
    )

    # ── Method 2: Mondrian split-conformal ────────────────────────────────────
    logger.info("=== Method 2: Mondrian split-conformal ===")
    y_pred_mond, y_int_mond, diag_mond = create_pd_intervals_mondrian(
        model, X_cal, y_cal, X_test, grade_cal, grade_test, alpha=ALPHA
    )
    results.append(
        _coverage_diagnostics(y_test_arr, y_int_mond, grade_test_arr, "mondrian_splitconf")
    )

    # ── Method 3: CQR ─────────────────────────────────────────────────────────
    logger.info("=== Method 3: CQR (asymmetric quantile) ===")
    try:
        y_int_cqr, cqr_diag = _run_cqr(model, X_cal, y_cal, X_test, y_test, cat_names, ALPHA)
        results.append(
            _coverage_diagnostics(y_test_arr, y_int_cqr, grade_test_arr, "cqr_asymmetric")
        )
        cqr_status = "success"
    except Exception as exc:
        logger.error("CQR failed: {}", exc)
        y_int_cqr = None
        cqr_status = f"failed: {exc}"

    # ── Build comparison table ─────────────────────────────────────────────────
    comparison_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "group_coverages"} for r in results]
    )
    comparison_path = Path("data/processed/cqr_comparison.parquet")
    comparison_df.to_parquet(comparison_path, index=False)
    logger.info("Saved CQR comparison: {}", comparison_path)

    # Save intervals for CQR
    if y_int_cqr is not None:
        intervals_df = pd.DataFrame(
            {
                "pd_low_cqr_90": y_int_cqr[:, 0],
                "pd_high_cqr_90": y_int_cqr[:, 1],
                "pd_width_cqr_90": y_int_cqr[:, 1] - y_int_cqr[:, 0],
            }
        )
        cqr_intervals_path = Path("data/processed/conformal_intervals_cqr.parquet")
        intervals_df.to_parquet(cqr_intervals_path, index=False)
        logger.info("Saved CQR intervals: {}", cqr_intervals_path)

    # Build status JSON
    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={"alpha": ALPHA, "low_risk_threshold": LOW_RISK_THRESHOLD},
    )
    status = {
        **metadata,
        "alpha": ALPHA,
        "n_test": len(y_test),
        "cqr_status": cqr_status,
        "methods": {
            r["method"]: {k: v for k, v in r.items() if k not in ("group_coverages",)}
            for r in results
        },
        "per_group_coverage": {r["method"]: r.get("group_coverages", {}) for r in results},
        "summary": {
            "best_coverage_method": max(results, key=lambda x: x["empirical_coverage"])["method"],
            "tightest_method": min(results, key=lambda x: x["avg_width"])["method"],
            "most_eligible_method": max(results, key=lambda x: x["n_eligible"])["method"],
        },
    }
    status_path = Path("models/cqr_comparison_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    # Summary table
    logger.info("=== CQR Comparison Summary ===")
    logger.info(
        "\n"
        + comparison_df[
            ["method", "empirical_coverage", "min_group_coverage", "avg_width", "n_eligible"]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
