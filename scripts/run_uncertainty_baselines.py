"""Compare conformal prediction intervals against classical uncertainty set methods.

Computes bootstrap, parametric (Gaussian), and ellipsoidal (grade-conditional)
intervals on the same test data, then benchmarks coverage, width, min-group
coverage, and a portfolio feasibility proxy against our canonical Mondrian
conformal intervals.

Paper Estrella artifact — see docs/backlog-papers-unified.md.

Usage:
    uv run python scripts/run_uncertainty_baselines.py
    uv run python scripts/run_uncertainty_baselines.py --bootstrap-B 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.conformal import apply_probability_calibrator
from src.models.pd_contract import (
    load_contract,
    resolve_calibrator_path,
    resolve_model_path,
)
from src.utils.artifact_metadata import build_artifact_metadata
from src.utils.io_utils import load_pickle_compat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2026-03-16.1"
TARGET_COL = "default_flag"
GROUP_COL = "grade"
NOMINAL_COVERAGE = 0.90
Z_90 = 1.6449  # z for 90% two-sided interval (5th / 95th percentile)
LOW_RISK_THRESHOLD = 0.10

# Paths
CONFORMAL_PATH = ROOT / "data" / "processed" / "conformal_intervals_mondrian.parquet"
TEST_PRED_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"
CALIBRATION_FE_PATH = ROOT / "data" / "processed" / "calibration_fe.parquet"
TEST_FE_PATH = ROOT / "data" / "processed" / "test_fe.parquet"
CONTRACT_PATH = ROOT / "models" / "pd_model_contract.json"

OUT_COMPARISON = ROOT / "data" / "processed" / "uncertainty_baselines_comparison.parquet"
OUT_BY_GRADE = ROOT / "data" / "processed" / "uncertainty_baselines_by_grade.parquet"
OUT_STATUS = ROOT / "models" / "uncertainty_baselines_status.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Empirical coverage: fraction of y_true inside [low, high]."""
    return float(np.mean((y_true >= low) & (y_true <= high)))


def _avg_width(low: np.ndarray, high: np.ndarray) -> float:
    """Mean interval width."""
    return float(np.mean(high - low))


def _min_group_coverage(
    y_true: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    grades: np.ndarray,
) -> float:
    """Worst-case group coverage across grades."""
    coverages = []
    for g in np.unique(grades):
        mask = grades == g
        if mask.sum() == 0:
            continue
        coverages.append(_coverage(y_true[mask], low[mask], high[mask]))
    return float(min(coverages)) if coverages else 0.0


def _fundable_count(high: np.ndarray, threshold: float = LOW_RISK_THRESHOLD) -> int:
    """Number of loans whose upper PD bound is below the threshold."""
    return int(np.sum(high < threshold))


def _coverage_by_grade(
    y_true: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    grades: np.ndarray,
) -> dict[str, float]:
    """Per-grade coverage dict."""
    result: dict[str, float] = {}
    for g in sorted(np.unique(grades)):
        mask = grades == g
        result[str(g)] = _coverage(y_true[mask], low[mask], high[mask])
    return result


# ---------------------------------------------------------------------------
# Model / calibrator loading
# ---------------------------------------------------------------------------
def _load_model_and_calibrator() -> tuple[CatBoostClassifier, object | None]:
    """Load canonical PD model and calibrator."""
    model_path = resolve_model_path()
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"Loaded PD model: {model_path}")

    cal_path = resolve_calibrator_path()
    calibrator = None
    if cal_path is not None:
        calibrator = load_pickle_compat(cal_path)
        logger.info(f"Loaded calibrator: {type(calibrator).__name__}")
    else:
        logger.warning("No calibrator found. Using raw probabilities.")

    return model, calibrator


def _resolve_features(model: CatBoostClassifier) -> tuple[list[str], list[str]]:
    """Resolve feature names and categorical list from contract or model."""
    contract = load_contract(CONTRACT_PATH)
    if isinstance(contract, dict):
        feats = contract.get("feature_names", [])
        cats = contract.get("categorical_features", [])
        if feats:
            cats = [c for c in cats if c in feats]
            logger.info(f"Using {len(feats)} contract features ({len(cats)} categorical)")
            return list(feats), cats

    model_features = list(getattr(model, "feature_names_", []) or [])
    if model_features:
        cat_idxs = set(model.get_cat_feature_indices())
        cats = [f for i, f in enumerate(model_features) if i in cat_idxs]
        return model_features, cats

    raise ValueError("Cannot resolve feature list from contract or model metadata.")


def _build_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Build model matrix with stable order and consistent dtypes."""
    X = df.copy()
    cat_set = set(categorical)
    for col in features:
        if col not in X.columns:
            X[col] = "UNKNOWN" if col in cat_set else np.nan
    X = X[features].copy()
    for col in features:
        if col in cat_set:
            X[col] = X[col].astype("string").fillna("UNKNOWN").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _predict_calibrated(
    model: CatBoostClassifier,
    calibrator: object | None,
    X: pd.DataFrame,
) -> np.ndarray:
    """Get calibrated PD predictions."""
    raw = model.predict_proba(X)[:, 1]
    return apply_probability_calibrator(calibrator, raw)


# ---------------------------------------------------------------------------
# Interval constructors
# ---------------------------------------------------------------------------
def _conformal_intervals(
    conformal_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed Mondrian conformal intervals.

    Returns:
        (y_true, y_pred, pd_low, pd_high, grades)
    """
    y_true = conformal_df["y_true"].values
    y_pred = conformal_df["y_pred"].values
    pd_low = conformal_df["pd_low_90"].values
    pd_high = conformal_df["pd_high_90"].values
    grades = conformal_df["grade"].values
    return y_true, y_pred, pd_low, pd_high, grades


def _bootstrap_intervals(
    y_pred_test: np.ndarray,
    residuals_cal: np.ndarray,
    B: int = 200,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap residual intervals at 90% nominal.

    Resample calibration residuals B times, compute 5th and 95th percentile
    of bootstrap quantiles, then shift test predictions.
    """
    rng = np.random.default_rng(rng_seed)
    n_cal = len(residuals_cal)
    boot_q5 = np.empty(B)
    boot_q95 = np.empty(B)

    for b in range(B):
        sample = rng.choice(residuals_cal, size=n_cal, replace=True)
        boot_q5[b] = np.percentile(sample, 5)
        boot_q95[b] = np.percentile(sample, 95)

    q_low = float(np.mean(boot_q5))
    q_high = float(np.mean(boot_q95))
    logger.info(f"Bootstrap ({B} resamples): q5={q_low:.4f}, q95={q_high:.4f}")

    pd_low = np.clip(y_pred_test + q_low, 0.0, 1.0)
    pd_high = np.clip(y_pred_test + q_high, 0.0, 1.0)
    return pd_low, pd_high


def _parametric_intervals(
    y_pred_test: np.ndarray,
    residuals_cal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Parametric (Gaussian) intervals at 90% nominal.

    Assumes calibration residuals ~ Normal(mu, sigma).
    """
    mu = float(np.mean(residuals_cal))
    sigma = float(np.std(residuals_cal, ddof=1))
    logger.info(f"Parametric: mu={mu:.4f}, sigma={sigma:.4f}")

    pd_low = np.clip(y_pred_test + mu - Z_90 * sigma, 0.0, 1.0)
    pd_high = np.clip(y_pred_test + mu + Z_90 * sigma, 0.0, 1.0)
    return pd_low, pd_high


def _ellipsoidal_intervals(
    y_pred_test: np.ndarray,
    residuals_cal: np.ndarray,
    grades_cal: np.ndarray,
    grades_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ellipsoidal (grade-conditional parametric) intervals at 90% nominal.

    Computes per-grade sigma from calibration residuals, then applies
    grade-specific z * sigma bands. This is the 1D analogue of ellipsoidal
    uncertainty sets used in robust optimization.
    """
    pd_low = np.empty_like(y_pred_test)
    pd_high = np.empty_like(y_pred_test)

    all_grades = sorted(set(np.unique(grades_cal)) | set(np.unique(grades_test)))
    global_sigma = float(np.std(residuals_cal, ddof=1))

    for g in all_grades:
        mask_cal = grades_cal == g
        mask_test = grades_test == g
        if mask_cal.sum() < 10:
            sigma_g = global_sigma
            logger.warning(f"Grade {g}: <10 calibration samples, using global sigma")
        else:
            sigma_g = float(np.std(residuals_cal[mask_cal], ddof=1))

        if mask_test.sum() == 0:
            continue

        pd_low[mask_test] = np.clip(y_pred_test[mask_test] - Z_90 * sigma_g, 0.0, 1.0)
        pd_high[mask_test] = np.clip(y_pred_test[mask_test] + Z_90 * sigma_g, 0.0, 1.0)
        logger.debug(f"Grade {g}: sigma={sigma_g:.4f}, n_test={mask_test.sum()}")

    return pd_low, pd_high


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(bootstrap_B: int = 200, run_tag: str | None = None) -> None:
    """Run uncertainty baseline comparison."""
    logger.info("=== Uncertainty Baselines Comparison ===")

    # ------------------------------------------------------------------
    # 1. Load conformal intervals (baseline)
    # ------------------------------------------------------------------
    if not CONFORMAL_PATH.exists():
        logger.error(f"Conformal intervals not found: {CONFORMAL_PATH}")
        raise FileNotFoundError(CONFORMAL_PATH)
    conformal_df = pd.read_parquet(CONFORMAL_PATH)
    y_true_conf, y_pred_conf, low_conf, high_conf, grades_conf = _conformal_intervals(conformal_df)
    logger.info(f"Loaded conformal intervals: {len(conformal_df):,} rows")

    # ------------------------------------------------------------------
    # 2. Load model, calibrator, and data for non-conformal methods
    # ------------------------------------------------------------------
    model, calibrator = _load_model_and_calibrator()
    features, categorical = _resolve_features(model)

    cal_df = pd.read_parquet(CALIBRATION_FE_PATH)
    test_df = pd.read_parquet(TEST_FE_PATH)
    test_pred_df = pd.read_parquet(TEST_PRED_PATH)

    # Build calibrated predictions on calibration set
    X_cal = _build_feature_matrix(cal_df, features, categorical)
    y_cal = cal_df[TARGET_COL].values.astype(float)
    y_pred_cal = _predict_calibrated(model, calibrator, X_cal)

    # Calibrated predictions on test set
    X_test = _build_feature_matrix(test_df, features, categorical)
    y_pred_test = _predict_calibrated(model, calibrator, X_test)

    # True labels and grades for test set
    y_true_test = test_pred_df["y_true"].values.astype(float)
    grades_cal = cal_df[GROUP_COL].values
    grades_test = (
        test_df[GROUP_COL].values if GROUP_COL in test_df.columns else conformal_df["grade"].values
    )

    # Calibration residuals: y_actual - y_predicted
    residuals_cal = y_cal - y_pred_cal
    logger.info(
        f"Calibration residuals: mean={np.mean(residuals_cal):.4f}, "
        f"std={np.std(residuals_cal):.4f}, n={len(residuals_cal):,}"
    )

    # ------------------------------------------------------------------
    # 3. Compute intervals for each method
    # ------------------------------------------------------------------
    methods: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # 3a. Conformal (baseline — already computed)
    methods["conformal_mondrian"] = (y_true_conf, low_conf, high_conf, grades_conf)

    # 3b. Bootstrap
    low_boot, high_boot = _bootstrap_intervals(y_pred_test, residuals_cal, B=bootstrap_B)
    methods["bootstrap"] = (y_true_test, low_boot, high_boot, grades_test)

    # 3c. Parametric (Gaussian)
    low_param, high_param = _parametric_intervals(y_pred_test, residuals_cal)
    methods["parametric_gaussian"] = (y_true_test, low_param, high_param, grades_test)

    # 3d. Ellipsoidal (grade-conditional parametric)
    low_ellip, high_ellip = _ellipsoidal_intervals(
        y_pred_test, residuals_cal, grades_cal, grades_test
    )
    methods["ellipsoidal_grade"] = (y_true_test, low_ellip, high_ellip, grades_test)

    # ------------------------------------------------------------------
    # 4. Compute metrics per method
    # ------------------------------------------------------------------
    comparison_rows = []
    grade_rows = []

    for method_name, (y_true, low, high, grades) in methods.items():
        cov = _coverage(y_true, low, high)
        width = _avg_width(low, high)
        min_gcov = _min_group_coverage(y_true, low, high, grades)
        fundable = _fundable_count(high)

        comparison_rows.append(
            {
                "method": method_name,
                "nominal_coverage": NOMINAL_COVERAGE,
                "empirical_coverage": round(cov, 6),
                "avg_width": round(width, 6),
                "min_group_coverage": round(min_gcov, 6),
                "fundable_loans_pd_high_lt_010": fundable,
                "n_obs": len(y_true),
            }
        )
        logger.info(
            f"{method_name:25s} | cov={cov:.4f} | width={width:.4f} | "
            f"min_gcov={min_gcov:.4f} | fundable={fundable:,}"
        )

        # Per-grade coverage
        gcov = _coverage_by_grade(y_true, low, high, grades)
        for g, c in gcov.items():
            grade_rows.append(
                {
                    "method": method_name,
                    "grade": g,
                    "coverage": round(c, 6),
                }
            )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_parquet(OUT_COMPARISON, index=False)
    logger.info(f"Saved comparison table: {OUT_COMPARISON}")

    grade_df = pd.DataFrame(grade_rows)
    grade_df.to_parquet(OUT_BY_GRADE, index=False)
    logger.info(f"Saved per-grade coverage: {OUT_BY_GRADE}")

    # Status JSON
    meta = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={
            "methods": list(methods.keys()),
            "n_test": int(len(y_true_test)),
            "n_calibration": int(len(y_cal)),
            "bootstrap_B": bootstrap_B,
            "nominal_coverage": NOMINAL_COVERAGE,
        },
    )
    summary = {
        **meta,
        "results": {row["method"]: row for row in comparison_rows},
    }
    OUT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATUS.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved status JSON: {OUT_STATUS}")

    logger.info("=== Uncertainty baselines comparison complete ===")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare conformal intervals against classical uncertainty baselines."
    )
    parser.add_argument(
        "--bootstrap-B",
        type=int,
        default=200,
        help="Number of bootstrap resamples (default: 200).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Pipeline run tag for artifact metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(bootstrap_B=args.bootstrap_B, run_tag=args.run_tag)
