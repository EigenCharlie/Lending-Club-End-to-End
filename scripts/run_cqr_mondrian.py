"""CQR Mondrian Hybrid — grade-conditional asymmetric conformal intervals.

Combines the best of two worlds:
- CQR (Conformalized Quantile Regression): asymmetric intervals adapting
  to local uncertainty (not symmetric around prediction)
- Mondrian (group-conditional): separate calibration per grade group,
  guaranteeing conditional coverage P(Y ∈ C(X) | grade=g) ≥ 1-alpha

Standard CQR: fit global quantile regressors, conformalize globally.
CQR Mondrian: fit global quantile regressors, conformalize PER GRADE.

This produces grade-specific conformalization offsets for the asymmetric
quantile bounds — tighter intervals for low-risk grades (A, B) and wider
for high-risk grades (E, F, G).

Usage:
    uv run python scripts/run_cqr_mondrian.py
"""

from __future__ import annotations

import json
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
ALPHA = 0.10
LOW_RISK_THRESHOLD = 0.10
N_CAL_TRAIN_SPLIT = 0.5  # fraction of cal set used to train quantile regressors


def _load_data() -> tuple:
    """Load calibration and test FE data with feature type handling."""
    model_path = Path("models/pd_canonical.cbm")
    contract_path = Path("models/pd_model_contract.json")
    cal_path = Path("data/processed/calibration_fe.parquet")
    test_path = Path("data/processed/test_fe.parquet")

    for p in [model_path, contract_path, cal_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    from catboost import CatBoostClassifier

    clf = CatBoostClassifier()
    clf.load_model(str(model_path))

    with open(contract_path) as f:
        contract = json.load(f)
    feature_cols: list[str] = contract["feature_names"]

    cat_idx = set(clf.get_cat_feature_indices())
    cat_names = {feature_cols[i] for i in cat_idx if i < len(feature_cols)}
    num_cols = [c for c in feature_cols if c not in cat_names]

    cal = pd.read_parquet(cal_path)
    test = pd.read_parquet(test_path)
    logger.info("Loaded cal ({:,}) test ({:,})", len(cal), len(test))

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
    grade_cal = cal["grade"].fillna("UNKNOWN").astype(str)
    grade_test = test["grade"].fillna("UNKNOWN").astype(str)

    # Numeric-only features for quantile regressors (MAPIE converts to numpy)
    X_cal_num = cal[num_cols].copy()
    X_test_num = test[num_cols].copy()
    for col in num_cols:
        X_cal_num[col] = pd.to_numeric(X_cal_num[col], errors="coerce").fillna(0.0)
        X_test_num[col] = pd.to_numeric(X_test_num[col], errors="coerce").fillna(0.0)

    return clf, X_cal, X_test, y_cal, y_test, grade_cal, grade_test, X_cal_num, X_test_num, num_cols


def _fit_quantile_regressors(
    X_train_num: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    n_trees: int = 200,
) -> tuple:
    """Fit lower and upper quantile regressors on numeric features."""
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

    def make_qreg(q: float) -> CatBoostRegressor:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            iterations=n_trees,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )

    logger.info("Fitting quantile regressor q={:.3f} on {:,} samples...", q_lo, len(X_train_num))
    qreg_lo = make_qreg(q_lo)
    qreg_lo.fit(X_train_num, y_train)

    logger.info("Fitting quantile regressor q={:.3f}...", q_hi)
    qreg_hi = make_qreg(q_hi)
    qreg_hi.fit(X_train_num, y_train)

    return qreg_lo, qreg_hi


def _cqr_global(
    qreg_lo: CatBoostRegressor,
    qreg_hi: CatBoostRegressor,
    X_cal_num: pd.DataFrame,
    y_cal: pd.Series,
    X_test_num: pd.DataFrame,
    alpha: float,
) -> np.ndarray:
    """Standard CQR: global calibration of residuals."""
    q_lo_pred = qreg_lo.predict(X_cal_num)
    q_hi_pred = qreg_hi.predict(X_cal_num)
    y_cal_arr = np.asarray(y_cal, dtype=float)

    # CQR nonconformity scores
    scores = np.maximum(q_lo_pred - y_cal_arr, y_cal_arr - q_hi_pred)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    offset = float(np.quantile(scores, q_level, method="higher"))

    lo_pred = qreg_lo.predict(X_test_num) - offset
    hi_pred = qreg_hi.predict(X_test_num) + offset
    return np.column_stack([np.clip(lo_pred, 0, 1), np.clip(hi_pred, 0, 1)])


def _cqr_mondrian(
    qreg_lo: CatBoostRegressor,
    qreg_hi: CatBoostRegressor,
    X_cal_num: pd.DataFrame,
    y_cal: pd.Series,
    X_test_num: pd.DataFrame,
    grade_cal: pd.Series,
    grade_test: pd.Series,
    alpha: float,
    min_group_size: int = 500,
) -> tuple[np.ndarray, dict]:
    """CQR Mondrian: grade-conditional calibration of quantile residuals.

    Fits global quantile regressors but computes conformalization offset
    separately per grade group (Mondrian partition).

    Nonconformity score: max(q_lo(x) - y, y - q_hi(x))
    Per-grade offset: (n_g+1)(1-alpha)/n_g quantile of group scores.
    """
    q_lo_pred_cal = qreg_lo.predict(X_cal_num)
    q_hi_pred_cal = qreg_hi.predict(X_cal_num)
    y_cal_arr = np.asarray(y_cal, dtype=float)
    g_cal = np.asarray(grade_cal, dtype=str)
    g_test = np.asarray(grade_test, dtype=str)

    scores = np.maximum(q_lo_pred_cal - y_cal_arr, y_cal_arr - q_hi_pred_cal)

    # Global fallback quantile
    n_global = len(scores)
    q_lev_global = np.ceil((n_global + 1) * (1 - alpha)) / n_global
    global_offset = float(np.quantile(scores, min(q_lev_global, 1.0), method="higher"))

    # Per-grade offsets
    all_grades = sorted(set(g_cal).union(set(g_test)))
    group_offsets: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    fallback_groups: list[str] = []

    for g in all_grades:
        mask = g_cal == g
        n_g = int(mask.sum())
        group_counts[g] = n_g
        if n_g >= min_group_size:
            q_lev = np.ceil((n_g + 1) * (1 - alpha)) / n_g
            group_offsets[g] = float(np.quantile(scores[mask], min(q_lev, 1.0), method="higher"))
        else:
            group_offsets[g] = global_offset
            fallback_groups.append(g)

    logger.info(
        "CQR Mondrian offsets: {} groups, {} fallbacks | offsets: {}",
        len(all_grades),
        len(fallback_groups),
        {g: f"{v:.4f}" for g, v in sorted(group_offsets.items())},
    )

    # Apply per-group offsets to test predictions
    q_lo_test = qreg_lo.predict(X_test_num)
    q_hi_test = qreg_hi.predict(X_test_num)
    offsets_test = np.array([group_offsets.get(str(g), global_offset) for g in g_test])

    lo_final = np.clip(q_lo_test - offsets_test, 0.0, 1.0)
    hi_final = np.clip(q_hi_test + offsets_test, 0.0, 1.0)
    intervals = np.column_stack([lo_final, hi_final])

    diag = {
        "global_offset": global_offset,
        "group_offsets": group_offsets,
        "group_counts": group_counts,
        "fallback_groups": fallback_groups,
        "avg_width": float((hi_final - lo_final).mean()),
    }
    return intervals, diag


def _coverage_report(
    y_test: np.ndarray,
    intervals: np.ndarray,
    grade_test: np.ndarray,
    method: str,
    alpha: float,
) -> dict:
    covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
    widths = intervals[:, 1] - intervals[:, 0]
    group_cov: dict[str, float] = {}
    group_width: dict[str, float] = {}
    for g in sorted(set(grade_test)):
        mask = grade_test == g
        if mask.sum() > 0:
            group_cov[g] = float(covered[mask].mean())
            group_width[g] = float(widths[mask].mean())
    return {
        "method": method,
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float((1 - alpha) - covered.mean()),
        "min_group_coverage": float(min(group_cov.values())) if group_cov else float("nan"),
        "avg_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "std_width": float(widths.std()),
        "n_eligible": int((intervals[:, 1] < LOW_RISK_THRESHOLD).sum()),
        "eligible_pct": float((intervals[:, 1] < LOW_RISK_THRESHOLD).mean()),
        "group_coverage": group_cov,
        "group_avg_width": group_width,
    }


def main() -> int:
    run_tag = resolve_run_tag(None, allow_untracked=True)
    logger.info("CQR Mondrian hybrid starting | run_tag={}", run_tag)

    (clf, X_cal, X_test, y_cal, y_test, grade_cal, grade_test, X_cal_num, X_test_num, num_cols) = (
        _load_data()
    )

    y_test_arr = np.asarray(y_test, dtype=float)
    g_test_arr = np.asarray(grade_test, dtype=str)

    # Split calibration set: first half trains quantile regressors, second half for conformalization
    n_cal = len(X_cal)
    split = int(n_cal * N_CAL_TRAIN_SPLIT)
    idx = np.arange(n_cal)
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, conf_idx = idx[:split], idx[split:]

    X_qr_train = X_cal_num.iloc[train_idx]
    y_qr_train = y_cal.iloc[train_idx]
    X_qr_conf = X_cal_num.iloc[conf_idx]
    y_qr_conf = y_cal.iloc[conf_idx]
    grade_conf = grade_cal.iloc[conf_idx]

    logger.info(
        "QR train: {:,} | Conformalize: {:,} | Test: {:,}",
        len(X_qr_train),
        len(X_qr_conf),
        len(X_test),
    )

    # Fit quantile regressors once (shared by CQR global and CQR Mondrian)
    qreg_lo, qreg_hi = _fit_quantile_regressors(X_qr_train, y_qr_train, ALPHA)

    results: list[dict] = []
    intervals_dict: dict[str, np.ndarray] = {}

    # ── Method 1: Symmetric global (reference) ───────────────────────────────
    logger.info("=== Method 1: Symmetric global split-conformal ===")
    y_pred_g, y_int_g = create_pd_intervals(clf, X_cal, y_cal, X_test, alpha=ALPHA)
    r = _coverage_report(y_test_arr, y_int_g, g_test_arr, "symmetric_global", ALPHA)
    results.append({k: v for k, v in r.items() if k not in ("group_coverage", "group_avg_width")})
    intervals_dict["symmetric_global"] = y_int_g
    logger.info(
        "  cov={:.4f}, width={:.4f}, min_grp={:.4f}",
        r["empirical_coverage"],
        r["avg_width"],
        r["min_group_coverage"],
    )

    # ── Method 2: Symmetric Mondrian (reference) ─────────────────────────────
    logger.info("=== Method 2: Symmetric Mondrian split-conformal ===")
    y_pred_m, y_int_m, _ = create_pd_intervals_mondrian(
        clf, X_cal, y_cal, X_test, grade_cal, grade_test, alpha=ALPHA
    )
    r = _coverage_report(y_test_arr, y_int_m, g_test_arr, "symmetric_mondrian", ALPHA)
    results.append({k: v for k, v in r.items() if k not in ("group_coverage", "group_avg_width")})
    intervals_dict["symmetric_mondrian"] = y_int_m
    logger.info(
        "  cov={:.4f}, width={:.4f}, min_grp={:.4f}",
        r["empirical_coverage"],
        r["avg_width"],
        r["min_group_coverage"],
    )

    # ── Method 3: CQR Global ──────────────────────────────────────────────────
    logger.info("=== Method 3: CQR Global (asymmetric, global calibration) ===")
    y_int_cqr_g = _cqr_global(qreg_lo, qreg_hi, X_qr_conf, y_qr_conf, X_test_num, ALPHA)
    r = _coverage_report(y_test_arr, y_int_cqr_g, g_test_arr, "cqr_global", ALPHA)
    results.append({k: v for k, v in r.items() if k not in ("group_coverage", "group_avg_width")})
    intervals_dict["cqr_global"] = y_int_cqr_g
    logger.info(
        "  cov={:.4f}, width={:.4f}, min_grp={:.4f}",
        r["empirical_coverage"],
        r["avg_width"],
        r["min_group_coverage"],
    )

    # ── Method 4: CQR Mondrian (asymmetric, grade-conditional) ───────────────
    logger.info("=== Method 4: CQR Mondrian (asymmetric + grade-conditional) ===")
    y_int_cqr_m, diag_m = _cqr_mondrian(
        qreg_lo,
        qreg_hi,
        X_qr_conf,
        y_qr_conf,
        X_test_num,
        grade_conf,
        grade_test,
        ALPHA,
    )
    r = _coverage_report(y_test_arr, y_int_cqr_m, g_test_arr, "cqr_mondrian", ALPHA)
    results.append({k: v for k, v in r.items() if k not in ("group_coverage", "group_avg_width")})
    intervals_dict["cqr_mondrian"] = y_int_cqr_m
    logger.info(
        "  cov={:.4f}, width={:.4f}, min_grp={:.4f}",
        r["empirical_coverage"],
        r["avg_width"],
        r["min_group_coverage"],
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(results)
    comp_path = Path("data/processed/cqr_mondrian_comparison.parquet")
    comparison_df.to_parquet(comp_path, index=False)
    logger.info("Saved comparison: {}", comp_path)

    # Save CQR Mondrian intervals
    cqr_m_df = pd.DataFrame(
        {
            "pd_low_cqr_mondrian_90": y_int_cqr_m[:, 0],
            "pd_high_cqr_mondrian_90": y_int_cqr_m[:, 1],
            "pd_width_cqr_mondrian_90": y_int_cqr_m[:, 1] - y_int_cqr_m[:, 0],
            "grade": grade_test.values,
        }
    )
    int_path = Path("data/processed/conformal_intervals_cqr_mondrian.parquet")
    cqr_m_df.to_parquet(int_path, index=False)
    logger.info("Saved CQR Mondrian intervals: {}", int_path)

    # Per-group coverage for all methods
    group_cov_records: list[dict] = []
    for method_name, intervals in intervals_dict.items():
        covered = (y_test_arr >= intervals[:, 0]) & (y_test_arr <= intervals[:, 1])
        widths = intervals[:, 1] - intervals[:, 0]
        for g in sorted(set(g_test_arr)):
            mask = g_test_arr == g
            if mask.sum() > 0:
                group_cov_records.append(
                    {
                        "method": method_name,
                        "grade": g,
                        "n": int(mask.sum()),
                        "coverage": float(covered[mask].mean()),
                        "avg_width": float(widths[mask].mean()),
                    }
                )
    group_df = pd.DataFrame(group_cov_records)
    grp_path = Path("data/processed/cqr_mondrian_group_coverage.parquet")
    group_df.to_parquet(grp_path, index=False)
    logger.info("Saved per-group coverage: {}", grp_path)

    # Status JSON
    metadata = build_artifact_metadata(
        schema_version=SCHEMA_VERSION,
        run_tag=run_tag,
        allow_untracked=True,
        extra={"alpha": ALPHA, "n_numeric_features": len(num_cols)},
    )
    status: dict[str, object] = {
        **metadata,
        "alpha": ALPHA,
        "n_test": int(len(y_test)),
        "cqr_mondrian_group_offsets": diag_m["group_offsets"],
        "methods": {r["method"]: r for r in results},
        "summary": {
            "best_coverage": max(results, key=lambda x: x["empirical_coverage"])["method"],
            "tightest": min(results, key=lambda x: x["avg_width"])["method"],
            "best_min_group_cov": max(results, key=lambda x: x["min_group_coverage"])["method"],
        },
    }
    status_path = Path("models/cqr_mondrian_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info("Saved status: {}", status_path)

    # Summary table
    logger.info("=== CQR Mondrian Comparison ===")
    display = comparison_df[
        ["method", "empirical_coverage", "min_group_coverage", "avg_width", "n_eligible"]
    ]
    logger.info("\n" + display.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
