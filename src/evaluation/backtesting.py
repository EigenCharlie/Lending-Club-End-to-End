"""Backtesting and out-of-time validation for credit risk models.

Includes:
- Cohort analysis across vintage periods.
- Population Stability Index (PSI) for distribution drift.
- Kupiec (1995) Proportion of Failures (POF) test for unconditional coverage.
- Christoffersen (1998) test for conditional coverage and independence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluation.coverage_tests import christoffersen_test, kupiec_pof_test
from src.evaluation.metrics import classification_metrics


def cohort_analysis(
    df: pd.DataFrame,
    y_true_col: str = "default_flag",
    y_prob_col: str = "pd_predicted",
    cohort_col: str = "issue_quarter",
) -> pd.DataFrame:
    """Evaluate model performance across vintage cohorts."""
    results = []
    for cohort, group in df.groupby(cohort_col):
        if len(group) < 50:
            continue
        metrics = classification_metrics(
            group[y_true_col].values,
            group[y_prob_col].values,
        )
        metrics["cohort"] = cohort
        metrics["n_loans"] = len(group)
        metrics["default_rate"] = group[y_true_col].mean()
        results.append(metrics)

    result = pd.DataFrame(results)
    logger.info(f"Cohort analysis: {len(result)} cohorts evaluated")
    return result


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute PSI to detect distribution drift between train and test."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[np.isfinite(expected_arr)]
    actual_arr = actual_arr[np.isfinite(actual_arr)]
    if expected_arr.size == 0 or actual_arr.size == 0:
        return 0.0

    bin_edges = np.percentile(expected_arr, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        # Degenerate feature with near-constant values.
        return 0.0
    bin_edges[-1] += 1e-6

    expected_pct = np.histogram(expected_arr, bins=bin_edges)[0] / len(expected_arr)
    actual_pct = np.histogram(actual_arr, bins=bin_edges)[0] / len(actual_arr)

    # Avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    logger.info(
        f"PSI = {psi:.4f} ({'stable' if psi < 0.1 else 'drift detected' if psi < 0.25 else 'significant drift'})"
    )
    return psi


def ks_two_sample_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test for train-vs-test drift."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[np.isfinite(expected_arr)]
    actual_arr = actual_arr[np.isfinite(actual_arr)]

    if expected_arr.size == 0 or actual_arr.size == 0:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0}

    stat, pvalue = stats.ks_2samp(expected_arr, actual_arr)
    return {"ks_statistic": float(stat), "ks_pvalue": float(pvalue)}


def cramervonmises_two_sample_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """Two-sample Cramér-von Mises test for train-vs-test drift."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[np.isfinite(expected_arr)]
    actual_arr = actual_arr[np.isfinite(actual_arr)]

    if expected_arr.size == 0 or actual_arr.size == 0:
        return {"cvm_statistic": 0.0, "cvm_pvalue": 1.0}

    result = stats.cramervonmises_2samp(expected_arr, actual_arr)
    return {"cvm_statistic": float(result.statistic), "cvm_pvalue": float(result.pvalue)}


def classifier_two_sample_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    *,
    max_rows_per_split: int = 50_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Classifier Two-Sample Test (C2ST) / adversarial validation.

    Returns AUC of a binary classifier trying to distinguish train rows (label 0)
    from test rows (label 1). Values near 0.5 indicate low distribution shift.
    """
    if not features:
        return {"c2st_auc": 0.5, "n_rows": 0}

    train = train_df[features].copy()
    test = test_df[features].copy()

    if len(train) > max_rows_per_split:
        train = train.sample(n=max_rows_per_split, random_state=random_state)
    if len(test) > max_rows_per_split:
        test = test.sample(n=max_rows_per_split, random_state=random_state)

    train["__c2st_label"] = 0
    test["__c2st_label"] = 1
    all_df = pd.concat([train, test], axis=0, ignore_index=True)
    all_df = all_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    y = all_df.pop("__c2st_label").to_numpy(dtype=int)
    X = all_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if X.shape[0] < 200 or X.shape[1] == 0:
        return {"c2st_auc": 0.5, "n_rows": int(X.shape[0])}

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=250,
        learning_rate=0.05,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    return {"c2st_auc": float(auc), "n_rows": int(X.shape[0])}


def drift_monitoring_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    *,
    psi_threshold: float = 0.25,
    ks_pvalue_threshold: float = 0.01,
    cvm_pvalue_threshold: float = 0.01,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build per-feature drift monitoring table.

    Produces PSI + KS + CvM metrics and per-metric pass/fail flags.
    """
    rows: list[dict[str, float | int | str | bool]] = []
    for feature in features:
        if feature not in train_df.columns or feature not in test_df.columns:
            continue

        tr = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        te = pd.to_numeric(test_df[feature], errors="coerce").to_numpy(dtype=float)

        tr = tr[np.isfinite(tr)]
        te = te[np.isfinite(te)]
        if tr.size < 30 or te.size < 30:
            continue

        psi = population_stability_index(tr, te, n_bins=n_bins)
        ks = ks_two_sample_test(tr, te)
        cvm = cramervonmises_two_sample_test(tr, te)

        rows.append(
            {
                "feature": feature,
                "train_n": int(tr.size),
                "test_n": int(te.size),
                "psi": float(psi),
                "ks_statistic": float(ks["ks_statistic"]),
                "ks_pvalue": float(ks["ks_pvalue"]),
                "cvm_statistic": float(cvm["cvm_statistic"]),
                "cvm_pvalue": float(cvm["cvm_pvalue"]),
                "pass_psi": bool(psi <= psi_threshold),
                "pass_ks": bool(ks["ks_pvalue"] >= ks_pvalue_threshold),
                "pass_cvm": bool(cvm["cvm_pvalue"] >= cvm_pvalue_threshold),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "train_n",
                "test_n",
                "psi",
                "ks_statistic",
                "ks_pvalue",
                "cvm_statistic",
                "cvm_pvalue",
                "pass_psi",
                "pass_ks",
                "pass_cvm",
            ]
        )
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


def filter_high_psi_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    psi_threshold: float = 0.25,
    n_bins: int = 10,
) -> dict[str, list[str] | pd.DataFrame]:
    """Identify and filter features with high PSI (distribution drift).

    Equivalent to feature-engine's DropHighPSIFeatures but using the
    project's existing PSI implementation.

    Args:
        train_df: Training data (reference distribution).
        test_df: Test/production data (current distribution).
        features: List of numeric feature names to evaluate.
        psi_threshold: Features with PSI above this are flagged for removal.
        n_bins: Number of bins for PSI computation.

    Returns:
        Dict with:
        - 'stable_features': Features with PSI <= threshold.
        - 'drifted_features': Features with PSI > threshold.
        - 'psi_table': DataFrame with per-feature PSI values.
    """
    psi_records: list[dict[str, float | str]] = []
    for feat in features:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[feat], errors="coerce").to_numpy(dtype=float)
        te = pd.to_numeric(test_df[feat], errors="coerce").to_numpy(dtype=float)
        tr = tr[np.isfinite(tr)]
        te = te[np.isfinite(te)]
        if tr.size < 30 or te.size < 30:
            psi_records.append({"feature": feat, "psi": 0.0, "stable": True})
            continue
        psi = population_stability_index(tr, te, n_bins=n_bins)
        psi_records.append(
            {
                "feature": feat,
                "psi": float(psi),
                "stable": bool(psi <= psi_threshold),
            }
        )

    psi_table = pd.DataFrame(psi_records).sort_values("psi", ascending=False).reset_index(drop=True)
    stable = [r["feature"] for r in psi_records if r["stable"]]
    drifted = [r["feature"] for r in psi_records if not r["stable"]]

    if drifted:
        logger.warning(
            f"PSI filter: {len(drifted)} features drifted (PSI > {psi_threshold}): {drifted}"
        )
    else:
        logger.info(f"PSI filter: all {len(stable)} features stable (PSI <= {psi_threshold})")

    return {"stable_features": stable, "drifted_features": drifted, "psi_table": psi_table}


def interval_violations(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return binary violation indicators (1 if y_true outside [lower, upper])."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    outside = (y < lo) | (y > hi)
    return outside.astype(float)


def winkler_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute Winkler interval score per observation.

    Lower is better. Inside-interval score is width; outside gets linear penalty.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    widths = np.maximum(0.0, hi - lo)

    score = widths.copy()
    below = y < lo
    above = y > hi
    penalty_scale = 2.0 / max(float(alpha), 1e-8)

    score[below] = widths[below] + penalty_scale * (lo[below] - y[below])
    score[above] = widths[above] + penalty_scale * (y[above] - hi[above])
    return score


__all__ = [
    "cohort_analysis",
    "population_stability_index",
    "ks_two_sample_test",
    "cramervonmises_two_sample_test",
    "classifier_two_sample_test",
    "drift_monitoring_report",
    "interval_violations",
    "winkler_interval_score",
    "kupiec_pof_test",
    "christoffersen_test",
]
