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


def kupiec_pof_test(
    violations: np.ndarray,
    alpha: float,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    """Kupiec (1995) Proportion of Failures test for unconditional coverage.

    Tests H₀: the true violation rate equals the nominal rate α.
    Uses a likelihood-ratio statistic distributed as χ²(1).

    Args:
        violations: Binary array (1 = observation outside interval).
        alpha: Nominal violation rate (e.g., 0.10 for 90% coverage).
        confidence: Confidence level for the test (default 0.95).

    Returns:
        Dict with lr_statistic, p_value, reject, n_violations, n_total,
        violation_rate, nominal_alpha.
    """
    violations = np.asarray(violations, dtype=float)
    n_total = violations.size
    n_violations = int(violations.sum())

    if n_total == 0:
        return {
            "lr_statistic": 0.0,
            "p_value": 1.0,
            "reject": False,
            "n_violations": 0,
            "n_total": 0,
            "violation_rate": 0.0,
            "nominal_alpha": alpha,
        }

    p_hat = n_violations / n_total
    v = n_violations
    t = n_total

    # Avoid log(0) edge cases
    if p_hat == 0.0 or p_hat == 1.0:
        # Degenerate: if p_hat matches alpha exactly (unlikely), LR=0
        if abs(p_hat - alpha) < 1e-10:
            lr = 0.0
        else:
            # Use large LR to indicate strong rejection
            lr = 2 * t * abs(np.log(max(alpha, 1e-15)) - np.log(max(p_hat, 1e-15)))
    else:
        log_l0 = v * np.log(alpha) + (t - v) * np.log(1 - alpha)
        log_l1 = v * np.log(p_hat) + (t - v) * np.log(1 - p_hat)
        lr = -2 * (log_l0 - log_l1)

    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    reject = p_value < (1 - confidence)

    logger.info(
        f"Kupiec POF: violations={n_violations}/{n_total} "
        f"(rate={p_hat:.4f}, nominal={alpha:.4f}), "
        f"LR={lr:.4f}, p={p_value:.4f}, reject={reject}"
    )
    return {
        "lr_statistic": float(lr),
        "p_value": p_value,
        "reject": reject,
        "n_violations": n_violations,
        "n_total": n_total,
        "violation_rate": p_hat,
        "nominal_alpha": alpha,
    }


def christoffersen_test(
    violations: np.ndarray,
    alpha: float,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    """Christoffersen (1998) conditional coverage test.

    Combines the Kupiec unconditional coverage test with a test for
    independence of violations (no temporal clustering). Uses a joint
    likelihood-ratio statistic distributed as χ²(2).

    Args:
        violations: Binary array ordered by time (1 = violation).
        alpha: Nominal violation rate.
        confidence: Confidence level for the test.

    Returns:
        Dict with lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc,
        reject_uc, reject_ind, reject_cc, transition_matrix.
    """
    violations = np.asarray(violations, dtype=float)
    n_total = violations.size

    # Unconditional coverage component
    uc = kupiec_pof_test(violations, alpha, confidence)

    if n_total < 2:
        return {
            "lr_uc": uc["lr_statistic"],
            "p_uc": uc["p_value"],
            "reject_uc": uc["reject"],
            "lr_ind": 0.0,
            "p_ind": 1.0,
            "reject_ind": False,
            "lr_cc": uc["lr_statistic"],
            "p_cc": uc["p_value"],
            "reject_cc": uc["reject"],
            "transition_matrix": {"n00": 0, "n01": 0, "n10": 0, "n11": 0},
        }

    # Build transition counts
    v0 = violations[:-1]
    v1 = violations[1:]
    n00 = int(((v0 == 0) & (v1 == 0)).sum())
    n01 = int(((v0 == 0) & (v1 == 1)).sum())
    n10 = int(((v0 == 1) & (v1 == 0)).sum())
    n11 = int(((v0 == 1) & (v1 == 1)).sum())

    # Transition probabilities
    row0 = n00 + n01
    row1 = n10 + n11

    # Overall violation rate under independence (H₀)
    pi = (n01 + n11) / max(n_total - 1, 1)

    # Row-conditional violation rates (H₁)
    pi01 = n01 / row0 if row0 > 0 else 0.0
    pi11 = n11 / row1 if row1 > 0 else 0.0

    # Independence LR
    _eps = 1e-15

    def _safe_log(x: float) -> float:
        return np.log(max(x, _eps))

    # L(π) under independence
    log_l0_ind = 0.0
    if row0 > 0:
        log_l0_ind += n00 * _safe_log(1 - pi) + n01 * _safe_log(pi)
    if row1 > 0:
        log_l0_ind += n10 * _safe_log(1 - pi) + n11 * _safe_log(pi)

    # L(π01, π11) under dependence
    log_l1_ind = 0.0
    if row0 > 0:
        log_l1_ind += n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
    if row1 > 0:
        log_l1_ind += n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)

    lr_ind = -2 * (log_l0_ind - log_l1_ind)
    lr_ind = max(lr_ind, 0.0)  # Numerical safety

    p_ind = float(1 - stats.chi2.cdf(lr_ind, df=1))
    reject_ind = p_ind < (1 - confidence)

    # Joint conditional coverage: LR_cc = LR_uc + LR_ind
    lr_cc = uc["lr_statistic"] + lr_ind
    p_cc = float(1 - stats.chi2.cdf(lr_cc, df=2))
    reject_cc = p_cc < (1 - confidence)

    logger.info(
        f"Christoffersen: LR_uc={uc['lr_statistic']:.4f} (p={uc['p_value']:.4f}), "
        f"LR_ind={lr_ind:.4f} (p={p_ind:.4f}), "
        f"LR_cc={lr_cc:.4f} (p={p_cc:.4f}), reject_cc={reject_cc}"
    )
    return {
        "lr_uc": uc["lr_statistic"],
        "p_uc": uc["p_value"],
        "reject_uc": uc["reject"],
        "lr_ind": float(lr_ind),
        "p_ind": p_ind,
        "reject_ind": reject_ind,
        "lr_cc": float(lr_cc),
        "p_cc": p_cc,
        "reject_cc": reject_cc,
        "transition_matrix": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }
