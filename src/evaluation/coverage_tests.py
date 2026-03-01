"""Statistical coverage tests for conformal prediction intervals.

Includes:
- Kupiec (1995) Proportion of Failures (POF) test for unconditional coverage.
- Christoffersen (1998) test for conditional coverage and independence.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy import stats


def kupiec_pof_test(
    violations: np.ndarray,
    alpha: float,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    """Kupiec (1995) Proportion of Failures test for unconditional coverage.

    Tests H0: the true violation rate equals the nominal rate alpha.
    Uses a likelihood-ratio statistic distributed as chi2(1).

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
    likelihood-ratio statistic distributed as chi2(2).

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

    # Overall violation rate under independence (H0)
    pi = (n01 + n11) / max(n_total - 1, 1)

    # Row-conditional violation rates (H1)
    pi01 = n01 / row0 if row0 > 0 else 0.0
    pi11 = n11 / row1 if row1 > 0 else 0.0

    # Independence LR
    _eps = 1e-15

    def _safe_log(x: float) -> float:
        return np.log(max(x, _eps))

    # L(pi) under independence
    log_l0_ind = 0.0
    if row0 > 0:
        log_l0_ind += n00 * _safe_log(1 - pi) + n01 * _safe_log(pi)
    if row1 > 0:
        log_l0_ind += n10 * _safe_log(1 - pi) + n11 * _safe_log(pi)

    # L(pi01, pi11) under dependence
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
