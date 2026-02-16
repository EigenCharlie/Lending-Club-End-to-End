"""Plotting utilities for credit risk analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot reliability diagram (calibration curve)."""
    from sklearn.calibration import calibration_curve

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot(mean_pred, fraction_pos, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_conformal_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_intervals: np.ndarray,
    n_samples: int = 100,
    title: str = "Conformal Prediction Intervals",
) -> plt.Figure:
    """Plot prediction intervals with true values."""
    fig, ax = plt.subplots(figsize=(14, 6))
    idx = np.argsort(y_pred)[:n_samples]

    # Accept both MAPIE raw shape (n, 2, 1) and squeezed shape (n, 2).
    if y_intervals.ndim == 3:
        low = y_intervals[idx, 0, 0]
        high = y_intervals[idx, 1, 0]
    elif y_intervals.ndim == 2:
        low = y_intervals[idx, 0]
        high = y_intervals[idx, 1]
    else:
        raise ValueError(f"Unexpected y_intervals shape: {y_intervals.shape}")

    ax.fill_between(range(len(idx)), low, high, alpha=0.3, label="90% CI")
    ax.scatter(range(len(idx)), y_true[idx], s=10, c="red", label="True", zorder=5)
    ax.plot(range(len(idx)), y_pred[idx], "b-", linewidth=0.8, label="Predicted")
    ax.set_xlabel("Sample (sorted by prediction)")
    ax.set_ylabel("Probability of Default")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_portfolio_allocation(
    allocation: dict[int, float],
    loan_amounts: np.ndarray,
    title: str = "Portfolio Allocation",
) -> plt.Figure:
    """Plot portfolio allocation distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    funded = [i for i, v in allocation.items() if v > 0.01]
    amounts = [allocation[i] * loan_amounts[i] for i in funded]

    axes[0].hist(amounts, bins=30, edgecolor="black")
    axes[0].set_title("Distribution of Funded Amounts")
    axes[0].set_xlabel("Funded Amount ($)")

    funded_pct = len(funded) / len(allocation) * 100
    axes[1].pie(
        [len(funded), len(allocation) - len(funded)],
        labels=[f"Funded ({len(funded)})", f"Rejected ({len(allocation)-len(funded)})"],
        autopct="%1.1f%%",
    )
    axes[1].set_title("Approval Rate")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
