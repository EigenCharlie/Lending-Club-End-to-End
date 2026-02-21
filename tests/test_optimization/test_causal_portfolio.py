"""Unit tests for CATE-adjusted portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.causal_portfolio import (
    apply_cate_adjustment,
    build_cate_adjusted_portfolio,
)


@pytest.fixture
def small_cate_portfolio():
    """Small synthetic portfolio with CATE estimates."""
    rng = np.random.RandomState(42)
    n = 10
    loans = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(5000, 20000, n),
            "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
        }
    )
    pd_point = rng.uniform(0.05, 0.30, n)
    pd_low = pd_point * 0.8
    pd_high = pd_point * 1.3
    cate = rng.uniform(-0.01, 0.05, n)  # mix of positive and negative
    lgd = np.full(n, 0.45)
    int_rates = rng.uniform(0.08, 0.25, n)
    return loans, pd_point, pd_low, pd_high, cate, lgd, int_rates


# ── apply_cate_adjustment ──


def test_cate_adjustment_reduces_pd_for_positive_cate():
    """Positive CATE + negative delta → PD decreases."""
    pd_point = np.array([0.20, 0.30])
    int_rates = np.array([0.15, 0.20])
    cate = np.array([0.05, 0.10])  # both positive → eligible

    pd_adj, rates_adj = apply_cate_adjustment(pd_point, int_rates, cate, delta_rate=-1.0)

    assert np.all(pd_adj < pd_point), "PD should decrease for positive CATE"
    assert np.all(rates_adj < int_rates), "Rates should decrease"


def test_cate_adjustment_zero_cate_no_change():
    """Zero CATE → no adjustment."""
    pd_point = np.array([0.15, 0.25])
    int_rates = np.array([0.12, 0.18])
    cate = np.zeros(2)

    pd_adj, rates_adj = apply_cate_adjustment(pd_point, int_rates, cate)

    np.testing.assert_array_almost_equal(pd_adj, pd_point)
    np.testing.assert_array_almost_equal(rates_adj, int_rates)


def test_cate_adjustment_clips_pd_to_valid_range():
    """Extreme adjustment should be clipped to [clip_min, clip_max]."""
    pd_point = np.array([0.01])
    int_rates = np.array([0.10])
    cate = np.array([0.50])  # very large CATE

    pd_adj, _ = apply_cate_adjustment(pd_point, int_rates, cate, delta_rate=-5.0, clip_pd_min=0.001)
    assert pd_adj[0] >= 0.001, "PD should not go below clip_pd_min"


def test_cate_adjustment_clips_rate_to_minimum():
    """Rates should not go below clip_rate_min."""
    pd_point = np.array([0.20])
    int_rates = np.array([0.05])
    cate = np.array([0.10])

    _, rates_adj = apply_cate_adjustment(
        pd_point, int_rates, cate, delta_rate=-10.0, clip_rate_min=0.01
    )
    assert rates_adj[0] >= 0.01, "Rate should not go below minimum"


def test_cate_adjustment_negative_cate_no_change():
    """Negative CATE (rate increase helps) → no adjustment."""
    pd_point = np.array([0.20])
    int_rates = np.array([0.15])
    cate = np.array([-0.05])

    pd_adj, rates_adj = apply_cate_adjustment(pd_point, int_rates, cate, delta_rate=-1.0)

    np.testing.assert_array_almost_equal(pd_adj, pd_point)
    np.testing.assert_array_almost_equal(rates_adj, int_rates)


# ── build_cate_adjusted_portfolio ──


def test_build_cate_portfolio_returns_comparison_df(small_cate_portfolio):
    """Comparison DataFrame should have required structure."""
    loans, pd_point, pd_low, pd_high, cate, lgd, int_rates = small_cate_portfolio
    result = build_cate_adjusted_portfolio(
        loans,
        pd_point,
        pd_low,
        pd_high,
        cate,
        lgd,
        int_rates,
        total_budget=50_000,
        max_portfolio_pd=0.30,
    )
    df = result["comparison_df"]
    assert len(df) == 2
    assert set(df["scenario"]) == {"baseline", "cate_adjusted"}
    assert "objective_value" in df.columns
    assert "n_funded" in df.columns


def test_build_cate_portfolio_zero_cate_equals_baseline(small_cate_portfolio):
    """All-zero CATE → identical solutions."""
    loans, pd_point, pd_low, pd_high, _, lgd, int_rates = small_cate_portfolio
    cate_zero = np.zeros(len(loans))
    result = build_cate_adjusted_portfolio(
        loans,
        pd_point,
        pd_low,
        pd_high,
        cate_zero,
        lgd,
        int_rates,
        total_budget=50_000,
        max_portfolio_pd=0.30,
    )
    df = result["comparison_df"]
    base_obj = df.loc[df["scenario"] == "baseline", "objective_value"].iloc[0]
    adj_obj = df.loc[df["scenario"] == "cate_adjusted", "objective_value"].iloc[0]
    assert base_obj == pytest.approx(adj_obj, rel=0.01), "Zero CATE should yield same objective"


def test_build_cate_portfolio_budget_respected(small_cate_portfolio):
    """Both solutions should respect budget constraint."""
    loans, pd_point, pd_low, pd_high, cate, lgd, int_rates = small_cate_portfolio
    budget = 50_000
    result = build_cate_adjusted_portfolio(
        loans,
        pd_point,
        pd_low,
        pd_high,
        cate,
        lgd,
        int_rates,
        total_budget=budget,
        max_portfolio_pd=0.30,
    )
    assert result["baseline"]["total_allocated"] <= budget * 1.01
    assert result["cate_adjusted"]["total_allocated"] <= budget * 1.01
