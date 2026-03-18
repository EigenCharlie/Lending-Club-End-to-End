"""Tests for scripts/run_ts_ecl_intervals.py — unit tests for TS→ECL propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_ts_df(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    months = pd.date_range("2020-10-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month": months,
            "point_forecast": rng.uniform(0.005, 0.015, size=n),
            "adverse_90": rng.uniform(0.03, 0.06, size=n),
            "optimistic_90": rng.uniform(-0.01, 0.005, size=n),
            "point_model": ["AutoARIMA"] * n,
            "interval_model": ["AutoARIMA"] * n,
            "official_status": ["diagnostic"] * n,
        }
    )


def _make_grade_ecl() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Grade": ["A", "B", "C", "D", "E"],
            "Avg Loan": [14000, 13000, 14000, 13000, 12000],
            "PD_12m": [0.029, 0.046, 0.064, 0.096, 0.134],
            "PD_lifetime": [0.178, 0.266, 0.352, 0.456, 0.634],
            "ECL_Stage1": [186, 276, 413, 572, 838],
            "ECL_Stage2": [1124, 1593, 2265, 3114, 4891],
            "Stage2/Stage1": [6.0, 5.8, 5.5, 5.4, 5.8],
        }
    )


def _make_conf_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    grades = rng.choice(["A", "B", "C", "D", "E"], size=n)
    return pd.DataFrame(
        {
            "y_true": rng.binomial(1, 0.22, size=n),
            "y_pred": rng.beta(2, 8, size=n).astype(np.float32),
            "width_90": rng.beta(1, 2, size=n).astype(np.float32),
            "grade": grades,
            "loan_amnt": rng.uniform(5000, 35000, size=n).astype(np.float32),
        }
    )


def test_ecl_portfolio_positive() -> None:
    """Portfolio ECL should be positive for reasonable inputs."""
    from scripts.run_ts_ecl_intervals import _ecl_portfolio

    grade_ecl = _make_grade_ecl()
    conf_df = _make_conf_df()
    result = _ecl_portfolio(conf_df, grade_ecl, pd_mult=1.0, lgd=0.40, discount_rate=0.05)
    assert result["total_ecl"] > 0
    assert result["total_ecl"] == result["ecl_stage1"] + result["ecl_stage2"]


def test_ecl_increases_with_pd_mult() -> None:
    """Higher PD multiplier → higher ECL."""
    from scripts.run_ts_ecl_intervals import _ecl_portfolio

    grade_ecl = _make_grade_ecl()
    conf_df = _make_conf_df(seed=42)
    ecl_low = _ecl_portfolio(conf_df, grade_ecl, pd_mult=0.8, lgd=0.40, discount_rate=0.05)
    ecl_high = _ecl_portfolio(conf_df, grade_ecl, pd_mult=1.5, lgd=0.40, discount_rate=0.05)
    assert ecl_high["total_ecl"] > ecl_low["total_ecl"]


def test_ts_ecl_output_rows() -> None:
    """Output should have one row per forecast month."""
    from scripts.run_ts_ecl_intervals import _compute_ts_ecl_intervals

    ts_df = _make_ts_df(n=6)
    grade_ecl = _make_grade_ecl()
    conf_df = _make_conf_df(n=300)
    result_df, summary = _compute_ts_ecl_intervals(
        ts_df,
        conf_df,
        grade_ecl,
        lgd=0.40,
        discount_rate=0.05,
    )
    assert len(result_df) == 6
    assert "ecl_point" in result_df.columns
    assert "ecl_adverse_90" in result_df.columns
    assert "ecl_optimistic_90" in result_df.columns


def test_ts_ecl_adverse_geq_optimistic() -> None:
    """Adverse ECL should be >= optimistic ECL (by definition of multipliers)."""
    from scripts.run_ts_ecl_intervals import _compute_ts_ecl_intervals

    ts_df = _make_ts_df(n=12, seed=1)
    grade_ecl = _make_grade_ecl()
    conf_df = _make_conf_df(n=300, seed=1)
    result_df, _ = _compute_ts_ecl_intervals(
        ts_df,
        conf_df,
        grade_ecl,
        lgd=0.40,
        discount_rate=0.05,
    )
    # Adverse should be >= optimistic for each month
    valid = result_df["ecl_adverse_90"] >= result_df["ecl_optimistic_90"]
    assert valid.all(), f"Adverse < optimistic in {(~valid).sum()} months"


def test_ts_ecl_summary_keys() -> None:
    from scripts.run_ts_ecl_intervals import _compute_ts_ecl_intervals

    ts_df = _make_ts_df(n=6)
    grade_ecl = _make_grade_ecl()
    conf_df = _make_conf_df(n=200)
    _, summary = _compute_ts_ecl_intervals(
        ts_df,
        conf_df,
        grade_ecl,
        lgd=0.40,
        discount_rate=0.05,
    )
    required_keys = {
        "mean_ecl_point",
        "mean_ecl_adverse_90",
        "mean_ecl_optimistic_90",
        "ecl_band_width_90",
        "ecl_band_width_90_pct_of_point",
    }
    assert required_keys.issubset(summary.keys())
