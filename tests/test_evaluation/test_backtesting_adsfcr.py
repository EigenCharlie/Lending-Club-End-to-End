from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtesting import (
    bootstrap_pd_gap_test,
    bootstrap_slice_gap_report,
    classifier_two_sample_test,
    hosmer_lemeshow_test,
    jeffreys_interval,
    normal_approximation_backtest,
    two_sided_exact_binomial_test,
)


def test_classifier_two_sample_test_emits_drivers_and_materiality() -> None:
    rng = np.random.RandomState(42)
    train = pd.DataFrame(
        {
            "x1": rng.normal(0.0, 1.0, size=400),
            "x2": rng.normal(0.0, 1.0, size=400),
        }
    )
    test = pd.DataFrame(
        {
            "x1": rng.normal(2.5, 1.0, size=400),
            "x2": rng.normal(0.0, 1.0, size=400),
        }
    )

    result = classifier_two_sample_test(train, test, ["x1", "x2"], max_rows_per_split=400)

    assert result["c2st_auc"] > 0.6
    assert result["materiality"] in {"high", "severe"}
    assert result["effective_driver_count"] >= 1
    assert isinstance(result["top_drivers"], list)
    assert result["top_drivers"][0]["feature"] == "x1"


def test_pd_backtesting_primitives_return_expected_shapes() -> None:
    exact = two_sided_exact_binomial_test(n_defaults=12, n_obs=100, pd_ref=0.10)
    jeff = jeffreys_interval(n_defaults=12, n_obs=100)
    ztest = normal_approximation_backtest(n_defaults=12, n_obs=100, pd_ref=0.10)
    hl = hosmer_lemeshow_test(
        y_true=np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1] * 20, dtype=float),
        y_prob=np.linspace(0.05, 0.95, 200),
        n_groups=10,
    )

    assert 0.0 <= exact["p_value"] <= 1.0
    assert 0.0 <= jeff["lower"] <= jeff["upper"] <= 1.0
    assert 0.0 <= ztest["p_value"] <= 1.0
    assert hl["n_groups"] >= 2
    assert 0.0 <= hl["hl_p_value"] <= 1.0


def test_bootstrap_gap_test_detects_material_gap() -> None:
    y_true = np.r_[np.ones(300), np.zeros(700)]
    y_prob = np.full_like(y_true, 0.15, dtype=float)
    result = bootstrap_pd_gap_test(y_true, y_prob, n_boot=300, max_sample_size=1000)

    assert result["n_obs"] == 1000
    assert result["abs_gap_bp"] > 100.0
    assert result["zero_inside_ci"] is False
    assert result["materiality"] in {"high", "severe"}


def test_bootstrap_slice_gap_report_returns_ranked_rows() -> None:
    frame = pd.DataFrame(
        {
            "default_flag": np.r_[np.ones(300), np.zeros(700)],
            "pd_calibrated": np.r_[np.full(500, 0.10), np.full(500, 0.25)],
            "issue_quarter": ["2020Q1"] * 500 + ["2020Q2"] * 500,
        }
    )
    report = bootstrap_slice_gap_report(
        frame,
        group_col="issue_quarter",
        n_boot=200,
        min_rows=100,
        max_sample_size=500,
    )

    assert len(report) == 2
    assert list(report.columns)[:2] == ["slice_name", "slice_value"]
