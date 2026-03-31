from __future__ import annotations

import pandas as pd

from src.evaluation.encoding_stability import (
    bucket_stability_report,
    categorical_psi,
    woe_stability_report,
)


def test_categorical_psi_is_zero_for_matching_distributions() -> None:
    series = pd.Series(["a", "a", "b", "c", "c"])
    assert categorical_psi(series, series) == 0.0


def test_woe_stability_report_flags_large_shift() -> None:
    train = pd.DataFrame(
        {
            "grade_woe": [-1.0, -0.5, 0.5, 1.0] * 30,
            "default_flag": [1, 1, 0, 0] * 30,
        }
    )
    test = pd.DataFrame(
        {
            "grade_woe": [2.0, 2.2, 2.4, 2.6] * 30,
            "default_flag": [0, 0, 1, 1] * 30,
        }
    )
    report = woe_stability_report(train, test)

    assert len(report) == 1
    assert bool(report.loc[0, "overall_pass"]) is False


def test_bucket_stability_report_captures_rank_break() -> None:
    train = pd.DataFrame(
        {
            "risk_bucket": ["low", "med", "high"] * 40,
            "default_flag": [0, 0, 1] * 40,
        }
    )
    test = pd.DataFrame(
        {
            "risk_bucket": ["low", "med", "high"] * 40,
            "default_flag": [1, 0, 0] * 40,
        }
    )
    report = bucket_stability_report(train, test, feature_cols=["risk_bucket"])

    assert len(report) == 1
    assert bool(report.loc[0, "overall_pass"]) is False
