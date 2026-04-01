"""Tests for scripts/run_crpto_vs_spo_stability.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyepo", reason="pyepo not installed")

from scripts.run_crpto_vs_spo_stability import (
    _assign_periods,
    _evaluate_period_coverage,
)


@pytest.fixture
def test_issue_d() -> pd.Series:
    """Realistic issue_d series from known OOT test set counts.

    Uses random dates within each period boundary to match the expected counts.
    """
    rng = np.random.RandomState(42)
    segments = [
        ("2018-01-01", "2018-07-01", 110_839),
        ("2018-07-01", "2019-01-01", 86_339),
        ("2019-01-01", "2019-07-01", 50_245),
        ("2019-07-01", "2020-01-01", 25_160),
        ("2020-01-01", "2021-01-01", 4_286),
    ]
    dates = []
    for start, end, n in segments:
        s = pd.Timestamp(start).value // 10**9
        e = pd.Timestamp(end).value // 10**9 - 1
        epoch_secs = rng.randint(s, e, size=n)
        dates.extend(pd.to_datetime(epoch_secs, unit="s").tolist())
    return pd.Series(dates, name="issue_d")


def test_assign_periods_counts(test_issue_d: pd.Series) -> None:
    """Period assignment matches known OOT loan counts."""
    periods = _assign_periods(test_issue_d)
    counts = periods.value_counts().to_dict()
    assert counts["2018H1"] == 110_839
    assert counts["2018H2"] == 86_339
    assert counts["2019H1"] == 50_245
    assert counts["2019H2"] == 25_160
    assert counts["2020"] == 4_286


def test_assign_periods_covers_all(test_issue_d: pd.Series) -> None:
    """Union of periods covers the full test set with no empty strings."""
    periods = _assign_periods(test_issue_d)
    assert (periods != "").all()
    assert len(periods) == len(test_issue_d)


def test_evaluate_period_coverage_perfect() -> None:
    """Synthetic data where y_true is always inside [low, high] → coverage=1.0."""
    ci = pd.DataFrame(
        {
            "y_true": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pd_low_90": [0.05, 0.15, 0.25, 0.35, 0.45],
            "pd_high_90": [0.15, 0.25, 0.35, 0.45, 0.55],
            "grade": ["A", "A", "B", "B", "C"],
        }
    )
    result = _evaluate_period_coverage(ci)
    assert result["coverage_90"] == 1.0
    assert result["min_grade_coverage_90"] == 1.0
    assert result["avg_width_90"] == pytest.approx(0.1)


def test_evaluate_period_coverage_zero() -> None:
    """Synthetic data where y_true is always outside [low, high] → coverage=0.0."""
    ci = pd.DataFrame(
        {
            "y_true": [0.5, 0.6, 0.7, 0.8, 0.9],
            "pd_low_90": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pd_high_90": [0.1, 0.1, 0.1, 0.1, 0.1],
            "grade": ["A", "A", "B", "B", "C"],
        }
    )
    result = _evaluate_period_coverage(ci)
    assert result["coverage_90"] == 0.0
    assert result["min_grade_coverage_90"] == 0.0


@pytest.mark.slow
def test_output_schema() -> None:
    """Verify JSON structure of the generated artifact."""
    path = Path("data/processed/crpto_vs_spo_stability.json")
    assert path.exists(), f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    assert "schema_version" in data
    assert "config" in data
    assert "per_period" in data
    assert "stability_summary" in data
    assert len(data["per_period"]) == 5
    for _period_name, period_data in data["per_period"].items():
        assert "n_loans" in period_data
        assert "default_rate" in period_data
        assert "regret" in period_data
        assert "coverage_90" in period_data
        for method in ("two_stage", "spo_plus", "conformal_robust"):
            assert method in period_data["regret"]
            assert "mean" in period_data["regret"][method]
            assert "std" in period_data["regret"][method]
