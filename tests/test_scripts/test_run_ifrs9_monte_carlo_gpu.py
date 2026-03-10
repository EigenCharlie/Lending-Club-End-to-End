from __future__ import annotations

import numpy as np

from scripts.run_ifrs9_monte_carlo_gpu import _scenario_multipliers, _tail_metrics


def test_scenario_multipliers_stay_in_expected_ranges() -> None:
    shocks = np.zeros((4, 5), dtype=np.float32)
    mult = _scenario_multipliers(shocks)
    assert np.all(mult["pd_mult"] >= 0.70)
    assert np.all(mult["pd_mult"] <= 1.80)
    assert np.all(mult["lgd_mult"] >= 0.80)
    assert np.all(mult["lgd_mult"] <= 1.50)
    assert np.all(mult["ead_mult"] >= 0.90)
    assert np.all(mult["ead_mult"] <= 1.20)
    assert np.all(mult["discount_rate"] >= 0.03)
    assert np.all(mult["discount_rate"] <= 0.12)


def test_tail_metrics_returns_expected_keys() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0], dtype=float)
    out = _tail_metrics(values)
    assert set(out) == {
        "mean",
        "std",
        "p50",
        "p90",
        "p95",
        "p99",
        "expected_shortfall_95",
    }
    assert out["p95"] >= out["p90"]
