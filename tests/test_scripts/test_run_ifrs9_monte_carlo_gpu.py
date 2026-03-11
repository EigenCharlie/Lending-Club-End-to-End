from __future__ import annotations

import numpy as np

from scripts.run_ifrs9_monte_carlo_gpu import (
    _build_subportfolio_groups,
    _generate_shocks,
    _load_macro_center,
    _scenario_multipliers,
    _shock_correlation_matrix,
    _summarize_subportfolio_tails,
    _tail_metrics,
)


def test_scenario_multipliers_stay_in_expected_ranges() -> None:
    shocks = np.zeros((4, 5), dtype=np.float32)
    mult = _scenario_multipliers(
        shocks,
        macro_center={"pd_mult": 1.0, "lgd_mult": 1.0, "ead_mult": 1.0, "discount_rate": 0.05},
    )
    assert np.all(mult["pd_mult"] >= 0.70)
    assert np.all(mult["pd_mult"] <= 2.25)
    assert np.all(mult["lgd_mult"] >= 0.80)
    assert np.all(mult["lgd_mult"] <= 1.65)
    assert np.all(mult["ead_mult"] >= 0.90)
    assert np.all(mult["ead_mult"] <= 1.35)
    assert np.all(mult["discount_rate"] >= 0.03)
    assert np.all(mult["discount_rate"] <= 0.16)


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


def test_shock_correlation_profile_is_psd_and_symmetric() -> None:
    corr = _shock_correlation_matrix("moderate_credit")
    assert corr.shape == (5, 5)
    assert np.allclose(corr, corr.T)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals > 0)


def test_antithetic_generator_returns_requested_shape() -> None:
    rng = np.random.default_rng(42)
    shocks = _generate_shocks(
        rng=rng,
        n_scenarios=9,
        correlation_profile="stress_credit",
        antithetic=True,
    )
    assert shocks.shape == (9, 5)
    assert np.isfinite(shocks).all()


def test_load_macro_center_falls_back_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("scripts.run_ifrs9_monte_carlo_gpu.ROOT", tmp_path)
    center = _load_macro_center("severe")
    assert center["profile"] == "severe"
    assert center["pd_mult"] == 1.0


def test_build_subportfolio_groups_and_summary_shapes() -> None:
    base = {
        "pd_point": np.array([0.05, 0.08, 0.12]),
        "grade": np.array(["A", "B", "A"]),
        "dpd": np.array([0.0, 45.0, 120.0]),
        "pd_orig": np.array([0.03, 0.05, 0.09]),
    }
    test = __import__("pandas").DataFrame({"term": ["36 months", "60 months", "36 months"]})
    groups, matrix = _build_subportfolio_groups(base, test)
    assert matrix.shape[1] == 3
    assert len(groups) == matrix.shape[0]
    cpu = np.ones((len(groups), 4), dtype=float)
    gpu = np.ones((len(groups), 4), dtype=float) * 1.01
    summary = _summarize_subportfolio_tails(
        groups=groups, cpu_group_totals=cpu, gpu_group_totals=gpu
    )
    assert not summary.empty
    assert {"group_type", "group_value", "cpu_mean", "gpu_mean"}.issubset(summary.columns)
