from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.ifrs9_diagnostics import (
    adf_power_diagnostic,
    recursive_regression_paths,
    recursive_regression_status,
    scenario_interval_uncertainty,
    scenario_sign_coherence,
    sensitivity_surface_summary,
)


def test_recursive_regression_status_reports_stable_signs() -> None:
    n = 72
    ds = pd.date_range("2015-01-01", periods=n, freq="MS")
    avg_int_rate = np.linspace(8.0, 14.0, n)
    avg_dti = np.linspace(10.0, 18.0, n)
    avg_loan_amnt = np.linspace(9000.0, 12000.0, n)
    loan_count = np.linspace(1000, 1500, n)
    y = 0.01 + 0.003 * (avg_int_rate - avg_int_rate.mean()) + 0.001 * (avg_dti - avg_dti.mean())
    frame = pd.DataFrame(
        {
            "ds": ds,
            "default_rate": y,
            "avg_int_rate": avg_int_rate,
            "avg_dti": avg_dti,
            "avg_loan_amnt": avg_loan_amnt,
            "loan_count": loan_count,
        }
    )

    paths = recursive_regression_paths(
        frame,
        time_col="ds",
        target_col="default_rate",
        feature_cols=["avg_int_rate", "avg_dti", "avg_loan_amnt", "loan_count"],
        min_window=36,
        weight_col="loan_count",
    )
    status = recursive_regression_status(paths, min_sign_match_share=0.75)

    assert not paths.empty
    assert status["overall_pass"] is True
    assert status["min_sign_match_share"] >= 0.75


def test_scenario_sign_coherence_flags_breaks() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["baseline", "mild_stress", "adverse", "severe"],
            "pd_mult": [1.0, 1.1, 1.2, 1.3],
            "stage2_share": [0.2, 0.3, 0.28, 0.4],
            "stage3_share": [0.1, 0.1, 0.2, 0.25],
            "total_ecl": [100.0, 120.0, 118.0, 170.0],
            "total_ecl_high": [110.0, 130.0, 140.0, 190.0],
        }
    )
    coherence = scenario_sign_coherence(frame)

    assert not coherence.empty
    assert coherence.loc[coherence["metric"] == "pd_mult", "overall_pass"].item() is True
    assert coherence.loc[coherence["metric"] == "stage2_share", "overall_pass"].item() is False
    assert coherence.loc[coherence["metric"] == "total_ecl", "overall_pass"].item() is False


def test_adf_power_diagnostic_returns_power_grid() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, size=80)
    series = pd.Series(np.cumsum(noise * 0.05) + noise * 0.01)

    diag = adf_power_diagnostic(
        series, n_simulations=20, candidate_phis=(0.9, 0.95), random_state=7
    )

    assert diag["available"] is True
    assert diag["n_obs"] == 80
    assert set(diag["power_by_phi"]) == {"0.90", "0.95"}
    assert 0.0 <= float(diag["power_by_phi"]["0.90"]) <= 1.0


def test_ifrs9_uncertainty_and_sensitivity_helpers_return_expected_shapes() -> None:
    scenarios = pd.DataFrame(
        {
            "point_forecast": [0.01, 0.02, 0.03],
            "optimistic_90": [-0.01, -0.01, 0.0],
            "adverse_90": [0.03, 0.05, 0.08],
            "optimistic_95": [-0.02, -0.02, -0.01],
            "adverse_95": [0.04, 0.06, 0.09],
        }
    )
    grid = pd.DataFrame(
        {
            "pd_mult": [0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
            "lgd_mult": [0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1],
            "discount_rate": [
                0.04,
                0.05,
                0.06,
                0.04,
                0.05,
                0.06,
                0.04,
                0.05,
                0.06,
                0.04,
                0.05,
                0.06,
            ],
            "total_ecl": [
                95.0,
                100.0,
                106.0,
                115.0,
                122.0,
                129.0,
                138.0,
                146.0,
                154.0,
                162.0,
                171.0,
                180.0,
            ],
        }
    )
    uncertainty = scenario_interval_uncertainty(scenarios)
    sensitivity = sensitivity_surface_summary(grid)

    assert uncertainty["available"] is True
    assert uncertainty["mean_width_90"] > 0.0
    assert sensitivity["available"] is True
    assert sensitivity["dominant_driver"] in {"pd_mult", "lgd_mult", "discount_rate"}
