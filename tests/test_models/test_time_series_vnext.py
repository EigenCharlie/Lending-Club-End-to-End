"""Tests for research-only time-series vNext helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.time_series_vnext import (
    TargetSpec,
    evaluate_target_champions,
    generate_joint_sample_paths,
    inverse_transform_array,
    target_specs_from_config,
)


def test_target_specs_from_config_reads_candidates() -> None:
    config = {
        "targets": {
            "candidates": [
                {"name": "raw_rate", "column": "y", "transform": "identity"},
                {
                    "name": "logit_rate",
                    "column": "y_logit",
                    "transform": "logit",
                    "lower_bound": 1e-5,
                    "upper_bound": 0.99999,
                },
            ]
        }
    }
    specs = target_specs_from_config(config)
    assert [spec.name for spec in specs] == ["raw_rate", "logit_rate"]
    assert specs[1].transform == "logit"


def test_inverse_transform_array_for_logit_returns_probabilities() -> None:
    spec = TargetSpec(name="logit_rate", column="y_logit", transform="logit")
    restored = inverse_transform_array(np.asarray([-2.0, 0.0, 2.0]), spec)
    assert restored[0] < restored[1] < restored[2]
    assert np.all((restored > 0.0) & (restored < 1.0))


def test_evaluate_target_champions_skips_interval_only_candidates_for_point() -> None:
    metrics = pd.DataFrame(
        [
            {
                "target_variant": "raw_rate",
                "model": "SeasonalNaive",
                "mae": 0.10,
                "mase": 1.00,
                "rmsse": 1.00,
                "abs_bias": 0.020,
                "coverage_90": 0.88,
                "coverage_gap_90": 0.02,
                "winkler_90": 0.11,
                "wis_90": 0.10,
                "avg_interval_width_90": 0.10,
                "family": "statistical",
                "point_eligible": True,
                "interval_eligible": True,
            },
            {
                "target_variant": "raw_rate",
                "model": "AutoARIMA",
                "mae": 0.08,
                "mase": 0.80,
                "rmsse": 0.90,
                "abs_bias": 0.010,
                "coverage_90": 0.87,
                "coverage_gap_90": 0.03,
                "winkler_90": 0.10,
                "wis_90": 0.09,
                "avg_interval_width_90": 0.09,
                "family": "statistical",
                "point_eligible": True,
                "interval_eligible": True,
            },
            {
                "target_variant": "raw_rate",
                "model": "MAPIE_ENBPI",
                "mae": np.nan,
                "mase": np.nan,
                "rmsse": np.nan,
                "abs_bias": np.nan,
                "coverage_90": 0.91,
                "coverage_gap_90": 0.01,
                "winkler_90": 0.05,
                "wis_90": 0.04,
                "avg_interval_width_90": 0.07,
                "family": "adaptive",
                "interval_subfamily": "adaptive",
                "point_eligible": False,
                "interval_eligible": True,
            },
        ]
    )
    config = {
        "point_champion": {
            "must_beat_seasonal_naive": True,
            "max_rmsse_vs_seasonal_naive": 1.0,
        },
        "interval_policy": {
            "max_coverage_gap": 0.03,
            "eligible_interval_families": ["statistical", "adaptive"],
        },
    }
    champions = evaluate_target_champions(metrics, config)
    assert champions["point"]["model"] == "AutoARIMA"
    assert champions["interval"]["model"] == "MAPIE_ENBPI"


def test_generate_joint_sample_paths_returns_expected_methods() -> None:
    canonical_forecasts = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "y": [0.02, 0.021, 0.022, 0.023],
            "y_lo_90": [0.015, 0.016, 0.017, 0.018],
            "y_hi_90": [0.025, 0.026, 0.027, 0.028],
        }
    )
    backtest_predictions = pd.DataFrame(
        {
            "model": ["AutoARIMA"] * 8,
            "cutoff": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-01",
                    "2023-08-01",
                    "2023-08-01",
                    "2023-09-01",
                    "2023-09-01",
                    "2023-09-01",
                    "2023-09-01",
                ]
            ),
            "ds": pd.to_datetime(
                [
                    "2023-09-01",
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                    "2024-01-01",
                ]
            ),
            "horizon_step": [1, 2, 3, 4, 1, 2, 3, 4],
            "y_true": [0.018, 0.019, 0.021, 0.022, 0.019, 0.020, 0.0215, 0.0225],
            "y_pred": [0.017, 0.0195, 0.0205, 0.021, 0.018, 0.019, 0.021, 0.022],
        }
    )

    summary, samples = generate_joint_sample_paths(
        canonical_forecasts,
        backtest_predictions,
        point_model="AutoARIMA",
        n_samples=32,
        random_seed=7,
    )

    assert {"gaussian_copula", "schaake_shuffle"} == set(summary["method"])
    assert {"method", "sample_id"}.issubset(samples.columns)
    assert summary["estimated_ar1_rho"].map(math.isfinite).all()
    assert len(samples) == 64
