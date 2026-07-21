"""Contract tests for the conformal reopen workflow wrapper."""

from __future__ import annotations

from scripts.search.run_conformal_reopen_search import _acceptance_pass, _aggregate_inner_search


def test_acceptance_pass_respects_thresholds() -> None:
    policy_status = {
        "overall_pass": True,
        "coverage_90": 0.918,
        "min_group_coverage_90": 0.89,
        "avg_width_90": 0.79,
        "warning_alerts": 3,
        "total_alerts": 4,
    }
    validation_cfg = {
        "acceptance": {
            "warning_alerts_max": 5,
            "total_alerts_max": 5,
            "coverage_deviation_90_max": 0.03,
            "min_group_coverage_90_min": 0.88,
            "avg_width_90_max": 0.80,
        }
    }

    assert _acceptance_pass(policy_status, validation_cfg) is True


def test_aggregate_inner_search_ranks_stronger_candidate_first() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "partition": "score_decile_mondrian",
                "partition_probability_source": "calibrated",
                "n_score_bins": 10,
                "fallback_mode": "grade_then_global",
                "alpha_used_90": 0.10,
                "alpha_used_95": 0.05,
                "score_scale_family": "none",
                "min_group_size": 250,
                "calibration_fraction": 0.75,
                "global_ok": True,
                "group_ok": True,
                "width_ok": True,
                "is_pareto": True,
                "empirical_coverage": 0.91,
                "coverage_gap": 0.01,
                "avg_interval_width": 0.76,
                "min_group_coverage": 0.89,
                "winkler_90": 0.82,
                "stability_over_time": 0.02,
                "max_monthly_gap": 0.05,
            },
            {
                "partition": "score_decile_mondrian",
                "partition_probability_source": "calibrated",
                "n_score_bins": 10,
                "fallback_mode": "grade_then_global",
                "alpha_used_90": 0.10,
                "alpha_used_95": 0.05,
                "score_scale_family": "none",
                "min_group_size": 250,
                "calibration_fraction": 0.75,
                "global_ok": True,
                "group_ok": True,
                "width_ok": True,
                "is_pareto": True,
                "empirical_coverage": 0.905,
                "coverage_gap": 0.005,
                "avg_interval_width": 0.75,
                "min_group_coverage": 0.895,
                "winkler_90": 0.80,
                "stability_over_time": 0.02,
                "max_monthly_gap": 0.05,
            },
            {
                "partition": "grade",
                "partition_probability_source": "raw",
                "n_score_bins": 5,
                "fallback_mode": "grade_then_global",
                "alpha_used_90": 0.10,
                "alpha_used_95": 0.05,
                "score_scale_family": "bernoulli_sqrt",
                "min_group_size": 500,
                "calibration_fraction": 1.0,
                "global_ok": False,
                "group_ok": False,
                "width_ok": True,
                "is_pareto": False,
                "empirical_coverage": 0.95,
                "coverage_gap": 0.05,
                "avg_interval_width": 0.88,
                "min_group_coverage": 0.84,
                "winkler_90": 1.10,
                "stability_over_time": 0.08,
                "max_monthly_gap": 0.12,
            },
        ]
    )

    aggregated = _aggregate_inner_search(rows)

    assert aggregated.iloc[0]["partition"] == "score_decile_mondrian"
    assert aggregated.iloc[0]["selection_rank"] == 1
