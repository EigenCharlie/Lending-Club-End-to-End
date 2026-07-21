"""Tests for the research-only TS/IFRS9 vNext orchestration script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import run_time_series_vnext as vnext_mod


def test_main_writes_namespaced_vnext_outputs_without_touching_canonical(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    portfolio = pd.DataFrame(
        {
            "unique_id": ["portfolio"] * 36,
            "ds": pd.date_range("2019-01-01", periods=36, freq="MS"),
            "y": [0.02 + (i * 0.0001) for i in range(36)],
            "y_logit": [-3.9 + (i * 0.01) for i in range(36)],
            "loan_count": [100.0] * 36,
            "default_count": [2.0] * 36,
        }
    )
    panel = pd.DataFrame(
        {
            "series_level": ["grade_term"] * 36,
            "unique_id": ["grade_term::A__36"] * 36,
            "ds": pd.date_range("2019-01-01", periods=36, freq="MS"),
            "default_count": [2.0] * 36,
            "loan_count": [20.0] * 36,
            "grade": ["A"] * 36,
            "term_months": [36] * 36,
            "default_rate": [0.10] * 36,
        }
    )
    monkeypatch.setattr(vnext_mod, "_load_history_vnext", lambda: (portfolio, panel))
    monkeypatch.setattr(vnext_mod, "load_future_covariates", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(
        vnext_mod,
        "compute_forecastability_report",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "series_level": ["grade_term"],
                    "unique_id": ["grade_term::A__36"],
                    "route": ["structured_statistical"],
                }
            ),
            {"available": True, "series_evaluated": 1},
        ),
    )
    monkeypatch.setattr(
        vnext_mod,
        "evaluate_hierarchical_reconciliation",
        lambda *args, **kwargs: (
            pd.DataFrame({"target": ["default_count"], "method": ["BottomUp"], "mae": [1.0]}),
            {"available": True},
        ),
    )
    monkeypatch.setattr(
        vnext_mod,
        "compute_forecastability_diagnostics",
        lambda *args, **kwargs: {"recent_actual_mean_12m": 0.02, "n_periods": 36},
    )

    def _backtest(*args, **kwargs):
        spec = kwargs.get("target_spec", args[2])
        predictions = pd.DataFrame(
            {
                "cutoff": [pd.Timestamp("2021-12-01"), pd.Timestamp("2021-12-01")] * 2,
                "ds": [
                    pd.Timestamp("2022-01-01"),
                    pd.Timestamp("2022-02-01"),
                    pd.Timestamp("2022-01-01"),
                    pd.Timestamp("2022-02-01"),
                ],
                "horizon_step": [1, 2, 1, 2],
                "unique_id": ["portfolio"] * 4,
                "model": ["SeasonalNaive", "SeasonalNaive", "AutoARIMA", "AutoARIMA"],
                "target_variant": [spec.name] * 4,
                "family": ["statistical"] * 4,
                "interval_subfamily": ["native_statistical"] * 4,
                "y_true": [0.02, 0.021, 0.02, 0.021],
                "y_pred": [0.021, 0.022, 0.019, 0.021],
                "lo_90": [0.015, 0.016, 0.014, 0.016],
                "hi_90": [0.027, 0.028, 0.024, 0.026],
                "lo_95": [0.013, 0.014, 0.012, 0.014],
                "hi_95": [0.029, 0.030, 0.026, 0.028],
            }
        )
        metrics = pd.DataFrame(
            [
                {
                    "target_variant": spec.name,
                    "model": "SeasonalNaive",
                    "mae": 0.10,
                    "mase": 1.00,
                    "rmsse": 1.00,
                    "abs_bias": 0.020,
                    "coverage_90": 0.87,
                    "coverage_95": 0.93,
                    "coverage_gap_90": 0.03,
                    "avg_interval_width_90": 0.12,
                    "avg_interval_width_95": 0.16,
                    "winkler_90": 0.12,
                    "winkler_95": 0.15,
                    "pinball_90": 0.04,
                    "wis_90": 0.11,
                    "family": "statistical",
                    "interval_subfamily": "native_statistical",
                    "point_eligible": True,
                    "interval_eligible": True,
                },
                {
                    "target_variant": spec.name,
                    "model": "AutoARIMA",
                    "mae": 0.08 if spec.name == "raw_rate" else 0.09,
                    "mase": 0.80 if spec.name == "raw_rate" else 0.90,
                    "rmsse": 0.90,
                    "abs_bias": 0.010,
                    "coverage_90": 0.88,
                    "coverage_95": 0.94,
                    "coverage_gap_90": 0.02,
                    "avg_interval_width_90": 0.10,
                    "avg_interval_width_95": 0.14,
                    "winkler_90": 0.10,
                    "winkler_95": 0.13,
                    "pinball_90": 0.03,
                    "wis_90": 0.09,
                    "family": "statistical",
                    "interval_subfamily": "native_statistical",
                    "point_eligible": True,
                    "interval_eligible": True,
                },
            ]
        )
        return predictions, metrics

    monkeypatch.setattr(vnext_mod, "run_portfolio_backtest_vnext", _backtest)
    monkeypatch.setattr(
        vnext_mod,
        "benchmark_mapie_time_series_intervals_vnext",
        lambda *args, **kwargs: {
            "available": True,
            "best_method": "enbpi",
            "candidate_methods_tested": ["enbpi"],
            "results": [
                {
                    "method": "enbpi",
                    "coverage_90": 0.91,
                    "coverage_gap_90": 0.01,
                    "avg_interval_width_90": 0.08,
                    "winkler_90": 0.07,
                    "wis_90": 0.06,
                    "pinball_90": 0.02,
                }
            ],
        },
    )
    monkeypatch.setattr(
        vnext_mod,
        "forecast_portfolio_models_vnext",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "unique_id": ["portfolio", "portfolio"],
                "ds": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                "AutoARIMA": [0.021, 0.022],
                "AutoARIMA-lo-90": [0.016, 0.017],
                "AutoARIMA-hi-90": [0.026, 0.027],
                "AutoARIMA-lo-95": [0.014, 0.015],
                "AutoARIMA-hi-95": [0.028, 0.029],
            }
        ),
    )
    monkeypatch.setattr(
        vnext_mod,
        "forecast_panel_bottom_up",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "unique_id": ["grade_term::A__36"],
                    "ds": [pd.Timestamp("2022-01-01")],
                    "default_rate": [0.10],
                }
            ),
            {"available": True},
        ),
    )
    monkeypatch.setattr(
        vnext_mod,
        "generate_joint_sample_paths",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "method": ["gaussian_copula", "schaake_shuffle"],
                    "n_samples": [32, 32],
                    "mean_path_avg": [0.022, 0.0215],
                    "p05_path_avg": [0.020, 0.019],
                    "p95_path_avg": [0.024, 0.0245],
                    "mean_path_sum": [0.044, 0.043],
                    "p05_path_sum": [0.040, 0.039],
                    "p50_path_sum": [0.044, 0.043],
                    "p95_path_sum": [0.048, 0.049],
                    "prob_any_month_above_recent_x110": [0.1, 0.2],
                    "prob_three_consecutive_above_recent_x110": [0.0, 0.0],
                    "three_consecutive_signal_present": [False, False],
                }
            ),
            pd.DataFrame({"method": ["gaussian_copula"], "sample_id": [0], "2022-01-01": [0.021]}),
        ),
    )
    monkeypatch.setattr(vnext_mod, "infer_run_tag", lambda: "test-run")

    pd.DataFrame(
        {
            "Grade": ["A"],
            "PD_12m": [0.05],
            "PD_lifetime": [0.12],
            "ECL_Stage1": [100.0],
            "ECL_Stage2": [200.0],
        }
    ).to_parquet("data/processed/ifrs9_ecl_comparison.parquet", index=False)
    pd.DataFrame(
        {
            "y_true": [0.02],
            "y_pred": [0.02],
            "width_90": [0.04],
            "grade": ["A"],
            "loan_amnt": [10000.0],
        }
    ).to_parquet("data/processed/conformal_intervals_mondrian.parquet", index=False)

    project_root = Path(__file__).resolve().parent.parent.parent
    vnext_mod.main(config_path=str(project_root / "configs" / "time_series_vnext.yaml"))

    assert Path("models/time_series_vnext_status.json").exists()
    assert Path("models/time_series_policy_review.json").exists()
    assert Path("models/time_series_status.json").exists() is False
    assert Path("data/processed/ts_backtest_metrics_vnext.parquet").exists()
    assert Path("data/processed/ts_interval_eval_vnext.parquet").exists()
    assert Path("data/processed/ts_ifrs9_scenarios_vnext.parquet").exists()
    assert Path("data/processed/ts_ecl_intervals_vnext.parquet").exists()

    status = vnext_mod._load_optional_json(Path("models/time_series_vnext_status.json"))
    assert status["selected_target_variant"] == "raw_rate"
    assert status["summary"]["operational_interval_model"] == "AutoARIMA"
    assert "backtest-only" in status["operational_interval_note"]
