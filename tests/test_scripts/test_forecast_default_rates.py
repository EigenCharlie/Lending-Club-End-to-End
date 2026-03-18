"""Tests for forecast_default_rates artifact loading fallbacks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import forecast_default_rates as forecast_mod


def test_load_history_rebuilds_missing_governed_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    def _frame(start: str, periods: int) -> pd.DataFrame:
        issue_d = pd.date_range(start, periods=periods, freq="MS")
        return pd.DataFrame(
            {
                "id": [f"{start}-{i}" for i in range(periods)],
                "issue_d": issue_d,
                "loan_amnt": [10_000 + 100 * i for i in range(periods)],
                "default_flag": [i % 5 == 0 for i in range(periods)],
                "grade": ["A" if i % 2 == 0 else "B" for i in range(periods)],
                "term": ["36 months" if i % 2 == 0 else "60 months" for i in range(periods)],
                "int_rate": ["12.5%"] * periods,
                "dti": [15.0] * periods,
            }
        )

    _frame("2015-01-01", 24).to_parquet(data_dir / "train.parquet", index=False)
    _frame("2017-01-01", 12).to_parquet(data_dir / "calibration.parquet", index=False)
    _frame("2018-01-01", 12).to_parquet(data_dir / "test.parquet", index=False)

    portfolio, panel = forecast_mod._load_history()

    assert not portfolio.empty
    assert not panel.empty
    assert (data_dir / "time_series_full.parquet").exists()
    assert (data_dir / "time_series_panel.parquet").exists()


def test_load_history_rebuilds_irregular_panel_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    base = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "issue_d": pd.to_datetime(["2018-01-01", "2018-03-01", "2018-04-01"]),
            "loan_amnt": [10000, 12000, 14000],
            "default_flag": [0, 1, 0],
            "grade": ["A", "A", "A"],
            "term": ["36 months", "36 months", "36 months"],
            "int_rate": ["12.5%"] * 3,
            "dti": [15.0] * 3,
        }
    )
    base.to_parquet(data_dir / "train.parquet", index=False)
    base.to_parquet(data_dir / "calibration.parquet", index=False)
    base.to_parquet(data_dir / "test.parquet", index=False)

    pd.DataFrame({"ds": pd.to_datetime(["2018-01-01", "2018-03-01"])}).to_parquet(
        data_dir / "time_series_full.parquet",
        index=False,
    )
    stale_panel = pd.DataFrame(
        {
            "series_level": ["grade_term", "grade_term"],
            "unique_id": ["grade_term::A__36", "grade_term::A__36"],
            "ds": pd.to_datetime(["2018-01-01", "2018-03-01"]),
            "loan_count": [1.0, 1.0],
            "default_count": [0.0, 1.0],
            "grade": ["A", "A"],
            "term_months": [36, 36],
        }
    )
    stale_panel.to_parquet(data_dir / "time_series_panel.parquet", index=False)

    _, panel = forecast_mod._load_history()
    bottom = panel.loc[panel["unique_id"] == "grade_term::A__36"].sort_values("ds")

    assert pd.Timestamp("2018-02-01") in set(bottom["ds"])


def test_main_with_v2_config_writes_namespaced_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    portfolio = pd.DataFrame(
        {
            "unique_id": ["portfolio"] * 24,
            "ds": pd.date_range("2018-01-01", periods=24, freq="MS"),
            "y": [0.02] * 24,
        }
    )
    panel = pd.DataFrame(
        {
            "series_level": ["grade_term"] * 24,
            "unique_id": ["grade_term::A__36"] * 24,
            "ds": pd.date_range("2018-01-01", periods=24, freq="MS"),
            "default_count": [1.0] * 24,
            "loan_count": [10.0] * 24,
            "grade": ["A"] * 24,
            "term_months": [36] * 24,
            "default_rate": [0.1] * 24,
        }
    )
    monkeypatch.setattr(forecast_mod, "_load_history", lambda: (portfolio, panel))
    monkeypatch.setattr(
        forecast_mod,
        "build_future_covariates_contract",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "month_sin": [0.0] * 12,
                "month_cos": [1.0] * 12,
                "macro_event_count_12m": [0] * 12,
            }
        ),
    )
    monkeypatch.setattr(
        forecast_mod,
        "load_future_covariates",
        lambda cfg: pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "month_sin": [0.0] * 12,
                "month_cos": [1.0] * 12,
                "macro_event_count_12m": [0] * 12,
            }
        ),
    )
    backtest_metrics = pd.DataFrame(
        [
            {
                "model": "AutoARIMA",
                "mae": 0.05,
                "mase": 0.90,
                "rmsse": 0.95,
                "abs_bias": 0.01,
                "coverage_90": 0.89,
                "coverage_gap_90": 0.01,
                "winkler_90": 0.10,
                "wis_90": 0.09,
                "pinball_90": 0.02,
                "avg_interval_width_90": 0.10,
                "family": "statistical",
                "interval_subfamily": "native_statistical",
                "point_eligible": True,
                "interval_eligible": True,
            }
        ]
    )
    monkeypatch.setattr(
        forecast_mod,
        "run_portfolio_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "cutoff": [pd.Timestamp("2019-12-01")],
                    "ds": [pd.Timestamp("2020-01-01")],
                    "horizon_step": [1],
                    "unique_id": ["portfolio"],
                    "model": ["AutoARIMA"],
                    "y_true": [0.02],
                    "y_pred": [0.02],
                    "lo_90": [0.01],
                    "hi_90": [0.03],
                    "lo_95": [0.0],
                    "hi_95": [0.04],
                }
            ),
            backtest_metrics,
        ),
    )
    monkeypatch.setattr(
        forecast_mod,
        "select_time_series_champions",
        lambda metrics, config: {
            "point": {
                "model": "AutoARIMA",
                "promotable": True,
                "mae": 0.05,
                "mase": 0.90,
                "rmsse": 0.95,
                "abs_bias": 0.01,
                "reasons": [],
            },
            "interval": {
                "model": "AutoARIMA",
                "promotable": True,
                "coverage_gap_90": 0.01,
                "avg_interval_width_90": 0.10,
                "winkler_90": 0.10,
                "wis_90": 0.09,
                "pinball_90": 0.02,
                "family": "statistical",
                "reasons": [],
            },
            "baseline": {"model": "SeasonalNaive", "mae": 0.06, "rmsse": 1.0},
        },
    )
    monkeypatch.setattr(
        forecast_mod,
        "forecast_portfolio_models",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "unique_id": ["portfolio"],
                "ds": [pd.Timestamp("2020-01-01")],
                "AutoARIMA": [0.02],
                "AutoARIMA-lo-90": [0.01],
                "AutoARIMA-hi-90": [0.03],
                "AutoARIMA-lo-95": [0.0],
                "AutoARIMA-hi-95": [0.04],
            }
        ),
    )
    monkeypatch.setattr(
        forecast_mod,
        "build_canonical_forecast_frame",
        lambda f, c: f.assign(
            y=f["AutoARIMA"],
            point_model="AutoARIMA",
            interval_model="AutoARIMA",
            y_lo_90=f["AutoARIMA-lo-90"],
            y_hi_90=f["AutoARIMA-hi-90"],
            y_lo_95=f["AutoARIMA-lo-95"],
            y_hi_95=f["AutoARIMA-hi-95"],
            point_promotable=True,
            interval_promotable=True,
            official_status="official",
        ),
    )
    monkeypatch.setattr(forecast_mod, "build_ifrs9_temporal_scenarios", lambda df: df.copy())
    monkeypatch.setattr(
        forecast_mod,
        "forecast_panel_bottom_up",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "unique_id": ["grade_term::A__36"],
                    "ds": [pd.Timestamp("2020-01-01")],
                    "default_rate": [0.1],
                }
            ),
            {"available": True},
        ),
    )
    monkeypatch.setattr(
        forecast_mod,
        "compute_forecastability_diagnostics",
        lambda *args, **kwargs: {"n_periods": 24, "recent_actual_mean_12m": 0.02},
    )
    monkeypatch.setattr(
        forecast_mod,
        "compute_forecastability_report",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "series_level": ["portfolio"],
                    "unique_id": ["portfolio"],
                    "route": ["structured_statistical"],
                }
            ),
            {"available": True, "series_evaluated": 1},
        ),
    )
    monkeypatch.setattr(
        forecast_mod,
        "benchmark_mapie_time_series_intervals",
        lambda *args, **kwargs: {
            "available": True,
            "best_method": "enbpi",
            "candidate_methods_tested": ["enbpi"],
            "results": [
                {
                    "method": "enbpi",
                    "coverage_gap_90": 0.05,
                    "avg_interval_width_90": 0.08,
                    "winkler_90": 0.14,
                    "wis_90": 0.13,
                    "pinball_90": 0.03,
                    "rolling_coverage_summary": {
                        "min_rolling_coverage_6": 0.5,
                        "last_rolling_coverage_6": 0.8,
                    },
                }
            ],
            "rolling_coverage_summary": {
                "min_rolling_coverage_6": 0.5,
                "last_rolling_coverage_6": 0.8,
            },
        },
    )
    monkeypatch.setattr(
        forecast_mod,
        "evaluate_hierarchical_reconciliation",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "target": ["default_count"],
                    "method": ["base"],
                    "series_level": ["portfolio"],
                    "mae": [1.0],
                    "abs_bias": [0.5],
                    "n_obs": [12],
                }
            ),
            {"available": True},
        ),
    )
    monkeypatch.setattr(forecast_mod, "infer_run_tag", lambda: "test-run")

    _project_root = Path(__file__).resolve().parent.parent.parent
    forecast_mod.main(
        config_path=str(_project_root / "configs" / "time_series_v2.yaml"),
        horizon=12,
    )

    assert Path("models/time_series_research_status.json").exists()
    assert Path("models/time_series_status.json").exists() is False
    assert Path("data/processed/ts_backtest_metrics_v2.parquet").exists()
    assert Path("data/processed/ts_interval_eval.parquet").exists()
