"""Tests for conformal policy validation v2 checks."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import yaml

from scripts import validate_conformal_policy as policy_mod


def test_validate_conformal_policy_includes_statistical_checks(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "conformal_results_mondrian.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics_90": {"empirical_coverage": 0.91, "avg_interval_width": 0.4},
                "metrics_95": {"empirical_coverage": 0.96, "avg_interval_width": 0.6},
            },
            f,
        )

    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.9, 0.89]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )

    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "coverage_90": [0.9, 0.91],
            "coverage_95": [0.95, 0.96],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)

    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )

    rng = np.random.RandomState(42)
    y_true = rng.uniform(0.0, 1.0, 200)
    intervals = pd.DataFrame(
        {
            "y_true": y_true,
            "pd_low_90": np.clip(y_true - 0.15, 0.0, 1.0),
            "pd_high_90": np.clip(y_true + 0.15, 0.0, 1.0),
            "pd_low_95": np.clip(y_true - 0.20, 0.0, 1.0),
            "pd_high_95": np.clip(y_true + 0.20, 0.0, 1.0),
        }
    )
    intervals.to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

    cfg = {
        "policy": {
            "target_coverage_90_min": 0.90,
            "target_coverage_95_min": 0.95,
            "min_group_coverage_90_min": 0.88,
            "max_avg_width_90": 0.8,
            "max_critical_alerts": 0,
            "max_total_alerts": 5,
            "max_warning_alerts": 5,
            "max_winkler_90": 10.0,
            "max_winkler_95": 10.0,
            "min_kupiec_pvalue_90": 0.0,
            "min_kupiec_pvalue_95": 0.0,
            "min_christoffersen_pvalue_90": 0.0,
            "min_christoffersen_pvalue_95": 0.0,
        },
        "artifacts": {
            "conformal_results_path": str(model_dir / "conformal_results_mondrian.pkl"),
            "group_metrics_path": str(data_dir / "conformal_group_metrics_mondrian.parquet"),
            "backtest_monthly_path": str(data_dir / "conformal_backtest_monthly.parquet"),
            "backtest_alerts_path": str(data_dir / "conformal_backtest_alerts.parquet"),
            "intervals_path": str(data_dir / "conformal_intervals_mondrian.parquet"),
        },
        "output": {
            "policy_status_json": str(model_dir / "conformal_policy_status.json"),
            "policy_status_json_v2": str(model_dir / "conformal_policy_status_v2.json"),
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))
    status_v2 = json.loads(
        (model_dir / "conformal_policy_status_v2.json").read_text(encoding="utf-8")
    )
    checks = pd.read_parquet(data_dir / "conformal_policy_checks.parquet")

    assert "winkler_90" in status
    assert "kupiec_pvalue_90" in status
    assert "christoffersen_pvalue_90" in status
    assert status["checks_total"] >= 13
    assert "statistical_coverage" in set(checks["scope"])
    assert status_v2["checks_total"] == status["checks_total"]
