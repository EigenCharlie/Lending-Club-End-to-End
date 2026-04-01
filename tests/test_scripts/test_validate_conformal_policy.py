"""Tests for conformal policy validation v2 checks."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import validate_conformal_policy as policy_mod


@pytest.fixture(autouse=True)
def _set_run_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_TAG", "run-conformal-test")


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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))
    checks = pd.read_parquet(data_dir / "conformal_policy_checks.parquet")

    assert status["schema_version"]
    assert status["generated_at_utc"]
    assert status["run_tag"]
    assert "winkler_90" in status
    assert "kupiec_pvalue_90" in status
    assert "christoffersen_pvalue_90" in status
    assert status["checks_total"] >= 13
    assert "statistical_coverage" in set(checks["scope"])
    assert "lgd_ead_conformal_status" in status
    assert "strict_overall_pass" in status
    assert "methodological_justification_pass" in status
    assert "failing_statistical_checks" in status
    assert "sample_size_context" in status


def test_validate_conformal_policy_falls_back_to_official_baseline_run_tag(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIPELINE_RUN_TAG", raising=False)
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
    pd.DataFrame({"group": ["A"], "coverage_90": [0.9]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01"]),
            "coverage_90": [0.91],
            "coverage_95": [0.96],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )
    pd.DataFrame(
        {
            "y_true": np.linspace(0.1, 0.9, 10),
            "pd_low_90": np.linspace(0.0, 0.8, 10),
            "pd_high_90": np.linspace(0.2, 1.0, 10),
            "pd_low_95": np.linspace(0.0, 0.75, 10),
            "pd_high_95": np.linspace(0.25, 1.0, 10),
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    monkeypatch.setattr(policy_mod, "resolve_official_baseline_run_tag", lambda: "run-official")
    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))
    assert status["run_tag"] == "run-official"


def test_validate_conformal_policy_supports_artifact_namespace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed" / "conformal_gap" / "shadow_ns"
    model_dir = tmp_path / "models" / "conformal_gap" / "shadow_ns"
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
    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.91, 0.90]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "coverage_90": [0.91, 0.92],
            "coverage_95": [0.95, 0.96],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )
    pd.DataFrame(
        {
            "y_true": np.linspace(0.1, 0.9, 20),
            "pd_low_90": np.linspace(0.0, 0.75, 20),
            "pd_high_90": np.linspace(0.25, 1.0, 20),
            "pd_low_95": np.linspace(0.0, 0.70, 20),
            "pd_high_95": np.linspace(0.30, 1.0, 20),
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

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
        "artifacts": {},
        "output": {},
    }
    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path), artifact_namespace="shadow_ns")

    status_path = (
        tmp_path / "models" / "conformal_gap" / "shadow_ns" / "conformal_policy_status.json"
    )
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["artifact_namespace"] == "shadow_ns"
    assert not (tmp_path / "models" / "conformal_policy_status.json").exists()


def test_validate_conformal_policy_allows_methodological_justification_for_stats_only_failures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "conformal_results_mondrian.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics_90": {"empirical_coverage": 0.915, "avg_interval_width": 0.4},
                "metrics_95": {"empirical_coverage": 0.958, "avg_interval_width": 0.6},
            },
            f,
        )

    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.91, 0.90]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "coverage_90": [0.91, 0.92],
            "coverage_95": [0.95, 0.96],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )
    pd.DataFrame(
        {
            "y_true": np.linspace(0.05, 0.95, 50),
            "pd_low_90": np.linspace(0.00, 0.80, 50),
            "pd_high_90": np.linspace(0.20, 1.00, 50),
            "pd_low_95": np.linspace(0.00, 0.75, 50),
            "pd_high_95": np.linspace(0.25, 1.00, 50),
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

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
            "min_kupiec_pvalue_90": 0.01,
            "min_kupiec_pvalue_95": 0.01,
            "min_christoffersen_pvalue_90": 0.01,
            "min_christoffersen_pvalue_95": 0.01,
            "statistical_tests_role": "strict_diagnostics",
            "allow_methodological_justification": True,
            "max_coverage_deviation_for_statistical_warning_90": 0.03,
            "max_coverage_deviation_for_statistical_warning_95": 0.03,
            "min_christoffersen_independence_pvalue_90": 0.01,
            "min_christoffersen_independence_pvalue_95": 0.01,
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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    def _fake_kupiec(_violations, *, alpha=None, nominal_alpha=None):
        return {
            "lr_pof": 20.0,
            "p_value": 0.0,
            "reject_h0": True,
            "n": 50,
            "n_fail": 4,
            "fail_rate": 0.08,
            "nominal_alpha": float(alpha if alpha is not None else nominal_alpha),
        }

    def _fake_christoffersen(_violations, *, alpha=None):
        return {
            "lr_uc": 12.0,
            "p_uc": 0.0,
            "lr_ind": 0.8,
            "p_ind": 0.45,
            "lr_cc": 12.8,
            "p_cc": 0.0,
            "reject_cc": True,
        }

    def _fake_winkler(_y_true, _lower, _upper, *, alpha):
        return np.full(120, 1.203 if float(alpha) == 0.10 else 1.10)

    monkeypatch.setattr(policy_mod, "kupiec_pof_test", _fake_kupiec)
    monkeypatch.setattr(policy_mod, "christoffersen_test", _fake_christoffersen)
    monkeypatch.setattr(policy_mod, "winkler_interval_score", _fake_winkler)

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))

    assert status["overall_pass"] is True  # methodological justification elevates overall
    assert status["strict_overall_pass"] is False
    assert status["non_statistical_checks_pass"] is True
    assert status["methodological_justification_pass"] is True
    assert status["methodological_justification_status"] == "eligible_statistical_warning_only"
    assert status["failing_non_statistical_checks"] == []
    assert set(status["failing_statistical_checks"]) == {
        "kupiec_pvalue_90",
        "kupiec_pvalue_95",
        "christoffersen_pvalue_90",
        "christoffersen_pvalue_95",
    }


def test_validate_conformal_policy_reports_winkler_sensitivity_without_overwriting_main_status(
    tmp_path,
) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "conformal_results_mondrian.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics_90": {"empirical_coverage": 0.912, "avg_interval_width": 0.75},
                "metrics_95": {"empirical_coverage": 0.955, "avg_interval_width": 0.90},
            },
            f,
        )

    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.90, 0.89]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "coverage_90": [0.91, 0.92, 0.90],
            "coverage_95": [0.95, 0.96, 0.95],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )

    y_true = np.linspace(0.10, 0.90, 120)
    pd_low_90 = np.clip(y_true - 0.15, 0.0, 1.0)
    pd_high_90 = np.clip(y_true + 0.15, 0.0, 1.0)
    pd_low_95 = np.clip(y_true - 0.20, 0.0, 1.0)
    pd_high_95 = np.clip(y_true + 0.20, 0.0, 1.0)

    miss_idx = np.arange(108, 120)
    pd_low_90[miss_idx] = np.clip(y_true[miss_idx] - 0.758, 0.0, 1.0)
    pd_high_90[miss_idx] = np.clip(y_true[miss_idx] - 0.458, 0.0, 1.0)
    pd_low_95[miss_idx] = np.clip(y_true[miss_idx] - 0.80, 0.0, 1.0)
    pd_high_95[miss_idx] = np.clip(y_true[miss_idx] - 0.42, 0.0, 1.0)

    intervals = pd.DataFrame(
        {
            "y_true": y_true,
            "pd_low_90": pd_low_90,
            "pd_high_90": pd_high_90,
            "pd_low_95": pd_low_95,
            "pd_high_95": pd_high_95,
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
            "max_winkler_90": 1.00,
            "max_winkler_95": 10.0,
            "min_kupiec_pvalue_90": 0.0,
            "min_kupiec_pvalue_95": 0.0,
            "min_christoffersen_pvalue_90": 0.0,
            "min_christoffersen_pvalue_95": 0.0,
        },
        "policy_sensitivity": {
            "max_winkler_90_values": [1.20, 1.22, 1.25],
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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))

    assert status["overall_pass"] is False
    assert status["failing_non_statistical_checks"] == ["winkler_90"]
    assert "policy_sensitivity" in status
    results = status["policy_sensitivity"]["results"]
    assert [row["max_winkler_90"] for row in results] == [1.2, 1.22, 1.25]
    assert results[0]["strict_overall_pass"] is False
    assert any(row["strict_overall_pass"] for row in results[1:])


def test_validate_conformal_policy_allows_compensated_winkler_band(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "conformal_results_mondrian.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics_90": {"empirical_coverage": 0.928, "avg_interval_width": 0.756},
                "metrics_95": {"empirical_coverage": 0.960, "avg_interval_width": 0.936},
            },
            f,
        )

    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.887, 0.92]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "coverage_90": [0.92, 0.93, 0.94],
            "coverage_95": [0.96, 0.96, 0.97],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )

    y_true = np.linspace(0.10, 0.90, 120)
    low90 = np.clip(y_true - 0.378, 0.0, 1.0)
    high90 = np.clip(y_true + 0.378, 0.0, 1.0)
    low95 = np.clip(y_true - 0.468, 0.0, 1.0)
    high95 = np.clip(y_true + 0.468, 0.0, 1.0)
    pd.DataFrame(
        {
            "y_true": y_true,
            "pd_low_90": low90,
            "pd_high_90": high90,
            "pd_low_95": low95,
            "pd_high_95": high95,
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

    cfg = {
        "policy": {
            "target_coverage_90_min": 0.90,
            "target_coverage_95_min": 0.95,
            "min_group_coverage_90_min": 0.88,
            "max_avg_width_90": 0.80,
            "max_critical_alerts": 0,
            "max_total_alerts": 5,
            "max_warning_alerts": 5,
            "max_winkler_90": 1.00,
            "enable_compensated_winkler_90": True,
            "compensated_winkler_90_max": 1.22,
            "compensated_min_coverage_90": 0.92,
            "compensated_min_group_coverage_90": 0.885,
            "compensated_max_avg_width_90": 0.80,
            "max_winkler_95": 10.0,
            "min_kupiec_pvalue_90": 0.01,
            "min_kupiec_pvalue_95": 0.01,
            "min_christoffersen_pvalue_90": 0.01,
            "min_christoffersen_pvalue_95": 0.01,
            "statistical_tests_role": "strict_diagnostics",
            "allow_methodological_justification": True,
            "max_coverage_deviation_for_statistical_warning_90": 0.03,
            "max_coverage_deviation_for_statistical_warning_95": 0.03,
            "min_christoffersen_independence_pvalue_90": 0.01,
            "min_christoffersen_independence_pvalue_95": 0.01,
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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    def _fake_kupiec(_violations, *, alpha=None, nominal_alpha=None):
        return {
            "lr_pof": 20.0,
            "p_value": 0.0,
            "reject_h0": True,
            "n": 120,
            "n_fail": 8,
            "fail_rate": 8 / 120,
            "nominal_alpha": float(alpha if alpha is not None else nominal_alpha),
        }

    def _fake_christoffersen(_violations, *, alpha=None):
        return {
            "lr_uc": 12.0,
            "p_uc": 0.0,
            "lr_ind": 0.8,
            "p_ind": 0.45,
            "lr_cc": 12.8,
            "p_cc": 0.0,
            "reject_cc": True,
        }

    monkeypatch.setattr(policy_mod, "kupiec_pof_test", _fake_kupiec)
    monkeypatch.setattr(policy_mod, "christoffersen_test", _fake_christoffersen)

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    policy_mod.main(str(cfg_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))

    assert status["strict_overall_pass"] is False
    assert status["non_statistical_checks_pass"] is True
    assert status["methodological_justification_pass"] is True
    assert status["failing_non_statistical_checks"] == []
    assert status["winkler_90_policy_pass"] is True


def test_validate_conformal_policy_sensitivity_config_overrides_policy_sensitivity(
    tmp_path,
) -> None:
    """--sensitivity-config flag overrides policy_sensitivity from the main config."""
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    with open(model_dir / "conformal_results_mondrian.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics_90": {"empirical_coverage": 0.912, "avg_interval_width": 0.75},
                "metrics_95": {"empirical_coverage": 0.955, "avg_interval_width": 0.90},
            },
            f,
        )

    pd.DataFrame({"group": ["A", "B"], "coverage_90": [0.90, 0.89]}).to_parquet(
        data_dir / "conformal_group_metrics_mondrian.parquet", index=False
    )
    pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "coverage_90": [0.91, 0.92],
            "coverage_95": [0.95, 0.96],
        }
    ).to_parquet(data_dir / "conformal_backtest_monthly.parquet", index=False)
    pd.DataFrame(columns=["severity"]).to_parquet(
        data_dir / "conformal_backtest_alerts.parquet", index=False
    )

    y_true = np.linspace(0.10, 0.90, 120)
    pd.DataFrame(
        {
            "y_true": y_true,
            "pd_low_90": np.clip(y_true - 0.378, 0.0, 1.0),
            "pd_high_90": np.clip(y_true + 0.378, 0.0, 1.0),
            "pd_low_95": np.clip(y_true - 0.468, 0.0, 1.0),
            "pd_high_95": np.clip(y_true + 0.468, 0.0, 1.0),
        }
    ).to_parquet(data_dir / "conformal_intervals_mondrian.parquet", index=False)

    # Main config has NO policy_sensitivity (or different values)
    cfg = {
        "policy": {
            "target_coverage_90_min": 0.90,
            "target_coverage_95_min": 0.95,
            "min_group_coverage_90_min": 0.88,
            "max_avg_width_90": 0.8,
            "max_critical_alerts": 0,
            "max_total_alerts": 5,
            "max_warning_alerts": 5,
            "max_winkler_90": 1.00,
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
            "policy_checks_parquet": str(data_dir / "conformal_policy_checks.parquet"),
        },
    }

    # Sensitivity config injects thresholds to test
    sensitivity_cfg = {"policy_sensitivity": {"max_winkler_90_values": [1.20, 1.25]}}

    cfg_path = tmp_path / "conformal_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    sens_path = tmp_path / "conformal_policy_sensitivity.yaml"
    sens_path.write_text(yaml.safe_dump(sensitivity_cfg), encoding="utf-8")

    policy_mod.main(str(cfg_path), sensitivity_config_path=str(sens_path))

    status = json.loads((model_dir / "conformal_policy_status.json").read_text(encoding="utf-8"))

    assert "policy_sensitivity" in status
    results = status["policy_sensitivity"]["results"]
    assert [row["max_winkler_90"] for row in results] == [1.20, 1.25]
