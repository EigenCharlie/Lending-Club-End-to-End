"""Tests for scripts/log_mlflow_experiment_suite.py helper functions.

These test the pure utility functions (_to_metrics, _to_params, _git_sha)
without requiring MLflow or DagsHub connections.
"""

from __future__ import annotations

from unittest.mock import patch


def _import_helpers():
    """Import helpers from the script module."""
    from scripts.log_mlflow_experiment_suite import _git_sha, _to_metrics, _to_params

    return _to_metrics, _to_params, _git_sha


class TestToMetrics:
    def test_flat_dict(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"auc": 0.72, "gini": 0.44})
        assert result == {"auc": 0.72, "gini": 0.44}

    def test_nested_dict_flattens_with_prefix(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"model": {"auc": 0.7, "ks": 0.3}})
        assert result == {"model_auc": 0.7, "model_ks": 0.3}

    def test_skips_nan_and_inf(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"good": 1.0, "nan_val": float("nan"), "inf_val": float("inf")})
        assert result == {"good": 1.0}

    def test_booleans_become_float(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"pass": True, "fail": False})
        assert result == {"pass": 1.0, "fail": 0.0}

    def test_skips_non_numeric(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"name": "catboost", "score": 0.9})
        assert result == {"score": 0.9}

    def test_prefix_argument(self) -> None:
        _to_metrics, _, _ = _import_helpers()
        result = _to_metrics({"auc": 0.7}, prefix="pd_")
        assert result == {"pd_auc": 0.7}


class TestToParams:
    def test_basic_types(self) -> None:
        _, _to_params, _ = _import_helpers()
        result = _to_params({"lr": 0.01, "name": "catboost", "depth": 6, "verbose": True})
        assert result == {"lr": 0.01, "name": "catboost", "depth": 6, "verbose": True}

    def test_skips_none(self) -> None:
        _, _to_params, _ = _import_helpers()
        result = _to_params({"a": 1, "b": None, "c": "x"})
        assert "b" not in result
        assert result == {"a": 1, "c": "x"}

    def test_list_joined(self) -> None:
        _, _to_params, _ = _import_helpers()
        result = _to_params({"features": ["a", "b", "c"]})
        assert result == {"features": "a,b,c"}

    def test_complex_type_becomes_str(self) -> None:
        _, _to_params, _ = _import_helpers()
        result = _to_params({"config": {"nested": True}})
        assert isinstance(result["config"], str)


class TestGitSha:
    @patch("subprocess.check_output", return_value="abc123def456\n")
    def test_returns_sha(self, mock_subprocess) -> None:
        _, _, _git_sha = _import_helpers()
        assert _git_sha() == "abc123def456"

    @patch("subprocess.check_output", side_effect=FileNotFoundError("git not found"))
    def test_returns_unknown_on_error(self, mock_subprocess) -> None:
        _, _, _git_sha = _import_helpers()
        assert _git_sha() == "unknown"


def test_log_conformal_includes_explicit_policy_metrics(monkeypatch) -> None:
    import scripts.log_mlflow_experiment_suite as suite_mod

    policy_status = {
        "coverage_90": 0.90,
        "coverage_95": 0.95,
        "avg_width_90": 0.72,
        "min_group_coverage_90": 0.88,
        "checks_passed": 6,
        "checks_total": 7,
        "overall_pass": False,
        "critical_alerts": 0,
        "warning_alerts": 2,
        "policy_config": "configs/conformal_policy.yaml",
    }
    captured: dict = {}

    def fake_log_run(**kwargs):
        captured.update(kwargs)
        return "run-id"

    monkeypatch.setattr(suite_mod, "_load_json", lambda *_args, **_kwargs: policy_status)
    monkeypatch.setattr(suite_mod, "_log_run", fake_log_run)

    run_id = suite_mod._log_conformal("20260217", {"git_sha": "abc"})

    assert run_id == "run-id"
    metrics = captured["metrics"]
    params = captured["params"]

    assert metrics["checks_passed"] == 6.0
    assert metrics["checks_total"] == 7.0
    assert metrics["checks_passed_ratio"] == 6.0 / 7.0
    assert metrics["overall_pass"] == 0.0
    assert params["checks_total"] == 7
    assert params["overall_pass"] is False


def test_configure_tracking_non_interactive_uses_local_fallback(monkeypatch, tmp_path) -> None:
    import scripts.log_mlflow_experiment_suite as suite_mod

    monkeypatch.setattr(suite_mod, "ROOT", tmp_path)
    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)

    calls = {"init": 0}
    captured: dict[str, str] = {}

    def fake_init_dagshub(**_kwargs):
        calls["init"] += 1

    monkeypatch.setattr(suite_mod, "init_dagshub", fake_init_dagshub)
    monkeypatch.setattr(
        suite_mod.mlflow,
        "set_tracking_uri",
        lambda uri: captured.setdefault("uri", uri),
    )

    suite_mod._configure_tracking_non_interactive("owner", "repo")

    assert calls["init"] == 0
    assert captured["uri"].startswith("file:")


def test_resolve_official_baseline_run_tag_prefers_registry(monkeypatch, tmp_path) -> None:
    import scripts.log_mlflow_experiment_suite as suite_mod

    registry = tmp_path / "configs" / "baselines" / "canonical_operational_baseline.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        '{"official_run_tag":"2026-03-11-C-official-selector-v3-freeze"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(suite_mod, "BASELINE_REGISTRY_PATH", registry)
    monkeypatch.delenv("OFFICIAL_BASELINE_RUN_TAG", raising=False)

    assert (
        suite_mod._resolve_official_baseline_run_tag(None)
        == "2026-03-11-C-official-selector-v3-freeze"
    )


def test_resolve_official_baseline_run_tag_cli_overrides_env_and_registry(
    monkeypatch, tmp_path
) -> None:
    import scripts.log_mlflow_experiment_suite as suite_mod

    registry = tmp_path / "configs" / "baselines" / "canonical_operational_baseline.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text('{"official_run_tag":"registry-tag"}', encoding="utf-8")
    monkeypatch.setattr(suite_mod, "BASELINE_REGISTRY_PATH", registry)
    monkeypatch.setenv("OFFICIAL_BASELINE_RUN_TAG", "env-tag")

    assert suite_mod._resolve_official_baseline_run_tag("cli-tag") == "cli-tag"


def test_log_time_series_uses_backtest_metrics_and_status(monkeypatch, tmp_path) -> None:
    import json

    import pandas as pd

    import scripts.log_mlflow_experiment_suite as suite_mod

    (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "unique_id": ["portfolio", "portfolio"],
            "ds": pd.to_datetime(["2026-04-01", "2026-05-01"]),
            "y": [0.11, 0.12],
            "y_lo_90": [0.09, 0.10],
            "y_hi_90": [0.13, 0.14],
            "point_model": ["AutoARIMA", "AutoARIMA"],
            "interval_model": ["AutoARIMA", "AutoARIMA"],
            "official_status": ["official", "official"],
        }
    ).to_parquet(tmp_path / "data/processed/ts_forecasts.parquet", index=False)
    pd.DataFrame(
        {
            "model": ["AutoARIMA", "SeasonalNaive"],
            "mae": [0.02, 0.03],
            "mase": [0.80, 1.00],
            "rmsse": [0.90, 1.00],
            "fva_mae_pct": [0.20, 0.00],
            "coverage_90": [0.89, 0.94],
            "coverage_gap_90": [0.01, 0.04],
            "winkler_90": [0.08, 0.12],
            "avg_interval_width_90": [0.04, 0.06],
        }
    ).to_parquet(tmp_path / "data/processed/ts_backtest_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2025-12-01", "2025-12-01"]),
            "ds": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "horizon_step": [1, 2],
            "unique_id": ["portfolio", "portfolio"],
            "model": ["AutoARIMA", "AutoARIMA"],
            "y_true": [0.10, 0.12],
            "y_pred": [0.11, 0.12],
        }
    ).to_parquet(tmp_path / "data/processed/ts_backtest_predictions.parquet", index=False)
    pd.DataFrame({"month": pd.to_datetime(["2026-04-01"]), "point_forecast": [0.11]}).to_parquet(
        tmp_path / "data/processed/ts_ifrs9_scenarios.parquet",
        index=False,
    )
    (tmp_path / "data/processed/ts_diagnostics.json").write_text(
        json.dumps({"seasonal_strength": 0.55, "variance_ratio": 1.08}),
        encoding="utf-8",
    )
    (tmp_path / "models/time_series_status.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {
                    "point_model": "AutoARIMA",
                    "interval_model": "AutoARIMA",
                    "recent_actual_mean_12m": 0.105,
                },
                "point_champion": {"model": "AutoARIMA", "promotable": True},
                "interval_champion": {
                    "model": "AutoARIMA",
                    "promotable": True,
                    "coverage_90": 0.89,
                },
                "config": {"exogenous_enabled": False},
            }
        ),
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_log_run(**kwargs):
        captured.update(kwargs)
        return "run-id"

    monkeypatch.setattr(suite_mod, "ROOT", tmp_path)
    monkeypatch.setattr(suite_mod, "_log_run", fake_log_run)

    run_id = suite_mod._log_time_series("20260307", {"git_sha": "abc"})

    assert run_id == "run-id"
    assert captured["metrics"]["point_champion_mase"] == 0.80
    assert captured["metrics"]["interval_champion_coverage_90"] == 0.89
    assert captured["params"]["point_model"] == "AutoARIMA"
    assert captured["params"]["official_status"] == "official"
    assert "data/processed/ts_backtest_metrics.parquet" in captured["artifacts"]


def test_log_time_series_accepts_nested_diagnostics_schema(monkeypatch, tmp_path) -> None:
    import json

    import pandas as pd

    import scripts.log_mlflow_experiment_suite as suite_mod

    (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "unique_id": ["portfolio"],
            "ds": pd.to_datetime(["2026-04-01"]),
            "y": [0.11],
            "y_lo_90": [0.09],
            "y_hi_90": [0.13],
            "point_model": ["AutoARIMA"],
            "interval_model": ["AutoARIMA"],
            "official_status": ["official"],
        }
    ).to_parquet(tmp_path / "data/processed/ts_forecasts.parquet", index=False)
    pd.DataFrame(
        {
            "model": ["AutoARIMA"],
            "mae": [0.02],
            "mase": [0.80],
            "rmsse": [0.90],
            "fva_mae_pct": [0.20],
            "coverage_90": [0.89],
            "coverage_gap_90": [0.01],
            "winkler_90": [0.08],
            "avg_interval_width_90": [0.04],
        }
    ).to_parquet(tmp_path / "data/processed/ts_backtest_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2025-12-01"]),
            "ds": pd.to_datetime(["2026-01-01"]),
            "horizon_step": [1],
            "unique_id": ["portfolio"],
            "model": ["AutoARIMA"],
            "y_true": [0.10],
            "y_pred": [0.11],
        }
    ).to_parquet(tmp_path / "data/processed/ts_backtest_predictions.parquet", index=False)
    (tmp_path / "data/processed/ts_diagnostics.json").write_text(
        json.dumps(
            {
                "stl": {"seasonal_strength": 0.44},
                "variance_ratio": {"k": 12, "value": 0.77},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "models/time_series_status.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "summary": {"recent_actual_mean_12m": 0.105},
                "point_champion": {"model": "AutoARIMA", "promotable": True},
                "interval_champion": {"model": "AutoARIMA", "promotable": False},
                "config": {"exogenous_enabled": False},
            }
        ),
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_log_run(**kwargs):
        captured.update(kwargs)
        return "run-id"

    monkeypatch.setattr(suite_mod, "ROOT", tmp_path)
    monkeypatch.setattr(suite_mod, "_log_run", fake_log_run)

    run_id = suite_mod._log_time_series("20260313", {"git_sha": "abc"})

    assert run_id == "run-id"
    assert captured["metrics"]["seasonal_strength"] == 0.44
    assert captured["metrics"]["variance_ratio"] == 0.77
