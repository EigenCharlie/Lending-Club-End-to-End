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
