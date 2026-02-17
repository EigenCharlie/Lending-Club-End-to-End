"""Tests for src/utils/mlflow_utils.py — DagsHub init + experiment logging."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


class TestInitDagshub:
    """Test init_dagshub with dagshub module mocked at import level."""

    def _call_init(self, **kwargs):
        """Import and call init_dagshub with dagshub mocked."""
        mock_dagshub = MagicMock()
        with patch.dict(sys.modules, {"dagshub": mock_dagshub}):
            from src.utils.mlflow_utils import init_dagshub

            init_dagshub(**kwargs)
        return mock_dagshub

    @patch.dict("os.environ", {"DAGSHUB_USER": "testuser", "DAGSHUB_REPO": "testrepo"}, clear=False)
    def test_uses_env_vars_when_no_args(self) -> None:
        mock_dagshub = self._call_init()
        mock_dagshub.init.assert_called_once_with(
            repo_owner="testuser", repo_name="testrepo", mlflow=True, dvc=False
        )

    def test_explicit_args_override_env(self) -> None:
        mock_dagshub = self._call_init(repo_owner="alice", repo_name="my-repo")
        mock_dagshub.init.assert_called_once_with(
            repo_owner="alice", repo_name="my-repo", mlflow=True, dvc=False
        )

    @patch.dict("os.environ", {"DAGSHUB_TOKEN": "tok123"}, clear=False)
    def test_copies_dagshub_token_to_user_token(self) -> None:
        import os

        os.environ.pop("DAGSHUB_USER_TOKEN", None)
        self._call_init(repo_owner="a", repo_name="b")
        assert os.environ.get("DAGSHUB_USER_TOKEN") == "tok123"

    def test_enable_dvc_flag(self) -> None:
        mock_dagshub = self._call_init(repo_owner="a", repo_name="b", enable_dvc=True)
        mock_dagshub.init.assert_called_once_with(
            repo_owner="a", repo_name="b", mlflow=True, dvc=True
        )


class TestLogExperiment:
    @patch("src.utils.mlflow_utils.mlflow")
    def test_returns_run_id(self, mock_mlflow: MagicMock) -> None:
        mock_run = MagicMock()
        mock_run.info.run_id = "run_abc123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from src.utils.mlflow_utils import log_experiment

        run_id = log_experiment(
            run_name="test_run",
            params={"lr": 0.01},
            metrics={"auc": 0.72},
        )
        assert run_id == "run_abc123"
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.01})
        mock_mlflow.log_metrics.assert_called_once_with({"auc": 0.72})


class TestLogConformalExperiment:
    @patch("src.utils.mlflow_utils.mlflow")
    def test_prefixes_conformal_params_and_metrics(self, mock_mlflow: MagicMock) -> None:
        mock_run = MagicMock()
        mock_run.info.run_id = "run_cp_001"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from src.utils.mlflow_utils import log_conformal_experiment

        run_id = log_conformal_experiment(
            run_name="cp_test",
            base_model_params={"depth": 6},
            conformal_params={"confidence_level": 0.9},
            classification_metrics={"auc": 0.71},
            conformal_metrics={"coverage_90": 0.92},
        )
        assert run_id == "run_cp_001"

        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["depth"] == 6
        assert logged_params["cp_confidence_level"] == 0.9

        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert logged_metrics["auc"] == 0.71
        assert logged_metrics["cp_coverage_90"] == 0.92

        mock_mlflow.set_tags.assert_called_once_with({"experiment_type": "conformal_prediction"})
