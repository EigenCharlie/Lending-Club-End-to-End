"""Tests for scripts/run_fairness_audit.py threshold resolution behavior."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from scripts import run_fairness_audit as fairness_mod


@pytest.fixture(autouse=True)
def _set_run_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_TAG", "run-fairness-test")


def test_run_fairness_uses_threshold_artifact(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    preds = pd.DataFrame({"pd_calibrated": [0.20, 0.60, 0.30, 0.90]})
    test_df = pd.DataFrame(
        {
            "default_flag": [0, 1, 0, 1],
            "home_ownership": ["RENT", "RENT", "OWN", "OWN"],
            "annual_inc": [50_000, 60_000, 55_000, 90_000],
            "verification_status": ["Verified", "Not Verified", "Verified", "Not Verified"],
        }
    )

    pred_path = data_dir / "test_predictions.parquet"
    data_path = data_dir / "test_fe.parquet"
    preds.to_parquet(pred_path, index=False)
    test_df.to_parquet(data_path, index=False)

    threshold_path = model_dir / "decision_threshold.json"
    threshold_path.write_text(json.dumps({"selected_threshold": 0.70}), encoding="utf-8")

    cfg = {
        "policy": {
            "dpd_threshold": 0.5,
            "eo_gap_threshold": 0.5,
            "dir_threshold": 0.5,
            "prediction_threshold": 0.50,
            "outcome_mode": "approval",
        },
        "threshold_policy": {
            "use_artifact": True,
            "artifact_path": str(threshold_path),
            "selected_threshold_key": "selected_threshold",
        },
        "decision_policy": {
            "auto_select": False,
            "artifact_path": str(model_dir / "missing_fairness_decision_policy.json"),
        },
        "attributes": [
            {"name": "home_ownership", "column": "home_ownership"},
            {"name": "annual_inc_quartile", "column": "annual_inc", "binning": "quartile"},
            {"name": "verification_status", "column": "verification_status"},
        ],
        "artifacts": {
            "test_predictions_path": str(pred_path),
            "test_data_path": str(data_path),
        },
        "output": {
            "audit_parquet": str(data_dir / "fairness_audit.parquet"),
            "frontier_parquet": str(data_dir / "fairness_threshold_frontier.parquet"),
            "status_json": str(model_dir / "fairness_audit_status.json"),
        },
        "fairlearn_sidecar": {
            "enabled": True,
            "status_json": str(model_dir / "fairlearn_fairness_status.json"),
            "group_metrics_parquet": str(data_dir / "fairlearn_group_metrics.parquet"),
            "bootstrap_samples": 5,
        },
        "intersectional": {"enabled": True, "max_order": 2, "min_group_size": 1},
        "threshold_frontier": {"enabled": True, "window_radius": 0.10, "step": 0.10},
    }

    cfg_path = tmp_path / "fairness_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    fairness_mod.main(str(cfg_path))

    status = json.loads((model_dir / "fairness_audit_status.json").read_text(encoding="utf-8"))
    fairlearn_status = json.loads(
        (model_dir / "fairlearn_fairness_status.json").read_text(encoding="utf-8")
    )
    fairlearn_groups = pd.read_parquet(data_dir / "fairlearn_group_metrics.parquet")
    frontier = pd.read_parquet(data_dir / "fairness_threshold_frontier.parquet")
    assert status["schema_version"]
    assert status["generated_at_utc"]
    assert status["run_tag"]
    assert status["prediction_threshold"] == 0.70
    assert status["primary_threshold"] == 0.70
    assert status["prediction_threshold_source"] == "artifact"
    assert status["outcome_mode"] == "approval"
    assert status["n_intersectional_attributes"] > 0
    assert fairlearn_status["run_tag"] == "run-fairness-test"
    assert fairlearn_status["n_attributes"] >= 1
    assert not fairlearn_groups.empty
    assert not frontier.empty
    assert frontier["is_primary_threshold"].any()


def test_run_fairness_auto_selects_threshold_and_writes_decision_policy(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    preds = pd.DataFrame({"pd_calibrated": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]})
    test_df = pd.DataFrame(
        {
            "default_flag": [0, 0, 1, 1, 0, 1],
            "home_ownership": ["RENT", "RENT", "OWN", "OWN", "OWN", "RENT"],
            "annual_inc": [50_000, 55_000, 60_000, 65_000, 70_000, 75_000],
            "verification_status": [
                "Verified",
                "Not Verified",
                "Verified",
                "Not Verified",
                "Verified",
                "Not Verified",
            ],
        }
    )
    pred_path = data_dir / "test_predictions.parquet"
    data_path = data_dir / "test_fe.parquet"
    preds.to_parquet(pred_path, index=False)
    test_df.to_parquet(data_path, index=False)

    cfg = {
        "policy": {
            "dpd_threshold": 0.5,
            "eo_gap_threshold": 0.5,
            "dir_threshold": 0.5,
            "prediction_threshold": 0.50,
            "outcome_mode": "approval",
        },
        "threshold_policy": {"use_artifact": False},
        "decision_policy": {
            "auto_select": True,
            "artifact_path": str(model_dir / "fairness_decision_policy.json"),
        },
        "attributes": [
            {"name": "home_ownership", "column": "home_ownership"},
            {"name": "verification_status", "column": "verification_status"},
        ],
        "artifacts": {
            "test_predictions_path": str(pred_path),
            "test_data_path": str(data_path),
        },
        "output": {
            "audit_parquet": str(data_dir / "fairness_audit.parquet"),
            "frontier_parquet": str(data_dir / "fairness_threshold_frontier.parquet"),
            "status_json": str(model_dir / "fairness_audit_status.json"),
        },
        "fairlearn_sidecar": {
            "enabled": True,
            "status_json": str(model_dir / "fairlearn_fairness_status.json"),
            "group_metrics_parquet": str(data_dir / "fairlearn_group_metrics.parquet"),
            "bootstrap_samples": 5,
        },
        "intersectional": {"enabled": True, "max_order": 2, "min_group_size": 1},
        "threshold_frontier": {"enabled": True, "thresholds": [0.35, 0.40, 0.45, 0.50]},
    }
    cfg_path = tmp_path / "fairness_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    fairness_mod.main(str(cfg_path), run_tag="run-fairness-auto")

    status = json.loads((model_dir / "fairness_audit_status.json").read_text(encoding="utf-8"))
    decision_policy = json.loads(
        (model_dir / "fairness_decision_policy.json").read_text(encoding="utf-8")
    )
    assert status["run_tag"] == "run-fairness-auto"
    assert status["prediction_threshold_source"] == "decision_policy_artifact_auto_selected"
    assert status["decision_policy"]["path"].endswith("fairness_decision_policy.json")
    assert decision_policy["global_threshold"] in [0.35, 0.4, 0.45, 0.5]
