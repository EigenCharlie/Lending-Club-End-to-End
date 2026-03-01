"""Tests for scripts/run_fairness_audit.py threshold resolution behavior."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from scripts import run_fairness_audit as fairness_mod


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
        },
        "threshold_policy": {
            "use_artifact": True,
            "artifact_path": str(threshold_path),
            "selected_threshold_key": "selected_threshold",
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
            "status_json": str(model_dir / "fairness_audit_status.json"),
            "status_json_v2": str(model_dir / "fairness_audit_status_v2.json"),
        },
    }

    cfg_path = tmp_path / "fairness_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    fairness_mod.main(str(cfg_path))

    status = json.loads((model_dir / "fairness_audit_status.json").read_text(encoding="utf-8"))
    status_v2 = json.loads(
        (model_dir / "fairness_audit_status_v2.json").read_text(encoding="utf-8")
    )
    assert status["prediction_threshold"] == 0.70
    assert status["prediction_threshold_source"] == "artifact"
    assert status_v2["prediction_threshold"] == status["prediction_threshold"]
