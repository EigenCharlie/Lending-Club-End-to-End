"""Tests for scripts/export_streamlit_artifacts.py."""

from __future__ import annotations

import json
import pickle

from scripts import export_streamlit_artifacts as export_mod


def test_export_pipeline_summary_uses_current_metric_keys(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    data_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    with open(model_dir / "pipeline_results.pkl", "wb") as f:
        pickle.dump({"pd_auc": 0.71, "price_of_robustness": 10.0}, f)

    with open(model_dir / "pd_training_record.pkl", "wb") as f:
        pickle.dump(
            {
                "best_calibration": "Platt Sigmoid",
                "final_test_metrics": {
                    "auc_roc": 0.723,
                    "gini": 0.446,
                    "brier_score": 0.154,
                    "ece": 0.013,
                },
            },
            f,
        )

    with open(model_dir / "survival_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "cox_concordance_index": 0.67,
                "rsf_c_index_test": 0.68,
                "n_loans": 123,
                "n_events": 22,
                "event_rate": 0.1789,
            },
            f,
        )

    (model_dir / "conformal_policy_status.json").write_text(
        json.dumps(
            {
                "coverage_90": 0.92,
                "coverage_95": 0.96,
                "overall_pass": True,
                "checks_passed": 7,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(export_mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr(export_mod, "DATA_DIR", data_dir)

    export_mod.export_pipeline_summary()

    summary = json.loads((data_dir / "pipeline_summary.json").read_text(encoding="utf-8"))

    assert summary["pd_model"]["final_auc"] == 0.723
    assert summary["pd_model"]["final_brier"] == 0.154
    assert summary["conformal"]["n_checks_passed"] == 7
