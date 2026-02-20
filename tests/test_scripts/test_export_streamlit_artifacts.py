"""Tests for scripts/export_streamlit_artifacts.py."""

from __future__ import annotations

import json
import pickle

import pandas as pd

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


def test_export_model_comparison_uses_real_hpo_trials_and_metadata(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    data_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    preds = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 1, 0],
            "y_prob_lr": [0.1, 0.6, 0.2, 0.7, 0.8, 0.3],
            "y_prob_cb_default": [0.15, 0.62, 0.18, 0.72, 0.77, 0.28],
            "y_prob_cb_tuned": [0.12, 0.68, 0.16, 0.75, 0.82, 0.24],
            "y_prob_final": [0.11, 0.70, 0.17, 0.78, 0.84, 0.22],
        }
    )
    preds.to_parquet(data_dir / "test_predictions.parquet", index=False)

    with open(model_dir / "pd_training_record.pkl", "wb") as f:
        pickle.dump(
            {
                "best_calibration": "Platt Sigmoid",
                "hpo_trials_executed": 77,
                "hpo_best_validation_auc": 0.7312,
                "validation_scheme": "temporal_train_val_cal_test",
                "feature_count_default": 44,
                "feature_count_tuned": 44,
                "final_test_metrics": {"auc_roc": 0.75},
                "calibration_selection_report": {
                    "selected_method": "platt",
                    "selection_reason": "feasible_multi_metric",
                    "auc_drop_limit": 0.0015,
                    "candidates": [
                        {
                            "method": "platt",
                            "mean_brier": 0.15,
                            "mean_ece": 0.02,
                            "mean_auc_drop": 0.0007,
                            "stability": 0.0002,
                            "folds_used": 4,
                        }
                    ],
                },
            },
            f,
        )

    monkeypatch.setattr(export_mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr(export_mod, "DATA_DIR", data_dir)

    export_mod.export_model_comparison()

    payload = json.loads((data_dir / "model_comparison.json").read_text(encoding="utf-8"))

    assert payload["optuna_n_trials"] == 77
    assert payload["hpo_trials_executed"] == 77
    assert payload["hpo_best_validation_auc"] == 0.7312
    assert payload["validation_scheme"] == "temporal_train_val_cal_test"
    assert payload["feature_count_default"] == 44
    assert payload["feature_count_tuned"] == 44
    assert payload["calibration_selection_report"]["selected_method"] == "platt"
