"""Tests for scripts/export_streamlit_artifacts.py."""

from __future__ import annotations

import json
import pickle

import numpy as np
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
    (model_dir / "causal_effect_status.json").write_text(
        json.dumps(
            {
                "run_tag": "run-test-123",
                "ate": 0.012,
                "cate_mean": 0.009,
                "cate_std": 0.003,
                "official_method": {
                    "identification": "DoWhy backdoor identification/refutation",
                    "heterogeneity": "EconML CausalForestDML",
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "causal_policy_rule.json").write_text(
        json.dumps(
            {
                "run_tag": "run-test-123",
                "selected_rule": "high_only",
                "selected_metrics": {
                    "total_net_value": 1234.0,
                    "bootstrap_p05_net": 1111.0,
                    "action_rate": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "causal_policy_oot_status.json").write_text(
        json.dumps({"run_tag": "run-test-123", "avg_action_rate": 0.22, "p05_monthly_net": 77.0}),
        encoding="utf-8",
    )
    (model_dir / "cate_portfolio_status.json").write_text(
        json.dumps({"run_tag": "run-test-123", "feasible_adjusted": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(export_mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr(export_mod, "DATA_DIR", data_dir)
    monkeypatch.setenv("PIPELINE_RUN_TAG", "run-test-123")

    export_mod.export_pipeline_summary()

    summary = json.loads((data_dir / "pipeline_summary.json").read_text(encoding="utf-8"))

    assert summary["run_tag"] == "run-test-123"
    assert summary["pd_model"]["final_auc"] == 0.723
    assert summary["pd_model"]["final_brier"] == 0.154
    assert summary["conformal"]["n_checks_passed"] == 7
    assert summary["causal"]["ate"] == 0.012
    assert summary["causal"]["selected_rule"] == "high_only"
    assert summary["causal"]["run_tag_coherence"]["coherent"] is True


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


def test_export_explainability_bundle_builds_required_artifacts(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    shap_summary = pd.DataFrame(
        {
            "feature": ["int_rate", "dti", "fico_score"],
            "mean_abs_shap": [0.12, 0.08, 0.05],
        }
    )
    shap_summary.to_parquet(data_dir / "shap_summary.parquet", index=False)

    shap_raw = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "case_id": ["c0", "c1", "c2", "c3"],
            "issue_quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q3"],
            "grade": ["A", "B", "B", "C"],
            "score_raw": [0.10, 0.20, 0.30, 0.40],
            "pd_calibrated": [0.12, 0.22, 0.33, 0.41],
            "shap_int_rate": [0.10, 0.08, 0.09, 0.11],
            "shap_dti": [0.05, 0.06, 0.04, 0.03],
            "shap_fico_score": [-0.02, -0.01, -0.03, -0.04],
            "val_int_rate": [5.0, 7.5, 10.0, 12.5],
            "val_dti": [8.0, 12.0, 15.0, 20.0],
            "val_fico_score": [700.0, 680.0, 660.0, 640.0],
        }
    )
    shap_raw.to_parquet(data_dir / "shap_raw_top20.parquet", index=False)

    permutation = pd.DataFrame(
        {
            "feature": ["int_rate", "dti", "fico_score"],
            "auc_drop": [0.05, 0.03, 0.01],
        }
    )
    permutation.to_parquet(data_dir / "permutation_importance.parquet", index=False)

    pdp_ice = pd.DataFrame(
        {
            "feature": ["int_rate", "int_rate", "dti", "dti"],
            "grid_value": [5.0, 10.0, 8.0, 15.0],
            "observation_id": [0, 1, 0, 1],
            "ice_pred": [0.1, 0.2, 0.15, 0.25],
            "pdp_pred": [0.1, 0.2, 0.15, 0.25],
        }
    )
    pdp_ice.to_parquet(data_dir / "pdp_ice_top5.parquet", index=False)

    shap_local_cases = pd.DataFrame(
        {
            "case_id": ["c3", "c3", "c3", "c0", "c0", "c0"],
            "segment": [
                "alto_riesgo",
                "alto_riesgo",
                "alto_riesgo",
                "bajo_riesgo",
                "bajo_riesgo",
                "bajo_riesgo",
            ],
            "row_id": [3, 3, 3, 0, 0, 0],
            "issue_quarter": ["2020Q3", "2020Q3", "2020Q3", "2020Q1", "2020Q1", "2020Q1"],
            "grade": ["C", "C", "C", "A", "A", "A"],
            "score_raw": [0.4, 0.4, 0.4, 0.1, 0.1, 0.1],
            "predicted_pd": [0.41, 0.41, 0.41, 0.12, 0.12, 0.12],
            "pd_calibrated": [0.41, 0.41, 0.41, 0.12, 0.12, 0.12],
            "pd_low_90": [0.2, 0.2, 0.2, 0.01, 0.01, 0.01],
            "pd_high_90": [0.8, 0.8, 0.8, 0.3, 0.3, 0.3],
            "pd_low_95": [0.15, 0.15, 0.15, 0.0, 0.0, 0.0],
            "pd_high_95": [0.9, 0.9, 0.9, 0.35, 0.35, 0.35],
            "feature": ["int_rate", "dti", "fico_score", "fico_score", "int_rate", "dti"],
            "feature_value": ["12.5", "20.0", "640", "700", "5.0", "8.0"],
            "shap_value": [0.11, 0.03, -0.04, -0.02, 0.10, 0.05],
            "feature_family": [
                "pricing",
                "capacity",
                "credit_quality",
                "credit_quality",
                "pricing",
                "capacity",
            ],
            "controllable": [True, False, False, False, True, False],
            "monotonic_expected": ["up", "up", "down", "down", "up", "up"],
        }
    )
    shap_local_cases.to_parquet(data_dir / "shap_local_cases.parquet", index=False)

    taxonomy_path = tmp_path / "configs" / "explainability_taxonomy.json"
    taxonomy_path.parent.mkdir(parents=True)
    taxonomy_path.write_text(
        json.dumps(
            {
                "default": {
                    "family": "derived",
                    "controllable": False,
                    "monotonic_expected": "none",
                    "business_label": "Variable derivada",
                },
                "features": {
                    "int_rate": {
                        "family": "pricing",
                        "controllable": True,
                        "monotonic_expected": "up",
                        "business_label": "Tasa",
                    },
                    "dti": {
                        "family": "capacity",
                        "controllable": False,
                        "monotonic_expected": "up",
                        "business_label": "DTI",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeModel:
        def predict_proba(self, X):
            x = np.asarray(pd.DataFrame(X).fillna(0.0), dtype=float)
            score = x[:, 0] * 0.02 + x[:, 1] * 0.01 - x[:, 2] * 0.0001
            prob = 1.0 / (1.0 + np.exp(-score))
            return np.column_stack([1.0 - prob, prob])

    X = pd.DataFrame(
        {
            "int_rate": [5.0, 7.5, 10.0, 12.5],
            "dti": [8.0, 12.0, 15.0, 20.0],
            "fico_score": [700.0, 680.0, 660.0, 640.0],
        }
    )
    meta = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "case_id": ["c0", "c1", "c2", "c3"],
            "issue_quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q3"],
            "grade": ["A", "B", "B", "C"],
            "score_raw": [0.1, 0.2, 0.3, 0.4],
            "pd_calibrated": [0.12, 0.22, 0.33, 0.41],
        }
    )

    monkeypatch.setattr(export_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(export_mod, "FEATURE_TAXONOMY_PATH", taxonomy_path)
    monkeypatch.setattr(
        export_mod,
        "_load_pd_explainability_context",
        lambda sample_size=4000: (
            FakeModel(),
            X.copy(),
            np.array([0, 1, 0, 1]),
            ["int_rate", "dti", "fico_score"],
            [],
            meta.copy(),
        ),
    )

    export_mod.export_explainability_bundle()

    global_df = pd.read_parquet(data_dir / "explainability_global.parquet")
    local_df = pd.read_parquet(data_dir / "explainability_local_cases.parquet")
    ale_df = pd.read_parquet(data_dir / "ale_curves.parquet")
    interaction_df = pd.read_parquet(data_dir / "shap_interactions_or_redundancy.parquet")

    assert {"feature", "mean_abs_shap", "permutation_auc_drop"}.issubset(global_df.columns)
    assert {"case_id", "segmento", "top_positive_reasons", "intervalo_conformal"}.issubset(
        local_df.columns
    )
    assert {"feature", "midpoint", "ale_value"}.issubset(ale_df.columns)
    assert {"feature_a", "feature_b", "redundancy_flag"}.issubset(interaction_df.columns)
