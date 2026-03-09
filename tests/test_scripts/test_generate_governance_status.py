"""Tests for extended governance status generation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from scripts import generate_governance_status as gov_mod


def test_generate_governance_status_emits_explanation_and_fairness_fields(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    rng = np.random.RandomState(7)
    n_train = 400
    n_test = 200
    train = pd.DataFrame(
        {
            "issue_d": pd.date_range("2018-01-01", periods=n_train, freq="D"),
            "fico_score": rng.normal(700, 20, size=n_train),
            "dti": rng.normal(15, 4, size=n_train),
            "int_rate": rng.normal(12, 2, size=n_train),
            "default_flag": rng.binomial(1, 0.2, size=n_train),
            "grade": rng.choice(["A", "B", "C"], size=n_train),
        }
    )
    test = pd.DataFrame(
        {
            "issue_d": pd.date_range("2020-01-01", periods=n_test, freq="D"),
            "fico_score": rng.normal(680, 25, size=n_test),
            "dti": rng.normal(17, 5, size=n_test),
            "int_rate": rng.normal(13, 2.5, size=n_test),
            "default_flag": rng.binomial(1, 0.25, size=n_test),
            "grade": rng.choice(["A", "B", "C"], size=n_test),
        }
    )
    train.to_parquet(data_dir / "train_fe.parquet", index=False)
    test.to_parquet(data_dir / "test_fe.parquet", index=False)

    shap_rows = []
    for idx, row in test.iterrows():
        quarter = "2020Q1" if idx < 100 else "2020Q2"
        shap_rows.append(
            {
                "row_id": int(idx),
                "case_id": f"case-{idx}",
                "issue_quarter": quarter,
                "grade": str(row["grade"]),
                "pd_calibrated": float(0.15 + 0.001 * idx),
                "shap_int_rate": float(0.04 + 0.0005 * idx),
                "shap_dti": float(0.02 + 0.0002 * idx),
                "shap_fico_score": float(-0.03 - 0.0003 * idx),
                "val_int_rate": float(row["int_rate"]),
                "val_dti": float(row["dti"]),
                "val_fico_score": float(row["fico_score"]),
            }
        )
    pd.DataFrame(shap_rows).to_parquet(data_dir / "shap_raw_top20.parquet", index=False)

    (model_dir / "fairness_audit_status.json").write_text(
        json.dumps(
            {
                "overall_pass": True,
                "primary_threshold": 0.5,
                "prediction_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "challenger_promotion_report.json").write_text(
        json.dumps({"challenger_promotable": False}),
        encoding="utf-8",
    )

    cfg = {
        "governance_checks": {
            "ks_pvalue_min": 0.01,
            "cvm_pvalue_min": 0.01,
            "c2st_auc_max": 0.90,
            "max_feature_breach_ratio": 1.0,
            "c2st_max_rows_per_split": 500,
            "psi_bins": 8,
            "random_state": 42,
            "explanation_rank_overlap_top10_min": 0.10,
            "explanation_shap_psi_max": 5.0,
            "reason_code_stability_min": 0.0,
            "explanation_min_rows_per_slice": 20,
        },
        "retraining_triggers": {"psi_threshold": 1.0},
        "governance_output": {
            "drift_monitoring_path": str(data_dir / "drift_monitoring.parquet"),
            "explanation_drift_path": str(data_dir / "explanation_drift.parquet"),
            "fairness_status_path": str(model_dir / "fairness_audit_status.json"),
            "fairness_frontier_path": str(data_dir / "fairness_threshold_frontier.parquet"),
            "challenger_promotion_report_path": str(model_dir / "challenger_promotion_report.json"),
            "governance_status_path": str(model_dir / "governance_status.json"),
        },
    }
    cfg_path = tmp_path / "mrm_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    gov_mod.main(config_path=str(cfg_path), run_tag="run-governance-test")

    status = json.loads((model_dir / "governance_status.json").read_text(encoding="utf-8"))
    explanation = pd.read_parquet(data_dir / "explanation_drift.parquet")

    assert status["run_tag"] == "run-governance-test"
    assert "pass_explainability" in status["checks"]
    assert "pass_fairness" in status["checks"]
    assert "pass_score_psi" in status["checks"]
    assert "warnings" in status
    assert "warn_c2st" in status["warnings"]
    assert "distribution_warning_ratio" in status["summary"]
    assert "challenger_promotable" in status
    assert "explanation_drift_path" in status["artifacts"]
    assert not explanation.empty
