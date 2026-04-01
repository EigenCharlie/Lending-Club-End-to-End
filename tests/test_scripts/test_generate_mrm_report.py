"""Tests for scripts/generate_mrm_report.py compliance gate behavior."""

from __future__ import annotations

import json
import pickle

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from scripts import generate_mrm_report as mrm_mod


def test_generate_mrm_report_overall_pass_with_pipeline_summary(tmp_path) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports" / "mrm"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    (data_dir / "pipeline_summary.json").write_text(
        json.dumps({"pipeline": {"batch_size": 10}, "pd_model": {"final_auc": 0.72}}),
        encoding="utf-8",
    )
    (model_dir / "conformal_policy_status.json").write_text(
        json.dumps({"overall_pass": True}), encoding="utf-8"
    )
    (model_dir / "governance_status.json").write_text(
        json.dumps({"overall_pass": True}), encoding="utf-8"
    )
    (model_dir / "fairness_audit_status.json").write_text(
        json.dumps({"overall_pass": True}), encoding="utf-8"
    )

    cfg = {
        "model": {
            "name": "CorePD",
            "version": "1.0",
            "owner": "owner",
            "champion_artifact": "models/pd_canonical.cbm",
        },
        "governance": {"validation_frequency_days": 90, "review_cadence": "quarterly"},
        "retraining_triggers": {
            "psi_threshold": 0.25,
            "auc_degradation_threshold": 0.03,
            "coverage_degradation_threshold": 0.02,
        },
        "challenger": {"criteria": [], "promotion_requires": []},
        "artifacts": {
            "pipeline_summary": str(data_dir / "pipeline_summary.json"),
            "conformal_status": str(model_dir / "conformal_policy_status.json"),
            "governance_status": str(model_dir / "governance_status.json"),
            "fairness_status": str(model_dir / "fairness_audit_status.json"),
        },
        "output": {"mrm_report_json": str(report_dir / "mrm_validation_report.json")},
    }
    cfg_path = tmp_path / "mrm_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    mrm_mod.main(str(cfg_path))

    report = json.loads((report_dir / "mrm_validation_report.json").read_text(encoding="utf-8"))
    summary = report["compliance_summary"]
    assert summary["overall_pass"] is True
    assert summary["n_passing"] == 4
    assert summary["subsystems"]["pipeline"] is True
    assert "diagnostic_statuses" in report
    assert "bootstrap_validation" in report["diagnostic_statuses"]
    assert "calibration_mapping" in report["diagnostic_statuses"]
    assert "model_shift" in report["diagnostic_statuses"]
    assert "skops_governance" in report


def test_generate_mrm_report_exports_skops_sidecar(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports" / "mrm"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with open(model_dir / "pd_logreg_baseline.pkl", "wb") as fh:
        pickle.dump({"model": model, "feature_names": ["x"], "fill_values": {"x": 0.0}}, fh)

    (data_dir / "pipeline_summary.json").write_text(
        json.dumps({"pipeline": {"batch_size": 10}, "pd_model": {"final_auc": 0.72}}),
        encoding="utf-8",
    )
    for name in [
        "conformal_policy_status.json",
        "governance_status.json",
        "fairness_audit_status.json",
    ]:
        (model_dir / name).write_text(json.dumps({"overall_pass": True}), encoding="utf-8")

    cfg = {
        "model": {
            "name": "CorePD",
            "version": "1.0",
            "owner": "owner",
            "champion_artifact": "models/pd_canonical.cbm",
        },
        "governance": {"validation_frequency_days": 90, "review_cadence": "quarterly"},
        "retraining_triggers": {
            "psi_threshold": 0.25,
            "auc_degradation_threshold": 0.03,
            "coverage_degradation_threshold": 0.02,
        },
        "challenger": {"criteria": [], "promotion_requires": []},
        "artifacts": {
            "pipeline_summary": str(data_dir / "pipeline_summary.json"),
            "conformal_status": str(model_dir / "conformal_policy_status.json"),
            "governance_status": str(model_dir / "governance_status.json"),
            "fairness_status": str(model_dir / "fairness_audit_status.json"),
        },
        "output": {"mrm_report_json": str(report_dir / "mrm_validation_report.json")},
    }
    cfg_path = tmp_path / "mrm_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    mrm_mod.main(str(cfg_path))

    report = json.loads((report_dir / "mrm_validation_report.json").read_text(encoding="utf-8"))
    skops_sidecar = report["skops_governance"]
    assert skops_sidecar["exports"][0]["status"] == "exported"
    assert (report_dir / "skops" / "pd_logreg_baseline.skops").exists()
    assert (report_dir / "corepd_model_card.json").exists()
    assert (report_dir / "corepd_model_card.html").exists()


def test_generate_mrm_report_prefers_explicit_or_env_run_tag_over_pipeline_summary(
    tmp_path, monkeypatch
) -> None:
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports" / "mrm"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    (data_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_tag": "stale-pipeline-run",
                "pipeline": {"batch_size": 10},
                "pd_model": {"final_auc": 0.72},
            }
        ),
        encoding="utf-8",
    )
    for name in [
        "conformal_policy_status.json",
        "governance_status.json",
        "fairness_audit_status.json",
    ]:
        (model_dir / name).write_text(json.dumps({"overall_pass": True}), encoding="utf-8")

    cfg = {
        "model": {"name": "CorePD", "version": "1.0", "owner": "owner"},
        "governance": {"validation_frequency_days": 90, "review_cadence": "quarterly"},
        "retraining_triggers": {"psi_threshold": 0.25},
        "challenger": {"criteria": [], "promotion_requires": []},
        "artifacts": {
            "pipeline_summary": str(data_dir / "pipeline_summary.json"),
            "conformal_status": str(model_dir / "conformal_policy_status.json"),
            "governance_status": str(model_dir / "governance_status.json"),
            "fairness_status": str(model_dir / "fairness_audit_status.json"),
        },
        "output": {"mrm_report_json": str(report_dir / "mrm_validation_report.json")},
    }
    cfg_path = tmp_path / "mrm_policy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_RUN_TAG", "env-run-tag")
    mrm_mod.main(str(cfg_path))

    report = json.loads((report_dir / "mrm_validation_report.json").read_text(encoding="utf-8"))
    assert report["run_tag"] == "env-run-tag"
