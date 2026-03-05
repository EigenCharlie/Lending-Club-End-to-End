"""Tests for run comparison promotion gates."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_comparison as rc


def test_gate_artifact_coherence_passes_with_consistent_metadata() -> None:
    cur_metrics = {
        "dvc_metrics_meta": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:00:00+00:00",
            "run_tag": "run-xyz",
        },
        "pipeline_summary": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:05:00+00:00",
            "run_tag": "run-xyz",
        },
        "conformal_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:10:00+00:00",
            "run_tag": "run-xyz",
        },
        "fairness_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:15:00+00:00",
            "run_tag": "run-xyz",
        },
        "governance_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:20:00+00:00",
            "run_tag": "run-xyz",
        },
        "ab_simulation_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:25:00+00:00",
            "run_tag": "run-xyz",
        },
    }
    gate = rc._gate_artifact_coherence(cur_metrics, "run-xyz")
    assert gate.passed is True
    assert gate.details["run_tag_matches_expected"] is True
    assert gate.details["all_have_metadata"] is True


def test_gate_artifact_coherence_fails_on_missing_or_mismatched_run_tag() -> None:
    cur_metrics = {
        "dvc_metrics_meta": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:00:00+00:00",
            "run_tag": "run-xyz",
        },
        "pipeline_summary": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:05:00+00:00",
            "run_tag": "run-xyz",
        },
        "conformal_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:10:00+00:00",
            "run_tag": "other-run",
        },
        "fairness_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:15:00+00:00",
        },
        "governance_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:20:00+00:00",
            "run_tag": "run-xyz",
        },
        "ab_simulation_status": {
            "schema_version": "2026-03-01.1",
            "generated_at_utc": "2026-03-01T10:25:00+00:00",
            "run_tag": "run-xyz",
        },
    }
    gate = rc._gate_artifact_coherence(cur_metrics, "run-xyz")
    assert gate.passed is False
    assert gate.details["run_tag_matches_expected"] is False
    assert gate.details["all_have_metadata"] is False
    assert "models/conformal_policy_status.json" in gate.details["mismatched_run_tag_artifacts"]
    assert "models/fairness_audit_status.json" in gate.details["missing_metadata_artifacts"]


def test_gate_conformal_passes_with_statistical_warning() -> None:
    baseline = {
        "conformal_status": {
            "coverage_90": 0.90,
            "coverage_95": 0.95,
            "min_group_coverage_90": 0.88,
            "winkler_90": 1.00,
            "critical_alerts": 0,
        }
    }
    current = {
        "conformal_status": {
            "coverage_90": 0.905,
            "coverage_95": 0.952,
            "min_group_coverage_90": 0.889,
            "winkler_90": 1.06,
            "critical_alerts": 0,
            "overall_pass": False,
            "kupiec_pvalue_90": 0.001,
            "kupiec_pvalue_95": 0.20,
            "christoffersen_pvalue_90": 0.004,
            "christoffersen_pvalue_95": 0.30,
        }
    }

    gate = rc._gate_conformal(baseline, current)

    assert gate.passed is True
    assert gate.details["checks"]["conformal_promotion_pass"] is True
    assert gate.details["diagnostics"]["statistical_warning"] is True
    assert "kupiec_pvalue_90" in gate.details["diagnostics"]["failing_statistical_tests"]
    assert "christoffersen_pvalue_90" in gate.details["diagnostics"]["failing_statistical_tests"]


def test_gate_conformal_fails_on_material_coverage_regression() -> None:
    baseline = {
        "conformal_status": {
            "coverage_90": 0.90,
            "coverage_95": 0.95,
            "min_group_coverage_90": 0.88,
            "winkler_90": 1.00,
            "critical_alerts": 0,
        }
    }
    current = {
        "conformal_status": {
            "coverage_90": 0.84,
            "coverage_95": 0.90,
            "min_group_coverage_90": 0.82,
            "winkler_90": 1.05,
            "critical_alerts": 0,
            "kupiec_pvalue_90": 0.50,
            "kupiec_pvalue_95": 0.50,
            "christoffersen_pvalue_90": 0.50,
            "christoffersen_pvalue_95": 0.50,
        }
    }

    gate = rc._gate_conformal(baseline, current)

    assert gate.passed is False
    assert gate.details["checks"]["coverage90_ok"] is False
    assert gate.details["checks"]["coverage95_ok"] is False
    assert gate.details["checks"]["min_group_coverage90_ok"] is False


def test_gate_ab_no_regression_ignores_significance_when_return_ok() -> None:
    baseline = {
        "ab_simulation_status": {
            "metrics_a": {"total_return": 50_000.0},
            "metrics_b": {"total_return": 49_000.0},
        }
    }
    current = {
        "ab_simulation_status": {
            "metrics_a": {"total_return": 49_500.0},
            "metrics_b": {"total_return": 48_000.0},
            "comparison": {"p_value": 0.60, "significant": False},
            "no_regression": {
                "diff_total_return": -1500.0,
                "tolerance_total_return": 2500.0,
                "passed": True,
            },
        }
    }
    gate = rc._gate_ab_no_regression(baseline, current)
    assert gate.passed is True
    assert gate.details["diagnostics"]["significant"] is False
    assert gate.details["diagnostics"]["significance_role"] == "diagnostic"


def test_gate_ab_no_regression_fails_on_material_regression() -> None:
    baseline = {
        "ab_simulation_status": {
            "metrics_a": {"total_return": 50_000.0},
            "metrics_b": {"total_return": 49_000.0},
        }
    }
    current = {
        "ab_simulation_status": {
            "metrics_a": {"total_return": 44_000.0},
            "metrics_b": {"total_return": 40_000.0},
            "comparison": {"p_value": 0.20, "significant": False},
            "no_regression": {
                "diff_total_return": -4000.0,
                "tolerance_total_return": 1000.0,
                "passed": False,
            },
        }
    }
    gate = rc._gate_ab_no_regression(baseline, current)
    assert gate.passed is False
    assert gate.details["checks"]["self_no_regression_ok"] is False


def test_gate_fairness_absolute_business_passes_with_business_threshold(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rc,
        "_load_fairness_policy_contract",
        lambda: {
            "prediction_threshold": 0.50,
            "outcome_mode": "approval",
            "use_artifact": False,
            "policy_path": "configs/fairness_policy.yaml",
        },
    )
    current = {
        "fairness_status": {
            "overall_pass": True,
            "n_passed": 3,
            "n_attributes": 3,
            "prediction_threshold": 0.50,
            "prediction_threshold_source": "policy_default",
            "outcome_mode": "approval",
        }
    }

    gate = rc._gate_fairness_absolute_business({}, current)
    assert gate.passed is True
    assert gate.details["checks"]["threshold_match_ok"] is True
    assert gate.details["checks"]["threshold_source_ok"] is True


def test_gate_fairness_absolute_business_fails_on_threshold_or_source_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rc,
        "_load_fairness_policy_contract",
        lambda: {
            "prediction_threshold": 0.50,
            "outcome_mode": "approval",
            "use_artifact": False,
            "policy_path": "configs/fairness_policy.yaml",
        },
    )
    current = {
        "fairness_status": {
            "overall_pass": True,
            "n_passed": 3,
            "n_attributes": 3,
            "prediction_threshold": 0.05,
            "prediction_threshold_source": "artifact",
            "outcome_mode": "default",
        }
    }

    gate = rc._gate_fairness_absolute_business({}, current)
    assert gate.passed is False
    assert gate.details["checks"]["threshold_match_ok"] is False
    assert gate.details["checks"]["threshold_source_ok"] is False
    assert gate.details["checks"]["outcome_mode_ok"] is False


def test_write_compare_exports_conformal_promotion_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "OUT_ROOT", tmp_path / "reports" / "run_comparisons")

    baseline = {
        "schema_version": rc.SCHEMA_VERSION,
        "run_tag": "baseline",
        "generated_at_utc": "2026-02-27T00:00:00+00:00",
        "git": {"head": "base", "branch": "main", "status_short": ""},
        "metrics": {
            "dvc_metrics": {},
            "dvc_metrics_meta": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:00:00+00:00",
                "run_tag": "run-x",
            },
            "pipeline_summary": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:01:00+00:00",
                "run_tag": "run-x",
            },
            "conformal_status": {
                "coverage_90": 0.90,
                "coverage_95": 0.95,
                "min_group_coverage_90": 0.88,
                "winkler_90": 1.00,
                "critical_alerts": 0,
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:00:00+00:00",
                "run_tag": "run-x",
            },
            "fairness_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:05:00+00:00",
                "run_tag": "run-x",
            },
            "governance_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:07:00+00:00",
                "run_tag": "run-x",
            },
            "survival_summary": {},
            "model_comparison": {},
            "ab_simulation_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:10:00+00:00",
                "run_tag": "run-x",
            },
        },
        "artifacts": {},
    }
    baseline_path = tmp_path / "baseline_snapshot.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    current_snapshot = {
        "schema_version": rc.SCHEMA_VERSION,
        "run_tag": "current",
        "generated_at_utc": "2026-02-27T01:00:00+00:00",
        "git": {"head": "cur", "branch": "feature", "status_short": ""},
        "metrics": {
            "dvc_metrics": {},
            "dvc_metrics_meta": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:00:00+00:00",
                "run_tag": "run-x",
            },
            "pipeline_summary": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:01:00+00:00",
                "run_tag": "run-x",
            },
            "conformal_status": {
                "coverage_90": 0.91,
                "coverage_95": 0.95,
                "min_group_coverage_90": 0.89,
                "winkler_90": 1.02,
                "critical_alerts": 0,
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:00:00+00:00",
                "run_tag": "run-x",
                "kupiec_pvalue_90": 0.001,
                "kupiec_pvalue_95": 0.50,
                "christoffersen_pvalue_90": 0.20,
                "christoffersen_pvalue_95": 0.20,
            },
            "fairness_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:05:00+00:00",
                "run_tag": "run-x",
            },
            "governance_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:07:00+00:00",
                "run_tag": "run-x",
            },
            "survival_summary": {},
            "model_comparison": {},
            "ab_simulation_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T01:10:00+00:00",
                "run_tag": "run-x",
            },
        },
        "artifacts": {},
    }
    monkeypatch.setattr(rc, "_snapshot_payload", lambda _run_tag: current_snapshot)

    json_path, md_path = rc._write_compare("run-x", baseline_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")

    assert "conformal_promotion_pass" in report
    assert "artifact_coherence_pass" in report
    assert report["artifact_coherence_pass"] is True
    assert "conformal_statistical_warning" in report
    assert "conformal_failing_statistical_tests" in report
    assert report["conformal_promotion_pass"] is True
    assert report["conformal_statistical_warning"] is True
    assert "ab_no_regression_pass" in report
    assert "fairness_absolute_business_pass" in report
    assert report["ab_gate_mode"] == "no_regression"
    assert report["ab_significance_role"] == "diagnostic"
    assert "`kupiec_pvalue_90`" in md


def test_write_compare_accepts_relative_baseline_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "OUT_ROOT", tmp_path / "reports" / "run_comparisons")
    monkeypatch.chdir(tmp_path)

    baseline = {
        "schema_version": rc.SCHEMA_VERSION,
        "run_tag": "baseline",
        "generated_at_utc": "2026-02-27T00:00:00+00:00",
        "git": {"head": "base", "branch": "main", "status_short": ""},
        "metrics": {
            "dvc_metrics": {},
            "dvc_metrics_meta": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:00:00+00:00",
                "run_tag": "run-rel",
            },
            "pipeline_summary": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:01:00+00:00",
                "run_tag": "run-rel",
            },
            "conformal_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:00:00+00:00",
                "run_tag": "run-rel",
            },
            "fairness_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:05:00+00:00",
                "run_tag": "run-rel",
            },
            "governance_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:07:00+00:00",
                "run_tag": "run-rel",
            },
            "survival_summary": {},
            "model_comparison": {},
            "ab_simulation_status": {
                "schema_version": "2026-03-01.1",
                "generated_at_utc": "2026-02-27T00:10:00+00:00",
                "run_tag": "run-rel",
            },
        },
        "artifacts": {},
    }
    baseline_rel = Path("baseline_snapshot.json")
    baseline_rel.write_text(json.dumps(baseline), encoding="utf-8")

    current_snapshot = {
        "schema_version": rc.SCHEMA_VERSION,
        "run_tag": "current",
        "generated_at_utc": "2026-02-27T01:00:00+00:00",
        "git": {"head": "cur", "branch": "feature", "status_short": ""},
        "metrics": baseline["metrics"],
        "artifacts": {},
    }
    monkeypatch.setattr(rc, "_snapshot_payload", lambda _run_tag: current_snapshot)

    json_path, _ = rc._write_compare("run-rel", baseline_rel)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["baseline_path"] == "baseline_snapshot.json"
