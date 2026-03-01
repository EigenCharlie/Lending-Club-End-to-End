"""Tests for run comparison promotion gates."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_comparison as rc


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
            "conformal_status": {
                "coverage_90": 0.90,
                "coverage_95": 0.95,
                "min_group_coverage_90": 0.88,
                "winkler_90": 1.00,
                "critical_alerts": 0,
            },
            "fairness_status": {},
            "survival_summary": {},
            "model_comparison": {},
            "pipeline_summary": {},
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
            "conformal_status": {
                "coverage_90": 0.91,
                "coverage_95": 0.95,
                "min_group_coverage_90": 0.89,
                "winkler_90": 1.02,
                "critical_alerts": 0,
                "kupiec_pvalue_90": 0.001,
                "kupiec_pvalue_95": 0.50,
                "christoffersen_pvalue_90": 0.20,
                "christoffersen_pvalue_95": 0.20,
            },
            "fairness_status": {},
            "survival_summary": {},
            "model_comparison": {},
            "pipeline_summary": {},
        },
        "artifacts": {},
    }
    monkeypatch.setattr(rc, "_snapshot_payload", lambda _run_tag: current_snapshot)

    json_path, md_path = rc._write_compare("run-x", baseline_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")

    assert "conformal_promotion_pass" in report
    assert "conformal_statistical_warning" in report
    assert "conformal_failing_statistical_tests" in report
    assert report["conformal_promotion_pass"] is True
    assert report["conformal_statistical_warning"] is True
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
            "conformal_status": {},
            "fairness_status": {},
            "survival_summary": {},
            "model_comparison": {},
            "pipeline_summary": {},
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
