"""Tests for scripts/run_calibration_mapping_shadow_validation.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_calibration_mapping_shadow_validation as shadow_mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_shadow_validation_cuts_early_when_no_pd_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"generate": 0, "benchmark": 0, "backtest": 0, "validate": 0}

    def _mapping_stub(run_tag: str | None = None) -> None:
        _write_json(
            tmp_path / "models" / "calibration_mapping_status.json",
            {
                "stage_a_pass": False,
                "current_candidate": {"candidate_id": "current_identity"},
                "best_candidate": {"candidate_id": "current_identity"},
            },
        )

    monkeypatch.setattr(shadow_mod, "mapping_diagnostics_main", _mapping_stub)
    monkeypatch.setattr(
        shadow_mod,
        "generate_conformal_main",
        lambda **kwargs: calls.__setitem__("generate", calls["generate"] + 1),
    )
    monkeypatch.setattr(
        shadow_mod,
        "benchmark_conformal_main",
        lambda **kwargs: calls.__setitem__("benchmark", calls["benchmark"] + 1),
    )
    monkeypatch.setattr(
        shadow_mod,
        "backtest_conformal_main",
        lambda **kwargs: calls.__setitem__("backtest", calls["backtest"] + 1),
    )
    monkeypatch.setattr(
        shadow_mod,
        "validate_conformal_policy_main",
        lambda **kwargs: calls.__setitem__("validate", calls["validate"] + 1),
    )

    shadow_mod.main(run_tag="run-shadow-test")

    status = json.loads(
        (tmp_path / "models" / "calibration_mapping_shadow_impact_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["recommendation"] == "keep_current_calibrator"
    assert status["shadow_validation_executed"] is False
    assert calls == {"generate": 0, "benchmark": 0, "backtest": 0, "validate": 0}


def test_shadow_validation_runs_namespaced_flow_without_touching_canonical(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical_policy = {
        "overall_pass": True,
        "avg_width_90": 0.75,
        "min_group_coverage_90": 0.89,
    }
    canonical_variant = {
        "selected_variant": "score_decile_mondrian",
        "selected_metrics": {"stability_over_time": 0.04},
    }
    _write_json(tmp_path / "models" / "conformal_policy_status.json", canonical_policy)
    _write_json(tmp_path / "models" / "conformal_variant_selection_status.json", canonical_variant)

    def _mapping_stub(run_tag: str | None = None) -> None:
        _write_json(
            tmp_path / "models" / "calibration_mapping_status.json",
            {
                "stage_a_pass": True,
                "shadow_namespace": "shadow_ns",
                "shadow_candidate_path": "models/calibration_mapping_shadow/shadow_ns/pd_shadow_calibrator.pkl",
                "current_candidate": {
                    "candidate_id": "current_identity",
                    "abs_global_gap_bp": 100.0,
                },
                "best_candidate": {
                    "candidate_id": "logit_intercept_shift",
                    "abs_global_gap_bp": 80.0,
                },
            },
        )

    def _generate_stub(**kwargs) -> None:
        data_dir = tmp_path / "data" / "processed" / "conformal_gap" / "shadow_ns"
        models_dir = tmp_path / "models" / "conformal_gap" / "shadow_ns"
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "conformal_results_mondrian.pkl").write_bytes(b"placeholder")
        (data_dir / "conformal_intervals_mondrian.parquet").write_bytes(b"placeholder")

    def _benchmark_stub(**kwargs) -> None:
        _write_json(
            tmp_path
            / "models"
            / "conformal_gap"
            / "shadow_ns"
            / "conformal_variant_selection_status.json",
            {
                "selected_variant": "score_decile_mondrian",
                "selected_metrics": {"stability_over_time": 0.045},
            },
        )

    def _backtest_stub(**kwargs) -> None:
        data_dir = tmp_path / "data" / "processed" / "conformal_gap" / "shadow_ns"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "conformal_backtest_monthly.parquet").write_bytes(b"placeholder")
        (data_dir / "conformal_backtest_alerts.parquet").write_bytes(b"placeholder")

    def _validate_stub(**kwargs) -> None:
        _write_json(
            tmp_path / "models" / "conformal_gap" / "shadow_ns" / "conformal_policy_status.json",
            {
                "overall_pass": True,
                "avg_width_90": 0.76,
                "min_group_coverage_90": 0.889,
            },
        )

    monkeypatch.setattr(shadow_mod, "mapping_diagnostics_main", _mapping_stub)
    monkeypatch.setattr(shadow_mod, "generate_conformal_main", _generate_stub)
    monkeypatch.setattr(shadow_mod, "benchmark_conformal_main", _benchmark_stub)
    monkeypatch.setattr(shadow_mod, "backtest_conformal_main", _backtest_stub)
    monkeypatch.setattr(shadow_mod, "validate_conformal_policy_main", _validate_stub)

    shadow_mod.main(run_tag="run-shadow-test")

    status = json.loads(
        (tmp_path / "models" / "calibration_mapping_shadow_impact_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["recommendation"] == "shadow_candidate_pd_and_conformal"
    assert status["shadow_validation_executed"] is True

    untouched_policy = json.loads(
        (tmp_path / "models" / "conformal_policy_status.json").read_text(encoding="utf-8")
    )
    assert untouched_policy == canonical_policy
