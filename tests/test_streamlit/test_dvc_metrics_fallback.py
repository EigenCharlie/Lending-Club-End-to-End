"""Tests for DVC metrics loading fallback behavior in Streamlit utils."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import streamlit_app.utils as utils


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_dvc_metrics_summary_falls_back_to_pipeline_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data" / "processed"
    dvc_dir = tmp_path / "reports" / "dvc"
    _write_json(
        data_dir / "pipeline_summary.json",
        {
            "pd_model": {
                "final_auc": 0.71,
                "final_gini": 0.42,
                "final_brier": 0.16,
                "final_ece": 0.01,
            },
            "conformal": {"coverage_90": 0.91, "coverage_95": 0.95},
            "pipeline": {
                "interval_width_mean": 0.8,
                "ecl_expected": 100.0,
                "ecl_conservative": 150.0,
                "robust_return": 10.0,
                "nonrobust_return": 20.0,
                "price_of_robustness": 10.0,
            },
        },
    )

    monkeypatch.setattr(utils, "DATA_DIR", data_dir)
    monkeypatch.setattr(utils, "DVC_REPORTS_DIR", dvc_dir)
    utils.load_dvc_metrics_summary.clear()

    metrics = utils.load_dvc_metrics_summary()

    assert metrics["pd.auc"] == pytest.approx(0.71)
    assert metrics["conformal.coverage90"] == pytest.approx(0.91)
    assert metrics["conformal.avg_width90"] == pytest.approx(0.8)
    assert metrics["ifrs9.ecl_baseline"] == pytest.approx(100.0)
    assert metrics["ifrs9.ecl_severe"] == pytest.approx(150.0)
    assert metrics["ifrs9.severe_uplift_pct"] == pytest.approx(50.0)
    assert metrics["optimization.price_of_robustness"] == pytest.approx(10.0)

    utils.load_dvc_metrics_summary.clear()


def test_load_dvc_metrics_summary_prefers_dvc_metrics_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data" / "processed"
    dvc_dir = tmp_path / "reports" / "dvc"
    _write_json(
        data_dir / "pipeline_summary.json",
        {"pd_auc": 0.5},
    )
    _write_json(
        dvc_dir / "metrics_summary.json",
        {"metrics": {"pd.auc": 0.77, "pd.ece": 0.02}},
    )

    monkeypatch.setattr(utils, "DATA_DIR", data_dir)
    monkeypatch.setattr(utils, "DVC_REPORTS_DIR", dvc_dir)
    utils.load_dvc_metrics_summary.clear()

    metrics = utils.load_dvc_metrics_summary()

    assert metrics["pd.auc"] == pytest.approx(0.77)
    assert metrics["pd.ece"] == pytest.approx(0.02)

    utils.load_dvc_metrics_summary.clear()
