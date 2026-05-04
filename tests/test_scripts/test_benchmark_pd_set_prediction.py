"""Tests for namespaced PD set prediction benchmark paths."""

from __future__ import annotations

from scripts.benchmark_pd_set_prediction import _build_paths


def test_build_paths_uses_namespaced_locations() -> None:
    paths = _build_paths("abc/def")

    assert str(paths["cases"]).endswith(
        "data/processed/conformal_gap/abc_def/pd_set_prediction_cases.parquet"
    )
    assert str(paths["status"]).endswith(
        "models/conformal_gap/abc_def/pd_set_prediction_status.json"
    )
