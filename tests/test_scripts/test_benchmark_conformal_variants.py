"""Tests for scripts/benchmark_conformal_variants.py."""

from __future__ import annotations

from scripts.benchmark_conformal_variants import _build_output_paths


def test_build_output_paths_uses_namespaced_shadow_locations() -> None:
    paths = _build_output_paths("abc/def")

    assert str(paths["data_dir"]).endswith("data/processed/conformal_gap/abc_def")
    assert str(paths["models_dir"]).endswith("models/conformal_gap/abc_def")
    assert str(paths["selection_status"]).endswith(
        "models/conformal_gap/abc_def/conformal_variant_selection_status.json"
    )
