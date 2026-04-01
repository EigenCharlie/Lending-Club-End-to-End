"""Tests for scripts/validate_conformal_experiment.py."""

from __future__ import annotations

from scripts.validate_conformal_experiment import _build_paths


def test_build_paths_uses_namespaced_artifact_locations() -> None:
    paths = _build_paths("abc/def")

    assert str(paths["data_dir"]).endswith("data/processed/conformal_gap/abc_def")
    assert str(paths["models_dir"]).endswith("models/conformal_gap/abc_def")
    assert str(paths["policy_status"]).endswith(
        "models/conformal_gap/abc_def/conformal_policy_status.json"
    )
