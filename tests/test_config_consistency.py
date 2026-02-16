"""Tests for config/code consistency.

Validates that configuration files match actual code usage patterns,
preventing the class of bug where configs drift from implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pd_model.yaml"
CONTRACT_PATH = PROJECT_ROOT / "models" / "pd_model_contract.json"


@pytest.fixture
def pd_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestCalibrationConfig:
    def test_calibration_method_is_valid(self, pd_config: dict) -> None:
        method = pd_config["calibration"]["method"]
        assert method in {"isotonic", "platt", "venn_abers"}, (
            f"Unknown calibration method: {method}"
        )

    def test_calibration_method_matches_thesis(self, pd_config: dict) -> None:
        """Thesis selected Platt sigmoid (ECE=0.0128). Config must agree."""
        method = pd_config["calibration"]["method"]
        assert method == "platt", (
            f"Config says '{method}' but thesis/CLAUDE.md specifies Platt sigmoid"
        )


class TestConformalConfig:
    def test_uses_mapie_13_params(self, pd_config: dict) -> None:
        """Config should use MAPIE 1.3+ parameter names."""
        conf = pd_config.get("conformal", {})
        legacy_keys = {"alpha", "method", "cv"}
        found_legacy = legacy_keys & set(conf.keys())
        assert not found_legacy, (
            f"Found legacy MAPIE <1.0 params: {found_legacy}. "
            "Use confidence_level/prefit/estimator instead."
        )

    def test_confidence_level_valid(self, pd_config: dict) -> None:
        conf = pd_config.get("conformal", {})
        level = conf.get("confidence_level", 0.9)
        assert 0.5 < level < 1.0, f"confidence_level={level} outside valid range (0.5, 1.0)"


class TestModelContract:
    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_paths_are_posix(self) -> None:
        """Model contract paths must not contain Windows backslashes."""
        contract = json.loads(CONTRACT_PATH.read_text())
        for key in ("model_path", "calibrator_path"):
            path = contract.get(key, "")
            if path:
                assert "\\" not in path, (
                    f"Contract {key}='{path}' contains Windows backslashes. "
                    "Use Path.as_posix() in pd_contract.py."
                )

    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_contract_features_not_empty(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text())
        features = contract.get("feature_names", [])
        assert len(features) > 0, "Model contract has no features"

    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_no_missing_features_in_splits(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text())
        missing = contract.get("split_missing_features", {})
        for split_name, missing_list in missing.items():
            assert len(missing_list) == 0, (
                f"Split '{split_name}' is missing features: {missing_list}"
            )
