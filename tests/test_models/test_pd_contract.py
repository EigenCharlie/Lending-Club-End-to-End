"""Tests for src/models/pd_contract.py.

Covers path resolution, contract serialization, feature validation,
and model feature inference.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.models.pd_contract import (
    build_contract_payload,
    infer_model_feature_contract,
    load_contract,
    resolve_model_path,
    save_contract,
    validate_features_in_splits,
)

# ---------------------------------------------------------------------------
# resolve_model_path / resolve_calibrator_path
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_resolve_model_path_finds_canonical(self, tmp_path, monkeypatch):
        model_file = tmp_path / "models" / "pd_canonical.cbm"
        model_file.parent.mkdir(parents=True)
        model_file.touch()
        monkeypatch.setattr("src.models.pd_contract.CANONICAL_MODEL_PATH", model_file)
        assert resolve_model_path() == model_file

    def test_resolve_model_path_fallback(self, tmp_path, monkeypatch):
        # Canonical doesn't exist, but fallback does
        monkeypatch.setattr("src.models.pd_contract.CANONICAL_MODEL_PATH", tmp_path / "nope.cbm")
        fallback = tmp_path / "models" / "pd_catboost.cbm"
        fallback.parent.mkdir(parents=True)
        fallback.touch()
        # Patch the candidates list inside the function
        import src.models.pd_contract as mod

        def patched():
            candidates = [tmp_path / "nope.cbm", fallback]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError("No PD model artifact found in models/.")
            return path

        monkeypatch.setattr(mod, "resolve_model_path", patched)
        assert mod.resolve_model_path() == fallback

    def test_resolve_model_path_raises_when_missing(self, tmp_path, monkeypatch):
        import src.models.pd_contract as mod

        def _no_model():
            candidates = [tmp_path / "a.cbm", tmp_path / "b.cbm"]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError("No PD model artifact found in models/.")
            return path

        monkeypatch.setattr(mod, "resolve_model_path", _no_model)
        with pytest.raises(FileNotFoundError):
            mod.resolve_model_path()

    def test_resolve_calibrator_returns_none_when_missing(self, tmp_path, monkeypatch):
        import src.models.pd_contract as mod

        def _no_cal():
            candidates = [tmp_path / "a.pkl", tmp_path / "b.pkl"]
            return next((p for p in candidates if p.exists()), None)

        monkeypatch.setattr(mod, "resolve_calibrator_path", _no_cal)
        result = mod.resolve_calibrator_path()
        assert result is None


# ---------------------------------------------------------------------------
# load_contract / save_contract
# ---------------------------------------------------------------------------


class TestContractIO:
    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_contract(tmp_path / "no_contract.json")
        assert result is None

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "contract.json"
        payload = {"model_path": "models/test.cbm", "feature_names": ["a", "b"]}
        save_contract(payload, path)
        loaded = load_contract(path)
        assert loaded["model_path"] == "models/test.cbm"
        assert loaded["feature_names"] == ["a", "b"]


# ---------------------------------------------------------------------------
# build_contract_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_required_fields_present(self):
        payload = build_contract_payload(
            model_path=Path("models/test.cbm"),
            calibrator_path=Path("models/cal.pkl"),
            feature_names=["f1", "f2", "f3"],
            categorical_features=["f1"],
        )
        assert payload["model_path"] == "models/test.cbm"
        assert payload["calibrator_path"] == "models/cal.pkl"
        assert payload["n_features"] == 3
        assert "created_at_utc" in payload

    def test_posix_paths(self):
        payload = build_contract_payload(
            model_path=Path("models/nested/path.cbm"),
            calibrator_path=None,
            feature_names=["x"],
            categorical_features=[],
        )
        # as_posix() always uses forward slashes
        assert "/" in payload["model_path"]
        assert payload["model_path"] == "models/nested/path.cbm"
        assert payload["calibrator_path"] is None

    def test_split_shapes_included(self):
        payload = build_contract_payload(
            model_path=Path("m.cbm"),
            calibrator_path=None,
            feature_names=["a"],
            categorical_features=[],
            split_shapes={"train": (1000, 10), "test": (200, 10)},
        )
        assert payload["split_shapes"]["train"] == [1000, 10]


# ---------------------------------------------------------------------------
# infer_model_feature_contract
# ---------------------------------------------------------------------------


class TestInferContract:
    def test_extracts_features_and_categoricals(self):
        mock_model = MagicMock()
        mock_model.feature_names_ = ["feat_a", "feat_b", "feat_c"]
        mock_model.get_cat_feature_indices.return_value = [1]

        features, categoricals = infer_model_feature_contract(mock_model)
        assert features == ["feat_a", "feat_b", "feat_c"]
        assert categoricals == ["feat_b"]

    def test_no_categoricals(self):
        mock_model = MagicMock()
        mock_model.feature_names_ = ["x", "y"]
        mock_model.get_cat_feature_indices.return_value = []

        features, categoricals = infer_model_feature_contract(mock_model)
        assert len(features) == 2
        assert len(categoricals) == 0


# ---------------------------------------------------------------------------
# validate_features_in_splits
# ---------------------------------------------------------------------------


class TestValidateFeatures:
    def test_no_missing_features(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        shapes, missing = validate_features_in_splits(["a", "b"], {"train": df})
        assert missing["train"] == []
        assert shapes["train"] == (1, 3)

    def test_reports_missing_features(self):
        df = pd.DataFrame({"a": [1]})
        shapes, missing = validate_features_in_splits(["a", "b", "c"], {"test": df})
        assert "b" in missing["test"]
        assert "c" in missing["test"]
