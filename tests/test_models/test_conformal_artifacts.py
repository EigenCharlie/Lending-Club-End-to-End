"""Tests for canonical conformal artifact loading contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import conformal_artifacts as ca


def test_load_conformal_intervals_uses_canonical_only(tmp_path, monkeypatch) -> None:
    canonical = tmp_path / "conformal_intervals_mondrian.parquet"
    legacy = tmp_path / "conformal_intervals.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(canonical, index=False)
    pd.DataFrame({"x": [3, 4]}).to_parquet(legacy, index=False)

    monkeypatch.setattr(ca, "CANONICAL_INTERVALS_PATH", canonical)
    path, is_legacy = ca.resolve_intervals_path(allow_legacy_fallback=True)
    assert path == canonical
    assert is_legacy is False


def test_resolve_intervals_path_raises_without_canonical_even_if_legacy_exists(
    tmp_path, monkeypatch
) -> None:
    canonical = tmp_path / "missing.parquet"
    legacy = tmp_path / "conformal_intervals.parquet"
    pd.DataFrame({"x": [3, 4]}).to_parquet(legacy, index=False)

    monkeypatch.setattr(ca, "CANONICAL_INTERVALS_PATH", canonical)
    with pytest.raises(FileNotFoundError):
        ca.resolve_intervals_path(allow_legacy_fallback=True)
