"""Unit tests for API router guardrails and alias normalization."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routers.conformal import _normalize_alpha
from api.routers.ecl import _normalize_scenario


def test_normalize_alpha_accepts_supported_levels():
    assert _normalize_alpha(0.10) == 0.10
    assert _normalize_alpha(0.05) == 0.05


def test_normalize_alpha_rejects_unsupported_level():
    with pytest.raises(HTTPException) as exc:
        _normalize_alpha(0.07)
    assert exc.value.status_code == 422
    assert "Valores permitidos" in str(exc.value.detail)


def test_normalize_scenario_accepts_canonical_names():
    assert _normalize_scenario("baseline") == "baseline"
    assert _normalize_scenario("mild_stress") == "mild_stress"
    assert _normalize_scenario("adverse") == "adverse"
    assert _normalize_scenario("severe") == "severe"


def test_normalize_scenario_accepts_aliases():
    assert _normalize_scenario("base") == "baseline"
    assert _normalize_scenario("optimistic") == "mild_stress"
    assert _normalize_scenario(" Baseline ") == "baseline"


def test_normalize_scenario_rejects_unknown_value():
    with pytest.raises(HTTPException) as exc:
        _normalize_scenario("impossible_scenario")
    assert exc.value.status_code == 422
    assert "scenario no soportado" in str(exc.value.detail)
