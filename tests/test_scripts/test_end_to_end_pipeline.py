"""Tests for scripts/end_to_end_pipeline.py fail-fast behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from scripts import end_to_end_pipeline as pipeline_mod


def _stub_module(monkeypatch, name: str, fn) -> None:
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(main=fn))


def test_main_fail_fast_returns_non_zero(monkeypatch) -> None:
    persisted: dict[str, str] = {}

    def persist(status: dict[str, str], elapsed: float) -> None:
        persisted.update(status)
        persisted["_elapsed"] = f"{elapsed:.2f}"

    def prepare_fail() -> None:
        raise RuntimeError("prepare failed")

    _stub_module(monkeypatch, "src.data.build_datasets", lambda: None)
    _stub_module(monkeypatch, "src.data.prepare_dataset", prepare_fail)
    monkeypatch.setattr(pipeline_mod, "_persist_status", persist)

    exit_code = pipeline_mod.main(run_name="test", continue_on_error=False)

    assert exit_code == 1
    assert persisted["data"].startswith("error:")


def test_main_continue_on_error_keeps_running(monkeypatch) -> None:
    persisted: dict[str, str] = {}

    def persist(status: dict[str, str], elapsed: float) -> None:
        persisted.update(status)
        persisted["_elapsed"] = f"{elapsed:.2f}"

    def prepare_fail() -> None:
        raise RuntimeError("prepare failed")

    _stub_module(monkeypatch, "src.data.build_datasets", lambda: None)
    _stub_module(monkeypatch, "src.data.prepare_dataset", prepare_fail)
    _stub_module(monkeypatch, "scripts.train_pd_model", lambda *args, **kwargs: None)
    _stub_module(monkeypatch, "scripts.forecast_default_rates", lambda **kwargs: None)
    _stub_module(monkeypatch, "scripts.generate_conformal_intervals", lambda: None)
    _stub_module(monkeypatch, "scripts.estimate_causal_effects", lambda *args, **kwargs: None)
    _stub_module(monkeypatch, "scripts.optimize_portfolio", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_mod, "_persist_status", persist)

    exit_code = pipeline_mod.main(run_name="test", continue_on_error=True)

    assert exit_code == 0
    assert persisted["data"].startswith("error:")
    assert persisted["pd_model"] == "ok"
    assert persisted["optimization"] == "ok"
    assert persisted["modeva_governance_side_task"] == "skipped"
