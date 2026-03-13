"""Tests for new public pipeline entrypoints."""

from __future__ import annotations

from scripts import (
    run_canonical_rebuild,
    run_champion_search,
    run_insights_factory,
    run_smoke_pipeline,
)


def test_run_champion_search_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_champion_search, "_main", _fake_main)
    assert run_champion_search.main(["--run-tag", "champion-test"]) == 0
    assert captured["argv"] == ["--run-tag", "champion-test"]
    assert captured["kwargs"]["default_pipeline_family"] == "champion_search"
    assert captured["kwargs"]["default_sampling_profile"] == "mega64plus"


def test_run_canonical_rebuild_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_canonical_rebuild, "_main", _fake_main)
    assert run_canonical_rebuild.main(["--run-tag", "canonical-test"]) == 0
    assert captured["kwargs"]["default_pipeline_family"] == "canonical_rebuild"
    assert captured["kwargs"]["default_sampling_profile"] == "champion64safe"


def test_run_smoke_pipeline_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, run_name, continue_on_error, skip_make_dataset):
        captured["run_name"] = run_name
        captured["continue_on_error"] = continue_on_error
        captured["skip_make_dataset"] = skip_make_dataset
        return 0

    monkeypatch.setattr(run_smoke_pipeline, "_main", _fake_main)
    monkeypatch.setattr(
        run_smoke_pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "run_name": "smoke-test",
                "continue_on_error": True,
                "skip_make_dataset": False,
            },
        )(),
    )
    assert run_smoke_pipeline.main() == 0
    assert captured["run_name"] == "smoke-test"
    assert captured["continue_on_error"] is True


def test_insights_factory_requires_upstream_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_insights_factory, "PRIMARY_BASELINE", tmp_path / "missing-primary.json")
    monkeypatch.setattr(run_insights_factory, "LEGACY_BASELINE", tmp_path / "missing-legacy.json")
    try:
        run_insights_factory._resolve_upstream_canonical_run_tag(None)
    except FileNotFoundError as exc:
        assert "canonical baseline registry" in str(exc)
    else:
        raise AssertionError("Expected missing baseline registry to fail")
