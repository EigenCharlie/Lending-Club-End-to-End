"""Tests for new public pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path

from scripts import (
    run_canonical_rebuild,
    run_champion_search,
    run_insights_factory,
    run_smoke_pipeline,
)
from scripts.search import run_paper2_ifrs9_search, run_pd_hpo_local


def test_run_champion_search_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_champion_search, "_main", _fake_main)
    assert run_champion_search.main(["--run-tag", "champion-test"]) == 0
    assert captured["argv"] == ["--run-tag", "champion-test"]
    assert captured["kwargs"]["default_pipeline_family"] == "search_pd"
    assert captured["kwargs"]["default_sampling_profile"] == "mega64plus"
    assert captured["kwargs"]["default_include_rapids"] is False
    assert captured["kwargs"]["default_include_notebooks"] is False


def test_run_canonical_rebuild_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_canonical_rebuild, "_main", _fake_main)
    assert run_canonical_rebuild.main(["--run-tag", "canonical-test"]) == 0
    assert captured["kwargs"]["default_pipeline_family"] == "core_canonical"
    assert captured["kwargs"]["default_sampling_profile"] == "champion64safe"
    assert captured["kwargs"]["default_include_notebooks"] is False


def test_run_paper2_ifrs9_search_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_paper2_ifrs9_search, "_main", _fake_main)
    assert run_paper2_ifrs9_search.main(["--run-tag", "paper2-search"]) == 0
    assert captured["argv"] == ["--run-tag", "paper2-search"]
    assert captured["kwargs"]["default_pipeline_family"] == "search_paper2_ifrs9"
    assert captured["kwargs"]["default_sampling_profile"] == "mega64safe"
    assert captured["kwargs"]["default_include_rapids"] is False
    assert captured["kwargs"]["default_include_notebooks"] is False


def test_run_pd_hpo_local_builds_config_and_delegates(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    repo_root = Path(__file__).resolve().parents[2]
    generated = repo_root / "models" / "search_pd" / "pd-hpo-local-test" / "pd_model_hpo_local.yaml"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("model: {}\n", encoding="utf-8")

    def _fake_build(**kwargs):
        captured["build_kwargs"] = kwargs
        return generated

    def _fake_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(run_pd_hpo_local, "build_pd_hpo_local_config", _fake_build)
    monkeypatch.setattr(run_pd_hpo_local, "_main", _fake_main)

    assert (
        run_pd_hpo_local.main(
            [
                "--run-tag",
                "pd-hpo-local",
                "--base-search-run-tag",
                "pd-blockwise-exhaustive-2026-04-02-0940",
            ]
        )
        == 0
    )
    assert (
        captured["build_kwargs"]["base_search_run_tag"] == "pd-blockwise-exhaustive-2026-04-02-0940"
    )
    assert "--pd-config-override" in captured["argv"]
    assert "search_pd_hpo_local_exhaustive" in captured["argv"]
    assert captured["kwargs"]["default_pipeline_family"] == "search_pd"
    assert captured["kwargs"]["default_sampling_profile"] == "mega64plus"
    generated.unlink(missing_ok=True)


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
