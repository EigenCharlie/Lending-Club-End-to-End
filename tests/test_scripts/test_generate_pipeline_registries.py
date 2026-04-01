"""Tests for pipeline registry snapshots."""

from __future__ import annotations

import json

from scripts import generate_pipeline_registries as registry_mod
from src.utils import pipeline_topology


def test_pipeline_topology_resolves_legacy_aliases() -> None:
    assert pipeline_topology.resolve_pipeline_family("canonical_rebuild") == "core_canonical"
    assert pipeline_topology.resolve_pipeline_family("champion_search") == "search_pd"
    assert pipeline_topology.resolve_pipeline_family("challenger_promotion") == "search_pd"


def test_generate_pipeline_registries_writes_json_snapshots(tmp_path, monkeypatch) -> None:
    registry_dir = tmp_path / "configs" / "pipeline_registry"
    out_dir = tmp_path / "models" / "pipeline_registry"
    registry_dir.mkdir(parents=True)
    (registry_dir / "pipeline_matrix.yaml").write_text(
        "schema_version: '1'\npipelines:\n  - pipeline: core_canonical\n",
        encoding="utf-8",
    )
    (registry_dir / "search_registry.yaml").write_text(
        "schema_version: '1'\nsearches:\n  - search: pd_hpo\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_mod, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(registry_mod, "OUT_DIR", out_dir)

    assert registry_mod.main() == 0
    pipeline_json = json.loads((out_dir / "pipeline_matrix.json").read_text(encoding="utf-8"))
    search_json = json.loads((out_dir / "search_registry.json").read_text(encoding="utf-8"))
    assert pipeline_json["pipelines"][0]["pipeline"] == "core_canonical"
    assert search_json["searches"][0]["search"] == "pd_hpo"
