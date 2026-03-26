"""Guardrails for the Streamlit -> Quarto migration registry."""

from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY_PATH = Path("docs/STREAMLIT_QUARTO_MIGRATION_REGISTRY.yml")
PAGES_DIR = Path("streamlit_app/pages")


def test_registry_covers_all_streamlit_pages() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    page_ids = {path.stem for path in PAGES_DIR.glob("*.py")}
    registry_ids = {entry["page_id"] for entry in registry["pages"]}
    assert page_ids <= registry_ids, (
        f"Registry is missing active pages: {sorted(page_ids - registry_ids)}"
    )
    active_entries = [entry for entry in registry["pages"] if entry["page_id"] in page_ids]
    assert active_entries, "Expected registry entries for active companion pages"
    for entry in active_entries:
        decisions = {block["decision"] for block in entry["blocks"]}
        assert "keep_streamlit_only" in decisions or "migrate_to_quarto" in decisions


def test_registry_blocks_expose_required_decisions() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    valid_decisions = {"already_in_quarto", "migrate_to_quarto", "keep_streamlit_only", "drop"}
    valid_coverage = {"none", "partial", "full"}
    valid_interaction = {"none", "light", "strong"}
    valid_stability = {"stable", "exploratory"}
    valid_dependency = {"can_crystallize", "interaction_required"}
    valid_target_surfaces = {"quarto", "streamlit_lab", "docs_internal", "drop"}

    for entry in registry["pages"]:
        blocks = entry.get("blocks", [])
        assert blocks, f"{entry['page_id']}: expected at least one block"
        for block in blocks:
            assert block["decision"] in valid_decisions
            assert block["cobertura_quarto"] in valid_coverage
            assert block["interactividad_real"] in valid_interaction
            assert block["estabilidad_editorial"] in valid_stability
            assert block["dependencia_streamlit"] in valid_dependency
            assert block["target_surface"] in valid_target_surfaces
            assert "valor_defendible" in block and block["valor_defendible"]
