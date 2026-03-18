"""Contract tests for Streamlit page storytelling metadata."""

from __future__ import annotations

from pathlib import Path

from streamlit_app.content.page_contracts import (
    PAGE_CONTRACTS,
    PAGE_TYPES,
    build_page_contract_registry,
)

PAGES_DIR = Path(__file__).resolve().parents[2] / "streamlit_app" / "pages"


def test_all_pages_have_story_contracts() -> None:
    page_ids = {p.stem for p in PAGES_DIR.glob("*.py")}
    registry = build_page_contract_registry()
    assert page_ids == set(registry), (
        f"Missing contracts for pages: {sorted(page_ids - set(registry))}"
    )
    assert page_ids == set(PAGE_CONTRACTS), "PAGE_CONTRACTS should be complete for all pages"


def test_page_types_are_valid() -> None:
    for page_id, contract in PAGE_CONTRACTS.items():
        assert contract.page_type in PAGE_TYPES, (
            f"{page_id}: invalid page_type {contract.page_type}"
        )


def test_next_pages_reference_existing_pages() -> None:
    page_ids = set(PAGE_CONTRACTS)
    for page_id, contract in PAGE_CONTRACTS.items():
        for next_page in contract.next_pages:
            assert next_page in page_ids, f"{page_id}: next_page `{next_page}` not found"


def test_research_contracts_remain_expert_first() -> None:
    for page_id, contract in PAGE_CONTRACTS.items():
        if contract.page_type in {"research", "paper_draft"}:
            assert contract.primary_audience == "Técnico", (
                f"{page_id}: research should be expert-first"
            )
            assert "how_to_read" not in contract.required_sections


def test_contracts_expose_book_and_pipeline_metadata() -> None:
    valid_axes = {"operational_pipeline", "insight_factory", "book_foundations"}
    valid_scopes = {"canonical", "insight", "research", "shared"}
    for page_id, contract in PAGE_CONTRACTS.items():
        assert contract.narrative_axis in valid_axes, (
            f"{page_id}: invalid narrative_axis {contract.narrative_axis}"
        )
        assert contract.artifact_scope in valid_scopes, (
            f"{page_id}: invalid artifact_scope {contract.artifact_scope}"
        )
        assert contract.pipeline_role, f"{page_id}: pipeline_role should not be empty"
        assert contract.book_chapter, f"{page_id}: book_chapter should not be empty"
