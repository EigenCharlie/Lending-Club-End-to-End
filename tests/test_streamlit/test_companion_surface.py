"""Tests for the reduced local Streamlit companion surface."""

from __future__ import annotations

from streamlit_app.content.companion_surface import (
    ACTIVE_COMPANION_LABS,
    ACTIVE_COMPANION_PAGE_IDS,
)
from streamlit_app.content.page_contracts import PAGE_CONTRACTS


def test_companion_surface_has_five_labs() -> None:
    assert len(ACTIVE_COMPANION_LABS) == 5
    assert len(ACTIVE_COMPANION_PAGE_IDS) == 5


def test_companion_surface_pages_exist_in_contract_registry() -> None:
    for page_id in ACTIVE_COMPANION_PAGE_IDS:
        assert page_id in PAGE_CONTRACTS, f"Missing contract for active lab {page_id}"


def test_companion_surface_titles_are_unique() -> None:
    titles = [lab.title for lab in ACTIVE_COMPANION_LABS]
    assert len(titles) == len(set(titles))
