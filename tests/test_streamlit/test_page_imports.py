"""Smoke tests: verify all 20 Streamlit pages are syntactically valid.

These tests parse each page as Python AST, verifying there are no syntax
errors, and check that essential imports are resolvable. They do NOT
execute the pages (which would require a running Streamlit runtime).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PAGES_DIR = Path(__file__).resolve().parents[2] / "streamlit_app" / "pages"

# Discover all page files dynamically
PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=[p.stem for p in PAGE_FILES])
def test_page_syntax_valid(page_path: Path) -> None:
    """Each page file should be valid Python (no syntax errors)."""
    source = page_path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(page_path))
    except SyntaxError as exc:
        pytest.fail(f"Syntax error in {page_path.name}: {exc}")


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=[p.stem for p in PAGE_FILES])
def test_page_has_streamlit_import(page_path: Path) -> None:
    """Each page should import streamlit."""
    source = page_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "streamlit" in imports or "streamlit_app" in imports, (
        f"{page_path.name} does not import streamlit or streamlit_app"
    )


def test_all_20_pages_discovered() -> None:
    """Ensure we have exactly 20 page files."""
    assert len(PAGE_FILES) == 20, (
        f"Expected 20 pages, found {len(PAGE_FILES)}: {[p.name for p in PAGE_FILES]}"
    )


def test_streamlit_app_utils_importable() -> None:
    """Core utils module should import without errors."""
    try:
        mod = importlib.import_module("streamlit_app.utils")
        assert mod is not None
    except Exception as exc:
        pytest.fail(f"Cannot import streamlit_app.utils: {exc}")


def test_streamlit_app_theme_importable() -> None:
    """Theme module should import without errors."""
    try:
        mod = importlib.import_module("streamlit_app.theme")
        assert mod is not None
    except Exception as exc:
        pytest.fail(f"Cannot import streamlit_app.theme: {exc}")
