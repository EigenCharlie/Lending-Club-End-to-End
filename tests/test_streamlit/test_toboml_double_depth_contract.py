"""Targeted concept-capsule usage checks for Streamlit pages."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = PROJECT_ROOT / "streamlit_app" / "pages"
CONCEPT_CAPSULES_PATH = PROJECT_ROOT / "streamlit_app" / "components" / "concept_capsules.py"
STORY_SHELL_PATH = PROJECT_ROOT / "streamlit_app" / "components" / "story_shell.py"


def _has_call(path: Path, fn_name: str) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == fn_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == fn_name:
            return True
    return False


def test_concept_capsule_has_executive_and_technical_blocks() -> None:
    source = CONCEPT_CAPSULES_PATH.read_text(encoding="utf-8")
    assert "Cápsula conceptual (lectura ejecutiva + técnica)" in source
    assert "Supuestos y límites operativos" in source


def test_story_shell_header_does_not_inject_concept_stack_globally() -> None:
    source = STORY_SHELL_PATH.read_text(encoding="utf-8")
    assert "render_concept_stack(" not in source
