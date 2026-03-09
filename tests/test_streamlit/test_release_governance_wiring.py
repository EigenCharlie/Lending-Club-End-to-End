"""Guardrails for release governance block removal in key Streamlit pages."""

from __future__ import annotations

import ast
from pathlib import Path

TARGETS = [
    Path("streamlit_app/pages/executive_summary.py"),
    Path("streamlit_app/pages/model_governance.py"),
    Path("streamlit_app/pages/thesis_defense.py"),
    Path("streamlit_app/pages/tesis_especializacion.py"),
    Path("streamlit_app/pages/notebook_evidence.py"),
]


def _has_release_governance_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "streamlit_app.components.release_governance":
            continue
        for alias in node.names:
            if alias.name == "render_release_governance":
                return True
    return False


def _has_release_governance_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "render_release_governance":
            return True
    return False


def test_priority_pages_do_not_render_release_governance_block() -> None:
    violations: list[str] = []
    for path in TARGETS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_import = _has_release_governance_import(tree)
        has_call = _has_release_governance_call(tree)
        if has_import or has_call:
            violations.append(f"{path}:import={has_import},call={has_call}")
    assert not violations, (
        "Release governance block must stay hidden from reader pages: " + ", ".join(violations)
    )
