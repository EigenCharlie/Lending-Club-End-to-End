"""Guardrails for governance artifact wiring in key Streamlit pages."""

from __future__ import annotations

import ast
from pathlib import Path

TARGETS = [
    Path("streamlit_app/pages/executive_summary.py"),
    Path("streamlit_app/pages/thesis_defense.py"),
]


def _resolve_governance_source(path: Path) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "governance" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name):
            continue
        if node.value.func.id not in {"load_json", "try_load_json"}:
            continue
        if not node.value.args:
            continue
        first_arg = node.value.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
    return None


def test_priority_pages_load_governance_status_artifact() -> None:
    violations: list[str] = []
    for path in TARGETS:
        source_name = _resolve_governance_source(path)
        if source_name != "governance_status":
            violations.append(f"{path}:{source_name}")

    assert not violations, (
        "Governance in priority pages must come from models/governance_status.json: "
        + ", ".join(sorted(violations))
    )
