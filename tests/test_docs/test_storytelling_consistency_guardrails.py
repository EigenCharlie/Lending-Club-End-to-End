"""Guardrails for high-level storytelling consistency in Streamlit pages."""

from __future__ import annotations

from pathlib import Path

TARGETS = [
    Path("streamlit_app/pages/executive_summary.py"),
    Path("streamlit_app/pages/thesis_contribution.py"),
    Path("streamlit_app/pages/tech_stack.py"),
]

FORBIDDEN_PATTERNS = [
    "CatBoost + Platt",
    "2.26 millones de préstamos",
    "2.26M de préstamos",
]


def test_no_stale_storytelling_claims_in_priority_pages() -> None:
    violations: list[str] = []
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                violations.append(f"{path}:{pattern}")

    assert not violations, (
        "Found stale storytelling claims in: "
        + ", ".join(sorted(violations))
        + ". Keep dataset/calibration narrative aligned with canonical artifacts."
    )
