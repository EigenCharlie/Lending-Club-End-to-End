"""Guardrails for selective storytelling intros in Streamlit pages."""

from __future__ import annotations

from pathlib import Path

NARRATIVE_COMPONENT = Path("streamlit_app/components/narrative.py")

REQUIRED_INTRO_PAGES = [
    Path("streamlit_app/pages/data_story.py"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/model_interpretability.py"),
    Path("streamlit_app/pages/causal_intelligence.py"),
    Path("streamlit_app/pages/portfolio_optimizer.py"),
]

RESEARCH_STYLE_PAGES: list[Path] = []


def test_required_pages_include_storytelling_intro() -> None:
    missing: list[str] = []
    for path in REQUIRED_INTRO_PAGES:
        text = path.read_text(encoding="utf-8")
        if "storytelling_intro(" not in text:
            missing.append(str(path))

    assert not missing, (
        "Missing storytelling_intro in required pages: "
        + ", ".join(missing)
        + ". Keep high-level pages beginner-friendly."
    )


def test_research_pages_do_not_force_storytelling_intro() -> None:
    violations: list[str] = []
    for path in RESEARCH_STYLE_PAGES:
        text = path.read_text(encoding="utf-8")
        if "storytelling_intro(" in text:
            violations.append(str(path))

    assert not violations, (
        "Research pages should keep a direct expert tone (no forced storytelling_intro): "
        + ", ".join(violations)
    )


def test_storytelling_intro_keeps_core_questions() -> None:
    text = NARRATIVE_COMPONENT.read_text(encoding="utf-8")
    required_patterns = [
        "Qué resuelve esta técnica",
        "Por qué importa en negocio",
        "Decisión que habilita",
    ]
    missing = [pattern for pattern in required_patterns if pattern not in text]

    assert not missing, (
        "storytelling_intro is missing core narrative prompts: "
        + ", ".join(missing)
        + ". Keep the intro beginner-friendly and decision-oriented."
    )
