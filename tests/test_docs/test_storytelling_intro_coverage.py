"""Guardrails for selective storytelling intros in Streamlit pages."""

from __future__ import annotations

from pathlib import Path

NARRATIVE_COMPONENT = Path("streamlit_app/components/narrative.py")

REQUIRED_INTRO_PAGES = [
    Path("streamlit_app/pages/executive_summary.py"),
    Path("streamlit_app/pages/data_story.py"),
    Path("streamlit_app/pages/feature_engineering.py"),
    Path("streamlit_app/pages/model_laboratory.py"),
    Path("streamlit_app/pages/uncertainty_quantification.py"),
    Path("streamlit_app/pages/causal_intelligence.py"),
    Path("streamlit_app/pages/survival_analysis.py"),
    Path("streamlit_app/pages/time_series_outlook.py"),
    Path("streamlit_app/pages/portfolio_optimizer.py"),
    Path("streamlit_app/pages/ifrs9_provisions.py"),
    Path("streamlit_app/pages/glossary_fundamentals.py"),
    Path("streamlit_app/pages/model_governance.py"),
]

RESEARCH_STYLE_PAGES = [
    Path("streamlit_app/pages/paper_1_cp_robust_opt.py"),
    Path("streamlit_app/pages/paper_2_ifrs9_e2e.py"),
    Path("streamlit_app/pages/paper_3_mondrian.py"),
    Path("streamlit_app/pages/research_best_practices.py"),
    Path("streamlit_app/pages/research_landscape.py"),
    Path("streamlit_app/pages/thesis_contribution.py"),
    Path("streamlit_app/pages/thesis_end_to_end.py"),
    Path("streamlit_app/pages/thesis_defense.py"),
]


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
