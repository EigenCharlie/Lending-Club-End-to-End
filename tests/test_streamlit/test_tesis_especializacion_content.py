"""Content and smoke checks for tesis_especializacion page."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE_PATH = (
    Path(__file__).resolve().parents[2] / "streamlit_app" / "pages" / "tesis_especializacion.py"
)


def test_tesis_especializacion_has_all_chapter_markers() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    required_markers = [
        "Capitulo 1. Apertura del problema y objetivos de la tesis",
        "Capitulo 2. Contexto financiero y planteamiento del problema",
        "Capitulo 3. Marco teorico y estado del arte",
        "Capitulo 4. Datos y preparacion (Lending Club)",
        "Capitulo 5. Metodologia y diseño experimental",
        "Capitulo 6. Resultados PD",
        "Capitulo 7. Resultados LGD/EAD",
        "Capitulo 8. Conformal prediction y evaluacion comparativa",
        "Capitulo 9. Impacto regulatorio (IFRS9)",
        "Capitulo 10. Conclusiones y proximos pasos",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    assert not missing, f"Missing chapter markers: {missing}"


def test_tesis_especializacion_covers_required_section_ids() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    required_section_ids = [
        "intro_eda",
        "modelado_base",
        "conformal_prediction",
        "evaluacion_comparativa",
        "impacto_ifrs9",
        "fairness_secundario",
        "crisp_dm",
        "conclusiones",
    ]
    missing = [item for item in required_section_ids if item not in text]
    assert not missing, f"Missing required section ids: {missing}"


def test_tesis_especializacion_has_explicit_fallback_messages() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    required_fallbacks = [
        "No hay `ifrs9_scenario_summary.parquet` en esta corrida.",
        "No hay `fairness_audit.parquet` para esta corrida.",
        "No hay `conformal_intervals_ead.parquet` en esta corrida.",
    ]
    missing = [msg for msg in required_fallbacks if msg not in text]
    assert not missing, f"Missing fallback messages: {missing}"


def test_tesis_especializacion_hides_section_id_labels_in_ui() -> None:
    text = PAGE_PATH.read_text(encoding="utf-8")
    assert "section_id:" not in text


def test_tesis_especializacion_page_renders_without_exceptions() -> None:
    at = AppTest.from_file(PAGE_PATH, default_timeout=35)
    at.run(timeout=35)
    assert len(at.exception) == 0
    assert any("Tesis de Especializacion" in str(item.value) for item in at.title)
