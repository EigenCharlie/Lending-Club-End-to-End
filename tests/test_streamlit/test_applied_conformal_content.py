"""Narrative guardrails for Applied Conformal Prediction updates."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_PAGE = PROJECT_ROOT / "streamlit_app" / "pages" / "uncertainty_quantification.py"
TESIS_PAGE = PROJECT_ROOT / "streamlit_app" / "pages" / "tesis_especializacion.py"
GOV_PAGE = PROJECT_ROOT / "streamlit_app" / "pages" / "model_governance.py"
CP_BLOCKS = PROJECT_ROOT / "streamlit_app" / "components" / "conformal_applied_blocks.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_uncertainty_page_has_applied_cp_sections() -> None:
    text = _read(UNCERTAINTY_PAGE)
    required = [
        "0) Qué está garantizado y qué no",
        "0.1) Estabilidad por tamaño de muestra",
        "0.2) Validez vs eficiencia",
        "0.3) Cuándo usar qué método",
        "0.4) Exchangeability stress checklist",
    ]
    missing = [marker for marker in required if marker not in text]
    assert not missing, f"Missing applied conformal sections in uncertainty page: {missing}"


def test_tesis_page_has_limits_and_evidence_ladder() -> None:
    text = _read(TESIS_PAGE)
    required = [
        "Tres garantias y tres limites de Conformal Prediction",
        "Evidence ladder (principio -> artefacto -> decision)",
        "Escalamiento metodologico (sin cambiar pipeline canonico)",
    ]
    missing = [marker for marker in required if marker not in text]
    assert not missing, f"Missing applied conformal sections in tesis page: {missing}"


def test_tesis_page_avoids_absolute_conditional_guarantee_claims() -> None:
    lower = _read(TESIS_PAGE).lower()
    forbidden = [
        "garantia condicional por grupo",
        "garantía condicional por grupo",
        "garantizando cobertura condicional",
    ]
    found = [term for term in forbidden if term in lower]
    assert not found, f"Overclaim wording should be removed from tesis page: {found}"


def test_model_governance_has_cp_micro_panel() -> None:
    text = _read(GOV_PAGE)
    assert "Micro-panel CP: ruptura de supuestos y respuesta" in text


def test_conformal_blocks_include_marginal_vs_conditional_limit() -> None:
    text = _read(CP_BLOCKS)
    assert "Cobertura marginal en muestra finita" in text
    assert "Cobertura condicional exacta para todo x" in text
