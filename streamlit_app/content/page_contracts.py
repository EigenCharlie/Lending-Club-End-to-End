"""Narrative contracts for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PAGE_TYPES = {
    "executive",
    "analytics",
    "decision",
    "governance",
    "research",
    "paper_draft",
    "appendix",
    "journey",
    "glossary",
    "exploration",
}


@dataclass(frozen=True)
class PageStoryContract:
    page_id: str
    page_type: str
    primary_audience: str
    secondary_audience: tuple[str, ...] = ()
    goal: str = ""
    business_value: str = ""
    key_decision: str = ""
    required_sections: tuple[str, ...] = ()
    next_pages: tuple[str, ...] = ()
    glossary_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChartNarrativeSpec:
    chart_id: str
    headline_claim: str
    what_to_look_at: str
    common_misread: str
    decision_implication: str


_MANUAL_CONTRACTS: dict[str, PageStoryContract] = {
    "executive_summary": PageStoryContract(
        page_id="executive_summary",
        page_type="executive",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        goal="Conectar la tesis end-to-end en una sola vista ejecutiva.",
        business_value="Permite evaluar retorno, robustez, provisión y gobernanza en conjunto.",
        key_decision="Definir si la política propuesta es defendible para operación.",
        required_sections=("hook", "kpis", "evidence", "decision", "next_step"),
        next_pages=("thesis_end_to_end", "model_laboratory", "portfolio_optimizer"),
        glossary_terms=("canónico", "calibración", "conformal", "price of robustness"),
    ),
    "glossary_fundamentals": PageStoryContract(
        page_id="glossary_fundamentals",
        page_type="glossary",
        primary_audience="General",
        secondary_audience=("Negocio", "Técnico"),
        goal="Alinear vocabulario del proyecto para lectura consistente del dashboard.",
        business_value="Reduce malentendidos y discusiones de comité por definiciones ambiguas.",
        key_decision="Elegir métricas y técnicas con lenguaje compartido.",
        required_sections=("intro", "terms", "guide"),
        next_pages=("thesis_end_to_end", "executive_summary"),
        glossary_terms=("canónico", "coverage", "ece", "brier", "ks", "gini"),
    ),
    "thesis_end_to_end": PageStoryContract(
        page_id="thesis_end_to_end",
        page_type="journey",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        goal="Explicar la secuencia E2E desde datos hasta decisiones y gobernanza.",
        business_value="Alinea expectativas sobre qué parte del sistema produce cada resultado.",
        key_decision="Priorizar qué módulo revisar/defender según la audiencia.",
        next_pages=("data_architecture", "data_story", "model_laboratory"),
    ),
    "model_laboratory": PageStoryContract(
        page_id="model_laboratory",
        page_type="analytics",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        goal="Justificar la elección del modelo PD y su calibración.",
        business_value="Evita usar probabilidades mal calibradas en pricing, límites e IFRS9.",
        key_decision="Adoptar baseline/challenger y campeón canónico.",
        required_sections=("model_compare", "calibration", "explainability", "caveats"),
        next_pages=("uncertainty_quantification", "model_governance"),
        glossary_terms=("calibración", "ece", "brier", "ks", "gini", "baseline vs canonical"),
    ),
    "uncertainty_quantification": PageStoryContract(
        page_id="uncertainty_quantification",
        page_type="analytics",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        goal="Mostrar cómo la incertidumbre PD se traduce en intervalos con cobertura controlada.",
        business_value="Evita decisiones frágiles basadas en PD puntuales sobreconfiadas.",
        key_decision="Definir cobertura objetivo y tolerancia de ancho para operación.",
        required_sections=("intuition", "kpis", "backtest", "group_checks", "decision"),
        next_pages=("portfolio_optimizer", "ifrs9_provisions", "model_governance"),
        glossary_terms=("conformal", "coverage", "canónico"),
    ),
    "portfolio_optimizer": PageStoryContract(
        page_id="portfolio_optimizer",
        page_type="decision",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        goal="Traducir PD + incertidumbre en asignación de capital bajo restricciones.",
        business_value="Cuantificar el costo de robustez y el impacto comercial de la política.",
        key_decision="Elegir tolerancia de riesgo y aversión a incertidumbre.",
        required_sections=("kpis", "tradeoff", "frontier", "policy"),
        next_pages=("ifrs9_provisions", "model_governance"),
        glossary_terms=("price of robustness", "conformal", "canónico"),
    ),
    "ifrs9_provisions": PageStoryContract(
        page_id="ifrs9_provisions",
        page_type="decision",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        goal="Conectar modelos de riesgo con provisión contable bajo escenarios.",
        business_value="Mide impacto en P&L, reservas y resiliencia ante stress.",
        key_decision="Definir nivel de provisión y sensibilidad aceptable por escenario.",
        required_sections=("ifrs9_intro", "ecl_kpis", "scenario_compare", "sensitivity", "caveats"),
        next_pages=("model_governance",),
        glossary_terms=("canónico", "calibración", "conformal"),
    ),
    "model_governance": PageStoryContract(
        page_id="model_governance",
        page_type="governance",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        goal="Verificar confiabilidad operativa del sistema de riesgo y sus artefactos.",
        business_value="Reduce riesgo de operación con métricas, políticas y contratos inconsistentes.",
        key_decision="Operar, recalibrar o bloquear despliegue.",
        required_sections=("global_status", "artifacts", "checks", "fairness", "mrm"),
        next_pages=("tech_stack",),
        glossary_terms=("canónico", "coverage", "baseline vs canonical"),
    ),
    "thesis_contribution": PageStoryContract(
        page_id="thesis_contribution",
        page_type="research",
        primary_audience="Técnico",
        goal="Definir aportes, claims y posicionamiento científico de la tesis.",
        next_pages=("research_landscape", "paper_1_cp_robust_opt"),
    ),
    "research_landscape": PageStoryContract(
        page_id="research_landscape",
        page_type="research",
        primary_audience="Técnico",
        goal="Mapear literatura y huecos de investigación relevantes para la tesis.",
        next_pages=("paper_1_cp_robust_opt", "paper_2_ifrs9_e2e", "paper_3_mondrian"),
    ),
    "paper_1_cp_robust_opt": PageStoryContract(
        page_id="paper_1_cp_robust_opt",
        page_type="paper_draft",
        primary_audience="Técnico",
        goal="Documentar draft del paper CP + robust optimization.",
    ),
    "paper_2_ifrs9_e2e": PageStoryContract(
        page_id="paper_2_ifrs9_e2e",
        page_type="paper_draft",
        primary_audience="Técnico",
        goal="Documentar draft del paper IFRS9 end-to-end.",
    ),
    "paper_3_mondrian": PageStoryContract(
        page_id="paper_3_mondrian",
        page_type="paper_draft",
        primary_audience="Técnico",
        goal="Documentar draft del paper Mondrian conformal.",
    ),
    "gpu_benchmark": PageStoryContract(
        page_id="gpu_benchmark",
        page_type="appendix",
        primary_audience="Técnico",
        goal="Presentar benchmark GPU como anexo técnico independiente.",
    ),
    "tesis_especializacion": PageStoryContract(
        page_id="tesis_especializacion",
        page_type="research",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        goal="Alinear propuesta de especialización con evidencia ejecutable y estado real de artefactos.",
        business_value="Documento defendible para comité académico y comité técnico sin claims no verificables.",
        key_decision="Definir versión oficial técnica y alcance de especialización vs agenda de maestría.",
        required_sections=(
            "intro_eda",
            "dataset",
            "modelado_base",
            "conformal_prediction",
            "evaluacion_comparativa",
            "impacto_ifrs9",
            "fairness_secundario",
            "crisp_dm",
            "conclusiones",
        ),
        next_pages=("thesis_contribution", "research_landscape"),
        glossary_terms=("conformal_prediction", "ECE", "MPIW", "Kupiec", "Christoffersen"),
    ),
}


_TYPE_BY_PAGE: dict[str, str] = {
    "ab_testing_simulation": "decision",
    "causal_intelligence": "analytics",
    "chat_with_data": "exploration",
    "data_architecture": "journey",
    "data_story": "analytics",
    "feature_engineering": "analytics",
    "notebook_evidence": "journey",
    "research_best_practices": "research",
    "survival_analysis": "analytics",
    "tech_stack": "governance",
    "thesis_defense": "journey",
    "time_series_outlook": "analytics",
}


def _default_contract(page_id: str) -> PageStoryContract:
    page_type = _TYPE_BY_PAGE.get(page_id, "analytics")
    return PageStoryContract(
        page_id=page_id,
        page_type=page_type,
        primary_audience="Técnico" if page_type in {"research", "paper_draft"} else "Negocio",
        secondary_audience=("Técnico",) if page_type not in {"research", "paper_draft"} else (),
        goal=f"Presentar resultados y narrativa de `{page_id}` con estructura consistente.",
        required_sections=("evidence", "interpretation"),
    )


def build_page_contract_registry() -> dict[str, PageStoryContract]:
    """Return complete page contract registry covering all Streamlit pages."""
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    registry: dict[str, PageStoryContract] = {}
    for path in sorted(pages_dir.glob("*.py")):
        page_id = path.stem
        registry[page_id] = _MANUAL_CONTRACTS.get(page_id, _default_contract(page_id))
    return registry


PAGE_CONTRACTS = build_page_contract_registry()


def get_page_contract(page_id: str) -> PageStoryContract:
    """Fetch page contract by page id."""
    return PAGE_CONTRACTS.get(page_id, _default_contract(page_id))
