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
    narrative_axis: str = "insight_factory"
    pipeline_role: str = "supporting"
    artifact_scope: str = "insight"
    book_chapter: str = ""
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
        narrative_axis="operational_pipeline",
        pipeline_role="overview",
        artifact_scope="canonical",
        book_chapter="00-executive-map",
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
        narrative_axis="book_foundations",
        pipeline_role="cross_cutting",
        artifact_scope="shared",
        book_chapter="01-glossary-and-foundations",
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
        narrative_axis="operational_pipeline",
        pipeline_role="spine",
        artifact_scope="canonical",
        book_chapter="02-operational-pipeline",
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
        narrative_axis="operational_pipeline",
        pipeline_role="pd_modeling",
        artifact_scope="canonical",
        book_chapter="03-pd-modeling-and-calibration",
        goal="Justificar la elección del modelo PD y su calibración.",
        business_value="Evita usar probabilidades mal calibradas en pricing, límites e IFRS9.",
        key_decision="Adoptar baseline/challenger y campeón canónico.",
        required_sections=("model_compare", "calibration", "explainability", "caveats"),
        next_pages=("model_interpretability", "uncertainty_quantification", "model_governance"),
        glossary_terms=("calibración", "ece", "brier", "ks", "gini", "baseline vs canonical"),
    ),
    "model_interpretability": PageStoryContract(
        page_id="model_interpretability",
        page_type="analytics",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        narrative_axis="insight_factory",
        pipeline_role="complementary_analysis",
        artifact_scope="insight",
        book_chapter="08-explainability-and-insights",
        goal="Convertir la interpretabilidad del score PD en una capacidad operativa y defendible.",
        business_value="Permite explicar drivers, casos individuales y estabilidad del modelo con artefactos canónicos.",
        key_decision="Adoptar un lenguaje común de explicabilidad para operación, auditoría y defensa académica.",
        required_sections=("map", "global", "local", "effects", "stability"),
        next_pages=("uncertainty_quantification", "model_governance"),
        glossary_terms=("canónico", "prediction_interval", "concept_drift"),
    ),
    "uncertainty_quantification": PageStoryContract(
        page_id="uncertainty_quantification",
        page_type="analytics",
        primary_audience="Negocio",
        secondary_audience=("Técnico",),
        narrative_axis="operational_pipeline",
        pipeline_role="conformal_uncertainty",
        artifact_scope="canonical",
        book_chapter="04-conformal-and-uncertainty",
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
        narrative_axis="operational_pipeline",
        pipeline_role="portfolio_decision",
        artifact_scope="canonical",
        book_chapter="06-portfolio-policy-and-selection",
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
        narrative_axis="operational_pipeline",
        pipeline_role="ifrs9_translation",
        artifact_scope="canonical",
        book_chapter="07-ifrs9-and-governance",
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
        narrative_axis="operational_pipeline",
        pipeline_role="governance",
        artifact_scope="canonical",
        book_chapter="07-ifrs9-and-governance",
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
        narrative_axis="book_foundations",
        pipeline_role="research_positioning",
        artifact_scope="research",
        book_chapter="10-research-agenda-and-contributions",
        goal="Definir aportes, claims y posicionamiento científico de la tesis.",
        next_pages=("research_landscape", "paper_1_cp_robust_opt"),
    ),
    "roadmap_backlog": PageStoryContract(
        page_id="roadmap_backlog",
        page_type="research",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        narrative_axis="book_foundations",
        pipeline_role="program_governance",
        artifact_scope="shared",
        book_chapter="16-quarto-and-publication-contracts",
        goal="Concentrar el estado promovido del champion actual y el backlog operativo/research en una sola vista auditable.",
        business_value="Evita perder contexto entre sesiones y alinea decisiones operativas con la agenda de tesis, libro y papers.",
        key_decision="Priorizar correctamente el siguiente bloque de trabajo sin reabrir decisiones ya promovidas.",
        required_sections=("status", "promotions", "backlog", "handoff"),
        next_pages=("executive_summary", "model_governance", "research_landscape"),
        glossary_terms=(
            "baseline oficial",
            "champion_search",
            "canonical_rebuild",
            "insights_factory",
        ),
    ),
    "papers_backlog": PageStoryContract(
        page_id="papers_backlog",
        page_type="research",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        narrative_axis="book_foundations",
        pipeline_role="publication_roadmap",
        artifact_scope="shared",
        book_chapter="16-quarto-and-publication-contracts",
        goal="Traducir el backlog técnico a un roadmap editorial de papers y Quarto, dejando claro qué bloquea cada publicación.",
        business_value="Conecta el avance del pipeline con publicaciones, libro y entregables académicos sin perder trazabilidad operativa.",
        key_decision="Decidir qué trabajo siguiente maximiza valor para papers/Q1 y qué puede quedar como soporte editorial paralelo.",
        required_sections=("scope", "differences", "papers", "quarto", "handoff"),
        next_pages=("roadmap_backlog", "research_landscape", "thesis_contribution"),
        glossary_terms=("paper estrella", "paper-grade", "Quarto", "research backlog"),
    ),
    "research_landscape": PageStoryContract(
        page_id="research_landscape",
        page_type="research",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="research_landscape",
        artifact_scope="research",
        book_chapter="10-research-agenda-and-contributions",
        goal="Mapear literatura y huecos de investigación relevantes para la tesis.",
        next_pages=("paper_1_cp_robust_opt", "paper_2_ifrs9_e2e", "paper_3_mondrian"),
    ),
    "paper_estrella_predict_optimize": PageStoryContract(
        page_id="paper_estrella_predict_optimize",
        page_type="paper_draft",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="paper_draft",
        artifact_scope="research",
        book_chapter="11-paper-predict-then-optimize",
        goal="Paper Estrella: predict-then-optimize con bound alpha-Gamma, SPO+ regret y baselines de uncertainty sets. Absorbe Paper 1.",
    ),
    "paper_1_cp_robust_opt": PageStoryContract(
        page_id="paper_1_cp_robust_opt",
        page_type="paper_draft",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="paper_draft",
        artifact_scope="research",
        book_chapter="11-paper-cp-robust-opt",
        goal="Draft historico del paper CP + robust optimization (absorbido por Paper Estrella).",
    ),
    "paper_2_ifrs9_e2e": PageStoryContract(
        page_id="paper_2_ifrs9_e2e",
        page_type="paper_draft",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="paper_draft",
        artifact_scope="research",
        book_chapter="12-paper-ifrs9-e2e",
        goal="Documentar draft del paper IFRS9 end-to-end.",
    ),
    "paper_3_mondrian": PageStoryContract(
        page_id="paper_3_mondrian",
        page_type="paper_draft",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="paper_draft",
        artifact_scope="research",
        book_chapter="13-paper-mondrian",
        goal="Documentar draft del paper Mondrian conformal.",
    ),
    "gpu_benchmark": PageStoryContract(
        page_id="gpu_benchmark",
        page_type="appendix",
        primary_audience="Técnico",
        narrative_axis="insight_factory",
        pipeline_role="gpu_benchmark",
        artifact_scope="insight",
        book_chapter="14-appendix-gpu-benchmarks",
        goal="Presentar benchmark GPU como anexo técnico independiente.",
    ),
    "tesis_especializacion": PageStoryContract(
        page_id="tesis_especializacion",
        page_type="research",
        primary_audience="Técnico",
        secondary_audience=("Negocio",),
        narrative_axis="book_foundations",
        pipeline_role="academic_context",
        artifact_scope="research",
        book_chapter="09-specialization-to-masters-bridge",
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

_NARRATIVE_AXIS_BY_PAGE: dict[str, str] = {
    "data_architecture": "operational_pipeline",
    "executive_summary": "operational_pipeline",
    "feature_engineering": "operational_pipeline",
    "ifrs9_provisions": "operational_pipeline",
    "model_governance": "operational_pipeline",
    "model_laboratory": "operational_pipeline",
    "portfolio_optimizer": "operational_pipeline",
    "survival_analysis": "operational_pipeline",
    "thesis_defense": "operational_pipeline",
    "thesis_end_to_end": "operational_pipeline",
    "time_series_outlook": "operational_pipeline",
    "uncertainty_quantification": "operational_pipeline",
    "ab_testing_simulation": "insight_factory",
    "causal_intelligence": "insight_factory",
    "chat_with_data": "insight_factory",
    "data_story": "insight_factory",
    "gpu_benchmark": "insight_factory",
    "notebook_evidence": "insight_factory",
    "research_best_practices": "insight_factory",
    "research_landscape": "insight_factory",
    "tech_stack": "book_foundations",
}

_PIPELINE_ROLE_BY_PAGE: dict[str, str] = {
    "data_architecture": "data_and_lineage",
    "feature_engineering": "feature_engineering",
    "survival_analysis": "survival_modeling",
    "time_series_outlook": "time_series_forecasting",
    "causal_intelligence": "causal_policy_research",
    "ab_testing_simulation": "decision_validation",
    "data_story": "dataset_insight",
    "notebook_evidence": "notebook_atlas",
    "research_best_practices": "publication_tooling",
    "tech_stack": "book_architecture",
}

_ARTIFACT_SCOPE_BY_PAGE: dict[str, str] = {
    "chat_with_data": "shared",
    "glossary_fundamentals": "shared",
    "tech_stack": "shared",
}

_BOOK_CHAPTER_BY_PAGE: dict[str, str] = {
    "ab_testing_simulation": "06-portfolio-policy-and-selection",
    "causal_intelligence": "08-explainability-and-insights",
    "chat_with_data": "15-streamlit-exploration",
    "data_architecture": "02-operational-pipeline",
    "data_story": "08-explainability-and-insights",
    "feature_engineering": "03-pd-modeling-and-calibration",
    "notebook_evidence": "14-appendix-notebook-atlas",
    "research_best_practices": "16-quarto-and-publication-contracts",
    "survival_analysis": "05-survival-time-series-and-causal",
    "tech_stack": "16-quarto-and-publication-contracts",
    "thesis_defense": "02-operational-pipeline",
    "time_series_outlook": "05-survival-time-series-and-causal",
}


def _default_contract(page_id: str) -> PageStoryContract:
    page_type = _TYPE_BY_PAGE.get(page_id, "analytics")
    narrative_axis = _NARRATIVE_AXIS_BY_PAGE.get(page_id, "insight_factory")
    artifact_scope = _ARTIFACT_SCOPE_BY_PAGE.get(
        page_id,
        "canonical" if narrative_axis == "operational_pipeline" else "insight",
    )
    return PageStoryContract(
        page_id=page_id,
        page_type=page_type,
        primary_audience="Técnico" if page_type in {"research", "paper_draft"} else "Negocio",
        secondary_audience=("Técnico",) if page_type not in {"research", "paper_draft"} else (),
        narrative_axis=narrative_axis,
        pipeline_role=_PIPELINE_ROLE_BY_PAGE.get(page_id, "supporting"),
        artifact_scope=artifact_scope,
        book_chapter=_BOOK_CHAPTER_BY_PAGE.get(page_id, f"zz-{page_id}"),
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
