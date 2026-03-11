"""Canonical concept map that aligns Streamlit pages with TOBoML concepts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptCard:
    """Normalized concept card used by the reusable capsule renderer."""

    key: str
    label: str
    level: str
    what_is: str
    why_business: str
    common_misread: str
    decision_enabled: str
    pass_when: str
    warn_when: str
    fail_when: str
    anti_patterns: tuple[str, ...] = ()


CONCEPT_REGISTRY: dict[str, ConceptCard] = {
    "aleatoric": ConceptCard(
        key="aleatoric",
        label="Incertidumbre Aleatoric",
        level="core",
        what_is="Variabilidad irreducible del sistema generador de datos (ruido inherente).",
        why_business="Define el piso mínimo de error esperable y evita exigir precisión irreal.",
        common_misread="Asumir que más tuning siempre eliminará este ruido.",
        decision_enabled="Fijar expectativas realistas de performance y buffers prudenciales.",
        pass_when="Se reporta explícitamente como límite estructural del modelo.",
        warn_when="Se menciona ruido pero sin impacto en decisiones de riesgo.",
        fail_when="Se promete precisión puntual sin reconocer ruido irreducible.",
    ),
    "epistemic": ConceptCard(
        key="epistemic",
        label="Incertidumbre Epistemic",
        level="core",
        what_is="Incertidumbre reducible por falta de datos, cobertura o conocimiento.",
        why_business="Permite priorizar inversión en datos, segmentación y monitoreo.",
        common_misread="Confundirla con aleatoric y no diseñar planes de mitigación.",
        decision_enabled="Definir dónde recolectar más señal o recalibrar por segmento.",
        pass_when="Se conecta explícitamente con acciones de mejora de datos/modelo.",
        warn_when="Se menciona incertidumbre pero sin distinguir fuente reducible.",
        fail_when="Se interpreta todo error como inherente y no accionable.",
    ),
    "confidence_interval": ConceptCard(
        key="confidence_interval",
        label="Confidence Interval",
        level="core",
        what_is="Intervalo sobre parámetros estimados del modelo, no sobre observaciones futuras.",
        why_business="Evita usar intervalos de parámetros como si fueran bandas operativas de riesgo.",
        common_misread="Interpretarlo como rango de predicción por préstamo.",
        decision_enabled="Separar inferencia estadística de decisión operativa en cartera.",
        pass_when="La página distingue parámetro vs predicción en el texto.",
        warn_when="Se usa el término intervalo sin aclarar su objeto.",
        fail_when="Se decide política usando CI como si fuera PI.",
    ),
    "prediction_interval": ConceptCard(
        key="prediction_interval",
        label="Prediction Interval",
        level="core",
        what_is="Rango probable para valores futuros observables (ej. PD por préstamo).",
        why_business="Es el insumo correcto para robustez, pricing prudencial e IFRS9 forward-looking.",
        common_misread="Creer que ancho grande implica automáticamente mal modelo.",
        decision_enabled="Definir política de riesgo por cobertura objetivo y ancho tolerable.",
        pass_when="Se reportan cobertura y ancho junto con implicación operativa.",
        warn_when="Solo se reporta el punto central sin banda de incertidumbre.",
        fail_when="Se omite incertidumbre en decisiones de alto impacto.",
    ),
    "mcar_mar_mnar": ConceptCard(
        key="mcar_mar_mnar",
        label="MCAR/MAR/MNAR",
        level="advanced",
        what_is="Taxonomía de mecanismos de faltantes: aleatorio total, condicional o no aleatorio.",
        why_business="Determina estrategia correcta de imputación y sesgo esperado.",
        common_misread="Aplicar imputación única sin evaluar mecanismo de faltantes.",
        decision_enabled="Elegir tratamiento de missing con criterio y trazabilidad.",
        pass_when="Se documenta mecanismo probable y estrategia usada.",
        warn_when="Se imputa masivamente sin diagnóstico explícito.",
        fail_when="Se ignoran faltantes informativos en variables críticas.",
    ),
    "data_leakage": ConceptCard(
        key="data_leakage",
        label="Data Leakage",
        level="core",
        what_is="Uso de información no disponible al momento real de decisión.",
        why_business="Infla métricas offline y destruye confiabilidad en producción.",
        common_misread="Tomar AUC alto como evidencia suficiente sin revisar fuga.",
        decision_enabled="Bloquear variables post-evento y endurecer contratos de features.",
        pass_when="Existe control explícito de variables de fuga y validación OOT.",
        warn_when="Hay mención de leakage pero sin guardrails automatizados.",
        fail_when="Se incluyen features post-default o de futuro en entrenamiento.",
    ),
    "nested_cv": ConceptCard(
        key="nested_cv",
        label="Nested Cross-Validation",
        level="advanced",
        what_is="Esquema que separa tuning y evaluación para reducir optimismo de selección.",
        why_business="Mejora estimación realista del desempeño esperado en despliegue.",
        common_misread="Usar el mismo CV para elegir hiperparámetros y reportar score final.",
        decision_enabled="Diseñar protocolo de evaluación más defendible ante auditoría.",
        pass_when="La evaluación final está separada de la búsqueda de hiperparámetros.",
        warn_when="Se reutiliza validación interna de tuning como métrica final.",
        fail_when="No existe separación entre selección y evaluación de modelo.",
    ),
    "covariate_shift": ConceptCard(
        key="covariate_shift",
        label="Covariate Shift",
        level="core",
        what_is="Cambio en distribución de covariables entre entrenamiento y operación.",
        why_business="Afecta calibración y robustez de decisiones aunque el modelo no cambie.",
        common_misread="Atribuir todo deterioro a sobreajuste sin revisar shift de entrada.",
        decision_enabled="Activar recalibración o ajuste de política por cambios de mix.",
        pass_when="Se monitorean distribuciones por cohorte/segmento.",
        warn_when="Solo se monitorea performance agregada sin diagnóstico de entrada.",
        fail_when="Cambios de mix pasan sin alertas ni plan de respuesta.",
    ),
    "concept_drift": ConceptCard(
        key="concept_drift",
        label="Concept Drift",
        level="core",
        what_is="Cambio en la relación entre variables de entrada y objetivo.",
        why_business="Puede invalidar reglas de decisión aunque inputs parezcan estables.",
        common_misread="Tratar drift de concepto como simple ruido temporal.",
        decision_enabled="Escalar monitor -> recalibrar -> bloquear según severidad.",
        pass_when="Existe criterio de escalamiento documentado por severidad.",
        warn_when="Se detecta drift pero no hay acción operativa definida.",
        fail_when="Se mantiene operación igual pese a degradación sostenida.",
    ),
    "c2st": ConceptCard(
        key="c2st",
        label="C2ST / Two-Sample Tests",
        level="advanced",
        what_is="Pruebas de diferencia distribucional (KS, Cramér-von Mises, C2ST).",
        why_business="Formaliza alertas de drift y reduce decisiones por intuición visual.",
        common_misread="Usar solo gráficos sin contraste estadístico reproducible.",
        decision_enabled="Definir umbrales objetivos de monitoreo de estabilidad.",
        pass_when="Hay prueba formal y umbral explícito por cohorte/segmento.",
        warn_when="Solo hay evidencia visual sin test estadístico.",
        fail_when="No existe monitoreo distribucional en producción.",
    ),
    "class_imbalance": ConceptCard(
        key="class_imbalance",
        label="Class Imbalance + i.i.d. Caveat",
        level="advanced",
        what_is="Desbalance de clases y riesgos de romper supuestos i.i.d. con re-muestreo.",
        why_business="Evita políticas sesgadas por minorías mal representadas.",
        common_misread="Aplicar SMOTE u oversampling sin validar impacto temporal/causal.",
        decision_enabled="Elegir métricas, umbrales y estrategia de entrenamiento más robusta.",
        pass_when="Se discuten métricas apropiadas y limitaciones del re-muestreo.",
        warn_when="Se reporta accuracy sin contexto de desbalance.",
        fail_when="Se optimiza solo accuracy en dataset desbalanceado.",
        anti_patterns=(
            "No usar SMOTE de forma ciega en problemas temporales de crédito.",
            "No evaluar solo accuracy cuando la clase minoritaria es la crítica.",
        ),
    ),
    "iid_caveat": ConceptCard(
        key="iid_caveat",
        label="Caveat i.i.d.",
        level="advanced",
        what_is="Advertencia sobre el supuesto de observaciones independientes e idénticamente distribuidas.",
        why_business="Evita sobre-confiar en validaciones que no representan operación real.",
        common_misread="Asumir i.i.d. en series temporales o cohortes de originación.",
        decision_enabled="Elegir validación temporal y monitoreo por régimen.",
        pass_when="Se declara explícitamente cuándo i.i.d. no aplica.",
        warn_when="Se aplica CV aleatorio en contextos secuenciales.",
        fail_when="Se ignora estructura temporal al evaluar y decidir.",
    ),
    "extrapolation": ConceptCard(
        key="extrapolation",
        label="Extrapolation Risk",
        level="advanced",
        what_is="Predicción fuera del soporte observado en entrenamiento.",
        why_business="Reduce decisiones frágiles cuando cambian extremos de cartera o mercado.",
        common_misread="Confiar en predicciones fuera de rango como si fueran interpolación.",
        decision_enabled="Limitar políticas fuera de soporte o activar restricciones prudenciales.",
        pass_when="Se señala explícitamente el riesgo fuera de dominio observado.",
        warn_when="Se interpreta forecast lejano sin advertencia de soporte.",
        fail_when="Se extrapola política sin controles ni límites.",
    ),
    "convex_hull": ConceptCard(
        key="convex_hull",
        label="Convex Hull",
        level="advanced",
        what_is="Región geométrica que aproxima el soporte observado para validar extrapolación.",
        why_business="Ayuda a detectar decisiones en zonas de baja evidencia histórica.",
        common_misread="Ignorar que muchos métodos tabulares extrapolan mal fuera del hull.",
        decision_enabled="Etiquetar decisiones fuera de soporte y aplicar prudencia adicional.",
        pass_when="Se usa como referencia conceptual para soporte observado.",
        warn_when="Se decide en extremos sin evaluación de dominio.",
        fail_when="No hay ningún control de fuera-de-soporte en políticas.",
    ),
    "optimizer_curse": ConceptCard(
        key="optimizer_curse",
        label="Optimizer's Curse",
        level="advanced",
        what_is="Sesgo optimista por seleccionar el mejor entre muchas configuraciones ruidosas.",
        why_business="Previene sobreprometer rendimiento que no se sostiene OOT.",
        common_misread="Interpretar mejor trial como mejora real sin ajuste por selección.",
        decision_enabled="Aplicar validación robusta, seeds múltiples y reporte conservador.",
        pass_when="Se discute sesgo optimista y variabilidad por random_state.",
        warn_when="Se reporta solo el mejor resultado sin dispersión.",
        fail_when="Se elige política por máximo puntual no replicable.",
        anti_patterns=(
            "Now change random_state: validar estabilidad antes de concluir.",
            "No confundir mejor trial con ganancia estructural.",
        ),
    ),
    "no_free_lunch": ConceptCard(
        key="no_free_lunch",
        label="No Free Lunch",
        level="core",
        what_is="No existe modelo o configuración universalmente mejor para todo dataset/objetivo.",
        why_business="Obliga a justificar trade-offs por contexto operativo real.",
        common_misread="Buscar técnica 'ganadora absoluta' sin considerar objetivo de decisión.",
        decision_enabled="Elegir método por costo de error, regulación y estabilidad, no por moda.",
        pass_when="La página explicita trade-offs y contexto de selección.",
        warn_when="Se promueve una técnica como solución universal.",
        fail_when="No se explicita por qué una técnica es adecuada al contexto.",
    ),
    "proper_scoring_rules": ConceptCard(
        key="proper_scoring_rules",
        label="Proper Scoring Rules",
        level="core",
        what_is="Métricas probabilísticas (log loss, Brier) que incentivan probabilidades honestas.",
        why_business="Mejoran decisiones que dependen de probabilidad, no solo ranking.",
        common_misread="Usar AUC como sustituto de calidad probabilística.",
        decision_enabled="Elegir calibrador/modelo para pricing, límites e IFRS9.",
        pass_when="Se reportan métricas de ranking y probabilidad en conjunto.",
        warn_when="Solo se muestran métricas de discriminación.",
        fail_when="Se decide con scores no calibrados ni scoring proper.",
        anti_patterns=("No confundir AUC alto con probabilidad bien calibrada.",),
    ),
    "decision_threshold": ConceptCard(
        key="decision_threshold",
        label="Decision Threshold",
        level="core",
        what_is="Umbral operativo que convierte probabilidad continua en acción binaria.",
        why_business="Define trade-off explícito entre crecimiento, riesgo y costo de error.",
        common_misread="Fijar 0.5 por defecto sin análisis de costo y desbalance.",
        decision_enabled="Alinear originación/provisión con apetito de riesgo.",
        pass_when="El umbral se justifica por costo, política o restricción regulatoria.",
        warn_when="Existe umbral pero no su racional económico.",
        fail_when="No hay umbral explícito ni sensibilidad de decisión.",
    ),
}

REQUIRED_CONCEPT_KEYS: tuple[str, ...] = tuple(CONCEPT_REGISTRY.keys())


PAGE_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "ab_testing_simulation": (
        "confidence_interval",
        "prediction_interval",
        "decision_threshold",
    ),
    "causal_intelligence": (
        "no_free_lunch",
        "iid_caveat",
        "decision_threshold",
    ),
    "chat_with_data": (
        "data_leakage",
        "covariate_shift",
        "c2st",
    ),
    "data_architecture": (
        "mcar_mar_mnar",
        "data_leakage",
        "concept_drift",
    ),
    "data_story": (
        "no_free_lunch",
        "covariate_shift",
        "class_imbalance",
    ),
    "executive_summary": (
        "prediction_interval",
        "aleatoric",
        "epistemic",
        "proper_scoring_rules",
    ),
    "feature_engineering": (
        "mcar_mar_mnar",
        "data_leakage",
        "class_imbalance",
    ),
    "glossary_fundamentals": REQUIRED_CONCEPT_KEYS,
    "gpu_benchmark": (
        "no_free_lunch",
        "extrapolation",
    ),
    "ifrs9_provisions": (
        "prediction_interval",
        "decision_threshold",
        "concept_drift",
    ),
    "model_governance": (
        "covariate_shift",
        "concept_drift",
        "c2st",
    ),
    "model_interpretability": (
        "prediction_interval",
        "decision_threshold",
        "concept_drift",
    ),
    "model_laboratory": (
        "proper_scoring_rules",
        "decision_threshold",
        "class_imbalance",
        "optimizer_curse",
        "nested_cv",
    ),
    "notebook_evidence": (
        "proper_scoring_rules",
        "concept_drift",
        "optimizer_curse",
    ),
    "paper_1_cp_robust_opt": (
        "extrapolation",
        "convex_hull",
        "optimizer_curse",
        "nested_cv",
    ),
    "paper_2_ifrs9_e2e": (
        "confidence_interval",
        "prediction_interval",
        "decision_threshold",
    ),
    "paper_3_mondrian": (
        "iid_caveat",
        "concept_drift",
        "c2st",
    ),
    "portfolio_optimizer": (
        "extrapolation",
        "convex_hull",
        "optimizer_curse",
    ),
    "research_best_practices": (
        "nested_cv",
        "data_leakage",
        "optimizer_curse",
    ),
    "research_landscape": (
        "nested_cv",
        "class_imbalance",
        "extrapolation",
        "optimizer_curse",
        "no_free_lunch",
    ),
    "survival_analysis": (
        "prediction_interval",
        "iid_caveat",
        "concept_drift",
    ),
    "tech_stack": (
        "optimizer_curse",
        "no_free_lunch",
        "nested_cv",
        "data_leakage",
    ),
    "tesis_especializacion": (
        "prediction_interval",
        "proper_scoring_rules",
        "data_leakage",
        "concept_drift",
        "optimizer_curse",
    ),
    "thesis_contribution": (
        "prediction_interval",
        "optimizer_curse",
        "no_free_lunch",
    ),
    "thesis_defense": (
        "no_free_lunch",
        "optimizer_curse",
        "decision_threshold",
    ),
    "thesis_end_to_end": (
        "data_leakage",
        "decision_threshold",
        "covariate_shift",
    ),
    "time_series_outlook": (
        "prediction_interval",
        "concept_drift",
        "iid_caveat",
    ),
    "uncertainty_quantification": (
        "confidence_interval",
        "prediction_interval",
        "aleatoric",
        "epistemic",
    ),
}


PAGE_ANTI_PATTERNS: dict[str, tuple[str, ...]] = {
    "model_interpretability": (
        "No confundir SHAP o permutation importance con causalidad.",
        "No leer ALE/PDP como política de intervención individual.",
    ),
    "model_laboratory": (
        "No confundir AUC alto con probabilidad bien calibrada.",
        "Now change random_state antes de consolidar claim de mejora.",
    ),
    "feature_engineering": (
        "No imputar todo igual sin distinguir MCAR/MAR/MNAR.",
        "No codificar target leakage en variables post-evento.",
    ),
    "portfolio_optimizer": (
        "No extrapolar política fuera del soporte observado sin caveat.",
        "No presentar robustez sin cuantificar su costo económico.",
    ),
    "chat_with_data": (
        "Consulta SQL no implica causalidad ni validez de política.",
        "No usar resultados exploratorios como evidencia final sin validación.",
    ),
    "research_landscape": (
        "No vender una técnica como universal (no-free-lunch).",
        "No reportar solo mejor resultado sin variabilidad y límites.",
    ),
}


PAGE_FOCUS_NOTES: dict[str, str] = {
    "executive_summary": "Lectura rápida: traducir conceptos técnicos a impacto de decisión.",
    "model_interpretability": "Lectura interpretativa: separar atribución, efecto, caso local y estabilidad.",
    "model_laboratory": "Lectura técnica: separar ranking, calibración y umbral de decisión.",
    "uncertainty_quantification": "Lectura de riesgo: cobertura vs ancho, no una sola métrica.",
    "portfolio_optimizer": "Lectura de política: retorno esperado vs resiliencia en peor caso.",
    "model_governance": "Lectura operativa: cuándo monitorear, recalibrar o bloquear despliegue.",
}


def get_concept(key: str) -> ConceptCard | None:
    """Return a concept card by key."""
    return CONCEPT_REGISTRY.get(key)


def get_page_concepts(page_id: str) -> tuple[ConceptCard, ...]:
    """Return concept cards assigned to a page."""
    keys = PAGE_CONCEPT_MAP.get(page_id, ())
    cards = [CONCEPT_REGISTRY[key] for key in keys if key in CONCEPT_REGISTRY]
    return tuple(cards)


def get_page_anti_patterns(page_id: str) -> tuple[str, ...]:
    """Return anti-pattern tips for a page."""
    return PAGE_ANTI_PATTERNS.get(page_id, ())


def get_page_focus_note(page_id: str) -> str:
    """Return optional focus note for a page."""
    return PAGE_FOCUS_NOTES.get(page_id, "")


def build_concept_index_rows() -> list[dict[str, str]]:
    """Build a tabular index row set for glossary pages and tests."""
    rows: list[dict[str, str]] = []
    for key, card in CONCEPT_REGISTRY.items():
        pages = sorted(page for page, concepts in PAGE_CONCEPT_MAP.items() if key in concepts)
        rows.append(
            {
                "key": key,
                "concepto": card.label,
                "nivel": card.level,
                "paginas_objetivo": ", ".join(pages),
                "n_paginas": str(len(pages)),
            }
        )
    return rows
