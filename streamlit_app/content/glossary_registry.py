"""Canonical glossary snippets for contextual help across pages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryTerm:
    key: str
    label: str
    short_definition: str
    why_it_matters: str


GLOSSARY_REGISTRY: dict[str, GlossaryTerm] = {
    "canonico": GlossaryTerm(
        key="canonico",
        label="Canónico",
        short_definition=(
            "Fuente oficial (single source of truth) que se usa para reporting, monitoreo y "
            "decisiones cuando existen múltiples artefactos o versiones."
        ),
        why_it_matters=(
            "Evita inconsistencias: todos leen la misma métrica/artefacto y no una variante "
            "intermedia o legacy."
        ),
    ),
    "calibracion": GlossaryTerm(
        key="calibracion",
        label="Calibración",
        short_definition=(
            "Ajuste post-entrenamiento para que las probabilidades predichas reflejen mejor "
            "las frecuencias observadas."
        ),
        why_it_matters=(
            "Una PD mal calibrada distorsiona pricing, límites e IFRS9 aunque el AUC sea bueno."
        ),
    ),
    "conformal": GlossaryTerm(
        key="conformal",
        label="Conformal Prediction",
        short_definition=(
            "Método para construir intervalos de predicción con cobertura empírica controlada."
        ),
        why_it_matters=(
            "Convierte una PD puntual en un rango utilizable para decisiones robustas y "
            "monitoreo de incertidumbre."
        ),
    ),
    "coverage": GlossaryTerm(
        key="coverage",
        label="Coverage (Cobertura)",
        short_definition=(
            "Porcentaje de casos donde el valor real cae dentro del intervalo de predicción."
        ),
        why_it_matters=(
            "Cobertura por debajo del objetivo implica intervalos optimistas; muy por encima "
            "implica conservadurismo costoso."
        ),
    ),
    "ece": GlossaryTerm(
        key="ece",
        label="ECE",
        short_definition=(
            "Expected Calibration Error: cuánto se alejan las probabilidades predichas de las "
            "frecuencias reales."
        ),
        why_it_matters=(
            "Ayuda a validar si la PD puede usarse de forma confiable en pricing, provisión y "
            "optimización."
        ),
    ),
    "brier": GlossaryTerm(
        key="brier",
        label="Brier Score",
        short_definition=(
            "Error cuadrático medio entre probabilidad predicha y resultado observado."
        ),
        why_it_matters=(
            "Resume calidad probabilística; menor es mejor. Penaliza probabilidades mal calibradas."
        ),
    ),
    "ks": GlossaryTerm(
        key="ks",
        label="KS",
        short_definition=(
            "Kolmogorov-Smirnov: máxima separación entre score de buenos y malos."
        ),
        why_it_matters=(
            "Sirve para definir cutoffs operativos y medir separabilidad útil en crédito."
        ),
    ),
    "gini": GlossaryTerm(
        key="gini",
        label="Gini",
        short_definition="Métrica de discriminación derivada de AUC: Gini = 2*AUC - 1.",
        why_it_matters=(
            "Es estándar histórico en credit scoring y facilita comparación con benchmarks bancarios."
        ),
    ),
    "price_of_robustness": GlossaryTerm(
        key="price_of_robustness",
        label="Price of Robustness",
        short_definition=(
            "Costo económico de proteger el portafolio contra un peor caso plausible de riesgo."
        ),
        why_it_matters=(
            "Cuantifica el trade-off entre retorno esperado y resiliencia de la política."
        ),
    ),
    "baseline_vs_canonical": GlossaryTerm(
        key="baseline_vs_canonical",
        label="Baseline vs Canónico",
        short_definition=(
            "Baseline = referencia simple/challenger. Canónico = artefacto oficial adoptado para "
            "reporting y decisión."
        ),
        why_it_matters=(
            "Evita confundir comparativas exploratorias con la versión operativa aprobada."
        ),
    ),
}


ALIASES = {
    "canónico": "canonico",
    "canonico": "canonico",
    "brier score": "brier",
    "conformal prediction": "conformal",
    "price of robustness": "price_of_robustness",
}


def normalize_term_key(term: str) -> str:
    """Normalize term labels to glossary keys."""
    clean = term.strip().lower()
    return ALIASES.get(clean, clean)


def get_glossary_term(term: str) -> GlossaryTerm | None:
    """Return glossary entry by key or label alias."""
    return GLOSSARY_REGISTRY.get(normalize_term_key(term))
