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
    "aleatoric": GlossaryTerm(
        key="aleatoric",
        label="Incertidumbre Aleatoric",
        short_definition="Ruido irreducible del sistema generador de datos.",
        why_it_matters=(
            "Define el piso de error alcanzable y evita prometer precisión irreal."
        ),
    ),
    "epistemic": GlossaryTerm(
        key="epistemic",
        label="Incertidumbre Epistemic",
        short_definition="Incertidumbre reducible por falta de datos o cobertura de señal.",
        why_it_matters=(
            "Permite priorizar acciones de mejora de datos, segmentación y recalibración."
        ),
    ),
    "confidence_interval": GlossaryTerm(
        key="confidence_interval",
        label="Confidence Interval",
        short_definition=(
            "Intervalo para parámetros del modelo; no representa directamente una predicción futura."
        ),
        why_it_matters=(
            "Evita usar intervalos de inferencia como si fueran bandas de decisión por préstamo."
        ),
    ),
    "prediction_interval": GlossaryTerm(
        key="prediction_interval",
        label="Prediction Interval",
        short_definition=(
            "Rango para observaciones futuras, útil para decisiones bajo incertidumbre."
        ),
        why_it_matters=(
            "Es el formato correcto para robustez, provisión prudencial y límites operativos."
        ),
    ),
    "mcar_mar_mnar": GlossaryTerm(
        key="mcar_mar_mnar",
        label="MCAR/MAR/MNAR",
        short_definition="Taxonomía de mecanismos de faltantes (aleatorio total, condicional o no aleatorio).",
        why_it_matters=(
            "Ayuda a elegir una estrategia de imputación coherente y con menor sesgo."
        ),
    ),
    "data_leakage": GlossaryTerm(
        key="data_leakage",
        label="Data Leakage",
        short_definition=(
            "Uso accidental de información futura o post-evento durante entrenamiento/evaluación."
        ),
        why_it_matters=(
            "Inflar métricas offline por leakage conduce a decisiones frágiles en producción."
        ),
    ),
    "nested_cv": GlossaryTerm(
        key="nested_cv",
        label="Nested Cross-Validation",
        short_definition=(
            "Esquema que separa tuning y evaluación para reducir sesgo optimista de selección."
        ),
        why_it_matters=(
            "Mejora la credibilidad de la métrica reportada para despliegue real."
        ),
    ),
    "covariate_shift": GlossaryTerm(
        key="covariate_shift",
        label="Covariate Shift",
        short_definition="Cambio de distribución de variables de entrada entre train y operación.",
        why_it_matters=(
            "Puede deteriorar calibración y decisiones incluso con el mismo modelo."
        ),
    ),
    "concept_drift": GlossaryTerm(
        key="concept_drift",
        label="Concept Drift",
        short_definition="Cambio de la relación entre features y target a lo largo del tiempo.",
        why_it_matters=(
            "Requiere monitoreo y escalamiento para evitar degradación silenciosa."
        ),
    ),
    "c2st": GlossaryTerm(
        key="c2st",
        label="C2ST / Two-Sample Tests",
        short_definition=(
            "Pruebas formales para detectar diferencias distribucionales entre muestras."
        ),
        why_it_matters=(
            "Convierte alertas de drift en criterios auditables y reproducibles."
        ),
    ),
    "class_imbalance": GlossaryTerm(
        key="class_imbalance",
        label="Class Imbalance",
        short_definition="Desbalance de clases con riesgo de métricas engañosas.",
        why_it_matters=(
            "Obliga a usar métricas y umbrales adecuados al costo real del error."
        ),
    ),
    "iid_caveat": GlossaryTerm(
        key="iid_caveat",
        label="Caveat i.i.d.",
        short_definition=(
            "Advertencia sobre el supuesto de muestras independientes e idénticamente distribuidas."
        ),
        why_it_matters=(
            "Evita validación incorrecta en series temporales y cohortes crediticias."
        ),
    ),
    "extrapolation": GlossaryTerm(
        key="extrapolation",
        label="Extrapolation Risk",
        short_definition="Predicción fuera del rango observado en entrenamiento.",
        why_it_matters=(
            "Fuera de soporte histórico, la confianza de la decisión debe disminuir."
        ),
    ),
    "convex_hull": GlossaryTerm(
        key="convex_hull",
        label="Convex Hull",
        short_definition=(
            "Aproximación geométrica del soporte observado para detectar zonas fuera de dominio."
        ),
        why_it_matters=(
            "Ayuda a separar interpolación confiable de extrapolación riesgosa."
        ),
    ),
    "optimizer_curse": GlossaryTerm(
        key="optimizer_curse",
        label="Optimizer's Curse",
        short_definition="Sesgo optimista al escoger el mejor de muchos intentos ruidosos.",
        why_it_matters=(
            "Previene claims inflados al exigir estabilidad por seeds y splits."
        ),
    ),
    "no_free_lunch": GlossaryTerm(
        key="no_free_lunch",
        label="No Free Lunch",
        short_definition="No existe método universalmente mejor para todos los problemas.",
        why_it_matters=(
            "Fuerza selección contextual por trade-offs reales, no por moda técnica."
        ),
    ),
    "proper_scoring_rules": GlossaryTerm(
        key="proper_scoring_rules",
        label="Proper Scoring Rules",
        short_definition=(
            "Métricas probabilísticas (p.ej. log loss, Brier) que incentivan probabilidades honestas."
        ),
        why_it_matters=(
            "Clave para pricing, provisión y decisiones que dependen de probabilidad calibrada."
        ),
    ),
    "decision_threshold": GlossaryTerm(
        key="decision_threshold",
        label="Decision Threshold",
        short_definition=(
            "Umbral que traduce una probabilidad continua en una decisión binaria operativa."
        ),
        why_it_matters=(
            "Hace explícito el trade-off entre crecimiento, riesgo y costo de error."
        ),
    ),
    "nonconformity_score": GlossaryTerm(
        key="nonconformity_score",
        label="Nonconformity Score",
        short_definition=(
            "Medida de qué tan atípica es una observación frente al comportamiento calibrado."
        ),
        why_it_matters=(
            "Define el cuantil conformal y, por tanto, el ancho efectivo del intervalo."
        ),
    ),
    "marginal_coverage": GlossaryTerm(
        key="marginal_coverage",
        label="Marginal Coverage",
        short_definition=(
            "Cobertura promedio sobre la población; no garantiza cobertura exacta para cada x."
        ),
        why_it_matters=(
            "Evita sobreinterpretar el target 90/95 como promesa individual por observación."
        ),
    ),
    "conditional_coverage": GlossaryTerm(
        key="conditional_coverage",
        label="Conditional Coverage",
        short_definition=(
            "Cobertura condicionada a covariables específicas; difícil de garantizar sin supuestos fuertes."
        ),
        why_it_matters=(
            "Aclara límites metodológicos y reduce sobreclaims en comités técnicos."
        ),
    ),
    "calibration_size": GlossaryTerm(
        key="calibration_size",
        label="Calibration Set Size",
        short_definition=(
            "Tamaño de la muestra usada para estimar cuantiles conformales."
        ),
        why_it_matters=(
            "n pequeño aumenta la varianza de cobertura observada, sobre todo por subgrupo."
        ),
    ),
    "weighted_conformal": GlossaryTerm(
        key="weighted_conformal",
        label="Weighted Conformal",
        short_definition=(
            "Variante conformal que pondera calibración para escenarios con covariate shift."
        ),
        why_it_matters=(
            "Mejora robustez cuando la distribución de operación difiere del histórico."
        ),
    ),
    "adaptive_conformal": GlossaryTerm(
        key="adaptive_conformal",
        label="Adaptive Conformal",
        short_definition=(
            "Familia de métodos que recalibran dinámicamente ante no estacionariedad."
        ),
        why_it_matters=(
            "Permite sostener cobertura útil en entornos con drift y cambios de régimen."
        ),
    ),
    "jackknife_plus": GlossaryTerm(
        key="jackknife_plus",
        label="Jackknife+",
        short_definition=(
            "Método conformal que reutiliza datos para inferencia robusta con fuerte cobertura empírica."
        ),
        why_it_matters=(
            "Alternativa cuando separar un set de calibración fijo reduce demasiada data de entrenamiento."
        ),
    ),
    "cqr": GlossaryTerm(
        key="cqr",
        label="Conformalized Quantile Regression (CQR)",
        short_definition=(
            "Conformal sobre cuantiles para intervalos adaptativos en heteroscedasticidad."
        ),
        why_it_matters=(
            "Puede lograr mejor equilibrio cobertura-ancho en regresión (LGD/EAD)."
        ),
    ),
}


ALIASES = {
    "canónico": "canonico",
    "canonico": "canonico",
    "brier score": "brier",
    "conformal prediction": "conformal",
    "price of robustness": "price_of_robustness",
    "aleatoric uncertainty": "aleatoric",
    "epistemic uncertainty": "epistemic",
    "intervalo de confianza": "confidence_interval",
    "intervalo de prediccion": "prediction_interval",
    "prediction interval": "prediction_interval",
    "confidence interval": "confidence_interval",
    "mcar/mar/mnar": "mcar_mar_mnar",
    "nested cv": "nested_cv",
    "covariate shift": "covariate_shift",
    "concept drift": "concept_drift",
    "class imbalance": "class_imbalance",
    "i.i.d caveat": "iid_caveat",
    "optimizer's curse": "optimizer_curse",
    "optimizer curse": "optimizer_curse",
    "no free lunch": "no_free_lunch",
    "proper scoring rules": "proper_scoring_rules",
    "decision threshold": "decision_threshold",
    "nonconformity score": "nonconformity_score",
    "marginal coverage": "marginal_coverage",
    "conditional coverage": "conditional_coverage",
    "calibration size": "calibration_size",
    "weighted conformal": "weighted_conformal",
    "adaptive conformal": "adaptive_conformal",
    "jackknife+": "jackknife_plus",
    "jackknife plus": "jackknife_plus",
    "cqr": "cqr",
}


def normalize_term_key(term: str) -> str:
    """Normalize term labels to glossary keys."""
    clean = term.strip().lower()
    return ALIASES.get(clean, clean)


def get_glossary_term(term: str) -> GlossaryTerm | None:
    """Return glossary entry by key or label alias."""
    return GLOSSARY_REGISTRY.get(normalize_term_key(term))
