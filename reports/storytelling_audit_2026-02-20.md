# Storytelling Audit (Streamlit)
Fecha: 2026-02-20

## 1) Objetivo
Revisar consistencia narrativa en páginas Streamlit para:

1. Reducir redundancia.
2. Evitar claims rígidos/inconsistentes.
3. Mejorar comprensión para lectores no expertos.
4. Asegurar que cada página responda una decisión de negocio.

## 2) Hallazgos principales

### A. Inconsistencias detectadas
- Referencias rígidas a calibración específica (`Platt`) en páginas donde el pipeline selecciona método de forma dinámica.
- Cifras históricas de volumen (`2.26M`) no alineadas con narrativa actual del proyecto (`2.93M raw`, `1.86M resueltos`).
- `pipeline_summary.json` desactualizado respecto a artefactos canónicos de optimización/conformal.

### B. Riesgos narrativos
- Varias páginas explican “qué es la técnica” pero no cierran con “qué decisión habilita”.
- Exceso de texto técnico antes de situar valor de negocio para audiencia general.
- Repetición de contexto transversal sin un marco uniforme de lectura.

## 3) Cambios aplicados

### 3.1 Sincronización de artefactos
- Ejecutado: `uv run python scripts/export_streamlit_artifacts.py`
- Resultado: `pipeline_summary.json` actualizado con estado actual (conformal 7/7, robust return actualizado).

### 3.2 Introducción narrativa estándar (uso selectivo)
Se añadió helper reusable:
- `streamlit_app/components/narrative.py`
  - función nueva: `storytelling_intro(...)`

Se aplicó solo en páginas donde agrega contexto para audiencia mixta (negocio/no experta), incluyendo:
- `streamlit_app/pages/executive_summary.py`
- `streamlit_app/pages/data_story.py`
- `streamlit_app/pages/feature_engineering.py`
- `streamlit_app/pages/model_laboratory.py`
- `streamlit_app/pages/uncertainty_quantification.py`
- `streamlit_app/pages/portfolio_optimizer.py`
- `streamlit_app/pages/ifrs9_provisions.py`
- `streamlit_app/pages/causal_intelligence.py`
- `streamlit_app/pages/survival_analysis.py`
- `streamlit_app/pages/time_series_outlook.py`
- `streamlit_app/pages/model_governance.py`
- `streamlit_app/pages/glossary_fundamentals.py`

Estructura añadida en cada caso:
- Qué resuelve la técnica.
- Por qué importa en negocio.
- Decisión que habilita.
- Ruta sugerida de lectura.

Se removió explícitamente de páginas de investigación/papers/tesis y de páginas muy técnicas
para mantener tono experto y evitar texto introductorio redundante.

### 3.3 Corrección de claims rígidos
- `streamlit_app/pages/executive_summary.py`
  - “CatBoost + Platt” -> “CatBoost + calibración”.
  - texto de contexto de Lending Club sin cifra inconsistente de 2.26M.
- `streamlit_app/pages/tech_stack.py`
  - referencias a “Platt” cambiadas a “calibración probabilística” cuando aplica arquitectura general.
- `streamlit_app/pages/thesis_contribution.py`
  - actualización de narrativa de tamaño de dataset (`2.93M raw`, `1.86M resueltos`, `142` variables raw).

### 3.4 Guía de storytelling y fuentes externas
- `streamlit_app/pages/research_best_practices.py`
  - nueva sección “Storytelling para audiencia no experta”.
  - checklist accionable.
  - enlaces externos de buenas prácticas.

## 4) Buenas prácticas adoptadas (resumen operativo)

1. Empezar por la decisión, no por la técnica.
2. Separar claramente:
   - contexto,
   - evidencia,
   - recomendación.
3. Mostrar umbrales/targets cuando haya métrica.
4. Evitar snapshots rígidos en texto; preferir lectura de artefactos.
5. Mantener continuidad narrativa con llamada a la siguiente página.

## 5) Páginas con mayor impacto para lectores no expertos

Orden recomendado de lectura:

1. `streamlit_app/pages/executive_summary.py`
2. `streamlit_app/pages/glossary_fundamentals.py`
3. `streamlit_app/pages/model_laboratory.py`
4. `streamlit_app/pages/uncertainty_quantification.py`
5. `streamlit_app/pages/portfolio_optimizer.py`
6. `streamlit_app/pages/ifrs9_provisions.py`

## 6) Referencias externas usadas

- Microsoft Learn (dashboard design):
  https://learn.microsoft.com/en-us/power-bi/guidance/dashboard-design
- Microsoft Learn (effective storytelling in Power BI):
  https://learn.microsoft.com/en-us/training/modules/power-bi-effective-storytelling/
- Tableau Blueprint (data storytelling):
  https://www.tableau.com/learn/blueprint/data-storytelling
- Adaptive dashboards research:
  https://arxiv.org/abs/2404.11131

## 7) Próximo paso recomendado
Implementar y mantener tests de consistencia narrativa para prevenir regresiones:

- patrones prohibidos (claims rígidos de métricas/calibración),
- verificación de presencia de bloque `storytelling_intro` solo en páginas objetivo,
- verificación de ausencia en sección de investigación (tono experto),
- validación de lectura de artefactos canónicos en lugar de cifras fijas.

Estado actual:
- `tests/test_docs/test_storytelling_consistency_guardrails.py` protege claims rígidos detectados.
- `tests/test_docs/test_storytelling_intro_coverage.py` protege cobertura selectiva y ausencia en research pages.
