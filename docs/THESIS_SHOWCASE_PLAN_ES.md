# Plan de Ejecucion — Thesis Showcase Platform
Fecha: 2026-02-15

## 1) Objetivo

Priorizar el proyecto como plataforma de tesis y portafolio sobre dataset fijo:

- Storytelling integral en Streamlit.
- Capa de datos visible y defendible (DuckDB + dbt + Feast).
- Trazabilidad metodologica completa desde notebook 01 al 09.

## 2) Decision de Arquitectura

### Arquitectura elegida

1. **Streamlit (capa principal de experiencia)**:
   narrativas, metricas, visualizaciones, discusion metodologica y resultados.
2. **DuckDB (motor local)**:
   consultas analiticas reproducibles sobre artefactos parquet.
3. **dbt (capa de gobernanza)**:
   linaje, tests de calidad y documentacion automatica.
4. **Feast (capa de feature store showcase)**:
   entidades, feature views y feature services para consistencia train/serve.
5. **FastAPI + MCP (opcional)**:
   integraciones externas y extensiones de consumo.

### Racional

- No hay requerimiento de inferencia online sobre datos nuevos.
- El valor principal es comunicar rigor tecnico y coherencia de decision.
- Esta arquitectura minimiza complejidad operativa y maximiza impacto de presentacion.

## 3) Alcance de Cambios (Sesion actual)

### 3.1 Documentacion

- Alinear `README.md` con estado real y estrategia Streamlit-first.
- Actualizar `SESSION_STATE.md` como fuente de verdad vigente.
- Corregir menciones de "stubs" en `CLAUDE.md` y `docs/PROJECT_JUSTIFICATION.md`.
- Agregar adenda de alineacion en `reports/INFORME_INTEGRAL_PROYECTO_RIESGO_CREDITO_ES.md`.

### 3.2 Streamlit

- Crear pagina `Thesis End-to-End`:
  - metodologia completa del notebook 09
  - KPIs de impacto
  - trade-off robust vs non-robust
  - nexo IFRS9 y optimizacion
  - comandos de reproducibilidad
- Integrar pagina en navegacion principal.

### 3.3 Chat with Data

- Mantener SQL manual.
- Habilitar opcion NL->SQL con Grok via `GROK_API_KEY` (opcional).
- Sin hardcode de secretos en codigo.

### 3.4 Docker Compose

- Desacoplar Streamlit de dependencia obligatoria de API para modo showcase.

## 4) Siguientes Iteraciones

1. Pagina dedicada a "Triada de Poder" (DuckDB + dbt + Feast) con diagramas y lineage.
2. Pagina de "Defensa de Tesis" con:
   - preguntas frecuentes de jurado
   - amenazas a validez
   - limites y trabajo futuro
3. Mejorar Chat with Data:
   - plantillas por audiencia
   - validaciones SQL adicionales
   - prompting guiado por esquema activo
4. Automatizar smoke checks:
   - import de paginas Streamlit
   - verificacion de artefactos obligatorios
   - tests de regresion de metricas clave

## 5) Seguridad y Secretos

- Las llaves API (como `GROK_API_KEY`) se manejan solo por variables de entorno.
- No incluir secretos en markdown, codigo fuente ni commits.
