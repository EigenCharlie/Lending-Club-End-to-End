"""Chat con datos: SQL read-only + asistente opcional NL->SQL."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import os
import re

import plotly.express as px
import streamlit as st

from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.components.story_shell import (
    render_caveats,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import query_duckdb, suggest_sql_with_grok

READ_ONLY_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)

SCHEMA_CONTEXT = """
Staging:
- main_staging.stg_loan_master
- main_staging.stg_test_predictions
- main_staging.stg_conformal_intervals
- main_staging.stg_time_series
- main_staging.stg_portfolio_allocations
- main_staging.stg_cate_estimates
- main_staging.stg_ifrs9_ecl
- main_staging.stg_governance

Marts:
- main_credit_risk.fct_loan_risk_assessment
- main_credit_risk.fct_portfolio_allocation
- main_credit_risk.fct_ifrs9_ecl
- main_credit_risk.dim_loan_grades
- main_analytics.fct_model_performance_by_cohort
- main_analytics.fct_conformal_coverage
- main_analytics.fct_causal_effects
- main_analytics.fct_forecasts
- main_feature_store.obt_loan_features
"""


def _is_read_only(sql: str) -> bool:
    return not bool(READ_ONLY_SQL.search(sql))


def _select_template(template_name: str, templates_map: dict[str, str]) -> None:
    st.session_state["chat_sql"] = templates_map[template_name].strip()
    st.session_state["chat_template_name"] = template_name
    st.session_state["run_chat_query"] = True


st.title("💬 Chat con Datos")
page_contract = get_page_contract("chat_with_data")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página ofrece trazabilidad técnica: permite contrastar con SQL explícito los insights mostrados en las páginas narrativas."
)
st.markdown(
    "Explora la base analítica directamente con SQL. "
    "Opcionalmente puedes convertir preguntas en lenguaje natural a SQL usando Grok."
)
st.markdown(
    """
Esta página es el laboratorio exploratorio del proyecto: permite verificar hipótesis y reproducir resultados sin salir de
la aplicación. Su propósito no es reemplazar notebooks técnicos, sino ofrecer una capa interactiva para validar tablas,
segmentaciones y métricas clave del pipeline sobre DuckDB. Todas las consultas se ejecutan en modo lectura para mantener
seguridad y consistencia de artefactos.
"""
)
st.warning(
    "Aviso metodológico: una consulta SQL puede describir asociaciones, pero no prueba causalidad ni valida una política por sí sola."
)

st.subheader("Consultas rápidas")
templates = {
    "Default por grade": """
SELECT grade, count(*) AS n_prestamos,
       round(avg(default_flag), 4) AS tasa_default,
       round(avg(loan_amnt), 0) AS monto_promedio
FROM main_staging.stg_loan_master
GROUP BY grade ORDER BY grade""",
    "Desempeño por cohorte": """
WITH joined AS (
    SELECT
        date_trunc('month', l.issue_d) AS cohort_month,
        l.grade,
        p.y_true::DOUBLE AS y_true,
        p.pd_calibrated::DOUBLE AS y_prob_final
    FROM main_staging.stg_test_predictions p
    INNER JOIN main_staging.stg_loan_master l
        ON CAST(l.id AS VARCHAR) = CAST(p.loan_id AS VARCHAR)
)
SELECT
    cohort_month,
    grade,
    count(*) AS n_loans,
    round(avg(y_true), 4) AS actual_default_rate,
    round(avg(y_prob_final), 4) AS avg_predicted_pd,
    round(avg(abs(y_prob_final - y_true)), 4) AS mae
FROM joined
GROUP BY cohort_month, grade
ORDER BY cohort_month DESC, grade
LIMIT 60""",
    "Cobertura conformal": """
WITH base AS (
    SELECT
        grade,
        CASE
            WHEN pd_point < 0.10 THEN 'bajo'
            WHEN pd_point < 0.20 THEN 'medio'
            ELSE 'alto'
        END AS risk_category,
        y_true::DOUBLE AS y_true,
        pd_point::DOUBLE AS pd_point,
        pd_low::DOUBLE AS pd_low,
        pd_high::DOUBLE AS pd_high
    FROM main_staging.stg_conformal_intervals
)
SELECT
    grade,
    risk_category,
    count(*) AS n_loans,
    round(avg(CASE WHEN y_true BETWEEN pd_low AND pd_high THEN 1 ELSE 0 END), 4) AS empirical_coverage,
    round(avg(pd_high - pd_low), 4) AS avg_interval_width,
    round(avg(pd_point), 4) AS avg_pd
FROM base
GROUP BY grade, risk_category
ORDER BY grade, risk_category""",
    "Resumen IFRS9": """
SELECT * FROM main_credit_risk.fct_ifrs9_ecl""",
    "Asignaciones portafolio": """
SELECT * FROM main_credit_risk.fct_portfolio_allocation LIMIT 30""",
    "Tablas disponibles": """
SELECT table_schema, table_name, table_type
FROM information_schema.tables
ORDER BY table_schema, table_name""",
}
audit_templates = {
    "Leakage guard (columnas sospechosas)": """
SELECT
    table_schema,
    table_name,
    column_name
FROM information_schema.columns
WHERE table_schema IN ('main_staging', 'main_feature_store')
  AND (
      lower(column_name) LIKE '%recover%'
      OR lower(column_name) LIKE '%pymnt%'
      OR lower(column_name) LIKE '%collection%'
      OR lower(column_name) LIKE '%out_prncp%'
      OR lower(column_name) LIKE '%last_pymnt%'
  )
ORDER BY table_schema, table_name, column_name""",
    "Drift básico por cohorte (share grade)": """
WITH base AS (
    SELECT
        date_trunc('quarter', issue_d) AS cohort_quarter,
        grade
    FROM main_staging.stg_loan_master
    WHERE issue_d IS NOT NULL
),
agg AS (
    SELECT
        cohort_quarter,
        grade,
        count(*) AS n_loans
    FROM base
    GROUP BY cohort_quarter, grade
)
SELECT
    cohort_quarter,
    grade,
    n_loans,
    round(n_loans::DOUBLE / sum(n_loans) OVER (PARTITION BY cohort_quarter), 4) AS grade_share
FROM agg
ORDER BY cohort_quarter DESC, grade""",
    "Calibración rápida por deciles": """
WITH base AS (
    SELECT
        y_true::DOUBLE AS y_true,
        pd_calibrated::DOUBLE AS pd_hat
    FROM main_staging.stg_test_predictions
),
bucketed AS (
    SELECT
        ntile(10) OVER (ORDER BY pd_hat) AS decile,
        y_true,
        pd_hat
    FROM base
)
SELECT
    decile,
    count(*) AS n_loans,
    round(avg(pd_hat), 4) AS avg_predicted_pd,
    round(avg(y_true), 4) AS observed_default_rate,
    round(avg(abs(pd_hat - y_true)), 4) AS avg_abs_error
FROM bucketed
GROUP BY decile
ORDER BY decile""",
}

if "chat_sql" not in st.session_state:
    st.session_state["chat_sql"] = "SELECT * FROM main_credit_risk.dim_loan_grades"
if "chat_template_name" not in st.session_state:
    st.session_state["chat_template_name"] = "Consulta inicial"
if "run_chat_query" not in st.session_state:
    st.session_state["run_chat_query"] = False

col1, col2, col3 = st.columns(3)
for idx, name in enumerate(templates.keys()):
    col = [col1, col2, col3][idx % 3]
    col.button(
        name,
        width="stretch",
        key=f"quick_query_{idx}",
        on_click=_select_template,
        args=(name, templates),
    )

st.caption(f"Plantilla activa: **{st.session_state.get('chat_template_name', 'N/A')}**")

st.subheader("Plantillas auditables (leakage/drift/checks básicos)")
st.caption(
    "Estas consultas no reemplazan validación estadística completa, pero sirven como primer filtro auditable antes de sacar conclusiones."
)
audit_col1, audit_col2, audit_col3 = st.columns(3)
for idx, name in enumerate(audit_templates.keys()):
    col = [audit_col1, audit_col2, audit_col3][idx % 3]
    col.button(
        name,
        width="stretch",
        key=f"audit_query_{idx}",
        on_click=_select_template,
        args=(name, audit_templates),
    )

st.subheader("Asistente lenguaje natural -> SQL (opcional)")
grok_enabled = bool(os.getenv("GROK_API_KEY", "").strip())
if grok_enabled:
    question = st.text_input(
        "Escribe tu pregunta",
        placeholder="Ejemplo: tendencia mensual de default para grades C a G",
    )
    if st.button("Generar SQL con Grok"):
        if not question.strip():
            st.warning("Primero escribe una pregunta.")
        else:
            try:
                with st.status("Generando SQL desde lenguaje natural...", expanded=False) as _nl_status:
                    suggestion = suggest_sql_with_grok(question, SCHEMA_CONTEXT)
                    _nl_status.update(label="SQL generado", state="complete")
                sql_suggested = suggestion.get("sql", "")
                if not _is_read_only(sql_suggested):
                    st.error("La consulta generada no es read-only. Ajusta manualmente.")
                else:
                    st.session_state["chat_sql"] = sql_suggested
                    st.session_state["chat_template_name"] = "Generada por Grok"
                    st.session_state["run_chat_query"] = False
                    st.success("SQL generado. Revísalo antes de ejecutar.")
                    rationale = suggestion.get("rationale", "")
                    if rationale:
                        st.caption(rationale)
                    st.rerun()
            except Exception as exc:
                st.error(f"Error generando SQL con Grok: {exc}")
else:
    st.info("Configura `GROK_API_KEY` para habilitar el asistente NL->SQL.")

sql = st.text_area(
    "Consulta SQL",
    key="chat_sql",
    height=150,
    help="Solo se permiten consultas de lectura.",
)

run_query_clicked = st.button("Ejecutar consulta", type="primary")
should_run_query = (run_query_clicked or st.session_state.get("run_chat_query", False)) and bool(
    sql.strip()
)
if should_run_query:
    st.session_state["run_chat_query"] = False
    if not _is_read_only(sql):
        st.error("Solo se permiten consultas de lectura.")
    else:
        try:
            with st.status("Ejecutando consulta SQL (read-only)...", expanded=False) as _sql_status:
                result = query_duckdb(sql)
                _sql_status.update(label=f"Consulta completada ({len(result)} filas)", state="complete")
            st.success(f"Filas retornadas: {len(result)}")
            st.dataframe(result, width="stretch", height=420)
            if len(result) == 0:
                st.warning(
                    "La consulta se ejecutó sin error pero retornó 0 filas. "
                    "Prueba ajustar filtros o usar `Tablas disponibles` para validar fuentes."
                )

            if len(result.columns) >= 2 and len(result) <= 120:
                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                cat_cols = [c for c in result.columns if c not in numeric_cols]
                if numeric_cols and cat_cols:
                    st.subheader("Visualización automática")
                    fig = px.bar(
                        result,
                        x=cat_cols[0],
                        y=numeric_cols[0],
                        title=f"{numeric_cols[0]} por {cat_cols[0]}",
                    )
                    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
                    fig.update_traces(marker_color="#00D4AA")
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "Propósito: dar lectura rápida al resultado SQL sin salir de la página. "
                        "Insight: útil para validar segmentaciones y tendencias antes de análisis profundo."
                    )
        except Exception as exc:
            st.error(f"Error en consulta: {exc}")

with st.expander("Referencia de esquemas (dbt + DuckDB)"):
    st.markdown(
        """
**Staging (vistas sobre parquet):**
- `main_staging.stg_loan_master`
- `main_staging.stg_test_predictions`
- `main_staging.stg_conformal_intervals`
- `main_staging.stg_time_series`
- `main_staging.stg_portfolio_allocations`
- `main_staging.stg_cate_estimates`
- `main_staging.stg_ifrs9_ecl`
- `main_staging.stg_governance`

**Marts analíticos:**
- `main_credit_risk.fct_loan_risk_assessment`
- `main_credit_risk.fct_portfolio_allocation`
- `main_credit_risk.fct_ifrs9_ecl`
- `main_credit_risk.dim_loan_grades`
- `main_analytics.fct_model_performance_by_cohort`
- `main_analytics.fct_conformal_coverage`
- `main_analytics.fct_causal_effects`
- `main_analytics.fct_forecasts`
- `main_feature_store.obt_loan_features`
"""
    )

st.markdown(
    """
Como cierre, `Chat con Datos` funciona como mecanismo de transparencia del proyecto: cualquier insight mostrado en páginas
narrativas puede contrastarse aquí con SQL explícito. Eso fortalece trazabilidad y facilita que lectores técnicos o de
negocio exploren el mismo stack con distintos niveles de profundidad.
"""
)
render_caveats(
    [
        "SQL exploratorio no demuestra causalidad: para decisiones de política usar páginas causales y validación de supuestos.",
        "El asistente NL→SQL puede proponer consultas válidas pero analíticamente inadecuadas; revisar siempre.",
        "Resultados dependen de la capa dbt/DuckDB disponible en el entorno local y su frescura.",
    ]
)
render_page_feedback("chat_with_data")

next_page_teaser(
    "Resumen Ejecutivo",
    "Regresa al hub principal del proyecto.",
    "pages/executive_summary.py",
)
