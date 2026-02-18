"""Dashboard integral de riesgo de credito.

Run: uv run streamlit run streamlit_app/app.py
"""

import streamlit as st

from streamlit_app.theme import inject_custom_css
from streamlit_app.utils import load_runtime_status

st.set_page_config(
    page_title="Riesgo de Credito E2E",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
runtime_status = load_runtime_status()
tests_total = int(runtime_status.get("test_suite_total", 0) or 0)
pages_total = int(runtime_status.get("streamlit_pages_total", 0) or 0)
tests_label = str(tests_total) if tests_total > 0 else "N/D"
pages_label = str(pages_total) if pages_total > 0 else "N/D"

# ── Navigation ──
pg = st.navigation(
    {
        "Inicio": [
            st.Page(
                "pages/executive_summary.py", title="Resumen Ejecutivo", icon="🏠", default=True
            ),
            st.Page("pages/glossary_fundamentals.py", title="Glosario y Fundamentos", icon="📖"),
        ],
        "Recorrido End-to-End": [
            st.Page("pages/thesis_contribution.py", title="Contribución de Tesis", icon="🎯"),
            st.Page("pages/research_landscape.py", title="Panorama de Investigación", icon="🔬"),
            st.Page("pages/thesis_end_to_end.py", title="Visión End-to-End", icon="🧭"),
            st.Page("pages/data_architecture.py", title="Arquitectura y Linaje de Datos", icon="🗂️"),
            st.Page("pages/thesis_defense.py", title="Mapa Integrado de Métodos", icon="🧩"),
            st.Page("pages/notebook_evidence.py", title="Atlas de Evidencia", icon="📚"),
        ],
        "Analítica": [
            st.Page("pages/feature_engineering.py", title="Ingeniería de Features", icon="🔧"),
            st.Page("pages/data_story.py", title="Historia de Datos", icon="📊"),
            st.Page("pages/model_laboratory.py", title="Laboratorio de Modelos", icon="🔬"),
            st.Page(
                "pages/uncertainty_quantification.py",
                title="Cuantificación de Incertidumbre",
                icon="📐",
            ),
            st.Page("pages/time_series_outlook.py", title="Panorama Temporal", icon="📈"),
            st.Page("pages/survival_analysis.py", title="Análisis de Supervivencia", icon="⏳"),
            st.Page("pages/causal_intelligence.py", title="Inteligencia Causal", icon="🧬"),
        ],
        "Decisiones": [
            st.Page("pages/portfolio_optimizer.py", title="Optimizador de Portafolio", icon="💼"),
            st.Page("pages/ifrs9_provisions.py", title="Provisiones IFRS9", icon="🏦"),
        ],
        "Gobernanza": [
            st.Page("pages/model_governance.py", title="Gobernanza del Modelo", icon="🛡️"),
            st.Page("pages/tech_stack.py", title="Stack Tecnológico", icon="🛠️"),
        ],
        "Exploración": [
            st.Page("pages/chat_with_data.py", title="Chat con Datos", icon="💬"),
        ],
    }
)

# ── Sidebar info ──
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
<div style="font-size: 0.78em; color: #5F6B7A; line-height: 1.6;">
<b>Proyecto de Tesis</b><br>
Carlos Vergara<br>
1.35M préstamos · 2007-2020<br>
{tests_label} tests · {pages_label} páginas<br>
<i>CatBoost + Conformal + Pyomo</i>
</div>
""",
        unsafe_allow_html=True,
    )

pg.run()
