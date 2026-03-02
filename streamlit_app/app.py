"""Dashboard integral de riesgo de credito.

Run: uv run streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamlit_app.components.dvc_kpi_spine import build_metric_cards
from streamlit_app.content.page_contracts import PAGE_CONTRACTS
from streamlit_app.theme import inject_custom_css
from streamlit_app.utils import load_dvc_metrics_summary, load_runtime_status

st.set_page_config(
    page_title="Riesgo de Credito E2E",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
runtime_status = load_runtime_status()
dvc_metrics = load_dvc_metrics_summary()
tests_total = int(runtime_status.get("test_suite_total", 0) or 0)
pages_total = int(runtime_status.get("streamlit_pages_total", 0) or 0)
tests_label = str(tests_total) if tests_total > 0 else "N/D"
pages_label = str(pages_total) if pages_total > 0 else "N/D"
contracts_total = len(PAGE_CONTRACTS)

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
            st.Page("pages/ab_testing_simulation.py", title="Simulación A/B", icon="🧪"),
            st.Page("pages/ifrs9_provisions.py", title="Provisiones IFRS9", icon="🏦"),
        ],
        "Gobernanza": [
            st.Page("pages/model_governance.py", title="Gobernanza del Modelo", icon="🛡️"),
            st.Page("pages/tech_stack.py", title="Stack Tecnológico", icon="🛠️"),
        ],
        "Exploración": [
            st.Page("pages/chat_with_data.py", title="Chat con Datos", icon="💬"),
        ],
        "Investigación": [
            st.Page(
                "pages/tesis_especializacion.py",
                title="Tesis Especialización",
                icon="🎓",
            ),
            st.Page("pages/thesis_contribution.py", title="Contribución de Tesis", icon="🎯"),
            st.Page("pages/research_landscape.py", title="Panorama de Investigación", icon="🔬"),
            st.Page("pages/paper_1_cp_robust_opt.py", title="Paper 1: CP + Robust Opt", icon="🧪"),
            st.Page("pages/paper_2_ifrs9_e2e.py", title="Paper 2: IFRS9 E2E", icon="🏦"),
            st.Page("pages/paper_3_mondrian.py", title="Paper 3: Mondrian", icon="📐"),
            st.Page(
                "pages/research_best_practices.py",
                title="Buenas Prácticas y Herramientas",
                icon="🧰",
            ),
        ],
        "Anexos": [
            st.Page("pages/gpu_benchmark.py", title="Benchmark RAPIDS GPU", icon="⚡"),
        ],
    }
)

# ── Sidebar info ──
def _render_sidebar_health() -> None:
    st.markdown("#### Estado del proyecto")
    if dvc_metrics:
        cards = build_metric_cards(dvc_metrics, "executive")[:3]
        for card in cards:
            st.metric(card["label"], card["value"], help=card.get("help"))
    else:
        st.caption("KPIs DVC no disponibles en este entorno.")


with st.sidebar:
    st.markdown("---")
    st.markdown(
        f"""
<div style="font-size: 0.78em; color: #5F6B7A; line-height: 1.6;">
<b>Proyecto de Tesis</b><br>
Carlos Vergara<br>
1.35M préstamos · 2007-2020<br>
{tests_label} tests · {pages_label} páginas · {contracts_total} contratos<br>
<i>CatBoost + Conformal + Pyomo</i>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("Snapshot canónico (DVC) para lectura rápida")
    if hasattr(st, "fragment"):
        @st.fragment
        def _sidebar_fragment() -> None:
            _render_sidebar_health()

        _sidebar_fragment()
    else:
        _render_sidebar_health()

pg.run()
