"""Dashboard integral de riesgo de credito.

Run: uv run streamlit run streamlit_app/app.py
"""
# ruff: noqa: E402

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
        "Libro y Fundamentos": [
            st.Page(
                "pages/executive_summary.py",
                title="Resumen Ejecutivo",
                icon=":material/home:",
                default=True,
            ),
            st.Page(
                "pages/glossary_fundamentals.py",
                title="Glosario y Fundamentos",
                icon=":material/menu_book:",
            ),
        ],
        "Pipeline Operativo": [
            st.Page(
                "pages/thesis_end_to_end.py",
                title="Visión End-to-End",
                icon=":material/explore:",
            ),
            st.Page(
                "pages/data_architecture.py",
                title="Arquitectura y Linaje de Datos",
                icon=":material/account_tree:",
            ),
            st.Page(
                "pages/thesis_defense.py",
                title="Mapa Integrado de Métodos",
                icon=":material/hub:",
            ),
            st.Page(
                "pages/feature_engineering.py",
                title="Ingeniería de Features",
                icon=":material/build:",
            ),
            st.Page(
                "pages/model_laboratory.py",
                title="Laboratorio de Modelos",
                icon=":material/science:",
            ),
            st.Page(
                "pages/uncertainty_quantification.py",
                title="Cuantificación de Incertidumbre",
                icon=":material/straighten:",
            ),
            st.Page(
                "pages/time_series_outlook.py",
                title="Panorama Temporal",
                icon=":material/timeline:",
            ),
            st.Page(
                "pages/survival_analysis.py",
                title="Análisis de Supervivencia",
                icon=":material/hourglass_bottom:",
            ),
            st.Page(
                "pages/portfolio_optimizer.py",
                title="Optimizador de Portafolio",
                icon=":material/work:",
            ),
            st.Page(
                "pages/ifrs9_provisions.py",
                title="Provisiones IFRS9",
                icon=":material/account_balance:",
            ),
        ],
        "Insight Factory": [
            st.Page(
                "pages/notebook_evidence.py",
                title="Atlas de Evidencia",
                icon=":material/library_books:",
            ),
            st.Page(
                "pages/data_story.py",
                title="Historia de Datos",
                icon=":material/bar_chart:",
            ),
            st.Page(
                "pages/model_interpretability.py",
                title="Explicabilidad e Interpretabilidad",
                icon=":material/psychology:",
            ),
            st.Page(
                "pages/causal_intelligence.py",
                title="Inteligencia Causal",
                icon=":material/genetics:",
            ),
            st.Page(
                "pages/ab_testing_simulation.py",
                title="Simulación A/B",
                icon=":material/experiment:",
            ),
            st.Page(
                "pages/chat_with_data.py",
                title="Chat con Datos",
                icon=":material/chat:",
            ),
            st.Page(
                "pages/gpu_benchmark.py",
                title="Benchmark RAPIDS GPU",
                icon=":material/bolt:",
            ),
        ],
        "Gobernanza y Libro": [
            st.Page(
                "pages/model_governance.py",
                title="Gobernanza del Modelo",
                icon=":material/shield:",
            ),
            st.Page(
                "pages/roadmap_backlog.py",
                title="Roadmap y Backlog",
                icon=":material/checklist:",
            ),
            st.Page(
                "pages/papers_backlog.py",
                title="Backlog Papers y Quarto",
                icon=":material/article:",
            ),
            st.Page(
                "pages/tech_stack.py",
                title="Stack Tecnológico",
                icon=":material/handyman:",
            ),
            st.Page(
                "pages/tesis_especializacion.py",
                title="Tesis Especialización",
                icon=":material/school:",
            ),
            st.Page(
                "pages/thesis_contribution.py",
                title="Contribución de Tesis",
                icon=":material/target:",
            ),
            st.Page(
                "pages/research_landscape.py",
                title="Panorama de Investigación",
                icon=":material/search:",
            ),
            st.Page(
                "pages/paper_1_cp_robust_opt.py",
                title="Paper 1: CP + Robust Opt",
                icon=":material/experiment:",
            ),
            st.Page(
                "pages/paper_2_ifrs9_e2e.py",
                title="Paper 2: IFRS9 E2E",
                icon=":material/account_balance:",
            ),
            st.Page(
                "pages/paper_3_mondrian.py",
                title="Paper 3: Mondrian",
                icon=":material/straighten:",
            ),
            st.Page(
                "pages/research_best_practices.py",
                title="Buenas Prácticas y Herramientas",
                icon=":material/construction:",
            ),
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
    st.caption(
        f"**Proyecto de Tesis** · Carlos Vergara\n\n"
        f"1.35M préstamos · 2007-2020\n\n"
        f"{tests_label} tests · {pages_label} páginas · {contracts_total} contratos\n\n"
        f"_canonical_rebuild + champion_search + insights_factory_"
    )
    st.caption("Snapshot canónico (DVC) para lectura rápida")
    if hasattr(st, "fragment"):
        @st.fragment
        def _sidebar_fragment() -> None:
            _render_sidebar_health()

        _sidebar_fragment()
    else:
        _render_sidebar_health()

try:
    pg.run()
except FileNotFoundError as exc:
    st.error(
        "Archivo de datos no encontrado. Ejecute el pipeline antes de usar esta página.",
        icon=":material/folder_off:",
    )
    st.caption(f"Detalle: `{exc}`")
except KeyError as exc:
    st.error(
        f"Clave o columna faltante: `{exc}`",
        icon=":material/key_off:",
    )
    st.caption(
        "Los artefactos pueden estar desactualizados. "
        "Ejecute: `uv run python scripts/run_canonical_rebuild.py`"
    )
except IndexError as exc:
    st.error(
        f"Datos insuficientes: `{exc}`",
        icon=":material/data_alert:",
    )
    st.caption(
        "El artefacto existe pero no contiene las filas esperadas. "
        "Re-ejecute el pipeline para regenerar los datos."
    )
except Exception as exc:
    st.error(
        f"Error inesperado: `{type(exc).__name__}: {exc}`",
        icon=":material/error:",
    )
    st.caption(
        "Sugerencia: `uv run python scripts/run_canonical_rebuild.py` para regenerar artefactos."
    )
