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
from streamlit_app.content.companion_surface import ACTIVE_COMPANION_LABS
from streamlit_app.content.page_contracts import PAGE_CONTRACTS
from streamlit_app.theme import inject_custom_css
from streamlit_app.utils import load_dvc_metrics_summary, load_runtime_status

st.set_page_config(
    page_title="Lending Club Companion Lab",
    page_icon="🧪",
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
        "Companion Labs": [
            st.Page(lab.path, title=lab.title, icon=lab.icon, default=lab.default)
            for lab in ACTIVE_COMPANION_LABS
        ]
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
        f"**Quarto = source of truth** · Streamlit = companion local opcional\n\n"
        f"1.35M préstamos · 2007-2020\n\n"
        f"{tests_label} tests · {pages_label} páginas en snapshot · {contracts_total} contratos\n\n"
        f"_5 labs activos · showcase público congelado_"
    )
    st.caption("Este companion solo conserva exploración interactiva que no vale la pena duplicar en Quarto.")
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
