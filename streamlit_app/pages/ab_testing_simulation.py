"""Simulación A/B: portafolio robusto vs no-robusto."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.express as px
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import try_load_json, try_load_parquet

st.title("🧪 Simulación A/B de Estrategias")
st.caption(
    "Comparación retroactiva de portafolio robusto vs no-robusto "
    "usando defaults reales del conjunto OOT como ground truth."
)
storytelling_intro(
    page_goal=(
        "Validar empíricamente si la estrategia robusta (conformal) supera a la no-robusta en retorno realizado."
    ),
    business_value=(
        "Traduce la teoría de robustez en evidencia económica medible: ¿cuánto vale la incertidumbre cuantificada?"
    ),
    key_decision=(
        "Determinar si adoptar la estrategia robusta genera valor estadísticamente significativo."
    ),
    how_to_read=[
        "Compara retorno total de cada estrategia y verifica significancia estadística.",
        "Revisa el intervalo de confianza: si no cruza cero, la diferencia es confiable.",
        "Analiza la tabla de métricas para entender el trade-off funding vs pérdida.",
    ],
)

results = try_load_parquet("ab_simulation_results")
summary = try_load_parquet("ab_simulation_summary")
status = try_load_json("ab_simulation_status", directory="models", default={})

if not results.empty:
    row = results.iloc[0]
    comparison = status.get("comparison", {})

    kpi_row(
        [
            {"label": "Retorno A (no-robusto)", "value": f"${row.get('strategy_a_return', 0):,.0f}"},
            {"label": "Retorno B (robusto)", "value": f"${row.get('strategy_b_return', 0):,.0f}"},
            {
                "label": "Diferencia",
                "value": f"${row.get('diff', 0):+,.0f}",
            },
            {
                "label": "p-value",
                "value": f"{row.get('p_value', 1.0):.4f}",
            },
        ],
        n_cols=4,
    )

    sig = row.get("significant", False)
    if sig:
        st.success("La diferencia es estadísticamente significativa (bootstrap, α=0.05).")
    else:
        st.warning("La diferencia NO es estadísticamente significativa.")

    st.subheader("Intervalo de confianza (Bootstrap)")
    st.markdown(
        f"- **CI 95%**: [{row.get('ci_low', 0):,.2f}, {row.get('ci_high', 0):,.2f}]\n"
        f"- **Loans funded A**: {row.get('n_funded_a', 0):,}\n"
        f"- **Loans funded B**: {row.get('n_funded_b', 0):,}"
    )

    if not summary.empty:
        st.subheader("Comparación detallada de métricas")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        fig = px.bar(
            summary,
            x="metric",
            y=["control", "treatment"],
            barmode="group",
            title="Control (no-robusto) vs Treatment (robusto)",
            labels={"value": "Valor", "metric": "Métrica"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**Interpretación:** esta simulación aplica dos estrategias de asignación al mismo conjunto OOT (2018-2020)
usando los defaults reales como ground truth. La estrategia robusta usa `pd_high` (conformal upper bound)
para la restricción de PD del portafolio, mientras que la no-robusta usa `pd_point`.

La comparación es retrospectiva (no un experimento aleatorizado en producción), pero proporciona
evidencia empírica sobre el valor económico de la incertidumbre cuantificada.
"""
    )
else:
    st.info(
        "Ejecuta `scripts/simulate_ab_test.py` para generar la simulación A/B."
    )

next_page_teaser(
    "Provisiones IFRS9",
    "Cálculo de ECL bajo escenarios regulatorios con incertidumbre conformal.",
    "pages/ifrs9_provisions.py",
)
