"""Simulación A/B: portafolio robusto vs no-robusto."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.express as px
import streamlit as st

from streamlit_app.components.decision_panels import decision_checklist
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    OFFICIAL_GPU_REPLAY_TAG,
    load_rapids_tradeoff_full_ab_status,
    load_rapids_tradeoff_full_policy,
    try_load_gpu_replay_json,
    try_load_json,
    try_load_parquet,
)


def _std_norm_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


st.title("🧪 Simulación A/B de Estrategias")
st.caption(
    "Comparación retroactiva de portafolio robusto vs no-robusto "
    "usando defaults reales del conjunto OOT como ground truth."
)
page_contract = get_page_contract("ab_testing_simulation")
render_page_header(page_contract)
render_key_takeaway(
    "La simulación A/B convierte la discusión de robustez en evidencia económica retrospectiva, separando intuición de resultado cuantificado."
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
ab_attribution_status = try_load_json("ab_attribution_status", directory="models", default={})
ab_grade_df = try_load_parquet("ab_attribution_by_grade")
ab_cohort_df = try_load_parquet("ab_attribution_by_cohort")
ab_amount_df = try_load_parquet("ab_attribution_by_amount")
gpu_status = try_load_gpu_replay_json(
    OFFICIAL_GPU_REPLAY_TAG, "artifacts/models/ab_simulation_status.json"
)
gpu_full_ab = load_rapids_tradeoff_full_ab_status()
gpu_full_policy = load_rapids_tradeoff_full_policy()

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
    ci_low = float(row.get("ci_low", 0.0))
    ci_high = float(row.get("ci_high", 0.0))
    ci_width = ci_high - ci_low
    se_boot = (ci_width / (2 * 1.96)) if ci_width > 0 else 0.0
    effect = float(row.get("diff", 0.0))
    z_alpha = 1.96
    if se_boot > 0:
        delta = abs(effect) / se_boot
        approx_power = float((1 - _std_norm_cdf(z_alpha - delta)) + _std_norm_cdf(-z_alpha - delta))
    else:
        approx_power = 0.0
    st.markdown(
        f"- **CI 95% del uplift medio**: [{ci_low:,.2f}, {ci_high:,.2f}]\n"
        f"- **Loans funded A**: {row.get('n_funded_a', 0):,}\n"
        f"- **Loans funded B**: {row.get('n_funded_b', 0):,}\n"
        f"- **SE bootstrap aprox.**: {se_boot:,.2f}\n"
        f"- **Potencia aprox. (post-hoc)**: {approx_power:.1%}"
    )
    st.info(
        "Diferencia clave: el CI del uplift resume incertidumbre del efecto promedio A-B; "
        "no es un intervalo de predicción del retorno de un periodo futuro individual."
    )
    st.caption(
        "Riesgo de falsos positivos: si se prueban muchas variantes de política sin corrección "
        "por multiplicidad, la probabilidad de encontrar mejoras espurias aumenta."
    )

    if not summary.empty:
        st.subheader("Comparación detallada de métricas")
        st.dataframe(summary, width="stretch", hide_index=True)

        fig = px.bar(
            summary,
            x="metric",
            y=["control", "treatment"],
            barmode="group",
            title="Control (no-robusto) vs Treatment (robusto)",
            labels={"value": "Valor", "metric": "Métrica"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
        st.plotly_chart(fig, width="stretch")

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

if gpu_status:
    st.markdown("### Replay RAPIDS de la misma prueba A/B")
    st.markdown(
        """
Además del artefacto canónico CPU, ya existe un replay GPU equivalente sobre el mismo bloque de decisión.
Eso importa porque esta página dejó de ser solo un resultado económico; ahora también demuestra que la comparación robusto vs no robusto puede evaluarse rápido en GPU cuando el universo de candidatos crece.
"""
    )
    gpu_table = [
        {
            "Engine": "GPU replay (cuopt)",
            "Control total return": gpu_status.get("metrics_a", {}).get("total_return", 0.0),
            "Treatment total return": gpu_status.get("metrics_b", {}).get("total_return", 0.0),
            "Diff": gpu_status.get("no_regression", {}).get("diff_total_return", 0.0),
            "No regression": gpu_status.get("no_regression", {}).get("passed", False),
        }
    ]
    st.dataframe(gpu_table, width="stretch", hide_index=True)
    full_ab = gpu_full_ab.get("no_regression", {})
    full_selected = gpu_full_policy.get("selected_policy", {})
    if full_ab:
        st.markdown("#### Qué pasó cuando escalamos el A/B al universo full")
        st.dataframe(
            [
                {
                    "Universe": gpu_full_ab.get("n_candidates_used", 0),
                    "Control total return": gpu_full_ab.get("metrics_a", {}).get("total_return", 0.0),
                    "Treatment total return": gpu_full_ab.get("metrics_b", {}).get("total_return", 0.0),
                    "Diff": full_ab.get("diff_total_return", 0.0),
                    "No regression": full_ab.get("passed", False),
                    "Selected gamma": full_selected.get("gamma"),
                    "Policy mode": full_selected.get("policy_mode"),
                }
            ],
            width="stretch",
            hide_index=True,
        )
        st.info(
            "La lectura más valiosa del replay full no es solo el tiempo. Es que, al escalar el universo, la policy seleccionada converge a una versión casi no robusta. Eso indica que el reto restante del A/B es de diseño de policy selection, no de solver ni de hardware."
        )

if ab_attribution_status:
    sharpe = ab_attribution_status.get("sharpe_metrics", {})
    with st.expander("Sharpe-like ratio y atribución por grade/cohort/monto", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bootstrap replications", f"{sharpe.get('n_bootstrap', 0):,}")
        c2.metric("ROIC estrategia A", f"{float(sharpe.get('roic_a', 0)):.2%}")
        c3.metric("ROIC estrategia B", f"{float(sharpe.get('roic_b', 0)):.2%}")
        c4.metric("Sharpe-like diff", f"{float(sharpe.get('sharpe_like_diff', 0)):.3f}")
        st.caption(
            f"Calmar-like A: {float(sharpe.get('calmar_like_a', 0)):.2f} | "
            f"Calmar-like B: {float(sharpe.get('calmar_like_b', 0)):.2f} | "
            f"{sharpe.get('note', '')}"
        )
        if not ab_grade_df.empty:
            st.markdown("**Atribución por grade (capital asignado + retorno real vs esperado)**")
            st.dataframe(ab_grade_df, hide_index=True, width="stretch")
        if not ab_cohort_df.empty:
            st.markdown("**Atribución por cohorte temporal (segmento OOT)**")
            st.dataframe(ab_cohort_df, hide_index=True, width="stretch")
        if not ab_amount_df.empty:
            st.markdown("**Atribución por bucket de monto**")
            st.dataframe(ab_amount_df, hide_index=True, width="stretch")
        st.caption(
            "Atribución calculada sobre portfolio_allocations canónico + conformal_intervals_mondrian (OOT test set). "
            "allocated_k = capital asignado en miles $. actual_return_k = retorno con defaults realizados."
        )

decision_checklist(
    "Checklist de lectura A/B",
    [
        "Revisar p-value e intervalo de confianza antes de concluir superioridad.",
        "Diferenciar significancia estadística de materialidad económica.",
        "Recordar que la comparación es retrospectiva (no experimento online).",
    ],
)
render_caveats(
    [
        "Resultados sensibles al periodo OOT y supuestos de simulación de estrategia.",
        "CI bootstrap y potencia post-hoc no sustituyen un experimento online aleatorizado con protocolo pre-registrado.",
        "No sustituye un rollout controlado real con monitoreo de comportamiento en producción.",
    ]
)
render_page_feedback("ab_testing_simulation")

next_page_teaser(
    "Provisiones IFRS9",
    "Cálculo de ECL bajo escenarios regulatorios con incertidumbre conformal.",
    "pages/ifrs9_provisions.py",
)
