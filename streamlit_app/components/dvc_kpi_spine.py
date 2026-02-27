"""Reusable DVC-backed KPI spine for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass

import plotly.express as px
import streamlit as st

from streamlit_app.components.context_help import methodology_dialog, metric_help_popover
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    format_pct,
    load_dvc_conformal_backtest,
    load_dvc_metrics_summary,
    load_dvc_robustness_frontier,
    safe_metric_get,
)


@dataclass(frozen=True)
class DvcKpiSpec:
    key: str
    label: str
    format: str
    directionality: str
    business_interpretation: str
    thresholds: tuple[float, float] | None = None


DVC_KPI_SPECS: dict[str, DvcKpiSpec] = {
    "pd.auc": DvcKpiSpec("pd.auc", "AUC", "float4", "higher_better", "Mejor ranking de riesgo"),
    "pd.gini": DvcKpiSpec("pd.gini", "Gini", "float4", "higher_better", "Mayor poder discriminante"),
    "pd.ks": DvcKpiSpec("pd.ks", "KS", "float4", "higher_better", "Mayor separación útil para cutoffs"),
    "pd.brier": DvcKpiSpec("pd.brier", "Brier", "float4", "lower_better", "Menor error probabilístico"),
    "pd.ece": DvcKpiSpec("pd.ece", "ECE", "float4", "lower_better", "Mejor calibración"),
    "conformal.coverage90": DvcKpiSpec(
        "conformal.coverage90", "Coverage 90%", "pct1", "contextual", "Cobertura cerca del objetivo 90%"
    ),
    "conformal.coverage95": DvcKpiSpec(
        "conformal.coverage95", "Coverage 95%", "pct1", "contextual", "Cobertura cerca del objetivo 95%"
    ),
    "conformal.avg_width90": DvcKpiSpec(
        "conformal.avg_width90", "Ancho promedio 90%", "float3", "contextual", "Costo de conservadurismo"
    ),
    "ifrs9.ecl_baseline": DvcKpiSpec(
        "ifrs9.ecl_baseline", "ECL baseline", "money_short", "contextual", "Provisión contable base"
    ),
    "ifrs9.ecl_severe": DvcKpiSpec(
        "ifrs9.ecl_severe", "ECL severe", "money_short", "contextual", "Provisión bajo escenario severo"
    ),
    "ifrs9.severe_uplift_pct": DvcKpiSpec(
        "ifrs9.severe_uplift_pct", "Uplift severe", "pct1_from_percent", "contextual", "Sensibilidad de provisión"
    ),
    "optimization.robust_return": DvcKpiSpec(
        "optimization.robust_return", "Retorno robusto", "money_short", "higher_better", "Retorno protegido"
    ),
    "optimization.nonrobust_return": DvcKpiSpec(
        "optimization.nonrobust_return", "Retorno no robusto", "money_short", "higher_better", "Retorno optimista"
    ),
    "optimization.price_of_robustness": DvcKpiSpec(
        "optimization.price_of_robustness",
        "Price of Robustness",
        "money_short",
        "contextual",
        "Costo por proteger downside",
    ),
}


CONTEXT_KPI_KEYS: dict[str, list[str]] = {
    "executive": [
        "pd.auc",
        "pd.ece",
        "conformal.coverage90",
        "conformal.avg_width90",
        "optimization.price_of_robustness",
        "ifrs9.ecl_severe",
    ],
    "model": ["pd.auc", "pd.gini", "pd.ks", "pd.brier", "pd.ece"],
    "uncertainty": [
        "conformal.coverage90",
        "conformal.coverage95",
        "conformal.avg_width90",
    ],
    "ifrs9": ["ifrs9.ecl_baseline", "ifrs9.ecl_severe", "ifrs9.severe_uplift_pct"],
    "optimization": [
        "optimization.robust_return",
        "optimization.nonrobust_return",
        "optimization.price_of_robustness",
    ],
    "governance": ["pd.ece", "conformal.coverage90", "conformal.coverage95"],
}


def get_context_metric_keys(context: str) -> list[str]:
    """Return metric keys configured for a page context."""
    return CONTEXT_KPI_KEYS.get(context, CONTEXT_KPI_KEYS["executive"])


def _format_metric_value(value: float | None, spec: DvcKpiSpec) -> str:
    if value is None:
        return "N/D"
    if spec.format == "float4":
        return f"{value:.4f}"
    if spec.format == "float3":
        return f"{value:.3f}"
    if spec.format == "pct1":
        return format_pct(value, decimals=1)
    if spec.format == "pct1_from_percent":
        return f"{value:.1f}%"
    if spec.format == "money_short":
        return format_number(value, prefix="$")
    return str(value)


def build_metric_cards(metrics: dict[str, float], context: str) -> list[dict[str, str | None]]:
    """Build metric card payloads from DVC metrics for a context."""
    cards: list[dict[str, str | None]] = []
    for key in get_context_metric_keys(context):
        spec = DVC_KPI_SPECS[key]
        value = safe_metric_get(metrics, key)
        cards.append(
            {
                "label": spec.label,
                "value": _format_metric_value(value, spec),
                "help": spec.business_interpretation,
            }
        )
    return cards


def render_metric_delta_explainer(metric_key: str, current_value: float | None, baseline_ref: float | None = None) -> None:
    """Explain a metric in business terms, optionally with a baseline delta."""
    spec = DVC_KPI_SPECS.get(metric_key)
    if spec is None:
        return
    metric_help_popover(metric_key, label=f"{spec.label}: interpretación")
    if current_value is None:
        return
    if baseline_ref is None:
        st.caption(f"{spec.label}: {spec.business_interpretation}.")
        return
    delta = current_value - baseline_ref
    st.caption(
        f"{spec.label}: cambio de {delta:+.4f} vs referencia. {spec.business_interpretation}."
    )


def render_global_kpi_spine(context: str = "executive") -> None:
    """Render DVC-backed KPI cards for a page context."""
    metrics = load_dvc_metrics_summary()
    if not metrics:
        st.info(
            "No se encontró `reports/dvc/metrics_summary.json`. Ejecuta `dvc repro export_dvc_metrics` "
            "y luego `dvc pull`/`dvc push` según corresponda."
        )
        return
    st.markdown("### 📌 KPIs canónicos (DVC)")
    kpi_row(build_metric_cards(metrics, context), n_cols=3)


def render_conformal_health_panel() -> None:
    """Render DVC-backed conformal backtest summary and trend chart."""
    metrics = load_dvc_metrics_summary()
    backtest = load_dvc_conformal_backtest()
    if not metrics:
        st.info("KPIs conformal canónicos no disponibles en `reports/dvc/metrics_summary.json`.")
        return

    st.markdown("### 📐 Salud conformal (snapshot canónico DVC)")
    kpi_row(build_metric_cards(metrics, "uncertainty"), n_cols=3)
    metric_help_popover("coverage", label="¿Cómo interpretar cobertura y ancho?")

    if backtest.empty:
        st.caption("No se encontró `reports/dvc/conformal_coverage_backtest.csv`.")
        return

    fig = px.line(
        backtest,
        x="month",
        y=["coverage_90", "target_90", "coverage_95", "target_95"],
        title="Cobertura mensual vs targets (DVC canónico)",
        labels={"value": "Cobertura", "month": "Mes", "variable": "Serie"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    methodology_dialog(
        "Cómo leer el panel conformal",
        """
- `coverage_90` y `coverage_95` son coberturas empíricas observadas por mes.
- `target_90` y `target_95` son los niveles objetivo declarados por política.
- `avg_width_90` (en KPIs) resume el costo de conservadurismo.

Interpretación operativa:
- Cobertura consistentemente bajo target -> intervalos optimistas.
- Cobertura muy alta + ancho alto -> robustez cara (menos eficiencia).
""",
        button_label="Ver detalle de cobertura y ancho",
    )


def render_robustness_frontier_panel() -> None:
    """Render DVC-backed robustness frontier view."""
    metrics = load_dvc_metrics_summary()
    frontier = load_dvc_robustness_frontier()
    st.markdown("### 💼 Frontera de robustez (DVC canónico)")
    if metrics:
        kpi_row(build_metric_cards(metrics, "optimization"), n_cols=3)
        metric_help_popover("price_of_robustness", label="¿Qué es Price of Robustness?")
    if frontier.empty:
        st.caption("No se encontró `reports/dvc/robustness_frontier.csv`.")
        return

    fig = px.line(
        frontier.sort_values(["risk_tolerance", "uncertainty_aversion"]),
        x="worst_case_pd",
        y="expected_return_net_point",
        color="policy",
        line_dash="uncertainty_aversion",
        hover_data=["risk_tolerance", "n_funded", "price_of_robustness_pct"],
        title="Trade-off retorno vs riesgo peor caso (DVC canónico)",
        labels={
            "worst_case_pd": "PD peor caso (portafolio)",
            "expected_return_net_point": "Retorno esperado neto",
        },
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
    st.plotly_chart(fig, width="stretch")
    methodology_dialog(
        "Interpretación de la frontera robusta",
        """
Cada punto representa una política factible con restricciones de riesgo y presupuesto.

- Eje X: riesgo de portafolio en peor caso (más a la izquierda = más resiliente).
- Eje Y: retorno esperado neto (más arriba = mejor caso medio).
- `price_of_robustness_pct`: costo relativo de proteger downside frente a solución no robusta.

No hay punto "mejor" universal: depende del apetito de riesgo y del costo tolerable de robustez.
""",
        button_label="Ver cómo leer la frontera robusta",
    )
