"""Panorama temporal: pronóstico gobernado de defaults y escenarios."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    get_notebook_image_path,
    try_load_json,
    try_load_parquet,
)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _format_pct(value: object) -> str:
    parsed = _safe_float(value)
    return f"{parsed:.1%}" if parsed is not None else "n/d"


def _format_num(value: object, decimals: int = 3) -> str:
    parsed = _safe_float(value)
    return f"{parsed:.{decimals}f}" if parsed is not None else "n/d"


def _first_text(frame: pd.DataFrame, column: str, default: str = "n/d") -> str:
    if column not in frame.columns or frame.empty:
        return default
    values = frame[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else default


def _build_horizon_quality(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    if predictions.empty or "model" not in predictions.columns:
        return pd.DataFrame()
    frame = predictions.loc[predictions["model"] == model].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["abs_error"] = (
        pd.to_numeric(frame["y_true"], errors="coerce") - pd.to_numeric(frame["y_pred"], errors="coerce")
    ).abs()
    if {"lo_90", "hi_90"}.issubset(frame.columns):
        frame["covered_90"] = (
            pd.to_numeric(frame["y_true"], errors="coerce").between(
                pd.to_numeric(frame["lo_90"], errors="coerce"),
                pd.to_numeric(frame["hi_90"], errors="coerce"),
                inclusive="both",
            )
        ).astype(float)
    else:
        frame["covered_90"] = np.nan

    out = (
        frame.groupby("horizon_step", as_index=False)
        .agg(
            n_obs=("ds", "size"),
            mae=("abs_error", "mean"),
            coverage_90=("covered_90", "mean"),
        )
        .sort_values("horizon_step")
        .reset_index(drop=True)
    )
    return out


def _build_diagnostics_table(
    diagnostics: dict,
    summary: dict,
    point_champion: dict,
    interval_champion: dict,
) -> pd.DataFrame:
    rows = [
        {"Diagnóstico": "ADF p-value", "Valor": _format_num(diagnostics.get("adf_pvalue"), 4)},
        {"Diagnóstico": "KPSS p-value", "Valor": _format_num(diagnostics.get("kpss_pvalue"), 4)},
        {
            "Diagnóstico": "Fuerza estacional",
            "Valor": _format_num(diagnostics.get("seasonal_strength"), 3),
        },
        {
            "Diagnóstico": "Variance ratio",
            "Valor": _format_num(diagnostics.get("variance_ratio"), 3),
        },
        {
            "Diagnóstico": "Entropía espectral",
            "Valor": _format_num(diagnostics.get("spectral_entropy"), 3),
        },
        {
            "Diagnóstico": "Permutation entropy",
            "Valor": _format_num(diagnostics.get("permutation_entropy"), 3),
        },
        {
            "Diagnóstico": "Promedio real reciente (12m)",
            "Valor": _format_pct(summary.get("recent_actual_mean_12m")),
        },
        {
            "Diagnóstico": "Champion puntual",
            "Valor": str(point_champion.get("model", "n/d")),
        },
        {
            "Diagnóstico": "Champion intervalar",
            "Valor": str(interval_champion.get("model", "n/d")),
        },
    ]
    return pd.DataFrame(rows)


st.title("📈 Panorama Temporal")
st.caption(
    "Pronóstico mensual de default con rolling-origin real, selección de champion y escenarios IFRS9."
)
page_contract = get_page_contract("time_series_outlook")
render_page_header(page_contract)
render_key_takeaway(
    "El bloque temporal aporta contexto de régimen para planeación e IFRS9, pero solo es oficial cuando existe evidencia artefact-driven de backtest y cobertura."
)
storytelling_intro(
    page_goal=(
        "Anticipar cómo puede evolucionar el default agregado del portafolio en los próximos 12 meses."
    ),
    business_value=(
        "Permite ajustar provisiones, apetito de riesgo y planeación antes de que el deterioro aparezca en defaults realizados."
    ),
    key_decision=(
        "Definir si el forecast es oficial para IFRS9/planeación o solo exploratorio por brechas de cobertura o evidencia."
    ),
    how_to_read=[
        "Empieza por el badge de estado: oficial o exploratorio.",
        "Luego revisa histórico vs forecast canónico y la tabla de backtest real.",
        "Cierra con cobertura por horizonte y escenarios IFRS9 derivados.",
    ],
)

history = try_load_parquet("time_series_full")
if history.empty:
    history = try_load_parquet("time_series")
forecasts = try_load_parquet("ts_forecasts")
scenarios = try_load_parquet("ts_ifrs9_scenarios")
backtest_predictions = try_load_parquet("ts_backtest_predictions")
if backtest_predictions.empty:
    backtest_predictions = try_load_parquet("ts_cv_stats")
backtest_metrics = try_load_parquet("ts_backtest_metrics")
panel_forecasts = try_load_parquet("ts_panel_forecasts")
status = try_load_json("time_series_status", directory="models", default={})
diagnostics = try_load_json("ts_diagnostics", default={})

if scenarios.empty and not forecasts.empty and {"ds", "y", "y_lo_90", "y_hi_90", "y_lo_95", "y_hi_95"}.issubset(
    forecasts.columns
):
    scenarios = forecasts[
        ["ds", "y", "y_lo_90", "y_hi_90", "y_lo_95", "y_hi_95", "point_model", "interval_model", "official_status"]
    ].rename(
        columns={
            "ds": "month",
            "y": "point_forecast",
            "y_lo_90": "optimistic_90",
            "y_hi_90": "adverse_90",
            "y_lo_95": "optimistic_95",
            "y_hi_95": "adverse_95",
        }
    )

summary = status.get("summary", {})
point_champion = status.get("point_champion", {})
interval_champion = status.get("interval_champion", {})
warnings = status.get("warnings", [])
status_level = str(status.get("status", "warn"))
point_model = str(point_champion.get("model") or _first_text(forecasts, "point_model"))
interval_model = str(interval_champion.get("model") or _first_text(forecasts, "interval_model"))
official_status = _first_text(forecasts, "official_status", default="desconocido")

if status_level == "pass":
    st.success(
        f"Estado `{official_status}`. Champion puntual: `{point_model}`. Champion intervalar: `{interval_model}`."
    )
elif status_level == "warn":
    st.warning(
        f"Estado `{official_status}` con observaciones. Champion puntual: `{point_model}`. Champion intervalar: `{interval_model}`."
    )
else:
    st.error(
        f"Estado `{official_status}` no promocionable. El forecast puede usarse como diagnóstico, no como baseline oficial."
    )

if warnings:
    st.caption("Warnings del status temporal: " + ", ".join(str(item) for item in warnings))

metric_cols = st.columns(4)
metric_cols[0].metric("Estado", official_status.capitalize())
metric_cols[1].metric("Champion puntual", point_model)
metric_cols[2].metric("MASE champion", _format_num(point_champion.get("mase"), 3))
metric_cols[3].metric("Coverage 90% champion", _format_pct(interval_champion.get("coverage_90")))

st.markdown(
    """
Este bloque ya no usa placeholders de validación. La lectura oficial se apoya en:
- backtest rolling-origin real guardado por corte y horizonte;
- selección explícita de champion puntual e intervalar;
- reconciliación bottom-up sobre conteos aditivos antes de derivar tasas.
"""
)

st.subheader("1) Histórico y forecast canónico")
if history.empty or forecasts.empty or "y" not in forecasts.columns:
    st.info("Forecast canónico no disponible: faltan `time_series_full.parquet` o `ts_forecasts.parquet`.")
else:
    fig = go.Figure()
    if "y" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history["ds"],
                y=history["y"],
                mode="lines",
                name="Histórico",
                line={"color": "#334155", "width": 2.5},
            )
        )

    for level, color in [(95, "rgba(255,217,61,0.16)"), (90, "rgba(0,212,170,0.22)")]:
        lo = f"y_lo_{level}"
        hi = f"y_hi_{level}"
        if lo in forecasts.columns and hi in forecasts.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecasts["ds"],
                    y=forecasts[hi],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecasts["ds"],
                    y=forecasts[lo],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=color,
                    name=f"Intervalo {level}%",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["y"],
            mode="lines+markers",
            name="Forecast canónico",
            line={"color": "#0B5ED7", "width": 2.6},
            marker={"size": 6, "color": "#0B5ED7"},
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Tasa de default: histórico + forecast oficial/diagnóstico",
        xaxis_title="Fecha",
        yaxis_title="Tasa de default",
        height=470,
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Punto canónico desde `{point_model}` y bandas desde `{interval_model}`. "
        f"Estado operativo: `{official_status}`."
    )

st.subheader("2) Champion rationale y backtest real")
if backtest_metrics.empty:
    st.info("Backtest rolling-origin no disponible. La promoción del forecast queda en estado `no disponible`.")
else:
    candidate_table = backtest_metrics.copy()
    candidate_table["rol"] = np.where(
        candidate_table["model"].eq(point_model),
        "champion_point",
        np.where(candidate_table["model"].eq(interval_model), "champion_interval", "challenger"),
    )
    candidate_table["coverage_gap_90"] = pd.to_numeric(
        candidate_table.get("coverage_gap_90"), errors="coerce"
    )
    candidate_table["mase"] = pd.to_numeric(candidate_table.get("mase"), errors="coerce")
    candidate_table["coverage_90"] = pd.to_numeric(candidate_table.get("coverage_90"), errors="coerce")
    candidate_table["fva_mae_pct"] = pd.to_numeric(
        candidate_table.get("fva_mae_pct"), errors="coerce"
    )
    candidate_table = candidate_table.sort_values(["mase", "rmsse", "abs_bias"]).reset_index(
        drop=True
    )

    display_cols = [
        "rol",
        "model",
        "family",
        "mase",
        "rmsse",
        "abs_bias",
        "fva_mae_pct",
        "coverage_90",
        "coverage_gap_90",
        "winkler_90",
        "avg_interval_width_90",
        "mean_abs_revision",
        "dm_pvalue_vs_seasonal_naive",
    ]
    available_cols = [col for col in display_cols if col in candidate_table.columns]
    st.dataframe(candidate_table[available_cols], width="stretch", hide_index=True)

    fig = px.scatter(
        candidate_table,
        x="mase",
        y="coverage_gap_90",
        size="avg_interval_width_90",
        color="family",
        symbol="rol",
        hover_name="model",
        title="Trade-off entre error, cobertura y ancho intervalar",
        labels={"mase": "MASE", "coverage_gap_90": "|coverage_90 - 0.90|"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "El champion puntual se elige por MASE con guardrail contra SeasonalNaive; "
        "el champion intervalar requiere gap de cobertura 90% dentro de política."
    )

st.subheader("3) Cobertura y error por horizonte")
horizon_quality = _build_horizon_quality(backtest_predictions, interval_model)
if horizon_quality.empty:
    st.info("Cobertura por horizonte no disponible: faltan predicciones reales de backtest.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.line(
            horizon_quality,
            x="horizon_step",
            y="coverage_90",
            markers=True,
            title=f"Cobertura 90% por horizonte ({interval_model})",
            labels={"horizon_step": "Paso", "coverage_90": "Coverage"},
        )
        fig.add_hline(y=0.90, line_dash="dash", line_color="#B91C1C")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        point_horizon = _build_horizon_quality(backtest_predictions, point_model)
        fig = px.bar(
            point_horizon,
            x="horizon_step",
            y="mae",
            title=f"MAE por horizonte ({point_model})",
            labels={"horizon_step": "Paso", "mae": "MAE"},
            color="mae",
            color_continuous_scale="Blues",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], coloraxis_showscale=False, height=360)
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Esta vista permite distinguir un modelo bueno en promedio de uno estable a lo largo del horizonte."
    )

st.subheader("4) Reconciliación coherente por grade")
if "series_level" in panel_forecasts.columns:
    grade_panel = panel_forecasts.loc[panel_forecasts["series_level"] == "grade"].copy()
else:
    grade_panel = pd.DataFrame()
if grade_panel.empty:
    st.info("Pronóstico jerárquico por `grade` no disponible; la página degrada esta sección a `no disponible`.")
else:
    fig = px.line(
        grade_panel,
        x="ds",
        y="default_rate",
        color="grade",
        markers=True,
        title="Bottom-up forecast por grade (tasas derivadas tras reconciliación)",
        labels={"ds": "Fecha", "default_rate": "Tasa de default"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "La reconciliación se hace sobre `default_count` y `loan_count`; la tasa se deriva después para evitar inconsistencias aditivas."
    )

st.subheader("5) Escenarios IFRS9 derivados")
if scenarios.empty:
    st.info("Escenarios IFRS9 temporales no disponibles.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scenarios["month"],
                y=scenarios["point_forecast"],
                mode="lines+markers",
                name="Punto",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=scenarios["month"],
                y=scenarios["adverse_90"],
                mode="lines",
                name="Adverso 90%",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=scenarios["month"],
                y=scenarios["optimistic_90"],
                mode="lines",
                name="Optimista 90%",
            )
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Escenario central vs bandas 90%",
            height=390,
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        scen_long = scenarios.melt(
            id_vars=["month"],
            value_vars=[
                "optimistic_90",
                "point_forecast",
                "adverse_90",
                "optimistic_95",
                "adverse_95",
            ],
            var_name="escenario",
            value_name="tasa",
        )
        fig = px.box(
            scen_long,
            x="escenario",
            y="tasa",
            points="all",
            title="Dispersión de tasas por escenario",
            labels={"escenario": "", "tasa": "Tasa"},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Estos escenarios ya salen del artifact canónico del forecast temporal y se consumen downstream en IFRS9."
    )

with st.expander("Diagnóstico temporal y backlog metodológico"):
    if diagnostics or summary:
        diag_table = _build_diagnostics_table(
            diagnostics=diagnostics,
            summary=summary,
            point_champion=point_champion,
            interval_champion=interval_champion,
        )
        st.dataframe(diag_table, width="stretch", hide_index=True)
    else:
        st.info("Diagnóstico temporal no disponible.")

    backlog = status.get("research_backlog", [])
    backlog_text = ", ".join(str(item) for item in backlog) if backlog else "ACI, EnbPI"
    st.markdown(
        f"""
**Nota metodológica**

- Las bandas oficiales priorizan el champion intervalar que cumple la política de cobertura.
- Si el modelo ML no cumple cobertura, sus bandas se consideran diagnósticas y no oficiales.
- Agenda explícita de research para no-exchangeability: `{backlog_text}`.
"""
    )

img = get_notebook_image_path("05_time_series_forecasting", "cell_017_out_01.png")
if img.exists():
    st.image(
        str(img),
        caption="Notebook 05: evidencia exploratoria previa a la versión gobernada del subsistema temporal.",
        width="stretch",
    )

render_caveats(
    [
        "Un forecast puntual útil no basta: la promoción oficial exige evidencia de backtest y cobertura intervalar.",
        "Las covariables exógenas solo deben comunicarse cuando exista contrato explícito de futuro; en otro caso la lectura oficial es univariada.",
    ]
)
render_page_feedback("time_series_outlook")

next_page_teaser(
    "Análisis de Supervivencia",
    "De la pregunta '¿quién incumple?' a '¿cuándo incumple?' para riesgo de horizonte.",
    "pages/survival_analysis.py",
)
