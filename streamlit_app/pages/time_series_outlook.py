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
        lo = pd.to_numeric(frame["lo_90"], errors="coerce")
        hi = pd.to_numeric(frame["hi_90"], errors="coerce")
        if lo.isna().all() or hi.isna().all():
            # Interval bounds are all None/NaN — no conformal intervals in this artifact
            frame["covered_90"] = np.nan
        else:
            frame["covered_90"] = (
                pd.to_numeric(frame["y_true"], errors="coerce").between(lo, hi, inclusive="both")
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
forecastability_status = try_load_json("time_series_forecastability_status", directory="models", default={})
hierarchy_status = try_load_json("time_series_hierarchy_status", directory="models", default={})
research_status = try_load_json("time_series_research_status", directory="models", default={})

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
final_interval_decision = status.get("final_interval_decision", {})
_interval_research_only = str(final_interval_decision.get("status", "")).lower() == "research_only"

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

if _interval_research_only:
    _gap = interval_champion.get("coverage_gap_90")
    _gap_str = f"{_gap:.3f}" if _gap is not None else "n/d"
    st.info(
        f"⚠️ **Intervalos (Research Only):** el champion intervalar `{interval_model}` no supera el gate de cobertura "
        f"(gap={_gap_str}, target≤0.03). Los intervalos pueden usarse como diagnóstico pero NO son promocionables. "
        "Resolución pendiente: ACI/TCP rolling window (ver backlog P3.2)."
    )

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
_no_coverage = horizon_quality.empty or horizon_quality["coverage_90"].isna().all()
if horizon_quality.empty:
    st.info("Cobertura por horizonte no disponible: faltan predicciones reales de backtest.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        if _no_coverage:
            # Interval bounds were None in predictions — show per-model coverage from metrics table
            if not backtest_metrics.empty and {"model", "coverage_90"}.issubset(backtest_metrics.columns):
                _cov_row = backtest_metrics[backtest_metrics["model"] == interval_model]
                _cov_val = float(_cov_row["coverage_90"].iloc[0]) if not _cov_row.empty else None
                if _cov_val is not None:
                    st.metric(f"Cobertura 90% global ({interval_model})", f"{_cov_val:.1%}")
                    st.caption("Cobertura promedio sobre todos los horizontes; desglose por horizonte no disponible porque los bounds del artifact de predicciones están vacíos.")
                else:
                    st.info("Cobertura por horizonte no disponible: intervalos sin datos en el artifact de predicciones.")
            else:
                st.info("Cobertura por horizonte no disponible: intervalos sin datos en el artifact de predicciones.")
        else:
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

_cv_img = get_notebook_image_path("05_time_series_forecasting", "cv_model_comparison.png")
_fan_img = get_notebook_image_path("05_time_series_forecasting", "ifrs9_scenario_fan_chart.png")
_ts_overview_img = get_notebook_image_path("05_time_series_forecasting", "portfolio_time_series_overview.png")
_cell_img = get_notebook_image_path("05_time_series_forecasting", "cell_017_out_01.png")

_nb_ts_imgs = [(_cv_img, "NB05: comparación cross-validation por modelo (AutoARIMA, ETS, LightGBM, Theta)."),
               (_fan_img, "NB05: fan chart de escenarios IFRS9 — base / severo / recuperación."),
               (_ts_overview_img, "NB05: series temporales del portafolio — volumen, tasa de default y montos.")]
_nb_ts_valid = [(p, c) for p, c in _nb_ts_imgs if p.exists()]
if not _nb_ts_valid and _cell_img.exists():
    _nb_ts_valid = [(_cell_img, "Notebook 05: evidencia exploratoria del subsistema temporal.")]

if _nb_ts_valid:
    with st.expander("Figuras del notebook de pronóstico temporal", expanded=True):
        _ts_cols = st.columns(min(len(_nb_ts_valid), 3))
        for _ci, (_p, _cap) in enumerate(_nb_ts_valid):
            with _ts_cols[_ci % 3]:
                st.image(str(_p), caption=_cap, width="stretch")

with st.expander("Forecastabilidad: rutas por serie y tipo de intermitencia", expanded=False):
    if forecastability_status:
        c1, c2 = st.columns(2)
        c1.metric("Series evaluadas", forecastability_status.get("series_evaluated", "N/A"))
        available = forecastability_status.get("available", False)
        c2.metric("Disponible", "Sí" if available else "No")
        routes = forecastability_status.get("routes", {})
        if routes:
            routes_df = pd.DataFrame(
                [{"Ruta": k, "N series": v} for k, v in routes.items()]
            )
            st.markdown("**Distribución de series por ruta de modelado**")
            st.dataframe(routes_df, hide_index=True, width="stretch")
        intermittency = forecastability_status.get("intermittency", {})
        levels = forecastability_status.get("levels", {})
        if intermittency or levels:
            col_a, col_b = st.columns(2)
            if intermittency:
                with col_a:
                    st.markdown("**Tipo de intermitencia**")
                    st.dataframe(
                        pd.DataFrame([{"Tipo": k, "N": v} for k, v in intermittency.items()]),
                        hide_index=True,
                    )
            if levels:
                with col_b:
                    st.markdown("**Niveles de jerarquía**")
                    st.dataframe(
                        pd.DataFrame([{"Nivel": k, "N": v} for k, v in levels.items()]),
                        hide_index=True,
                    )
        st.caption(
            "Intermitente/errático = series con muchos ceros o alta varianza; "
            "requiere métodos especializados (ADIDA, IMAPA) o retroceso a nivel portafolio."
        )
    else:
        st.info("No hay `models/time_series_forecastability_status.json` disponible.")

with st.expander("Jerarquía temporal: estado de reconciliación", expanded=False):
    if hierarchy_status:
        available_h = hierarchy_status.get("available", False)
        reason = hierarchy_status.get("reason", "")
        if available_h:
            st.success("Reconciliación jerárquica habilitada.")
        else:
            st.warning(f"Reconciliación jerárquica deshabilitada. Razón: `{reason}`")
        st.caption(
            "La reconciliación jerárquica alinea pronósticos grade-term → grade → portafolio. "
            "Cuando está deshabilitada, los pronósticos por subgrupo no suman al total."
        )
    else:
        st.info("No hay `models/time_series_hierarchy_status.json` disponible.")

with st.expander("Research status: estado completo del módulo de pronóstico", expanded=False):
    if research_status:
        rs_status = research_status.get("status", "N/A")
        warnings = research_status.get("warnings", [])
        summary_rs = research_status.get("summary", {})
        point_champ = research_status.get("point_champion", {})
        interval_champ = research_status.get("interval_champion", {})
        c1, c2, c3 = st.columns(3)
        color = "normal" if rs_status == "ok" else "inverse"
        c1.metric("Estado", rs_status.upper())
        c2.metric("Modelos evaluados", summary_rs.get("n_models_evaluated", "N/A"))
        c3.metric("Backtest rows", f"{int(summary_rs.get('n_backtest_rows', 0)):,}" if summary_rs.get('n_backtest_rows') else "N/A")
        if warnings:
            for w in warnings:
                st.warning(f"⚠ {w}")
        if summary_rs:
            st.markdown(
                f"**Punto:** {summary_rs.get('point_model', 'N/A')} "
                f"({'promotable' if summary_rs.get('point_promotable') else 'no promotable'}) | "
                f"**Intervalo:** {summary_rs.get('interval_model', 'N/A')} "
                f"({'promotable' if summary_rs.get('interval_promotable') else 'no promotable'})"
            )
        cfg = research_status.get("config", {})
        if cfg:
            st.caption(
                f"Horizon: {cfg.get('horizon', 'N/A')}m | Freq: {cfg.get('freq', 'N/A')} | "
                f"Season: {cfg.get('season_length', 'N/A')} | Exogenous: {cfg.get('exogenous_enabled', False)}"
            )
    else:
        st.info("No hay `models/time_series_research_status.json` disponible.")

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
