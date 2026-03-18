"""Análisis de supervivencia aplicado a riesgo de crédito."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    get_notebook_image_path,
    try_load_json,
    try_load_parquet,
)

st.title("⏳ Análisis de Supervivencia")
st.caption(
    "Modelamos tiempo hasta incumplimiento para complementar la PD puntual "
    "con una visión temporal útil en IFRS9 y gestión de ciclo de vida."
)
page_contract = get_page_contract("survival_analysis")
render_page_header(page_contract)
render_key_takeaway(
    "La PD puntual responde 'quién'; supervivencia añade 'cuándo', que es crucial para provisión lifetime y priorización preventiva."
)
storytelling_intro(
    page_goal=(
        "Estimar no solo quién puede caer en default, sino cuándo es más probable que ocurra."
    ),
    business_value=(
        "Mejora provisiones lifetime y priorización de seguimiento preventivo en cartera."
    ),
    key_decision=(
        "Definir qué segmentos requieren monitoreo temprano y ajustes prudenciales por horizonte temporal."
    ),
    how_to_read=[
        "Empieza con la intuición de dos préstamos con igual PD pero distinto tiempo al default.",
        "Revisa C-index y hazard ratios para entender drivers temporales.",
        "Termina en curvas PD lifetime para conexión directa con IFRS9.",
    ],
)

summary = try_load_json("pipeline_summary")
survival = summary.get("survival", {})

with st.expander("¿Por qué importa el tiempo hasta default?"):
    st.markdown(
        f"""
### Dos préstamos, misma PD, impacto muy diferente

| | Préstamo A | Préstamo B |
|---|---|---|
| PD | 10% | 10% |
| Default en | Mes 3 | Mes 48 |
| Pagos recibidos | 3 cuotas | 48 cuotas |
| Pérdida real | Alta (casi todo el saldo) | Baja (ya pagó 80%) |

Un modelo de clasificación binaria (PD) dice que ambos tienen el mismo riesgo.
**Survival analysis** dice que el Préstamo A es mucho más peligroso porque hace
default temprano, cuando aún se ha recuperado poco del capital.

### Conexión con IFRS9

La PD **lifetime** (probabilidad acumulada de default a cualquier horizonte) es
el insumo directo para **Stage 2** de IFRS9. Sin survival analysis, no podemos
estimar cuánto provisionar para la vida completa del préstamo.

### Modelos usados en este proyecto

| Modelo | Tipo | C-index | Ventaja |
|--------|------|---------|---------|
| **Cox PH** | Semi-paramétrico | {survival.get("cox_concordance", 0):.4f} | Interpretable (hazard ratios) |
| **RSF** | Ensemble (Random Survival Forest) | {survival.get("rsf_concordance", 0):.4f} | Captura no-linealidades |

**C-index** = probabilidad de que el modelo ordene correctamente qué préstamo hace
default primero. 0.5 = aleatorio, 1.0 = perfecto.
"""
    )

st.markdown(
    """
Esta página responde una pregunta que la clasificación binaria no cubre por sí sola: no solo quién incumple, sino cuándo
es más probable que lo haga. Esa dimensión temporal es crítica para provisiones lifetime, seguimiento preventivo y diseño
de acciones de mitigación. En el proyecto, supervivencia actúa como puente entre modelado predictivo y lectura de horizonte
que exige IFRS9.
"""
)

km_df = try_load_parquet("km_curve_data")
hazard = try_load_parquet("hazard_ratios")
lifetime_pd = try_load_parquet("lifetime_pd_table")
if lifetime_pd.empty:
    loan_master = try_load_parquet("loan_master")
    if not loan_master.empty and {"grade", "default_flag"}.issubset(loan_master.columns):
        grade_pd = (
            loan_master.groupby("grade", observed=True)["default_flag"]
            .mean()
            .sort_index()
            .clip(lower=0.0001, upper=0.9999)
        )
        lifetime_pd = pd.DataFrame(
            {
                "Grade": grade_pd.index,
                "PD_12m": grade_pd.values,
                "PD_24m": 1.0 - (1.0 - grade_pd.values) ** 2,
                "PD_36m": 1.0 - (1.0 - grade_pd.values) ** 3,
                "PD_48m": 1.0 - (1.0 - grade_pd.values) ** 4,
                "PD_60m": 1.0 - (1.0 - grade_pd.values) ** 5,
            }
        )
    else:
        lifetime_pd = pd.DataFrame(
            columns=["Grade", "PD_12m", "PD_24m", "PD_36m", "PD_48m", "PD_60m"]
        )
elif "Grade" not in lifetime_pd.columns:
    lifetime_pd = lifetime_pd.reset_index().rename(columns={"index": "Grade"})

kpi_row(
    [
        {"label": "C-index Cox", "value": f"{survival.get('cox_concordance', 0):.4f}"},
        {"label": "C-index RSF", "value": f"{survival.get('rsf_concordance', 0):.4f}"},
        {"label": "Horizonte", "value": "60 meses"},
        {
            "label": "Evento observado",
            "value": f"{summary.get('dataset', {}).get('event_rate', 0.185) * 100:.1f}%",
        },
    ]
)

st.dataframe(
    pd.DataFrame(
        [
            {
                "Métrica": "C-index",
                "Interpretación técnica": "Probabilidad de ordenar correctamente qué préstamo cae antes en default.",
                "Interpretación negocio": "Mejor priorización temporal para acciones de seguimiento.",
            },
            {
                "Métrica": "Hazard ratio",
                "Interpretación técnica": "Multiplicador del riesgo instantáneo por unidad de cambio de una variable.",
                "Interpretación negocio": "Permite identificar palancas con impacto proporcional en deterioro.",
            },
            {
                "Métrica": "PD lifetime",
                "Interpretación técnica": "Probabilidad acumulada de default a distintos horizontes (12-60m).",
                "Interpretación negocio": "Insumo directo para provisión Stage 2 IFRS9.",
            },
        ]
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("1) Curvas Kaplan-Meier por grade")
grades = sorted(km_df["grade"].dropna().unique().tolist())
selected_grades = st.multiselect("Grades visibles", grades, default=grades)
filtered = km_df[km_df["grade"].isin(selected_grades)]

fig = go.Figure()
for grade in selected_grades:
    subset = filtered[filtered["grade"] == grade]
    fig.add_trace(
        go.Scatter(
            x=subset["timeline"],
            y=subset["survival_prob"],
            mode="lines",
            name=f"Grade {grade}",
        )
    )
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
)
fig.update_layout(
    title="Probabilidad de supervivencia del préstamo en el tiempo",
    xaxis_title="Meses desde originación",
    yaxis_title="S(t)",
    yaxis={"tickformat": ".0%"},
    height=470,
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Propósito: estimar probabilidad de no default a lo largo del tiempo. Insight: la separación entre curvas por grade "
    "muestra diferencias estructurales en velocidad de deterioro."
)

_km_img = get_notebook_image_path("06_survival_analysis", "kaplan_meier_curves.png")
if _km_img.exists():
    with st.expander("Figura del notebook: curvas Kaplan-Meier completas (NB06)", expanded=False):
        st.image(
            str(_km_img),
            caption="NB06: curvas KM completas por grade A–G con bandas de confianza — versión publicable del análisis de supervivencia.",
            width="stretch",
        )

st.info(
    "Insight: el gradiente entre A y G no solo cambia la probabilidad de default, "
    "también acelera el tiempo esperado al evento."
)

st.subheader("2) Hazard ratios (Cox PH)")
hazard_sorted = hazard.sort_values("exp_coef")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=hazard_sorted["feature"],
        x=hazard_sorted["exp_coef"],
        orientation="h",
        marker_color="#00D4AA",
        error_x={
            "type": "data",
            "symmetric": False,
            "array": (hazard_sorted["ci_upper"] - hazard_sorted["exp_coef"]).tolist(),
            "arrayminus": (hazard_sorted["exp_coef"] - hazard_sorted["ci_lower"]).tolist(),
        },
    )
)
fig.add_vline(x=1.0, line_dash="dash", line_color="#FF6B6B")
fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    title="Efecto multiplicativo sobre el hazard de default",
    xaxis_title="Hazard ratio exp(coef)",
    height=420,
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Propósito: cuantificar efecto multiplicativo de cada driver sobre el riesgo instantáneo. Insight: HR>1 acelera default; "
    "HR<1 protege temporalmente la cartera."
)

st.subheader("3) Curvas de PD de vida (IFRS9)")
if lifetime_pd.empty or "Grade" not in lifetime_pd.columns:
    st.info("No hay `lifetime_pd_table.parquet` disponible; se omite visual de PD lifetime.")
else:
    lt_long = lifetime_pd.melt(id_vars="Grade", var_name="horizonte", value_name="pd")
    lt_long["mes"] = lt_long["horizonte"].str.extract(r"(\d+)").astype(int)
    fig = px.line(
        lt_long,
        x="mes",
        y="pd",
        color="Grade",
        markers=True,
        title="Evolución de PD acumulada por grade",
        labels={"mes": "Meses", "pd": "PD acumulada"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(yaxis={"tickformat": ".0%"}, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Propósito: proyectar PD acumulada por horizonte y grade. Insight: curvas más inclinadas implican mayor presión de provisión "
        "lifetime en IFRS9."
    )

st.markdown(
    """
**Cómo complementa al resto del stack:**
- ML clasifica riesgo de default.
- Supervivencia agrega dimensión temporal para provisiones y seguimiento.
- Causalidad ayuda a identificar acciones que puedan modificar trayectorias de riesgo.
"""
)
st.markdown(
    """
La historia final de esta sección es que dos clientes con PD similar pueden tener perfiles de deterioro temporal muy distintos.
Ese matiz cambia prioridades de monitoreo, valor presente de pérdidas y la conversación de política de riesgo. Por eso esta
capa temporal se integra explícitamente con causalidad e IFRS9 en el flujo completo.
"""
)

with st.expander("Nota metodológica: supuesto de hazards proporcionales"):
    st.markdown(
        f"""
**Observación**: El test de residuos de Schoenfeld (NB06) identifica 8 variables cuyo efecto sobre el hazard
no es constante en el tiempo, incluyendo `int_rate`, `dti` y `term`. Esto indica que el supuesto de **hazards
proporcionales** del modelo Cox PH no se cumple estrictamente.

**Implicación**: Los hazard ratios reportados representan un efecto *promedio* a lo largo del tiempo, pero el
impacto real de estas variables puede variar entre períodos tempranos y tardíos. Las curvas de PD lifetime
para horizontes largos (>36 meses) pueden tener sesgo si la estructura de riesgo cambia significativamente.

**Mitigación en este proyecto**:
- El modelo **Random Survival Forest (RSF)** no requiere el supuesto PH y obtiene un C-index comparable
  ({survival.get('rsf_concordance', 0):.4f} vs {survival.get('cox_concordance', 0):.4f}),
  validando que las conclusiones son robustas a esta violación.
- Las curvas de PD lifetime se utilizan principalmente para IFRS9 Stage 2, donde el horizonte relevante
  (12-36 meses) es el menos afectado por la violación PH.

**Trabajo futuro**: Un Cox **estratificado por `term`** (2 niveles: 36 y 60 meses) permitiría que cada
estrato tenga su propia función de hazard base, relajando parcialmente el supuesto PH para la variable
más influyente.
"""
    )

st.subheader("4) Evidencia del Notebook: PD lifetime (RSF)")
col_rsf, col_rsf_text = st.columns([3, 2])
with col_rsf:
    img = get_notebook_image_path("06_survival_analysis", "cell_025_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 06: curvas de PD lifetime por grade (Random Survival Forest).",
        )
with col_rsf_text:
    st.markdown(
        """
**¿Qué muestra esta gráfica?**

Curvas de PD acumulada estimadas por el **Random Survival Forest** (RSF)
para cada grade (A-G) a lo largo de 60 meses.

**Resultados clave:**
- **Grade A**: PD lifetime ~5% a 60 meses — riesgo estable y bajo.
- **Grade G**: PD lifetime >40% a 60 meses — deterioro acelerado.
- La mayor parte de la divergencia ocurre en los primeros 24 meses.

**Insight de negocio:**
Los grades D-G concentran el riesgo temprano. Para IFRS9, esto significa
que la provisión lifetime (Stage 2) para estos segmentos debe reflejar
una curva de pérdida agresiva en los primeros 2 años.

**Conexión con el pipeline:**
Estas curvas alimentan la PD lifetime para Stage 2 de IFRS9 y se
complementan con la información causal (NB07) para diseñar acciones
de retención temprana en los segmentos de mayor pendiente.
"""
    )

with st.expander("5) Competing Risks — CIF (Default vs Prepayment)", expanded=False):
    cif_ts = try_load_parquet("competing_risks_cif_timeseries")
    cif_summary = try_load_parquet("competing_risks_cif")
    cif_status = try_load_json("competing_risks_status", directory="models", default={})

    if not cif_ts.empty:
        fig_cif = go.Figure()
        fig_cif.add_trace(go.Scatter(
            x=cif_ts["time_months"], y=cif_ts["cif_default"],
            mode="lines", name="CIF Default (Charged Off)", line={"color": "red"},
        ))
        fig_cif.add_trace(go.Scatter(
            x=cif_ts["time_months"], y=cif_ts["cif_prepay"],
            mode="lines", name="CIF Prepayment (Fully Paid)", line={"color": "green"},
        ))
        if "cif_default_lo" in cif_ts.columns:
            fig_cif.add_trace(go.Scatter(
                x=pd.concat([cif_ts["time_months"], cif_ts["time_months"].iloc[::-1]]),
                y=pd.concat([cif_ts["cif_default_hi"], cif_ts["cif_default_lo"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(255,0,0,0.1)", line={"color": "rgba(255,255,255,0)"},
                name="CIF Default 95% CI",
            ))
        fig_cif.update_layout(
            title="Cumulative Incidence Functions — Default vs Prepayment (Aalen-Johansen)",
            xaxis_title="Months since issue", yaxis_title="Cumulative Incidence",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_cif, use_container_width=True)

        if cif_status:
            km_bias = cif_status.get("km_bias_vs_cif", {})
            ov = cif_status.get("overall", {})
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("CIF Default @36m", f"{ov.get('cif_default_36m', 0):.2%}")
            mc2.metric("CIF Default @final", f"{ov.get('cif_default_final', 0):.2%}")
            mc3.metric("KM overestimate @60m", f"+{km_bias.get('km_overestimate_t60m', 0):.2%}")
            st.caption(
                "El KM sobreestima la PD de default porque ignora prepayment como riesgo competidor. "
                "CIF (Aalen-Johansen) corrige este sesgo usando la sub-distribution hazard de cada evento."
            )

        if not cif_summary.empty:
            st.markdown("**CIF por grade (incidencia a distintos horizontes):**")
            grade_cols = [c for c in cif_summary.columns if "cif_default" in c or c == "grade"]
            st.dataframe(cif_summary[grade_cols].head(8), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No se encontro `competing_risks_cif_timeseries.parquet`. "
            "Ejecute: `uv run python scripts/run_competing_risks.py`"
        )

with st.expander("6) Impacto CIF en ECL — ¿Cuánto sobreestima el KM las reservas IFRS9?", expanded=False):
    cif_ecl = try_load_parquet("cif_ecl_impact")
    cif_ecl_status = try_load_json("cif_ecl_impact_status", directory="models", default={})

    if cif_ecl_status:
        ov = cif_ecl_status.get("overall", {})
        ei = cif_ecl_status.get("ecl_impact", {})
        ie1, ie2, ie3, ie4 = st.columns(4)
        ie1.metric("KM PD@60m", f"{ov.get('km_t60m', 0):.2%}")
        ie2.metric("CIF PD@60m", f"{ov.get('cif_t60m', 0):.2%}")
        ie3.metric("Sesgo KM", f"+{ov.get('km_overestimate_t60m_pp', 0):.2f}pp")
        ie4.metric("Factor correccion (CF)", f"{ov.get('cf_lifetime', 0):.4f}")

        col_a, col_b, col_c = st.columns(3)
        total_km = ei.get("total_ecl_km") or 0
        total_cif = ei.get("total_ecl_cif_adjusted") or 0
        excess = ei.get("total_excess_reserve") or 0
        excess_pct = ei.get("excess_reserve_pct") or 0
        col_a.metric("ECL total (KM-based)", f"${total_km/1e6:.1f}M" if total_km else "N/D")
        col_b.metric("ECL ajustado (CIF)", f"${total_cif/1e6:.1f}M" if total_cif else "N/D")
        col_c.metric("Reserva en exceso", f"${excess/1e6:.1f}M ({excess_pct:.1f}%)" if excess else "N/D")

        st.caption(
            "El KM ignora prepayment como riesgo competidor, inflando la PD lifetime en un "
            f"{(1 - ov.get('cf_lifetime', 1)) * 100:.1f}%. "
            "Esto se traduce en reservas ECL excesivas en Stage 2. "
            "La corrección CIF reduce el ECL aplicando el factor CF a cada grade."
        )

    if not cif_ecl.empty:
        display_cols = [c for c in [
            "grade", "pd_lifetime_km", "cif_t60m", "cf_lifetime",
            "ecl_stage2_km", "ecl_stage2_cif", "excess_ecl_stage2_per_loan",
        ] if c in cif_ecl.columns]
        fmt_df = cif_ecl[display_cols].copy()
        for col in ["pd_lifetime_km", "cif_t60m", "cf_lifetime"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.4f}")
        for col in ["ecl_stage2_km", "ecl_stage2_cif", "excess_ecl_stage2_per_loan"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"${x:,.1f}")
        fmt_df.columns = [c.replace("_", " ").title() for c in fmt_df.columns]
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)
        st.caption(
            "ECL Stage 2 por loan (promedio). Exceso = diferencia que el KM sobreestima vs CIF. "
            "Grado A: +$300/loan × 72K loans ≈ $22M en exceso solo Stage 2."
        )
    else:
        st.info(
            "No se encontro `cif_ecl_impact.parquet`. "
            "Ejecute: `uv run python scripts/run_cif_ecl_impact.py`"
        )

with st.expander(
    "7) SICR conformal — Trigger IFRS9 por ancho de intervalo y sensibilidad ECL al alpha",
    expanded=False,
):
    sicr_status = try_load_json("sicr_conformal_status", directory="models", default={})
    sicr_grid = try_load_parquet("sicr_conformal_grid")
    ecl_alpha = try_load_parquet("ecl_alpha_sensitivity")

    if not sicr_status and sicr_grid.empty:
        st.info(
            "No se encontraron artefactos SICR conformal. "
            "Ejecute: `uv run python scripts/run_sicr_conformal.py`"
        )
    else:
        st.markdown(
            "**SICR conformal**: el ancho del intervalo de predicción (PD_high − PD_low) "
            "actúa como señal adicional de SICR (Significant Increase in Credit Risk). "
            "Loans con alta incertidumbre pero PD_12m moderada son candidatos a Stage 2. "
            "Contribución Paper 2 (JBF): formalización del trigger y sensibilidad del ECL al nivel de confianza."
        )

        # ── Part A: optimal threshold ───────────────────────────────────────────
        st.markdown("##### Parte A — Umbral óptimo del ancho conformal (t*)")
        pa = sicr_status.get("part_a", {})
        if pa:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("t* óptimo", f"{pa.get('optimal_width_threshold', 0):.2f}")
            c2.metric("F1 (t*)", f"{pa.get('optimal_f1', 0):.4f}" if pa.get("optimal_f1") else "N/D")
            c3.metric("Precision (t*)", f"{pa.get('optimal_precision', 0):.2%}" if pa.get("optimal_precision") else "N/D")
            c4.metric("ECL adicional (t*)", f"${pa.get('ecl_additional_at_optimal', 0)/1e6:.1f}M" if pa.get("ecl_additional_at_optimal") else "N/D")

        if not sicr_grid.empty:
            pd_col = "pd_threshold" if "pd_threshold" in sicr_grid.columns else None
            # Show curves for the default pd_threshold (0.20)
            if pd_col and sicr_grid[pd_col].nunique() > 1:
                grid_20 = sicr_grid[sicr_grid[pd_col] == 0.20].copy()
            else:
                grid_20 = sicr_grid.copy()

            if not grid_20.empty and "width_threshold" in grid_20.columns:
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(
                    x=grid_20["width_threshold"], y=grid_20["f1_width"],
                    name="F1", mode="lines+markers", line={"color": "#3B82F6"},
                ))
                fig_a.add_trace(go.Scatter(
                    x=grid_20["width_threshold"], y=grid_20["precision_width"],
                    name="Precision", mode="lines", line={"color": "#10B981", "dash": "dash"},
                ))
                fig_a.add_trace(go.Scatter(
                    x=grid_20["width_threshold"], y=grid_20["recall_width_of_missed"],
                    name="Recall (missed)", mode="lines", line={"color": "#F59E0B", "dash": "dot"},
                ))
                opt_t = pa.get("optimal_width_threshold", 0.30)
                fig_a.add_vline(x=opt_t, line={"dash": "dash", "color": "red"}, annotation_text=f"t*={opt_t}")
                fig_a.update_layout(
                    title="F1 / Precision / Recall vs umbral de ancho conformal (PD_thr=0.20)",
                    xaxis_title="Width threshold (t)",
                    yaxis_title="Score",
                    height=350,
                    legend={"orientation": "h"},
                    margin={"t": 40, "b": 40},
                )
                st.plotly_chart(fig_a, use_container_width=True)

            if "ecl_additional" in grid_20.columns:
                fig_ecl = go.Figure()
                fig_ecl.add_trace(go.Bar(
                    x=grid_20["width_threshold"],
                    y=grid_20["ecl_additional"] / 1e6,
                    name="ECL adicional ($M)",
                    marker_color="#6366F1",
                ))
                fig_ecl.update_layout(
                    title="ECL adicional por umbral de ancho (Stage 2 incremental)",
                    xaxis_title="Width threshold (t)",
                    yaxis_title="ECL adicional ($M)",
                    height=300,
                    margin={"t": 40, "b": 40},
                )
                st.plotly_chart(fig_ecl, use_container_width=True)
                st.caption(
                    f"t*={opt_t}: El trigger captura ~76% de defaults no detectados por PD solo, "
                    "con ECL adicional de ~$57M. Threshold mayor → menor cobertura y menor coste regulatorio."
                )

        # ── Part A.2: Stage composition at each threshold ────────────────────────
        sicr_trigger_opt = try_load_parquet("sicr_trigger_optimization")
        if not sicr_trigger_opt.empty:
            st.markdown("**Composición Stage 1 / Stage 2 por umbral de ancho**")
            _opt_view = sicr_trigger_opt[
                [c for c in ["width_threshold", "stage2_share", "n_stage2", "defaults_in_stage2",
                              "defaults_in_stage1", "precision_sicr", "recall_defaults", "f1_sicr"]
                 if c in sicr_trigger_opt.columns]
            ].copy()
            if "stage2_share" in _opt_view.columns:
                _opt_view["stage2_share"] = _opt_view["stage2_share"].apply(lambda x: f"{x:.1%}")
            if "precision_sicr" in _opt_view.columns:
                _opt_view["precision_sicr"] = _opt_view["precision_sicr"].apply(lambda x: f"{x:.3f}")
            if "recall_defaults" in _opt_view.columns:
                _opt_view["recall_defaults"] = _opt_view["recall_defaults"].apply(lambda x: f"{x:.3f}")
            if "f1_sicr" in _opt_view.columns:
                _opt_view["f1_sicr"] = _opt_view["f1_sicr"].apply(lambda x: f"{x:.4f}")
            st.dataframe(_opt_view, width="stretch", hide_index=True)
            st.caption(
                "stage2_share: % de loans que pasan a Stage 2 con este umbral. "
                "recall_defaults: % de defaults reales capturados. "
                "precision_sicr: % de los clasificados Stage 2 que son realmente default."
            )

        # ── Part B: alpha sensitivity ────────────────────────────────────────────
        st.markdown("##### Parte B — Sensibilidad del ECL al nivel de confianza alpha")
        _pb = sicr_status.get("part_b", {})
        _ecl_10 = _pb.get("ecl_at_alpha_010")
        _ecl_05 = _pb.get("ecl_at_alpha_005")
        _ecl_20 = _pb.get("ecl_at_alpha_020")
        if _ecl_10 or _ecl_05:
            _sb1, _sb2, _sb3 = st.columns(3)
            _sb1.metric("ECL adicional (alpha=0.10, 90% CI)", f"${_ecl_10/1e6:.1f}M" if _ecl_10 else "N/D")
            _sb2.metric("ECL adicional (alpha=0.05, 95% CI)", f"${_ecl_05/1e6:.1f}M" if _ecl_05 else "N/D")
            if _ecl_10 and _ecl_05:
                _incr = (_ecl_05 - _ecl_10) / _ecl_10 * 100
                _sb3.metric("Incremento 90%→95% CI", f"{_incr:+.0f}%")
            else:
                _sb3.metric("Incremento 90%→95% CI", "N/D")
        if not ecl_alpha.empty and "alpha" in ecl_alpha.columns:
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=ecl_alpha["confidence_level"] if "confidence_level" in ecl_alpha.columns else 1 - ecl_alpha["alpha"],
                y=ecl_alpha["ecl_additional"] / 1e6,
                mode="lines+markers",
                name="ECL adicional ($M)",
                line={"color": "#EF4444"},
                marker={"size": 8},
            ))
            if "n_additional_sicr" in ecl_alpha.columns:
                fig_b.add_trace(go.Scatter(
                    x=ecl_alpha["confidence_level"] if "confidence_level" in ecl_alpha.columns else 1 - ecl_alpha["alpha"],
                    y=ecl_alpha["n_additional_sicr"] / 1000,
                    mode="lines+markers",
                    name="Loans SICR adicionales (K)",
                    line={"color": "#8B5CF6", "dash": "dash"},
                    yaxis="y2",
                ))
            fig_b.update_layout(
                title="Coste regulatorio del nivel de confianza conformal (IFRS9 Stage 2)",
                xaxis_title="Nivel de confianza (1 − alpha)",
                yaxis_title="ECL adicional ($M)",
                yaxis2={"title": "Loans SICR adicionales (K)", "overlaying": "y", "side": "right"},
                height=350,
                legend={"orientation": "h"},
                margin={"t": 40, "b": 40},
            )
            st.plotly_chart(fig_b, use_container_width=True)

            pb = sicr_status.get("part_b", {})
            ecl_10_v = pb.get("ecl_at_alpha_010") or pb.get("ecl_additional_alpha_10") or 0
            ecl_05_v = pb.get("ecl_at_alpha_005") or pb.get("ecl_additional_alpha_01") or 0
            delta_pct = ((ecl_05_v - ecl_10_v) / max(ecl_10_v, 1)) * 100 if ecl_10_v else 0

            st.caption(
                f"Pasar de alpha=0.10 (90% CI) a alpha=0.05 (95% CI) incrementa el ECL adicional en ~{delta_pct:+.0f}% "
                f"(${ecl_10_v/1e6:.1f}M → ${ecl_05_v/1e6:.1f}M). "
                "Este es el coste regulatorio de exigir mayor certeza en los intervalos conformales."
            )
        else:
            st.info("Ejecute `run_sicr_conformal.py` para ver la sensibilidad ECL al alpha.")


with st.expander("8) BMA vs Conformal — Comparación de conjuntos de incertidumbre (Paper 2 / Paper Estrella)", expanded=False):
    bma_status = try_load_json("bma_comparison_status", directory="models", default={})
    bma_grade = try_load_parquet("bma_grade_coverage")

    if not bma_status and bma_grade.empty:
        st.info("Ejecute: `uv run python scripts/run_bma_comparison.py`")
    else:
        st.markdown(
            "**BMA (Bayesian Model Averaging)** usa el ensemble LR + CatBoost_default + CatBoost_tuned. "
            "Comparación clave vs conformal Mondrian: cobertura global, eficiencia (ancho), y cobertura por grade."
        )
        res = bma_status.get("results", {}).get("equal", bma_status.get("results", {}))
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cobertura BMA (equal)", f"{res.get('overall_bma_coverage', 0):.2%}")
            c2.metric("Cobertura CP Mondrian", f"{res.get('overall_cp_coverage', 0):.2%}")
            c3.metric("Ancho medio BMA", f"{res.get('mean_bma_width', 0):.4f}")
            c4.metric(
                "Ancho medio CP",
                f"{res.get('mean_cp_width', 0):.4f}",
                delta=f"-{res.get('width_reduction_cp_vs_bma_pct', 0):.1f}% vs BMA",
                delta_color="normal",
            )
            st.metric(
                "Min cobertura por grade BMA",
                f"{res.get('min_grade_bma_coverage', 0):.2%}",
                help="BMA puede violar la cobertura nominal en grupos específicos.",
            )

        if not bma_grade.empty:
            eq = bma_grade[bma_grade["scheme"] == "equal"] if "scheme" in bma_grade.columns else bma_grade
            if not eq.empty:
                fig_bma = go.Figure()
                fig_bma.add_trace(go.Bar(x=eq["grade"], y=eq["bma_coverage"], name="BMA coverage", marker_color="#F59E0B"))
                fig_bma.add_trace(go.Bar(x=eq["grade"], y=eq["cp_coverage"], name="CP coverage", marker_color="#3B82F6"))
                fig_bma.add_hline(y=0.90, line={"dash": "dash", "color": "red"}, annotation_text="90% nominal")
                fig_bma.update_layout(
                    title="Cobertura por grade: BMA vs Conformal Mondrian",
                    xaxis_title="Grade", yaxis_title="Coverage",
                    barmode="group", height=320,
                    margin={"t": 40, "b": 40},
                    yaxis={"range": [0.80, 1.0]},
                )
                st.plotly_chart(fig_bma, use_container_width=True)
                st.caption(
                    "BMA puede caer por debajo del 90% nominal en grades específicos. "
                    "CP Mondrian garantiza cobertura por grade en el conjunto de calibración "
                    "(desvíos en test reflejan drift OOT, no violación de la garantía)."
                )

with st.expander("9) TS Forecast → ECL Intervals — Incertidumbre macro en provisiones IFRS9", expanded=False):
    ts_ecl_status = try_load_json("ts_ecl_intervals_status", directory="models", default={})
    ts_ecl_df = try_load_parquet("ts_ecl_intervals")

    if not ts_ecl_status and ts_ecl_df.empty:
        st.info("Ejecute: `uv run python scripts/run_ts_ecl_intervals.py`")
    else:
        st.markdown(
            "Propagación de la incertidumbre del forecast TS (AutoARIMA, banda 90%) al ECL de la cartera. "
            "Cuantifica la **banda de provisión regulatoria** derivada del intervalo de predicción macroeconómica."
        )
        ts_f = ts_ecl_status.get("ts_forecast", {})
        if ts_f:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ECL optimista", f"${ts_f.get('mean_ecl_optimistic_90', 0)/1e6:.1f}M")
            c2.metric("ECL punto", f"${ts_f.get('mean_ecl_point', 0)/1e6:.1f}M")
            c3.metric("ECL adverso (90%)", f"${ts_f.get('mean_ecl_adverse_90', 0)/1e6:.1f}M")
            c4.metric(
                "Banda ECL 90%",
                f"${ts_f.get('ecl_band_width_90', 0)/1e6:.1f}M",
                delta=f"{ts_f.get('ecl_band_width_90_pct_of_point', 0):.0f}% del punto",
                delta_color="off",
            )

        if not ts_ecl_df.empty and "ecl_point" in ts_ecl_df.columns:
            fig_ts = go.Figure()
            x = ts_ecl_df["month"].astype(str).tolist()
            fig_ts.add_trace(go.Scatter(x=x, y=ts_ecl_df["ecl_adverse_90"] / 1e6, name="Adverso 90%", line={"color": "#EF4444", "dash": "dash"}, fill="tonexty"))
            fig_ts.add_trace(go.Scatter(x=x, y=ts_ecl_df["ecl_point"] / 1e6, name="Punto", line={"color": "#6366F1"}, fill="tonexty"))
            fig_ts.add_trace(go.Scatter(x=x, y=ts_ecl_df["ecl_optimistic_90"] / 1e6, name="Optimista 90%", line={"color": "#10B981", "dash": "dash"}))
            fig_ts.update_layout(
                title="ECL mensual: banda de incertidumbre TS forecast (90%)",
                xaxis_title="Mes", yaxis_title="ECL ($M)",
                height=320, margin={"t": 40, "b": 40},
                legend={"orientation": "h"},
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            st.caption(
                "La banda adverso–optimista refleja la incertidumbre del forecast macroeconómico. "
                "Cuando el punto forecast ≈ 0 (sin cambio esperado), la banda es asimétrica y muy amplia. "
                "Nuevo input para ICAAP stress testing: la elección del nivel de confianza del forecast define el colchón regulatorio."
            )

with st.expander("10) Coste de Mala Clasificación Stage 1 vs Stage 2 (IFRS9 asimétrico)", expanded=False):
    stage_status = try_load_json("stage_misclassification_status", directory="models", default={})
    stage_cost = try_load_parquet("stage_misclassification_cost")
    stage_grade = try_load_parquet("stage_cost_by_grade")

    if not stage_status and stage_cost.empty:
        st.info("Ejecute: `uv run python scripts/run_stage_misclassification_cost.py`")
    else:
        st.markdown(
            "**Coste asimétrico IFRS9**: FN (bajo-provisión) pesa más que FP (sobre-provisión). "
            f"Peso regulatorio por defecto: w_fn={stage_status.get('w_fn', 5.0)}, w_fp={stage_status.get('w_fp', 1.0)}."
        )
        pa = stage_status.get("part_a_pure_pd", {})
        pb = stage_status.get("part_b_combined", {})
        if pa and pb:
            c1, c2, c3 = st.columns(3)
            c1.metric("Coste total PD solo", f"${pa.get('total_cost_M', 0):.1f}M")
            c2.metric("Coste total PD+Width", f"${pb.get('total_cost_M', 0):.1f}M")
            c3.metric(
                "Reducción por trigger width",
                f"{pb.get('cost_reduction_vs_pure_pd_pct', 0):.1f}%",
                delta_color="normal",
            )
            r1, r2 = st.columns(2)
            r1.metric("Tasa FN (PD solo)", f"{pa.get('fn_rate', 0):.2%}", help="Defaults not staged as Stage 2")
            r2.metric("Tasa FN (combinado)", f"{pb.get('fn_rate', 0):.2%}")

        if not stage_cost.empty and "total_cost" in stage_cost.columns:
            combined_grid = stage_cost[stage_cost["use_width_trigger"] == True].copy()  # noqa: E712
            if not combined_grid.empty and "pd_threshold" in combined_grid.columns:
                for pd_thr in sorted(combined_grid["pd_threshold"].unique())[:3]:
                    sub = combined_grid[combined_grid["pd_threshold"] == pd_thr].sort_values("width_threshold")
                    if sub.empty:
                        continue
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Scatter(
                        x=sub["width_threshold"], y=sub["total_cost"] / 1e6,
                        name=f"Total (pd_thr={pd_thr})", mode="lines+markers",
                    ))
                    fig_cost.add_trace(go.Scatter(
                        x=sub["width_threshold"], y=sub["fn_cost"] / 1e6,
                        name="FN cost", mode="lines", line={"dash": "dash", "color": "#EF4444"},
                    ))
                    fig_cost.add_trace(go.Scatter(
                        x=sub["width_threshold"], y=sub["fp_cost"] / 1e6,
                        name="FP cost", mode="lines", line={"dash": "dot", "color": "#10B981"},
                    ))
                    fig_cost.update_layout(
                        title=f"Coste misclasificación vs umbral width (pd_thr={pd_thr})",
                        xaxis_title="Width threshold (t)",
                        yaxis_title="Coste ($M, asimétrico)",
                        height=280, margin={"t": 40, "b": 30},
                        legend={"orientation": "h"},
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                    break

        if not stage_grade.empty and "grade" in stage_grade.columns:
            st.markdown("**Coste por grade (umbral óptimo combinado)**")
            disp_cols = [c for c in ["grade", "n_loans", "default_rate", "n_fn", "n_fp", "fn_cost", "fp_cost", "total_cost"] if c in stage_grade.columns]
            fmt = stage_grade[disp_cols].copy()
            for col in ["fn_cost", "fp_cost", "total_cost"]:
                if col in fmt.columns:
                    fmt[col] = fmt[col].apply(lambda x: f"${x/1e6:.2f}M")
            if "default_rate" in fmt.columns:
                fmt["default_rate"] = fmt["default_rate"].apply(lambda x: f"{x:.2%}")
            st.dataframe(fmt, use_container_width=True, hide_index=True)
            st.caption(
                "FP (sobre-provisión) domina el coste total porque la mayoría de loans son Stage 1 (no defaults). "
                "El trigger width reduce FN pero añade FP. El balance óptimo depende de w_fn/w_fp."
            )


render_caveats(
    [
        "Las conclusiones temporales dependen del modelo elegido (Cox/RSF) y de la estabilidad del proceso de pagos.",
        "Las curvas lifetime deben leerse junto con PD calibrada y contexto macro, no de forma aislada.",
        "CIF asume eventos mutuamente excluyentes (default vs prepayment). Loans con restructuracion son censurados.",
        "El factor de corrección CF aplica el ratio CIF/KM global a todos los grades; per-grade requeriría KM por grade.",
    ]
)
render_page_feedback("survival_analysis")

next_page_teaser(
    "Inteligencia Causal",
    "Pasamos de correlaciones a efectos causales y reglas de intervención.",
    "pages/causal_intelligence.py",
)
