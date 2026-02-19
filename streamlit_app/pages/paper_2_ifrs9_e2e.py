"""Paper 2 draft: IFRS9 end-to-end pipeline with distribution-free uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.paper_scaffold import render_phase_tracker
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    download_table,
    format_number,
    format_pct,
    try_load_json,
    try_load_parquet,
)

st.title("🏦 Paper 2 — Working Draft")
st.caption("An End-to-End ML Pipeline for IFRS9 with Distribution-Free Uncertainty")
st.warning(
    "Borrador de trabajo para revisión académica. El foco es mostrar el máximo material "
    "técnico disponible para evaluar factibilidad y aporte científico."
)

pipeline_summary = try_load_json("pipeline_summary", directory="data", default={})
ifrs9_summary = try_load_parquet("ifrs9_scenario_summary")
ifrs9_grid = try_load_parquet("ifrs9_sensitivity_grid")
ifrs9_grade = try_load_parquet("ifrs9_scenario_grade_summary")

pipeline = pipeline_summary.get("pipeline", {})
stages = pipeline.get("stages", {}) if isinstance(pipeline.get("stages"), dict) else {}

def _safe_money(value: float) -> str:
    return format_number(float(value), prefix="$") if np.isfinite(value) else "N/D"


def _safe_pct(value: float) -> str:
    return format_pct(float(value), 2) if np.isfinite(value) else "N/D"


def _pick_scenario_row(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.DataFrame, str | None]:
    if df.empty or "scenario" not in df.columns:
        return pd.DataFrame(), None
    names = df["scenario"].astype(str).str.lower()
    for candidate in candidates:
        mask = names == candidate.lower()
        if mask.any():
            return df.loc[mask].copy(), str(df.loc[mask, "scenario"].iloc[0])
    return pd.DataFrame(), None


def _pick_worst_scenario_row(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if df.empty or not {"scenario", "total_ecl"}.issubset(df.columns):
        return pd.DataFrame(), None
    ecl = pd.to_numeric(df["total_ecl"], errors="coerce")
    if not ecl.notna().any():
        return pd.DataFrame(), None
    idx = ecl.idxmax()
    return df.loc[[idx]].copy(), str(df.loc[idx, "scenario"])


baseline = (
    ifrs9_summary.loc[ifrs9_summary["scenario"] == "baseline"].copy()
    if (not ifrs9_summary.empty and "scenario" in ifrs9_summary.columns)
    else pd.DataFrame()
)
severe, severe_label = _pick_scenario_row(ifrs9_summary, ["severe_stress", "severe"])
if severe.empty:
    severe, severe_label = _pick_worst_scenario_row(ifrs9_summary)

baseline_ecl = float(baseline["total_ecl"].iloc[0]) if (not baseline.empty and "total_ecl" in baseline.columns) else np.nan
severe_ecl = float(severe["total_ecl"].iloc[0]) if (not severe.empty and "total_ecl" in severe.columns) else np.nan
uplift = (severe_ecl / baseline_ecl - 1.0) if np.isfinite(baseline_ecl) and baseline_ecl > 0 and np.isfinite(severe_ecl) else np.nan
severe_label_display = severe_label if severe_label else "escenario severo"
scenario_values = (
    sorted(ifrs9_summary["scenario"].astype(str).unique().tolist())
    if (not ifrs9_summary.empty and "scenario" in ifrs9_summary.columns)
    else []
)
scenario_list_display = ", ".join(f"`{name}`" for name in scenario_values) if scenario_values else "`baseline`, `adverse`, `severe`"

stage1_n = int(stages.get("S1", 0) or 0)
stage2_n = int(stages.get("S2", 0) or 0)
stage3_n = int(stages.get("S3", 0) or 0)

st.markdown("## 0) Metadata de Propuesta")
meta_df = pd.DataFrame(
    [
        {"Campo": "Estado", "Valor": "Working Draft"},
        {"Campo": "Venue sugerido", "Valor": "Journal of Banking & Finance"},
        {"Campo": "Pregunta", "Valor": "Como integrar incertidumbre conformal en un pipeline IFRS9 accionable"},
        {"Campo": "Dataset", "Valor": "Lending Club (split OOT)"},
        {"Campo": "ECL baseline", "Valor": _safe_money(baseline_ecl)},
        {"Campo": f"ECL {severe_label_display}", "Valor": _safe_money(severe_ecl)},
        {"Campo": f"Uplift {severe_label_display} vs baseline", "Valor": _safe_pct(uplift)},
    ]
)
st.dataframe(meta_df, use_container_width=True, hide_index=True)
if severe_label and severe_label != "severe_stress":
    st.caption(
        f"Nota de trazabilidad: el escenario severo en el artefacto actual se llama `{severe_label}`."
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("ECL Baseline", _safe_money(baseline_ecl))
k2.metric(f"ECL {severe_label_display}", _safe_money(severe_ecl))
k3.metric("Uplift Severe", _safe_pct(uplift))
k4.metric("Stage 2 Loans", f"{stage2_n:,}")

st.markdown("## 1) Abstract (Draft)")
st.markdown(
    f"""
Presentamos un pipeline integral para IFRS9 que conecta PD calibrada, intervalos conformales
(Mondrian), staging y calculo de ECL por escenario. En ventana OOT, la distribucion de
stages es S1={stage1_n:,}, S2={stage2_n:,}, S3={stage3_n:,}. Bajo escenario baseline,
el ECL agregado es {_safe_money(baseline_ecl)}, mientras que en {severe_label_display}
asciende a {_safe_money(severe_ecl)} ({_safe_pct(uplift)}).

El framework incorpora senal de incertidumbre conformal para SICR: observaciones con ancho
conformal alto y PD no decreciente pueden migrar a Stage 2 aun sin superar el umbral absoluto
clasico. El resultado es una lectura prudencial de provisiones con sensibilidad explicita
a incertidumbre estadistica.
"""
)

st.markdown("## 2) Introduction")
st.markdown(
    """
IFRS9 exige medicion prospectiva de perdida esperada (ECL) y clasificacion de deterioro
(Stage 1/2/3). En la practica, muchos pipelines usan una sola PD puntual y reglas SICR
basadas solo en cambio absoluto o relativo de PD. Este enfoque oculta incertidumbre de modelo
justo en el punto donde la decision contable es mas sensible.

Este draft propone: (i) ECL por rango (`low`, `point`, `high`) usando conformal prediction,
(ii) trigger SICR enriquecido con ancho conformal, y (iii) mapa de sensibilidad de provisiones
que explicita la fragilidad de resultados frente a supuestos macro y parametros prudenciales.
"""
)

st.markdown("## 3) Related Work (Resumen para borrador)")
related = pd.DataFrame(
    [
        ["Bárcena Saavedra et al. (2024)", "IFRS9 lifetime PD", "Supervivencia/competing risks para ECL lifetime"],
        ["Bellini et al. (2024)", "Credit risk modelling", "Puente regulatorio entre modelado y reporting"],
        ["Gibbs & Candes (2021)", "Adaptive conformal", "Fundamento para monitoreo bajo drift temporal"],
        ["Vovk & Petej (2014)", "Venn-Abers", "Calibracion con enfoque intervalar y validez"],
    ],
    columns=["Referencia", "Eje", "Relevancia para este draft"],
)
st.dataframe(related, use_container_width=True, hide_index=True)

st.markdown("## 4) Data and Experimental Protocol")
st.markdown(
    """
- Split temporal OOT fijo en pipeline productivo del repositorio.
- PD calibrada + intervalos conformales por subgrupo.
- Stage assignment IFRS9 con regla clasica y trigger por incertidumbre.
- Escenarios macro disponibles: {scenario_list_display} y grilla de sensibilidad
  (`pd_mult`, `lgd_mult`, `discount_rate`).
"""
)

st.markdown("## 5) Methods")
st.markdown("### 5.1 Expected Credit Loss by Horizon")
st.latex(
    r"\mathrm{ECL}_{i,s}^{(h)} = \mathrm{PD}_{i,s}^{(h)} \times \mathrm{LGD}_{i,s} \times \mathrm{EAD}_{i,s} \times \mathrm{DF}_{i,s},\quad h\in\{\text{low},\text{point},\text{high}\}"
)
st.caption("Equation 1. ECL por prestamo, escenario y horizonte de incertidumbre.")

st.markdown("### 5.2 SICR Trigger with Conformal Width")
st.latex(
    r"\mathrm{Stage2}_i = \mathbb{1}\left[\Delta \widehat{PD}_i > \theta_{PD} \;\lor\; (w_i > q_{0.9}(w)\wedge \Delta \widehat{PD}_i \ge 0)\right]"
)
st.caption("Equation 2. Trigger SICR combinado: deterioro de PD o incertidumbre alta sin mejora de PD.")

st.markdown("### 5.3 Scenario Sensitivity")
st.latex(
    r"\mathrm{ECL}_{\mathrm{portfolio}}(\gamma_{PD},\gamma_{LGD},r)=\sum_i \mathrm{ECL}_{i}(\gamma_{PD}\cdot PD_i,\gamma_{LGD}\cdot LGD_i,r)"
)
st.caption("Equation 3. Grilla de sensibilidad de provisiones frente a multiplicadores y tasa de descuento.")

st.markdown("## 6) Results")

if not ifrs9_summary.empty and {"scenario", "stage1_share", "stage2_share", "stage3_share"}.issubset(ifrs9_summary.columns):
    stage_share = ifrs9_summary[["scenario", "stage1_share", "stage2_share", "stage3_share"]].melt(
        id_vars=["scenario"],
        var_name="stage",
        value_name="share",
    )
    fig1 = px.bar(
        stage_share,
        x="scenario",
        y="share",
        color="stage",
        barmode="stack",
        title="Figure 1. Stage Composition by Scenario",
        labels={"scenario": "Escenario", "share": "Proporcion"},
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Figure 1. Composicion de Stage 1/2/3 bajo cada escenario IFRS9.")

if not ifrs9_summary.empty and {"scenario", "total_ecl_low", "total_ecl_point", "total_ecl_high"}.issubset(ifrs9_summary.columns):
    ecl_range_plot = ifrs9_summary[["scenario", "total_ecl_low", "total_ecl_point", "total_ecl_high"]].copy()
    ecl_range_plot["err_up"] = (ecl_range_plot["total_ecl_high"] - ecl_range_plot["total_ecl_point"]).clip(lower=0)
    ecl_range_plot["err_down"] = (ecl_range_plot["total_ecl_point"] - ecl_range_plot["total_ecl_low"]).clip(lower=0)

    fig2 = px.bar(
        ecl_range_plot,
        x="scenario",
        y="total_ecl_point",
        error_y="err_up",
        error_y_minus="err_down",
        title="Figure 2. ECL Point Estimate with Conformal Range",
        labels={"scenario": "Escenario", "total_ecl_point": "ECL point"},
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Figure 2. ECL point con barras de incertidumbre conformal (`low` a `high`).")

if not ifrs9_grid.empty and {"pd_mult", "lgd_mult", "discount_rate", "total_ecl"}.issubset(ifrs9_grid.columns):
    discount_values = sorted(ifrs9_grid["discount_rate"].dropna().unique().tolist())
    target_discount = 0.05 if 0.05 in discount_values else (discount_values[0] if discount_values else None)

    if target_discount is not None:
        grid_slice = ifrs9_grid.loc[ifrs9_grid["discount_rate"] == target_discount].copy()
        heat = (
            grid_slice.pivot_table(index="pd_mult", columns="lgd_mult", values="total_ecl", aggfunc="mean")
            / 1_000_000
        )
        fig3 = px.imshow(
            heat,
            labels={"x": "LGD multiplier", "y": "PD multiplier", "color": "ECL (MM)"},
            title=f"Figure 3. ECL Sensitivity Heatmap (discount_rate={target_discount:.2f})",
            text_auto=".1f",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Figure 3. Sensibilidad de ECL ante variaciones de PD y LGD.")

if not ifrs9_grade.empty and {"scenario", "grade", "total_ecl", "stage3_share"}.issubset(ifrs9_grade.columns):
    baseline_grade = ifrs9_grade.loc[ifrs9_grade["scenario"] == "baseline"].copy()
    if not baseline_grade.empty:
        fig4 = px.bar(
            baseline_grade.sort_values("total_ecl", ascending=False),
            x="grade",
            y="total_ecl",
            color="stage3_share",
            title="Figure 4. Baseline ECL by Grade",
            labels={"total_ecl": "ECL baseline", "stage3_share": "Stage 3 share"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Figure 4. ECL baseline por grade con intensidad de Stage 3.")

st.markdown("### Tablas principales")
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("**Table 1. IFRS9 Scenario Summary**")
    if ifrs9_summary.empty:
        st.info("No se encontro `ifrs9_scenario_summary.parquet`.")
    else:
        st.dataframe(ifrs9_summary, use_container_width=True, hide_index=True)
        download_table(ifrs9_summary, "paper2_table1_scenario_summary.csv")

with col_t2:
    st.markdown("**Table 2. IFRS9 Sensitivity Grid**")
    if ifrs9_grid.empty:
        st.info("No se encontro `ifrs9_sensitivity_grid.parquet`.")
    else:
        st.dataframe(ifrs9_grid, use_container_width=True, hide_index=True)
        download_table(ifrs9_grid, "paper2_table2_sensitivity_grid.csv")

with st.expander("Appendix Tables"):
    st.markdown("**Table A1. IFRS9 Grade Summary by Scenario**")
    if ifrs9_grade.empty:
        st.info("No se encontro `ifrs9_scenario_grade_summary.parquet`.")
    else:
        st.dataframe(ifrs9_grade, use_container_width=True, hide_index=True)
        download_table(ifrs9_grade, "paper2_tableA1_grade_summary.csv")

st.markdown("## 7) Discussion")
st.markdown(
    """
El enfoque muestra que la incertidumbre no solo afecta intervalos de prediccion, sino decisiones
contables de staging y magnitud de provisiones. Incorporar rango conformal permite discutir
prudencia de forma cuantitativa frente a escenarios macro y supuestos de severidad.
"""
)

st.markdown("## 8) Threats to Validity")
st.markdown(
    """
- El trigger SICR por incertidumbre requiere calibracion institucional de umbrales.
- Escenarios macro por multiplicadores son una aproximacion; falta curva macro estructural.
- La validez externa depende de recalibrar PD/LGD/EAD en otra cartera y jurisdiccion.
"""
)

st.markdown("## 9) Reproducibility Package")
st.code(
    "\n".join(
        [
            "uv run dvc repro run_ifrs9_sensitivity build_pipeline_results export_streamlit_artifacts",
            "uv run python scripts/run_paper_notebook_suite.py",
            "uv run pytest -q tests/test_evaluation/test_ifrs9.py",
        ]
    ),
    language="bash",
)

st.markdown("## 10) Fases del Paper y Estado Actual")
render_phase_tracker(
    [
        {
            "Fase": "1. Research Question + Scope",
            "Estado": "Completada",
            "Evidencia": "Secciones 0-2 de esta pagina",
            "Criterio de cierre": "Aprobacion del framing por revisor experto.",
        },
        {
            "Fase": "2. Related Work + Positioning",
            "Estado": "Completada",
            "Evidencia": "Tabla de related work (Seccion 3)",
            "Criterio de cierre": "Agregar/ajustar comparadores sugeridos por el profesor.",
        },
        {
            "Fase": "3. Methods + Experimental Design",
            "Estado": "Completada",
            "Evidencia": "Ecuaciones 1-3 y protocolo (Secciones 4-5)",
            "Criterio de cierre": "Validar notacion final y supuestos prudenciales.",
        },
        {
            "Fase": "4. Results + Robustness",
            "Estado": "Completada",
            "Evidencia": "Figuras 1-4, Tablas 1-2-A1",
            "Criterio de cierre": "Refinar visuales y destacar resultados principales.",
        },
        {
            "Fase": "5. Discussion + Limitations",
            "Estado": "En progreso",
            "Evidencia": "Secciones 7-8",
            "Criterio de cierre": "Expandir implicaciones regulatorias tras feedback.",
        },
    ]
)

st.markdown("### Puntos a Revisar / Complementar")
st.markdown(
    """
- **Section 1 (Abstract)**: ajustar nivel de claim segun tolerancia del revisor.
- **Section 2 (Introduction)**: reforzar diferencia exacta frente a enfoques IFRS9 clasicos.
- **Section 3 (Related Work / Table)**: decidir si incluir benchmark explicito con scorecards tradicionales.
- **Section 5 (Methods / Eq. 1)**: confirmar convencion de horizonte lifetime vs 12m en la notacion final.
- **Section 5 (Methods / Eq. 2)**: justificar cuantil de ancho conformal usado para trigger SICR.
- **Figure 1**: validar legibilidad de stages para presentacion a profesor.
- **Figure 2**: confirmar si mostrar tambien ECL relativo (%) ademas de valores absolutos.
- **Figure 3**: considerar contornos/iso-lineas para lectura mas rapida de sensibilidad.
- **Figure 4**: revisar si ordenar por riesgo esperado o por contribucion marginal a ECL.
- **Table 1 / Table 2 / Table A1**: podar columnas para version principal y mover detalle a anexo.
"""
)
