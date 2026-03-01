"""Tesis de Especialización: Conformal Prediction para Riesgo Crediticio.

Consolida todos los resultados del proyecto como evidencia para la tesis
de Especialización en Analítica y Ciencia de Datos (UTP).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.story_shell import render_key_takeaway
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_pct,
    load_json,
    try_load_parquet,
)

st.title("🎓 Tesis de Especialización")
st.caption(
    "Conformal Prediction para la Calibración y Cuantificación de "
    "Incertidumbre en Modelos de Riesgo Crediticio — UTP 2026"
)
page_contract = get_page_contract("tesis_especializacion")
render_key_takeaway(
    "Esta página consolida toda la evidencia experimental del proyecto como "
    "soporte para la tesis de especialización. Cada tab corresponde a un "
    "capítulo o sección del documento final."
)

# ── Load data ──
comparison = load_json("model_comparison")
final_metrics = comparison.get("final_test_metrics", {}) if comparison else {}
cal_report = comparison.get("calibration_selection_report", {}) if comparison else {}
conformal_bt = try_load_parquet("conformal_backtest_monthly", default=None)
ifrs9_summary = try_load_parquet("ifrs9_scenario_summary", default=None)
fairness_data = try_load_parquet("fairness_audit", default=None)

# Load EDA summary
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
_eda_path = _DATA_DIR / "eda_summary.json"
eda = {}
if _eda_path.exists():
    with open(_eda_path) as f:
        eda = json.load(f)


def render_eda_section() -> None:
    """Renderiza el bloque de análisis exploratorio (EDA) del dataset."""
    st.subheader("Análisis Exploratorio del Dataset")

    st.markdown(
        """
El dataset de **LendingClub** es la fuente de datos pública más grande y
referenciada en la literatura de riesgo crediticio peer-to-peer. Contiene el
historial completo de préstamos originados entre 2007 y 2020, con más de 140
variables originales que cubren información del solicitante (ingresos, historial
crediticio, empleo), del préstamo (monto, tasa, plazo) y del resultado
(pagos, defaults, recuperaciones).
"""
    )

    if eda:
        # ── Key metrics ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Préstamos (train)", f"{eda.get('n_loans', 0):,.0f}")
        c2.metric("Default rate", format_pct(eda.get("default_rate", 0)))
        c3.metric("Tasa interés media", f"{eda.get('int_rate', {}).get('mean', 0):.1f}%")
        c4.metric("DTI medio", f"{eda.get('dti', {}).get('mean', 0):.1f}")

        c5, c6, c7, c8 = st.columns(4)
        loan_amnt = eda.get("loan_amnt", {})
        annual_inc = eda.get("annual_inc", {})
        c5.metric("Monto medio", f"${loan_amnt.get('mean', 0):,.0f}")
        c6.metric("Monto mediano", f"${loan_amnt.get('median', 0):,.0f}")
        c7.metric("Ingreso medio", f"${annual_inc.get('mean', 0):,.0f}")
        c8.metric("Ingreso mediano", f"${annual_inc.get('median', 0):,.0f}")

        # ── Default Rate by Grade ──
        st.subheader("Tasa de Default por Grade")
        st.markdown(
            """
El sistema de grades de LendingClub (A–G) refleja la evaluación de riesgo
al momento de la originación. La relación entre grade y default rate es
**monotónicamente creciente**: los préstamos Grade A tienen una tasa de
default del 5.6%, mientras que los Grade G alcanzan el 47.7% — casi 9 veces
mayor. Esta relación es fundamental para la predicción conformal: los
intervalos Mondrian (grupo-condicional) aprovechan esta estructura para
proveer cobertura calibrada *dentro de cada grade*.
"""
        )

        dr_by_grade = eda.get("default_rate_by_grade", {})
        if dr_by_grade:
            grades = list(dr_by_grade.keys())
            rates = [v * 100 for v in dr_by_grade.values()]
            fig_dr = px.bar(
                x=grades,
                y=rates,
                labels={"x": "Grade", "y": "Default Rate (%)"},
                title="Tasa de Default por Grade (Train Set)",
                text=[f"{r:.1f}%" for r in rates],
                color=rates,
                color_continuous_scale="RdYlGn_r",
            )
            fig_dr.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_dr.update_traces(textposition="outside")
            fig_dr.update_coloraxes(showscale=False)
            st.plotly_chart(fig_dr, width="stretch")

        # ── Loan Count by Grade ──
        st.subheader("Distribución de Préstamos por Grade")
        st.markdown(
            """
La composición del portafolio está concentrada en los grades B y C, que
representan más del 58% de los préstamos. Los grades extremos (F, G)
constituyen menos del 3% de la cartera. Esta distribución asimétrica tiene
implicaciones para el conformal prediction: los subgrupos pequeños (F, G)
requieren tratamiento especial en Mondrian conformal para garantizar
cobertura suficiente (floor de cobertura mínima por grupo).
"""
        )

        lc_by_grade = eda.get("loan_count_by_grade", {})
        if lc_by_grade:
            fig_lc = px.bar(
                x=list(lc_by_grade.keys()),
                y=list(lc_by_grade.values()),
                labels={"x": "Grade", "y": "Número de Préstamos"},
                title="Número de Préstamos por Grade (Train Set)",
                text=[f"{v:,.0f}" for v in lc_by_grade.values()],
            )
            fig_lc.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_lc.update_traces(textposition="outside", marker_color="#0B5ED7")
            st.plotly_chart(fig_lc, width="stretch")

        # ── Financial Variables ──
        st.subheader("Variables Financieras Clave")
        st.markdown(
            """
Las variables financieras del solicitante determinan tanto el riesgo de
default como la incertidumbre asociada. Préstamos con altas tasas de interés
y DTI elevado tienden a tener intervalos conformales más amplios,
reflejando mayor incertidumbre en la predicción.
"""
        )

        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            fin_vars = {
                "Monto del préstamo": loan_amnt.get("mean", 0),
                "Ingreso anual": annual_inc.get("mean", 0),
            }
            fig_fin1 = go.Figure(go.Bar(
                x=list(fin_vars.keys()),
                y=list(fin_vars.values()),
                text=[f"${v:,.0f}" for v in fin_vars.values()],
                textposition="outside",
                marker_color=["#0B5ED7", "#198754"],
            ))
            fig_fin1.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title="Promedios: Monto e Ingreso",
                yaxis_title="USD",
            )
            st.plotly_chart(fig_fin1, width="stretch")

        with col_fin2:
            rate_vars = {
                "Tasa de interés (%)": eda.get("int_rate", {}).get("mean", 0),
                "DTI (%)": eda.get("dti", {}).get("mean", 0),
            }
            fig_fin2 = go.Figure(go.Bar(
                x=list(rate_vars.keys()),
                y=list(rate_vars.values()),
                text=[f"{v:.1f}%" for v in rate_vars.values()],
                textposition="outside",
                marker_color=["#F59F00", "#DC3545"],
            ))
            fig_fin2.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title="Promedios: Tasa e Indicadores",
                yaxis_title="%",
            )
            st.plotly_chart(fig_fin2, width="stretch")

        # ── Term Distribution ──
        st.subheader("Distribución por Plazo")
        term_dist = eda.get("term_distribution", {})
        if term_dist:
            term_labels = [f"{int(float(k))} meses" for k in term_dist]
            term_values = list(term_dist.values())
            fig_term = px.pie(
                names=term_labels,
                values=term_values,
                title="Distribución por Plazo del Préstamo",
                color_discrete_sequence=["#0B5ED7", "#F59F00"],
            )
            fig_term.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_term.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_term, width="stretch")

            st.markdown(
                f"""
El **{term_values[0] / sum(term_values) * 100:.0f}%** de los préstamos son a
36 meses y el **{term_values[1] / sum(term_values) * 100:.0f}%** a 60 meses.
Los préstamos a mayor plazo tienen mayor tasa de default por la mayor
exposición temporal, lo que afecta directamente el ancho de los intervalos
conformales y el cálculo de provisiones lifetime bajo IFRS 9.
"""
            )

        # ── Missing Values ──
        null_pcts = eda.get("null_pcts", {})
        if null_pcts:
            st.subheader("Valores Faltantes")
            st.markdown(
                """
CatBoost maneja valores faltantes de forma nativa (sin imputación), lo cual
es una ventaja sobre otros frameworks que requieren preprocesamiento
adicional. Las variables con mayor proporción de missings son:
"""
            )
            null_df = pd.DataFrame([
                {"Variable": k, "% Missing": f"{v * 100:.1f}%"}
                for k, v in null_pcts.items()
            ])
            st.dataframe(null_df, width="stretch", hide_index=True)
    else:
        st.info(
            "Datos del EDA no disponibles. Ejecute el pipeline para generar "
            "`data/processed/eda_summary.json`."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "Intro & Abstract",
    "Dataset & EDA",
    "Modelado Base",
    "Conformal Prediction",
    "Evaluación Comparativa",
    "Impacto IFRS9",
    "Fairness",
    "CRISP-DM",
    "Conclusiones",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 0: Intro & Abstract
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Intro & Abstract")

    # ── Abstract ──
    st.markdown(
        """
> **Conformal Prediction para la Calibración y Cuantificación de
> Incertidumbre en Modelos de Riesgo Crediticio**
>
> **Carlos Alfredo Vergara Rojas**
>
> Universidad Tecnológica de Pereira — Programa de Especialización en
> Analítica y Ciencia de Datos Aplicada, 2026
>
> Docente directora: Alejandra María Restrepo Franco
"""
    )

    st.markdown(
        """
### Resumen de la propuesta

Los modelos de riesgo crediticio en la industria financiera producen
estimaciones puntuales de la probabilidad de incumplimiento (PD), la pérdida
dado incumplimiento (LGD) y la exposición en caso de incumplimiento (EAD) sin
cuantificar la incertidumbre asociada a dichas predicciones. Esta limitación
tiene consecuencias directas: **subestimación del riesgo** que genera pérdidas
inesperadas, **provisiones excesivas** por posturas conservadoras ante la
incertidumbre no medida, y **desconfianza regulatoria** por falta de
transparencia en los intervalos de confianza de los modelos.

La **predicción conformal** (*Conformal Prediction*, CP), propuesta
originalmente por Vovk, Gammerman y Shafer (2005), ofrece una solución
rigurosa: intervalos de predicción con **garantías formales de cobertura**
$$P(Y \\in C(X)) \\geq 1 - \\alpha$$ bajo el supuesto de intercambiabilidad de
los datos, sin requerir supuestos paramétricos sobre la distribución
subyacente. Esta propiedad de cobertura *distribution-free* convierte a CP en
una herramienta especialmente atractiva para el sector financiero regulado,
donde la robustez estadística y la auditabilidad son requisitos fundamentales.

Esta tesis implementa y evalúa un pipeline completo de conformal prediction
aplicado a modelos de riesgo crediticio sobre el dataset público de
**LendingClub** (2.26 millones de préstamos, 2007-2020), incluyendo cinco
variantes de intervalos predictivos: **Split Conformal**, **Mondrian Conformal**
(grupo-condicional por grade), **Venn-Abers Predictors**, **Conformalized
Quantile Regression** (CQR) para LGD/EAD, e **intervalos residuales** como
benchmark no-conformal. Los resultados se evalúan mediante tests formales de
backtesting (**Kupiec**, **Christoffersen**), métricas de calibración
(**ECE**, **Brier Score**), eficiencia de intervalos (**MPIW**) y fairness
conformal, y se conectan directamente con el cálculo de provisiones bajo
**IFRS 9** y los requerimientos de **Basilea III**.
"""
    )

    st.markdown("---")
    st.markdown(
        """
### Pregunta de investigación

> **¿Cómo mejorar la calibración y la cuantificación de la incertidumbre en
> los modelos de riesgo crediticio (PD, LGD, EAD) mediante la aplicación de
> técnicas de predicción conformal, y cuál es su desempeño comparado frente a
> métodos tradicionales de calibración?**

### Objetivo general

Implementar y evaluar técnicas de predicción conformal para mejorar la
calibración y cuantificación de la incertidumbre en modelos de riesgo
crediticio (PD, LGD, EAD), utilizando el dataset público de LendingClub
como caso de estudio, y comparar su desempeño frente a métodos tradicionales
de calibración.

### Objetivos específicos

1. **OE1** — Realizar una revisión del estado del arte sobre calibración de
   modelos, cuantificación de incertidumbre y predicción conformal aplicada
   al riesgo crediticio.
2. **OE2** — Construir y preparar un dataset a partir de los registros
   públicos de LendingClub (2007-2020), definiendo las variables objetivo
   (PD, LGD, EAD) y realizando el preprocesamiento necesario.
3. **OE3** — Diseñar e implementar experimentos aplicando técnicas de
   Conformal Prediction (Split Conformal, Mondrian Conformal, Venn-Abers
   Predictors, CQR) sobre modelos base de PD, LGD y EAD.
4. **OE4** — Comparar el desempeño de Conformal Prediction con métodos
   tradicionales de calibración (Platt Scaling, Isotonic Regression,
   intervalos residuales) mediante métricas de calibración, cobertura,
   eficiencia y discriminación.
5. **OE5** — Evaluar el impacto de la predicción conformal en la estimación
   de provisiones bajo IFRS 9, la asignación de capital regulatorio bajo
   Basilea III, y las métricas de backtesting formal (Kupiec, Christoffersen).
6. **OE6** — Proponer lineamientos para la adopción de Conformal Prediction
   en entornos bancarios reales, incluyendo fairness conformal, gobernanza
   y escalabilidad.
"""
    )

    # ── Importance, Status, Competitors ──
    st.markdown("---")
    st.markdown(
        """
### Importancia para la industria y la academia

#### Perspectiva de la industria financiera

El sector de riesgo crediticio opera bajo un dilema fundamental: los modelos
de machine learning han demostrado **mayor poder discriminativo** que los
modelos estadísticos tradicionales (regresión logística), pero adolecen de
**falta de calibración** y **opacidad** — dos características que los
reguladores financieros no pueden tolerar. Las probabilidades de default
estimadas por gradient boosting (CatBoost, XGBoost, LightGBM) no son
necesariamente probabilidades bien calibradas; un modelo que dice
"P(default)=0.15" no garantiza que el 15% de esos préstamos efectivamente
entre en default.

La predicción conformal resuelve este problema de raíz al proveer
**intervalos con cobertura estadística garantizada**. Para una entidad
financiera, esto significa:

- **Pricing de préstamos más preciso**: El intervalo [PD_low, PD_high]
  permite tarifar con márgenes de seguridad informados, no arbitrarios.
- **Provisiones IFRS9 justificables**: El ancho del intervalo conformal
  sirve como señal objetiva de SICR (Significant Increase in Credit Risk),
  mejorando la migración entre Stages 1→2 con criterios auditables.
- **Base para optimización robusta**: Los intervalos conformales proveen
  conjuntos de incertidumbre que pueden alimentar modelos de optimización
  de portafolio en fases posteriores del proyecto.
- **Backtesting regulatorio formal**: Los tests de Kupiec y Christoffersen
  validan que la cobertura prometida se cumple — requisito de Basilea III
  para modelos internos (IRB).

#### Perspectiva académica

La aplicación de predicción conformal al riesgo crediticio es una línea de
investigación **emergente pero aún fragmentada**. La literatura existente
se ha concentrado en PD de forma aislada (Bellotti, 2017; Javanmardi &
Vovk, 2023). Esta tesis contribuye a cerrar varios vacíos:

- **Cobertura conjunta PD + LGD + EAD**: Primera aplicación documentada
  de CP sobre los tres componentes del riesgo crediticio simultáneamente.
- **Mondrian conformal con tuning multi-objetivo**: Extensión del conformal
  grupo-condicional con selección Pareto y cobertura mínima garantizada
  por subgrupo.
- **Fairness conformal**: Evaluación de disparidades en cobertura y ancho
  de intervalos entre grupos demográficos — un ángulo completamente nuevo
  en la intersección de fairness algorítmica y predicción conformal.
- **Base para predict-then-optimize**: Los intervalos conformales generados
  aquí sientan las bases para su integración con optimización robusta de
  portafolio (trabajo futuro en la maestría).

#### Contexto colombiano

En Colombia, la adopción de técnicas avanzadas de cuantificación de
incertidumbre en modelos de riesgo es incipiente. Las entidades financieras
supervisadas por la Superintendencia Financiera de Colombia (SFC) utilizan
modelos de riesgo que cumplen con la normativa local pero generalmente no
incorporan intervalos de confianza con garantías formales. Esta investigación
posiciona la predicción conformal como una herramienta viable para el
contexto regulatorio colombiano, alineada con las mejores prácticas
internacionales de Basilea III e IFRS 9.
"""
    )

    st.markdown("---")
    st.markdown("### Estatus del proyecto")
    st.markdown(
        """
A continuación se responden las cinco preguntas clave de diagnóstico del
estado del proyecto, siguiendo el marco de evaluación propuesto.
"""
    )

    # ── 1. Definición del "Dolor" ──
    with st.container(border=True):
        st.markdown("#### 1. Definición del Dolor (El Problema)")
        st.markdown(
            """
**Pregunta**: *¿Cuál es el problema específico que mi proyecto resuelve y
quién es el dueño de ese problema en la organización?*

**Respuesta**: El problema es la **ausencia de cuantificación formal de
incertidumbre** en los modelos de riesgo crediticio. Los modelos actuales
producen una PD puntual (ej: 0.12) sin indicar si esa estimación es confiable
(intervalo estrecho) o altamente incierta (intervalo amplio). Esto genera:

- **Provisiones mal calibradas** → P&L volátil e inesperado.
- **Decisiones de originación subóptimas** → se rechazan buenos clientes
  o se aprueban malos.
- **Riesgo de incumplimiento regulatorio** → los supervisores exigen
  backtesting formal de los intervalos de confianza.

**Dueño del problema**: Dirección de Riesgo Crediticio (CRO / Chief Risk
Officer), con impacto directo en Comités de Crédito, Provisiones (CFO) y
Validación de Modelos (MRM / Model Risk Management).

**¿Es un problema real?** Sí. Los reguladores internacionales (BIS, EBA) y
locales (SFC Colombia) exigen que los modelos internos demuestren cobertura
adecuada de sus intervalos de confianza. La predicción conformal ofrece
exactamente esto con garantías matemáticas.
"""
        )

    # ── 2. Estatus del "Cerebro" ──
    with st.container(border=True):
        st.markdown("#### 2. Estatus del Cerebro (El Modelo / Analítica)")
        st.markdown(
            """
**Pregunta**: *A hoy, ¿tengo datos reales disponibles y una técnica analítica
(o algoritmo) ya seleccionada?*

**Respuesta**: **Sí, completamente implementado y validado.**

- **Datos**: Dataset LendingClub con **2.26 millones** de préstamos reales
  (2007-2020), con splits temporales estrictos Out-of-Time.
- **Modelos base**: CatBoost (gradient boosting) como modelo principal +
  Regresión Logística como baseline. Tuning con Optuna (400 trials).
- **Técnica conformal**: 5 variantes implementadas (Split, Mondrian,
  Venn-Abers, CQR, Residual) usando MAPIE 1.3.0 y crepes.
- **Calibración**: Selección automática Platt vs Isotonic por validación
  temporal multi-métrica.

El proyecto **no está en fase de ideación** — es un **prototipo experimental
completo** con 418 tests, pipeline DVC de 24 stages, y dashboard Streamlit
de 27 páginas.
"""
        )

    # ── 3. Tangibilidad ──
    with st.container(border=True):
        st.markdown("#### 3. Tangibilidad (El Producto)")
        st.markdown(
            """
**Pregunta**: *¿Cómo se imagina el usuario final consumiendo mi resultado
(un reporte, una API, un dashboard, un mensaje de alerta)?*

**Respuesta**: El resultado se consume en **tres formatos complementarios**:

1. **Dashboard Streamlit** (27 páginas): Interfaz interactiva donde el
   analista de riesgo, el comité de crédito o el validador de modelos
   puede explorar métricas, comparar variantes conformales, visualizar
   intervalos por grade, y auditar fairness — todo en tiempo real.

2. **API REST (FastAPI)**: 15 endpoints para inferencia de PD con intervalos
   conformales, consultas IFRS9, y métricas de portafolio. Permite
   integración programática con sistemas downstream.

3. **Reportes automatizados**: MRM report (SR 11-7), fairness audit,
   backtesting report — generados por scripts y consumidos por el
   área de validación de modelos.

El usuario final ya puede **interactuar con todos los resultados** a
través de este mismo dashboard.
"""
        )

    # ── 4. Madurez Tecnológica ──
    with st.container(border=True):
        st.markdown("#### 4. Madurez Tecnológica (TRL Inicial)")
        st.markdown(
            """
**Pregunta**: *Siendo 1 "solo tengo la idea" y 9 "el sistema ya funciona en
producción", ¿en qué número del 1 al 9 ubico mi proyecto en este momento?*

**Respuesta**: **TRL 6-7 (Prototipo validado en entorno representativo)**
"""
        )

        trl_data = [
            {"TRL": "1-2", "Descripción": "Concepto e idea", "Estado": "Superado"},
            {"TRL": "3", "Descripción": "Prueba de concepto analítica", "Estado": "Superado"},
            {"TRL": "4", "Descripción": "Validación en laboratorio (datos reales)", "Estado": "Superado"},
            {"TRL": "5", "Descripción": "Validación en entorno relevante", "Estado": "Superado"},
            {"TRL": "6", "Descripción": "Prototipo demostrado (dashboard + API + tests)", "Estado": "Actual"},
            {"TRL": "7", "Descripción": "Sistema prototipo en entorno operativo", "Estado": "Parcial"},
            {"TRL": "8-9", "Descripción": "Sistema completo en producción", "Estado": "Pendiente"},
        ]
        st.dataframe(pd.DataFrame(trl_data), width="stretch", hide_index=True)

        st.markdown(
            """
**Justificación del TRL 6-7**:
- Pipeline end-to-end ejecutable y reproducible (DVC 24 stages).
- 418 tests unitarios e integración.
- Dashboard interactivo operativo (Streamlit 27 páginas).
- API REST funcional (FastAPI 15 endpoints).
- CI/CD con GitHub Actions (lint + test + smoke).
- MLflow tracking + DagsHub remote.

**Lo que falta para TRL 8-9**: Despliegue en infraestructura bancaria real,
integración con core bancario, datos de producción protegidos, monitoring
en tiempo real.
"""
        )

    # ── 5. Brecha de Innovación ──
    with st.container(border=True):
        st.markdown("#### 5. Brecha de Innovación")
        st.markdown(
            """
**Pregunta**: *¿Qué es lo que me impide entregar este proyecto mañana
mismo? (¿Falta de datos? ¿Falta de conocimiento técnico? ¿Falta de
infraestructura?)*

**Respuesta**: El proyecto como **tesis de especialización está
prácticamente completo**. Las brechas restantes no son bloqueantes para
la entrega académica:

| Brecha | Tipo | Impacto | Estado |
|---|---|---|---|
| Datos bancarios reales | Datos | Generalización limitada a P2P | Mitigado: LendingClub es el dataset estándar de la literatura |
| Infraestructura cloud | Infra | No hay deploy en producción | No requerido para tesis |
| Conformal adaptativo | Técnico | No maneja concept drift | Documentado como trabajo futuro |
| Validación con regulador | Proceso | No hay feedback de SFC | Fuera de alcance de la tesis |

**Conclusión**: Ninguna de estas brechas impide la entrega. El proyecto
supera ampliamente el alcance original de la propuesta (5 variantes vs
3 propuestas, tuning multi-objetivo, fairness conformal, backtesting
formal Kupiec/Christoffersen).
"""
        )

    # ── Quién más resuelve esto ──
    st.markdown("---")
    st.markdown("### ¿Quién más en el mundo está resolviendo esto y cómo lo hace?")
    st.markdown(
        """
| Autor / Grupo | Año | Enfoque | Diferencia con esta tesis |
|---|---|---|---|
| **Bellotti** (Nottingham) | 2017 | CP para credit scoring (SCP básico) | Solo PD, sin LGD/EAD, sin fairness |
| **Javanmardi & Vovk** (Royal Holloway) | 2023 | Venn-Abers para PD bancario | Solo Venn-Abers, sin Mondrian, sin IFRS9 |
| **Angelopoulos & Bates** (Berkeley) | 2023 | Tutorial general de CP | No aplicado a riesgo crediticio |
| **Romano, Patterson & Candes** (Stanford) | 2019 | CQR para regresión | Método genérico, no aplicado a LGD/EAD |
| **Fontana, Zeni & Vantini** (Politecnico Milano) | 2023 | Revisión unificada de CP | Teórico, sin implementación financiera |
| **MAPIE team** (Quantmetry) | 2023-2025 | Librería MAPIE | Herramienta, no aplicación a crédito |
| **Esta tesis** | 2026 | CP aplicada a riesgo crediticio integral | **5 variantes, PD+LGD+EAD, IFRS9, fairness conformal, Mondrian tuned, backtesting formal** |

**Diferenciadores clave de esta tesis**:
- Única implementación documentada que aplica CP simultáneamente a **PD, LGD y EAD**.
- Primer análisis de **fairness conformal** (disparidad de cobertura entre grupos protegidos).
- **Mondrian conformal con tuning Pareto multi-objetivo** — no existe en la literatura revisada.
- **Backtesting formal** (Kupiec + Christoffersen) aplicado a intervalos conformales.
- Pipeline **reproducible** con 418 tests, DVC, MLflow, CI/CD.
"""
    )

    # ── Objectives-Evidence Mapping (moved to end) ──
    st.markdown("---")
    st.subheader("Mapeo de Objetivos Específicos → Evidencia")
    objectives_data = [
        {
            "Objetivo": "OE1: Revisión estado del arte",
            "Estado": "Completo",
            "Evidencia": "Cap. 5 Marco Teórico (informe) + 25 referencias revisadas",
        },
        {
            "Objetivo": "OE2: Construir dataset LendingClub",
            "Estado": "Completo",
            "Evidencia": "src/data/ (3 módulos), NB01-NB02, 2.26M préstamos, splits OOT",
        },
        {
            "Objetivo": "OE3: Aplicar CP (Split, CQR, Venn-Abers)",
            "Estado": "Completo",
            "Evidencia": "src/models/conformal.py — Split, Mondrian, Venn-Abers, CQR, Residual",
        },
        {
            "Objetivo": "OE4: Comparar CP vs calibración tradicional",
            "Estado": "Completo",
            "Evidencia": "ECE, Brier, coverage, MPIW + benchmark_conformal_variants.py",
        },
        {
            "Objetivo": "OE5: Evaluar impacto regulatorio IFRS9",
            "Estado": "Completo",
            "Evidencia": "src/evaluation/ifrs9.py — ECL con/sin conformal, staging",
        },
        {
            "Objetivo": "OE6: Proponer lineamientos de adopción",
            "Estado": "Completo",
            "Evidencia": "configs/ (7 YAML), MRM report, fairness conformal audit",
        },
    ]
    st.dataframe(pd.DataFrame(objectives_data), width="stretch", hide_index=True)

# ── Tab 1: Dataset & EDA ──
with tabs[1]:
    st.header("Dataset & EDA")

    st.markdown(
        """
### Fuente: LendingClub Loan Data (2007–2020)

El dataset contiene **2.26 millones** de préstamos resueltos del mercado
peer-to-peer más grande de Estados Unidos. Seleccionado por su riqueza,
representatividad y uso extensivo en la literatura de riesgo crediticio.
"""
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Préstamos totales", "2.26M")
    col2.metric("Variables originales", "140+")
    col3.metric("Período", "2007–2020")

    st.markdown("---")
    render_eda_section()

    st.markdown("---")
    st.subheader("Splits Temporales (Out-of-Time)")
    st.markdown(
        """
**Mejora respecto a la propuesta**: la propuesta original planteaba splits
aleatorios 60/20/20. El proyecto implementa **splits temporales estrictos**
(OOT), que son metodológicamente superiores para riesgo crediticio porque
respetan la estructura cronológica y evitan data leakage temporal.
"""
    )

    splits_data = [
        {"Split": "Train", "Filas": "1,346,311", "Default Rate": "18.52%", "Rango": "2007-06 a 2017-03"},
        {"Split": "Calibración", "Filas": "237,584", "Default Rate": "22.20%", "Rango": "2017-03 a 2017-12"},
        {"Split": "Test (OOT)", "Filas": "276,869", "Default Rate": "21.98%", "Rango": "2018-01 a 2020-09"},
    ]
    st.dataframe(pd.DataFrame(splits_data), width="stretch", hide_index=True)

    # ── Splits visualization ──
    fig_splits = go.Figure()
    split_names = ["Train", "Calibración", "Test (OOT)"]
    split_sizes = [1346311, 237584, 276869]
    split_colors = ["#0B5ED7", "#F59F00", "#DC3545"]
    fig_splits.add_trace(go.Bar(
        x=split_names,
        y=split_sizes,
        text=[f"{s:,.0f}" for s in split_sizes],
        textposition="outside",
        marker_color=split_colors,
    ))
    fig_splits.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Tamaño de Cada Split Temporal",
        yaxis_title="Número de Préstamos",
    )
    st.plotly_chart(fig_splits, width="stretch")

    st.subheader("Tres Datasets Analíticos")
    datasets_info = [
        {"Dataset": "loan_master.parquet", "Descripción": "Una fila por préstamo (PD, LGD, supervivencia)", "Uso": "PD, conformal prediction"},
        {"Dataset": "ead_dataset.parquet", "Descripción": "Solo defaults (EAD modeling)", "Uso": "EAD conformal intervals"},
        {"Dataset": "time_series.parquet", "Descripción": "Agregados mensuales (118 filas)", "Uso": "Forecasting temporal"},
    ]
    st.dataframe(pd.DataFrame(datasets_info), width="stretch", hide_index=True)

    st.info(
        "**Data leakage**: Se removieron 15+ variables post-loan "
        "(total_pymnt, recoveries, etc.) en `src/data/make_dataset.py` para "
        "prevenir fugas de información del futuro al modelo."
    )

# ── Tab 2: Modelado Base ──
with tabs[2]:
    st.header("Modelado Base (PD / LGD / EAD)")

    st.markdown(
        """
### Adaptación respecto a la propuesta

La propuesta planteaba LightGBM y XGBoost. El proyecto usa **CatBoost** como
modelo principal de gradient boosting por:

- **Manejo nativo de categóricas**: no requiere encoding manual.
- **Ordered boosting**: reduce overfitting en datos temporales.
- **Manejo nativo de NaN**: no requiere imputación.

Se mantiene **Regresión Logística** como baseline interpretable (coincide con la propuesta).
"""
    )

    st.subheader("Métricas PD (Test OOT)")
    if final_metrics:
        pd_cols = st.columns(4)
        pd_cols[0].metric("AUC-ROC", format_pct(final_metrics.get("auc_roc", 0)))
        pd_cols[1].metric("Gini", format_pct(final_metrics.get("gini", 0)))
        pd_cols[2].metric("Brier Score", f"{final_metrics.get('brier_score', 0):.4f}")
        pd_cols[3].metric("ECE", f"{final_metrics.get('ece', 0):.4f}")

        best_cal = str(comparison.get("best_calibration", "N/A")) if comparison else "N/A"
        st.success(f"**Calibrador seleccionado automáticamente**: {best_cal} (selección por validación temporal multi-métrica)")

        # ── Metrics comparison chart ──
        metric_names = ["AUC-ROC", "Gini", "1 - Brier", "1 - ECE"]
        metric_values = [
            final_metrics.get("auc_roc", 0),
            final_metrics.get("gini", 0),
            1 - final_metrics.get("brier_score", 0),
            1 - final_metrics.get("ece", 0),
        ]
        fig_metrics = go.Figure(go.Bar(
            x=metric_names,
            y=[v * 100 for v in metric_values],
            text=[f"{v * 100:.1f}%" for v in metric_values],
            textposition="outside",
            marker_color=["#0B5ED7", "#198754", "#F59F00", "#6F42C1"],
        ))
        fig_metrics.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Métricas del Modelo PD Calibrado (Test OOT)",
            yaxis_title="% (mayor es mejor)",
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig_metrics, width="stretch")

        st.markdown(
            """
**Interpretación**: AUC-ROC y Gini miden poder discriminativo (capacidad de
separar defaults de no-defaults). Brier Score y ECE miden calibración
(qué tan bien las probabilidades predichas reflejan las frecuencias reales).
Un modelo ideal tiene alto AUC-ROC/Gini Y bajo Brier/ECE. Los valores
"1 - Brier" y "1 - ECE" se muestran para que todas las métricas sean
"mayor es mejor".
"""
        )
    else:
        st.warning("Métricas PD no disponibles. Ejecute el pipeline primero.")

    st.subheader("Modelos LGD y EAD")
    st.markdown(
        """
| Componente | Modelo | Enfoque |
|---|---|---|
| **LGD** | CatBoost (two-stage) | Stage 1: P(LGD>0) clasificador + Stage 2: LGD regressor |
| **EAD** | CatBoost Regressor | Entrenado sobre defaults-only |

Los modelos de LGD y EAD son esenciales para el cálculo de ECL bajo IFRS 9.
Los intervalos conformales CQR se aplican sobre estos modelos para cuantificar
la incertidumbre en cada componente de la fórmula ECL = PD × LGD × EAD.
"""
    )

    st.subheader("Calibración: Platt vs Isotonic")
    st.markdown(
        """
Ambos métodos están implementados (`src/models/calibration.py`).
La selección se hace **automáticamente** mediante validación temporal:
se evalúa ECE, Brier Score y AUC-ROC en folds temporales anclados,
y se selecciona el método que minimiza ECE sin degradar AUC más de 0.15%.

**¿Por qué importa la calibración?** Un modelo con AUC-ROC = 0.85 puede
tener probabilidades mal calibradas (ej: predecir PD=0.30 cuando la tasa
real es 0.20). La calibración ajusta las probabilidades para que reflejen
frecuencias reales, lo cual es prerequisito para que los intervalos
conformales sean informativos.
"""
    )

# ── Tab 3: Conformal Prediction ──
with tabs[3]:
    st.header("Conformal Prediction — Núcleo Experimental")

    st.markdown(
        """
### ¿Qué es Conformal Prediction y por qué importa?

La predicción conformal es un framework matemático que transforma cualquier
modelo de ML en uno que produce **intervalos de predicción con garantías
formales de cobertura**. A diferencia de los intervalos de confianza
tradicionales (bootstrap, bayesianos), los intervalos conformales:

- **No requieren supuestos distribucionales**: funcionan con cualquier modelo.
- **Tienen cobertura garantizada**: P(Y ∈ [low, high]) ≥ 1-α en muestras finitas.
- **Son computacionalmente eficientes**: no requieren re-entrenar el modelo.
- **Son adaptativos**: intervalos más anchos donde el modelo es más incierto.

### Variantes Implementadas

El proyecto implementa **5 variantes** de intervalos predictivos, superando
las 3 propuestas originalmente (Split, CQR, Venn-Abers):
"""
    )

    variants_data = [
        {
            "Variante": "Split Conformal (Global)",
            "Target": "PD",
            "Librería": "MAPIE 1.3.0",
            "Garantía": "Cobertura marginal ≥ 1-α",
        },
        {
            "Variante": "Mondrian Conformal (por Grade)",
            "Target": "PD",
            "Librería": "Custom + MAPIE",
            "Garantía": "Cobertura condicional por grupo ≥ 1-α",
        },
        {
            "Variante": "Venn-Abers Predictors",
            "Target": "PD",
            "Librería": "crepes",
            "Garantía": "Calibración multiprobabilística",
        },
        {
            "Variante": "CQR (Split Conformal Regression)",
            "Target": "LGD / EAD",
            "Librería": "MAPIE 1.3.0",
            "Garantía": "Cobertura marginal ≥ 1-α (regresión)",
        },
        {
            "Variante": "Residual Intervals (Benchmark)",
            "Target": "PD / LGD / EAD",
            "Librería": "NumPy",
            "Garantía": "Sin garantía formal (benchmark)",
        },
    ]
    st.dataframe(pd.DataFrame(variants_data), width="stretch", hide_index=True)

    # ── Visual: How Conformal Prediction works ──
    st.subheader("¿Cómo funciona Split Conformal?")
    st.markdown(
        """
El flujo de Split Conformal Prediction sigue tres pasos:

1. **Entrenar** el modelo base en el conjunto de entrenamiento.
2. **Calibrar** los scores de no-conformidad en un conjunto de calibración
   separado: calcular los residuos |y - ŷ| para cada ejemplo.
3. **Predecir** con intervalos: para un nuevo ejemplo, el intervalo es
   [ŷ - q, ŷ + q] donde q es el quantil (1-α) de los residuos de calibración.

La **garantía matemática** es que si los datos son intercambiables
(condición más débil que i.i.d.), la cobertura empírica será ≥ 1-α.
"""
    )

    st.subheader("Tuning Conformal (Mondrian)")
    st.markdown(
        """
El conformal Mondrian extiende el Split Conformal calculando quantiles
**separados por subgrupo** (en nuestro caso, por grade crediticio). Esto
garantiza cobertura *dentro de cada grade*, no solo globalmente.

El proyecto incluye un sistema sofisticado de tuning conformal:

- **Pareto multi-objetivo**: optimiza simultáneamente cobertura global,
  cobertura mínima por grupo, y ancho de intervalos.
- **Selección jerárquica**: 8 tiers de prioridad para elegir la mejor
  configuración.
- **Multiplicadores por grupo**: ajuste fino de radios conformales por grade.
- **Floor de cobertura por grupo**: garantía de cobertura mínima condicional.
"""
    )

    st.subheader("PD Conformal — Backtesting Temporal")
    if conformal_bt is not None and len(conformal_bt) > 0:
        numeric_bt = conformal_bt.select_dtypes(include="number")

        # ── Coverage over time chart ──
        if "month" in conformal_bt.columns and "coverage_90" in conformal_bt.columns:
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Scatter(
                x=conformal_bt["month"],
                y=conformal_bt["coverage_90"],
                mode="lines+markers",
                name="Cobertura empírica (90%)",
                line={"color": "#0B5ED7", "width": 2},
                marker={"size": 5},
            ))
            if "target_90" in conformal_bt.columns:
                fig_cov.add_hline(
                    y=conformal_bt["target_90"].iloc[0],
                    line_dash="dash",
                    line_color="#DC3545",
                    annotation_text="Target 90%",
                )
            fig_cov.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title="Cobertura Conformal al 90% — Evolución Temporal",
                xaxis_title="Mes",
                yaxis_title="Cobertura Empírica",
                yaxis_range=[0.7, 1.05],
            )
            st.plotly_chart(fig_cov, width="stretch")

            st.markdown(
                """
**Interpretación**: Cada punto representa la cobertura empírica mensual de los
intervalos conformales al 90%. La línea roja punteada es el target (90%).
Idealmente, la cobertura empírica debe estar **siempre por encima** del target.
Fluctuaciones mensuales son normales, pero tendencias descendentes sostenidas
indicarían degradación del modelo (concept drift).
"""
            )

        # ── Width over time chart ──
        if "month" in conformal_bt.columns and "avg_width_90" in conformal_bt.columns:
            fig_width = go.Figure()
            fig_width.add_trace(go.Scatter(
                x=conformal_bt["month"],
                y=conformal_bt["avg_width_90"],
                mode="lines+markers",
                name="Ancho promedio (90%)",
                line={"color": "#198754", "width": 2},
                marker={"size": 5},
            ))
            fig_width.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title="Ancho Promedio de Intervalos (MPIW) — Evolución Temporal",
                xaxis_title="Mes",
                yaxis_title="MPIW (Mean Prediction Interval Width)",
            )
            st.plotly_chart(fig_width, width="stretch")

            st.markdown(
                """
**Interpretación**: El MPIW (Mean Prediction Interval Width) mide la
eficiencia de los intervalos. Intervalos más estrechos son más informativos
(mayor precisión), siempre que mantengan la cobertura requerida. Un aumento
del MPIW puede indicar mayor incertidumbre en el portafolio.
"""
            )

        st.markdown("**Estadísticas descriptivas del backtesting:**")
        st.dataframe(
            numeric_bt.describe().round(4),
            width="stretch",
        )
    else:
        st.info("Datos de backtesting conformal no disponibles. Ejecute `scripts/backtest_conformal_coverage.py`.")

# ── Tab 4: Evaluación Comparativa ──
with tabs[4]:
    st.header("Evaluación Comparativa")

    st.markdown(
        """
### Framework de Evaluación

La evaluación se estructura en tres niveles complementarios, cada uno
respondiendo a una pregunta diferente:

| Nivel | Pregunta | Métricas |
|---|---|---|
| **Calibración** | ¿Las probabilidades predichas reflejan frecuencias reales? | ECE, Brier Score |
| **Cobertura** | ¿Los intervalos contienen el valor real con la frecuencia prometida? | Coverage, Kupiec, Christoffersen |
| **Eficiencia** | ¿Los intervalos son lo más estrechos posible? | MPIW |
"""
    )

    st.subheader("Métricas de Evaluación")
    st.markdown(
        """
| Métrica | Fórmula | Propósito |
|---|---|---|
| **ECE** | Expected Calibration Error (10 bins) | Calibración de probabilidades |
| **Brier Score** | Mean squared error de probabilidades | Calibración + discriminación |
| **Coverage** | P(y ∈ [low, high]) | Cobertura empírica vs nominal |
| **MPIW** | Mean Prediction Interval Width | Eficiencia de intervalos |
| **Kupiec POF** | LR test contra χ²(1) | Cobertura incondicional (Basel) |
| **Christoffersen** | LR test contra χ²(2) | Cobertura condicional + independencia |
"""
    )

    st.subheader("Tests Formales de Backtesting")
    st.markdown(
        """
Los tests de **Kupiec (1995)** y **Christoffersen (1998)** son requerimientos
estándar de validación bajo Basilea III para modelos internos de riesgo.

**Kupiec POF Test**: Evalúa si la tasa de violaciones observada es
consistente con la tasa nominal α. Bajo H₀: la tasa de violaciones es
exactamente α. Si el p-value > 0.05, no rechazamos: la cobertura es
adecuada.

**Christoffersen Test**: Combina la prueba de cobertura incondicional
con una prueba de independencia temporal de las violaciones.
Rechazar la independencia indica clustering de violaciones — es decir,
los fallos del modelo no son aleatorios sino que ocurren en rachas,
lo cual es especialmente peligroso en riesgo crediticio (ej: crisis).
"""
    )

    # ── Visual comparison if backtest data available ──
    if conformal_bt is not None and len(conformal_bt) > 0:
        st.subheader("Comparación: Cobertura 90% vs 95%")
        if all(c in conformal_bt.columns for c in ["coverage_90", "coverage_95"]):
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Box(
                y=conformal_bt["coverage_90"],
                name="Cobertura 90%",
                marker_color="#0B5ED7",
            ))
            fig_comp.add_trace(go.Box(
                y=conformal_bt["coverage_95"],
                name="Cobertura 95%",
                marker_color="#198754",
            ))
            fig_comp.add_hline(y=0.90, line_dash="dash", line_color="#0B5ED7",
                               annotation_text="Target 90%")
            fig_comp.add_hline(y=0.95, line_dash="dash", line_color="#198754",
                               annotation_text="Target 95%")
            fig_comp.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title="Distribución de Cobertura Mensual (90% vs 95%)",
                yaxis_title="Cobertura Empírica",
                yaxis_range=[0.7, 1.05],
                showlegend=False,
            )
            st.plotly_chart(fig_comp, width="stretch")

            st.markdown(
                """
**Interpretación**: Los box plots muestran la distribución de la cobertura
empírica mensual. La mediana debe estar **por encima** de la línea punteada
correspondiente. Baja varianza indica estabilidad del modelo en el tiempo.
"""
            )

    st.markdown(
        """
### Código de Ejemplo

```python
from src.evaluation.backtesting import kupiec_pof_test, christoffersen_test

# violations: binary array (1 = y_true outside interval)
kupiec = kupiec_pof_test(violations, alpha=0.10)
christo = christoffersen_test(violations, alpha=0.10)

print(f"Kupiec p-value: {kupiec['p_value']:.4f}")
print(f"Christoffersen p-value (joint): {christo['p_cc']:.4f}")
```
"""
    )

# ── Tab 5: Impacto IFRS9 ──
with tabs[5]:
    st.header("Impacto Regulatorio — IFRS9")

    st.markdown(
        """
### ¿Qué es IFRS 9 y cómo se conecta con Conformal Prediction?

**IFRS 9** (International Financial Reporting Standard 9) es la norma
contable que define cómo las instituciones financieras deben calcular y
reportar las provisiones por pérdidas crediticias esperadas (ECL).

La fórmula central es: **ECL = PD × LGD × EAD × Discount Factor**

El estándar define tres stages de deterioro crediticio:
"""
    )

    staging_data = [
        {"Stage": "Stage 1", "Trigger": "Sin SICR", "PD Usada": "12-month PD", "Provisión": "ECL a 12 meses"},
        {"Stage": "Stage 2", "Trigger": "SICR detectado", "PD Usada": "Lifetime PD", "Provisión": "ECL lifetime (mayor)"},
        {"Stage": "Stage 3", "Trigger": "Credit-impaired (90+ DPD)", "PD Usada": "PD ≈ 1.0", "Provisión": "ECL lifetime máximo"},
    ]
    st.dataframe(pd.DataFrame(staging_data), width="stretch", hide_index=True)

    st.subheader("Enhanced SICR con Conformal Prediction")
    st.markdown(
        """
La contribución principal de esta tesis al marco IFRS 9 es usar el **ancho
del intervalo conformal** (PD_high - PD_point) como señal adicional de
**SICR** (Significant Increase in Credit Risk):

- Un préstamo con PD_point = 0.10 y intervalo estrecho [0.08, 0.12] es
  **poco incierto** → probablemente Stage 1.
- Un préstamo con PD_point = 0.10 pero intervalo amplio [0.03, 0.25] es
  **muy incierto** → candidato a migrar a Stage 2, incluso con la misma PD
  puntual.

Esto permite detectar deterioro crediticio **antes** de que se manifieste
en indicadores tradicionales (días de mora, calificación).
"""
    )

    # ── IFRS9 visualization ──
    if ifrs9_summary is not None and len(ifrs9_summary) > 0:
        st.subheader("Resumen de Escenarios IFRS9")
        st.dataframe(ifrs9_summary.head(20), width="stretch", hide_index=True)

        # Try to visualize ECL by stage if the columns exist
        if "Stage" in ifrs9_summary.columns or "stage" in ifrs9_summary.columns:
            stage_col = "Stage" if "Stage" in ifrs9_summary.columns else "stage"
            stage_counts = ifrs9_summary[stage_col].value_counts().sort_index()
            fig_stage = px.bar(
                x=stage_counts.index.astype(str),
                y=stage_counts.values,
                labels={"x": "Stage IFRS9", "y": "Número de Préstamos"},
                title="Distribución por Stage IFRS9",
                text=stage_counts.values,
                color=stage_counts.index.astype(str),
                color_discrete_map={"1": "#198754", "2": "#F59F00", "3": "#DC3545"},
            )
            fig_stage.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
            fig_stage.update_traces(textposition="outside")
            st.plotly_chart(fig_stage, width="stretch")
    else:
        st.info("Datos IFRS9 no disponibles. Ejecute `scripts/run_ifrs9_sensitivity.py`.")

    st.markdown(
        """
### ECL Ranges con Conformal

Con los intervalos conformales se computa un **rango de ECL**:

| Escenario | PD usada | Interpretación |
|---|---|---|
| ECL_low | PD_low (límite inferior conformal) | Escenario optimista |
| ECL_point | PD_point (calibrada) | Estimación central |
| ECL_high | PD_high (límite superior conformal) | Escenario conservador |

Esto permite a la Dirección de Riesgos reportar provisiones con **rangos
de incertidumbre cuantificados**, en lugar de un solo número puntual.
"""
    )

# ── Tab 6: Fairness ──
with tabs[6]:
    st.header("Fairness y Gobernanza")

    st.markdown(
        """
### ¿Por qué evaluar fairness en intervalos conformales?

La predicción conformal garantiza cobertura *marginal* (en promedio sobre
toda la población), pero esta garantía podría **no mantenerse de forma
equitativa** entre subgrupos. Un modelo conformal que cubre el 90% global
podría cubrir 95% para un grupo y solo 80% para otro — violando principios
de fairness algorítmica.

Esta tesis introduce el concepto de **fairness conformal**: evaluar si los
intervalos de predicción son equitativos entre grupos protegidos, tanto
en cobertura como en ancho.

### Métricas Tradicionales de Fairness
"""
    )

    fairness_metrics = [
        {"Métrica": "DPD (Demographic Parity Difference)", "Umbral": "≤ 0.10", "Qué mide": "Diferencia en tasa de positivos entre grupos"},
        {"Métrica": "EO Gap (Equalized Odds)", "Umbral": "≤ 0.10", "Qué mide": "Diferencia en TPR/FPR entre grupos"},
        {"Métrica": "DIR (Disparate Impact Ratio)", "Umbral": "≥ 0.80", "Qué mide": "Regla de los 4/5 (EEOC)"},
    ]
    st.dataframe(pd.DataFrame(fairness_metrics), width="stretch", hide_index=True)

    if fairness_data is not None and len(fairness_data) > 0:
        st.subheader("Auditoría de Fairness")
        st.dataframe(fairness_data, width="stretch", hide_index=True)

        # Try to visualize fairness metrics
        numeric_fair = fairness_data.select_dtypes(include="number")
        if len(numeric_fair.columns) > 0:
            fig_fair = px.bar(
                fairness_data,
                x=fairness_data.columns[0],
                y=numeric_fair.columns[:3].tolist(),
                barmode="group",
                title="Métricas de Fairness por Grupo",
            )
            fig_fair.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_fair, width="stretch")
    else:
        st.info("Datos de fairness no disponibles. Ejecute `scripts/run_fairness_audit.py`.")

    st.subheader("Fairness Conformal (Contribución Original)")
    st.markdown(
        """
Además de la auditoría tradicional, el proyecto evalúa si los intervalos
conformales exhiben **disparidades de cobertura o ancho** entre grupos
protegidos:

| Métrica Conformal | Umbral | Qué mide |
|---|---|---|
| **Coverage Disparity** | ≤ 0.05 | Max coverage - Min coverage entre grupos |
| **Width Ratio** | ≤ 2.0 | Max avg_width / Min avg_width entre grupos |

**¿Por qué es una contribución original?** La intersección entre fairness
algorítmica y predicción conformal es un área de investigación incipiente.
No existen trabajos publicados que evalúen sistemáticamente si las garantías
de cobertura conformal se mantienen equitativamente entre subgrupos
demográficos en modelos de riesgo crediticio.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 7: CRISP-DM
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Metodología CRISP-DM")
    st.markdown(
        """
Esta sección presenta los resultados y decisiones del proyecto siguiendo el
flujo estándar de la metodología **CRISP-DM** (*Cross-Industry Standard
Process for Data Mining*), adaptada al contexto de riesgo crediticio con
predicción conformal. Cada fase documenta qué se hizo, qué retos se
enfrentaron y qué resultados se obtuvieron.
"""
    )

    # ── Phase 1: Business Understanding ──
    st.markdown("---")
    st.subheader("1. Comprensión del Negocio (*Business Understanding*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

Se identificó el problema central del riesgo crediticio: los modelos de ML
producen estimaciones puntuales de PD, LGD y EAD **sin cuantificar la
incertidumbre** asociada. Esto impacta directamente en:

- **Provisiones IFRS 9** mal calibradas (sobre o sub-provisión).
- **Decisiones de originación** sin margen de seguridad cuantificado.
- **Validación regulatoria** (Basilea III) que exige backtesting formal de
  intervalos de confianza.

#### Definición del problema de negocio

Se formuló la pregunta de investigación:

> *¿Cómo mejorar la calibración y cuantificación de la incertidumbre en
> modelos de riesgo crediticio mediante predicción conformal?*

#### Criterios de éxito definidos

| Criterio | Métrica | Umbral |
|---|---|---|
| Cobertura conformal | Coverage empírica al 90% nominal | ≥ 88% |
| Eficiencia de intervalos | MPIW (ancho promedio) | Minimizar |
| Calibración PD | ECE (Expected Calibration Error) | ≤ 0.05 |
| Backtesting formal | Kupiec / Christoffersen p-value | > 0.05 (no rechazo) |
| Fairness conformal | Coverage disparity entre grupos | ≤ 0.05 |
| Cobertura grupo-condicional | Mondrian coverage mínima por grade | ≥ 85% |

#### Retos

- Definir métricas de éxito que fueran **simultáneamente relevantes** para
  el negocio (provisiones, pricing), la regulación (backtesting Basilea) y
  la academia (calibración, cobertura CP).
- Alinear un framework de investigación académica (tesis de especialización)
  con entregables prácticos y operativos.
"""
        )

    # ── Phase 2: Data Understanding ──
    st.markdown("---")
    st.subheader("2. Comprensión de los Datos (*Data Understanding*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

Se seleccionó el dataset de **LendingClub** (2.26M préstamos, 2007-2020)
como caso de estudio, por ser:

- El dataset de referencia más grande y público para riesgo crediticio P2P.
- Ampliamente usado en la literatura (Lessmann et al., 2015; Baesens et al., 2016).
- Suficientemente rico (140+ variables) para modelar PD, LGD y EAD.
"""
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Préstamos", "2.26M")
        col2.metric("Variables originales", "140+")
        col3.metric("Default rate global", "~20%")

        st.markdown(
            """
#### Hallazgos del EDA

- **Tasa de default**: 18-22% según período (no estacionaria).
- **Variables categóricas dominantes**: `grade`, `sub_grade`, `home_ownership`,
  `purpose`, `emp_length` — CatBoost las maneja nativamente.
- **Valores faltantes**: Concentrados en variables de empleo y crédito
  secundario — CatBoost los maneja sin imputación.
- **Variables leaky identificadas**: 15+ variables post-loan que contienen
  información del futuro (`total_pymnt`, `recoveries`, `last_pymnt_d`, etc.).

#### Tres datasets analíticos construidos

| Dataset | Filas | Propósito |
|---|---|---|
| `loan_master.parquet` | 2.26M | PD, supervivencia |
| `ead_dataset.parquet` | ~418K defaults | Modelado EAD |
| `time_series.parquet` | 118 meses | Forecasting temporal |

#### Retos

- **Identificación de data leakage**: Requirió análisis detallado de cada
  variable para determinar si contenía información posterior al momento
  de originación. Error en esta fase habría invalidado todos los resultados.
- **Default rate no estacionaria**: La tasa de default varía significativamente
  entre períodos (18.5% train vs 22.2% calibración), lo que justifica la
  decisión de usar splits temporales en lugar de aleatorios.
"""
        )

    # ── Phase 3: Data Preparation ──
    st.markdown("---")
    st.subheader("3. Preparación de los Datos (*Data Preparation*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

**Splits temporales Out-of-Time (OOT)** — decisión metodológica fundamental
que mejora la propuesta original (splits aleatorios 60/20/20):
"""
        )

        splits_data = [
            {"Split": "Train", "Filas": "1,346,311", "Default Rate": "18.52%", "Período": "2007-06 → 2017-03"},
            {"Split": "Calibración", "Filas": "237,584", "Default Rate": "22.20%", "Período": "2017-03 → 2017-12"},
            {"Split": "Test (OOT)", "Filas": "276,869", "Default Rate": "21.98%", "Período": "2018-01 → 2020-09"},
        ]
        st.dataframe(pd.DataFrame(splits_data), width="stretch", hide_index=True)

        st.markdown(
            """
#### Feature engineering

- **Variables WOE** (Weight of Evidence) via OptBinning — transformación
  estándar de la industria crediticia para variables continuas.
- **Validación con Pandera schemas** — contratos de datos formales para
  detectar drift y anomalías en el pipeline.
- **Feature contract persistido** en `models/pd_model_contract.json` —
  garantiza que todos los modelos downstream usan las mismas features.

#### Retos

- **Balance entre splits**: El conjunto de calibración (237K) es
  suficientemente grande para conformal prediction (MAPIE recomienda ≥1000),
  pero no tan grande como para reducir excesivamente el conjunto de
  entrenamiento.
- **Tres conjuntos disjuntos obligatorios**: Conformal prediction requiere
  un conjunto de calibración **separado** del entrenamiento y del test.
"""
        )

    # ── Phase 4: Modeling ──
    st.markdown("---")
    st.subheader("4. Modelado (*Modeling*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

Se implementaron tres capas de modelado para esta tesis, cada una
construyendo sobre la anterior:

**Capa 1 — Modelos base (PD, LGD, EAD)**

| Modelo | Target | Enfoque |
|---|---|---|
| Regresión Logística | PD | Baseline interpretable |
| CatBoost Default | PD | Gradient boosting sin tuning |
| CatBoost Tuned | PD | HPO con Optuna (400 trials) |
| CatBoost Two-Stage | LGD | Clasificador + Regresor |
| CatBoost Regressor | EAD | Defaults-only |
"""
        )

        if final_metrics:
            st.markdown("**Métricas PD (Test OOT):**")
            m_cols = st.columns(4)
            m_cols[0].metric("AUC-ROC", format_pct(final_metrics.get("auc_roc", 0)))
            m_cols[1].metric("Gini", format_pct(final_metrics.get("gini", 0)))
            m_cols[2].metric("Brier Score", f"{final_metrics.get('brier_score', 0):.4f}")
            m_cols[3].metric("ECE", f"{final_metrics.get('ece', 0):.4f}")

        st.markdown(
            """
**Capa 2 — Calibración**

Selección automática entre Platt Scaling e Isotonic Regression mediante
validación temporal multi-métrica.

**Capa 3 — Predicción conformal (5 variantes)**

| Variante | Target | Garantía de cobertura |
|---|---|---|
| Split Conformal Global | PD | P(y ∈ C(x)) ≥ 1-α marginal |
| Mondrian por Grade | PD | P(y ∈ C(x) \\| g) ≥ 1-α por grupo |
| Venn-Abers | PD | Calibración multiprobabilística |
| CQR (MAPIE) | LGD, EAD | P(y ∈ C(x)) ≥ 1-α regresión |
| Residual (benchmark) | PD, LGD, EAD | Sin garantía formal |

#### Retos

- **MAPIE 1.3.0 API breaking changes**: La migración de `MapieRegressor` a
  `SplitConformalRegressor` requirió reescribir toda la interfaz conformal.
- **ProbabilityRegressor wrapper**: MAPIE espera un regresor, pero CatBoost
  es un clasificador. Se creó un wrapper que expone `predict_proba()[:, 1]`
  como `predict()`.
- **crepes API**: `WrapClassifier.predict_p()` retorna `ndarray(n, 2)`,
  no un tuple — descubierto en testing, no documentado claramente.
"""
        )

    # ── Phase 5: Evaluation ──
    st.markdown("---")
    st.subheader("5. Evaluación (*Evaluation*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

Evaluación integral en cuatro dimensiones:

**Dimensión 1 — Calibración y discriminación**: AUC-ROC, Gini, Brier Score, ECE.

**Dimensión 2 — Intervalos conformales**: Coverage empírica, MPIW, Kupiec
POF test, Christoffersen test.

**Dimensión 3 — Impacto regulatorio (IFRS 9)**: ECL con/sin conformal,
distribución por Stage, sensibilidad a escenarios.

**Dimensión 4 — Fairness**: DPD, EO gap, DIR, Coverage Disparity, Width Ratio.
"""
        )

        if conformal_bt is not None and len(conformal_bt) > 0:
            st.markdown("**Estadísticas de backtesting temporal:**")
            st.dataframe(
                conformal_bt.select_dtypes(include="number").describe().round(4),
                width="stretch",
            )

        st.markdown(
            """
#### Retos

- **Evaluar 5 variantes con métricas comparables**: Cada variante tiene
  propiedades distintas (marginal vs condicional, PD vs regresión).
- **Kupiec/Christoffersen con muestras pequeñas**: Para grupos individuales
  (ej: Grade G), el tamaño de muestra puede ser insuficiente para los tests
  chi-cuadrado.
"""
        )

    # ── Phase 6: Deployment ──
    st.markdown("---")
    st.subheader("6. Despliegue (*Deployment*)")
    with st.container(border=True):
        st.markdown(
            """
#### Qué hicimos

El alcance de la tesis de especialización no incluye despliegue en
producción bancaria. Sin embargo, se implementaron elementos clave de
reproducibilidad y presentación de resultados:

| Componente | Tecnología | Estado |
|---|---|---|
| **Esta página Streamlit** | Streamlit (autocontenida) | Operativo |
| **Informe MD** | Markdown (reports/tesis_especializacion.md) | Completo |
| **Pipeline reproducible** | DVC (24 stages) | Operativo |
| **Experiment tracking** | MLflow + DagsHub | Operativo |
| **CI/CD** | GitHub Actions (lint + test + smoke) | Operativo |
| **Test suite** | pytest (418 tests) | 100% passing |

**Modelo canónico como contrato**

El modelo canónico (`models/pd_canonical.cbm`) junto con su calibrador
(`models/pd_canonical_calibrator.pkl`) y contrato de features
(`models/pd_model_contract.json`) forman una **unidad inmutable** que
garantiza consistencia entre entrenamiento e inferencia conformal.
"""
        )

    # ── CRISP-DM Summary ──
    st.markdown("---")
    st.subheader("Resumen CRISP-DM")
    st.markdown(
        """
El ciclo CRISP-DM se ejecutó de forma **iterativa**, no lineal. Las flechas
bidireccionales entre fases reflejan la realidad del proyecto:

- **Business ↔ Data Understanding**: La identificación de data leakage
  (Data Understanding) reformuló los criterios de éxito (Business).
- **Data Preparation ↔ Modeling**: El descubrimiento de que CatBoost
  maneja categóricas y NaN nativamente simplificó la preparación.
- **Modeling ↔ Evaluation**: Los resultados de backtesting (Kupiec/
  Christoffersen) retroalimentaron el tuning de los parámetros conformales.
- **Evaluation → Deployment**: Las métricas de fairness conformal generaron
  un nuevo requisito de gobernanza que se incorporó en el MRM report.
"""
    )

    crisp_summary = [
        {
            "Fase CRISP-DM": "1. Business Understanding",
            "Entregable principal": "Pregunta de investigación + criterios de éxito",
            "Artefacto clave": "configs/*.yaml (7 archivos de política)",
        },
        {
            "Fase CRISP-DM": "2. Data Understanding",
            "Entregable principal": "EDA completo + identificación data leakage",
            "Artefacto clave": "notebooks/01_eda_lending_club.ipynb",
        },
        {
            "Fase CRISP-DM": "3. Data Preparation",
            "Entregable principal": "3 datasets + splits OOT + features WOE",
            "Artefacto clave": "data/processed/*.parquet + feature_config.pkl",
        },
        {
            "Fase CRISP-DM": "4. Modeling",
            "Entregable principal": "5 modelos base + 5 variantes conformal",
            "Artefacto clave": "models/pd_canonical.cbm + conformal intervals",
        },
        {
            "Fase CRISP-DM": "5. Evaluation",
            "Entregable principal": "Backtesting, IFRS9, fairness, métricas",
            "Artefacto clave": "model_comparison.json + backtesting reports",
        },
        {
            "Fase CRISP-DM": "6. Deployment",
            "Entregable principal": "Página Streamlit + Informe MD + DVC pipeline",
            "Artefacto clave": "tesis_especializacion.py + tesis_especializacion.md",
        },
    ]
    st.dataframe(pd.DataFrame(crisp_summary), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 8: Conclusiones
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("Conclusiones y Contribuciones")

    st.markdown(
        """
### Contribuciones principales

1. **5 variantes de intervalos conformales para riesgo crediticio**:
   Split Conformal, Mondrian by Grade, Venn-Abers, CQR (LGD/EAD),
   Residual (benchmark) — vs. 3 planteadas en la propuesta original.

2. **Cobertura conjunta PD + LGD + EAD**: Primera aplicación documentada
   de conformal prediction sobre los tres componentes del riesgo crediticio
   simultáneamente.

3. **Tuning conformal multi-objetivo (Mondrian)**: Optimización Pareto +
   selección jerárquica + multiplicadores por grupo + floor de cobertura —
   va más allá de la propuesta original y no existe en la literatura revisada.

4. **Enhanced SICR con incertidumbre conformal**: La amplitud del intervalo
   conformal como señal de SICR para staging IFRS9 (contribución original).

5. **Backtesting formal (Kupiec + Christoffersen)**: Tests estadísticos
   estándar de Basilea III aplicados a intervalos conformales.

6. **Fairness conformal**: Auditoría de disparidades en cobertura y ancho
   de intervalos por grupo protegido (contribución original).

### Limitaciones

- Dataset peer-to-peer (no bancario tradicional) — generalización limitada.
- Conformal coverage es marginal, no condicional (excepto Mondrian).
- LGD/EAD usan solo defaults (~18% de la población) — tamaño de calibración menor.
- Intercambiabilidad asumida en splits temporales (asunción aproximada).

### Trabajo Futuro y Proyección hacia la Maestría

Esta tesis de especialización sienta las bases para un trabajo de mayor
alcance en la **maestría**, donde se planea:

- **Pipeline predict-then-optimize**: Los intervalos conformales generados
  en esta tesis pueden servir como **conjuntos de incertidumbre** para
  modelos de optimización robusta de portafolio. La idea central es:
  en lugar de optimizar usando solo la PD puntual, usar el rango
  [PD_low, PD_high] para encontrar portafolios que sean óptimos bajo
  *cualquier* realización de PD dentro del intervalo conformal.
  Esto conecta la cuantificación de incertidumbre (esta tesis) con la
  toma de decisiones óptima bajo incertidumbre (maestría), usando
  herramientas como Pyomo y el solver HiGHS.
- **Inferencia causal**: Integrar efectos causales (CATE) con los
  intervalos conformales para decisiones de tratamiento crediticio.
- **Adaptive conformal prediction** para manejo de concept drift.
- Conformalized Quantile Regression con LightGBM como base (comparación).
- Aplicación a datasets bancarios reales (con datos protegidos).
- Integración con online learning para actualización en tiempo real.
"""
    )

    st.subheader("Stack Tecnológico")
    tech_data = [
        {"Categoría": "ML", "Herramientas": "CatBoost, scikit-learn, Optuna, SHAP"},
        {"Categoría": "Conformal", "Herramientas": "MAPIE 1.3.0, crepes"},
        {"Categoría": "Evaluación", "Herramientas": "scipy.stats (Kupiec, Christoffersen), Pandera"},
        {"Categoría": "MLOps", "Herramientas": "DVC, MLflow, DagsHub"},
        {"Categoría": "Dashboard", "Herramientas": "Streamlit"},
        {"Categoría": "Dev", "Herramientas": "uv, ruff, pytest, pre-commit"},
    ]
    st.dataframe(pd.DataFrame(tech_data), width="stretch", hide_index=True)
