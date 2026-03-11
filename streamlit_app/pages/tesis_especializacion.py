"""Pagina integral de tesis de especializacion con trazabilidad artefactual."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from streamlit_app.components.conformal_applied_blocks import (
    build_cp_concept_matrix_rows,
    build_cp_evidence_ladder_rows,
    build_cp_method_menu_rows,
    render_cp_guarantees_and_limits,
    render_exchangeability_stress_checklist,
)
from streamlit_app.components.context_help import methodology_dialog
from streamlit_app.components.decision_panels import tradeoff_panel
from streamlit_app.components.narrative import claim_evidence_implication, storytelling_intro
from streamlit_app.components.story_shell import (
    render_decision_box,
    render_key_takeaway,
    render_page_header,
    render_section_checkpoint,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.content.tesis_especializacion_context import (
    EVALUATOR_COMMENTS,
    GENERAL_OBJECTIVE,
    RESEARCH_QUESTION,
    SPECIFIC_OBJECTIVES,
    THESIS_TITLE,
)
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    format_number,
    format_pct,
    try_load_json,
    try_load_parquet,
    try_load_report_json,
    page_error_boundary,
)

# Section tokens are kept for narrative contract checks but intentionally not rendered in UI.

with page_error_boundary("tesis_especializacion"):
    NARRATIVE_SECTION_IDS: tuple[str, ...] = (
        "intro_eda",
        "dataset",
        "modelado_base",
        "conformal_prediction",
        "evaluacion_comparativa",
        "impacto_ifrs9",
        "fairness_secundario",
        "crisp_dm",
        "conclusiones",
    )


    def _status_badge(passed: bool) -> str:
        return "PASS" if passed else "WARN/FAIL"


    def _to_dataframe(value: object) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        return pd.DataFrame()


    def _path_status(paths: list[str]) -> str:
        existing = [Path(path).exists() for path in paths]
        if existing and all(existing):
            return "Completo"
        if any(existing):
            return "Parcial"
        return "Pendiente"


    # ---------------------------------------------------------------------------
    # Pipeline conformal flow diagram (Chapter 5)
    # ---------------------------------------------------------------------------
    _TESIS_NODES = [
        {
            "id": "raw",
            "label": "Lending Club Raw",
            "layer": "data",
            "detail": "2.93M filas \u00b7 142 cols",
        },
        {
            "id": "clean",
            "label": "Limpieza + Splits OOT",
            "layer": "data",
            "detail": "Train/Cal/Test temporales",
        },
        {
            "id": "fe",
            "label": "Feature Engineering",
            "layer": "data",
            "detail": "WOE/IV \u00b7 60 features",
        },
        {
            "id": "pd",
            "label": "CatBoost PD + Calibraci\u00f3n",
            "layer": "model",
            "detail": "Platt/Isotonic \u00b7 AUC OOT",
        },
        {
            "id": "conformal",
            "label": "Conformal Mondrian",
            "layer": "model",
            "detail": "Intervalos [PD_low, PD_high]",
        },
        {
            "id": "eval",
            "label": "Evaluaci\u00f3n: Coverage + Backtest",
            "layer": "eval",
            "detail": "35 meses \u00b7 7 grades",
        },
        {
            "id": "ifrs9",
            "label": "Impacto IFRS9: ECL + SICR",
            "layer": "eval",
            "detail": "4 escenarios \u00b7 3 stages",
        },
    ]
    _TESIS_EDGES = [
        {"source": "raw", "target": "clean", "desc": "Filtro + OOT"},
        {"source": "clean", "target": "fe", "desc": "WOE/IV"},
        {"source": "fe", "target": "pd", "desc": "60 features"},
        {"source": "pd", "target": "conformal", "desc": "PD calibrada"},
        {"source": "conformal", "target": "eval", "desc": "Cobertura"},
        {"source": "conformal", "target": "ifrs9", "desc": "Uncertainty"},
    ]
    _TESIS_LAYER_COLORS = {
        "data": {"bg": "#F1F5F9", "border": "#64748B", "accent": "#334155"},
        "model": {"bg": "#EFF6FF", "border": "#3B82F6", "accent": "#1D4ED8"},
        "eval": {"bg": "#ECFDF5", "border": "#10B981", "accent": "#047857"},
    }
    _TESIS_LAYER_BADGES = {
        "data": "DATOS",
        "model": "MODELO",
        "eval": "EVALUACI\u00d3N",
    }
    _TESIS_POSITIONS = {
        "raw": (20, 100),
        "clean": (260, 100),
        "fe": (500, 100),
        "pd": (740, 100),
        "conformal": (980, 100),
        "eval": (1220, 50),
        "ifrs9": (1220, 210),
    }
    _TESIS_NODE_MAP = {n["id"]: n for n in _TESIS_NODES}
    _TESIS_FLOW_STATE_KEY = "tesis_pipeline_flow_state"


    def _render_conformal_pipeline_flow() -> None:
        """Render the specialization-scoped pipeline flow diagram."""
        flow_nodes = []
        for n in _TESIS_NODES:
            lc = _TESIS_LAYER_COLORS[n["layer"]]
            ntype = "input" if n["id"] == "raw" else ("output" if n["id"] == "ifrs9" else "default")
            flow_nodes.append(
                StreamlitFlowNode(
                    id=n["id"],
                    pos=_TESIS_POSITIONS[n["id"]],
                    data={"content": f"**{n['label']}**\n\n{n['detail']}"},
                    node_type=ntype,
                    source_position="right",
                    target_position="left",
                    style={
                        "background": lc["bg"],
                        "border": f"2.5px solid {lc['border']}",
                        "borderRadius": "10px",
                        "padding": "10px 14px",
                        "fontSize": "12px",
                        "fontFamily": "Inter, Arial, sans-serif",
                        "color": lc["accent"],
                        "width": 195,
                        "textAlign": "center",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                    },
                )
            )
        flow_edges = []
        for i, e in enumerate(_TESIS_EDGES):
            src_layer = _TESIS_NODE_MAP[e["source"]]["layer"]
            color = _TESIS_LAYER_COLORS[src_layer]["border"]
            flow_edges.append(
                StreamlitFlowEdge(
                    id=f"te{i}",
                    source=e["source"],
                    target=e["target"],
                    edge_type="smoothstep",
                    label=e["desc"],
                    label_show_bg=True,
                    label_bg_style={
                        "fill": "#FFFFFF",
                        "fillOpacity": 0.92,
                        "stroke": color,
                        "strokeWidth": 0.5,
                        "rx": 4,
                        "ry": 4,
                    },
                    label_style={
                        "fontSize": "10px",
                        "fontFamily": "Inter, Arial, sans-serif",
                        "fontWeight": "600",
                        "fill": color,
                    },
                    marker_end={"type": "arrowclosed"},
                    animated=False,
                    style={"stroke": color, "strokeWidth": 2, "opacity": 0.7},
                )
            )

        if _TESIS_FLOW_STATE_KEY not in st.session_state:
            st.session_state[_TESIS_FLOW_STATE_KEY] = StreamlitFlowState(
                nodes=flow_nodes, edges=flow_edges
            )

        state = streamlit_flow(
            "tesis_pipeline_flow",
            st.session_state[_TESIS_FLOW_STATE_KEY],
            layout=ManualLayout(),
            fit_view=True,
            show_minimap=False,
            show_controls=True,
            allow_zoom=True,
            pan_on_drag=True,
            hide_watermark=True,
            height=380,
            get_node_on_click=True,
            style={"backgroundColor": "#FAFBFC"},
        )
        st.session_state[_TESIS_FLOW_STATE_KEY] = state
        selected = state.selected_id
        if selected and selected in _TESIS_NODE_MAP:
            sel = _TESIS_NODE_MAP[selected]
            st.info(f"**{sel['label']}** \u2014 {sel['detail']}")

        legend_cols = st.columns(3)
        for col, (lk, lv) in zip(legend_cols, _TESIS_LAYER_COLORS.items(), strict=False):
            with col:
                st.markdown(
                    f'<div style="background:{lv["bg"]}; border:2px solid {lv["border"]}; '
                    f"border-radius:8px; padding:6px 10px; text-align:center; font-size:13px; "
                    f'color:{lv["accent"]}; font-weight:600;">{_TESIS_LAYER_BADGES[lk]}</div>',
                    unsafe_allow_html=True,
                )


    def _render_story_block(title: str, body: str) -> None:
        st.markdown(f"**{title}**")
        st.markdown(body)


    def _humanize_gap_reason(raw_reason: str, component: str) -> str:
        mapping = {
            "missing_lgd_column": "No se encontro una variable objetivo LGD valida en el dataset curado de esta corrida.",
            "missing_ead_column": "No se encontro una variable objetivo EAD valida en el dataset curado de esta corrida.",
            "missing_target_column": f"No se encontro la variable objetivo esperada para {component}.",
            "file_not_found": "El artefacto esperado no fue generado en esta corrida.",
        }
        return mapping.get(raw_reason, raw_reason.replace("_", " "))


    def _render_capitulo_1_apertura_y_objetivos() -> None:
        st.subheader("Capitulo 1. Apertura del problema y objetivos de la tesis")
        st.markdown(f"**Proyecto:** {THESIS_TITLE}")

        storytelling_intro(
            page_goal=(
                "Demostrar que conformal prediction mejora la calibracion y cuantificacion "
                "de incertidumbre en modelos PD/LGD/EAD para riesgo crediticio."
            ),
            business_value=(
                "Decisiones de riesgo crediticio defendibles ante auditoria y regulacion, "
                "no solo rankings de accuracy."
            ),
            key_decision=(
                "Adoptar intervalos conformales como capa complementaria a calibracion "
                "en el pipeline de riesgo."
            ),
            how_to_read=[
                "Contexto y problema (por que importa)",
                "Marco teorico (fundamentos de calibracion y conformal)",
                "Datos y metodologia (como se construyo la evidencia)",
                "Resultados PD, LGD/EAD y conformal (que se obtuvo)",
                "Impacto IFRS9 y conclusiones (para que sirve)",
            ],
        )

        _render_story_block(
            "Por que esta tesis importa",
            (
                "En riesgo crediticio, una probabilidad puntual puede ser suficiente para clasificar, pero no para "
                "decidir con seguridad cuanto provisionar, cuanto capital reservar o que tan robusta es una politica. "
                "El problema central es pasar de predicciones puntuales a decisiones defendibles bajo incertidumbre."
            ),
        )
        _render_story_block(
            "Pregunta de investigacion",
            RESEARCH_QUESTION,
        )
        _render_story_block(
            "Objetivo general",
            GENERAL_OBJECTIVE,
        )
        _render_story_block(
            "Alcance de esta tesis vs continuidad a maestria",
            (
                "La tesis de especializacion es el protagonista: demostrar un flujo defendible de calibracion y "
                "cuantificacion de incertidumbre sobre PD/LGD/EAD con evidencia ejecutable. La agenda de maestria "
                "se presenta solo como continuidad natural de este cierre, no como cambio de foco."
            ),
        )
        scope_df = pd.DataFrame(
            [
                {
                    "Capa": "Tesis de especializacion (cierre actual)",
                    "Pregunta": "El pipeline actual ya permite defender calibracion + conformal + impacto IFRS9?",
                    "Resultado esperado": "Version defendible ante directora y jurados con artefactos canonicos.",
                },
                {
                    "Capa": "Continuidad a maestria (siguiente etapa)",
                    "Pregunta": "Que extensiones de investigacion se habilitan con esta base?",
                    "Resultado esperado": "Agenda focalizada en Paper 1 (CP + robust opt) y Paper 2 (IFRS9 E2E).",
                },
            ]
        )
        st.dataframe(scope_df, hide_index=True, width="stretch")

        evaluator_df = pd.DataFrame(
            [
                {
                    "Comentario clave del evaluador": item.evaluator_comment,
                    "Respuesta implementada en la pagina": item.project_response,
                }
                for item in EVALUATOR_COMMENTS
            ]
        )
        st.markdown("**Ajustes solicitados en evaluacion y respuesta del proyecto**")
        st.dataframe(evaluator_df, hide_index=True, width="stretch")

        render_section_checkpoint(
            "Checkpoint Capitulo 1",
            [
                "Pregunta de investigacion definida y acotada",
                "6 objetivos especificos trazables a artefactos ejecutables",
                "Ajustes del evaluador incorporados en la narrativa",
            ],
        )


    def _render_capitulo_2_contexto_financiero() -> None:
        st.subheader("Capitulo 2. Contexto financiero y planteamiento del problema")
        _render_story_block(
            "El problema en una frase",
            (
                "Si una entidad financiera confia solo en predicciones puntuales, la decision puede parecer precisa, "
                "pero en realidad ignora el margen de error. Ese vacio termina impactando pricing, cupos, provisiones y "
                "la defensa tecnica del modelo frente a auditoria."
            ),
        )
        _render_story_block(
            "Del riesgo tecnico al riesgo de negocio",
            (
                "La tesis se centra en cerrar esa brecha: pasar de un modelo que clasifica bien a un modelo que, "
                "ademas, expresa incertidumbre de forma util para decision y gobernanza."
            ),
        )

        problem_df = pd.DataFrame(
            [
                {
                    "Problema": "Predicciones puntuales sin banda de incertidumbre",
                    "Impacto de negocio": "Subestimacion del riesgo y perdidas inesperadas",
                    "Impacto regulatorio": "Dificultad para defensa ante auditoria/model risk",
                },
                {
                    "Problema": "Probabilidades mal calibradas",
                    "Impacto de negocio": "Pricing, cupos y provisiones distorsionadas",
                    "Impacto regulatorio": "IFRS9 exige estimaciones defendibles y consistentes",
                },
                {
                    "Problema": "Cobertura no monitoreada por segmentos",
                    "Impacto de negocio": "Sobre y subproteccion de grupos de cartera",
                    "Impacto regulatorio": "Riesgo de decisiones inequitativas y alertas de gobernanza",
                },
            ]
        )
        st.markdown("**Mapa de dolor: que pasa cuando no medimos incertidumbre**")
        st.dataframe(problem_df, hide_index=True, width="stretch")
        st.markdown(
            "Lectura guiada: la columna de negocio muestra costos economicos directos; la columna regulatoria "
            "muestra por que ese mismo problema escala a riesgo de validacion y gobernanza."
        )
        _render_story_block(
            "Decision que esta en juego",
            (
                "La tesis no se limita a predecir defaults: busca que cada decision critica (originacion, pricing, "
                "cupos y provisiones) tenga una lectura explicita de incertidumbre. Ese puente entre modelado y "
                "decision regulada es el hilo conductor del resto de capitulos."
            ),
        )

        nodes = [
            "Predicciones puntuales",
            "Mala calibracion",
            "Incertidumbre no cuantificada",
            "Perdidas inesperadas",
            "Provisiones excesivas",
            "Desconfianza regulatoria",
        ]
        link_source = [0, 1, 2, 2, 2]
        link_target = [3, 4, 3, 4, 5]
        link_value = [3, 3, 2, 2, 3]

        fig = go.Figure(
            data=[
                go.Sankey(
                    textfont={"color": "#FFFFFF", "size": 14},
                    node={
                        "label": nodes,
                        "pad": 20,
                        "thickness": 18,
                        "color": ["#1D4ED8", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"],
                        "line": {"color": "rgba(255,255,255,0.35)", "width": 1},
                    },
                    link={
                        "source": link_source,
                        "target": link_target,
                        "value": link_value,
                        "color": "rgba(255,255,255,0.30)",
                    },
                )
            ]
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_layout(
            title="Diagrama problema-impacto",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font={"color": "#FFFFFF"},
        )
        st.plotly_chart(fig, width="stretch")
        _render_story_block(
            "Insight del capitulo",
            (
                "El problema no es solo de accuracy; es de decision bajo incertidumbre. Por eso la narrativa pasa "
                "de 'que tan bien predice' a 'que tan confiable es la prediccion para decidir'."
            ),
        )

        render_decision_box(
            "Innovacion propuesta: usar el ancho del intervalo conformal (PD_high - PD_point) como senal "
            "adicional de deterioro crediticio (SICR) para clasificacion de stages IFRS9.",
        )


    def _render_capitulo_3_marco_teorico() -> None:
        st.subheader("Capitulo 3. Marco teorico y estado del arte")
        _render_story_block(
            "Puente teorico",
            (
                "El marco teorico integra tres capas: riesgo crediticio (PD/LGD/EAD), calibracion probabilistica "
                "y prediccion conformal. La calibracion corrige nivel probabilistico; conformal agrega garantias "
                "de cobertura para cuantificar incertidumbre."
            ),
        )
        _render_story_block(
            "Idea fuerza para la defensa",
            (
                "No se propone reemplazar calibracion por conformal. Se propone una arquitectura complementaria: "
                "primero probabilidades bien calibradas, luego bandas de incertidumbre que puedan auditarse."
            ),
        )
        st.latex(r"\mathrm{ECL} = PD \times LGD \times EAD")
        st.latex(r"P(Y \in C(X)) \ge 1 - \alpha")
        st.markdown("**Tres garantias y tres limites de Conformal Prediction**")
        render_cp_guarantees_and_limits()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Garantia": "Validez marginal finita",
                        "Limite asociado": "No implica cobertura exacta para todo x.",
                    },
                    {
                        "Garantia": "Model-agnostic",
                        "Limite asociado": "No corrige un modelo base sistematicamente sesgado.",
                    },
                    {
                        "Garantia": "Distribucion libre (operativa)",
                        "Limite asociado": "Puede degradarse si se rompe exchangeability.",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(pd.DataFrame(build_cp_concept_matrix_rows()), hide_index=True, width="stretch")

        # Calibration vs Conformal complementarity Sankey
        st.markdown("**Calibracion y Conformal: complemento, no reemplazo**")
        sankey_nodes = [
            "CatBoost PD base",
            "Calibracion (Platt/Isotonic)",
            "PD calibrada",
            "Conformal Mondrian",
            "Intervalo [PD_low, PD_high]",
            "Decision bajo incertidumbre",
        ]
        sankey_fig = go.Figure(
            data=[
                go.Sankey(
                    textfont={"color": "#FFFFFF", "size": 13},
                    node={
                        "label": sankey_nodes,
                        "pad": 20,
                        "thickness": 18,
                        "color": ["#3B82F6", "#10B981", "#0B5ED7", "#F59E0B", "#8B5CF6", "#047857"],
                        "line": {"color": "rgba(255,255,255,0.35)", "width": 1},
                    },
                    link={
                        "source": [0, 1, 2, 3, 4],
                        "target": [1, 2, 3, 4, 5],
                        "value": [3, 3, 3, 3, 3],
                        "color": "rgba(255,255,255,0.25)",
                    },
                )
            ]
        )
        sankey_fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        sankey_fig.update_layout(
            title="Flujo: modelo base -> calibracion -> conformal -> decision",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font={"color": "#FFFFFF"},
            height=320,
        )
        st.plotly_chart(sankey_fig, width="stretch")
        st.markdown(
            "Lectura: la calibracion corrige el *nivel* probabilistico (que 12% sea realmente 12%); "
            "conformal agrega *bandas de incertidumbre* con garantia de cobertura. Son capas secuenciales, no alternativas."
        )

        methodology_dialog(
            "Como se construyen los intervalos conformales (5 pasos)",
            (
                "1. **Entrenar** el modelo PD base (CatBoost) en el set de entrenamiento.\n\n"
                "2. **Calibrar** las probabilidades con Platt o Isotonic en un set separado.\n\n"
                "3. **Calcular residuos de no conformidad** en el set de calibracion conformal: "
                "`score_i = |y_i - PD_calibrada_i|`.\n\n"
                "4. **Obtener el cuantil** `q` de los scores al nivel `1 - alpha` (ej. 90%). "
                "En version **Mondrian**, este cuantil se calcula *por grupo* (grade A, B, ..., G).\n\n"
                "5. **Construir el intervalo** para cada observacion nueva: "
                "`[PD_calibrada - q_grupo, PD_calibrada + q_grupo]`, "
                "truncado a [0, 1]. El resultado es un intervalo con garantia de cobertura finita."
            ),
            button_label="Ver: como se construyen los intervalos conformales",
        )

        methods_rows = [
            {
                "Metodo": "Platt Scaling",
                "Que garantiza": "Mejor calibracion probabilistica (PD)",
                "Que no garantiza": "No genera cobertura intervalar por si solo",
                "Cuando usar": "Modelo binario con buena separacion",
                "Riesgo de sobreclaim": "Confundir mejor ECE con garantia de incertidumbre",
            },
            {
                "Metodo": "Isotonic Regression",
                "Que garantiza": "Calibracion flexible sin forma parametrica fija",
                "Que no garantiza": "No protege contra drift o shift por si solo",
                "Cuando usar": "Suficiente data de calibracion y no linealidad clara",
                "Riesgo de sobreclaim": "Asumir estabilidad fuera de muestra por defecto",
            },
            {
                "Metodo": "Venn-Abers",
                "Que garantiza": "Salida probabilistica mas robusta en calibracion",
                "Que no garantiza": "No reemplaza validacion de cobertura conformal",
                "Cuando usar": "Cuando la calibracion es cuello de botella principal",
                "Riesgo de sobreclaim": "Tomar intervalos VA como bandas conformales completas",
            },
        ]
        methods_rows.extend(build_cp_method_menu_rows("mondrian_selected_cfg"))
        methods_df = pd.DataFrame(methods_rows)
        st.markdown("**Comparativo de metodos y trade-offs**")
        st.dataframe(methods_df, hide_index=True, width="stretch")
        st.markdown(
            "Lectura guiada: el foco no es elegir el metodo 'ganador universal', sino el que mejor "
            "equilibra garantia estadistica y utilidad operativa para este caso de uso."
        )
        st.markdown("**Como se aplica Conformal Prediction en PD/LGD/EAD**")
        cp_role_df = pd.DataFrame(
            [
                {
                    "Componente": "PD (clasificacion)",
                    "Salida conformal": "Intervalo de probabilidad o set de clases plausibles",
                    "Para que sirve": "Cuantificar incertidumbre de default y evitar decisiones de cupo/precio sobreconfiadas.",
                    "Decision impactada": "Originacion, limite y pricing por segmento",
                },
                {
                    "Componente": "LGD (regresion)",
                    "Salida conformal": "Intervalo de severidad de perdida",
                    "Para que sirve": "Capturar variabilidad de recuperacion y evitar subestimacion de perdida severa.",
                    "Decision impactada": "Provisiones y stress de severidad",
                },
                {
                    "Componente": "EAD (regresion)",
                    "Salida conformal": "Intervalo de exposicion al default",
                    "Para que sirve": "Expresar incertidumbre en saldo expuesto al momento de incumplimiento.",
                    "Decision impactada": "Capital economico y sensibilidad IFRS9",
                },
            ]
        )
        st.dataframe(cp_role_df, hide_index=True, width="stretch")
        st.markdown(
            "Lectura guiada: CP no cumple el mismo rol en los tres componentes. En PD mejora la prudencia de "
            "decisiones de originacion; en LGD y EAD fortalece provisiones y sensibilidad de escenarios."
        )

        st.markdown("**Cinco papers clave del marco teorico (referenciados en el anteproyecto)**")
        papers_df = pd.DataFrame(
            [
                {
                    "Paper": "Vovk, Gammerman & Shafer (2022), Algorithmic Learning in a Random World",
                    "Aporte": "Fundamento formal de conformal prediction con garantia de cobertura finita.",
                    "Relacion con la tesis": "Marco base para justificar garantias en PD/LGD/EAD sin supuestos parametricos fuertes.",
                },
                {
                    "Paper": "Angelopoulos & Bates (2023) - A Gentle Introduction to Conformal Prediction",
                    "Aporte": "Sintesis moderna de metodos distribution-free y guias de implementacion practica.",
                    "Relacion con la tesis": "Guia de diseño experimental y lectura de trade-off cobertura-ancho.",
                },
                {
                    "Paper": "Romano, Patterson & Candes (2019) - Conformalized Quantile Regression",
                    "Aporte": "CQR para intervalos adaptativos en regresion heteroscedastica.",
                    "Relacion con la tesis": "Marco directo para intervalos en LGD/EAD.",
                },
                {
                    "Paper": "Vovk & Petej (2014) - Venn-Abers Predictors",
                    "Aporte": "Probabilidades multiprobabilisticas con mejor calibracion.",
                    "Relacion con la tesis": "Puente entre calibracion probabilistica y salida intervalar para PD.",
                },
                {
                    "Paper": "Bellotti (2017) - Reliable Region Predictions for Automated Credit Scoring",
                    "Aporte": "Aplicacion temprana de conformal en scoring crediticio.",
                    "Relacion con la tesis": "Conecta teoria conformal con decisiones de riesgo en credito.",
                },
            ]
        )
        st.dataframe(papers_df, hide_index=True, width="stretch")
        _render_story_block(
            "Sintesis del estado del arte para la defensa",
            (
                "La literatura valida tres ideas: (1) calibrar es obligatorio antes de usar probabilidades en negocio, "
                "(2) conformal agrega garantias de cobertura que la calibracion sola no ofrece, y (3) en credito "
                "aun hay poca evidencia integrada PD/LGD/EAD con impacto regulatorio. Ese hueco define el aporte de la tesis."
            ),
        )

        with st.expander(
            "Adopcion industrial de conformal prediction (contexto para jurados)", expanded=False
        ):
            st.markdown(
                "Conformal prediction no es solo una tecnica academica. "
                "En los ultimos años ha ganado traccion en industria:\n\n"
                "- **AstraZeneca / farmaceutica**: intervalos conformales para prediccion de ensayos clinicos.\n"
                "- **Volvo / automotriz**: monitoreo de incertidumbre en sistemas autonomos.\n"
                "- **Fintech / banca digital**: cuantificacion de riesgo por segmento sin supuestos distribucionales.\n"
                "- **Reguladores**: IFRS9 y Basel III piden evidencia de incertidumbre, no solo metricas puntuales.\n"
                "- **MAPIE (biblioteca de referencia)**: usada en produccion por empresas como Capgemini y startups de ML.\n\n"
                "La tesis se situa en la interseccion de credito + conformal + regulacion, "
                "un area donde la evidencia integrada es aun escasa."
            )


    def _render_capitulo_4_datos_y_preparacion(
        eda: dict,
        dataset_shapes: dict,
        dataset_dictionary: object,
        pd_contract: dict,
    ) -> None:
        st.subheader("Capitulo 4. Datos y preparacion (Lending Club)")
        _render_story_block(
            "Que datos sostienen la evidencia",
            (
                "Se trabaja sobre Lending Club en esquema temporal OOT (train, calibration y test). "
                "La narrativa de resultados se apoya en que los patrones de riesgo por grade y plazo "
                "sean coherentes con el comportamiento esperado de cartera."
            ),
        )

        n_dict_items = len(dataset_dictionary) if isinstance(dataset_dictionary, list) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prestamos train", f"{int(eda.get('n_loans', 0)):,}")
        c2.metric("Default rate train", format_pct(float(eda.get("default_rate", 0.0))))
        c3.metric("Diccionario de campos", str(n_dict_items))
        c4.metric("Features contrato PD", str(pd_contract.get("n_features", "n/a")))
        _render_story_block(
            "Narrativa de calidad de datos",
            (
                "En esta tesis, la preparacion no es un paso cosmetico: define si las metricas posteriores son "
                "defendibles. La estrategia fue fijar contratos de columnas, separar temporalmente train/calibration/test "
                "y evitar leakage entre observaciones de distintos periodos."
            ),
        )

        if isinstance(dataset_dictionary, list) and dataset_dictionary:
            dict_df = pd.DataFrame(dataset_dictionary)
            if "tipo" in dict_df.columns:
                type_df = (
                    dict_df["tipo"]
                    .astype(str)
                    .value_counts(dropna=False)
                    .rename_axis("Tipo de variable")
                    .reset_index(name="Cantidad")
                )
                st.markdown("**Composicion del diccionario de datos por tipo**")
                st.dataframe(type_df, hide_index=True, width="stretch")

        # Splits OOT
        split_rows = []
        for split_name in ["train", "calibration", "test"]:
            key = f"data/processed/{split_name}.parquet"
            shape = dataset_shapes.get(key, {}) if isinstance(dataset_shapes, dict) else {}
            split_rows.append(
                {
                    "Split": split_name,
                    "Filas": int(shape.get("rows", 0) or 0),
                    "Columnas": int(shape.get("cols", 0) or 0),
                }
            )
        st.markdown("**Splits temporales (OOT) usados en la corrida actual**")
        st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
        st.markdown(
            "Lectura guiada: el split temporal evita mezclar periodos futuros en entrenamiento y mejora "
            "la validez de los resultados cuando se discuten decisiones reales de riesgo."
        )

        default_rate_by_grade = eda.get("default_rate_by_grade", {}) if isinstance(eda, dict) else {}
        if isinstance(default_rate_by_grade, dict) and default_rate_by_grade:
            dr_df = pd.DataFrame(
                {
                    "grade": list(default_rate_by_grade.keys()),
                    "default_rate": [float(v) for v in default_rate_by_grade.values()],
                }
            )
            fig_dr = px.bar(
                dr_df,
                x="grade",
                y="default_rate",
                text=dr_df["default_rate"].map(lambda x: f"{x * 100:.1f}%"),
                title="Default rate por grade",
                color="default_rate",
                color_continuous_scale="Reds",
            )
            fig_dr.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_dr.update_coloraxes(showscale=False)
            fig_dr.update_traces(textposition="outside")
            st.plotly_chart(fig_dr, width="stretch")
            st.markdown(
                "Insight: el gradiente de default por grade confirma que la senal de riesgo del dataset "
                "es consistente y util para justificar modelado PD por segmentos."
            )

        loan_count_by_grade = eda.get("loan_count_by_grade", {}) if isinstance(eda, dict) else {}
        if isinstance(loan_count_by_grade, dict) and loan_count_by_grade:
            lc_df = pd.DataFrame(
                {
                    "grade": list(loan_count_by_grade.keys()),
                    "n": [int(v) for v in loan_count_by_grade.values()],
                }
            )
            fig_lc = px.bar(
                lc_df,
                x="grade",
                y="n",
                text=lc_df["n"].map(lambda x: f"{x:,}"),
                title="Distribucion de prestamos por grade",
                color_discrete_sequence=["#0B5ED7"],
            )
            fig_lc.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_lc.update_traces(textposition="outside")
            st.plotly_chart(fig_lc, width="stretch")
            st.markdown(
                "Insight: la masa de cartera por grade ayuda a interpretar estabilidad de metricas "
                "y posibles sesgos de representacion en clases de mayor riesgo."
            )

        term_distribution = eda.get("term_distribution", {}) if isinstance(eda, dict) else {}
        if isinstance(term_distribution, dict) and term_distribution:
            term_df = pd.DataFrame(
                {
                    "term": [f"{int(float(k))} meses" for k in term_distribution],
                    "n": [int(v) for v in term_distribution.values()],
                }
            )
            fig_term = px.pie(
                term_df,
                names="term",
                values="n",
                title="Distribucion por plazo de prestamo",
                color_discrete_sequence=["#0B5ED7", "#F59F00"],
            )
            fig_term.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_term, width="stretch")

        null_pcts = eda.get("null_pcts", {}) if isinstance(eda, dict) else {}
        if isinstance(null_pcts, dict) and null_pcts:
            null_df = pd.DataFrame(
                [
                    {
                        "Variable": key,
                        "Missing_%": float(value) * 100,
                    }
                    for key, value in null_pcts.items()
                ]
            ).sort_values("Missing_%", ascending=False)
            st.markdown("**Top variables con faltantes (porcentaje)**")
            st.dataframe(null_df.head(12), hide_index=True, width="stretch")
            st.markdown(
                "Insight: documentar faltantes evita sobreinterpretar variables con baja calidad y "
                "fortalece la trazabilidad de decisiones de limpieza e imputacion."
            )
        _render_story_block(
            "Conclusion de preparacion de datos",
            (
                "El dataset mantiene senal economica coherente por grade y plazo, con faltantes controlados y "
                "splits temporales consistentes. Esto habilita discutir resultados de modelado como evidencia "
                "metodologica y no solo como salida de un notebook."
            ),
        )


    def _render_capitulo_5_metodologia_y_diseno() -> None:
        st.subheader("Capitulo 5. Metodologia y diseño experimental")
        _render_story_block(
            "Narrativa metodologica",
            (
                "Esta seccion responde a la pregunta de viabilidad: como pasar de una idea amplia a una "
                "ejecucion defendible. La estrategia fue descomponer el problema en objetivos especificos "
                "trazables y ordenar la implementacion con CRISP-DM."
            ),
        )

        objective_rows = [
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[0],
                "Evidencia": "docs/conformal_prediction_research_2026.md + docs/conformal_prediction_quick_reference.md",
                "Estado": _path_status(
                    [
                        "docs/conformal_prediction_research_2026.md",
                        "docs/conformal_prediction_quick_reference.md",
                    ]
                ),
            },
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[1],
                "Evidencia": "src/data/make_dataset.py + data/processed/train.parquet + data/processed/test.parquet",
                "Estado": _path_status(
                    [
                        "src/data/make_dataset.py",
                        "data/processed/train.parquet",
                        "data/processed/test.parquet",
                    ]
                ),
            },
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[2],
                "Evidencia": "src/models/conformal.py + scripts/generate_conformal_intervals.py",
                "Estado": _path_status(
                    [
                        "src/models/conformal.py",
                        "scripts/generate_conformal_intervals.py",
                    ]
                ),
            },
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[3],
                "Evidencia": "data/processed/model_comparison.json + data/processed/conformal_variant_benchmark.parquet",
                "Estado": _path_status(
                    [
                        "data/processed/model_comparison.json",
                        "data/processed/conformal_variant_benchmark.parquet",
                    ]
                ),
            },
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[4],
                "Evidencia": "data/processed/ifrs9_scenario_summary.parquet + src/evaluation/backtesting.py",
                "Estado": _path_status(
                    [
                        "data/processed/ifrs9_scenario_summary.parquet",
                        "src/evaluation/backtesting.py",
                    ]
                ),
            },
            {
                "Objetivo especifico": SPECIFIC_OBJECTIVES[5],
                "Evidencia": "models/conformal_policy_status.json + models/fairness_audit_status.json + docs/RUNBOOK.md",
                "Estado": _path_status(
                    [
                        "models/conformal_policy_status.json",
                        "models/fairness_audit_status.json",
                        "docs/RUNBOOK.md",
                    ]
                ),
            },
        ]
        st.markdown("**Trazabilidad objetivo especifico -> evidencia ejecutable**")
        st.dataframe(pd.DataFrame(objective_rows), hide_index=True, width="stretch")

        crisp_df = pd.DataFrame(
            [
                {
                    "Fase CRISP-DM": "1. Business Understanding",
                    "Implementacion": "Problema de riesgo + criterios de exito + alcance",
                    "Artefactos": "Propuesta, evaluacion, contrato narrativo",
                },
                {
                    "Fase CRISP-DM": "2. Data Understanding",
                    "Implementacion": "EDA Lending Club + distribuciones + drift",
                    "Artefactos": "eda_summary.json, dataset_shapes_summary.json",
                },
                {
                    "Fase CRISP-DM": "3. Data Preparation",
                    "Implementacion": "Splits OOT + feature contract + control de leakage",
                    "Artefactos": "train/calibration/test, pd_model_contract.json",
                },
                {
                    "Fase CRISP-DM": "4. Modeling",
                    "Implementacion": "PD + calibracion + conformal + LGD/EAD",
                    "Artefactos": "model_comparison.json, conformal_lgd_ead_status.json",
                },
                {
                    "Fase CRISP-DM": "5. Evaluation",
                    "Implementacion": "Conformal checks + fairness + IFRS9 + backtesting",
                    "Artefactos": "conformal_policy_checks.parquet, fairness_audit.parquet, ifrs9_scenario_summary.parquet",
                },
                {
                    "Fase CRISP-DM": "6. Deployment",
                    "Implementacion": "Narrativa Streamlit + runbook + reproducibilidad",
                    "Artefactos": "tesis_especializacion.py, docs/RUNBOOK.md, reports/dvc/metrics_summary.json",
                },
            ]
        )
        st.markdown("**Lectura metodologica CRISP-DM del trabajo ejecutado**")
        st.dataframe(crisp_df, hide_index=True, width="stretch")
        st.markdown(
            "Interpretacion: el valor de CRISP-DM aqui no es solo orden documental; es control de alcance. "
            "Cada fase tiene evidencia concreta y eso evita depender de afirmaciones no verificables."
        )

        st.markdown("**Pipeline conformal para riesgo crediticio (scope de especializacion)**")
        _render_conformal_pipeline_flow()
        st.markdown(
            "Lectura del diagrama: el flujo muestra las 7 etapas del scope de especializacion, "
            "desde datos crudos hasta impacto IFRS9. Cada nodo es clickeable para ver detalle."
        )

        st.markdown("**Desarrollo detallado fase por fase (lectura extendida de la tabla CRISP-DM)**")

        phase_details = [
            {
                "fase": "1. Business Understanding",
                "que_hicimos": (
                    "Aterrizamos el problema desde el anteproyecto: pasar de un modelo que solo rankea riesgo "
                    "a un sistema que soporte decisiones bajo incertidumbre. Tambien se incorporo el ajuste del "
                    "evaluador sobre viabilidad, priorizando entregables trazables sobre alcance excesivo."
                ),
                "evidencia": (
                    "Propuesta + evaluacion del anteproyecto + contrato narrativo de la pagina con objetivos "
                    "especificos y capitulos obligatorios."
                ),
                "resultado": (
                    "Se definio un alcance defendible para especializacion: PD/LGD/EAD con calibracion, "
                    "conformal e impacto IFRS9 en la corrida canónica."
                ),
            },
            {
                "fase": "2. Data Understanding",
                "que_hicimos": (
                    "Caracterizamos Lending Club por volumen, default rate, distribucion por grade y plazo, "
                    "y calidad de variables criticas. Esta lectura valida que el dataset tiene senal real de riesgo "
                    "y soporte suficiente para evaluacion por segmentos."
                ),
                "evidencia": (
                    "`eda_summary.json`, `dataset_shapes_summary.json`, `dataset_dictionary.json`, graficas por "
                    "grade/plazo/faltantes en esta misma pagina."
                ),
                "resultado": (
                    "Se confirmo que los patrones de riesgo son consistentes y que los faltantes relevantes "
                    "pueden gestionarse sin comprometer interpretabilidad."
                ),
            },
            {
                "fase": "3. Data Preparation",
                "que_hicimos": (
                    "Se consolidaron splits temporales OOT (train/calibration/test), contratos de columnas y "
                    "controles para evitar leakage. Tambien se materializo la variable objetivo LGD para cerrar "
                    "la brecha detectada en corridas previas."
                ),
                "evidencia": (
                    "`train.parquet`, `calibration.parquet`, `test.parquet`, `pd_model_contract.json` y "
                    "estado LGD/EAD en `conformal_lgd_ead_status.json`."
                ),
                "resultado": (
                    "Base de datos lista para modelado reproducible, con variables objetivo y trazabilidad "
                    "por componente."
                ),
            },
            {
                "fase": "4. Modeling",
                "que_hicimos": (
                    "Entrenamos PD con calibracion seleccionada por validacion temporal y aplicamos conformal "
                    "para cuantificar incertidumbre. En LGD se compararon variantes (two-stage, split directo, "
                    "CQR y adaptativo por grade-tiempo) y se selecciono la que cumple guardrails."
                ),
                "evidencia": (
                    "`model_comparison.json`, `conformal_policy_status.json`, `lgd_variant_benchmark.parquet`, "
                    "`conformal_intervals_lgd.parquet`, `conformal_intervals_ead.parquet`."
                ),
                "resultado": (
                    "Pipeline de modelado con decision metodologica explicita, no basada en una sola metrica."
                ),
            },
            {
                "fase": "5. Evaluation",
                "que_hicimos": (
                    "Evaluamos desempeño discriminativo y calibracion en PD, cobertura/eficiencia en conformal, "
                    "estabilidad temporal por mes/grade, fairness como diagnostico y sensibilidad IFRS9 por escenario."
                ),
                "evidencia": (
                    "`roc_curve_data.parquet`, `calibration_curve_data.parquet`, `conformal_policy_checks.parquet`, "
                    "`conformal_backtest_monthly.parquet`, `fairness_audit.parquet`, `ifrs9_scenario_summary.parquet`."
                ),
                "resultado": (
                    "La tesis demuestra no solo desempeño, sino gobernanza tecnica de resultados bajo incertidumbre."
                ),
            },
            {
                "fase": "6. Deployment",
                "que_hicimos": (
                    "Empaquetamos la historia en Streamlit con trazabilidad por artefacto, fallbacks cuando faltan "
                    "insumos y contrato narrativo para evitar degradacion de contenido entre iteraciones."
                ),
                "evidencia": (
                    "`tesis_especializacion.py`, `docs/RUNBOOK.md`, `reports/tesis_especializacion.md`, "
                    "`reports/dvc/metrics_summary.json`."
                ),
                "resultado": (
                    "Entrega reproducible para direccion y jurados, alineada con una lectura tipo articulo tecnico."
                ),
            },
        ]
        for item in phase_details:
            st.markdown(f"**{item['fase']}**")
            st.markdown(item["que_hicimos"])
            st.caption(f"Evidencia ejecutable: {item['evidencia']}")
            st.markdown(f"Resultado de la fase: {item['resultado']}")

        render_section_checkpoint(
            "Checkpoint Metodologia",
            [
                "CRISP-DM con 6 fases y artefactos trazables",
                "Pipeline conformal de 7 etapas (scope de especializacion)",
                "Cada objetivo especifico mapeado a evidencia ejecutable",
            ],
        )


    def _render_capitulo_6_resultados_pd(
        comparison: dict,
        roc_curve: pd.DataFrame,
        calibration_curve: pd.DataFrame,
        shap_summary: pd.DataFrame,
        permutation_importance: pd.DataFrame,
    ) -> None:
        st.subheader("Capitulo 6. Resultados PD")
        _render_story_block(
            "Que queremos demostrar en PD",
            (
                "La meta no es solo discriminar buenos y malos pagadores; tambien es asegurar que la probabilidad "
                "estimada sea util para decisiones de negocio y consistente con la frecuencia observada."
            ),
        )

        final_metrics = comparison.get("final_test_metrics", {}) if isinstance(comparison, dict) else {}
        cal_report = (
            comparison.get("calibration_selection_report", {}) if isinstance(comparison, dict) else {}
        )

        cols = st.columns(6)
        cols[0].metric("AUC OOT", f"{float(final_metrics.get('auc_roc', 0.0)):.4f}")
        cols[1].metric("Gini", f"{float(final_metrics.get('gini', 0.0)):.4f}")
        cols[2].metric("Brier", f"{float(final_metrics.get('brier_score', 0.0)):.4f}")
        cols[3].metric("ECE", f"{float(final_metrics.get('ece', 0.0)):.4f}")
        cols[4].metric("KS", f"{float(final_metrics.get('ks_statistic', 0.0)):.4f}")
        cols[5].metric("Calibrador", str(comparison.get("best_calibration", "n/a")))

        st.markdown("**Criterio de seleccion del calibrador**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Razon de seleccion": cal_report.get("selection_reason", "n/a"),
                        "Limite de degradacion AUC": cal_report.get("auc_drop_limit", "n/a"),
                        "Esquema de validacion": comparison.get("validation_scheme", "n/a"),
                        "Alcance del dataset": comparison.get("dataset_scope", "n/a"),
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        pd_reading_df = pd.DataFrame(
            [
                {
                    "Metrica": "AUC / Gini / KS",
                    "Que valida": "Capacidad de separacion entre buenos y malos pagadores.",
                    "Riesgo si falla": "Politica de originacion con segmentacion debil.",
                    "Decision soportada": "Seleccion de modelo champion en ranking de riesgo.",
                },
                {
                    "Metrica": "Brier / ECE / Curva de calibracion",
                    "Que valida": "Que la PD estimada se parezca a la frecuencia observada.",
                    "Riesgo si falla": "Pricing, cupos y provisiones sesgados por probabilidad mal escalada.",
                    "Decision soportada": "Eleccion y defensa del calibrador final.",
                },
            ]
        )
        st.markdown("**Como interpretar PD para decision (no solo para benchmark)**")
        st.dataframe(pd_reading_df, hide_index=True, width="stretch")

        if not roc_curve.empty and all(c in roc_curve.columns for c in ["model", "fpr", "tpr"]):
            fig_roc = px.line(
                roc_curve,
                x="fpr",
                y="tpr",
                color="model",
                title="Curva ROC por modelo",
                labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
            )
            fig_roc.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#5F6B7A"}
            )
            fig_roc.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_roc, width="stretch")
            claim_evidence_implication(
                claim="CatBoost calibrado discrimina significativamente mejor que el baseline",
                evidence="AUC OOT del modelo calibrado vs Logistic Regression visible en la curva ROC",
                implication="Poder de separacion suficiente para soportar decisiones de originacion y pricing.",
            )
        else:
            st.info("No hay `roc_curve_data.parquet` para esta corrida.")

        if not calibration_curve.empty and all(
            c in calibration_curve.columns for c in ["model", "predicted_prob", "observed_freq"]
        ):
            fig_cal = px.line(
                calibration_curve,
                x="predicted_prob",
                y="observed_freq",
                color="model",
                markers=True,
                title="Curva de calibracion (probabilidad predicha vs frecuencia observada)",
            )
            fig_cal.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line={"dash": "dash", "color": "#5F6B7A"},
            )
            fig_cal.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_cal, width="stretch")
            claim_evidence_implication(
                claim="La calibracion mantiene coherencia entre PD predicha y frecuencia observada",
                evidence="Curva de calibracion proxima a la diagonal con ECE bajo",
                implication="Las probabilidades son defendibles para pricing, cupos y provisiones.",
            )
        else:
            st.info("No hay `calibration_curve_data.parquet` para esta corrida.")

        col_l, col_r = st.columns(2)
        with col_l:
            if not shap_summary.empty and all(
                c in shap_summary.columns for c in ["feature", "mean_abs_shap"]
            ):
                top_shap = shap_summary.sort_values("mean_abs_shap", ascending=False).head(12)
                fig_shap = px.bar(
                    top_shap,
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Top variables por importancia SHAP",
                )
                fig_shap.update_layout(**PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig_shap, width="stretch")
            else:
                st.info("No hay `shap_summary.parquet` disponible.")

        with col_r:
            if not permutation_importance.empty and all(
                c in permutation_importance.columns for c in ["feature", "auc_drop"]
            ):
                top_perm = permutation_importance.sort_values("auc_drop", ascending=False).head(12)
                fig_perm = px.bar(
                    top_perm,
                    x="auc_drop",
                    y="feature",
                    orientation="h",
                    title="Top variables por caida de AUC (permutation)",
                    color_discrete_sequence=["#F59F00"],
                )
                fig_perm.update_layout(**PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig_perm, width="stretch")
            else:
                st.info("No hay `permutation_importance.parquet` disponible.")
        st.markdown(
            "Cierre del capitulo: se valida PD en dos niveles complementarios, discriminacion (ROC/AUC/KS) "
            "y calibracion (Brier/ECE/curva), para evitar decisiones basadas en probabilidades mal escaladas."
        )


    def _render_capitulo_7_resultados_lgd_ead(
        lgd_ead_status: dict,
        ead_intervals: pd.DataFrame,
        lgd_variant_benchmark: pd.DataFrame,
        lgd_coverage_by_grade: pd.DataFrame,
        lgd_coverage_by_year: pd.DataFrame,
        lgd_guardrails: dict,
    ) -> None:
        st.subheader("Capitulo 7. Resultados LGD/EAD")
        _render_story_block(
            "Lectura de equilibrio PD/LGD/EAD",
            (
                "La brecha LGD paso de disponibilidad a calidad. Ya existe target LGD, se compararon variantes "
                "de modelado/conformal y se selecciono una politica adaptativa por grado-tiempo con guardrails "
                "explícitos de cobertura, ancho y sesgo."
            ),
        )
        _render_story_block(
            "Historia tecnica del cierre de brecha",
            (
                "Primero se materializo LGD en el pipeline. Luego se evaluaron cuatro variantes (two-stage, "
                "directo split, CQR y adaptativo online). Finalmente se eligio la variante con cumplimiento "
                "integral de guardrails, no solo por una metrica aislada."
            ),
        )

        lgd = lgd_ead_status.get("lgd", {}) if isinstance(lgd_ead_status, dict) else {}
        ead = lgd_ead_status.get("ead", {}) if isinstance(lgd_ead_status, dict) else {}
        lgd_available = bool(lgd.get("available", False))
        ead_available = bool(ead.get("available", False))
        lgd_reason = _humanize_gap_reason(str(lgd.get("reason", "ok")), "LGD")
        ead_reason = _humanize_gap_reason(str(ead.get("reason", "ok")), "EAD")
        selected_variant = str(lgd.get("selected_variant", "n/a"))
        guardrails_payload = lgd_guardrails if isinstance(lgd_guardrails, dict) else {}
        guardrails_status = (
            lgd.get("guardrails", {}) if isinstance(lgd.get("guardrails", {}), dict) else {}
        )
        guardrails_overall = bool(
            guardrails_payload.get("overall_pass", guardrails_status.get("overall_pass", False))
        )

        status_df = pd.DataFrame(
            [
                {
                    "Componente": "PD",
                    "Estado": "Disponible",
                    "Detalle": "Integrado en modelo principal con calibracion + conformal",
                    "Artefacto": "data/processed/model_comparison.json",
                },
                {
                    "Componente": "LGD",
                    "Estado": "Disponible" if lgd_available else "No disponible",
                    "Detalle": lgd_reason
                    if not lgd_available
                    else f"Intervalos conformales disponibles ({selected_variant})",
                    "Artefacto": str(lgd.get("intervals_path", "n/a")),
                },
                {
                    "Componente": "EAD",
                    "Estado": "Disponible" if ead_available else "No disponible",
                    "Detalle": ead_reason
                    if not ead_available
                    else "Intervalos conformales disponibles",
                    "Artefacto": str(ead.get("intervals_path", "n/a")),
                },
            ]
        )
        st.markdown("**Matriz de estado por componente (PD/LGD/EAD)**")
        st.dataframe(status_df, hide_index=True, width="stretch")
        if lgd_available:
            lgd_m90 = (lgd.get("conformal", {}) or {}).get("metrics_90", {})
            lgd_m95 = (lgd.get("conformal", {}) or {}).get("metrics_95", {})
            if isinstance(lgd_m90, dict) and isinstance(lgd_m95, dict) and lgd_m90 and lgd_m95:
                st.markdown("**Metricas LGD conformal (corrida actual)**")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Coverage LGD 90%", format_pct(float(lgd_m90.get("empirical_coverage", 0.0))))
                c2.metric("Coverage LGD 95%", format_pct(float(lgd_m95.get("empirical_coverage", 0.0))))
                c3.metric("Width LGD 90%", f"{float(lgd_m90.get('avg_interval_width', 0.0)):.4f}")
                c4.metric("Width LGD 95%", f"{float(lgd_m95.get('avg_interval_width', 0.0)):.4f}")
                c5.metric("Guardrails LGD", _status_badge(guardrails_overall))

                cov90 = float(lgd_m90.get("empirical_coverage", 0.0))
                cov95 = float(lgd_m95.get("empirical_coverage", 0.0))
                cov90_target = float(lgd_m90.get("target_coverage", 0.90))
                cov95_target = float(lgd_m95.get("target_coverage", 0.95))
                if cov90 >= cov90_target and cov95 >= cov95_target and guardrails_overall:
                    st.success(
                        "LGD cumple cobertura objetivo y guardrails de calidad en la corrida vigente. "
                        "La brecha principal queda cerrada para discusión académica."
                    )
                else:
                    st.warning(
                        "LGD está disponible, pero todavía hay checks de calidad por cerrar en cobertura/guardrails."
                    )

                st.markdown(
                    "Interpretacion: la meta ya no es 'tener LGD', sino sostener cobertura defendible con "
                    "eficiencia de ancho y sesgo controlado bajo drift temporal."
                )

                tradeoff_panel(
                    decision_label="Variante LGD conformal seleccionada",
                    upside="Provision calibrada con cobertura por grade y estabilidad temporal",
                    downside="Variantes adaptativas pueden sobre-cubrir en grades de bajo riesgo (mayor ancho)",
                    monitoring="Coverage + width por grade y año de originacion en cada corrida canonica",
                )

                if not lgd_variant_benchmark.empty:
                    bench = lgd_variant_benchmark.copy()
                    numeric_cols = [
                        "coverage_90",
                        "coverage_95",
                        "avg_width_90",
                        "avg_width_95",
                        "bias",
                        "width_inflation_90",
                        "width_inflation_95",
                    ]
                    for col in numeric_cols:
                        if col in bench.columns:
                            bench[col] = bench[col].astype(float).round(4)
                    preferred_cols = [
                        c
                        for c in [
                            "variant",
                            "coverage_90",
                            "coverage_95",
                            "avg_width_90",
                            "avg_width_95",
                            "bias",
                            "overall_pass",
                        ]
                        if c in bench.columns
                    ]
                    st.markdown("**Benchmark de variantes LGD (evidencia de selección)**")
                    st.dataframe(bench[preferred_cols], hide_index=True, width="stretch")

                if not lgd_coverage_by_grade.empty:
                    st.markdown("**Cobertura LGD por grade (support-aware)**")
                    st.dataframe(lgd_coverage_by_grade, hide_index=True, width="stretch")
                    grade_cov = lgd_coverage_by_grade.copy()
                    if all(c in grade_cov.columns for c in ["grade", "coverage_90", "coverage_95"]):
                        grade_long = grade_cov.melt(
                            id_vars="grade",
                            value_vars=["coverage_90", "coverage_95"],
                            var_name="nivel",
                            value_name="coverage",
                        )
                        fig_grade = px.bar(
                            grade_long,
                            x="grade",
                            y="coverage",
                            color="nivel",
                            barmode="group",
                            title="Cobertura LGD por grade (90/95)",
                        )
                        fig_grade.update_layout(**PLOTLY_TEMPLATE["layout"])
                        st.plotly_chart(fig_grade, width="stretch")

                if not lgd_coverage_by_year.empty:
                    st.markdown("**Cobertura LGD por año de originación**")
                    st.dataframe(lgd_coverage_by_year, hide_index=True, width="stretch")
                    year_cov = lgd_coverage_by_year.copy()
                    if "issue_year" in year_cov.columns:
                        year_cov["issue_year"] = pd.to_numeric(year_cov["issue_year"], errors="coerce")
                    if all(c in year_cov.columns for c in ["issue_year", "coverage_90", "coverage_95"]):
                        fig_year = go.Figure()
                        fig_year.add_trace(
                            go.Scatter(
                                x=year_cov["issue_year"],
                                y=year_cov["coverage_90"],
                                mode="lines+markers",
                                name="Coverage 90%",
                            )
                        )
                        fig_year.add_trace(
                            go.Scatter(
                                x=year_cov["issue_year"],
                                y=year_cov["coverage_95"],
                                mode="lines+markers",
                                name="Coverage 95%",
                            )
                        )
                        fig_year.update_layout(**PLOTLY_TEMPLATE["layout"])
                        fig_year.update_layout(title="Estabilidad temporal de cobertura LGD")
                        st.plotly_chart(fig_year, width="stretch")
        if not lgd_available:
            st.markdown("**Diagnostico tecnico de la brecha LGD**")
            lgd_diag_df = pd.DataFrame(
                [
                    {
                        "Chequeo": "Estado reportado por pipeline canónico",
                        "Resultado": str(lgd.get("reason", "n/a")),
                        "Interpretacion": "El flujo de entrenamiento LGD no encontro la variable objetivo esperada (`lgd`).",
                    },
                    {
                        "Chequeo": "Artefacto de intervalos LGD",
                        "Resultado": "Disponible"
                        if Path("data/processed/conformal_intervals_lgd.parquet").exists()
                        else "No disponible",
                        "Interpretacion": "No hay evidencia ejecutable de cobertura LGD para esta corrida.",
                    },
                    {
                        "Chequeo": "Modelos two-stage LGD entrenados",
                        "Resultado": (
                            "Si"
                            if (
                                Path("models/lgd_stage1_clf.pkl").exists()
                                and Path("models/lgd_stage2_reg.pkl").exists()
                            )
                            else "No"
                        ),
                        "Interpretacion": "No hay salida de entrenamiento LGD en `models/` para el run actual.",
                    },
                    {
                        "Chequeo": "Construccion de target en pipeline de datos",
                        "Resultado": "Pendiente",
                        "Interpretacion": "La preparacion actual conserva `default_flag`, pero no materializa una columna `lgd` en los splits canónicos.",
                    },
                ]
            )
            st.dataframe(lgd_diag_df, hide_index=True, width="stretch")
            st.markdown(
                "Conclusion de causa raiz: hoy el problema no es de Conformal Prediction; es de variable objetivo. "
                "Hasta no construir y persistir `lgd` en train/calibration/test, no se pueden generar intervalos conformales LGD."
            )

        if not ead_intervals.empty and all(
            c in ead_intervals.columns for c in ["width_90", "width_95"]
        ):
            width_90 = pd.to_numeric(ead_intervals["width_90"], errors="coerce").dropna()
            width_95 = pd.to_numeric(ead_intervals["width_95"], errors="coerce").dropna()
            c1, c2, c3 = st.columns(3)
            c1.metric("EAD intervalos", f"{len(ead_intervals):,}")
            c2.metric("Ancho medio 90%", f"{float(width_90.mean()):.2f}")
            c3.metric("Ancho medio 95%", f"{float(width_95.mean()):.2f}")
            st.markdown(
                "Interpretacion: para la reunion de direccion es mas util discutir estabilidad resumen "
                "(promedios y percentiles) que una distribucion cruda de anchos."
            )

            unique_90 = int(width_90.round(8).nunique())
            unique_95 = int(width_95.round(8).nunique())
            if unique_90 <= 1 and unique_95 <= 1:
                st.info(
                    "En esta corrida el ancho EAD es practicamente constante (P10=P50=P90), por lo que la tabla "
                    "de percentiles no agrega informacion adicional."
                )
            else:
                quant_df = pd.DataFrame(
                    {
                        "Percentil": ["P10", "P50", "P90"],
                        "Width_90": [
                            float(width_90.quantile(0.10)),
                            float(width_90.quantile(0.50)),
                            float(width_90.quantile(0.90)),
                        ],
                        "Width_95": [
                            float(width_95.quantile(0.10)),
                            float(width_95.quantile(0.50)),
                            float(width_95.quantile(0.90)),
                        ],
                    }
                )
                st.dataframe(quant_df, hide_index=True, width="stretch")
                st.markdown(
                    "Lectura guiada: los percentiles permiten validar si el ancho de intervalos es operativamente "
                    "razonable para decisiones por segmento y escenario."
                )
        else:
            st.info("No hay `conformal_intervals_ead.parquet` en esta corrida.")

        st.markdown("**Cierre operativo de brecha LGD (para direccion de tesis)**")
        if lgd_available:
            lgd_plan = pd.DataFrame(
                [
                    {
                        "Accion": "Sostener monitoreo mensual de cobertura LGD por grade/año",
                        "Estado": "En curso",
                        "Criterio de cierre": "Alertas tempranas activas y sin degradación de guardrails",
                    },
                    {
                        "Accion": "Consolidar anexos metodológicos (online conformal + maduración)",
                        "Estado": "En curso",
                        "Criterio de cierre": "Capítulo técnico y defensa alineados con artefactos canónicos",
                    },
                    {
                        "Accion": "Validar robustez en corrida adicional pre-jurados",
                        "Estado": "Pendiente",
                        "Criterio de cierre": "Repetir cumplimiento de guardrails en nueva corrida canónica",
                    },
                ]
            )
        else:
            lgd_plan = pd.DataFrame(
                [
                    {
                        "Accion": "Materializar target LGD en splits canónicos",
                        "Estado": "Pendiente",
                        "Criterio de cierre": "`train/calibration/test` con columna `lgd`",
                    },
                    {
                        "Accion": "Entrenar variantes LGD y ejecutar benchmark conformal",
                        "Estado": "Pendiente",
                        "Criterio de cierre": "Artefactos `lgd_variant_benchmark` y `conformal_intervals_lgd` disponibles",
                    },
                ]
            )
        st.dataframe(lgd_plan, hide_index=True, width="stretch")


    def _render_capitulo_8_conformal_y_comparativa(
        conformal_status: dict,
        conformal_checks: pd.DataFrame,
        conformal_backtest: pd.DataFrame,
        conformal_backtest_grade: pd.DataFrame,
        conformal_intervals_mondrian: pd.DataFrame,
        conformal_group_metrics: pd.DataFrame,
        conformal_variant_benchmark: pd.DataFrame,
        conformal_alerts: pd.DataFrame,
        conformal_variant_by_group: pd.DataFrame,
    ) -> None:
        st.subheader("Capitulo 8. Conformal prediction y evaluacion comparativa")
        _render_story_block(
            "Que responde esta evaluacion",
            (
                "Si el modelo predice una banda al 90% o 95%, la pregunta critica es si esa promesa se cumple en "
                "el tiempo y en subgrupos. Este capitulo mide cobertura, eficiencia y estabilidad para responder "
                "esa pregunta de forma auditable."
            ),
        )
        _render_story_block(
            "Intuicion operativa para jurados",
            (
                "Sin conformal, una PD puntual parece precisa pero no dice que tan fragile es la decision. "
                "Con conformal, cada observacion trae un rango [low, high] con cobertura objetivo. "
                "Eso permite discutir decision bajo incertidumbre, no solo accuracy promedio."
            ),
        )

        left, right = st.columns(2)
        with left:
            st.markdown(
                """
    **Sin conformal prediction**
    - Prestamo: `PD = 12%`
    - Un solo numero, sin banda de error.
    - No hay forma de saber si es 8% o 16%.
    - Riesgo: sobreconfianza en cupo, pricing y provisiones.
    - Ante auditoria: "el modelo dice 12%" sin defensa tecnica.

    *Analogia: decir "manana llueve" sin probabilidad ni rango.*
    """
            )
        with right:
            st.markdown(
                """
    **Con conformal prediction**
    - Prestamo: `PD = [8%, 16%]` al 90%
    - Banda de incertidumbre auditable con garantia de cobertura.
    - El decisor sabe el peor caso (16%) y el mejor (8%).
    - Beneficio: decision robusta ante variacion real.
    - Ante auditoria: "el modelo garantiza 90% de acierto en la banda".

    *Analogia: "probabilidad de lluvia entre 60-80%" — accionable.*
    """
            )

        cols = st.columns(8)
        cols[0].metric("Coverage 90%", format_pct(float(conformal_status.get("coverage_90", 0.0))))
        cols[1].metric("Coverage 95%", format_pct(float(conformal_status.get("coverage_95", 0.0))))
        cols[2].metric(
            "Min group cov 90%", format_pct(float(conformal_status.get("min_group_coverage_90", 0.0)))
        )
        cols[3].metric("Avg width 90%", f"{float(conformal_status.get('avg_width_90', 0.0)):.4f}")
        cols[4].metric("Winkler 90%", f"{float(conformal_status.get('winkler_90', 0.0)):.4f}")
        cols[5].metric(
            "Checks",
            f"{int(conformal_status.get('checks_passed', 0))}/{int(conformal_status.get('checks_total', 0))}",
        )
        cols[6].metric("Alerts", str(int(conformal_status.get("total_alerts", 0))))
        cols[7].metric(
            "Estado estricto", _status_badge(bool(conformal_status.get("overall_pass", False)))
        )
        st.markdown(
            "Lectura guiada: cobertura mide garantia empirica; ancho mide costo de robustez. "
            "El objetivo no es maximizar una metrica aislada, sino sostener equilibrio cobertura-ancho."
        )
        st.markdown("**Evidence ladder (principio -> artefacto -> decision)**")
        st.dataframe(pd.DataFrame(build_cp_evidence_ladder_rows()), hide_index=True, width="stretch")

        metric_meaning = pd.DataFrame(
            [
                {
                    "Metrica": "Coverage 90/95",
                    "Que significa": "Frecuencia empirica de acierto del intervalo",
                    "Lectura para tesis": "Si cae bajo meta, la politica de incertidumbre no es defendible",
                },
                {
                    "Metrica": "Avg width",
                    "Que significa": "Conservadurismo promedio del intervalo",
                    "Lectura para tesis": "Mas ancho protege mas, pero reduce eficiencia economica",
                },
                {
                    "Metrica": "Min group coverage 90",
                    "Que significa": "Peor cobertura observada por segmento",
                    "Lectura para tesis": "Evita que el promedio global esconda fallos en grupos criticos",
                },
            ]
        )
        st.dataframe(metric_meaning, hide_index=True, width="stretch")

        methodology_dialog(
            "Por que Mondrian y no solo Split global",
            (
                "**Problema del Split global:** si el portafolio tiene 70% Grade A y 5% Grade G, "
                "un conformal global puede reportar 90% de cobertura promedio, pero Grade G puede "
                "tener solo 58%. El promedio global esconde fallos en subgrupos criticos.\n\n"
                "**Solucion Mondrian:** calcula cuantiles de no conformidad *por grupo* (grade). "
                "Cada grade obtiene su propio radio de intervalo y mejora la cobertura por particion.\n\n"
                "**Costo:** los intervalos de grupos pequenos (F, G) pueden ser mas anchos, "
                "pero la garantia por segmento es operativamente mas valiosa que un promedio engañoso.\n\n"
                "**Evidencia en esta tesis:** Global split = 58.69% min group coverage vs "
                "Mondrian seleccionado = 89.79% min group coverage (+31 puntos porcentuales)."
            ),
            button_label="Ver: por que Mondrian y no Split global",
        )

        if not conformal_intervals_mondrian.empty and all(
            c in conformal_intervals_mondrian.columns for c in ["width_90", "y_pred", "grade"]
        ):
            st.markdown("**Estructura de incertidumbre por observacion y segmento (PD)**")
            c1, c2 = st.columns(2)
            with c1:
                sample = conformal_intervals_mondrian.sample(
                    min(80000, len(conformal_intervals_mondrian)),
                    random_state=11,
                )
                fig_hist = px.histogram(
                    sample,
                    x="width_90",
                    color="grade",
                    nbins=55,
                    barmode="overlay",
                    opacity=0.65,
                    title="Ancho de intervalos (90%) por grade",
                )
                fig_hist.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
                st.plotly_chart(fig_hist, width="stretch")
            with c2:
                sample = conformal_intervals_mondrian.sample(
                    min(35000, len(conformal_intervals_mondrian)),
                    random_state=19,
                )
                fig_scatter = px.scatter(
                    sample,
                    x="y_pred",
                    y="width_90",
                    color="grade",
                    opacity=0.25,
                    title="PD puntual vs ancho de incertidumbre",
                )
                fig_scatter.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
                st.plotly_chart(fig_scatter, width="stretch")

            width_by_grade = (
                conformal_intervals_mondrian.groupby("grade", observed=True)["width_90"]
                .median()
                .sort_values(ascending=False)
            )
            if not width_by_grade.empty:
                widest_grade = str(width_by_grade.index[0])
                narrowest_grade = str(width_by_grade.index[-1])
                widest_width = float(width_by_grade.iloc[0])
                narrowest_width = float(width_by_grade.iloc[-1])
                pd_width_corr = float(
                    conformal_intervals_mondrian["y_pred"].corr(
                        conformal_intervals_mondrian["width_90"]
                    )
                )
                st.markdown(
                    f"Insight: mayor ancho mediano en grade **{widest_grade}** ({widest_width:.3f}) "
                    f"y menor en **{narrowest_grade}** ({narrowest_width:.3f}); "
                    f"correlacion PD-width={pd_width_corr:.2f}."
                )

        if not conformal_checks.empty:
            st.markdown("**Policy checks conformal (corrida actual)**")
            st.dataframe(conformal_checks, hide_index=True, width="stretch")
            if "passed" in conformal_checks.columns:
                failed = conformal_checks[~conformal_checks["passed"]]
                if failed.empty:
                    st.success("Todas las reglas de politica conformal pasan en el snapshot actual.")
                else:
                    st.warning(
                        f"Hay {len(failed)} regla(s) fuera de politica; requiere recalibracion o ajuste segmentado."
                    )

        if not conformal_group_metrics.empty and all(
            c in conformal_group_metrics.columns for c in ["group", "coverage_90"]
        ):
            st.markdown("**Cobertura por grupo (Mondrian)**")
            group_plot = conformal_group_metrics.rename(columns={"group": "grade"}).copy()
            fig_group = go.Figure()
            fig_group.add_trace(
                go.Bar(
                    x=group_plot["grade"],
                    y=group_plot["coverage_90"],
                    marker_color="#00D4AA",
                    name="Cobertura 90%",
                )
            )
            fig_group.add_hline(
                y=0.90, line_dash="dash", line_color="#FF6B6B", annotation_text="Meta 90%"
            )
            fig_group.update_layout(**PLOTLY_TEMPLATE["layout"])
            fig_group.update_layout(
                title="Cobertura empirica por grade",
                yaxis={"tickformat": ".0%"},
                height=390,
            )
            st.plotly_chart(fig_group, width="stretch")
            st.dataframe(group_plot, hide_index=True, width="stretch")
            worst_group = group_plot.loc[group_plot["coverage_90"].idxmin()]
            best_group = group_plot.loc[group_plot["coverage_90"].idxmax()]
            under_target = group_plot[group_plot["coverage_90"] < 0.90]
            st.markdown(
                f"Lectura: peor grade **{worst_group['grade']}** ({float(worst_group['coverage_90']):.1%}), "
                f"mejor grade **{best_group['grade']}** ({float(best_group['coverage_90']):.1%}), "
                f"grades bajo meta: **{len(under_target)}**."
            )

        if not conformal_backtest.empty and all(
            c in conformal_backtest.columns
            for c in ["month", "coverage_90", "coverage_95", "avg_width_90"]
        ):
            bt_df = conformal_backtest.copy()
            bt_df["month"] = pd.to_datetime(bt_df["month"], errors="coerce")

            fig_cov = go.Figure()
            fig_cov.add_trace(
                go.Scatter(
                    x=bt_df["month"],
                    y=bt_df["coverage_90"],
                    mode="lines+markers",
                    name="Coverage 90%",
                    line={"color": "#0B5ED7"},
                )
            )
            fig_cov.add_trace(
                go.Scatter(
                    x=bt_df["month"],
                    y=bt_df["coverage_95"],
                    mode="lines+markers",
                    name="Coverage 95%",
                    line={"color": "#198754"},
                )
            )
            fig_cov.add_hline(y=0.90, line_dash="dash", line_color="#0B5ED7")
            fig_cov.add_hline(y=0.95, line_dash="dash", line_color="#198754")
            fig_cov.update_layout(**PLOTLY_TEMPLATE["layout"], title="Cobertura mensual 90/95")
            st.plotly_chart(fig_cov, width="stretch")
            st.markdown(
                "Lectura guiada: las lineas punteadas son las metas de cobertura; desalineaciones sostenidas "
                "indican necesidad de recalibracion o rediseno del score no conformidad."
            )

            fig_width = px.line(
                bt_df,
                x="month",
                y="avg_width_90",
                markers=True,
                title="Ancho promedio mensual (90%)",
                color_discrete_sequence=["#F59F00"],
            )
            fig_width.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_width, width="stretch")
            st.markdown(
                "Lectura guiada: el ancho no se interpreta aislado; se evalua junto con cobertura para "
                "entender el costo de robustez."
            )

            if not conformal_backtest_grade.empty and all(
                c in conformal_backtest_grade.columns for c in ["month", "grade", "coverage_90"]
            ):
                heat_df = conformal_backtest_grade.copy()
                heat_df["month"] = pd.to_datetime(heat_df["month"], errors="coerce").dt.strftime(
                    "%Y-%m"
                )
                pivot = heat_df.pivot(index="grade", columns="month", values="coverage_90").sort_index()
                if not pivot.empty:
                    fig_heat = px.imshow(
                        pivot,
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                        title="Mapa de calor cobertura 90% (grade x mes)",
                        labels={"color": "Cobertura"},
                    )
                    fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"], height=390)
                    st.plotly_chart(fig_heat, width="stretch")

                    monthly_below = int((bt_df["coverage_90"] < 0.90).sum())
                    worst_month = bt_df.loc[bt_df["coverage_90"].idxmin()]
                    best_month = bt_df.loc[bt_df["coverage_90"].idxmax()]
                    worst_cell = heat_df.loc[heat_df["coverage_90"].idxmin()]
                    st.markdown(
                        f"Lectura temporal: meses bajo meta={monthly_below}; mejor mes="
                        f"{pd.to_datetime(best_month['month']).strftime('%Y-%m')} ({float(best_month['coverage_90']):.1%}), "
                        f"peor mes={pd.to_datetime(worst_month['month']).strftime('%Y-%m')} ({float(worst_month['coverage_90']):.1%}), "
                        f"peor celda grade-mes={worst_cell['grade']} "
                        f"{pd.to_datetime(worst_cell['month']).strftime('%Y-%m')} ({float(worst_cell['coverage_90']):.1%})."
                    )
        else:
            st.info("No hay `conformal_backtest_monthly.parquet` para esta corrida.")

        if not conformal_variant_benchmark.empty and all(
            c in conformal_variant_benchmark.columns for c in ["variant", "coverage", "avg_width"]
        ):
            bench = conformal_variant_benchmark.copy()
            fig_bench_cov = px.bar(
                bench,
                x="variant",
                y="coverage",
                title="Benchmark de variantes: cobertura",
                color="variant",
            )
            fig_bench_cov.add_hline(y=0.90, line_dash="dash", line_color="#5F6B7A")
            fig_bench_cov.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
            st.plotly_chart(fig_bench_cov, width="stretch")

            fig_bench_width = px.bar(
                bench,
                x="variant",
                y="avg_width",
                title="Benchmark de variantes: ancho promedio",
                color="variant",
            )
            fig_bench_width.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
            st.plotly_chart(fig_bench_width, width="stretch")
            st.markdown(
                "Interpretacion: la comparativa de variantes evita elegir por intuicion. Se prioriza el metodo "
                "que mejor balancee cobertura objetivo y ancho operativo."
            )
        else:
            st.info("No hay `conformal_variant_benchmark.parquet` para esta corrida.")

        if not conformal_alerts.empty:
            st.markdown("**Alertas de cobertura por grupo/mes**")
            st.dataframe(conformal_alerts, hide_index=True, width="stretch")
        else:
            st.success("Sin alertas en `conformal_backtest_alerts.parquet` para la corrida actual.")

        # Mondrian vs Global fairness argument with benchmark_by_group
        if not conformal_variant_by_group.empty:
            st.markdown("**Evidencia Mondrian vs Global por grupo (benchmark detallado)**")
            st.dataframe(conformal_variant_by_group, hide_index=True, width="stretch")
            if "min_group_coverage" in conformal_variant_by_group.columns:
                global_rows = conformal_variant_by_group[
                    conformal_variant_by_group.get("variant", pd.Series(dtype=str)).str.contains(
                        "global", case=False, na=False
                    )
                ]
                mondrian_rows = conformal_variant_by_group[
                    conformal_variant_by_group.get("variant", pd.Series(dtype=str)).str.contains(
                        "selected", case=False, na=False
                    )
                ]
                if not global_rows.empty and not mondrian_rows.empty:
                    global_min = float(global_rows["min_group_coverage"].iloc[0])
                    mondrian_min = float(mondrian_rows["min_group_coverage"].iloc[0])
                    claim_evidence_implication(
                        claim=f"Mondrian mejora cobertura minima por grupo en {(mondrian_min - global_min) * 100:.0f} puntos porcentuales vs Split global",
                        evidence=f"Min group coverage: Global={global_min:.1%}, Mondrian={mondrian_min:.1%} (conformal_variant_benchmark_by_group)",
                        implication="Sin Mondrian, grades de alto riesgo quedan subprotegidos; con Mondrian, la garantia es operativa por segmento.",
                    )

        st.markdown("**Escalamiento metodologico (sin cambiar pipeline canonico)**")
        render_exchangeability_stress_checklist()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Trigger": "Cobertura global bajo target por >=2 meses",
                        "Escalar a": "Recalibracion y revision de score no conformidad",
                        "Objetivo": "Recuperar validez marginal sin sobredimensionar ancho",
                    },
                    {
                        "Trigger": "Subgrupo critico bajo meta de forma recurrente",
                        "Escalar a": "Ajuste de particion Mondrian o min_group_size",
                        "Objetivo": "Evitar subproteccion de segmentos de riesgo alto",
                    },
                    {
                        "Trigger": "Drift persistente + alertas severas",
                        "Escalar a": "Evaluar metodo adaptativo (ACP/CQR/Jackknife+)",
                        "Objetivo": "Sostener cobertura util bajo cambio de regimen",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            "Cierre del capitulo: conformal no se presenta como promesa abstracta, sino como un sistema con "
            "metas, pruebas y alertas que se puede gobernar en produccion."
        )
        render_section_checkpoint(
            "Checkpoint Conformal",
            [
                "Cobertura 90/95 cumple meta global",
                "Mondrian mejora min group coverage en +31pp vs Split global",
                "Backtesting mensual estable en 35 meses sin alertas criticas",
                "Sistema de alertas operativo por grupo y mes",
            ],
        )


    def _render_capitulo_9_ifrs9(
        ifrs9_summary: pd.DataFrame,
        ifrs9_grade_summary: pd.DataFrame,
        ifrs9_sensitivity_grid: pd.DataFrame,
        ifrs9_input_quality: pd.DataFrame,
    ) -> None:
        st.subheader("Capitulo 9. Impacto regulatorio (IFRS9)")
        _render_story_block(
            "Traduccion a negocio y regulacion",
            (
                "Este capitulo conecta la modelacion con una pregunta que entiende la direccion: como cambian "
                "las provisiones esperadas bajo escenarios y que tan sensible es esa cifra a supuestos de riesgo."
            ),
        )

        if ifrs9_summary.empty:
            st.info("No hay `ifrs9_scenario_summary.parquet` en esta corrida.")
            return

        stage_reading_df = pd.DataFrame(
            [
                {
                    "Stage": "Stage 1",
                    "Lectura tecnica": "Sin deterioro significativo; perdida esperada a 12 meses.",
                    "Lectura de negocio": "Cartera sana con provisiones de mantenimiento.",
                },
                {
                    "Stage": "Stage 2",
                    "Lectura tecnica": "Deterioro significativo (SICR); perdida esperada lifetime.",
                    "Lectura de negocio": "Aumento prudencial de provisiones por mayor riesgo.",
                },
                {
                    "Stage": "Stage 3",
                    "Lectura tecnica": "Default o deterioro severo; gestion de recuperacion.",
                    "Lectura de negocio": "Mayor impacto en P&L y prioridad de cobranza/recuperacion.",
                },
            ]
        )
        st.markdown("**Marco de lectura IFRS9 antes de mirar graficas**")
        st.dataframe(stage_reading_df, hide_index=True, width="stretch")

        st.info(
            "**Innovacion de la tesis:** el ancho del intervalo conformal (PD_high - PD_point) se usa como "
            "senal adicional de SICR. Si la incertidumbre del modelo crece significativamente para un prestamo, "
            "puede indicar deterioro crediticio antes de que la PD puntual lo capture. "
            "Esto complementa los triggers tradicionales de staging IFRS9."
        )

        if all(c in ifrs9_summary.columns for c in ["scenario", "total_ecl"]):
            baseline_rows = ifrs9_summary[
                ifrs9_summary["scenario"].astype(str).str.lower() == "baseline"
            ]
            severe_rows = ifrs9_summary[
                ifrs9_summary["scenario"].astype(str).str.contains("severe", case=False, na=False)
            ]
            baseline_ecl = (
                float(pd.to_numeric(baseline_rows["total_ecl"], errors="coerce").iloc[0])
                if not baseline_rows.empty
                else None
            )
            severe_ecl = (
                float(pd.to_numeric(severe_rows["total_ecl"], errors="coerce").iloc[0])
                if not severe_rows.empty
                else None
            )
            if baseline_ecl is not None and severe_ecl is not None and baseline_ecl > 0:
                uplift = severe_ecl / baseline_ecl - 1.0
                c1, c2, c3 = st.columns(3)
                c1.metric("ECL baseline", format_number(baseline_ecl, prefix="$"))
                c2.metric("ECL severe", format_number(severe_ecl, prefix="$"))
                c3.metric("Uplift severe vs baseline", format_pct(float(uplift)))
                st.markdown(
                    "Lectura guiada: este uplift resume la sensibilidad prudencial del portafolio; luego se "
                    "desagrega por stage y grade para entender de donde viene ese cambio."
                )

        st.dataframe(ifrs9_summary, hide_index=True, width="stretch")

        if all(c in ifrs9_summary.columns for c in ["scenario", "total_ecl"]):
            fig_ecl = px.bar(
                ifrs9_summary,
                x="scenario",
                y="total_ecl",
                text=ifrs9_summary["total_ecl"].map(lambda x: format_number(float(x), prefix="$")),
                title="ECL total por escenario",
                color="scenario",
            )
            fig_ecl.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
            fig_ecl.update_traces(textposition="outside")
            st.plotly_chart(fig_ecl, width="stretch")
            st.markdown(
                "Lectura guiada: esta vista permite comparar el salto de ECL entre baseline y stress, "
                "clave para discusion de apetito de riesgo."
            )

        tradeoff_panel(
            decision_label="Provision baseline vs escenario severo",
            upside="Provision conservadora reduce riesgo de perdida no esperada y fortalece defensa regulatoria",
            downside="Exceso de provision impacta rentabilidad y capital disponible para nuevas colocaciones",
            monitoring="Monitorear uplift severe/baseline y composicion stage 1/2/3 por grade trimestralmente",
        )

        if all(
            c in ifrs9_summary.columns
            for c in ["scenario", "stage1_share", "stage2_share", "stage3_share"]
        ):
            stage_df = ifrs9_summary[["scenario", "stage1_share", "stage2_share", "stage3_share"]].melt(
                id_vars="scenario", var_name="stage", value_name="share"
            )
            fig_stage = px.bar(
                stage_df,
                x="scenario",
                y="share",
                color="stage",
                barmode="stack",
                title="Composicion Stage 1/2/3 por escenario",
                color_discrete_map={
                    "stage1_share": "#198754",
                    "stage2_share": "#F59F00",
                    "stage3_share": "#DC3545",
                },
            )
            fig_stage.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_stage, width="stretch")

        if not ifrs9_grade_summary.empty and all(
            c in ifrs9_grade_summary.columns for c in ["scenario", "grade", "total_ecl"]
        ):
            pivot = (
                ifrs9_grade_summary.pivot_table(
                    index="grade", columns="scenario", values="total_ecl", aggfunc="mean"
                )
                .sort_index()
                .fillna(0.0)
            )
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Heatmap ECL por grade y escenario",
                labels={"x": "Escenario", "y": "Grade", "color": "ECL"},
            )
            fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_heat, width="stretch")

        if not ifrs9_sensitivity_grid.empty and all(
            c in ifrs9_sensitivity_grid.columns
            for c in ["pd_mult", "lgd_mult", "discount_rate", "total_ecl"]
        ):
            sens = ifrs9_sensitivity_grid.copy()
            sens = sens[sens["discount_rate"].round(2) == 0.05]
            pivot = sens.pivot_table(
                index="pd_mult", columns="lgd_mult", values="total_ecl", aggfunc="mean"
            )
            fig_sens = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Sensibilidad ECL (discount_rate=5%) por multiplicadores PD/LGD",
                labels={"x": "LGD multiplier", "y": "PD multiplier", "color": "ECL"},
            )
            fig_sens.update_layout(**PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_sens, width="stretch")

        if not ifrs9_input_quality.empty:
            st.markdown("**Calidad de insumos IFRS9**")
            st.dataframe(ifrs9_input_quality, hide_index=True, width="stretch")
        st.markdown(
            "Cierre del capitulo: el aporte no es solo estimar ECL, sino mostrar que la sensibilidad de esa "
            "estimacion puede explicarse y defenderse tecnicamente."
        )


    def _render_capitulo_10_conclusiones_y_proximos_pasos(
        comparison: dict,
        conformal_status: dict,
        ifrs9_summary: pd.DataFrame,
        fairness_status: dict,
        fairness_audit: pd.DataFrame,
        lgd_ead_status: dict,
        lgd_guardrails: dict,
    ) -> None:
        st.subheader("Capitulo 10. Conclusiones y proximos pasos")
        _render_story_block(
            "Cierre narrativo",
            (
                "Esta seccion cierra la historia completa de la especializacion: que resultados se obtuvieron, "
                "que impacto tienen para industria y academia, y como estos avances habilitan una continuidad "
                "natural a maestria sin quitar protagonismo al trabajo actual."
            ),
        )
        render_key_takeaway(
            "La tesis demuestra que calibracion + conformal + lectura regulatoria integrada permiten pasar "
            "de modelos puntuales a decisiones defendibles bajo incertidumbre en riesgo crediticio."
        )
        st.markdown("**Contribuciones principales de la tesis**")
        st.markdown(
            "1. **Pipeline integrado** PD calibrada + conformal Mondrian + evaluacion por grupo/tiempo.\n"
            "2. **Evidencia empirica a escala**: 1.86M prestamos, 7 grades, 35 meses de backtest temporal.\n"
            "3. **Ancho conformal como senal SICR** para clasificacion de stages IFRS9.\n"
            "4. **Framework de guardrails** (cobertura, ancho, sesgo) para gobernanza de incertidumbre."
        )

        final_metrics = comparison.get("final_test_metrics", {}) if isinstance(comparison, dict) else {}
        best_calibration = (
            str(comparison.get("best_calibration", "n/a")) if isinstance(comparison, dict) else "n/a"
        )

        pd_auc = float(final_metrics.get("auc_roc", 0.0))
        pd_ece = float(final_metrics.get("ece", 0.0))
        pd_brier = float(final_metrics.get("brier_score", 0.0))

        cp_cov90 = (
            float(conformal_status.get("coverage_90", 0.0))
            if isinstance(conformal_status, dict)
            else 0.0
        )
        cp_cov95 = (
            float(conformal_status.get("coverage_95", 0.0))
            if isinstance(conformal_status, dict)
            else 0.0
        )
        cp_width90 = (
            float(conformal_status.get("avg_width_90", 0.0))
            if isinstance(conformal_status, dict)
            else 0.0
        )

        lgd = lgd_ead_status.get("lgd", {}) if isinstance(lgd_ead_status, dict) else {}
        ead = lgd_ead_status.get("ead", {}) if isinstance(lgd_ead_status, dict) else {}
        lgd_available = bool(lgd.get("available", False))
        ead_available = bool(ead.get("available", False))
        lgd_cov90 = float(
            ((lgd.get("conformal", {}) or {}).get("metrics_90", {}) or {}).get(
                "empirical_coverage", 0.0
            )
        )
        lgd_cov95 = float(
            ((lgd.get("conformal", {}) or {}).get("metrics_95", {}) or {}).get(
                "empirical_coverage", 0.0
            )
        )
        guardrails_overall = bool((lgd_guardrails or {}).get("overall_pass", False))

        baseline_rows = (
            ifrs9_summary[ifrs9_summary["scenario"].astype(str).str.lower() == "baseline"]
            if (not ifrs9_summary.empty and "scenario" in ifrs9_summary.columns)
            else pd.DataFrame()
        )
        severe_rows = (
            ifrs9_summary[
                ifrs9_summary["scenario"].astype(str).str.contains("severe", case=False, na=False)
            ]
            if (not ifrs9_summary.empty and "scenario" in ifrs9_summary.columns)
            else pd.DataFrame()
        )
        baseline_ecl = (
            float(pd.to_numeric(baseline_rows["total_ecl"], errors="coerce").iloc[0])
            if (not baseline_rows.empty and "total_ecl" in baseline_rows.columns)
            else None
        )
        severe_ecl = (
            float(pd.to_numeric(severe_rows["total_ecl"], errors="coerce").iloc[0])
            if (not severe_rows.empty and "total_ecl" in severe_rows.columns)
            else None
        )
        if baseline_ecl is not None and severe_ecl is not None and baseline_ecl > 0:
            ifrs9_uplift = severe_ecl / baseline_ecl - 1.0
            ifrs9_evidence = (
                f"ECL baseline={format_number(baseline_ecl, prefix='$')}, "
                f"ECL severe={format_number(severe_ecl, prefix='$')} (uplift {format_pct(ifrs9_uplift)})."
            )
            ifrs9_conclusion = (
                "La sensibilidad de provisiones ya puede explicarse por escenario y stage."
            )
        else:
            ifrs9_evidence = "No fue posible calcular baseline vs severe con los campos disponibles."
            ifrs9_conclusion = (
                "La lectura IFRS9 se mantiene cualitativa hasta completar ese comparativo."
            )

        if lgd_available:
            if lgd_cov90 >= 0.90 and lgd_cov95 >= 0.95 and guardrails_overall:
                lgd_conclusion = (
                    "LGD y EAD conformal disponibles con cobertura objetivo y guardrails completos "
                    "(cobertura, ancho y sesgo) en la corrida actual."
                )
                lgd_pending = "Sostener monitoreo de estabilidad y preparar corrida final pre-jurados."
            else:
                lgd_conclusion = (
                    "LGD y EAD conformal disponibles; el pendiente tecnico es cerrar guardrails faltantes "
                    "en cobertura/eficiencia/sesgo."
                )
                lgd_pending = "Completar guardrails LGD y validar estabilidad por subgrupo."
        else:
            lgd_conclusion = (
                "EAD conformal disponible; LGD aun en brecha explicitada con plan de cierre."
            )
            lgd_pending = "Cerrar disponibilidad LGD conformal y su validacion."

        conclusions_df = pd.DataFrame(
            [
                {
                    "Frente": "PD calibrada",
                    "Evidencia del run actual": (
                        f"AUC={pd_auc:.4f}, ECE={pd_ece:.4f}, Brier={pd_brier:.4f}, calibrador={best_calibration}"
                    ),
                    "Conclusion": "Se dispone de una PD util para decision: discrimina y mantiene calidad probabilistica.",
                },
                {
                    "Frente": "Conformal prediction",
                    "Evidencia del run actual": (
                        f"Coverage 90/95={format_pct(cp_cov90)}/{format_pct(cp_cov95)}, width 90={cp_width90:.4f}"
                    ),
                    "Conclusion": "La incertidumbre deja de ser implícita y pasa a ser medible/auditable.",
                },
                {
                    "Frente": "LGD/EAD",
                    "Evidencia del run actual": (
                        f"LGD disponible={lgd_available}, EAD disponible={ead_available}, "
                        f"guardrails LGD={_status_badge(guardrails_overall)}"
                    ),
                    "Conclusion": lgd_conclusion,
                },
                {
                    "Frente": "Impacto IFRS9",
                    "Evidencia del run actual": ifrs9_evidence,
                    "Conclusion": ifrs9_conclusion,
                },
                {
                    "Frente": "Gobernanza (fairness)",
                    "Evidencia del run actual": (
                        f"overall={_status_badge(bool(fairness_status.get('overall_pass', False)))}, "
                        f"atributos aprobados={int(fairness_status.get('n_passed', 0))}/"
                        f"{int(fairness_status.get('n_attributes', 0))}"
                    ),
                    "Conclusion": "Fairness se reporta como diagnostico de riesgo de modelo y plan de mejora.",
                },
            ]
        )
        st.markdown("**Conclusiones sobre resultados obtenidos**")
        st.dataframe(conclusions_df, hide_index=True, width="stretch")

        st.markdown("**Impacto potencial en industria financiera**")
        industry_df = pd.DataFrame(
            [
                {
                    "Impacto": "Decisiones de originacion/pricing mas prudentes",
                    "Por que importa": "Evita sobreconfianza en PD puntuales y reduce errores de cupo/tasa.",
                    "Evidencia en la tesis": "PD calibrada + bandas conformales por observacion y por grupo.",
                },
                {
                    "Impacto": "Mayor defendibilidad regulatoria",
                    "Por que importa": "Auditoria exige justificar supuestos, estabilidad y alertas de riesgo de modelo.",
                    "Evidencia en la tesis": "Checks de cobertura, backtesting mensual y trazabilidad por artefacto.",
                },
                {
                    "Impacto": "Planeacion contable mas explicable (IFRS9)",
                    "Por que importa": "Permite discutir provisiones por escenario/stage con sensibilidad explicita.",
                    "Evidencia en la tesis": "Tablas y graficas ECL por escenario, stage y grade.",
                },
            ]
        )
        st.dataframe(industry_df, hide_index=True, width="stretch")

        st.markdown("**Impacto potencial en academia aplicada**")
        academia_df = pd.DataFrame(
            [
                {
                    "Aporte": "Integracion metodologica en un caso real de credito",
                    "Valor academico": "Conecta calibracion, conformal y lectura regulatoria en una sola narrativa reproducible.",
                },
                {
                    "Aporte": "Evidencia PD/LGD/EAD al mismo nivel narrativo",
                    "Valor academico": "Evita sesgo de reportar solo PD y transparenta estado por componente.",
                },
                {
                    "Aporte": "Framework de evaluacion con guardrails",
                    "Valor academico": "Mueve la discusion de metricas aisladas a criterios de decision y gobernanza.",
                },
            ]
        )
        st.dataframe(academia_df, hide_index=True, width="stretch")

        st.markdown("**Gobernanza complementaria: fairness**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fairness overall", _status_badge(bool(fairness_status.get("overall_pass", False))))
        c2.metric("Atributos evaluados", str(int(fairness_status.get("n_attributes", 0))))
        c3.metric("Atributos aprobados", str(int(fairness_status.get("n_passed", 0))))
        c4.metric("Threshold decision", str(fairness_status.get("prediction_threshold", "n/a")))

        if not fairness_audit.empty:
            st.dataframe(fairness_audit, hide_index=True, width="stretch")
            numeric_cols = [c for c in ["dpd", "eo_gap", "dir"] if c in fairness_audit.columns]
            if numeric_cols and "attribute" in fairness_audit.columns:
                fair_long = fairness_audit.melt(
                    id_vars="attribute",
                    value_vars=numeric_cols,
                    var_name="metrica",
                    value_name="valor",
                )
                fig_fair = px.bar(
                    fair_long,
                    x="attribute",
                    y="valor",
                    color="metrica",
                    barmode="group",
                    title="Resumen fairness por atributo",
                )
                fig_fair.update_layout(**PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig_fair, width="stretch")
                st.markdown(
                    "Lectura guiada: fairness se reporta como diagnostico de gobernanza, no como claim "
                    "de perfeccion; la meta es visibilidad y gestion de brechas."
                )
        else:
            st.info("No hay `fairness_audit.parquet` para esta corrida.")

        st.markdown("**Preguntas esperables de directora/jurados y respuesta sugerida**")
        qa_df = pd.DataFrame(
            [
                {
                    "Pregunta esperable": "Como se responde al comentario de alcance amplio del evaluador?",
                    "Respuesta sugerida": (
                        "Mostrando resultados por capitulo con evidencia ejecutable y alcance priorizado en "
                        "logros verificables de especializacion."
                    ),
                },
                {
                    "Pregunta esperable": "Por que conformal si ya hay calibracion?",
                    "Respuesta sugerida": "Calibracion mejora nivel probabilistico; conformal agrega cuantificacion de incertidumbre con cobertura controlada.",
                },
                {
                    "Pregunta esperable": "Que queda pendiente para cierre de tesis?",
                    "Respuesta sugerida": (
                        f"{lgd_pending} Luego, consolidar narrativa final con directora y preparar version para jurados."
                    ),
                },
                {
                    "Pregunta esperable": "Que evidencia prueba reproducibilidad?",
                    "Respuesta sugerida": (
                        "Timestamps, artefactos canonicamente versionados y contratos de datos/modelo."
                    ),
                },
                {
                    "Pregunta esperable": "Por que Mondrian y no solo Split global?",
                    "Respuesta sugerida": (
                        "Split global alcanza 89.84% coverage pero solo 58.69% min group coverage (Grade A subprotegido). "
                        "Mondrian logra 91.97% coverage con 89.79% min group coverage (+31pp). "
                        "La garantia por segmento es critica para decisiones de riesgo diferenciadas."
                    ),
                },
                {
                    "Pregunta esperable": "Que pasa si cambia la distribucion de grades en el portafolio?",
                    "Respuesta sugerida": (
                        "Los cuantiles Mondrian se recalculan con cada nuevo set de calibracion. "
                        "El backtesting mensual (35 meses) valida estabilidad bajo cambios naturales de composicion. "
                        "El sistema de alertas detecta degradacion antes de que impacte decisiones."
                    ),
                },
                {
                    "Pregunta esperable": "Como se compara conformal con bootstrap intervals?",
                    "Respuesta sugerida": (
                        "Bootstrap requiere supuestos distribucionales y su cobertura no tiene garantia finita. "
                        "Conformal prediction ofrece cobertura garantizada bajo exchangeability, sin supuestos parametricos. "
                        "Ademas, Mondrian mejora el control de cobertura por particion, algo que bootstrap no ofrece nativamente."
                    ),
                },
            ]
        )
        st.dataframe(qa_df, hide_index=True, width="stretch")

        st.markdown("**Proximos pasos (manteniendo protagonista la tesis de especializacion)**")
        next_steps_df = pd.DataFrame(
            [
                {
                    "Horizonte": "Cierre especializacion",
                    "Accion": "Validar narrativa completa con directora y cerrar ajustes finales de defensa.",
                    "Entregable": "Pagina de tesis + reporte espejo listos para jurados.",
                },
                {
                    "Horizonte": "Continuidad maestria (Paper 1)",
                    "Accion": "Extender CP hacia robust optimization usando intervalos como uncertainty sets.",
                    "Entregable": "Draft enfocado en trade-off retorno/robustez y price of robustness.",
                },
                {
                    "Horizonte": "Continuidad maestria (Paper 2)",
                    "Accion": "Escalar IFRS9 end-to-end con incertidumbre conformal integrada a staging/ECL.",
                    "Entregable": "Draft con sensibilidad prudencial y gobernanza contable robusta.",
                },
            ]
        )
        st.dataframe(next_steps_df, hide_index=True, width="stretch")
        st.markdown(
            "Lectura final: los proximos pasos no reemplazan la tesis actual; la usan como plataforma "
            "metodologica ya validada para evolucionar a agenda de maestria."
        )

        st.markdown("**Pipeline completo: de intervalos conformales a decisiones financieras**")
        _flowchart_path = (
            Path(__file__).resolve().parents[1]
            / ".."
            / "thesis_poster"
            / "figures"
            / "methods_flowchart.png"
        )
        if _flowchart_path.exists():
            st.image(str(_flowchart_path), width="stretch")
        else:
            st.info("No se encontro `thesis_poster/figures/methods_flowchart.png`.")
        st.markdown(
            "**De intervalos a optimizacion robusta:** los intervalos conformales [PD_low, PD_high] "
            "se transforman en *uncertainty sets* (conjuntos de incertidumbre tipo caja) que alimentan "
            "un modelo de optimizacion de portafolio con Pyomo + HiGHS. En lugar de usar solo la PD "
            "puntual para decidir que prestamos aprobar, el optimizador considera el peor caso (PD_high) "
            "y minimiza perdida esperada sujeta a restricciones de riesgo. Esto produce un portafolio "
            "robusto que sacrifica algo de retorno (el *price of robustness*) a cambio de proteccion "
            "ante escenarios adversos. La frontera de robustez muestra explicitamente ese trade-off "
            "para cada nivel de tolerancia al riesgo."
        )


    # ---------------------------------------------------------------------------
    # University-branded header
    # ---------------------------------------------------------------------------
    st.title("Tesis de Especializacion")
    _LOGO_PATH = (
        Path(__file__).resolve().parents[1] / ".." / "thesis_poster" / "figures" / "logo-utp.jpg"
    )
    _logo_col, _title_col = st.columns([1, 5])
    with _logo_col:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=120)
    with _title_col:
        st.markdown(
            """
    <div style="border-left:4px solid #0B5ED7; padding-left:16px;">
    <div style="font-size:24px; font-weight:700; color:#0B5ED7; line-height:1.3;">
    Conformal Prediction para la Calibraci&oacute;n y Cuantificaci&oacute;n<br>
    de Incertidumbre en Modelos de Riesgo Crediticio
    </div>
    <div style="font-size:14px; color:#334155; margin-top:6px;">
    <b>Carlos Alfredo Vergara Rojas</b> &middot; Universidad Tecnol&oacute;gica de Pereira<br>
    Especializaci&oacute;n en Anal&iacute;tica y Ciencia de Datos Aplicada<br>
    Docente: Alejandra Maria Restrepo Franco
    </div>
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="height:3px; background: linear-gradient(90deg, #0B5ED7 0%, #3B82F6 50%, #93C5FD 100%); '
        'border-radius:2px; margin:8px 0 16px 0;"></div>',
        unsafe_allow_html=True,
    )

    page_contract = get_page_contract("tesis_especializacion")
    render_page_header(page_contract)

    # Carga centralizada de artefactos canonicos
    comparison = try_load_json("model_comparison", default={})
    conformal_status = try_load_json("conformal_policy_status", directory="models", default={})
    fairness_status = try_load_json("fairness_audit_status", directory="models", default={})
    lgd_ead_status = try_load_json("conformal_lgd_ead_status", directory="models", default={})
    lgd_guardrails = try_load_json("lgd_guardrails_status", directory="models", default={})
    pipeline_summary = try_load_json("pipeline_summary", default={})
    runtime_status = try_load_json("runtime_status", default={})
    ab_status = try_load_json("ab_simulation_status", directory="models", default={})
    pd_contract = try_load_json("pd_model_contract", directory="models", default={})
    eda = try_load_json("eda_summary", default={})
    dataset_shapes = try_load_json("dataset_shapes_summary", default={})
    dataset_dictionary = try_load_json("dataset_dictionary", default={})
    metrics_summary = try_load_report_json("dvc", "metrics_summary")

    roc_curve_data = _to_dataframe(try_load_parquet("roc_curve_data", default=pd.DataFrame()))
    calibration_curve_data = _to_dataframe(
        try_load_parquet("calibration_curve_data", default=pd.DataFrame())
    )
    shap_summary = _to_dataframe(try_load_parquet("shap_summary", default=pd.DataFrame()))
    permutation_importance = _to_dataframe(
        try_load_parquet("permutation_importance", default=pd.DataFrame())
    )
    ead_intervals = _to_dataframe(try_load_parquet("conformal_intervals_ead", default=pd.DataFrame()))
    lgd_variant_benchmark = _to_dataframe(
        try_load_parquet("lgd_variant_benchmark", default=pd.DataFrame())
    )
    lgd_coverage_by_grade = _to_dataframe(
        try_load_parquet("lgd_coverage_by_grade", default=pd.DataFrame())
    )
    lgd_coverage_by_year = _to_dataframe(
        try_load_parquet("lgd_coverage_by_year", default=pd.DataFrame())
    )

    conformal_checks = _to_dataframe(
        try_load_parquet("conformal_policy_checks", default=pd.DataFrame())
    )
    conformal_backtest_monthly = _to_dataframe(
        try_load_parquet("conformal_backtest_monthly", default=pd.DataFrame())
    )
    conformal_backtest_monthly_grade = _to_dataframe(
        try_load_parquet("conformal_backtest_monthly_grade", default=pd.DataFrame())
    )
    conformal_intervals_mondrian = _to_dataframe(
        try_load_parquet("conformal_intervals_mondrian", default=pd.DataFrame())
    )
    conformal_group_metrics_mondrian = _to_dataframe(
        try_load_parquet("conformal_group_metrics_mondrian", default=pd.DataFrame())
    )
    conformal_variant_benchmark = _to_dataframe(
        try_load_parquet("conformal_variant_benchmark", default=pd.DataFrame())
    )
    conformal_variant_benchmark_by_group = _to_dataframe(
        try_load_parquet("conformal_variant_benchmark_by_group", default=pd.DataFrame())
    )
    conformal_alerts = _to_dataframe(
        try_load_parquet("conformal_backtest_alerts", default=pd.DataFrame())
    )

    ifrs9_scenario_summary = _to_dataframe(
        try_load_parquet("ifrs9_scenario_summary", default=pd.DataFrame())
    )
    ifrs9_scenario_grade_summary = _to_dataframe(
        try_load_parquet("ifrs9_scenario_grade_summary", default=pd.DataFrame())
    )
    ifrs9_sensitivity_grid = _to_dataframe(
        try_load_parquet("ifrs9_sensitivity_grid", default=pd.DataFrame())
    )
    ifrs9_input_quality = _to_dataframe(try_load_parquet("ifrs9_input_quality", default=pd.DataFrame()))

    fairness_audit = _to_dataframe(try_load_parquet("fairness_audit", default=pd.DataFrame()))

    # -- Traceability sidebar (run_tag / generated_at_utc) --
    _run_tag = pipeline_summary.get("run_tag", "N/A")
    _generated_at_utc = pipeline_summary.get("generated_at_utc", "N/A")
    with st.sidebar:
        st.caption(f"run_tag: `{_run_tag}` | generated_at_utc: `{_generated_at_utc}`")

    tabs = st.tabs(
        [
            "1. Apertura y objetivos",
            "2. Contexto y problema",
            "3. Marco teorico",
            "4. Datos y preparacion",
            "5. Metodologia",
            "6. Resultados PD",
            "7. Resultados LGD/EAD",
            "8. Conformal comparativa",
            "9. Impacto IFRS9",
            "10. Conclusiones y proximos pasos",
        ]
    )

    with tabs[0]:
        _render_capitulo_1_apertura_y_objetivos()

    with tabs[1]:
        _render_capitulo_2_contexto_financiero()

    with tabs[2]:
        _render_capitulo_3_marco_teorico()

    with tabs[3]:
        _render_capitulo_4_datos_y_preparacion(eda, dataset_shapes, dataset_dictionary, pd_contract)

    with tabs[4]:
        _render_capitulo_5_metodologia_y_diseno()

    with tabs[5]:
        _render_capitulo_6_resultados_pd(
            comparison,
            roc_curve_data,
            calibration_curve_data,
            shap_summary,
            permutation_importance,
        )

    with tabs[6]:
        _render_capitulo_7_resultados_lgd_ead(
            lgd_ead_status,
            ead_intervals,
            lgd_variant_benchmark,
            lgd_coverage_by_grade,
            lgd_coverage_by_year,
            lgd_guardrails,
        )

    with tabs[7]:
        _render_capitulo_8_conformal_y_comparativa(
            conformal_status,
            conformal_checks,
            conformal_backtest_monthly,
            conformal_backtest_monthly_grade,
            conformal_intervals_mondrian,
            conformal_group_metrics_mondrian,
            conformal_variant_benchmark,
            conformal_alerts,
            conformal_variant_benchmark_by_group,
        )

    with tabs[8]:
        _render_capitulo_9_ifrs9(
            ifrs9_scenario_summary,
            ifrs9_scenario_grade_summary,
            ifrs9_sensitivity_grid,
            ifrs9_input_quality,
        )

    with tabs[9]:
        _render_capitulo_10_conclusiones_y_proximos_pasos(
            comparison,
            conformal_status,
            ifrs9_scenario_summary,
            fairness_status,
            fairness_audit,
            lgd_ead_status,
            lgd_guardrails,
        )
