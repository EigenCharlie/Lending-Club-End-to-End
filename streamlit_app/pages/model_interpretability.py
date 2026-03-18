"""Pagina dedicada a explicabilidad e interpretabilidad operacional del modelo PD."""

from __future__ import annotations

import json
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

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.context_help import (
    chart_help_popover,
    methodology_dialog,
    term_popover,
)
from streamlit_app.components.dvc_kpi_spine import render_global_kpi_spine
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import narrative_block, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_decision_box,
    render_key_takeaway,
    render_next_steps,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import get_notebook_image_path, try_load_json, try_load_parquet


def _parse_json_payload(value: object, default: object) -> object:
    if value in (None, "", "nan"):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _load_explainability_global() -> pd.DataFrame:
    global_df = try_load_parquet("explainability_global")
    if not global_df.empty:
        out = global_df.copy()
    else:
        shap_summary = try_load_parquet("shap_summary")
        permutation_df = try_load_parquet("permutation_importance")
        if shap_summary.empty:
            return pd.DataFrame()
        out = shap_summary.copy()
        if not permutation_df.empty:
            perm_cols = permutation_df[["feature", "auc_drop"]].rename(
                columns={"auc_drop": "permutation_auc_drop"}
            )
            out = out.merge(perm_cols, on="feature", how="left")

    if "feature_family" not in out.columns:
        out["feature_family"] = "unknown"
    if "business_label" not in out.columns:
        out["business_label"] = out["feature"].astype(str)
    if "controllable" not in out.columns:
        out["controllable"] = False
    if "monotonic_expected" not in out.columns:
        out["monotonic_expected"] = "none"
    if "preferred_effect_view" not in out.columns:
        out["preferred_effect_view"] = "none"
    if "permutation_auc_drop" not in out.columns:
        out["permutation_auc_drop"] = np.nan
    if "joint_rank" not in out.columns:
        out["joint_rank"] = np.arange(1, len(out) + 1)
    return out.sort_values(["joint_rank", "mean_abs_shap"], ascending=[True, False]).reset_index(
        drop=True
    )


def _format_interval(interval_payload: dict[str, object], low_key: str, high_key: str) -> str:
    low = interval_payload.get(low_key)
    high = interval_payload.get(high_key)
    try:
        return f"[{float(low):.3f}, {float(high):.3f}]"
    except Exception:
        return "N/D"


def _build_driver_map_figure(global_df: pd.DataFrame) -> go.Figure:
    df = global_df.copy()
    df["permutation_auc_drop"] = pd.to_numeric(df["permutation_auc_drop"], errors="coerce")
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df["controllable_label"] = np.where(df["controllable"].astype(bool), "Controlable", "No controlable")
    label_features = set(df.nsmallest(min(8, len(df)), "joint_rank")["feature"].astype(str).tolist())
    df["label"] = df["feature"].astype(str).where(df["feature"].astype(str).isin(label_features), "")

    fig = px.scatter(
        df,
        x="permutation_auc_drop",
        y="mean_abs_shap",
        color="feature_family",
        symbol="controllable_label",
        hover_name="feature",
        text="label",
        hover_data={
            "business_label": True,
            "joint_rank": True,
            "preferred_effect_view": True,
            "permutation_auc_drop": ":.4f",
            "mean_abs_shap": ":.4f",
        },
        labels={
            "permutation_auc_drop": "Sensibilidad (AUC drop por permutacion)",
            "mean_abs_shap": "Atribucion media |SHAP|",
            "feature_family": "Familia",
            "controllable_label": "Tipo",
        },
        title="Mapa de drivers: atribucion vs sensibilidad",
    )
    fig.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=0.5, color="white")))
    median_x = float(df["permutation_auc_drop"].dropna().median()) if df["permutation_auc_drop"].notna().any() else 0.0
    median_y = float(df["mean_abs_shap"].dropna().median()) if df["mean_abs_shap"].notna().any() else 0.0
    fig.add_vline(x=median_x, line_dash="dot", line_color="#94A3B8")
    fig.add_hline(y=median_y, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=470)
    return fig


def _build_family_figure(global_df: pd.DataFrame) -> go.Figure:
    family_df = (
        global_df.groupby("feature_family", as_index=False)
        .agg(total_abs_shap=("mean_abs_shap", "sum"), n_features=("feature", "count"))
        .sort_values("total_abs_shap", ascending=True)
    )
    fig = px.bar(
        family_df,
        x="total_abs_shap",
        y="feature_family",
        orientation="h",
        text="n_features",
        labels={"total_abs_shap": "Masa total |SHAP|", "feature_family": "Familia"},
        title="Concentracion por familia de negocio",
    )
    fig.update_traces(texttemplate="%{text} vars", marker_line_width=0)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=470, showlegend=False)
    return fig


def _build_global_rank_figure(global_df: pd.DataFrame, top_n: int) -> go.Figure:
    df = global_df.nsmallest(min(top_n, len(global_df)), "joint_rank").copy()
    df = df.sort_values("mean_abs_shap", ascending=True)
    fig = px.bar(
        df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="feature_family",
        hover_data={
            "business_label": True,
            "permutation_auc_drop": ":.4f",
            "joint_rank": True,
            "controllable": True,
        },
        labels={"mean_abs_shap": "Impacto medio |SHAP|", "feature": "Variable"},
        title=f"Top {top_n} drivers globales",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=max(360, top_n * 28))
    return fig


def _prepare_case_reason_frames(case_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    positive = pd.DataFrame(_parse_json_payload(case_row.get("top_positive_reasons"), []))
    negative = pd.DataFrame(_parse_json_payload(case_row.get("top_negative_reasons"), []))
    family_summary = _parse_json_payload(case_row.get("feature_family_summary"), {})

    frames: list[pd.DataFrame] = []
    if not positive.empty:
        pos = positive.copy()
        pos["direction"] = "Sube PD"
        frames.append(pos)
    if not negative.empty:
        neg = negative.copy()
        neg["direction"] = "Baja PD"
        frames.append(neg)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined["feature"] = combined["feature"].astype(str)
        combined["feature_value"] = combined["feature_value"].astype(str)
        combined["shap_value"] = pd.to_numeric(combined["shap_value"], errors="coerce").fillna(0.0)
        combined["label"] = combined["feature"].astype(str) + " = " + combined["feature_value"].astype(str)
        combined["abs_shap"] = combined["shap_value"].abs()
        combined = combined.sort_values("shap_value", ascending=True)

    return combined, positive, negative, family_summary


def _build_local_reason_figure(reasons: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if reasons.empty:
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, title=title)
        return fig

    colors = ["#B42318" if value > 0 else "#0B7285" for value in reasons["shap_value"]]
    fig.add_trace(
        go.Bar(
            x=reasons["shap_value"],
            y=reasons["label"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Contribucion SHAP: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=360,
        title=title,
        xaxis_title="Contribucion al score del modelo",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def _build_ale_figure(ale_subset: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.line(
        ale_subset.sort_values("midpoint"),
        x="midpoint",
        y="ale_value",
        markers=True,
        labels={"midpoint": feature, "ale_value": "ALE sobre PD"},
        title=f"ALE canónico para {feature}",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
    return fig


def _build_pdp_ice_figure(pdp_subset: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()
    if pdp_subset.empty:
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, title=f"PDP/ICE para {feature}")
        return fig

    obs_ids = pdp_subset["observation_id"].drop_duplicates()
    n_obs = min(40, len(obs_ids))
    sample_ids = (
        obs_ids.sample(n_obs, random_state=17).tolist()
        if len(obs_ids) > n_obs
        else obs_ids.tolist()
    )
    for obs_id in sample_ids:
        row_df = pdp_subset[pdp_subset["observation_id"] == obs_id].sort_values("grid_value")
        fig.add_trace(
            go.Scatter(
                x=row_df["grid_value"],
                y=row_df["ice_pred"],
                mode="lines",
                line=dict(color="rgba(11, 94, 215, 0.10)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    pdp_line = (
        pdp_subset.groupby("grid_value", as_index=False)["pdp_pred"].mean().sort_values("grid_value")
    )
    fig.add_trace(
        go.Scatter(
            x=pdp_line["grid_value"],
            y=pdp_line["pdp_pred"],
            mode="lines+markers",
            line=dict(color="#0B5ED7", width=3),
            name="PDP",
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=380,
        title=f"PDP/ICE de respaldo para {feature}",
        xaxis_title=feature,
        yaxis_title="PD promedio",
        showlegend=False,
    )
    return fig


def _build_redundancy_heatmap(interaction_df: pd.DataFrame) -> go.Figure:
    plot_df = interaction_df.copy()
    flagged = plot_df[plot_df["redundancy_flag"].astype(bool)].copy()
    if not flagged.empty:
        plot_df = flagged
    plot_df = plot_df.head(18)
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, title="Mapa de redundancia")
        return fig

    features = sorted(set(plot_df["feature_a"].astype(str)).union(plot_df["feature_b"].astype(str)))
    matrix = pd.DataFrame(0.0, index=features, columns=features)
    np.fill_diagonal(matrix.values, 1.0)
    for _, row in plot_df.iterrows():
        feature_a = str(row["feature_a"])
        feature_b = str(row["feature_b"])
        raw_value = pd.to_numeric(pd.Series([row["shap_spearman"]]), errors="coerce").iloc[0]
        value = float(np.nan_to_num(raw_value))
        matrix.loc[feature_a, feature_b] = value
        matrix.loc[feature_b, feature_a] = value

    fig = px.imshow(
        matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={"x": "Feature", "y": "Feature", "color": "Spearman SHAP"},
        title="Mapa aproximado de interacciones/redundancia SHAP",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    return fig


def _build_shap_dependence_figure(shap_raw: pd.DataFrame, feature: str) -> go.Figure:
    shap_col = f"shap_{feature}"
    val_col = f"val_{feature}"
    sample = shap_raw[[val_col, shap_col]].copy()
    sample = sample.dropna().sample(min(2500, len(sample.dropna())), random_state=17)
    fig = px.scatter(
        sample,
        x=val_col,
        y=shap_col,
        opacity=0.35,
        labels={val_col: f"Valor de {feature}", shap_col: "Contribucion SHAP"},
        title=f"Dependencia SHAP para {feature}",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
    return fig


def _render_case_panel(case_row: pd.Series) -> None:
    interval_payload = _parse_json_payload(case_row.get("intervalo_conformal"), {})
    combined, positive, negative, family_summary = _prepare_case_reason_frames(case_row)

    kpi_row(
        [
            {"label": "Case ID", "value": str(case_row.get("case_id", "N/D"))},
            {"label": "Quarter", "value": str(case_row.get("issue_quarter", "N/D"))},
            {"label": "Grade", "value": str(case_row.get("grade", "N/D"))},
            {"label": "Score raw", "value": f"{float(case_row.get('score_raw', 0.0)):.4f}"},
            {"label": "PD calibrada", "value": f"{float(case_row.get('pd_calibrada', 0.0)):.4f}"},
        ],
        n_cols=5,
    )
    st.markdown(f"**Reason code canónico:** {case_row.get('reason_code_text', 'N/D')}")
    col_chart, col_meta = st.columns([1.2, 1.0], gap="large")
    with col_chart:
        st.plotly_chart(
            _build_local_reason_figure(
                combined,
                title=f"Contribuciones locales para {case_row.get('segmento', 'caso')}",
            ),
            width="stretch",
        )
    with col_meta:
        st.markdown("**Lectura operativa**")
        st.markdown(
            f"- Intervalo 90%: {_format_interval(interval_payload, 'pd_low_90', 'pd_high_90')}"
        )
        st.markdown(
            f"- Intervalo 95%: {_format_interval(interval_payload, 'pd_low_95', 'pd_high_95')}"
        )
        if family_summary:
            top_families = sorted(family_summary.items(), key=lambda item: (-item[1], item[0]))
            st.markdown(
                "**Familias dominantes:** "
                + ", ".join(f"{family} ({count})" for family, count in top_families[:5])
            )
        if not combined.empty:
            dominant = combined.iloc[combined["abs_shap"].idxmax()]
            st.markdown(
                f"**Driver dominante:** `{dominant['feature']}` con contribucion "
                f"{float(dominant['shap_value']):+.4f}."
            )

    col_pos, col_neg = st.columns(2, gap="large")
    with col_pos:
        st.markdown("**Factores que suben la PD**")
        if positive.empty:
            st.info("Sin factores positivos para este caso.")
        else:
            st.dataframe(positive, width="stretch", hide_index=True)
    with col_neg:
        st.markdown("**Factores que bajan la PD**")
        if negative.empty:
            st.info("Sin factores negativos para este caso.")
        else:
            st.dataframe(negative, width="stretch", hide_index=True)


def _render_notebook_gallery() -> None:
    gallery = [
        {
            "label": "EDA",
            "stem": "01_eda_lending_club",
            "file": "cell_025_out_00.png",
            "title": "Grade x term: estructura de riesgo desde el dato",
            "caption": "Notebook 01: interacción grade x term.",
            "insight": (
                "Antes de cualquier explicación post-hoc, ya había estructura económica visible: "
                "plazo y calidad crediticia interactúan de forma no trivial."
            ),
            "bridge": "Ese patrón reaparece luego en SHAP, ALE y redundancia de drivers.",
        },
        {
            "label": "WOE",
            "stem": "02_feature_engineering",
            "file": "cell_020_out_00.png",
            "title": "WOE binning: monotonicidad y señal económica",
            "caption": "Notebook 02: WOE binning en features top.",
            "insight": (
                "La interpretabilidad no empieza en SHAP: empieza en cómo construimos variables "
                "con señal estable y legible para negocio."
            ),
            "bridge": "Esto sustenta la taxonomía de familias y señales monotónicas esperadas.",
        },
        {
            "label": "SHAP",
            "stem": "03_pd_modeling",
            "file": "cell_017_out_00.png",
            "title": "SHAP global original del notebook de modelado",
            "caption": "Notebook 03: beeswarm e importancia global.",
            "insight": (
                "El notebook validó primero los drivers globales; la página nueva los convierte "
                "en bundle canónico reutilizable por laboratorio, gobernanza y despliegue."
            ),
            "bridge": "Es la base visual de los artefactos `explainability_global` y `shap_raw_top20`.",
        },
        {
            "label": "Intervalos",
            "stem": "04_conformal_prediction",
            "file": "cell_020_out_01.png",
            "title": "Predicciones locales con intervalos conformales",
            "caption": "Notebook 04: ejemplos por préstamo.",
            "insight": (
                "Un caso local no debe leerse solo con top drivers: tambien importa cuanta "
                "incertidumbre rodea la PD puntual."
            ),
            "bridge": "Por eso cada caso arquetipico de esta pagina incluye PD calibrada + intervalo.",
        },
        {
            "label": "Waterfall",
            "stem": "13_model_explainability",
            "file": "shap_waterfall_examples.png",
            "title": "SHAP waterfall: casos locales con reason codes",
            "caption": "Notebook 13: waterfall de casos individuales — quién paga y quién no.",
            "insight": (
                "El waterfall muestra exactamente cómo cada feature empuja la PD hacia arriba o abajo "
                "respecto al valor base. Es el formato más directo para reason codes en auditoría."
            ),
            "bridge": "Esta visualización conecta directamente con los reason codes de la sección de casos locales.",
        },
        {
            "label": "SHAP vs Perm",
            "stem": "13_model_explainability",
            "file": "shap_vs_permutation.png",
            "title": "SHAP vs permutation importance: consistencia de ranking",
            "caption": "Notebook 13: comparación de dos métodos de importancia.",
            "insight": (
                "Cuando SHAP y permutation importance coinciden en el top-10, el ranking de drivers "
                "es robusto. Divergencias señalan features con interacciones no capturadas por SHAP marginal."
            ),
            "bridge": "Sustenta la elección de SHAP como método primario ante comité o regulador.",
        },
        {
            "label": "Monotonicidad",
            "stem": "13_model_explainability",
            "file": "monotonicity_verification.png",
            "title": "Verificación de monotonicidad económica",
            "caption": "Notebook 13: chequeo de que el modelo cumple restricciones económicas esperadas.",
            "insight": (
                "Un modelo puede tener AUC alto pero violar sentido económico (ej. mayor ingreso → mayor PD). "
                "Este chequeo es la primera línea de defensa ante gobernanza y auditoría."
            ),
            "bridge": "Validación necesaria antes de presentar el modelo a comité de crédito o regulador.",
        },
        {
            "label": "Familias",
            "stem": "13_model_explainability",
            "file": "feature_family_decomposition.png",
            "title": "Descomposición de importancia por familia de features",
            "caption": "Notebook 13: masa SHAP total por familia (riesgo, ingresos, historial, producto).",
            "insight": (
                "Agrupar drivers por familia revela qué dimensión del riesgo domina el modelo: "
                "¿calidad crediticia histórica, carga financiera actual o características del producto?"
            ),
            "bridge": "Insumo directo para el narrative de gobernanza: explicar el modelo a no-técnicos.",
        },
    ]
    valid_items = [
        item
        for item in gallery
        if get_notebook_image_path(item["stem"], item["file"]).exists()
    ]
    if not valid_items:
        st.info("No hay imagenes de notebooks disponibles en este entorno.")
        return

    tabs = st.tabs([item["label"] for item in valid_items])
    for tab, item in zip(tabs, valid_items):
        with tab:
            img_path = get_notebook_image_path(item["stem"], item["file"])
            col_img, col_txt = st.columns([1.25, 1.0], gap="large")
            with col_img:
                st.image(str(img_path), caption=item["caption"], width="stretch")
            with col_txt:
                st.markdown(f"**{item['title']}**")
                st.markdown(f"**Qué añade:** {item['insight']}")
                st.markdown(f"**Cómo conecta:** {item['bridge']}")


st.title("🧠 Explicabilidad e Interpretabilidad")
st.caption(
    "Pagina canónica para explicar el modelo PD con drivers globales, efectos, reason codes, "
    "intervalos y estabilidad de explicaciones."
)
page_contract = get_page_contract("model_interpretability")
render_page_header(page_contract)
render_key_takeaway(
    "No toda tecnica responde la misma pregunta: aqui se separan atribucion, sensibilidad, efecto, caso local e interpretabilidad estable en el tiempo."
)
term_popover("canónico", label="Bundle canónico")
term_popover("prediction_interval", label="Intervalos en casos locales")
term_popover("concept_drift", label="Drift explicativo")
storytelling_intro(
    page_goal=(
        "Responder con artefactos canónicos qué variables mandan, cómo afectan la PD, "
        "por qué un caso salió así y si la explicación se mantiene estable."
    ),
    business_value=(
        "Permite defender decisiones ante comité, tesis y auditoría sin depender de una sola "
        "figura exploratoria o de una explicación improvisada."
    ),
    key_decision=(
        "Usar un lenguaje común para explicabilidad operativa: drivers globales, efectos, "
        "reason codes y estabilidad."
    ),
    how_to_read=[
        "Empieza por el mapa conceptual para no mezclar preguntas distintas.",
        "Mira luego drivers globales y efectos para entender la lógica general del modelo.",
        "Cierra con casos locales y estabilidad para validar operación y gobernanza.",
    ],
)
render_decision_box(
    "Usar esta página como fuente oficial de interpretabilidad del score PD y no como galería aislada de SHAP.",
    owner="Data Science + Model Risk",
    cadence="retrain, recalibracion y monitoreo mensual",
)
render_global_kpi_spine("model")
methodology_dialog(
    "Cómo se organiza la interpretabilidad en esta página",
    """
Siguiendo la taxonomía del libro, esta página separa cinco preguntas distintas:

1. **Qué variables importan**: SHAP global + permutation importance.
2. **Cómo cambia la PD si una variable se mueve**: ALE y PDP/ICE como respaldo.
3. **Por qué un caso salió así**: SHAP local + reason code + intervalo conformal.
4. **Qué drivers se pisan o interactúan**: redundancia/interacciones SHAP aproximadas.
5. **Si la explicación sigue siendo la misma en el tiempo**: explanation drift y benchmark challenger.

La idea central es evitar mezclar atribución, sensibilidad, efecto promedio y causalidad.
""",
    button_label="Ver taxonomía interpretativa",
)

audience = audience_selector()
st.caption(f"Vista de explicación activa: **{audience}**")

narrative_block(
    audience,
    general=(
        "Esta página convierte la interpretabilidad en una capacidad operativa. "
        "No muestra solo gráficos bonitos: muestra qué parte del modelo entendemos y con qué artefactos lo defendemos."
    ),
    business=(
        "La lectura correcta para negocio es: qué drivers se pueden explicar, qué drivers son accionables, "
        "qué casos requieren justificación y si esa explicación sigue estable en el tiempo."
    ),
    technical=(
        "La lectura técnica separa importancia global, perfiles de efecto, explicaciones locales, "
        "redundancia de drivers, conformal local y drift explicativo."
    ),
)

global_df = _load_explainability_global()
local_cases = try_load_parquet("explainability_local_cases")
ale_curves = try_load_parquet("ale_curves")
pdp_ice = try_load_parquet("pdp_ice_top5")
interaction_df = try_load_parquet("shap_interactions_or_redundancy")
explanation_drift = try_load_parquet("explanation_drift")
shap_raw = try_load_parquet("shap_raw_top20")
governance = try_load_json("governance_status", directory="models", default={})
challenger_report = try_load_json("challenger_promotion_report", directory="models", default={})
fairness_status = try_load_json("fairness_audit_status", directory="models", default={})

bundle_ready = not global_df.empty and not local_cases.empty and not ale_curves.empty
if bundle_ready:
    st.success("Bundle canónico de explicabilidad disponible para lectura operacional.")
else:
    st.warning(
        "Faltan artefactos del bundle canónico; la página degrada de forma segura con lo que sí exista."
    )

redundancy_pairs = (
    int(interaction_df["redundancy_flag"].astype(bool).sum()) if "redundancy_flag" in interaction_df.columns else 0
)
ale_features = int(ale_curves["feature"].nunique()) if not ale_curves.empty else 0
explainability_pass = bool(
    governance.get("explainability_pass", (governance.get("checks", {}) or {}).get("pass_explainability", False))
)

kpi_row(
    [
        {"label": "Drivers globales", "value": str(len(global_df)) if not global_df.empty else "0"},
        {"label": "Casos arquetípicos", "value": str(len(local_cases)) if not local_cases.empty else "0"},
        {"label": "Features con ALE", "value": str(ale_features)},
        {"label": "Pares redundantes", "value": str(redundancy_pairs)},
        {"label": "Explainability pass", "value": "PASS" if explainability_pass else "REVISAR"},
    ],
    n_cols=5,
)

tabs = st.tabs(
    [
        "Mapa conceptual",
        "Drivers globales",
        "Casos locales",
        "Efectos e interacciones",
        "Estabilidad y evidencia",
    ]
)

with tabs[0]:
    st.subheader("1) Qué pregunta responde cada técnica")
    method_rows = pd.DataFrame(
        [
            {
                "Pregunta": "Qué variables mandan en el modelo",
                "Concepto del libro": "Global modular interpretation",
                "Método en el proyecto": "SHAP global + permutation importance",
                "Artefacto canónico": "explainability_global.parquet",
                "Uso de decisión": "Priorizar drivers y separar atribución de sensibilidad",
            },
            {
                "Pregunta": "Cómo cambia la PD cuando una variable se mueve",
                "Concepto del libro": "Global effect profiles",
                "Método en el proyecto": "ALE + PDP/ICE de respaldo",
                "Artefacto canónico": "ale_curves.parquet / pdp_ice_top5.parquet",
                "Uso de decisión": "Explicar forma del efecto y revisar monotonicidad económica",
            },
            {
                "Pregunta": "Por qué un préstamo concreto salió así",
                "Concepto del libro": "Local single-prediction interpretation",
                "Método en el proyecto": "Reason codes SHAP + intervalo conformal",
                "Artefacto canónico": "explainability_local_cases.parquet",
                "Uso de decisión": "Defender decisiones caso a caso y no solo el promedio",
            },
            {
                "Pregunta": "Qué drivers se pisan o se refuerzan",
                "Concepto del libro": "Feature interactions",
                "Método en el proyecto": "Redundancia/interacciones SHAP aproximadas",
                "Artefacto canónico": "shap_interactions_or_redundancy.parquet",
                "Uso de decisión": "Simplificar narrativa y detectar explicaciones redundantes",
            },
            {
                "Pregunta": "Si la explicación sigue siendo la misma en el tiempo",
                "Concepto del libro": "Stability + governance layer",
                "Método en el proyecto": "Explanation drift + benchmark challenger",
                "Artefacto canónico": "explanation_drift.parquet + challenger_promotion_report.json",
                "Uso de decisión": "Monitorear estabilidad y promoción del challenger",
            },
        ]
    )
    st.dataframe(method_rows, width="stretch", hide_index=True)

    card_cols = st.columns(4, gap="large")
    cards = [
        (
            "Atribución",
            "SHAP y permutation importance",
            "Responde qué variables pesan más y cuáles dañan más el ranking si se rompen.",
        ),
        (
            "Efecto",
            "ALE y PDP/ICE",
            "Responde cómo cambia la PD cuando una variable se mueve manteniendo el resto del contexto.",
        ),
        (
            "Caso local",
            "Reason codes + intervalo",
            "Responde por qué este caso salió así y con qué incertidumbre debe leerse.",
        ),
        (
            "Estabilidad",
            "Explanation drift + challenger",
            "Responde si la explicación sigue siendo defendible y si existe un challenger más interpretable.",
        ),
    ]
    for col, (title, subtitle, body) in zip(card_cols, cards):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(subtitle)
            st.markdown(body)

    st.markdown(
        """
La regla editorial de esta página es deliberada: **no confundir importancia con efecto, ni efecto con causalidad**.
Ese es uno de los aportes más útiles del libro al proyecto. En el dashboard ahora cada pregunta tiene su artefacto
canónico y su uso de negocio explícito.
"""
    )

with tabs[1]:
    st.subheader("2) Drivers globales")
    if global_df.empty:
        st.info("No hay artefactos globales de explicabilidad en este entorno.")
    else:
        chart_help_popover(
            "global_driver_map",
            what_to_look_at=(
                "Busca el cuadrante superior derecho: variables con alta atribución SHAP y alta sensibilidad "
                "por permutación."
            ),
            common_misread=(
                "Un driver con SHAP alto no es automáticamente accionable ni causal; solo explica la lógica "
                "del modelo actual."
            ),
        )
        top_driver = str(global_df.iloc[0]["feature"])
        top_controllable_df = global_df[global_df["controllable"].astype(bool)]
        top_controllable = (
            str(top_controllable_df.iloc[0]["feature"]) if not top_controllable_df.empty else "N/D"
        )
        ale_ready_df = global_df[global_df["preferred_effect_view"].astype(str).eq("ale")]
        ale_ready = str(ale_ready_df.iloc[0]["feature"]) if not ale_ready_df.empty else "N/D"
        kpi_row(
            [
                {"label": "Driver #1", "value": top_driver},
                {"label": "Top controlable", "value": top_controllable},
                {"label": "Primer driver con ALE", "value": ale_ready},
                {
                    "label": "Family leader",
                    "value": str(
                        global_df.groupby("feature_family")["mean_abs_shap"].sum().idxmax()
                    ),
                },
            ],
            n_cols=4,
        )

        col_map, col_family = st.columns([1.3, 0.9], gap="large")
        with col_map:
            st.plotly_chart(_build_driver_map_figure(global_df), width="stretch")
        with col_family:
            st.plotly_chart(_build_family_figure(global_df), width="stretch")

        top_n = st.slider(
            "Top drivers a mostrar",
            min_value=8,
            max_value=min(20, len(global_df)),
            value=min(12, len(global_df)),
            step=1,
            key="model_interpretability_topn",
        )
        st.plotly_chart(_build_global_rank_figure(global_df, top_n=top_n), width="stretch")

        table_cols = [
            col
            for col in [
                "feature",
                "business_label",
                "mean_abs_shap",
                "permutation_auc_drop",
                "feature_family",
                "controllable",
                "monotonic_expected",
                "preferred_effect_view",
                "max_abs_spearman_corr",
            ]
            if col in global_df.columns
        ]
        table_view = global_df.loc[:, table_cols].head(top_n).copy()
        if "controllable" in table_view.columns:
            table_view["controllable"] = table_view["controllable"].map({True: "Si", False: "No"})
        st.dataframe(table_view, width="stretch", hide_index=True)
        st.caption(
            "SHAP explica la contribución media de cada driver dentro del modelo; permutation importance "
            "mide cuánto se deteriora el ranking si esa señal se rompe. Ambas cosas son útiles, pero no son lo mismo."
        )

with tabs[2]:
    st.subheader("3) Casos locales y reason codes")
    if local_cases.empty:
        st.info("No hay `explainability_local_cases.parquet` disponible.")
    else:
        case_order = {
            "bajo_riesgo": 0,
            "cercano_umbral": 1,
            "alto_riesgo": 2,
            "cohorte_drift": 3,
        }
        local_cases_view = local_cases.copy()
        local_cases_view["case_order"] = local_cases_view["segmento"].map(case_order).fillna(99)
        local_cases_view = local_cases_view.sort_values(["case_order", "pd_calibrada"]).reset_index(drop=True)

        st.caption(
            "Cada caso arquetípico responde una pregunta operativa distinta: caso benigno, caso cercano al umbral, "
            "caso severo y caso de cohorte con drift."
        )
        case_tabs = st.tabs(
            [
                str(segmento).replace("_", " ").title()
                for segmento in local_cases_view["segmento"].astype(str).tolist()
            ]
        )
        for tab, (_, case_row) in zip(case_tabs, local_cases_view.iterrows()):
            with tab:
                _render_case_panel(case_row)

        st.markdown(
            """
La lectura correcta de un caso local es triple:

- **score raw**: salida interna del modelo antes de calibración;
- **PD calibrada**: probabilidad operativa que consume el resto del pipeline;
- **intervalo conformal**: cuánta incertidumbre rodea esa PD.
"""
        )

with tabs[3]:
    st.subheader("4) Efectos globales e interacciones")
    feature_pool: list[str] = []
    if not ale_curves.empty:
        feature_pool.extend(ale_curves["feature"].astype(str).unique().tolist())
    if not pdp_ice.empty:
        feature_pool.extend(pdp_ice["feature"].astype(str).unique().tolist())
    if not feature_pool and not global_df.empty:
        feature_pool.extend(global_df["feature"].astype(str).head(5).tolist())
    feature_options = sorted(set(feature_pool))

    if not feature_options:
        st.info("No hay curvas de efecto disponibles en este entorno.")
    else:
        selected_feature = st.selectbox(
            "Variable para perfiles de efecto",
            options=feature_options,
            index=0,
            key="model_interpretability_effect_feature",
        )
        feature_meta = global_df[global_df["feature"].astype(str) == str(selected_feature)].head(1)
        if not feature_meta.empty:
            row = feature_meta.iloc[0]
            kpi_row(
                [
                    {"label": "Familia", "value": str(row.get("feature_family", "N/D"))},
                    {
                        "label": "Controlable",
                        "value": "Si" if bool(row.get("controllable", False)) else "No",
                    },
                    {
                        "label": "Señal esperada",
                        "value": str(row.get("monotonic_expected", "N/D")),
                    },
                    {
                        "label": "Vista preferida",
                        "value": str(row.get("preferred_effect_view", "N/D")).upper(),
                    },
                    {
                        "label": "Max corr",
                        "value": f"{float(row.get('max_abs_spearman_corr', np.nan)):.3f}"
                        if pd.notna(row.get("max_abs_spearman_corr", np.nan))
                        else "N/D",
                    },
                ],
                n_cols=5,
            )

        col_ale, col_pdp = st.columns(2, gap="large")
        with col_ale:
            feature_ale = ale_curves[ale_curves["feature"].astype(str) == str(selected_feature)].copy()
            if feature_ale.empty:
                st.info("Sin ALE para esta variable.")
            else:
                st.plotly_chart(_build_ale_figure(feature_ale, selected_feature), width="stretch")
                st.caption(
                    "ALE es la vista primaria cuando hay dependencia entre variables porque reduce el sesgo "
                    "que PDP puede introducir bajo correlación."
                )
        with col_pdp:
            feature_pdp = pdp_ice[pdp_ice["feature"].astype(str) == str(selected_feature)].copy()
            if feature_pdp.empty:
                st.info("Sin PDP/ICE de respaldo para esta variable.")
            else:
                st.plotly_chart(_build_pdp_ice_figure(feature_pdp, selected_feature), width="stretch")
                st.caption(
                    "PDP/ICE queda como diagnóstico secundario para ver heterogeneidad de trayectorias individuales."
                )

        if not shap_raw.empty:
            shap_col = f"shap_{selected_feature}"
            val_col = f"val_{selected_feature}"
            if shap_col in shap_raw.columns and val_col in shap_raw.columns:
                with st.expander("Ver dependencia SHAP suplementaria", expanded=False):
                    st.plotly_chart(
                        _build_shap_dependence_figure(shap_raw, selected_feature),
                        width="stretch",
                    )

    st.markdown("#### Interacciones y redundancia")
    if interaction_df.empty:
        st.info("No hay artefacto de redundancia/interacciones SHAP disponible.")
    else:
        flagged = interaction_df[interaction_df["redundancy_flag"].astype(bool)].copy()
        lead_pair = flagged.iloc[0] if not flagged.empty else (interaction_df.iloc[0] if not interaction_df.empty else pd.Series({"feature_a": "N/D", "feature_b": "N/D", "shap_spearman": 0.0}))
        st.caption(
            "Pareja líder detectada: "
            f"`{lead_pair['feature_a']}` x `{lead_pair['feature_b']}` "
            f"(Spearman SHAP={float(lead_pair['shap_spearman']):+.3f})."
        )
        col_heatmap, col_table = st.columns([1.15, 0.95], gap="large")
        with col_heatmap:
            st.plotly_chart(_build_redundancy_heatmap(interaction_df), width="stretch")
        with col_table:
            st.dataframe(interaction_df.head(12), width="stretch", hide_index=True)
        st.caption(
            "Esto no son interacciones SHAP exactas: es una aproximación operativa para detectar drivers "
            "que cuentan historias muy parecidas y simplificar el relato a negocio."
        )

with tabs[4]:
    st.subheader("5) Estabilidad de explicaciones y conexión con gobernanza")
    if explanation_drift.empty:
        st.info("No hay `explanation_drift.parquet`; la capa de estabilidad interpretativa queda incompleta.")
    else:
        drift_row = explanation_drift.iloc[0]
        kpi_row(
            [
                {
                    "label": "Overlap top-10",
                    "value": f"{float(drift_row.get('rank_overlap_top10', 0.0)):.3f}",
                },
                {
                    "label": "Max SHAP PSI",
                    "value": f"{float(drift_row.get('max_shap_psi_top5', 0.0)):.3f}",
                },
                {
                    "label": "Reason code match",
                    "value": f"{float(drift_row.get('reason_code_match_rate', 0.0)):.3f}",
                },
                {
                    "label": "Drift explicativo",
                    "value": "PASS" if bool(drift_row.get("passed_all", False)) else "FAIL",
                },
            ],
            n_cols=4,
        )
        st.dataframe(explanation_drift, width="stretch", hide_index=True)
        with st.expander("Ver detalle de PSI SHAP y coincidencia de reason codes"):
            feature_details = pd.DataFrame(
                _parse_json_payload(drift_row.get("feature_psi_details"), [])
            )
            reason_details = pd.DataFrame(
                _parse_json_payload(drift_row.get("reason_code_details"), [])
            )
            if not feature_details.empty:
                st.markdown("**PSI por driver top-5**")
                st.dataframe(feature_details, width="stretch", hide_index=True)
            if not reason_details.empty:
                st.markdown("**Coincidencia de razones por banda de PD**")
                st.dataframe(reason_details, width="stretch", hide_index=True)

    checks = governance.get("checks", {}) if isinstance(governance, dict) else {}
    if governance:
        st.markdown("#### Lectura de gobernanza")
        kpi_row(
            [
                {"label": "Explainability", "value": "PASS" if checks.get("pass_explainability") else "FAIL"},
                {"label": "Fairness", "value": "PASS" if checks.get("pass_fairness") else "FAIL"},
                {
                    "label": "Predictive drift",
                    "value": "PASS" if checks.get("pass_predictive_drift") else "FAIL",
                },
                {"label": "Overall pass", "value": "PASS" if governance.get("overall_pass") else "FAIL"},
                {
                    "label": "Challenger",
                    "value": "Promovible"
                    if governance.get("challenger_promotable", False)
                    else "Benchmark",
                },
            ],
            n_cols=5,
        )
        if checks.get("pass_explainability") and not governance.get("overall_pass", False):
            st.warning(
                "La interpretabilidad está en verde, pero la gobernanza total sigue en revisión por drift predictivo y/o fairness."
            )

    if challenger_report:
        st.markdown("#### Challenger monotónico desde la óptica de interpretabilidad")
        interp = challenger_report.get("interpretability", {})
        deltas = challenger_report.get("deltas", {})
        gains = interp.get("gains", {})
        comparison_df = pd.DataFrame(
            [
                {
                    "Métrica": "Drivers efectivos",
                    "Champion": interp.get("champion_effective_driver_count"),
                    "Challenger": interp.get("challenger_effective_driver_count"),
                },
                {
                    "Métrica": "Estabilidad de explicaciones",
                    "Champion": interp.get("champion_explanation_stability"),
                    "Challenger": interp.get("challenger_explanation_stability"),
                },
                {
                    "Métrica": "Monotonic violation rate",
                    "Champion": interp.get("champion_monotonic_violation_rate"),
                    "Challenger": interp.get("challenger_monotonic_violation_rate"),
                },
            ]
        )
        kpi_row(
            [
                {
                    "label": "Promovible",
                    "value": "SI" if challenger_report.get("challenger_promotable", False) else "NO",
                },
                {"label": "Ganancias de interpretabilidad", "value": str(int(interp.get("gain_count", 0)))},
                {"label": "AUC drop", "value": f"{float(deltas.get('auc_drop', 0.0)):.4f}"},
                {
                    "label": "Brier increase",
                    "value": f"{float(deltas.get('brier_increase_pct', 0.0)) * 100:.2f}%",
                },
            ],
            n_cols=4,
        )
        st.dataframe(comparison_df, width="stretch", hide_index=True)
        st.caption(
            "El challenger sí mejora parsimonia y consistencia monotónica, pero hoy no se promueve porque el costo en AUC supera el guardrail definido."
        )
        with st.expander("Ver ganancias cualitativas del challenger"):
            st.json(gains)

    st.markdown("#### Evidencia visual de notebooks")
    _render_notebook_gallery()

st.markdown(
    """
**Conclusión de la página:**
- La explicabilidad del proyecto ya no vive solo en SHAP global; ahora es un bundle operativo con drivers, efectos, casos, intervalos y estabilidad.
- ALE se usa como vista principal cuando hay dependencia, mientras PDP/ICE queda como diagnóstico secundario.
- La interpretación es defendible hoy como capacidad analítica, aunque la gobernanza total siga condicionada por drift predictivo y fairness.
"""
)

render_caveats(
    [
        "SHAP y permutation importance describen la lógica del modelo actual; no prueban causalidad.",
        "ALE y PDP/ICE son perfiles de efecto promedio o condicional, no recomendaciones individuales de intervención.",
        "Reason codes siguen siendo dependientes del modelo y del set de features disponible.",
        "Una explicación estable no compensa por sí sola fallos de fairness o drift predictivo.",
    ]
)
render_page_feedback("model_interpretability")
render_next_steps(
    [
        (
            "Cuantificación de Incertidumbre",
            "Conectar la explicacion del score con los intervalos conformales que consume el resto del pipeline.",
            "pages/uncertainty_quantification.py",
        ),
        (
            "Gobernanza del Modelo",
            "Ver cómo esta capa de explicabilidad entra al semáforo formal de drift, fairness y promoción del challenger.",
            "pages/model_governance.py",
        ),
    ]
)
