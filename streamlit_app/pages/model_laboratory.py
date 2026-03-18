"""Laboratorio de modelos PD: desempeño, calibración y handoff interpretativo."""

# ruff: noqa: E402

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import re
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
    methodology_dialog,
    metric_help_popover,
    term_popover,
)
from streamlit_app.components.dvc_kpi_spine import render_global_kpi_spine
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import narrative_block, next_page_teaser, storytelling_intro
from streamlit_app.components.story_shell import (
    render_caveats,
    render_decision_box,
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    get_notebook_image_path,
    get_operational_threshold,
    get_pd_internal_threshold,
    load_pd_calibration_diagnostics,
    try_load_json,
    try_load_parquet,
)


@st.cache_data(ttl=600, max_entries=1)
def load_logreg_baseline_coefficients() -> pd.DataFrame:
    """Load baseline logistic coefficients and convert them to odds ratios.

    The stored baseline model is a sklearn Pipeline(StandardScaler + LogisticRegression),
    so coefficients map to a +1 standard deviation change in each feature.
    """
    model_path = _REPO_ROOT / "models" / "pd_logreg_baseline.pkl"
    if not model_path.exists():
        return pd.DataFrame()
    try:
        import joblib

        pipe = joblib.load(model_path)
        scaler = pipe.named_steps.get("scaler")
        clf = pipe.named_steps.get("clf")
        feature_names = getattr(scaler, "feature_names_in_", None)
        coefs = getattr(clf, "coef_", None)
        if feature_names is None or coefs is None:
            return pd.DataFrame()
        coef_values = np.asarray(coefs).ravel()
        if len(feature_names) != len(coef_values):
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "feature": [str(f) for f in feature_names],
                "coef_log_odds": coef_values.astype(float),
                "odds_ratio_plus_1sd": np.exp(coef_values.astype(float)),
            }
        )
        df["abs_coef"] = df["coef_log_odds"].abs()
        df["direction"] = np.where(df["coef_log_odds"] >= 0, "↑ default odds", "↓ default odds")
        return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, max_entries=1)
def read_pd_model_has_time_flag() -> bool | None:
    """Read `has_time` from `configs/pd_model.yaml` without requiring PyYAML in Streamlit Cloud."""
    cfg_path = _REPO_ROOT / "configs" / "pd_model.yaml"
    if not cfg_path.exists():
        return None
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(r"(?m)^[ \t]*has_time:[ \t]*(true|false)\b", text)
    if not match:
        return None
    return match.group(1).lower() == "true"


@st.cache_data(ttl=600, max_entries=1)
def load_package_versions_for_model_lab() -> pd.DataFrame:
    """Versions relevant to modeling and explainability in this page."""
    packages = [
        "catboost",
        "scikit-learn",
        "shap",
        "scikit-survival",
        "pyarrow",
        "mlflow",
        "cvxpy",
        "dowhy",
        "econml",
    ]
    rows = []
    for pkg in packages:
        try:
            version = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            version = "not installed (main env)"
        rows.append({"package": pkg, "version": version})
    return pd.DataFrame(rows)


def prob_to_odds(prob: float) -> float:
    """Convert probability to odds with clipping near the boundaries."""
    p = min(max(float(prob), 1e-9), 1 - 1e-9)
    return p / (1 - p)


def _parse_json_payload(value: object, default: object) -> object:
    if value in (None, "", "nan"):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _echarts_model_auc_option(
    models_df: pd.DataFrame,
    *,
    best_model: str,
    selected_model: str | None = None,
) -> dict:
    """Build a compact ECharts option for clickable model comparison (AUC focus)."""
    if models_df.empty or "model" not in models_df.columns or "auc" not in models_df.columns:
        return {}

    df = models_df.copy()
    df = df.dropna(subset=["model", "auc"]).copy()
    if df.empty:
        return {}

    if "brier" not in df.columns:
        df["brier"] = np.nan
    if "gini" not in df.columns:
        df["gini"] = np.nan

    df["model"] = df["model"].astype(str)
    df = df.sort_values("auc", ascending=True)
    categories = df["model"].tolist()
    series_data = []
    for _, row in df.iterrows():
        model_name = str(row["model"])
        auc_val = float(row["auc"])
        is_best = model_name == str(best_model)
        is_selected = selected_model is not None and model_name == selected_model
        color = "#0B5ED7" if is_best else "#6C757D"
        if is_selected:
            color = "#00A389"
        series_data.append(
            {
                "value": round(auc_val, 4),
                "model_name": model_name,
                "auc": round(auc_val, 4),
                "brier": None if pd.isna(row["brier"]) else round(float(row["brier"]), 4),
                "gini": None if pd.isna(row["gini"]) else round(float(row["gini"]), 4),
                "is_best": is_best,
                "itemStyle": {"color": color},
            }
        )

    min_auc = max(0.0, float(df["auc"].min()) - 0.03)
    max_auc = min(1.0, float(df["auc"].max()) + 0.01)
    return {
        "animationDuration": 350,
        "grid": {"left": 180, "right": 18, "top": 34, "bottom": 24},
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}<br/>AUC: {c}",
        },
        "xAxis": {
            "type": "value",
            "min": round(min_auc, 3),
            "max": round(max_auc, 3),
            "name": "AUC",
            "nameLocation": "middle",
            "nameGap": 24,
        },
        "yAxis": {"type": "category", "data": categories, "axisLabel": {"fontSize": 11}},
        "series": [
            {
                "name": "AUC",
                "type": "bar",
                "data": series_data,
                "label": {"show": True, "position": "right", "formatter": "{c}"},
                "emphasis": {"focus": "self"},
            }
        ],
    }


def _plotly_model_auc_figure(
    models_df: pd.DataFrame,
    *,
    best_model: str,
    selected_model: str | None = None,
) -> go.Figure:
    """Stable Plotly fallback for model AUC comparison."""
    df = models_df.copy()
    if df.empty:
        return go.Figure()
    if "brier" not in df.columns:
        df["brier"] = np.nan
    if "gini" not in df.columns:
        df["gini"] = np.nan
    df["model"] = df["model"].astype(str)
    df = df.sort_values("auc", ascending=True)

    colors = []
    for model_name in df["model"]:
        if selected_model is not None and model_name == selected_model:
            colors.append("#00A389")
        elif model_name == str(best_model):
            colors.append("#0B5ED7")
        else:
            colors.append("#6C757D")

    customdata = np.column_stack(
        [
            df["brier"].fillna(np.nan).astype(float).values,
            df["gini"].fillna(np.nan).astype(float).values,
        ]
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["auc"],
                y=df["model"],
                orientation="h",
                marker={"color": colors},
                text=[f"{v:.4f}" for v in df["auc"].astype(float)],
                textposition="outside",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "AUC: %{x:.4f}<br>"
                    "Brier: %{customdata[0]:.4f}<br>"
                    "Gini: %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        ]
    )
    min_auc = max(0.0, float(df["auc"].min()) - 0.03)
    max_auc = min(1.0, float(df["auc"].max()) + 0.01)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=max(260, len(df) * 52))
    fig.update_xaxes(range=[min_auc, max_auc], title="AUC")
    fig.update_yaxes(title="")
    return fig


st.title("🔬 Laboratorio de Modelos")
st.caption(
    "Comparación de modelos PD y calibración; la interpretabilidad detallada vive en su página dedicada."
)
page_contract = get_page_contract("model_laboratory")
render_page_header(page_contract)
render_key_takeaway(
    "La decisión correcta aquí no es solo el mejor AUC, sino el mejor trade-off entre discriminación y calidad probabilística para uso operativo."
)
term_popover("calibración", label="Por qué importa la calibración")
st.markdown(
    """
Este capítulo traduce teoría de modelado tabular a una decisión concreta de arquitectura PD. La intención no es exhibir
un benchmark superficial, sino documentar por qué se escoge un modelo específico, cómo se valida fuera de muestra temporal
y por qué la calibración probabilística es tan importante como el ranking. En riesgo de crédito, una PD mal calibrada puede
desalinear pricing, límites y provisiones aunque el AUC sea alto.
"""
)
audience = audience_selector()
st.caption(f"Vista de explicación activa: **{audience}**")

storytelling_intro(
    page_goal="Determinar qué arquitectura de PD usar y por qué, no solo quién gana un benchmark.",
    business_value="Reduce errores de aprobación y evita usar probabilidades mal calibradas en pricing e IFRS9.",
    key_decision="Adoptar el modelo final y su política de calibración para operación.",
    how_to_read=[
        "Mirar primero comparativo de modelos y métricas finales.",
        "Validar calibración (Brier/ECE) además de AUC/KS.",
        "Usar el resumen interpretativo solo como chequeo rápido; el detalle vive en la página dedicada.",
    ],
)
render_decision_box(
    "Adoptar el campeón canónico solo si mantiene buen ranking (AUC/KS) y buena calibración (Brier/ECE).",
    owner="Model Risk / Data Science",
    cadence="retrain o recalibración",
)
render_global_kpi_spine("model")
metric_help_popover("baseline_vs_canonical", label="Baseline vs canónico")
methodology_dialog(
    "Cómo leer las métricas del laboratorio",
    """
- `AUC` y `KS` miden ranking/separación.
- `Brier` y `ECE` miden calidad probabilística (calibración).
- Un modelo puede mejorar AUC pero empeorar calibración; eso impacta pricing e IFRS9.
- La decisión del campeón requiere mirar ambas familias de métricas.
""",
    button_label="Ver guía rápida de métricas",
)

focus_items = [
    ("comparison", "Comparativo"),
    ("lr_odds", "LR Odds"),
    ("catboost_params", "CatBoost Avanzado"),
    ("calibration", "Calibración"),
    ("venn_abers", "Venn-Abers"),
    ("shap", "Interpretabilidad"),
    ("upgrades", "Upgrades"),
]
id_to_label = dict(focus_items)
label_to_id = {label: item_id for item_id, label in focus_items}

focus_section = str(st.session_state.get("model_lab_focus", "comparison"))
if focus_section not in id_to_label:
    focus_section = "comparison"
default_focus_label = id_to_label[focus_section]

focus_label = st.segmented_control(
    "Navegación rápida",
    options=[label for _, label in focus_items],
    default=default_focus_label,
    key="model_lab_nav_segmented",
)
if not isinstance(focus_label, str):
    focus_label = default_focus_label
focus_section = label_to_id.get(focus_label, focus_section)
st.session_state["model_lab_focus"] = focus_section
st.caption("Navegación rápida nativa: selecciona sección para enfocar el contenido.")

focus_labels = {
    "comparison": "Comparativo de arquitecturas y decisión del campeón",
    "lr_odds": "Regresión logística: log-odds, odds y odds ratios",
    "catboost_params": "CatBoost avanzado: hiperparámetros aplicables al proyecto",
    "calibration": "Calibración probabilística y selección de método",
    "venn_abers": "Venn-Abers: calibración canónica con garantías conformales",
    "shap": "Resumen interpretativo y puente a la página dedicada",
    "upgrades": "Mejoras habilitadas por upgrades recientes",
}
st.info(f"Sección enfocada: **{focus_labels.get(focus_section, focus_section)}**")

scope_default = str(st.session_state.get("model_lab_scope_mode", "Mostrar todo"))
if scope_default not in {"Mostrar todo", "Solo sección enfocada"}:
    scope_default = "Mostrar todo"
    st.session_state["model_lab_scope_mode"] = scope_default
scope_mode = st.segmented_control(
    "Vista de contenido",
    options=["Mostrar todo", "Solo sección enfocada"],
    selection_mode="single",
    default=scope_default,
    key="model_lab_scope_mode",
    help="Reduce scroll en páginas largas cuando usas la navegación rápida.",
    width="content",
)
if isinstance(scope_mode, list):
    scope_mode = scope_mode[0] if scope_mode else scope_default
scope_mode = str(scope_mode or scope_default)
focus_only_mode = scope_mode == "Solo sección enfocada"
if focus_only_mode:
    st.caption("Modo foco activo: se muestran solo los bloques relacionados con la sección seleccionada.")


def _show_sections(*section_ids: str) -> bool:
    return (not focus_only_mode) or (focus_section in set(section_ids))

narrative_block(
    audience,
    general=(
        "El objetivo de este bloque es construir una PD confiable para decisiones de riesgo. "
        "No basta con ranking; la calibración probabilística es crítica para IFRS9 y optimización."
    ),
    business=(
        "Interpretación de negocio: un mejor score reduce errores de asignación, "
        "pero una mala calibración distorsiona pricing y provisiones."
    ),
    technical=(
        "Se evaluaron baseline logístico y variantes de CatBoost (default, tuned, calibrado) "
        "con foco en AUC, KS, Brier y ECE."
    ),
)

if audience == "General":
    st.markdown(
        """
**En simple:** el modelo produce una probabilidad de incumplimiento por préstamo.
Un AUC más alto significa que ordena mejor quién es más riesgoso, y una buena calibración
significa que el porcentaje predicho se parece al porcentaje que realmente incumple.
"""
    )
elif audience == "Negocio":
    st.markdown(
        """
**En clave de negocio:** un modelo puede tener buen ranking (AUC) pero mala calibración.
Para pricing, provisiones y límites de aprobación necesitas ambas cosas. Por eso se evaluó
CatBoost calibrado, no solo el mejor score bruto.
"""
    )
else:
    st.markdown(
        """
**En clave técnica:** se optimiza separación (`AUC`, `KS`) y calidad probabilística (`Brier`, `ECE`).
Función conceptual: minimizar pérdida logarítmica y luego recalibrar para reducir error de probabilidad.
"""
    )
    st.latex(r"\text{AUC} = P(s(x^+) > s(x^-))")
    st.latex(
        r"\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,\qquad \text{ECE}=\sum_b w_b\left|\hat{p}_b-\hat{y}_b\right|"
    )

comparison = try_load_json("model_comparison", directory="data", default={})
st.caption(
    "Contrato de thresholds: el cutoff interno PD para screening/search se reporta por separado del "
    f"threshold operativo de aprobación (`{get_operational_threshold():.2f}`); "
    f"threshold interno actual `{get_pd_internal_threshold():.2f}`."
)
models = pd.DataFrame(comparison.get("models", []))
final = comparison.get("final_test_metrics", {})
cal_report = comparison.get("calibration_selection_report", {})
hpo_trials = int(comparison.get("hpo_trials_executed", comparison.get("optuna_n_trials", 0)))
feature_count_tuned = int(comparison.get("feature_count_tuned", 0))
test_predictions = try_load_parquet("test_predictions")
pd_model_has_time = read_pd_model_has_time_flag()
calib_diagnostics = load_pd_calibration_diagnostics()

if _show_sections("comparison"):
    st.subheader("Comparativo de arquitecturas")
    if not models.empty:
        models_view = models.copy()
        models_view["es_mejor"] = models_view["model"].eq(comparison.get("best_model", ""))
        st.dataframe(models_view, width="stretch", hide_index=True)

        compare_focus_model = st.session_state.get("model_lab_compare_focus_model")
        if not isinstance(compare_focus_model, str) or compare_focus_model not in models_view["model"].astype(str).tolist():
            compare_focus_model = str(comparison.get("best_model", models_view.iloc[-1]["model"]))

        st.caption(
            "Comparativo estable (Plotly): selecciona un modelo para enfocar el detalle y comparar AUC/Brier."
        )
        model_options = models_view["model"].astype(str).tolist()
        selected_idx = model_options.index(compare_focus_model) if compare_focus_model in model_options else 0
        compare_focus_model = st.selectbox(
            "Modelo en foco",
            options=model_options,
            index=selected_idx,
            key="model_lab_compare_focus_model",
        )
        fig_compare = _plotly_model_auc_figure(
            models_view,
            best_model=str(comparison.get("best_model", "")),
            selected_model=compare_focus_model,
        )
        st.plotly_chart(fig_compare, width="stretch")

        selected_row_df = models_view[models_view["model"].astype(str) == str(compare_focus_model)]
        if not selected_row_df.empty:
            selected_row = selected_row_df.iloc[0]
            c_auc, c_brier, c_gini = st.columns(3)
            c_auc.metric("AUC (modelo seleccionado)", f"{float(selected_row.get('auc', 0.0)):.4f}")
            c_brier.metric("Brier", f"{float(selected_row.get('brier', 0.0)):.4f}")
            c_gini.metric("Gini", f"{float(selected_row.get('gini', 0.0)):.4f}")
            is_best_model = bool(selected_row.get("es_mejor", False))
            st.caption(
                f"Modelo seleccionado: `{compare_focus_model}`"
                + (" (campeón actual)" if is_best_model else "")
            )
            if "CatBoost" in str(compare_focus_model):
                st.markdown(
                    "Lectura: CatBoost mejora discriminación (AUC/Gini) y el modelo calibrado además "
                    "mejora fuertemente el Brier, que es la señal clave para uso operativo de PD."
                )
            else:
                st.markdown(
                    "Lectura: Logistic Regression sigue siendo el baseline regulatorio por interpretabilidad y trazabilidad, "
                    "aunque su techo de AUC/Brier es menor en este dataset."
                )

    with st.expander(
        "¿Qué es CatBoost y por qué se usa en credit scoring?",
        expanded=focus_section in {"comparison", "catboost_intro"},
    ):
        narrative_block(
            audience,
            general="CatBoost es un algoritmo de inteligencia artificial que aprende patrones de datos "
            "para predecir quién va a incumplir un préstamo. Es como un equipo de analistas que "
            "revisan miles de variables simultáneamente y encuentran las combinaciones más predictivas.",
            business="CatBoost es un algoritmo de gradient boosting desarrollado por Yandex, dominante en "
            "competencias de datos tabulares (Kaggle). Es el estándar en credit scoring moderno junto "
            "con XGBoost y LightGBM, adoptado por JPMorgan, Capital One, Nubank, Mercado Libre.",
            technical="CatBoost implementa ordered boosting con manejo nativo de categorías (evita target "
            "leakage en encoding), tratamiento nativo de NaN, y regularización oblivious trees. "
            f"Tuneado con Optuna ({hpo_trials} trials ejecutados) optimizando AUC en validación temporal.",
        )
        st.markdown(
            """
**¿Por qué CatBoost en este proyecto?**
- Maneja variables tabulares heterogéneas y valores faltantes nativamente (sin imputation)
- Captura no linealidades e interacciones frecuentes en riesgo de crédito
- Logra mejor balance entre discriminación (AUC/KS) y estabilidad que el baseline lineal
- Encoding nativo de categorías evita target leakage que afecta a otros frameworks
"""
        )
        if feature_count_tuned > 0:
            st.caption(f"Contrato actual del modelo final: {feature_count_tuned} features.")

    with st.expander(
        "Discusión técnica: Logistic Regression vs CatBoost",
        expanded=focus_section in {"comparison", "lr_vs_catboost"},
    ):
        st.markdown(
            """
**Por qué Logistic Regression sigue siendo baseline de referencia en riesgo de crédito**
- Alta trazabilidad regulatoria: su estructura lineal sobre log-odds permite auditoría directa de signos y magnitudes.
- Interpretabilidad operativa: facilita scorecards, documentación metodológica y explicaciones a comités de riesgo.
- Estabilidad y simplicidad: menos grados de libertad, menor riesgo de sobreajuste en setups bien especificados.
- Gobierno de modelo: resulta ideal como benchmark/challenger por su comportamiento predecible.

**Limitaciones de Logistic Regression en datos Lending Club**
- Supuesto de linealidad en log-odds: muchas relaciones reales son no lineales y con umbrales.
- Aditividad estricta: interacciones complejas deben diseñarse manualmente.
- Dependencia fuerte del feature engineering: bins/WOE/interacciones impactan mucho el techo de desempeño.
- Menor capacidad para capturar heterogeneidad de segmentos cuando el riesgo cambia por combinaciones de variables.

**Por qué CatBoost puede superar a LR sin perder gobernanza**
- Mejora discriminación en tabular complejo al modelar no linealidades e interacciones de forma nativa.
- Maneja categorías y faltantes de forma robusta, reduciendo fragilidad de preprocesamiento manual.
- Mantiene explicabilidad práctica con SHAP global/local, permutation importance y PDP/ICE.
- Conserva control probabilístico vía calibración explícita (Platt/Isotonic) y validación temporal OOT.
- Se integra a un contrato de features y artefactos auditables (HPO, calibración, métricas y reportes).

**Decisión de arquitectura**
- `Logistic Regression` permanece como baseline regulatorio y benchmark interpretable.
- `CatBoost tuneado + calibrado` se elige como modelo final cuando entrega mejor trade-off entre AUC/KS y calidad probabilística (Brier/ECE) sin romper trazabilidad.
"""
        )

if _show_sections("lr_odds"):
    with st.expander(
        "Cómo leer la Regresión Logística (odds, log-odds y odds ratios)",
        expanded=focus_section == "lr_odds",
    ):
        st.markdown(
            """
**Por qué sigue siendo tan usada en riesgo de crédito**
- Convierte una combinación lineal de variables en una probabilidad interpretable.
- Permite explicar dirección y magnitud del efecto (signo y coeficiente).
- Es fácil de auditar, documentar y defender ante comités/regulador.
- Funciona muy bien como benchmark/challenger estable, aunque no sea el campeón.
"""
        )
        st.latex(r"\log\left(\frac{p}{1-p}\right)=\beta_0+\sum_j \beta_j x_j")
        st.latex(r"\text{odds}=\frac{p}{1-p},\qquad p=\frac{\text{odds}}{1+\text{odds}}")

        col_prob, col_compare = st.columns(2)
        with col_prob:
            p_demo = st.slider(
                "PD de ejemplo para convertir a odds",
                min_value=0.01,
                max_value=0.99,
                value=0.20,
                step=0.01,
            )
            odds_demo = prob_to_odds(p_demo)
            st.metric("Odds (p/(1-p))", f"{odds_demo:.3f}")
            st.metric("Log-odds", f"{np.log(odds_demo):.3f}")
        with col_compare:
            p_a = st.slider("PD A", min_value=0.01, max_value=0.99, value=0.10, step=0.01, key="lr_pd_a")
            p_b = st.slider("PD B", min_value=0.01, max_value=0.99, value=0.20, step=0.01, key="lr_pd_b")
            odds_a = prob_to_odds(p_a)
            odds_b = prob_to_odds(p_b)
            st.metric("Odds ratio (B/A)", f"{(odds_b / odds_a):.2f}x")
            st.caption(
                "Ejemplo: pasar de 10% a 20% PD no duplica la probabilidad en términos lineales, "
                "pero sí multiplica las odds ~2.25x."
            )

        if not test_predictions.empty and {"pd_logreg", "pd_calibrated"}.issubset(test_predictions.columns):
            q = (
                test_predictions[["pd_logreg", "pd_calibrated"]]
                .quantile([0.1, 0.5, 0.9])
                .rename(index={0.1: "P10", 0.5: "P50", 0.9: "P90"})
                .reset_index()
                .rename(columns={"index": "percentil"})
            )
            for col in ["pd_logreg", "pd_calibrated"]:
                q[f"{col}_odds"] = q[col].map(prob_to_odds)
            st.dataframe(
                q.rename(
                    columns={
                        "pd_logreg": "PD LR",
                        "pd_calibrated": "PD final calibrada",
                        "pd_logreg_odds": "Odds LR",
                        "pd_calibrated_odds": "Odds final",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Tabla calculada sobre `data/processed/test_predictions.parquet` para aterrizar la lectura de odds en el snapshot OOT."
            )

        lr_coef = load_logreg_baseline_coefficients()
        if lr_coef.empty:
            st.info("No se pudo cargar `models/pd_logreg_baseline.pkl` para mostrar coeficientes del baseline.")
        else:
            top_k = st.slider("Top coeficientes LR a mostrar", min_value=5, max_value=15, value=8, step=1)
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("**Variables que aumentan las odds de default (baseline LR)**")
                pos = (
                    lr_coef[lr_coef["coef_log_odds"] > 0]
                    .sort_values("coef_log_odds", ascending=False)
                    .head(top_k)[["feature", "coef_log_odds", "odds_ratio_plus_1sd"]]
                )
                st.dataframe(pos, width="stretch", hide_index=True)
            with col_neg:
                st.markdown("**Variables que reducen las odds de default (baseline LR)**")
                neg = (
                    lr_coef[lr_coef["coef_log_odds"] < 0]
                    .sort_values("coef_log_odds", ascending=True)
                    .head(top_k)[["feature", "coef_log_odds", "odds_ratio_plus_1sd"]]
                )
                st.dataframe(neg, width="stretch", hide_index=True)
            st.caption(
                "Estos coeficientes están sobre variables estandarizadas (pipeline con `StandardScaler`), "
                "por lo que el odds ratio se interpreta como cambio por +1 desviación estándar."
            )

if _show_sections("catboost_params"):
    with st.expander(
        "CatBoost avanzado: hiperparámetros útiles (y qué sí aplica aquí)",
        expanded=focus_section == "catboost_params",
    ):
        has_time_enabled = bool(pd_model_has_time) if pd_model_has_time is not None else False
        catboost_rows = [
            {
                "parámetro": "has_time",
                "qué resuelve": "Preserva el orden temporal y evita permutaciones que rompen estructura secuencial.",
                "aplica en este proyecto": "Sí (split temporal/OOT).",
                "estado": (
                    f"ACTIVO en `configs/pd_model.yaml` (`has_time={str(has_time_enabled).lower()}`)"
                    if pd_model_has_time is not None
                    else "No se pudo leer el flag desde `configs/pd_model.yaml`"
                ),
            },
            {
                "parámetro": "fixed_binary_splits",
                "qué resuelve": "Forzar splits binarios predefinidos en el árbol (caso especializado).",
                "aplica en este proyecto": "Potencialmente para umbrales regulados, pero con cautela.",
                "estado": "NO activo; requiere diseño y validación de fronteras versionadas.",
            },
            {
                "parámetro": "feature_weights",
                "qué resuelve": "Favorecer/desfavorecer variables en el score de splits (priors suaves).",
                "aplica en este proyecto": "Sí, útil para priorizar señales estables vs ruidosas.",
                "estado": "Candidato de experimento HPO guiado por MRM + SHAP.",
            },
            {
                "parámetro": "first_feature_use_penalties",
                "qué resuelve": "Penalizar la primera aparición de features costosas.",
                "aplica en este proyecto": "Más útil si el scoring online tiene costos por consulta/API.",
                "estado": "NO activo; pipeline actual es batch/offline.",
            },
            {
                "parámetro": "per_object_feature_penalties",
                "qué resuelve": "Penalización por fila/objeto (matriz de costos/availability heterogénea).",
                "aplica en este proyecto": "Útil si una fuente está disponible solo en ciertos segmentos.",
                "estado": "NO activo; requiere matriz operativa por observación.",
            },
        ]
        st.dataframe(pd.DataFrame(catboost_rows), width="stretch", hide_index=True)
        st.info(
            "Las imágenes que compartiste son útiles como ideas de diseño, pero algunas simplifican la API: "
            "`fixed_binary_splits` en la documentación de CatBoost no se usa como diccionario de thresholds por nombre "
            "de feature y además es una opción orientada a GPU. Para umbrales regulatorios suele ser más robusto usar "
            "binning/WOE o fronteras de cuantización gobernadas."
        )
        st.markdown(
            """
**Cómo sacar provecho aquí (sin inventar features)**
- Mantener `has_time=True` (ya está activo) porque la validación es temporal.
- Probar `feature_weights` en un challenger con hipótesis explícitas (ej. mayor peso a señales internas estables).
- Validar con SHAP + backtesting temporal si el prior mejora estabilidad, no solo AUC.
- Reservar `*_penalties` para un futuro score online con costos de adquisición por señal.
"""
        )

if _show_sections("comparison", "calibration"):
    # Calibration comparison
    with st.expander(
        "Comparación de métodos de calibración",
        expanded=focus_section == "calibration",
    ):
        candidates = cal_report.get("candidates", []) if isinstance(cal_report, dict) else []
        if candidates:
            rows = []
            auc_drop_limit = float(cal_report.get("auc_drop_limit", 0.0015))
            for c in candidates:
                rows.append(
                    {
                        "metodo": str(c.get("method", "")),
                        "folds": int(c.get("folds_used", 0)),
                        "mean_brier": float(c.get("mean_brier", 0.0)),
                        "mean_ece": float(c.get("mean_ece", 0.0)),
                        "mean_auc_drop": float(c.get("mean_auc_drop", 0.0)),
                        "stability": float(c.get("stability", 0.0)),
                        "cumple_auc_drop": float(c.get("mean_auc_drop", 9.0)) <= auc_drop_limit,
                    }
                )
            cal_df = pd.DataFrame(rows).sort_values(
                by=["mean_brier", "mean_ece", "stability"], ascending=[True, True, True]
            )
            st.dataframe(cal_df, width="stretch", hide_index=True)
            selected = cal_report.get("selected_method", comparison.get("best_calibration", "N/D"))
            reason = cal_report.get("selection_reason", "n/a")
            st.caption(
                f"Método seleccionado: `{selected}` | razón: `{reason}` | restricción AUC drop <= {auc_drop_limit:.4f}"
            )
        else:
            st.info(
                "No hay reporte detallado de selección de calibración en artefactos; "
                "se muestra únicamente el método ganador."
            )

    kpi_row(
        [
            {"label": "Mejor modelo", "value": comparison.get("best_model", "N/D")},
            {"label": "AUC final", "value": f"{final.get('auc_roc', 0):.4f}"},
            {"label": "KS", "value": f"{final.get('ks_statistic', 0):.4f}"},
            {"label": "Brier", "value": f"{final.get('brier_score', 0):.4f}"},
            {"label": "ECE", "value": f"{final.get('ece', 0):.4f}"},
            {"label": "Calibración", "value": comparison.get("best_calibration", "N/D")},
        ],
        n_cols=3,
    )

    with st.expander(
        "Venn-Abers: calibración canónica con garantías conformales",
        expanded=focus_section == "venn_abers",
    ):
        va_candidates = calib_diagnostics.get("candidate_comparison", [])
        selected_method = calib_diagnostics.get("selected_method", "n/d")
        if va_candidates:
            va_rows = []
            for c in va_candidates:
                va_rows.append(
                    {
                        "método": str(c.get("method", "")),
                        "ECE (OOT)": f"{float(c.get('ece', 0)):.4f}",
                        "Brier (OOT)": f"{float(c.get('brier', 0)):.4f}",
                        "AUC (OOT)": f"{float(c.get('auc', 0)):.4f}",
                        "seleccionado": str(c.get("method", "")) == selected_method,
                    }
                )
            st.dataframe(pd.DataFrame(va_rows), width="stretch", hide_index=True)
            va_meta = calib_diagnostics.get("venn_abers", {})
            avg_w = va_meta.get("avg_width")
            med_w = va_meta.get("median_width")
            unbias = va_meta.get("unbiasedness_in_the_large")
            c1, c2, c3 = st.columns(3)
            c1.metric("Método canónico", selected_method.replace("_", "-").title())
            if avg_w is not None:
                c2.metric("VA avg_width (bounds)", f"{float(avg_w):.5f}", help="Ancho medio del intervalo [p0, p1]. Cerca de 0 = calibración muy estable.")
            if med_w is not None:
                c3.metric("VA median_width", f"{float(med_w):.5f}")
            if unbias is not None:
                st.caption(
                    f"{'✅' if unbias else '⚠️'} `unbiasedness_in_the_large={'True' if unbias else 'False'}` — "
                    + ("El modelo es marginalmente insesgado en el conjunto OOT." if unbias
                       else "Leve shift de prevalencia cal→test (esperado en split OOT estricto; no afecta la validez conformal).")
                )
        else:
            st.info("Artefacto `models/pd_calibration_diagnostics.json` no disponible.")

        st.markdown(
            """
**¿Qué es Venn-Abers y por qué es coherente con el stack conformal?**

Venn-Abers (Vovk & Petej 2012) es un método de calibración post-hoc que produce **pares de probabilidad** $(p_0, p_1)$
con garantía de calibración finita bajo intercambiabilidad. No requiere hipótesis distribucionales.

| Propiedad | Platt | Isotonic | **Venn-Abers** |
|-----------|-------|----------|----------------|
| Garantía distribución-libre | ✗ | ✗ | **✓** |
| Muestra finita | ✗ | ✗ | **✓** |
| Produce bounds (incertidumbre de calibración) | ✗ | ✗ | **✓** |
| Monotonía | ✓ | ✓ | ✓ |

**Coherencia arquitectónica:** el proyecto usa conformal prediction (MAPIE Mondrian) para los intervalos de PD.
Usar Venn-Abers para la calibración base crea un stack de incertidumbre coherente:
*calibración conformal → intervalos conformales → optimización robusta*.

**Costo computacional:** O(n log n) vs O(1) de Platt — asumible para batch scoring (276K préstamos).
"""
        )
        st.caption(
            "Referencia: Vovk V. & Petej I. (2012). Venn-Abers predictors. UAI. | "
            "Artefacto: `models/pd_calibration_diagnostics.json` | "
            "Implementación: `src/models/venn_abers.py`"
        )

    metricas_interpretacion = pd.DataFrame(
        [
            {
                "Métrica": "AUC",
                "Qué mide": "Capacidad de ordenar riesgo entre default y no default",
                "Interpretación técnica": (
                    f"{final.get('auc_roc', 0):.4f} implica discriminación sólida en OOT para "
                    "datos tabulares reales."
                ),
                "Interpretación negocio": "Permite priorizar mejor qué solicitudes revisar/restringir.",
            },
            {
                "Métrica": "KS",
                "Qué mide": "Separación máxima entre distribuciones de score",
                "Interpretación técnica": (
                    f"{final.get('ks_statistic', 0):.4f} indica separación útil para estrategias por umbrales."
                ),
                "Interpretación negocio": "Facilita definir cutoffs de aprobación según apetito de riesgo.",
            },
            {
                "Métrica": "Brier",
                "Qué mide": "Error cuadrático de probabilidad",
                "Interpretación técnica": "Valor bajo mejora consistencia probabilística del score.",
                "Interpretación negocio": "Reduce sesgo en estimación de pérdidas esperadas.",
            },
            {
                "Métrica": "ECE",
                "Qué mide": "Error promedio de calibración",
                "Interpretación técnica": (
                    f"{final.get('ece', 0):.4f} sugiere muy buena calibración global."
                ),
                "Interpretación negocio": "Mayor confianza al usar PD en IFRS9 y pricing.",
            },
        ]
    )
    st.dataframe(metricas_interpretacion, width="stretch", hide_index=True)

    st.subheader("Curvas ROC")
    roc_df = try_load_parquet("roc_curve_data")
    if roc_df.empty or "model" not in roc_df.columns:
        st.info("No hay `roc_curve_data.parquet` oficial disponible para esta vista.")
    else:
        available_models = sorted(roc_df["model"].dropna().unique().tolist())
        default_models = [
            m for m in ["catboost_calibrated", "catboost_tuned", "logreg"] if m in available_models
        ]
        selected_models = st.multiselect(
            "Modelos a comparar",
            options=available_models,
            default=default_models or available_models[:2],
        )
        roc_filtered = roc_df[roc_df["model"].isin(selected_models)]

        fig = px.line(
            roc_filtered,
            x="fpr",
            y="tpr",
            color="model",
            title="Discriminación: ROC por modelo",
            labels={"fpr": "FPR", "tpr": "TPR", "model": "Modelo"},
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#888"})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=470)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: medir discriminación entre buenos y malos pagadores. "
            "Insight: CatBoost calibrado/tuned domina baseline logístico en casi todo el rango."
        )

    st.subheader("Calibración probabilística")
    cal_df = try_load_parquet("calibration_curve_data")
    if cal_df.empty or "model" not in cal_df.columns:
        st.info("No hay `calibration_curve_data.parquet` oficial disponible para esta vista.")
    else:
        fig = go.Figure()
        for model_name in sorted(cal_df["model"].dropna().unique()):
            subset = cal_df[cal_df["model"] == model_name]
            fig.add_trace(
                go.Scatter(
                    x=subset["predicted_prob"],
                    y=subset["observed_freq"],
                    mode="markers+lines",
                    name=model_name,
                )
            )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#999"})
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Probabilidad predicha vs frecuencia observada",
            xaxis_title="Probabilidad predicha",
            yaxis_title="Frecuencia observada",
            height=430,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Propósito: evaluar calidad de probabilidad. Insight: cercanía a la diagonal indica menor sesgo de calibración; "
            "esto es clave para IFRS9 y pricing."
        )

    col_nb1, col_nb2 = st.columns(2)
    with col_nb1:
        img = get_notebook_image_path("03_pd_modeling", "cell_013_out_00.png")
        if img.exists():
            st.image(
                str(img),
                caption="Notebook 03: historial de Optuna e importancia de hiperparámetros.",
                width="stretch",
            )
    with col_nb2:
        img = get_notebook_image_path("03_pd_modeling", "cell_020_out_00.png")
        if img.exists():
            st.image(
                str(img),
                caption="Notebook 03: ROC y Precision-Recall de modelos comparados.",
                width="stretch",
            )

    narrative_block(
        audience,
        general="La cercanía a la diagonal indica mejor calibración.",
        business="Probabilidades calibradas mejoran decisiones de apetito de riesgo y provisiones.",
        technical=(
            "ECE final bajo y Brier estable sustentan uso de la PD como insumo cuantitativo en capas posteriores."
        ),
    )

if _show_sections("shap"):
    st.subheader("Resumen interpretativo")
    if focus_section == "shap":
        st.caption(
            "La interpretación detallada ahora vive en la página dedicada `Explicabilidad e Interpretabilidad`."
        )
    explainability_global = try_load_parquet("explainability_global")
    explainability_local = try_load_parquet("explainability_local_cases")
    explanation_drift = try_load_parquet("explanation_drift")
    if explainability_global.empty and explainability_local.empty:
        st.info(
            "No hay artefactos de explicabilidad en este entorno; se omite el resumen interpretativo."
        )
    else:
        kpi_cards = [
            {
                "label": "Drivers globales",
                "value": str(len(explainability_global)) if not explainability_global.empty else "N/D",
            },
            {
                "label": "Casos locales",
                "value": str(len(explainability_local)) if not explainability_local.empty else "N/D",
            },
            {
                "label": "Overlap top-10",
                "value": (
                    f"{float(explanation_drift['rank_overlap_top10'].min()):.3f}"
                    if not explanation_drift.empty
                    else "N/D"
                ),
            },
            {
                "label": "Reason code drift",
                "value": (
                    "PASS"
                    if (not explanation_drift.empty and bool(explanation_drift["passed_all"].all()))
                    else "N/D"
                ),
            },
        ]
        kpi_row(kpi_cards, n_cols=4)

        if not explainability_global.empty:
            top_n = min(8, len(explainability_global))
            summary_df = explainability_global.head(top_n).sort_values("mean_abs_shap")
            fig = px.bar(
                summary_df,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                color="feature_family" if "feature_family" in summary_df.columns else None,
                labels={"mean_abs_shap": "Impacto medio |SHAP|", "feature": "Variable"},
                title="Top drivers globales (resumen)",
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, width="stretch")

        col_local, col_handoff = st.columns([1.0, 1.0], gap="large")
        with col_local:
            if explainability_local.empty:
                st.info("No hay casos locales explicados en este entorno.")
            else:
                local_cols = [
                    col
                    for col in ["segmento", "score_raw", "pd_calibrada", "reason_code_text"]
                    if col in explainability_local.columns
                ]
                st.dataframe(
                    explainability_local.loc[:, local_cols],
                    width="stretch",
                    hide_index=True,
                )
        with col_handoff:
            st.markdown(
                """
**Qué ya no resolvemos aquí**
- drivers globales canónicos completos
- casos locales con intervalos conformales
- ALE vs PDP/ICE
- redundancia/interacciones SHAP
- explanation drift y challenger desde la óptica de interpretabilidad
"""
            )
            st.markdown(
                """
**Por qué moverlo**
- El laboratorio debe concentrarse en selección de modelo y calibración.
- La interpretabilidad merece una narrativa propia y reutilizable por gobernanza.
- Así evitamos duplicar visuales y definiciones entre páginas.
"""
            )
            try:
                st.page_link(
                    "pages/model_interpretability.py",
                    label="Abrir la página dedicada de explicabilidad",
                    icon="➡️",
                )
            except Exception:
                st.caption("➡️ pages/model_interpretability.py")

        st.caption(
            "Lectura correcta: aquí solo validamos que el modelo ganador siga siendo interpretable; "
            "la defensa completa de drivers, effects y reason codes vive en la página dedicada."
        )

if _show_sections("upgrades"):
    with st.expander(
        "Mejoras habilitadas por upgrades recientes (medium-risk)",
        expanded=focus_section == "upgrades",
    ):
        versions_df = load_package_versions_for_model_lab()
        st.dataframe(versions_df, width="stretch", hide_index=True)
        st.markdown(
            """
**Qué sí mejoró ya en el proyecto**
- `scikit-learn 1.8`: habilita `d2_brier_score` para evaluar calidad probabilística relativa a un baseline.
- `shap 0.50`: mejor compatibilidad/soporte moderno para workflows de explicabilidad (notebooks y análisis local).
- `scikit-survival 0.27`: mejora alineación con stack actualizado de `scikit-learn`.
- `pyarrow 23`: mejoras de IO/compatibilidad útiles para artefactos Parquet del pipeline.
- `mlflow 3.10`: mejoras del stack de tracking/tracing para gobernanza experimental.
- `cvxpy 1.8.1` y `dowhy 0.14`: base más moderna para optimización y causalidad en módulos aguas abajo.
"""
        )

        try:
            from sklearn.metrics import d2_brier_score as _d2_brier_score
        except ImportError:
            _d2_brier_score = None

        if (
            _d2_brier_score is not None
            and not test_predictions.empty
            and {"y_true", "pd_logreg", "pd_calibrated"}.issubset(test_predictions.columns)
        ):
            d2_rows = []
            y_true = test_predictions["y_true"].to_numpy()
            for label, col in [
                ("Logistic Regression (baseline)", "pd_logreg"),
                ("CatBoost calibrado (final)", "pd_calibrated"),
            ]:
                y_prob = test_predictions[col].to_numpy()
                d2_rows.append({"modelo": label, "d2_brier_score": float(_d2_brier_score(y_true, y_prob))})
            st.dataframe(pd.DataFrame(d2_rows), width="stretch", hide_index=True)
            st.caption(
                "Lectura rápida: cuanto mayor es `D² Brier`, mejor explica la variabilidad de la probabilidad "
                "frente a un predictor de referencia constante."
            )

        if versions_df.loc[versions_df["package"].eq("econml"), "version"].astype(str).str.contains("not installed").any():
            st.caption(
                "Nota de arquitectura: `econml` se movió fuera del entorno principal para evitar bloquear "
                "upgrades de `scikit-learn` y `shap`. Los workflows causales siguen disponibles en un env separado."
            )

# ── Rare Event Calibration Diagnostics ──────────────────────────────────
rare_event_status = try_load_json("pd_rare_event_calibration_status", directory="models", default={})
with st.expander("Rare Event Calibration Diagnostics", expanded=False):
    if rare_event_status:
        re_cols = st.columns(3)
        _re_summ = rare_event_status.get("summary", rare_event_status.get("global", {}))
        _re_method = rare_event_status.get("method") or rare_event_status.get("config", {}).get("calibration_method") or "auto"
        _re_brier = _re_summ.get("brier") or rare_event_status.get("brier_score")
        _re_ece = _re_summ.get("ece") or rare_event_status.get("ece")
        re_cols[0].metric("Método", _re_method)
        re_cols[1].metric("Brier Score", f"{_re_brier:.4f}" if isinstance(_re_brier, float) else "N/D")
        re_cols[2].metric("ECE", f"{_re_ece:.4f}" if isinstance(_re_ece, float) else "N/D")
        if "class_balance" in rare_event_status:
            st.markdown(f"**Balance de clase (default rate):** {rare_event_status['class_balance']}")
        if "notes" in rare_event_status:
            st.caption(rare_event_status["notes"])
        st.caption(
            "Lectura: estas métricas reflejan la calidad de calibración bajo desbalance de clase severo. "
            "Un Brier Score bajo y ECE cercano a cero indican calibración confiable para eventos raros."
        )
    else:
        st.info(
            "No se encontró `models/pd_rare_event_calibration_status.json`. "
            "Ejecute el pipeline de calibración de eventos raros para generar este artefacto."
        )

# ── PD Conformal Gap Analysis ────────────────────────────────────────────
with st.expander("Análisis de gap conformal y atribución de ancho de intervalo", expanded=False):
    _gap_summary = try_load_json("conformal_gap_summary", directory="models", default={})
    _gap_exp = try_load_parquet("pd_conformal_gap_experiments")
    _gap_top = try_load_parquet("pd_conformal_gap_top_candidates")
    _width_attr = try_load_parquet("pd_conformal_width_attribution")

    if _gap_summary:
        st.markdown(
            f"**Propósito:** {_gap_summary.get('purpose', 'Exploración del espacio de configuración conformal.')}  \n"
            f"**Decisión:** {_gap_summary.get('decision', 'N/D')}  \n"
            f"**Justificación:** {_gap_summary.get('decision_rationale', 'N/D')}"
        )
        _nc = _gap_summary.get("n_candidates", 0)
        st.caption(f"Candidatos evaluados: {_nc}")

    if not _gap_top.empty:
        st.markdown("**Top candidatos seleccionados (Pareto-óptimos)**")
        _top_cols = [c for c in ["selection_rank", "partition", "scaled_scores", "coverage_90",
                                  "min_group_coverage_90", "avg_width_90", "strict_overall_pass",
                                  "checks_passed"] if c in _gap_top.columns]
        st.dataframe(_gap_top[_top_cols], width="stretch", hide_index=True)

    if not _width_attr.empty:
        st.markdown("**Atribución de ancho por etapa de experimento**")
        _attr_cols = [c for c in ["dataset_scope", "stage", "coverage_90", "min_group_coverage_90",
                                   "avg_width_90", "winkler_90"] if c in _width_attr.columns]
        st.dataframe(_width_attr[_attr_cols], width="stretch", hide_index=True)
        st.caption("Cómo el ancho del intervalo varía según el dataset de referencia y la etapa del experimento.")

    if _gap_exp.empty and not _gap_summary:
        st.info("Ejecuta `scripts/run_conformal_gap_analysis.py` para generar estos diagnósticos.")

# ── Rare Event Calibration Report (Parquet detail) ────────────────────────
with st.expander("Reporte detallado: calibración para eventos raros", expanded=False):
    _rare_report = try_load_parquet("pd_rare_event_calibration_report")
    if not _rare_report.empty:
        st.markdown(
            "Diagnóstico de calibración por decil de score y por tipo de slice. "
            "Las barras de ECE revelan zonas donde el modelo sobreestima o subestima la PD real."
        )
        _decile_view = _rare_report[_rare_report["report_type"] == "decile"] if "report_type" in _rare_report.columns else _rare_report
        if not _decile_view.empty:
            _decile_cols = [c for c in ["score_decile", "n", "prevalence", "mean_score", "brier", "ece_component"] if c in _decile_view.columns]
            st.dataframe(_decile_view[_decile_cols], width="stretch", hide_index=True)
            st.caption("Prevalencia = default rate real en cada decil. mean_score = PD predicha promedio. ece_component = contribución al ECE global.")
    else:
        st.info("Ejecuta `scripts/analyze_pd_rare_event_calibration.py` para generar el reporte detallado.")

# ── PD Slice Performance + HPO Seed Replay ────────────────────────────────
with st.expander("Performance por slice temporal/grade y estabilidad HPO", expanded=False):
    _slice_perf = try_load_json("pd_slice_performance", directory="models", default={})
    _hpo_replay = try_load_json("pd_hpo_seed_replay_status", directory="models", default={})

    if _slice_perf:
        st.markdown("**Performance por slice** (AUC, Brier, ECE por combinación temporal/grade)")
        _slices = _slice_perf.get("slice_performance", [])
        if _slices:
            _slice_df = pd.DataFrame(_slices)
            st.dataframe(_slice_df, width="stretch", hide_index=True)
        _if_diag = _slice_perf.get("isolation_forest", {})
        if _if_diag:
            st.markdown(f"**Isolation Forest (anomalías):** {_if_diag.get('n_anomalies', 0)} slices anómalos detectados de {_if_diag.get('n_total', 0)} total.")

    if _hpo_replay:
        st.markdown("**Estabilidad HPO (replay de seeds)**")
        replay = _hpo_replay.get("replay", {})
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Calibración seleccionada", str(_hpo_replay.get("selected_calibration_method", "N/D")))
        hc2.metric("AUC validación", f"{_hpo_replay.get('validation_auc', 0):.4f}" if _hpo_replay.get("validation_auc") else "N/D")
        hc3.metric("AUC OOT", f"{_hpo_replay.get('oot_auc', 0):.4f}" if _hpo_replay.get("oot_auc") else "N/D")
        hc4.metric("Brier", f"{_hpo_replay.get('brier', 0):.4f}" if _hpo_replay.get("brier") else "N/D")
        if replay:
            _n_seeds = replay.get("n_seeds", "N/D")
            _auc_std = replay.get("auc_std", None)
            st.caption(
                f"Seeds evaluados: {_n_seeds}. "
                + (f"Std AUC entre seeds: {_auc_std:.5f} — " if _auc_std else "")
                + "Baja varianza entre seeds confirma estabilidad del proceso de entrenamiento."
            )

    if not _slice_perf and not _hpo_replay:
        st.info("Ejecuta los scripts de diagnóstico de PD para generar estos artefactos.")

st.markdown(
    """
**Conclusión del laboratorio:**
- Se eligió CatBoost calibrado por equilibrio entre discriminación y confiabilidad probabilística.
- La interpretabilidad del campeón sigue alineada con teoría de riesgo, pero su desarrollo completo vive en la página dedicada.
- Este bloque alimenta directamente incertidumbre conformal y decisiones robustas.
"""
)
st.markdown(
    """
En narrativa de proyecto, aquí se fija el “motor probabilístico” que el resto del stack consume. A partir de este punto,
la discusión deja de ser únicamente predictiva y pasa a ser decisional: cuánto confiar en cada PD, cómo protegerse frente a
incertidumbre y cómo convertir esa información en políticas de cartera y provisión más robustas.
"""
)
render_caveats(
    [
        "El baseline logístico sigue siendo referencia regulatoria aunque no sea el campeón.",
        "Las métricas mostradas son snapshot; deben leerse junto a estabilidad temporal y gobernanza.",
        "SHAP explica correlaciones locales del modelo, no causalidad.",
    ]
)
render_page_feedback("model_laboratory")

next_page_teaser(
    "Explicabilidad e Interpretabilidad",
    "Drivers globales, ALE, reason codes, casos locales e interpretación estable del score PD.",
    "pages/model_interpretability.py",
)
