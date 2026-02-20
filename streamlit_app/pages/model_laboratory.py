"""Laboratorio de modelos PD: desempeño, calibración e interpretabilidad."""

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

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import narrative_block, next_page_teaser, storytelling_intro
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import get_notebook_image_path, load_json, load_parquet, try_load_parquet

st.title("🔬 Laboratorio de Modelos")
st.caption("Comparación de modelos PD, calibración y explicabilidad (SHAP).")
st.markdown(
    """
Este capítulo traduce teoría de modelado tabular a una decisión concreta de arquitectura PD. La intención no es exhibir
un benchmark superficial, sino documentar por qué se escoge un modelo específico, cómo se valida fuera de muestra temporal
y por qué la calibración probabilística es tan importante como el ranking. En riesgo de crédito, una PD mal calibrada puede
desalinear pricing, límites y provisiones aunque el AUC sea alto.
"""
)
audience = audience_selector()

storytelling_intro(
    page_goal="Determinar qué arquitectura de PD usar y por qué, no solo quién gana un benchmark.",
    business_value="Reduce errores de aprobación y evita usar probabilidades mal calibradas en pricing e IFRS9.",
    key_decision="Adoptar el modelo final y su política de calibración para operación.",
    how_to_read=[
        "Mirar primero comparativo de modelos y métricas finales.",
        "Validar calibración (Brier/ECE) además de AUC/KS.",
        "Usar SHAP para explicar por qué el modelo decide así.",
    ],
)

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
    st.latex(r"\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,\qquad \text{ECE}=\sum_b w_b\left|\hat{p}_b-\hat{y}_b\right|")

comparison = load_json("model_comparison")
models = pd.DataFrame(comparison.get("models", []))
final = comparison.get("final_test_metrics", {})
cal_report = comparison.get("calibration_selection_report", {})
hpo_trials = int(
    comparison.get("hpo_trials_executed", comparison.get("optuna_n_trials", 0))
)
feature_count_tuned = int(comparison.get("feature_count_tuned", 0))

st.subheader("Comparativo de arquitecturas")
if not models.empty:
    models_view = models.copy()
    models_view["es_mejor"] = models_view["model"].eq(comparison.get("best_model", ""))
    st.dataframe(models_view, use_container_width=True, hide_index=True)

with st.expander("¿Qué es CatBoost y por qué se usa en credit scoring?", expanded=False):
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

with st.expander("Discusión técnica: Logistic Regression vs CatBoost", expanded=False):
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

# Calibration comparison
with st.expander("Comparación de métodos de calibración", expanded=False):
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
        st.dataframe(cal_df, use_container_width=True, hide_index=True)
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
st.dataframe(metricas_interpretacion, use_container_width=True, hide_index=True)

st.subheader("Curvas ROC")
roc_df = load_parquet("roc_curve_data")
available_models = sorted(roc_df["model"].dropna().unique().tolist())
default_models = [m for m in ["catboost_calibrated", "catboost_tuned", "logreg"] if m in available_models]
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
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Propósito: medir discriminación entre buenos y malos pagadores. "
    "Insight: CatBoost calibrado/tuned domina baseline logístico en casi todo el rango."
)

st.subheader("Calibración probabilística")
cal_df = load_parquet("calibration_curve_data")
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
st.plotly_chart(fig, use_container_width=True)
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
            use_container_width=True,
        )
with col_nb2:
    img = get_notebook_image_path("03_pd_modeling", "cell_020_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="Notebook 03: ROC y Precision-Recall de modelos comparados.",
            use_container_width=True,
        )

narrative_block(
    audience,
    general="La cercanía a la diagonal indica mejor calibración.",
    business="Probabilidades calibradas mejoran decisiones de apetito de riesgo y provisiones.",
    technical=(
        "ECE final bajo y Brier estable sustentan uso de la PD como insumo cuantitativo en capas posteriores."
    ),
)

st.subheader("Interpretabilidad con SHAP")
shap_summary = try_load_parquet("shap_summary")
if shap_summary.empty:
    st.info("No hay artefactos SHAP en este entorno; se omite la sección de interpretabilidad avanzada.")
else:
    top_n = st.slider("Top variables por importancia SHAP", min_value=8, max_value=25, value=15, step=1)
    shap_top = shap_summary.head(top_n).sort_values("mean_abs_shap")

    fig = px.bar(
        shap_top,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=f"Top {top_n} variables más influyentes",
        labels={"mean_abs_shap": "Impacto medio |SHAP|", "feature": "Variable"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=max(360, top_n * 26))
    fig.update_traces(marker_color="#00D4AA")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Propósito: identificar variables que explican el score. Insight: tasa de interés, plazo y calidad crediticia "
        "concentran mayor aporte al riesgo."
    )

    if audience in ("Negocio", "Técnico"):
        shap_raw = try_load_parquet("shap_raw_top20")
        shap_features = sorted(
            [c.replace("shap_", "") for c in shap_raw.columns if c.startswith("shap_")]
        )
        if shap_raw.empty or not shap_features:
            st.info("No hay `shap_raw_top20.parquet`; se omite dependencia SHAP.")
        else:
            selected_feat = st.selectbox("Variable para explorar dependencia SHAP", shap_features, index=0)
            shap_col = f"shap_{selected_feat}"
            val_col = f"val_{selected_feat}"
            if shap_col in shap_raw.columns and val_col in shap_raw.columns:
                sample = shap_raw.sample(min(3000, len(shap_raw)), random_state=17)
                fig = px.scatter(
                    sample,
                    x=val_col,
                    y=shap_col,
                    opacity=0.35,
                    title=f"Dependencia SHAP: {selected_feat}",
                    labels={val_col: f"Valor de {selected_feat}", shap_col: "Contribución SHAP"},
                )
                fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Propósito: ver dirección y magnitud local del efecto por variable. "
                    "Insight: el impacto no es lineal en todo el dominio."
                )

img = get_notebook_image_path("03_pd_modeling", "cell_017_out_00.png")
if img.exists():
    st.image(
        str(img),
        caption="Notebook 03: SHAP beeswarm + importancia global (figura original).",
        use_container_width=True,
    )

st.markdown(
    """
**Conclusión del laboratorio:**
- Se eligió CatBoost calibrado por equilibrio entre discriminación y confiabilidad probabilística.
- SHAP muestra drivers coherentes con teoría de riesgo de crédito (tasa, score, carga financiera).
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

next_page_teaser(
    "Cuantificación de Incertidumbre",
    "Pasamos de probabilidades puntuales a bandas de riesgo con cobertura empírica.",
    "pages/uncertainty_quantification.py",
)
