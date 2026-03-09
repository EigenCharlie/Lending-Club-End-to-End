"""Ingeniería de features: transformación de variables crudas a señales predictivas."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.audience_toggle import audience_selector
from streamlit_app.components.narrative import (
    narrative_block,
    next_page_teaser,
    storytelling_intro,
)
from streamlit_app.components.story_shell import (
    render_key_takeaway,
    render_page_feedback,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import (
    get_notebook_image_path,
    load_json,
    try_load_parquet,
)

st.title("🔧 Ingeniería de Features")
st.caption(
    "De 142 variables crudas a 60 features predictivas: limpieza, transformación, "
    "WOE encoding y selección por Information Value."
)
page_contract = get_page_contract("feature_engineering")
render_page_header(page_contract)
render_key_takeaway(
    "La ingeniería de features define el techo de desempeño y la interpretabilidad del sistema; errores aquí se propagan a todo el pipeline downstream."
)

audience = audience_selector()
storytelling_intro(
    page_goal=("Transformar variables crudas en señales estables y útiles para predecir default."),
    business_value=(
        "La calidad de features define gran parte del desempeño y la interpretabilidad del score."
    ),
    key_decision=(
        "Decidir qué variables conservar, transformar o eliminar para maximizar valor predictivo sin leakage."
    ),
    how_to_read=[
        "Empieza por el pipeline de 142→110→60 para entender la depuración.",
        "Revisa features creadas y su intuición de riesgo.",
        "Usa IV/WOE para validar por qué entran al modelo final.",
    ],
)

narrative_block(
    audience,
    general="Las variables crudas del dataset necesitan transformación antes de alimentar un modelo. "
    "Es como preparar ingredientes antes de cocinar: seleccionar los mejores, limpiarlos y cortarlos "
    "en el formato adecuado.",
    business="La ingeniería de features es la etapa que más impacta el poder predictivo del modelo. "
    "Aquí se decide qué información entra al score, cómo se codifica y qué se descarta. "
    "Una mala selección de variables puede invalidar todo el pipeline downstream.",
    technical="Pipeline: Raw CSV (142 cols) → eliminación de leakage, nulls, IDs, constantes (→110 cols) → "
    "feature engineering (ratios, buckets, WOE, flags, interacciones) → selección por IV → "
    "60 features finales (CatBoost). WOE vía OptBinning con supervision monotónica.",
)

# ── 1. Pipeline Visual ──
st.subheader("1) Pipeline de transformación")

pipeline_data = pd.DataFrame(
    [
        {
            "Etapa": "1. Datos crudos",
            "Columnas": "142",
            "Acción": "CSV original de Kaggle (2.93M filas)",
            "Razón": "Punto de partida",
        },
        {
            "Etapa": "2. Limpieza (leakage + nulidad + IDs)",
            "Columnas": "142 → 110",
            "Acción": "Remover leakage, nulls >80%, IDs, constantes, texto libre",
            "Razón": "Variables sin valor predictivo o que generan trampa",
        },
        {
            "Etapa": "3. Feature engineering",
            "Columnas": "110 + 15 creadas",
            "Acción": "Ratios, logs, buckets, WOE encoding, flags, interacciones",
            "Razón": "Capturar señales no lineales y relaciones",
        },
        {
            "Etapa": "4. Reemplazar/eliminar redundantes",
            "Columnas": "−50 reemplazadas",
            "Acción": "sub_grade→grade_woe, timestamps→features temporales, etc.",
            "Razón": "WOE/buckets reemplazan originales",
        },
        {
            "Etapa": "5. Selección final",
            "Columnas": "60",
            "Acción": "Ranking por IV + importancia SHAP",
            "Razón": "Balance poder predictivo vs complejidad",
        },
    ]
)
st.dataframe(pipeline_data, width="stretch", hide_index=True)

st.info(
    "**Punto crítico — Data Leakage**: Se removieron 15 variables que solo existen después de que "
    "el préstamo termina (total_pymnt, recoveries, collection_recovery_fee, etc.). Incluirlas "
    "inflaría artificialmente el AUC pero produciría un modelo inútil en producción."
)

# ── 2. Features Creadas ──
st.subheader("2) Variables creadas (Feature Engineering)")

narrative_block(
    audience,
    general="Se crearon nuevas variables combinando las originales para capturar patrones que "
    "una sola variable no revela. Por ejemplo, el ratio préstamo/ingreso dice más sobre "
    "el riesgo que el monto del préstamo solo.",
    business="Cada variable creada tiene una justificación de negocio: mide carga financiera, "
    "comportamiento crediticio o estabilidad del solicitante. No son transformaciones arbitrarias.",
    technical="Transformaciones: logarítmicas (normalizar distribuciones sesgadas), ratios (capturar "
    "interacciones), bucketing por cuantiles (discretizar continuas), WOE encoding (relación "
    "monotónica con target), flags de missing (missingness informativa).",
)

features_created = pd.DataFrame(
    [
        {
            "Feature": "loan_to_income",
            "Fórmula/Método": "loan_amnt / annual_inc",
            "Tipo": "Ratio",
            "Intuición de riesgo": "Carga del préstamo relativa a capacidad de pago. Ratio alto = mayor riesgo.",
        },
        {
            "Feature": "revol_utilization",
            "Fórmula/Método": "revol_bal × revol_util / 100",
            "Tipo": "Ratio",
            "Intuición de riesgo": "Uso real del crédito revolvente. Alta utilización señala estrés financiero.",
        },
        {
            "Feature": "log_loan_amnt",
            "Fórmula/Método": "log(loan_amnt)",
            "Tipo": "Log transform",
            "Intuición de riesgo": "Normaliza distribución sesgada a la derecha. Mejora linealidad.",
        },
        {
            "Feature": "log_annual_inc",
            "Fórmula/Método": "log(annual_inc)",
            "Tipo": "Log transform",
            "Intuición de riesgo": "Comprime la cola larga de ingresos altos. Estabiliza varianza.",
        },
        {
            "Feature": "int_rate_bucket",
            "Fórmula/Método": "Quantile binning (deciles)",
            "Tipo": "Bucket",
            "Intuición de riesgo": "Captura relación no-lineal entre tasa y default. Cada bucket tiene su tasa de default propia.",
        },
        {
            "Feature": "dti_bucket",
            "Fórmula/Método": "Quantile binning (deciles)",
            "Tipo": "Bucket",
            "Intuición de riesgo": "Segmenta DTI en bandas de riesgo homogéneo.",
        },
        {
            "Feature": "grade_woe",
            "Fórmula/Método": "WOE encoding (OptBinning)",
            "Tipo": "WOE",
            "Intuición de riesgo": "Transforma grade A-G a escala continua ponderada por default rate. Mayor WOE = menor riesgo.",
        },
        {
            "Feature": "purpose_woe",
            "Fórmula/Método": "WOE encoding (OptBinning)",
            "Tipo": "WOE",
            "Intuición de riesgo": "Codifica propósito del préstamo según su asociación histórica con default.",
        },
        {
            "Feature": "home_ownership_woe",
            "Fórmula/Método": "WOE encoding (OptBinning)",
            "Tipo": "WOE",
            "Intuición de riesgo": "Situación de vivienda como proxy de estabilidad financiera.",
        },
        {
            "Feature": "emp_length_cat",
            "Fórmula/Método": "Binned ordinal",
            "Tipo": "Bucket",
            "Intuición de riesgo": "Antigüedad laboral agrupada como proxy de estabilidad.",
        },
        {
            "Feature": "log_annual_inc_miss",
            "Fórmula/Método": "Indicador 1/0 de nulo",
            "Tipo": "Flag",
            "Intuición de riesgo": "La ausencia de dato de ingreso puede ser informativa (auto-reporte incompleto).",
        },
        {
            "Feature": "dti_miss",
            "Fórmula/Método": "Indicador 1/0 de nulo",
            "Tipo": "Flag",
            "Intuición de riesgo": "DTI nulo puede indicar falta de historial de deuda o dato no verificable.",
        },
        {
            "Feature": "days_since_delinq_miss",
            "Fórmula/Método": "Indicador 1/0 de nulo",
            "Tipo": "Flag",
            "Intuición de riesgo": "Nulo = nunca hubo morosidad (buen signo). Informativo para el modelo.",
        },
        {
            "Feature": "int_rate_bucket__grade",
            "Fórmula/Método": "Interacción bucket × grade",
            "Tipo": "Interacción",
            "Intuición de riesgo": "Captura que el impacto de la tasa depende del grade asignado.",
        },
    ]
)
st.dataframe(features_created, width="stretch", hide_index=True)

# ── 3. IV Ranking ──
st.subheader("3) Ranking por Information Value (IV)")

narrative_block(
    audience,
    general="Information Value mide qué tan útil es cada variable para distinguir entre préstamos "
    "que pagan y los que no. Es como un examen de admisión para variables: solo las que "
    "aportan información real pasan.",
    business="Las variables con IV bajo (<0.02) se descartan porque no ayudan a predecir default. "
    "Las de IV alto (>0.3) son las más valiosas para el scoring y deben monitorearse activamente.",
    technical="IV = Σ (D%−ND%) × ln(D%/ND%) sobre bins. Rangos: <0.02 débil, 0.02-0.1 útil, "
    "0.1-0.3 fuerte, >0.3 muy fuerte. WOE features tienden a IV alto por diseño.",
)

iv_data = load_json("feature_importance_iv")
iv_scores = iv_data.get("iv_scores", {})

if iv_scores:
    n_top = st.slider("Número de features a mostrar", 8, min(30, len(iv_scores)), 20)
    top_iv = dict(list(iv_scores.items())[:n_top])

    iv_df = pd.DataFrame({"feature": list(top_iv.keys()), "iv": list(top_iv.values())})
    iv_df = iv_df.sort_values("iv", ascending=True)

    fig = px.bar(
        iv_df,
        x="iv",
        y="feature",
        orientation="h",
        title=f"Top {n_top} features por Information Value",
        labels={"iv": "Information Value", "feature": ""},
        color="iv",
        color_continuous_scale="YlOrRd",
    )
    # Add IV threshold lines
    for threshold, label, color in [
        (0.02, "Débil", "#94A3B8"),
        (0.1, "Fuerte", "#F59E0B"),
        (0.3, "Muy fuerte", "#EF4444"),
    ]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color, annotation_text=label)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=max(350, n_top * 28), coloraxis_showscale=False
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Líneas verticales: umbrales de IV. <0.02=débil (no predictiva), 0.02-0.1=útil, "
        "0.1-0.3=fuerte, >0.3=muy fuerte. Las features WOE dominan el ranking."
    )

    # Feature family summary
    feature_lists = iv_data.get("feature_lists", {})
    st.markdown(
        f"""
**Resumen de familias de features:**
- Numéricas: **{len(feature_lists.get("numeric", []))}** variables
- Categóricas: **{len(feature_lists.get("categorical", []))}** variables
- WOE: **{len(feature_lists.get("woe", []))}** variables
- Flags: **{len(feature_lists.get("flag", []))}** variables
- Interacciones: **{len(feature_lists.get("interaction", []))}** variables
- Total CatBoost: **{len(feature_lists.get("catboost", []))}** features
"""
    )
else:
    perm = try_load_parquet("permutation_importance")
    shap = try_load_parquet("shap_summary")
    fallback_df = pd.DataFrame()
    score_col = "score"
    x_label = ""
    title = ""
    color_scale = "Teal"
    fallback_source = ""

    if not perm.empty and {"feature", "auc_drop"}.issubset(perm.columns):
        fallback_df = perm[["feature", "auc_drop"]].copy()
        fallback_df["feature"] = fallback_df["feature"].astype(str)
        fallback_df[score_col] = pd.to_numeric(fallback_df["auc_drop"], errors="coerce")
        fallback_df = fallback_df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        x_label = "AUC drop al permutar feature"
        title = "Ranking predictivo (fallback): permutation importance"
        color_scale = "Teal"
        fallback_source = "permutation"
    elif not shap.empty and {"feature", "mean_abs_shap"}.issubset(shap.columns):
        fallback_df = shap[["feature", "mean_abs_shap"]].copy()
        fallback_df["feature"] = fallback_df["feature"].astype(str)
        fallback_df[score_col] = pd.to_numeric(fallback_df["mean_abs_shap"], errors="coerce")
        fallback_df = fallback_df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        x_label = "Mean absolute SHAP"
        title = "Ranking explicativo (fallback): SHAP"
        color_scale = "Viridis"
        fallback_source = "shap"

    if not fallback_df.empty:
        n_max = min(30, len(fallback_df))
        n_top = st.slider("Número de features a mostrar", 8, n_max, min(20, n_max))
        plot_df = fallback_df.head(n_top).sort_values(score_col, ascending=True)
        fig = px.bar(
            plot_df,
            x=score_col,
            y="feature",
            orientation="h",
            title=title,
            labels={score_col: x_label, "feature": ""},
            color=score_col,
            color_continuous_scale=color_scale,
        )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=max(350, n_top * 28), coloraxis_showscale=False
        )
        st.plotly_chart(fig, width="stretch")
        if fallback_source == "permutation":
            st.info(
                "Fallback activo: `iv_scores` está vacío en el snapshot actual; se usa permutation importance para mantener el ranking."
            )
        else:
            st.info(
                "Fallback activo: `iv_scores` vacío y sin permutation importance; se usa SHAP como proxy explicativo."
            )
    else:
        st.warning(
            "No hay datos para ranking (`iv_scores`, `permutation_importance` o `shap_summary`). "
            "Ejecuta export de artefactos para poblar esta sección."
        )

    feature_lists = iv_data.get("feature_lists", {})
    st.markdown(
        f"""
**Resumen de familias de features:**
- Numéricas: **{len(feature_lists.get("numeric", []))}** variables
- Categóricas: **{len(feature_lists.get("categorical", []))}** variables
- WOE: **{len(feature_lists.get("woe", []))}** variables
- Flags: **{len(feature_lists.get("flag", []))}** variables
- Interacciones: **{len(feature_lists.get("interaction", []))}** variables
- Total CatBoost: **{len(feature_lists.get("catboost", []))}** features
"""
    )

# ── 4. WOE Explained ──
st.subheader("4) Weight of Evidence (WOE) explicado")

col_woe1, col_woe2 = st.columns(2)

with col_woe1:
    st.markdown(
        """
**¿Qué es WOE?**

Weight of Evidence transforma categorías en valores numéricos que reflejan su
relación con el default. Para cada categoría (bin):

$$WOE_i = \\ln\\left(\\frac{\\%\\ No\\ Default_i}{\\%\\ Default_i}\\right)$$

- **WOE positivo** → categoría con menor proporción de defaults (bajo riesgo)
- **WOE negativo** → categoría con mayor proporción de defaults (alto riesgo)
- **WOE = 0** → proporción de defaults igual al promedio
"""
    )

with col_woe2:
    st.markdown(
        """
**¿Por qué WOE en credit scoring?**

1. **Interpretabilidad**: Transforma categorías a escala de riesgo monotónica
2. **Manejo de categorías raras**: Agrupa bins con pocos datos
3. **Estándar regulatorio**: Aceptado por supervisores bancarios (EBA, OCC)
4. **Compatibilidad**: Funciona tanto en LogReg como en gradient boosting
5. **OptBinning**: Librería que optimiza bins respetando monotonicidad
"""
    )

st.info(
    "En este proyecto, WOE se aplica a **grade**, **purpose** y **home_ownership**. "
    "Estas tres features WOE están entre las de mayor IV del modelo, confirmando que "
    "la transformación captura información predictiva relevante."
)

# ── 5. Variables Eliminadas ──
st.subheader("5) Variables eliminadas: anatomía de la limpieza")

st.markdown(
    """
El dataset crudo tiene **142 columnas**. Tras limpieza se retienen **110 columnas** (32 eliminadas
por nulidad, leakage, IDs y constantes). Luego, la ingeniería de features transforma, reemplaza y
selecciona hasta llegar a **60 features finales** para CatBoost.
"""
)

eliminated = pd.DataFrame(
    [
        {
            "Etapa": "1. Limpieza inicial",
            "Razón": "Alta nulidad (>80%)",
            "Eliminadas": 14,
            "Ejemplos": "mths_since_last_major_derog, annual_inc_joint, dti_joint, il_util",
            "Justificación": "Datos insuficientes para aprender patrones confiables",
        },
        {
            "Etapa": "1. Limpieza inicial",
            "Razón": "Data leakage (post-loan)",
            "Eliminadas": 10,
            "Ejemplos": "total_pymnt, recoveries, collection_recovery_fee, out_prncp",
            "Justificación": "Solo existen después de que termina el préstamo — usarlas sería trampa",
        },
        {
            "Etapa": "1. Limpieza inicial",
            "Razón": "Identificadores",
            "Eliminadas": 2,
            "Ejemplos": "id, member_id",
            "Justificación": "Únicos por fila, sin poder predictivo",
        },
        {
            "Etapa": "1. Limpieza inicial",
            "Razón": "Constantes / quasi-constantes",
            "Eliminadas": 3,
            "Ejemplos": "policy_code, pymnt_plan",
            "Justificación": "Un solo valor para todos los préstamos",
        },
        {
            "Etapa": "1. Limpieza inicial",
            "Razón": "Texto libre no estructurado",
            "Eliminadas": 3,
            "Ejemplos": "emp_title, title, desc",
            "Justificación": "Requeriría NLP; reemplazado por purpose y emp_length",
        },
        {
            "Etapa": "2. Feature engineering",
            "Razón": "Reemplazadas por encoding WOE/bucket",
            "Eliminadas": 45,
            "Ejemplos": "sub_grade → grade_woe, addr_state, timestamps crudos",
            "Justificación": "Representación más eficiente generada",
        },
        {
            "Etapa": "2. Feature engineering",
            "Razón": "Duplicadas / redundantes",
            "Eliminadas": 5,
            "Ejemplos": "funded_amnt (≈loan_amnt), funded_amnt_inv",
            "Justificación": "Información ya capturada por otra variable",
        },
    ]
)
st.dataframe(eliminated, width="stretch", hide_index=True)

st.warning(
    "**Resultado neto: 142 columnas originales → 110 tras limpieza → 60 features finales** "
    "(con +15 variables creadas y selección por IV/SHAP). "
    "La limpieza rigurosa es la primera línea de defensa contra modelos sobreajustados o con leakage."
)

# ── 6. Notebook Images ──
st.subheader("6) Evidencia visual del Notebook 02")

col_img1, col_img2 = st.columns(2)
with col_img1:
    img = get_notebook_image_path("02_feature_engineering", "cell_017_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="WOE binning: transformación monotónica supervisada.",
            width="stretch",
        )
with col_img2:
    img = get_notebook_image_path("02_feature_engineering", "cell_018_out_00.png")
    if img.exists():
        st.image(
            str(img),
            caption="IV ranking: selección de features por poder predictivo.",
            width="stretch",
        )

# ── Closing ──
st.markdown(
    """
**Conexión con el pipeline:** Las 60 features finales alimentan el modelo CatBoost (NB03),
que produce la PD calibrada. A su vez, esa PD entra en el conformal prediction (NB04) para
generar intervalos de incertidumbre, que luego informan la optimización de portafolio (NB08)
y las provisiones IFRS9 (NB09). **Sin features bien diseñadas, toda la cadena downstream
hereda ruido y sesgo.**
"""
)
render_page_feedback("feature_engineering")

next_page_teaser(
    "Historia de Datos",
    "Explora las distribuciones del dataset y los patrones de riesgo que motivaron esta ingeniería de variables.",
    "pages/data_story.py",
)
