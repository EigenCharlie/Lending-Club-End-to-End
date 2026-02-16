"""Atlas de evidencia: lectura guiada de figuras originales de notebooks."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.utils import get_notebook_image_path, load_notebook_image_manifest


def _render_figure_panel(stem: str, figure_meta: dict, idx: int) -> None:
    img_path = get_notebook_image_path(stem, figure_meta["file"])
    if not img_path.exists():
        return
    with st.container(border=True):
        st.markdown(f"#### Figura {idx}. {figure_meta['title']}")
        col_img, col_txt = st.columns([1.25, 1.0], gap="large")
        with col_img:
            st.image(str(img_path), caption=figure_meta["caption"], width=820)
        with col_txt:
            st.markdown(f"**Propósito analítico:** {figure_meta['purpose']}")
            st.markdown(f"**Insight principal:** {figure_meta['insight']}")
            st.markdown(f"**Interpretación de negocio:** {figure_meta['business']}")
            st.markdown(f"**Conexión metodológica:** {figure_meta['connection']}")


st.title("📚 Atlas de Evidencia de Notebooks")
st.caption(
    "Este atlas funciona como capítulos de un libro dinámico: cada notebook tiene un hilo conductor, "
    "preguntas explícitas, hallazgos clave y conexión con el siguiente módulo del pipeline."
)
st.markdown(
    """
No se trata de mostrar muchas imágenes sin contexto, sino de explicar **por qué cada evidencia importa**.
El orden sigue la lógica real del proyecto: primero entendemos datos y sesgos de origen, luego construimos señal
predictiva, después cuantificamos incertidumbre, y finalmente traducimos eso a políticas causales, optimización de
portafolio e impacto IFRS9. Cuando una figura aparece aquí, su objetivo es dejar claro qué problema resuelve y cómo
su salida alimenta al siguiente bloque.
"""
)

NOTEBOOK_META = {
    "01_eda_lending_club": {
        "titulo": "Notebook 01 — EDA",
        "objetivo": "Entender estructura del dataset, gradientes de riesgo y señales de drift temporal.",
        "aporte": "Define decisiones de limpieza, segmentación y validación out-of-time para todo el pipeline.",
        "story": (
            "Este capítulo demuestra que el default no es ruido aleatorio: sigue patrones por grade, plazo y nivel "
            "de tasa. Esa constatación justifica el split temporal y explica por qué en riesgo de crédito no basta un "
            "holdout aleatorio. El EDA también delimita qué variables pueden inducir leakage y cuáles sí representan "
            "información disponible al momento de originar el crédito."
        ),
        "closing": (
            "Conclusión del capítulo: el problema tiene estructura económica observable y dinámica temporal; por eso la "
            "arquitectura metodológica parte de limpieza rigurosa y validación OOT."
        ),
        "curated": [
            {
                "file": "cell_024_out_00.png",
                "title": "Default por bucket de tasa y volumen",
                "caption": "Figura original del Notebook 01.",
                "purpose": "Relacionar precio del crédito con tasa de incumplimiento y masa de originación.",
                "insight": "La tasa de default aumenta de forma monótona al subir bucket de interés.",
                "business": "Permite justificar pricing basado en riesgo y límites de exposición por tramo de tasa.",
                "connection": "Prepara las variables de bucket e interacciones usadas en feature engineering.",
            },
            {
                "file": "cell_025_out_00.png",
                "title": "Heatmap grade x term",
                "caption": "Figura original del Notebook 01.",
                "purpose": "Visualizar interacciones entre calidad crediticia y horizonte contractual.",
                "insight": "Plazos largos amplifican el riesgo especialmente en grades bajos.",
                "business": "No todo crecimiento de volumen es rentable: el mix grade/plazo cambia pérdidas esperadas.",
                "connection": "Motiva variables cruzadas como `int_rate_bucket__grade` y reglas de segmentación.",
            },
        ],
    },
    "02_feature_engineering": {
        "titulo": "Notebook 02 — Feature Engineering",
        "objetivo": "Construir variables interpretables y medir poder predictivo (WOE/IV).",
        "aporte": "Entrega datasets modelables y un contrato de features trazable.",
        "story": (
            "En este capítulo se convierte un universo inicial de variables en señales accionables de riesgo. "
            "La técnica WOE/IV no se usa solo por tradición de scorecards: se usa para cuantificar poder de separación, "
            "evitar variables irrelevantes y mantener una narrativa interpretable para negocio y control interno."
        ),
        "closing": (
            "Conclusión del capítulo: la calidad del modelo final depende más de la ingeniería de señal que del algoritmo "
            "por sí solo."
        ),
        "curated": [
            {
                "file": "cell_008_out_00.png",
                "title": "Distribución de ratios financieros por estado",
                "caption": "Figura original del Notebook 02.",
                "purpose": "Contrastar separabilidad entre buenos y malos pagadores en variables de capacidad de pago.",
                "insight": "Ratios de carga financiera muestran desplazamiento consistente hacia zonas de mayor riesgo.",
                "business": "Ayuda a diseñar políticas de admisión con umbrales comprensibles para comité.",
                "connection": "Estos ratios terminan entre los drivers de SHAP en el modelo PD.",
            },
            {
                "file": "cell_020_out_00.png",
                "title": "WOE binning en features top",
                "caption": "Figura original del Notebook 02.",
                "purpose": "Ver dirección del riesgo por bins y verificar monotonicidad económica.",
                "insight": "Subgrade e interés concentran gran parte del poder de discriminación.",
                "business": "Permite explicar por qué ciertas variables tienen mayor peso en score y decisión.",
                "connection": "Define un espacio de features estable para entrenamiento y calibración.",
            },
            {
                "file": "cell_023_out_00.png",
                "title": "Ranking IV",
                "caption": "Figura original del Notebook 02.",
                "purpose": "Priorizar variables por aporte incremental de información.",
                "insight": "El top de IV coincide con variables de teoría crediticia clásica.",
                "business": "Reduce complejidad de modelo sin perder señal relevante.",
                "connection": "Base para selección final en CatBoost y análisis de interpretabilidad.",
            },
        ],
    },
    "03_pd_modeling": {
        "titulo": "Notebook 03 — PD Modeling",
        "objetivo": "Comparar modelos, optimizar hiperparámetros y calibrar probabilidades.",
        "aporte": "Entrega la PD calibrada que alimenta todo el pipeline downstream.",
        "story": (
            "Aquí se valida que un buen ranking no es suficiente: el proyecto exige probabilidad confiable porque esa PD "
            "se usa en IFRS9, política causal y optimización. Por eso se combina discriminación (AUC/KS) con calidad "
            "probabilística (Brier/ECE)."
        ),
        "closing": (
            "Conclusión del capítulo: CatBoost calibrado logra el mejor equilibrio entre separación y confiabilidad "
            "probabilística fuera de muestra temporal."
        ),
        "curated": [
            {
                "file": "cell_013_out_00.png",
                "title": "Optuna: trayectoria de búsqueda e importancia de hiperparámetros",
                "caption": "Figura original del Notebook 03.",
                "purpose": "Documentar convergencia y sensibilidad del tuning.",
                "insight": "El `learning_rate` domina variabilidad del desempeño validado.",
                "business": "Justifica técnica y operativamente el set final de hiperparámetros.",
                "connection": "Conecta con estabilidad del score usado en módulos posteriores.",
            },
            {
                "file": "cell_020_out_00.png",
                "title": "ROC y Precision-Recall comparado",
                "caption": "Figura original del Notebook 03.",
                "purpose": "Comparar capacidad de ranking entre baseline y variantes CatBoost.",
                "insight": "CatBoost supera consistentemente el baseline logístico en ambas curvas.",
                "business": "Mejor ordenamiento implica mejor priorización de originación y revisión.",
                "connection": "Define la PD base para conformal prediction.",
            },
            {
                "file": "cell_017_out_00.png",
                "title": "SHAP global (beeswarm + importancia)",
                "caption": "Figura original del Notebook 03.",
                "purpose": "Explicar drivers globales y dirección de impacto de cada variable.",
                "insight": "Tasa, plazo y score crediticio lideran la explicación del riesgo.",
                "business": "Mejora explicabilidad ante comité de riesgo y auditoría de modelo.",
                "connection": "Los drivers explicados se usan luego para narrativa causal y de políticas.",
            },
        ],
    },
    "04_conformal_prediction": {
        "titulo": "Notebook 04 — Conformal Prediction",
        "objetivo": "Convertir PD puntual en intervalos con cobertura empírica controlada.",
        "aporte": "Introduce incertidumbre finita como insumo directo para decisión.",
        "story": (
            "Conformal no compite con el modelo PD, lo complementa. El valor aquí es transformar una predicción puntual "
            "en un rango con garantía empírica de cobertura, incluyendo control por grupos (Mondrian). Esto evita que una "
            "métrica global oculte errores concentrados en segmentos específicos."
        ),
        "closing": (
            "Conclusión del capítulo: la incertidumbre deja de ser cualitativa y pasa a ser cuantitativa, auditable y utilizable "
            "por optimización e IFRS9."
        ),
        "curated": [
            {
                "file": "cell_018_out_00.png",
                "title": "Marginal vs Mondrian por grade",
                "caption": "Figura original del Notebook 04.",
                "purpose": "Comparar cobertura y ancho por esquema de conformalización.",
                "insight": "Mondrian mejora control por subgrupo a costa de mayor ancho promedio.",
                "business": "Permite elegir política de incertidumbre según apetito de riesgo.",
                "connection": "Anchos resultantes impactan el costo de robustez en optimización.",
            },
            {
                "file": "cell_022_out_01.png",
                "title": "Cobertura objetivo vs cobertura observada",
                "caption": "Figura original del Notebook 04.",
                "purpose": "Validar calibración empírica del intervalo frente a la meta.",
                "insight": "La cobertura sigue de cerca el objetivo entre 70%-99%.",
                "business": "Da confianza para usar intervalos en decisiones de capital y provisiones.",
                "connection": "Sustenta policy checks conformal y reglas de gobernanza.",
            },
            {
                "file": "cell_020_out_01.png",
                "title": "Ejemplos de intervalos por préstamo",
                "caption": "Figura original del Notebook 04.",
                "purpose": "Mostrar heterogeneidad de incertidumbre a nivel individual.",
                "insight": "Préstamos con PD similar pueden tener incertidumbre muy distinta.",
                "business": "Evita tratar casos ambiguos igual que casos bien identificados.",
                "connection": "El optimizador usa esta diferencia en peor-caso (`PD_high`).",
            },
        ],
    },
    "05_time_series_forecasting": {
        "titulo": "Notebook 05 — Forecasting",
        "objetivo": "Proyectar tasa agregada de default y su incertidumbre temporal.",
        "aporte": "Añade componente forward-looking para lectura IFRS9 y planeación.",
        "story": (
            "Este capítulo cambia la escala: pasamos de riesgo por préstamo a dinámica agregada del portafolio. "
            "La comparación multi-modelo y el backtesting temporal evitan seleccionar un pronóstico por conveniencia visual."
        ),
        "closing": (
            "Conclusión del capítulo: el pronóstico aporta visión de ciclo y fortalece la discusión de escenarios macro."
        ),
        "curated": [
            {
                "file": "cell_017_out_01.png",
                "title": "Comparativo de modelos por MAE/RMSE",
                "caption": "Figura original del Notebook 05.",
                "purpose": "Evaluar error temporal fuera de muestra de modelos candidatos.",
                "insight": "No siempre el mejor modelo por MAE coincide con RMSE.",
                "business": "Permite elegir modelo según costo de errores grandes vs promedio.",
                "connection": "Ese output alimenta escenarios IFRS9 en la página regulatoria.",
            },
            {
                "file": "cell_006_out_01.png",
                "title": "Diagnóstico de nivel y variabilidad temporal",
                "caption": "Figura original del Notebook 05.",
                "purpose": "Observar cambios de régimen y estabilidad de la serie.",
                "insight": "La no estacionariedad explica por qué se requieren ventanas temporales y recalibración.",
                "business": "Refuerza monitoreo continuo en vez de decisiones estáticas de riesgo.",
                "connection": "Conecta con gobernanza temporal y alertas de drift.",
            },
        ],
    },
    "06_survival_analysis": {
        "titulo": "Notebook 06 — Survival",
        "objetivo": "Estimar cuándo ocurre el default, no solo si ocurre.",
        "aporte": "Aporta curvas lifetime para lectura de horizonte y provisión.",
        "story": (
            "En crédito, el tiempo al evento cambia completamente la gestión del riesgo. Dos carteras con misma PD anual "
            "pueden tener impacto de caja distinto si el deterioro llega antes o después. Por eso se incorporan Cox y RSF "
            "como capa temporal complementaria."
        ),
        "closing": (
            "Conclusión del capítulo: la dimensión temporal mejora decisiones de seguimiento y fortalece IFRS9 lifetime."
        ),
        "curated": [
            {
                "file": "cell_025_out_00.png",
                "title": "Curvas de PD lifetime por grade (RSF)",
                "caption": "Figura original del Notebook 06.",
                "purpose": "Comparar acumulación de riesgo por horizonte y calificación.",
                "insight": "Grades bajos acumulan riesgo más rápido y a niveles más altos.",
                "business": "Permite dimensionar provisión y estrategia de cobranza por perfil.",
                "connection": "Se integra con stage y ECL en IFRS9.",
            },
            {
                "file": "cell_009_out_00.png",
                "title": "Kaplan-Meier por segmento",
                "caption": "Figura original del Notebook 06.",
                "purpose": "Visualizar supervivencia del préstamo en distintos grupos.",
                "insight": "La separación entre curvas confirma heterogeneidad de tiempo-a-default.",
                "business": "Ayuda a priorizar monitoreo y acciones tempranas por cohortes de riesgo.",
                "connection": "Complementa lectura causal: qué grupo intervenir y con qué urgencia.",
            },
        ],
    },
    "07_causal_inference": {
        "titulo": "Notebook 07 — Causalidad",
        "objetivo": "Estimar efectos de intervención y evitar decisiones por correlación espuria.",
        "aporte": "Convierte análisis en política accionable con valor económico esperado.",
        "story": (
            "Este capítulo responde una pregunta crítica: si modifico una palanca (por ejemplo tasa), ¿cambia realmente "
            "la probabilidad de default o solo estoy viendo sesgo de selección? El uso de DoWhy + EconML permite separar "
            "señal causal de correlación ingenua y construir reglas con evidencia cuantitativa."
        ),
        "closing": (
            "Conclusión del capítulo: la causalidad no reemplaza al score; lo complementa para decidir intervenciones con "
            "impacto y costo medible."
        ),
        "curated": [
            {
                "file": "cell_020_out_01.png",
                "title": "Correlación naive vs estimadores causales",
                "caption": "Figura original del Notebook 07.",
                "purpose": "Mostrar sesgo de estimador ingenuo frente a modelos causales ajustados.",
                "insight": "La regresión naive sobreestima el efecto de tasa respecto al efecto causal.",
                "business": "Evita políticas comerciales que parecen rentables en correlación pero no en causalidad.",
                "connection": "Define el set de reglas candidatas para política de intervención.",
            },
            {
                "file": "cell_026_out_01.png",
                "title": "Sensibilidad por grupo y recomendación de tasa",
                "caption": "Figura original del Notebook 07.",
                "purpose": "Identificar dónde intervenir tiene mayor retorno de riesgo.",
                "insight": "No todos los segmentos responden igual a un ajuste de tasa.",
                "business": "Permite focalizar descuentos/acciones donde generan más valor neto.",
                "connection": "La regla causal seleccionada se integra con optimización de portafolio.",
            },
        ],
    },
    "08_portfolio_optimization": {
        "titulo": "Notebook 08 — Optimización",
        "objetivo": "Decidir asignación de capital bajo restricciones de riesgo e incertidumbre.",
        "aporte": "Cuantifica explícitamente el trade-off retorno vs robustez.",
        "story": (
            "Aquí se materializa la tesis de investigación de operaciones: predicción e incertidumbre se transforman en una "
            "decisión de cartera. La solución no robusta maximiza retorno esperado puntual; la robusta sacrifica parte de ese "
            "retorno para protegerse frente a errores de estimación representados por intervalos conformal."
        ),
        "closing": (
            "Conclusión del capítulo: el `price of robustness` deja de ser abstracto y se convierte en métrica de política."
        ),
        "curated": [
            {
                "file": "cell_021_out_54.png",
                "title": "Frontera eficiente con punto robusto y no robusto",
                "caption": "Figura original del Notebook 08.",
                "purpose": "Visualizar alternativas factibles de riesgo-retorno.",
                "insight": "La solución robusta se ubica en menor retorno, pero con control de downside.",
                "business": "Hace explícita la prima por prudencia que acepta el comité de riesgo.",
                "connection": "Se conecta con resultados IFRS9 y narrativa ejecutiva de capital.",
            },
            {
                "file": "cell_015_out_14.png",
                "title": "Sensibilidad retorno/aprobaciones al límite PD",
                "caption": "Figura original del Notebook 08.",
                "purpose": "Cuantificar cuánto cambia la política al mover tolerancia de riesgo.",
                "insight": "Pequeños cambios en PD cap pueden alterar fuertemente volumen financiado.",
                "business": "Útil para definir apetito de riesgo en términos operativos y comerciales.",
                "connection": "Informa parametrización de política robusta por escenarios.",
            },
        ],
    },
    "09_end_to_end_pipeline": {
        "titulo": "Notebook 09 — Integración E2E",
        "objetivo": "Unificar score, incertidumbre, causalidad, optimización e IFRS9 en una sola corrida.",
        "aporte": "Demuestra trazabilidad completa desde datos hasta decisión.",
        "story": (
            "Este cierre integra todo el recorrido. Ya no vemos módulos aislados sino una cadena reproducible donde la salida "
            "de una técnica alimenta a la siguiente: PD calibrada -> intervalo conformal -> optimización robusta e IFRS9, con "
            "lectura causal para acciones de política. Es el punto donde el proyecto deja de ser un conjunto de notebooks y se "
            "convierte en un sistema analítico coherente."
        ),
        "closing": (
            "Conclusión del capítulo: el valor diferencial está en la integración metodológica completa y auditable."
        ),
        "curated": [
            {
                "file": "cell_018_out_00.png",
                "title": "Dashboard integrado E2E",
                "caption": "Figura original del Notebook 09.",
                "purpose": "Concentrar en un panel único métricas de score, incertidumbre y asignación.",
                "insight": "La lectura conjunta permite detectar tensiones entre retorno, cobertura y provisión.",
                "business": "Facilita conversación ejecutiva con una sola fuente visual.",
                "connection": "Se replica en la app Streamlit como capa de comunicación final.",
            },
            {
                "file": "cell_009_out_02.png",
                "title": "Staging IFRS9 y rango ECL",
                "caption": "Figura original del Notebook 09.",
                "purpose": "Mostrar cómo incertidumbre de PD se traduce a rango de provisión.",
                "insight": "La diferencia entre escenario puntual y conservador es material.",
                "business": "Permite planificar reservas con visión prudencial y no solo puntual.",
                "connection": "Cierra el vínculo entre conformal y cumplimiento regulatorio.",
            },
        ],
    },
}

manifest = load_notebook_image_manifest()
tabs = st.tabs([NOTEBOOK_META[k]["titulo"] for k in NOTEBOOK_META])

for tab, stem in zip(tabs, NOTEBOOK_META, strict=False):
    meta = NOTEBOOK_META[stem]
    with tab:
        st.markdown(f"### {meta['titulo']}")
        st.markdown(f"**Objetivo del capítulo:** {meta['objetivo']}")
        st.markdown(f"**Contribución al pipeline:** {meta['aporte']}")
        st.markdown(meta["story"])
        for idx, figure_meta in enumerate(meta["curated"], start=1):
            _render_figure_panel(stem, figure_meta, idx)
        st.markdown(f"**Cierre del capítulo:** {meta['closing']}")

        with st.expander("Galería completa extraída de este notebook (miniaturas)"):
            rows = [
                row for row in manifest if row.get("notebook_stem") == stem and row.get("image_path", "").endswith(".png")
            ]
            rows = sorted(rows, key=lambda r: (r.get("cell_index", 0), r.get("output_index", 0)))
            cols = st.columns(3)
            for idx, row in enumerate(rows):
                p = Path(row["image_path"])
                abs_path = Path.cwd() / p
                if abs_path.exists():
                    cols[idx % 3].image(
                        str(abs_path),
                        caption=f"{p.name} | cell {row.get('cell_index')}",
                        width=360,
                    )

st.markdown(
    """
Como lectura final del atlas, la evidencia visual no busca reemplazar el análisis cuantitativo, sino hacerlo más verificable.
Cada figura documenta una decisión metodológica concreta y, en conjunto, demuestra que el pipeline evolucionó como una cadena
coherente de hipótesis, validación y traducción a impacto de negocio/regulatorio.
"""
)

next_page_teaser(
    "Visión End-to-End",
    "Volver al hilo principal y conectar la evidencia visual con la narrativa completa del proyecto.",
    "pages/thesis_end_to_end.py",
)
