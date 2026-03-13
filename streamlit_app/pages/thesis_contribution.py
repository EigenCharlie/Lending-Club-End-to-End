"""Contribución central de la tesis: predict-then-optimize con conformal prediction."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_mermaid import st_mermaid

from streamlit_app.components.context_help import methodology_dialog
from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import next_page_teaser
from streamlit_app.components.story_shell import (
    render_caveats,
    render_key_takeaway,
    render_page_header,
)
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import format_pct, load_json, load_parquet, load_runtime_status

st.title("🎯 Contribución de Tesis")
st.caption(
    "Predict-then-Optimize con Conformal Prediction: decisiones de portafolio "
    "bajo incertidumbre cuantificada con garantías matemáticas."
)
page_contract = get_page_contract("thesis_contribution")
render_page_header(page_contract)
render_key_takeaway(
    "Esta página articula el claim de tesis completo; debe leerse como puente entre la narrativa aplicada del dashboard y la defensa académica de novelty."
)
methodology_dialog(
    "Cómo leer la contribución de tesis (modo experto)",
    """
Orden sugerido:
1. Pregunta de investigación y claim.
2. Dataset como plataforma de convergencia metodológica.
3. Pipeline conceptual (calibración -> conformal -> robust optimization).
4. KPIs y trade-off de robustez.
5. Conexión a IFRS9 y reproducibilidad.
""",
    button_label="Ver mapa de lectura de la contribución",
)
st.caption(
    "Lectura de claims: las afirmaciones metodológicas se interpretan contra evidencia ejecutable "
    "del snapshot canónico actual. Fairness conformal avanzado se discute en `research_landscape.py`."
)

comparison = load_json("model_comparison")
final_metrics = comparison.get("final_test_metrics", {})
best_calibration = str(comparison.get("best_calibration", "calibración seleccionada"))
cal_report = comparison.get("calibration_selection_report", {})
cal_reason = str(cal_report.get("selection_reason", "n/a"))
cal_auc_drop_limit = float(cal_report.get("auc_drop_limit", 0.0015))

# ── Research Question ──
st.markdown(
    """
## Pregunta de investigación

> **¿Cómo tomar decisiones óptimas de asignación de portafolio crediticio cuando la
> probabilidad de default tiene incertidumbre inherente?**

El enfoque tradicional usa predicciones puntuales de PD como si fueran exactas. Esto produce
portafolios frágiles: una pequeña desviación del modelo invalida la decisión. Esta tesis propone
un pipeline que **cuantifica la incertidumbre** y la **incorpora directamente en la optimización**.
"""
)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET MOTIVATION — why Lending Club enables all seven disciplines
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("El dataset como plataforma de investigación")

st.markdown(
    """
### ¿Por qué Lending Club?

El dataset de **Lending Club** (2.93 millones de registros crudos; 1.86 millones resueltos) no se eligió por
conveniencia: es uno de los pocos datasets públicos que reúne las condiciones necesarias
para demostrar un pipeline end-to-end de riesgo de crédito con rigor académico.

- **Cobertura temporal completa**: abarca desde la post-crisis financiera de 2008 hasta el
  inicio de la pandemia COVID-19, capturando al menos un ciclo crediticio completo con
  períodos de expansión y estrés.
- **Volumen y diversidad**: 2.93M de registros crudos con 142 variables originales
  (atributos del prestatario, términos del préstamo, comportamiento de pago, variables
  macroeconómicas implícitas).
- **Reproducibilidad**: dataset público en Kaggle, permitiendo que cualquier investigador
  replique los resultados.
"""
)

st.markdown(
    """
### Siete disciplinas, un dataset

La verdadera fortaleza de este dataset es que habilita la convergencia de múltiples
disciplinas en un único flujo analítico:

**1. Machine Learning para credit scoring** — Los modelos de PD (Probability of Default)
son la piedra angular del riesgo de crédito. CatBoost aprovecha el manejo nativo de
variables categóricas y valores nulos, eliminando la necesidad de preprocesamiento manual.
La ingeniería de features con WOE (Weight of Evidence) via OptBinning transforma variables
crudas en predictores con poder discriminativo medido por Information Value.

**2. Calibración de probabilidades** — Un modelo con AUC alto no necesariamente produce
probabilidades confiables. La calibración seleccionada por validación temporal convierte
scores en probabilidades que reflejan frecuencias reales de default, requisito fundamental
antes de cualquier cuantificación de incertidumbre o toma de decisiones.

**3. Conformal Prediction** — Cuantificar incertidumbre sin asumir ninguna distribución
es el corazón de este proyecto. Split Conformal Prediction genera intervalos
[PD_low, PD_high] con garantía matemática de cobertura en muestra finita. La variante
Mondrian extiende esta garantía a nivel de subgrupo (loan grade A, B, ..., G), asegurando
que cada segmento de riesgo tenga su propia cobertura controlada.

**4. Investigación de Operaciones** — La optimización de portafolio transforma predicciones
en decisiones de asignación de capital. Pyomo formula el problema como un LP con
restricciones de presupuesto, concentración y PD máxima. HiGHS resuelve el problema en
fracciones de segundo. La innovación: usar los intervalos conformal como conjuntos de
incertidumbre box para optimización robusta, protegiendo contra el peor caso plausible.

**5. Inferencia Causal** — ¿Qué pasa si la tasa de interés sube 1 punto porcentual?
La correlación no basta para responder políticas crediticias. Double/Debiased Machine
Learning (DML) y Causal Forests (via econml y dowhy) estiman efectos causales
heterogéneos, eliminando el sesgo de selección que contamina regresiones ingenuas.

**6. Series de Tiempo** — Las tasas de default agregadas mensuales revelan patrones
estacionales y cambios de régimen. El subsistema oficial combina baselines
estadísticos (SeasonalNaive, ARIMA/ETS/Theta), challengers híbridos tipo
STL+CatBoost residual y panel global bottom-up para `grade x term`. Las
covariables exógenas solo se activan bajo contrato explícito de futuro
(`ts_future_covariates.parquet`); si ese artefacto no existe, la narrativa oficial
se mantiene univariada. Las bandas canónicas hoy se apoyan en intervalos
estadísticos y ACI/EnbPI quedan como agenda de investigación para no-exchangeability.

**7. Análisis de Supervivencia** — No solo importa *si* un préstamo incumple, sino
*cuándo*. Cox Proportional Hazards y Random Survival Forests estiman la función de
riesgo condicional al tiempo, generando las curvas de PD lifetime necesarias para el
cálculo de provisiones IFRS9 Stage 2 (deterioro significativo del crédito).
"""
)

st.success(
    "**Fábrica de insights**: estas siete disciplinas no operan en silos — convergen "
    "en un pipeline end-to-end donde la salida de una alimenta la entrada de otra. "
    "Desde un CSV crudo hasta un portafolio optimizado con garantías matemáticas de "
    "cobertura, este dataset demuestra que un enfoque integrado produce decisiones "
    "más robustas que la suma de sus componentes individuales."
)

# ── Pipeline Diagram ──
st.subheader("1) Pipeline: del modelo a la decisión robusta")

st_mermaid(
    f"""
    graph LR
        A[CatBoost PD] --> B[Calibración {best_calibration}]
        B --> C[MAPIE Mondrian<br/>Conformal Prediction]
        C --> D["[PD_low, PD_high]<br/>Intervalos con garantía"]
        D --> E[Box Uncertainty Sets]
        E --> F[Pyomo Robust<br/>Optimization + HiGHS]
        F --> G[Portafolio Óptimo<br/>Robusto]

        style A fill:#1a1a2e,stroke:#00D4AA,color:#e0e0e0
        style B fill:#1a1a2e,stroke:#00D4AA,color:#e0e0e0
        style C fill:#16213e,stroke:#FFD93D,color:#e0e0e0
        style D fill:#16213e,stroke:#FFD93D,color:#e0e0e0
        style E fill:#0f3460,stroke:#FF6B6B,color:#e0e0e0
        style F fill:#0f3460,stroke:#FF6B6B,color:#e0e0e0
        style G fill:#1a1a2e,stroke:#00D4AA,color:#e0e0e0
    """,
    height=200,
)

st.markdown(
    f"""
**Cada etapa tiene un propósito preciso:**
1. **CatBoost PD**: modelo de clasificación robusto con manejo nativo de categorías y nulos.
2. **Calibración ({best_calibration})**: convierte scores en probabilidades verdaderas
   (ECE test actual={final_metrics.get("ece", 0):.4f}).
3. **Conformal Prediction Mondrian**: genera intervalos `[PD_low, PD_high]` con garantía de
   cobertura empírica por grupo (grade), sin supuestos distribucionales.
4. **Box Uncertainty Sets**: encapsula los intervalos como conjuntos de incertidumbre para optimización.
5. **Robust Optimization**: resuelve el problema de asignación bajo el peor caso plausible dentro del
   conjunto de incertidumbre (Pyomo + HiGHS).
"""
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Calibración — por qué es imprescindible y cómo se complementa con Conformal
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2) ¿Por qué calibrar antes de cuantificar incertidumbre?")

st.markdown(
    """
Muchos modelos de ML entregan probabilidades que pueden estar **mal calibradas**.
Eso significa que el valor numérico no coincide con la frecuencia real observada.
La **calibración** ajusta ese sesgo para que la probabilidad predicha se acerque a la
frecuencia de incumplimiento en segmentos de riesgo comparables.
"""
)

col_cal_left, col_cal_right = st.columns(2)
with col_cal_left:
    st.markdown(
        """
#### Sin calibración
```
Probabilidad estimada     = 12.0%
Frecuencia real (mismo bin) = 8.0%
```
El modelo sobreestima el riesgo en 4pp.
Si usamos 12.0% en el optimizador, seremos
innecesariamente conservadores.
"""
    )
with col_cal_right:
    st.markdown(
        f"""
#### Con calibración {best_calibration}
```
Probabilidad estimada     = 12.0%
Probabilidad calibrada    = 8.2%
Frecuencia real (mismo bin) = 8.0%
```
Ahora la probabilidad calibrada refleja mejor la realidad.
El optimizador trabaja con datos honestos.
"""
    )

st.info(
    "**Regla de oro:** La calibración corrige el *nivel* de las probabilidades "
    "(que sean honestas). Conformal prediction genera *intervalos* alrededor de esas "
    "probabilidades (que capturen la incertidumbre). Son complementarios, no sustitutos."
)

# ── Platt vs Isotonic vs Venn-Abers ──
with st.expander("Métodos de calibración: Platt, Isotonic y Venn-Abers"):
    st.markdown(
        f"""
### ¿Qué hace cada método?

| Método | Mecanismo | Ventajas | Limitaciones |
|--------|-----------|----------|-------------|
| **Platt Scaling** | Ajusta una función sigmoide: `P(y=1) = 1/(1 + exp(-az - b))` | Suave, generalizable, pocos parámetros (a, b) | Asume relación sigmoid entre score y probabilidad |
| **Isotonic Regression** | Ajuste no-paramétrico monotónico (step function) | Flexible, no asume forma funcional | Riesgo de overfitting con calibración sets pequeños |
| **Venn-Abers** | Genera **dos** calibraciones isotonic (una asumiendo y=0, otra y=1) y reporta un **intervalo** [p_low, p_high] | Produce intervalos con garantía de validez | Más complejo, computacionalmente costoso, menos conocido |

### ¿Qué quedó seleccionado en este run?

- Método ganador: **{best_calibration}**
- Regla de selección: priorizar menor Brier, luego menor ECE, luego estabilidad fold-to-fold.
- Restricción aplicada: degradación media de AUC <= **{cal_auc_drop_limit:.4f}**.
- Motivo registrado: `{cal_reason}`.

### ¿Qué pasaría con Venn-Abers en vez de Platt?

Venn-Abers es fascinante porque ya produce un **intervalo de probabilidad** [p_low, p_high]
como output, no un punto. La pregunta natural es: ¿necesitamos Conformal Prediction si
Venn-Abers ya da intervalos?

La respuesta es **sí**, por tres razones:

| Dimensión | Venn-Abers | Conformal Prediction |
|-----------|-----------|---------------------|
| **Tipo de garantía** | Validez probabilística (las probabilidades son "válidas") | Cobertura empírica (el % de veces que el valor real cae en el intervalo es controlable) |
| **Ancho del intervalo** | Determinado por la discrepancia entre dos calibradores isotonic | Determinado por el nivel de confianza elegido (90%, 95%) |
| **Control por grupo** | No tiene variante Mondrian estándar | **Mondrian** permite garantías por subgrupo (grade A, B, C...) |
| **Interpretación** | "La PD verdadera está entre p_low y p_high" | "Con 90% de probabilidad, el evento observado cae en este rango" |

**Conclusión**: Venn-Abers y Conformal resuelven problemas **diferentes**:
- Venn-Abers calibra de forma conservadora (intervalos de *probabilidad*).
- Conformal cuantifica incertidumbre de *predicción* con cobertura controlable.

En este proyecto, el calibrador seleccionado ({best_calibration}) ajusta el punto central (PD honesta) y Conformal genera el
intervalo operativo [PD_low, PD_high] que consume el optimizador. Si usáramos Venn-Abers,
tendríamos intervalos de calibración, pero **no** la garantía de cobertura marginal finita
que ofrece Conformal Prediction y que es esencial para la robustez del optimizador.
"""
    )

# ── Calibration → Conformal flow ──
with st.expander("¿Cómo se complementan calibración y conformal prediction?"):
    st.markdown(
        f"""
### El flujo completo, paso a paso

```
Paso 1: CatBoost produce un score bruto
        → score = 0.15 (no es una probabilidad confiable)

Paso 2: Calibración ({best_calibration}) ajusta el score
        → PD_point = 0.12 (ahora sí es una probabilidad honesta)

Paso 3: Conformal Prediction genera el intervalo
        → [PD_low, PD_high] = [0.06, 0.18] con 90% de garantía

Paso 4: El optimizador usa PD_high = 0.18 como peor caso
        → Decisión robusta que soporta la incertidumbre real
```

### ¿Por qué no saltar la calibración?

Si alimentamos Conformal Prediction con scores **no calibrados**:
- Los intervalos serán **técnicamente válidos** (la cobertura se cumple).
- Pero el **centro del intervalo** estará sesgado.
- Un PD_point de 0.15 cuando la realidad es 0.08 produce un intervalo
  desplazado: [0.09, 0.21] en vez de [0.02, 0.14].
- El optimizador tomaría decisiones demasiado conservadoras.

**La calibración centra el intervalo; conformal controla su ancho.**

### ¿Y si no usamos Conformal Prediction?

Sin Conformal, solo tenemos PD_point = 0.12. El optimizador asume que es exacta.
Si el modelo tiene un error de ±5pp (común en riesgo de crédito), el portafolio
"óptimo" puede ser subóptimo o incluso peligroso.

Con Conformal: el optimizador sabe que la PD puede ser hasta 0.18 y se protege.
La pérdida de retorno (precio de robustez) es el costo de esa protección.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Comparison table — Why Conformal?
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3) ¿Por qué Conformal Prediction y no otro método?")

comparison_data = pd.DataFrame(
    [
        {
            "Método": "Punto (sin intervalos)",
            "Garantía de cobertura": "Ninguna",
            "Supuestos": "Modelo perfecto",
            "Muestra finita": "No",
            "Veredicto": "Frágil — ignora incertidumbre",
        },
        {
            "Método": "Bootstrap",
            "Garantía de cobertura": "Asintótica",
            "Supuestos": "n → ∞",
            "Muestra finita": "Aproximada",
            "Veredicto": "Razonable pero sin garantía formal",
        },
        {
            "Método": "Bayesiano (credible intervals)",
            "Garantía de cobertura": "Condicional al prior",
            "Supuestos": "Distribución correcta",
            "Muestra finita": "Depende del prior",
            "Veredicto": "Fuerte si el prior es correcto",
        },
        {
            "Método": "Venn-Abers",
            "Garantía de cobertura": "Validez probabilística",
            "Supuestos": "Exchangeability",
            "Muestra finita": "Sí",
            "Veredicto": "Calibración conservadora, no cobertura operativa",
        },
        {
            "Método": "Conformal Prediction",
            "Garantía de cobertura": "Marginal exacta",
            "Supuestos": "Exchangeability",
            "Muestra finita": "Sí (matemática)",
            "Veredicto": "Distribución-libre + cobertura controlable",
        },
    ]
)
st.dataframe(comparison_data, width="stretch", hide_index=True)

st.success(
    "**Ventaja clave**: Conformal Prediction es el único método que ofrece garantías de cobertura "
    "en muestra finita sin asumir una distribución específica. Esto es exactamente lo que necesita "
    "un optimizador que debe ser robusto ante errores del modelo."
)

with st.expander("¿Qué es Conformal Prediction? (explicación desde cero)"):
    st.markdown(
        """
### Para quienes nunca han oído de Conformal Prediction

Imagina que tienes un modelo que predice la temperatura de mañana: **25°C**.
¿Pero qué tan seguro es? Podría ser 23°C o 28°C.

**Conformal Prediction** dice: "No sé la distribución del error, pero puedo mirar
los errores pasados del modelo en datos que ya conozco la respuesta, y construir
un intervalo que contenga la respuesta correcta el 90% de las veces."

**¿Cómo funciona?**
1. Entrena un modelo normal (CatBoost, cualquiera).
2. En un set de calibración separado, calcula los errores del modelo.
3. Toma el cuantil 90% de esos errores. Ese es el "radio" del intervalo.
4. Para datos nuevos: predicción ± radio = intervalo con 90% de cobertura.

**¿Por qué es revolucionario?**
- No asume que los errores son normales, ni simétricos, ni homogéneos.
- Funciona con **cualquier** modelo (neural nets, árboles, regresión).
- La garantía de cobertura es **matemática**, no empírica ni asintótica.
- Solo requiere que los datos sean **exchangeable** (intercambiables).

**¿Qué agrega Mondrian?**
El conformal básico da un solo radio para todos. Pero un préstamo Grade A
tiene menos incertidumbre que uno Grade G. **Mondrian** calcula un radio
*diferente* por grupo, dando intervalos más justos y operativos.

### Referencia académica
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*.
- Angelopoulos & Bates (2023). *Conformal Prediction: A Gentle Introduction*.
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Key Results
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4) Resultados de impacto")

policy = load_json("conformal_policy_status", directory="models")
robust = load_parquet("portfolio_robustness_summary")
ifrs9 = load_parquet("ifrs9_scenario_summary")
runtime_status = load_runtime_status()
test_suite_total = int(runtime_status.get("test_suite_total", 0) or 0)
test_suite_label = str(test_suite_total) if test_suite_total > 0 else "N/D"

# IFRS9 uses 'total_ecl' column
baseline_ecl = (
    float(ifrs9.loc[ifrs9["scenario"] == "baseline", "total_ecl"].iloc[0])
    if "baseline" in ifrs9["scenario"].values
    else 0
)
severe_ecl = (
    float(ifrs9.loc[ifrs9["scenario"] == "severe", "total_ecl"].iloc[0])
    if "severe" in ifrs9["scenario"].values
    else 0
)

# Robustness summary uses 'risk_tolerance', 'baseline_nonrobust_return', 'best_robust_return'
tol_col = "risk_tolerance" if "risk_tolerance" in robust.columns else "tolerance"
ret_nonrobust_col = (
    "baseline_nonrobust_return"
    if "baseline_nonrobust_return" in robust.columns
    else "nonrobust_return"
)
ret_robust_col = "best_robust_return" if "best_robust_return" in robust.columns else "robust_return"

tol_10 = robust[robust[tol_col] == 0.10] if tol_col in robust.columns else pd.DataFrame()
if not tol_10.empty:
    robust_return = float(tol_10[ret_robust_col].iloc[0])
    nonrobust_return = float(tol_10[ret_nonrobust_col].iloc[0])
    price_of_robustness = nonrobust_return - robust_return
else:
    robust_return = 0
    nonrobust_return = 0
    price_of_robustness = 0

kpi_row(
    [
        {"label": "Cobertura 90% (Mondrian)", "value": format_pct(policy.get("coverage_90", 0))},
        {
            "label": "Ancho promedio 90%",
            "value": f"{float(policy.get('avg_width_90', 0.0)):.3f}",
        },
        {"label": "Retorno robusto (tol=10%)", "value": f"${robust_return:,.0f}"},
        {"label": "Precio de robustez", "value": f"${price_of_robustness:,.0f}"},
        {"label": "ECL baseline", "value": f"${baseline_ecl / 1e6:,.1f}M"},
        {"label": "ECL severo", "value": f"${severe_ecl / 1e6:,.1f}M"},
    ],
    n_cols=3,
)

st.markdown(
    f"""
**Lectura de los KPIs:**
- **Cobertura 90%**: en el snapshot canónico actual, la cobertura observada es
  **{format_pct(policy.get("coverage_90", 0))}** frente a meta de 90%.
- **Ancho promedio 90%**: resume el nivel de conservadurismo de los intervalos conformales.
- **Precio de robustez**: la diferencia de retorno entre asumir PD exacta vs usar el peor caso
  conformal. Es el costo de la protección.
- **ECL baseline vs severo**: cómo cambian las provisiones regulatorias bajo estrés.
  El uplift actual es **{format_pct((severe_ecl / baseline_ecl - 1) if baseline_ecl else 0)}**,
  mostrando sensibilidad material a escenarios adversos.
"""
)

# ── Robustness Trade-off ──
st.subheader("5) Trade-off: retorno vs robustez")

if not robust.empty and tol_col in robust.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=robust[tol_col],
            y=robust[ret_nonrobust_col],
            mode="lines+markers",
            name="Sin robustez (PD puntual)",
            line={"color": "#FF6B6B", "width": 2.5},
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=robust[tol_col],
            y=robust[ret_robust_col],
            mode="lines+markers",
            name="Con robustez (PD_high conformal)",
            line={"color": "#00D4AA", "width": 2.5},
            marker={"size": 8},
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Frontera de robustez: retorno esperado vs tolerancia de riesgo",
        xaxis_title="Tolerancia de PD máxima del portafolio",
        yaxis_title="Retorno esperado ($)",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "La brecha entre las curvas es el **precio de robustez**: lo que cuesta protegerse "
        "contra el peor caso plausible. Una brecha pequeña indica que la protección es barata."
    )

st.markdown(
    """
**Lectura del trade-off:**
- La curva roja asume que la PD puntual es exacta. Maximiza retorno pero es **frágil**.
- La curva verde usa `PD_high` (límite superior conformal) como constraint. Sacrifica retorno
  pero **garantiza** que el portafolio soporta el peor caso con 90% de probabilidad.
- La diferencia entre ambas es el **precio de la robustez**: cuánto cuesta la protección.
- **Este trade-off es la contribución operativa central de la tesis.**
"""
)

# ── IFRS9 Connection ──
st.subheader("6) Conexión con IFRS9")
st.markdown(
    """
Los intervalos conformal no solo alimentan la optimización — también mejoran la gobernanza regulatoria:

| Uso en IFRS9 | Descripción |
|---|---|
| **ECL por rango** | Provisionar con `PD_high` en vez de `PD_point` para lectura prudencial |
| **SICR signal** | Ancho del intervalo (`PD_high - PD_point`) como señal adicional de deterioro significativo |
| **Stress testing** | Escenarios con multiplicadores derivados de bandas de pronóstico temporal |
| **Gobernanza** | Monitoreo de cobertura y backtesting temporal documenta calidad de incertidumbre ante auditoría |
"""
)

# ── Reproducibility ──
st.subheader("7) Reproducibilidad")
st.code(
    """
# Clonar y configurar
git clone <repo> && cd Lending-Club-End-to-End

# Instalar dependencias
uv sync --extra dev

# Ejecutar rebuild canónico
uv run python scripts/run_canonical_rebuild.py

# Verificar tests
uv run pytest -x

# Lanzar dashboard
uv run streamlit run streamlit_app/app.py
""",
    language="bash",
)

st.markdown(
    f"""
**Stack tecnológico**: Python 3.12 · CatBoost · MAPIE 1.3 · Pyomo + HiGHS · DuckDB · dbt · Feast · Streamlit

**{test_suite_label} tests** validan datos, features, modelos, conformal, IFRS9, optimización, MLflow, Streamlit e integración end-to-end.
"""
)
render_caveats(
    [
        "El claim de tesis integra múltiples módulos; la fortaleza depende de la calidad de cada componente y su alineación temporal.",
        "La generalización a otras carteras requiere recalibración y adaptación institucional.",
    ]
)

next_page_teaser(
    "Visión End-to-End",
    "Narrativa completa del pipeline con métricas detalladas por componente.",
    "pages/thesis_end_to_end.py",
)
