"""Anexo: RAPIDS GPU benchmark — aceleración GPU para riesgo de crédito."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.components.metric_cards import kpi_row
from streamlit_app.components.narrative import storytelling_intro
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import download_table

# ---------------------------------------------------------------------------
# Data loading helpers — reads from reports/gpu_benchmark/
# Supports both parquet (new) and CSV (legacy) artifacts.
# ---------------------------------------------------------------------------

GPU_BENCH_DIR = Path(__file__).resolve().parents[2] / "reports" / "gpu_benchmark"

# Consistent color palette
COLORS = {
    "pandas_cpu": "#E45756",
    "pandas_cudf": "#F58518",
    "polars_cpu": "#4C78A8",
    "polars_cudf": "#72B7B2",
    "duckdb": "#54A24B",
    "sklearn_cpu": "#A8B5C4",
    "cuml_gpu": "#0B5ED7",
    "networkx_cpu": "#A8B5C4",
    "cugraph_gpu": "#0B5ED7",
    "networkx_cugraph_backend": "#198754",
    "scipy_highs_cpu": "#A8B5C4",
    "scipy_milp_cpu": "#6F42C1",
    "cuopt_gpu": "#0B5ED7",
    "numpy_cpu": "#A8B5C4",
    "scipy_cpu": "#A8B5C4",
    "cupy_gpu": "#0B5ED7",
}


@st.cache_data(ttl=300)
def _load_bench(name: str) -> pd.DataFrame:
    """Load benchmark artifact (try parquet first, then CSV)."""
    pq = GPU_BENCH_DIR / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = GPU_BENCH_DIR / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_meta() -> dict:
    """Load benchmark metadata JSON."""
    import json

    path = GPU_BENCH_DIR / "gpu_bench_meta.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _safe_float(val: object, default: float = 0.0) -> float:
    """Safely convert to float, returning default for NaN/None."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("⚡ Benchmark RAPIDS GPU")
st.caption(
    "Evaluación exhaustiva de aceleración GPU (NVIDIA RAPIDS 26.02) sobre los "
    "datos reales del proyecto Lending Club — Side project independiente."
)

storytelling_intro(
    page_goal=(
        "Medir el valor práctico de aceleración GPU (NVIDIA RAPIDS) sobre los datos reales "
        "del proyecto Lending Club (1.86M préstamos, 110 columnas)."
    ),
    business_value=(
        "Identificar qué componentes del pipeline de riesgo se benefician de GPU "
        "y cuáles no justifican la complejidad, para decisiones de infraestructura."
    ),
    key_decision=(
        "Decidir qué librerías usar para cada tarea: procesamiento de datos, "
        "machine learning, grafos, optimización y cómputo numérico."
    ),
    how_to_read=[
        "Cada sección compara backends CPU vs GPU para una tarea específica.",
        "Los tiempos son medianas de múltiples ejecuciones con warmup.",
        "Las barras muestran tiempos absolutos — menor es mejor.",
        "Los quality gates validan que GPU produce resultados correctos.",
    ],
)

# ── Load all artifacts ──
cudf_bench = _load_bench("cudf_polars_benchmark")
cuml_bench = _load_bench("cuml_benchmark")
cugraph_bench = _load_bench("cugraph_benchmark")
cuopt_bench = _load_bench("cuopt_benchmark")
cupy_bench = _load_bench("cupy_benchmark")
scaling = _load_bench("gpu_bench_scaling")
meta = _load_meta()

# ── Compute KPIs from available data ──
has_any_data = not cudf_bench.empty or not cuml_bench.empty

sections_tested = sum(1 for df in [cudf_bench, cuml_bench, cugraph_bench, cuopt_bench, cupy_bench] if not df.empty)

# Best speedup across all sections
all_speedups: list[float] = []
if not cudf_bench.empty and "speedup_vs_pandas_cpu" in cudf_bench.columns:
    ok = cudf_bench[cudf_bench["status"] == "ok"] if "status" in cudf_bench.columns else cudf_bench
    all_speedups.extend(ok["speedup_vs_pandas_cpu"].dropna().tolist())
if not cuml_bench.empty and "fit_speedup_vs_cpu" in cuml_bench.columns:
    all_speedups.extend(cuml_bench["fit_speedup_vs_cpu"].dropna().tolist())

best_speedup = f"{max(all_speedups):.1f}x" if all_speedups else "N/D"

# Total algorithms benchmarked
n_algos = 0
for df, col in [(cudf_bench, "mode"), (cuml_bench, "task"), (cugraph_bench, "task"), (cuopt_bench, "task"), (cupy_bench, "task")]:
    if not df.empty and col in df.columns:
        n_algos += df[col].nunique()

kpi_row(
    [
        {"label": "Mejor speedup", "value": best_speedup},
        {"label": "Secciones", "value": f"{sections_tested}/5"},
        {"label": "Métodos evaluados", "value": str(n_algos)},
        {
            "label": "Hardware",
            "value": meta.get("hardware", {}).get("gpu", "RTX 3080") if meta else "RTX 3080",
        },
    ],
    n_cols=4,
)

if not has_any_data:
    st.warning(
        "No se encontraron artefactos de benchmark en `reports/gpu_benchmark/`. "
        "Ejecuta el notebook RAPIDS para generar los resultados."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1: DataFrame Processing — pandas vs cuDF vs Polars vs DuckDB
# ══════════════════════════════════════════════════════════════════════════════

st.header("1) Procesamiento de DataFrames")

st.markdown(
    """
**Workload**: lectura de parquet (1.35M filas, 110 columnas) → selección de columnas →
parsing de strings (int_rate, term) → filtrado multi-condición → groupby multi-clave →
join → sort → head(5000). Este pipeline replica operaciones típicas de
`build_datasets.py` y `prepare_dataset.py` del proyecto.

**Backends evaluados**:
| Backend | Descripción | Tipo |
|---------|-------------|------|
| `pandas_cpu` | pandas vanilla (baseline) | CPU |
| `pandas_cudf` | pandas via `python -m cudf.pandas` (zero-code-change) | GPU |
| `polars_cpu` | Polars lazy collect | CPU |
| `polars_cudf` | Polars lazy con `pl.GPUEngine()` | GPU |
| `duckdb` | SQL in-process (motor analítico columnar) | CPU |
"""
)

if not cudf_bench.empty:
    ok = cudf_bench[cudf_bench["status"] == "ok"].copy() if "status" in cudf_bench.columns else cudf_bench.copy()

    if "mode" in ok.columns and "median_seconds" in ok.columns:
        chart_df = ok[["mode", "median_seconds"]].dropna().copy()
        chart_df = chart_df.sort_values("median_seconds", ascending=False)

        # Compute speedup vs each backend for the comparison table
        backends = chart_df.set_index("mode")["median_seconds"].to_dict()
        pandas_time = backends.get("pandas_cpu", 1.0)

        # ── Chart 1: Absolute times (all backends) ──
        chart_df["color"] = chart_df["mode"].map(COLORS).fillna("#999999")

        fig = px.bar(
            chart_df,
            y="mode",
            x="median_seconds",
            orientation="h",
            labels={"mode": "Backend", "median_seconds": "Tiempo mediano (s)"},
            title="Tiempo de ejecución por backend (menor = mejor)",
            color="mode",
            color_discrete_map=COLORS,
        )
        for _, row in chart_df.iterrows():
            speedup = pandas_time / max(row["median_seconds"], 1e-12)
            fig.add_annotation(
                x=row["median_seconds"],
                y=row["mode"],
                text=f" {row['median_seconds']:.3f}s ({speedup:.1f}x vs pandas)",
                showarrow=False,
                xanchor="left",
                font={"size": 11},
            )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=max(280, len(chart_df) * 55), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ── Cross-comparison matrix ──
        st.subheader("Matriz de speedup cruzado")
        st.markdown(
            "Cada celda muestra cuántas veces más rápido es el backend de la **columna** "
            "respecto al backend de la **fila**. Valores > 1 indican ventaja."
        )
        modes = sorted(backends.keys())
        matrix_rows = []
        for row_mode in modes:
            row_data = {"Backend": row_mode}
            for col_mode in modes:
                if row_mode == col_mode:
                    row_data[col_mode] = "—"
                else:
                    ratio = backends[row_mode] / max(backends[col_mode], 1e-12)
                    row_data[col_mode] = f"{ratio:.1f}x"
            matrix_rows.append(row_data)
        matrix_df = pd.DataFrame(matrix_rows).set_index("Backend")
        st.dataframe(matrix_df, use_container_width=True)

        # ── Detailed results table ──
        with st.expander("Datos detallados de benchmark"):
            display_cols = ["mode", "median_seconds", "mean_seconds", "std_seconds"]
            if "speedup_vs_pandas_cpu" in ok.columns:
                display_cols.append("speedup_vs_pandas_cpu")
            if "rows_out" in ok.columns:
                display_cols.append("rows_out")
            if "consistency_pass" in ok.columns:
                display_cols.append("consistency_pass")
            show = [c for c in display_cols if c in ok.columns]
            st.dataframe(ok[show].sort_values("median_seconds"), use_container_width=True, hide_index=True)

        # ── Analysis text ──
        st.markdown(
            """
**Análisis detallado:**

- **Polars GPU** (`pl.GPUEngine()`) lidera gracias a que compila el plan lazy completo
  a operaciones cuDF en GPU, eliminando transferencias intermedias CPU↔GPU. El plan
  incluye scan, filter, groupby, join y sort en una sola pasada GPU.

- **Polars CPU** ya es significativamente más rápido que pandas porque usa un motor
  lazy columnar con ejecución paralela multi-hilo y predicados pushdown.

- **cuDF pandas accelerator** (`python -m cudf.pandas`) logra ~18x sobre pandas sin
  cambiar una sola línea de código. Intercepta llamadas a la API de pandas y las
  despacha a kernels CUDA. El overhead viene de la traducción de operaciones.

- **DuckDB** es un motor SQL analítico columnar in-process. Su ventaja principal es
  el bajo consumo de memoria (spill-to-disk) y la capacidad de procesar datasets más
  grandes que la RAM disponible — algo que ni pandas ni cuDF pueden hacer nativamente.
  [Referencia: codecentric.de benchmark](https://www.codecentric.de/en/knowledge-hub/blog/duckdb-vs-polars-performance-and-memory-with-massive-parquet-data)
  encontró que DuckDB usa solo 1.3 GB de RAM para 140 GB de datos, vs 17 GB de Polars.

- **pandas CPU** es el baseline más lento porque opera fila-por-fila en muchas
  operaciones y no paraleliza automáticamente.

**Operaciones representadas del pipeline real** (`build_datasets.py`):
- Parsing de `int_rate` (string con '%' → float)
- Parsing de `term` (regex extract → int)
- Filtrado multi-condición (loan_amnt, annual_inc, term)
- GroupBy multi-clave con agregaciones (count, sum, mean)
- Joins (left merge en grade y purpose)
"""
        )

        download_table(chart_df, "gpu_bench_dataframe_comparison.csv", "Descargar resultados")
else:
    st.info("No hay datos de benchmark cuDF/Polars/DuckDB disponibles.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Machine Learning — scikit-learn CPU vs cuML GPU
# ══════════════════════════════════════════════════════════════════════════════

st.header("2) Machine Learning (scikit-learn vs cuML)")

st.markdown(
    """
**Dataset**: `train_fe.parquet` (250K sample) con 27 features numéricos del pipeline de
feature engineering. Incluye ratios financieros (`loan_to_income`, `rev_utilization`),
scores crediticios (`fico_score`), e interacciones (`fico_x_dti`).

**Protocolo**: Cada algoritmo se entrena y predice múltiples veces con warmup.
Se reporta el **fit time** mediano y la **métrica de calidad** para validar que la
aceleración GPU no degrada los resultados.
"""
)

if not cuml_bench.empty and "task" in cuml_bench.columns:
    tasks = sorted(cuml_bench["task"].unique())

    # ── Chart per algorithm (solves the scale problem) ──
    for task_name in tasks:
        task_data = cuml_bench[cuml_bench["task"] == task_name].copy()
        if task_data.empty:
            continue

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                task_data,
                x="backend",
                y="fit_seconds",
                color="backend",
                title=f"{task_name.replace('_', ' ').title()}: Tiempo de entrenamiento",
                labels={"backend": "Backend", "fit_seconds": "Fit time (s)"},
                color_discrete_map=COLORS,
                text_auto=".3f",
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            for _, r in task_data.iterrows():
                backend = str(r.get("backend", ""))
                fit_s = _safe_float(r.get("fit_seconds"))
                metric = str(r.get("metric", ""))
                metric_val = _safe_float(r.get("metric_value"))
                speedup = _safe_float(r.get("fit_speedup_vs_cpu"))

                if "gpu" in backend:
                    if speedup > 1.0:
                        st.metric("Speedup GPU", f"{speedup:.1f}x", f"{(speedup - 1) * 100:.0f}% más rápido")
                    elif speedup > 0:
                        st.metric("Speedup GPU", f"{speedup:.2f}x", f"{(1 - speedup) * 100:.0f}% más lento", delta_color="inverse")
                    st.caption(f"**{metric}**: CPU={_safe_float(cuml_bench[(cuml_bench['task'] == task_name) & (cuml_bench['backend'] == 'sklearn_cpu')]['metric_value'].values[0]) if len(cuml_bench[(cuml_bench['task'] == task_name) & (cuml_bench['backend'] == 'sklearn_cpu')]) else 'N/D':.4f} vs GPU={metric_val:.4f}")

    # ── Analysis per algorithm ──
    st.markdown(
        """
**Análisis por algoritmo:**

- **Random Forest**: El mayor beneficio de GPU. La construcción de árboles en
  paralelo escala linealmente con CUDA cores. cuML paraleliza la evaluación de
  splits y la construcción simultánea de todos los árboles (250 estimators).

- **KMeans**: Speedup moderado. Las iteraciones de Lloyd's algorithm (asignación
  de centroides + recálculo) se paralelizan bien en GPU. El cuello de botella es
  el cómputo de distancias `O(n×k×d)` — ideal para GPU.

- **PCA**: La descomposición SVD subyacente es una operación de álgebra lineal
  densa que aprovecha las unidades tensoriales de GPU. `cuML.PCA` usa
  `cuSOLVER` internamente.

- **KNN**: El cómputo de distancias `O(n×m×d)` para k-nearest neighbors se
  paraleliza masivamente en GPU. cuML usa `FAISS`-like indices internos.

- **Logistic Regression**: Puede ser **más lento en GPU** para datasets pequeños.
  El algoritmo LBFGS requiere operaciones secuenciales (line search) y el
  overhead de transferencia CPU↔GPU domina cuando el cómputo por iteración
  es rápido. Este es un caso donde GPU **no** es la mejor opción.

**Lección clave**: No todo se beneficia de GPU. Algoritmos con iteraciones
secuenciales o datasets pequeños (< 100K filas) pueden ser más lentos en GPU
por el overhead de transferencia y lanzamiento de kernels CUDA.
"""
    )

    # ── Quality parity table ──
    quality = _load_bench("cuml_quality_checks")
    if quality.empty:
        quality = _load_bench("benchmark_quality_checks_all_sections")
        if not quality.empty and "section" in quality.columns:
            quality = quality[quality["section"] == "cuml"]

    if not quality.empty:
        with st.expander("Paridad de métricas CPU vs GPU (quality gates)"):
            st.dataframe(quality, use_container_width=True, hide_index=True)
            st.markdown(
                """
Los **quality gates** validan que la aceleración GPU no degrada la calidad del modelo.
Para cada algoritmo se define una tolerancia máxima de diferencia:
- Clasificación (AUC): tolerancia 0.010–0.025
- Clustering (silhouette): tolerancia 0.040
- PCA (explained variance): tolerancia 0.050

Si `quality_pass = True`, la implementación GPU es funcionalmente equivalente a CPU.
"""
            )

    download_table(cuml_bench, "gpu_bench_ml_comparison.csv", "Descargar resultados ML")
else:
    st.info("No hay datos de benchmark cuML disponibles.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Graph Analytics — NetworkX vs cuGraph
# ══════════════════════════════════════════════════════════════════════════════

st.header("3) Análisis de Grafos (NetworkX vs cuGraph)")

st.markdown(
    """
**Grafo**: Bipartito loan→atributo. Cada préstamo se conecta a nodos de
`grade`, `purpose`, `sub_grade` y `verification_status`. Esto produce un grafo
denso que modela relaciones implícitas entre préstamos.

**Aplicación en riesgo de crédito**: Detectar comunidades de alto riesgo,
nodos centrales (préstamos que comparten muchos atributos de riesgo), y
concentración de exposición por subgrafo.

**Backends**:
- `networkx_cpu`: NetworkX puro (Python, single-threaded)
- `cugraph_gpu`: cuGraph nativo (CUDA, paralelo)
- `networkx_cugraph_backend`: NetworkX con dispatch automático a cuGraph via `backend="cugraph"`
"""
)

if not cugraph_bench.empty and "task" in cugraph_bench.columns:
    tasks = sorted(cugraph_bench["task"].unique())

    # Check if we actually have GPU data
    has_gpu_data = any(
        "gpu" in str(b) or "cugraph" in str(b)
        for b in cugraph_bench["backend"].unique()
        if b != "networkx_cpu"
    )

    # ── Chart: grouped bar per task ──
    fig = px.bar(
        cugraph_bench,
        x="task",
        y="seconds",
        color="backend",
        barmode="group",
        title="Tiempo por tarea de grafo y backend",
        labels={"task": "Tarea", "seconds": "Tiempo mediano (s)", "backend": "Backend"},
        color_discrete_map=COLORS,
        text_auto=".3f",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ── Speedup table ──
    if "speedup_vs_cpu" in cugraph_bench.columns:
        gpu_rows = cugraph_bench[cugraph_bench["backend"] != "networkx_cpu"].copy()
        if not gpu_rows.empty:
            st.subheader("Speedup por tarea")
            for _, r in gpu_rows.iterrows():
                speedup = _safe_float(r.get("speedup_vs_cpu"))
                task = str(r.get("task", ""))
                backend = str(r.get("backend", ""))
                if speedup > 0:
                    icon = "🟢" if speedup > 1.0 else "🔴"
                    st.caption(f"{icon} **{task}** ({backend}): {speedup:.1f}x speedup vs NetworkX CPU")

    if not has_gpu_data:
        st.warning(
            "Solo hay datos de CPU disponibles. Los benchmarks GPU de cuGraph "
            "requieren un entorno RAPIDS con cuGraph instalado. Ejecuta el notebook "
            "en un entorno RAPIDS para ver las comparaciones completas."
        )

    with st.expander("Datos detallados de benchmark de grafos"):
        st.dataframe(cugraph_bench, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Tareas evaluadas:**

- **graph_build**: Construcción del grafo desde edgelist. En CPU esto es O(E) con
  overhead de Python dict. cuGraph usa CSR/CSC nativo en GPU memory.

- **connected_components**: BFS/Union-Find para encontrar componentes conexas.
  Altamente paralelizable — cada nodo puede explorarse independientemente.

- **pagerank**: Iteraciones de power method (multiplicación matriz-vector repetida).
  Natural para GPU: cada iteración es una SpMV (sparse matrix-vector multiply).
  Nota: verificar `sum_pagerank ≈ 1.0` y `converged_rate = 1.0`.

- **louvain**: Detección de comunidades por optimización de modularidad.
  cuGraph implementa el algoritmo de Louvain con coarsening multi-nivel en GPU.

**nx-cugraph dispatch**: Permite usar la API de NetworkX (`nx.pagerank(G, backend="cugraph")`)
con aceleración transparente. El grafo se transfiere a GPU automáticamente.
"""
    )
else:
    st.info("No hay datos de benchmark cuGraph disponibles.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Optimization — HiGHS vs cuOpt
# ══════════════════════════════════════════════════════════════════════════════

st.header("4) Optimización de Portafolio (HiGHS vs cuOpt)")

st.markdown(
    """
**Problema**: LP de selección de portafolio con restricciones de presupuesto
(20% del total), riesgo (PD-weighted ≤ 10% del presupuesto), y concentración
por `purpose` (≤ 35% por categoría). Este es el mismo tipo de problema que
resuelve `scripts/optimize_portfolio.py` con Pyomo+HiGHS.

**Backends**:
- `scipy_highs_cpu`: SciPy `linprog(method="highs")` — solver LP open-source
- `scipy_milp_cpu`: SciPy `milp()` — solver MILP (referencia)
- `cuopt_gpu`: NVIDIA cuOpt `DataModel` + `Solve` API
"""
)

if not cuopt_bench.empty and "task" in cuopt_bench.columns:
    # Separate LP results from smoke tests/errors
    lp_data = cuopt_bench[cuopt_bench["task"] == "portfolio_lp"].copy()
    milp_data = cuopt_bench[cuopt_bench["task"] == "portfolio_milp_reference"].copy()
    other_data = cuopt_bench[~cuopt_bench["task"].isin(["portfolio_lp", "portfolio_milp_reference"])].copy()

    # ── LP comparison chart ──
    if not lp_data.empty:
        # Only show backends with valid times
        valid_lp = lp_data[lp_data["seconds"].notna() & (lp_data["seconds"] > 0)].copy()

        if not valid_lp.empty:
            fig = px.bar(
                valid_lp,
                x="backend",
                y="seconds",
                color="backend",
                title=f"Portfolio LP: Tiempo de resolución ({int(valid_lp['n_variables'].iloc[0]):,} variables)",
                labels={"backend": "Solver", "seconds": "Tiempo mediano (s)"},
                color_discrete_map=COLORS,
                text_auto=".4f",
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350, showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # Show LP results with objective comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Resultados LP:**")
            for _, r in lp_data.iterrows():
                backend = str(r.get("backend", ""))
                seconds = _safe_float(r.get("seconds"))
                status = str(r.get("status", ""))
                objective = _safe_float(r.get("objective"))
                n_vars = int(_safe_float(r.get("n_variables")))

                if "error" in status.lower() or seconds == 0:
                    st.error(f"**{backend}**: Error — {status}")
                else:
                    st.success(f"**{backend}**: {seconds:.4f}s, objetivo={objective:,.0f}, {n_vars:,} variables")

        with col2:
            if not milp_data.empty:
                st.markdown("**Referencia MILP:**")
                for _, r in milp_data.iterrows():
                    seconds = _safe_float(r.get("seconds"))
                    objective = _safe_float(r.get("objective"))
                    n_vars = int(_safe_float(r.get("n_variables")))
                    st.info(f"MILP ({n_vars:,} vars): {seconds:.4f}s, objetivo={objective:,.0f}")

    # ── Error details ──
    error_rows = cuopt_bench[cuopt_bench["status"].astype(str).str.contains("error", case=False, na=False)]
    if not error_rows.empty:
        with st.expander("Errores detectados en benchmarks de optimización"):
            for _, r in error_rows.iterrows():
                st.warning(f"**{r.get('task', '')}** ({r.get('backend', '')}): {r.get('status', '')}")
            st.markdown(
                """
**Nota sobre cuOpt**: La API de cuOpt ha cambiado significativamente entre versiones.
El error `'Problem' object has no attribute 'set_objective_data'` indica que la versión
instalada no soporta la API legacy `Problem`. El notebook usa la API actual
`DataModel` + `Solve` que es la correcta para RAPIDS 26.02.

Si ves errores, verifica:
1. Que cuOpt está instalado: `pip install cuopt`
2. Que la versión es compatible con RAPIDS 26.02
3. Que la GPU tiene suficiente VRAM para el problema
"""
            )

    with st.expander("Datos completos de optimización"):
        st.dataframe(cuopt_bench, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Contexto del proyecto**: El pipeline de optimización de portafolio usa
**Pyomo + HiGHS** como solver principal (ver `src/optimization/portfolio_model.py`).
cuOpt es una alternativa GPU que puede acelerar problemas grandes (>10K variables).

**Conexión con la tesis**: Los **uncertainty sets conformales** (PD_low, PD_high)
del predictor conformal Mondrian se usan como restricciones de incertidumbre en
la optimización robusta. Un solver más rápido permite evaluar más escenarios
de robustez en menos tiempo.

**Scaling behavior**: LP solvers GPU son más eficientes a escala:
- < 5K variables: CPU (HiGHS) generalmente gana por menor overhead
- 5K–20K variables: GPU empieza a ser competitivo
- &gt; 20K variables: GPU tiene ventaja significativa (paralelismo en simplex/IPM)
"""
    )
else:
    st.info("No hay datos de benchmark cuOpt disponibles.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Numerical Computing — NumPy/SciPy vs CuPy
# ══════════════════════════════════════════════════════════════════════════════

st.header("5) Cómputo Numérico (NumPy/SciPy vs CuPy)")

st.markdown(
    """
**CuPy** es un reemplazo drop-in de NumPy/SciPy que ejecuta operaciones en GPU.
Los arrays CuPy viven en VRAM y las operaciones usan cuBLAS, cuSOLVER, cuSPARSE.

**Tareas evaluadas:**
- **Monte Carlo ECL**: Simulación de 100K escenarios × 10K préstamos
  (PD × LGD × EAD) — operación central en provisioning IFRS9
- **SVD**: Descomposición en valores singulares de la matriz de features
  (100K × 27) — usado en PCA y reducción dimensional
- **Sparse MatMul**: Multiplicación de matrices dispersas CSR (50K × 50K,
  densidad 0.1%) — relevante para grafos y regularización
"""
)

if not cupy_bench.empty and "task" in cupy_bench.columns:
    tasks = sorted(cupy_bench["task"].unique())

    for task_name in tasks:
        task_data = cupy_bench[cupy_bench["task"] == task_name].copy()
        if task_data.empty or len(task_data) < 1:
            continue

        col1, col2 = st.columns([2, 1])

        with col1:
            valid_data = task_data[task_data["seconds"].notna() & (task_data["seconds"] > 0)]
            if not valid_data.empty:
                fig = px.bar(
                    valid_data,
                    x="backend",
                    y="seconds",
                    color="backend",
                    title=f"{task_name.replace('_', ' ').title()}: CPU vs GPU",
                    labels={"backend": "Backend", "seconds": "Tiempo mediano (s)"},
                    color_discrete_map=COLORS,
                    text_auto=".4f",
                )
                fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, showlegend=False)
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            cpu_row = task_data[task_data["backend"].str.contains("cpu", case=False, na=False)]
            gpu_row = task_data[task_data["backend"].str.contains("gpu", case=False, na=False)]
            if not cpu_row.empty and not gpu_row.empty:
                cpu_s = _safe_float(cpu_row.iloc[0]["seconds"])
                gpu_s = _safe_float(gpu_row.iloc[0]["seconds"])
                if gpu_s > 0 and cpu_s > 0:
                    speedup = cpu_s / gpu_s
                    if speedup > 1.0:
                        st.metric("Speedup GPU", f"{speedup:.1f}x", f"{(speedup - 1) * 100:.0f}% más rápido")
                    else:
                        st.metric("Speedup GPU", f"{speedup:.2f}x", f"{(1 - speedup) * 100:.0f}% más lento", delta_color="inverse")
            elif not cpu_row.empty:
                st.info("Solo CPU disponible")

    st.markdown(
        """
**Por qué Monte Carlo ECL es ideal para GPU:**
La simulación `ECL = PD × LGD × EAD` es embarrassingly parallel: cada escenario
es independiente. Con 100K escenarios × 10K préstamos = 1 billón de multiplicaciones
flotantes — esto es exactamente lo que las miles de CUDA cores de una GPU hacen bien.

**Conexión con IFRS9**: El cálculo de Expected Credit Loss bajo múltiples escenarios
macroeconómicos (base, stress, severe) es una operación Monte Carlo. Acelerar esto
permite evaluar más escenarios y obtener distribuciones de ECL más robustas.

**SVD y reducción dimensional**: La descomposición SVD es una operación de álgebra
lineal densa (LAPACK → cuSOLVER) que escala como O(min(m,n)² × max(m,n)). Para
matrices grandes, GPU tiene ventaja significativa.
"""
    )

    with st.expander("Datos detallados de benchmark CuPy"):
        st.dataframe(cupy_bench, use_container_width=True, hide_index=True)
else:
    st.info(
        "No hay datos de benchmark CuPy disponibles. "
        "Ejecuta el notebook RAPIDS para generar benchmarks de Monte Carlo ECL, SVD y sparse matmul."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Scaling Analysis
# ══════════════════════════════════════════════════════════════════════════════

st.header("6) Curva de Escalamiento: GPU vs Tamaño del Dataset")

st.markdown(
    """
Un factor clave en la decisión CPU vs GPU es el **tamaño del dataset**.
Para datos pequeños, el overhead de transferencia CPU→GPU y lanzamiento de
kernels CUDA puede superar el beneficio del paralelismo. El **punto de cruce**
indica el tamaño mínimo donde GPU empieza a ser ventajoso.
"""
)

if not scaling.empty:
    x_col = "pct_data" if "pct_data" in scaling.columns else scaling.columns[0]
    y_col = "speedup_x" if "speedup_x" in scaling.columns else scaling.columns[-1]
    color_col = "method" if "method" in scaling.columns else None

    fig = px.line(
        scaling,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title="Speedup GPU vs % del dataset",
        labels={x_col: "Fracción del dataset", y_col: "Speedup (x)"},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#5F6B7A", annotation_text="1x = paridad CPU/GPU")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Show the data
    with st.expander("Datos de escalamiento"):
        st.dataframe(scaling, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Interpretación:**
- El overhead fijo de GPU (transferencia de datos, compilación de kernels) es
  constante ~10-50ms independiente del tamaño.
- Con datos pequeños (< 10K filas), este overhead domina → GPU es más lento.
- Con datos grandes (> 100K filas), el paralelismo masivo (miles de CUDA cores
  ejecutando simultáneamente) amortiza el overhead.
- El **punto de cruce** varía por operación: operaciones element-wise (cuDF)
  cruzan antes que operaciones con dependencias (cuML iterativo).
"""
    )
else:
    st.info(
        "No hay datos de escalamiento disponibles. "
        "Ejecuta el notebook RAPIDS con la sección de scaling analysis para ver "
        "cómo varía el speedup con el tamaño del dataset."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Resumen Comparativo por Librería
# ══════════════════════════════════════════════════════════════════════════════

st.header("7) Resumen Comparativo por Librería")

st.markdown(
    """
Cada librería RAPIDS tiene un perfil diferente de rendimiento. No existe una
solución universal — la elección depende de la tarea, el tamaño del dataset,
y las restricciones de infraestructura.
"""
)

# Build per-library summary cards
summaries: list[dict] = []

# cuDF
if not cudf_bench.empty and "speedup_vs_pandas_cpu" in cudf_bench.columns:
    ok = cudf_bench[cudf_bench["status"] == "ok"] if "status" in cudf_bench.columns else cudf_bench
    speeds = ok["speedup_vs_pandas_cpu"].dropna()
    if len(speeds):
        summaries.append({
            "Librería": "cuDF / Polars GPU",
            "Mejor speedup": f"{speeds.max():.1f}x",
            "Mediana speedup": f"{speeds.median():.1f}x",
            "Caso de uso": "ETL, feature engineering, data wrangling",
            "Fortaleza": "Zero-code-change acceleration",
            "Debilidad": "Requiere datos en VRAM (10 GB limit en RTX 3080)",
            "Recomendación": "Usar para datasets > 100K filas con operaciones tabulares",
        })

# cuML
if not cuml_bench.empty and "fit_speedup_vs_cpu" in cuml_bench.columns:
    gpu = cuml_bench[cuml_bench["backend"] == "cuml_gpu"]
    speeds = gpu["fit_speedup_vs_cpu"].dropna()
    if len(speeds):
        best_task = gpu.loc[gpu["fit_speedup_vs_cpu"].idxmax(), "task"] if len(gpu) else "N/D"
        worst_task = gpu.loc[gpu["fit_speedup_vs_cpu"].idxmin(), "task"] if len(gpu) else "N/D"
        summaries.append({
            "Librería": "cuML",
            "Mejor speedup": f"{speeds.max():.1f}x ({best_task})",
            "Mediana speedup": f"{speeds.median():.1f}x",
            "Caso de uso": "Entrenamiento e inferencia ML",
            "Fortaleza": "Algoritmos paralelos (RF, KMeans, PCA)",
            "Debilidad": f"Overhead en algoritmos iterativos ({worst_task})",
            "Recomendación": "Usar para RF, KMeans, KNN; evaluar caso por caso para LR",
        })

# cuGraph
if not cugraph_bench.empty and "speedup_vs_cpu" in cugraph_bench.columns:
    speeds = cugraph_bench["speedup_vs_cpu"].dropna()
    if len(speeds):
        summaries.append({
            "Librería": "cuGraph",
            "Mejor speedup": f"{speeds.max():.1f}x",
            "Mediana speedup": f"{speeds.median():.1f}x",
            "Caso de uso": "Análisis de grafos de crédito",
            "Fortaleza": "PageRank, Louvain, componentes conexas",
            "Debilidad": "Overhead de transferencia para grafos pequeños",
            "Recomendación": "Usar para grafos con > 100K aristas",
        })

# cuOpt
if not cuopt_bench.empty:
    valid = cuopt_bench[cuopt_bench["seconds"].notna() & (cuopt_bench["seconds"] > 0)]
    if not valid.empty:
        summaries.append({
            "Librería": "cuOpt",
            "Mejor speedup": "Depende de escala",
            "Mediana speedup": "Depende de escala",
            "Caso de uso": "Optimización LP/MILP de portafolio",
            "Fortaleza": "Problemas grandes (>10K variables)",
            "Debilidad": "API en evolución, overhead para problemas pequeños",
            "Recomendación": "Usar para problemas de optimización a escala",
        })

# CuPy
if not cupy_bench.empty and "speedup_vs_cpu" in cupy_bench.columns:
    speeds = cupy_bench["speedup_vs_cpu"].dropna()
    if len(speeds):
        summaries.append({
            "Librería": "CuPy",
            "Mejor speedup": f"{speeds.max():.1f}x",
            "Mediana speedup": f"{speeds.median():.1f}x",
            "Caso de uso": "Monte Carlo, SVD, álgebra lineal",
            "Fortaleza": "Drop-in NumPy/SciPy, embarrassingly parallel ops",
            "Debilidad": "Requiere refactoring para operaciones no element-wise",
            "Recomendación": "Usar para simulaciones Monte Carlo y SVD",
        })

if summaries:
    summary_df = pd.DataFrame(summaries)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Recommendation matrix ──
    st.subheader("Matriz de decisión: ¿Cuándo usar GPU?")
    st.markdown(
        """
| Escenario | Recomendación | Librería |
|-----------|--------------|----------|
| ETL/Feature engineering > 100K filas | **GPU** | Polars GPU o cudf.pandas |
| ETL/Feature engineering < 100K filas | **CPU** | Polars CPU o DuckDB |
| Datasets más grandes que RAM | **CPU** | DuckDB (spill-to-disk) |
| Random Forest / KMeans / KNN training | **GPU** | cuML |
| Logistic Regression / SVM training | **CPU** | scikit-learn |
| Monte Carlo simulation (> 10K escenarios) | **GPU** | CuPy |
| Álgebra lineal densa (SVD, eigenvalues) | **GPU** | CuPy |
| Grafos grandes (> 100K aristas) | **GPU** | cuGraph |
| Grafos pequeños | **CPU** | NetworkX |
| LP/MILP > 10K variables | **GPU** | cuOpt |
| LP/MILP < 10K variables | **CPU** | HiGHS (Pyomo) |
| Portabilidad de código | **CPU** | Narwhals (write-once API) |
"""
    )
else:
    st.info("No hay datos suficientes para generar el resumen comparativo.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Hardware y Metodología
# ══════════════════════════════════════════════════════════════════════════════

st.header("8) Hardware y Metodología")

col_hw, col_method = st.columns(2)

with col_hw:
    st.subheader("Especificaciones")
    if meta and "hardware" in meta:
        hw = meta["hardware"]
        st.markdown(
            f"""
| Componente | Especificación |
|------------|---------------|
| **GPU** | {hw.get('gpu', 'NVIDIA GeForce RTX 3080')} |
| **VRAM** | {hw.get('vram', '10 GB GDDR6X')} |
| **CPU** | {hw.get('cpu', 'AMD Ryzen 5 5600X')} |
| **RAM** | {hw.get('ram', '24 GB DDR4')} |
| **OS** | {hw.get('os', 'WSL2 / Linux')} |
| **CUDA** | Driver 591.86, CUDA 13.1 |
"""
        )
    else:
        st.markdown(
            """
| Componente | Especificación |
|------------|---------------|
| **GPU** | NVIDIA GeForce RTX 3080 |
| **VRAM** | 10 GB GDDR6X |
| **CPU** | AMD Ryzen 5 5600X (6-core, 12-thread) |
| **RAM** | 24 GB DDR4 |
| **OS** | WSL2 / Linux |
| **CUDA** | Driver 591.86, CUDA 13.1 |
"""
        )

with col_method:
    st.subheader("Protocolo")
    st.markdown(
        """
**Medición:**
- Cada benchmark ejecuta múltiples repeticiones con warmup
- Se reporta la **mediana** (robusta a outliers) con IQR
- Sincronización GPU explícita (`cp.cuda.Stream.null.synchronize()`)

**RMM (RAPIDS Memory Manager):**
- Pool allocator con 6 GB iniciales
- RTX 3080 safe (10 GB VRAM total, 4 GB para OS/driver)
- CuPy integrado via `rmm_cupy_allocator`

**Quality gates:**
- Row-count parity (cuDF)
- Checksum relative error ≤ 0.5% (cuDF)
- Metric tolerance per algorithm (cuML)
- Convergence verification (cuGraph PageRank)
- Objective parity (cuOpt LP)
"""
    )

with st.expander("Versiones de librerías"):
    if meta and "library_versions" in meta:
        versions = meta["library_versions"]
        ver_df = pd.DataFrame(
            [{"Librería": k, "Versión": v} for k, v in sorted(versions.items())]
        )
        st.dataframe(ver_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
Ejecuta el notebook RAPIDS para registrar las versiones exactas en `gpu_bench_meta.json`.

**Versiones objetivo (RAPIDS 26.02):**
- cuDF, cuML, cuGraph, cuOpt: 26.02
- CuPy: 13.x
- Polars: 1.x (con GPUEngine)
- DuckDB: 1.x
- Narwhals: 1.x
"""
        )

with st.expander("Sobre DuckDB vs Polars (referencia externa)"):
    st.markdown(
        """
Según el [benchmark de codecentric.de](https://www.codecentric.de/en/knowledge-hub/blog/duckdb-vs-polars-performance-and-memory-with-massive-parquet-data)
con datos masivos (2 GB → 2 TB):

| Configuración | Memoria (140 GB dataset) | Velocidad | Mejor para |
|---------------|-------------------------|-----------|-----------|
| **DuckDB** | 1.3 GB | Más rápido | Entornos con restricción de memoria |
| **Polars (default)** | 17 GB | Competitivo | Datos particionados con RAM disponible |
| **Polars (async)** | 750 MB | Más lento | Archivos grandes no particionados |

**Hallazgo clave**: Particionar los datos reduce dramáticamente el uso de memoria
para ambos motores (8x DuckDB, 4x Polars). DuckDB mantiene un consumo de memoria
constante independiente del tamaño del dataset gracias a su motor streaming.

En nuestro caso (190 MB parquet), ambos motores operan cómodamente en RAM, por lo que
la diferencia principal es la velocidad de ejecución de la query analítica.
"""
    )

# ── Footer ──
st.markdown("---")
st.caption(
    "Este benchmark es un **side project independiente** del pipeline principal de tesis. "
    "Los resultados provienen de `reports/gpu_benchmark/` y se generan ejecutando "
    "`notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb` en un entorno RAPIDS."
)
