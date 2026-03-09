"""Anexo: RAPIDS 26.02 GPU Benchmark — Credit Risk at GPU Speed."""

# ruff: noqa: E402

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
import streamlit as st

from streamlit_app.components.story_shell import render_key_takeaway, render_page_header
from streamlit_app.components.v2_echarts import render_v2_echarts
from streamlit_app.content.page_contracts import get_page_contract
from streamlit_app.theme import PLOTLY_TEMPLATE
from streamlit_app.utils import download_table

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
GPU_DIR = Path(__file__).resolve().parents[2] / "reports" / "gpu_benchmark"

COLORS = {
    "pandas_cpu": "#E45756",
    "cudf_pandas_gpu": "#F58518",
    "polars_cpu": "#4C78A8",
    "polars_gpu": "#72B7B2",
    "duckdb": "#54A24B",
    "sklearn_cpu": "#B3B8BD",
    "cuml_gpu": "#0B5ED7",
    "networkx_cpu": "#B3B8BD",
    "nx_cugraph_gpu": "#198754",
    "cugraph_gpu": "#0B5ED7",
    "scipy_highs_cpu": "#B3B8BD",
    "scipy_milp_cpu": "#6F42C1",
    "cuopt_gpu": "#0B5ED7",
    "cuopt_milp_gpu": "#0B5ED7",
    "numpy_cpu": "#B3B8BD",
    "scipy_cpu": "#B3B8BD",
    "cupy_gpu": "#0B5ED7",
}

BACKEND_LABELS = {
    "pandas_cpu": "Pandas (CPU)",
    "cudf_pandas_gpu": "cuDF.pandas (GPU)",
    "polars_cpu": "Polars (CPU)",
    "polars_gpu": "Polars GPU Engine",
    "duckdb": "DuckDB (CPU)",
    "sklearn_cpu": "scikit-learn (CPU)",
    "cuml_gpu": "cuML (GPU)",
    "networkx_cpu": "NetworkX (CPU)",
    "nx_cugraph_gpu": "nx-cugraph (GPU)",
    "cugraph_gpu": "cuGraph (GPU)",
    "scipy_highs_cpu": "SciPy HiGHS (CPU)",
    "scipy_milp_cpu": "SciPy MILP (CPU)",
    "cuopt_gpu": "cuOpt (GPU)",
    "cuopt_milp_gpu": "cuOpt MILP (GPU)",
    "numpy_cpu": "NumPy (CPU)",
    "scipy_cpu": "SciPy (CPU)",
    "cupy_gpu": "CuPy (GPU)",
}


@st.cache_data(ttl=300)
def _load(name: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        p = GPU_DIR / f"{name}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _meta() -> dict:
    p = GPU_DIR / "gpu_bench_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _sf(val: object, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def _echarts_ml_speedup_option(gpu_valid: pd.DataFrame) -> dict:
    """Build ECharts 5 option for ML speedup pilot (clickable horizontal bars)."""
    df = gpu_valid.copy()
    df["task_label"] = df["task"].str.replace("_", " ").str.title()
    categories = df["task_label"].tolist()
    series_data = [
        {
            "value": round(_sf(row["fit_speedup_vs_cpu"]), 4),
            "task_id": str(row["task"]),
            "task_label": str(row["task_label"]),
            "itemStyle": {
                "color": COLORS["cuml_gpu"] if _sf(row["fit_speedup_vs_cpu"]) >= 1 else "#E45756"
            },
        }
        for _, row in df.iterrows()
    ]
    return {
        "animationDuration": 500,
        "grid": {"left": 170, "right": 28, "top": 36, "bottom": 30},
        "tooltip": {"trigger": "item"},
        "xAxis": {
            "type": "value",
            "name": "Speedup (x)",
            "nameLocation": "middle",
            "nameGap": 28,
            "min": 0,
            "axisLabel": {"formatter": "{value}x"},
        },
        "yAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"fontSize": 11},
        },
        "series": [
            {
                "name": "cuML vs CPU",
                "type": "bar",
                "data": series_data,
                "label": {"show": True, "position": "right", "formatter": "{c}x"},
                "emphasis": {"focus": "self"},
                "markLine": {
                    "silent": True,
                    "symbol": ["none", "none"],
                    "lineStyle": {"type": "dashed", "color": "#E45756"},
                    "label": {"formatter": "1x parity"},
                    "data": [{"xAxis": 1}],
                },
            }
        ],
    }


def _echarts_graph_speedup_option(gpu_data: pd.DataFrame) -> dict:
    """Build grouped-bar ECharts option for graph speedup pilot (clickable bars)."""
    df = gpu_data.copy()
    df = df.dropna(subset=["speedup_vs_cpu"])
    if df.empty:
        return {}

    df["task_label"] = df["task"].str.replace("_", " ").str.title()
    task_order = (
        df.groupby(["task", "task_label"], as_index=False)["speedup_vs_cpu"]
        .max()
        .sort_values("speedup_vs_cpu", ascending=True)
    )

    backend_order = ["nx_cugraph_gpu", "cugraph_gpu"]
    series = []
    for idx, backend in enumerate(backend_order):
        backend_df = df[df["backend"] == backend].set_index("task")
        data_points = []
        for row in task_order.itertuples(index=False):
            task_id = str(row.task)
            task_label = str(row.task_label)
            speedup = (
                backend_df.loc[task_id, "speedup_vs_cpu"] if task_id in backend_df.index else np.nan
            )
            speed = None if pd.isna(speedup) else round(_sf(speedup), 4)
            data_points.append(
                {
                    "value": speed,
                    "task_id": task_id,
                    "task_label": task_label,
                    "backend_id": backend,
                    "backend_label": _label(backend),
                    "itemStyle": {"color": COLORS.get(backend, "#4C78A8")},
                }
            )

        series_cfg = {
            "name": _label(backend),
            "type": "bar",
            "barMaxWidth": 18,
            "emphasis": {"focus": "series"},
            "label": {"show": True, "position": "right", "formatter": "{c}x"},
            "data": data_points,
        }
        if idx == 0:
            series_cfg["markLine"] = {
                "silent": True,
                "symbol": ["none", "none"],
                "lineStyle": {"type": "dashed", "color": "#E45756"},
                "label": {"formatter": "1x parity"},
                "data": [{"xAxis": 1}],
            }
        series.append(series_cfg)

    return {
        "animationDuration": 500,
        "grid": {"left": 180, "right": 24, "top": 48, "bottom": 30},
        "legend": {"top": 4},
        "tooltip": {
            "trigger": "item",
            "formatter": "{a}<br/>{b}: {c}x",
        },
        "xAxis": {
            "type": "value",
            "name": "Speedup vs NetworkX CPU",
            "nameLocation": "middle",
            "nameGap": 28,
            "min": 0,
            "axisLabel": {"formatter": "{value}x"},
        },
        "yAxis": {
            "type": "category",
            "data": task_order["task_label"].tolist(),
            "axisLabel": {"fontSize": 11},
        },
        "series": series,
    }


# ===================================================================
# Page
# ===================================================================

st.title("Can a $700 GPU Accelerate a Full Credit Risk Pipeline?")
page_contract = get_page_contract("gpu_benchmark")
render_page_header(page_contract)
render_key_takeaway(
    "Anexo técnico independiente: compara aceleración GPU en tareas de datos/ML/grafos/optimización, sin alterar el pipeline canónico de la tesis."
)
st.markdown(
    """
*A hands-on benchmark of NVIDIA RAPIDS 26.02 on 1.86 million Lending Club loans,
running on a consumer RTX 3080 (10 GB VRAM) under WSL2.*

---

**The question every data scientist with a GPU eventually asks:**
*"I have this expensive graphics card — can it actually speed up my data work, or is it
just for gaming and deep learning?"*

This benchmark answers that question **end-to-end** for a real credit risk pipeline.
We test every stage of the thesis workflow — from raw data wrangling through ML model
training, graph analytics, portfolio optimization, and Monte Carlo simulation — using
**five RAPIDS libraries** against their CPU counterparts. For each technique we measure
not just speed, but **output correctness** and explain **where it fits in our pipeline**.
"""
)

# -- Load all artifacts --
df_bench = _load("cudf_polars_benchmark")
ml_bench = _load("cuml_benchmark")
gr_bench = _load("cugraph_benchmark")
opt_bench = _load("cuopt_benchmark")
cp_bench = _load("cupy_benchmark")
meta = _meta()

# -- Hero KPIs --
cols = st.columns(5)
hero_data = [
    ("Dataset", "1.86M loans"),
    ("GPU", "RTX 3080 10GB"),
    ("RAPIDS", "26.02"),
    ("Libraries", "5 tested"),
    ("Best Speedup", "164x"),
]
for c, (label, val) in zip(cols, hero_data, strict=False):
    c.metric(label, val)

# ===================================================================
# 1. DataFrame Processing
# ===================================================================

st.markdown("---")
st.header("1. DataFrame Processing: The Great Five-Way Race")

st.markdown(
    """
> **Where this matters in our pipeline:**
> Every run of `make_dataset.py` and `prepare_dataset.py` parses the full
> `lending_club_cleaned.parquet` (1.86M rows, 110 columns) — type-casting string
> percentages, filtering by credit quality, and computing grade-level aggregates.
> The dbt pipeline (`dbt_project/`) runs similar groupby-join-window patterns on
> the same data. Faster wrangling means faster iteration during development and
> shorter end-to-end pipeline execution.

We run a realistic analytics workload that mirrors the actual pipeline:

1. **Read** parquet (select 11 columns from 110 available)
2. **Parse** string fields (`int_rate` "13.5%" -> 13.5, `term` "36 months" -> 36)
3. **Filter** (loan_amnt >= $5K, income > $20K, grades A-E) -> 1.6M rows survive
4. **GroupBy** grade x year: count loans, sum funded, mean rate, mean default rate
5. **GroupBy** purpose: loans per purpose, default rate
6. **Join** the two aggregates via grade -> top-purpose mapping
7. **Window** function: rank within year by default rate
8. **Sort** by year, grade
"""
)

if not df_bench.empty and "mode" in df_bench.columns:
    has_status = "status" in df_bench.columns
    ok = df_bench[df_bench["status"] == "ok"].copy() if has_status else df_bench.copy()
    ok = ok.dropna(subset=["median_seconds"])

    if not ok.empty:
        ok = ok.sort_values("median_seconds", ascending=True)
        pandas_t = ok.loc[ok["mode"] == "pandas_cpu", "median_seconds"]
        base_t = pandas_t.values[0] if not pandas_t.empty else 1.0
        ok["speedup"] = base_t / ok["median_seconds"]
        ok["label"] = ok["mode"].map(_label)

        # -- Bar chart --
        fig = px.bar(
            ok.sort_values("median_seconds", ascending=False),
            y="label",
            x="median_seconds",
            orientation="h",
            color="mode",
            color_discrete_map=COLORS,
            labels={"label": "", "median_seconds": "Seconds (median of 3 runs)"},
        )
        for _, r in ok.iterrows():
            fig.add_annotation(
                x=r["median_seconds"],
                y=_label(r["mode"]),
                text=f"  {r['median_seconds']:.3f}s  ({r['speedup']:.0f}x)",
                showarrow=False,
                xanchor="left",
                font={"size": 12},
            )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=max(280, len(ok) * 60),
            showlegend=False,
            title="1.86M Loans x 110 Columns: End-to-End Analytics",
        )
        st.plotly_chart(fig, width="stretch")

        # -- Cross-comparison matrix --
        times = ok.set_index("mode")["median_seconds"].to_dict()
        modes = list(ok["mode"])
        matrix = []
        for rm in modes:
            row = {"vs": _label(rm)}
            for cm in modes:
                if rm == cm:
                    row[_label(cm)] = "--"
                else:
                    row[_label(cm)] = f"{times[rm] / max(times[cm], 1e-12):.1f}x"
            matrix.append(row)
        st.markdown("**Cross-comparison matrix** (row is N times slower than column):")
        st.dataframe(pd.DataFrame(matrix).set_index("vs"), width="stretch")

        download_table(ok, "dataframe_benchmark.csv", "Download results")

    st.markdown(
        """
### Key Findings

**Polars dominates.** Both CPU and GPU variants finish in ~130ms — **34x faster than
pandas**. The GPU engine (`cudf-polars`) provides marginal improvement over CPU Polars
because the bottleneck at this scale is I/O and query planning, not compute.

**cuDF pandas accelerator is the surprise winner for zero-effort migration.**
With `cudf.pandas.install()` and *zero code changes* to your pandas code, you get
**13x speedup** (0.34s vs 4.5s). This is the lowest-effort GPU acceleration available.

**DuckDB is the memory champion.** At 0.37s (12x vs pandas) it's fast, but its real
strength is memory efficiency — DuckDB can process datasets far larger than RAM via
its streaming engine. For our 190 MB parquet both fit comfortably, but for production
batch scoring on millions of loans DuckDB's spill-to-disk capability is invaluable.

**How this applies to our project:**
| Pipeline stage | Current tool | Opportunity |
|----------------|-------------|-------------|
| `make_dataset.py` (clean + split) | pandas | `cudf.pandas.install()` -> 13x with zero changes |
| `build_datasets.py` (aggregates) | pandas | Polars rewrite -> 34x on groupby-heavy workloads |
| dbt pipeline (SQL analytics) | DuckDB | Already optimal — 12x vs pandas with SQL expressiveness |
| Streamlit data loading | pandas | `cudf.pandas` for hot-path artifact loading |
"""
    )
else:
    st.info("No DataFrame benchmark data found.")

# ===================================================================
# 2. Machine Learning
# ===================================================================

st.markdown("---")
st.header("2. Machine Learning: Where GPU Really Shines (and Where It Doesn't)")

st.markdown(
    """
> **Where this matters in our pipeline:**
> The thesis trains a **CatBoost PD model** (comparable to Random Forest in GPU behavior),
> a **Logistic Regression baseline**, runs **PCA** for feature analysis, and uses
> **KMeans** for borrower segmentation. Each of these has a cuML GPU counterpart.
> Understanding where GPU helps (and where it doesn't) directly informs whether to
> GPU-accelerate `train_pd_model.py` and `feature_engineering.py`.

We train **7 algorithms** on `train_fe.parquet` — the same 47 engineered features
(WOE-encoded, numerical) used in the thesis PD model. Each runs on scikit-learn (CPU)
and cuML 26.02 (GPU), comparing **speed** and **output quality**.

| Algorithm | Train size | Test size | Features | Relevance to project |
|-----------|-----------|----------|----------|---------------------|
| **Logistic Regression** | 500K | 100K | 47 | LR baseline in `train_pd_model.py` |
| **Random Forest** | 500K | 100K | 47 | Tree-model proxy for CatBoost GPU potential |
| **KMeans** | 100K | -- | 47 | Borrower segmentation by risk profile |
| **PCA** | 500K | 100K | 47 | Feature importance analysis in NB02 |
| **KNN** | 80K | 20K | 47 | Nearest-neighbor imputation & scoring |
| **UMAP** | 50K | -- | 47 | 2D visualization of borrower clusters |
| **HDBSCAN** | 50K | -- | 47 | Anomaly detection for fraud/outlier flagging |
"""
)

if not ml_bench.empty and "task" in ml_bench.columns:
    tasks = ml_bench["task"].unique()
    ml_focus_task = str(st.session_state.get("gpu_benchmark_ml_focus_task", ""))

    # -- Overview speedup chart --
    gpu_rows = ml_bench[ml_bench["backend"] == "cuml_gpu"].copy()
    gpu_valid = gpu_rows[gpu_rows["fit_speedup_vs_cpu"].notna()].copy()
    if not gpu_valid.empty:
        gpu_valid = gpu_valid.sort_values("fit_speedup_vs_cpu", ascending=True)
        gpu_valid["task_label"] = gpu_valid["task"].str.replace("_", " ").str.title()
        fig = px.bar(
            gpu_valid,
            y="task_label",
            x="fit_speedup_vs_cpu",
            orientation="h",
            color_discrete_sequence=["#0B5ED7"],
            labels={"task_label": "", "fit_speedup_vs_cpu": "Speedup (x)"},
            title="GPU Speedup by Algorithm (fit time, higher = better)",
        )
        fig.add_vline(
            x=1.0, line_dash="dash", line_color="#E45756", annotation_text="GPU slower | GPU faster"
        )
        for _, r in gpu_valid.iterrows():
            s = r["fit_speedup_vs_cpu"]
            fig.add_annotation(
                x=max(s, 0.1),
                y=r["task_label"],
                text=f"  {s:.1f}x",
                showarrow=False,
                xanchor="left",
                font={"size": 12},
            )
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=max(300, len(gpu_valid) * 50),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("ECharts 5 pilot (clic para enfocar algoritmo)", expanded=False):
            st.caption(
                "Piloto adicional: ECharts 5 sobre Components v2 para callbacks de click. "
                "Plotly sigue siendo la visualización canónica; este bloque prueba UX interactiva."
            )
            echarts_clicked = render_v2_echarts(
                _echarts_ml_speedup_option(gpu_valid),
                key="gpu_benchmark_ml_speedup_echarts_pilot",
                height_px=max(320, len(gpu_valid) * 52),
            )
            clicked_data = (
                echarts_clicked.get("data") if isinstance(echarts_clicked, dict) else None
            )
            if isinstance(clicked_data, dict):
                task_id = clicked_data.get("task_id")
                if isinstance(task_id, str) and task_id:
                    st.session_state["gpu_benchmark_ml_focus_task"] = task_id
                    ml_focus_task = task_id

            c_focus, c_reset = st.columns([4, 1])
            with c_focus:
                if ml_focus_task:
                    st.info(
                        "Foco actual desde ECharts: "
                        f"`{ml_focus_task.replace('_', ' ').title()}`. "
                        "Se abrirá automáticamente su expander más abajo."
                    )
                else:
                    st.caption(
                        "Haz clic en una barra del gráfico ECharts para enfocar un algoritmo."
                    )
            with c_reset:
                if st.button("Reset", key="gpu_ml_echarts_focus_reset"):
                    st.session_state.pop("gpu_benchmark_ml_focus_task", None)
                    ml_focus_task = ""

    # -- Per-algorithm context --
    PROJECT_CONTEXT = {
        "logistic_regression": (
            "The **LR baseline** in `train_pd_model.py` trains on 1.35M rows with "
            "sklearn `lbfgs`. cuML's L-BFGS is 64x faster but didn't converge to "
            "the same solution — AUC dropped from 0.69 to 0.55. For production PD "
            "models, **correctness trumps speed**: we keep sklearn for LR."
        ),
        "random_forest": (
            "Our thesis uses **CatBoost** (not RF), but RF is the best proxy for "
            "tree-based GPU acceleration. At 7.5x speedup with <0.2% AUC difference, "
            "this validates that **GPU tree training works reliably**. CatBoost also "
            "has GPU support (`task_type='GPU'`) that we use in `train_pd_model.py`."
        ),
        "kmeans": (
            "We use grade-based segmentation in `build_datasets.py`, but **KMeans "
            "on 100K rows is too small** for GPU to win — the data transfer overhead "
            "dominates. At 1M+ rows, cuML KMeans would overtake sklearn."
        ),
        "pca": (
            "PCA runs in NB02 for **feature importance analysis** and dimensionality "
            "reduction. At 500K x 47, GPU is only 1.4x faster — the matrix is too "
            "narrow for cuSOLVER to dominate. Wider matrices (500+ features) would "
            "see a bigger GPU advantage."
        ),
        "knn": (
            "KNN is useful for **imputation** (filling missing LGD values) and as "
            "a non-parametric scoring baseline. The 2.3x fit speedup is modest, but "
            "the **7.5x predict speedup** matters for real-time inference where "
            "every millisecond counts."
        ),
        "umap": (
            "UMAP produces the **2D borrower embeddings** displayed in the Streamlit "
            "dashboard. Since `umap-learn` wasn't installed (CPU baseline failed), "
            "we only have the GPU time. cuML UMAP is typically 50-100x faster than "
            "CPU — essential for interactive exploration of 1M+ borrowers."
        ),
        "hdbscan": (
            "HDBSCAN is our **anomaly detection** tool — identifying unusual loan "
            "applications that don't cluster with any risk group. At **77x speedup** "
            "with identical cluster assignments, this is a clear GPU win. In production, "
            "this enables real-time fraud flagging on incoming applications."
        ),
    }

    for task_name in sorted(tasks):
        task_data = ml_bench[ml_bench["task"] == task_name]
        cpu_row = task_data[task_data["backend"] == "sklearn_cpu"]
        gpu_row = task_data[task_data["backend"] == "cuml_gpu"]

        if gpu_row.empty:
            continue

        gpu_fit = _sf(gpu_row.iloc[0].get("fit_seconds"))
        gpu_metric_val = gpu_row.iloc[0].get("metric_value")
        gpu_metric = str(gpu_row.iloc[0].get("metric", ""))

        cpu_fit = _sf(cpu_row.iloc[0].get("fit_seconds")) if not cpu_row.empty else None
        cpu_metric_val = cpu_row.iloc[0].get("metric_value") if not cpu_row.empty else None

        speedup = _sf(gpu_row.iloc[0].get("fit_speedup_vs_cpu"))

        with st.expander(
            f"**{task_name.replace('_', ' ').title()}** — {speedup:.1f}x speedup",
            expanded=ml_focus_task == task_name,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("CPU fit", f"{cpu_fit:.3f}s" if cpu_fit else "N/A")
            c2.metric("GPU fit", f"{gpu_fit:.3f}s")
            c3.metric(
                "Speedup",
                f"{speedup:.1f}x",
                delta="faster" if speedup > 1 else "slower",
                delta_color="normal" if speedup > 1 else "inverse",
            )

            # Quality comparison
            if cpu_metric_val is not None and gpu_metric != "error":
                cpu_v = _sf(cpu_metric_val)
                gpu_v = _sf(gpu_metric_val)
                diff = abs(cpu_v - gpu_v)
                rel = diff / max(abs(cpu_v), 1e-10) * 100
                st.markdown(
                    f"**Quality check** ({gpu_metric}): CPU = `{cpu_v:.6f}`, "
                    f"GPU = `{gpu_v:.6f}` — diff = {diff:.6f} ({rel:.2f}% relative)"
                )
                if rel < 1:
                    st.success("Outputs are essentially identical.")
                elif rel < 5:
                    st.warning("Small numerical differences — acceptable for most use cases.")
                else:
                    st.error("Significant divergence — investigate before using in production.")

            # Project context
            ctx = PROJECT_CONTEXT.get(task_name)
            if ctx:
                st.info(ctx)

    st.markdown(
        """
### The Full Picture

| Algorithm | Speedup | Quality | Verdict |
|-----------|---------|---------|---------|
| **HDBSCAN** | **77x** | Identical | **Clear GPU win** — embarrassingly parallel |
| **LogReg** | **64x** | 21% AUC gap | **Misleading** — cuML didn't converge |
| **Random Forest** | **7.5x** | 0.15% AUC diff | **Solid win** — tree parallelism |
| **KNN** | **7.5x** | Identical AUC | Great for inference workloads |
| **PCA** | **1.4x** | Identical | Marginal at this scale |
| **KMeans** | **0.3x** | Diff silhouette | **GPU slower** at 100K rows |

### Lessons for Our Pipeline

1. **CatBoost `task_type='GPU'`** is the highest-impact change. Our RF benchmark
   shows 7.5x speedup for tree models — CatBoost on GPU typically achieves 3-8x
   for boosted trees, cutting `train_pd_model.py` from minutes to seconds.

2. **HDBSCAN on GPU** enables real-time anomaly scoring. At 77x, we could flag
   unusual applications in the FastAPI `/predict` endpoint without blocking.

3. **Don't GPU-accelerate LogReg.** The convergence differences are a production risk.
   Our LR baseline should stay on sklearn.

4. **cuml.accel** (RAPIDS 26.02) can dispatch sklearn calls to GPU automatically
   via `import cuml.accel; cuml.accel.install()` — same idea as cudf.pandas.
"""
    )

    download_table(ml_bench, "ml_benchmark.csv", "Download ML results")
else:
    st.info("No ML benchmark data found.")

# ===================================================================
# 3. Graph Analytics
# ===================================================================

st.markdown("---")
st.header("3. Graph Analytics: Where GPU Acceleration Is Transformative")

st.markdown(
    """
> **Where this matters in our pipeline:**
> Graph analytics isn't in the core thesis pipeline *yet*, but it's the natural
> extension for **fraud detection** and **borrower network analysis**. PageRank can
> identify which loan attributes (grade, purpose) are most "central" to default risk.
> Louvain community detection can find **risk clusters** — groups of borrowers with
> similar profiles that default together. Betweenness centrality reveals which
> attributes **bridge** between low-risk and high-risk communities.

We build a **borrower-attribute bipartite graph** from 200K loans in `train.parquet`:
each loan connects to its `grade`, `purpose`, `sub_grade`, and `home_ownership`
nodes — creating a graph with **200K nodes and 800K edges**.

Three backends compete:
- **NetworkX CPU** — pure Python, single-threaded (the baseline)
- **nx-cugraph** — zero-code-change GPU backend (`backend="cugraph"` parameter)
- **cuGraph direct** — native CUDA API (maximum performance, requires code changes)
"""
)

if not gr_bench.empty and "task" in gr_bench.columns:
    graph_focus_task = st.session_state.get("gpu_benchmark_graph_focus_task")
    graph_focus_backend = st.session_state.get("gpu_benchmark_graph_focus_backend")
    graph_speedup_data = pd.DataFrame()

    # -- Speedup chart --
    gpu_data = gr_bench[gr_bench["backend"] != "networkx_cpu"].copy()
    if not gpu_data.empty and "speedup_vs_cpu" in gpu_data.columns:
        gpu_data = gpu_data.dropna(subset=["speedup_vs_cpu"])
        graph_speedup_data = gpu_data.copy()
        gpu_data = gpu_data.sort_values("speedup_vs_cpu", ascending=True)
        gpu_data["backend_label"] = gpu_data["backend"].map(_label)
        gpu_data["task_label"] = gpu_data["task"].str.replace("_", " ").str.title()

        fig = px.bar(
            gpu_data,
            y="task_label",
            x="speedup_vs_cpu",
            color="backend_label",
            orientation="h",
            barmode="group",
            color_discrete_map={_label(k): v for k, v in COLORS.items()},
            labels={
                "task_label": "",
                "speedup_vs_cpu": "Speedup vs NetworkX CPU (x)",
                "backend_label": "Backend",
            },
            title="Graph Algorithm Speedup (200K nodes, 800K edges)",
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#E45756")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
        st.plotly_chart(fig, width="stretch")

    if not graph_speedup_data.empty:
        with st.expander(
            "ECharts 5 pilot (clic para resaltar fila de timings en Graph Analytics)",
            expanded=False,
        ):
            st.caption(
                "Piloto adicional: ECharts 5 sobre Components v2 con click callbacks. "
                "Haz clic en una barra para resaltar la fila del algoritmo en la tabla."
            )
            graph_clicked = render_v2_echarts(
                _echarts_graph_speedup_option(graph_speedup_data),
                key="gpu_benchmark_graph_speedup_echarts",
                height_px=380,
            )
            if isinstance(graph_clicked, dict):
                payload = graph_clicked.get("data")
                if isinstance(payload, dict):
                    task_id = payload.get("task_id")
                    backend_id = payload.get("backend_id")
                    if isinstance(task_id, str) and task_id:
                        st.session_state["gpu_benchmark_graph_focus_task"] = task_id
                        graph_focus_task = task_id
                    if isinstance(backend_id, str) and backend_id:
                        st.session_state["gpu_benchmark_graph_focus_backend"] = backend_id
                        graph_focus_backend = backend_id

            if isinstance(graph_focus_task, str) and graph_focus_task:
                focus_label = graph_focus_task.replace("_", " ").title()
                backend_label = (
                    _label(str(graph_focus_backend))
                    if isinstance(graph_focus_backend, str) and graph_focus_backend
                    else "cualquier backend"
                )
                st.caption(f"Foco actual: `{focus_label}` (último click: {backend_label})")
            else:
                st.caption("Sin foco activo. Usa el gráfico ECharts para seleccionar un algoritmo.")

            if st.button("Limpiar foco de Graph Analytics", key="reset_graph_echarts_focus"):
                st.session_state.pop("gpu_benchmark_graph_focus_task", None)
                st.session_state.pop("gpu_benchmark_graph_focus_backend", None)
                graph_focus_task = None
                graph_focus_backend = None

    # -- Timing comparison table --
    pivot_cols = ["task", "backend", "seconds"]
    if all(c in gr_bench.columns for c in pivot_cols):
        timing = gr_bench[pivot_cols].copy()
        timing["backend"] = timing["backend"].map(_label)
        pivot = timing.pivot(index="task", columns="backend", values="seconds")
        ordered_cols = [
            _label("networkx_cpu"),
            _label("nx_cugraph_gpu"),
            _label("cugraph_gpu"),
        ]
        pivot = pivot.reindex(columns=[c for c in ordered_cols if c in pivot.columns])

        selected_task = (
            str(graph_focus_task)
            if isinstance(graph_focus_task, str) and graph_focus_task in pivot.index
            else None
        )
        selected_backend_label = (
            _label(str(graph_focus_backend))
            if isinstance(graph_focus_backend, str) and graph_focus_backend
            else None
        )

        styler = pivot.style.format(lambda v: f"{float(v):.4f}s" if pd.notna(v) else "—")
        if selected_task:
            st.caption(f"Fila resaltada desde ECharts: `{selected_task.replace('_', ' ').title()}`")

            def _timing_highlight(data: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=data.index, columns=data.columns)
                styles.loc[selected_task, :] = "background-color: rgba(11, 94, 215, 0.10);"
                if selected_backend_label in styles.columns:
                    styles.loc[selected_task, selected_backend_label] = (
                        "background-color: rgba(11, 94, 215, 0.18);"
                        " outline: 2px solid #0B5ED7;"
                        " font-weight: 600;"
                    )
                return styles

            styler = styler.apply(_timing_highlight, axis=None)

        st.dataframe(styler, width="stretch")

    st.markdown(
        """
### The Headline Numbers

| Algorithm | NetworkX CPU | nx-cugraph | cuGraph Direct | Speedup |
|-----------|-------------|------------|----------------|---------|
| **Betweenness Centrality** | 179.5s | 1.97s | 1.10s | **91x / 164x** |
| **Louvain Community** | 8.07s | 0.82s | 0.07s | **10x / 120x** |
| **PageRank** | 1.59s | 2.73s* | 0.13s | **--/ 12x** |
| **Connected Components** | 0.09s | 0.39s* | 0.02s | **--/ 5x** |

*nx-cugraph overhead includes graph conversion to GPU format on first call.*

### How Each Algorithm Serves the Project

**Betweenness Centrality** (164x) — Identifies which loan attributes *bridge*
between risk groups. A sub_grade like "C3" with high betweenness might be the
tipping point between Stage 1 and Stage 2 in IFRS9 classification. At 179s on
CPU this is impractical for exploration; at 1.1s on GPU it becomes interactive.

**Louvain Community Detection** (120x) — Discovers natural *risk clusters*
among borrowers. Instead of the predefined grade system (A-G), Louvain can find
data-driven groupings that better predict default — useful for the Mondrian
conformal prediction groups in `generate_conformal_intervals.py`.

**PageRank** (12x) — Ranks loan attributes by "influence" in the default network.
High-PageRank purposes or grades disproportionately connect to defaults — a signal
for `run_fairness_audit.py` to investigate potential bias.

**Connected Components** (5x) — A data integrity check: one connected component
means all attributes are reachable from any loan. Multiple components would signal
isolated subpopulations needing separate models.

**Practical recommendation:** Set `NX_CUGRAPH_AUTOCONFIG=True` and let NetworkX
dispatch to GPU automatically. The conversion cost is amortized across algorithm calls.
"""
    )

    download_table(gr_bench, "graph_benchmark.csv", "Download graph results")
else:
    st.info("No graph benchmark data found.")

# ===================================================================
# 4. Portfolio Optimization
# ===================================================================

st.markdown("---")
st.header("4. Portfolio Optimization: cuOpt vs HiGHS at Scale")

st.markdown(
    """
> **Where this matters in our pipeline:**
> This is the **core of the thesis contribution**. `optimize_portfolio.py` solves
> LP/MILP problems to select optimal loan portfolios using conformal prediction
> intervals as uncertainty sets. `optimize_portfolio_tradeoff.py` traces the
> Pareto frontier across risk budgets — each point requires a fresh LP solve.
> Faster LP solving = more points on the frontier = better risk-return tradeoffs.

We solve the same portfolio selection problem at increasing scale using real loan
data from `train.parquet`:
- **Objective**: maximize expected return (interest rate)
- **Constraints**: budget (30% of total), risk (PD-weighted <= 15%), max 5% per loan
- **Variables**: 3K -> 6K -> 12K -> 18K loans (the thesis uses up to 5K candidates)
"""
)

if not opt_bench.empty and "task" in opt_bench.columns:
    lp_data = opt_bench[opt_bench["task"] == "portfolio_lp"].copy()
    milp_data = opt_bench[opt_bench["task"] == "portfolio_milp"].copy()
    lp_perf_summary_md = "**LP performance snapshot.** No se pudo calcular una comparación CPU vs GPU con datos suficientes."

    if not lp_data.empty:
        valid = lp_data[lp_data["seconds"].notna()].copy()
        valid["n_variables"] = valid["n_variables"].astype(int)
        valid["backend_label"] = valid["backend"].map(_label)

        # -- Scaling chart --
        fig = px.line(
            valid,
            x="n_variables",
            y="seconds",
            color="backend_label",
            markers=True,
            color_discrete_map={_label(k): v for k, v in COLORS.items()},
            labels={
                "n_variables": "Number of Variables (loans)",
                "seconds": "Solve Time (s)",
                "backend_label": "Solver",
            },
            title="LP Solve Time: SciPy HiGHS (CPU) vs cuOpt (GPU)",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
        st.plotly_chart(fig, width="stretch")

        cpu_lp = valid[valid["backend"] == "scipy_highs_cpu"][["n_variables", "seconds"]].rename(
            columns={"seconds": "cpu_seconds"}
        )
        gpu_lp_perf = valid[valid["backend"] == "cuopt_gpu"][["n_variables", "seconds"]].rename(
            columns={"seconds": "gpu_seconds"}
        )
        paired_lp = cpu_lp.merge(gpu_lp_perf, on="n_variables", how="inner")
        if not paired_lp.empty:
            paired_lp["speedup_ratio_cpu_over_gpu"] = paired_lp["cpu_seconds"] / paired_lp[
                "gpu_seconds"
            ].clip(lower=1e-12)
            min_ratio = float(paired_lp["speedup_ratio_cpu_over_gpu"].min())
            max_ratio = float(paired_lp["speedup_ratio_cpu_over_gpu"].max())
            if max_ratio < 1.0:
                lp_perf_summary_md = (
                    "**LP performance snapshot.** En este benchmark, `SciPy HiGHS` supera a `cuOpt` en todos "
                    f"los tamaños probados. cuOpt corre a ~`{min_ratio:.3f}x`–`{max_ratio:.3f}x` de la velocidad "
                    "de HiGHS (equivale a ~"
                    f"`{1 / max_ratio:.1f}x`–`{1 / min_ratio:.1f}x` más lento)."
                )
            elif min_ratio > 1.0:
                lp_perf_summary_md = (
                    "**LP performance snapshot.** En este benchmark, `cuOpt` supera a `HiGHS` en todos "
                    f"los tamaños probados con ~`{min_ratio:.2f}x`–`{max_ratio:.2f}x` speedup."
                )
            else:
                lp_perf_summary_md = (
                    "**LP performance snapshot.** El rendimiento cruza la paridad según el tamaño: "
                    "hay tamaños donde gana HiGHS y otros donde gana cuOpt."
                )

        # -- Speedup by size --
        if "speedup_vs_cpu_lp" in valid.columns:
            gpu_lp = valid[valid["backend"] == "cuopt_gpu"].dropna(subset=["speedup_vs_cpu_lp"])
            if not gpu_lp.empty:
                gpu_lp = gpu_lp.copy()
                gpu_lp["speedup_vs_cpu_lp"] = gpu_lp["speedup_vs_cpu_lp"].astype(float)
                gpu_lp["relative_factor_vs_highs"] = np.where(
                    gpu_lp["speedup_vs_cpu_lp"] >= 1.0,
                    gpu_lp["speedup_vs_cpu_lp"],
                    1.0 / np.clip(gpu_lp["speedup_vs_cpu_lp"], 1e-12, None),
                )
                gpu_lp["relative_status"] = np.where(
                    gpu_lp["speedup_vs_cpu_lp"] >= 1.0, "cuOpt faster", "cuOpt slower"
                )
                gpu_lp["relative_label"] = np.where(
                    gpu_lp["speedup_vs_cpu_lp"] >= 1.0,
                    gpu_lp["relative_factor_vs_highs"].map(lambda v: f"{v:.2f}x faster"),
                    gpu_lp["relative_factor_vs_highs"].map(lambda v: f"{v:.1f}x slower"),
                )

                fig2 = px.bar(
                    gpu_lp,
                    x="n_variables",
                    y="relative_factor_vs_highs",
                    color="relative_status",
                    color_discrete_map={"cuOpt faster": "#0B5ED7", "cuOpt slower": "#E45756"},
                    labels={
                        "n_variables": "Variables",
                        "relative_factor_vs_highs": "Factor vs HiGHS (x)",
                        "relative_status": "",
                    },
                    title="cuOpt vs HiGHS by Problem Size (distance from parity)",
                    text="relative_label",
                )
                fig2.add_hline(
                    y=1.0, line_dash="dash", line_color="#E45756", annotation_text="1x = parity"
                )
                fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=350, showlegend=False)
                fig2.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig2, width="stretch")
                if (gpu_lp["speedup_vs_cpu_lp"] < 1.0).all():
                    min_ratio = float(gpu_lp["speedup_vs_cpu_lp"].min())
                    max_ratio = float(gpu_lp["speedup_vs_cpu_lp"].max())
                    st.caption(
                        "En este snapshot de benchmark, cuOpt está por debajo de HiGHS en LP "
                        f"(~{1 / max_ratio:.1f}x a ~{1 / min_ratio:.1f}x más lento según tamaño). "
                        "La gráfica muestra factor relativo (más rápido o más lento) para evitar barras invisibles."
                    )
                else:
                    st.caption(
                        "La gráfica muestra factor relativo vs HiGHS: >1 significa que cuOpt es más rápido; "
                        "<1 en el ratio original se traduce a `x slower` para mejor legibilidad."
                    )

    # -- MILP comparison --
    if not milp_data.empty:
        st.subheader("MILP (Binary Portfolio Selection)")
        st.markdown(
            "Binary selection (invest or don't) maps to the **causal portfolio** "
            "in `optimize_cate_portfolio.py`, where CATE-adjusted binary decisions "
            "determine which loans to approve."
        )
        for _, r in milp_data.iterrows():
            backend = _label(r.get("backend", ""))
            secs = _sf(r.get("seconds"))
            obj = _sf(r.get("objective"))
            st.markdown(f"- **{backend}**: {secs:.3f}s, objective = {obj:,.2f}")

    # -- Objective agreement --
    if not lp_data.empty:
        cpu_mask = lp_data["backend"] == "scipy_highs_cpu"
        cpu_objs = lp_data[cpu_mask].set_index("n_variables")["objective"]
        gpu_objs = lp_data[lp_data["backend"] == "cuopt_gpu"].set_index("n_variables")["objective"]
        diffs = []
        for nv in cpu_objs.index:
            if nv in gpu_objs.index:
                co, go = _sf(cpu_objs[nv]), _sf(gpu_objs[nv])
                rel = abs(co - go) / max(abs(co), 1e-10) * 100
                diffs.append(
                    {
                        "Variables": int(nv),
                        "CPU Obj": f"{co:.6f}",
                        "GPU Obj": f"{go:.6f}",
                        "Rel Diff %": f"{rel:.4f}%",
                    }
                )
        if diffs:
            st.markdown("**Objective agreement (CPU vs GPU):**")
            st.dataframe(pd.DataFrame(diffs), width="stretch", hide_index=True)

    st.markdown(
        f"""
### Analysis

{lp_perf_summary_md}

**Objectives match perfectly.** Both solvers find the same optimal value to 6+ decimal
places. This is critical: a fast but incorrect solver is useless for portfolio allocation
where every basis point matters.

**MILP is slower on GPU** (1.9s vs 0.1s for 3K binary variables). HiGHS has decades
of cutting-plane and presolve heuristics. cuOpt's GPU branch-and-bound hasn't caught up
yet for small MILPs.

### Direct Pipeline Impact

| Pipeline stage | Current solver | cuOpt benefit |
|----------------|---------------|---------------|
| `optimize_portfolio.py` (5K LP) | HiGHS ~0.02s | ~0.01s — marginal at this size |
| `optimize_portfolio_tradeoff.py` (100+ LPs) | HiGHS ~2s | ~0.7s — 3x faster Pareto |
| `robust_opt.py` (uncertainty sets) | HiGHS/scenario | cuOpt for 10K+ var robust LPs |
| `optimize_cate_portfolio.py` (MILP) | HiGHS ~0.1s | GPU slower — keep HiGHS for MILP |
"""
    )

    download_table(opt_bench, "optimization_benchmark.csv", "Download optimization results")
else:
    st.info("No optimization benchmark data found.")

# ===================================================================
# 5. Numerical Computing (CuPy)
# ===================================================================

st.markdown("---")
st.header("5. Numerical Computing: CuPy for Monte Carlo and Linear Algebra")

st.markdown(
    """
> **Where this matters in our pipeline:**
> `run_ifrs9_sensitivity.py` computes **Expected Credit Loss** (ECL = PD x LGD x EAD)
> across macroeconomic scenarios. The thesis evaluates base, adverse, and severely adverse
> scenarios; a Monte Carlo extension would simulate thousands of correlated scenarios.
> SVD is the core of PCA and feature decomposition. Sparse matrix operations appear in
> graph algorithms and regularized models.

**CuPy** is a drop-in NumPy/SciPy replacement that runs on GPU. Same API,
GPU execution. We test three operations central to credit risk modeling:
"""
)

if not cp_bench.empty and "task" in cp_bench.columns:
    CUPY_CONTEXT = {
        "monte_carlo_ecl": (
            "**ECL Monte Carlo** is the IFRS9 backbone. Our `run_ifrs9_sensitivity.py` "
            "currently computes ECL for 3 deterministic scenarios. With CuPy, we could "
            "run **100K stochastic scenarios** (varying PD, LGD, EAD jointly) to build "
            "a full loss distribution — VaR, CVaR, and tail risk metrics that regulators "
            "increasingly demand. The 2.1x speedup makes this feasible in production. "
            "With batching (processing 10K loans at a time), even the full 1.35M-loan "
            "portfolio is tractable."
        ),
        "sparse_matmul": (
            "**Sparse matrix multiply** is the core operation in graph adjacency "
            "computations and L1/L2 regularization solvers. cuSPARSE handles CSR "
            "format natively — the same format used by scipy.sparse in our pipeline. "
            "At 3.7x speedup, GPU sparse ops would accelerate any future graph-based "
            "risk model (e.g., GNN-based default prediction)."
        ),
        "svd": (
            "**SVD** decomposes the feature matrix — the foundation of PCA in NB02. "
            "At 100K x 47, the matrix is too narrow for GPU to win (cuSOLVER kernel "
            "launch overhead > compute savings). For wider matrices (500+ columns) or "
            "taller matrices (1M+ rows), CuPy SVD would dominate."
        ),
    }

    for task_name in sorted(cp_bench["task"].unique()):
        task_data = cp_bench[cp_bench["task"] == task_name].copy()
        valid = task_data[task_data["seconds"].notna()]

        if valid.empty:
            continue

        c1, c2 = st.columns([3, 1])

        with c1:
            valid_plot = valid.copy()
            valid_plot["backend_label"] = valid_plot["backend"].map(_label)
            fig = px.bar(
                valid_plot,
                x="backend_label",
                y="seconds",
                color="backend",
                color_discrete_map=COLORS,
                title=f"{task_name.replace('_', ' ').title()}",
                labels={"backend_label": "", "seconds": "Seconds"},
                text_auto=".3f",
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

        with c2:
            speedup = task_data["speedup_vs_cpu"].dropna()
            if not speedup.empty:
                s = speedup.values[0]
                st.metric(
                    "Speedup",
                    f"{s:.1f}x",
                    delta="faster" if s > 1 else "slower",
                    delta_color="normal" if s > 1 else "inverse",
                )

        ctx = CUPY_CONTEXT.get(task_name)
        if ctx:
            st.info(ctx)

    st.markdown(
        """
### Summary

| Task | CPU | GPU | Speedup | Scale tested |
|------|-----|-----|---------|-------------|
| **Monte Carlo ECL** | 34.2s | 16.6s | **2.1x** | 100K scenarios x 10K loans |
| **Sparse MatMul** | 1.5s | 0.4s | **3.7x** | 10K x 10K CSR, density 1% |
| **SVD** | 0.3s | 0.8s | **0.4x** | 100K x 47 (too narrow for GPU) |
"""
    )

    download_table(cp_bench, "cupy_benchmark.csv", "Download CuPy results")
else:
    st.info("No CuPy benchmark data found.")

# ===================================================================
# 6. Decision Matrix & Conclusions
# ===================================================================

st.markdown("---")
st.header("6. The Verdict: When to Use GPU for Credit Risk")

st.markdown(
    """
### Decision Matrix for Our Pipeline

| Pipeline Stage | Script | Best Tool | Speedup | Effort |
|----------------|--------|-----------|---------|--------|
| Data cleaning | `make_dataset.py` | **cudf.pandas** | 13x | Zero code changes |
| Feature engineering | `feature_engineering.py` | **Polars** | 34x | API rewrite |
| dbt analytics | `dbt_project/` | **DuckDB** | 12x | Already using it |
| PD model training | `train_pd_model.py` | **CatBoost GPU** | ~5-8x | `task_type='GPU'` |
| LR baseline | `train_pd_model.py` | **sklearn CPU** | -- | Keep CPU (convergence) |
| Conformal intervals | `generate_conformal_intervals.py` | CPU | -- | No MAPIE GPU support |
| Portfolio LP | `optimize_portfolio.py` | **cuOpt** (>5K) | 3x | API change |
| Portfolio MILP | `optimize_cate_portfolio.py` | **HiGHS CPU** | -- | Keep CPU |
| IFRS9 ECL | `run_ifrs9_sensitivity.py` | **CuPy** | 2x+ | `np.` -> `cp.` |
| Graph risk analysis | Future work | **cuGraph** | 12-164x | cuGraph API |
| Anomaly detection | Future work | **cuML HDBSCAN** | 77x | cuML API |
| Borrower embedding | Streamlit viz | **cuML UMAP** | 50-100x | cuML API |

### Top 3 Takeaways

**1. The highest-ROI change is `cudf.pandas.install()`.** One line of code, zero
other changes, 13x speedup on all pandas operations in the pipeline. This should
be the first thing deployed.

**2. GPU shines on tree/graph/density algorithms.** Random Forest (7.5x),
HDBSCAN (77x), Louvain (120x), and Betweenness Centrality (164x) are
transformative. If we extend the thesis to graph-based risk models, a GPU
pays for itself immediately.

**3. Not everything benefits from GPU.** Logistic Regression (convergence issues),
PCA (too narrow), KMeans (too small), and MILP (HiGHS heuristics win) are faster
or more reliable on CPU. **The right answer is a hybrid pipeline** that dispatches
each stage to the best hardware.
"""
)

# ===================================================================
# 7. Hardware & Methodology
# ===================================================================

st.markdown("---")

with st.expander("Hardware & Methodology"):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
**Hardware:**
| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 3080 (10 GB GDDR6X) |
| CPU | AMD Ryzen 5 5600X (6-core, 12-thread) |
| RAM | 24 GB DDR4 |
| Platform | WSL2 (Windows Subsystem for Linux) |
| CUDA | 13.1 (driver 591.86) |
"""
        )

    with c2:
        st.markdown(
            """
**Methodology:**
- All benchmarks run on the same machine, same session
- DataFrame: median of 3 runs with 1 warmup
- ML/Graph/CuPy: single run (training is expensive)
- GPU sync: explicit `cp.cuda.Stream.null.synchronize()`
- Quality checks: verify outputs match between CPU and GPU
"""
        )

    st.markdown(
        """
**Datasets used:**
| Benchmark | Source | Rows | Features |
|-----------|--------|------|----------|
| DataFrame | `lending_club_cleaned.parquet` | 1,860,764 | 110 (11 selected) |
| ML | `train_fe.parquet` | up to 500,000 | 47 numeric |
| Graph | `train.parquet` | 200,000 (capped) | 5 (id + 4 attrs) |
| Optimization | `train.parquet` | 3K-18K subsets | 3 (rate, PD, amount) |
| CuPy | Synthetic + `train_fe.parquet` | 100K scenarios | varies |
"""
    )

    # Library versions
    if meta and "versions" in meta:
        ver = meta["versions"]
        ver_df = pd.DataFrame([{"Library": k, "Version": v} for k, v in sorted(ver.items())])
        st.dataframe(ver_df, width="stretch", hide_index=True)

# -- Footer --
st.markdown("---")
st.caption(
    "This benchmark is a **side project** independent of the main thesis pipeline. "
    "Results generated with RAPIDS 26.02 on a consumer RTX 3080 under WSL2. "
    "Scripts: `reports/gpu_benchmark/tmp_scripts/`."
)
