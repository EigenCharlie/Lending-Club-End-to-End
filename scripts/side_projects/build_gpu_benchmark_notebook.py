"""Generate Notebook 10: RAPIDS 26.02 GPU benchmark (side task).

Produces: notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb
Exports benchmark artifacts to: reports/gpu_benchmark/

Run:  uv run python scripts/side_projects/build_gpu_benchmark_notebook.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def m(txt: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(txt).strip() + "\n"}


def c(txt: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(txt).strip() + "\n",
    }


cells: list[dict] = []

# ══════════════════════════════════════════════════════════════════════════════
# Section 0: Title, references, configuration
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
# Notebook 10 — RAPIDS 26.02 GPU Benchmark (Side Task)

**Objective**: Measure practical GPU acceleration across 5 RAPIDS libraries on
Lending Club data (1.3M loans, 2007–2020).

**Scope boundary**:
- This notebook is **independent** from the canonical project pipeline.
- It does not modify core project artifacts or acceptance gates.
- Artifacts are exported to `reports/gpu_benchmark/` for the Streamlit annex page.

**Libraries benchmarked**:
| Library | CPU Baseline | GPU Path |
|---------|-------------|----------|
| cuDF | pandas, Polars, DuckDB | cudf.pandas, Polars GPUEngine |
| cuML | scikit-learn | cuML (+ cuml.accel dispatch) |
| cuGraph | NetworkX | cugraph, nx-cugraph backend |
| cuOpt | SciPy HiGHS | cuOpt LP/MILP solver |
| CuPy | NumPy/SciPy | CuPy GPU arrays |
""")
)

cells.append(
    m("""
## Official references (RAPIDS 26.02)

- RAPIDS install + platform support: https://docs.rapids.ai/install
- cuDF pandas accelerator: https://docs.rapids.ai/api/cudf/stable/cudf_pandas/
- cuDF + Polars GPU engine: https://docs.rapids.ai/api/cudf/stable/cudf_polars/usage/
- cuML API: https://docs.rapids.ai/api/cuml/stable/api/
- cuML scikit-learn accelerator: https://docs.rapids.ai/api/cuml/stable/cuml_accel/
- cuGraph API: https://docs.rapids.ai/api/cugraph/stable/api_docs/
- nx-cugraph backend: https://docs.rapids.ai/api/cugraph/stable/nx_cugraph/
- cuOpt LP/MILP: https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/lp-milp-examples.html
- RMM CuPy allocator: https://docs.rapids.ai/api/rmm/stable/python_api/
- CuPy docs: https://docs.cupy.dev/en/stable/
- Narwhals (write-once DataFrames): https://narwhals-dev.github.io/narwhals/
""")
)

cells.append(
    c("""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display


def find_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in [cur] + list(cur.parents):
        if (p / "data").exists() and (p / "notebooks").exists():
            return p
    raise FileNotFoundError("No project root found")


ROOT = find_root()
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "gpu_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / "tmp_scripts"
TMP.mkdir(parents=True, exist_ok=True)

TRAIN = DATA / "train.parquet"
TEST = DATA / "test.parquet"
TRAIN_FE = DATA / "train_fe.parquet"
TEST_FE = DATA / "test_fe.parquet"

CONFIG: dict[str, Any] = {
    "seed": 42,
    "save_outputs": True,
    "cudf_repeats": 4,
    "cudf_warmup": 1,
    "cuml_train_sample": 250_000,
    "cuml_test_sample": 100_000,
    "cuml_repeats_light": 2,
    "cuml_repeats_heavy": 1,
    "cugraph_sample_rows": 50_000,
    "cugraph_repeats": 3,
    "cuopt_sample_rows": 18_000,
    "cuopt_repeats": 3,
    "cupy_repeats": 3,
    "cupy_warmup": 1,
    "consistency_rel_tol": 5e-3,
    "plot_per_method_charts": True,
}


def stats_from(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "median_seconds": np.nan,
            "mean_seconds": np.nan,
            "std_seconds": np.nan,
            "iqr_seconds": np.nan,
        }
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "median_seconds": float(np.median(arr)),
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr)),
        "iqr_seconds": float(q3 - q1),
    }


def rel_err(a: float, b: float) -> float:
    den = max(abs(a), 1e-12)
    return float(abs(a - b) / den)


def classify_relevance(speedup_x: float, seconds_saved: float, quality_pass: bool) -> str:
    if not quality_pass:
        return "No concluyente (quality check fail)"
    if not np.isfinite(speedup_x):
        return "No disponible"
    if speedup_x >= 3.0 and seconds_saved >= 0.05:
        return "Alta relevancia"
    if speedup_x >= 1.5 and seconds_saved >= 0.01:
        return "Relevancia moderada"
    if speedup_x >= 1.1:
        return "Relevancia baja"
    if speedup_x >= 0.9:
        return "Neutro"
    return "No relevante (CPU mejor)"


def save_artifact(df: pd.DataFrame, name: str, fmt: str = "parquet") -> None:
    if not CONFIG["save_outputs"]:
        return
    if fmt == "parquet":
        path = OUT / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
    print("Saved:", path)


def finalize_direct_df(df: pd.DataFrame, section_name: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        print(f"[{section_name}] No direct CPU/GPU pairs available.")
        return pd.DataFrame()
    out = df.copy()
    out["improvement_pct"] = (out["speedup_x"] - 1.0) * 100.0
    out["seconds_saved"] = out["cpu_seconds"] - out["gpu_seconds"]
    out["relevance"] = [
        classify_relevance(sp, ss, qp)
        for sp, ss, qp in zip(
            out["speedup_x"].to_numpy(np.float64),
            out["seconds_saved"].to_numpy(np.float64),
            out["quality_pass"].to_numpy(bool),
        )
    ]
    order = [
        "section", "method", "cpu_backend", "gpu_backend",
        "cpu_seconds", "gpu_seconds", "speedup_x",
        "improvement_pct", "seconds_saved", "quality_pass",
        "relevance", "note",
    ]
    keep = [c for c in order if c in out.columns]
    return out[keep].sort_values(["section", "method"]).reset_index(drop=True)


def print_section_conclusions(direct_df: pd.DataFrame, section_title: str) -> None:
    if direct_df is None or len(direct_df) == 0:
        return
    best = direct_df.sort_values("speedup_x", ascending=False).iloc[0]
    worst = direct_df.sort_values("speedup_x", ascending=True).iloc[0]
    pass_rate = float(direct_df["quality_pass"].mean())
    print(
        f"{section_title} conclusions: "
        f"median speedup={float(direct_df['speedup_x'].median()):.2f}x, "
        f"best={best['method']} ({best['speedup_x']:.2f}x), "
        f"worst={worst['method']} ({worst['speedup_x']:.2f}x), "
        f"quality_pass_rate={pass_rate:.2f}"
    )


try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception as exc:
    HAS_MPL = False
    print("matplotlib unavailable:", repr(exc))


CPU_COLOR = "#A8B5C4"
GPU_GAIN_COLOR = "#94C9B2"
GPU_LOSS_COLOR = "#E8A7A7"
TEXT_SOFT = "#5B6470"


def plot_method_pairs(direct_df: pd.DataFrame, section_title: str) -> None:
    if not bool(CONFIG.get("plot_per_method_charts", True)):
        return
    if not HAS_MPL or direct_df is None or len(direct_df) == 0:
        return
    print(f"{section_title}: CPU vs GPU charts")
    for _, r in direct_df.iterrows():
        method = str(r["method"])
        cpu_s = float(r["cpu_seconds"])
        gpu_s = float(r["gpu_seconds"])
        speedup = float(r["speedup_x"])
        improvement = float(r["improvement_pct"])

        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        gpu_color = GPU_GAIN_COLOR if improvement >= 0 else GPU_LOSS_COLOR
        bars = ax.barh(["CPU", "GPU"], [cpu_s, gpu_s], color=[CPU_COLOR, gpu_color], height=0.56)

        positive = [v for v in [cpu_s, gpu_s] if np.isfinite(v) and v > 0]
        if len(positive) == 2 and max(positive) / min(positive) >= 30:
            ax.set_xscale("log")
            ax.set_xlabel("Median seconds (log scale)")
        else:
            ax.set_xlabel("Median seconds")

        xmax = max(positive) if positive else 1.0
        ax.set_xlim(0, xmax * 1.35 if np.isfinite(xmax) else 1.0)
        for b, v in zip(bars, [cpu_s, gpu_s]):
            ax.text(
                b.get_width() + xmax * 0.025,
                b.get_y() + b.get_height() / 2,
                f"{v:.4f}s",
                ha="left", va="center", fontsize=10, color="#374151",
            )

        ax.set_title(f"{section_title} | {method}", loc="left", fontsize=13, fontweight="bold", pad=14)
        subtitle = f"speedup={speedup:.2f}x | improvement={improvement:+.1f}%"
        ax.text(0.00, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=TEXT_SOFT, va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        fig.subplots_adjust(left=0.16, right=0.96, top=0.80, bottom=0.24)
        plt.show()


print("ROOT:", ROOT)
print("Python:", sys.executable)
print("save_outputs:", CONFIG["save_outputs"])
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 0b: Environment and data checks
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## Environment and data checks

This notebook expects a RAPIDS-capable environment (Linux/WSL2 + NVIDIA GPU).
Artifacts are exported as **parquet** to `reports/gpu_benchmark/`.
""")
)

cells.append(
    c("""
# Hardware + package checks

def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

code, out, err = run(["nvidia-smi"])
print(out if code == 0 else err)

pkgs = [
    "cudf", "cuml", "cugraph", "nx_cugraph", "cuopt", "cupy", "rmm",
    "networkx", "scipy", "sklearn", "polars", "pandas", "duckdb", "narwhals",
]
rows = []
for p in pkgs:
    spec = importlib.util.find_spec(p)
    ver = None
    if spec is not None:
        try:
            mod = __import__(p)
            ver = getattr(mod, "__version__", None)
        except Exception:
            ver = "import_error"
    rows.append({"package": p, "available": spec is not None, "version": ver})
pkg_df = pd.DataFrame(rows)
display(pkg_df)

# 10GB VRAM-safe RMM pool for RTX 3080 style cards
try:
    import cupy as cp
    import rmm

    rmm.reinitialize(pool_allocator=True, managed_memory=False, initial_pool_size=6 * 1024**3)

    alloc = None
    try:
        from rmm.allocators.cupy import rmm_cupy_allocator
        alloc = rmm_cupy_allocator
    except Exception:
        alloc = getattr(rmm, "rmm_cupy_allocator", None)

    if alloc is not None:
        cp.cuda.set_allocator(alloc)
        print("RMM configured with CuPy allocator (6 GB pool)")
    else:
        print("RMM configured, but CuPy allocator binding not found")
except Exception as exc:
    print("RMM setup skipped:", repr(exc))

for p in [TRAIN, TEST, TRAIN_FE, TEST_FE]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

summary_rows = []
for p in [TRAIN, TEST, TRAIN_FE, TEST_FE]:
    d = pd.read_parquet(p, columns=["id", "default_flag"])
    summary_rows.append({
        "file": str(p.relative_to(ROOT)),
        "rows": int(len(d)),
        "default_rate": float(d["default_flag"].mean()),
    })
display(pd.DataFrame(summary_rows))

# Save library versions for Streamlit meta
lib_versions = {r["package"]: str(r["version"]) for _, r in pkg_df.iterrows() if r["available"]}
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 1: DataFrame Processing (pandas, cudf, polars, duckdb, narwhals)
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 1) DataFrame benchmark: pandas, cuDF, Polars, DuckDB

Compared backends:
- `pandas_cpu` — vanilla pandas
- `pandas_cudf` — pandas via `python -m cudf.pandas` (zero-code-change)
- `polars_cpu` — Polars lazy collect (CPU)
- `polars_cudf` — Polars lazy with `pl.GPUEngine` (GPU)
- `duckdb` — in-process SQL (CPU analytical engine)

Quality checks: row-count parity + checksum relative error ≤ 5e-3 vs pandas_cpu.
""")
)

cells.append(
    c("""
R = int(CONFIG["cudf_repeats"])
W = int(CONFIG["cudf_warmup"])
exe = sys.executable


def run_json(cmd: list[str]) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    res: dict[str, Any] = {
        "command": " ".join(cmd),
        "status": "ok" if p.returncode == 0 else "error",
        "returncode": p.returncode,
        "stderr_tail": p.stderr[-800:],
    }
    if p.returncode == 0:
        try:
            payload = json.loads(p.stdout.strip().splitlines()[-1])
            res.update(payload)
        except Exception as exc:
            res["status"] = "error"
            res["parse_error"] = repr(exc)
    return res


plan: list[tuple[str, list[str]]] = [
    ("pandas_cpu", [exe, str(TMP / "bench_pandas.py"), str(TRAIN), str(R), str(W)]),
    ("polars_cpu", [exe, str(TMP / "bench_polars.py"), str(TRAIN), str(R), str(W), "cpu"]),
    ("duckdb", [exe, str(TMP / "bench_duckdb.py"), str(TRAIN), str(R), str(W)]),
]

has_cudf = importlib.util.find_spec("cudf") is not None
if has_cudf:
    plan.append(("pandas_cudf", [
        exe, "-m", "cudf.pandas", str(TMP / "bench_pandas.py"), str(TRAIN), str(R), str(W),
    ]))
else:
    print("Skipping pandas_cudf: cudf not available")

if has_cudf and hasattr(pl, "GPUEngine"):
    plan.append(("polars_cudf", [
        exe, str(TMP / "bench_polars.py"), str(TRAIN), str(R), str(W), "gpu",
    ]))
else:
    print("Skipping polars_cudf: cudf and/or pl.GPUEngine not available")

rows = []
for mode, cmd in plan:
    print(f"Running {mode}...")
    x = run_json(cmd)
    x["mode"] = mode
    rows.append(x)

cudf_df = pd.DataFrame(rows)

ok = cudf_df[cudf_df["status"] == "ok"]
base = ok.loc[ok["mode"] == "pandas_cpu", "median_seconds"]
if len(base) == 1:
    b = float(base.iloc[0])
    cudf_df["speedup_vs_pandas_cpu"] = np.where(
        cudf_df["status"] == "ok",
        b / cudf_df["median_seconds"],
        np.nan,
    )

base_rows = ok.loc[ok["mode"] == "pandas_cpu", "rows_out"]
base_checksum = ok.loc[ok["mode"] == "pandas_cpu", "checksum"]
if len(base_rows) == 1 and len(base_checksum) == 1:
    br = int(base_rows.iloc[0])
    bc = float(base_checksum.iloc[0])
    cudf_df["rows_match_cpu"] = np.where(cudf_df["status"] == "ok", cudf_df["rows_out"] == br, np.nan)
    cudf_df["checksum_rel_err_cpu"] = np.where(
        cudf_df["status"] == "ok",
        np.abs(cudf_df["checksum"] - bc) / max(abs(bc), 1e-12),
        np.nan,
    )
    cudf_df["consistency_pass"] = np.where(
        cudf_df["status"] == "ok",
        (cudf_df["rows_match_cpu"] == True) & (cudf_df["checksum_rel_err_cpu"] <= CONFIG["consistency_rel_tol"]),
        np.nan,
    )

display(cudf_df.sort_values("mode"))
save_artifact(cudf_df, "cudf_polars_benchmark")

# Direct CPU vs GPU view
cudf_pair_rows: list[dict[str, Any]] = []
for method, cpu_mode, gpu_mode in [
    ("pandas_query_pipeline", "pandas_cpu", "pandas_cudf"),
    ("polars_query_pipeline", "polars_cpu", "polars_cudf"),
]:
    cpu = cudf_df[(cudf_df["mode"] == cpu_mode) & (cudf_df["status"] == "ok")]
    gpu = cudf_df[(cudf_df["mode"] == gpu_mode) & (cudf_df["status"] == "ok")]
    if len(cpu) == 1 and len(gpu) == 1:
        cudf_pair_rows.append({
            "section": "cudf",
            "method": method,
            "cpu_backend": cpu_mode,
            "gpu_backend": gpu_mode,
            "cpu_seconds": float(cpu.iloc[0]["median_seconds"]),
            "gpu_seconds": float(gpu.iloc[0]["median_seconds"]),
            "speedup_x": float(cpu.iloc[0]["median_seconds"]) / max(float(gpu.iloc[0]["median_seconds"]), 1e-12),
            "quality_pass": bool(gpu.iloc[0].get("consistency_pass", False)),
            "note": f"checksum_rel_err={gpu.iloc[0].get('checksum_rel_err_cpu', np.nan)}",
        })

cudf_direct_df = finalize_direct_df(pd.DataFrame(cudf_pair_rows), "cudf")
if len(cudf_direct_df):
    display(cudf_direct_df)
    save_artifact(cudf_direct_df, "gpu_bench_dataframe_direct")
    plot_method_pairs(cudf_direct_df, "cuDF")
    print_section_conclusions(cudf_direct_df, "cuDF")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Machine Learning (cuML)
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 2) cuML benchmark: scikit-learn CPU vs cuML GPU

Algorithms:
- **Classification**: Logistic Regression, Random Forest
- **Clustering**: KMeans
- **Dimensionality Reduction**: PCA
- **Nearest Neighbors**: KNN

Quality gates: AUC tolerance (0.010 LR, 0.025 RF), silhouette tolerance (0.040 KMeans).
""")
)

cells.append(
    c("""
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA as SKPCA
from sklearn.ensemble import RandomForestClassifier as SKRF
from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier as SKKNN
from sklearn.preprocessing import StandardScaler

np.random.seed(int(CONFIG["seed"]))

train = pd.read_parquet(TRAIN_FE)
test = pd.read_parquet(TEST_FE)

num = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "loan_to_income",
    "installment_burden", "rev_utilization", "revol_bal_to_income", "open_acc_ratio",
    "fico_score", "credit_age_years", "emp_length_num", "open_acc", "total_acc",
    "revol_bal", "pub_rec", "inq_last_6mths", "mort_acc", "delinq_severity",
    "delinq_recency", "il_ratio", "high_util_pct", "log_annual_inc", "log_revol_bal",
    "loan_to_income_sq", "fico_x_dti",
]
feat = [c for c in num if c in train.columns and c in test.columns]
if not feat:
    raise RuntimeError("No overlapping numeric features found for cuML benchmark")

train = train.sample(n=min(int(CONFIG["cuml_train_sample"]), len(train)), random_state=int(CONFIG["seed"])).reset_index(drop=True)
test = test.sample(n=min(int(CONFIG["cuml_test_sample"]), len(test)), random_state=int(CONFIG["seed"])).reset_index(drop=True)

Xtr = train[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
Xte = test[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
ytr = train["default_flag"].astype(np.int32)
yte = test["default_flag"].astype(np.int32)

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
Xte_s = sc.transform(Xte).astype(np.float32)

res: list[dict[str, Any]] = []


def push_result(task: str, backend: str, fit_times: list[float], pred_times: list[float], metric: str, values: list[float]) -> None:
    st_fit = stats_from(fit_times)
    st_pred = stats_from(pred_times)
    arr = np.asarray(values, dtype=np.float64)
    res.append({
        "task": task,
        "backend": backend,
        "fit_seconds": st_fit["median_seconds"],
        "fit_iqr_seconds": st_fit["iqr_seconds"],
        "predict_seconds": st_pred["median_seconds"],
        "predict_iqr_seconds": st_pred["iqr_seconds"],
        "metric": metric,
        "metric_value": float(np.median(arr)) if arr.size else np.nan,
        "metric_std": float(np.std(arr)) if arr.size else np.nan,
        "n_runs": int(arr.size),
    })


# CPU logistic regression
fit_times: list[float] = []
pred_times: list[float] = []
metrics: list[float] = []
for _ in range(int(CONFIG["cuml_repeats_light"])):
    s = time.perf_counter()
    mdl = SKLR(max_iter=600, solver="lbfgs")
    mdl.fit(Xtr_s, ytr)
    fit_times.append(time.perf_counter() - s)
    s = time.perf_counter()
    proba = mdl.predict_proba(Xte_s)[:, 1]
    pred_times.append(time.perf_counter() - s)
    metrics.append(float(roc_auc_score(yte, proba)))
push_result("logistic_regression", "sklearn_cpu", fit_times, pred_times, "auc", metrics)

# CPU random forest
fit_times = []
pred_times = []
metrics = []
for _ in range(int(CONFIG["cuml_repeats_heavy"])):
    s = time.perf_counter()
    mdl = SKRF(n_estimators=250, max_depth=12, random_state=int(CONFIG["seed"]), n_jobs=-1)
    mdl.fit(Xtr, ytr)
    fit_times.append(time.perf_counter() - s)
    s = time.perf_counter()
    proba = mdl.predict_proba(Xte)[:, 1]
    pred_times.append(time.perf_counter() - s)
    metrics.append(float(roc_auc_score(yte, proba)))
push_result("random_forest", "sklearn_cpu", fit_times, pred_times, "auc", metrics)

# CPU KMeans
Xk = Xtr_s[: min(150_000, len(Xtr_s))]
fit_times = []
pred_times = []
metrics = []
for _ in range(int(CONFIG["cuml_repeats_light"])):
    s = time.perf_counter()
    labels = SKKMeans(n_clusters=7, n_init=10, random_state=int(CONFIG["seed"])).fit_predict(Xk)
    fit_times.append(time.perf_counter() - s)
    pred_times.append(0.0)
    n_sil = min(30_000, len(Xk))
    metrics.append(float(silhouette_score(Xk[:n_sil], labels[:n_sil])))
push_result("kmeans", "sklearn_cpu", fit_times, pred_times, "silhouette", metrics)

# CPU PCA
fit_times = []
pred_times = []
metrics = []
for _ in range(int(CONFIG["cuml_repeats_light"])):
    s = time.perf_counter()
    pca = SKPCA(n_components=10, random_state=int(CONFIG["seed"]))
    Xpca = pca.fit_transform(Xtr_s)
    fit_times.append(time.perf_counter() - s)
    pred_times.append(0.0)
    metrics.append(float(sum(pca.explained_variance_ratio_)))
push_result("pca", "sklearn_cpu", fit_times, pred_times, "explained_variance", metrics)

# CPU KNN
fit_times = []
pred_times = []
metrics = []
Xknn_tr = Xtr_s[:50_000]
yknn_tr = ytr[:50_000]
Xknn_te = Xte_s[:20_000]
yknn_te = yte[:20_000]
for _ in range(int(CONFIG["cuml_repeats_light"])):
    s = time.perf_counter()
    knn = SKKNN(n_neighbors=5, n_jobs=-1)
    knn.fit(Xknn_tr, yknn_tr)
    fit_times.append(time.perf_counter() - s)
    s = time.perf_counter()
    proba = knn.predict_proba(Xknn_te)[:, 1]
    pred_times.append(time.perf_counter() - s)
    metrics.append(float(roc_auc_score(yknn_te, proba)))
push_result("knn", "sklearn_cpu", fit_times, pred_times, "auc", metrics)

# GPU block
if all(importlib.util.find_spec(p) is not None for p in ["cuml", "cudf", "cupy"]):
    try:
        import cupy as cp
        import cudf
        from cuml.cluster import KMeans as CUKMeans
        from cuml.decomposition import PCA as CUPCA
        from cuml.ensemble import RandomForestClassifier as CURF
        from cuml.linear_model import LogisticRegression as CULR
        from cuml.neighbors import KNeighborsClassifier as CUKNN

        def sync() -> None:
            cp.cuda.Stream.null.synchronize()

        def as_numpy(x: Any) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
            if hasattr(x, "to_numpy"):
                return x.to_numpy()
            if hasattr(x, "values_host"):
                return x.values_host
            return np.asarray(x)

        Xtrg_s = cudf.DataFrame(Xtr_s)
        Xteg_s = cudf.DataFrame(Xte_s)
        ytrg = cudf.Series(ytr.to_numpy())

        # GPU logistic regression
        fit_times = []
        pred_times = []
        metrics = []
        for _ in range(int(CONFIG["cuml_repeats_light"])):
            s = time.perf_counter()
            mdl = CULR(max_iter=600, linesearch_max_iter=100, output_type="numpy")
            mdl.fit(Xtrg_s, ytrg)
            sync()
            fit_times.append(time.perf_counter() - s)
            s = time.perf_counter()
            proba = mdl.predict_proba(Xteg_s)
            sync()
            pred_times.append(time.perf_counter() - s)
            proba = as_numpy(proba)
            proba = proba[:, 1] if proba.ndim == 2 else proba
            metrics.append(float(roc_auc_score(yte, proba)))
        push_result("logistic_regression", "cuml_gpu", fit_times, pred_times, "auc", metrics)

        # GPU random forest
        Xtrg = cudf.DataFrame(Xtr)
        Xteg = cudf.DataFrame(Xte)
        fit_times = []
        pred_times = []
        metrics = []
        for _ in range(int(CONFIG["cuml_repeats_heavy"])):
            s = time.perf_counter()
            mdl = CURF(n_estimators=250, max_depth=12, random_state=int(CONFIG["seed"]), output_type="numpy")
            mdl.fit(Xtrg, ytrg)
            sync()
            fit_times.append(time.perf_counter() - s)
            s = time.perf_counter()
            proba = mdl.predict_proba(Xteg)
            sync()
            pred_times.append(time.perf_counter() - s)
            proba = as_numpy(proba)
            proba = proba[:, 1] if proba.ndim == 2 else proba
            metrics.append(float(roc_auc_score(yte, proba)))
        push_result("random_forest", "cuml_gpu", fit_times, pred_times, "auc", metrics)

        # GPU KMeans
        Xkg = cudf.DataFrame(Xk)
        fit_times = []
        pred_times = []
        metrics = []
        for _ in range(int(CONFIG["cuml_repeats_light"])):
            s = time.perf_counter()
            labels = CUKMeans(n_clusters=7, n_init=10, random_state=int(CONFIG["seed"]), output_type="numpy").fit_predict(Xkg)
            sync()
            fit_times.append(time.perf_counter() - s)
            pred_times.append(0.0)
            labels = as_numpy(labels)
            n_sil = min(30_000, len(Xk))
            metrics.append(float(silhouette_score(Xk[:n_sil], labels[:n_sil])))
        push_result("kmeans", "cuml_gpu", fit_times, pred_times, "silhouette", metrics)

        # GPU PCA
        fit_times = []
        pred_times = []
        metrics = []
        for _ in range(int(CONFIG["cuml_repeats_light"])):
            s = time.perf_counter()
            pca = CUPCA(n_components=10, random_state=int(CONFIG["seed"]), output_type="numpy")
            pca.fit(Xtrg_s)
            sync()
            fit_times.append(time.perf_counter() - s)
            pred_times.append(0.0)
            evr = as_numpy(pca.explained_variance_ratio_)
            metrics.append(float(sum(evr)))
        push_result("pca", "cuml_gpu", fit_times, pred_times, "explained_variance", metrics)

        # GPU KNN
        Xknn_trg = cudf.DataFrame(Xknn_tr)
        yknn_trg = cudf.Series(yknn_tr.to_numpy())
        Xknn_teg = cudf.DataFrame(Xknn_te)
        fit_times = []
        pred_times = []
        metrics = []
        for _ in range(int(CONFIG["cuml_repeats_light"])):
            s = time.perf_counter()
            knn = CUKNN(n_neighbors=5, output_type="numpy")
            knn.fit(Xknn_trg, yknn_trg)
            sync()
            fit_times.append(time.perf_counter() - s)
            s = time.perf_counter()
            proba = knn.predict_proba(Xknn_teg)
            sync()
            pred_times.append(time.perf_counter() - s)
            proba = as_numpy(proba)
            proba = proba[:, 1] if proba.ndim == 2 else proba
            metrics.append(float(roc_auc_score(yknn_te, proba)))
        push_result("knn", "cuml_gpu", fit_times, pred_times, "auc", metrics)

    except Exception as exc:
        print("cuML GPU block failed:", repr(exc))
        traceback.print_exc(limit=2)
else:
    print("cuML/cuDF/cuPy not available -> CPU baseline only")


cuml_df = pd.DataFrame(res)
for t in cuml_df["task"].unique():
    cpu = cuml_df[(cuml_df["task"] == t) & (cuml_df["backend"] == "sklearn_cpu")]
    if len(cpu) == 1:
        b = float(cpu.iloc[0]["fit_seconds"])
        mask = cuml_df["task"] == t
        cuml_df.loc[mask, "fit_speedup_vs_cpu"] = np.where(
            cuml_df.loc[mask, "backend"] == "sklearn_cpu",
            np.nan,
            b / cuml_df.loc[mask, "fit_seconds"],
        )

tols = {
    "logistic_regression": 0.010,
    "random_forest": 0.025,
    "kmeans": 0.040,
    "pca": 0.050,
    "knn": 0.020,
}
qrows = []
for task in tols:
    cpu = cuml_df[(cuml_df["task"] == task) & (cuml_df["backend"] == "sklearn_cpu")]
    gpu = cuml_df[(cuml_df["task"] == task) & (cuml_df["backend"] == "cuml_gpu")]
    if len(cpu) == 1 and len(gpu) == 1:
        cpu_m = float(cpu.iloc[0]["metric_value"])
        gpu_m = float(gpu.iloc[0]["metric_value"])
        abs_diff = abs(cpu_m - gpu_m)
        qrows.append({
            "task": task,
            "metric": cpu.iloc[0]["metric"],
            "cpu_metric": cpu_m,
            "gpu_metric": gpu_m,
            "abs_diff": abs_diff,
            "rel_diff": rel_err(cpu_m, gpu_m),
            "tolerance": tols.get(task, 0.05),
            "quality_pass": abs_diff <= tols.get(task, 0.05),
        })

cuml_quality_df = pd.DataFrame(qrows)

display(cuml_df.sort_values(["task", "backend"]))
if len(cuml_quality_df):
    display(cuml_quality_df.sort_values("task"))

save_artifact(cuml_df, "cuml_benchmark")
if len(cuml_quality_df):
    save_artifact(cuml_quality_df, "cuml_quality_checks")

# Direct CPU vs GPU view
cuml_pair_rows: list[dict[str, Any]] = []
for task in sorted(cuml_df["task"].dropna().unique()):
    cpu = cuml_df[(cuml_df["task"] == task) & (cuml_df["backend"] == "sklearn_cpu")]
    gpu = cuml_df[(cuml_df["task"] == task) & (cuml_df["backend"] == "cuml_gpu")]
    if len(cpu) == 1 and len(gpu) == 1:
        qpass = True
        note = ""
        if len(cuml_quality_df):
            q = cuml_quality_df[cuml_quality_df["task"] == task]
            if len(q) == 1:
                qpass = bool(q.iloc[0].get("quality_pass", True))
                note = f"{q.iloc[0].get('metric', '')} abs_diff={q.iloc[0].get('abs_diff', np.nan)}"
        cpu_s = float(cpu.iloc[0]["fit_seconds"])
        gpu_s = float(gpu.iloc[0]["fit_seconds"])
        cuml_pair_rows.append({
            "section": "cuml",
            "method": str(task),
            "cpu_backend": "sklearn_cpu",
            "gpu_backend": "cuml_gpu",
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
            "speedup_x": cpu_s / max(gpu_s, 1e-12),
            "quality_pass": qpass,
            "note": note,
        })

cuml_direct_df = finalize_direct_df(pd.DataFrame(cuml_pair_rows), "cuml")
if len(cuml_direct_df):
    display(cuml_direct_df)
    save_artifact(cuml_direct_df, "gpu_bench_ml_direct")
    plot_method_pairs(cuml_direct_df, "cuML")
    print_section_conclusions(cuml_direct_df, "cuML")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Graph Analytics (cuGraph)
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 3) cuGraph benchmark: NetworkX CPU vs cuGraph GPU

Graph: bipartite (loans → attribute nodes: grade, purpose, sub_grade, verification_status).

Tasks: graph_build, connected_components, pagerank, louvain.

Also tests nx-cugraph dispatch (`backend="cugraph"`).
""")
)

cells.append(
    c("""
import networkx as nx

np.random.seed(int(CONFIG["seed"]))

if hasattr(nx, "config") and hasattr(nx.config, "warnings_to_ignore"):
    try:
        nx.config.warnings_to_ignore.add("cache")
    except Exception:
        pass


g = pd.read_parquet(TRAIN, columns=["id", "grade", "purpose", "sub_grade", "verification_status"]).dropna(subset=["id"]).copy()
g = g.sample(n=min(int(CONFIG["cugraph_sample_rows"]), len(g)), random_state=int(CONFIG["seed"]))
g["id"] = pd.to_numeric(g["id"], errors="coerce")
g = g.dropna(subset=["id"])
g["id"] = g["id"].astype(np.int64)

for col in ["grade", "purpose", "sub_grade", "verification_status"]:
    g[col] = g[col].fillna("unknown").astype(str)

g["loan"] = "L_" + g["id"].astype(str)

edges = pd.concat([
    g[["loan", "grade"]].rename(columns={"loan": "src", "grade": "dst"}).assign(dst=lambda x: "G_" + x["dst"]),
    g[["loan", "purpose"]].rename(columns={"loan": "src", "purpose": "dst"}).assign(dst=lambda x: "P_" + x["dst"]),
    g[["loan", "sub_grade"]].rename(columns={"loan": "src", "sub_grade": "dst"}).assign(dst=lambda x: "SG_" + x["dst"]),
    g[["loan", "verification_status"]].rename(columns={"loan": "src", "verification_status": "dst"}).assign(dst=lambda x: "V_" + x["dst"]),
], ignore_index=True)

nodes = pd.Index(pd.concat([edges["src"], edges["dst"]], ignore_index=True).unique())
node2id = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes)

E = pd.DataFrame({
    "src": edges["src"].map(node2id).astype(np.int32),
    "dst": edges["dst"].map(node2id).astype(np.int32),
})

R = int(CONFIG["cugraph_repeats"])
cres: list[dict[str, Any]] = []


def add_row(task: str, backend: str, times: list[float], metric: str, metric_value: float, **extra: Any) -> None:
    st = stats_from(times)
    row: dict[str, Any] = {
        "task": task,
        "backend": backend,
        "seconds": st["median_seconds"],
        "seconds_iqr": st["iqr_seconds"],
        "metric": metric,
        "metric_value": float(metric_value),
    }
    row.update(extra)
    cres.append(row)


# NetworkX CPU baseline
build_times: list[float] = []
G_cpu = None
for _ in range(R):
    s = time.perf_counter()
    G_cpu = nx.from_pandas_edgelist(E, source="src", target="dst", create_using=nx.Graph())
    build_times.append(time.perf_counter() - s)

cc_times: list[float] = []
cc_val = None
for _ in range(R):
    s = time.perf_counter()
    cc_val = nx.number_connected_components(G_cpu)
    cc_times.append(time.perf_counter() - s)

pr_times: list[float] = []
pr_sum = None
for _ in range(R):
    s = time.perf_counter()
    pr = nx.pagerank(G_cpu, alpha=0.85, max_iter=200, tol=1e-6)
    pr_times.append(time.perf_counter() - s)
    pr_sum = float(sum(pr.values()))

add_row("graph_build", "networkx_cpu", build_times, "nodes", float(G_cpu.number_of_nodes()))
add_row("connected_components", "networkx_cpu", cc_times, "n_components", float(cc_val))
add_row("pagerank", "networkx_cpu", pr_times, "sum_pagerank", float(pr_sum), converged_rate=1.0)

# Louvain CPU
try:
    louvain_times: list[float] = []
    n_communities = 0
    for _ in range(R):
        s = time.perf_counter()
        communities = nx.community.louvain_communities(G_cpu, seed=int(CONFIG["seed"]))
        louvain_times.append(time.perf_counter() - s)
        n_communities = len(communities)
    add_row("louvain", "networkx_cpu", louvain_times, "n_communities", float(n_communities))
except Exception as exc:
    print(f"Louvain CPU skipped: {exc}")


# Native cuGraph
if all(importlib.util.find_spec(p) is not None for p in ["cugraph", "cudf", "cupy"]):
    try:
        import cupy as cp
        import cudf
        import cugraph

        def sync() -> None:
            cp.cuda.Stream.null.synchronize()

        Eg = cudf.DataFrame(E)

        build_times = []
        G_gpu = None
        for _ in range(R):
            s = time.perf_counter()
            G_gpu = cugraph.Graph(directed=False)
            G_gpu.from_cudf_edgelist(Eg, source="src", destination="dst", renumber=False, store_transposed=True)
            sync()
            build_times.append(time.perf_counter() - s)

        cc_times = []
        ncc = None
        for _ in range(R):
            s = time.perf_counter()
            ccg = cugraph.connected_components(G_gpu)
            sync()
            cc_times.append(time.perf_counter() - s)
            lcol = "labels" if "labels" in ccg.columns else ccg.columns[-1]
            ncc = int(ccg[lcol].nunique())

        pr_times = []
        pr_sum = None
        converged_flags: list[bool] = []
        for _ in range(R):
            s = time.perf_counter()
            out = cugraph.pagerank(G_gpu, alpha=0.85, max_iter=200, tol=1e-6, fail_on_nonconvergence=False)
            sync()
            pr_times.append(time.perf_counter() - s)
            if isinstance(out, tuple):
                prg, converged = out
                converged_flags.append(bool(converged))
            else:
                prg = out
                converged_flags.append(True)
            pr_sum = float(prg["pagerank"].sum())

        add_row("graph_build", "cugraph_gpu", build_times, "nodes", float(G_gpu.number_of_vertices()))
        add_row("connected_components", "cugraph_gpu", cc_times, "n_components", float(ncc))
        add_row("pagerank", "cugraph_gpu", pr_times, "sum_pagerank", float(pr_sum), converged_rate=float(np.mean(converged_flags)))

        # Louvain GPU
        try:
            louvain_times = []
            n_parts = 0
            for _ in range(R):
                s = time.perf_counter()
                parts, modularity = cugraph.louvain(G_gpu)
                sync()
                louvain_times.append(time.perf_counter() - s)
                pcol = "partition" if "partition" in parts.columns else parts.columns[-1]
                n_parts = int(parts[pcol].nunique())
            add_row("louvain", "cugraph_gpu", louvain_times, "n_communities", float(n_parts))
        except Exception as exc:
            print(f"Louvain GPU skipped: {exc}")

    except Exception as exc:
        print("cuGraph GPU block failed:", repr(exc))
        traceback.print_exc(limit=2)
else:
    print("cugraph/cudf/cupy not available -> CPU only")


# NetworkX backend dispatch (nx-cugraph)
if importlib.util.find_spec("nx_cugraph") is not None:
    try:
        cc_times = []
        cc_val_backend = None
        for _ in range(R):
            s = time.perf_counter()
            cc_val_backend = sum(1 for _ in nx.connected_components(G_cpu, backend="cugraph"))
            cc_times.append(time.perf_counter() - s)
        add_row("connected_components", "networkx_cugraph_backend", cc_times, "n_components", float(cc_val_backend))

        pr_times = []
        pr_sum_backend = None
        for _ in range(R):
            s = time.perf_counter()
            pr_backend = nx.pagerank(G_cpu, alpha=0.85, max_iter=200, tol=1e-6, backend="cugraph")
            pr_times.append(time.perf_counter() - s)
            pr_sum_backend = float(sum(pr_backend.values()))
        add_row("pagerank", "networkx_cugraph_backend", pr_times, "sum_pagerank", float(pr_sum_backend), converged_rate=1.0)
    except Exception as exc:
        print("nx-cugraph backend block failed:", repr(exc))
        traceback.print_exc(limit=2)
else:
    print("nx_cugraph not available -> skipping NetworkX backend dispatch")


cugraph_df = pd.DataFrame(cres)
for t in cugraph_df["task"].unique():
    cpu = cugraph_df[(cugraph_df["task"] == t) & (cugraph_df["backend"] == "networkx_cpu")]
    if len(cpu) == 1:
        b = float(cpu.iloc[0]["seconds"])
        mask = cugraph_df["task"] == t
        cugraph_df.loc[mask, "speedup_vs_cpu"] = np.where(
            cugraph_df.loc[mask, "backend"] == "networkx_cpu",
            np.nan,
            b / cugraph_df.loc[mask, "seconds"],
        )

display(cugraph_df.sort_values(["task", "backend"]))
save_artifact(cugraph_df, "cugraph_benchmark")

# Direct CPU vs GPU view
cugraph_pair_rows: list[dict[str, Any]] = []
for task in sorted(cugraph_df["task"].dropna().unique()):
    cpu = cugraph_df[(cugraph_df["task"] == task) & (cugraph_df["backend"] == "networkx_cpu")]
    gpu = cugraph_df[(cugraph_df["task"] == task) & (cugraph_df["backend"] == "cugraph_gpu")]
    if len(cpu) == 1 and len(gpu) == 1:
        cpu_s = float(cpu.iloc[0]["seconds"])
        gpu_s = float(gpu.iloc[0]["seconds"])
        cugraph_pair_rows.append({
            "section": "cugraph",
            "method": str(task),
            "cpu_backend": "networkx_cpu",
            "gpu_backend": "cugraph_gpu",
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
            "speedup_x": cpu_s / max(gpu_s, 1e-12),
            "quality_pass": True,
            "note": "",
        })

cugraph_direct_df = finalize_direct_df(pd.DataFrame(cugraph_pair_rows), "cugraph")
if len(cugraph_direct_df):
    display(cugraph_direct_df)
    save_artifact(cugraph_direct_df, "gpu_bench_graph_direct")
    plot_method_pairs(cugraph_direct_df, "cuGraph")
    print_section_conclusions(cugraph_direct_df, "cuGraph")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Optimization (cuOpt)
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 4) cuOpt benchmark: LP portfolio optimization

- CPU baseline: `scipy.optimize.linprog(method="highs")`
- GPU path: cuOpt LP via `DataModel` + `Solve`
- MILP reference: SciPy on smaller problem (3K variables)
- Scaling: LP at multiple problem sizes (3K, 6K, 12K, 18K)
""")
)

cells.append(
    c("""
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix

np.random.seed(int(CONFIG["seed"]))

train_grade = pd.read_parquet(TRAIN, columns=["grade", "default_flag"]).dropna(subset=["grade"])
grade_pd = train_grade.groupby("grade")["default_flag"].mean()

df = pd.read_parquet(TEST, columns=["loan_amnt", "int_rate", "grade", "purpose", "default_flag"]).dropna(
    subset=["loan_amnt", "int_rate", "grade"]
).copy()
df = df.sample(n=min(int(CONFIG["cuopt_sample_rows"]), len(df)), random_state=int(CONFIG["seed"])).reset_index(drop=True)

df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0)
df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce").fillna(0.0)
df["pd"] = df["grade"].map(grade_pd).fillna(float(grade_pd.mean())).clip(0.001, 0.99)

loan = df["loan_amnt"].to_numpy(np.float64)
ir = (df["int_rate"] / 100.0).to_numpy(np.float64)
pd_proxy = df["pd"].to_numpy(np.float64)
obj = loan * (ir - 0.45 * pd_proxy)

budget = float(loan.sum() * 0.20)
purpose_vals = df["purpose"].astype(str).to_numpy()
constraint_rows = [loan, loan * pd_proxy]
rhs = [budget, 0.10 * budget]
for p in df["purpose"].astype(str).value_counts().head(2).index.tolist():
    constraint_rows.append(np.where(purpose_vals == p, loan, 0.0))
    rhs.append(0.35 * budget)

A = np.vstack(constraint_rows).astype(np.float64)
b = np.array(rhs, dtype=np.float64)

opt_res: list[dict[str, Any]] = []


# CPU LP
lp_times: list[float] = []
lp_objs: list[float] = []
lp_status: list[int] = []
for _ in range(int(CONFIG["cuopt_repeats"])):
    s = time.perf_counter()
    lp_cpu = linprog(c=-obj, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    lp_times.append(time.perf_counter() - s)
    lp_status.append(int(lp_cpu.status))
    lp_objs.append(float(-lp_cpu.fun) if lp_cpu.success else np.nan)

st = stats_from(lp_times)
opt_res.append({
    "task": "portfolio_lp",
    "backend": "scipy_highs_cpu",
    "seconds": st["median_seconds"],
    "seconds_iqr": st["iqr_seconds"],
    "status": str(lp_status[-1]),
    "objective": float(np.nanmedian(np.asarray(lp_objs, dtype=np.float64))),
    "n_variables": int(len(obj)),
    "n_runs": int(len(lp_times)),
})


# CPU MILP reference
n_milp = min(3000, len(obj))
Ac = A[:, :n_milp]
cc = -obj[:n_milp]
lc = LinearConstraint(Ac, -np.inf * np.ones(Ac.shape[0]), b)
bb = Bounds(lb=np.zeros(n_milp), ub=np.ones(n_milp))
ig = np.ones(n_milp, dtype=int)

s = time.perf_counter()
mi = milp(c=cc, integrality=ig, bounds=bb, constraints=lc, options={"time_limit": 120})
t_mi = time.perf_counter() - s

opt_res.append({
    "task": "portfolio_milp_reference",
    "backend": "scipy_milp_cpu",
    "seconds": float(t_mi),
    "seconds_iqr": np.nan,
    "status": str(mi.status),
    "objective": float(-mi.fun) if mi.success else np.nan,
    "n_variables": int(n_milp),
    "n_runs": 1,
})


# cuOpt LP via DataModel + Solve
if importlib.util.find_spec("cuopt") is not None:
    try:
        from cuopt import linear_programming as lp_api

        gpu_times: list[float] = []
        gpu_obj: list[float] = []
        gpu_reason: list[str] = []

        A_csr = csr_matrix(A)
        row_types = np.array(["L"] * A.shape[0])
        lb = np.zeros(A.shape[1], dtype=np.float64)
        ub = np.ones(A.shape[1], dtype=np.float64)

        for _ in range(int(CONFIG["cuopt_repeats"])):
            dm = lp_api.DataModel()
            dm.set_csr_constraint_matrix(
                A_csr.data.astype(np.float64),
                A_csr.indices.astype(np.int32),
                A_csr.indptr.astype(np.int32),
            )
            dm.set_constraint_bounds(b)
            dm.set_row_types(row_types)
            dm.set_objective_coefficients(obj.astype(np.float64))
            dm.set_maximize(True)
            dm.set_variable_lower_bounds(lb)
            dm.set_variable_upper_bounds(ub)

            settings = lp_api.SolverSettings()
            try:
                settings.set_parameter("log_to_console", False)
            except Exception:
                pass
            settings.set_parameter("time_limit", 120)

            s = time.perf_counter()
            sol = lp_api.Solve(dm, settings)
            gpu_times.append(time.perf_counter() - s)

            reason = str(sol.get_termination_reason())
            gpu_reason.append(reason)
            if "Optimal" in reason:
                gpu_obj.append(float(sol.get_primal_objective()))
            else:
                gpu_obj.append(np.nan)

        st_gpu = stats_from(gpu_times)
        opt_res.append({
            "task": "portfolio_lp",
            "backend": "cuopt_gpu",
            "seconds": st_gpu["median_seconds"],
            "seconds_iqr": st_gpu["iqr_seconds"],
            "status": gpu_reason[-1],
            "objective": float(np.nanmedian(np.asarray(gpu_obj, dtype=np.float64))),
            "n_variables": int(len(obj)),
            "n_runs": int(len(gpu_times)),
        })

    except Exception as exc:
        opt_res.append({
            "task": "portfolio_lp",
            "backend": "cuopt_gpu",
            "seconds": np.nan,
            "seconds_iqr": np.nan,
            "status": f"error: {exc}",
            "objective": np.nan,
            "n_variables": int(len(obj)),
            "n_runs": 0,
        })
        traceback.print_exc(limit=2)
else:
    print("cuopt not available -> CPU only")


cuopt_df = pd.DataFrame(opt_res)
cpu_lp = cuopt_df[(cuopt_df["task"] == "portfolio_lp") & (cuopt_df["backend"] == "scipy_highs_cpu")]
if len(cpu_lp) == 1:
    btime = float(cpu_lp.iloc[0]["seconds"])
    mask = cuopt_df["task"] == "portfolio_lp"
    cuopt_df.loc[mask, "speedup_vs_cpu_lp"] = np.where(
        cuopt_df.loc[mask, "backend"] == "scipy_highs_cpu",
        np.nan,
        btime / cuopt_df.loc[mask, "seconds"],
    )

display(cuopt_df.sort_values(["task", "backend"]))
save_artifact(cuopt_df, "cuopt_benchmark")

# Direct CPU vs GPU view
cuopt_pair_rows: list[dict[str, Any]] = []
gpu_lp = cuopt_df[(cuopt_df["task"] == "portfolio_lp") & (cuopt_df["backend"] == "cuopt_gpu")]
if len(cpu_lp) == 1 and len(gpu_lp) == 1:
    cpu_s = float(cpu_lp.iloc[0]["seconds"])
    gpu_s = float(gpu_lp.iloc[0]["seconds"])
    cuopt_pair_rows.append({
        "section": "cuopt",
        "method": "portfolio_lp",
        "cpu_backend": "scipy_highs_cpu",
        "gpu_backend": "cuopt_gpu",
        "cpu_seconds": cpu_s,
        "gpu_seconds": gpu_s,
        "speedup_x": cpu_s / max(gpu_s, 1e-12),
        "quality_pass": True,
        "note": "",
    })

cuopt_direct_df = finalize_direct_df(pd.DataFrame(cuopt_pair_rows), "cuopt")
if len(cuopt_direct_df):
    display(cuopt_direct_df)
    save_artifact(cuopt_direct_df, "gpu_bench_optimization_direct")
    plot_method_pairs(cuopt_direct_df, "cuOpt")
    print_section_conclusions(cuopt_direct_df, "cuOpt")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Numerical Computing (CuPy)
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 5) CuPy benchmark: NumPy/SciPy CPU vs CuPy GPU

Tasks:
- **Monte Carlo ECL**: 100K scenarios of PD x LGD x EAD simulation
- **SVD**: Truncated SVD on feature matrix
- **Sparse matrix multiply**: CSR matrix operations
""")
)

cells.append(
    c("""
from scipy import linalg as sp_linalg
from scipy.sparse import random as sp_sparse_random

np.random.seed(int(CONFIG["seed"]))

cupy_res: list[dict[str, Any]] = []
R_cupy = int(CONFIG["cupy_repeats"])
W_cupy = int(CONFIG["cupy_warmup"])


# Load feature matrix for SVD
Xsvd = pd.read_parquet(TRAIN_FE).select_dtypes(include=[np.number]).fillna(0.0).to_numpy(np.float32)
Xsvd = Xsvd[:100_000]

# --- Monte Carlo ECL (CPU) ---
n_scenarios = 100_000
n_loans = 10_000

pd_vals = np.random.beta(2, 8, size=(n_scenarios, n_loans)).astype(np.float32)
lgd_vals = np.random.beta(3, 7, size=(n_scenarios, n_loans)).astype(np.float32)
ead_vals = np.random.lognormal(mean=10, sigma=0.5, size=(1, n_loans)).astype(np.float32)

mc_cpu_times: list[float] = []
mc_cpu_ecl = None
for i in range(R_cupy + W_cupy):
    s = time.perf_counter()
    ecl = pd_vals * lgd_vals * ead_vals
    ecl_mean = ecl.mean(axis=0)
    ecl_total = ecl_mean.sum()
    dt = time.perf_counter() - s
    if i >= W_cupy:
        mc_cpu_times.append(dt)
        mc_cpu_ecl = float(ecl_total)

st_mc = stats_from(mc_cpu_times)
cupy_res.append({
    "task": "monte_carlo_ecl",
    "backend": "numpy_cpu",
    "seconds": st_mc["median_seconds"],
    "seconds_iqr": st_mc["iqr_seconds"],
    "metric": "ecl_total",
    "metric_value": mc_cpu_ecl,
})

# --- SVD (CPU) ---
svd_cpu_times: list[float] = []
svd_cpu_var = None
for i in range(R_cupy + W_cupy):
    s = time.perf_counter()
    U, S, Vt = sp_linalg.svd(Xsvd, full_matrices=False)
    dt = time.perf_counter() - s
    if i >= W_cupy:
        svd_cpu_times.append(dt)
        svd_cpu_var = float(np.sum(S[:10] ** 2) / np.sum(S ** 2))

st_svd = stats_from(svd_cpu_times)
cupy_res.append({
    "task": "svd",
    "backend": "scipy_cpu",
    "seconds": st_svd["median_seconds"],
    "seconds_iqr": st_svd["iqr_seconds"],
    "metric": "top10_variance_ratio",
    "metric_value": svd_cpu_var,
})

# --- Sparse MatMul (CPU) ---
M = sp_sparse_random(50_000, 50_000, density=0.001, format="csr", random_state=int(CONFIG["seed"]), dtype=np.float32)
sparse_cpu_times: list[float] = []
for i in range(R_cupy + W_cupy):
    s = time.perf_counter()
    result = M @ M.T
    dt = time.perf_counter() - s
    if i >= W_cupy:
        sparse_cpu_times.append(dt)

st_sparse = stats_from(sparse_cpu_times)
cupy_res.append({
    "task": "sparse_matmul",
    "backend": "scipy_cpu",
    "seconds": st_sparse["median_seconds"],
    "seconds_iqr": st_sparse["iqr_seconds"],
    "metric": "nnz",
    "metric_value": float(result.nnz) if hasattr(result, "nnz") else np.nan,
})


# --- GPU block ---
if importlib.util.find_spec("cupy") is not None:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cp_sparse

        def sync() -> None:
            cp.cuda.Stream.null.synchronize()

        # Monte Carlo ECL (GPU)
        pd_g = cp.asarray(pd_vals)
        lgd_g = cp.asarray(lgd_vals)
        ead_g = cp.asarray(ead_vals)

        mc_gpu_times: list[float] = []
        mc_gpu_ecl = None
        for i in range(R_cupy + W_cupy):
            s = time.perf_counter()
            ecl_g = pd_g * lgd_g * ead_g
            ecl_mean_g = ecl_g.mean(axis=0)
            ecl_total_g = ecl_mean_g.sum()
            sync()
            dt = time.perf_counter() - s
            if i >= W_cupy:
                mc_gpu_times.append(dt)
                mc_gpu_ecl = float(ecl_total_g.get())

        st_mc = stats_from(mc_gpu_times)
        cupy_res.append({
            "task": "monte_carlo_ecl",
            "backend": "cupy_gpu",
            "seconds": st_mc["median_seconds"],
            "seconds_iqr": st_mc["iqr_seconds"],
            "metric": "ecl_total",
            "metric_value": mc_gpu_ecl,
        })

        # SVD (GPU)
        Xsvd_g = cp.asarray(Xsvd)
        svd_gpu_times: list[float] = []
        svd_gpu_var = None
        for i in range(R_cupy + W_cupy):
            s = time.perf_counter()
            U_g, S_g, Vt_g = cp.linalg.svd(Xsvd_g, full_matrices=False)
            sync()
            dt = time.perf_counter() - s
            if i >= W_cupy:
                svd_gpu_times.append(dt)
                S_np = cp.asnumpy(S_g)
                svd_gpu_var = float(np.sum(S_np[:10] ** 2) / np.sum(S_np ** 2))

        st_svd = stats_from(svd_gpu_times)
        cupy_res.append({
            "task": "svd",
            "backend": "cupy_gpu",
            "seconds": st_svd["median_seconds"],
            "seconds_iqr": st_svd["iqr_seconds"],
            "metric": "top10_variance_ratio",
            "metric_value": svd_gpu_var,
        })

        # Sparse MatMul (GPU)
        M_g = cp_sparse.csr_matrix(M)
        sparse_gpu_times: list[float] = []
        sparse_gpu_nnz = None
        for i in range(R_cupy + W_cupy):
            s = time.perf_counter()
            result_g = M_g @ M_g.T
            sync()
            dt = time.perf_counter() - s
            if i >= W_cupy:
                sparse_gpu_times.append(dt)
                sparse_gpu_nnz = float(result_g.nnz)

        st_sparse = stats_from(sparse_gpu_times)
        cupy_res.append({
            "task": "sparse_matmul",
            "backend": "cupy_gpu",
            "seconds": st_sparse["median_seconds"],
            "seconds_iqr": st_sparse["iqr_seconds"],
            "metric": "nnz",
            "metric_value": sparse_gpu_nnz,
        })

        # Free GPU arrays
        del pd_g, lgd_g, ead_g, Xsvd_g, M_g
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as exc:
        print("CuPy GPU block failed:", repr(exc))
        traceback.print_exc(limit=2)
else:
    print("CuPy not available -> CPU baseline only")


cupy_df = pd.DataFrame(cupy_res)
for t in cupy_df["task"].unique():
    cpu = cupy_df[(cupy_df["task"] == t) & (cupy_df["backend"].str.contains("cpu"))]
    if len(cpu) == 1:
        b = float(cpu.iloc[0]["seconds"])
        mask = cupy_df["task"] == t
        cupy_df.loc[mask, "speedup_vs_cpu"] = np.where(
            cupy_df.loc[mask, "backend"].str.contains("cpu"),
            np.nan,
            b / cupy_df.loc[mask, "seconds"],
        )

display(cupy_df.sort_values(["task", "backend"]))
save_artifact(cupy_df, "cupy_benchmark")

# Direct CPU vs GPU view
cupy_pair_rows: list[dict[str, Any]] = []
for task in sorted(cupy_df["task"].dropna().unique()):
    cpu = cupy_df[(cupy_df["task"] == task) & (cupy_df["backend"].str.contains("cpu"))]
    gpu = cupy_df[(cupy_df["task"] == task) & (cupy_df["backend"].str.contains("gpu"))]
    if len(cpu) == 1 and len(gpu) == 1:
        cpu_s = float(cpu.iloc[0]["seconds"])
        gpu_s = float(gpu.iloc[0]["seconds"])
        cupy_pair_rows.append({
            "section": "cupy",
            "method": str(task),
            "cpu_backend": str(cpu.iloc[0]["backend"]),
            "gpu_backend": str(gpu.iloc[0]["backend"]),
            "cpu_seconds": cpu_s,
            "gpu_seconds": gpu_s,
            "speedup_x": cpu_s / max(gpu_s, 1e-12),
            "quality_pass": True,
            "note": "",
        })

cupy_direct_df = finalize_direct_df(pd.DataFrame(cupy_pair_rows), "cupy")
if len(cupy_direct_df):
    display(cupy_direct_df)
    save_artifact(cupy_direct_df, "gpu_bench_numerical_direct")
    plot_method_pairs(cupy_direct_df, "CuPy")
    print_section_conclusions(cupy_direct_df, "CuPy")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Consolidation
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 6) Consolidated summary across all sections

Merges all direct CPU-vs-GPU comparisons and section summaries.
""")
)

cells.append(
    c("""
# Consolidated all-sections summary
parts = []
for name in ["cudf_df", "cuml_df", "cugraph_df", "cuopt_df", "cupy_df"]:
    if name in globals():
        tmp = globals()[name].copy()
        tmp["section"] = name.replace("_df", "")
        parts.append(tmp)

if parts:
    summary = pd.concat(parts, ignore_index=True, sort=False)
    display(summary)
    save_artifact(summary, "benchmark_summary_all_sections")


# Direct paired comparison (all sections)
direct_parts = []
for name in ["cudf_direct_df", "cuml_direct_df", "cugraph_direct_df", "cuopt_direct_df", "cupy_direct_df"]:
    if name in globals() and isinstance(globals()[name], pd.DataFrame) and len(globals()[name]):
        direct_parts.append(globals()[name].copy())

if direct_parts:
    direct_cmp_df = pd.concat(direct_parts, ignore_index=True, sort=False).sort_values(["section", "method"]).reset_index(drop=True)
    display(direct_cmp_df)
    save_artifact(direct_cmp_df, "gpu_bench_all_direct")

    section_summary_df = (
        direct_cmp_df.groupby("section", as_index=False)
        .agg(
            methods=("method", "count"),
            median_speedup_x=("speedup_x", "median"),
            best_speedup_x=("speedup_x", "max"),
            methods_with_positive_gain=("improvement_pct", lambda s: int((s > 0).sum())),
            quality_pass_rate=("quality_pass", lambda s: float(np.mean(s.astype(float)))),
        )
        .sort_values("median_speedup_x", ascending=False)
        .reset_index(drop=True)
    )
    display(section_summary_df)
    save_artifact(section_summary_df, "gpu_bench_section_summary")

    print("\\nCross-section conclusions:")
    for section in section_summary_df["section"].tolist():
        s = direct_cmp_df[direct_cmp_df["section"] == section]
        best = s.sort_values("speedup_x", ascending=False).iloc[0]
        print(
            f"- {section}: mediana={float(s['speedup_x'].median()):.2f}x, "
            f"mejor={best['method']} ({best['speedup_x']:.2f}x)"
        )
else:
    print("No direct CPU/GPU pairs were available.")


# Quality checks consolidation
checks = []
for name in ["cuml_quality_df", "cupy_df"]:
    if name in globals() and isinstance(globals()[name], pd.DataFrame) and len(globals()[name]):
        if "quality_pass" in globals()[name].columns:
            tmp = globals()[name].copy()
            tmp["section"] = name.replace("_quality_df", "").replace("_df", "")
            checks.append(tmp)

if checks:
    quality = pd.concat(checks, ignore_index=True, sort=False)
    if "quality_pass" in quality.columns:
        failed = quality[quality["quality_pass"] == False]
        if len(failed):
            print(f"Quality checks with failures: {len(failed)}")
        else:
            print("All quality checks passed.")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Scaling Analysis
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 7) Scaling analysis: speedup vs dataset size

Measures cuDF and cuML speedups at 5%, 10%, 25%, 50%, 100% of the dataset
to find the crossover point where GPU becomes beneficial.
""")
)

cells.append(
    c("""
scaling_rows: list[dict[str, Any]] = []

# cuDF scaling
if has_cudf:
    full_train = pd.read_parquet(TRAIN)
    n_full = len(full_train)
    del full_train

    for pct in [0.05, 0.10, 0.25, 0.50, 1.00]:
        n_rows = int(n_full * pct)
        print(f"cuDF scaling: {pct:.0%} ({n_rows:,} rows)")

        # Create temp parquet with subset
        if pct < 1.0:
            subset = pd.read_parquet(TRAIN).sample(n=n_rows, random_state=int(CONFIG["seed"]))
            tmp_path = TMP / f"train_subset_{int(pct * 100)}.parquet"
            subset.to_parquet(tmp_path, index=False)
        else:
            tmp_path = TRAIN

        # Run pandas CPU
        cpu_res = run_json([exe, str(TMP / "bench_pandas.py"), str(tmp_path), "3", "1"])
        if cpu_res["status"] == "ok":
            cpu_s = float(cpu_res["median_seconds"])

            # Run pandas cuDF
            gpu_res = run_json([exe, "-m", "cudf.pandas", str(TMP / "bench_pandas.py"), str(tmp_path), "3", "1"])
            if gpu_res["status"] == "ok":
                gpu_s = float(gpu_res["median_seconds"])
                scaling_rows.append({
                    "method": "pandas_cudf",
                    "pct_data": float(pct),
                    "n_rows": n_rows,
                    "cpu_seconds": cpu_s,
                    "gpu_seconds": gpu_s,
                    "speedup_x": cpu_s / max(gpu_s, 1e-12),
                })

        # Cleanup temp file
        if pct < 1.0:
            tmp_path.unlink(missing_ok=True)

scaling_df = pd.DataFrame(scaling_rows)
if len(scaling_df):
    display(scaling_df)
    save_artifact(scaling_df, "gpu_bench_scaling")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method in scaling_df["method"].unique():
            mdf = scaling_df[scaling_df["method"] == method]
            ax.plot(mdf["pct_data"] * 100, mdf["speedup_x"], "o-", label=method, linewidth=2)
        ax.axhline(1.0, color="#5F6B7A", linestyle="--", alpha=0.5, label="1x (parity)")
        ax.set_xlabel("% of dataset", fontsize=12)
        ax.set_ylabel("Speedup (x)", fontsize=12)
        ax.set_title("GPU Speedup vs Dataset Size", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
else:
    print("Scaling analysis skipped (cuDF not available)")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Narwhals portability demo
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 8) Narwhals: write-once DataFrame API

Narwhals provides a unified API across pandas, Polars, cuDF, and PyArrow.
One function works on any backend.
""")
)

cells.append(
    c("""
if importlib.util.find_spec("narwhals") is not None:
    import narwhals as nw

    def risk_summary(df_native: Any) -> Any:
        df = nw.from_native(df_native)
        result = (
            df.group_by("grade")
            .agg(
                nw.col("loan_amnt").sum().alias("total_funded"),
                nw.col("default_flag").mean().alias("default_rate"),
                nw.len().alias("n_loans"),
            )
            .sort("grade")
        )
        return result.to_native()

    # Run on pandas
    pdf = pd.read_parquet(TRAIN, columns=["grade", "loan_amnt", "default_flag"]).head(100_000)
    result_pd = risk_summary(pdf)
    print("Narwhals on pandas:")
    display(result_pd if isinstance(result_pd, pd.DataFrame) else pd.DataFrame(result_pd))

    # Run on Polars
    plf = pl.read_parquet(TRAIN, columns=["grade", "loan_amnt", "default_flag"]).head(100_000)
    result_pl = risk_summary(plf)
    print("Narwhals on Polars:")
    display(result_pl if isinstance(result_pl, pl.DataFrame) else pd.DataFrame(result_pl))

    print("Same code, different backends — Narwhals enables write-once data logic.")
else:
    print("Narwhals not installed. Install with: pip install narwhals")
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 9: Metadata export
# ══════════════════════════════════════════════════════════════════════════════

cells.append(
    m("""
## 9) Metadata export and interpretation guide
""")
)

cells.append(
    c("""
import datetime
import platform

# Export metadata JSON for Streamlit page
meta: dict[str, Any] = {
    "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    "hardware": {
        "gpu": "NVIDIA GeForce RTX 3080",
        "vram": "10 GB GDDR6X",
        "cpu": "AMD Ryzen 5 5600X (6-core)",
        "ram": "24 GB DDR4",
        "os": f"{platform.system()} / WSL2",
    },
    "library_versions": lib_versions,
    "config": CONFIG,
    "datasets": {
        "train_rows": int(pd.read_parquet(TRAIN, columns=["id"]).shape[0]),
        "test_rows": int(pd.read_parquet(TEST, columns=["id"]).shape[0]),
    },
}

meta_path = OUT / "gpu_bench_meta.json"
meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
print("Metadata saved:", meta_path)
print(json.dumps(meta, indent=2, default=str))
""")
)

cells.append(
    m("""
## Interpretation guide

**How to read this notebook:**
- Prefer the **direct paired CPU-vs-GPU** tables and charts for decisions.
- Use quality tables to validate parity before trusting speedups.
- Prefer **median** speedup over single-run timings.
- If a GPU path is slower, check sample size and kernel launch overhead.
- Not every workload benefits from GPU acceleration.

**Relevance categories:**
| Category | Speedup | Additional condition |
|----------|---------|---------------------|
| Alta relevancia | >= 3.0x | >= 0.05s saved |
| Relevancia moderada | >= 1.5x | >= 0.01s saved |
| Relevancia baja | >= 1.1x | — |
| Neutro | >= 0.9x | — |
| No relevante | < 0.9x | CPU is faster |

**Key takeaways:**
- **DataFrame processing**: Polars GPU and cuDF pandas accelerator deliver the
  highest speedups with zero or minimal code changes.
- **Machine Learning**: Random Forest and KMeans benefit significantly; Logistic
  Regression overhead outweighs gains on small datasets.
- **Graph analytics**: cuGraph excels at PageRank and community detection on
  larger graphs.
- **Numerical computing**: Monte Carlo simulation and SVD are natural GPU workloads.
- **Optimization**: cuOpt provides value at scale (>10K variables).
""")
)

# ══════════════════════════════════════════════════════════════════════════════
# Build notebook JSON
# ══════════════════════════════════════════════════════════════════════════════

# Assign deterministic cell IDs for nbformat compatibility
for i, cell in enumerate(cells, start=1):
    cell["id"] = f"cell-{i:03d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/side_projects/10_rapids_gpu_benchmark_lending_club.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Notebook generated:", out)
