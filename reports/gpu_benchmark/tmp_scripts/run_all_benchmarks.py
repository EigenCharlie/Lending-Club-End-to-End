"""Comprehensive RAPIDS GPU benchmark runner.

Runs all benchmarks (DataFrame, ML, Graph, Optimization, CuPy) and exports
CSV artifacts to reports/gpu_benchmark/.

Usage:
    .venv/bin/python reports/gpu_benchmark/tmp_scripts/run_all_benchmarks.py
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT / "reports" / "gpu_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PQ = PROJECT / "data" / "processed" / "train.parquet"
CLEANED_PQ = PROJECT / "data" / "interim" / "lending_club_cleaned.parquet"
TRAIN_FE_PQ = PROJECT / "data" / "processed" / "train_fe.parquet"

REPEATS = 3
WARMUP = 1
BENCH_PROFILE = "current"

PROFILE_CONFIGS = {
    "current": {
        "repeats": 3,
        "warmup": 1,
        "ml_rows_cap": 250_000,
        "ml_train_cap": 200_000,
        "graph_rows_cap": 50_000,
        "opt_lp_sizes": [3000, 6000, 12000, 18000],
        "cupy_n_scenarios": 100_000,
        "cupy_svd_rows": 100_000,
        "cupy_sparse_n": 50_000,
    },
    "full_data": {
        "repeats": 3,
        "warmup": 1,
        "ml_rows_cap": 500_000,
        "ml_train_cap": 400_000,
        "graph_rows_cap": 200_000,
        "opt_lp_sizes": [5000, 10000, 20000, 35000, 50000],
        "cupy_n_scenarios": 120_000,
        "cupy_svd_rows": 150_000,
        "cupy_sparse_n": 60_000,
    },
    "stress_gpu": {
        "repeats": 2,
        "warmup": 1,
        "ml_rows_cap": 800_000,
        "ml_train_cap": 650_000,
        "graph_rows_cap": 350_000,
        "opt_lp_sizes": [10000, 25000, 50000, 75000, 100000],
        "cupy_n_scenarios": 200_000,
        "cupy_svd_rows": 200_000,
        "cupy_sparse_n": 75_000,
    },
}


def _profile_cfg() -> dict[str, object]:
    return PROFILE_CONFIGS.get(BENCH_PROFILE, PROFILE_CONFIGS["current"])


def timer(func, repeats=None, warmup=None):
    """Run func multiple times, return (median_seconds, last_result)."""
    if repeats is None:
        repeats = REPEATS
    if warmup is None:
        warmup = WARMUP
    results = []
    out = None
    for i in range(warmup + repeats):
        s = time.perf_counter()
        out = func()
        dt = time.perf_counter() - s
        if i >= warmup:
            results.append(dt)
    arr = np.array(results)
    return float(np.median(arr)), out


# ===================================================================
# 1. DataFrame Processing Benchmark
# ===================================================================
def run_dataframe_benchmarks():
    """Benchmark pandas, cudf.pandas, polars, polars-gpu, duckdb."""
    print("\n" + "=" * 60)
    print("SECTION 1: DataFrame Processing")
    print("=" * 60)
    rows = []
    pq_path = str(TRAIN_PQ)

    # --- Pandas CPU ---
    print("  [1/5] pandas CPU ...")
    try:
        def pandas_workload():
            df = pd.read_parquet(pq_path, columns=[
                "id", "issue_d", "loan_amnt", "int_rate", "annual_inc",
                "term", "grade", "purpose", "default_flag",
            ])
            df["int_rate"] = pd.to_numeric(
                df["int_rate"].astype(str).str.rstrip("%"), errors="coerce"
            )
            df["term_m"] = pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
            df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
            df["issue_year"] = df["issue_d"].dt.year.astype("Int64")
            f = df[
                (df["loan_amnt"] >= 5000)
                & (df["annual_inc"] > 20000)
                & (df["term_m"].isin([36, 60]))
            ]
            a = (
                f.groupby(["grade", "issue_year"], dropna=False)
                .agg(loans=("id", "count"), funded=("loan_amnt", "sum"),
                     dr=("default_flag", "mean"))
                .reset_index()
            )
            b = (
                f.groupby("purpose", dropna=False)
                .agg(p_loans=("id", "count"), p_dr=("default_flag", "mean"))
                .reset_index()
            )
            g = f[["grade", "purpose"]].drop_duplicates("grade")
            z = a.merge(g, on="grade", how="left").merge(b, on="purpose", how="left")
            z["checksum_col"] = z["dr"].fillna(0.0) * z["funded"].fillna(0.0)
            return z.sort_values(["issue_year", "grade"]).head(5000)

        med, out = timer(pandas_workload)
        ck = float(out["checksum_col"].sum())
        rows.append({"mode": "pandas_cpu", "median_seconds": med,
                      "rows_out": len(out), "checksum": ck, "status": "ok"})
        print(f"    -> {med:.4f}s, {len(out)} rows")
    except Exception as e:
        rows.append({"mode": "pandas_cpu", "status": f"error: {e}"})
        print(f"    -> FAILED: {e}")

    # --- cuDF pandas accelerator ---
    print("  [2/5] cuDF pandas accelerator ...")
    try:
        import cudf.pandas
        cudf.pandas.install()

        # Re-import pandas to get accelerated version
        import importlib
        pd_accel = importlib.import_module("pandas")

        def cudf_pandas_workload():
            df = pd_accel.read_parquet(pq_path, columns=[
                "id", "issue_d", "loan_amnt", "int_rate", "annual_inc",
                "term", "grade", "purpose", "default_flag",
            ])
            df["int_rate"] = pd_accel.to_numeric(
                df["int_rate"].astype(str).str.rstrip("%"), errors="coerce"
            )
            df["term_m"] = pd_accel.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
            df["issue_d"] = pd_accel.to_datetime(df["issue_d"], errors="coerce")
            df["issue_year"] = df["issue_d"].dt.year.astype("Int64")
            f = df[
                (df["loan_amnt"] >= 5000)
                & (df["annual_inc"] > 20000)
                & (df["term_m"].isin([36, 60]))
            ]
            a = (
                f.groupby(["grade", "issue_year"], dropna=False)
                .agg(loans=("id", "count"), funded=("loan_amnt", "sum"),
                     dr=("default_flag", "mean"))
                .reset_index()
            )
            b = (
                f.groupby("purpose", dropna=False)
                .agg(p_loans=("id", "count"), p_dr=("default_flag", "mean"))
                .reset_index()
            )
            g = f[["grade", "purpose"]].drop_duplicates("grade")
            z = a.merge(g, on="grade", how="left").merge(b, on="purpose", how="left")
            z["checksum_col"] = z["dr"].fillna(0.0) * z["funded"].fillna(0.0)
            return z.sort_values(["issue_year", "grade"]).head(5000)

        med, out = timer(cudf_pandas_workload)
        ck = float(out["checksum_col"].sum())
        rows.append({"mode": "pandas_cudf", "median_seconds": med,
                      "rows_out": len(out), "checksum": ck, "status": "ok"})
        print(f"    -> {med:.4f}s, {len(out)} rows")
    except Exception as e:
        rows.append({"mode": "pandas_cudf", "status": f"error: {e}"})
        print(f"    -> FAILED: {e}")

    # --- Polars CPU ---
    print("  [3/5] Polars CPU ...")
    try:
        import polars as pl

        def polars_cpu_workload():
            lf = pl.scan_parquet(pq_path).select([
                "id", "issue_d", "loan_amnt", "int_rate", "annual_inc",
                "term", "grade", "purpose", "default_flag",
            ])
            lf = lf.with_columns([
                pl.col("int_rate").cast(pl.Utf8).str.replace("%", "", literal=True)
                  .cast(pl.Float64, strict=False),
                pl.col("term").cast(pl.Utf8).str.extract(r"(\d+)", group_index=1)
                  .cast(pl.Int32, strict=False).alias("term_m"),
                pl.col("issue_d").cast(pl.Date, strict=False),
            ]).with_columns(pl.col("issue_d").dt.year().alias("issue_year"))
            f = lf.filter(
                (pl.col("loan_amnt") >= 5000)
                & (pl.col("annual_inc") > 20000)
                & (pl.col("term_m").is_in([36, 60]))
            )
            a = f.group_by(["grade", "issue_year"]).agg([
                pl.len().alias("loans"),
                pl.sum("loan_amnt").alias("funded"),
                pl.mean("default_flag").alias("dr"),
            ])
            b = f.group_by("purpose").agg([
                pl.len().alias("p_loans"),
                pl.mean("default_flag").alias("p_dr"),
            ])
            g = f.select(["grade", "purpose"]).unique(subset=["grade"], keep="first")
            z = (
                a.join(g, on="grade", how="left")
                .join(b, on="purpose", how="left")
                .with_columns(
                    (pl.col("dr").fill_null(0.0) * pl.col("funded").fill_null(0.0))
                    .alias("checksum_col")
                )
                .sort(["issue_year", "grade"])
                .limit(5000)
            )
            return z.collect()

        med, out = timer(polars_cpu_workload)
        ck = float(out["checksum_col"].sum())
        rows.append({"mode": "polars_cpu", "median_seconds": med,
                      "rows_out": out.height, "checksum": ck, "status": "ok"})
        print(f"    -> {med:.4f}s, {out.height} rows")
    except Exception as e:
        rows.append({"mode": "polars_cpu", "status": f"error: {e}"})
        print(f"    -> FAILED: {e}")

    # --- Polars GPU ---
    print("  [4/5] Polars GPU Engine ...")
    try:
        import polars as pl

        def polars_gpu_workload():
            lf = pl.scan_parquet(pq_path).select([
                "id", "issue_d", "loan_amnt", "int_rate", "annual_inc",
                "term", "grade", "purpose", "default_flag",
            ])
            lf = lf.with_columns([
                pl.col("int_rate").cast(pl.Utf8).str.replace("%", "", literal=True)
                  .cast(pl.Float64, strict=False),
                pl.col("term").cast(pl.Utf8).str.extract(r"(\d+)", group_index=1)
                  .cast(pl.Int32, strict=False).alias("term_m"),
                pl.col("issue_d").cast(pl.Date, strict=False),
            ]).with_columns(pl.col("issue_d").dt.year().alias("issue_year"))
            f = lf.filter(
                (pl.col("loan_amnt") >= 5000)
                & (pl.col("annual_inc") > 20000)
                & (pl.col("term_m").is_in([36, 60]))
            )
            a = f.group_by(["grade", "issue_year"]).agg([
                pl.len().alias("loans"),
                pl.sum("loan_amnt").alias("funded"),
                pl.mean("default_flag").alias("dr"),
            ])
            b = f.group_by("purpose").agg([
                pl.len().alias("p_loans"),
                pl.mean("default_flag").alias("p_dr"),
            ])
            g = f.select(["grade", "purpose"]).unique(subset=["grade"], keep="first")
            z = (
                a.join(g, on="grade", how="left")
                .join(b, on="purpose", how="left")
                .with_columns(
                    (pl.col("dr").fill_null(0.0) * pl.col("funded").fill_null(0.0))
                    .alias("checksum_col")
                )
                .sort(["issue_year", "grade"])
                .limit(5000)
            )
            return z.collect(engine=pl.GPUEngine(raise_on_fail=True))

        med, out = timer(polars_gpu_workload)
        ck = float(out["checksum_col"].sum())
        rows.append({"mode": "polars_gpu", "median_seconds": med,
                      "rows_out": out.height, "checksum": ck, "status": "ok"})
        print(f"    -> {med:.4f}s, {out.height} rows")
    except Exception as e:
        rows.append({"mode": "polars_gpu", "status": f"error: {e}"})
        print(f"    -> FAILED: {e}")

    # --- DuckDB ---
    print("  [5/5] DuckDB ...")
    try:
        import duckdb

        SQL = f"""
        WITH raw AS (
            SELECT id, issue_d, loan_amnt,
                   CAST(REGEXP_REPLACE(CAST(int_rate AS VARCHAR), '%', '') AS DOUBLE) AS int_rate,
                   annual_inc,
                   CAST(REGEXP_EXTRACT(CAST(term AS VARCHAR), '(\\d+)', 1) AS INT) AS term_m,
                   grade, purpose, default_flag
            FROM read_parquet('{pq_path}')
        ),
        filtered AS (
            SELECT *, YEAR(CAST(issue_d AS DATE)) AS issue_year
            FROM raw
            WHERE loan_amnt >= 5000 AND annual_inc > 20000 AND term_m IN (36, 60)
        ),
        agg_grade AS (
            SELECT grade, issue_year, COUNT(id) AS loans,
                   SUM(loan_amnt) AS funded, AVG(default_flag) AS dr
            FROM filtered GROUP BY grade, issue_year
        ),
        agg_purpose AS (
            SELECT purpose, COUNT(id) AS p_loans, AVG(default_flag) AS p_dr
            FROM filtered GROUP BY purpose
        ),
        grade_purpose AS (
            SELECT DISTINCT ON (grade) grade, purpose FROM filtered
        ),
        joined AS (
            SELECT a.*, gp.purpose, ap.p_loans, ap.p_dr,
                   COALESCE(a.dr, 0.0) * COALESCE(a.funded, 0.0) AS checksum_col
            FROM agg_grade a
            LEFT JOIN grade_purpose gp ON a.grade = gp.grade
            LEFT JOIN agg_purpose ap ON gp.purpose = ap.purpose
        )
        SELECT * FROM joined ORDER BY issue_year, grade LIMIT 5000
        """

        def duckdb_workload():
            conn = duckdb.connect()
            out = conn.execute(SQL).fetchdf()
            conn.close()
            return out

        med, out = timer(duckdb_workload)
        ck = float(out["checksum_col"].sum())
        rows.append({"mode": "duckdb", "median_seconds": med,
                      "rows_out": len(out), "checksum": ck, "status": "ok"})
        print(f"    -> {med:.4f}s, {len(out)} rows")
    except Exception as e:
        rows.append({"mode": "duckdb", "status": f"error: {e}"})
        print(f"    -> FAILED: {e}")

    # Compute speedups
    df = pd.DataFrame(rows)
    pandas_time = df.loc[df["mode"] == "pandas_cpu", "median_seconds"]
    if not pandas_time.empty:
        base = pandas_time.values[0]
        df["speedup_vs_pandas_cpu"] = base / df["median_seconds"]
    df.to_csv(OUT_DIR / "cudf_polars_benchmark.csv", index=False)
    print(f"\n  Saved: cudf_polars_benchmark.csv ({len(df)} rows)")
    return df


# ===================================================================
# 2. Machine Learning Benchmark
# ===================================================================
def run_ml_benchmarks():
    """Benchmark sklearn CPU vs cuML GPU."""
    print("\n" + "=" * 60)
    print("SECTION 2: Machine Learning")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(str(TRAIN_FE_PQ))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "default_flag"][:30]
    cfg = _profile_cfg()
    df_sub = df[num_cols + ["default_flag"]].dropna().head(int(cfg["ml_rows_cap"]))
    X = df_sub[num_cols].values.astype(np.float32)
    y = df_sub["default_flag"].values.astype(np.int32)
    n_train = min(int(cfg["ml_train_cap"]), max(1, len(X) - max(1000, len(X) // 5)))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    print(f"  Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")

    rows = []

    # Tasks to benchmark
    tasks = [
        ("logistic_regression", "classification"),
        ("random_forest", "classification"),
        ("kmeans", "clustering"),
        ("pca", "dimensionality_reduction"),
        ("knn", "classification"),
        ("umap", "dimensionality_reduction"),
        ("hdbscan", "clustering"),
    ]

    for task_name, task_type in tasks:
        print(f"\n  --- {task_name} ---")

        # --- sklearn CPU ---
        print(f"    sklearn CPU ...")
        try:
            if task_name == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                model = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
                s = time.perf_counter()
                model.fit(X_train, y_train)
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_test)[:, 1]
                pred_t = time.perf_counter() - s
                metric_val = roc_auc_score(y_test, preds)
                metric_name = "auc"
            elif task_name == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import roc_auc_score
                model = RandomForestClassifier(n_estimators=100, max_depth=12,
                                                n_jobs=-1, random_state=42)
                s = time.perf_counter()
                model.fit(X_train, y_train)
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_test)[:, 1]
                pred_t = time.perf_counter() - s
                metric_val = roc_auc_score(y_test, preds)
                metric_name = "auc"
            elif task_name == "kmeans":
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                model = KMeans(n_clusters=7, n_init=1, max_iter=100, random_state=42)
                X_km = X_train[:50000]
                s = time.perf_counter()
                labels = model.fit_predict(X_km)
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                metric_val = silhouette_score(X_km[:10000], labels[:10000])
                metric_name = "silhouette"
            elif task_name == "pca":
                from sklearn.decomposition import PCA
                model = PCA(n_components=10)
                s = time.perf_counter()
                model.fit(X_train)
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                _ = model.transform(X_test)
                pred_t = time.perf_counter() - s
                metric_val = float(model.explained_variance_ratio_.sum())
                metric_name = "explained_var"
            elif task_name == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import roc_auc_score
                X_knn_tr, y_knn_tr = X_train[:50000], y_train[:50000]
                X_knn_te, y_knn_te = X_test[:10000], y_test[:10000]
                model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                s = time.perf_counter()
                model.fit(X_knn_tr, y_knn_tr)
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_knn_te)[:, 1]
                pred_t = time.perf_counter() - s
                metric_val = roc_auc_score(y_knn_te, preds)
                metric_name = "auc"
            elif task_name == "umap":
                from umap import UMAP
                X_um = X_train[:30000]
                model = UMAP(n_components=2, n_neighbors=15, random_state=42)
                s = time.perf_counter()
                emb = model.fit_transform(X_um)
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                metric_val = float(emb.shape[0])
                metric_name = "n_embedded"
            elif task_name == "hdbscan":
                from sklearn.cluster import HDBSCAN as SkHDBSCAN
                X_hdb = X_train[:30000]
                model = SkHDBSCAN(min_cluster_size=50)
                s = time.perf_counter()
                labels = model.fit_predict(X_hdb)
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                metric_val = float(n_clusters)
                metric_name = "n_clusters"

            rows.append({
                "task": task_name, "backend": "sklearn_cpu",
                "fit_seconds": fit_t, "predict_seconds": pred_t,
                "metric": metric_name, "metric_value": metric_val,
            })
            print(f"      fit={fit_t:.3f}s, pred={pred_t:.3f}s, {metric_name}={metric_val:.4f}")
        except Exception as e:
            rows.append({"task": task_name, "backend": "sklearn_cpu",
                         "fit_seconds": None, "predict_seconds": None,
                         "metric": "error", "metric_value": str(e)})
            print(f"      FAILED: {e}")

        # --- cuML GPU ---
        print(f"    cuML GPU ...")
        try:
            import cuml
            import cupy as cp

            X_train_gpu = cp.asarray(X_train)
            y_train_gpu = cp.asarray(y_train)
            X_test_gpu = cp.asarray(X_test)

            if task_name == "logistic_regression":
                from cuml.linear_model import LogisticRegression as cuLogReg
                from sklearn.metrics import roc_auc_score as cpu_auc
                model = cuLogReg(max_iter=200)
                s = time.perf_counter()
                model.fit(X_train_gpu, y_train_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_test_gpu)[:, 1]
                cp.cuda.Stream.null.synchronize()
                pred_t = time.perf_counter() - s
                metric_val = cpu_auc(y_test, cp.asnumpy(preds))
                metric_name = "auc"
            elif task_name == "random_forest":
                from cuml.ensemble import RandomForestClassifier as cuRF
                from sklearn.metrics import roc_auc_score as cpu_auc
                model = cuRF(n_estimators=100, max_depth=12, random_state=42)
                s = time.perf_counter()
                model.fit(X_train_gpu, y_train_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_test_gpu)[:, 1]
                cp.cuda.Stream.null.synchronize()
                pred_t = time.perf_counter() - s
                metric_val = cpu_auc(y_test, cp.asnumpy(preds))
                metric_name = "auc"
            elif task_name == "kmeans":
                from cuml.cluster import KMeans as cuKMeans
                X_km_gpu = cp.asarray(X_train[:50000].astype(np.float32))
                model = cuKMeans(n_clusters=7, n_init=1, max_iter=100,
                                 random_state=42)
                s = time.perf_counter()
                labels = model.fit_predict(X_km_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                from sklearn.metrics import silhouette_score
                metric_val = silhouette_score(
                    X_train[:50000][:10000],
                    cp.asnumpy(labels)[:10000]
                )
                metric_name = "silhouette"
            elif task_name == "pca":
                from cuml.decomposition import PCA as cuPCA
                model = cuPCA(n_components=10)
                s = time.perf_counter()
                model.fit(X_train_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                _ = model.transform(X_test_gpu)
                cp.cuda.Stream.null.synchronize()
                pred_t = time.perf_counter() - s
                metric_val = float(cp.asnumpy(
                    model.explained_variance_ratio_
                ).sum())
                metric_name = "explained_var"
            elif task_name == "knn":
                from cuml.neighbors import KNeighborsClassifier as cuKNN
                from sklearn.metrics import roc_auc_score as cpu_auc
                X_knn_tr_gpu = cp.asarray(X_train[:50000])
                y_knn_tr_gpu = cp.asarray(y_train[:50000])
                X_knn_te_gpu = cp.asarray(X_test[:10000])
                model = cuKNN(n_neighbors=5)
                s = time.perf_counter()
                model.fit(X_knn_tr_gpu, y_knn_tr_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                s = time.perf_counter()
                preds = model.predict_proba(X_knn_te_gpu)[:, 1]
                cp.cuda.Stream.null.synchronize()
                pred_t = time.perf_counter() - s
                metric_val = cpu_auc(y_test[:10000], cp.asnumpy(preds))
                metric_name = "auc"
            elif task_name == "umap":
                from cuml.manifold import UMAP as cuUMAP
                X_um_gpu = cp.asarray(X_train[:30000])
                model = cuUMAP(n_components=2, n_neighbors=15, random_state=42)
                s = time.perf_counter()
                emb = model.fit_transform(X_um_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                metric_val = float(emb.shape[0])
                metric_name = "n_embedded"
            elif task_name == "hdbscan":
                from cuml.cluster import HDBSCAN as cuHDBSCAN
                X_hdb_gpu = cp.asarray(X_train[:30000])
                model = cuHDBSCAN(min_cluster_size=50)
                s = time.perf_counter()
                labels = model.fit_predict(X_hdb_gpu)
                cp.cuda.Stream.null.synchronize()
                fit_t = time.perf_counter() - s
                pred_t = 0.0
                labels_cpu = cp.asnumpy(labels)
                n_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)
                metric_val = float(n_clusters)
                metric_name = "n_clusters"

            rows.append({
                "task": task_name, "backend": "cuml_gpu",
                "fit_seconds": fit_t, "predict_seconds": pred_t,
                "metric": metric_name, "metric_value": metric_val,
            })
            print(f"      fit={fit_t:.3f}s, pred={pred_t:.3f}s, {metric_name}={metric_val:.4f}")
        except Exception as e:
            traceback.print_exc()
            rows.append({"task": task_name, "backend": "cuml_gpu",
                         "fit_seconds": None, "predict_seconds": None,
                         "metric": "error", "metric_value": str(e)})
            print(f"      FAILED: {e}")

    # Compute speedups
    df = pd.DataFrame(rows)
    cpu_times = df[df["backend"] == "sklearn_cpu"].set_index("task")["fit_seconds"]
    gpu_times = df[df["backend"] == "cuml_gpu"].set_index("task")["fit_seconds"]
    speedups = cpu_times / gpu_times
    df["fit_speedup_vs_cpu"] = df.apply(
        lambda r: speedups.get(r["task"]) if r["backend"] == "cuml_gpu" else None,
        axis=1,
    )
    df.to_csv(OUT_DIR / "cuml_benchmark.csv", index=False)
    print(f"\n  Saved: cuml_benchmark.csv ({len(df)} rows)")
    return df


# ===================================================================
# 3. Graph Analytics Benchmark
# ===================================================================
def run_graph_benchmarks():
    """Benchmark NetworkX CPU vs cuGraph GPU."""
    print("\n" + "=" * 60)
    print("SECTION 3: Graph Analytics")
    print("=" * 60)

    # Build a bipartite-like graph: loan_id -> grade, loan_id -> purpose
    df = pd.read_parquet(str(TRAIN_PQ), columns=["id", "grade", "purpose", "sub_grade"])
    df = df.dropna(subset=["grade", "purpose"]).head(int(_profile_cfg()["graph_rows_cap"]))

    # Create edges: borrower -> grade_node, borrower -> purpose_node
    edges_grade = df[["id", "grade"]].copy()
    edges_grade.columns = ["src", "dst"]
    edges_grade["dst"] = "G_" + edges_grade["dst"]

    edges_purpose = df[["id", "purpose"]].copy()
    edges_purpose.columns = ["src", "dst"]
    edges_purpose["dst"] = "P_" + edges_purpose["dst"]

    edges_subgrade = df[["id", "sub_grade"]].dropna().copy()
    edges_subgrade.columns = ["src", "dst"]
    edges_subgrade["dst"] = "SG_" + edges_subgrade["dst"]

    edge_df = pd.concat([edges_grade, edges_purpose, edges_subgrade], ignore_index=True)

    # Create numeric node IDs
    all_nodes = pd.concat([edge_df["src"], edge_df["dst"]]).unique()
    node_map = {n: i for i, n in enumerate(all_nodes)}
    edge_df["src_id"] = edge_df["src"].map(node_map).astype(np.int32)
    edge_df["dst_id"] = edge_df["dst"].map(node_map).astype(np.int32)
    n_nodes = len(all_nodes)
    n_edges = len(edge_df)
    print(f"  Graph: {n_nodes} nodes, {n_edges} edges")

    rows = []

    # --- NetworkX CPU ---
    print("\n  NetworkX CPU:")
    try:
        import networkx as nx

        # Build
        s = time.perf_counter()
        G_nx = nx.Graph()
        G_nx.add_edges_from(zip(edge_df["src_id"], edge_df["dst_id"], strict=False))
        build_t = time.perf_counter() - s
        rows.append({"task": "graph_build", "backend": "networkx_cpu",
                      "seconds": build_t, "metric": "nodes",
                      "metric_value": G_nx.number_of_nodes()})
        print(f"    build: {build_t:.4f}s ({G_nx.number_of_nodes()} nodes)")

        # Connected components
        s = time.perf_counter()
        cc = list(nx.connected_components(G_nx))
        cc_t = time.perf_counter() - s
        rows.append({"task": "connected_components", "backend": "networkx_cpu",
                      "seconds": cc_t, "metric": "n_components",
                      "metric_value": len(cc)})
        print(f"    connected_components: {cc_t:.4f}s ({len(cc)} components)")

        # PageRank
        s = time.perf_counter()
        pr = nx.pagerank(G_nx, alpha=0.85, max_iter=100, tol=1e-5)
        pr_t = time.perf_counter() - s
        pr_sum = sum(pr.values())
        rows.append({"task": "pagerank", "backend": "networkx_cpu",
                      "seconds": pr_t, "metric": "sum_pagerank",
                      "metric_value": pr_sum})
        print(f"    pagerank: {pr_t:.4f}s (sum={pr_sum:.4f})")

        # Louvain
        s = time.perf_counter()
        communities = nx.community.louvain_communities(G_nx, seed=42)
        louv_t = time.perf_counter() - s
        rows.append({"task": "louvain", "backend": "networkx_cpu",
                      "seconds": louv_t, "metric": "n_communities",
                      "metric_value": len(communities)})
        print(f"    louvain: {louv_t:.4f}s ({len(communities)} communities)")

        # Betweenness centrality (sampled for speed)
        k_sample = min(100, n_nodes)
        s = time.perf_counter()
        bc = nx.betweenness_centrality(G_nx, k=k_sample, seed=42)
        bc_t = time.perf_counter() - s
        bc_max = max(bc.values())
        rows.append({"task": "betweenness_centrality", "backend": "networkx_cpu",
                      "seconds": bc_t, "metric": "max_bc",
                      "metric_value": bc_max})
        print(f"    betweenness_centrality (k={k_sample}): {bc_t:.4f}s (max={bc_max:.6f})")

    except Exception as e:
        traceback.print_exc()
        print(f"  NetworkX FAILED: {e}")

    # --- cuGraph GPU ---
    print("\n  cuGraph GPU:")
    try:
        import cudf as cudf_lib
        import cugraph

        # Build
        edge_cudf = cudf_lib.DataFrame({
            "src": edge_df["src_id"].values,
            "dst": edge_df["dst_id"].values,
        })

        s = time.perf_counter()
        G_cu = cugraph.Graph()
        G_cu.from_cudf_edgelist(edge_cudf, source="src", destination="dst",
                                 renumber=True)
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        build_t = time.perf_counter() - s
        rows.append({"task": "graph_build", "backend": "cugraph_gpu",
                      "seconds": build_t, "metric": "nodes",
                      "metric_value": float(G_cu.number_of_vertices())})
        print(f"    build: {build_t:.4f}s ({G_cu.number_of_vertices()} nodes)")

        # Connected components
        s = time.perf_counter()
        cc_df = cugraph.connected_components(G_cu, connection="weak")
        cp.cuda.Stream.null.synchronize()
        cc_t = time.perf_counter() - s
        n_comp = cc_df["labels"].nunique()
        rows.append({"task": "connected_components", "backend": "cugraph_gpu",
                      "seconds": cc_t, "metric": "n_components",
                      "metric_value": float(n_comp)})
        print(f"    connected_components: {cc_t:.4f}s ({n_comp} components)")

        # PageRank
        s = time.perf_counter()
        pr_df = cugraph.pagerank(G_cu, alpha=0.85, max_iter=100, tol=1e-5)
        cp.cuda.Stream.null.synchronize()
        pr_t = time.perf_counter() - s
        pr_sum = float(pr_df["pagerank"].sum())
        rows.append({"task": "pagerank", "backend": "cugraph_gpu",
                      "seconds": pr_t, "metric": "sum_pagerank",
                      "metric_value": pr_sum})
        print(f"    pagerank: {pr_t:.4f}s (sum={pr_sum:.4f})")

        # Louvain
        s = time.perf_counter()
        louv_df, modularity = cugraph.louvain(G_cu)
        cp.cuda.Stream.null.synchronize()
        louv_t = time.perf_counter() - s
        n_comm = louv_df["partition"].nunique()
        rows.append({"task": "louvain", "backend": "cugraph_gpu",
                      "seconds": louv_t, "metric": "n_communities",
                      "metric_value": float(n_comm)})
        print(f"    louvain: {louv_t:.4f}s ({n_comm} communities, mod={modularity:.4f})")

        # Betweenness centrality
        s = time.perf_counter()
        bc_df = cugraph.betweenness_centrality(G_cu, k=k_sample, seed=42)
        cp.cuda.Stream.null.synchronize()
        bc_t = time.perf_counter() - s
        bc_max = float(bc_df["betweenness_centrality"].max())
        rows.append({"task": "betweenness_centrality", "backend": "cugraph_gpu",
                      "seconds": bc_t, "metric": "max_bc",
                      "metric_value": bc_max})
        print(f"    betweenness_centrality (k={k_sample}): {bc_t:.4f}s (max={bc_max:.6f})")

    except Exception as e:
        traceback.print_exc()
        print(f"  cuGraph FAILED: {e}")

    # Compute speedups
    df = pd.DataFrame(rows)
    cpu_times = df[df["backend"] == "networkx_cpu"].set_index("task")["seconds"]
    gpu_times = df[df["backend"] == "cugraph_gpu"].set_index("task")["seconds"]
    speedups = {}
    for task in cpu_times.index:
        if task in gpu_times.index and gpu_times[task] > 0:
            speedups[task] = cpu_times[task] / gpu_times[task]
    df["speedup_vs_cpu"] = df.apply(
        lambda r: speedups.get(r["task"]) if r["backend"] == "cugraph_gpu" else None,
        axis=1,
    )
    df.to_csv(OUT_DIR / "cugraph_benchmark.csv", index=False)
    print(f"\n  Saved: cugraph_benchmark.csv ({len(df)} rows)")
    return df


# ===================================================================
# 4. Optimization Benchmark
# ===================================================================
def run_optimization_benchmarks():
    """Benchmark SciPy HiGHS vs cuOpt for LP/MILP."""
    print("\n" + "=" * 60)
    print("SECTION 4: Optimization (LP/MILP)")
    print("=" * 60)

    rows = []
    # Load real loan data for the optimization problem
    df = pd.read_parquet(str(TRAIN_PQ), columns=["loan_amnt", "int_rate", "default_flag"])
    df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
    df = df.dropna()

    sizes = [int(v) for v in _profile_cfg()["opt_lp_sizes"]]

    for n_vars in sizes:
        print(f"\n  --- LP with {n_vars} variables ---")
        sub = df.head(n_vars)
        expected_return = (sub["int_rate"].values / 100.0).astype(np.float64)
        pd_default = sub["default_flag"].values.astype(np.float64)
        loan_amounts = sub["loan_amnt"].values.astype(np.float64)
        budget = loan_amounts.sum() * 0.3  # 30% budget
        max_alloc = 0.05  # max 5% per loan

        # --- SciPy HiGHS ---
        print(f"    SciPy HiGHS ...")
        try:
            from scipy.optimize import linprog
            from scipy.sparse import eye as speye

            # Maximize expected_return @ x => minimize -expected_return @ x
            c = -expected_return

            # Constraints:
            # 1. Budget: sum(loan_amnt * x) <= budget
            # 2. Risk: sum(pd * x) <= risk_budget
            risk_budget = n_vars * 0.15  # allow avg 15% default weight

            A_ub = np.vstack([
                loan_amounts.reshape(1, -1),
                pd_default.reshape(1, -1),
            ])
            b_ub = np.array([budget, risk_budget])

            bounds = [(0.0, max_alloc)] * n_vars

            s = time.perf_counter()
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            scipy_t = time.perf_counter() - s
            obj = -res.fun if res.success else None
            rows.append({
                "task": "portfolio_lp", "backend": "scipy_highs_cpu",
                "seconds": scipy_t, "status": res.status,
                "objective": obj, "n_variables": n_vars,
            })
            print(f"      {scipy_t:.4f}s, obj={obj:.2f}, status={res.status}")
        except Exception as e:
            rows.append({"task": "portfolio_lp", "backend": "scipy_highs_cpu",
                         "seconds": None, "status": f"error: {e}",
                         "objective": None, "n_variables": n_vars})
            print(f"      FAILED: {e}")

        # --- cuOpt GPU ---
        print(f"    cuOpt GPU ...")
        try:
            from cuopt.linear_programming import DataModel, Solve, SolverSettings

            dm = DataModel()

            # Objective: maximize expected_return @ x
            dm.set_objective_coefficients(expected_return)
            dm.set_maximize(True)

            # Constraints in CSR format:
            # Row 0: budget constraint: sum(loan_amnt * x) <= budget
            # Row 1: risk constraint: sum(pd * x) <= risk_budget
            n_constraints = 2
            A_values = np.concatenate([loan_amounts, pd_default]).astype(np.float64)
            A_indices = np.concatenate([
                np.arange(n_vars, dtype=np.int32),
                np.arange(n_vars, dtype=np.int32),
            ])
            A_offsets = np.array([0, n_vars, 2 * n_vars], dtype=np.int32)

            dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
            dm.set_constraint_bounds(np.array([budget, risk_budget], dtype=np.float64))
            dm.set_row_types(np.array(["L", "L"]))

            # Variable bounds
            dm.set_variable_lower_bounds(np.zeros(n_vars, dtype=np.float64))
            dm.set_variable_upper_bounds(np.full(n_vars, max_alloc, dtype=np.float64))

            s = time.perf_counter()
            solution = Solve(dm)
            cuopt_t = time.perf_counter() - s

            obj = float(solution.get_primal_objective())
            rows.append({
                "task": "portfolio_lp", "backend": "cuopt_gpu",
                "seconds": cuopt_t, "status": "optimal",
                "objective": obj, "n_variables": n_vars,
            })
            print(f"      {cuopt_t:.4f}s, obj={obj:.2f}")
        except Exception as e:
            traceback.print_exc()
            rows.append({"task": "portfolio_lp", "backend": "cuopt_gpu",
                         "seconds": None, "status": f"error: {e}",
                         "objective": None, "n_variables": n_vars})
            print(f"      FAILED: {e}")

    # MILP reference (one size)
    print(f"\n  --- MILP with 3000 variables ---")
    n_milp = 3000
    sub = df.head(n_milp)
    expected_return = (sub["int_rate"].values / 100.0).astype(np.float64)
    pd_default = sub["default_flag"].values.astype(np.float64)
    loan_amounts = sub["loan_amnt"].values.astype(np.float64)
    budget = loan_amounts.sum() * 0.3
    risk_budget = n_milp * 0.15

    # SciPy MILP
    print(f"    SciPy MILP ...")
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        c_milp = -expected_return
        A_milp = np.vstack([loan_amounts.reshape(1, -1), pd_default.reshape(1, -1)])
        constraints = LinearConstraint(A_milp, ub=[budget, risk_budget])
        integrality = np.ones(n_milp)  # all binary
        bounds_milp = Bounds(lb=0, ub=1)

        s = time.perf_counter()
        res = milp(c_milp, constraints=constraints, integrality=integrality,
                   bounds=bounds_milp)
        milp_t = time.perf_counter() - s
        obj = -res.fun if res.success else None
        rows.append({
            "task": "portfolio_milp", "backend": "scipy_milp_cpu",
            "seconds": milp_t, "status": 0 if res.success else -1,
            "objective": obj, "n_variables": n_milp,
        })
        print(f"      {milp_t:.4f}s, obj={obj:.2f}")
    except Exception as e:
        rows.append({"task": "portfolio_milp", "backend": "scipy_milp_cpu",
                     "seconds": None, "status": f"error: {e}",
                     "objective": None, "n_variables": n_milp})
        print(f"      FAILED: {e}")

    # cuOpt MILP
    print(f"    cuOpt MILP ...")
    try:
        from cuopt.linear_programming import DataModel, Solve

        dm = DataModel()
        dm.set_objective_coefficients(-expected_return)  # minimize negative = maximize

        A_values = np.concatenate([loan_amounts, pd_default]).astype(np.float64)
        A_indices = np.concatenate([
            np.arange(n_milp, dtype=np.int32),
            np.arange(n_milp, dtype=np.int32),
        ])
        A_offsets = np.array([0, n_milp, 2 * n_milp], dtype=np.int32)

        dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
        dm.set_constraint_bounds(np.array([budget, risk_budget], dtype=np.float64))
        dm.set_row_types(np.array(["L", "L"]))
        dm.set_variable_lower_bounds(np.zeros(n_milp, dtype=np.float64))
        dm.set_variable_upper_bounds(np.ones(n_milp, dtype=np.float64))
        dm.set_variable_types(np.array(["I"] * n_milp))  # integer (binary)

        s = time.perf_counter()
        solution = Solve(dm)
        milp_t = time.perf_counter() - s
        obj = -float(solution.get_primal_objective())
        rows.append({
            "task": "portfolio_milp", "backend": "cuopt_milp_gpu",
            "seconds": milp_t, "status": "optimal",
            "objective": obj, "n_variables": n_milp,
        })
        print(f"      {milp_t:.4f}s, obj={obj:.2f}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": "portfolio_milp", "backend": "cuopt_milp_gpu",
                     "seconds": None, "status": f"error: {e}",
                     "objective": None, "n_variables": n_milp})
        print(f"      FAILED: {e}")

    # Compute speedups
    df_out = pd.DataFrame(rows)
    # LP speedups per size
    for nv in sizes:
        cpu_row = df_out[(df_out["backend"] == "scipy_highs_cpu") &
                         (df_out["n_variables"] == nv)]
        gpu_row = df_out[(df_out["backend"] == "cuopt_gpu") &
                         (df_out["n_variables"] == nv)]
        if not cpu_row.empty and not gpu_row.empty:
            ct = cpu_row["seconds"].values[0]
            gt = gpu_row["seconds"].values[0]
            if ct and gt and gt > 0:
                df_out.loc[gpu_row.index, "speedup_vs_cpu_lp"] = ct / gt

    df_out.to_csv(OUT_DIR / "cuopt_benchmark.csv", index=False)
    print(f"\n  Saved: cuopt_benchmark.csv ({len(df_out)} rows)")
    return df_out


# ===================================================================
# 5. CuPy Numerical Computing Benchmark
# ===================================================================
def run_cupy_benchmarks():
    """Benchmark NumPy/SciPy CPU vs CuPy GPU."""
    print("\n" + "=" * 60)
    print("SECTION 5: Numerical Computing (CuPy)")
    print("=" * 60)
    rows = []

    # --- Monte Carlo ECL simulation ---
    print("\n  --- Monte Carlo ECL (100K scenarios) ---")
    n_loans = 10000
    n_scenarios = int(_profile_cfg()["cupy_n_scenarios"])
    rng = np.random.default_rng(42)
    pd_vals = rng.uniform(0.01, 0.40, n_loans).astype(np.float64)
    lgd_vals = rng.uniform(0.20, 0.80, n_loans).astype(np.float64)
    ead_vals = rng.uniform(5000, 50000, n_loans).astype(np.float64)

    # NumPy CPU
    print("    NumPy CPU ...")
    try:
        s = time.perf_counter()
        defaults = rng.random((n_scenarios, n_loans)) < pd_vals
        losses = defaults * lgd_vals * ead_vals
        ecl_dist = losses.sum(axis=1)
        ecl_mean = ecl_dist.mean()
        ecl_var95 = np.percentile(ecl_dist, 95)
        numpy_t = time.perf_counter() - s
        rows.append({"task": "monte_carlo_ecl", "backend": "numpy_cpu",
                      "seconds": numpy_t, "metric": "ecl_mean",
                      "metric_value": float(ecl_mean)})
        print(f"      {numpy_t:.4f}s, ECL_mean={ecl_mean:.0f}, VaR95={ecl_var95:.0f}")
    except Exception as e:
        rows.append({"task": "monte_carlo_ecl", "backend": "numpy_cpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # CuPy GPU
    print("    CuPy GPU ...")
    try:
        import cupy as cp

        pd_gpu = cp.asarray(pd_vals)
        lgd_gpu = cp.asarray(lgd_vals)
        ead_gpu = cp.asarray(ead_vals)

        s = time.perf_counter()
        rng_gpu = cp.random.default_rng(42)
        defaults = rng_gpu.random((n_scenarios, n_loans)) < pd_gpu
        losses = defaults * lgd_gpu * ead_gpu
        ecl_dist = losses.sum(axis=1)
        ecl_mean = float(ecl_dist.mean())
        ecl_var95 = float(cp.percentile(ecl_dist, 95))
        cp.cuda.Stream.null.synchronize()
        cupy_t = time.perf_counter() - s
        rows.append({"task": "monte_carlo_ecl", "backend": "cupy_gpu",
                      "seconds": cupy_t, "metric": "ecl_mean",
                      "metric_value": ecl_mean})
        print(f"      {cupy_t:.4f}s, ECL_mean={ecl_mean:.0f}, VaR95={ecl_var95:.0f}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": "monte_carlo_ecl", "backend": "cupy_gpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # --- SVD ---
    print("\n  --- SVD (features matrix) ---")
    df = pd.read_parquet(str(TRAIN_FE_PQ))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "default_flag"][:30]
    X = df[num_cols].dropna().head(int(_profile_cfg()["cupy_svd_rows"])).values.astype(np.float64)
    print(f"    Matrix: {X.shape}")

    # NumPy CPU
    print("    NumPy CPU ...")
    try:
        s = time.perf_counter()
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        numpy_t = time.perf_counter() - s
        rows.append({"task": "svd", "backend": "numpy_cpu",
                      "seconds": numpy_t, "metric": "top_sv",
                      "metric_value": float(S[0])})
        print(f"      {numpy_t:.4f}s, top_sv={S[0]:.2f}")
    except Exception as e:
        rows.append({"task": "svd", "backend": "numpy_cpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # CuPy GPU
    print("    CuPy GPU ...")
    try:
        import cupy as cp

        X_gpu = cp.asarray(X)
        s = time.perf_counter()
        U, S, Vt = cp.linalg.svd(X_gpu, full_matrices=False)
        cp.cuda.Stream.null.synchronize()
        cupy_t = time.perf_counter() - s
        rows.append({"task": "svd", "backend": "cupy_gpu",
                      "seconds": cupy_t, "metric": "top_sv",
                      "metric_value": float(S[0])})
        print(f"      {cupy_t:.4f}s, top_sv={float(S[0]):.2f}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": "svd", "backend": "cupy_gpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # --- Sparse matrix multiply ---
    print("\n  --- Sparse Matrix Multiply ---")
    from scipy import sparse as sp

    n_sp = int(_profile_cfg()["cupy_sparse_n"])
    density = 0.01
    print(f"    Sparse: {n_sp}x{n_sp}, density={density}")

    # SciPy CPU
    print("    SciPy CPU ...")
    try:
        A = sp.random(n_sp, n_sp, density=density, format="csr", dtype=np.float64,
                      random_state=42)
        B = sp.random(n_sp, n_sp, density=density, format="csr", dtype=np.float64,
                      random_state=43)
        s = time.perf_counter()
        C = A @ B
        scipy_t = time.perf_counter() - s
        nnz = C.nnz
        rows.append({"task": "sparse_matmul", "backend": "scipy_cpu",
                      "seconds": scipy_t, "metric": "nnz",
                      "metric_value": float(nnz)})
        print(f"      {scipy_t:.4f}s, nnz={nnz}")
    except Exception as e:
        rows.append({"task": "sparse_matmul", "backend": "scipy_cpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # CuPy GPU
    print("    CuPy GPU ...")
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cusp

        A_gpu = cusp.csr_matrix(A)
        B_gpu = cusp.csr_matrix(B)
        s = time.perf_counter()
        C_gpu = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()
        cupy_t = time.perf_counter() - s
        nnz_gpu = C_gpu.nnz
        rows.append({"task": "sparse_matmul", "backend": "cupy_gpu",
                      "seconds": cupy_t, "metric": "nnz",
                      "metric_value": float(nnz_gpu)})
        print(f"      {cupy_t:.4f}s, nnz={nnz_gpu}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": "sparse_matmul", "backend": "cupy_gpu",
                      "seconds": None, "metric": "error", "metric_value": str(e)})
        print(f"      FAILED: {e}")

    # Compute speedups
    df_out = pd.DataFrame(rows)
    for task in ["monte_carlo_ecl", "svd", "sparse_matmul"]:
        cpu_backend = "numpy_cpu" if task != "sparse_matmul" else "scipy_cpu"
        cpu_row = df_out[(df_out["task"] == task) & (df_out["backend"] == cpu_backend)]
        gpu_row = df_out[(df_out["task"] == task) & (df_out["backend"] == "cupy_gpu")]
        if not cpu_row.empty and not gpu_row.empty:
            ct = cpu_row["seconds"].values[0]
            gt = gpu_row["seconds"].values[0]
            if ct and gt and gt > 0:
                df_out.loc[gpu_row.index, "speedup_vs_cpu"] = ct / gt

    df_out.to_csv(OUT_DIR / "cupy_benchmark.csv", index=False)
    print(f"\n  Saved: cupy_benchmark.csv ({len(df_out)} rows)")
    return df_out


# ===================================================================
# 6. Metadata
# ===================================================================
def save_metadata():
    """Save hardware and version metadata."""
    import subprocess
    meta = {
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3080",
            "vram_mb": 10240,
            "cpu": "AMD Ryzen 5 5600X",
            "ram_gb": 24,
            "platform": "WSL2",
        },
        "versions": {},
        "config": {
            "profile": BENCH_PROFILE,
            "repeats": REPEATS,
            "warmup": WARMUP,
        },
    }
    for lib in ["cudf", "cuml", "cugraph", "cuopt", "cupy", "polars", "duckdb",
                "pandas", "numpy", "scipy", "sklearn"]:
        try:
            m = __import__(lib)
            meta["versions"][lib] = m.__version__
        except Exception:
            meta["versions"][lib] = "not installed"

    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                           "--format=csv,noheader"], capture_output=True, text=True)
        meta["hardware"]["driver"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.gr,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if line:
            vals = [x.strip() for x in line.split(",")]
            if len(vals) == 5:
                meta["hardware"].update(
                    {
                        "graphics_clock_mhz": int(vals[0]),
                        "temperature_c": int(vals[1]),
                        "gpu_utilization_pct": int(vals[2]),
                        "memory_used_mb": int(vals[3]),
                        "memory_total_mb_reported": int(vals[4]),
                    }
                )
    except Exception:
        pass
    try:
        r = subprocess.run(["free", "-m"], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            meta["hardware"]["wsl_free_m"] = r.stdout
    except Exception:
        pass

    with open(OUT_DIR / "gpu_bench_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved: gpu_bench_meta.json")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPIDS GPU Benchmark Suite")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="current")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    BENCH_PROFILE = args.profile
    cfg = _profile_cfg()
    REPEATS = int(args.repeats if args.repeats is not None else cfg["repeats"])
    WARMUP = int(args.warmup if args.warmup is not None else cfg["warmup"])

    print("=" * 60)
    print("RAPIDS GPU Benchmark Suite")
    print("=" * 60)
    print(f"Profile: {BENCH_PROFILE} (repeats={REPEATS}, warmup={WARMUP})")

    t0 = time.perf_counter()

    df_df = run_dataframe_benchmarks()
    df_ml = run_ml_benchmarks()
    df_graph = run_graph_benchmarks()
    df_opt = run_optimization_benchmarks()
    df_cupy = run_cupy_benchmarks()
    save_metadata()

    total = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"ALL BENCHMARKS COMPLETE in {total:.1f}s")
    print(f"{'=' * 60}")
