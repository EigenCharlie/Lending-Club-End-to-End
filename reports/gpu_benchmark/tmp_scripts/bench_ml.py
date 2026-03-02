"""ML benchmark: sklearn CPU vs cuML GPU.

Uses train_fe.parquet with all numeric features.
Tests: LogisticRegression, RandomForest, KMeans, PCA, KNN, UMAP, HDBSCAN
Reports: fit time, predict time, metrics, output agreement.
"""
import gc
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, silhouette_score

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_FE = os.path.join(PROJECT, "data", "processed", "train_fe.parquet")
OUT = os.path.join(PROJECT, "reports", "gpu_benchmark")

# ── Prepare data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(TRAIN_FE)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "default_flag"]
# Use all numeric features for maximum stress
X_all = df[num_cols].values.astype(np.float32)
y_all = df["default_flag"].values.astype(np.int32)

# Handle NaN: fill with column means for sklearn (cuML also handles NaN poorly for some algos)
col_means = np.nanmean(X_all, axis=0)
nan_mask = np.isnan(X_all)
X_all[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

N_TRAIN = int(os.getenv("LC_RAPIDS_ML_N_TRAIN", "500000"))
N_TEST = int(os.getenv("LC_RAPIDS_ML_N_TEST", "100000"))
X_train = X_all[:N_TRAIN]
y_train = y_all[:N_TRAIN]
X_test = X_all[N_TRAIN:N_TRAIN + N_TEST]
y_test = y_all[N_TRAIN:N_TRAIN + N_TEST]

# Smaller subsets for expensive algos
N_KM = int(os.getenv("LC_RAPIDS_ML_N_KM", "100000"))
N_KNN = int(os.getenv("LC_RAPIDS_ML_N_KNN", "80000"))
N_KNN_TEST = int(os.getenv("LC_RAPIDS_ML_N_KNN_TEST", "20000"))
N_UMAP = int(os.getenv("LC_RAPIDS_ML_N_UMAP", "50000"))
N_HDBSCAN = int(os.getenv("LC_RAPIDS_ML_N_HDBSCAN", "50000"))

print(f"  Full: {len(X_all):,} rows, {X_all.shape[1]} features")
print(f"  Train: {N_TRAIN:,}, Test: {N_TEST:,}")
print(f"  KMeans: {N_KM:,}, KNN: {N_KNN:,}/{N_KNN_TEST:,}")
print(f"  UMAP/HDBSCAN: {N_UMAP:,}/{N_HDBSCAN:,}")

rows = []


def run_benchmark(task_name, cpu_fn, gpu_fn):
    """Run CPU and GPU benchmarks, comparing outputs."""
    print(f"\n{'─'*50}")
    print(f"  {task_name}")
    print(f"{'─'*50}")
    cpu_result = None
    gpu_result = None

    # CPU
    print(f"  [CPU] sklearn ...")
    try:
        gc.collect()
        cpu_result = cpu_fn()
        rows.append({"task": task_name, "backend": "sklearn_cpu", **cpu_result})
        print(f"    fit={cpu_result['fit_seconds']:.3f}s, "
              f"pred={cpu_result['predict_seconds']:.3f}s, "
              f"{cpu_result['metric']}={cpu_result['metric_value']:.6f}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": task_name, "backend": "sklearn_cpu",
                     "fit_seconds": None, "predict_seconds": None,
                     "metric": "error", "metric_value": str(e)})
        print(f"    FAILED: {e}")

    # GPU
    print(f"  [GPU] cuML ...")
    try:
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        gpu_result = gpu_fn()
        rows.append({"task": task_name, "backend": "cuml_gpu", **gpu_result})
        print(f"    fit={gpu_result['fit_seconds']:.3f}s, "
              f"pred={gpu_result['predict_seconds']:.3f}s, "
              f"{gpu_result['metric']}={gpu_result['metric_value']:.6f}")
    except Exception as e:
        traceback.print_exc()
        rows.append({"task": task_name, "backend": "cuml_gpu",
                     "fit_seconds": None, "predict_seconds": None,
                     "metric": "error", "metric_value": str(e)})
        print(f"    FAILED: {e}")

    # Compare
    if (cpu_result and gpu_result
            and cpu_result.get("fit_seconds") and gpu_result.get("fit_seconds")
            and gpu_result["fit_seconds"] > 0):
        speedup = cpu_result["fit_seconds"] / gpu_result["fit_seconds"]
        if cpu_result["metric"] == gpu_result["metric"] and cpu_result["metric"] != "error":
            cv = cpu_result["metric_value"]
            gv = gpu_result["metric_value"]
            if isinstance(cv, (int, float)) and isinstance(gv, (int, float)):
                diff = abs(cv - gv)
                rel = diff / max(abs(cv), 1e-10) * 100
                print(f"  [COMPARE] Speedup: {speedup:.2f}x | "
                      f"Metric diff: {diff:.6f} ({rel:.2f}% relative)")


# ═══════════════════════════════════════════
# 1. Logistic Regression
# ═══════════════════════════════════════════
def lr_cpu():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=-1, C=1.0)
    s = time.perf_counter()
    model.fit(X_train, y_train)
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_test)[:, 1]
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_test, preds)}

def lr_gpu():
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLR
    X_tr, y_tr = cp.asarray(X_train), cp.asarray(y_train)
    X_te = cp.asarray(X_test)
    model = cuLR(max_iter=300, C=1.0)
    s = time.perf_counter()
    model.fit(X_tr, y_tr)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_te)[:, 1]
    cp.cuda.Stream.null.synchronize()
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_test, cp.asnumpy(preds))}

run_benchmark("logistic_regression", lr_cpu, lr_gpu)


# ═══════════════════════════════════════════
# 2. Random Forest
# ═══════════════════════════════════════════
def rf_cpu():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1, random_state=42)
    s = time.perf_counter()
    model.fit(X_train, y_train)
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_test)[:, 1]
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_test, preds)}

def rf_gpu():
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    X_tr, y_tr = cp.asarray(X_train), cp.asarray(y_train)
    X_te = cp.asarray(X_test)
    model = cuRF(n_estimators=200, max_depth=16, random_state=42)
    s = time.perf_counter()
    model.fit(X_tr, y_tr)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_te)[:, 1]
    cp.cuda.Stream.null.synchronize()
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_test, cp.asnumpy(preds))}

run_benchmark("random_forest", rf_cpu, rf_gpu)


# ═══════════════════════════════════════════
# 3. KMeans
# ═══════════════════════════════════════════
def km_cpu():
    from sklearn.cluster import KMeans
    X_km = X_train[:N_KM]
    model = KMeans(n_clusters=7, n_init=3, max_iter=300, random_state=42)
    s = time.perf_counter()
    labels = model.fit_predict(X_km)
    fit_t = time.perf_counter() - s
    sil = silhouette_score(X_km[:10000], labels[:10000])
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "silhouette", "metric_value": sil}

def km_gpu():
    import cupy as cp
    from cuml.cluster import KMeans as cuKM
    X_km_gpu = cp.asarray(X_train[:N_KM])
    model = cuKM(n_clusters=7, n_init=3, max_iter=300, random_state=42)
    s = time.perf_counter()
    labels = model.fit_predict(X_km_gpu)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    sil = silhouette_score(X_train[:N_KM][:10000], cp.asnumpy(labels)[:10000])
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "silhouette", "metric_value": sil}

run_benchmark("kmeans", km_cpu, km_gpu)


# ═══════════════════════════════════════════
# 4. PCA
# ═══════════════════════════════════════════
def pca_cpu():
    from sklearn.decomposition import PCA
    model = PCA(n_components=10)
    s = time.perf_counter()
    model.fit(X_train)
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    _ = model.transform(X_test)
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "explained_var", "metric_value": float(model.explained_variance_ratio_.sum())}

def pca_gpu():
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    X_tr_gpu = cp.asarray(X_train)
    X_te_gpu = cp.asarray(X_test)
    model = cuPCA(n_components=10)
    s = time.perf_counter()
    model.fit(X_tr_gpu)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    _ = model.transform(X_te_gpu)
    cp.cuda.Stream.null.synchronize()
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "explained_var",
            "metric_value": float(cp.asnumpy(model.explained_variance_ratio_).sum())}

run_benchmark("pca", pca_cpu, pca_gpu)


# ═══════════════════════════════════════════
# 5. KNN
# ═══════════════════════════════════════════
def knn_cpu():
    from sklearn.neighbors import KNeighborsClassifier
    X_tr = X_train[:N_KNN]
    y_tr = y_train[:N_KNN]
    X_te = X_test[:N_KNN_TEST]
    y_te = y_test[:N_KNN_TEST]
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    s = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_te)[:, 1]
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_te, preds)}

def knn_gpu():
    import cupy as cp
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    X_tr = cp.asarray(X_train[:N_KNN])
    y_tr = cp.asarray(y_train[:N_KNN])
    X_te = cp.asarray(X_test[:N_KNN_TEST])
    model = cuKNN(n_neighbors=5)
    s = time.perf_counter()
    model.fit(X_tr, y_tr)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    s = time.perf_counter()
    preds = model.predict_proba(X_te)[:, 1]
    cp.cuda.Stream.null.synchronize()
    pred_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": pred_t,
            "metric": "auc", "metric_value": roc_auc_score(y_test[:N_KNN_TEST], cp.asnumpy(preds))}

run_benchmark("knn", knn_cpu, knn_gpu)


# ═══════════════════════════════════════════
# 6. UMAP
# ═══════════════════════════════════════════
def umap_cpu():
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn not installed (CPU)")
    X_um = X_train[:N_UMAP]
    model = UMAP(n_components=2, n_neighbors=15, random_state=42)
    s = time.perf_counter()
    emb = model.fit_transform(X_um)
    fit_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "n_embedded", "metric_value": float(emb.shape[0])}

def umap_gpu():
    import cupy as cp
    from cuml.manifold import UMAP as cuUMAP
    X_um = cp.asarray(X_train[:N_UMAP])
    model = cuUMAP(n_components=2, n_neighbors=15, random_state=42)
    s = time.perf_counter()
    emb = model.fit_transform(X_um)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "n_embedded", "metric_value": float(emb.shape[0])}

run_benchmark("umap", umap_cpu, umap_gpu)


# ═══════════════════════════════════════════
# 7. HDBSCAN
# ═══════════════════════════════════════════
def hdbscan_cpu():
    from sklearn.cluster import HDBSCAN as SkHDB
    X_hdb = X_train[:N_HDBSCAN]
    model = SkHDB(min_cluster_size=50)
    s = time.perf_counter()
    labels = model.fit_predict(X_hdb)
    fit_t = time.perf_counter() - s
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "n_clusters", "metric_value": float(n_cl)}

def hdbscan_gpu():
    import cupy as cp
    from cuml.cluster import HDBSCAN as cuHDB
    X_hdb = cp.asarray(X_train[:N_HDBSCAN])
    model = cuHDB(min_cluster_size=50)
    s = time.perf_counter()
    labels = model.fit_predict(X_hdb)
    cp.cuda.Stream.null.synchronize()
    fit_t = time.perf_counter() - s
    labels_cpu = cp.asnumpy(labels)
    n_cl = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)
    return {"fit_seconds": fit_t, "predict_seconds": 0.0,
            "metric": "n_clusters", "metric_value": float(n_cl)}

run_benchmark("hdbscan", hdbscan_cpu, hdbscan_gpu)


# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
df = pd.DataFrame(rows)
# Compute speedups
cpu_fit = df[df["backend"] == "sklearn_cpu"].set_index("task")["fit_seconds"]
gpu_fit = df[df["backend"] == "cuml_gpu"].set_index("task")["fit_seconds"]
speedups = cpu_fit / gpu_fit
df["fit_speedup_vs_cpu"] = df.apply(
    lambda r: speedups.get(r["task"]) if r["backend"] == "cuml_gpu" else None, axis=1)
df.to_csv(os.path.join(OUT, "cuml_benchmark.csv"), index=False)
print(f"\nSaved cuml_benchmark.csv ({len(df)} rows)")
print(df[["task", "backend", "fit_seconds", "metric", "metric_value"]].to_string(index=False))
