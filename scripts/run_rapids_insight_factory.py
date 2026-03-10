"""RAPIDS insight factory on real project workloads.

This script focuses on GPU-heavy exploratory workloads that complement the
canonical CPU pipeline:

- cuDF for heavy ETL/aggregation on engineered features
- cuML for large-scale segmentation
- cuGraph for loan-similarity graph analytics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - intended for RAPIDS env
    import cudf
except Exception:  # pragma: no cover
    cudf = None

try:  # pragma: no cover - intended for RAPIDS env
    import cugraph
except Exception:  # pragma: no cover
    cugraph = None

try:  # pragma: no cover - intended for RAPIDS env
    from cuml.cluster import KMeans as CuKMeans
    from cuml.neighbors import NearestNeighbors as CuNearestNeighbors
except Exception:  # pragma: no cover
    CuKMeans = None
    CuNearestNeighbors = None


def _artifact_root(run_tag: str) -> Path:
    raw = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    if raw:
        return Path(raw)
    return ROOT / "reports" / "gpu_insights" / run_tag / "artifacts"


def _select_numeric_features(df: pd.DataFrame, max_features: int = 24) -> list[str]:
    excluded = {"id", "default_flag"}
    preferred = [
        "loan_to_income",
        "revol_bal_to_income",
        "installment",
        "emp_length_num",
        "fico_score",
        "credit_age_years",
        "installment_burden",
        "log_revol_bal",
        "int_rate",
        "log_annual_inc",
        "annual_inc",
        "open_acc_ratio",
        "loan_amnt",
        "total_acc",
        "pub_rec",
        "il_ratio",
        "loan_to_income_sq",
        "open_acc",
        "rev_utilization",
        "high_util_pct",
        "dti",
        "inq_last_6mths",
        "mort_acc",
        "fico_x_dti",
    ]
    numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    ordered = [c for c in preferred if c in numeric]
    extras = [c for c in numeric if c not in ordered]
    return (ordered + extras)[:max_features]


def _prepare_issue_quarter(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("Q").astype(str)


def _run_cudf_etl(
    *,
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    test_with_intervals = test_fe.merge(
        intervals[["id", "pd_high_90", "width_90"]],
        on="id",
        how="left",
    )
    test_with_intervals["pd_high_90"] = test_with_intervals["pd_high_90"].fillna(0.0)
    test_with_intervals["width_90"] = test_with_intervals["width_90"].fillna(0.0)
    train_with_intervals = train_fe.copy()
    train_with_intervals["pd_high_90"] = 0.0
    train_with_intervals["width_90"] = 0.0
    train_with_intervals["split"] = "train"
    test_with_intervals["split"] = "test"
    combined = pd.concat([train_with_intervals, test_with_intervals], ignore_index=True)
    combined["issue_quarter"] = _prepare_issue_quarter(combined["issue_d"])

    group_cols = [
        "split",
        "issue_quarter",
        "grade",
        "term",
        "home_ownership",
        "verification_status",
    ]
    agg_spec = {
        "id": "count",
        "default_flag": "mean",
        "int_rate": "mean",
        "loan_amnt": "mean",
        "fico_score": "mean",
        "rev_utilization": "mean",
        "pd_high_90": "mean",
        "width_90": "mean",
    }

    cpu_start = time.perf_counter()
    cpu_out = (
        combined.groupby(group_cols, dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"id": "n_loans", "default_flag": "default_rate"})
    )
    cpu_seconds = time.perf_counter() - cpu_start

    if cudf is None:  # pragma: no cover
        raise RuntimeError("cudf is not available in this environment.")
    gtrain = cudf.from_pandas(train_fe)
    gtest = cudf.from_pandas(test_fe)
    gintervals = cudf.from_pandas(intervals[["id", "pd_high_90", "width_90"]])
    gpu_start = time.perf_counter()
    gtest = gtest.merge(gintervals, on="id", how="left")
    gtest["pd_high_90"] = gtest["pd_high_90"].fillna(0.0)
    gtest["width_90"] = gtest["width_90"].fillna(0.0)
    gtrain["pd_high_90"] = 0.0
    gtrain["width_90"] = 0.0
    gtrain["split"] = "train"
    gtest["split"] = "test"
    gtrain["issue_quarter"] = cudf.Series(_prepare_issue_quarter(train_fe["issue_d"]))
    gtest["issue_quarter"] = cudf.Series(_prepare_issue_quarter(test_fe["issue_d"]))
    gall = cudf.concat([gtrain, gtest], ignore_index=True)
    gpu_out = (
        gall.groupby(group_cols, dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"id": "n_loans", "default_flag": "default_rate"})
        .to_pandas()
    )
    gpu_seconds = time.perf_counter() - gpu_start

    cpu_sorted = cpu_out.sort_values(group_cols).reset_index(drop=True)
    gpu_sorted = gpu_out.sort_values(group_cols).reset_index(drop=True)
    checksum_cols = [
        "n_loans",
        "default_rate",
        "int_rate",
        "loan_amnt",
        "fico_score",
        "rev_utilization",
    ]
    aligned = cpu_sorted.merge(
        gpu_sorted,
        on=group_cols,
        how="outer",
        suffixes=("_cpu", "_gpu"),
    ).fillna(0.0)
    matched = cpu_sorted.merge(
        gpu_sorted,
        on=group_cols,
        how="inner",
        suffixes=("_cpu", "_gpu"),
    ).fillna(0.0)
    cpu_values = matched[[f"{col}_cpu" for col in checksum_cols]].to_numpy(dtype=float)
    gpu_values = matched[[f"{col}_gpu" for col in checksum_cols]].to_numpy(dtype=float)
    rel_err = np.abs(cpu_values - gpu_values) / np.maximum(np.abs(cpu_values), 1e-9)

    summary = {
        "stage": "cudf_etl",
        "rows_input": int(len(combined)),
        "rows_output": int(len(aligned)),
        "matched_groups": int(len(matched)),
        "row_count_diff": int(abs(len(cpu_sorted) - len(gpu_sorted))),
        "cpu_seconds": float(cpu_seconds),
        "gpu_seconds": float(gpu_seconds),
        "speedup_gpu_vs_cpu": float(cpu_seconds / max(gpu_seconds, 1e-9)),
        "max_rel_err_pct_matched": float(rel_err.max() * 100.0) if len(matched) else 0.0,
    }
    return summary, gpu_sorted


def _run_cuml_segmentation(
    *,
    train_fe: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    feature_cols = _select_numeric_features(train_fe)
    sample = train_fe.sample(n=min(sample_size, len(train_fe)), random_state=random_state).copy()
    X = sample[feature_cols].fillna(0.0).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cpu_start = time.perf_counter()
    km_cpu = KMeans(n_clusters=8, n_init=10, random_state=random_state)
    cpu_labels = km_cpu.fit_predict(Xs)
    cpu_seconds = time.perf_counter() - cpu_start

    if CuKMeans is None:  # pragma: no cover
        raise RuntimeError("cuML KMeans is not available in this environment.")
    gpu_start = time.perf_counter()
    km_gpu = CuKMeans(n_clusters=8, random_state=random_state, n_init=10, max_iter=300)
    gpu_labels = np.asarray(km_gpu.fit_predict(Xs), dtype=int)
    gpu_seconds = time.perf_counter() - gpu_start

    cpu_counts = pd.Series(cpu_labels).value_counts().sort_index()
    gpu_counts = pd.Series(gpu_labels).value_counts().sort_index()
    cpu_sorted_sizes = np.sort(cpu_counts.reindex(range(8), fill_value=0).to_numpy(dtype=float))
    gpu_sorted_sizes = np.sort(gpu_counts.reindex(range(8), fill_value=0).to_numpy(dtype=float))
    summary = {
        "stage": "cuml_segmentation",
        "rows_input": int(len(sample)),
        "n_features": int(len(feature_cols)),
        "cpu_seconds": float(cpu_seconds),
        "gpu_seconds": float(gpu_seconds),
        "speedup_gpu_vs_cpu": float(cpu_seconds / max(gpu_seconds, 1e-9)),
        "cpu_inertia": float(km_cpu.inertia_),
        "gpu_inertia": float(float(km_gpu.inertia_)),
        "inertia_rel_diff_pct": float(
            abs(float(km_cpu.inertia_) - float(km_gpu.inertia_))
            / max(abs(float(km_cpu.inertia_)), 1e-9)
            * 100.0
        ),
        "cluster_size_l1_diff_sorted": float(np.abs(cpu_sorted_sizes - gpu_sorted_sizes).sum()),
    }
    cluster_summary = pd.DataFrame(
        {
            "cluster": range(8),
            "cpu_count": cpu_counts.reindex(range(8), fill_value=0).values,
            "gpu_count": gpu_counts.reindex(range(8), fill_value=0).values,
        }
    )
    return summary, cluster_summary


def _build_knn_edges_cpu(X: np.ndarray, k: int) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    src = np.repeat(np.arange(len(X), dtype=np.int32), k)
    dst = idx[:, 1 : k + 1].reshape(-1).astype(np.int32)
    return pd.DataFrame({"src": src, "dst": dst})


def _run_cugraph_similarity(
    *,
    train_fe: pd.DataFrame,
    sample_size: int,
    knn_k: int,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    feature_cols = _select_numeric_features(train_fe, max_features=12)
    sample = train_fe.sample(n=min(sample_size, len(train_fe)), random_state=random_state).copy()
    X = sample[feature_cols].fillna(0.0).astype(np.float32)
    Xs = StandardScaler().fit_transform(X)

    cpu_start = time.perf_counter()
    cpu_edges = _build_knn_edges_cpu(Xs, knn_k)
    import networkx as nx

    g_cpu = nx.from_pandas_edgelist(cpu_edges, source="src", target="dst")
    cpu_components = nx.number_connected_components(g_cpu)
    cpu_pagerank = nx.pagerank(g_cpu, alpha=0.85, max_iter=200, tol=1e-6)
    cpu_seconds = time.perf_counter() - cpu_start

    if cudf is None or cugraph is None or CuNearestNeighbors is None:  # pragma: no cover
        raise RuntimeError(
            "cudf/cugraph/cuML NearestNeighbors are not available in this environment."
        )

    gpu_start = time.perf_counter()
    Xg = cudf.DataFrame(Xs)
    nn_gpu = CuNearestNeighbors(n_neighbors=knn_k + 1, output_type="numpy")
    nn_gpu.fit(Xg)
    _, idx_gpu = nn_gpu.kneighbors(Xg)
    src_gpu = np.repeat(np.arange(len(Xs), dtype=np.int32), knn_k)
    dst_gpu = idx_gpu[:, 1 : knn_k + 1].reshape(-1).astype(np.int32)
    gpu_edges = cudf.DataFrame({"src": src_gpu, "dst": dst_gpu})
    g_gpu = cugraph.Graph(directed=False)
    g_gpu.from_cudf_edgelist(gpu_edges, source="src", destination="dst", renumber=False)
    cc_gpu = cugraph.connected_components(g_gpu)
    pr_gpu = cugraph.pagerank(g_gpu)
    gpu_seconds = time.perf_counter() - gpu_start

    summary = {
        "stage": "cugraph_similarity",
        "rows_input": int(len(sample)),
        "n_features": int(len(feature_cols)),
        "knn_k": int(knn_k),
        "cpu_seconds": float(cpu_seconds),
        "gpu_seconds": float(gpu_seconds),
        "speedup_gpu_vs_cpu": float(cpu_seconds / max(gpu_seconds, 1e-9)),
        "cpu_components": int(cpu_components),
        "gpu_components": int(cc_gpu["labels"].nunique()),
        "cpu_pagerank_sum": float(sum(cpu_pagerank.values())),
        "gpu_pagerank_sum": float(pr_gpu["pagerank"].sum()),
    }
    detail = pd.DataFrame(
        {
            "metric": ["components", "pagerank_sum", "edges"],
            "cpu_value": [cpu_components, sum(cpu_pagerank.values()), len(cpu_edges)],
            "gpu_value": [
                int(cc_gpu["labels"].nunique()),
                float(pr_gpu["pagerank"].sum()),
                len(src_gpu),
            ],
        }
    )
    return summary, detail


def main(
    *,
    run_tag: str,
    cluster_sample: int,
    graph_sample: int,
    knn_k: int,
    random_state: int,
) -> None:
    if cudf is None:
        raise RuntimeError("This script must run in the RAPIDS env.")

    root = _artifact_root(run_tag)
    data_dir = root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_fe = pd.read_parquet(ROOT / "data" / "processed" / "train_fe.parquet")
    test_fe = pd.read_parquet(ROOT / "data" / "processed" / "test_fe.parquet")
    intervals = pd.read_parquet(
        ROOT / "data" / "processed" / "conformal_intervals_mondrian.parquet"
    )

    summaries: list[dict[str, object]] = []

    etl_summary, etl_df = _run_cudf_etl(train_fe=train_fe, test_fe=test_fe, intervals=intervals)
    summaries.append(etl_summary)
    etl_df.to_parquet(data_dir / "cudf_etl_summary.parquet", index=False)
    logger.info(
        "cuDF ETL | rows={} cpu_s={:.3f} gpu_s={:.3f} speedup={:.2f}x",
        etl_summary["rows_input"],
        etl_summary["cpu_seconds"],
        etl_summary["gpu_seconds"],
        etl_summary["speedup_gpu_vs_cpu"],
    )

    cuml_summary, cluster_df = _run_cuml_segmentation(
        train_fe=train_fe,
        sample_size=cluster_sample,
        random_state=random_state,
    )
    summaries.append(cuml_summary)
    cluster_df.to_parquet(data_dir / "cuml_segmentation_summary.parquet", index=False)
    logger.info(
        "cuML segmentation | rows={} cpu_s={:.3f} gpu_s={:.3f} speedup={:.2f}x",
        cuml_summary["rows_input"],
        cuml_summary["cpu_seconds"],
        cuml_summary["gpu_seconds"],
        cuml_summary["speedup_gpu_vs_cpu"],
    )

    cugraph_summary, graph_df = _run_cugraph_similarity(
        train_fe=train_fe,
        sample_size=graph_sample,
        knn_k=knn_k,
        random_state=random_state,
    )
    summaries.append(cugraph_summary)
    graph_df.to_parquet(data_dir / "cugraph_similarity_summary.parquet", index=False)
    logger.info(
        "cuGraph similarity | rows={} cpu_s={:.3f} gpu_s={:.3f} speedup={:.2f}x",
        cugraph_summary["rows_input"],
        cugraph_summary["cpu_seconds"],
        cugraph_summary["gpu_seconds"],
        cugraph_summary["speedup_gpu_vs_cpu"],
    )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_parquet(data_dir / "rapids_insight_factory_summary.parquet", index=False)
    (root / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "2026-03-10.1",
                "run_tag": run_tag,
                "artifact_root": str(root),
                "stages": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="2026-03-10-rapids-insight-factory")
    parser.add_argument("--cluster-sample", type=int, default=150_000)
    parser.add_argument("--graph-sample", type=int, default=20_000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(
        run_tag=args.run_tag,
        cluster_sample=args.cluster_sample,
        graph_sample=args.graph_sample,
        knn_k=args.knn_k,
        random_state=args.random_state,
    )
