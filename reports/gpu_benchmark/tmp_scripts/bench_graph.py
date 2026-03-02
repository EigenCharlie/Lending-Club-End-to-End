"""Graph analytics benchmark: NetworkX CPU vs cuGraph GPU vs nx-cugraph.

Builds a borrower-attribute graph from the full training set.
Tests: graph build, connected_components, pagerank, louvain, betweenness_centrality
"""
import gc
import os
import time
import traceback

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN = os.path.join(PROJECT, "data", "processed", "train.parquet")
OUT = os.path.join(PROJECT, "reports", "gpu_benchmark")

# ── Build edge list ─────────────────────────────────────────────────
print("Building edge list from training data...")
df = pd.read_parquet(TRAIN, columns=["id", "grade", "purpose", "sub_grade", "home_ownership"])
df = df.dropna(subset=["grade", "purpose"])
# Use all data (1.3M+ rows) for a large graph
MAX_NODES = int(os.getenv("LC_RAPIDS_GRAPH_MAX_NODES", "200000"))  # tractable cap
df = df.head(MAX_NODES)

# Edges: borrower → grade, borrower → purpose, borrower → sub_grade
edges = []
for col, prefix in [("grade", "G_"), ("purpose", "P_"), ("sub_grade", "SG_"),
                     ("home_ownership", "H_")]:
    sub = df[["id", col]].dropna()
    sub = sub.rename(columns={col: "dst"})
    sub["dst"] = prefix + sub["dst"].astype(str)
    edges.append(sub.rename(columns={"id": "src"}))

edge_df = pd.concat(edges, ignore_index=True)
all_nodes = pd.concat([edge_df["src"], edge_df["dst"]]).unique()
node_map = {n: i for i, n in enumerate(all_nodes)}
edge_df["src_id"] = edge_df["src"].map(node_map).astype(np.int32)
edge_df["dst_id"] = edge_df["dst"].map(node_map).astype(np.int32)
n_nodes = len(all_nodes)
n_edges = len(edge_df)
print(f"  Graph: {n_nodes:,} nodes, {n_edges:,} edges")

K_SAMPLE = min(int(os.getenv("LC_RAPIDS_GRAPH_K_SAMPLE", "200")), n_nodes)
rows = []

# ═══════════════════════════════════════════
# 1. NetworkX CPU
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("NetworkX CPU")
print("=" * 60)

import networkx as nx

# Build
s = time.perf_counter()
G_nx = nx.Graph()
G_nx.add_edges_from(zip(edge_df["src_id"].values, edge_df["dst_id"].values, strict=False))
build_t = time.perf_counter() - s
rows.append({"task": "graph_build", "backend": "networkx_cpu",
             "seconds": build_t, "metric": "nodes", "metric_value": G_nx.number_of_nodes()})
print(f"  build: {build_t:.4f}s ({G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges)")

# Connected components
s = time.perf_counter()
cc = list(nx.connected_components(G_nx))
cc_t = time.perf_counter() - s
rows.append({"task": "connected_components", "backend": "networkx_cpu",
             "seconds": cc_t, "metric": "n_components", "metric_value": len(cc)})
print(f"  connected_components: {cc_t:.4f}s ({len(cc)} components)")

# PageRank
s = time.perf_counter()
pr = nx.pagerank(G_nx, alpha=0.85, max_iter=100, tol=1e-5)
pr_t = time.perf_counter() - s
pr_sum = sum(pr.values())
rows.append({"task": "pagerank", "backend": "networkx_cpu",
             "seconds": pr_t, "metric": "sum_pagerank", "metric_value": pr_sum})
print(f"  pagerank: {pr_t:.4f}s (sum={pr_sum:.4f})")

# Louvain
s = time.perf_counter()
communities = nx.community.louvain_communities(G_nx, seed=42)
louv_t = time.perf_counter() - s
rows.append({"task": "louvain", "backend": "networkx_cpu",
             "seconds": louv_t, "metric": "n_communities", "metric_value": len(communities)})
print(f"  louvain: {louv_t:.4f}s ({len(communities)} communities)")

# Betweenness centrality (sampled)
s = time.perf_counter()
bc = nx.betweenness_centrality(G_nx, k=K_SAMPLE, seed=42)
bc_t = time.perf_counter() - s
bc_max = max(bc.values())
rows.append({"task": "betweenness_centrality", "backend": "networkx_cpu",
             "seconds": bc_t, "metric": "max_bc", "metric_value": bc_max})
print(f"  betweenness_centrality (k={K_SAMPLE}): {bc_t:.4f}s (max={bc_max:.6f})")

# ═══════════════════════════════════════════
# 2. nx-cugraph (zero-code-change GPU backend)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("nx-cugraph (zero-code-change GPU backend)")
print("=" * 60)

try:
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
    # Force reload to pick up the backend
    import importlib
    importlib.reload(nx)

    # Build (same graph, just measure dispatch overhead)
    s = time.perf_counter()
    G_nx2 = nx.Graph()
    G_nx2.add_edges_from(zip(edge_df["src_id"].values, edge_df["dst_id"].values, strict=False))
    build_t2 = time.perf_counter() - s
    rows.append({"task": "graph_build", "backend": "nx_cugraph_gpu",
                 "seconds": build_t2, "metric": "nodes", "metric_value": G_nx2.number_of_nodes()})
    print(f"  build: {build_t2:.4f}s ({G_nx2.number_of_nodes():,} nodes)")

    # PageRank via nx-cugraph
    s = time.perf_counter()
    pr2 = nx.pagerank(G_nx2, alpha=0.85, max_iter=100, tol=1e-5, backend="cugraph")
    pr_t2 = time.perf_counter() - s
    pr_sum2 = sum(pr2.values())
    rows.append({"task": "pagerank", "backend": "nx_cugraph_gpu",
                 "seconds": pr_t2, "metric": "sum_pagerank", "metric_value": pr_sum2})
    print(f"  pagerank: {pr_t2:.4f}s (sum={pr_sum2:.4f})")

    # Louvain via nx-cugraph
    s = time.perf_counter()
    communities2 = nx.community.louvain_communities(G_nx2, seed=42, backend="cugraph")
    louv_t2 = time.perf_counter() - s
    rows.append({"task": "louvain", "backend": "nx_cugraph_gpu",
                 "seconds": louv_t2, "metric": "n_communities", "metric_value": len(communities2)})
    print(f"  louvain: {louv_t2:.4f}s ({len(communities2)} communities)")

    # Betweenness via nx-cugraph
    s = time.perf_counter()
    bc2 = nx.betweenness_centrality(G_nx2, k=K_SAMPLE, seed=42, backend="cugraph")
    bc_t2 = time.perf_counter() - s
    bc_max2 = max(bc2.values())
    rows.append({"task": "betweenness_centrality", "backend": "nx_cugraph_gpu",
                 "seconds": bc_t2, "metric": "max_bc", "metric_value": bc_max2})
    print(f"  betweenness_centrality (k={K_SAMPLE}): {bc_t2:.4f}s (max={bc_max2:.6f})")

    # Connected components via nx-cugraph
    s = time.perf_counter()
    cc2 = list(nx.connected_components(G_nx2, backend="cugraph"))
    cc_t2 = time.perf_counter() - s
    rows.append({"task": "connected_components", "backend": "nx_cugraph_gpu",
                 "seconds": cc_t2, "metric": "n_components", "metric_value": len(cc2)})
    print(f"  connected_components: {cc_t2:.4f}s ({len(cc2)} components)")

except Exception as e:
    traceback.print_exc()
    print(f"  nx-cugraph FAILED: {e}")


# ═══════════════════════════════════════════
# 3. cuGraph direct API
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("cuGraph (direct GPU API)")
print("=" * 60)

try:
    import cudf as cudf_lib
    import cugraph
    import cupy as cp

    edge_cudf = cudf_lib.DataFrame({
        "src": edge_df["src_id"].values,
        "dst": edge_df["dst_id"].values,
    })

    # Build
    s = time.perf_counter()
    G_cu = cugraph.Graph()
    G_cu.from_cudf_edgelist(edge_cudf, source="src", destination="dst", renumber=True)
    cp.cuda.Stream.null.synchronize()
    build_t3 = time.perf_counter() - s
    rows.append({"task": "graph_build", "backend": "cugraph_gpu",
                 "seconds": build_t3, "metric": "nodes",
                 "metric_value": float(G_cu.number_of_vertices())})
    print(f"  build: {build_t3:.4f}s ({G_cu.number_of_vertices():,} nodes)")

    # Connected components
    s = time.perf_counter()
    cc_df = cugraph.connected_components(G_cu, connection="weak")
    cp.cuda.Stream.null.synchronize()
    cc_t3 = time.perf_counter() - s
    n_comp = cc_df["labels"].nunique()
    rows.append({"task": "connected_components", "backend": "cugraph_gpu",
                 "seconds": cc_t3, "metric": "n_components", "metric_value": float(n_comp)})
    print(f"  connected_components: {cc_t3:.4f}s ({n_comp} components)")

    # PageRank
    s = time.perf_counter()
    pr_df = cugraph.pagerank(G_cu, alpha=0.85, max_iter=100, tol=1e-5)
    cp.cuda.Stream.null.synchronize()
    pr_t3 = time.perf_counter() - s
    pr_sum3 = float(pr_df["pagerank"].sum())
    rows.append({"task": "pagerank", "backend": "cugraph_gpu",
                 "seconds": pr_t3, "metric": "sum_pagerank", "metric_value": pr_sum3})
    print(f"  pagerank: {pr_t3:.4f}s (sum={pr_sum3:.4f})")

    # Louvain
    s = time.perf_counter()
    louv_df, modularity = cugraph.louvain(G_cu)
    cp.cuda.Stream.null.synchronize()
    louv_t3 = time.perf_counter() - s
    n_comm = louv_df["partition"].nunique()
    rows.append({"task": "louvain", "backend": "cugraph_gpu",
                 "seconds": louv_t3, "metric": "n_communities", "metric_value": float(n_comm)})
    print(f"  louvain: {louv_t3:.4f}s ({n_comm} communities, modularity={modularity:.4f})")

    # Betweenness centrality
    s = time.perf_counter()
    bc_df = cugraph.betweenness_centrality(G_cu, k=K_SAMPLE, seed=42)
    cp.cuda.Stream.null.synchronize()
    bc_t3 = time.perf_counter() - s
    bc_max3 = float(bc_df["betweenness_centrality"].max())
    rows.append({"task": "betweenness_centrality", "backend": "cugraph_gpu",
                 "seconds": bc_t3, "metric": "max_bc", "metric_value": bc_max3})
    print(f"  betweenness_centrality (k={K_SAMPLE}): {bc_t3:.4f}s (max={bc_max3:.6f})")

except Exception as e:
    traceback.print_exc()
    print(f"  cuGraph FAILED: {e}")


# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
df_out = pd.DataFrame(rows)
# Compute speedups vs networkx_cpu
cpu_times = df_out[df_out["backend"] == "networkx_cpu"].set_index("task")["seconds"]
for backend in ["nx_cugraph_gpu", "cugraph_gpu"]:
    gpu_times = df_out[df_out["backend"] == backend].set_index("task")["seconds"]
    for task in gpu_times.index:
        if task in cpu_times.index and gpu_times[task] > 0:
            mask = (df_out["backend"] == backend) & (df_out["task"] == task)
            df_out.loc[mask, "speedup_vs_cpu"] = cpu_times[task] / gpu_times[task]

df_out.to_csv(os.path.join(OUT, "cugraph_benchmark.csv"), index=False)
print(f"\nSaved cugraph_benchmark.csv ({len(df_out)} rows)")
print(df_out[["task", "backend", "seconds", "speedup_vs_cpu"]].to_string(index=False))
