"""Optimization + CuPy benchmark.

Optimization: SciPy HiGHS vs cuOpt for LP/MILP portfolio problems.
CuPy: NumPy/SciPy CPU vs CuPy GPU for Monte Carlo, SVD, Sparse MatMul.
"""
import gc
import os
import time
import traceback

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN = os.path.join(PROJECT, "data", "processed", "train.parquet")
TRAIN_FE = os.path.join(PROJECT, "data", "processed", "train_fe.parquet")
OUT = os.path.join(PROJECT, "reports", "gpu_benchmark")


# ═══════════════════════════════════════════
# OPTIMIZATION BENCHMARKS
# ═══════════════════════════════════════════
print("=" * 60)
print("OPTIMIZATION: SciPy HiGHS vs cuOpt GPU")
print("=" * 60)

# Load real loan data
df = pd.read_parquet(TRAIN, columns=["loan_amnt", "int_rate", "default_flag"])
df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
df = df.dropna()
print(f"  Loaded {len(df):,} loans for optimization")

opt_rows = []
sizes = [int(x) for x in os.getenv("LC_RAPIDS_OPT_LP_SIZES", "3000,6000,12000,18000").split(",") if x]

for n_vars in sizes:
    print(f"\n  --- LP with {n_vars} variables ---")
    sub = df.head(n_vars)
    expected_return = (sub["int_rate"].values / 100.0).astype(np.float64)
    pd_default = sub["default_flag"].values.astype(np.float64)
    loan_amounts = sub["loan_amnt"].values.astype(np.float64)
    budget = loan_amounts.sum() * 0.3
    risk_budget = n_vars * 0.15
    max_alloc = 0.05

    # SciPy HiGHS
    print(f"    SciPy HiGHS ...")
    try:
        from scipy.optimize import linprog
        c = -expected_return
        A_ub = np.vstack([loan_amounts.reshape(1, -1), pd_default.reshape(1, -1)])
        b_ub = np.array([budget, risk_budget])
        bounds = [(0.0, max_alloc)] * n_vars
        s = time.perf_counter()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        t = time.perf_counter() - s
        obj = -res.fun if res.success else None
        opt_rows.append({"task": "portfolio_lp", "backend": "scipy_highs_cpu",
                         "seconds": t, "status": 0, "objective": obj, "n_variables": n_vars})
        print(f"      {t:.4f}s, obj={obj:.2f}")
    except Exception as e:
        opt_rows.append({"task": "portfolio_lp", "backend": "scipy_highs_cpu",
                         "seconds": None, "status": f"error: {e}", "objective": None, "n_variables": n_vars})

    # cuOpt GPU
    print(f"    cuOpt GPU ...")
    try:
        from cuopt.linear_programming import DataModel, Solve
        dm = DataModel()
        dm.set_objective_coefficients(expected_return)
        dm.set_maximize(True)
        A_values = np.concatenate([loan_amounts, pd_default])
        A_indices = np.concatenate([np.arange(n_vars, dtype=np.int32), np.arange(n_vars, dtype=np.int32)])
        A_offsets = np.array([0, n_vars, 2 * n_vars], dtype=np.int32)
        dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
        dm.set_constraint_bounds(np.array([budget, risk_budget]))
        dm.set_row_types(np.array(["L", "L"]))
        dm.set_variable_lower_bounds(np.zeros(n_vars))
        dm.set_variable_upper_bounds(np.full(n_vars, max_alloc))

        s = time.perf_counter()
        solution = Solve(dm)
        t = time.perf_counter() - s
        obj = float(solution.get_primal_objective())
        status = solution.get_termination_reason()
        opt_rows.append({"task": "portfolio_lp", "backend": "cuopt_gpu",
                         "seconds": t, "status": status, "objective": obj, "n_variables": n_vars})
        print(f"      {t:.4f}s, obj={obj:.2f}, status={status}")
    except Exception as e:
        traceback.print_exc()
        opt_rows.append({"task": "portfolio_lp", "backend": "cuopt_gpu",
                         "seconds": None, "status": f"error: {e}", "objective": None, "n_variables": n_vars})

# MILP
default_milp = int(os.getenv("LC_RAPIDS_OPT_MILP_SIZE", "3000"))
print(f"\n  --- MILP with {default_milp} variables ---")
n_milp = default_milp
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
    integrality = np.ones(n_milp)
    bounds_milp = Bounds(lb=0, ub=1)
    s = time.perf_counter()
    res = milp(c_milp, constraints=constraints, integrality=integrality, bounds=bounds_milp)
    t = time.perf_counter() - s
    obj = -res.fun if res.success else None
    opt_rows.append({"task": "portfolio_milp", "backend": "scipy_milp_cpu",
                     "seconds": t, "status": 0 if res.success else -1, "objective": obj, "n_variables": n_milp})
    print(f"      {t:.4f}s, obj={obj:.2f}")
except Exception as e:
    opt_rows.append({"task": "portfolio_milp", "backend": "scipy_milp_cpu",
                     "seconds": None, "status": f"error: {e}", "objective": None, "n_variables": n_milp})

# cuOpt MILP
print(f"    cuOpt MILP ...")
try:
    from cuopt.linear_programming import DataModel, Solve
    dm = DataModel()
    dm.set_objective_coefficients(-expected_return)  # minimize negative
    A_values = np.concatenate([loan_amounts, pd_default])
    A_indices = np.concatenate([np.arange(n_milp, dtype=np.int32), np.arange(n_milp, dtype=np.int32)])
    A_offsets = np.array([0, n_milp, 2 * n_milp], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
    dm.set_constraint_bounds(np.array([budget, risk_budget]))
    dm.set_row_types(np.array(["L", "L"]))
    dm.set_variable_lower_bounds(np.zeros(n_milp))
    dm.set_variable_upper_bounds(np.ones(n_milp))
    dm.set_variable_types(np.array(["I"] * n_milp))

    s = time.perf_counter()
    solution = Solve(dm)
    t = time.perf_counter() - s
    obj = -float(solution.get_primal_objective())
    opt_rows.append({"task": "portfolio_milp", "backend": "cuopt_milp_gpu",
                     "seconds": t, "status": "optimal", "objective": obj, "n_variables": n_milp})
    print(f"      {t:.4f}s, obj={obj:.2f}")
except Exception as e:
    traceback.print_exc()
    opt_rows.append({"task": "portfolio_milp", "backend": "cuopt_milp_gpu",
                     "seconds": None, "status": f"error: {e}", "objective": None, "n_variables": n_milp})

# Speedups
df_opt = pd.DataFrame(opt_rows)
for nv in sizes:
    cpu_row = df_opt[(df_opt["backend"] == "scipy_highs_cpu") & (df_opt["n_variables"] == nv)]
    gpu_row = df_opt[(df_opt["backend"] == "cuopt_gpu") & (df_opt["n_variables"] == nv)]
    if not cpu_row.empty and not gpu_row.empty:
        ct = cpu_row["seconds"].values[0]
        gt = gpu_row["seconds"].values[0]
        if ct and gt and gt > 0:
            df_opt.loc[gpu_row.index, "speedup_vs_cpu_lp"] = ct / gt

df_opt.to_csv(os.path.join(OUT, "cuopt_benchmark.csv"), index=False)
print(f"\nSaved cuopt_benchmark.csv ({len(df_opt)} rows)")


# ═══════════════════════════════════════════
# CUPY BENCHMARKS
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("CUPY: NumPy/SciPy CPU vs CuPy GPU")
print("=" * 60)

cupy_rows = []

# --- Monte Carlo ECL ---
print("\n  --- Monte Carlo ECL (100K scenarios, 10K loans) ---")
n_loans = 10_000
n_scenarios = int(os.getenv("LC_RAPIDS_CUPY_N_SCENARIOS", "100000"))
rng = np.random.default_rng(42)
pd_vals = rng.uniform(0.01, 0.40, n_loans).astype(np.float64)
lgd_vals = rng.uniform(0.20, 0.80, n_loans).astype(np.float64)
ead_vals = rng.uniform(5000, 50000, n_loans).astype(np.float64)

# NumPy
print("    NumPy CPU ...")
s = time.perf_counter()
defaults = rng.random((n_scenarios, n_loans)) < pd_vals
losses = defaults * lgd_vals * ead_vals
ecl_dist = losses.sum(axis=1)
ecl_mean = ecl_dist.mean()
ecl_var95 = np.percentile(ecl_dist, 95)
numpy_t = time.perf_counter() - s
cupy_rows.append({"task": "monte_carlo_ecl", "backend": "numpy_cpu",
                  "seconds": numpy_t, "metric": "ecl_mean", "metric_value": float(ecl_mean)})
print(f"      {numpy_t:.4f}s, ECL_mean={ecl_mean:,.0f}, VaR95={ecl_var95:,.0f}")

# CuPy
print("    CuPy GPU ...")
import cupy as cp
pd_gpu = cp.asarray(pd_vals)
lgd_gpu = cp.asarray(lgd_vals)
ead_gpu = cp.asarray(ead_vals)
s = time.perf_counter()
rng_gpu = cp.random.default_rng(42)
defaults = rng_gpu.random((n_scenarios, n_loans)) < pd_gpu
losses = defaults * lgd_gpu * ead_gpu
ecl_dist = losses.sum(axis=1)
ecl_mean_gpu = float(ecl_dist.mean())
ecl_var95_gpu = float(cp.percentile(ecl_dist, 95))
cp.cuda.Stream.null.synchronize()
cupy_t = time.perf_counter() - s
cupy_rows.append({"task": "monte_carlo_ecl", "backend": "cupy_gpu",
                  "seconds": cupy_t, "metric": "ecl_mean", "metric_value": ecl_mean_gpu})
print(f"      {cupy_t:.4f}s, ECL_mean={ecl_mean_gpu:,.0f}, VaR95={ecl_var95_gpu:,.0f}")

# --- SVD ---
print("\n  --- SVD (100K x 47 features matrix) ---")
df_fe = pd.read_parquet(TRAIN_FE)
num_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns if c != "default_flag"]
X = df_fe[num_cols].dropna().head(int(os.getenv("LC_RAPIDS_CUPY_SVD_ROWS", "100000"))).values.astype(np.float64)
print(f"    Matrix: {X.shape}")

# NumPy
print("    NumPy CPU ...")
s = time.perf_counter()
U, S, Vt = np.linalg.svd(X, full_matrices=False)
numpy_t = time.perf_counter() - s
cupy_rows.append({"task": "svd", "backend": "numpy_cpu",
                  "seconds": numpy_t, "metric": "top_sv", "metric_value": float(S[0])})
print(f"      {numpy_t:.4f}s, top_sv={S[0]:.2f}")

# CuPy
print("    CuPy GPU ...")
X_gpu = cp.asarray(X)
s = time.perf_counter()
U_g, S_g, Vt_g = cp.linalg.svd(X_gpu, full_matrices=False)
cp.cuda.Stream.null.synchronize()
cupy_t = time.perf_counter() - s
cupy_rows.append({"task": "svd", "backend": "cupy_gpu",
                  "seconds": cupy_t, "metric": "top_sv", "metric_value": float(S_g[0])})
print(f"      {cupy_t:.4f}s, top_sv={float(S_g[0]):.2f}")

# --- Sparse MatMul ---
print("\n  --- Sparse Matrix Multiply (50K x 50K, density=0.01) ---")
from scipy import sparse as sp
n_sp = int(os.getenv("LC_RAPIDS_CUPY_SPARSE_N", "50000"))
density = 0.01

# SciPy
print("    SciPy CPU ...")
A = sp.random(n_sp, n_sp, density=density, format="csr", dtype=np.float64, random_state=42)
B = sp.random(n_sp, n_sp, density=density, format="csr", dtype=np.float64, random_state=43)
s = time.perf_counter()
C = A @ B
scipy_t = time.perf_counter() - s
cupy_rows.append({"task": "sparse_matmul", "backend": "scipy_cpu",
                  "seconds": scipy_t, "metric": "nnz", "metric_value": float(C.nnz)})
print(f"      {scipy_t:.4f}s, nnz={C.nnz:,}")

# CuPy
print("    CuPy GPU ...")
import cupyx.scipy.sparse as cusp
A_gpu = cusp.csr_matrix(A)
B_gpu = cusp.csr_matrix(B)
s = time.perf_counter()
C_gpu = A_gpu @ B_gpu
cp.cuda.Stream.null.synchronize()
cupy_t = time.perf_counter() - s
cupy_rows.append({"task": "sparse_matmul", "backend": "cupy_gpu",
                  "seconds": cupy_t, "metric": "nnz", "metric_value": float(C_gpu.nnz)})
print(f"      {cupy_t:.4f}s, nnz={C_gpu.nnz:,}")

# Speedups
df_cupy = pd.DataFrame(cupy_rows)
for task in ["monte_carlo_ecl", "svd", "sparse_matmul"]:
    cpu_b = "numpy_cpu" if task != "sparse_matmul" else "scipy_cpu"
    cpu_row = df_cupy[(df_cupy["task"] == task) & (df_cupy["backend"] == cpu_b)]
    gpu_row = df_cupy[(df_cupy["task"] == task) & (df_cupy["backend"] == "cupy_gpu")]
    if not cpu_row.empty and not gpu_row.empty:
        ct = cpu_row["seconds"].values[0]
        gt = gpu_row["seconds"].values[0]
        if ct and gt and gt > 0:
            df_cupy.loc[gpu_row.index, "speedup_vs_cpu"] = ct / gt

df_cupy.to_csv(os.path.join(OUT, "cupy_benchmark.csv"), index=False)
print(f"\nSaved cupy_benchmark.csv ({len(df_cupy)} rows)")

# --- Metadata ---
import json
meta = {
    "hardware": {"gpu": "NVIDIA GeForce RTX 3080", "vram_mb": 10240,
                 "cpu": "AMD Ryzen 5 5600X", "ram_gb": 24, "platform": "WSL2"},
    "versions": {}, "dataset": {"n_rows": 1_860_764, "parquet": "lending_club_cleaned.parquet"},
}
for lib in ["cudf", "cuml", "cugraph", "cuopt", "cupy", "polars", "duckdb",
            "pandas", "numpy", "scipy", "sklearn"]:
    try:
        m = __import__(lib)
        meta["versions"][lib] = m.__version__
    except Exception:
        meta["versions"][lib] = "not installed"
with open(os.path.join(OUT, "gpu_bench_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved gpu_bench_meta.json")

print("\nALL DONE!")
