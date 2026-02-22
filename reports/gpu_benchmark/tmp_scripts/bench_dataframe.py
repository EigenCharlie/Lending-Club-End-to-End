"""DataFrame benchmark: pandas vs cudf.pandas vs polars vs polars-gpu vs duckdb.

Uses the full lending_club_cleaned.parquet (1.86M rows, 110 cols).
Workload: read parquet → type cast → filter → multi-groupby → join → window → sort
"""
import gc
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PQ = os.path.join(PROJECT, "data", "interim", "lending_club_cleaned.parquet")
OUT = os.path.join(PROJECT, "reports", "gpu_benchmark")
REPEATS = 3
WARMUP = 1

COLS = ["id", "loan_amnt", "int_rate", "annual_inc", "term", "grade",
        "purpose", "default_flag", "issue_d", "sub_grade", "home_ownership"]


def median_time(func, repeats=REPEATS, warmup=WARMUP):
    results = []
    out = None
    for i in range(warmup + repeats):
        gc.collect()
        s = time.perf_counter()
        out = func()
        dt = time.perf_counter() - s
        if i >= warmup:
            results.append(dt)
    return float(np.median(results)), out


rows = []

# ═══════════════════════════════════════════
# 1. PANDAS CPU
# ═══════════════════════════════════════════
print("=" * 60)
print("1. Pandas CPU")

def pandas_workload():
    df = pd.read_parquet(PQ, columns=COLS)
    n_orig = len(df)
    df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
    df["term_months"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_d"].dt.year
    mask = (
        (df["loan_amnt"] >= 5000) & (df["annual_inc"] > 20000)
        & (df["grade"].isin(["A", "B", "C", "D", "E"]))
    )
    f = df[mask]
    n_filt = len(f)
    g1 = (f.groupby(["grade", "issue_year"], dropna=False)
           .agg(n_loans=("id", "count"), total_funded=("loan_amnt", "sum"),
                avg_rate=("int_rate", "mean"), default_rate=("default_flag", "mean"))
           .reset_index())
    g2 = (f.groupby("purpose", dropna=False)
           .agg(purpose_loans=("id", "count"), purpose_dr=("default_flag", "mean"))
           .reset_index())
    # Top purpose per grade (no lambda — use idxmax approach)
    gpc = f.groupby(["grade", "purpose"]).size().reset_index(name="cnt")
    idx = gpc.groupby("grade")["cnt"].idxmax()
    gp = gpc.loc[idx, ["grade", "purpose"]].rename(columns={"purpose": "top_purpose"})
    z = (g1.merge(gp, on="grade", how="left")
           .merge(g2, left_on="top_purpose", right_on="purpose", how="left"))
    z["rank_in_year"] = z.groupby("issue_year")["default_rate"].rank(ascending=False, method="dense")
    z = z.sort_values(["issue_year", "grade"])
    ck = float(z["default_rate"].fillna(0).sum() + z["total_funded"].fillna(0).sum())
    return {"rows": len(z), "checksum": ck, "n_filtered": n_filt, "n_original": n_orig}

try:
    t, out = median_time(pandas_workload)
    rows.append({"mode": "pandas_cpu", "median_seconds": t, "status": "ok", **out})
    print(f"  -> {t:.4f}s | {out['n_original']:,} -> {out['n_filtered']:,} -> {out['rows']} agg rows")
except Exception as e:
    traceback.print_exc()
    rows.append({"mode": "pandas_cpu", "status": f"error: {e}"})


# ═══════════════════════════════════════════
# 2. CUDF.PANDAS (zero-code-change GPU)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("2. cuDF.pandas (zero-code-change GPU)")

try:
    import cudf.pandas
    cudf.pandas.install()
    import importlib
    _pd = importlib.import_module("pandas")

    def cudf_pandas_workload():
        df = _pd.read_parquet(PQ, columns=COLS)
        n_orig = len(df)
        df["int_rate"] = _pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
        df["term_months"] = _pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        df["issue_d"] = _pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["issue_year"] = df["issue_d"].dt.year
        mask = (
            (df["loan_amnt"] >= 5000) & (df["annual_inc"] > 20000)
            & (df["grade"].isin(["A", "B", "C", "D", "E"]))
        )
        f = df[mask]
        n_filt = len(f)
        g1 = (f.groupby(["grade", "issue_year"], dropna=False)
               .agg(n_loans=("id", "count"), total_funded=("loan_amnt", "sum"),
                    avg_rate=("int_rate", "mean"), default_rate=("default_flag", "mean"))
               .reset_index())
        g2 = (f.groupby("purpose", dropna=False)
               .agg(purpose_loans=("id", "count"), purpose_dr=("default_flag", "mean"))
               .reset_index())
        gpc = f.groupby(["grade", "purpose"]).size().reset_index(name="cnt")
        idx = gpc.groupby("grade")["cnt"].idxmax()
        gp = gpc.loc[idx, ["grade", "purpose"]].rename(columns={"purpose": "top_purpose"})
        z = (g1.merge(gp, on="grade", how="left")
               .merge(g2, left_on="top_purpose", right_on="purpose", how="left"))
        z["rank_in_year"] = z.groupby("issue_year")["default_rate"].rank(ascending=False, method="dense")
        z = z.sort_values(["issue_year", "grade"])
        ck = float(z["default_rate"].fillna(0).sum() + z["total_funded"].fillna(0).sum())
        return {"rows": len(z), "checksum": ck, "n_filtered": n_filt, "n_original": n_orig}

    t, out = median_time(cudf_pandas_workload)
    rows.append({"mode": "cudf_pandas_gpu", "median_seconds": t, "status": "ok", **out})
    print(f"  -> {t:.4f}s | {out['n_original']:,} -> {out['n_filtered']:,} -> {out['rows']} agg rows")
except Exception as e:
    traceback.print_exc()
    rows.append({"mode": "cudf_pandas_gpu", "status": f"error: {e}"})


# ═══════════════════════════════════════════
# 3. POLARS CPU
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Polars CPU (lazy)")

try:
    import polars as pl

    def polars_cpu_workload():
        lf = pl.scan_parquet(PQ).select(COLS)
        lf = lf.with_columns([
            pl.col("int_rate").cast(pl.Utf8).str.replace("%", "", literal=True)
              .cast(pl.Float64, strict=False),
            pl.col("term").cast(pl.Utf8).str.extract(r"(\d+)", group_index=1)
              .cast(pl.Int32, strict=False).alias("term_months"),
            pl.col("issue_d").str.to_date("%b-%Y", strict=False),
        ]).with_columns(pl.col("issue_d").dt.year().alias("issue_year"))
        f = lf.filter(
            (pl.col("loan_amnt") >= 5000) & (pl.col("annual_inc") > 20000)
            & (pl.col("grade").is_in(["A", "B", "C", "D", "E"]))
        )
        g1 = f.group_by(["grade", "issue_year"]).agg([
            pl.len().alias("n_loans"),
            pl.sum("loan_amnt").alias("total_funded"),
            pl.mean("int_rate").alias("avg_rate"),
            pl.mean("default_flag").alias("default_rate"),
        ])
        g2 = f.group_by("purpose").agg([
            pl.len().alias("purpose_loans"),
            pl.mean("default_flag").alias("purpose_dr"),
        ])
        gp = (f.group_by(["grade", "purpose"]).agg(pl.len().alias("cnt"))
                .sort(["grade", "cnt"], descending=[False, True])
                .group_by("grade").first()
                .rename({"purpose": "top_purpose"})
                .select(["grade", "top_purpose"]))
        z = (g1.join(gp, on="grade", how="left")
               .join(g2, left_on="top_purpose", right_on="purpose", how="left"))
        z = z.with_columns(
            pl.col("default_rate").rank(descending=True, method="dense")
            .over("issue_year").alias("rank_in_year")
        )
        z = z.sort(["issue_year", "grade"])
        result = z.collect()
        ck = float(result["default_rate"].fill_null(0).sum()
                   + result["total_funded"].fill_null(0).sum())
        return {"rows": result.height, "checksum": ck}

    t, out = median_time(polars_cpu_workload)
    rows.append({"mode": "polars_cpu", "median_seconds": t, "status": "ok", **out})
    print(f"  -> {t:.4f}s | {out['rows']} agg rows")
except Exception as e:
    traceback.print_exc()
    rows.append({"mode": "polars_cpu", "status": f"error: {e}"})


# ═══════════════════════════════════════════
# 4. POLARS GPU
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Polars GPU (cudf-polars engine)")

try:
    import polars as pl

    def polars_gpu_workload():
        # Avoid str.to_date (unsupported on GPU) — extract year via string ops
        lf = pl.scan_parquet(PQ).select(COLS)
        lf = lf.with_columns([
            pl.col("int_rate").cast(pl.Utf8).str.replace("%", "", literal=True)
              .cast(pl.Float64, strict=False),
            pl.col("term").cast(pl.Utf8).str.extract(r"(\d+)", group_index=1)
              .cast(pl.Int32, strict=False).alias("term_months"),
            # Extract year from "Mon-YYYY" without date parsing
            pl.col("issue_d").cast(pl.Utf8).str.extract(r"-(\d{4})", group_index=1)
              .cast(pl.Int32, strict=False).alias("issue_year"),
        ])
        f = lf.filter(
            (pl.col("loan_amnt") >= 5000) & (pl.col("annual_inc") > 20000)
            & (pl.col("grade").is_in(["A", "B", "C", "D", "E"]))
        )
        g1 = f.group_by(["grade", "issue_year"]).agg([
            pl.len().alias("n_loans"),
            pl.sum("loan_amnt").alias("total_funded"),
            pl.mean("int_rate").alias("avg_rate"),
            pl.mean("default_flag").alias("default_rate"),
        ])
        g2 = f.group_by("purpose").agg([
            pl.len().alias("purpose_loans"),
            pl.mean("default_flag").alias("purpose_dr"),
        ])
        gp = (f.group_by(["grade", "purpose"]).agg(pl.len().alias("cnt"))
                .sort(["grade", "cnt"], descending=[False, True])
                .group_by("grade").first()
                .rename({"purpose": "top_purpose"})
                .select(["grade", "top_purpose"]))
        z = (g1.join(gp, on="grade", how="left")
               .join(g2, left_on="top_purpose", right_on="purpose", how="left"))
        # rank() not supported on GPU — skip window func, add post-collect
        z = z.sort(["issue_year", "grade"])
        result = z.collect(engine=pl.GPUEngine(raise_on_fail=True))
        # Add rank on CPU (trivial cost on small agg result)
        result = result.with_columns(
            pl.col("default_rate").rank(descending=True, method="dense")
            .over("issue_year").alias("rank_in_year")
        )
        ck = float(result["default_rate"].fill_null(0).sum()
                   + result["total_funded"].fill_null(0).sum())
        return {"rows": result.height, "checksum": ck}

    t, out = median_time(polars_gpu_workload)
    rows.append({"mode": "polars_gpu", "median_seconds": t, "status": "ok", **out})
    print(f"  -> {t:.4f}s | {out['rows']} agg rows")
except Exception as e:
    traceback.print_exc()
    rows.append({"mode": "polars_gpu", "status": f"error: {e}"})


# ═══════════════════════════════════════════
# 5. DUCKDB
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("5. DuckDB (SQL analytics)")

try:
    import duckdb

    SQL = f"""
    WITH raw AS (
        SELECT id, loan_amnt, annual_inc, grade, purpose, default_flag, sub_grade,
               home_ownership,
               TRY_CAST(REGEXP_REPLACE(CAST(int_rate AS VARCHAR), '%', '') AS DOUBLE) AS int_rate,
               TRY_CAST(REGEXP_EXTRACT(CAST(term AS VARCHAR), '(\\d+)', 1) AS INT) AS term_months,
               YEAR(STRPTIME(issue_d, '%b-%Y')) AS issue_year
        FROM read_parquet('{PQ}')
    ),
    filtered AS (
        SELECT * FROM raw
        WHERE loan_amnt >= 5000 AND annual_inc > 20000 AND grade IN ('A','B','C','D','E')
    ),
    agg_grade AS (
        SELECT grade, issue_year,
               COUNT(*) AS n_loans, SUM(loan_amnt) AS total_funded,
               AVG(int_rate) AS avg_rate, AVG(default_flag) AS default_rate
        FROM filtered GROUP BY grade, issue_year
    ),
    agg_purpose AS (
        SELECT purpose, COUNT(*) AS purpose_loans, AVG(default_flag) AS purpose_dr
        FROM filtered GROUP BY purpose
    ),
    grade_purpose_cnt AS (
        SELECT grade, purpose, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY grade ORDER BY COUNT(*) DESC) AS rn
        FROM filtered GROUP BY grade, purpose
    ),
    top_purpose AS (
        SELECT grade, purpose AS top_purpose FROM grade_purpose_cnt WHERE rn = 1
    ),
    joined AS (
        SELECT a.*, tp.top_purpose, ap.purpose_loans, ap.purpose_dr,
               DENSE_RANK() OVER (PARTITION BY a.issue_year ORDER BY a.default_rate DESC) AS rank_in_year
        FROM agg_grade a
        LEFT JOIN top_purpose tp ON a.grade = tp.grade
        LEFT JOIN agg_purpose ap ON tp.top_purpose = ap.purpose
    )
    SELECT * FROM joined ORDER BY issue_year, grade
    """

    def duckdb_workload():
        conn = duckdb.connect()
        out = conn.execute(SQL).fetchdf()
        conn.close()
        ck = float(out["default_rate"].fillna(0).sum() + out["total_funded"].fillna(0).sum())
        return {"rows": len(out), "checksum": ck}

    t, out = median_time(duckdb_workload)
    rows.append({"mode": "duckdb", "median_seconds": t, "status": "ok", **out})
    print(f"  -> {t:.4f}s | {out['rows']} agg rows")
except Exception as e:
    traceback.print_exc()
    rows.append({"mode": "duckdb", "status": f"error: {e}"})


# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
df = pd.DataFrame(rows)
pandas_t = df.loc[df["mode"] == "pandas_cpu", "median_seconds"]
if not pandas_t.empty:
    base = pandas_t.values[0]
    df["speedup_vs_pandas_cpu"] = base / df["median_seconds"].where(df["median_seconds"] > 0)
df.to_csv(os.path.join(OUT, "cudf_polars_benchmark.csv"), index=False)
print(f"\nSaved cudf_polars_benchmark.csv ({len(df)} rows)")
print(df[["mode", "median_seconds", "status"]].to_string(index=False))
