import json
import sys
import time

import numpy as np
import polars as pl

path = sys.argv[1]
repeats = int(sys.argv[2])
warmup = int(sys.argv[3])
mode = sys.argv[4]


def run_once(mode):
    lf = pl.scan_parquet(path).select(
        [
            "id",
            "issue_d",
            "loan_amnt",
            "int_rate",
            "annual_inc",
            "term",
            "grade",
            "purpose",
            "default_flag",
        ]
    )
    lf = lf.with_columns(
        [
            pl.col("int_rate")
            .cast(pl.Utf8)
            .str.replace("%", "", literal=True)
            .cast(pl.Float64, strict=False),
            pl.col("term")
            .cast(pl.Utf8)
            .str.extract(r"(\d+)", group_index=1)
            .cast(pl.Int32, strict=False)
            .alias("term_m"),
            pl.col("issue_d").cast(pl.Date, strict=False),
        ]
    ).with_columns(pl.col("issue_d").dt.year().alias("issue_year"))

    f = lf.filter(
        (pl.col("loan_amnt") >= 5000)
        & (pl.col("annual_inc") > 20000)
        & (pl.col("term_m").is_in([36, 60]))
    )
    a = f.group_by(["grade", "issue_year"]).agg(
        [
            pl.len().alias("loans"),
            pl.sum("loan_amnt").alias("funded"),
            pl.mean("default_flag").alias("dr"),
        ]
    )
    b = f.group_by("purpose").agg(
        [
            pl.len().alias("p_loans"),
            pl.mean("default_flag").alias("p_dr"),
        ]
    )
    g = f.select(["grade", "purpose"]).unique(subset=["grade"], keep="first")

    z = (
        a.join(g, on="grade", how="left")
        .join(b, on="purpose", how="left")
        .with_columns(
            (pl.col("dr").fill_null(0.0) * pl.col("funded").fill_null(0.0)).alias("checksum_col")
        )
        .sort(["issue_year", "grade"])
        .limit(5000)
    )

    if mode == "gpu":
        if not hasattr(pl, "GPUEngine"):
            raise RuntimeError("Polars GPUEngine is not available")
        out = z.collect(engine=pl.GPUEngine(raise_on_fail=True))
    else:
        out = z.collect()
    return out


t = []
out = None
for i in range(repeats + warmup):
    s = time.perf_counter()
    out = run_once(mode)
    dt = time.perf_counter() - s
    if i >= warmup:
        t.append(dt)

arr = np.asarray(t, dtype=np.float64)
q1, q3 = np.percentile(arr, [25, 75])
print(
    json.dumps(
        {
            "median_seconds": float(np.median(arr)),
            "mean_seconds": float(np.mean(arr)),
            "std_seconds": float(np.std(arr)),
            "iqr_seconds": float(q3 - q1),
            "rows_out": int(out.height),
            "checksum": float(out["checksum_col"].sum()),
        }
    )
)
