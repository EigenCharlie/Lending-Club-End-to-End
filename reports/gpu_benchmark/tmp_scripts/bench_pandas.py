import json
import sys
import time

import numpy as np
import pandas as pd

path = sys.argv[1]
repeats = int(sys.argv[2])
warmup = int(sys.argv[3])
cols = [
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


def run_once():
    df = pd.read_parquet(path, columns=cols)
    df["int_rate"] = pd.to_numeric(df["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
    df["term_m"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df["issue_year"] = df["issue_d"].dt.year.astype("Int64")

    f = df[(df["loan_amnt"] >= 5000) & (df["annual_inc"] > 20000) & (df["term_m"].isin([36, 60]))]
    a = (
        f.groupby(["grade", "issue_year"], dropna=False)
        .agg(loans=("id", "count"), funded=("loan_amnt", "sum"), dr=("default_flag", "mean"))
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
    z = z.sort_values(["issue_year", "grade"]).head(5000)
    return z


t = []
out = None
for i in range(repeats + warmup):
    s = time.perf_counter()
    out = run_once()
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
            "rows_out": int(len(out)),
            "checksum": float(out["checksum_col"].sum()),
        }
    )
)
