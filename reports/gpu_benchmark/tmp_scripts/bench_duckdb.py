"""DuckDB benchmark: same analytical workload as bench_pandas.py and bench_polars.py."""

import json
import sys
import time

import duckdb
import numpy as np

path = sys.argv[1]
repeats = int(sys.argv[2])
warmup = int(sys.argv[3])

SQL = f"""
WITH raw AS (
    SELECT
        id,
        issue_d,
        loan_amnt,
        CAST(REGEXP_REPLACE(CAST(int_rate AS VARCHAR), '%', '') AS DOUBLE) AS int_rate,
        annual_inc,
        CAST(REGEXP_EXTRACT(CAST(term AS VARCHAR), '(\\d+)', 1) AS INT) AS term_m,
        grade,
        purpose,
        default_flag
    FROM read_parquet('{path}')
),
filtered AS (
    SELECT *,
           YEAR(CAST(issue_d AS DATE)) AS issue_year
    FROM raw
    WHERE loan_amnt >= 5000
      AND annual_inc > 20000
      AND term_m IN (36, 60)
),
agg_grade AS (
    SELECT
        grade,
        issue_year,
        COUNT(id) AS loans,
        SUM(loan_amnt) AS funded,
        AVG(default_flag) AS dr
    FROM filtered
    GROUP BY grade, issue_year
),
agg_purpose AS (
    SELECT
        purpose,
        COUNT(id) AS p_loans,
        AVG(default_flag) AS p_dr
    FROM filtered
    GROUP BY purpose
),
grade_purpose AS (
    SELECT DISTINCT ON (grade) grade, purpose
    FROM filtered
),
joined AS (
    SELECT
        a.*,
        gp.purpose,
        ap.p_loans,
        ap.p_dr,
        COALESCE(a.dr, 0.0) * COALESCE(a.funded, 0.0) AS checksum_col
    FROM agg_grade a
    LEFT JOIN grade_purpose gp ON a.grade = gp.grade
    LEFT JOIN agg_purpose ap ON gp.purpose = ap.purpose
)
SELECT *
FROM joined
ORDER BY issue_year, grade
LIMIT 5000
"""

t = []
out = None
for i in range(repeats + warmup):
    conn = duckdb.connect()
    s = time.perf_counter()
    out = conn.execute(SQL).fetchdf()
    dt = time.perf_counter() - s
    conn.close()
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
