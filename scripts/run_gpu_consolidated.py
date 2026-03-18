"""GPU consolidated summary: clean CPU vs GPU table from benchmark_summary_all_sections.

Reads existing benchmark data from reports/gpu_benchmark/ and produces:
    models/gpu_consolidated_summary.json   — KPI summary
    data/processed/gpu_consolidated_table.parquet  — clean per-task comparison table
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
GPU_BENCH_DIR = PROJECT_ROOT / "reports" / "gpu_benchmark"
SCHEMA_VERSION = "2026-03-17.1"


def _build_cudf_summary(df: pd.DataFrame) -> list[dict]:
    """cuDF ETL section: CPU frameworks vs GPU."""
    sub = df[df["section"] == "cudf"].dropna(subset=["median_seconds"])
    if sub.empty:
        return []
    rows = []
    cpu_ref = sub[sub["mode"].str.contains("pandas_cpu", na=False)]["median_seconds"]
    cpu_baseline = float(cpu_ref.mean()) if not cpu_ref.empty else 1.0
    for _, row in sub.iterrows():
        speedup = (
            cpu_baseline / float(row["median_seconds"])
            if float(row["median_seconds"]) > 0
            else None
        )
        rows.append(
            {
                "section": "cuDF ETL",
                "task": str(row.get("mode", row.get("task", ""))),
                "backend": str(row.get("mode", "")),
                "cpu_seconds": None,
                "gpu_seconds": float(row["median_seconds"])
                if "gpu" in str(row.get("mode", "")).lower()
                else None,
                "cpu_seconds_alt": float(row["median_seconds"])
                if "cpu" in str(row.get("mode", "")).lower()
                else None,
                "speedup_vs_pandas": float(row.get("speedup_vs_pandas_cpu", speedup or 1.0)),
            }
        )
    return rows


def _build_section_summary(df: pd.DataFrame, section: str, label: str) -> list[dict]:
    """Generic: pair cpu and gpu rows by task."""
    sub = df[df["section"] == section].copy()
    if sub.empty:
        return []
    rows = []
    for task, grp in sub.groupby("task", dropna=False):
        cpu_row = grp[grp["backend"].str.contains("cpu", case=False, na=False)]
        gpu_row = grp[~grp["backend"].str.contains("cpu", case=False, na=False)]
        cpu_s = (
            float(cpu_row["fit_seconds"].mean())
            if not cpu_row.empty
            and "fit_seconds" in cpu_row.columns
            and not cpu_row["fit_seconds"].isna().all()
            else None
        )
        gpu_s = (
            float(gpu_row["fit_seconds"].mean())
            if not gpu_row.empty
            and "fit_seconds" in gpu_row.columns
            and not gpu_row["fit_seconds"].isna().all()
            else None
        )
        if cpu_s is None:
            cpu_s = (
                float(cpu_row["seconds"].mean())
                if not cpu_row.empty
                and "seconds" in cpu_row.columns
                and not cpu_row["seconds"].isna().all()
                else None
            )
        if gpu_s is None:
            gpu_s = (
                float(gpu_row["seconds"].mean())
                if not gpu_row.empty
                and "seconds" in gpu_row.columns
                and not gpu_row["seconds"].isna().all()
                else None
            )
        speedup = (cpu_s / gpu_s) if (cpu_s and gpu_s and gpu_s > 0) else None
        rows.append(
            {
                "section": label,
                "task": str(task),
                "backend_cpu": str(cpu_row["backend"].iloc[0]) if not cpu_row.empty else "cpu",
                "backend_gpu": str(gpu_row["backend"].iloc[0]) if not gpu_row.empty else "gpu",
                "cpu_seconds": round(cpu_s, 4) if cpu_s is not None else None,
                "gpu_seconds": round(gpu_s, 4) if gpu_s is not None else None,
                "speedup_cpu_vs_gpu": round(speedup, 2) if speedup is not None else None,
            }
        )
    return rows


def main() -> None:
    logger.info("GPU consolidated summary — clean CPU vs GPU comparison table")

    bench_path = GPU_BENCH_DIR / "benchmark_summary_all_sections.parquet"
    meta_path = GPU_BENCH_DIR / "gpu_bench_meta.json"

    if not bench_path.exists():
        logger.error(f"Not found: {bench_path}")
        return

    df = pd.read_parquet(bench_path)
    logger.info(f"Benchmark rows: {len(df)} | sections: {df['section'].unique().tolist()}")

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hardware = meta.get("hardware", {})

    # Build clean rows per section
    all_rows: list[dict] = []

    # cuDF ETL
    cudf_rows = _build_cudf_summary(df)
    all_rows.extend(cudf_rows)

    # cuML models
    cuml_rows = _build_section_summary(df, "cuml", "cuML Models")
    all_rows.extend(cuml_rows)

    # cuGraph analytics
    cugraph_rows = _build_section_summary(df, "cugraph", "cuGraph Analytics")
    all_rows.extend(cugraph_rows)

    # cuOpt (OR)
    cuopt_path = GPU_BENCH_DIR / "cuopt_benchmark.parquet"
    if cuopt_path.exists():
        opt_df = pd.read_parquet(cuopt_path)
        for _, row in opt_df.iterrows():
            cpu_s = row.get("cpu_seconds") or row.get("highs_seconds")
            gpu_s = row.get("gpu_seconds") or row.get("cuopt_seconds")
            speedup = (
                (float(cpu_s) / float(gpu_s)) if cpu_s and gpu_s and float(gpu_s) > 0 else None
            )
            all_rows.append(
                {
                    "section": "cuOpt Portfolio",
                    "task": str(row.get("stage", row.get("scenario", "portfolio_opt"))),
                    "backend_cpu": "HiGHS (CPU)",
                    "backend_gpu": "cuOpt (GPU)",
                    "cpu_seconds": round(float(cpu_s), 4) if cpu_s else None,
                    "gpu_seconds": round(float(gpu_s), 4) if gpu_s else None,
                    "speedup_cpu_vs_gpu": round(speedup, 2) if speedup else None,
                }
            )

    # CuPy IFRS9 Monte Carlo
    cupy_path = GPU_BENCH_DIR / "cupy_benchmark.parquet"
    if cupy_path.exists():
        cp_df = pd.read_parquet(cupy_path)
        for _, row in cp_df.iterrows():
            cpu_s = row.get("numpy_seconds") or row.get("cpu_seconds")
            gpu_s = row.get("cupy_seconds") or row.get("gpu_seconds")
            speedup = (
                (float(cpu_s) / float(gpu_s)) if cpu_s and gpu_s and float(gpu_s) > 0 else None
            )
            all_rows.append(
                {
                    "section": "CuPy IFRS9 MC",
                    "task": str(row.get("scenario", row.get("task", "ifrs9_montecarlo"))),
                    "backend_cpu": "NumPy (CPU)",
                    "backend_gpu": "CuPy (GPU)",
                    "cpu_seconds": round(float(cpu_s), 4) if cpu_s else None,
                    "gpu_seconds": round(float(gpu_s), 4) if gpu_s else None,
                    "speedup_cpu_vs_gpu": round(speedup, 2) if speedup else None,
                }
            )

    if not all_rows:
        logger.warning("No rows produced — check benchmark files.")
        return

    table = pd.DataFrame(all_rows)
    out_tbl = DATA_PROC / "gpu_consolidated_table.parquet"
    table.to_parquet(out_tbl, index=False)
    logger.success(f"Saved {out_tbl} ({len(table)} rows)")

    # Summary KPIs
    with_speedup = table.dropna(subset=["speedup_cpu_vs_gpu"])
    max_row = (
        with_speedup.loc[with_speedup["speedup_cpu_vs_gpu"].idxmax()]
        if not with_speedup.empty
        else None
    )
    mean_speedup = (
        float(with_speedup["speedup_cpu_vs_gpu"].mean()) if not with_speedup.empty else 0.0
    )

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "hardware": hardware,
        "n_tasks": len(table),
        "n_sections": int(table["section"].nunique()),
        "sections": table["section"].unique().tolist(),
        "mean_speedup_vs_cpu": round(mean_speedup, 2),
        "max_speedup": round(float(max_row["speedup_cpu_vs_gpu"]), 2)
        if max_row is not None
        else None,
        "max_speedup_task": str(max_row["task"]) if max_row is not None else None,
        "max_speedup_section": str(max_row["section"]) if max_row is not None else None,
        "table_path": str(out_tbl),
        "source": str(bench_path),
    }

    out_status = MODELS_DIR / "gpu_consolidated_summary.json"
    out_status.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.success(f"Saved {out_status}")


if __name__ == "__main__":
    main()
