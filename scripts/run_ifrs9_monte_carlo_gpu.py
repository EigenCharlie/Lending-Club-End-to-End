"""Monte Carlo IFRS9 benchmark with NumPy CPU and CuPy GPU paths.

This is a RAPIDS/GPU extension and does not replace the canonical
`run_ifrs9_sensitivity.py` artifact path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ifrs9_sensitivity import (  # noqa: E402
    _load_intervals,
    _load_lifetime_table,
    _load_raw_splits,
    _load_temporal_context,
    _prepare_base_vectors,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover - only exercised in RAPIDS env
    cp = None


def _artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    raw = str(os.environ.get("GPU_REPLAY_ARTIFACT_ROOT", "")).strip()
    return (Path(raw) / path) if raw else path


def _build_lifetime_base(
    pd12: np.ndarray,
    grade: np.ndarray,
    lifetime_table: pd.DataFrame | None,
) -> np.ndarray:
    if lifetime_table is not None and "PD_60m" in lifetime_table.columns:
        table = lifetime_table.copy()
        if table.index.name is None and "Grade" in table.columns:
            table = table.set_index("Grade")
        mapped = pd.Series(grade).map(table["PD_60m"]).to_numpy(dtype=float)
        fallback = np.clip(1.0 - np.power(1.0 - pd12, 5.0), 0.0, 1.0)
        return np.where(np.isfinite(mapped), np.clip(mapped, 0.0, 1.0), fallback)
    return np.clip(1.0 - np.power(1.0 - pd12, 5.0), 0.0, 1.0)


def _scenario_multipliers(shocks: np.ndarray) -> dict[str, np.ndarray]:
    macro = shocks[:, 0]
    pd_noise = shocks[:, 1]
    lgd_noise = shocks[:, 2]
    ead_noise = shocks[:, 3]
    disc_noise = shocks[:, 4]
    return {
        "pd_mult": np.clip(np.exp(0.10 * macro + 0.20 * pd_noise), 0.70, 1.80),
        "lgd_mult": np.clip(np.exp(0.05 * macro + 0.10 * lgd_noise), 0.80, 1.50),
        "ead_mult": np.clip(np.exp(0.03 * macro + 0.06 * ead_noise), 0.90, 1.20),
        "discount_rate": np.clip(0.05 + 0.01 * macro + 0.01 * disc_noise, 0.03, 0.12),
    }


def _shock_correlation_matrix(profile: str) -> np.ndarray:
    profiles = {
        "independent": np.eye(5, dtype=np.float64),
        "moderate_credit": np.array(
            [
                [1.00, 0.55, 0.35, 0.20, 0.25],
                [0.55, 1.00, 0.40, 0.25, 0.20],
                [0.35, 0.40, 1.00, 0.30, 0.15],
                [0.20, 0.25, 0.30, 1.00, 0.10],
                [0.25, 0.20, 0.15, 0.10, 1.00],
            ],
            dtype=np.float64,
        ),
        "stress_credit": np.array(
            [
                [1.00, 0.75, 0.55, 0.35, 0.40],
                [0.75, 1.00, 0.60, 0.40, 0.35],
                [0.55, 0.60, 1.00, 0.45, 0.30],
                [0.35, 0.40, 0.45, 1.00, 0.20],
                [0.40, 0.35, 0.30, 0.20, 1.00],
            ],
            dtype=np.float64,
        ),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown correlation profile: {profile}")
    return profiles[profile]


def _generate_shocks(
    *,
    rng: np.random.Generator,
    n_scenarios: int,
    correlation_profile: str,
    antithetic: bool,
) -> np.ndarray:
    corr = _shock_correlation_matrix(correlation_profile)
    chol = np.linalg.cholesky(corr)
    if antithetic:
        half = int(np.ceil(n_scenarios / 2))
        base = rng.standard_normal((half, 5), dtype=np.float32).astype(np.float64)
        paired = np.vstack([base, -base])[:n_scenarios]
        return (paired @ chol.T).astype(np.float32)
    raw = rng.standard_normal((n_scenarios, 5), dtype=np.float32).astype(np.float64)
    return (raw @ chol.T).astype(np.float32)


def _tail_metrics(values: np.ndarray) -> dict[str, float]:
    p95 = float(np.quantile(values, 0.95))
    tail = values[values >= p95]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": p95,
        "p99": float(np.quantile(values, 0.99)),
        "expected_shortfall_95": float(np.mean(tail)) if len(tail) else p95,
    }


def _run_numpy_chunk(
    *,
    base: dict[str, np.ndarray],
    lifetime_base: np.ndarray,
    base_lgd: float,
    multipliers: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pd12 = np.clip(base["pd_point"][:, None] * multipliers["pd_mult"][None, :], 0.0, 1.0)
    lifetime_pd = np.clip(lifetime_base[:, None] * multipliers["pd_mult"][None, :], 0.0, 1.0)
    lgd = np.clip(base_lgd * multipliers["lgd_mult"][None, :], 0.0, 1.0)
    ead = base["loan_amnt"][:, None] * multipliers["ead_mult"][None, :]
    discount = 1.0 / (1.0 + multipliers["discount_rate"][None, :])

    dpd = base["dpd"][:, None]
    pd_orig = base["pd_orig"][:, None]
    sicr_mask = (pd12 - pd_orig) > 0.02
    stage3 = np.broadcast_to(dpd >= 90.0, pd12.shape)
    stage2 = (~stage3) & ((dpd >= 30.0) | sicr_mask)
    effective_pd = np.where(stage3, 1.0, np.where(stage2, lifetime_pd, pd12))

    ecl = effective_pd * lgd * ead * discount
    stage2_share = np.mean(stage2, axis=0)
    stage3_share = np.mean(stage3, axis=0)
    total_ecl = np.sum(ecl, axis=0)
    return total_ecl.astype(float), stage2_share.astype(float), stage3_share.astype(float)


def _run_cupy_chunk(
    *,
    base_gpu: dict[str, Any],
    lifetime_base_gpu: Any,
    base_lgd: float,
    multipliers: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is not available in this environment.")
    pd_mult = cp.asarray(multipliers["pd_mult"], dtype=cp.float32)
    lgd_mult = cp.asarray(multipliers["lgd_mult"], dtype=cp.float32)
    ead_mult = cp.asarray(multipliers["ead_mult"], dtype=cp.float32)
    disc = cp.asarray(multipliers["discount_rate"], dtype=cp.float32)

    pd12 = cp.clip(base_gpu["pd_point"][:, None] * pd_mult[None, :], 0.0, 1.0)
    pd_high = cp.clip(base_gpu["pd_high"][:, None] * pd_mult[None, :], 0.0, 1.0)
    del pd_high
    lifetime_pd = cp.clip(lifetime_base_gpu[:, None] * pd_mult[None, :], 0.0, 1.0)
    lgd = cp.clip(cp.float32(base_lgd) * lgd_mult[None, :], 0.0, 1.0)
    ead = base_gpu["loan_amnt"][:, None] * ead_mult[None, :]
    discount = 1.0 / (1.0 + disc[None, :])

    dpd = base_gpu["dpd"][:, None]
    pd_orig = base_gpu["pd_orig"][:, None]
    sicr_mask = (pd12 - pd_orig) > cp.float32(0.02)
    stage3 = cp.broadcast_to(dpd >= cp.float32(90.0), pd12.shape)
    stage2 = (~stage3) & ((dpd >= cp.float32(30.0)) | sicr_mask)
    effective_pd = cp.where(stage3, 1.0, cp.where(stage2, lifetime_pd, pd12))
    ecl = effective_pd * lgd * ead * discount

    total_ecl = cp.asnumpy(cp.sum(ecl, axis=0)).astype(float)
    stage2_share = cp.asnumpy(cp.mean(stage2, axis=0)).astype(float)
    stage3_share = cp.asnumpy(cp.mean(stage3, axis=0)).astype(float)
    return total_ecl, stage2_share, stage3_share


def main(
    *,
    n_scenarios: int = 8192,
    chunk_size: int = 256,
    seed: int = 42,
    base_lgd: float = 0.45,
    correlation_profile: str = "moderate_credit",
    antithetic: bool = True,
) -> None:
    intervals = _load_intervals()
    train_raw, test_raw = _load_raw_splits()
    lifetime_table = _load_lifetime_table()
    temporal_context = _load_temporal_context()
    base, quality = _prepare_base_vectors(intervals=intervals, train=train_raw, test=test_raw)
    lifetime_base = _build_lifetime_base(base["pd_point"], base["grade"], lifetime_table)

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(int(n_scenarios) / int(chunk_size)))

    base_gpu = None
    lifetime_base_gpu = None
    if cp is not None:
        base_gpu = {
            "pd_point": cp.asarray(base["pd_point"], dtype=cp.float32),
            "pd_high": cp.asarray(base["pd_high"], dtype=cp.float32),
            "loan_amnt": cp.asarray(base["loan_amnt"], dtype=cp.float32),
            "dpd": cp.asarray(base["dpd"], dtype=cp.float32),
            "pd_orig": cp.asarray(base["pd_orig"], dtype=cp.float32),
        }
        lifetime_base_gpu = cp.asarray(lifetime_base, dtype=cp.float32)

    cpu_started = time.perf_counter()
    cpu_totals: list[np.ndarray] = []
    cpu_stage2: list[np.ndarray] = []
    cpu_stage3: list[np.ndarray] = []
    multipliers_cache: list[dict[str, np.ndarray]] = []
    scenario_ids: list[np.ndarray] = []
    all_shocks = _generate_shocks(
        rng=rng,
        n_scenarios=int(n_scenarios),
        correlation_profile=correlation_profile,
        antithetic=antithetic,
    )
    for chunk_idx in range(n_chunks):
        start = chunk_idx * int(chunk_size)
        size = min(int(chunk_size), int(n_scenarios) - start)
        shocks = all_shocks[start : start + size]
        multipliers = _scenario_multipliers(shocks)
        multipliers["pd_mult"] = multipliers["pd_mult"] * float(
            temporal_context.get("baseline_pd_mult", 1.0) or 1.0
        )
        totals, s2, s3 = _run_numpy_chunk(
            base=base,
            lifetime_base=lifetime_base,
            base_lgd=base_lgd,
            multipliers=multipliers,
        )
        cpu_totals.append(totals)
        cpu_stage2.append(s2)
        cpu_stage3.append(s3)
        multipliers_cache.append(multipliers)
        scenario_ids.append(np.arange(start, start + size, dtype=int))
    cpu_seconds = time.perf_counter() - cpu_started

    if cp is None:
        raise RuntimeError("CuPy is not available; run this script in the RAPIDS env.")

    gpu_started = time.perf_counter()
    gpu_totals: list[np.ndarray] = []
    gpu_stage2: list[np.ndarray] = []
    gpu_stage3: list[np.ndarray] = []
    for multipliers in multipliers_cache:
        totals, s2, s3 = _run_cupy_chunk(
            base_gpu=base_gpu,
            lifetime_base_gpu=lifetime_base_gpu,
            base_lgd=base_lgd,
            multipliers=multipliers,
        )
        gpu_totals.append(totals)
        gpu_stage2.append(s2)
        gpu_stage3.append(s3)
    cp.cuda.Stream.null.synchronize()
    gpu_seconds = time.perf_counter() - gpu_started

    cpu_total_ecl = np.concatenate(cpu_totals)
    gpu_total_ecl = np.concatenate(gpu_totals)
    cpu_stage2_share = np.concatenate(cpu_stage2)
    gpu_stage2_share = np.concatenate(gpu_stage2)
    cpu_stage3_share = np.concatenate(cpu_stage3)
    gpu_stage3_share = np.concatenate(gpu_stage3)
    scenario_id = np.concatenate(scenario_ids)

    distribution = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "cpu_total_ecl": cpu_total_ecl,
            "gpu_total_ecl": gpu_total_ecl,
            "abs_diff_total_ecl": np.abs(cpu_total_ecl - gpu_total_ecl),
            "cpu_stage2_share": cpu_stage2_share,
            "gpu_stage2_share": gpu_stage2_share,
            "cpu_stage3_share": cpu_stage3_share,
            "gpu_stage3_share": gpu_stage3_share,
        }
    )
    distribution["rel_diff_total_ecl_pct"] = (
        distribution["abs_diff_total_ecl"] / distribution["cpu_total_ecl"].clip(lower=1e-9) * 100.0
    )

    summary = {
        "schema_version": "2026-03-10.1",
        "run_tag": str(os.environ.get("PIPELINE_RUN_TAG", "")).strip() or "untracked",
        "n_loans": int(len(base["pd_point"])),
        "n_scenarios": int(n_scenarios),
        "chunk_size": int(chunk_size),
        "n_chunks": int(n_chunks),
        "seed": int(seed),
        "correlation_profile": correlation_profile,
        "antithetic": bool(antithetic),
        "temporal_source": str(temporal_context.get("source", "unknown")),
        "temporal_pd_baseline_mult": float(temporal_context.get("baseline_pd_mult", 1.0) or 1.0),
        "cpu_seconds": float(cpu_seconds),
        "gpu_seconds": float(gpu_seconds),
        "speedup_gpu_vs_cpu": float(cpu_seconds / max(gpu_seconds, 1e-9)),
        "cpu_tail": _tail_metrics(cpu_total_ecl),
        "gpu_tail": _tail_metrics(gpu_total_ecl),
        "max_abs_diff_total_ecl": float(distribution["abs_diff_total_ecl"].max()),
        "mean_abs_diff_total_ecl": float(distribution["abs_diff_total_ecl"].mean()),
        "max_rel_diff_total_ecl_pct": float(distribution["rel_diff_total_ecl_pct"].max()),
        "mean_rel_diff_total_ecl_pct": float(distribution["rel_diff_total_ecl_pct"].mean()),
        "input_quality": quality.to_dict(orient="records")[0],
    }

    data_dir = _artifact_path("data/processed")
    model_dir = _artifact_path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = data_dir / "ifrs9_mc_distribution.parquet"
    tail_path = data_dir / "ifrs9_mc_tail_metrics.json"
    distribution.to_parquet(distribution_path, index=False)
    tail_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved IFRS9 Monte Carlo distribution: {}", distribution_path)
    logger.info("Saved IFRS9 Monte Carlo summary: {}", tail_path)
    logger.info(
        "IFRS9 Monte Carlo benchmark | loans={} scenarios={} cpu_s={:.3f} gpu_s={:.3f} speedup={:.2f}x",
        len(base["pd_point"]),
        n_scenarios,
        cpu_seconds,
        gpu_seconds,
        summary["speedup_gpu_vs_cpu"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-lgd", type=float, default=0.45)
    parser.add_argument(
        "--correlation-profile",
        choices=["independent", "moderate_credit", "stress_credit"],
        default="moderate_credit",
    )
    parser.add_argument("--no-antithetic", action="store_true")
    args = parser.parse_args()
    main(
        n_scenarios=args.n_scenarios,
        chunk_size=args.chunk_size,
        seed=args.seed,
        base_lgd=args.base_lgd,
        correlation_profile=args.correlation_profile,
        antithetic=not args.no_antithetic,
    )
