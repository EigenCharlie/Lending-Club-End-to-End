#!/usr/bin/env python3
"""Run A/B robust-vs-non-robust sensitivity grid with stronger statistical power."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.simulate_ab_test import main as run_ab_once

SCHEMA_VERSION = "2026-03-01.1"


def _parse_int_grid(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("Sensitivity grid is empty.")
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A/B sensitivity grid.")
    parser.add_argument("--candidate-grid", default="10000,0")
    parser.add_argument("--boot-grid", default="1000,3000,5000")
    parser.add_argument("--seed-grid", default="42,52,62")
    parser.add_argument("--total_budget", type=float, default=1_000_000)
    parser.add_argument("--max_portfolio_pd", type=float, default=0.10)
    parser.add_argument("--no_regression_tolerance_pct", type=float, default=0.05)
    parser.add_argument("--out-dir", default="reports/ab_sensitivity")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    candidate_grid = _parse_int_grid(args.candidate_grid)
    boot_grid = _parse_int_grid(args.boot_grid)
    seed_grid = _parse_int_grid(args.seed_grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_tag = (
        str(args.run_tag or "").strip() or str(os.environ.get("PIPELINE_RUN_TAG", "")).strip()
    )
    if not resolved_run_tag:
        resolved_run_tag = "untracked"

    rows: list[dict[str, object]] = []
    for max_candidates in candidate_grid:
        for n_boot in boot_grid:
            for seed in seed_grid:
                stem = f"cand{max_candidates}_boot{n_boot}_seed{seed}"
                status_path = out_dir / f"{stem}.status.json"
                results_path = out_dir / f"{stem}.results.parquet"
                summary_path = out_dir / f"{stem}.summary.parquet"

                run_ab_once(
                    total_budget=float(args.total_budget),
                    max_portfolio_pd=float(args.max_portfolio_pd),
                    max_candidates=int(max_candidates),
                    n_boot=int(n_boot),
                    seed=int(seed),
                    no_regression_tolerance_pct=float(args.no_regression_tolerance_pct),
                    results_path=str(results_path),
                    summary_path=str(summary_path),
                    status_path=str(status_path),
                    run_tag=resolved_run_tag,
                )

                status = json.loads(status_path.read_text(encoding="utf-8"))
                comp = status.get("comparison", {})
                no_reg = status.get("no_regression", {})
                rows.append(
                    {
                        "max_candidates": int(max_candidates),
                        "n_boot": int(n_boot),
                        "seed": int(seed),
                        "p_value": float(comp.get("p_value", float("nan"))),
                        "significant": bool(comp.get("significant", False)),
                        "diff_total_return": float(
                            no_reg.get(
                                "diff_total_return",
                                (status.get("metrics_b", {}).get("total_return", 0.0))
                                - (status.get("metrics_a", {}).get("total_return", 0.0)),
                            )
                        ),
                        "no_regression_pass": bool(no_reg.get("passed", False)),
                        "status_path": str(status_path),
                    }
                )

    results_df = (
        pd.DataFrame(rows).sort_values(["max_candidates", "n_boot", "seed"]).reset_index(drop=True)
    )
    aggregate = (
        results_df.groupby(["max_candidates", "n_boot"], as_index=False)
        .agg(
            runs=("seed", "count"),
            p_value_median=("p_value", "median"),
            significance_rate=("significant", "mean"),
            no_regression_pass_rate=("no_regression_pass", "mean"),
            diff_total_return_median=("diff_total_return", "median"),
        )
        .sort_values(
            ["no_regression_pass_rate", "diff_total_return_median"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    detail_path = out_dir / f"ab_sensitivity_detail_{ts}.parquet"
    agg_path = out_dir / f"ab_sensitivity_aggregate_{ts}.parquet"
    status_out = out_dir / "ab_sensitivity_status.json"
    results_df.to_parquet(detail_path, index=False)
    aggregate.to_parquet(agg_path, index=False)

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_tag": resolved_run_tag,
        "candidate_grid": candidate_grid,
        "boot_grid": boot_grid,
        "seed_grid": seed_grid,
        "detail_path": str(detail_path),
        "aggregate_path": str(agg_path),
        "gate_mode": "no_regression",
        "significance_role": "diagnostic",
        "best_row": aggregate.iloc[0].to_dict() if not aggregate.empty else None,
    }
    status_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"[ab_sensitivity] detail: {detail_path}")
    print(f"[ab_sensitivity] aggregate: {agg_path}")
    print(f"[ab_sensitivity] status: {status_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
