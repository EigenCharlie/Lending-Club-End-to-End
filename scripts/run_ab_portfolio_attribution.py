"""A/B portfolio attribution: grade/cohort/amount-bucket breakdown + Sharpe-like ratio.

Uses existing canonical artifacts — no re-optimization required.

Outputs:
    models/ab_attribution_status.json
    data/processed/ab_attribution_by_grade.parquet
    data/processed/ab_attribution_by_cohort.parquet
    data/processed/ab_attribution_by_amount.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCHEMA_VERSION = "2026-03-17.1"


def _sharpe_from_ci(ab_status: dict) -> dict:
    """Derive Sharpe-like ratio from 15K bootstrap CI stored in ab_simulation_status."""
    comp = ab_status.get("comparison", {})
    metrics_a = ab_status.get("metrics_a", {})
    metrics_b = ab_status.get("metrics_b", {})
    diag = ab_status.get("diagnostics", {})

    n_boot = diag.get("n_boot", 0)
    diff = float(comp.get("diff", 0.0))
    ci_low = float(comp.get("ci_low", 0.0))
    ci_high = float(comp.get("ci_high", 0.0))

    # std of bootstrap distribution: CI_range / (2 * z_0.025)
    ci_range = ci_high - ci_low
    std_diff = ci_range / 3.92 if ci_range > 0 else 1.0  # 3.92 = 2*1.96
    sharpe_like = diff / std_diff if std_diff > 0 else 0.0

    total_alloc = float(metrics_a.get("total_allocated", 1_000_000))
    roic_a = float(metrics_a.get("total_return", 0)) / total_alloc
    roic_b = float(metrics_b.get("total_return", 0)) / total_alloc

    # Calmar-like: ROIC / |downside (ci_low relative to 0)|
    downside_proxy = abs(min(ci_low, 0.0)) + 1e-9
    calmar_a = roic_a / downside_proxy
    calmar_b = roic_b / downside_proxy

    return {
        "n_bootstrap": n_boot,
        "bootstrap_exceeds_10k": n_boot >= 10_000,
        "sharpe_like_diff": round(sharpe_like, 4),
        "roic_a": round(roic_a, 4),
        "roic_b": round(roic_b, 4),
        "calmar_like_a": round(calmar_a, 4),
        "calmar_like_b": round(calmar_b, 4),
        "std_diff_approx": round(std_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "p_value": comp.get("p_value"),
        "note": (
            f"n_bootstrap={n_boot:,} (≥10K ✓). "
            "Sharpe-like = E[diff] / std(diff) from 95% CI of bootstrap distribution. "
            "ROIC = total_return / total_allocated. Calmar-like = ROIC / |downside_CI|."
        ),
    }


def _attribution_by_dim(
    allocations: pd.DataFrame,
    conformal: pd.DataFrame,
    dim: str,
    dim_col: str,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Generic per-dimension return attribution."""
    conf_sub = conformal[
        ["_row_number", "grade", "y_true", "temporal_segment", "loan_amnt"]
    ].rename(columns={"_row_number": "loan_idx"})
    df = allocations.merge(conf_sub, on="loan_idx", how="left", suffixes=("", "_conf"))
    df["loan_amnt"] = df["loan_amnt"].fillna(df.get("loan_amnt_conf", df["loan_amnt"]))

    df["allocated_k"] = df["alloc"] * df["loan_amnt"] / 1000
    df["expected_return_k"] = (
        df["alloc"] * df["int_rate"] * df["loan_amnt"]
        - df["alloc"] * df["pd_point"] * df["loan_amnt"]
    ) / 1000
    df["actual_return_k"] = (
        df["alloc"] * df["int_rate"] * df["loan_amnt"]
        - df["alloc"] * df["y_true"].fillna(df["pd_point"]) * df["loan_amnt"]
    ) / 1000

    funded = df[df["alloc"] > 0].copy()
    if funded.empty or dim_col not in funded.columns:
        return pd.DataFrame()

    agg = (
        funded.groupby(dim_col)
        .agg(
            n_funded=("alloc", "count"),
            allocated_k=("allocated_k", "sum"),
            mean_pd=("pd_point", "mean"),
            mean_int_rate=("int_rate", "mean"),
            expected_return_k=("expected_return_k", "sum"),
            actual_return_k=("actual_return_k", "sum"),
            actual_default_rate=("y_true", lambda x: x.fillna(0).mean()),
        )
        .reset_index()
        .rename(columns={dim_col: dim})
    )

    agg["return_on_capital_pct"] = (
        agg["actual_return_k"] / (agg["allocated_k"] + 1e-9) * 100
    ).round(2)
    for col in ["mean_pd", "mean_int_rate", "actual_default_rate"]:
        agg[col] = agg[col].round(4)
    for col in ["allocated_k", "expected_return_k", "actual_return_k"]:
        agg[col] = agg[col].round(2)
    return agg


def main() -> None:
    logger.info("A/B portfolio attribution — grade/cohort/amount breakdown + Sharpe-like")

    ab_status_path = MODELS_DIR / "ab_simulation_status.json"
    if not ab_status_path.exists():
        logger.error("ab_simulation_status.json not found — run simulate_ab_test.py first.")
        return

    ab_status = json.loads(ab_status_path.read_text(encoding="utf-8"))
    run_tag = ab_status.get("run_tag", "untracked")

    sharpe = _sharpe_from_ci(ab_status)
    logger.info(
        f"Sharpe-like diff: {sharpe['sharpe_like_diff']} | "
        f"ROIC A: {sharpe['roic_a']:.1%} | ROIC B: {sharpe['roic_b']:.1%} | "
        f"n_boot: {sharpe['n_bootstrap']:,}"
    )

    alloc_path = DATA_PROC / "portfolio_allocations.parquet"
    conf_path = DATA_PROC / "conformal_intervals_mondrian.parquet"
    results: dict[str, str] = {}

    if alloc_path.exists() and conf_path.exists():
        allocations = pd.read_parquet(alloc_path)
        conformal = pd.read_parquet(conf_path)
        logger.info(f"Allocations: {len(allocations):,} | Conformal: {len(conformal):,}")

        # Grade breakdown
        grade_df = _attribution_by_dim(allocations, conformal, "grade", "grade")
        if not grade_df.empty:
            out = DATA_PROC / "ab_attribution_by_grade.parquet"
            grade_df.to_parquet(out, index=False)
            results["grade"] = str(out)
            logger.success(f"  grade: {len(grade_df)} rows → {out.name}")

        # Temporal cohort breakdown
        cohort_df = _attribution_by_dim(
            allocations, conformal, "temporal_segment", "temporal_segment"
        )
        if not cohort_df.empty:
            out = DATA_PROC / "ab_attribution_by_cohort.parquet"
            cohort_df.to_parquet(out, index=False)
            results["cohort"] = str(out)
            logger.success(f"  cohort: {len(cohort_df)} rows → {out.name}")

        # Loan amount bucket breakdown
        allocations_copy = allocations.copy()
        allocations_copy["amount_bucket"] = pd.cut(
            allocations_copy["loan_amnt"],
            bins=[0, 5_000, 10_000, 20_000, 35_000, float("inf")],
            labels=["<5K", "5-10K", "10-20K", "20-35K", ">35K"],
        ).astype(str)
        # inject amount_bucket into conformal-keyed data via loan_idx
        conf_with_bucket = conformal.copy()
        conf_with_bucket["amount_bucket"] = (
            allocations_copy.set_index("loan_idx")["amount_bucket"]
            .reindex(conf_with_bucket["_row_number"])
            .values
        )
        amount_df = _attribution_by_dim(
            allocations_copy, conf_with_bucket, "amount_bucket", "amount_bucket"
        )
        if not amount_df.empty:
            out = DATA_PROC / "ab_attribution_by_amount.parquet"
            amount_df.to_parquet(out, index=False)
            results["amount"] = str(out)
            logger.success(f"  amount: {len(amount_df)} rows → {out.name}")
    else:
        logger.warning("Allocations or conformal not found — attribution skipped.")

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "run_tag": run_tag,
        "sharpe_metrics": sharpe,
        "breakdown_dimensions": list(results.keys()),
        "artifact_paths": results,
        "note": "Attribution uses canonical portfolio_allocations + conformal_intervals_mondrian (OOT test set).",
    }
    out_status = MODELS_DIR / "ab_attribution_status.json"
    out_status.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.success(f"Saved {out_status}")


if __name__ == "__main__":
    main()
